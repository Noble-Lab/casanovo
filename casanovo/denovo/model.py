"""A de novo peptide sequencing model."""

import collections
import inspect
import itertools
import logging
import warnings
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import depthcharge.constants
import einops
import lightning.pytorch as pl
import numpy as np
import torch
import tqdm
from depthcharge.tokenizers import PeptideTokenizer

from .. import config
from ..data import ms_io, psm
from ..data.db_utils import PROTON
from ..denovo.transformers import PeptideDecoder, SpectrumEncoder
from . import evaluate
from .muon import MuonWithAuxAdamW

logger = logging.getLogger("casanovo")

H2O_MASS = depthcharge.constants.H2O
ISOTOPE_SPACING = depthcharge.constants.C13
# Precise mass control (PMC) decoding: the mass discretization step, the
# headroom that keeps rounding from carrying a valid path off the end of
# the mass axis, a cap on the backtracking table, and the frame
# probability below which a token is not considered at that frame. The
# last prunes hard because CTC output is blank-dominated: only a handful
# of frames per spectrum carry a plausible residue.
PMC_RESOLUTION = 0.01
PMC_MASS_GUARD = 0.1
PMC_MAX_POINTER_BYTES = 1_000_000_000
PMC_MIN_EMIT_PROB = 1e-4


class Spec2Pep(pl.LightningModule):
    """
    A Transformer model for de novo peptide sequencing.

    Use this model in conjunction with a pytorch-lightning Trainer.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality used by the transformer model.
    n_head : int
        The number of attention heads in each layer. ``dim_model`` must
        be divisible by ``n_head``.
    dim_feedforward : int
        The dimensionality of the fully connected layers in the
        transformer model.
    n_layers : int
        The number of transformer layers.
    dropout : float
        The dropout probability for all layers.
    dim_intensity : Optional[int]
        The number of features to use for encoding peak intensity. The
        remaining (``dim_model - dim_intensity``) are reserved for
        encoding the m/z value. If ``None``, the intensity will be
        projected up to ``dim_model`` using a linear layer, then summed
        with the m/z encoding for each peak.
    decoder_frames : int
        How many frames the decoder emits, which CTC aligns to the
        shorter peptide. It has to exceed the longest peptide plus one
        frame per repeated residue, and self-attention grows with its
        square.
    residues : str | Dict[str, float]
        The amino acid dictionary and their masses. By default
        ("canonical") this is only the 20 canonical amino acids, with
        cysteine carbamidomethylated. If "massivekb", this dictionary
        will include the modifications found in MassIVE-KB.
        Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int
        The maximum precursor charge to consider.
    precursor_mass_tol : float
        The precursor mass tolerance (in ppm). It sets the window that
        precise mass control searches when the greedy prediction misses
        the precursor mass. Set to ``inf`` to turn precise mass control
        off.
    isotope_error_range : Tuple[int, int]
        The isotope errors, from low to high and inclusive, that precise
        mass control considers. Each one moves the window down by that
        many isotope spacings (1.00335 Da) of the neutral precursor mass.
    min_peptide_len : int
        The minimum length of predicted peptides.
    n_beams : int
        Number of beams used during beam search decoding.
    top_match : int
        Number of PSMs to return for each spectrum.
    n_log : int
        The number of epochs to wait between logging messages.
    warmup_iters : int
        The number of iterations for the linear warm-up of the learning
        rate.
    cosine_schedule_period_iters : int
        The number of iterations for the cosine half period of the
        learning rate.
    out_writer : ms_io.MztabWriter | None
        The output writer for the prediction results.
    calculate_precision : bool
        Calculate the validation set precision during training.
        This is expensive.
    tokenizer: PeptideTokenizer | None
        Tokenizer object to process peptide sequences.
    inter_ctc_layers : Sequence[int]
        Decoder layers (1-based) that also score their hidden states for
        an auxiliary CTC loss. Scoring reuses the output layer, so this
        adds no parameters and changes nothing at inference. Layers
        outside the stack are dropped with a warning.
    inter_ctc_weight : float
        Share of the training loss carried by those auxiliary
        predictions; the final layer carries the rest.
    muon_lr : float
        Learning rate for the Muon parameter group, the hidden weight
        matrices. Muon updates have unit RMS, so this is on a different
        scale than the auxiliary group's.
    muon_momentum : float
        Momentum for the Muon parameter group.
    **kwargs : Dict
        Additional keyword arguments for the auxiliary AdamW group. Only
        valid Adam parameters are retained; any other values are ignored.
    """

    def __init__(
        self,
        dim_model: int = 512,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 9,
        dropout: float = 0.0,
        decoder_frames: int = 100,
        residues: str | Dict[str, float] = "canonical",
        max_charge: int = 5,
        precursor_mass_tol: float = 50,
        isotope_error_range: Tuple[int, int] = (0, 3),
        min_peptide_len: int = 6,
        n_beams: int = 1,
        top_match: int = 1,
        n_log: int = 10,
        warmup_iters: int = 100_000,
        cosine_schedule_period_iters: int = 600_000,
        inter_ctc_layers: Sequence[int] = (),
        inter_ctc_weight: float = 0.5,
        muon_lr: float = 0.002,
        muon_momentum: float = 0.95,
        out_writer: Optional[ms_io.MztabWriter] = None,
        calculate_precision: bool = False,
        tokenizer: PeptideTokenizer | None = None,
        **kwargs: Dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Reversed, as `model_runner` builds it. Precise mass control needs
        # that direction, and depthcharge defaults to the other, so a bare
        # model would otherwise lose mass control without saying so.
        self.tokenizer = tokenizer or PeptideTokenizer(reverse=True)
        # Vocabulary: tokenizer tokens incl. padding (0), plus a
        # dedicated CTC blank class as the last index.
        self.vocab_size = len(self.tokenizer) + 2
        self.blank_token = self.vocab_size - 1
        # Build the model.
        self.encoder = SpectrumEncoder(
            d_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
        )
        # Intermediate CTC: which decoder layers score their own hidden
        # states for an auxiliary loss, and how much of the loss those
        # auxiliary predictions carry. Empty list disables it, and the
        # model is then identical to the plain CTC decoder.
        self.inter_ctc_layers = tuple(inter_ctc_layers or ())
        self.inter_ctc_weight = float(inter_ctc_weight)
        # Outside [0, 1] the mix rewards a worse final prediction; NaN
        # fails the comparison too.
        if not 0.0 <= self.inter_ctc_weight <= 1.0:
            raise ValueError(
                f"inter_ctc_weight ({inter_ctc_weight}) must be between 0 "
                "and 1"
            )
        self.decoder = PeptideDecoder(
            n_tokens=self.tokenizer,
            d_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            max_charge=max_charge,
            inter_ctc_layers=self.inter_ctc_layers,
        )
        # The decoder keeps only the layers it has, so take its list back as
        # the effective one. Holding the raw request here would let the gate
        # in _forward_step ask for intermediates from a decoder that scores
        # nothing, and the auxiliary loss would be skipped without a word.
        self.inter_ctc_layers = self.decoder.inter_ctc_layers
        self.softmax = torch.nn.Softmax(2)
        self.ctc_loss = torch.nn.CTCLoss(
            blank=self.blank_token, zero_infinity=True
        )
        self._ctc_infeasible_warned = False
        self._pmc_size_warned = False
        self._pmc_vocab_warned = False
        self._pmc_reverse_warned = False
        # Optimizer settings.
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        self.muon_lr = muon_lr
        self.muon_momentum = muon_momentum
        # `kwargs` will contain additional arguments as well as
        # unrecognized arguments, including deprecated ones. Remove the
        # deprecated ones.
        for k in config._config_deprecated:
            kwargs.pop(k, None)
            warnings.warn(
                f"Deprecated hyperparameter '{k}' removed from the model.",
                DeprecationWarning,
            )
        # Keep only valid Adam arguments; other configuration values
        # (e.g. loaded from a checkpoint) must not reach the optimizer.
        adam_kwargs = set(inspect.signature(torch.optim.Adam).parameters)
        self.opt_kwargs = {k: v for k, v in kwargs.items() if k in adam_kwargs}

        # Data properties.
        self.decoder_frames = decoder_frames
        self.residues = residues
        self.precursor_mass_tol = precursor_mass_tol
        if isotope_error_range[0] > isotope_error_range[1]:
            raise ValueError(
                f"isotope_error_range {tuple(isotope_error_range)} must run "
                "from low to high"
            )
        self.isotope_error_range = isotope_error_range
        self.min_peptide_len = min_peptide_len
        self.n_beams = n_beams
        self.top_match = top_match

        # Logging.
        self.calculate_precision = calculate_precision
        self.n_log = n_log
        self._history = []
        # Count of spectra for which decoding returned no valid peptide.
        self.n_missing_predictions = 0
        # Per-file validation metadata; set by ModelRunner.train() before fit.
        self.val_stems: list = []
        self.n_main_loaders: int = 0

        # Output writer during predicting.
        self.out_writer = out_writer

        self._init_tokenizer_state()

    def _init_tokenizer_state(self) -> None:
        """
        Derive the decoding tables from the current tokenizer.

        Called again by ``ModelRunner`` after it swaps in a tokenizer
        with a different vocabulary, so that the stop token, the
        N-terminal and negative-mass token indices, and the token masses
        follow the new token ids.
        """
        # On a rebuild the buffers already sit on the model's device.
        device = (
            self.nterm_idx.device
            if "nterm_idx" in self._buffers
            else torch.device("cpu")
        )
        self.stop_token = self.tokenizer.stop_int
        # Get n-term mod tokens
        self.n_term = [
            aa
            for aa in self.tokenizer.index
            if aa.startswith("[") and aa.endswith("]-")
        ]
        # Register tensor buffers for negative mass amino acid indices
        self.register_buffer(
            "neg_mass_idx",
            torch.tensor(
                [
                    self.tokenizer.index[aa]  # all negative‑mass AAs
                    for aa, mass in self.tokenizer.residues.items()
                    if mass < 0
                ],
                dtype=torch.int,
                device=device,
            ),
            persistent=False,
        )

        # Register tensor buffer for N-terminal modification indices
        self.register_buffer(
            "nterm_idx",
            torch.tensor(
                [self.tokenizer.index[aa] for aa in self.n_term],
                dtype=torch.int,
                device=device,
            ),
            persistent=False,
        )

        # Amino acid token masses, in float64 for precise mass control.
        # A plain CPU tensor rather than a buffer: MPS has no float64, so
        # moving it with the model would fail there.
        self.token_masses = torch.zeros(self.vocab_size, dtype=torch.float64)
        # Populate token masses from tokenizer residues
        for aa, mass in self.tokenizer.residues.items():
            idx = self.tokenizer.index.get(aa)
            if idx is not None:
                self.token_masses[idx] = mass

    @property
    def device(self) -> torch.device:
        """
        The device on which the model is currently running.

        Returns
        -------
        torch.device
            The device on which the model is currently running.
        """
        return next(self.parameters()).device

    def _process_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert a SpectrumDataset batch to tensors.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, which contains keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``, and
            ``precursor_charge``, each pointing to tensors with the
            corresponding data. The ``seq`` key is optional and
            contains the peptide sequences for training.

        Returns
        -------
        mzs : torch.Tensor of shape (batch_size, max_peaks)
            The m/z values for each spectrum.
        intensities : torch.Tensor of shape (batch_size, max_peaks)
            The intensity values for each spectrum.
        precursors : torch.Tensor of shape (batch_size, 3)
            A tensor with the precursor neutral mass, precursor charge,
            and precursor m/z.
        seqs : np.ndarray
            The spectrum identifiers (during de novo sequencing) or
            peptide sequences (during training).
        """
        precursor_mzs = batch["precursor_mz"].squeeze(0)
        precursor_charges = batch["precursor_charge"].squeeze(0)
        precursor_masses = (precursor_mzs - 1.007276) * precursor_charges
        precursors = torch.vstack(
            [precursor_masses, precursor_charges, precursor_mzs]
        ).T

        mzs = batch["mz_array"]
        intensities = batch["intensity_array"]
        seqs = batch.get("seq")

        return mzs, intensities, precursors, seqs

    def forward(self, batch):
        return self._forward_step(batch)

    def _forward_step(
        self,
        batch: Dict[str, torch.Tensor],
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        The forward learning step for non-autoregressive decoding.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset.

        Returns
        -------
        scores : torch.Tensor of shape
                (n_spectra, decoder_frames + 1, n_amino_acids)
            The frame-level amino acid scores for each prediction. The
            extra frame is the precursor token, which the decoder puts
            at position 0 and the output layer scores like any other.
        seqs : torch.Tensor of shape (n_spectra, length) or None
            The ground truth tokens for training, or None for inference.
        intermediates : list of torch.Tensor
            Only when ``return_intermediates``: the scores from each
            layer in ``inter_ctc_layers``, for the auxiliary CTC losses.
            Empty when that list is.
        """
        mzs, ints, precursors, seqs = self._process_batch(batch)
        memories, mem_masks = self.encoder(mzs, ints)

        # Decode a fixed number of frames; the CTC loss aligns them to
        # the (shorter) ground truth peptide.
        zero_tokens = torch.zeros(
            (mzs.shape[0], self.decoder_frames),
            dtype=torch.long,
            device=self.device,
        )
        # Score the intermediate layers only when the caller uses them;
        # inference would discard them.
        if return_intermediates and self.inter_ctc_layers:
            scores, intermediates = self.decoder.forward_with_intermediates(
                tokens=zero_tokens,
                memory=memories,
                memory_key_padding_mask=mem_masks,
                precursors=precursors,
            )
        else:
            scores = self.decoder(
                tokens=zero_tokens,
                memory=memories,
                memory_key_padding_mask=mem_masks,
                precursors=precursors,
            )
            intermediates = []

        if return_intermediates:
            return scores, seqs, intermediates
        return scores, seqs

    def _ctc_losses(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
        """
        Compute the CTC losses for a batch.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, which contains keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``, and
            ``precursor_charge``, each pointing to tensors with the
            corresponding data. The ``seq`` key is optional and
            contains the peptide sequences for training.

        Returns
        -------
        loss : torch.Tensor
            The final layer's CTC loss.
        total : torch.Tensor
            The objective the optimizer sees: ``loss`` mixed with the
            intermediate CTC loss when ``inter_ctc_layers`` is set, else
            ``loss`` itself.
        aux : Optional[torch.Tensor]
            The mean intermediate CTC loss, or ``None`` without
            intermediate layers.
        batch_size : int
            The number of spectra in the batch.
        """
        pred, truth, intermediates = self._forward_step(
            batch, return_intermediates=True
        )

        log_probs = _ctc_log_probs(pred)  # (T, B, V)
        input_lengths = torch.full(
            (truth.shape[0],),
            log_probs.shape[0],
            dtype=torch.long,
            device=truth.device,
        )
        target_lengths = (truth != 0).sum(dim=1)
        # CTC needs one frame per token plus a blank between repeated
        # tokens; longer peptides are unalignable and get zero loss
        # (zero_infinity), so warn that they do not contribute.
        repeats = ((truth[:, 1:] == truth[:, :-1]) & (truth[:, 1:] != 0)).sum(
            dim=1
        )
        infeasible = target_lengths + repeats > input_lengths
        if infeasible.any() and not self._ctc_infeasible_warned:
            self._ctc_infeasible_warned = True
            logger.warning(
                "%d peptide(s) in this batch need more CTC frames than "
                "decoder_frames + 1 = %d and will contribute zero loss. "
                "Increase decoder_frames to include them in training.",
                infeasible.sum().item(),
                self.decoder_frames + 1,
            )
        loss = self._ctc(log_probs, truth, input_lengths, target_lengths)
        if not intermediates:
            return loss, loss, None, truth.shape[0]

        # Intermediate CTC: the same objective on each intermediate
        # layer's own prediction, so the layers below the last one are
        # supervised directly rather than only through it.
        aux = torch.stack(
            [
                self._ctc(
                    _ctc_log_probs(scores),
                    truth,
                    input_lengths,
                    target_lengths,
                )
                for scores in intermediates
            ]
        ).mean()
        total = (1 - self.inter_ctc_weight) * loss + (
            self.inter_ctc_weight * aux
        )
        return loss, total, aux, truth.shape[0]

    def _ctc(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        The CTC loss, computed on the CPU when the inputs are on MPS.

        PyTorch has no MPS kernel for CTC loss. Moving its inputs to the
        CPU keeps the rest of training on MPS, and the move is
        differentiable, so gradients still reach the model.

        Parameters
        ----------
        log_probs : torch.Tensor of shape (n_frames, batch, n_tokens)
            Frame-level log-probabilities.
        targets : torch.Tensor of shape (batch, max_target_len)
            The target token indices.
        input_lengths : torch.Tensor of shape (batch,)
            The number of frames per spectrum.
        target_lengths : torch.Tensor of shape (batch,)
            The number of target tokens per spectrum.

        Returns
        -------
        torch.Tensor
            The CTC loss, on the device of ``log_probs``.
        """
        device = log_probs.device
        if device.type != "mps":
            return self.ctc_loss(
                log_probs, targets, input_lengths, target_lengths
            )
        return self.ctc_loss(
            log_probs.cpu(),
            targets.cpu(),
            input_lengths.cpu(),
            target_lengths.cpu(),
        ).to(device)

    def _log_ctc_losses(
        self,
        prefix: str,
        loss: torch.Tensor,
        total: torch.Tensor,
        aux: Optional[torch.Tensor],
        **log_kwargs,
    ) -> None:
        """
        Log the CTC losses under ``prefix``.

        The final layer's CTC loss is what gets logged as
        ``{prefix}_CTCLoss``, so it stays comparable with runs that
        predate intermediate CTC and keeps driving best-checkpoint
        selection. The combined objective is what the optimizer sees;
        logging that instead would mix in the shallower layers' worse
        predictions and both compare wrongly against earlier curves and
        pick checkpoints on the wrong quantity.

        Parameters
        ----------
        prefix : str
            Logging key to describe the current stage.
        loss, total, aux
            The losses returned by ``_ctc_losses``.
        **log_kwargs
            Keyword arguments passed on to ``self.log``.
        """
        self.log(f"{prefix}_CTCLoss", loss.detach(), **log_kwargs)
        if aux is not None:
            self.log(
                f"{prefix}_CTCLoss_intermediate", aux.detach(), **log_kwargs
            )
            self.log(f"{prefix}_CTCLoss_total", total.detach(), **log_kwargs)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        *args,
    ) -> torch.Tensor:
        """
        A single training step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, which contains keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``, and
            ``precursor_charge``, each pointing to tensors with the
            corresponding data. The ``seq`` key is optional and
            contains the peptide sequences for training.

        Returns
        -------
        torch.Tensor
            The loss of the training step.
        """
        loss, total, aux, batch_size = self._ctc_losses(batch)
        self._log_ctc_losses(
            "train",
            loss,
            total,
            aux,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return total

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """
        A single validation step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, which contains keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``, and
            ``precursor_charge``, each pointing to tensors with the
            corresponding data. The ``seq`` key is optional and
            contains the peptide sequences for training.
        batch_idx : int
            Index of the current batch within its dataloader.
        dataloader_idx : int
            Index of the dataloader this batch comes from. Dataloaders
            0..n_main_loaders-1 are "main" validation files that contribute
            to the aggregate ``valid_CTCLoss`` used for checkpoint selection.
            Higher indices are "tracking" files logged per-file only.

        Returns
        -------
        torch.Tensor
            The loss of the validation step.
        """
        loss, total, aux, batch_size = self._ctc_losses(batch)
        log_kwargs = dict(
            add_dataloader_idx=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        # Determine per-file stem and main/tracking classification.
        n_main = self.n_main_loaders if self.n_main_loaders > 0 else 1
        is_main = dataloader_idx < n_main

        if self.val_stems and dataloader_idx < len(self.val_stems):
            stem = self.val_stems[dataloader_idx]
            self.log(f"valid_CTCLoss/{stem}", loss.detach(), **log_kwargs)

        if is_main:
            # Contributes to the aggregate ``valid_CTCLoss`` monitored by
            # ModelCheckpoint for best-checkpoint selection.
            self._log_ctc_losses("valid", loss, total, aux, **log_kwargs)

        if not self.calculate_precision or not is_main:
            return total

        # Calculate and log amino acid and peptide match evaluation
        # metrics from the predicted peptides (main files only).
        peptides_true = self.tokenizer.detokenize(batch["seq"])
        logits, _ = self.forward(batch)
        # Withheld as predict_step withholds them; None is what the
        # evaluator counts as a missing prediction.
        peptides_pred = [
            (
                "".join(
                    self.tokenizer.detokenize(
                        torch.tensor([tokens]), join=False
                    )[0]
                )
                if tokens and len(tokens) >= self.min_peptide_len
                else None
            )
            for tokens in self._ctc_decode(logits)[0]
        ]
        aa_precision, _, pep_precision = evaluate.aa_match_metrics(
            *evaluate.aa_match_batch(
                peptides_true, peptides_pred, self.tokenizer.residues
            )
        )

        batch_size = len(peptides_true)
        log_args = dict(
            add_dataloader_idx=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "pep_precision", pep_precision, **log_args, batch_size=batch_size
        )
        self.log(
            "aa_precision", aa_precision, **log_args, batch_size=batch_size
        )
        return total

    def _ctc_decode(
        self, logits: torch.Tensor
    ) -> Tuple[List[List[int]], List[List[float]], List[Optional[float]]]:
        """
        Greedy CTC decoding of frame-level logits.

        Collapse repeated frame predictions, remove blanks, and truncate
        at the stop token. N-terminal modification tokens are only
        valid at the peptide's N-terminus; elsewhere they are re-decided
        from their frame with all N-terminal tokens masked out, and
        dropped if the re-decision yields a blank or stop token.

        Parameters
        ----------
        logits : torch.Tensor of shape (n_spectra, n_frames, n_tokens)
            The frame-level amino acid scores.

        Returns
        -------
        sequences : List[List[int]]
            The decoded token indices for each spectrum.
        scores : List[List[float]]
            The confidence for each decoded token, taken as the maximum
            probability over the frames merged into that token.
        stop_scores : List[Optional[float]]
            The stop's own confidence, where one was emitted. It is
            scored with the peptide but is not part of it, so it is
            returned apart from the tokens rather than among them.
        """
        probs = torch.softmax(logits, dim=-1)
        frame_confs, frame_tokens = probs.max(dim=-1)
        nterm_idx = set(self.nterm_idx.tolist())
        # Padding (0) is treated as non-emitting, like the blank.
        silent = (0, self.blank_token)

        sequences, scores, stop_scores = [], [], []
        for b, (tokens, confs) in enumerate(
            zip(frame_tokens.tolist(), frame_confs.tolist())
        ):
            seq, conf, frames, prev = [], [], [], self.blank_token
            for j, (token, prob) in enumerate(zip(tokens, confs)):
                if token not in silent:
                    if token != prev:
                        seq.append(token)
                        conf.append(prob)
                        frames.append(j)
                    elif prob > conf[-1]:
                        conf[-1] = prob
                        frames[-1] = j
                prev = token
            # The stop ends the peptide but is scored with it, the way
            # upstream's peptide score counts it, so its confidence is
            # kept aside rather than dropped with the frames after it.
            stop_conf = None
            if self.stop_token in seq:
                stop = seq.index(self.stop_token)
                stop_conf = conf[stop]
                seq, conf, frames = seq[:stop], conf[:stop], frames[:stop]
            # With a reversed tokenizer the N-terminus is the last token.
            nterm_pos = len(seq) - 1 if self.tokenizer.reverse else 0
            fixed_seq, fixed_conf = [], []
            for i, (token, prob, frame) in enumerate(zip(seq, conf, frames)):
                if i != nterm_pos and token in nterm_idx:
                    masked = logits[b, frame].clone()
                    masked[self.nterm_idx] = -float("inf")
                    token = int(masked.argmax())
                    if token in silent or token == self.stop_token:
                        continue
                    prob = torch.softmax(masked, dim=0)[token].item()
                fixed_seq.append(token)
                fixed_conf.append(prob)
            sequences.append(fixed_seq)
            scores.append(fixed_conf)
            stop_scores.append(stop_conf)
        return sequences, scores, stop_scores

    def _residue_mass_windows(
        self, precursor_mass: float, guard: float = 0.0
    ) -> List[Tuple[float, float]]:
        """
        Acceptable total residue mass windows for a precursor.

        One window per isotope error in ``isotope_error_range``, each
        spanning the precursor mass tolerance (in ppm), expressed in
        residue-sum space (i.e. with the water mass removed).

        Parameters
        ----------
        precursor_mass : float
            The observed precursor neutral mass.
        guard : float
            Extra widening (in Da) applied to both window edges.

        Returns
        -------
        List[Tuple[float, float]]
            The (lower, upper) bounds of each window.
        """
        tol = self.precursor_mass_tol * precursor_mass / 1e6
        return [
            (center - tol - guard, center + tol + guard)
            for iso in range(
                self.isotope_error_range[0], self.isotope_error_range[1] + 1
            )
            for center in [precursor_mass - iso * ISOTOPE_SPACING - H2O_MASS]
        ]

    def _fits_precursor_mass(
        self, tokens: List[int], precursor_mass: float
    ) -> bool:
        """
        Check whether a peptide matches the precursor mass.

        Parameters
        ----------
        tokens : List[int]
            The decoded token indices.
        precursor_mass : float
            The observed precursor neutral mass.

        Returns
        -------
        bool
            True if the total residue mass falls within the precursor
            mass tolerance for any allowed isotope error.
        """
        mass = (
            self.token_masses[torch.tensor(tokens, dtype=torch.long)]
            .sum()
            .item()
        )
        return any(
            lo <= mass <= hi
            for lo, hi in self._residue_mass_windows(precursor_mass)
        )

    def _pmc_grid(
        self, precursor_mass: float, n_frames: int, vocab: int
    ) -> Optional[Tuple[float, int, int, float, List[Tuple[float, float]]]]:
        """
        Size the mass axis the PMC search runs over.

        Returns ``(resolution, zero_bin, n_bins, hi_max, accept_windows)``,
        or None when no finite window bounds the search.
        """
        # Mass discretization. The pointer table is the memory
        # bottleneck, so pick the finest resolution whose table fits in
        # PMC_MAX_POINTER_BYTES rather than giving up on mass control
        # for heavy precursors. The guard widens the mass axis to hold
        # accumulated rounding, which grows with the resolution and the
        # number of emissions; acceptance tests the exact mass.
        base = self._residue_mass_windows(precursor_mass, PMC_MASS_GUARD)
        hi_base = max(hi for _, hi in base)
        if hi_base <= 0:
            return None
        # An unbounded window constrains nothing: precursor_mass_tol is
        # "inf", which is how PMC is turned off. Bail out rather than build
        # a mass axis of infinite extent, whose resolution would come out
        # inf and whose bin count would be NaN. Only an empty greedy
        # peptide reaches here under an infinite tolerance, since a
        # non-empty one already "fits" and predict_step skips the search.
        if not np.isfinite(hi_base):
            return None
        max_bins = max(4, PMC_MAX_POINTER_BYTES // (n_frames * vocab))
        resolution = max(PMC_RESOLUTION, hi_base / (max_bins - 2))
        guard = max(PMC_MASS_GUARD, float(np.sqrt(n_frames)) * resolution)
        axis_windows = self._residue_mass_windows(precursor_mass, guard)
        hi_max = max(hi for _, hi in axis_windows)
        # What the readout accepts: the true windows, unguarded. The
        # states carry exact masses, so this is the same predicate
        # `_fits_precursor_mass` applies to the finished peptide.
        accept_windows = self._residue_mass_windows(precursor_mass)
        # Negative-mass tokens (the ammonia-loss N-terminal modification)
        # push the running total below zero, so the axis starts below it.
        # Only N-terminal modifications may be negative and only one may be
        # emitted, so one token's worth of headroom is enough.
        neg_mass = min(
            (self.token_masses[c].item() for c in self.neg_mass_idx.tolist()),
            default=0.0,
        )
        # One bin above that for the stop, which carries no mass but moves
        # the state anyway so that it cannot be emitted twice. A path that
        # emitted one runs one bin above a path of the same mass that did
        # not, so the axis needs the room.
        zero_bin = int(np.ceil(max(0.0, -neg_mass) / resolution)) + 1
        n_bins = zero_bin + int(hi_max / resolution) + 3
        if n_bins > max_bins:
            # The widened guard pushed the table back over budget. The
            # offset bins have to come out of the budget too, so solve for
            # the resolution with them already reserved.
            resolution = hi_max / max(1, max_bins - zero_bin - 3)
            zero_bin = int(np.ceil(max(0.0, -neg_mass) / resolution)) + 1
            n_bins = zero_bin + int(hi_max / resolution) + 3
        # The floor of four bins above can still leave the table over
        # budget for very long frame counts. Decline the search then;
        # the caller keeps the greedy peptide.
        if n_frames * n_bins * vocab > PMC_MAX_POINTER_BYTES:
            return None
        if resolution > PMC_RESOLUTION and not self._pmc_size_warned:
            self._pmc_size_warned = True
            logger.warning(
                "Coarsening precise mass control decoding to %.4f Da for "
                "large precursor masses to keep the DP table under %d "
                "bytes.",
                resolution,
                PMC_MAX_POINTER_BYTES,
            )
        return resolution, zero_bin, n_bins, hi_max, accept_windows

    def _pmc_alphabet(
        self, vocab: int, hi_max: float, resolution: float, device
    ) -> Optional[
        Tuple[Any, Any, Any, List[bool], List[bool], Dict[int, int]]
    ]:
        """
        Which tokens the PMC search may emit, and by how much mass.

        Returns ``(emit_idx, delta_idx, emit_exact, emit_pinned,
        emit_is_stop, deltas)``, or None when nothing is emittable within
        the mass axis.
        """
        # Emittable tokens and their discretized masses. Only padding and
        # the blank never emit; the blank stays traversable, as CTC needs.
        excluded = {0, self.blank_token}
        # Pinned tokens may only be emitted at one point on the mass
        # axis, which is what lets a token of zero or negative mass in at
        # all: it cannot then repeat without bound. An N-terminal
        # modification opens the peptide, and the stop opens PMC's own
        # order, since the reversed tokenizer puts it last and the frames
        # are flipped. Both are therefore emitted from the zero-mass bin.
        pinned = set(self.nterm_idx.tolist()) | {self.stop_token}
        # A negative N-terminal modification lowers the running total
        # before any residue lands, so a residue may weigh up to its
        # magnitude more than the axis top: [Ammonia-loss]-W totals
        # about 169 Da although W alone is about 186 Da.
        neg_mass = min(
            (self.token_masses[c].item() for c in self.neg_mass_idx.tolist()),
            default=0.0,
        )
        mass_cap = hi_max - min(0.0, neg_mass)
        emit_tokens, emit_deltas, emit_pinned, emit_is_stop = [], [], [], []
        for c in range(min(vocab, self.token_masses.shape[0])):
            mass = self.token_masses[c].item()
            if c in excluded or mass > mass_cap:
                continue
            if mass <= 0 and c not in pinned:
                # A non-terminal residue of zero or negative mass could
                # repeat without bound, which the fixed axis cannot hold.
                continue
            emit_tokens.append(c)
            # The stop carries no mass, so pinning it to the zero-mass bin
            # leaves that bin still satisfying the pin: a blank could
            # return there and emit another. Step it one bin instead. The
            # bin is not the mass, which `state_mass` carries exactly and
            # the readout tests, so this only moves the state out of reach
            # of its own pin. One stop per path, without a state bit.
            emit_deltas.append(
                1 if c == self.stop_token else round(mass / resolution)
            )
            emit_pinned.append(c in pinned)
            emit_is_stop.append(c == self.stop_token)
        if not emit_tokens:
            return None
        deltas = dict(zip(emit_tokens, emit_deltas))
        emit_idx = torch.tensor(emit_tokens, device=device)
        delta_idx = torch.tensor(emit_deltas, device=device)
        # The unrounded masses, added to the running total alongside the
        # rounded ones that move the state along the axis.
        emit_exact = self.token_masses.to(device)[emit_idx]
        return (
            emit_idx,
            delta_idx,
            emit_exact,
            emit_pinned,
            emit_is_stop,
            deltas,
        )

    def _pmc_transitions(
        self,
        delta_idx,
        emit_pinned: List[bool],
        emit_is_stop: List[bool],
        n_bins: int,
        zero_bin: int,
        device,
    ) -> Tuple[Any, Any]:
        """
        Where each emission comes from on the mass axis, and whether it may.

        Returns ``(src_bins, valid)``, both of shape (emissions, bins).
        """
        # Per-token mass shifts as gather indices over the mass axis:
        # src_bins[e, m] = m - delta_e. A negative delta moves the source
        # ABOVE the target, so both ends of the axis have to be checked.
        bins = torch.arange(n_bins, device=device)
        raw_src = bins.unsqueeze(0) - delta_idx.unsqueeze(1)
        valid = (raw_src >= 0) & (raw_src < n_bins)
        src_bins = raw_src.clamp(0, n_bins - 1)
        # A pinned token may only be emitted before any residue mass has
        # accumulated. "Nothing emitted yet" is the zero-mass bin, so
        # requiring the source to be `zero_bin` means requiring the
        # target to be zero_bin + delta.
        pinned_only = torch.tensor(emit_pinned, device=device).unsqueeze(1)
        is_stop = torch.tensor(emit_is_stop, device=device).unsqueeze(1)
        from_open = bins.unsqueeze(0) == zero_bin + delta_idx.unsqueeze(1)
        # The stop steps one bin without carrying mass, so a path that
        # emitted one sits at zero_bin + 1 with nothing emitted yet. An
        # N-terminal modification opens the peptide after it, since the
        # frame flip puts the stop first, so it may start from either.
        # The stop itself may not: leaving it with the single source is
        # what stops a second one. Only a modification of exactly minus
        # one bin could step back and re-open that door, and the real
        # ones are +25.98, +42.01, -17.03 and +43.01 Da, thousands away.
        from_stopped = bins.unsqueeze(0) == zero_bin + 1 + delta_idx.unsqueeze(
            1
        )
        valid &= ~pinned_only | torch.where(
            is_stop, from_open, from_open | from_stopped
        )
        return src_bins, valid

    def _pmc_decode(
        self, logits: torch.Tensor, precursor_mass: float
    ) -> Optional[Tuple[List[int], List[float], Optional[float]]]:
        """
        Precise mass control (PMC) decoding for a single spectrum.

        Knapsack-like dynamic programming over the CTC lattice: find
        the highest-probability CTC path whose collapsed peptide's total
        residue mass matches the precursor mass within tolerance,
        considering all allowed isotope errors.

        The DP state is (discretized emitted mass, last path symbol);
        the last symbol is needed to apply the CTC collapse rules
        (a repeated symbol without an intervening blank does not emit).
        Each state also carries the exact residue mass of its best path,
        and the readout tests that rather than the bin, so rounding
        cannot make the search commit to a path the final mass check
        rejects. Each state keeps only its best path, though, so the
        search can miss a match: two paths that share a bin and a last
        symbol but differ in exact mass merge into the higher-scoring
        one, and if that one falls outside the window while the other
        is inside, the search returns no match and the greedy peptide is
        kept. Coarser bins, used for heavy precursors, make such merges
        more likely.

        Padding is excluded from the search, as is any residue of zero
        or negative mass that is not pinned (it could repeat without
        bound, which a fixed axis cannot hold). A pinned token may only
        be emitted before any residue mass has accumulated, which is
        also what bounds the axis below zero: only pinned tokens may
        carry a negative mass, so at most one such step is ever taken.
        N-terminal modifications are pinned because they open the
        peptide, and the stop because the frame flip puts it at the
        opening of this search's order; it is returned separately, since
        it is scored with the peptide but is not part of it. The stop
        also steps one bin without carrying any mass, so that emitting
        it moves the path out of reach of its own pin and no second one
        can follow; the readout tests the exact mass, not the bin, so
        the offset costs only a bin of headroom. As an
        approximation for speed, a token takes part in a
        frame only where its probability exceeds ``PMC_MIN_EMIT_PROB``:
        it cannot be emitted there, and a run of it cannot span that
        frame either, since the state is dropped rather than carried.
        Frames with no such token reduce to a blank-only update, which
        is most of them.

        The mass grid is discretized at ``PMC_RESOLUTION``, coarsened
        for heavy precursors so that the backtracking table stays under
        ``PMC_MAX_POINTER_BYTES``. Bin ``zero_bin`` holds zero mass, with
        the bins below it reserved for the negative excursion.

        Parameters
        ----------
        logits : torch.Tensor of shape (n_frames, n_tokens)
            The frame-level amino acid scores for one spectrum.
        precursor_mass : float
            The observed precursor neutral mass.

        Returns
        -------
        Optional[Tuple[List[int], List[float], Optional[float]]]
            The decoded token indices, their per-token confidences, and
            the stop's confidence if the path emitted one, or None if no
            mass-matching path exists. None also means the search was
            declined: the DP table would be too large, the vocabulary
            outgrew the int8 backtracking pointers, or the tokenizer is
            not reversed. The caller keeps the greedy peptide either way.
        """
        # MPS has no float64, which the exact masses need, so PMC runs
        # on the CPU there.
        if logits.device.type == "mps":
            logits = logits.cpu()
        device = logits.device
        n_frames, vocab = logits.shape
        # The stop is pinned to the zero-mass bin, which is where the frame
        # flip puts it for a reversed tokenizer: it closes that tokenizer's
        # order, so flipping opens this search's. Without the flip the stop
        # closes the frames instead, at full residue mass, and the pin
        # cannot place it there. `model_runner` always builds a reversed
        # tokenizer, so this only guards a directly constructed model.
        if not self.tokenizer.reverse:
            if not self._pmc_reverse_warned:
                self._pmc_reverse_warned = True
                logger.warning(
                    "Skipping precise mass control: it places the stop token "
                    "by flipping the frames, which needs a reversed "
                    "tokenizer."
                )
            return None
        # A reversed tokenizer emits the peptide C-terminus first, which
        # would put an N-terminal modification last. Decoding the frames
        # back to front puts it first, so the "N-terminal tokens open the
        # peptide" rule below is the only one needed; the emitted tokens
        # are flipped back before returning.
        logits = logits.flip(0)

        grid = self._pmc_grid(precursor_mass, n_frames, vocab)
        if grid is None:
            return None
        resolution, zero_bin, n_bins, hi_max, accept_windows = grid

        alphabet = self._pmc_alphabet(vocab, hi_max, resolution, device)
        if alphabet is None:
            return None
        (
            emit_idx,
            delta_idx,
            emit_exact,
            emit_pinned,
            emit_is_stop,
            deltas,
        ) = alphabet
        emit_i8 = emit_idx.unsqueeze(1).to(torch.int8)

        src_bins, valid = self._pmc_transitions(
            delta_idx, emit_pinned, emit_is_stop, n_bins, zero_bin, device
        )

        log_probs = logits.log_softmax(-1)
        neg_inf = float("-inf")
        blank = self.blank_token
        # score[m, c]: best log-probability of any path prefix with
        # discretized emitted mass m and last symbol c.
        score = torch.full((n_bins, vocab), neg_inf, device=device)
        score[zero_bin, blank] = 0.0
        # state_mass[m, c]: the exact residue mass of that same best
        # prefix, accumulated from the unrounded token masses. NaN marks
        # a state no path has reached; it compares false against every
        # window, so an unreachable state can never be read out.
        state_mass = torch.full(
            (n_bins, vocab), float("nan"), dtype=torch.float64, device=device
        )
        state_mass[zero_bin, blank] = 0.0
        # pointers[t, m, c]: last symbol of the predecessor state; a
        # pointer equal to c itself encodes a repeat (no new emission).
        # Kept on the model device; backtracking reads single entries.
        # int8 holds the symbol index, so the alphabet has to fit in it:
        # above 127 the cast would wrap and backtracking would follow the
        # wrong symbols without any error. Nothing checks the residue
        # dictionary against this bound, so a model trains normally and
        # only reaches it here. Decline the search rather than raise; the
        # caller then keeps the greedy peptide, as it does whenever no
        # mass-matching path exists.
        if vocab > 127:
            if not self._pmc_vocab_warned:
                self._pmc_vocab_warned = True
                logger.warning(
                    "Skipping precise mass control: it holds at most 127 "
                    "tokens, and this vocabulary has %d.",
                    vocab,
                )
            return None
        pointers = torch.empty(
            (n_frames, n_bins, vocab), dtype=torch.int8, device=device
        )
        # Tokens below PMC_MIN_EMIT_PROB at a frame drop out of that
        # frame entirely: they cannot be emitted, and an ongoing run of
        # one cannot continue through it, because the state is left at
        # -inf rather than carried forward. Frames with no candidate at
        # all reduce to a blank-only update, which is the usual case and
        # the reason this is affordable: measured 7 of 101 frames doing
        # emission work on a typical spectrum, 120 ms against 1,378 ms
        # with the threshold removed.
        min_lp = float(np.log(PMC_MIN_EMIT_PROB))
        candidates = log_probs[:, emit_idx] > min_lp  # (n_frames, E)
        for t in range(n_frames):
            lp = log_probs[t]
            sel = candidates[t].nonzero(as_tuple=True)[0]
            new_score = torch.full_like(score, neg_inf)
            new_ptr = torch.full(
                (n_bins, vocab), -1, dtype=torch.int8, device=device
            )
            new_mass = torch.full_like(state_mass, float("nan"))
            if sel.numel() == 0:
                # Blank-only frame: every state transitions to blank.
                best, best_arg = score.max(dim=1)
                new_score[:, blank] = best + lp[blank]
                new_ptr[:, blank] = best_arg.to(torch.int8)
                # A blank emits nothing, so the mass rides along.
                new_mass[:, blank] = state_mass.gather(
                    1, best_arg.unsqueeze(1)
                ).squeeze(1)
                score = new_score
                state_mass = new_mass
                pointers[t] = new_ptr
                continue
            top2 = score.topk(2, dim=1)
            best, best_arg = top2.values[:, 0], top2.indices[:, 0]
            second, second_arg = top2.values[:, 1], top2.indices[:, 1]
            # The blank column's predecessor mass; the emission columns
            # read theirs off `emit_arg` below, which already names the
            # predecessor the score chose.
            mass_best = state_mass.gather(1, best_arg.unsqueeze(1)).squeeze(1)
            sel_tokens = emit_idx[sel]
            # Best predecessor excluding each candidate token itself,
            # shifted by that token's mass: a new emission of token e at
            # mass m extends the best path at mass m - delta_e whose
            # last symbol differs from e.
            is_self = best_arg.unsqueeze(0) == sel_tokens.unsqueeze(1)
            excl_val = torch.where(
                is_self, second.unsqueeze(0), best.unsqueeze(0)
            )
            excl_arg = torch.where(
                is_self, second_arg.unsqueeze(0), best_arg.unsqueeze(0)
            )
            emit_val = excl_val.gather(1, src_bins[sel]) + lp[
                sel_tokens
            ].unsqueeze(1)
            emit_val = emit_val.masked_fill(~valid[sel], neg_inf)
            emit_arg = excl_arg.gather(1, src_bins[sel])
            # The exact mass follows the same predecessor the score
            # does, state (src_bins, emit_arg), gaining the unrounded
            # residue mass rather than its bin.
            emit_mass = state_mass.view(-1)[
                src_bins[sel] * vocab + emit_arg
            ] + emit_exact[sel].unsqueeze(1)
            # Continue the current run: no new emission.
            repeat = score[:, sel_tokens].T + lp[sel_tokens].unsqueeze(1)
            repeat_mass = state_mass[:, sel_tokens].T
            use_emit = emit_val > repeat
            emit_score = torch.where(use_emit, emit_val, repeat)
            emit_state_mass = torch.where(use_emit, emit_mass, repeat_mass)
            emit_ptr = torch.where(
                use_emit,
                emit_arg.to(torch.int8),
                emit_i8[sel].expand_as(use_emit),
            )
            # Blank keeps the mass and may follow any symbol.
            new_score[:, blank] = best + lp[blank]
            new_ptr[:, blank] = best_arg.to(torch.int8)
            new_mass[:, blank] = mass_best
            new_score[:, sel_tokens] = emit_score.T
            new_ptr[:, sel_tokens] = emit_ptr.T
            new_mass[:, sel_tokens] = emit_state_mass.T
            score = new_score
            state_mass = new_mass
            pointers[t] = new_ptr

        # Read out the best final state whose exact mass is in tolerance.
        allowed = torch.zeros_like(score, dtype=torch.bool)
        for lo, hi in accept_windows:
            allowed |= (state_mass >= lo) & (state_mass <= hi)
        score = score.masked_fill(~allowed, neg_inf)
        best_val, flat_idx = score.flatten().max(dim=0)
        if not torch.isfinite(best_val):
            return None
        m = int(flat_idx.item()) // vocab
        c = int(flat_idx.item()) % vocab

        # Backtrack, aggregating each emission's confidence as the
        # maximum probability over its merged frames.
        probs = log_probs.exp().cpu()
        emissions = []
        run_conf = None
        for t in reversed(range(n_frames)):
            prev = int(pointers[t, m, c])
            if c != blank:
                prob = probs[t, c].item()
                run_conf = prob if run_conf is None else max(run_conf, prob)
                if prev != c:
                    # Frame t started this emission of c.
                    emissions.append((c, run_conf))
                    run_conf = None
                    m -= deltas[c]
            c = prev

        # Undo the frame flip so the caller still receives the tokens in
        # the tokenizer's own order. The guard above left only the
        # reversed case, so this is unconditional.
        tokens = [c for c, _ in emissions]
        confs = [s for _, s in emissions]
        # The stop is scored like the peptide's other emissions but is
        # not part of it, so it leaves here on its own, as it does from
        # `_ctc_decode`. The pin allows only one, and taking the best
        # keeps the readout total if that ever stops holding.
        stop_conf = max(
            (s for c, s in zip(tokens, confs) if c == self.stop_token),
            default=None,
        )
        if stop_conf is not None:
            keep = [c != self.stop_token for c in tokens]
            tokens = [c for c, k in zip(tokens, keep) if k]
            confs = [s for s, k in zip(confs, keep) if k]
        # Agrees with the readout by construction; kept as the guard on
        # an empty decode.
        if not tokens or not self._fits_precursor_mass(tokens, precursor_mass):
            return None
        return tokens, confs, stop_conf

    def predict_step(
        self, batch: Dict[str, torch.Tensor], *args
    ) -> List[psm.PepSpecMatch]:
        """
        A single prediction step: greedy CTC decoding, then precise mass
        control when the greedy peptide misses the precursor window.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, containing keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``,
            ``precursor_charge``, plus metadata like ``peak_file`` and
            ``scan_id``.

        Returns
        -------
        predictions : List[psm.PepSpecMatch]
            Predicted PSMs for the given batch of spectra.
        """
        logits, _ = self._forward_step(batch)
        sequences, scores, stop_scores = self._ctc_decode(logits)

        # Precise mass control: when the greedy peptide does not match
        # the precursor mass, search for the best CTC path that does.
        # If PMC finds none, the greedy peptide is kept, unpenalized
        # (_peptide_score).
        _, _, precursors, _ = self._process_batch(batch)
        for i, tokens in enumerate(sequences):
            precursor_mass = precursors[i, 0].item()
            fit = bool(tokens) and self._fits_precursor_mass(
                tokens, precursor_mass
            )
            if not fit:
                pmc = self._pmc_decode(logits[i], precursor_mass)
                # PMC only returns mass-matching paths. A correction
                # shorter than min_peptide_len would be dropped below,
                # so keep the greedy peptide instead.
                if pmc is not None and len(pmc[0]) >= self.min_peptide_len:
                    sequences[i], scores[i], stop_scores[i] = pmc

        predictions = []
        for filename, scan, charge, prec_mz, tokens, confs, stop in zip(
            batch["peak_file"],
            batch["scan_id"],
            batch["precursor_charge"],
            batch["precursor_mz"],
            sequences,
            scores,
            stop_scores,
        ):
            # Beam search withheld these; CTC decoding replaced it
            # without replacing the rule.
            if not tokens or len(tokens) < self.min_peptide_len:
                self.n_missing_predictions += 1
                continue

            peptide = "".join(
                self.tokenizer.detokenize(torch.tensor([tokens]), join=False)[
                    0
                ]
            )
            aa_scores = np.array(confs)
            if self.tokenizer.reverse:
                aa_scores = aa_scores[::-1]

            # The peptide score is the product of the residue scores
            # (see _peptide_score). The product also survives the
            # N-terminal score merge in `on_predict_batch_end`, which a
            # mean does not. The stop is scored with the peptide, as
            # upstream counts it, but is not reported among the residues.
            scored = aa_scores
            if stop is not None:
                scored = np.append(aa_scores, stop)
            peptide_score = float(_peptide_score(scored))

            predictions.append(
                psm.PepSpecMatch(
                    sequence=peptide,
                    spectrum_id=(filename, scan),
                    peptide_score=peptide_score,
                    charge=int(charge),
                    calc_mz=np.nan,
                    exp_mz=float(prec_mz.item()),
                    aa_scores=aa_scores,
                )
            )

        return predictions

    def on_train_epoch_end(self) -> None:
        """
        Log the training loss at the end of each epoch.
        """
        if "train_CTCLoss" in self.trainer.callback_metrics:
            train_loss = (
                self.trainer.callback_metrics["train_CTCLoss"].detach().item()
            )
        else:
            train_loss = np.nan
        metrics = {"step": self.trainer.global_step, "train": train_loss}
        self._history.append(metrics)
        self._log_history()

    def on_validation_epoch_end(self) -> None:
        """
        Log the validation metrics at the end of each epoch.
        """
        callback_metrics = self.trainer.callback_metrics
        metrics = {"step": self.trainer.global_step}

        if "valid_CTCLoss" in callback_metrics:
            metrics["valid"] = (
                callback_metrics["valid_CTCLoss"].detach().item()
            )

        for stem in self.val_stems:
            key = f"valid_CTCLoss/{stem}"
            if key in callback_metrics:
                metrics[f"valid/{stem}"] = (
                    callback_metrics[key].detach().item()
                )

        if self.calculate_precision:
            if "aa_precision" in callback_metrics:
                metrics["valid_aa_precision"] = (
                    callback_metrics["aa_precision"].detach().item()
                )
            if "pep_precision" in callback_metrics:
                metrics["valid_pep_precision"] = (
                    callback_metrics["pep_precision"].detach().item()
                )
        self._history.append(metrics)
        self._log_history()

    def on_predict_start(self) -> None:
        """Reset the count of spectra without a prediction."""
        self.n_missing_predictions = 0

    def on_predict_epoch_end(self) -> None:
        """Aggregate the missing-prediction count across devices."""
        self.n_missing_predictions = int(
            self.all_gather(
                torch.tensor(self.n_missing_predictions, device=self.device)
            ).sum()
        )

    def on_predict_batch_end(
        self, outputs: List[psm.PepSpecMatch], *args
    ) -> None:
        """
        Write the predicted PSMs to the output file.

        Parameters
        ----------
        outputs : List[psm.PepSpecMatch]
            The predicted PSMs for the processed batch.
        """
        if self.out_writer is None:
            return

        for spec_match in outputs:
            if not spec_match.sequence:
                continue

            # N terminal scores should be combined with first token
            if len(spec_match.aa_scores) >= 2 and any(
                spec_match.sequence.startswith(mod) for mod in self.n_term
            ):
                spec_match.aa_scores[1] *= spec_match.aa_scores[0]
                spec_match.aa_scores = spec_match.aa_scores[1:]

            # Compute the precursor m/z if not already set (e.g. from the
            # peptide database in DB search mode).
            if np.isnan(spec_match.calc_mz):
                spec_match.calc_mz = self.tokenizer.calculate_precursor_ions(
                    spec_match.sequence,
                    torch.tensor(spec_match.charge),
                ).item()

            self.out_writer.psms.append(spec_match)

    def on_train_start(self):
        """Log optimizer settings."""
        self.log("hp/optimizer_warmup_iters", self.warmup_iters)
        self.log(
            "hp/optimizer_cosine_schedule_period_iters",
            self.cosine_schedule_period_iters,
        )

    def _log_history(self) -> None:
        """
        Write log to console, if requested.
        """
        # Log only if all output for the current epoch is recorded.
        if len(self._history) == 0:
            return
        if len(self._history) == 1:
            header = "Step\tTrain loss\tValid loss\t"
            if self.calculate_precision:
                header += "Peptide precision\tAA precision"

            logger.info(header)
        metrics = self._history[-1]
        if metrics["step"] % self.n_log == 0:
            msg = "%i\t%.6f\t%.6f"
            vals = [
                metrics["step"],
                metrics.get("train", np.nan),
                metrics.get("valid", np.nan),
            ]

            if self.calculate_precision:
                msg += "\t%.6f\t%.6f"
                vals += [
                    metrics.get("valid_pep_precision", np.nan),
                    metrics.get("valid_aa_precision", np.nan),
                ]

            logger.info(msg, *vals)

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], Dict[str, Any]]:
        """
        Initialize the optimizer.

        Hidden weight matrices are optimized with Muon; embeddings, the
        output head, and all vector/scalar parameters use an auxiliary
        AdamW group, per the Muon usage guidance. A single cosine
        learning rate scheduler rescales both groups' base learning
        rates by the same warmup/decay factor.

        Returns
        -------
        Tuple[List[torch.optim.Optimizer], Dict[str, Any]]
            The initialized optimizer and its learning rate scheduler.
        """
        aux_ids = set()
        for module in self.modules():
            if isinstance(module, torch.nn.Embedding):
                aux_ids.update(id(p) for p in module.parameters())
        aux_ids.update(id(p) for p in self.decoder.final.parameters())
        muon_params, aux_params = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            # Squeeze first: ndim alone calls the encoder's spectrum
            # token, shaped (1, 1, d_model), a matrix, and Muon would
            # normalize its update to a size the gradient cannot affect.
            if p.squeeze().ndim >= 2 and id(p) not in aux_ids:
                muon_params.append(p)
            else:
                aux_params.append(p)
        optimizer = MuonWithAuxAdamW(
            [
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=self.muon_lr,
                    momentum=self.muon_momentum,
                    weight_decay=self.opt_kwargs.get("weight_decay", 0.0),
                ),
                dict(params=aux_params, use_muon=False, **self.opt_kwargs),
            ]
        )
        # Apply learning rate scheduler per step.
        lr_scheduler = CosineWarmupScheduler(
            optimizer, self.warmup_iters, self.cosine_schedule_period_iters
        )
        return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}


class DbSpec2Pep(Spec2Pep):
    """
    Subclass of Spec2Pep for the use of Casanovo as an MS/MS database
    search score function.

    Uses teacher forcing to 'query' Casanovo to score a peptide-spectrum
    pair. Note that this does *not* involve training, but rather that
    teacher forcing is used for predicting.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        The forward step.

        Score candidate peptides against an encoder output the batch
        already carries. Each spectrum is encoded once and all of its
        candidates are scored against that, so nothing here encodes.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A candidate batch from ``_psm_batches``, carrying the
            candidate tokens in ``seq`` and the encoder outputs
            ``memory``, ``mem_masks``, and ``precursors``.

        Returns
        -------
        scores : torch.Tensor of shape (B, length, n_amino_acids)
            The individual amino acid scores for each prediction,
            converted to probabilities using a softmax.
        tokens : torch.Tensor of shape (B, length)
            The ground truth tokens for each spectrum.

        Notes
        -----
        Here ``B`` denotes the number of peptide–spectrum pairs in the
        current candidate batch (or the number of spectra for a plain
        forward pass).
        """
        if (
            "memory" in batch
            and "mem_masks" in batch
            and "precursors" in batch
        ):
            memories, mem_masks = batch["memory"], batch["mem_masks"]
            precursors = batch["precursors"]
            tokens = batch["seq"]
            # These are real peptides padded to the longest candidate in
            # the batch, unlike the de novo frames, so the padding has to
            # be masked or a candidate's score depends on what it was
            # batched with.
            logits = self.decoder(
                tokens=tokens,
                memory=memories,
                memory_key_padding_mask=mem_masks,
                precursors=precursors,
                tgt_key_padding_mask=tokens
                == self.decoder.token_encoder.padding_idx,
            )
            probs = self.softmax(logits)
            return probs, tokens
        # Upstream fell back to `_forward_step` here, which was equivalent
        # because it fed the candidate to the decoder and differed only in
        # encoding the spectrum again. This decoder takes no peptide: it
        # decodes blank frames that CTC aligns to the target afterwards, so
        # `_forward_step` returns frame logits that say nothing about the
        # candidate, and the same ones for every candidate of a spectrum.
        # There is no scoring this way, so say so rather than return them.
        raise ValueError(
            "Database search scores candidates against a cached encoder "
            "output; this batch carries none. `_psm_batches` attaches it."
        )

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        *args,
    ) -> List[ms_io.PepSpecMatch]:
        """
        A single prediction step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, which contains keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``, and
            ``precursor_charge``, each pointing to tensors with the
            corresponding data.

        Returns
        -------
        predictions: List[ms_io.PepSpecMatch]
            The predicted PSMs for the processed batch.
        """
        predictions = collections.defaultdict(list)

        with torch.inference_mode():
            # Pre-compute encoder outputs for the entire batch.
            mzs, intensities, precursors_all, _ = self._process_batch(batch)
            memories, mem_masks = self.encoder(mzs, intensities)
            enc_cache = {
                "memory": memories,
                "mem_masks": mem_masks,
                "precursors_all": precursors_all,
            }

            for psm_batch in self._psm_batches(batch, enc_cache=enc_cache):
                pred_logits, truth = self.forward(psm_batch)
                peptide_scores, aa_scores_all = _calc_match_score(
                    pred_logits, truth
                )

                for (
                    filename,
                    scan,
                    precursor_charge,
                    precursor_mz,
                    peptide,
                    peptide_score,
                    curr_aa_scores,
                ) in zip(
                    psm_batch["peak_file"],
                    psm_batch["scan_id"],
                    psm_batch["precursor_charge"],
                    psm_batch["precursor_mz"],
                    psm_batch["original_seq_str"],
                    peptide_scores,
                    aa_scores_all,
                ):
                    # Omit stop token from reported AA scores.
                    curr_aa_scores = curr_aa_scores[:-1]
                    if self.tokenizer.reverse:
                        curr_aa_scores = curr_aa_scores[::-1]

                    spectrum_id = (filename, scan)
                    predictions[spectrum_id].append(
                        psm.PepSpecMatch(
                            sequence=peptide,
                            spectrum_id=spectrum_id,
                            peptide_score=peptide_score,
                            charge=int(precursor_charge),
                            calc_mz=np.nan,
                            exp_mz=precursor_mz.item(),
                            aa_scores=curr_aa_scores,
                        )
                    )

        # Filter the top-scoring prediction for each spectrum.
        predictions = list(
            itertools.chain.from_iterable(
                sorted(
                    spectrum_predictions,
                    key=lambda p: p.peptide_score,
                    reverse=True,
                )[: self.top_match]
                for spectrum_predictions in predictions.values()
            )
        )

        # Determine the parent proteins and calc_mz only for the retained PSMs.
        for pred in predictions:
            pred.protein = self.protein_database.get_associated_protein(
                pred.sequence
            )
            calc_mass = self.protein_database.db_peptides.loc[
                pred.sequence, "calc_mass"
            ]
            pred.calc_mz = float(calc_mass) / pred.charge + PROTON

        return predictions

    def _psm_batches(
        self,
        batch: Dict[str, torch.Tensor],
        enc_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Generates batches of candidate database PSMs.

        PSM batches consist of repeated spectrum information for each
        candidate peptide to be scored against each spectrum.
        This method ensures that the batches provided to the model
        are of a consistent size.

        FIXME: Move this logic to a subclassed DataLoader.
         This would also allow correctly setting the batch size (now the
         final batch will be (much) smaller depending on how many
         spectra remain).

        TODO: The batch creation and generation could potentially be
         improved using a producer-consumer pattern.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, which contains keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``, and
            ``precursor_charge``, each pointing to tensors with the
            corresponding data.
        enc_cache : Optional[Dict[str, torch.Tensor]]
            Optional cache of encoder outputs (``memory``, ``mem_masks``,
            and ``precursors_all``) to avoid re-computation.

        Yields
        ------
        Dict[str, torch.Tensor]
            Batches of candidate database PSMs ready for scoring. Each batch
            contains repeated spectrum information for each candidate peptide
            to be scored against each spectrum.
        """
        device = self.decoder.device
        batch_size = batch["precursor_charge"].shape[0]

        # Iterate precursor charges and m/z values per spectrum.
        charge_iter = batch["precursor_charge"].detach().cpu().tolist()
        mz_iter = batch["precursor_mz"].detach().cpu().tolist()

        # Use pre-computed encoder outputs if available; otherwise compute once here.
        if enc_cache is None:
            mzs, ints, precursors_all, _ = self._process_batch(batch)
            memories, mem_masks = self.encoder(mzs, ints)
        else:
            memories, mem_masks = enc_cache["memory"], enc_cache["mem_masks"]
            precursors_all = enc_cache["precursors_all"]

        # Determine the candidates to score for each spectrum and
        # compile them into new batches with the same size as the original batch.
        candidates = []
        for i, (precursor_charge, precursor_mz) in enumerate(
            zip(charge_iter, mz_iter)
        ):
            spec_cands = self.protein_database.get_candidates(
                precursor_mz, precursor_charge
            )
            candidates.extend((i, cand) for cand in spec_cands)

        if len(candidates) == 0:
            return

        # Yield PSM sub-batches with a progress bar.
        progress = tqdm.tqdm(
            total=len(candidates),
            desc="Scoring candidates",
            unit="PSM",
            leave=False,
        )
        try:
            for start in range(0, len(candidates), batch_size):
                batch_candidates = candidates[start : start + batch_size]

                # Repeat the spectrum information for each candidate to be matched.
                psm_batch = {key: [] for key in [*batch.keys(), "seq"]}
                for spec_i, cand in batch_candidates:
                    for key in batch:
                        psm_batch[key].append(batch[key][spec_i])
                    psm_batch["seq"].append(cand)

                # Convert tensor items to batched tensors on the correct device.
                for key in psm_batch:
                    if isinstance(psm_batch[key][0], torch.Tensor):
                        psm_batch[key] = torch.stack(psm_batch[key]).to(
                            self.decoder.device
                        )

                # Keep the original sequence string for downstream database lookup
                # (e.g., isoleucine ↔ leucine handling) and tokenize for scoring.
                psm_batch["original_seq_str"] = psm_batch["seq"]
                psm_batch["seq"] = self.tokenizer.tokenize(
                    psm_batch["seq"], add_stop=True
                ).to(self.decoder.device)

                # Attach the corresponding (pre)computed encoder outputs for these spectra.
                spec_idx = torch.tensor(
                    [i for i, _ in batch_candidates],
                    dtype=torch.long,
                    device=device,
                )
                psm_batch["memory"] = memories.index_select(0, spec_idx)
                psm_batch["mem_masks"] = mem_masks.index_select(0, spec_idx)
                psm_batch["precursors"] = precursors_all.index_select(
                    0, spec_idx
                )
                # Yield the PSM batch for processing.
                yield psm_batch

                progress.update(len(batch_candidates))
        finally:
            progress.close()


def _ctc_log_probs(scores: torch.Tensor) -> torch.Tensor:
    """
    Frame log-probabilities in the layout the CTC loss takes.

    Half-precision scores are promoted to float32 first: the CTC loss
    kernels take float32 or float64 only, and a log-softmax in half
    precision loses the small probabilities CTC sums over.

    Parameters
    ----------
    scores : torch.Tensor of shape (batch, n_frames, n_tokens)
        The frame-level scores.

    Returns
    -------
    torch.Tensor of shape (n_frames, batch, n_tokens)
        The log-probabilities, in float32 or float64.
    """
    if scores.dtype in (torch.float16, torch.bfloat16):
        scores = scores.float()
    return scores.log_softmax(-1).transpose(0, 1)


def _calc_match_score(
    batch_all_aa_scores: torch.Tensor,
    truth_aa_indices: torch.Tensor,
) -> Tuple[List[float], List[np.ndarray]]:
    """
    Calculate the score between the input spectra and associated
    peptide.

    This function now acts as a wrapper that prepares data for the unified
    _peptide_score function.

    Parameters
    ----------
    batch_all_aa_scores : torch.Tensor
        Amino acid scores for all amino acids in the vocabulary for
        every prediction made to generate the associated peptide (for an
        entire batch).
    truth_aa_indices : torch.Tensor
        Indices of the score for each actual amino acid in the peptide
        (for an entire batch).

    Returns
    -------
    peptide_scores: List[float]
        The peptide score for each PSM in the batch.
    aa_scores : List[np.ndarray]
        The amino acid scores for each PSM in the batch.
    """
    # Remove trailing token.
    batch_all_aa_scores = batch_all_aa_scores[:, :-1]

    # Get aa scores corresponding with true aas.
    per_aa_scores = torch.gather(
        batch_all_aa_scores, 2, truth_aa_indices.unsqueeze(-1)
    ).squeeze(-1)

    # Calculate peptide lengths.
    lengths = (truth_aa_indices != 0).sum(dim=1)

    # Fuse scores and lengths for a single GPU->CPU transfer.
    fused = torch.cat(
        [per_aa_scores, lengths.to(per_aa_scores.dtype).unsqueeze(1)], dim=1
    )
    fused_np = fused.detach().cpu().numpy()

    # Unpack scores and lengths on the CPU.
    per_aa_np = fused_np[:, :-1]
    lengths_np = fused_np[:, -1].astype(np.int32, copy=False)

    # Call the single, unified scoring function for batch calculation.
    peptide_scores = _peptide_score(per_aa_np, lengths=lengths_np).tolist()

    # Extract AA scores for each peptide based on its length.
    B = per_aa_np.shape[0]
    aa_scores = [per_aa_np[i, : lengths_np[i]] for i in range(B)]

    return peptide_scores, aa_scores


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm-up followed by cosine
    shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup_iters : int
        The number of iterations for the linear warm-up of the learning
        rate.
    cosine_schedule_period_iters : int
        The number of iterations for the cosine half period of the
        learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        cosine_schedule_period_iters: int,
    ):
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (
            1 + np.cos(np.pi * epoch / self.cosine_schedule_period_iters)
        )
        if epoch <= self.warmup_iters:
            lr_factor *= epoch / self.warmup_iters
        return lr_factor


def _peptide_score(
    aa_scores: Union[np.ndarray, torch.Tensor],
    lengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Calculate the peptide-level confidence score from the raw
    amino acid scores.

    The peptide score is the product of the raw amino acid scores.
    This function accepts both NumPy arrays and PyTorch tensors.
    NumPy inputs are converted to tensors for computation (zero-copy
    for contiguous CPU arrays) and the result is returned in the
    original type.

    Parameters
    ----------
    aa_scores : np.ndarray or torch.Tensor
        A 1D array of amino acid scores for a single peptide, or a 2D
        padded array for a batch of peptides.
    lengths : Optional[np.ndarray or torch.Tensor]
        An array of peptide lengths, required when `aa_scores` is a 2D
        (batched) array.

    Returns
    -------
    peptide_score : float, np.ndarray, or torch.Tensor
        The calculated peptide score or an array of scores for the batch.
    """
    # Track whether the input was numpy to return the appropriate type.
    return_numpy = isinstance(aa_scores, np.ndarray)

    # Convert numpy arrays to tensors without copying data (zero-copy for
    # contiguous CPU arrays), enabling a single unified computation path.
    if return_numpy:
        # Reversed views (negative strides) cannot be wrapped as tensors;
        # ascontiguousarray copies only those.
        aa_scores = torch.as_tensor(np.ascontiguousarray(aa_scores))

    eps = torch.finfo(torch.float64).eps
    log_scores = torch.log(torch.clamp(aa_scores, eps, 1))

    if aa_scores.ndim == 1:
        # FAST PATH: de novo inference — single peptide.
        peptide_score = torch.exp(torch.sum(log_scores))
        return peptide_score.item() if return_numpy else peptide_score

    # BATCH PATH: database search — padded batch of peptides.
    if lengths is None:
        raise ValueError("`lengths` must be provided for batched input.")
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(
            lengths, dtype=torch.long, device=aa_scores.device
        )
    else:
        lengths = lengths.to(dtype=torch.long, device=aa_scores.device)
    cumsum = torch.cumsum(log_scores, dim=1)
    batch_size = aa_scores.shape[0]
    idx = torch.arange(batch_size, device=aa_scores.device)
    peptide_scores = torch.exp(cumsum[idx, torch.clamp(lengths - 1, min=0)])
    return peptide_scores.numpy() if return_numpy else peptide_scores
