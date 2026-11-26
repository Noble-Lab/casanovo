"""A de novo peptide sequencing model."""

import collections
import heapq
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

import einops
import lightning.pytorch as pl
import numpy as np
import torch
from depthcharge.tokenizers import PeptideTokenizer

from .. import config
from ..data import ms_io, psm
from ..denovo.transformers import PeptideDecoder, SpectrumEncoder
from . import evaluate

logger = logging.getLogger("casanovo")


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
        The maximum allowable precursor mass tolerance (in ppm) for
        correct predictions.
    isotope_error_range : Tuple[int, int]
        Take into account the error introduced by choosing a
        non-monoisotopic peak for fragmentation by not penalizing
        predicted precursor m/z's that fit the specified isotope error:
        `abs(calc_mz - (precursor_mz - isotope * 1.00335 / precursor_charge))
        < precursor_mass_tol`
    min_peptide_len : int
        The minimum length of predicted peptides.
    n_beams : int
        Number of beams used during beam search decoding.
    top_match : int
        Number of PSMs to return for each spectrum.
    n_log : int
        The number of epochs to wait between logging messages.
    train_label_smoothing : float
        Unused; retained for configuration compatibility. The CTC loss
        does not support label smoothing.
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
    **kwargs : Dict
        Additional keyword arguments passed to the Adam optimizer.
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
        isotope_error_range: Tuple[int, int] = (0, 1),
        min_peptide_len: int = 6,
        n_beams: int = 1,
        top_match: int = 1,
        n_log: int = 10,
        train_label_smoothing: float = 0.01,
        warmup_iters: int = 100_000,
        cosine_schedule_period_iters: int = 600_000,
        inter_ctc_layers: Sequence[int] = (),
        inter_ctc_weight: float = 0.5,
        out_writer: Optional[ms_io.MztabWriter] = None,
        calculate_precision: bool = False,
        tokenizer: PeptideTokenizer | None = None,
        **kwargs: Dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = tokenizer or PeptideTokenizer()
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
        # Optimizer settings.
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        # `kwargs` will contain additional arguments as well as
        # unrecognized arguments, including deprecated ones. Remove the
        # deprecated ones.
        for k in config._config_deprecated:
            kwargs.pop(k, None)
            warnings.warn(
                f"Deprecated hyperparameter '{k}' removed from the model.",
                DeprecationWarning,
            )
        self.opt_kwargs = kwargs

        # Data properties.
        self.decoder_frames = decoder_frames
        self.residues = residues
        self.precursor_mass_tol = precursor_mass_tol
        self.isotope_error_range = isotope_error_range
        self.min_peptide_len = min_peptide_len
        self.n_beams = n_beams
        self.top_match = top_match
        self.stop_token = self.tokenizer.stop_int

        # Logging.
        self.calculate_precision = calculate_precision
        self.n_log = n_log
        self._history = []

        # Output writer during predicting.
        self.out_writer = out_writer

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
            ),
            persistent=False,
        )

        # Register tensor buffer for N-terminal modification indices
        self.register_buffer(
            "nterm_idx",
            torch.tensor(
                [self.tokenizer.index[aa] for aa in self.n_term],
                dtype=torch.int,
            ),
            persistent=False,
        )

        # Register tensor buffer for amino acid token masses
        self.register_buffer(
            "token_masses",
            torch.zeros(self.vocab_size, dtype=torch.float64),
            persistent=False,
        )
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
        if self.inter_ctc_layers or return_intermediates:
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

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        *args,
        mode: str = "train",
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
        mode : str
            Logging key to describe the current stage.

        Returns
        -------
        torch.Tensor
            The loss of the training step.
        """

        pred, truth, intermediates = self._forward_step(
            batch, return_intermediates=True
        )

        log_probs = pred.log_softmax(-1).transpose(0, 1)  # (T, B, V)
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
        loss = self.ctc_loss(log_probs, truth, input_lengths, target_lengths)
        # The final layer's CTC loss is what gets logged as `*_CTCLoss`, so it
        # stays comparable with runs that predate intermediate CTC and keeps
        # driving best-checkpoint selection. The combined objective below is
        # what the optimizer sees; logging that instead would mix in the
        # shallower layers' worse predictions and both compare wrongly against
        # earlier curves and pick checkpoints on the wrong quantity.
        self.log(
            f"{mode}_CTCLoss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=truth.shape[0],
        )
        if intermediates:
            # Intermediate CTC: the same objective on each intermediate
            # layer's own prediction, so the layers below the last one are
            # supervised directly rather than only through it.
            aux = torch.stack(
                [
                    self.ctc_loss(
                        scores.log_softmax(-1).transpose(0, 1),
                        truth,
                        input_lengths,
                        target_lengths,
                    )
                    for scores in intermediates
                ]
            ).mean()
            loss = (1 - self.inter_ctc_weight) * loss + (
                self.inter_ctc_weight * aux
            )
            for key, value in (
                (f"{mode}_CTCLoss_intermediate", aux),
                (f"{mode}_CTCLoss_total", loss),
            ):
                self.log(
                    key,
                    value.detach(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=truth.shape[0],
                )
        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], *args
    ) -> torch.Tensor:
        """
        A single validation step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, which contains keys:
            A batch from the SpectrumDataset, which contains keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``, and
            ``precursor_charge``, each pointing to tensors with the
            corresponding data. The ``seq`` key is optional and
            contains the peptide sequences for training.

        Returns
        -------
        torch.Tensor
            The loss of the validation step.
        """
        # Record the loss.
        loss = self.training_step(batch, mode="valid")
        if not self.calculate_precision:
            return loss

        # Calculate and log amino acid and peptide match evaluation
        # metrics from the predicted peptides.
        # FIXME: Remove work around when depthcharge reverse detokenization
        # bug is fixed.
        # peptides_true = self.tokenizer.detokenize(batch["seq"])
        peptides_true = [
            "".join(pep)
            for pep in self.tokenizer.detokenize(batch["seq"], join=False)
        ]
        logits, _ = self.forward(batch)
        peptides_pred = [
            (
                "".join(
                    self.tokenizer.detokenize(
                        torch.tensor([tokens]), join=False
                    )[0]
                )
                if tokens
                else ""
            )
            for tokens in self._ctc_decode(logits)[0]
        ]
        aa_precision, _, pep_precision = evaluate.aa_match_metrics(
            *evaluate.aa_match_batch(
                peptides_true, peptides_pred, self.tokenizer.residues
            )
        )

        batch_size = len(peptides_true)
        log_args = dict(on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "pep_precision", pep_precision, **log_args, batch_size=batch_size
        )
        self.log(
            "aa_precision", aa_precision, **log_args, batch_size=batch_size
        )
        return loss

    def _ctc_decode(
        self, logits: torch.Tensor
    ) -> Tuple[
        List[List[int]], List[List[float]], List[Optional[float]]
    ]:
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

    def predict_step(
        self, batch: Dict[str, torch.Tensor], *args
    ) -> List[psm.PepSpecMatch]:
        """
        A single prediction step: greedy CTC decoding.

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
                continue

            peptide = "".join(
                self.tokenizer.detokenize(torch.tensor([tokens]), join=False)[
                    0
                ]
            )
            aa_scores = np.array(confs)
            if self.tokenizer.reverse:
                aa_scores = aa_scores[::-1]

            # Upstream's aggregation, the product of the residue scores,
            # without its 1 subtracted from a peptide that misses the
            # precursor (see _peptide_score). The
            # product also survives the N-terminal score merge in
            # `on_predict_batch_end`, which a mean does not. The stop is
            # scored with the peptide, as upstream counts it, but is not
            # reported among the residues.
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
        metrics = {
            "step": self.trainer.global_step,
            "valid": callback_metrics["valid_CTCLoss"].detach().item(),
        }

        if self.calculate_precision:
            metrics["valid_aa_precision"] = (
                callback_metrics["aa_precision"].detach().item()
            )
            metrics["valid_pep_precision"] = (
                callback_metrics["pep_precision"].detach().item()
            )
        self._history.append(metrics)
        self._log_history()

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

            # Compute the precursor m/z of the predicted peptide.
            spec_match.calc_mz = self.tokenizer.calculate_precursor_ions(
                spec_match.sequence, torch.tensor(spec_match.charge)
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

        We use the Adam optimizer with a cosine learning rate scheduler.

        Returns
        -------
        Tuple[List[torch.optim.Optimizer], Dict[str, Any]]
            The initialized Adam optimizer and its learning rate
            scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), **self.opt_kwargs)
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

        # Determine the parent proteins only for the retained PSMs.
        for pred in predictions:
            pred.protein = self.protein_database.get_associated_protein(
                pred.sequence
            )

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
        charge_iter = batch["precursor_charge"]  # tensor[B]
        mz_iter = batch["precursor_mz"]  # tensor[B]

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
            for cand in self.protein_database.get_candidates(
                precursor_mz, precursor_charge
            ):
                candidates.append((i, cand))

            # Yield a batch if sufficient candidates are found or all spectra have been processed.
            while len(candidates) >= batch_size or (
                i == batch_size - 1 and len(candidates) > 0
            ):
                batch_candidates = candidates[:batch_size]

                # Repeat the spectrum information for each candidate to be matched.
                psm_batch = {key: [] for key in [*batch.keys(), "seq"]}
                for spec_i, cand in batch_candidates:
                    for key in batch.keys():
                        psm_batch[key].append(batch[key][spec_i])
                    psm_batch["seq"].append(cand)

                # Convert tensor items to batched tensors on the correct device.
                for key in psm_batch.keys():
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

                # Remove the processed candidates and continue.
                candidates = candidates[batch_size:]


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


def _calc_mass_error(
    calc_mz: float, obs_mz: float, charge: int, isotope: int = 0
) -> float:
    """
    Calculate the mass error in ppm between the theoretical m/z and the
    observed m/z, optionally accounting for an isotopologue mismatch.

    Parameters
    ----------
    calc_mz : float
        The theoretical m/z.
    obs_mz : float
        The observed m/z.
    charge : int
        The charge.
    isotope : int
        Correct for the given number of C13 isotopes (default: 0).

    Returns
    -------
    float
        The mass error in ppm.
    """
    return (calc_mz - (obs_mz - isotope * 1.00335 / charge)) / obs_mz * 10**6


def _peptide_score(
    aa_scores: np.ndarray,
    lengths: Optional[np.ndarray] = None,
) -> Union[float, np.ndarray]:
    """
    Calculate the peptide-level confidence score from the raw
    amino acid scores.

    The peptide score is the product of the raw amino acid scores.
    This function contains paths for both single peptide inputs
    (de novo mode) and batched peptide inputs (database search mode).

    Unlike upstream Casanovo, a peptide whose mass misses the precursor
    does not have 1 subtracted from its score: on chimeric spectra such
    a prediction is often the co-isolated peptide, a real peptide whose
    mass misses the recorded precursor, and the penalty would rank it
    below every mass-matching prediction.

    Parameters
    ----------
    aa_scores : np.ndarray
        A 1D array of amino acid scores for a single peptide, or a 2D
        padded array for a batch of peptides.
    lengths : Optional[np.ndarray]
        An array of peptide lengths, required when `aa_scores` is a 2D
        (batched) array.

    Returns
    -------
    peptide_score : float or np.ndarray
        The calculated peptide score or an array of scores for the batch.
    """
    eps = np.finfo(np.float64).eps

    # FAST PATH: de novo inference
    if aa_scores.ndim == 1:
        log_scores = np.log(np.clip(aa_scores, eps, 1))
        peptide_log_score = np.sum(log_scores)
        return np.exp(peptide_log_score)

    # BATCH PATH: database search
    else:
        if lengths is None:
            raise ValueError("`lengths` must be provided for batched input.")

        log_scores = np.log(np.clip(aa_scores, eps, 1))
        cumsum = np.cumsum(log_scores, axis=1)
        batch_size = aa_scores.shape[0]
        idx = np.arange(batch_size)
        peptide_log_scores = cumsum[idx, np.maximum(lengths - 1, 0)]
        return np.exp(peptide_log_scores)
