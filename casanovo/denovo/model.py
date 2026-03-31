"""A de novo peptide sequencing model."""

import collections
import heapq
import itertools
import logging
import warnings
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import einops
import lightning.pytorch as pl
import numpy as np
import torch
from depthcharge.tokenizers import PeptideTokenizer

from .chimera import ChimeraTokenizer
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
    max_peptide_len : int
        The maximum peptide length to decode.
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
        Smoothing factor when calculating the training loss.
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
    chimera : bool
        If ``True``, enable chimeric spectrum sequencing.  The tokenizer must
        be a :class:`~casanovo.denovo.chimera.ChimeraTokenizer`, the training
        data must contain ``"seq_compliment"`` batches (provided automatically
        by :class:`~casanovo.denovo.chimera.ChimeraAnnotatedSpectrumDataset`),
        and the loss is computed as a hard-minimum over the two possible
        peptide orderings.  During prediction, the separator token is used to
        split predictions into two :class:`~casanovo.data.psm.PepSpecMatch`
        objects per spectrum.
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
        max_peptide_len: int = 100,
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
        out_writer: Optional[ms_io.MztabWriter] = None,
        calculate_precision: bool = False,
        tokenizer: PeptideTokenizer | None = None,
        **kwargs: Dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = tokenizer or PeptideTokenizer()
        self.vocab_size = len(self.tokenizer) + 1
        # Build the model.
        self.encoder = SpectrumEncoder(
            d_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.decoder = PeptideDecoder(
            n_tokens=self.tokenizer,
            d_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            max_charge=max_charge,
        )
        self.softmax = torch.nn.Softmax(2)
        ignore_index = 0
        self.celoss = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=train_label_smoothing,
            reduction="none",
        )
        self.val_celoss = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="none"
        )
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
        self.max_peptide_len = max_peptide_len
        self.residues = residues
        self.precursor_mass_tol = precursor_mass_tol
        self.isotope_error_range = isotope_error_range
        self.min_peptide_len = min_peptide_len
        self.n_beams = n_beams
        self.top_match = top_match
        self.stop_token = self.tokenizer.stop_int

        self.chimera = isinstance(self.tokenizer, ChimeraTokenizer)
        if self.chimera:
            self.sep_token = self.tokenizer.index[
                self.tokenizer.chimeric_separator_token
            ]

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward learning step for non-autoregressive decoding.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset.

        Returns
        -------
        scores : torch.Tensor of shape (n_spectra, max_peptide_len, n_amino_acids)
            The individual amino acid scores for each prediction.
        seqs : torch.Tensor of shape (n_spectra, length)
            The ground truth tokens for training, or None for inference.
        """
        mzs, ints, precursors, seqs = self._process_batch(batch)
        memories, mem_masks = self.encoder(mzs, ints)

        zero_tokens = torch.zeros(
            (mzs.shape[0], self.max_peptide_len),
            dtype=torch.long,
            device=self.device,
        )
        scores = self.decoder(
            tokens=zero_tokens,
            memory=memories,
            memory_key_padding_mask=mem_masks,
            precursors=precursors,
        )

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

        pred, truth = self._forward_step(batch)

        batch_size, seq_len = truth.shape
        seq_len = min(seq_len, self.max_peptide_len)
        truth = truth[:, :seq_len]
        pred_sliced = pred[:, :seq_len, :]
        pred_flat = pred_sliced.reshape(-1, self.vocab_size)

        loss_fun = self.celoss if mode == "train" else self.val_celoss

        if self.chimera and "seq_compliment" in batch:
            # Chimera hard-min loss: pick the peptide ordering with lower
            # mask-normalised per-sequence CE, then average raw token losses
            # across the batch (consistent with the non-chimeric path).
            # Because the NAR decoder always receives zero tokens, a single
            # forward pass produces logits that can be scored against both
            # peptide orderings without a second decoder call.
            truth_comp = batch["seq_compliment"][:, :seq_len]
            mask = (truth != 0).float()
            n_real = mask.sum(dim=1).clamp(min=1)
            raw_a = (
                loss_fun(pred_flat, truth.flatten()).reshape(batch_size, seq_len)
            )
            raw_b = (
                loss_fun(pred_flat, truth_comp.flatten()).reshape(batch_size, seq_len)
            )
            # Mask-normalised loss used only for ordering selection.
            loss_a_norm = (raw_a * mask).sum(dim=1) / n_real
            loss_b_norm = (raw_b * mask).sum(dim=1) / n_real
            winner = (loss_b_norm < loss_a_norm).float().unsqueeze(1)
            # Divide by real tokens only to avoid padding dilution.
            n_real_total = mask.sum().clamp(min=1)
            loss = (raw_a * (1 - winner) + raw_b * winner).sum() / n_real_total
        else:
            raw = loss_fun(pred_flat, truth.flatten())
            n_real_total = (truth.flatten() != 0).float().sum().clamp(min=1)
            loss = raw.sum() / n_real_total
        self.log(
            f"{mode}_CELoss",
            loss.detach(),
            on_step=(mode == "train"),  # per-step logging for debugging
            on_epoch=True,
            sync_dist=True,
            batch_size=pred_flat.shape[0],
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

        # Calculate and log amino acid and peptide match evaluation metrics.
        # In chimera mode each chimeric spectrum contributes two (truth, pred)
        # sub-peptide pairs; single-peptide spectra contribute one.
        # FIXME: Remove work around when depthcharge reverse detokenization
        # bug is fixed.
        # peptides_true = self.tokenizer.detokenize(batch["seq"])
        peptides_true = [
            "".join(pep)
            for pep in self.tokenizer.detokenize(batch["seq"], join=False)
        ]
        pred_psms = self.predict_step(batch)

        if self.chimera:
            peptides_true, peptides_pred = self._build_chimera_eval_pairs(
                batch, peptides_true, pred_psms
            )
        else:
            peptides_pred = [m.sequence for m in pred_psms]

        aa_precision, _, pep_precision = evaluate.aa_match_metrics(
            *evaluate.aa_match_batch(
                peptides_true, peptides_pred, self.tokenizer.residues
            )
        )

        # Use the number of original spectra as the batch_size weight so that
        # Lightning's metric averaging stays spectrum-normalised even when
        # chimeric spectra produce two evaluation pairs.
        n_spectra = batch["precursor_charge"].shape[0]
        log_args = dict(on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "pep_precision", pep_precision, **log_args, batch_size=n_spectra
        )
        self.log(
            "aa_precision", aa_precision, **log_args, batch_size=n_spectra
        )
        return loss

    def _build_chimera_eval_pairs(
        self,
        batch: Dict[str, torch.Tensor],
        peptides_true: List[str],
        pred_psms: List[psm.PepSpecMatch],
    ) -> Tuple[List[str], List[str]]:
        """Pair ground-truth and predicted sub-peptides for chimera evaluation.

        For each spectrum, the ground-truth chimeric string ``"pep1:pep2"`` is
        split into individual sub-peptides.  Predicted sub-peptides (already
        split by :meth:`predict_step`) are grouped by spectrum ID and matched
        positionally.  If the model predicted fewer sub-peptides than the
        ground truth contains (e.g. no separator was emitted for a chimeric
        spectrum), the missing prediction is recorded as an empty string so
        that it counts as a miss.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            The current validation batch.
        peptides_true : List[str]
            Ground-truth sequences, one per spectrum (may contain ``":"``).
        pred_psms : List[psm.PepSpecMatch]
            Predicted PSMs returned by :meth:`predict_step`.

        Returns
        -------
        flat_true : List[str]
            Ground-truth sub-peptides, one entry per sub-peptide pair.
        flat_pred : List[str]
            Corresponding predicted sub-peptides (empty string for misses).
        """
        sep = self.tokenizer.chimeric_separator_token

        # Group predictions by spectrum ID.
        pred_by_spec: Dict[Tuple, List[str]] = {}
        for m in pred_psms:
            pred_by_spec.setdefault(m.spectrum_id, []).append(m.sequence)

        flat_true: List[str] = []
        flat_pred: List[str] = []

        for peak_file, scan_id, true_str in zip(
            batch["peak_file"], batch["scan_id"], peptides_true
        ):
            true_parts = true_str.split(sep) if sep in true_str else [true_str]
            preds = pred_by_spec.get((peak_file, scan_id), [])

            if len(true_parts) == 2 and len(preds) >= 2:
                # For chimeric spectra with two predictions, try both orderings
                # and pick the one with more exact sub-peptide matches.  This
                # avoids penalising the model when it predicts the correct
                # sub-peptides in the opposite order from the stored annotation.
                t0, t1 = true_parts
                p0, p1 = preds[0], preds[1]
                score_straight = (p0 == t0) + (p1 == t1)
                score_swapped = (p0 == t1) + (p1 == t0)
                if score_swapped > score_straight:
                    p0, p1 = p1, p0
                flat_true.extend([t0, t1])
                flat_pred.extend([p0, p1])
            else:
                for j, true_part in enumerate(true_parts):
                    flat_true.append(true_part)
                    flat_pred.append(preds[j] if j < len(preds) else "")

        return flat_true, flat_pred

    def predict_step(
        self, batch: Dict[str, torch.Tensor], *args
    ) -> List[psm.PepSpecMatch]:
        """
        A single prediction step (non-autoregressive with N-term check).

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, containing keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``, ``precursor_charge``,
            plus metadata like ``peak_file`` and ``scan_id``.

        Returns
        -------
        predictions : List[psm.PepSpecMatch]
            Predicted PSMs for the given batch of spectra.  In chimera mode
            up to two PSMs are emitted per spectrum (one per sub-peptide).
        """
        # Forward pass
        logits, _ = self._forward_step(batch)  # logits: (B, L, V)
        device = logits.device
        batch_size, L, V = logits.shape

        # N-term indices
        nterm_idx = self.nterm_idx.to(device)
        nterm_set = set(nterm_idx.tolist())

        # Probabilities and argmax tokens
        probs = torch.softmax(logits, dim=-1)  # (B, L, V)
        predicted_tokens = probs.argmax(dim=-1)  # (B, L)

        # Per-AA confidence
        per_aa_conf = probs.gather(-1, predicted_tokens.unsqueeze(-1)).squeeze(
            -1
        )  # (B, L)

        # Vectorized N-term cleanup for non-reverse, non-chimera case only.
        # Chimera and reverse cases are handled per-spectrum below.
        if not self.tokenizer.reverse and L > 1 and not self.chimera:
            mask = torch.isin(predicted_tokens[:, 1:], nterm_idx)
            predicted_tokens[:, 1:][mask] = 0

        predictions: List[psm.PepSpecMatch] = []

        for b, (
            filename,
            scan,
            charge,
            prec_mz,
            tokens,
            confs,
        ) in enumerate(
            zip(
                batch["peak_file"],
                batch["scan_id"],
                batch["precursor_charge"],
                batch["precursor_mz"],
                predicted_tokens,
                per_aa_conf,
            )
        ):
            if self.chimera:
                predictions.extend(
                    self._predict_chimera(
                        b, filename, scan, charge, prec_mz,
                        tokens, confs, logits, nterm_idx, nterm_set, L,
                    )
                )
            else:
                spec_match = self._predict_single(
                    b, filename, scan, charge, prec_mz,
                    tokens, confs, logits, nterm_idx, nterm_set, L,
                )
                if spec_match is not None:
                    predictions.append(spec_match)

        return predictions

    def _predict_single(
        self,
        b: int,
        filename: str,
        scan: str,
        charge: torch.Tensor,
        prec_mz: torch.Tensor,
        tokens: torch.Tensor,
        confs: torch.Tensor,
        logits: torch.Tensor,
        nterm_idx: torch.Tensor,
        nterm_set: set,
        L: int,
    ):
        """Build a single PSM from one spectrum's predicted tokens.

        Handles the per-spectrum N-term fix for the reverse-tokenizer case.
        For the non-reverse case the vectorized fix has already been applied
        before this call.
        """
        # Find STOP position
        stop_pos = L
        for j, t in enumerate(tokens):
            if t == self.stop_token or t == 0:
                stop_pos = j
                break

        if self.tokenizer.reverse:
            # Allowed N-term mod is at the last valid position (stop_pos - 1).
            valid_pos = max(0, stop_pos - 1)
            for j in range(stop_pos):
                tok = int(tokens[j].item())
                if tok in nterm_set and j != valid_pos:
                    masked = logits[b, j].clone()
                    masked[nterm_idx] = -float("inf")
                    newtok_idx = int(masked.argmax().item())
                    tokens[j] = newtok_idx
                    new_probs = torch.softmax(masked, dim=0)
                    confs[j] = new_probs[newtok_idx]

        valid_tokens = tokens[:stop_pos]
        valid_scores = confs[:stop_pos].detach().cpu().numpy()

        if len(valid_tokens) == 0:
            return None

        valid_tokens_cpu = valid_tokens.detach().cpu().unsqueeze(0)
        peptide = "".join(
            self.tokenizer.detokenize(valid_tokens_cpu, join=False)[0]
        )

        if self.tokenizer.reverse:
            valid_scores = valid_scores[::-1]

        return psm.PepSpecMatch(
            sequence=peptide,
            spectrum_id=(filename, scan),
            peptide_score=float(valid_scores.mean()),
            charge=int(charge),
            calc_mz=np.nan,
            exp_mz=float(prec_mz.item()),
            aa_scores=valid_scores,
        )

    def _predict_chimera(
        self,
        b: int,
        filename: str,
        scan: str,
        charge: torch.Tensor,
        prec_mz: torch.Tensor,
        tokens: torch.Tensor,
        confs: torch.Tensor,
        logits: torch.Tensor,
        nterm_idx: torch.Tensor,
        nterm_set: set,
        L: int,
    ) -> List[psm.PepSpecMatch]:
        """Build two PSMs from one chimeric spectrum's predicted tokens.

        Locates the separator token, applies per-sub-peptide N-term fixes,
        and returns up to two :class:`~casanovo.data.psm.PepSpecMatch` objects
        (one per sub-peptide).  The second peptide's PSM does not apply a
        precursor mass filter because its charge state may differ from the
        recorded precursor charge.

        Parameters
        ----------
        b : int
            Index into the batch dimension (used to index *logits*).
        tokens : torch.Tensor of shape (L,)
            Mutable copy of argmax token indices for this spectrum.
        confs : torch.Tensor of shape (L,)
            Corresponding per-position confidences.
        logits : torch.Tensor of shape (B, L, V)
            Full logit tensor from the forward pass.
        nterm_idx : torch.Tensor
            Token indices of all N-terminal modification tokens.
        nterm_set : set
            Same indices as a Python set for fast membership tests.
        L : int
            Sequence length (== ``logits.shape[1]``).

        Returns
        -------
        List[psm.PepSpecMatch]
            Zero, one or two PSMs depending on how many non-empty sub-peptides
            were found.
        """
        tokens = tokens.clone()
        confs = confs.clone()

        # Locate first STOP token (or padding zero).
        stop_pos = L
        for j, t in enumerate(tokens):
            if t == self.stop_token or t == 0:
                stop_pos = j
                break

        # Collect all separator positions before STOP.
        sep_positions = [
            j for j in range(stop_pos)
            if int(tokens[j].item()) == self.sep_token
        ]

        # Discard tokens after the second separator to ensure at most two
        # peptides are predicted.
        if len(sep_positions) >= 2:
            stop_pos = sep_positions[1]

        if len(sep_positions) == 0:
            # No separator found: fall back to single-peptide path.
            spec_match = self._predict_single(
                b, filename, scan, charge, prec_mz,
                tokens, confs, logits, nterm_idx, nterm_set, L,
            )
            return [spec_match] if spec_match is not None else []

        sep_pos = sep_positions[0]

        # Apply per-sub-peptide N-term fixes.
        if not self.tokenizer.reverse:
            # Forward tokenizer: valid N-term positions are 0 and sep_pos+1.
            invalid_ranges = [
                range(1, sep_pos),
                range(sep_pos + 2, stop_pos),
            ]
        else:
            # Reverse tokenizer: valid N-term positions are sep_pos-1 and
            # stop_pos-1 (last position of each reversed sub-peptide).
            invalid_ranges = [
                range(0, max(0, sep_pos - 1)),
                range(sep_pos + 1, max(sep_pos + 1, stop_pos - 1)),
            ]

        for rng in invalid_ranges:
            for j in rng:
                tok = int(tokens[j].item())
                if tok in nterm_set:
                    masked = logits[b, j].clone()
                    masked[nterm_idx] = -float("inf")
                    newtok_idx = int(masked.argmax().item())
                    tokens[j] = newtok_idx
                    new_probs = torch.softmax(masked, dim=0)
                    confs[j] = new_probs[newtok_idx]

        # Extract sub-sequences. sub1 is tokens before the first separator;
        # sub2 is tokens between the first separator and stop_pos (which is
        # either the original stop position or the second separator position).
        sub1_tok = tokens[:sep_pos]
        sub1_conf = confs[:sep_pos].detach().cpu().numpy()

        sub2_tok = tokens[sep_pos + 1 : stop_pos]
        sub2_conf = confs[sep_pos + 1 : stop_pos].detach().cpu().numpy()

        if self.tokenizer.reverse:
            sub1_conf = sub1_conf[::-1]
            sub2_conf = sub2_conf[::-1]

        results: List[psm.PepSpecMatch] = []
        for sub_tokens, sub_scores in [(sub1_tok, sub1_conf), (sub2_tok, sub2_conf)]:
            if len(sub_tokens) == 0:
                continue
            sub_tokens_cpu = sub_tokens.detach().cpu().unsqueeze(0)
            peptide = "".join(
                self.tokenizer.detokenize(sub_tokens_cpu, join=False)[0]
            )
            results.append(
                psm.PepSpecMatch(
                    sequence=peptide,
                    spectrum_id=(filename, scan),
                    peptide_score=float(sub_scores.mean()),
                    charge=int(charge),
                    calc_mz=np.nan,
                    exp_mz=float(prec_mz.item()),
                    aa_scores=sub_scores,
                )
            )

        return results

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        step = self.trainer.global_step
        if step % self.n_log == 0:
            loss_key = "train_CELoss_step"
            if loss_key in self.trainer.callback_metrics:
                loss = self.trainer.callback_metrics[loss_key].detach().item()
            else:
                loss = float("nan")  # mirrors on_train_epoch_end's np.nan pattern
            logger.debug("Step %i\tTrain loss: %.6f", step, loss)

    def on_train_epoch_end(self) -> None:
        """
        Log the training loss at the end of each epoch.
        """
        if "train_CELoss" in self.trainer.callback_metrics:
            train_loss = (
                self.trainer.callback_metrics["train_CELoss"].detach().item()
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
            "valid": callback_metrics["valid_CELoss"].detach().item(),
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
            # In chimera mode the recorded precursor belongs to the peptide
            # pair, not to an individual sub-peptide, so calc_mz is left as
            # NaN to avoid misleading comparisons against exp_mz.
            if not self.chimera:
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

        If the encoder output is already present in the batch, it is used
        directly by the decoder. Otherwise, the full forward pass including
        the encoder is performed.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset. It must contain ``seq``.
            For a full forward pass, it also needs ``mz_array``,
            ``intensity_array``, ``precursor_mz``, and ``precursor_charge``.
            Alternatively, it can contain precomputed encoder outputs:
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
            logits = self.decoder(
                tokens=tokens,
                memory=memories,
                memory_key_padding_mask=mem_masks,
                precursors=precursors,
            )
            probs = self.softmax(logits)
            return probs, tokens
        else:
            pred, truth = self._forward_step(batch)
            pred = self.softmax(pred)
            return pred, truth

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
    # In database search mode, fits_precursor_mz is implicitly True.
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
    fits_precursor_mz: Union[bool, np.ndarray] = True,
    lengths: Optional[np.ndarray] = None,
) -> Union[float, np.ndarray]:
    """
    Calculate the peptide-level confidence score from the raw
    amino acid scores.

    The peptide score is the product of the raw amino acid scores.
    This function contains paths for both single peptide inputs
    (de novo mode) and batched peptide inputs (database search mode).

    Parameters
    ----------
    aa_scores : np.ndarray
        A 1D array of amino acid scores for a single peptide, or a 2D
        padded array for a batch of peptides.
    fits_precursor_mz : bool or np.ndarray
        Flag or array of flags indicating whether predictions fit the
        precursor m/z filter.
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
        peptide_score = np.exp(peptide_log_score)

        if not fits_precursor_mz:
            peptide_score -= 1
        return peptide_score

    # BATCH PATH: database search
    else:
        if lengths is None:
            raise ValueError("`lengths` must be provided for batched input.")

        log_scores = np.log(np.clip(aa_scores, eps, 1))
        cumsum = np.cumsum(log_scores, axis=1)
        batch_size = aa_scores.shape[0]
        idx = np.arange(batch_size)
        peptide_log_scores = cumsum[idx, np.maximum(lengths - 1, 0)]
        peptide_scores = np.exp(peptide_log_scores)

        if isinstance(fits_precursor_mz, (bool, np.bool_)):
            if not fits_precursor_mz:
                peptide_scores -= 1
        else:
            peptide_scores[~fits_precursor_mz] -= 1

        return peptide_scores
