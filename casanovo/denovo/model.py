"""A de novo peptide sequencing model."""

import collections
import inspect
import itertools
import logging
import warnings
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

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

logger = logging.getLogger(__name__)


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
    **kwargs : Dict
        Additional keyword arguments passed to the Adam optimizer. Only
        valid Adam parameters are retained; any other values are ignored.
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
            ignore_index=ignore_index, label_smoothing=train_label_smoothing
        )
        self.val_celoss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
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
        # Keep only valid Adam arguments; other configuration values
        # (e.g. loaded from a checkpoint) must not reach the optimizer.
        adam_kwargs = set(inspect.signature(torch.optim.Adam).parameters)
        self.opt_kwargs = {k: v for k, v in kwargs.items() if k in adam_kwargs}

        # Data properties.
        self.max_peptide_len = max_peptide_len
        self.residues = residues
        self.min_peptide_len = min_peptide_len
        self.n_beams = n_beams
        self.top_match = top_match
        self.stop_token = self.tokenizer.stop_int

        # Logging.
        self.calculate_precision = calculate_precision
        self.n_log = n_log
        self._history = []
        # Count of spectra for which beam search returned no valid peptide.
        self.n_missing_predictions = 0
        # Per-file validation metadata; set by ModelRunner.train() before fit.
        self.val_stems: list = []
        self.n_main_loaders: int = 0

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

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        Predict peptide sequences for a batch of MS/MS spectra.

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
        pred_peptides : List[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide predictions.
            A peptide prediction consists of a tuple with the peptide
            score, the amino acid scores, and the predicted peptide
            sequence.
        """
        mzs, ints, precursors, _ = self._process_batch(batch)
        return self.beam_search_decode(mzs, ints, precursors)

    def beam_search_decode(
        self,
        mzs: torch.Tensor,
        intensities: torch.Tensor,
        precursors: torch.Tensor,
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        Beam search decoding of the spectrum predictions.

        Parameters
        ----------
        mzs : torch.Tensor of shape (n_spectra, max_peaks)
            The m/z values of spectra for which to predict peptide
            sequences. Axis 0 represents an MS/MS spectrum, axis 1
            contains the m/z values for each peak. These should be
            zero-padded, such that all the spectra in the batch are the
            same length.
        intensities: torch.Tensor of shape (n_spectra, max_peaks)
            The intensity values of spectra for which to predict peptide
            sequences. Axis 0 represents an MS/MS spectrum, axis 1
            contains the intensity values for each peak. These should
            be zero-padded, such that all the spectra in the batch are
            the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge
            (axis 1), and precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        pred_peptides : List[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide predictions.
            A peptide prediction consists of a tuple with the peptide
            score, the amino acid scores, and the predicted peptide
            sequence.
        """
        memories, mem_masks = self.encoder(mzs, intensities)

        # Get device from self for consistent placement
        device = self.device

        # Sizes.
        batch = mzs.shape[0]  # B
        length = self.max_peptide_len + 1  # L
        vocab = self.vocab_size  # V
        beam = self.n_beams  # S

        # Initialize scores and tokens.
        # Ensure tensors are on the correct device
        scores = torch.full(
            size=(batch, length, vocab, beam),
            fill_value=torch.nan,
            device=device,
        )

        tokens = torch.zeros(
            batch, length, beam, dtype=torch.int64, device=device
        )

        # Create cache for decoded beams. cache_tokens uses int32 to reduce
        # memory footprint; it is cast to int64 before detokenization.
        cache_tokens = torch.full(
            (batch, beam, length, length),
            0,
            dtype=torch.int32,
            device=device,
        )
        cache_scores = torch.full(
            (batch, beam, length, length),
            0.0,
            dtype=scores.dtype,
            device=device,
        )

        # Get the first prediction.
        pred = self.decoder(
            tokens=torch.zeros(batch, 0, dtype=torch.int64, device=device),
            memory=memories,
            memory_key_padding_mask=mem_masks,
            precursors=precursors,
        ).to(scores.dtype)
        top_indices = torch.topk(pred[:, 0, :], beam, dim=1)[1]
        tokens[:, 0, :] = top_indices
        scores[:, :1, :, :] = einops.repeat(pred, "B L V -> B L V S", S=beam)

        # Make all tensors the right shape for decoding.
        precursors = einops.repeat(precursors, "B L -> (B S) L", S=beam)
        mem_masks = einops.repeat(mem_masks, "B L -> (B S) L", S=beam)
        memories = einops.repeat(memories, "B L V -> (B S) L V", S=beam)
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")

        # Store temporary attributes for use by other methods
        self._batch_size = batch
        self._beam_size = beam

        try:
            # The main decoding loop.
            for step in range(0, self.max_peptide_len):
                # Track all finished beams (either terminated or stop token
                # predicted).
                finished_beams, discarded_beams = self._finish_beams(
                    tokens, step
                )

                # Cache peptide predictions from the finished beams (but not
                # the discarded beams).
                beams_to_cache = finished_beams & ~discarded_beams
                self._cache_finished_beams(
                    tokens,
                    scores,
                    step,
                    beams_to_cache,
                    cache_tokens,
                    cache_scores,
                )

                # Stop decoding when all current beams have been finished.
                # Continue with beams that have not been finished and not
                # discarded.
                finished_beams |= discarded_beams
                if torch.all(finished_beams):
                    break

                # Only update scores for active beams
                active_beams = ~finished_beams
                if torch.any(active_beams):
                    active_tokens = tokens[active_beams, : step + 1]
                    active_precursors = precursors[active_beams]
                    active_memories = memories[active_beams]
                    active_mem_masks = mem_masks[active_beams]

                    active_scores = self.decoder(
                        tokens=active_tokens,
                        precursors=active_precursors,
                        memory=active_memories,
                        memory_key_padding_mask=active_mem_masks,
                    ).to(scores.dtype)

                    scores[active_beams, : step + 2, :] = active_scores

                # Find the top-k beams with the highest scores and continue
                # decoding those.
                tokens, scores = self._get_topk_beams(
                    tokens, scores, finished_beams, batch, step + 1
                )
        finally:
            # Ensure temporary attributes are cleaned up in all cases to prevent memory leaks
            temp_attrs = ["_batch_size", "_beam_size"]
            for attr in temp_attrs:
                if hasattr(self, attr):
                    delattr(self, attr)

        # Return the peptide with the highest confidence score, within
        # the precursor m/z tolerance if possible.
        return list(self._get_top_peptide(cache_tokens, cache_scores))

    def _finish_beams(
        self,
        tokens: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Track all beams that have been finished.

        Beams are finished by predicting the stop token or because they
        were terminated due to violating minimum peptide length or
        invalid N-terminal modification placement.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
        step : int
            Index of the current decoding step.

        Returns
        -------
        finished_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams have
            been finished.
        discarded_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams should
            be discarded (e.g. because they were predicted to end but
            violate the minimum peptide length).
        """
        # Get device from self for consistent placement
        device = self.device
        batch_size = tokens.shape[0]

        # Use precomputed indices and ensure they're on the correct device
        nterm_idx = self.nterm_idx

        # Check the tokens at the current step
        current_tokens = tokens[:, step]

        # Initialize return tensors
        finished_beams = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )
        ends_stop_token = current_tokens == self.stop_token
        finished_beams[ends_stop_token] = True

        discarded_beams = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )
        discarded_beams[current_tokens == 0] = True

        # Discard beams with invalid modification combinations. At step 0 the
        # single token is at the position where an N-terminal modification is
        # allowed under either token order, so the check starts at step 1.
        if step > 0:
            final_pos = torch.full((batch_size,), step, device=device)
            final_pos[ends_stop_token] = step - 1

            # Vectorized check for multiple N-terminal modifications
            token_is_nterm = torch.isin(tokens, nterm_idx)
            num_modifications = token_is_nterm.sum(dim=1)
            has_n_term = num_modifications > 0

            # We only need to this check if there are any n-term mods
            if torch.any(has_n_term).item():
                # Catch multiple modifications, pretty straightforward
                multiple_mods = num_modifications[has_n_term] > 1

                # Vectorized check for internal N-terminal modifications.
                # This will fail to catch internal modifications in some cases
                # where there are multiple mods, but these are already discarded
                # by the previous check.
                # A reversed tokenizer emits the C-terminus first, so a valid
                # N-terminal modification is the *last* token generated; an
                # unreversed tokenizer emits it first.
                n_terminal_pos = (
                    final_pos[has_n_term] if self.tokenizer.reverse else 0
                )
                internal_mods = ~token_is_nterm[has_n_term, n_terminal_pos]

                # Only discard beams we have actually checked
                discarded_beams[has_n_term] |= multiple_mods | internal_mods

        # Calculate peptide lengths, and adjust for stop tokens
        peptide_lens = torch.full((batch_size,), step + 1, device=device)
        peptide_lens[ends_stop_token] -= 1

        # Discard beams that don't meet minimum peptide length
        too_short = peptide_lens < self.min_peptide_len
        discarded_beams[too_short & finished_beams] = True

        return finished_beams, discarded_beams

    def _cache_finished_beams(
        self,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        step: int,
        beams_to_cache: torch.Tensor,
        cache_tokens: torch.Tensor,
        cache_scores: torch.Tensor,
    ) -> None:
        """
        Cache terminated beams into fixed-size tensors.

        Storing candidates as tensors allows vectorized final selection and
        avoids per-step Python heap operations.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
        scores : torch.Tensor of shape
            (n_spectra * n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and
            all spectra.
        step : int
            Index of the current decoding step.
        beams_to_cache : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams are
            ready for caching.
        cache_tokens : torch.Tensor of shape
            (n_spectra, n_beams, max_length, max_length)
            Tensor cache for predicted tokens of finished beams.
        cache_scores : torch.Tensor of shape
            (n_spectra, n_beams, max_length, max_length)
            Tensor cache for raw token probabilities of finished beams.
        """
        batch, beam, _, _ = cache_tokens.shape
        vocab = scores.shape[-1]

        # [B, S, step + 1] actual tokens up to the current step.
        tokens_bsl = tokens.view(batch, beam, -1)[:, :, : step + 1]

        # Softmax over the vocabulary and gather the probability of each
        # selected token in one shot. Use the model's configured softmax so
        # that the normalization axis stays in sync with future changes.
        scores_view = scores[:, : step + 1, :].view(
            batch, beam, step + 1, vocab
        )
        smx = self.softmax(scores_view.transpose(2, 3)).transpose(2, 3)
        raw_scores = smx.gather(3, tokens_bsl.unsqueeze(-1)).squeeze(-1)

        # Masked write: only update slots where beams_to_cache is True.
        write_mask = beams_to_cache.view(batch, beam, 1)
        cache_tokens[:, :, step, : step + 1] = torch.where(
            write_mask,
            tokens_bsl,
            cache_tokens[:, :, step, : step + 1],
        )
        cache_scores[:, :, step, : step + 1] = torch.where(
            write_mask,
            raw_scores,
            cache_scores[:, :, step, : step + 1],
        )

    def _get_topk_beams(
        self,
        tokens: torch.tensor,
        scores: torch.tensor,
        finished_beams: torch.tensor,
        batch: int,
        step: int,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Find the top-k beams with the highest scores and continue
        decoding those.

        Stop decoding for beams that have been finished.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and
            all spectra.
        finished_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams are
            ready for caching.
        batch: int
            Number of spectra in the batch.
        step : int
            Index of the next decoding step.

        Returns
        -------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and
            all spectra.
        """
        beam = self.n_beams  # S
        vocab = self.vocab_size  # V
        device = self.device  # Get device from input tensor

        # Reshape to group by spectrum (B for "batch").
        tokens = einops.rearrange(tokens, "(B S) L -> B L S", S=beam)
        scores = einops.rearrange(scores, "(B S) L V -> B L V S", S=beam)

        # Get the previous tokens and scores.
        prev_tokens = einops.repeat(
            tokens[:, :step, :], "B L S -> B L V S", V=vocab
        )
        prev_scores = torch.gather(
            scores[:, :step, :, :], dim=2, index=prev_tokens
        )
        prev_scores = einops.repeat(
            prev_scores[:, :, 0, :], "B L S -> B L (V S)", V=vocab
        )

        # Get the scores for all possible beams at this step.
        step_scores = torch.zeros(
            batch, step + 1, beam * vocab, device=device
        ).type_as(scores)
        step_scores[:, :step, :] = prev_scores
        step_scores[:, step, :] = einops.rearrange(
            scores[:, step, :, :], "B V S -> B (V S)"
        )

        # Find all still active beams by masking out terminated beams.
        active_mask = (
            ~finished_beams.reshape(batch, beam).repeat(1, vocab)
        ).float()
        # Mask out the index '0', i.e. padding token, by default.
        active_mask[:, :beam] = 1e-8
        # Compute beam scores and select top-k candidates
        # Use nanmean to properly handle NaN values in scores
        mean_scores = torch.nanmean(step_scores, dim=1)

        # Apply mask and get top-k indices
        _, top_idx = torch.topk(mean_scores * active_mask, beam, dim=1)

        # Vectorized index conversion without loops, fully on GPU.
        v_idx = (top_idx // beam).to(torch.long)
        s_idx = (top_idx % beam).to(torch.long)

        # Create batch indices for gathering - flatten s_idx for indexing
        s_idx_flat = einops.rearrange(s_idx, "B S -> (B S)")
        b_idx = einops.repeat(
            torch.arange(batch, device=device), "B -> (B S)", S=beam
        )

        # Record the top K decodings.
        tokens_new = tokens.clone()
        tokens_new[:, :step, :] = einops.rearrange(
            prev_tokens[b_idx, :, 0, s_idx_flat], "(B S) L -> B L S", S=beam
        )
        tokens_new[:, step, :] = v_idx

        scores_new = scores.clone()
        scores_new[:, : step + 1, :, :] = einops.rearrange(
            scores[b_idx, : step + 1, :, s_idx_flat],
            "(B S) L V -> B L V S",
            S=beam,
        )

        # Reshape for return
        tokens_out = einops.rearrange(tokens_new, "B L S -> (B S) L")
        scores_out = einops.rearrange(scores_new, "B L V S -> (B S) L V")

        return tokens_out, scores_out

    def _get_top_peptide(
        self,
        cache_tokens: torch.Tensor,
        cache_scores: torch.Tensor,
    ) -> Iterable[List[Tuple[float, np.ndarray, str]]]:
        """
        Return the peptide with the highest confidence score for each
        spectrum from the cache tensors.

        Parameters
        ----------
        cache_tokens : torch.Tensor of shape
            (n_spectra, n_beams, max_length, max_length)
            Tensor cache for predicted tokens of finished beams.
        cache_scores : torch.Tensor of shape
            (n_spectra, n_beams, max_length, max_length)
            Tensor cache for raw token probabilities of finished beams.

        Returns
        -------
        pred_peptides : Iterable[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide predictions.
            A peptide prediction consists of a tuple with the peptide
            score, the amino acid-level scores, and the predicted peptide
            sequence.
        """
        batch, beam, length, _ = cache_tokens.shape
        device = cache_tokens.device
        eps = torch.finfo(cache_scores.dtype).eps

        # Flatten the candidate pool over beams and decoding steps.
        flat_tokens = cache_tokens.view(batch, beam * length, length)
        flat_raw = cache_scores.view(batch, beam * length, length)
        # Valid slots are those with non-zero raw probabilities. This is
        # equivalent to the previous explicit cache_mask because softmax
        # probabilities are strictly positive and unwritten slots are 0.
        flat_mask = flat_raw.any(dim=-1)

        # The actual decoding step for each history slot.
        step_idx = torch.arange(length, device=device).view(1, 1, length)
        flat_step = step_idx.expand(batch, beam, length).reshape(
            batch, beam * length
        )

        # The last real token is at position `step`.
        last_token = flat_tokens.gather(2, flat_step.unsqueeze(-1)).squeeze(-1)
        has_stop = last_token == self.stop_token

        # Compute peptide scores as the product of raw token probabilities.
        # For positions beyond `step`, raw scores are 0 and are masked out.
        pos_idx = torch.arange(length, device=device).view(1, 1, length)
        valid_pos = pos_idx <= flat_step.unsqueeze(-1)
        log_raw = torch.log(flat_raw.clamp(min=eps))
        log_score = (log_raw * valid_pos).sum(dim=-1)
        # Penalize candidates without a stop token by appending a 0 score.
        log_score = log_score + torch.where(
            has_stop,
            torch.zeros_like(log_score),
            torch.log(torch.tensor(eps, device=device, dtype=log_score.dtype)),
        )
        # Use -inf for invalid slots so they are never selected.
        log_score = log_score.masked_fill(~flat_mask, float("-inf"))

        # Select the top candidates for each spectrum in log space to avoid
        # exp() underflow to 0, which would tie with masked slots.
        # Fast path for the common case where only the top-1 match is needed.
        if self.top_match == 1:
            n_candidates = 1
            topk_idx = log_score.argmax(dim=1, keepdim=True)
            topk_log_scores = log_score.gather(1, topk_idx)
        else:
            n_candidates = min(self.top_match, beam * length)
            topk_log_scores, topk_idx = torch.topk(
                log_score, n_candidates, dim=1
            )

        # Move selection-related tensors to CPU once to avoid per-iteration
        # GPU synchronization.
        topk_idx_cpu = topk_idx.cpu()
        flat_step_cpu = flat_step.cpu()
        has_stop_cpu = has_stop.cpu()
        flat_mask_cpu = flat_mask.cpu()
        topk_peptide_scores_cpu = torch.exp(topk_log_scores.cpu())

        for i in range(batch):
            pred_peptides = []
            for k in range(n_candidates):
                idx = topk_idx_cpu[i, k].item()
                if not flat_mask_cpu[i, idx]:
                    continue

                step = int(flat_step_cpu[i, idx])
                pred_tokens = flat_tokens[i, idx, : step + 1].long()
                stop = bool(has_stop_cpu[i, idx])

                if stop:
                    pred_tokens = pred_tokens[:-1]
                    aa_scores = flat_raw[i, idx, :step].cpu().numpy()
                else:
                    aa_scores = flat_raw[i, idx, : step + 1].cpu().numpy()

                peptide_score = float(topk_peptide_scores_cpu[i, k])

                if self.tokenizer.reverse:
                    aa_scores = aa_scores[::-1]

                pred_peptides.append(
                    (
                        peptide_score,
                        aa_scores,
                        self.tokenizer.detokenize(
                            torch.unsqueeze(pred_tokens, 0)
                        )[0],
                    )
                )

                if len(pred_peptides) >= self.top_match:
                    break

            yield pred_peptides

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

    def _forward_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward learning step.

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
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        tokens : torch.Tensor of shape (n_spectra, length)
            The predicted tokens for each spectrum.
        """
        mzs, ints, precursors, tokens = self._process_batch(batch)
        memories, mem_masks = self.encoder(mzs, ints)
        scores = self.decoder(
            tokens=tokens,
            memory=memories,
            memory_key_padding_mask=mem_masks,
            precursors=precursors,
        )
        return scores, tokens

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
        pred, truth = self._forward_step(batch)
        pred = pred[:, :-1, :].reshape(-1, self.vocab_size)
        loss = self.celoss(pred, truth.flatten())
        self.log(
            "train_CELoss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=pred.shape[0],
        )
        return loss

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
            to the aggregate ``valid_CELoss`` used for checkpoint selection.
            Higher indices are "tracking" files logged per-file only.

        Returns
        -------
        torch.Tensor
            The loss of the validation step.
        """
        pred, truth = self._forward_step(batch)
        pred = pred[:, :-1, :].reshape(-1, self.vocab_size)
        loss = self.val_celoss(pred, truth.flatten())

        batch_size = pred.shape[0]
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
            self.log(f"valid_CELoss/{stem}", loss.detach(), **log_kwargs)

        if is_main:
            # Contributes to the aggregate ``valid_CELoss`` monitored by
            # ModelCheckpoint for best-checkpoint selection.
            self.log("valid_CELoss", loss.detach(), **log_kwargs)

        if not self.calculate_precision or not is_main:
            return loss

        # Calculate and log amino acid and peptide match evaluation
        # metrics from the predicted peptides (main files only).
        peptides_true = self.tokenizer.detokenize(batch["seq"])
        peptides_pred = [
            pred
            for spectrum_preds in self.forward(batch)
            for _, _, pred in spectrum_preds
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
        return loss

    def predict_step(
        self, batch: Dict[str, torch.Tensor], *args
    ) -> List[ms_io.PepSpecMatch]:
        """
        A single prediction step.

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
        predictions: List[psm.PepSpecMatch]
            Predicted PSMs for the given batch of spectra.
        """
        predictions = []
        for (
            filename,
            scan,
            precursor_charge,
            precursor_mz,
            spectrum_preds,
        ) in zip(
            batch["peak_file"],
            batch["scan_id"],
            batch["precursor_charge"],
            batch["precursor_mz"],
            self.forward(batch),
        ):
            if not spectrum_preds:
                self.n_missing_predictions += 1
                continue
            for peptide_score, aa_scores, peptide in spectrum_preds:
                predictions.append(
                    psm.PepSpecMatch(
                        sequence=peptide,
                        spectrum_id=(filename, scan),
                        peptide_score=peptide_score,
                        charge=int(precursor_charge),
                        calc_mz=np.nan,
                        exp_mz=precursor_mz.item(),
                        aa_scores=aa_scores,
                    )
                )

        return predictions

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
        metrics = {"step": self.trainer.global_step}

        if "valid_CELoss" in callback_metrics:
            metrics["valid"] = callback_metrics["valid_CELoss"].detach().item()

        for stem in self.val_stems:
            key = f"valid_CELoss/{stem}"
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
        aa_scores = torch.as_tensor(aa_scores)

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
