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

        # Create cache for decoded beams.
        pred_cache = collections.OrderedDict((i, []) for i in range(batch))

        # Get the first prediction.
        pred = self.decoder(
            tokens=torch.zeros(batch, 0, dtype=torch.int64, device=device),
            memory=memories,
            memory_key_padding_mask=mem_masks,
            precursors=precursors,
        )
        top_indices = torch.topk(pred[:, 0, :], beam, dim=1)[1]
        tokens[:, 0, :] = top_indices
        scores[:, :1, :, :] = einops.repeat(pred, "B L V -> B L V S", S=beam)

        # Initialize cumulative masses to track peptide masses
        token_masses = self.token_masses.to(device)
        cumulative_masses = torch.zeros(batch, beam, device=device)
        for b in range(batch):
            for s in range(beam):
                token_idx = tokens[b, 0, s].item()
                if token_idx < len(token_masses):  # ensure index is valid
                    cumulative_masses[b, s] = token_masses[token_idx]

        # Make all tensors the right shape for decoding.
        precursors = einops.repeat(precursors, "B L -> (B S) L", S=beam)
        mem_masks = einops.repeat(mem_masks, "B L -> (B S) L", S=beam)
        memories = einops.repeat(memories, "B L V -> (B S) L V", S=beam)
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")
        cumulative_masses = einops.rearrange(cumulative_masses, "B S -> (B S)")

        # Store temporary attributes for use by other methods
        self._cumulative_masses = cumulative_masses
        self._batch_size = batch
        self._beam_size = beam

        try:
            # The main decoding loop.
            for step in range(0, self.max_peptide_len):
                # Terminate beams exceeding the precursor m/z tolerance and
                # track all finished beams (either terminated or stop token
                # predicted).
                (
                    finished_beams,
                    beam_fits_precursor,
                    discarded_beams,
                ) = self._finish_beams(tokens, precursors, step)

                # Cache peptide predictions from the finished beams (but not
                # the discarded beams).
                beams_to_cache = finished_beams & ~discarded_beams
                if torch.any(beams_to_cache):
                    self._cache_finished_beams(
                        tokens,
                        scores,
                        step,
                        beams_to_cache,
                        beam_fits_precursor,
                        pred_cache,
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
                    )

                    scores[active_beams, : step + 2, :] = active_scores

                # Find the top-k beams with the highest scores and continue
                # decoding those.
                tokens, scores = self._get_topk_beams(
                    tokens, scores, finished_beams, batch, step + 1
                )
        finally:
            # Ensure temporary attributes are cleaned up in all cases to prevent memory leaks
            temp_attrs = ["_cumulative_masses", "_batch_size", "_beam_size"]
            for attr in temp_attrs:
                if hasattr(self, attr):
                    delattr(self, attr)

        # Return the peptide with the highest confidence score, within
        # the precursor m/z tolerance if possible.
        return list(self._get_top_peptide(pred_cache))

    def _finish_beams(
        self,
        tokens: torch.Tensor,
        precursors: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Track all beams that have been finished.

        Beams are finished by predicting the stop token or because they
        were terminated due to exceeding the precursor m/z tolerance.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
        precursors : torch.Tensor of shape (n_spectra * n_beams, 3)
            The measured precursor mass (axis 0), precursor charge
            (axis 1), and precursor m/z (axis 2) of each MS/MS spectrum.
        step : int
            Index of the current decoding step.

        Returns
        -------
        finished_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams have
            been finished.
        beam_fits_precursor: torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating if the current beams are within
            the precursor m/z tolerance.
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
        neg_mass_idx = self.neg_mass_idx
        token_masses = self.token_masses

        # Check the tokens at the current step
        current_tokens = tokens[:, step]

        # Initialize return tensors
        beam_fits_precursor = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )
        finished_beams = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )
        ends_stop_token = current_tokens == self.stop_token
        finished_beams[ends_stop_token] = True

        discarded_beams = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )
        discarded_beams[current_tokens == 0] = True

        # Discard beams with invalid modification combinations
        if step > 1:
            dim0 = torch.arange(batch_size, device=device)
            final_pos = torch.full((batch_size,), step, device=device)
            final_pos[ends_stop_token] = step - 1

            # Vectorized check for multiple N-terminal modifications
            last_token_is_nterm = torch.isin(
                tokens[dim0, final_pos], nterm_idx
            )
            prev_token_is_nterm = torch.isin(
                tokens[dim0, final_pos - 1], nterm_idx
            )
            multiple_mods = last_token_is_nterm & prev_token_is_nterm

            # Vectorized check for internal N-terminal modifications
            positions = torch.arange(tokens.shape[1], device=device)
            mask = (final_pos - 1).unsqueeze(1) >= positions
            token_mask = torch.where(mask, tokens, torch.zeros_like(tokens))
            internal_mods = torch.any(torch.isin(token_mask, nterm_idx), dim=1)

            discarded_beams = discarded_beams | multiple_mods | internal_mods

        # Get precursor information
        precursor_charges = precursors[:, 1]
        precursor_mzs = precursors[:, 2]

        # Calculate peptide lengths
        peptide_lens = torch.full((batch_size,), step + 1, device=device)
        # Adjust for stop tokens
        if self.tokenizer.reverse:
            has_stop_at_start = tokens[:, 0] == self.stop_token
            peptide_lens[has_stop_at_start] -= 1
        else:
            has_stop_at_end = ends_stop_token
            peptide_lens[has_stop_at_end] -= 1

        # Discard beams that don't meet minimum peptide length
        too_short = finished_beams & (peptide_lens < self.min_peptide_len)
        discarded_beams[too_short] = True

        # Mask for beams we need to check mass tolerance
        beams_to_check = ~discarded_beams

        if torch.any(beams_to_check):
            # For all beams that need checking, perform a batch-wise, high-precision
            # recalculation of their mass and check against the precursor m/z tolerance.
            idx = torch.nonzero(beams_to_check).squeeze(-1)

            # Get the effective token sequences for the current step (without stop tokens).
            sequences_to_check = []
            charges_to_check = precursor_charges[idx]

            for i, beam_idx in enumerate(idx):
                seq = tokens[beam_idx, : step + 1]
                # If the last token is a stop token, remove it.
                if seq[-1] == self.stop_token:
                    seq = seq[:-1]
                sequences_to_check.append(seq)

            # Find the maximum sequence length for padding.
            if sequences_to_check:
                max_len = max(len(seq) for seq in sequences_to_check)
                if max_len > 0:
                    # Create a padded tensor.
                    padded_sequences = torch.zeros(
                        len(sequences_to_check),
                        max_len,
                        dtype=torch.int64,
                        device=device,
                    )
                    for i, seq in enumerate(sequences_to_check):
                        if len(seq) > 0:
                            padded_sequences[i, : len(seq)] = seq

                    # Batch calculate precursor ions.
                    # The tokenizer requires input on the CPU.
                    recalc_mzs = self.tokenizer.calculate_precursor_ions(
                        padded_sequences.cpu(), charges_to_check.cpu()
                    ).to(device, dtype=torch.float64)

                    # Convert to neutral mass.
                    recalc_neutral_masses = (
                        recalc_mzs - 1.007276
                    ) * charges_to_check.double()

                    # Update cumulative mass with the high-precision value.
                    self._cumulative_masses[idx] = recalc_neutral_masses.to(
                        self._cumulative_masses.dtype
                    )

                    # This is the recalculated theoretical m/z.
                    current_mzs = recalc_mzs
                else:
                    # If all sequences are empty, set m/z to 0.
                    current_mzs = torch.zeros(
                        len(idx), dtype=torch.float64, device=device
                    )
            else:
                current_mzs = torch.zeros(
                    len(idx), dtype=torch.float64, device=device
                )

            precursor_mzs_obs = precursor_mzs[idx].double()

            # Create a tensor for the isotope error range (e.g., [0, 1]).
            isotope_range = torch.arange(
                self.isotope_error_range[0],
                self.isotope_error_range[1] + 1,
                device=device,
                dtype=torch.float64,
            )

            # Calculate the m/z correction for each isotope based on charge.
            isotope_corr = (
                isotope_range.unsqueeze(0)
                * 1.00335
                / charges_to_check.double().unsqueeze(1)
            )

            # Calculate the PPM difference between the current m/z and the observed m/z for all isotope corrections.
            delta_ppms = (
                (
                    current_mzs.unsqueeze(1)
                    - (precursor_mzs_obs.unsqueeze(1) - isotope_corr)
                )
                / precursor_mzs_obs.unsqueeze(1)
                * 1e6
            )

            # For each beam, check if any isotope correction brings the PPM error within tolerance.
            matches_any = (
                torch.abs(delta_ppms) < self.precursor_mass_tol
            ).any(dim=1)

            # Store which beams match the precursor tolerance
            temp_matches = torch.zeros_like(beam_fits_precursor)
            temp_matches[idx] = matches_any

            # Decide whether to force terminate only for beams that have not naturally ended AND are not within the tolerance.
            still_alive = ~finished_beams[idx]
            to_terminate = still_alive & ~matches_any

            # Handle cases where a negative mass AA could potentially correct the mass.
            if torch.any(to_terminate) and self.neg_mass_idx.numel() > 0:
                exceeding_indices = torch.where(to_terminate)[0]
                neg_masses = self.token_masses[self.neg_mass_idx]

                exceeding_masses = recalc_neutral_masses[exceeding_indices]
                exceeding_charges = charges_to_check[
                    exceeding_indices
                ].double()
                exceeding_precursor_mzs = precursor_mzs_obs[exceeding_indices]

                # Calculate potential m/z with each negative mass AA
                potential_masses = exceeding_masses.unsqueeze(
                    1
                ) + neg_masses.double().unsqueeze(0)
                potential_mzs = (
                    potential_masses.unsqueeze(2)
                    / exceeding_charges.unsqueeze(1).unsqueeze(2)
                    + 1.007276
                )

                isotope_corr_expanded = (
                    isotope_range.unsqueeze(0).unsqueeze(0)
                    * 1.00335
                    / exceeding_charges.unsqueeze(1).unsqueeze(2)
                )
                observed_mzs_expanded = (
                    exceeding_precursor_mzs.unsqueeze(1).unsqueeze(2)
                    - isotope_corr_expanded
                )
                delta_ppms_neg = (
                    (potential_mzs - observed_mzs_expanded)
                    / exceeding_precursor_mzs.unsqueeze(1).unsqueeze(2)
                    * 1e6
                )

                any_neg_aa_works = torch.any(
                    torch.abs(delta_ppms_neg) < self.precursor_mass_tol,
                    dim=(1, 2),
                )
                any_not_strictly_exceeding = torch.any(
                    delta_ppms_neg <= self.precursor_mass_tol, dim=(1, 2)
                )

                # Update the matches flag for beams that can be 'saved' by a negative mass AA.
                can_be_saved = any_neg_aa_works | any_not_strictly_exceeding
                temp_matches[idx[exceeding_indices]] |= can_be_saved
                to_terminate[exceeding_indices] = ~can_be_saved

            # Terminate beams that are confirmed to exceed the tolerance.
            to_terminate = idx[to_terminate]
            finished_beams[to_terminate] = True

            # update beam_fits_precursor for finished beams
            beam_fits_precursor |= temp_matches & finished_beams

        return finished_beams, beam_fits_precursor, discarded_beams

    def _cache_finished_beams(
        self,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        step: int,
        beams_to_cache: torch.Tensor,
        beam_fits_precursor: torch.Tensor,
        pred_cache: Dict[
            int, List[Tuple[float, float, np.ndarray, torch.Tensor]]
        ],
    ):
        """
        Cache terminated beams.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and
            all spectra.
        step : int
            Index of the current decoding step.
        beams_to_cache : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams are
            ready for caching.
        beam_fits_precursor: torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the beams are within the
            precursor m/z tolerance.
        pred_cache : Dict[
            int, List[Tuple[float, float, np.ndarray, torch.Tensor]]
        ]
            Priority queue with finished beams for each spectrum,
            ordered by peptide score. For each finished beam, a tuple
            with the (negated) peptide score, a random tie-breaking
            float, the amino acid-level scores, and the predicted tokens
            is stored.
        """
        # Find non-zero indices for more efficient iteration
        cache_indices = (
            torch.nonzero(beams_to_cache).squeeze(-1).cpu().tolist()
        )

        device = self.device  # Get device from input tensor

        # Get beam indices and spectrum indices from cache_indices
        for i in cache_indices:
            # Find the starting index of the spectrum.
            spec_idx = i // self.n_beams

            # Get the predicted tokens
            pred_tokens = tokens[i, : step + 1]

            # Omit the stop token from the peptide sequence (if predicted).
            has_stop_token = pred_tokens[-1] == self.stop_token
            pred_peptide = pred_tokens[:-1] if has_stop_token else pred_tokens

            # Don't cache this peptide if it was already predicted previously.
            pred_peptide_cpu = pred_peptide.cpu()
            duplicate = False
            for pred_cached in pred_cache[spec_idx]:
                # TODO: Add duplicate predictions with their highest score.
                if torch.equal(pred_cached[-1], pred_peptide_cpu):
                    duplicate = True
                    break

            if duplicate:
                continue

            # Calculate softmax scores directly with proper indexing
            smx = self.softmax(scores[i : i + 1, : step + 1, :])

            # Vectorized AA score extraction
            range_tensor = torch.arange(len(pred_tokens), device=device)
            aa_scores = smx[0, range_tensor, pred_tokens].cpu().numpy()

            # Add explicit score 0 for missing stop token
            if not has_stop_token:
                aa_scores = np.append(aa_scores, 0)

            # Calculate the peptide score using the appropriate scoring function
            peptide_score = _peptide_score(
                aa_scores, beam_fits_precursor[i].item()
            )

            # Omit the stop token from the amino acid-level scores.
            aa_scores = aa_scores[:-1]
            # Add the prediction to the cache (minimum priority queue,
            # maximum the number of beams elements).
            if len(pred_cache[spec_idx]) < self.n_beams:
                heapadd = heapq.heappush
            else:
                heapadd = heapq.heappushpop

            heapadd(
                pred_cache[spec_idx],
                (
                    peptide_score,
                    np.random.random_sample(),
                    aa_scores,
                    torch.clone(
                        pred_peptide_cpu
                    ),  # Ensure tensor is on CPU for storage
                ),
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

        token_masses = self.token_masses

        # Reshape to group by spectrum (B for "batch").
        tokens = einops.rearrange(tokens, "(B S) L -> B L S", S=beam)
        scores = einops.rearrange(scores, "(B S) L V -> B L V S", S=beam)
        cumulative_masses = einops.rearrange(
            self._cumulative_masses, "(B S) -> B S", S=beam
        )

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

        # Ensure we use the exact same scoring and topk mechanism as original code
        # Use nanmean to properly handle NaN values in scores
        mean_scores = torch.nanmean(step_scores, dim=1)

        # Apply mask and get top-k indices
        _, top_idx = torch.topk(mean_scores * active_mask, beam, dim=1)

        # Vectorized index conversion without loops
        indices = torch.unravel_index(top_idx.flatten(), (vocab, beam))
        v_idx = indices[0].reshape(top_idx.shape).to(device)
        s_idx = indices[1].reshape(top_idx.shape).to(device)

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

        # Vectorized cumulative mass update
        # Gather parent beam masses
        parent_masses = torch.gather(cumulative_masses, dim=1, index=s_idx)

        # Get new token masses and update
        new_token_masses = token_masses[v_idx]
        cumulative_masses_new = parent_masses + new_token_masses

        # Update class attribute with new cumulative masses
        self._cumulative_masses = einops.rearrange(
            cumulative_masses_new, "B S -> (B S)"
        )

        # Reshape for return
        tokens_out = einops.rearrange(tokens_new, "B L S -> (B S) L")
        scores_out = einops.rearrange(scores_new, "B L V S -> (B S) L V")

        return tokens_out, scores_out

    def _get_top_peptide(
        self,
        pred_cache: Dict[
            int, List[Tuple[float, float, np.ndarray, torch.Tensor]]
        ],
    ) -> Iterable[List[Tuple[float, np.ndarray, str]]]:
        """
        Return the peptide with the highest confidence score for each
        spectrum.

        Parameters
        ----------
        pred_cache : Dict[
            int, List[Tuple[float, float, np.ndarray, torch.Tensor]]
        ]
            Priority queue with finished beams for each spectrum,
            ordered by peptide score. For each finished beam, a tuple
            with the peptide score, a random tie-breaking float, the
            amino acid-level scores, and the predicted tokens is stored.

        Returns
        -------
        pred_peptides : Iterable[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide predictions.
            A peptide prediction consists of a tuple with the peptide
            score, the amino acid scores, and the predicted peptide
            sequence.
        """
        for peptides in pred_cache.values():
            if len(peptides) > 0:
                yield [
                    (
                        pep_score,
                        (
                            aa_scores[::-1]
                            if self.tokenizer.reverse
                            else aa_scores
                        ),
                        # FIXME: Remove work around when depthcharge reverse
                        #   detokenization bug is fixed.
                        # self.tokenizer.detokenize(
                        #     torch.unsqueeze(pred_tokens, 0)
                        # )[0],
                        "".join(
                            self.tokenizer.detokenize(
                                torch.unsqueeze(pred_tokens, 0),
                                join=False,
                            )[0]
                        ),
                    )
                    for pep_score, _, aa_scores, pred_tokens in heapq.nlargest(
                        self.top_match, peptides
                    )
                ]
            else:
                yield []

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
        pred = pred[:, :-1, :].reshape(-1, self.vocab_size)

        if mode == "train":
            loss = self.celoss(pred, truth.flatten())
        else:
            loss = self.val_celoss(pred, truth.flatten())
        self.log(
            f"{mode}_CELoss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=pred.shape[0],
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
        log_args = dict(on_step=False, on_epoch=True, sync_dist=True)
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

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, which contains keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``,
            ``precursor_charge``, and ``seq``, each pointing to tensors
            with the corresponding data.

        Returns
        -------
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction,
            converted to probabilities using a softmax.
        tokens : torch.Tensor of shape (n_spectra, length)
            The predicted tokens for each spectrum.
        """
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
        predictions: List[psm.PepSpecMatch]
            The predicted PSMs for the processed batch.
        """
        predictions = collections.defaultdict(list)
        for psm_batch in self._psm_batches(batch):
            pred, truth = self.forward(psm_batch)
            peptide_scores, aa_scores_all = _calc_match_score(pred, truth)

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
                # Omit stop token from reported AA scores
                curr_aa_scores = curr_aa_scores[:-1]

                spectrum_id = (filename, scan)
                if self.tokenizer.reverse:
                    curr_aa_scores = curr_aa_scores[::-1]

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

        # Filter the top-scoring prediction(s) for each spectrum.
        predictions = list(
            itertools.chain.from_iterable(
                [
                    *(
                        sorted(
                            spectrum_predictions,
                            key=lambda p: p.peptide_score,
                            reverse=True,
                        )[: self.top_match]
                        for spectrum_predictions in predictions.values()
                    )
                ]
            )
        )

        # Determine the peptide sequence and parent proteins only for
        # the retained PSMs.
        for pred in predictions:
            pred.protein = self.protein_database.get_associated_protein(
                pred.sequence
            )

        return predictions

    def _psm_batches(
        self, batch: Dict[str, torch.Tensor]
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

        Returns
        -------
        psm_batch : Generator[Dict[str, torch.Tensor], None, None]
            A generator that yields batches of candidate database PSMs
            ready for scoring. Each batch contains repeated spectrum
            information for each candidate peptide to be scored
            against each spectrum.
        """
        batch_size = batch["precursor_charge"].shape[0]

        # Determine the candidates to score for each spectrum and
        # compile into new batches with the same size as the original
        # batch.
        candidates = []
        for i, (precursor_charge, precursor_mz) in enumerate(
            zip(batch["precursor_charge"], batch["precursor_mz"])
        ):
            for candidate in self.protein_database.get_candidates(
                precursor_mz, precursor_charge
            ):
                candidates.append((i, candidate))

            # Yield a batch if sufficient candidates are found or all
            # spectra have been processed.
            while len(candidates) >= batch_size or (
                i == batch_size - 1 and len(candidates) > 0
            ):
                batch_candidates = candidates[:batch_size]
                # Repeat the spectrum information for each candidate
                # that should be matched to the spectrum.
                psm_batch = {key: [] for key in [*batch.keys(), "seq"]}
                for spec_i, candidate in batch_candidates:
                    for key in batch.keys():
                        psm_batch[key].append(batch[key][spec_i])
                    psm_batch["seq"].append(candidate)

                # Convert the batch elements to tensors.
                for key in psm_batch.keys():
                    if isinstance(psm_batch[key][0], torch.Tensor):
                        psm_batch[key] = torch.stack(psm_batch[key])
                        psm_batch[key] = psm_batch[key].to(self.decoder.device)
                # We need to keep the original sequence for the database
                # lookup in case of there is an isoleucine -> leucine swap
                psm_batch["original_seq_str"] = psm_batch["seq"]
                psm_batch["seq"] = self.tokenizer.tokenize(
                    psm_batch["seq"], add_stop=True
                )
                psm_batch["seq"] = psm_batch["seq"].to(self.decoder.device)

                # Yield the PSM batch for processing.
                yield psm_batch

                # Remove the processed candidates from the list.
                candidates = candidates[batch_size:]


def _calc_match_score(
    batch_all_aa_scores: torch.Tensor,
    truth_aa_indices: torch.Tensor,
) -> Tuple[List[float], List[np.ndarray]]:
    """
    Calculate the score between the input spectra and associated
    peptide.

    Take in teacher-forced scoring of amino acids of the peptides (in a
    batch) and use the truth labels to calculate a score between the
    input spectra and associated peptide.

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
    # Remove trailing token
    batch_all_aa_scores = batch_all_aa_scores[:, :-1]

    # Get aa scores corresponding with true aas
    per_aa_scores = torch.gather(
        batch_all_aa_scores, 2, truth_aa_indices.unsqueeze(-1)
    ).squeeze(-1)

    # Calculate peptide scores and aa scores
    per_aa_scores = per_aa_scores.cpu().detach().numpy()
    score_mask = (truth_aa_indices != 0).cpu().detach().numpy()
    peptide_scores, aa_scores = [], []
    for psm_score, psm_mask in zip(per_aa_scores, score_mask):
        psm_aa_scores = psm_score[psm_mask]
        psm_peptide_score = _peptide_score(psm_aa_scores, True)
        peptide_scores.append(psm_peptide_score)
        aa_scores.append(psm_aa_scores)

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


def _peptide_score(aa_scores: np.ndarray, fits_precursor_mz: bool) -> float:
    """
    Calculate the peptide-level confidence score from the raw
    amino acid scores.

    The peptide score is the product of the raw amino acid scores.

    Parameters
    ----------
    aa_scores : np.ndarray
        Amino acid level confidence scores.
    fits_precursor_mz : bool
        Flag indicating whether the prediction fits the precursor m/z
        filter.

    Returns
    -------
    peptide_score : float
        The peptide score.
    """
    aa_scores = np.clip(aa_scores, np.finfo(np.float64).eps, 1)
    peptide_score = np.exp(np.sum(np.log(aa_scores)))
    if not fits_precursor_mz:
        peptide_score -= 1
    return peptide_score
