"""A de novo peptide sequencing model."""

import collections
import heapq
import logging
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import depthcharge.masses
import einops
import torch
import numpy as np
import lightning.pytorch as pl
from torch.utils.tensorboard import SummaryWriter
from depthcharge.components import ModelMixin, PeptideDecoder, SpectrumEncoder

from casanovo.denovo import evaluate

from casanovo import config
from casanovo.data import ms_io

logger = logging.getLogger("casanovo")


class Spec2Pep(pl.LightningModule, ModelMixin):
    """
    A Transformer model for de novo peptide sequencing.

    Use this model in conjunction with a pytorch-lightning Trainer.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality used by the transformer model.
    n_head : int
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int
        The dimensionality of the fully connected layers in the transformer
        model.
    n_layers : int
        The number of transformer layers.
    dropout : float
        The dropout probability for all layers.
    dim_intensity : Optional[int]
        The number of features to use for encoding peak intensity. The remaining
        (``dim_model - dim_intensity``) are reserved for encoding the m/z value.
        If ``None``, the intensity will be projected up to ``dim_model`` using a
        linear layer, then summed with the m/z encoding for each peak.
    max_length : int
        The maximum peptide length to decode.
    residues : Union[Dict[str, float], str]
        The amino acid dictionary and their masses. By default ("canonical) this
        is only the 20 canonical amino acids, with cysteine carbamidomethylated.
        If "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int
        The maximum precursor charge to consider.
    precursor_mass_tol : float, optional
        The maximum allowable precursor mass tolerance (in ppm) for correct
        predictions.
    isotope_error_range : Tuple[int, int]
        Take into account the error introduced by choosing a non-monoisotopic
        peak for fragmentation by not penalizing predicted precursor m/z's that
        fit the specified isotope error:
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
    tb_summarywriter : Optional[str]
        Folder path to record performance metrics during training. If ``None``,
        don't use a ``SummaryWriter``.
    train_label_smoothing : float
        Smoothing factor when calculating the training loss.
    warmup_iters : int
        The number of iterations for the linear warm-up of the learning rate.
    cosine_schedule_period_iters : int
        The number of iterations for the cosine half period of the learning rate.
    out_writer : Optional[str]
        The output writer for the prediction results.
    calculate_precision : bool
        Calculate the validation set precision during training.
        This is expensive.
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
            dim_intensity: Optional[int] = None,
            max_length: int = 100,
            residues: Union[Dict[str, float], str] = "canonical",
            max_charge: int = 5,
            precursor_mass_tol: float = 50,
            isotope_error_range: Tuple[int, int] = (0, 1),
            min_peptide_len: int = 6,
            n_beams: int = 1,
            top_match: int = 1,
            n_log: int = 10,
            tb_summarywriter: Optional[
                torch.utils.tensorboard.SummaryWriter
            ] = None,
            train_label_smoothing: float = 0.01,
            warmup_iters: int = 100_000,
            cosine_schedule_period_iters: int = 600_000,
            out_writer: Optional[ms_io.MztabWriter] = None,
            calculate_precision: bool = False,
            random_seed: int = 42,
            **kwargs: Dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build the model.
        self.encoder = SpectrumEncoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            dim_intensity=dim_intensity,
        )
        self.decoder = PeptideDecoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            residues=residues,
            max_charge=max_charge,
        )
        self.softmax = torch.nn.Softmax(2)
        self.celoss = torch.nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=train_label_smoothing
        )
        self.val_celoss = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.rng = np.random.RandomState(random_seed)


        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)

        # Optimizer settings.
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        # `kwargs` will contain additional arguments as well as unrecognized
        # arguments, including deprecated ones. Remove the deprecated ones.
        for k in config._config_deprecated:
            kwargs.pop(k, None)
            warnings.warn(
                f"Deprecated hyperparameter '{k}' removed from the model.",
                DeprecationWarning,
            )
        self.opt_kwargs = kwargs

        # Data properties.
        self.max_length = max_length
        self.residues = residues
        self.precursor_mass_tol = precursor_mass_tol
        self.isotope_error_range = isotope_error_range
        self.min_peptide_len = min_peptide_len
        self.n_beams = n_beams
        self.top_match = top_match
        self.peptide_mass_calculator = depthcharge.masses.PeptideMass(
            self.residues
        )
        self.stop_token = self.decoder._aa2idx["$"]

        # Optimization 1: Precompute token masses - don't specify device yet
        self.token_masses = torch.zeros(self.decoder.vocab_size + 1)
        for aa, mass in self.peptide_mass_calculator.masses.items():
            if aa in self.decoder._aa2idx:
                idx = self.decoder._aa2idx[aa]
                self.token_masses[idx] = mass

        # Precompute negative mass tokens and N-terminal residue tokens - store as lists
        self.neg_mass_tokens_list = []
        for aa, mass in self.peptide_mass_calculator.masses.items():
            if mass < 0 and aa in self.decoder._aa2idx:
                self.neg_mass_tokens_list.append(self.decoder._aa2idx[aa])

        self.n_term_tokens_list = []
        for aa in self.peptide_mass_calculator.masses:
            if aa.startswith(("+", "-")) and aa in self.decoder._aa2idx:
                self.n_term_tokens_list.append(self.decoder._aa2idx[aa])

        # Isotope mass shift precomputation - don't specify device yet
        self.isotope_mass_shifts = torch.tensor(
            [1.00335 / charge for charge in range(1, max_charge + 1)]
        )

        # Optimization 4: Mass calculation cache
        self._peptide_mass_cache = {}

        # Logging.
        self.calculate_precision = calculate_precision
        self.n_log = n_log
        self._history = []
        if tb_summarywriter is not None:
            self.tb_summarywriter = SummaryWriter(tb_summarywriter)
        else:
            self.tb_summarywriter = tb_summarywriter

        # Output writer during predicting.
        self.out_writer = out_writer

    def forward(
            self, spectra: torch.Tensor, precursors: torch.Tensor
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        Predict peptide sequences for a batch of MS/MS spectra.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        pred_peptides : List[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide predictions. A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        return self.beam_search_decode(
            spectra.to(self.encoder.device),
            precursors.to(self.decoder.device),
        )

    def beam_search_decode(
            self, spectra: torch.Tensor, precursors: torch.Tensor
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        Beam search decoding of the spectrum predictions. Optimized version.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        pred_peptides : List[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide prediction(s). A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        # Ensure all tensors are on the correct device
        device = spectra.device

        # First time called, convert lists to tensors and move to correct device
        if not hasattr(self, 'neg_mass_tokens'):
            self.neg_mass_tokens = torch.tensor(self.neg_mass_tokens_list, device=device)
            self.n_term_tokens = torch.tensor(self.n_term_tokens_list, device=device)

        # Move all precomputed tensors to the correct device
        self.token_masses = self.token_masses.to(device)
        self.isotope_mass_shifts = self.isotope_mass_shifts.to(device)

        memories, mem_masks = self.encoder(spectra)

        # Sizes.
        batch = spectra.shape[0]  # B
        length = self.max_length + 1  # L
        vocab = self.decoder.vocab_size + 1  # V
        beam = self.n_beams  # S

        # Initialize scores and tokens.
        scores = torch.full(
            size=(batch, length, vocab, beam), fill_value=torch.nan
        )
        scores = scores.type_as(spectra)
        tokens = torch.zeros(batch, length, beam, dtype=torch.int64, device=device)

        # Create cache for decoded beams.
        pred_cache = collections.OrderedDict((i, []) for i in range(batch))

        # Get the first prediction.
        pred, _ = self.decoder(None, precursors, memories, mem_masks)
        tokens[:, 0, :] = torch.topk(pred[:, 0, :], beam, dim=1)[1]
        scores[:, :1, :, :] = einops.repeat(pred, "B L V -> B L V S", S=beam)

        # Optimization 5: Initialize beam mass accumulation tensor
        beam_masses = torch.zeros(batch * beam, device=device)
        # Update masses for first tokens
        first_tokens = einops.rearrange(tokens[:, 0, :], "B S -> (B S)")
        beam_masses += self.token_masses[first_tokens]

        # Make all tensors the right shape for decoding.
        precursors = einops.repeat(precursors, "B L -> (B S) L", S=beam)
        mem_masks = einops.repeat(mem_masks, "B L -> (B S) L", S=beam)
        memories = einops.repeat(memories, "B L V -> (B S) L V", S=beam)
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")

        # The main decoding loop.
        for step in range(0, self.max_length):
            # Terminate beams exceeding the precursor m/z tolerance and track
            # all finished beams (either terminated or stop token predicted).
            finished_beams, beam_fits_precursor, discarded_beams = self._finish_beams(
                tokens, precursors, step, beam_masses
            )
            # Cache peptide predictions from the finished beams (but not the
            # discarded beams).
            self._cache_finished_beams(
                tokens,
                scores,
                step,
                finished_beams & ~discarded_beams,
                beam_fits_precursor,
                pred_cache,
            )

            # Stop decoding when all current beams have been finished.
            # Continue with beams that have not been finished and not discarded.
            finished_beams |= discarded_beams
            if finished_beams.all():
                break
            # Update the scores.
            scores[~finished_beams, : step + 2, :], _ = self.decoder(
                tokens[~finished_beams, : step + 1],
                precursors[~finished_beams, :],
                memories[~finished_beams, :, :],
                mem_masks[~finished_beams, :],
            )
            # Find the top-k beams with the highest scores and continue decoding
            # those.
            tokens, scores, beam_masses = self._get_topk_beams(
                tokens, scores, finished_beams, batch, step + 1, beam_masses
            )

        # Return the peptide with the highest confidence score, within the
        # precursor m/z tolerance if possible.
        return list(self._get_top_peptide(pred_cache))

    def _finish_beams(
            self,
            tokens: torch.Tensor,
            precursors: torch.Tensor,
            step: int,
            beam_masses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Track all beams that have been finished, either by predicting the stop
        token or because they were terminated due to exceeding the precursor
        m/z tolerance. Optimized version uses vectorized operations and reduces
        detokenize operations.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
        precursors : torch.Tensor of size (n_spectra * n_beams, 3)
            The measured precursor mass, charge, and m/z for each beam.
        step : int
            Index of the current decoding step.
        beam_masses : torch.Tensor
            Current accumulated masses for each beam.

        Returns
        -------
        finished_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams have been
            finished.
        beam_fits_precursor: torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating if current beams are within precursor m/z
            tolerance.
        discarded_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams should be
            discarded (e.g. because they were predicted to end but violate the
            minimum peptide length).
        """
        device = tokens.device

        # Initialize result tensors
        batch_size = tokens.shape[0]
        finished_beams = torch.zeros(batch_size, dtype=torch.bool, device=device)
        beam_fits_precursor = torch.zeros(batch_size, dtype=torch.bool, device=device)
        discarded_beams = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Vectorized operations: find beams with stop tokens
        ends_stop_token = tokens[:, step] == self.stop_token
        finished_beams[ends_stop_token] = True

        # Vectorized operations: find beams with padding tokens
        discarded_beams[tokens[:, step] == 0] = True

        # Handle N-terminal modifications special case (vectorized)
        if step > 1:  # Only relevant for longer predictions
            dim0 = torch.arange(tokens.shape[0], device=device)
            final_pos = torch.full((ends_stop_token.shape[0],), step, device=device)
            final_pos[ends_stop_token] = step - 1

            # Multiple N-terminal modifications
            multiple_mods = torch.isin(
                tokens[dim0, final_pos], self.n_term_tokens
            ) & torch.isin(tokens[dim0, final_pos - 1], self.n_term_tokens)

            # N-terminal modifications occur at an internal position
            position_range = torch.arange(tokens.shape[1], device=device)
            mask = (final_pos - 1).unsqueeze(1) >= position_range
            masked_tokens = torch.where(mask, tokens, torch.zeros_like(tokens))
            internal_mods = torch.any(
                torch.isin(masked_tokens, self.n_term_tokens), dim=1
            )

            discarded_beams[multiple_mods | internal_mods] = True

        # Check sequence length (vectorized)
        pred_lengths = torch.zeros(batch_size, dtype=torch.int64, device=device)
        for i in range(batch_size):
            if discarded_beams[i]:
                continue

            sequence_length = step + 1
            # Calculate length not including stop token
            if ends_stop_token[i]:
                sequence_length -= 1
            pred_lengths[i] = sequence_length

        # Discard beams that are too short (vectorized)
        too_short = (finished_beams & (pred_lengths < self.min_peptide_len))
        discarded_beams[too_short] = True

        # Handle mass checks
        # Get relevant information for each beam
        precursor_charges = precursors[:, 1]
        precursor_mzs = precursors[:, 2]

        # Check mass for each beam (try to vectorize where possible)
        for i in range(batch_size):
            if discarded_beams[i]:
                continue

            # Get current beam mass and precursor info
            curr_mass = beam_masses[i]
            charge = precursor_charges[i].item()  # Must use item() here
            obs_mz = precursor_mzs[i].item()  # Must use item() here

            # Check if current mass is within tolerance
            calc_mz = curr_mass / charge + 1.007276  # Proton mass
            matches_precursor = False
            exceeds_precursor = False

            # Early termination check: check isotope error range
            for isotope in range(self.isotope_error_range[0], self.isotope_error_range[1] + 1):
                delta_mass_ppm = self._calc_mass_error_tensor(calc_mz, obs_mz, charge, isotope)

                if abs(delta_mass_ppm.item()) < self.precursor_mass_tol:
                    matches_precursor = True
                    break

            # If current doesn't match but not finished, try adding negative mass AA to see if could match
            if not matches_precursor and not finished_beams[i]:
                for neg_token in self.neg_mass_tokens:
                    neg_mass = self.token_masses[neg_token]
                    adjusted_mz = (curr_mass + neg_mass) / charge + 1.007276

                    # Check if adding negative mass AA would match
                    for isotope in range(self.isotope_error_range[0], self.isotope_error_range[1] + 1):
                        delta_ppm = self._calc_mass_error_tensor(adjusted_mz, obs_mz, charge, isotope)

                        if abs(delta_ppm.item()) < self.precursor_mass_tol:
                            # Can match, but don't terminate now
                            matches_precursor = False
                            exceeds_precursor = False
                            break

                    if matches_precursor or not exceeds_precursor:
                        break

                # If all negative mass AA can't bring mass back to tolerance range
                if not matches_precursor:
                    exceeds_precursor = True
            else:
                # Finished beam, check if exceeds tolerance
                exceeds_precursor = not matches_precursor

            # Update result tensors
            beam_fits_precursor[i] = matches_precursor
            if exceeds_precursor:
                finished_beams[i] = True

        return finished_beams, beam_fits_precursor, discarded_beams

    def _cache_finished_beams(
            self,
            tokens: torch.Tensor,
            scores: torch.Tensor,
            step: int,
            beams_to_cache: torch.Tensor,
            beam_fits_precursor: torch.Tensor,
            pred_cache: Dict[int, List[Tuple[float, float, np.ndarray, torch.Tensor]]],
    ):
        """
        Cache terminated beams. Optimized version reduces detokenize operations.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
        scores : torch.Tensor of shape (n_spectra * n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all spectra.
        step : int
            Index of the current decoding step.
        beams_to_cache : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams are ready for caching.
        beam_fits_precursor: torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the beams are within the
            precursor m/z tolerance.
        pred_cache : Dict[int, List[Tuple[float, float, np.ndarray, torch.Tensor]]]
            Priority queue with finished beams for each spectrum, ordered by
            peptide score. For each finished beam, a tuple with the (negated)
            peptide score, a random tie-breaking float, the amino acid-level
            scores, and the predicted tokens is stored.
        """
        # Only process if there are beams to cache
        if not torch.any(beams_to_cache):
            return

        # Extract indices of beams to cache
        cache_indices = torch.nonzero(beams_to_cache).squeeze(-1)

        for idx in cache_indices:
            # Find spectrum starting index
            spec_idx = idx.item() // self.n_beams

            # Extract predicted tokens
            pred_tokens = tokens[idx][: step + 1]
            # Check for stop token
            has_stop_token = pred_tokens[-1] == self.stop_token
            pred_peptide = pred_tokens[:-1] if has_stop_token else pred_tokens

            # Check if already predicted the same peptide
            duplicate = False
            for pred_cached in pred_cache[spec_idx]:
                if torch.equal(pred_cached[-1], pred_peptide):
                    duplicate = True
                    break

            if duplicate:
                continue

            # Get amino acid level scores
            smx = self.softmax(scores[idx: idx + 1, : step + 1, :])
            aa_scores = smx[0, range(len(pred_tokens)), pred_tokens].cpu().detach().numpy()

            # Add missing stop token explicit 0 score
            if not has_stop_token:
                aa_scores = np.append(aa_scores, 0)

            # Calculate updated amino acid level and peptide scores
            aa_scores, peptide_score = self._aa_pep_score(
                aa_scores, beam_fits_precursor[idx].item()
            )

            # Omit stop token from amino acid level scores
            aa_scores = aa_scores[:-1]

            # Add prediction to cache (minimum priority queue, max beams elements)
            if len(pred_cache[spec_idx]) < self.n_beams:
                heapadd = heapq.heappush
            else:
                heapadd = heapq.heappushpop

            heapadd(
                pred_cache[spec_idx],
                (
                    peptide_score,
                    self.rng.random_sample(),
                    aa_scores,
                    torch.clone(pred_peptide),
                ),
            )

    def _get_topk_beams(
            self,
            tokens: torch.tensor,
            scores: torch.tensor,
            finished_beams: torch.tensor,
            batch: int,
            step: int,
            beam_masses: torch.tensor,
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Find the top-k beams with the highest scores and continue decoding
        those. Optimized version uses vectorized operations and updates beam_masses.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
        scores : torch.Tensor of shape (n_spectra * n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all spectra.
        finished_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams are ready for caching.
        batch: int
            Number of spectra in the batch.
        step : int
            Index of the next decoding step.
        beam_masses : torch.Tensor of shape (n_spectra * n_beams)
            Current accumulated masses for each beam.

        Returns
        -------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Updated predicted amino acid tokens for all beams and all spectra.
        scores : torch.Tensor of shape (n_spectra * n_beams, max_length, n_amino_acids)
            Updated scores for the predicted amino acid tokens for all beams and all spectra.
        beam_masses : torch.Tensor of shape (n_spectra * n_beams)
            Updated accumulated masses for each beam.
        """
        device = tokens.device
        beam = self.n_beams  # S
        vocab = self.decoder.vocab_size + 1  # V

        # Reshape to group by spectrum (B for "batch").
        tokens = einops.rearrange(tokens, "(B S) L -> B L S", S=beam)
        scores = einops.rearrange(scores, "(B S) L V -> B L V S", S=beam)
        beam_masses = einops.rearrange(beam_masses, "(B S) -> B S", S=beam)

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
        step_scores = torch.zeros(batch, step + 1, beam * vocab, device=device).type_as(
            scores
        )
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

        # Figure out the top K decodings.
        _, top_idx = torch.topk(step_scores.nanmean(dim=1) * active_mask, beam)

        # Extract top K decoding indices
        top_idx_cpu = top_idx.cpu().numpy()
        v_idx = top_idx_cpu // beam
        s_idx = top_idx_cpu % beam

        # Create new tokens and scores
        tokens_new = torch.zeros_like(tokens)
        scores_new = torch.zeros_like(scores)
        beam_masses_new = torch.zeros_like(beam_masses)

        for b in range(batch):
            for s in range(beam):
                old_s = s_idx[b, s]
                new_token = torch.tensor(v_idx[b, s], device=device)

                # Copy previous step tokens
                tokens_new[b, :step, s] = tokens[b, :step, old_s]
                # Add new token
                tokens_new[b, step, s] = new_token

                # Copy scores
                for l in range(step + 1):
                    scores_new[b, l, :, s] = scores[b, l, :, old_s]

                # Update masses
                beam_masses_new[b, s] = beam_masses[b, old_s] + self.token_masses[new_token]

        # Reshape back to flattened shape
        tokens = einops.rearrange(tokens_new, "B L S -> (B S) L")
        scores = einops.rearrange(scores_new, "B L V S -> (B S) L V")
        beam_masses = einops.rearrange(beam_masses_new, "B S -> (B S)")

        return tokens, scores, beam_masses

    def _get_top_peptide(
            self,
            pred_cache: Dict[int, List[Tuple[float, float, np.ndarray, torch.Tensor]]],
    ) -> Iterable[List[Tuple[float, np.ndarray, str]]]:
        """
        Return the peptide with the highest confidence score for each spectrum.
        Optimized version only performs detokenize in the final output.

        Parameters
        ----------
        pred_cache : Dict[int, List[Tuple[float, float, np.ndarray, torch.Tensor]]]
            Priority queue with finished beams for each spectrum, ordered by
            peptide score. For each finished beam, a tuple with the peptide
            score, a random tie-breaking float, the amino acid-level scores,
            and the predicted tokens is stored.

        Returns
        -------
        pred_peptides : Iterable[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide prediction(s). A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        for peptides in pred_cache.values():
            if len(peptides) > 0:
                yield [
                    (
                        pep_score,
                        aa_scores[::-1] if self.decoder.reverse else aa_scores,
                        "".join(self.decoder.detokenize(pred_tokens)),  # Only detokenize here
                    )
                    for pep_score, _, aa_scores, pred_tokens in heapq.nlargest(
                        self.top_match, peptides
                    )
                ]
            else:
                yield []

    def _aa_pep_score(
            self,
            aa_scores: np.ndarray,
            fits_precursor_mz: bool
    ) -> Tuple[np.ndarray, float]:
        """
        Calculate amino acid and peptide-level confidence score from the raw amino
        acid scores.

        The peptide score is the mean of the raw amino acid scores. The amino acid
        scores are the mean of the raw amino acid scores and the peptide score.

        Parameters
        ----------
        aa_scores : np.ndarray
            Amino acid level confidence scores.
        fits_precursor_mz : bool
            Flag indicating whether the prediction fits the precursor m/z filter.

        Returns
        -------
        aa_scores : np.ndarray
            The amino acid scores.
        peptide_score : float
            The peptide score.
        """
        peptide_score = np.mean(aa_scores)
        aa_scores = (aa_scores + peptide_score) / 2
        if not fits_precursor_mz:
            peptide_score -= 1
        return aa_scores, peptide_score

    def _calc_mass_error_tensor(self, calc_mz, obs_mz, charge, isotope=0):
        """
        GPU tensor version of mass error calculation.

        Parameters
        ----------
        calc_mz : float or tensor
            The theoretical m/z.
        obs_mz : float or tensor
            The observed m/z.
        charge : int
            The charge.
        isotope : int
            Correct for the given number of C13 isotopes (default: 0).

        Returns
        -------
        float or tensor
            The mass error in ppm.
        """
        isotope_shift = isotope * 1.00335 / charge
        return (calc_mz - (obs_mz - isotope_shift)) / obs_mz * 10 ** 6

    def _forward_step(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        sequences: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward learning step.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.
        sequences : List[str] of length n_spectra
            The partial peptide sequences to predict.

        Returns
        -------
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        tokens : torch.Tensor of shape (n_spectra, length)
            The predicted tokens for each spectrum.
        """
        return self.decoder(sequences, precursors, *self.encoder(spectra))

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str]],
        *args,
        mode: str = "train",
    ) -> torch.Tensor:
        """
        A single training step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences as torch Tensors.
        mode : str
            Logging key to describe the current stage.

        Returns
        -------
        torch.Tensor
            The loss of the training step.
        """
        pred, truth = self._forward_step(*batch)
        pred = pred[:, :-1, :].reshape(-1, self.decoder.vocab_size + 1)
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
        )
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[str]], *args
    ) -> torch.Tensor:
        """
        A single validation step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences.

        Returns
        -------
        torch.Tensor
            The loss of the validation step.
        """
        # Record the loss.
        loss = self.training_step(batch, mode="valid")
        if not self.calculate_precision:
            return loss

        # Calculate and log amino acid and peptide match evaluation metrics from
        # the predicted peptides.
        peptides_pred, peptides_true = [], batch[2]
        for spectrum_preds in self.forward(batch[0], batch[1]):
            for _, _, pred in spectrum_preds:
                peptides_pred.append(pred)

        aa_precision, _, pep_precision = evaluate.aa_match_metrics(
            *evaluate.aa_match_batch(
                peptides_true,
                peptides_pred,
                self.decoder._peptide_mass.masses,
            )
        )
        log_args = dict(on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "Peptide precision at coverage=1",
            pep_precision,
            **log_args,
        )
        self.log(
            "AA precision at coverage=1",
            aa_precision,
            **log_args,
        )
        return loss

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], *args
    ) -> List[Tuple[np.ndarray, float, float, str, float, np.ndarray]]:
        """
        A single prediction step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            spectrum identifiers as torch Tensors.

        Returns
        -------
        predictions: List[Tuple[np.ndarray, float, float, str, float, np.ndarray]]
            Model predictions for the given batch of spectra containing spectrum
            ids, precursor information, peptide sequences as well as peptide
            and amino acid-level confidence scores.
        """
        predictions = []
        for (
            precursor_charge,
            precursor_mz,
            spectrum_i,
            spectrum_preds,
        ) in zip(
            batch[1][:, 1].cpu().detach().numpy(),
            batch[1][:, 2].cpu().detach().numpy(),
            batch[2],
            self.forward(batch[0], batch[1]),
        ):
            for peptide_score, aa_scores, peptide in spectrum_preds:
                predictions.append(
                    (
                        spectrum_i,
                        precursor_charge,
                        precursor_mz,
                        peptide,
                        peptide_score,
                        aa_scores,
                    )
                )

        return predictions

    def on_train_epoch_end(self) -> None:
        """
        Log the training loss at the end of each epoch.
        """
        train_loss = self.trainer.callback_metrics["train_CELoss"].detach()
        metrics = {
            "step": self.trainer.global_step,
            "train": train_loss.item(),
        }
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
                callback_metrics["AA precision at coverage=1"].detach().item()
            )
            metrics["valid_pep_precision"] = (
                callback_metrics["Peptide precision at coverage=1"]
                .detach()
                .item()
            )
        self._history.append(metrics)
        self._log_history()

    def on_predict_batch_end(
        self,
        outputs: List[Tuple[np.ndarray, List[str], torch.Tensor]],
        *args,
    ) -> None:
        """
        Write the predicted peptide sequences and amino acid scores to the
        output file.
        """
        if self.out_writer is None:
            return
        # Triply nested lists: results -> batch -> step -> spectrum.
        for (
            spectrum_i,
            charge,
            precursor_mz,
            peptide,
            peptide_score,
            aa_scores,
        ) in outputs:
            if len(peptide) == 0:
                continue
            self.out_writer.psms.append(
                (
                    peptide,
                    tuple(spectrum_i),
                    peptide_score,
                    charge,
                    precursor_mz,
                    self.peptide_mass_calculator.mass(peptide, charge),
                    ",".join(list(map("{:.5f}".format, aa_scores))),
                ),
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
            if self.tb_summarywriter is not None:
                for descr, key in [
                    ("loss/train_crossentropy_loss", "train"),
                    ("loss/val_crossentropy_loss", "valid"),
                    ("eval/val_pep_precision", "valid_pep_precision"),
                    ("eval/val_aa_precision", "valid_aa_precision"),
                ]:
                    metric_value = metrics.get(key, np.nan)
                    if not np.isnan(metric_value):
                        self.tb_summarywriter.add_scalar(
                            descr, metric_value, metrics["step"]
                        )

    def _calculate_peptide_mass(self, tokens, charge):
        """
        Cached method for calculating peptide mass.

        Parameters
        ----------
        tokens : torch.Tensor
            Tensor of peptide token indices.
        charge : int
            The charge of the peptide.

        Returns
        -------
        float
            The calculated peptide mass.
        """
        cache_key = tuple(tokens.cpu().numpy().tolist()) + (charge,)
        if cache_key not in self._peptide_mass_cache:
            # Calculate mass directly from tokens
            mass = 0.0
            for token in tokens:
                mass += self.token_masses[token].item()
            self._peptide_mass_cache[cache_key] = mass

        return self._peptide_mass_cache[cache_key]

    def configure_optimizers(
        self,
    ) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
        """
        Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for training.

        Returns
        -------
        Tuple[torch.optim.Optimizer, Dict[str, Any]]
            The initialized Adam optimizer and its learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), **self.opt_kwargs)
        # Apply learning rate scheduler per step.
        lr_scheduler = CosineWarmupScheduler(
            optimizer, self.warmup_iters, self.cosine_schedule_period_iters
        )
        return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm-up followed by cosine shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup_iters : int
        The number of iterations for the linear warm-up of the learning rate.
    cosine_schedule_period_iters : int
        The number of iterations for the cosine half period of the learning rate.
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


import numba


@numba.jit(nopython=True)
def _calc_mass_error_numba(calc_mz, obs_mz, charge, isotope=0):
    """
    Calculate the mass error in ppm between the theoretical m/z and the observed
    m/z, optionally accounting for an isotopologue mismatch. Numba JIT compiled version.

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
    return (calc_mz - (obs_mz - isotope * 1.00335 / charge)) / obs_mz * 10 ** 6


def _calc_mass_error(
    calc_mz: float, obs_mz: float, charge: int, isotope: int = 0
) -> float:
    """
    Calculate the mass error in ppm between the theoretical m/z and the observed
    m/z, optionally accounting for an isotopologue mismatch.

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


if __name__ == "__main__":
    import time
    import torch

    # Prefer GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize model with beam size of 1
    model = Spec2Pep(n_beams=1)
    model.to(device)

    # Create random input data:
    # 10 spectra, each with 100 peaks containing [m/z, intensity]
    n_spectra = 10
    n_peaks = 100
    spectra = torch.rand(n_spectra, n_peaks, 2, device=device)

    # Create precursor data: [precursor_mass, precursor_charge, precursor_mz]
    precursors = torch.zeros(n_spectra, 3, device=device)
    precursors[:, 0] = torch.rand(n_spectra, device=device) * 1000  # precursor mass
    precursors[:, 1] = torch.randint(1, model.hparams.max_charge + 1, (n_spectra,), device=device)  # charge
    precursors[:, 2] = torch.rand(n_spectra, device=device) * 1000  # precursor m/z

    # Warm-up run to avoid initialization overhead in timing
    with torch.no_grad():
        _ = model.forward(spectra, precursors)

    # Perform multiple runs for stable timing results
    n_runs = 10
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()  # Ensure GPU operations are complete
            start = time.perf_counter()
            _ = model.forward(spectra, precursors)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    # Calculate average processing time in milliseconds per spectrum
    avg_time = sum(times) / n_runs
    avg_time_per_spectrum = avg_time / n_spectra
    print(f"Average processing time per spectrum: {avg_time_per_spectrum * 1000:.2f} ms")