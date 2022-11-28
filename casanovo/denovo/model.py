"""A de novo peptide sequencing model."""
import heapq
import logging
import operator
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import depthcharge.masses
import einops
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.tensorboard import SummaryWriter
from depthcharge.components import ModelMixin, PeptideDecoder, SpectrumEncoder

from . import evaluate
from ..data import ms_io

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
    custom_encoder : Optional[Union[SpectrumEncoder, PairedSpectrumEncoder]]
        A pretrained encoder to use. The ``dim_model`` of the encoder must be
        the same as that specified by the ``dim_model`` parameter here.
    max_length : int
        The maximum peptide length to decode.
    residues: Union[Dict[str, float], str]
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
    n_beams: int
        Number of beams used during beam search decoding.
    n_log : int
        The number of epochs to wait between logging messages.
    tb_summarywriter: Optional[str]
        Folder path to record performance metrics during training. If ``None``,
        don't use a ``SummaryWriter``.
    warmup_iters: int
        The number of warm up iterations for the learning rate scheduler.
    max_iters: int
        The total number of iterations for the learning rate scheduler.
    out_writer: Optional[str]
        The output writer for the prediction results.
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
        custom_encoder: Optional[SpectrumEncoder] = None,
        max_length: int = 100,
        residues: Union[Dict[str, float], str] = "canonical",
        max_charge: int = 5,
        precursor_mass_tol: float = 50,
        isotope_error_range: Tuple[int, int] = (0, 1),
        n_beams: int = 5,
        n_log: int = 10,
        tb_summarywriter: Optional[
            torch.utils.tensorboard.SummaryWriter
        ] = None,
        warmup_iters: int = 100_000,
        max_iters: int = 600_000,
        out_writer: Optional[ms_io.MztabWriter] = None,
        **kwargs: Dict,
    ):
        super().__init__()

        # Build the model.
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
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
        self.celoss = torch.nn.CrossEntropyLoss(ignore_index=0)
        # Optimizer settings.
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.opt_kwargs = kwargs

        # Data properties.
        self.max_length = max_length
        self.residues = residues
        self.precursor_mass_tol = precursor_mass_tol
        self.isotope_error_range = isotope_error_range
        self.n_beams = n_beams
        self.peptide_mass_calculator = depthcharge.masses.PeptideMass(
            self.residues
        )
        self.stop_token = self.decoder._aa2idx["$"]

        # Logging.
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
    ) -> Tuple[List[List[str]], torch.Tensor]:
        """
        Predict peptide sequences for a batch of MS/MS spectra.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all of the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        peptides : List[List[str]]
            The predicted peptide sequences for each spectrum.
        aa_scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        """
        aa_scores, tokens = self.beam_search_decode(
            spectra.to(self.encoder.device),
            precursors.to(self.decoder.device),
        )

        return [self.decoder.detokenize(t) for t in tokens], aa_scores

    def beam_search_decode(
        self, spectra: torch.Tensor, precursors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Beam search decoding of the spectrum predictions.

        Return the highest scoring peptide, within the precursor m/z tolerance
        whenever possible.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all of the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        scores : torch.Tensor of shape (n_spectra, max_length, n_amino_acids)
            The individual amino acid scores for each prediction.
        tokens : torch.Tensor of shape (n_spectra, max_length)
            The predicted tokens for each spectrum.
        """
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
        tokens = torch.zeros(batch, length, beam, dtype=torch.int64)
        tokens = tokens.to(self.encoder.device)
        # Keep track whether terminated beams have fitting precursor m/z.
        beam_fits_prec_tol = torch.zeros(batch * beam, dtype=torch.bool)

        # Create cache for decoded beams.
        (
            cache_scores,
            cache_tokens,
            cache_next_idx,
            cache_pred_seq,
            cache_pred_score,
        ) = self._create_beamsearch_cache(scores, tokens)

        # Get the first prediction.
        pred, _ = self.decoder(None, precursors, memories, mem_masks)
        tokens[:, 0, :] = torch.topk(pred[:, 0, :], beam, dim=1)[1]
        scores[:, :1, :, :] = einops.repeat(pred, "B L V -> B L V S", S=beam)

        # Make all tensors the right shape for decoding.
        precursors = einops.repeat(precursors, "B L -> (B S) L", S=beam)
        mem_masks = einops.repeat(mem_masks, "B L -> (B S) L", S=beam)
        memories = einops.repeat(memories, "B L V -> (B S) L V", S=beam)
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")

        # The main decoding loop.
        for i in range(1, self.max_length + 1):
            # Terminate beams exceeding precursor m/z tolerance and track all
            # terminated beams.
            finished_beams_idx, tokens = self._terminate_finished_beams(
                tokens, precursors, beam_fits_prec_tol, i
            )
            # Cache terminated beams, group and order by fitting precursor m/z
            # and confidence score.
            self._cache_finished_beams(
                finished_beams_idx,
                cache_next_idx,
                cache_pred_seq,
                cache_pred_score,
                cache_tokens,
                cache_scores,
                tokens,
                scores,
                beam_fits_prec_tol,
                i,
            )
            # Reset precursor tolerance status of all beams.
            beam_fits_prec_tol = torch.zeros(batch * beam, dtype=torch.bool)

            # Stop decoding when all current beams are terminated.
            decoded = (tokens == self.stop_token).any(axis=1)
            if decoded.all():
                break
            # Update the scores.
            scores[~decoded, : i + 1, :], _ = self.decoder(
                tokens[~decoded, :i],
                precursors[~decoded, :],
                memories[~decoded, :, :],
                mem_masks[~decoded, :],
            )
            # Find top-k beams with highest scores and continue decoding those.
            scores, tokens = self._get_topk_beams(scores, tokens, batch, i)

        # Return the peptide with the highest confidence score, within the
        # precursor m/z tolerance if possible.
        output_tokens, output_scores = self._get_top_peptide(
            cache_pred_score, cache_tokens, cache_scores, batch
        )
        return self.softmax(output_scores), output_tokens

    def _create_beamsearch_cache(
        self, scores: torch.Tensor, tokens: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[int, int],
        Dict[int, Set[str]],
        Dict[int, List[List[Tuple[float, int]]]],
    ]:
        """
        Create cache tensor and dictionary to store and group terminated beams.

        Parameters
        ----------
        scores : torch.Tensor of shape
        (n_spectra, max_length, n_amino_acids, n_beams)
            Output scores of the model.
        tokens : torch.Tensor of size (n_spectra, max_length, n_beams)
            Output token of the model corresponding to amino acid sequences.

        Returns
        -------
        cache_scores : torch.Tensor of shape
        (n_spectra * n_beams, max_length, n_amino_acids)
            The score for each amino acid in cached peptides.
        cache_tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            The token for each amino acid in cached peptides.
        cache_next_idx : Dict[int, int]
            Next available tensor index to cache peptides for each spectrum.
        cache_pred_seq : Dict[int, Set[torch.Tensor]]
            Set of decoded peptide tokens for each spectrum.
        cache_pred_score : Dict[int, List[List[Tuple[float, int]]]
            Confidence score for each decoded peptide, separated as
            precursor m/z fitting vs not, for each spectrum.
        """
        batch, beam = scores.shape[0], scores.shape[-1]

        # Cache terminated beams and their scores.
        cache_scores = einops.rearrange(scores.clone(), "B L V S -> (B S) L V")
        cache_tokens = einops.rearrange(tokens.clone(), "B L S -> (B S) L")

        # Keep pointer to free rows in the cache and already cached predictions.
        cache_next_idx = {i: i * beam for i in range(batch)}
        # Keep already decoded peptides to avoid duplicates in cache.
        cache_pred_seq = {i: set() for i in range(batch)}
        # Store peptide scores to replace lower score peptides in cache with
        # higher score peptides during decoding.
        cache_pred_score = {i: [[], []] for i in range(batch)}

        return (
            cache_scores,
            cache_tokens,
            cache_next_idx,
            cache_pred_seq,
            cache_pred_score,
        )

    def _terminate_finished_beams(
        self,
        tokens: torch.Tensor,
        precursors: torch.Tensor,
        is_beam_prec_fit: torch.Tensor,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Terminate beams exceeding the precursor m/z tolerance.

        Track all terminated beams.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Output token of the model corresponding to amino acid sequences.
        precursors : torch.Tensor of size (n_spectra * n_beams, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.
        is_beam_prec_fit: torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating if current beams are within precursor m/z
            tolerance.
        idx : int
            Index to be considered in the current decoding step.

        Returns
        -------
        finished_beams_idx : torch.Tensor
            Indices of all finished beams on tokens tensor.
        tokens : torch.Tensor of size (n_spectra * n_beams, max_length)
            Output token of the model corresponding to amino acid sequences.
        """
        # Check for tokens with a negative mass (i.e. neutral loss).
        aa_neg_mass = [None]
        for aa, mass in self.peptide_mass_calculator.masses.items():
            if mass < 0:
                aa_neg_mass.append(aa)

        # Terminate beams that exceed the precursor m/z.
        for beam_i in range(len(tokens)):
            # Check only non-terminated beams.
            if self.stop_token not in tokens[beam_i]:
                # Finish if dummy was predicted at the previous step.
                if tokens[beam_i][idx - 1] == 0:
                    tokens[beam_i][idx - 1] = self.stop_token
                # Terminate the beam if it exceeds the precursor m/z tolerance.
                else:
                    precursor_charge = precursors[beam_i, 1].item()
                    precursor_mz = precursors[beam_i, 2].item()
                    # Only terminate if the m/z difference cannot be corrected
                    # anymore by a subsequently predicted AA with negative mass.
                    matches_precursor_mz = exceeds_precursor_mz = False
                    for aa in aa_neg_mass:
                        peptide = self.decoder.detokenize(tokens[beam_i][:idx])
                        if aa is not None:
                            peptide.append(aa)
                        try:
                            calc_mz = self.peptide_mass_calculator.mass(
                                seq=peptide, charge=precursor_charge
                            )
                            delta_mass_ppm = [
                                _calc_mass_error(
                                    calc_mz,
                                    precursor_mz,
                                    precursor_charge,
                                    isotope,
                                )
                                for isotope in range(
                                    self.isotope_error_range[0],
                                    self.isotope_error_range[1] + 1,
                                )
                            ]
                            # Terminate the beam if the calculated m/z for the
                            # predicted peptide (without potential additional
                            # AAs with negative mass) is within the precursor
                            # m/z tolerance.
                            matches_precursor_mz = aa is None and any(
                                abs(d) < self.precursor_mass_tol
                                for d in delta_mass_ppm
                            )
                            # Terminate the beam if the calculated m/z exceeds
                            # the precursor m/z + tolerance and hasn't been
                            # corrected by a subsequently predicted AA with
                            # negative mass.
                            exceeds_precursor_mz = aa is not None and all(
                                d > self.precursor_mass_tol
                                for d in delta_mass_ppm
                            )
                            if matches_precursor_mz or exceeds_precursor_mz:
                                break
                        except KeyError:
                            matches_precursor_mz = exceeds_precursor_mz = False
                    if matches_precursor_mz or exceeds_precursor_mz:
                        tokens[beam_i][idx] = self.stop_token
                        is_beam_prec_fit[beam_i] = matches_precursor_mz

        # Get the indices of finished beams.
        finished_idx = torch.where((tokens == self.stop_token).any(dim=1))[0]
        return finished_idx, tokens

    def _cache_finished_beams(
        self,
        finished_beams_idx: torch.Tensor,
        cache_next_idx: Dict[int, int],
        cache_pred_seq: Dict[int, Set[torch.Tensor]],
        cache_pred_score: Dict[int, List[List[Tuple[float, int]]]],
        cache_tokens: torch.Tensor,
        cache_scores: torch.Tensor,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        is_beam_prec_fit: torch.Tensor,
        idx: int,
    ):
        """
        Cache terminated beams.

        Group and order by fitting precursor m/z and confidence score.

        Parameters
        ----------
        finished_beams_idx : torch.Tensor
            Indices of all finished beams on tokens tensor.
        cache_next_idx : Dict[int, int]
            Next available tensor index to cache peptides for each spectrum.
        cache_pred_seq : Dict[int, Set[torch.Tensor]]
            Set of decoded peptide tokens for each spectrum.
        cache_pred_score : Dict[int, List[List[Tuple[float, int]]]
            Confidence score for each decoded peptide, separated as
            precursor m/z fitting vs not, for each spectrum.
        cache_tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            The token for each amino acid in cached peptides.
        cache_scores : torch.Tensor of shape
        (n_spectra * n_beams, max_length, n_amino_acids)
            The score for each amino acid in cached peptides.
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Output token of the model corresponding to amino acid sequences.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Output scores of the model.
        is_beam_prec_fit: torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating if current beams are within the precursor
            m/z tolerance.
        idx : int
            Index to be considered in the current decoding step.
        """
        beam = self.n_beams
        # Store finished beams in the cache.
        for i in finished_beams_idx:
            i = i.item()
            spec_idx = i // beam  # Find the starting index of the spectrum.
            # Check position of stop token (changes in case stopped early).
            stop_token_idx = idx - (not tokens[i][idx] == self.stop_token)
            # Check if predicted peptide already in cache.
            pred_seq = tokens[i][:stop_token_idx]
            is_peptide_cached = any(
                torch.equal(pep, pred_seq) for pep in cache_pred_seq[spec_idx]
            )
            # Don't cache this peptide if it was already predicted previously.
            if is_peptide_cached:
                continue
            smx = self.softmax(scores)
            aa_scores = [smx[i, j, k].item() for j, k in enumerate(pred_seq)]
            pep_score = _aa_to_pep_score(aa_scores)
            # Cache peptides with fitting (idx=0) or non-fitting (idx=1)
            # precursor m/z separately.
            cache_pred_score_idx = cache_pred_score[spec_idx]
            cache_i = int(not is_beam_prec_fit[i])
            # Directly cache if we don't already have k peptides cached.
            if cache_next_idx[spec_idx] < (spec_idx + 1) * beam:
                insert_idx = cache_next_idx[spec_idx]
                cache_next_idx[spec_idx] += 1  # Move the pointer.
                heap_update = heapq.heappush
            # If any prediction has a non-fitting precursor m/z and this
            # prediction has a fitting precursor m/z, replace the non-fitting
            # peptide with the lowest score, irrespective of the current
            # predicted score.
            elif is_beam_prec_fit[i] and len(cache_pred_score_idx[1]) > 0:
                _, insert_idx = heapq.heappop(cache_pred_score_idx[1])
                heap_update = heapq.heappush
            # Else, replace the lowest-scoring peptide with corresponding
            # fitting or non-fitting precursor m/z if the current predicted
            # score is higher.
            elif len(cache_pred_score_idx[cache_i]) > 0:
                # Peek at the top of the heap (lowest score).
                pop_pep_score, insert_idx = cache_pred_score_idx[cache_i][0]
                heap_update = heapq.heappushpop
                # Don't store this prediction if it has a lower score than all
                # previous predictions.
                if pep_score <= pop_pep_score:
                    continue
            # Finally, no matching cache found (we should never get here).
            else:
                continue
            # Store the current prediction in its relevant cache.
            cache_tokens[insert_idx, :] = tokens[i, :]
            cache_scores[insert_idx, :, :] = scores[i, :, :]
            heap_update(cache_pred_score_idx[cache_i], (pep_score, insert_idx))
            cache_pred_seq[spec_idx].add(pred_seq)

    def _get_top_peptide(
        self,
        cache_pred_score: Dict[int, List[List[Tuple[float, int]]]],
        cache_tokens: torch.tensor,
        cache_scores: torch.tensor,
        batch: int,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Return the peptide with the highest confidence score for each spectrum.

        If there are no peptides within the precursor m/z tolerance, return the
        highest-scoring peptide among the non-fitting predictions.

        Parameters
        ----------
        cache_pred_score : Dict[int, List[List[Tuple[float, int]]]
            Confidence score for each decoded peptide, separated as
            precursor m/z fitting vs not, for each spectrum.
        cache_tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            The token for each amino acid in cached peptides.
        cache_scores : torch.Tensor of shape
        (n_spectra * n_beams, max_length, n_amino_acids)
            The score for each amino acid in cached peptides.
        batch: int
            Number of spectra in the batch.

        Returns
        -------
        output_tokens : torch.Tensor of shape (n_spectra, max_length)
            The token for each amino acid in the output peptides.
        output_scores : torch.Tensor of shape
        (n_spectra, max_length, n_amino_acids)
            The score for each amino acid in cached peptides.
        """
        # Sizes.
        length = self.max_length + 1  # L
        vocab = self.decoder.vocab_size + 1  # V

        # Create output tensors for top scoring peptides and their scores.
        output_scores = torch.full(
            size=(batch, length, vocab), fill_value=torch.nan
        )
        output_scores = output_scores.type_as(cache_scores)
        output_tokens = torch.zeros(batch, length).type_as(cache_tokens)

        # Return the top scoring peptide (fitting precursor mass if possible).
        for spec_idx in range(batch):
            cache = cache_pred_score[spec_idx][
                len(cache_pred_score[spec_idx][0]) == 0
            ]
            # Skip this spectrum if it doesn't have any finished beams.
            if len(cache) == 0:
                continue
            _, top_score_idx = max(cache, key=operator.itemgetter(0))
            output_tokens[spec_idx, :] = cache_tokens[top_score_idx, :]
            output_scores[spec_idx, :, :] = cache_scores[top_score_idx, :, :]

        return output_tokens, output_scores

    def _get_topk_beams(
        self, scores: torch.tensor, tokens: torch.tensor, batch: int, idx: int
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Find top-k beams with highest confidences and continue decoding those.

        Discontinue decoding for beams where the stop token was predicted.

        Parameters
        ----------
        scores : torch.Tensor of shape
        (n_spectra * n_beams, max_length, n_amino_acids)
            Output scores of the model.
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Output token of the model corresponding to amino acid sequences.
        batch: int
            Number of spectra in the batch.
        idx : int
            Index to be considered in the current decoding step.

        Returns
        -------
        scores : torch.Tensor of shape
        (n_spectra * n_beams, max_length, n_amino_acids)
            Output scores of the model.
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Output token of the model corresponding to amino acid sequences.
        """
        beam = self.n_beams  # S
        vocab = self.decoder.vocab_size + 1  # V

        # Reshape to group by spectrum (B for "batch").
        scores = einops.rearrange(scores, "(B S) L V -> B L V S", S=beam)
        tokens = einops.rearrange(tokens, "(B S) L -> B L S", S=beam)
        prev_tokens = einops.repeat(
            tokens[:, :idx, :], "B L S -> B L V S", V=vocab
        )

        # Get the previous tokens and scores.
        prev_scores = torch.gather(
            scores[:, :idx, :, :], dim=2, index=prev_tokens
        )
        prev_scores = einops.repeat(
            prev_scores[:, :, 0, :], "B L S -> B L (V S)", V=vocab
        )

        # Get scores for all possible beams at this step.
        step_scores = torch.zeros(batch, idx + 1, beam * vocab).type_as(scores)
        step_scores[:, :idx, :] = prev_scores
        step_scores[:, idx, :] = einops.rearrange(
            scores[:, idx, :, :], "B V S -> B (V S)"
        )

        # Mask out terminated beams. Include delta mass induced termination.
        extended_prev_tokens = einops.repeat(
            tokens[:, : idx + 1, :], "B L S -> B L V S", V=vocab
        )
        finished_mask = (
            einops.rearrange(extended_prev_tokens, "B L V S -> B L (V S)")
            == self.stop_token
        ).any(axis=1)
        # Mask out the index '0', i.e. padding token, by default.
        finished_mask[:, :beam] = True

        # Figure out the top K decodings.
        _, top_idx = torch.topk(
            step_scores.nanmean(dim=1) * (~finished_mask).float(), beam
        )
        v_idx, s_idx = np.unravel_index(top_idx.cpu(), (vocab, beam))
        s_idx = einops.rearrange(s_idx, "B S -> (B S)")
        b_idx = einops.repeat(torch.arange(batch), "B -> (B S)", S=beam)

        # Record the top K decodings.
        tokens[:, :idx, :] = einops.rearrange(
            prev_tokens[b_idx, :, 0, s_idx], "(B S) L -> B L S", S=beam
        )
        tokens[:, idx, :] = torch.tensor(v_idx)
        scores[:, : idx + 1, :, :] = einops.rearrange(
            scores[b_idx, : idx + 1, :, s_idx], "(B S) L V -> B L V S", S=beam
        )
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")
        return scores, tokens

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
            such that all of the spectra in the batch are the same length.
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
        loss = self.celoss(pred, truth.flatten())
        self.log(
            "CELoss",
            {mode: loss.detach()},
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

        # Calculate and log amino acid and peptide match evaluation metrics from
        # the predicted peptides.
        peptides_pred_raw, _ = self.forward(batch[0], batch[1])
        # FIXME: Temporary fix to skip predictions with multiple stop tokens.
        peptides_pred, peptides_true = [], []
        for peptide_pred, peptide_true in zip(peptides_pred_raw, batch[2]):
            if len(peptide_pred) > 0:
                if peptide_pred[0] == "$":
                    peptide_pred = peptide_pred[1:]  # Remove stop token.
                if "$" not in peptide_pred and len(peptide_pred) > 0:
                    peptides_pred.append(peptide_pred)
                    peptides_true.append(peptide_true)
        aa_precision, aa_recall, pep_recall = evaluate.aa_match_metrics(
            *evaluate.aa_match_batch(
                peptides_pred, peptides_true, self.decoder._peptide_mass.masses
            )
        )
        log_args = dict(on_step=False, on_epoch=True, sync_dist=True)
        self.log("aa_precision", {"valid": aa_precision}, **log_args)
        self.log("aa_recall", {"valid": aa_recall}, **log_args)
        self.log("pep_recall", {"valid": pep_recall}, **log_args)

        return loss

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], *args
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]], torch.Tensor]:
        """
        A single prediction step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            spectrum identifiers as torch Tensors.

        Returns
        -------
        spectrum_idx : torch.Tensor
            The spectrum identifiers.
        precursors : torch.Tensor
            Precursor information for each spectrum.
        peptides : List[List[str]]
            The predicted peptide sequences for each spectrum.
        aa_scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        """
        peptides, aa_scores = self.forward(batch[0], batch[1])
        return batch[2], batch[1], peptides, aa_scores

    def on_train_epoch_end(self) -> None:
        """
        Log the training loss at the end of each epoch.
        """
        train_loss = self.trainer.callback_metrics["CELoss"]["train"].detach()
        self._history[-1]["train"] = train_loss
        self._log_history()

    def on_validation_epoch_end(self) -> None:
        """
        Log the validation metrics at the end of each epoch.
        """
        callback_metrics = self.trainer.callback_metrics
        metrics = {
            "epoch": self.trainer.current_epoch,
            "valid": callback_metrics["CELoss"]["valid"].detach(),
            "valid_aa_precision": callback_metrics["aa_precision"][
                "valid"
            ].detach(),
            "valid_aa_recall": callback_metrics["aa_recall"]["valid"].detach(),
            "valid_pep_recall": callback_metrics["pep_recall"][
                "valid"
            ].detach(),
        }
        self._history.append(metrics)
        self._log_history()

    def on_predict_epoch_end(
        self, results: List[List[Tuple[np.ndarray, List[str], torch.Tensor]]]
    ) -> None:
        """
        Write the predicted peptide sequences and amino acid scores to the
        output file.
        """
        if self.out_writer is None:
            return
        for batch in results:
            for step in batch:
                for spectrum_i, precursor, aa_tokens, aa_scores in zip(*step):
                    # Get peptide sequence, amino acid and peptide-level
                    # confidence scores to write to output file.
                    (
                        peptide,
                        aa_tokens,
                        peptide_score,
                        aa_scores,
                    ) = self._get_output_peptide_and_scores(
                        aa_tokens, aa_scores
                    )
                    # Compare the experimental vs calculated precursor m/z.
                    _, precursor_charge, precursor_mz = precursor
                    precursor_charge = int(precursor_charge.item())
                    precursor_mz = precursor_mz.item()
                    try:
                        calc_mz = self.peptide_mass_calculator.mass(
                            aa_tokens, precursor_charge
                        )
                        delta_mass_ppm = [
                            _calc_mass_error(
                                calc_mz,
                                precursor_mz,
                                precursor_charge,
                                isotope,
                            )
                            for isotope in range(
                                self.isotope_error_range[0],
                                self.isotope_error_range[1] + 1,
                            )
                        ]
                        is_within_precursor_mz_tol = any(
                            abs(d) < self.precursor_mass_tol
                            for d in delta_mass_ppm
                        )
                    except KeyError:
                        calc_mz, is_within_precursor_mz_tol = np.nan, False
                    # Subtract one if the precursor m/z tolerance is violated.
                    if not is_within_precursor_mz_tol:
                        peptide_score -= 1

                    self.out_writer.psms.append(
                        (
                            peptide,
                            spectrum_i,
                            peptide_score,
                            precursor_charge,
                            precursor_mz,
                            calc_mz,
                            aa_scores,
                        ),
                    )

    def _get_output_peptide_and_scores(
        self, aa_tokens: List[str], aa_scores: torch.Tensor
    ) -> Tuple[str, List[str], float, str]:
        """
        Get peptide to output, amino acid and peptide-level confidence scores.

        Parameters
        ----------
        aa_tokens : List[str]
            Amino acid tokens of the peptide sequence.
        aa_scores : torch.Tensor
            Amino acid-level confidence scores for the predicted sequence.

        Returns
        -------
        peptide : str
            Peptide sequence.
        aa_tokens : List[str]
            Amino acid tokens of the peptide sequence.
        peptide_score : str
            Peptide-level confidence score.
        aa_scores : str
            Amino acid-level confidence scores for the predicted sequence.
        """
        # Omit stop token.
        aa_tokens = aa_tokens[1:] if self.decoder.reverse else aa_tokens[:-1]
        peptide = "".join(aa_tokens)

        # If this is a non-finished beam (after exceeding `max_length`), return
        # a dummy (empty) peptide and NaN scores.
        if len(peptide) == 0:
            aa_tokens = []

        # Take scores corresponding to the predicted amino acids. Reverse tokens
        # to correspond with correct amino acids as needed.
        step = -1 if self.decoder.reverse else 1
        top_aa_scores = [
            aa_score[self.decoder._aa2idx[aa_token]].item()
            for aa_score, aa_token in zip(aa_scores, aa_tokens[::step])
        ][::step]

        # Get peptide-level score from amino acid-level scores.
        peptide_score = _aa_to_pep_score(top_aa_scores)
        aa_scores = ",".join(list(map("{:.5f}".format, top_aa_scores)))
        return peptide, aa_tokens, peptide_score, aa_scores

    def _log_history(self) -> None:
        """
        Write log to console, if requested.
        """
        # Log only if all output for the current epoch is recorded.
        if len(self._history) > 0 and len(self._history[-1]) == 6:
            if len(self._history) == 1:
                logger.info(
                    "Epoch\tTrain loss\tValid loss\tAA precision\tAA recall\t"
                    "Peptide recall"
                )
            metrics = self._history[-1]
            if metrics["epoch"] % self.n_log == 0:
                logger.info(
                    "%i\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f",
                    metrics["epoch"] + 1,
                    metrics.get("train", np.nan),
                    metrics.get("valid", np.nan),
                    metrics.get("valid_aa_precision", np.nan),
                    metrics.get("valid_aa_recall", np.nan),
                    metrics.get("valid_pep_recall", np.nan),
                )
                if self.tb_summarywriter is not None:
                    for descr, key in [
                        ("loss/train_crossentropy_loss", "train"),
                        ("loss/dev_crossentropy_loss", "valid"),
                        ("eval/dev_aa_precision", "valid_aa_precision"),
                        ("eval/dev_aa_recall", "valid_aa_recall"),
                        ("eval/dev_pep_recall", "valid_pep_recall"),
                    ]:
                        self.tb_summarywriter.add_scalar(
                            descr,
                            metrics.get(key, np.nan),
                            metrics["epoch"] + 1,
                        )

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
            optimizer, warmup=self.warmup_iters, max_iters=self.max_iters
        )
        return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm up followed by cosine shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup : int
        The number of warm up iterations.
    max_iters : torch.optim
        The total number of iterations.
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, warmup: int, max_iters: int
    ):
        self.warmup, self.max_iters = warmup, max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch / self.warmup
        return lr_factor


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


def _aa_to_pep_score(aa_scores: List[float]) -> float:
    """
    Calculate peptide-level confidence score from amino acid level scores.

    Parameters
    ----------
    aa_scores : List[float]
        Amino acid level confidence scores.

    Returns
    -------
    float
        Peptide confidence score.
    """
    return np.mean(aa_scores)
