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
    residues : Union[Dict[str, float], str]
        The amino acid dictionary and their masses. By default
        ("canonical) this is only the 20 canonical amino acids, with
        cysteine carbamidomethylated. If "massivekb", this dictionary
        will include the modifications found in MassIVE-KB.
        Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int
        The maximum precursor charge to consider.
    precursor_mass_tol : float, optional
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
    out_writer : Optional[ms_io.MztabWriter]
        The output writer for the prediction results.
    calculate_precision : bool
        Calculate the validation set precision during training.
        This is expensive.
    tokenizer: Optional[PeptideTokenizer]
        Tokenizer object to tokenize and detokenize peptide sequences.
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
        residues: Union[Dict[str, float], str] = "canonical",
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
        tokenizer: Optional[PeptideTokenizer] = None,
        **kwargs: Dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = (
            tokenizer if tokenizer is not None else PeptideTokenizer()
        )
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
            d_model=dim_model,
            n_tokens=self.tokenizer,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            max_charge=max_charge,
        )
        self.softmax = torch.nn.Softmax(2)
        self.celoss = torch.nn.CrossEntropyLoss(
            ignore_index=0,
            label_smoothing=train_label_smoothing,
            reduce=False,
        )
        self.val_celoss = torch.nn.CrossEntropyLoss(
            ignore_index=0, reduce=False
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

        # Logging.
        self.calculate_precision = calculate_precision
        self.n_log = n_log
        self._history = []

        # Output writer during predicting.
        self.out_writer: ms_io.MztabWriter = out_writer

    @property
    def device(self) -> torch.device:
        """The current device for first parameter of the model."""
        return next(self.parameters()).device

    @property
    def n_parameters(self):
        """The number of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self, batch: dict
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        Predict peptide sequences for a batch of MS/MS spectra.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) m/z values of MS/MS spectra,
            (ii) intensity values of MS/MS spectra,
            (iii) precursor information,
            (iv) peptide sequences as torch Tensors.

        Returns
        -------
        pred_peptides : List[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide predictions.
            A peptide predictions consists of a tuple with the peptide
            score, the amino acid scores, and the predicted peptide
            sequence.
        """
        mzs, ints, precursors, _, _ = self._process_batch(batch)
        return self.beam_search_decode(mzs, ints, precursors)

    def beam_search_decode(
        self, mzs: torch.Tensor, ints: torch.Tensor, precursors: torch.Tensor
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        Beam search decoding of the spectrum predictions.

        Parameters
        ----------
        mzs : torch.Tensor of shape (n_spectra, n_peaks)
            The m/z axis of spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        ints: torch.Tensor of shape (n_spectra, n_peaks)
            The m/z axis of spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 specifies
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
        memories, mem_masks = self.encoder(mzs, ints)

        # Sizes.
        batch = mzs.shape[0]  # B
        length = self.max_peptide_len + 1  # L
        vocab = self.vocab_size  # V
        beam = self.n_beams  # S

        # Initialize scores and tokens.
        scores = torch.full(
            size=(batch, length, vocab, beam), fill_value=torch.nan
        ).type_as(mzs)

        tokens = torch.zeros(
            batch, length, beam, dtype=torch.int64, device=self.encoder.device
        )

        # Create cache for decoded beams.
        pred_cache = collections.OrderedDict((i, []) for i in range(batch))

        # Get the first prediction.
        pred = self.decoder(
            tokens=torch.zeros(
                batch, 0, dtype=torch.int64, device=self.encoder.device
            ),
            memory=memories,
            memory_key_padding_mask=mem_masks,
            precursors=precursors,
        )
        tokens[:, 0, :] = torch.topk(pred[:, 0, :], beam, dim=1)[1]
        scores[:, :1, :, :] = einops.repeat(pred, "B L V -> B L V S", S=beam)

        # Make all tensors the right shape for decoding.
        precursors = einops.repeat(precursors, "B L -> (B S) L", S=beam)
        mem_masks = einops.repeat(mem_masks, "B L -> (B S) L", S=beam)
        memories = einops.repeat(memories, "B L V -> (B S) L V", S=beam)
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")

        # The main decoding loop.
        for step in range(0, self.max_peptide_len):
            # Terminate beams exceeding the precursor m/z tolerance and track
            # all finished beams (either terminated or stop token predicted).
            (
                finished_beams,
                beam_fits_precursor,
                discarded_beams,
            ) = self._finish_beams(tokens, precursors, step)
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
            scores[~finished_beams, : step + 2, :] = self.decoder(
                tokens=tokens[~finished_beams, : step + 1],
                precursors=precursors[~finished_beams, :],
                memory=memories[~finished_beams, :, :],
                memory_key_padding_mask=mem_masks[~finished_beams, :],
            )
            # Find the top-k beams with the highest scores and continue decoding
            # those.
            tokens, scores = self._get_topk_beams(
                tokens, scores, finished_beams, batch, step + 1
            )

        # Return the peptide with the highest confidence score, within the
        # precursor m/z tolerance if possible.
        return list(self._get_top_peptide(pred_cache))

    def _finish_beams(
        self,
        tokens: torch.Tensor,
        precursors: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Track all beams that have been finished, either by predicting the stop
        token or because they were terminated due to exceeding the precursor
        m/z tolerance.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        step : int
            Index of the current decoding step.

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
        # Check for tokens with a negative mass (i.e. neutral loss).
        aa_neg_mass_idx = [None]
        for aa, mass in self.tokenizer.residues.items():
            if mass < 0:
                # aa_neg_mass.append(aa)
                aa_neg_mass_idx.append(self.tokenizer.index[aa])

        # Find N-terminal residues.
        n_term = torch.Tensor(
            [
                self.tokenizer.index[aa]
                for aa in self.tokenizer.index
                if aa.startswith("[") and aa.endswith("]-")
            ]
        ).to(self.decoder.device)

        beam_fits_precursor = torch.zeros(
            tokens.shape[0], dtype=torch.bool
        ).to(self.encoder.device)
        # Beams with a stop token predicted in the current step can be finished.
        finished_beams = torch.zeros(tokens.shape[0], dtype=torch.bool).to(
            self.encoder.device
        )
        ends_stop_token = tokens[:, step] == self.stop_token
        finished_beams[ends_stop_token] = True
        # Beams with a dummy token predicted in the current step can be
        # discarded.
        discarded_beams = torch.zeros(tokens.shape[0], dtype=torch.bool).to(
            self.encoder.device
        )

        discarded_beams[tokens[:, step] == 0] = True
        # Discard beams with invalid modification combinations (i.e. N-terminal
        # modifications occur multiple times or in internal positions).
        if step > 1:  # Only relevant for longer predictions.
            dim0 = torch.arange(tokens.shape[0])
            final_pos = torch.full((ends_stop_token.shape[0],), step)
            final_pos[ends_stop_token] = step - 1
            # Multiple N-terminal modifications.
            multiple_mods = torch.isin(
                tokens[dim0, final_pos], n_term
            ) & torch.isin(tokens[dim0, final_pos - 1], n_term)
            # N-terminal modifications occur at an internal position.
            # Broadcasting trick to create a two-dimensional mask.
            mask = (final_pos - 1)[:, None] >= torch.arange(tokens.shape[1])
            internal_mods = torch.isin(
                torch.where(mask.to(self.encoder.device), tokens, 0), n_term
            ).any(dim=1)
            discarded_beams[multiple_mods | internal_mods] = True

        # Check which beams should be terminated or discarded based on the
        # predicted peptide.
        for i in range(len(finished_beams)):
            # Skip already discarded beams.
            if discarded_beams[i]:
                continue
            pred_tokens = tokens[i][: step + 1]
            peptide_len = len(pred_tokens)

            # Omit stop token.
            if self.tokenizer.reverse and pred_tokens[0] == self.stop_token:
                pred_tokens = pred_tokens[1:]
                peptide_len -= 1
            elif (
                not self.tokenizer.reverse
                and pred_tokens[-1] == self.stop_token
            ):
                pred_tokens = pred_tokens[:-1]
                peptide_len -= 1
            # Discard beams that were predicted to end but don't fit the minimum
            # peptide length.
            if finished_beams[i] and peptide_len < self.min_peptide_len:
                discarded_beams[i] = True
                continue
            # Discard beams that contain more than chimeric separator
            chimeric_separator = self.tokenizer.index[
                self.tokenizer.chimeric_separator_token
            ]
            if (pred_tokens == chimeric_separator).sum() > 1:
                discarded_beams[i] = True
                continue
            # Terminate the beam if it has not been finished by the model but
            # the peptide mass exceeds the precursor m/z to an extent that it
            # cannot be corrected anymore by a subsequently predicted AA with
            # negative mass.
            precursor_charge = precursors[i, 1]
            precursor_mz = precursors[i, 0]
            matches_precursor_mz = exceeds_precursor_mz = False

            # Send tokenizer masses to correct device for calculate_precursor_ions()
            self.tokenizer.masses = self.tokenizer.masses.type_as(precursor_mz)

            for aa in [None] if finished_beams[i] else aa_neg_mass_idx:
                if aa is None:
                    calc_peptide = pred_tokens
                else:
                    calc_peptide = pred_tokens.detach().clone()
                    calc_peptide = torch.cat(
                        (
                            calc_peptide,
                            torch.tensor([aa]).type_as(calc_peptide),
                        )
                    )
                try:
                    calc_mz = self.tokenizer.calculate_precursor_ions(
                        calc_peptide.unsqueeze(0),
                        precursor_charge.unsqueeze(0),
                    )[0]

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
                    # Terminate the beam if the calculated m/z for the predicted
                    # peptide (without potential additional AAs with negative
                    # mass) is within the precursor m/z tolerance.
                    matches_precursor_mz = aa is None and any(
                        abs(d) < self.precursor_mass_tol
                        for d in delta_mass_ppm
                    )
                    # Terminate the beam if the calculated m/z exceeds the
                    # precursor m/z + tolerance and hasn't been corrected by a
                    # subsequently predicted AA with negative mass.
                    if matches_precursor_mz:
                        exceeds_precursor_mz = False
                    else:
                        exceeds_precursor_mz = all(
                            d > self.precursor_mass_tol for d in delta_mass_ppm
                        )
                        exceeds_precursor_mz = (
                            finished_beams[i] or aa is not None
                        ) and exceeds_precursor_mz
                    if matches_precursor_mz or exceeds_precursor_mz:
                        break
                except KeyError:
                    matches_precursor_mz = exceeds_precursor_mz = False
            # Finish beams that fit or exceed the precursor m/z.
            # Don't finish beams that don't include a stop token if they don't
            # exceed the precursor m/z tolerance yet.
            if finished_beams[i]:
                beam_fits_precursor[i] = matches_precursor_mz
            elif exceeds_precursor_mz:
                finished_beams[i] = True
                beam_fits_precursor[i] = matches_precursor_mz
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
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        step : int
            Index of the current decoding step.
        beams_to_cache : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams are ready for
            caching.
        beam_fits_precursor: torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the beams are within the
            precursor m/z tolerance.
        pred_cache : Dict[
                int, List[Tuple[float, float, np.ndarray, torch.Tensor]]
        ]
            Priority queue with finished beams for each spectrum, ordered by
            peptide score. For each finished beam, a tuple with the (negated)
            peptide score, a random tie-breaking float, the amino acid-level
            scores, and the predicted tokens is stored.
        """
        for i in range(len(beams_to_cache)):
            if not beams_to_cache[i]:
                continue
            # Find the starting index of the spectrum.
            spec_idx = i // self.n_beams
            # FIXME: The next 3 lines are very similar as what's done in
            #  _finish_beams. Avoid code duplication?
            pred_tokens = tokens[i][: step + 1]
            # Omit the stop token from the peptide sequence (if predicted).
            has_stop_token = pred_tokens[-1] == self.stop_token
            pred_peptide = pred_tokens[:-1] if has_stop_token else pred_tokens
            # Don't cache this peptide if it was already predicted previously.
            if any(
                torch.equal(pred_cached[-1], pred_peptide)
                for pred_cached in pred_cache[spec_idx]
            ):
                # TODO: Add duplicate predictions with their highest score.
                continue
            smx = self.softmax(scores[i : i + 1, : step + 1, :])
            aa_scores = smx[0, range(len(pred_tokens)), pred_tokens].tolist()
            # Add an explicit score 0 for the missing stop token in case this
            # was not predicted (i.e. early stopping).
            if not has_stop_token:
                aa_scores.append(0)
            aa_scores = np.asarray(aa_scores)
            # Calculate the updated amino acid-level and the peptide scores.
            aa_scores, peptide_score = _aa_pep_score(
                aa_scores, beam_fits_precursor[i]
            )
            # Omit the stop token from the amino acid-level scores.
            aa_scores = aa_scores[:-1]
            # Add the prediction to the cache (minimum priority queue, maximum
            # the number of beams elements).
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
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Find the top-k beams with the highest scores and continue decoding
        those.

        Stop decoding for beams that have been finished.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        finished_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams are ready for
            caching.
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
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        """
        beam = self.n_beams  # S
        vocab = self.vocab_size  # V

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
        step_scores = torch.zeros(batch, step + 1, beam * vocab).type_as(
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
        # FIXME: Set this to a very small, yet non-zero value, to only
        # get padding after stop token.
        active_mask[:, :beam] = 1e-8

        # Figure out the top K decodings.
        _, top_idx = torch.topk(step_scores.nanmean(dim=1) * active_mask, beam)
        v_idx, s_idx = np.unravel_index(top_idx.cpu(), (vocab, beam))
        s_idx = einops.rearrange(s_idx, "B S -> (B S)")
        b_idx = einops.repeat(torch.arange(batch), "B -> (B S)", S=beam)

        # Record the top K decodings.
        tokens[:, :step, :] = einops.rearrange(
            prev_tokens[b_idx, :, 0, s_idx], "(B S) L -> B L S", S=beam
        )
        tokens[:, step, :] = torch.tensor(v_idx)
        scores[:, : step + 1, :, :] = einops.rearrange(
            scores[b_idx, : step + 1, :, s_idx], "(B S) L V -> B L V S", S=beam
        )
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")
        return tokens, scores

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
            For each spectrum, a list with the top peptide
            prediction(s). A peptide predictions consists of a tuple
            with the peptide score, the amino acid scores, and the
            predicted peptide sequence.
        """
        for peptides in pred_cache.values():
            if len(peptides) > 0:
                yield [
                    (
                        pep_score,
                        aa_scores,
                        pred_tokens,
                    )
                    for pep_score, _, aa_scores, pred_tokens in heapq.nlargest(
                        self.top_match, peptides
                    )
                ]
            else:
                yield []

    def _process_batch(self, batch):
        """Prepare batch returned from AnnotatedSpectrumDataset of the
            latest depthcharge version

        Each batch is a dict and contains these keys:
             ['peak_file', 'scan_id', 'ms_level', 'precursor_mz',
             'precursor_charge', 'mz_array', 'intensity_array',
             'seq']
        Returns
        -------
        spectra : torch.Tensor of shape (batch_size, n_peaks, 2)
            The padded mass spectra tensor with the m/z and intensity peak values
            for each spectrum.
        precursors : torch.Tensor of shape (batch_size, 3)
            A tensor with the precursor neutral mass, precursor charge, and
            precursor m/z.
        seqs : np.ndarray
            The spectrum identifiers (during de novo sequencing) or peptide
            sequences (during training).

        """
        for k in batch.keys():
            try:
                batch[k] = batch[k].squeeze(0)
            except:
                continue

        precursor_mzs = batch["precursor_mz"]
        precursor_charges = batch["precursor_charge"]
        precursor_masses = (precursor_mzs - 1.007276) * precursor_charges
        precursors = torch.vstack(
            [precursor_masses, precursor_charges, precursor_mzs]
        ).T  # .float()

        mzs, ints = batch["mz_array"], batch["intensity_array"]
        # spectra = torch.stack([mzs, ints], dim=2)

        if "seq" in batch:
            seqs = batch["seq"]

            if "seq_compliment" in batch:
                seqs_comp = (
                    batch["seq_compliment"]
                    if "seq_compliment" in batch
                    else None
                )

                # Chimera may be inferred in either order if both precursors
                # have the same charge, else the peptide with the reported
                # (and embedded) charge must be inferred first.
                charges_equal = (
                    batch["precursor_charge"] == batch["precursor_charge_two"]
                )
                seqs_comp[~charges_equal] = seqs[~charges_equal]
            else:
                seqs_comp = None
        else:
            seqs = None
            seqs_comp = None

        return mzs, ints, precursors, seqs, seqs_comp

    def _forward_step(
        self,
        batch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward learning step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) m/z values of MS/MS spectra,
            (ii) intensity values of MS/MS spectra,
            (iii) precursor information,
            (iv) peptide sequences as torch Tensors.

        Returns
        -------
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        tokens : torch.Tensor of shape (n_spectra, length)
            The predicted tokens for each spectrum.
        """
        mzs, ints, precursors, tokens, tokens_comp = self._process_batch(batch)
        memories, mem_masks = self.encoder(mzs, ints)
        decoded = self.decoder(
            tokens=tokens,
            memory=memories,
            memory_key_padding_mask=mem_masks,
            precursors=precursors,
        )

        # FIXME: No compliment tokens in DB search mode
        if tokens_comp is not None:
            decoded_comp = self.decoder(
                tokens=tokens_comp,
                memory=memories,
                memory_key_padding_mask=mem_masks,
                precursors=precursors,
            )
        else:
            tokens_comp = tokens
            decoded_comp = decoded

        return decoded, tokens, decoded_comp, tokens_comp

    def training_step(
        self,
        batch: dict,
        *args,
        mode: str = "train",
    ) -> torch.Tensor:
        """
        A single training step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) m/z values of MS/MS spectra,
            (ii) intensity values of MS/MS spectra,
            (iii) precursor information,
            (iv) peptide sequences as torch Tensors.
        mode : str
            Logging key to describe the current stage.

        Returns
        -------
        torch.Tensor
            The loss of the training step.
        """
        pred, truth, pred_comp, truth_comp = self._forward_step(batch)
        pred = pred[:, :-1, :].reshape(-1, self.vocab_size)
        pred_comp = pred_comp[:, :-1, :].reshape(-1, self.vocab_size)

        loss_fun = self.celoss if mode == "Train" else self.val_celoss
        loss_one = loss_fun(pred, truth.flatten()).unsqueeze(-1)
        loss_two = loss_fun(pred_comp, truth_comp.flatten()).unsqueeze(-1)
        loss = torch.cat((-loss_one, -loss_two), dim=1)
        loss = -torch.logsumexp(loss, dim=1)
        loss = torch.mean(loss)

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
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[str]], *args
    ) -> torch.Tensor:
        """
        A single validation step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) m/z values of MS/MS spectra,
            (ii) intensity values of MS/MS spectra,
            (iii) precursor information,
            (iv) peptide sequences as torch Tensors.

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
        peptides_true = [
            "".join(p)
            for p in self.tokenizer.detokenize(batch["seq"], join=False)
        ]
        peptides_pred = []
        for spectrum_preds in self.forward(batch):
            for _, _, pred in spectrum_preds:
                peptides_pred.append(pred)
        peptides_pred = [
            "".join(p)
            for p in self.tokenizer.detokenize(peptides_pred, join=False)
        ]
        batch_size = len(peptides_true)
        aa_precision, _, pep_precision = evaluate.aa_match_metrics(
            *evaluate.aa_match_batch(
                peptides_true,
                peptides_pred,
                self.tokenizer.residues,
            )
        )

        log_args = dict(on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "pep_precision", pep_precision, **log_args, batch_size=batch_size
        )
        self.log(
            "aa_precision", aa_precision, **log_args, batch_size=batch_size
        )
        return loss

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], *args
    ) -> List[ms_io.PepSpecMatch]:
        """
        A single prediction step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) m/z values of MS/MS spectra,
            (ii) intensity values of MS/MS spectra,
            (iii) precursor information,
            (iv) peptide sequences as torch Tensors.

        Returns
        -------
        predictions: List[ms_io.PepSpecMatch]
            Predicted PSMs for the given batch of spectra.
        """
        _, _, precursors, _, _ = self._process_batch(batch)
        prec_charges = precursors[:, 1].cpu().detach().numpy()
        prec_mzs = precursors[:, 2].cpu().detach().numpy()
        predictions = []
        for (
            precursor_charge,
            precursor_mz,
            scan,
            file_name,
            spectrum_preds,
        ) in zip(
            prec_charges,
            prec_mzs,
            batch["scan_id"],
            batch["peak_file"],
            self.forward(batch),
        ):
            for peptide_score, aa_scores, peptide in spectrum_preds:
                predictions.append(
                    (
                        scan,
                        precursor_charge,
                        precursor_mz,
                        peptide,
                        peptide_score,
                        aa_scores,
                        file_name,
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
        metrics = {
            "step": self.trainer.global_step,
            "train": train_loss,
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
                callback_metrics["aa_precision"].detach().item()
            )
            metrics["valid_pep_precision"] = (
                callback_metrics["pep_precision"].detach().item()
            )
        self._history.append(metrics)
        self._log_history()

    def on_predict_batch_end(
        self, outputs: List[ms_io.PepSpecMatch], *args
    ) -> None:
        """
        Write the predicted peptide sequences and amino acid scores to
        the output file.
        """
        if self.out_writer is None:
            return
        # Triply nested lists: results -> batch -> step -> spectrum.
        for (
            scan,
            charge,
            precursor_mz,
            peptide,
            peptide_score,
            aa_scores,
            file_name,
        ) in outputs:
            if len(peptide) == 0:
                continue

            # Compute mass and detokenize
            calc_mass = self.tokenizer.calculate_precursor_ions(
                peptide.unsqueeze(0), torch.tensor([charge]).type_as(peptide)
            )[0]
            peptide = "".join(
                self.tokenizer.detokenize(peptide.unsqueeze(0), join=False)[0]
            )

            self.out_writer.psms.append(
                psm.PepSpecMatch(
                    sequence=peptide,
                    spectrum_id=(file_name, scan),
                    peptide_score=peptide_score,
                    charge=int(charge),
                    calc_mz=calc_mass.item(),
                    exp_mz=precursor_mz,
                    aa_scores=aa_scores,
                )
            )

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

        This is used by pytorch-lightning when preparing the model for
        training.

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
    pair. Higher scores indicate a better match between the peptide and
    spectrum. The amino acid-level scores are also returned.

    Also note that although teacher-forcing is used within this method,
    there is *no training* involved. This is a prediction-only method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.psm_batch_size = None

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor | List],
        *args,
    ) -> List[ms_io.PepSpecMatch]:
        """
        A single prediction step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor | List]
            A batch of MS/MS spectra, as generated by a depthcharge
            dataloader.

        Returns
        -------
        predictions: List[ms_io.PepSpecMatch]
            Predicted PSMs for the given batch of spectra.
        """
        for batch_key in [
            "ms_level",
            "precursor_mz",
            "precursor_charge",
            "mz_array",
            "intensity_array",
        ]:
            batch[batch_key] = batch[batch_key].squeeze(0)

        predictions_all = collections.defaultdict(list)
        for psm_batch in self._psm_batches(batch):
            pred, truth, _, _ = self._forward_step(psm_batch)
            pred = self.softmax(pred)
            batch_peptide_scores, batch_aa_scores = _calc_match_score(
                pred,
                truth,
            )

            for (
                scan,
                charge,
                precursor_mz,
                peptide,
                peptide_score,
                aa_scores,
                file_name,
            ) in zip(
                psm_batch["scan_id"],
                psm_batch["precursor_charge"],
                psm_batch["precursor_mz"],
                self.tokenizer.detokenize(psm_batch["seq"]),
                batch_peptide_scores,
                batch_aa_scores,
                psm_batch["peak_file"],
            ):
                spectrum_id = (file_name, scan)
                predictions_all[spectrum_id].append(
                    psm.PepSpecMatch(
                        sequence=peptide,
                        spectrum_id=spectrum_id,
                        peptide_score=peptide_score,
                        charge=int(charge),
                        calc_mz=self.tokenizer.calculate_precursor_ions(
                            peptide, charge
                        ).item(),
                        exp_mz=precursor_mz.item(),
                        aa_scores=aa_scores,
                        protein=self.protein_database.get_associated_protein(
                            peptide
                        ),
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
                        for spectrum_predictions in predictions_all.values()
                    )
                ]
            )
        )
        return predictions

    def on_predict_batch_end(
        self, outputs: List[psm.PepSpecMatch], *args
    ) -> None:
        """
        Write top scoring batches to the outwriter

        Parameters
        ----------
        outputs : List[psm.PepSpecMatch]
            List of peptide-spectrum matches predicted in the batch.
        *args : tuple
            Additional arguments.
        """
        self.out_writer.psms.extend(outputs)

    def _psm_batches(
        self, batch: Dict[str, torch.Tensor | List]
    ) -> Generator[Dict[str, Union[torch.Tensor, list]], None, None]:
        """
        Generates batches of candidate database PSMs.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor | List]
            One predict batch, from a depthcharge dataloader

        Yields
        ------
        psm_batch : Dict[str, torch.Tensor | List]
            A batch of candidate database PSMs ready for scoring.
        """
        num_candidate_psms = 0
        psm_batch = self._initialize_psm_batch(batch)

        for i, (precursor_mz, precursor_charge) in enumerate(
            zip(batch["precursor_mz"], batch["precursor_charge"])
        ):
            candidate_peps = self.protein_database.get_candidates(
                precursor_mz.item(), precursor_charge.item()
            ).to_list()

            if len(candidate_peps) == 0:
                logger.debug(
                    "No candidate peptides found for spectrum %s with precursor "
                    "charge %d and precursor m/z %f",
                    f"{batch['peak_file'][i]}:{batch['scan_id']}",
                    precursor_charge,
                    precursor_mz,
                )
                continue

            while len(candidate_peps) > 0:
                peps_to_add = min(
                    self.psm_batch_size
                    - (num_candidate_psms % self.psm_batch_size),
                    len(candidate_peps),
                )

                for key in batch.keys():
                    psm_batch[key] += [batch[key][i]] * peps_to_add

                psm_batch["seq"] += candidate_peps[:peps_to_add]
                num_candidate_psms += peps_to_add

                if self._pep_batch_ready(num_candidate_psms):
                    yield self._finalize_psm_batch(psm_batch)
                    psm_batch = self._initialize_psm_batch(batch)

                candidate_peps = candidate_peps[peps_to_add:]

        if (
            not self._pep_batch_ready(num_candidate_psms)
            and num_candidate_psms > 0
        ):
            yield self._finalize_psm_batch(psm_batch)

    def _pep_batch_ready(self, num_candidate_psms: int) -> bool:
        """
        Checks if a batch of candidate PSMs is ready for processing.

        Parameters
        ----------
        num_candidate_psms : int
            Number of candidate PSMs processed so far.

        Returns
        -------
        bool
            True if the batch is ready, False otherwise.
        """
        return (
            num_candidate_psms % self.psm_batch_size == 0
        ) and num_candidate_psms != 0

    def _initialize_psm_batch(self, batch: Dict[str, Any]) -> Dict[str, List]:
        """
        Initializes a new candidate PSM batch.

        Parameters
        ----------
        batch : Dict[str, Any]
            Input batch data to base the initialization on, usually from a
            depthcharge dataloader.

        Returns
        -------
        psm_batch : Dict[str, List]
            A dictionary representing the initialized PSM batch.
        """
        psm_batch = {key: list() for key in batch.keys()}
        psm_batch["seq"] = list()
        return psm_batch

    def _finalize_psm_batch(
        self, psm_batch: Dict[str, List[Any]]
    ) -> Dict[str, torch.Tensor | List[Any]]:
        """
        Prepare a candidate PSM batch for scoring by the Casanovo model.

        Parameters
        ----------
        psm_batch : Dict[str, List[Any]]
            The current PSM batch to finalize.

        Returns
        -------
        finalized_batch : Dict[str, torch.Tensor | List[Any]]
            A finalized PSM batch ready for scoring.
        """
        for key in psm_batch.keys():
            if isinstance(psm_batch[key][0], torch.Tensor):
                psm_batch[key] = torch.stack(psm_batch[key])

        psm_batch["seq"] = self.tokenizer.tokenize(psm_batch["seq"])
        return psm_batch


def _calc_match_score(
    batch_all_aa_scores: torch.Tensor,
    truth_aa_indices: torch.Tensor,
    decoder_reverse: bool = False,
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
    decoder_reverse : bool
        Whether the decoder is reversed.

    Returns
    -------
    peptide_scores: List[float]
        The peptide score for each PSM in the batch.
    aa_scores : List[np.ndarray]
        The amino acid scores for each PSM in the batch.
    """
    # Remove trailing tokens from predictions based on decoder reversal.
    if not decoder_reverse:
        batch_all_aa_scores = batch_all_aa_scores[:, 1:]
    else:
        batch_all_aa_scores = batch_all_aa_scores[:, :-1]

    # Vectorized scoring using efficient indexing.
    rows = (
        torch.arange(batch_all_aa_scores.shape[0])
        .unsqueeze(-1)
        .expand(-1, batch_all_aa_scores.shape[1])
    )
    cols = torch.arange(0, batch_all_aa_scores.shape[1]).expand_as(rows)

    per_aa_scores = batch_all_aa_scores[rows, cols, truth_aa_indices]
    per_aa_scores = per_aa_scores.cpu().detach().numpy()
    per_aa_scores[per_aa_scores == 0] += 1e-10
    score_mask = (truth_aa_indices != 0).cpu().detach().numpy()
    peptide_scores, aa_scores = [], []
    for psm_score, psm_mask in zip(per_aa_scores, score_mask):
        psm_aa_scores, psm_peptide_score = _aa_pep_score(
            psm_score[psm_mask], True
        )
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


def _aa_pep_score(
    aa_scores: np.ndarray, fits_precursor_mz: bool
) -> Tuple[np.ndarray, float]:
    """
    Calculate amino acid and peptide-level confidence score from the raw
    amino acid scores.

    The peptide score is the mean of the raw amino acid scores. The
    amino acid scores are the mean of the raw amino acid scores and the
    peptide score.

    Parameters
    ----------
    aa_scores : np.ndarray
        Amino acid level confidence scores.
    fits_precursor_mz : bool
        Flag indicating whether the prediction fits the precursor m/z
        filter.

    Returns
    -------
    aa_scores : np.ndarray
        The amino acid scores.
    peptide_score : float
        The peptide score.
    """
    peptide_score = np.exp(np.mean(np.log(aa_scores)))
    aa_scores = (aa_scores + peptide_score) / 2
    if not fits_precursor_mz:
        peptide_score -= 1
    return aa_scores, peptide_score


def generate_tgt_mask(sz: int) -> torch.Tensor:
    """Generate a square mask for the sequence.

    Parameters
    ----------
    sz : int
        The length of the target sequence.
    """
    return ~torch.triu(torch.ones(sz, sz, dtype=torch.bool)).transpose(0, 1)
