"""A de novo peptide sequencing model."""
import csv
import os
import collections
import heapq
import itertools
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import re

import depthcharge.masses
import einops
import torch
import numpy as np
import lightning.pytorch as pl
from torch.utils.tensorboard import SummaryWriter
from depthcharge.components import ModelMixin, PeptideDecoder, SpectrumEncoder

from . import evaluate
from .. import config
from ..data import ms_io

logger = logging.getLogger("casanovo")


class FullAttentionDecoder(PeptideDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_tgt_mask(self, sz):
        """Override to return a mask of all False values."""
        return torch.zeros(sz, sz, dtype=torch.bool)


class Spec2Pep(pl.LightningModule, ModelMixin):
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
    tb_summarywriter : Optional[Path]
        Folder path to record performance metrics during training. If
        ``None``, don't use a ``SummaryWriter``.
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
        max_peptide_len: int = 100,
        residues: Union[Dict[str, float], str] = "canonical",
        max_charge: int = 5,
        precursor_mass_tol: float = 50,
        isotope_error_range: Tuple[int, int] = (0, 1),
        min_peptide_len: int = 6,
        n_beams: int = 1,
        top_match: int = 1,
        n_log: int = 10,
        tb_summarywriter: Optional[Path] = None,
        train_label_smoothing: float = 0.01,
        warmup_iters: int = 100_000,
        cosine_schedule_period_iters: int = 600_000,
        out_writer: Optional[ms_io.MztabWriter] = None,
        calculate_precision: bool = False,
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
        self.decoder = FullAttentionDecoder(
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
        self.peptide_mass_calculator = depthcharge.masses.PeptideMass(
            self.residues
        )
        self.stop_token = self.decoder._aa2idx["$"]

        # Logging.
        self.calculate_precision = calculate_precision
        self.n_log = n_log
        self._history = []
        if tb_summarywriter is not None:
            self.tb_summarywriter = SummaryWriter(str(tb_summarywriter))
        else:
            self.tb_summarywriter = None

        # Output writer during predicting.
        self.out_writer: None

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
            Axis 0 represents an MS/MS spectrum, axis 1 contains the
            peaks in the MS/MS spectrum, and axis 2 is essentially a
            2-tuple specifying the m/z-intensity pair for each peak.
            These should be zero-padded, such that all the spectra in
            the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge
            (axis 1), and precursor m/z (axis 2) of each MS/MS spectrum.
        sequences : List[str] of length n_spectra
            The partial peptide sequences to predict.

        Returns
        -------
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        tokens : torch.Tensor of shape (n_spectra, length)
            The input padded tokens.
        """
        if sequences is not None:  # Training
            padded_sequences = []
            pattern = r"([A-Z](?:[+-]\d+\.\d+)?|[+-]\d+\.\d+)"  # handles modifications

            for seq in sequences:
                parsed = re.findall(pattern, seq)
                padded_tensor = torch.tensor(
                    [0] * len(parsed),
                    dtype=torch.long,
                    device=precursors.device,
                )
                padded_sequences.append(padded_tensor)

            return self.decoder(
                padded_sequences, precursors, *self.encoder(spectra)
            )
        else:  # Inference
            batch_size = spectra.shape[0]
            padded_sequences = torch.zeros(
                (batch_size, self.max_peptide_len),
                dtype=torch.long,
                device=precursors.device,
            )
            return self.decoder(
                padded_sequences, precursors, *self.encoder(spectra)
            )

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
            A batch of (i) MS/MS spectra, (ii) precursor information,
            (iii) peptide sequences as torch Tensors.
        mode : str
            Logging key to describe the current stage.

        Returns
        -------
        torch.Tensor
            The loss of the training step.
        """
        pred, tokens = self._forward_step(*batch)
        sequences = batch[2]
        print(tokens)

        truth = [self.decoder.tokenize(s) for s in sequences]
        truth = torch.nn.utils.rnn.pad_sequence(truth, batch_first=True)

        # Align pred shape and remove the last token for prediction
        pred = pred.reshape(-1, self.decoder.vocab_size + 1)

        print(pred, truth)
        print(pred.shape, truth.shape)

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
            A batch of (i) MS/MS spectra, (ii) precursor information,
            (iii) peptide sequences.

        Returns
        -------
        torch.Tensor
            The loss of the validation step.
        """
        # Compute validation loss
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
    ) -> List[dict]:
        """
        A single prediction step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A batch of (i) MS/MS spectra, (ii) precursor information,
            (iii) spectrum identifiers as torch Tensors.

        Returns
        -------
        predictions: List[dict]
            List of dictionaries with prediction info.
        """

        spectra, precursor_info, spectrum_ids = batch
        pred, __ = self._forward_step(spectra, precursor_info, None)
        pred = self.softmax(pred)

        predicted_tokens = torch.argmax(pred, dim=-1)

        # Per-AA confidence: the probability of each predicted token
        per_aa_conf = torch.gather(
            pred, 2, predicted_tokens.unsqueeze(-1)
        ).squeeze(-1)

        # Metadata
        precursor_mz = precursor_info[:, 0].tolist()
        precursor_charge = precursor_info[:, 1].tolist()
        spectrum_ids = [sid for sid in spectrum_ids]  # adapt if needed

        predictions = []
        for (
            precursor_charge,
            precursor_mz,
            spectrum_i,
            pred_tokens,
            aa_scores,
        ) in zip(
            batch[1][:, 1].cpu().detach().numpy(),
            batch[1][:, 2].cpu().detach().numpy(),
            batch[2],
            predicted_tokens,
            per_aa_conf,
        ):
            peptide = self.decoder.detokenize(pred_tokens)[1:]
            aa_scores = aa_scores[: len(peptide)].detach().cpu().numpy()
            aa_scores = aa_scores[::-1] if self.decoder.reverse else aa_scores
            peptide = "".join(peptide)
            peptide_score = _aa_pep_score(aa_scores, True)

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
        batch: Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray],
        *args,
    ) -> List[ms_io.PepSpecMatch]:
        """
        A single prediction step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]
            A batch of (i) MS/MS spectra, (ii) precursor information,
            (iii) spectrum identifiers, (iv) candidate peptides.

        Returns
        -------
        predictions: List[ms_io.PepSpecMatch]
            Predicted PSMs for the given batch of spectra.
        """
        predictions_all = collections.defaultdict(list)
        for start_i in range(0, len(batch[0]), self.psm_batch_size):
            psm_batch = [
                b[start_i : start_i + self.psm_batch_size] for b in batch
            ]
            pred, truth = self.forward(
                psm_batch[0], psm_batch[1], psm_batch[3]
            )
            pred = self.softmax(pred)
            batch_peptide_scores, batch_aa_scores = _calc_match_score(
                pred, truth, self.decoder.reverse
            )
            for (
                charge,
                precursor_mz,
                spectrum_i,
                peptide_score,
                aa_scores,
                peptide,
            ) in zip(
                psm_batch[1][:, 1].cpu().detach().numpy(),
                psm_batch[1][:, 2].cpu().detach().numpy(),
                psm_batch[2],
                batch_peptide_scores,
                batch_aa_scores,
                psm_batch[3],
            ):
                spectrum_i = tuple(spectrum_i)
                predictions_all[spectrum_i].append(
                    ms_io.PepSpecMatch(
                        sequence=peptide,
                        spectrum_id=spectrum_i,
                        peptide_score=peptide_score,
                        charge=int(charge),
                        calc_mz=self.peptide_mass_calculator.mass(
                            peptide, charge
                        ),
                        exp_mz=precursor_mz,
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
