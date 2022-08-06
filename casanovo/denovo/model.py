"""A de novo peptide sequencing model."""
import csv
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import depthcharge.masses
import numpy as np
import pytorch_lightning as pl
import torch
from depthcharge.components import ModelMixin, PeptideDecoder, SpectrumEncoder
from depthcharge.models.embed.model import PairedSpectrumEncoder

from . import evaluate
from .. import __version__


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
    n_log : int
        The number of epochs to wait between logging messages.
    tb_summarywriter: Optional[torch.utils.tensorboard.SummaryWriter]
        Object to record performance metrics during training. If ``None``, don't
        use a ``SummarWriter``.
    warmup_iters: int
        The number of warm up iterations for the learning rate scheduler.
    max_iters: int
        The total number of iterations for the learning rate scheduler.
    out_filename: Optional[str]
        The output file name for the prediction results.
    **kwargs : Dict
        Additional keyword arguments passed to the Adam optimizer.
    """

    def __init__(
        self,
        dim_model: int = 128,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0.0,
        dim_intensity: Optional[int] = None,
        custom_encoder: Optional[
            Union[SpectrumEncoder, PairedSpectrumEncoder]
        ] = None,
        max_length: int = 100,
        residues: Union[Dict[str, float], str] = "canonical",
        max_charge: int = 5,
        precursor_mass_tol=50,
        n_log: int = 10,
        tb_summarywriter: Optional[
            torch.utils.tensorboard.SummaryWriter
        ] = None,
        warmup_iters: int = 100_000,
        max_iters: int = 600_000,
        out_filename: Optional[str] = None,
        **kwargs: Dict,
    ):
        super().__init__()

        # Build the model.
        if custom_encoder is not None:
            if isinstance(custom_encoder, PairedSpectrumEncoder):
                self.encoder = custom_encoder.encoder
            else:
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
        self.peptide_mass_calculator = depthcharge.masses.PeptideMass(
            self.residues
        )
        self.stop_token = self.decoder._aa2idx["$"]

        # Logging.
        self.n_log = n_log
        self._history = []
        self.tb_summarywriter = tb_summarywriter

        # Output file during predicting.
        self.out_filename = out_filename

    def forward(
        self, spectra: torch.Tensor, precursors: torch.Tensor
    ) -> Tuple[List[str], torch.Tensor]:
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
        peptides : List[str]
            The predicted peptide sequences for each spectrum.
        aa_scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        """
        aa_scores, tokens = self.greedy_decode(
            spectra.to(self.encoder.device),
            precursors.to(self.decoder.device),
        )
        return [self.decoder.detokenize(t) for t in tokens], aa_scores

    def greedy_decode(
        self, spectra: torch.Tensor, precursors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy decoding of the spectrum predictions.

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
        tokens : torch.Tensor of shape (n_spectra, max_length, n_amino_acids)
            The predicted tokens for each spectrum.
        """
        memories, mem_masks = self.encoder(spectra)
        # Initialize the scores.
        scores = torch.zeros(
            spectra.shape[0], self.max_length + 1, self.decoder.vocab_size + 1
        ).type_as(spectra)
        # Start with the first amino acid predictions.
        scores[:, :1, :], _ = self.decoder(
            None, precursors, memories, mem_masks
        )
        tokens = torch.argmax(scores, axis=2)
        # Keep predicting until a stop token is predicted or max_length is
        # reached.
        # The stop token does not count towards max_length.
        for i in range(2, self.max_length + 2):
            decoded = (tokens == self.stop_token).any(axis=1)
            if decoded.all():
                break
            scores[~decoded, :i, :], _ = self.decoder(
                tokens[~decoded, : (i - 1)],
                precursors[~decoded, :],
                memories[~decoded, :, :],
                mem_masks[~decoded, :],
            )
            tokens = torch.argmax(scores, axis=2)

        return self.softmax(scores), tokens

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
        mode: str = "train",
        *args,
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
            {mode: loss.item()},
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
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str], torch.Tensor]:
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
        peptides : List[str]
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
        train_loss = self.trainer.callback_metrics["CELoss"]["train"].item()
        self._history[-1]["train"] = train_loss
        self._log_history()

    def on_validation_epoch_end(self) -> None:
        """
        Log the validation metrics at the end of each epoch.
        """
        callback_metrics = self.trainer.callback_metrics
        metrics = {
            "epoch": self.trainer.current_epoch,
            "valid": callback_metrics["CELoss"]["valid"].item(),
            "valid_aa_precision": callback_metrics["aa_precision"][
                "valid"
            ].item(),
            "valid_aa_recall": callback_metrics["aa_recall"]["valid"].item(),
            "valid_pep_recall": callback_metrics["pep_recall"]["valid"].item(),
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
        if self.out_filename is None:
            return
        empty_token_score = torch.tensor(0.04)
        with open(self.out_filename, "a") as f_out:
            writer = csv.writer(f_out, delimiter="\t")
            writer.writerow(
                [
                    "PSH",
                    "sequence",
                    "PSM_ID",
                    "accession",
                    "unique",
                    "database",
                    "database_version",
                    "search_engine",
                    "search_engine_score[1]",
                    "modifications",
                    "retention_time",
                    "charge",
                    "exp_mass_to_charge",
                    "calc_mass_to_charge",
                    "spectra_ref",
                    "pre",
                    "post",
                    "start",
                    "end",
                    "opt_ms_run[1]_aa_scores",
                ]
            )
            for batch in results:
                for step in batch:
                    for spectrum_i, precursor, peptide, aa_scores in zip(
                        *step
                    ):
                        peptide = peptide[1:]
                        # Compare the experimental vs calculated precursor m/z.
                        _, precursor_charge, precursor_mz = precursor
                        precursor_charge = precursor_charge.item()
                        precursor_mz = precursor_mz.item()
                        calc_mz = self.peptide_mass_calculator.mass(
                            peptide, precursor_charge
                        )
                        delta_mass_ppm = (
                            abs(calc_mz - precursor_mz)
                            / precursor_mz
                            * 10**6
                        )
                        # Take the scores of the most probable amino acids.
                        top_aa_scores = torch.max(aa_scores, axis=1)[0]
                        # Find the index after the first stop token to check if
                        # decoding was stopped.
                        empty_index = torch.argmax(
                            torch.isclose(
                                top_aa_scores, empty_token_score
                            ).double()
                        )
                        if empty_index > 0:
                            # Omit the stop token.
                            top_aa_scores = top_aa_scores[: empty_index - 1]
                            peptide_score = torch.mean(top_aa_scores).item()
                            # Subtract one if the precursor m/z tolerance is
                            # violated.
                            if delta_mass_ppm > self.precursor_mass_tol:
                                peptide_score -= 1
                            aa_scores = ",".join(
                                map(str, top_aa_scores.cpu().numpy()[::-1])
                            )
                        else:
                            peptide_score, aa_scores = None, None
                        writer.writerow(
                            [
                                "PSM",
                                peptide[1:],
                                spectrum_i,
                                "null",
                                "null",
                                "null",
                                "null",
                                f"[MS, MS:1001456, Casanovo, {__version__}]",
                                peptide_score,
                                # FIXME: Modifications should be specified as
                                #  controlled vocabulary terms.
                                "null",
                                # FIXME: Can we get the retention time from the
                                #  data loader?
                                "null",
                                precursor_charge,
                                precursor_mz,
                                calc_mz,
                                f"ms_run[1]:index={spectrum_i}",
                                "null",
                                "null",
                                "null",
                                "null",
                                aa_scores,
                            ]
                        )

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
