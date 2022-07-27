"""A de novo peptide sequencing model"""
import logging, time, random, os, csv

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter

from depthcharge.components import SpectrumEncoder, PeptideDecoder, ModelMixin
from depthcharge.models.embed.model import PairedSpectrumEncoder
from .evaluate import batch_aa_match, calc_eval_metrics

LOGGER = logging.getLogger(__name__)


class Spec2Pep(pl.LightningModule, ModelMixin):
    """A Transformer model for de novo peptide sequencing.

    Use this model in conjunction with a pytorch-lightning Trainer.

    Parameters
    ----------
    dim_model : int, optional
        The latent dimensionality used by the Transformer model.
    n_head : int, optional
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    dim_intensity : int or None, optional
        The number of features to use for encoding peak intensity.
        The remaining (``dim_model - dim_intensity``) are reserved for
        encoding the m/z value. If ``None``, the intensity will be projected
        up to ``dim_model`` using a linear layer, then summed with the m/z
        emcoding for each peak.
    custom_encoder : SpectrumEncoder or PairedSpectrumEncoder, optional
        A pretrained encoder to use. The ``dim_model`` of the encoder must
        be the same as that specified by the ``dim_model`` parameter here.
    max_length : int, optional
        The maximum peptide length to decode.
    residues: Dict or str {"massivekb", "canonical"}, optional
        The amino acid dictionary and their masses. By default this is only
        the 20 canonical amino acids, with cysteine carbamidomethylated. If
        "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int, optional
        The maximum charge state to consider.
    n_log : int, optional
        The number of epochs to wait between logging messages.
    tb_summarywriter: torch.utils.tensorboard.SummaryWriter object or None, optional
        Object to record performance metrics during training. If ``None``, don't use a SummarWriter
    warmup_iters: int, optional
        Number of warm up iterations for learning rate scheduler
    max_iters: int, optional
        Total number of iterations for learning rate scheduler
    output_path: str, optional
        Path to write csv file with denovo peptide sequences
    **kwargs : Dict
        Keyword arguments passed to the Adam optimizer
    """

    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
        dim_intensity=None,
        custom_encoder=None,
        max_length=100,
        residues="canonical",
        max_charge=5,
        n_log=10,
        tb_summarywriter=None,
        warmup_iters=100000,
        max_iters=600000,
        output_path="",
        **kwargs,
    ):
        """Initialize a Spec2Pep model"""
        super().__init__()

        # Writable
        self.max_length = max_length
        self.n_log = n_log

        self.residues = residues

        # Build the model
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

        # Things for training
        self._history = []
        self.opt_kwargs = kwargs
        self.stop_token = self.decoder._aa2idx["$"]

        self.tb_summarywriter = tb_summarywriter

        self.warmup_iters = warmup_iters
        self.max_iters = max_iters

        # Store de novo sequences to be saved
        self.denovo_seqs = []
        self.output_path = output_path

    def forward(self, spectra, precursors):
        """Sequence a batch of mass spectra.

        Parameters
        ----------
        spectrum : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.
        precursors : torch.Tensor of size (n_spectra, 2)
            The measured precursor mass (axis 0) and charge (axis 1) of each
            tandem mass spectrum.

        Returns
        -------
        sequences : list or str
            The sequence for each spectrum.
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The score for each amino acid.
        """
        spectra = spectra.to(self.encoder.device)
        precursors = precursors.to(self.decoder.device)
        scores, tokens = self.greedy_decode(spectra, precursors)
        sequences = [self.decoder.detokenize(t) for t in tokens]
        return sequences, scores

    def predict_step(self, batch, *args):
        """Sequence a batch of mass spectra.

        Note that this is used within the context of a pytorch-lightning
        Trainer to generate a prediction.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            A batch is expected to contain mass spectra (index 0) and the
            precursor mass and charge (index 1). It may have more indices,
            but these will be ignored.


        Returns
        -------
        sequences : list or str
            The sequence for each spectrum.
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The score for each amino acid.
        """
        return self(batch[0], batch[1])

    def greedy_decode(self, spectra, precursors):
        """Greedy decode the spectra.

        Parameters
        ----------
        spectrum : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.
        precursors : torch.Tensor of size (n_spectra, 2)
            The measured precursor mass (axis 0) and charge (axis 1) of each
            tandem mass spectrum.

        Returns
        -------
        tokens : torch.Tensor of shape (n_spectra, max_length, n_amino_acids)
            The token sequence for each spectrum.
        scores : torch.Tensor of shape (n_spectra, max_length, n_amino_acids)
            The score for each amino acid.
        """
        memories, mem_masks = self.encoder(spectra)

        # initialize scores:
        scores = torch.zeros(
            spectra.shape[0],
            self.max_length + 1,
            self.decoder.vocab_size + 1,
        )
        scores = scores.type_as(spectra)

        # The first prediction:
        scores[:, :1, :], _ = self.decoder(
            None,
            precursors,
            memories,
            mem_masks,
        )

        tokens = torch.argmax(scores, axis=2)

        # Keep predicting until all have a stop token or max_length is reached.
        # Don't count the stop token toward max_length though.
        for idx in range(2, self.max_length + 2):
            decoded = (tokens == self.stop_token).any(axis=1)
            if decoded.all():
                break

            scores[~decoded, :idx, :], _ = self.decoder(
                tokens[~decoded, : (idx - 1)],
                precursors[~decoded, :],
                memories[~decoded, :, :],
                mem_masks[~decoded, :],
            )
            tokens = torch.argmax(scores, axis=2)

        return self.softmax(scores), tokens

    def _step(self, spectra, precursors, sequences):
        """The forward learning step.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.
        precursors : torch.Tensor of size (n_spectra, 2)
            The measured precursor mass (axis 0) and charge (axis 1) of each
            tandem mass spectrum.
        sequences : list or str of length n_spectra
            The partial peptide sequences to predict.

        Returns
        -------
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The raw scores for each amino acid at each position.
        tokens : torch.Tensor of shape (n_spectra, length)
            The best token at each sequence position
        """
        memory, mem_mask = self.encoder(spectra)
        scores, tokens = self.decoder(sequences, precursors, memory, mem_mask)
        return scores, tokens

    def training_step(self, batch, *args):
        """A single training step

        Note that this is used within the context of a pytorch-lightning
        Trainer to generate a prediction.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            A batch is expected to contain mass spectra (index 0), the
            precursor mass and charge (index 1), and the peptide sequence
            (index 2)

        Returns
        -------
        torch.Tensor
            The loss.
        """

        spectra, precursors, sequences = batch
        pred, truth = self._step(spectra, precursors, sequences)

        pred = pred[:, :-1, :].reshape(-1, self.decoder.vocab_size + 1)
        loss = self.celoss(pred, truth.flatten())
        self.log(
            "CELoss",
            {"train": loss.item()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, *args):
        """A single validation step

        Note that this is used within the context of a pytorch-lightning
        Trainer to generate a prediction.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            A batch is expected to contain mass spectra (index 0), the
            precursor mass and charge (index 1), and the peptide sequence
            (index 2)

        Returns
        -------
        torch.Tensor
            The loss.
        """
        spectra, precursors, sequences = batch
        pred, truth = self._step(spectra, precursors, sequences)
        pred = pred[:, :-1, :].reshape(-1, self.decoder.vocab_size + 1)
        loss = self.celoss(pred, truth.flatten())
        self.log(
            "CELoss",
            {"valid": loss.item()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # De novo sequence the batch
        pred_seqs, scores = self.predict_step(batch)

        # Temporary solution to predictions with multiple stop token '$', filter them out
        filtered_pred_seqs = []
        filtered_sequences = []
        for i in range(len(pred_seqs)):
            if len(pred_seqs[i]) > 0:
                ps = pred_seqs[i]
                if pred_seqs[i][0] == "$":
                    ps = pred_seqs[i][1:]  # Remove stop token
                if "$" not in ps and len(ps) > 0:
                    filtered_pred_seqs += [ps]
                    filtered_sequences += [sequences[i]]

        # Find AA and peptide matches
        all_aa_match, orig_total_num_aa, pred_total_num_aa = batch_aa_match(
            filtered_pred_seqs,
            filtered_sequences,
            self.decoder._peptide_mass.masses,
            "best",
        )

        # Calculate evaluation metrics based on matches
        aa_precision, aa_recall, pep_recall = calc_eval_metrics(
            all_aa_match, orig_total_num_aa, pred_total_num_aa
        )

        self.log(
            "aa_precision",
            {"valid": aa_precision},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "aa_recall",
            {"valid": aa_recall},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "pep_recall",
            {"valid": pep_recall},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def test_step(self, batch, *args):
        """A single test step

        Note that this is used within the context of a pytorch-lightning
        Trainer to generate a prediction.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            A batch is expected to contain mass spectra (index 0), the
            precursor mass and charge (index 1), and the spectrum identifier
            (index 2)

        """
        # De novo sequence the batch
        pred_seqs, scores = self.predict_step(batch)
        spectrum_order_id = batch[-1]
        self.denovo_seqs += [(spectrum_order_id, pred_seqs, scores)]

    def on_train_epoch_end(self):
        """Log the training loss.

        This is a pytorch-lightning hook.
        """
        train_loss = self.trainer.callback_metrics["CELoss"]["train"].item()
        self._history[-1]["train"] = train_loss

    def on_validation_epoch_end(self):
        """Log the epoch metrics to self.history.

        This is a pytorch-lightning hook.
        """
        metrics = {
            "epoch": self.trainer.current_epoch,
            "valid": self.trainer.callback_metrics["CELoss"]["valid"].item(),
            "valid_aa_precision": self.trainer.callback_metrics[
                "aa_precision"
            ]["valid"].item(),
            "valid_aa_recall": self.trainer.callback_metrics["aa_recall"][
                "valid"
            ].item(),
            "valid_pep_recall": self.trainer.callback_metrics["pep_recall"][
                "valid"
            ].item(),
        }
        self._history.append(metrics)

    def on_test_epoch_end(self):
        """Write de novo sequences and confidence scores to csv file.

        This is a pytorch-lightning hook.
        """
        with open(
            os.path.join(str(self.output_path), "casanovo_output.csv"), "w"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(
                ["spectrum_id", "denovo_seq", "peptide_score", "aa_scores"]
            )

            for batch in self.denovo_seqs:
                scores = batch[2].cpu()  # transfer to cpu in case in gpu

                for i in range(len(batch[0])):
                    top_scores = torch.max(scores[i], axis=1)[
                        0
                    ]  # take the score of most probable AA
                    empty_index = torch.where(top_scores == 0.04)[
                        0
                    ]  # find the indices of positions after stop token

                    if len(empty_index) > 0:  # check if decoding was stopped
                        last_index = (
                            empty_index[0] - 1
                        )  # select index of the last AA

                        if (
                            last_index >= 1
                        ):  # check if peptide is at least one AA long
                            top_scores_list = top_scores[
                                :last_index
                            ].tolist()  # omit the stop token
                            peptide_score = np.mean(top_scores_list)
                            aa_scores = list(reversed(top_scores_list))

                        else:
                            peptide_score = None
                            aa_scores = None

                    else:
                        peptide_score = None
                        aa_scores = None

                    writer.writerow(
                        [
                            batch[0][i],
                            batch[1][i][1:],
                            peptide_score,
                            aa_scores,
                        ]
                    )

    def on_epoch_end(self):
        """Print log to console, if requested."""

        if len(self._history) > 0:
            # Print only if all output for the current epoch is recorded
            if len(self._history[-1]) == 6:
                if len(self._history) == 1:
                    LOGGER.info(
                        "---------------------------------------------------------------------------------------------------------"
                    )
                    LOGGER.info(
                        "  Epoch |   Train Loss  |  Valid Loss | Valid AA precision | Valid AA recall | Valid Peptide recall "
                    )
                    LOGGER.info(
                        "---------------------------------------------------------------------------------------------------------"
                    )

                metrics = self._history[-1]
                if not metrics["epoch"] % self.n_log:
                    LOGGER.info(
                        "  %5i | %13.6f | %13.6f | %13.6f | %13.6f | %13.6f ",
                        metrics["epoch"],
                        metrics.get("train", np.nan),
                        metrics.get("valid", np.nan),
                        metrics.get("valid_aa_precision", np.nan),
                        metrics.get("valid_aa_recall", np.nan),
                        metrics.get("valid_pep_recall", np.nan),
                    )
                    # Add metrics to SummaryWriter object if provided
                    if self.tb_summarywriter is not None:
                        self.tb_summarywriter.add_scalar(
                            "loss/train_crossentropy_loss",
                            metrics.get("train", np.nan),
                            metrics["epoch"] + 1,
                        )
                        self.tb_summarywriter.add_scalar(
                            "loss/dev_crossentropy_loss",
                            metrics.get("valid", np.nan),
                            metrics["epoch"] + 1,
                        )

                        self.tb_summarywriter.add_scalar(
                            "eval/dev_aa_precision",
                            metrics.get("valid_aa_precision", np.nan),
                            metrics["epoch"] + 1,
                        )
                        self.tb_summarywriter.add_scalar(
                            "eval/dev_aa_recall",
                            metrics.get("valid_aa_recall", np.nan),
                            metrics["epoch"] + 1,
                        )
                        self.tb_summarywriter.add_scalar(
                            "eval/dev_pep_recall",
                            metrics.get("valid_pep_recall", np.nan),
                            metrics["epoch"] + 1,
                        )

    def configure_optimizers(self):
        """Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for
        training.

        Returns
        -------
        torch.optim.Adam
            The intialized Adam optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), **self.opt_kwargs)

        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.warmup_iters, max_iters=self.max_iters
        )

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Learning rate scheduler with linear warm up followed by cosine shaped decay.
    Parameters
    ----------
    optimizer :  torch.optim
        Optimizier object
    warmup :  int
        Number of warm up iterations
    max_iters :  torch.optim
        Total number of iterations
    """

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
