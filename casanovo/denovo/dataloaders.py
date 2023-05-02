"""Data loaders for the de novo sequencing task."""
import os
from typing import List, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
from depthcharge.data import AnnotatedSpectrumIndex

from ..data.datasets import AnnotatedSpectrumDataset, SpectrumDataset


class DeNovoDataModule(pl.LightningDataModule):
    """
    Data loader to prepare MS/MS spectra for a Spec2Pep predictor.

    Parameters
    ----------
    train_index : Optional[AnnotatedSpectrumIndex]
        The spectrum index file corresponding to the training data.
    valid_index : Optional[AnnotatedSpectrumIndex]
        The spectrum index file corresponding to the validation data.
    test_index : Optional[AnnotatedSpectrumIndex]
        The spectrum index file corresponding to the testing data.
    train_batch_size : int
        The batch size to use for training.
    eval_batch_size : int
        The batch size to use for inference.
    n_workers : int, optional
        The number of workers to use for data loading. By default, the number of
        available CPU cores on the current machine is used.
    random_state : Optional[int]
        The NumPy random state. ``None`` leaves mass spectra in the order they
        were parsed.
    """

    def __init__(
        self,
        train_index: Optional[AnnotatedSpectrumIndex] = None,
        valid_index: Optional[AnnotatedSpectrumIndex] = None,
        test_index: Optional[AnnotatedSpectrumIndex] = None,
        train_batch_size: int = 128,
        eval_batch_size: int = 1028,
        n_workers: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.train_index = train_index
        self.valid_index = valid_index
        self.test_index = test_index
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.n_workers = n_workers if n_workers is not None else os.cpu_count()
        self.rng = np.random.default_rng(random_state)
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None, annotated: bool = True) -> None:
        """
        Set up the PyTorch Datasets.

        Parameters
        ----------
        stage : str {"fit", "validate", "test"}
            The stage indicating which Datasets to prepare. All are prepared by
            default.
        annotated: bool
            True if peptide sequence annotations are available for the test
            data.
        """
        if stage in (None, "fit", "validate"):
            if self.train_index is not None:
                self.train_dataset = AnnotatedSpectrumDataset(
                    self.train_index, random_state=self.rng
                )
            if self.valid_index is not None:
                self.valid_dataset = AnnotatedSpectrumDataset(self.valid_index)
        if stage in (None, "test"):
            make_dataset = (
                AnnotatedSpectrumDataset if annotated else SpectrumDataset
            )
            if self.test_index is not None:
                self.test_dataset = make_dataset(self.test_index)

    def _make_loader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
    ) -> torch.utils.data.DataLoader:
        """
        Create a PyTorch DataLoader.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            A PyTorch Dataset.
        batch_size : int
            The batch size to use.

        Returns
        -------
        torch.utils.data.DataLoader
            A PyTorch DataLoader.
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=prepare_batch,
            pin_memory=True,
            num_workers=self.n_workers,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the training DataLoader."""
        return self._make_loader(self.train_dataset, self.train_batch_size)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the validation DataLoader."""
        return self._make_loader(self.valid_dataset, self.eval_batch_size)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the test DataLoader."""
        return self._make_loader(self.test_dataset, self.eval_batch_size)

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the predict DataLoader."""
        return self._make_loader(self.test_dataset, self.eval_batch_size)


def prepare_batch(
    batch: List[Tuple[torch.Tensor, float, int, str]]
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Collate MS/MS spectra into a batch.

    The MS/MS spectra will be padded so that they fit nicely as a tensor.
    However, the padded elements are ignored during the subsequent steps.

    Parameters
    ----------
    batch : List[Tuple[torch.Tensor, float, int, str]]
        A batch of data from an AnnotatedSpectrumDataset, consisting of for each
        spectrum (i) a tensor with the m/z and intensity peak values, (ii), the
        precursor m/z, (iii) the precursor charge, (iv) the spectrum identifier.

    Returns
    -------
    spectra : torch.Tensor of shape (batch_size, n_peaks, 2)
        The padded mass spectra tensor with the m/z and intensity peak values
        for each spectrum.
    precursors : torch.Tensor of shape (batch_size, 3)
        A tensor with the precursor neutral mass, precursor charge, and
        precursor m/z.
    spectrum_ids : np.ndarray
        The spectrum identifiers (during de novo sequencing) or peptide
        sequences (during training).
    """
    spectra, precursor_mzs, precursor_charges, spectrum_ids = list(zip(*batch))
    spectra = torch.nn.utils.rnn.pad_sequence(spectra, batch_first=True)
    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - 1.007276) * precursor_charges
    precursors = torch.vstack(
        [precursor_masses, precursor_charges, precursor_mzs]
    ).T.float()
    return spectra, precursors, np.asarray(spectrum_ids)
