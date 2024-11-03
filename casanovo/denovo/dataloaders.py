"""Data loaders for the de novo sequencing task."""

import functools
import logging
import os
from typing import List, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
from depthcharge.data import AnnotatedSpectrumIndex

from ..data import db_utils
from ..data.datasets import (
    AnnotatedSpectrumDataset,
    SpectrumDataset,
)


logger = logging.getLogger("casanovo")


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
    n_peaks : Optional[int]
        The number of top-n most intense peaks to keep in each spectrum. `None`
        retains all peaks.
    min_mz : float
        The minimum m/z to include. The default is 140 m/z, in order to exclude
        TMT and iTRAQ reporter ions.
    max_mz : float
        The maximum m/z to include.
    min_intensity : float
        Remove peaks whose intensity is below `min_intensity` percentage of the
        base peak intensity.
    remove_precursor_tol : float
        Remove peaks within the given mass tolerance in Dalton around the
        precursor mass.
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
        n_peaks: Optional[int] = 150,
        min_mz: float = 50.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        n_workers: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.train_index = train_index
        self.valid_index = valid_index
        self.test_index = test_index
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_intensity = min_intensity
        self.remove_precursor_tol = remove_precursor_tol
        self.n_workers = n_workers if n_workers is not None else os.cpu_count()
        self.rng = np.random.default_rng(random_state)
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.protein_database = None

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
            make_dataset = functools.partial(
                AnnotatedSpectrumDataset,
                n_peaks=self.n_peaks,
                min_mz=self.min_mz,
                max_mz=self.max_mz,
                min_intensity=self.min_intensity,
                remove_precursor_tol=self.remove_precursor_tol,
            )
            if self.train_index is not None:
                self.train_dataset = make_dataset(
                    self.train_index,
                    random_state=self.rng,
                )
            if self.valid_index is not None:
                self.valid_dataset = make_dataset(self.valid_index)
        if stage in (None, "test"):
            make_dataset = functools.partial(
                AnnotatedSpectrumDataset if annotated else SpectrumDataset,
                n_peaks=self.n_peaks,
                min_mz=self.min_mz,
                max_mz=self.max_mz,
                min_intensity=self.min_intensity,
                remove_precursor_tol=self.remove_precursor_tol,
            )
            if self.test_index is not None:
                self.test_dataset = make_dataset(self.test_index)

    def _make_loader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = False,
        collate_fn: Optional[callable] = None,
    ) -> torch.utils.data.DataLoader:
        """
        Create a PyTorch DataLoader.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            A PyTorch Dataset.
        batch_size : int
            The batch size to use.
        shuffle : bool
            Option to shuffle the batches.
        collate_fn : Optional[callable]
            A function to collate the data into a batch.

        Returns
        -------
        torch.utils.data.DataLoader
            A PyTorch DataLoader.
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=prepare_batch if collate_fn is None else collate_fn,
            pin_memory=True,
            num_workers=self.n_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the training DataLoader."""
        return self._make_loader(
            self.train_dataset, self.train_batch_size, shuffle=True
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the validation DataLoader."""
        return self._make_loader(self.valid_dataset, self.eval_batch_size)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the test DataLoader."""
        return self._make_loader(self.test_dataset, self.eval_batch_size)

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the predict DataLoader."""
        return self._make_loader(self.test_dataset, self.eval_batch_size)

    def db_dataloader(self) -> torch.utils.data.DataLoader:
        """Get a special dataloader for DB search"""
        return self._make_loader(
            self.test_dataset,
            self.eval_batch_size,
            collate_fn=functools.partial(
                prepare_psm_batch, protein_database=self.protein_database
            ),
        )


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


def prepare_psm_batch(
    batch: List[Tuple[torch.Tensor, float, int, str]],
    protein_database: db_utils.ProteinDatabase,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, List[str], List[str]]:
    """
    Collate MS/MS spectra into a batch for DB search.

    The MS/MS spectra will be padded so that they fit nicely as a tensor.
    However, the padded elements are ignored during the subsequent steps.

    Parameters
    ----------
    batch : List[Tuple[torch.Tensor, float, int, str]]
        A batch of data from an AnnotatedSpectrumDataset, consisting of for each
        spectrum (i) a tensor with the m/z and intensity peak values, (ii), the
        precursor m/z, (iii) the precursor charge, (iv) the spectrum identifier.
    protein_database : db_utils.ProteinDatabase
        The protein database to use for candidate peptide retrieval.

    Returns
    -------
    all_spectra : torch.Tensor of shape (batch_size, n_peaks, 2)
        The padded mass spectra tensor with the m/z and intensity peak values
        for each spectrum.
    all_precursors : torch.Tensor of shape (batch_size, 3)
        A tensor with the precursor neutral mass, precursor charge, and
        precursor m/z.
    all_spectrum_ids : np.ndarray
        The spectrum identifiers.
    all_peptides : List[str]
        The candidate peptides for each spectrum.
    """
    spectra, precursor_mzs, precursor_charges, spectrum_ids = list(zip(*batch))
    spectra = torch.nn.utils.rnn.pad_sequence(spectra, batch_first=True)

    precursor_mzs_t = torch.tensor(precursor_mzs)
    precursor_charges_t = torch.tensor(precursor_charges)
    precursor_masses_t = (precursor_mzs_t - 1.007276) * precursor_charges_t
    precursors = torch.vstack(
        [precursor_masses_t, precursor_charges_t, precursor_mzs_t]
    ).T.float()

    all_spectra = []
    all_precursors = []
    all_spectrum_ids = []
    all_peptides = []
    for idx in range(len(batch)):
        spec_peptides = protein_database.get_candidates(
            precursor_mzs[idx],
            precursor_charges[idx],
        )
        try:
            all_spectra.append(
                spectra[idx].unsqueeze(0).repeat(len(spec_peptides), 1, 1)
            )
            all_precursors.append(
                precursors[idx].unsqueeze(0).repeat(len(spec_peptides), 1)
            )
            all_spectrum_ids.extend([spectrum_ids[idx]] * len(spec_peptides))
            all_peptides.extend(spec_peptides)
        except ValueError:
            logger.warning(
                "No candidates found for spectrum %s", spectrum_ids[idx]
            )

    return (
        torch.cat(all_spectra, dim=0),
        torch.cat(all_precursors, dim=0),
        all_spectrum_ids,
        all_peptides,
    )
