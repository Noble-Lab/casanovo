"""Data loaders for the de novo sequencing task.

This module also ensure consistent train, validation, and test splits.
"""
import os
from functools import partial

import torch
import numpy as np
import pytorch_lightning as pl

from ..data import AnnotatedSpectrumDataset, SpectrumDataset


class DeNovoDataModule(pl.LightningDataModule):
    """Prepare data for a Spec2Pep.

    Parameters
    ----------
    train_index : AnnotatedSpectrumIndex
        The spectrum index file for training.
    valid_index : AnnotatedSpectrumIndex
        The spectrum index file for validation.
    test_index : AnnotatedSpectrumIndex
        The spectrum index file for testing.
    batch_size : int, optional
        The batch size to use for training and evaluations
    n_peaks : int, optional
        Keep only the top-n most intense peaks in any spectrum. ``None``
        retains all of the peaks.
    min_mz : float, optional
        The minimum m/z to include. The default is 140 m/z, in order to
        exclude TMT and iTRAQ reporter ions.
    max_mz : float, optional
        The maximum m/z to include. 
    min_intensity : float, optional
        Remove peaks whose intensity is below `min_intensity` percentage
        of the intensity of the most intense peak
    fragment_tol_mass : float, optional
        Fragment mass tolerance around the precursor mass in Da to remove the
        precursor peak.       
    num_workers : int, optional
        The number of workers to use for data loading. By default, the number
        of available CPU cores on the current machine is used.
    random_state : int or Generator, optional.
        The numpy random state. ``None`` leaves mass spectra in the order
        they were parsed.
    """

    def __init__(
        self,
        train_index=None,
        valid_index=None,
        test_index=None,
        batch_size=128,
        n_peaks=200,
        min_mz=140,
        max_mz=2500,
        min_intensity=0.01,
        fragment_tol_mass=2,        
        num_workers=None,
        random_state=None,
        preprocess_spec=False
    ):
        """Initialize the PairedSpectrumDataModule."""
        super().__init__()
        self.train_index = train_index
        self.valid_index = valid_index
        self.test_index = test_index
        self.batch_size = batch_size
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_intensity = min_intensity
        self.fragment_tol_mass = fragment_tol_mass       
        self.num_workers = num_workers
        self.rng = np.random.default_rng(random_state)
        self.preprocess_spec = preprocess_spec
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        if self.num_workers is None:
            self.num_workers = os.cpu_count()

    def setup(self, stage=None, annotated=True):
        """
        Sets up the PyTorch Datasets.

        :param stage: The stage indicating which Datasets to prepare. All are prepared by default.
        :type stage: str {"fit", "validate", "test"}
        :param annotated: True if peptide sequence annotations available for test data
        :type annotated: bool
        """

        if stage in (None, "fit", "validate"):
            make_dataset = partial(
                AnnotatedSpectrumDataset,
                n_peaks=self.n_peaks,
                min_mz=self.min_mz,
                max_mz=self.max_mz,
                min_intensity=self.min_intensity,
                fragment_tol_mass=self.fragment_tol_mass,                
                preprocess_spec=self.preprocess_spec,
            )
            if self.train_index is not None:
                self.train_dataset = make_dataset(
                    self.train_index,
                    random_state=self.rng,
                )
            if self.valid_index is not None:
                self.valid_dataset = make_dataset(self.valid_index)

        if stage in (None, "test"):
            if annotated == True:             
                make_dataset = partial(
                    AnnotatedSpectrumDataset,
                    n_peaks=self.n_peaks,
                    min_mz=self.min_mz,
                    max_mz=self.max_mz,
                    min_intensity=self.min_intensity,
                    fragment_tol_mass=self.fragment_tol_mass,                   
                    preprocess_spec=self.preprocess_spec
                )        
            else:    
                make_dataset = partial(
                    SpectrumDataset,
                    n_peaks=self.n_peaks,
                    min_mz=self.min_mz,
                    max_mz=self.max_mz,
                    min_intensity=self.min_intensity,
                    fragment_tol_mass=self.fragment_tol_mass,                      
                    preprocess_spec=self.preprocess_spec
                )                
            if self.test_index is not None:
                self.test_dataset = make_dataset(self.test_index)
                
    def _make_loader(self, dataset):
        """
        Creates a PyTorch DataLoader.

        :param dataset: A PyTorch dataset
        :type dataset: torch.utils.data.Dataset
        :return: A PyTorch DataLoader
        :rtype: torch.utils.data.DataLoader
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=prepare_batch,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        """Get the training DataLoader."""
        return self._make_loader(self.train_dataset)

    def val_dataloader(self):
        """Get the validation DataLoader."""
        return self._make_loader(self.valid_dataset)

    def test_dataloader(self):
        """Get the test DataLoader."""
        return self._make_loader(self.test_dataset)


def prepare_batch(batch):
    """
    This is the collate function

    The mass spectra must be padded so that they fit nicely as a tensor.
    However, the padded elements are ignored during the subsequent steps.

    :param batch: A batch of data from an AnnotatedSpectrumDataset.
    :type batch: tuple of tuple of torch.Tensor
    :return: spectra - The mass spectra to sequence, where ``X[:, :, 0]`` are the m/z values and ``X[:, :, 1]`` are their associated intensities.
    :rtype: torch.Tensor of shape (batch_size, n_peaks, 2)
    :return: precursors - The precursor mass and charge state.
    :rtype: torch.Tensor of shape (batch_size, 2)
    :returns: sequence_or_ids - The peptide sequence annotations in training, the spectrum identifier in de novo sequencing.
    :rtype: list of str
    """
    
    spec, mz, charge, sequence_or_ids = list(zip(*batch))
    charge = torch.tensor(charge)
    mass = (torch.tensor(mz) - 1.007276) * charge
    precursors = torch.vstack([mass, charge]).T.float()
    spec = torch.nn.utils.rnn.pad_sequence(spec, batch_first=True)
    return spec, precursors, sequence_or_ids
