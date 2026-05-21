"""Data loaders for the de novo sequencing task."""

import functools
import logging
import os
import pathlib
from typing import Optional, Sequence

import lightning.pytorch as pl
import numpy as np
import pyarrow as pa
import spectrum_utils.spectrum as sus
import torch.utils.data._utils.collate
from depthcharge.data import (
    AnnotatedSpectrumDataset,
    CustomField,
    SpectrumDataset,
    preprocessing,
)
from depthcharge.tokenizers import PeptideTokenizer
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe

logger = logging.getLogger("casanovo")


def _unique_stems(paths: list) -> list:
    """Return unique file stems for a list of paths.

    Extract the file stem from each path. When the same stem appears
    more than once, subsequent occurrences are disambiguated with a
    ``_1``, ``_2``, ... suffix. The suffix probe skips names that are
    already in use (e.g. an organic ``data_1`` file will not collide
    with a duplicate ``data``).

    Parameters
    ----------
    paths : list
        File paths (strings or Path objects) to extract stems from.

    Returns
    -------
    list of str
        Stems in the same order as *paths*, with duplicates
        disambiguated.
    """
    used: set = set()
    stems = []
    for p in paths:
        stem = pathlib.Path(p).stem
        if stem not in used:
            used.add(stem)
            stems.append(stem)
        else:
            i = 1
            while f"{stem}_{i}" in used:
                i += 1
            unique = f"{stem}_{i}"
            used.add(unique)
            stems.append(unique)
    return stems


class DeNovoDataModule(pl.LightningDataModule):
    """
    Data loader to prepare MS/MS spectra for a Spec2Pep predictor.

    Parameters
    ----------
    lance_dir : str
        Directory to store Lance spectrum index files.
    train_paths : Sequence[str], optional
        Spectrum Lance path(s) for model training.
    valid_paths : Sequence[str], optional
        Spectrum Lance path(s) for validation. Each file gets its own
        DataLoader and contributes to the aggregate ``valid_CELoss``.
    test_paths : Sequence[str], optional
        Spectrum Lance path(s) for evaluation or inference.
    tracking_paths : Sequence[str], optional
        Additional annotated spectrum files logged per-file for monitoring
        only (e.g. detecting catastrophic forgetting); excluded from the
        aggregate ``valid_CELoss`` used for checkpoint selection.
    train_batch_size : int
        The batch size to use for training.
    eval_batch_size : int
        The batch size to use for inference.
    min_peaks : Optional[int]
        The number of peaks for a spectrum to be considered valid.
    max_peaks : Optional[int]
        The number of top-n most intense peaks to keep in each spectrum.
        `None` retains all peaks.
    min_mz : float
        The minimum m/z to include. The default is 140 m/z, in order to
        exclude TMT and iTRAQ reporter ions.
    max_mz : float
        The maximum m/z to include.
    min_intensity : float
        Remove peaks whose intensity is below `min_intensity` percentage
        of the base peak intensity.
    remove_precursor_tol : float
        Remove peaks within the given mass tolerance in Dalton around
        the precursor mass.
    max_charge: int
        Remove PSMs which precursor charge higher than specified
        max_charge.
    tokenizer: Optional[PeptideTokenizer]
        Tokenizer for processing peptide sequences.
    shuffle: Optional[bool]
        Shuffle the training dataset or not. Default is True.
    shuffle_buffer_size: Optional[int]
        Number of samples to buffer for randomly shuffling the training
        data.
    n_workers : int, optional
        The number of workers to use for data loading. By default, the
        number of available CPU cores on the current machine is used.
    """

    def __init__(
        self,
        lance_dir: str,
        train_paths: Optional[Sequence[str]] = None,
        valid_paths: Optional[Sequence[str]] = None,
        test_paths: Optional[Sequence[str]] = None,
        tracking_paths: Optional[Sequence[str]] = None,
        train_batch_size: int = 128,
        eval_batch_size: int = 1028,
        min_peaks: Optional[int] = 20,
        max_peaks: Optional[int] = 150,
        min_mz: float = 50.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        max_charge: Optional[int] = 10,
        tokenizer: Optional[PeptideTokenizer] = None,
        shuffle: Optional[bool] = True,
        shuffle_buffer_size: Optional[int] = 10_000,
        n_workers: Optional[int] = None,
    ):
        super().__init__()

        self.lance_dir = lance_dir

        self.train_paths = train_paths
        self.valid_paths = list(valid_paths or [])
        self.test_paths = test_paths
        self.tracking_paths = list(tracking_paths or [])

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        # Spectrum preprocessing functions.
        self.preprocessing_fn = [
            preprocessing.set_mz_range(min_mz=min_mz, max_mz=max_mz),
            preprocessing.remove_precursor_peak(remove_precursor_tol, "Da"),
            preprocessing.scale_intensity("root", 1),
            preprocessing.filter_intensity(min_intensity, max_peaks),
            functools.partial(_discard_low_quality, min_peaks=min_peaks),
            _scale_to_unit_norm,
        ]
        self.valid_charge = np.arange(1, max_charge + 1)

        self.tokenizer = tokenizer or PeptideTokenizer()

        # Set to None to disable shuffling, otherwise Torch throws an error.
        self.shuffle = shuffle if shuffle else None
        self.shuffle_buffer_size = shuffle_buffer_size

        self.n_workers = n_workers if n_workers is not None else os.cpu_count()

        # Custom fields to read from the input files.
        self.custom_field_anno = CustomField(
            "seq", lambda x: x["params"]["seq"], pa.string()
        )
        self.train_dataset = None
        # Per-file validation datasets: main (monitored) + tracking (log-only).
        self.valid_datasets: list = []
        self.tracking_datasets: list = []
        # val_stems[i] is the filename stem for the i-th val dataloader.
        # Dataloaders 0..n_main_loaders-1 are main; the rest are tracking.
        self.val_stems: list = []
        self.n_main_loaders: int = 0
        self.test_dataset = None
        self.protein_database = None

    def setup(self, stage: str = None, annotated: bool = True) -> None:
        """
        Set up the PyTorch Datasets.

        Parameters
        ----------
        stage : str {"fit", "validate", "test"}
            The stage indicating which Datasets to prepare. All are
            prepared by default.
        annotated: bool
            True if peptide sequence annotations are available for the
            test data.
        """
        if stage in (None, "fit", "validate"):
            if self.train_paths is not None:
                self.train_dataset = self._make_dataset(
                    self.train_paths,
                    annotated=True,
                    mode="train",
                    shuffle=self.shuffle,
                )
                logger.info(
                    "Training dataset contains %d spectra.",
                    self._get_n_spectra(self.train_dataset),
                )
            # Build one dataset per validation file so each gets its own
            # DataLoader and its loss can be logged separately.
            self.valid_datasets = []
            for i, path in enumerate(self.valid_paths):
                self.valid_datasets.append(
                    self._make_dataset(
                        [path],
                        annotated=True,
                        mode=f"valid_{i}",
                        shuffle=False,
                    )
                )
            self.tracking_datasets = []
            for i, path in enumerate(self.tracking_paths):
                self.tracking_datasets.append(
                    self._make_dataset(
                        [path],
                        annotated=True,
                        mode=f"tracking_{i}",
                        shuffle=False,
                    )
                )
            self.n_main_loaders = len(self.valid_datasets)
            self.val_stems = _unique_stems(
                [*self.valid_paths, *self.tracking_paths]
            )
            if self.valid_datasets:
                total = sum(
                    self._get_n_spectra(ds) for ds in self.valid_datasets
                )
                logger.info("Validation dataset contains %d spectra.", total)
        if stage in (None, "test"):
            if self.test_paths is not None:
                self.test_dataset = self._make_dataset(
                    self.test_paths,
                    annotated=annotated,
                    mode="test",
                    shuffle=False,
                )
                logger.info(
                    "Test dataset contains %d spectra.",
                    self._get_n_spectra(self.test_dataset),
                )

    @staticmethod
    def _get_n_spectra(dataset: torch.utils.data.Dataset) -> int:
        """
        Get the number of spectra in a dataset.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The dataset from which to get the number of spectra. This
            may be wrapped in a ShufflerIterDataPipe.

        Returns
        -------
        int
            The number of spectra in the dataset.
        """
        if isinstance(dataset, ShufflerIterDataPipe):
            dataset = dataset.datapipe
        return dataset.n_spectra

    def _make_dataset(
        self, paths, annotated, mode, shuffle
    ) -> torch.utils.data.Dataset:
        """
        Make spectrum datasets.

        Parameters
        ----------
        paths : Iterable[str]
            Paths to read the spectrum input data from.
        annotated: bool
            True if peptide sequence annotations are available for the
            test data.
        mode: str {"train", "valid", "test"}
            The mode indicating name of lance instance
        shuffle: bool
            Shuffle the dataset or not.

        Returns
        -------
        torch.utils.data.Dataset
            A PyTorch Dataset for the given peak files.
        """
        custom_fields = [self.custom_field_anno] if annotated else []
        lance_path = pathlib.Path(f"{self.lance_dir}/{mode}.lance")

        parse_params = dict(
            preprocessing_fn=self.preprocessing_fn,
            valid_charge=self.valid_charge,
            custom_fields=custom_fields,
        )

        dataset_params = dict(
            batch_size=(
                self.train_batch_size
                if mode == "train"
                else self.eval_batch_size
            )
        )
        anno_dataset_params = dataset_params | dict(
            tokenizer=self.tokenizer,
            annotations="seq",
        )

        if annotated:
            Dataset, params = AnnotatedSpectrumDataset, anno_dataset_params
        else:
            Dataset, params = SpectrumDataset, dataset_params

        if (
            len(paths) == 1
            and pathlib.Path(paths[0]).suffix.lower() == ".lance"
        ):
            dataset = Dataset.from_lance(paths[0], **params)
        else:
            dataset = Dataset(
                spectra=paths,
                path=lance_path,
                parse_kwargs=parse_params,
                **params,
            )

        if shuffle:
            dataset = ShufflerIterDataPipe(
                dataset, buffer_size=self.shuffle_buffer_size
            )

        return dataset

    def _make_loader(
        self, dataset: torch.utils.data.Dataset, shuffle: bool = False
    ) -> torch.utils.data.DataLoader:
        """
        Create a PyTorch DataLoader.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            A PyTorch Dataset.
        shuffle : bool
            Option to shuffle the batches.

        Returns
        -------
        torch.utils.data.DataLoader
            A PyTorch DataLoader.
        """
        return DataLoader(
            dataset,
            batch_size=None,
            pin_memory=True,
            num_workers=self.n_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the training DataLoader."""
        return self._make_loader(self.train_dataset, shuffle=self.shuffle)

    def val_dataloader(self) -> list:
        """Get validation DataLoaders.

        Returns one DataLoader per validation file, ordered with main
        files first (indices ``0..n_main_loaders-1``) followed by
        tracking-only files. Lightning dispatches each loader's
        batches with a ``dataloader_idx`` that maps 1-to-1 to the
        entries in ``val_stems``.

        Returns
        -------
        list of torch.utils.data.DataLoader
            One loader per validation and tracking file.
        """
        return [
            self._make_loader(ds)
            for ds in self.valid_datasets + self.tracking_datasets
        ]

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the test DataLoader."""
        return self._make_loader(self.test_dataset)

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the predict DataLoader."""
        return self._make_loader(self.test_dataset)

    def db_dataloader(self) -> torch.utils.data.DataLoader:
        """Get a special dataloader for DB search."""
        return self._make_loader(self.test_dataset)


def _discard_low_quality(
    spectrum: sus.MsmsSpectrum, min_peaks: int
) -> sus.MsmsSpectrum:
    """
    Discard low quality spectra.

    Spectra are considered low quality if:
    - They have fewer than 20 peaks.

    Parameters
    ----------
    spectrum : sus.MsmsSpectrum
        The spectrum to check for low quality.
    min_peaks : int
        The minimum number of peaks required for a spectrum to be
        considered high quality.

    Returns
    -------
    sus.MsmsSpectrum
        The spectrum if it is of high quality, otherwise None.

    Raises
    ------
    ValueError
        If the spectrum is of low quality.
    """
    if len(spectrum.mz) < min_peaks:
        raise ValueError("Insufficient number of peaks")
    return spectrum


def _scale_to_unit_norm(spectrum: sus.MsmsSpectrum) -> sus.MsmsSpectrum:
    """
    Scale fragment ion intensities to unit norm.

    Parameters
    ----------
    spectrum : sus.MsmsSpectrum
        The spectrum for which to scale the fragment ion intensities.

    Returns
    -------
    sus.MsmsSpectrum
        The spectrum with scaled fragment ion intensities.
    """
    spectrum._inner._intensity = spectrum.intensity / np.linalg.norm(
        spectrum.intensity
    )
    return spectrum
