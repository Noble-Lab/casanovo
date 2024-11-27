"""Data loaders for the de novo sequencing task."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import lightning.pytorch as pl
import numpy as np
import pyarrow as pa
import torch
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


class DeNovoDataModule(pl.LightningDataModule):
    """
    Data loader to prepare MS/MS spectra for a Spec2Pep predictor.

    Parameters
    ----------
    train_paths : str, optional
            A spectrum lance path for model training.
    valid_paths : str, optional
        A spectrum lance path for validation.
    test_paths : str, optional
        A spectrum lance path for evaluation or inference.
    train_batch_size : int
        The batch size to use for training.
    eval_batch_size : int
        The batch size to use for inference.
    n_peaks : Optional[int]
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
    n_workers : int, optional
        The number of workers to use for data loading. By default, the number of
        available CPU cores on the current machine is used.
    max_charge: int
        Remove PSMs which precursor charge higher than specified max_charge
    tokenizer: Optional[PeptideTokenizer]
        Peptide tokenizer for tokenizing sequences
    random_state : Optional[int]
        The NumPy random state. ``None`` leaves mass spectra in the order they
        were parsed.
    shuffle: Optional[bool]
        Should the training dataset be shuffled? Suffling based on specified buffer_size
    buffer_size: Optional[int]
        See more here:
        https://huggingface.co/docs/datasets/v1.11.0/dataset_streaming.html#shuffling-the-dataset-shuffle
    """

    def __init__(
        self,
        train_paths: Optional[Iterable[str]] = None,
        valid_paths: Optional[Iterable[str]] = None,
        test_paths: Optional[str] = None,
        train_batch_size: int = 128,
        eval_batch_size: int = 1028,
        n_peaks: Optional[int] = 150,
        min_mz: float = 50.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        n_workers: Optional[int] = None,
        random_state: Optional[int] = None,
        max_charge: Optional[int] = 10,
        tokenizer: Optional[PeptideTokenizer] = None,
        lance_dir: Optional[str] = None,
        shuffle: Optional[bool] = True,
        buffer_size: Optional[int] = 100_000,
    ):
        super().__init__()
        self.train_paths = train_paths
        self.valid_paths = valid_paths
        self.test_paths = test_paths
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.tokenizer = (
            tokenizer if tokenizer is not None else PeptideTokenizer()
        )
        self.lance_dir = (
            lance_dir
            if lance_dir is not None
            else tempfile.TemporaryDirectory(suffix=".lance").name
        )

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.protein_database = None

        self.n_workers = n_workers if n_workers is not None else os.cpu_count()
        self.shuffle = (
            shuffle if shuffle else None
        )  # set to None if not wanted. Otherwise torch throws and error
        self.buffer_size = buffer_size

        self.valid_charge = np.arange(1, max_charge + 1)
        self.preprocessing_fn = [
            preprocessing.set_mz_range(min_mz=min_mz, max_mz=max_mz),
            preprocessing.remove_precursor_peak(remove_precursor_tol, "Da"),
            preprocessing.filter_intensity(min_intensity, n_peaks),
            preprocessing.scale_intensity("root", 1),
            scale_to_unit_norm,
        ]
        self.custom_field_test_mgf = [
            CustomField("title", lambda x: x["params"]["title"], pa.string()),
        ]
        self.custom_field_test_mzml = [
            CustomField("title", lambda x: x["id"], pa.string()),
        ]

        self.custom_field_anno = [
            CustomField("seq", lambda x: x["params"]["seq"], pa.string())
        ]

    def make_dataset(self, paths, annotated, mode, shuffle):
        """Make spectrum datasets.

        Parameters
        ----------
        paths : Iterable[str]
            Paths to input datasets
        annotated: bool
            True if peptide sequence annotations are available for the test
            data.
        mode: str {"train", "valid", "test"}
            The mode indicating name of lance instance
        shuffle: bool
            Indicates whether to shuffle training data based on buffer_size
        """
        custom_fields = self.custom_field_anno if annotated else []

        if mode == "test":
            if all([Path(f).suffix in (".mgf") for f in paths]):
                custom_fields = custom_fields + self.custom_field_test_mgf
            if all(
                [Path(f).suffix in (".mzml", ".mzxml", ".mzML") for f in paths]
            ):
                custom_fields = custom_fields + self.custom_field_test_mzml

        lance_path = f"{self.lance_dir}/{mode}.lance"

        parse_kwargs = dict(
            preprocessing_fn=self.preprocessing_fn,
            custom_fields=custom_fields,
            valid_charge=self.valid_charge,
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

        if any([Path(f).suffix in (".lance") for f in paths]):
            if annotated:
                dataset = AnnotatedSpectrumDataset.from_lance(
                    paths[0], **anno_dataset_params
                )
            else:
                dataset = SpectrumDataset.from_lance(
                    paths[0], **dataset_params
                )
        else:
            if annotated:
                dataset = AnnotatedSpectrumDataset(
                    spectra=paths,
                    path=lance_path,
                    parse_kwargs=parse_kwargs,
                    **anno_dataset_params,
                )
            else:
                dataset = SpectrumDataset(
                    spectra=paths,
                    path=lance_path,
                    parse_kwargs=parse_kwargs,
                    **dataset_params,
                )

        if shuffle:
            dataset = ShufflerIterDataPipe(
                dataset, buffer_size=self.buffer_size
            )

        return dataset

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
                self.train_dataset = self.make_dataset(
                    self.train_paths,
                    annotated=True,
                    mode="train",
                    shuffle=self.shuffle,
                )
            if self.valid_paths is not None:
                self.valid_dataset = self.make_dataset(
                    self.valid_paths,
                    annotated=True,
                    mode="valid",
                    shuffle=False,
                )
        if stage in (None, "test"):
            if self.test_paths is not None:
                self.test_dataset = self.make_dataset(
                    self.test_paths,
                    annotated=annotated,
                    mode="test",
                    shuffle=False,
                )

    def _make_loader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = False,
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

        Returns
        -------
        torch.utils.data.DataLoader
            A PyTorch DataLoader.
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=self.n_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the training DataLoader."""
        return self._make_loader(
            self.train_dataset, self.train_batch_size, shuffle=self.shuffle
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
        """Get a special dataloader for DB search."""
        return self._make_loader(self.test_dataset, self.eval_batch_size)


def scale_to_unit_norm(spectrum):
    """
    Scaling function used in Casanovo
    slightly differing from the depthcharge implementation
    """
    spectrum._inner._intensity = spectrum.intensity / np.linalg.norm(
        spectrum.intensity
    )
    return spectrum
