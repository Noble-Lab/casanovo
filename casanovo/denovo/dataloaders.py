"""Data loaders for the de novo sequencing task."""

import functools
import logging
import math
import os
import pathlib
from typing import Optional, Sequence

import lance
import lightning.pytorch as pl
import numpy as np
import pyteomics
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
        ms_level: Optional[int] = 2,
        scan_width: Optional[int] = 2,
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
        self.ms_level = ms_level
        self.scan_width = scan_width

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
            ms_level=self.ms_level,
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
            if not all(
                (pathlib.Path(path).suffix.lower() == ".mzml")
                for path in paths
            ) and (self.ms_level == 1):
                paths = self._dia_to_dataframe(paths, annotated)

            dataset = Dataset(
                spectra=paths,
                path=lance_path,
                parse_kwargs=parse_params,
                **params,
            )

        if shuffle:
            buffer_batches = max(
                1, math.ceil(self.shuffle_buffer_size / self.train_batch_size)
            )
            dataset = ShufflerIterDataPipe(dataset, buffer_size=buffer_batches)

        return dataset

    def _dia_to_dataframe(self, paths, annotated):
        for spectra in paths:
            f_to_mzrt_to_pep, max_mz, window_size, cycle_time = (
                self.get_centers(spectra)
            )
            for part in f_to_mzrt_to_pep.keys():
                prec_to_spec = self._extract_spectra(
                    spectra,
                    f_to_mzrt_to_pep,
                    part,
                    self.max_peaks,
                    (self.scan_width + 1) * cycle_time,
                    max_mz,
                )
                for key, value in prec_to_spec.items():
                    prec, rt, _ = key
                    if "ms1_scans" not in value:
                        skipped += 1
                        continue

                    scans = np.array(value["scans"], dtype=object)
                    rts = np.array(value["rts"])
                    ms1_scans = np.array(value["ms1_scans"], dtype=object)
                    ms1_rts = np.array(value["ms1_rts"])
                    window_width = value["window_width"]

                    abs_rts = [np.abs(x) for x in rts]
                    sorted_rt_idxs = np.argsort(abs_rts)[: self.scan_width]
                    rts = rts[sorted_rt_idxs]
                    scans = scans[sorted_rt_idxs]

                    abs_ms1_rts = [np.abs(x) for x in ms1_rts]
                    sorted_ms1_rt_idxs = np.argsort(abs_ms1_rts)[
                        : self.scan_width
                    ]
                    ms1_rts = ms1_rts[sorted_ms1_rt_idxs]
                    ms1_scans = ms1_scans[sorted_ms1_rt_idxs]

                    for charge in range(1, self.max_charge + 1):
                        mz_array = []
                        intensity_array = []
                        scan_window_array = []
                        ms_array = []

                        for scan, cur_rt in zip(scans, rts):
                            for mz, intensity in scan:
                                mz_array.append(mz)
                                intensity_array.append(intensity)
                                scan_window_array.append(cur_rt)
                                ms_array.append(2)

                        for scan, cur_rt in zip(ms1_scans, ms1_rts):
                            for mz, intensity in scan:

                                if abs(mz - prec) > window_width + 1:
                                    continue

                                mz_array.append(mz)
                                intensity_array.append(intensity)
                                scan_window_array.append(cur_rt)
                                ms_array.append(1)

                        record = {
                            "peak_file": pathlib.Path(spectra).name,
                            "scan_id": value["center_scan_id"],
                            "precursor_mz": prec,
                            "mz_array": np.asarray(mz_array, dtype=np.float32),
                            "intensity_array": np.asarray(
                                intensity_array, dtype=np.float32
                            ),
                            "scan_window_array": np.asarray(
                                scan_window_array, dtype=np.float32
                            ),
                            "ms_array": np.asarray(ms_array, dtype=np.int8),
                        }

                        if not annotated:
                            record["charge"] = charge

                        yield pl.DataFrame(record)

            logging.warning(
                f"{skipped} spectra were skipped due to missing MS1 scans for file {spectra}"
            )

    def _extract_spectra(
        mzml_file, f_to_mzrt_to_pep, part, top_n, time_width, max_mz
    ):
        prec_to_spec = {}
        with pyteomics.mzml.read(mzml_file) as reader:
            for spec in reader:
                cur_rt = 60 * spec["scanList"]["scan"][0]["scan start time"]
                if spec["ms level"] == 1:
                    for scan_rt in range(
                        int(cur_rt / 10) - 1, int(cur_rt / 10) + 1
                    ):
                        for scan_window in range(max_mz + 1):
                            if (scan_window, scan_rt) in f_to_mzrt_to_pep[
                                part
                            ]:
                                for mz, rt, charge in f_to_mzrt_to_pep[part][
                                    (scan_window, scan_rt)
                                ]:
                                    if np.abs(rt - cur_rt) < time_width:
                                        mzs = spec["m/z array"]
                                        intensities = spec["intensity array"]

                                        sorted_intensity_idxs = np.argsort(
                                            intensities
                                        )[-top_n:]
                                        intensities = intensities[
                                            sorted_intensity_idxs
                                        ]
                                        mzs = mzs[sorted_intensity_idxs]

                                        sorted_mz_idxs = np.argsort(mzs)
                                        intensities = intensities[
                                            sorted_mz_idxs
                                        ]
                                        mzs = mzs[sorted_mz_idxs]

                                        intensities = intensities**0.5
                                        if len(intensities) > 0:
                                            intensities = intensities / np.max(
                                                intensities
                                            )

                                        if (
                                            mz,
                                            rt,
                                            charge,
                                        ) not in prec_to_spec:
                                            prec_to_spec[(mz, rt, charge)] = {
                                                "center_scan_id": spec.get(
                                                    "params"
                                                ).get("id"),
                                            }
                                        if (
                                            "ms1_scans"
                                            not in prec_to_spec[
                                                (mz, rt, charge)
                                            ]
                                        ):
                                            prec_to_spec[(mz, rt, charge)][
                                                "ms1_scans"
                                            ] = []
                                            prec_to_spec[(mz, rt, charge)][
                                                "ms1_rts"
                                            ] = []
                                        prec_to_spec[(mz, rt, charge)][
                                            "ms1_scans"
                                        ].append(
                                            [x for x in zip(mzs, intensities)]
                                        )
                                        prec_to_spec[(mz, rt, charge)][
                                            "ms1_rts"
                                        ].append(cur_rt - rt)
                elif spec["ms level"] == 2:
                    window = spec["precursorList"]["precursor"][0][
                        "isolationWindow"
                    ]
                    window_center = window["isolation window target m/z"]
                    lower_offset = window["isolation window lower offset"]
                    upper_offset = window["isolation window upper offset"]

                    for scan_rt in range(
                        int(cur_rt / 10) - 1, int(cur_rt / 10) + 1
                    ):
                        for scan_window in range(
                            int((window_center - lower_offset) / 10) - 1,
                            int((window_center + upper_offset) / 10) + 1,
                        ):
                            if (scan_window, scan_rt) in f_to_mzrt_to_pep[
                                part
                            ]:
                                for mz, rt, charge in f_to_mzrt_to_pep[part][
                                    (scan_window, scan_rt)
                                ]:
                                    in_mz = (
                                        mz > window_center - lower_offset
                                        and mz < window_center + upper_offset
                                    )
                                    rt_diff = np.abs(rt - cur_rt)
                                    if in_mz and rt_diff < time_width:
                                        mzs = spec["m/z array"]
                                        intensities = spec["intensity array"]

                                        sorted_intensity_idxs = np.argsort(
                                            intensities
                                        )[-top_n:]
                                        intensities = intensities[
                                            sorted_intensity_idxs
                                        ]
                                        mzs = mzs[sorted_intensity_idxs]

                                        sorted_mz_idxs = np.argsort(mzs)
                                        intensities = intensities[
                                            sorted_mz_idxs
                                        ]
                                        mzs = mzs[sorted_mz_idxs]

                                        intensities = intensities**0.5
                                        if len(intensities) > 0:
                                            intensities = intensities / np.max(
                                                intensities
                                            )

                                        if (
                                            mz,
                                            rt,
                                            charge,
                                        ) not in prec_to_spec:
                                            prec_to_spec[(mz, rt, charge)] = {}
                                        if (
                                            "scans"
                                            not in prec_to_spec[
                                                (mz, rt, charge)
                                            ]
                                        ):
                                            prec_to_spec[(mz, rt, charge)][
                                                "scans"
                                            ] = []
                                            prec_to_spec[(mz, rt, charge)][
                                                "rts"
                                            ] = []
                                            prec_to_spec[(mz, rt, charge)][
                                                "window_width"
                                            ] = max(lower_offset, upper_offset)
                                        prec_to_spec[(mz, rt, charge)][
                                            "scans"
                                        ].append(
                                            [x for x in zip(mzs, intensities)]
                                        )
                                        prec_to_spec[(mz, rt, charge)][
                                            "rts"
                                        ].append(cur_rt - rt)
        return prec_to_spec

    def _get_centers(mzml_file):
        f_to_mzrt_to_pep = {}
        max_mz = 0
        num_spectra = 0
        part = 0
        last_rt = 0
        cycle_time = None
        with pyteomics.mzml.read(mzml_file, decode_binary=False) as reader:
            for spec in reader:
                if spec["ms level"] == 1:
                    cur_rt = (
                        60 * spec["scanList"]["scan"][0]["scan start time"]
                    )
                    cycle_time = cur_rt - last_rt
                    last_rt = cur_rt
                if spec["ms level"] == 2:
                    window = spec["precursorList"]["precursor"][0][
                        "isolationWindow"
                    ]
                    window_center = window["isolation window target m/z"]
                    lower_offset = window["isolation window lower offset"]
                    upper_offset = window["isolation window upper offset"]
                    window_size = upper_offset + lower_offset
                    cur_rt = (
                        60 * spec["scanList"]["scan"][0]["scan start time"]
                    )
                    if num_spectra % 50000 == 0:
                        part += 1
                        f_to_mzrt_to_pep[part] = {}

                    num_spectra += 1
                    key = (int(window_center / 10), int(cur_rt / 10))
                    max_mz = max(max_mz, int(window_center / 10))
                    if key in f_to_mzrt_to_pep[part]:
                        f_to_mzrt_to_pep[part][key].append(
                            (window_center, cur_rt, 1)
                        )
                    else:
                        f_to_mzrt_to_pep[part][key] = [
                            (window_center, cur_rt, 1)
                        ]

        return f_to_mzrt_to_pep, max_mz, window_size, cycle_time

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
