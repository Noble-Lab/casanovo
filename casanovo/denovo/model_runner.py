"""Training and testing functionality for the de novo peptide sequencing
model."""

import glob
import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Union
from datetime import datetime

import lightning.pytorch as pl
import torch

from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch.loggers import TensorBoardLogger

from depthcharge.tokenizers import PeptideTokenizer
from depthcharge.tokenizers.peptides import MskbPeptideTokenizer

from ..config import Config
from ..data import ms_io
from ..denovo.dataloaders import DeNovoDataModule
from ..denovo.model import Spec2Pep


logger = logging.getLogger("casanovo")


class ModelRunner:
    """A class to run Casanovo models.

    Parameters
    ----------
    config : Config object
        The casanovo configuration.
    model_filename : str, optional
        The model filename is required for eval and de novo modes,
        but not for training a model from scratch.
    """

    def __init__(
        self,
        config: Config,
        model_filename: Optional[str] = None,
    ) -> None:
        """Initialize a ModelRunner"""
        self.config = config
        self.model_filename = model_filename

        # Initialized later:
        self.tmp_dir = None
        self.trainer = None
        self.model = None
        self.loaders = None
        self.writer = None

        self.callbacks = []
        # Configure checkpoints.
        self.callbacks = [
            ModelCheckpoint(
                dirpath=config.model_save_folder_path,
                save_on_train_epoch_end=True,
            )
        ]

        if config.save_top_k is not None:
            self.callbacks.append(
                ModelCheckpoint(
                    dirpath=config.model_save_folder_path,
                    monitor="valid_CELoss",
                    mode="min",
                    save_top_k=config.save_top_k,
                    auto_insert_metric_name=True,
                    filename="{epoch}-{step}-{train_CELoss:.3f}-{valid_CELoss:.3f}",
                    save_last=True,
                )
            )

        # Configure early stopping
        if config.early_stopping_patience is not None:
            self.callbacks.append(
                EarlyStopping(
                    monitor="valid_CELoss",
                    min_delta=0.00,
                    patience=self.config.early_stopping_patience,
                    verbose=True,
                    check_finite=True,
                    mode="min",
                )
            )
        # Configure learning rate monitor
        if config.tb_summarywriter is not None:
            self.callbacks.append(
                LearningRateMonitor(logging_interval="step", log_momentum=True)
            )

    def __enter__(self):
        """Enter the context manager"""
        self.tmp_dir = tempfile.TemporaryDirectory()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Cleanup on exit"""
        self.tmp_dir.cleanup()
        self.tmp_dir = None
        if self.writer is not None:
            self.writer.save()

    def train(
        self,
        train_peak_path: Iterable[str],
        valid_peak_path: Iterable[str],
    ) -> None:
        """Train the Casanovo model.

        Parameters
        ----------
        train_peak_path : iterable of str
            The path to the MS data files for training.
        valid_peak_path : iterable of str
            The path to the MS data files for validation.

        Returns
        -------
        self
        """
        self.initialize_trainer(train=True)
        self.initialize_tokenizer()
        self.initialize_model(train=True)

        train_paths = self._get_input_paths(train_peak_path, True, "train")
        valid_paths = self._get_input_paths(valid_peak_path, True, "valid")
        self.initialize_data_module(train_paths, valid_paths)
        self.loaders.setup()
        # logger.info(f'TRAIN PSMs: {self.loaders.train_dataset.n_spectra}')
        # logger.info(f'VAL PSMs: {self.loaders.valid_dataset.n_spectra}')

        self.trainer.fit(
            self.model,
            self.loaders.train_dataloader(),
            self.loaders.val_dataloader(),
        )

    def evaluate(self, peak_path: Iterable[str]) -> None:
        """Evaluate peptide sequence preditions from a trained Casanovo model.

        Parameters
        ----------
        peak_path : iterable of str
            The path with MS data files for predicting peptide sequences.

        Returns
        -------
        self
        """
        self.initialize_trainer(train=False)
        self.initialize_tokenizer()
        self.initialize_model(train=False)

        test_paths = self._get_input_paths(peak_path, True, "test")
        self.initialize_data_module(test_paths=test_paths)
        self.loaders.setup(stage="test", annotated=True)

        self.trainer.validate(self.model, self.loaders.test_dataloader())

    def predict(self, peak_path: Iterable[str], output: str) -> None:
        """Predict peptide sequences with a trained Casanovo model.

        Parameters
        ----------
        peak_path : iterable of str
            The path with the MS data files for predicting peptide sequences.
        output : str
            Where should the output be saved?

        Returns
        -------
        self
        """
        self.writer = ms_io.MztabWriter(Path(output).with_suffix(".mztab"))
        self.writer.set_metadata(
            self.config,
            model=str(self.model_filename),
            config_filename=self.config.file,
        )

        self.initialize_trainer(train=False)
        self.initialize_tokenizer()
        self.initialize_model(train=False)
        self.model.out_writer = self.writer

        test_paths = self._get_input_paths(peak_path, False, "test")
        self.writer.set_ms_run(test_paths)
        self.initialize_data_module(test_paths=test_paths)
        self.loaders.setup(stage="test", annotated=False)
        self.trainer.predict(self.model, self.loaders.test_dataloader())

    def initialize_trainer(self, train: bool) -> None:
        """Initialize the lightning Trainer.

        Parameters
        ----------
        train : bool
            Determines whether to set the trainer up for model training
            or evaluation / inference.
        """
        trainer_cfg = dict(
            accelerator=self.config.accelerator,
            devices=1,
            enable_checkpointing=False,
            precision=self.config.precision,
            logger=False,
        )

        if train:
            if self.config.devices is None:
                devices = "auto"
            else:
                devices = self.config.devices

            if self.config.tb_summarywriter is not None:
                logger = TensorBoardLogger(
                    self.config.tb_summarywriter,
                    version=None,
                    name=f'model_{datetime.now().strftime("%Y%m%d_%H%M")}',
                    default_hp_metric=False,
                )
            else:
                logger = False

            additional_cfg = dict(
                devices=devices,
                callbacks=self.callbacks,
                enable_checkpointing=True,
                max_epochs=self.config.max_epochs,
                num_sanity_val_steps=self.config.num_sanity_val_steps,
                strategy=self._get_strategy(),
                val_check_interval=self.config.val_check_interval,
                check_val_every_n_epoch=None,
                logger=logger,
                accumulate_grad_batches=self.config.accumulate_grad_batches,
                gradient_clip_val=self.config.gradient_clip_val,
                gradient_clip_algorithm=self.config.gradient_clip_algorithm,
            )
            trainer_cfg.update(additional_cfg)

        self.trainer = pl.Trainer(**trainer_cfg)

    def initialize_model(self, train: bool) -> None:
        """Initialize the Casanovo model.

        Parameters
        ----------
        train : bool
            Determines whether to set the model up for model training or
            evaluation / inference.
        """
        try:
            tokenizer = self.tokenizer
        except AttributeError:
            raise RuntimeError("Please use `initialize_tokenizer()` first.")

        model_params = dict(
            dim_model=self.config.dim_model,
            n_head=self.config.n_head,
            dim_feedforward=self.config.dim_feedforward,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout,
            dim_intensity=self.config.dim_intensity,
            max_length=self.config.max_length,
            max_charge=self.config.max_charge,
            precursor_mass_tol=self.config.precursor_mass_tol,
            isotope_error_range=self.config.isotope_error_range,
            min_peptide_len=self.config.min_peptide_len,
            n_beams=self.config.n_beams,
            top_match=self.config.top_match,
            n_log=self.config.n_log,
            train_label_smoothing=self.config.train_label_smoothing,
            warmup_iters=self.config.warmup_iters,
            cosine_schedule_period_iters=self.config.cosine_schedule_period_iters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            out_writer=self.writer,
            calculate_precision=self.config.calculate_precision,
            tokenizer=tokenizer,
        )

        # Reconfigurable non-architecture related parameters for a loaded model.
        loaded_model_params = dict(
            max_length=self.config.max_length,
            precursor_mass_tol=self.config.precursor_mass_tol,
            isotope_error_range=self.config.isotope_error_range,
            n_beams=self.config.n_beams,
            min_peptide_len=self.config.min_peptide_len,
            top_match=self.config.top_match,
            n_log=self.config.n_log,
            train_label_smoothing=self.config.train_label_smoothing,
            warmup_iters=self.config.warmup_iters,
            cosine_schedule_period_iters=self.config.cosine_schedule_period_iters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            out_writer=self.writer,
            calculate_precision=self.config.calculate_precision,
        )

        if self.model_filename is None:
            # Train a model from scratch if no model file is provided.
            if train:
                self.model = Spec2Pep(**model_params)
                return
            # Else we're not training, so a model file must be provided.
            else:
                logger.error("A model file must be provided")
                raise ValueError("A model file must be provided")
        # Else a model file is provided (to continue training or for inference).

        if not Path(self.model_filename).exists():
            logger.error(
                "Could not find the model weights at file %s",
                self.model_filename,
            )
            raise FileNotFoundError("Could not find the model weights file")

        # First try loading model details from the weights file, otherwise use
        # the provided configuration.
        device = torch.empty(1).device  # Use the default device.
        try:
            self.model = Spec2Pep.load_from_checkpoint(
                self.model_filename, map_location=device, **loaded_model_params
            )

            architecture_params = set(model_params.keys()) - set(
                loaded_model_params.keys()
            )
            for param in architecture_params:
                if model_params[param] != self.model.hparams[param]:
                    warnings.warn(
                        f"Mismatching {param} parameter in "
                        f"model checkpoint ({self.model.hparams[param]}) "
                        f"vs config file ({model_params[param]}); "
                        "using the checkpoint."
                    )
        except RuntimeError:
            # This only doesn't work if the weights are from an older version
            try:
                self.model = Spec2Pep.load_from_checkpoint(
                    self.model_filename,
                    map_location=device,
                    **model_params,
                )
            except RuntimeError:
                raise RuntimeError(
                    "Weights file incompatible with the current version of "
                    "Casanovo."
                )

    def initialize_tokenizer(
        self,
    ) -> None:
        """Initialize the peptide tokenizer"""
        if self.config.mskb_tokenizer:
            tokenizer_cs = MskbPeptideTokenizer
        else:
            tokenizer_cs = PeptideTokenizer

        self.tokenizer = tokenizer_cs(
            residues=self.config.residues,
            replace_isoleucine_with_leucine=self.config.replace_isoleucine_with_leucine,
            reverse=self.config.reverse_peptides,
            start_token=None,
            stop_token="$",
        )

    def initialize_data_module(
        self,
        train_paths: Optional[str] = None,
        valid_paths: Optional[str] = None,
        test_paths: Optional[str] = None,
    ) -> None:
        """Initialize the data module

        Parameters
        ----------
        train_paths : str, optional
            A spectrum path for model training.
        valid_paths : str, optional
            A spectrum path for validation.
        test_paths : str, optional
            A spectrum path for evaluation or inference.
        """
        try:
            n_devices = self.trainer.num_devices
            train_bs = self.config.train_batch_size // n_devices
            eval_bs = self.config.predict_batch_size // n_devices
        except AttributeError:
            raise RuntimeError("Please use `initialize_trainer()` first.")

        try:
            tokenizer = self.tokenizer
        except AttributeError:
            raise RuntimeError("Please use `initialize_tokenizer()` first.")

        lance_dir = (
            Path(self.tmp_dir.name)
            if self.config.lance_dir is None
            else self.config.lance_dir
        )
        self.loaders = DeNovoDataModule(
            train_paths=train_paths,
            valid_paths=valid_paths,
            test_paths=test_paths,
            min_mz=self.config.min_mz,
            max_mz=self.config.max_mz,
            min_intensity=self.config.min_intensity,
            remove_precursor_tol=self.config.remove_precursor_tol,
            n_workers=self.config.n_workers,
            train_batch_size=train_bs,
            eval_batch_size=eval_bs,
            n_peaks=self.config.n_peaks,
            max_charge=self.config.max_charge,
            tokenizer=tokenizer,
            lance_dir=lance_dir,
            shuffle=self.config.shuffle,
            buffer_size=self.config.buffer_size,
        )

    def _get_input_paths(
        self,
        peak_path: Iterable[str],
        annotated: bool,
        mode: str,
    ) -> str:
        """Get the spectrum input paths.

        Parameters
        ----------
        peak_path : Iterable[str]
            The peak files/directories to check.
        annotated : bool
            Are the spectra expected to be annotated?
        mode : str
            Either train, valid or test to specify lance file name
        Returns
        -------
            The spectrum paths for training, evaluation, or inference.
        """
        ext = (".mgf", ".lance")
        if not annotated:
            ext += (".mzML", ".mzml", ".mzxml")  # FIXME: Check if these work

        filenames = _get_peak_filenames(peak_path, ext)
        if not filenames:
            not_found_err = f"Cound not find {mode} peak files"
            logger.error(not_found_err + " from %s", peak_path)
            raise FileNotFoundError(not_found_err)

        is_lance = any([Path(f).suffix in (".lance") for f in filenames])
        if is_lance:
            if len(filenames) > 1:
                lance_err = f"Multiple {mode} spectrum lance files specified"
                logger.error(lance_err)
                raise ValueError(lance_err)

        return filenames

    def _get_strategy(self) -> Union[str, DDPStrategy]:
        """Get the strategy for the Trainer.

        The DDP strategy works best when multiple GPUs are used. It can work
        for CPU-only, but definitely fails using MPS (the Apple Silicon chip)
        due to Gloo.

        Returns
        -------
        Union[str, DDPStrategy]
            The strategy parameter for the Trainer.

        """
        if self.config.accelerator in ("cpu", "mps"):
            return "auto"
        elif self.config.devices == 1:
            return "auto"
        elif torch.cuda.device_count() > 1:
            return DDPStrategy(find_unused_parameters=False, static_graph=True)
        else:
            return "auto"


def _get_peak_filenames(
    paths: Iterable[str], supported_ext: Iterable[str]
) -> List[str]:
    """
    Get all matching peak file names from the path pattern.

    Performs cross-platform path expansion akin to the Unix shell (glob, expand
    user, expand vars).

    Parameters
    ----------
    paths : Iterable[str]
        The path pattern(s).
    supported_ext : Iterable[str]
        Extensions of supported peak file formats.

    Returns
    -------
    List[str]
        The peak file names matching the path pattern.
    """
    found_files = set()
    for path in paths:
        path = os.path.expanduser(path)
        path = os.path.expandvars(path)
        for fname in glob.glob(path, recursive=True):
            if Path(fname).suffix.lower() in supported_ext:
                found_files.add(fname)

    return sorted(list(found_files))
