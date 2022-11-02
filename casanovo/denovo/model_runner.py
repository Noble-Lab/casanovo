"""Training and testing functionality for the de novo peptide sequencing
model."""
import glob
import logging
import operator
import os
import tempfile
import uuid
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from depthcharge.data import AnnotatedSpectrumIndex, SpectrumIndex
from pytorch_lightning.strategies import DDPStrategy

from .. import utils
from ..data import ms_io
from ..denovo.dataloaders import DeNovoDataModule
from ..denovo.model import Spec2Pep


logger = logging.getLogger("casanovo")


def predict(
    peak_path: str,
    model_filename: str,
    config: Dict[str, Any],
    out_writer: ms_io.MztabWriter,
) -> None:
    """
    Predict peptide sequences with a trained Casanovo model.

    Parameters
    ----------
    peak_path : str
        The path with peak files for predicting peptide sequences.
    model_filename : str
        The file name of the model weights (.ckpt file).
    config : Dict[str, Any]
        The configuration options.
    out_writer : ms_io.MztabWriter
        The mzTab writer to export the prediction results.
    """
    _execute_existing(peak_path, model_filename, config, False, out_writer)


def evaluate(
    peak_path: str, model_filename: str, config: Dict[str, Any]
) -> None:
    """
    Evaluate peptide sequence predictions from a trained Casanovo model.

    Parameters
    ----------
    peak_path : str
        The path with peak files for predicting peptide sequences.
    model_filename : str
        The file name of the model weights (.ckpt file).
    config : Dict[str, Any]
        The configuration options.
    """
    _execute_existing(peak_path, model_filename, config, True)


def _execute_existing(
    peak_path: str,
    model_filename: str,
    config: Dict[str, Any],
    annotated: bool,
    out_writer: Optional[ms_io.MztabWriter] = None,
) -> None:
    """
    Predict peptide sequences with a trained Casanovo model with/without
    evaluation.

    Parameters
    ----------
    peak_path : str
        The path with peak files for predicting peptide sequences.
    model_filename : str
        The file name of the model weights (.ckpt file).
    config : Dict[str, Any]
        The configuration options.
    annotated : bool
        Whether the input peak files are annotated (execute in evaluation mode)
        or not (execute in prediction mode only).
    out_writer : Optional[ms_io.MztabWriter]
        The mzTab writer to export the prediction results.
    """
    # Load the trained model.
    if not os.path.isfile(model_filename):
        logger.error(
            "Could not find the trained model weights at file %s",
            model_filename,
        )
        raise FileNotFoundError("Could not find the trained model weights")
    model = Spec2Pep().load_from_checkpoint(
        model_filename,
        dim_model=config["dim_model"],
        n_head=config["n_head"],
        dim_feedforward=config["dim_feedforward"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        dim_intensity=config["dim_intensity"],
        custom_encoder=config["custom_encoder"],
        max_length=config["max_length"],
        residues=config["residues"],
        max_charge=config["max_charge"],
        precursor_mass_tol=config["precursor_mass_tol"],
        isotope_error_range=config["isotope_error_range"],
        n_log=config["n_log"],
        out_writer=out_writer,
    )
    # Read the MS/MS spectra for which to predict peptide sequences.
    if annotated:
        peak_ext = (".mgf", ".h5", ".hdf5")
    else:
        peak_ext = (".mgf", ".mzml", ".mzxml", ".h5", ".hdf5")
    if len(peak_filenames := _get_peak_filenames(peak_path, peak_ext)) == 0:
        logger.error("Could not find peak files from %s", peak_path)
        raise FileNotFoundError("Could not find peak files")
    peak_is_index = any(
        [os.path.splitext(fn)[1] in (".h5", ".hdf5") for fn in peak_filenames]
    )
    if peak_is_index and len(peak_filenames) > 1:
        logger.error("Multiple HDF5 spectrum indexes specified")
        raise ValueError("Multiple HDF5 spectrum indexes specified")
    tmp_dir = tempfile.TemporaryDirectory()
    if peak_is_index:
        idx_filename, peak_filenames = peak_filenames[0], None
    else:
        idx_filename = os.path.join(tmp_dir.name, f"{uuid.uuid4().hex}.hdf5")
    SpectrumIdx = AnnotatedSpectrumIndex if annotated else SpectrumIndex
    valid_charge = np.arange(1, config["max_charge"] + 1)
    index = SpectrumIdx(
        idx_filename, peak_filenames, valid_charge=valid_charge
    )
    # Initialize the data loader.
    loaders = DeNovoDataModule(
        test_index=index,
        n_peaks=config["n_peaks"],
        min_mz=config["min_mz"],
        max_mz=config["max_mz"],
        min_intensity=config["min_intensity"],
        remove_precursor_tol=config["remove_precursor_tol"],
        n_workers=config["n_workers"],
        batch_size=config["predict_batch_size"],
    )
    loaders.setup(stage="test", annotated=annotated)

    # Create the Trainer object.
    trainer = pl.Trainer(
        accelerator="auto",
        auto_select_gpus=True,
        devices=_get_devices(),
        logger=config["logger"],
        max_epochs=config["max_epochs"],
        num_sanity_val_steps=config["num_sanity_val_steps"],
        strategy=_get_strategy(),
    )
    # Run the model with/without validation.
    run_trainer = trainer.validate if annotated else trainer.predict
    run_trainer(model, loaders.test_dataloader())
    # Clean up temporary files.
    tmp_dir.cleanup()


def train(
    peak_path: str,
    peak_path_val: str,
    model_filename: str,
    config: Dict[str, Any],
) -> None:
    """
    Train a Casanovo model.

    The model can be trained from scratch or by continuing training an existing
    model.

    Parameters
    ----------
    peak_path : str
        The path with peak files to be used as training data.
    peak_path_val : str
        The path with peak files to be used as validation data.
    model_filename : str
        The file name of the model weights (.ckpt file).
    config : Dict[str, Any]
        The configuration options.
    """
    # Read the MS/MS spectra to use for training and validation.
    ext = (".mgf", ".h5", ".hdf5")
    if len(train_filenames := _get_peak_filenames(peak_path, ext)) == 0:
        logger.error("Could not find training peak files from %s", peak_path)
        raise FileNotFoundError("Could not find training peak files")
    train_is_index = any(
        [os.path.splitext(fn)[1] in (".h5", ".hdf5") for fn in train_filenames]
    )
    if train_is_index and len(train_filenames) > 1:
        logger.error("Multiple training HDF5 spectrum indexes specified")
        raise ValueError("Multiple training HDF5 spectrum indexes specified")
    if (
        peak_path_val is None
        or len(val_filenames := _get_peak_filenames(peak_path_val, ext)) == 0
    ):
        logger.error(
            "Could not find validation peak files from %s", peak_path_val
        )
        raise FileNotFoundError("Could not find validation peak files")
    val_is_index = any(
        [os.path.splitext(fn)[1] in (".h5", ".hdf5") for fn in val_filenames]
    )
    if val_is_index and len(val_filenames) > 1:
        logger.error("Multiple validation HDF5 spectrum indexes specified")
        raise ValueError("Multiple validation HDF5 spectrum indexes specified")
    tmp_dir = tempfile.TemporaryDirectory()
    if train_is_index:
        train_idx_fn, train_filenames = train_filenames[0], None
    else:
        train_idx_fn = os.path.join(tmp_dir.name, f"{uuid.uuid4().hex}.hdf5")
    valid_charge = np.arange(1, config["max_charge"] + 1)
    train_index = AnnotatedSpectrumIndex(
        train_idx_fn, train_filenames, valid_charge=valid_charge
    )
    if val_is_index:
        val_idx_fn, val_filenames = val_filenames[0], None
    else:
        val_idx_fn = os.path.join(tmp_dir.name, f"{uuid.uuid4().hex}.hdf5")
    val_index = AnnotatedSpectrumIndex(
        val_idx_fn, val_filenames, valid_charge=valid_charge
    )
    # Initialize the data loaders.
    dataloader_params = dict(
        batch_size=config["train_batch_size"],
        n_peaks=config["n_peaks"],
        min_mz=config["min_mz"],
        max_mz=config["max_mz"],
        min_intensity=config["min_intensity"],
        remove_precursor_tol=config["remove_precursor_tol"],
        n_workers=config["n_workers"],
    )
    train_loader = DeNovoDataModule(
        train_index=train_index, **dataloader_params
    )
    train_loader.setup()
    val_loader = DeNovoDataModule(valid_index=val_index, **dataloader_params)
    val_loader.setup()
    # Initialize the model.
    model_params = dict(
        dim_model=config["dim_model"],
        n_head=config["n_head"],
        dim_feedforward=config["dim_feedforward"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        dim_intensity=config["dim_intensity"],
        custom_encoder=config["custom_encoder"],
        max_length=config["max_length"],
        residues=config["residues"],
        max_charge=config["max_charge"],
        precursor_mass_tol=config["precursor_mass_tol"],
        isotope_error_range=config["isotope_error_range"],
        n_log=config["n_log"],
        tb_summarywriter=config["tb_summarywriter"],
        warmup_iters=config["warmup_iters"],
        max_iters=config["max_iters"],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    if config["train_from_scratch"]:
        model = Spec2Pep(**model_params)
    else:
        if not os.path.isfile(model_filename):
            logger.error(
                "Could not find the model weights at file %s to continue "
                "training",
                model_filename,
            )
            raise FileNotFoundError(
                "Could not find the model weights to continue training"
            )
        model = Spec2Pep().load_from_checkpoint(model_filename, **model_params)
    # Create the Trainer object and (optionally) a checkpoint callback to
    # periodically save the model.
    if config["save_model"]:
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=config["model_save_folder_path"],
                save_top_k=-1,
                save_weights_only=config["save_weights_only"],
                every_n_train_steps=config["every_n_train_steps"],
            )
        ]
    else:
        callbacks = None

    trainer = pl.Trainer(
        accelerator="auto",
        auto_select_gpus=True,
        callbacks=callbacks,
        devices=_get_devices(),
        logger=config["logger"],
        max_epochs=config["max_epochs"],
        num_sanity_val_steps=config["num_sanity_val_steps"],
        strategy=_get_strategy(),
    )
    # Train the model.
    trainer.fit(
        model, train_loader.train_dataloader(), val_loader.val_dataloader()
    )
    # Clean up temporary files.
    tmp_dir.cleanup()


def _get_peak_filenames(
    path: str, supported_ext: Iterable[str] = (".mgf",)
) -> List[str]:
    """
    Get all matching peak file names from the path pattern.

    Performs cross-platform path expansion akin to the Unix shell (glob, expand
    user, expand vars).

    Parameters
    ----------
    path : str
        The path pattern.
    supported_ext : Iterable[str]
        Extensions of supported peak file formats. Default: MGF.

    Returns
    -------
    List[str]
        The peak file names matching the path pattern.
    """
    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    return [
        fn
        for fn in glob.glob(path, recursive=True)
        if os.path.splitext(fn.lower())[1] in supported_ext
    ]


def _get_strategy() -> Optional[DDPStrategy]:
    """
    Get the strategy for the Trainer.

    The DDP strategy works best when multiple GPUs are used. It can work for
    CPU-only, but definitely fails using MPS (the Apple Silicon chip) due to
    Gloo.

    Returns
    -------
    Optional[DDPStrategy]
        The strategy parameter for the Trainer.
    """
    if torch.cuda.device_count() > 1:
        return DDPStrategy(find_unused_parameters=False, static_graph=True)

    return None


def _get_devices() -> Union[int, str]:
    """
    Get the number of GPUs/CPUs for the Trainer to use.

    Returns
    -------
    Union[int, str]
        The number of GPUs/CPUs to use, or "auto" to let PyTorch Lightning
        determine the appropriate number of devices.
    """
    if any(
        operator.attrgetter(device + ".is_available")(torch)()
        for device in ["cuda", "backends.mps"]
    ):
        return -1
    elif not (n_workers := utils.n_workers()):
        return "auto"
    else:
        return n_workers
