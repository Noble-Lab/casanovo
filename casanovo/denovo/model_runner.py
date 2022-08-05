"""Training and testing functionality for the de novo peptide sequencing model."""
import logging
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from depthcharge.data import AnnotatedSpectrumIndex, SpectrumIndex

from casanovo.denovo.dataloaders import DeNovoDataModule
from casanovo.denovo.model import Spec2Pep


logger = logging.getLogger("casanovo")


def predict(
    peak_dir: str,
    model_filename: str,
    out_filename: str,
    config: Dict[str, Any],
) -> None:
    """
    Predict peptide sequences with a trained Casanovo model.

    Parameters
    ----------
    peak_dir : str
        The directory with peak files for predicting peptide sequences.
    model_filename : str
        The file name of the model weights (.ckpt file).
    out_filename : str
        The output file name for the prediction results (format: .csv).
    config : Dict[str, Any]
        The configuration options.
    """
    _execute_existing(peak_dir, model_filename, config, False, out_filename)


def evaluate(
    peak_dir: str, model_filename: str, config: Dict[str, Any]
) -> None:
    """
    Evaluate peptide sequence predictions from a trained Casanovo model.

    Parameters
    ----------
    peak_dir : str
        The directory with peak files for predicting peptide sequences.
    model_filename : str
        The file name of the model weights (.ckpt file).
    config : Dict[str, Any]
        The configuration options.
    """
    _execute_existing(peak_dir, model_filename, config, True)


def _execute_existing(
    peak_dir: str,
    model_filename: str,
    config: Dict[str, Any],
    annotated: bool,
    out_filename: Optional[str] = None,
) -> None:
    """
    Predict peptide sequences with a trained Casanovo model with/without evaluation.

    Parameters
    ----------
    peak_dir : str
        The directory with peak files for predicting peptide sequences.
    model_filename : str
        The file name of the model weights (.ckpt file).
    config : Dict[str, Any]
        The configuration options.
    annotated : bool
        Whether the input peak files are annotated (execute in evaluation mode) or not
        (execute in prediction mode only).
    out_filename : str
        The output file name for the prediction results (format: .csv).
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
        n_log=config["n_log"],
        out_filename=out_filename,
    )
    # Read the MS/MS spectra for which to predict peptide sequences.
    if not os.path.isdir(peak_dir):
        logger.error(
            "Could not find directory %s from which to read peak files",
            peak_dir,
        )
        raise FileNotFoundError(
            "Could not find the directory to read peak files"
        )
    peak_filenames = _get_peak_filenames(peak_dir)
    tmp_dir = tempfile.TemporaryDirectory()
    idx_filename = os.path.join(tmp_dir.name, f"{uuid.uuid4().hex}.hdf5")
    if annotated:
        index = AnnotatedSpectrumIndex(idx_filename, peak_filenames)
    else:
        index = SpectrumIndex(idx_filename, peak_filenames)
    # Initialize the data loader.
    loaders = DeNovoDataModule(
        test_index=index,
        n_peaks=config["n_peaks"],
        min_mz=config["min_mz"],
        max_mz=config["max_mz"],
        min_intensity=config["min_intensity"],
        remove_precursor_tol=config["remove_precursor_tol"],
        n_workers=config["num_workers"],
        batch_size=config["test_batch_size"],
    )
    loaders.setup(stage="test", annotated=annotated)
    # Create the Trainer object.
    trainer = pl.Trainer(
        accelerator="auto",
        auto_select_gpus=True,
        devices=-1,
        logger=config["logger"],
        max_epochs=config["max_epochs"],
        num_sanity_val_steps=config["num_sanity_val_steps"],
        strategy=config["strategy"],
    )
    # Run the model with/without validation.
    if annotated:
        trainer.validate(model, loaders.test_dataloader())
    else:
        trainer.predict(model, loaders.test_dataloader())
    # Clean up temporary files.
    tmp_dir.cleanup()


def train(
    peak_dir: str,
    peak_dir_val: str,
    model_filename: str,
    config: Dict[str, Any],
) -> None:
    """
    Train a Casanovo model.

    The model can be trained from scratch or by continuing training an existing model.

    Parameters
    ----------
    peak_dir : str
        The directory with peak files to be used as training data.
    peak_dir_val : str
        The directory with peak files to be used as validation data.
    model_filename : str
        The file name of the model weights (.ckpt file).
    config : Dict[str, Any]
        The configuration options.
    """
    # Read the MS/MS spectra to use for training and validation.
    if not os.path.isdir(peak_dir):
        logger.error(
            "Could not find directory %s from which to read peak files",
            peak_dir,
        )
        raise FileNotFoundError(
            "Could not find the directory to read peak files"
        )
    train_filenames = _get_peak_filenames(peak_dir)
    if not os.path.isdir(peak_dir_val):
        logger.error(
            "Could not find directory %s from which to read peak files",
            peak_dir_val,
        )
        raise FileNotFoundError(
            "Could not find the directory to read peak files"
        )
    val_filenames = _get_peak_filenames(peak_dir_val)
    tmp_dir = tempfile.TemporaryDirectory()
    train_idx_filename = os.path.join(tmp_dir.name, f"{uuid.uuid4().hex}.hdf5")
    val_idx_filename = os.path.join(tmp_dir.name, f"{uuid.uuid4().hex}.hdf5")
    train_index = AnnotatedSpectrumIndex(train_idx_filename, train_filenames)
    val_index = AnnotatedSpectrumIndex(val_idx_filename, val_filenames)
    # Initialize the data loaders.
    dataloader_params = dict(
        n_peaks=config["n_peaks"],
        min_mz=config["min_mz"],
        max_mz=config["max_mz"],
        min_intensity=config["min_intensity"],
        remove_precursor_tol=config["remove_precursor_tol"],
        num_workers=config["num_workers"],
        batch_size=config["train_batch_size"],
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
                "Could not find the model weights at file %s to continue training",
                model_filename,
            )
            raise FileNotFoundError(
                "Could not find the model weights to continue training"
            )
        model = Spec2Pep().load_from_checkpoint(model_filename, **model_params)
    # Create the Trainer object and (optionally) a checkpoint callback to periodically
    # save the model.
    if config["save_model"]:
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=config["model_save_folder_path"],
                save_weights_only=config["save_weights_only"],
                filename="{epoch}",
                every_n_epochs=config["every_n_epochs"],
                save_top_k=-1,
            )
        ]
    else:
        callbacks = None
    trainer = pl.Trainer(
        accelerator="auto",
        auto_select_gpus=True,
        callbacks=callbacks,
        devices=-1,
        logger=config["logger"],
        max_epochs=config["max_epochs"],
        num_sanity_val_steps=config["num_sanity_val_steps"],
        strategy=config["strategy"],
    )
    # Train the model.
    trainer.fit(
        model, train_loader.train_dataloader(), val_loader.val_dataloader()
    )
    # Clean up temporary files.
    tmp_dir.cleanup()


def _get_peak_filenames(peak_dir: str) -> List[str]:
    """
    Get the peak file names in the given directory.

    Parameters
    ----------
    peak_dir : str
        The directory in which to find peak files.

    Returns
    -------
    List[str]
        The peak file names from the given directory.
    """
    return [
        os.path.join(peak_dir, f)
        for f in os.listdir(peak_dir)
        if f.lower().endswith(".mgf")
    ]
