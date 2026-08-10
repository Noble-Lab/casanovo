"""The command line entry point for Casanovo."""

import datetime
import functools
import hashlib
import logging
import os
import re
import shutil
import sys
import time
import urllib.parse
import warnings
from pathlib import Path
from typing import Optional, Tuple

warnings.formatwarning = lambda message, category, *args, **kwargs: (
    f"{category.__name__}: {message}"
)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    ".*Consider increasing the value of the `num_workers` argument*",
)
warnings.filterwarnings(
    "ignore",
    ".*The PyTorch API of nested tensors is in prototype stage*",
)
warnings.filterwarnings(
    "ignore",
    ".*Converting mask without torch.bool dtype to bool*",
)

import appdirs
import github
import requests
import rich_click as click
import tqdm
from lightning.pytorch import seed_everything
from .shared_loading import (
    _SharedFileIOParams,
    _SharedParams,
    _DEFAULT_MODEL_ID,
    _get_model_weights,
    _is_valid_url,
    _get_weights_from_url,
    _setup_output,
)

from . import utils
from .config import Config
from .denovo import ModelRunner

logger = logging.getLogger(__name__)

click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""
click.rich_click.SHOW_ARGUMENTS = True

from .version import _get_version

__version__ = _get_version("casanovo")


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main() -> None:
    """
    Casanovo
    ========

    Casanovo is a state-of-the-art deep learning tool designed for de
    novo peptide sequencing. Powered by a transformer neural network,
    Casanovo "translates" peaks in MS/MS spectra into amino acid
    sequences.

    Links:

    - Documentation: https://casanovo.readthedocs.io
    - Official code repository: https://github.com/Noble-Lab/casanovo

    If you use Casanovo in your work, please cite:
    - Yilmaz, M., Fondrie, W. E., Bittremieux, W., Oh, S. & Noble, W. S.
    De novo mass spectrometry peptide sequencing with a transformer
    model. Proceedings of the 39th International Conference on Machine
    Learning - ICML '22 (2022).
    [https://proceedings.mlr.press/v162/yilmaz22a.html]().

    For more information on how to cite different versions of Casanovo,
    please see [https://casanovo.readthedocs.io/en/latest/cite.html]().

    """
    return


@main.command(cls=_SharedParams)
@click.argument(
    "peak_path",
    required=True,
    nargs=-1,
    type=click.Path(exists=True, dir_okay=True),
)
@click.option(
    "--evaluate",
    "-e",
    is_flag=True,
    default=False,
    help="""
    Run in evaluation mode. When this flag is set the peptide and amino acid  
    precision will be calculated and logged at the end of the sequencing run. 
    All input files must be annotated MGF files if running in evaluation 
    mode.
    """,
)
def sequence(
    peak_path: Tuple[str],
    model: Optional[str],
    config: Optional[str],
    output_dir: Optional[str],
    output_root: Optional[str],
    verbosity: str,
    force_overwrite: bool,
    evaluate: bool,
) -> None:
    """De novo sequence peptides from tandem mass spectra.

    PEAK_PATH must be one or more mzML, mzXML, or MGF files from which
    to sequence peptides. If evaluate is set to True PEAK_PATH must be
    one or more annotated MGF file.
    """
    output_path, output_root_name = _setup_output(
        output_dir, output_root, force_overwrite, verbosity, "casanovo"
    )

    start_time = time.time()
    utils.log_system_info(model="Casanovo")

    if not force_overwrite:
        utils.check_dir_file_exists(output_path, f"{output_root_name}.mztab")

    config, model = setup_model(
        model, config, output_path, output_root_name, False
    )

    with ModelRunner(
        config,
        model,
        output_path,
        output_root_name if output_root is not None else None,
        False,
    ) as runner:
        logger.info(
            "Sequencing %speptides from:",
            "and evaluating " if evaluate else "",
        )
        for peak_file in peak_path:
            logger.info("  %s", peak_file)

        results_path = output_path / f"{output_root_name}.mztab"
        runner.predict(peak_path, str(results_path), evaluate=evaluate)
        utils.log_annotate_report(
            runner.writer.psms, start_time=start_time, end_time=time.time()
        )


@main.command(cls=_SharedParams)
@click.argument(
    "peak_path",
    required=True,
    nargs=-1,
    type=click.Path(exists=True, dir_okay=True),
)
@click.argument(
    "fasta_path",
    required=True,
    nargs=1,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--export",
    is_flag=True,
    default=False,
    help="""
    Dumps peptides digested from data for debugging.
    Contains mass of peptide, sequence, and proteins 
    it is associated with
    """,
)
def db_search(
    peak_path: Tuple[str],
    fasta_path: str,
    model: Optional[str],
    config: Optional[str],
    output_dir: Optional[str],
    output_root: Optional[str],
    export: Optional[bool],
    verbosity: str,
    force_overwrite: bool,
) -> None:
    """Perform a database search on MS/MS data using Casanovo-DB.

    PEAK_PATH must be one or more mzML, mzXML, or MGF files.
    FASTA_PATH must be one FASTA file.
    """
    output_path, output_root_name = _setup_output(
        output_dir, output_root, force_overwrite, verbosity, "casanovo"
    )

    start_time = time.time()
    utils.log_system_info()

    if not force_overwrite:
        utils.check_dir_file_exists(output_path, f"{output_root_name}.mztab")

    config, model = setup_model(
        model, config, output_path, output_root_name, False
    )

    with ModelRunner(
        config,
        model,
        output_path,
        output_root_name if output_root is not None else None,
        False,
    ) as runner:
        logger.info("Performing database search on:")
        for peak_file in peak_path:
            logger.info("  %s", peak_file)

        logger.info("Using the following FASTA file:")
        logger.info("  %s", fasta_path)

        results_path = output_path / f"{output_root_name}.mztab"
        runner.db_search(peak_path, fasta_path, str(results_path))
        if export:
            if not force_overwrite:
                utils.check_dir_file_exists(
                    output_path, f"{output_root_name}.tsv"
                )
            runner.model.protein_database.export(output_path, output_root_name)
        utils.log_annotate_report(
            runner.writer.psms, start_time=start_time, end_time=time.time()
        )


@main.command(cls=_SharedParams)
@click.argument(
    "train_peak_path",
    required=True,
    nargs=-1,
    type=click.Path(exists=True, dir_okay=True),
)
@click.option(
    "-p",
    "--validation_peak_path",
    help="""
    An annotated MGF file for validation, like from MassIVE-KB. Use this
    option multiple times to specify multiple files. Loss from these files
    contributes to the aggregate valid_CELoss used for checkpoint selection.
    """,
    required=False,
    multiple=True,
    type=click.Path(exists=True, dir_okay=True),
)
@click.option(
    "-t",
    "--tracking_peak_path",
    help="""
    An annotated MGF file used to monitor validation loss during training
    without influencing checkpoint selection (useful for detecting
    catastrophic forgetting). Use this option multiple times to specify
    multiple files.
    """,
    required=False,
    multiple=True,
    type=click.Path(exists=True, dir_okay=True),
)
@click.option(
    "--load_all_states",
    help="""
    Flag to indicate whether all states are loaded when re-starting 
    training, or only the weights. Defaults to False.
    """,
    required=False,
    default=False,
    is_flag=True,
)
def train(
    train_peak_path: Tuple[str],
    validation_peak_path: Optional[Tuple[str]],
    tracking_peak_path: Optional[Tuple[str]],
    model: Optional[str],
    config: Optional[str],
    output_dir: Optional[str],
    output_root: Optional[str],
    verbosity: str,
    force_overwrite: bool,
    load_all_states: bool,
) -> None:
    """Train a Casanovo model on your own data.

    TRAIN_PEAK_PATH must be one or more annoated MGF files, such as
    those provided by MassIVE-KB, from which to train a new Casnovo
    model.
    """

    _is_valid_model(model, load_all_states)

    output_path, output_root_name = _setup_output(
        output_dir, output_root, force_overwrite, verbosity, "casanovo"
    )

    start_time = time.time()
    utils.log_system_info()

    config, model = setup_model(
        model, config, output_path, output_root_name, True
    )

    with ModelRunner(
        config,
        model,
        output_path,
        output_root_name if output_root is not None else None,
        not force_overwrite,
    ) as runner:
        logger.info("Training a model from:")
        for peak_file in train_peak_path:
            logger.info("  %s", peak_file)

        if len(validation_peak_path) == 0:
            validation_peak_path = train_peak_path

        logger.info("Using the following validation files:")
        for peak_file in validation_peak_path:
            logger.info("  %s", peak_file)

        if tracking_peak_path:
            logger.info("Using the following tracking-only validation files:")
            for peak_file in tracking_peak_path:
                logger.info("  %s", peak_file)

        runner.train(
            train_peak_path,
            validation_peak_path,
            model if load_all_states else None,
            tracking_peak_path,
        )

        utils.log_run_report(start_time=start_time, end_time=time.time())


@main.command()
def version() -> None:
    """Get the Casanovo version information."""
    _setup_output(None, None, True, "info", "casanovo")
    utils.log_system_info()


@main.command(cls=_SharedFileIOParams)
def configure(
    output_dir: str, output_root: str, verbosity: str, force_overwrite: bool
) -> None:
    """
    Generate a Casanovo configuration file to customize.

    The Casanovo configuration file is in the YAML format.
    """
    utils.log_system_info()
    output_path, _ = _setup_output(
        output_dir, output_root, force_overwrite, verbosity, "casanovo"
    )
    config_fname = output_root if output_root is not None else "casanovo"
    config_fname = Path(config_fname).with_suffix(".yaml")
    if not force_overwrite:
        utils.check_dir_file_exists(output_path, str(config_fname))

    config_path = str(output_path / config_fname)
    Config.copy_default(config_path)
    logger.info(f"Wrote {config_path}")


def _is_valid_model(model: Optional[str], load_all_states: bool) -> None:
    """
    Validate the model argument when --load_all_states is specified.

    Parameters
    ----------
    model : Optional[str]
        The model path or URL.
    load_all_states : bool
        Whether to load all model states for resuming training.

    Raises
    ------
    ValueError
        If load_all_states is True and model is a URL or non-existent file.
    UserWarning
        If load_all_states is True but model is not provided
    """
    if load_all_states:
        if model is None:
            logger.warning(
                "When --load_all_states is specified, --model must also be provided. "
                "Training will start from scratch without a provided model.",
                stacklevel=2,
            )
        elif _is_valid_url(model):
            raise ValueError(
                "Full model state cannot be loaded from a URL. "
                "Please provide a local file path when --load_all_states is True.",
            )
        elif not Path(model).is_file():
            raise ValueError(
                "When --load_all_states is True, the model path must point to an existing file.",
            )


def setup_model(
    model: str | None,
    config: str | None,
    output_dir: Path | str,
    output_root_name: str,
    is_train: bool,
) -> Tuple["Config", Optional[Path]]:
    """
    Set up Casanovo config and resolve model weights (.ckpt) path.

    Parameters
    ----------
    model : str | None
        May be a file system path, a URL pointing to a .ckpt file, or
        None. If `model` is a URL the weights will be downloaded and
        cached from `model`. If `model` is `None` the weights from the
        latest matching official release will be used (downloaded and
        cached).
    config : str | None
        Config file path. If None the default config will be used.
    output_dir: : Path | str
        The path to the output directory.
    output_root_name : str,
        The base name for the output files.
    is_train : bool
        Are we training? If not, we need to retrieve weights when the
        model is None.

    Return
    ------
    Tuple[Config, Path]
        Initialized Casanovo config, local path to model weights if any
        (may be `None` if training using random starting weights).
    """
    config = Config(config)
    seed_everything(seed=config["random_seed"], workers=True)

    cache_dir = Path(appdirs.user_cache_dir("casanovo", False, opinion=False))
    resolved_model: Optional[Path] = None

    version = tuple(
        int(x) if x else 0 for x in utils.split_version(__version__)
    )

    if model and Path(model).is_file():
        resolved_model = Path(model)
    elif model:
        if _is_valid_url(model):
            resolved_model = _get_weights_from_url(model, cache_dir)
        else:
            try:
                resolved_model = _get_model_weights(
                    model, cache_dir, version, _CKPT_RE
                )
            except github.RateLimitExceededException:
                logger.error(
                    "GitHub API rate limit exceeded while trying to download "
                    "the model weights. Please download compatible model "
                    "weights manually from the official Casanovo code website "
                    "(https://github.com/Noble-Lab/casanovo) and specify "
                    "these explicitly using the `--model` parameter when "
                    "running Casanovo."
                )
                raise PermissionError(
                    "GitHub API rate limit exceeded while trying to download "
                    "the model weights"
                ) from None
    elif not is_train:
        # Defaulting to default model
        logger.warning(
            "No model was specified. Using the default model '%s'. "
            "To make this choice explicit, use '--model %s'.",
            _DEFAULT_MODEL_ID,
            _DEFAULT_MODEL_ID,
        )
        model = _DEFAULT_MODEL_ID
        try:
            resolved_model = _get_model_weights(model, cache_dir, version)
        except github.RateLimitExceededException:
            logger.error(
                "GitHub API rate limit exceeded while trying to download "
                "the model weights. Please download compatible model "
                "weights manually from the official Casanovo code website "
                "(https://github.com/Noble-Lab/casanovo) and specify "
                "these explicitly using the `--model` parameter when "
                "running Casanovo."
            )
            raise PermissionError(
                "GitHub API rate limit exceeded while trying to download "
                "the model weights"
            ) from None

    logger.info("Casanovo version %s", str(__version__))
    logger.debug("model = %s", resolved_model)
    logger.debug("config = %s", config.file)
    logger.debug("output directory = %s", output_dir)
    logger.debug("output root name = %s", output_root_name)
    for key, value in config.items():
        logger.debug("%s = %s", str(key), str(value))

    return config, resolved_model


_CKPT_RE = re.compile(
    r"^casanovo_([a-z0-9][a-z0-9-]*)_v([0-9]+)-([0-9]+)-([0-9]+)\.ckpt$"
)

if __name__ == "__main__":
    main()
