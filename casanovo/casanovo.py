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
    _is_valid_model,
    setup_output,
    setup_model,
)

from . import utils
from .config import Config
from .denovo import ModelRunner

logger = logging.getLogger(__name__)

click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""
click.rich_click.SHOW_ARGUMENTS = True

from .version import _get_version
_MODEL_WEIGHT_REQUEST_TIMEOUT = 30


__version__ = _get_version("casanovo")

_CKPT_CASANOVO = re.compile(
    r"^casanovo_([a-z0-9][a-z0-9-]*)_v([0-9]+)-([0-9]+)-([0-9]+)\.ckpt$"
)


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
    output_path, output_root_name = setup_output(
        output_dir, output_root, force_overwrite, verbosity, "casanovo"
    )

    start_time = time.time()
    utils.log_system_info(model="Casanovo", version=__version__)

    if not force_overwrite:
        utils.check_dir_file_exists(output_path, f"{output_root_name}.mztab")

    config, model = setup_model(
        model,
        config,
        output_path,
        output_root_name,
        False,
        _CKPT_CASANOVO,
        __version__,
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
            runner.writer.psms,
            start_time=start_time,
            end_time=time.time(),
            n_missing_predictions=runner.model.n_missing_predictions,
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
    output_path, output_root_name = setup_output(
        output_dir, output_root, force_overwrite, verbosity, "casanovo"
    )

    start_time = time.time()
    utils.log_system_info(model="Casanovo", version=__version__)

    if not force_overwrite:
        utils.check_dir_file_exists(output_path, f"{output_root_name}.mztab")

    config, model = setup_model(
        model,
        config,
        output_path,
        output_root_name,
        False,
        _CKPT_CASANOVO,
        __version__,
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
            runner.writer.psms,
            start_time=start_time,
            end_time=time.time(),
            n_missing_predictions=runner.model.n_missing_predictions,
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

    output_path, output_root_name = setup_output(
        output_dir, output_root, force_overwrite, verbosity, "casanovo"
    )

    start_time = time.time()
    utils.log_system_info(model="Casanovo", version=__version__)

    config, model = setup_model(
        model,
        config,
        output_path,
        output_root_name,
        True,
        _CKPT_CASANOVO,
        __version__,
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
    setup_output(None, None, True, "info", "casanovo")
    utils.log_system_info(model="Casanovo", version=__version__)


@main.command(cls=_SharedFileIOParams)
def configure(
    output_dir: str, output_root: str, verbosity: str, force_overwrite: bool
) -> None:
    """
    Generate a Casanovo configuration file to customize.

    The Casanovo configuration file is in the YAML format.
    """
    utils.log_system_info(model="Casanovo", version=__version__)
    output_path, _ = setup_output(
        output_dir, output_root, force_overwrite, verbosity, "casanovo"
    )
    config_fname = output_root if output_root is not None else "casanovo"
    config_fname = Path(config_fname).with_suffix(".yaml")
    if not force_overwrite:
        utils.check_dir_file_exists(output_path, str(config_fname))

    config_path = str(output_path / config_fname)
    Config.copy_default(config_path)
    logger.info(f"Wrote {config_path}


if __name__ == "__main__":
    main()
