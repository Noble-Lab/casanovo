"""The command line entry point for Casanovo."""

import logging
import re
import shutil
import sys
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple, List

warnings.formatwarning = lambda message, category, *args, **kwargs: (
    f"{category.__name__}: {' '.join(str(message).split())}"
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

# The regex pattern below describes the the model weight
# naming pattern.
_CKPT_RE = re.compile(
    r"^casanovo_([a-z0-9][a-z0-9-]*)_v([0-9]+)-([0-9]+)-([0-9]+)\.ckpt$"
)
# The default model is orbitrap.
_DEFAULT_MODEL_ID = "orbitrap"

import appdirs
import github
import requests
import rich_click as click
from .shared_loading import (
    _SharedFileIOParams,
    _SharedParams,
    _is_valid_model,
    _setup_output,
    setup_model,
)

from . import __version__, utils
from .config import Config
from .denovo import ModelRunner

logger = logging.getLogger("casanovo")
# Handlers added by setup_logging, removed again on the next call.
_log_handlers: List[logging.Handler] = []
click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""
click.rich_click.SHOW_ARGUMENTS = True

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
    output_path, output_root_name = _setup_output(
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
            average_precision=runner.average_precision,
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
            average_precision=runner.average_precision,
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

    TRAIN_PEAK_PATH must be one or more annotated MGF files, such as
    those provided by MassIVE-KB, from which to train a new Casanovo
    model.
    """

    _is_valid_model(model, load_all_states)

    output_path, output_root_name = _setup_output(
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
    _setup_output(None, None, True, "info", "casanovo")
    utils.log_system_info(model="Casanovo", version=__version__)


@main.command(cls=_SharedFileIOParams)
def configure(
    output_dir: str, output_root: str, verbosity: str, force_overwrite: bool
) -> None:
    """
    Generate a Casanovo configuration file to customize.

    The Casanovo configuration file is in the YAML format.
    """
    output_path, _ = _setup_output(
        output_dir, output_root, force_overwrite, verbosity, "casanovo"
    )
    utils.log_system_info(model="Casanovo", version=__version__)
    config_fname = output_root if output_root is not None else "casanovo"
    config_fname = Path(config_fname).with_suffix(".yaml")
    if not force_overwrite:
        utils.check_dir_file_exists(output_path, str(config_fname))

    config_path = str(output_path / config_fname)
    Config.copy_default(config_path)
    logger.info(f"Wrote {config_path}")


def setup_logging(
    log_file_path: Path,
    verbosity: str,
) -> None:
    """
    Set up the logger.

    Logging occurs to the command-line and to the given log file.

    Parameters
    ----------
    log_file_path: Path
        The log file path.
    verbosity : str
        The logging level to use in the console.
    """
    logging_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    # Configure logging.
    # Reset warning capture first, because captureWarnings(True) does nothing
    # if it was already called, even if the warnings hook has since been
    # replaced (e.g. by pytest between tests).
    logging.captureWarnings(False)
    logging.captureWarnings(True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove the handlers from a previous call, so messages aren't duplicated
    # when logging is set up multiple times in the same process.
    for handler in _log_handlers:
        root_logger.removeHandler(handler)
        handler.close()
    _log_handlers.clear()

    # Formatters for file vs console:
    console_formatter = logging.Formatter("{levelname}: {message}", style="{")
    log_formatter = logging.Formatter(
        "{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : "
        "{message}",
        style="{",
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging_levels[verbosity.lower()])
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    file_handler = logging.FileHandler(log_file_path, encoding="utf8")
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    _log_handlers.extend([console_handler, file_handler])

    # Disable dependency non-critical log messages.
    logging.getLogger("depthcharge").setLevel(
        logging_levels[verbosity.lower()]
    )
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("github").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _normalize(s: str) -> str:
    """
    Normalizes model selector.

    Removes hyphens, underscores, and whitespace characters.

    Parameters
    ----------
    s: String
       The string to be normalized.

    Returns
    -------
    String
       The normalized model selector.
    """
    return re.sub(r"[-_\s]", "", s).lower()


def _parse_ckpt(filename: str) -> Optional[Tuple[str, Tuple[int, int, int]]]:
    """
    Parses the checkpoint filename.

    Extracts the major, minor model versions and model selector.

    Parameters
    ----------
    filename: String
        The name of the model file provided.

    Returns
    -------
    Optional[Tuple[str, Tuple[int, int, int]]]
        Method could return None if filename does not match _CKPT_RE
        Returns a Tuple with the String set to the model selector,
        and the integer tuple values set to the Casanovo compatibility
        version.
    """
    parsed_string = _CKPT_RE.match(os.path.basename(filename).lower())
    if not parsed_string:
        return None
    return parsed_string.group(1), (
        int(parsed_string.group(2)),
        int(parsed_string.group(3)),
        int(parsed_string.group(4)),
    )


def _resolve_selector(selector: str, candidates: List[str]) -> str:
    """
    Resolves the model selector to a present model.

    Resolve a model selector to a canonical model ID via:
      1. Exact normalized match
      2. Unique normalized prefix match
      3. Unique normalized substring match

    Parameters
    ----------
    selector: String
       The parsed selector from _parse_ckpt.
    candidates: List[str]
        The current models present in the cache.

    Returns
    -------
    String
        Best matching model to selector
    """
    norm = _normalize(selector)
    norm_map = {c: _normalize(c) for c in candidates}

    for matches in (
        [c for c, nc in norm_map.items() if nc == norm],
        [c for c, nc in norm_map.items() if nc.startswith(norm)],
        [c for c, nc in norm_map.items() if norm in nc],
    ):
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            options = ", ".join(sorted(matches))
            raise ValueError(
                f"Ambiguous model selector '{selector}'. Matching models:\n"
                + f"{options} \n Please specify one of these options."
            )

    available = "\n".join(f"  {c}" for c in sorted(candidates))
    raise ValueError(
        f"Unknown model selector '{selector}'.\nAvailable models:\n{available}"
    )


def _best_ckpt(
    checkpoints: List[Tuple[Path, Tuple[int, int, int]]],
    version: Tuple[int, int, int],
) -> Optional[Path]:
    """
    Pick best checkpoint by: exact patch > latest same minor > latest same major.

    Parameters
    ----------
    checkpoints: List[Tuple[Path, Tuple[int, int, int]]]
        Possible checkpoints.
    version: Tuple[int, int, int]
        Intended version.
    """
    maj, min_, pat = version
    for subset in (
        [x for x in checkpoints if x[1] == (maj, min_, pat)],
        [x for x in checkpoints if x[1][:2] == (maj, min_)],
        [x for x in checkpoints if x[1][0] == maj],
    ):
        if subset:
            return max(subset, key=lambda x: x[1])[0]
    return None


if __name__ == "__main__":
    main()
