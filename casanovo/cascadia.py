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
from typing import Optional, Tuple, List

import appdirs
import github
import requests
import rich_click as click
import tqdm

from . import utils
from .config import Config
from .denovo import ModelRunner
from .shared_loading import (
    _SharedFileIOParams,
    _SharedParams,
    _setup_output,
    setup_model,
)
from .version import _get_version

logger = logging.getLogger(__name__)

click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""
click.rich_click.SHOW_ARGUMENTS = True

__version__ = "0.1.0"

_CKPT_CASCADIA = re.compile(
    r"^cascadia_([a-z0-9][a-z0-9-]*)_v([0-9]+)-([0-9]+)-([0-9]+)\.ckpt$"
)


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main() -> None:
    """
    Cascadia
    ========
    Cascadia is a state-of-the-art deep learning tool designed for de
    novo peptide sequencing with DIA data. Powered by a transformer
    neural network, Casanovo "translates" peaks in MS/MS spectra into
    amino acid sequences.

    Links:

    - Documentation: https://cascadia.readthedocs.io/en/latest/index.html
    - Official code repository: https://github.com/Noble-Lab/cascadia

    If you use Casanovo in your work, please cite:
    - Sanders, Justin, et al.
    ‘A Transformer Model for de Novo Sequencing of
    Data-Independent Acquisition Mass Spectrometry Data’. bioRxiv, Cold Spring
    Harbor Laboratory, 2024, [https://doi.org/10.1101/2024.06.03.597251]().
    """
    return


@main.command(cls=_SharedParams)
@click.argument(
    "peak_path",
    required=True,
    nargs=-1,
    type=click.Path(exists=True, dir_okay=True),
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
    output_path, output_root_name = _setup_output(
        output_dir, output_root, force_overwrite, verbosity, "cascadia"
    )

    start_time = time.time()
    utils.log_system_info(model="Cascadia", version=__version__)

    if not force_overwrite:
        utils.check_dir_file_exists(output_path, f"{output_root_name}.mztab")

    config, model = setup_model(
        model,
        config,
        output_path,
        output_root_name,
        False,
        _CKPT_CASCADIA,
        __version__,
    )

    with ModelRunner(
        config,
        model,
        output_path,
        output_root_name if output_root is not None else None,
        False,
        casanovo=False,
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


@main.command(cls=_SharedFileIOParams)
def configure(
    output_dir: str, output_root: str, verbosity: str, force_overwrite: bool
) -> None:
    """
    Generate a Cascadia configuration file to customize.

    The Cascadia configuration file is in the YAML format.
    """
    utils.log_system_info(model="Cascadia", version=__version__)
    output_path, _ = _setup_output(
        output_dir, output_root, force_overwrite, verbosity, "cascadia"
    )
    config_fname = output_root if output_root is not None else "casanovo"
    config_fname = Path(config_fname).with_suffix(".yaml")
    if not force_overwrite:
        utils.check_dir_file_exists(output_path, str(config_fname))

    config_path = str(output_path / config_fname)
    Config.copy_default(config_path)
    logger.info(f"Wrote {config_path}")


@main.command()
def version() -> None:
    """Get the Cascadia version information."""
    _setup_output(None, None, True, "info", "cascadia")
    utils.log_system_info(model="Cascadia", version=__version__)


if __name__ == "__main__":
    main()
