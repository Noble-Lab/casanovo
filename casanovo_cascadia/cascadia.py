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
from lightning.pytorch import seed_everything

from . import __version__, utils
from .config import Config
from .denovo import ModelRunner
from .shared_loading import (
    _SharedFileIOParams,
    _SharedParams,
    _DEFAULT_MODEL_ID,
    _get_model_weights,
    _is_valid_url,
    _get_weights_from_url,
    _setup_output,
)

logger = logging.getLogger(__name__)

click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""
click.rich_click.SHOW_ARGUMENTS = True

from .version import _get_version

__version__ = _get_version("casanovo")

click.group(context_settings=dict(help_option_names=["-h", "--help"]))


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
    utils.log_system_info(model="Cascadia")  # set as as a bool?

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

    cache_dir = Path(appdirs.user_cache_dir("cascadia", False, opinion=False))
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

    logger.info("Cascadia version %s", str(__version__))
    logger.debug("model = %s", resolved_model)
    logger.debug("config = %s", config.file)
    logger.debug("output directory = %s", output_dir)
    logger.debug("output root name = %s", output_root_name)
    for key, value in config.items():
        logger.debug("%s = %s", str(key), str(value))

    return config, resolved_model


_CKPT_RE = re.compile(
    r"^cascadia_([a-z0-9][a-z0-9-]*)_v([0-9]+)-([0-9]+)-([0-9]+)\.ckpt$"
)


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
        output_dir, output_root, force_overwrite, verbosity, "cascadia"
    )
    config_fname = output_root if output_root is not None else "cascadia"
    config_fname = Path(config_fname).with_suffix(".yaml")
    if not force_overwrite:
        utils.check_dir_file_exists(output_path, str(config_fname))

    config_path = str(output_path / config_fname)
    Config.copy_default(config_path)
    logger.info(f"Wrote {config_path}")


@main.command()
def version() -> None:
    """Get the Casanovo version information."""
    _setup_output(None, None, True, "info", "cascadia")
    utils.log_system_info()
