"""The command line entry point for Casanovo."""
import datetime
import functools
import logging
import os
import re
import shutil
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

warnings.filterwarnings("ignore", category=DeprecationWarning)

import appdirs
import click
import github
import requests
import torch
import tqdm
import yaml
from lightning.pytorch import seed_everything

from . import __version__
from . import utils
from .data import ms_io
from .denovo import ModelRunner
from .config import Config

logger = logging.getLogger("casanovo")


class _SharedParams(click.Command):
    """Options shared between most Casanovo commands"""

    def __init__(self, *args, **kwargs) -> None:
        """Define shared options."""
        self.params = [
            click.Option(
                ("-m", "--model"),
                help="""
                The model weights (.ckpt file). If not provided, Casanovo
                will try to download the latest release.
                """,
                type=click.Path(exists=True, dir_okay=False),
            ),
            click.Option(
                ("-o", "--output"),
                help="The mzTab file to which results will be written.",
                type=click.Path(dir_okay=False),
            ),
            click.Option(
                ("-c", "--config"),
                help="""
                The YAML configuration file overriding the default options.
                """,
                type=click.Path(exists=True, dir_okay=False),
            ),
            click.Option(
                ("-v", "--verbosity"),
                help="""
                Set the verbosity of console logging messages. Log files are
                always set to 'debug'.
                """,
                type=click.Choice(
                    ["error", "warning", "info", "debug"],
                    case_sensitive=False,
                ),
                default="info",
            ),
            click.Argument(
                ("peak_path",),
                help="""
                One or more mzML, mzXML, or MGF files from which to sequence
                peptides.
                """,
                required=True,
                nargs=-1,
            ),
        ]


@click.group()
def main() -> None:
    """
    \b
    Casanovo
    ========

    Casanovo de novo sequences peptide from tandem mass spectra using a
    Transformer model. Casanovo currently supports mzML, mzXML, and MGF files
    for de novo sequencing and annotated MGF files, such as those from MassIVE-KB,
    for training new models.

    Learn more: https://casanovo.readthedocs.io
    Official code website: https://github.com/Noble-Lab/casanovo

    If you use Casanovo in your work, please cite:
    Yilmaz, M., Fondrie, W. E., Bittremieux, W., Oh, S. & Noble, W. S. De novo
    mass spectrometry peptide sequencing with a transformer model. Proceedings
    of the 39th International Conference on Machine Learning - ICML '22 (2022)
    doi:10.1101/2022.02.07.479481.
    """
    return


@main.command(cls=_SharedParams)
def sequence(
    peak_path: str,
    model: Optional[str],
    config: Optional[str],
    output: Optional[str],
    verbosity: str,
) -> None:
    """De novo sequence peptides from tandem mass spectra."""


def setup(
    model: Optional[str],
    config: Optional[str],
    output: Optional[str],
    train: bool = False,
) -> None:
    """Setup Casanovo for most commands.

    Parameters
    ----------
    config : Optional[str]
        The provided configuration file.
    output : Optional[str]
        The provided output file name.
    train : bool
        Are we training? If not, we need to retreive weights when the model
        is None.

    Return
    ------
    Path
        The output file path.
    """
    if output is None:
        output = f"casanovo_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    output = Path(output).expanduser().resolve()

    logging_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    # Configure logging.
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter(
        "{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : "
        "{message}",
        style="{",
    )
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging_levels[verbosity.lower()])
    console_handler.setFormatter(log_formatter)
    root.addHandler(console_handler)
    file_handler = logging.FileHandler(f"{output}.log")
    file_handler.setFormatter(log_formatter)
    root.addHandler(file_handler)
    # Disable dependency non-critical log messages.
    logging.getLogger("depthcharge").setLevel(logging.INFO)
    logging.getLogger("github").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Read parameters from the config file.
    config = Config(config)

    seed_everything(seed=config["random_seed"], workers=True)

    # Download model weights if these were not specified (except when training).
    if model is None and mode != "train":
        try:
            model = _get_model_weights()
        except github.RateLimitExceededException:
            logger.error(
                "GitHub API rate limit exceeded while trying to download the "
                "model weights. Please download compatible model weights "
                "manually from the official Casanovo code website "
                "(https://github.com/Noble-Lab/casanovo) and specify these "
                "explicitly using the `--model` parameter when running "
                "Casanovo."
            )
            raise PermissionError(
                "GitHub API rate limit exceeded while trying to download the "
                "model weights"
            ) from None

    # Log the active configuration.
    logger.info("Casanovo version %s", str(__version__))
    logger.debug("mode = %s", mode)
    logger.debug("model = %s", model)
    logger.debug("peak_path = %s", peak_path)
    logger.debug("peak_path_val = %s", peak_path_val)
    logger.debug("config = %s", config.file)
    logger.debug("output = %s", output)
    for key, value in config.items():
        logger.debug("%s = %s", str(key), str(value))


@click.option(
    "--model",
    help="The file name of the model weights (.ckpt file).",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--peak_path",
    required=True,
    help="The file path with peak files for predicting peptide sequences or "
    "training Casanovo.",
    multiple=True,
)
@click.option(
    "--peak_path_val",
    help="The file path with peak files to be used as validation data during "
    "training.",
    multiple=True,
)
@click.option(
    "--config",
    help="The file name of the configuration file with custom options. If not "
    "specified, a default configuration will be used.",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--output",
    help="The base output file name to store logging (extension: .log) and "
    "(optionally) prediction results (extension: .mztab).",
    type=click.Path(dir_okay=False),
)
def main(
    mode: str,
    model: Optional[str],
    peak_path: str,
    peak_path_val: Optional[str],
    config: Optional[str],
    output: Optional[str],
):
    """
    \b
    Casanovo
    ========

    Casanovo de novo sequences peptide from tandem mass spectra using a
    Transformer model. Casanovo currently supports mzML, mzXML, and MGF files
    for de novo sequencing and annotated MGF files, such as those from MassIVE-KB,
    for training new models.

    Learn more: https://casanovo.readthedocs.io
    Official code website: https://github.com/Noble-Lab/casanovo

    If you use Casanovo in your work, please cite:
    Yilmaz, M., Fondrie, W. E., Bittremieux, W., Oh, S. & Noble, W. S. De novo
    mass spectrometry peptide sequencing with a transformer model. Proceedings
    of the 39th International Conference on Machine Learning - ICML '22 (2022)
    doi:10.1101/2022.02.07.479481.
    """
    # Run Casanovo in the specified mode.
    with ModelRunner(config, model) as model_runner:
        if mode == "denovo":
            logger.info("Predict peptide sequences with Casanovo.")
            model_runner.predict(peak_path, output)
        elif mode == "eval":
            logger.info("Evaluate a trained Casanovo model.")
            model_runner.evaluate(peak_path)
        elif mode == "train":
            logger.info("Train the Casanovo model.")
            model_runner.train(peak_path, peak_path_val)


def _get_model_weights() -> str:
    """
    Use cached model weights or download them from GitHub.

    If no weights file (extension: .ckpt) is available in the cache directory,
    it will be downloaded from a release asset on GitHub.
    Model weights are retrieved by matching release version. If no model weights
    for an identical release (major, minor, patch), alternative releases with
    matching (i) major and minor, or (ii) major versions will be used.
    If no matching release can be found, no model weights will be downloaded.

    Note that the GitHub API is limited to 60 requests from the same IP per
    hour.

    Returns
    -------
    str
        The name of the model weights file.
    """
    cache_dir = appdirs.user_cache_dir("casanovo", False, opinion=False)
    os.makedirs(cache_dir, exist_ok=True)
    version = utils.split_version(__version__)
    version_match: Tuple[Optional[str], Optional[str], int] = None, None, 0
    # Try to find suitable model weights in the local cache.
    for filename in os.listdir(cache_dir):
        root, ext = os.path.splitext(filename)
        if ext == ".ckpt":
            file_version = tuple(
                g for g in re.match(r".*_v(\d+)_(\d+)_(\d+)", root).groups()
            )
            match = (
                sum(m)
                if (m := [i == j for i, j in zip(version, file_version)])[0]
                else 0
            )
            if match > version_match[2]:
                version_match = os.path.join(cache_dir, filename), None, match
    # Provide the cached model weights if found.
    if version_match[2] > 0:
        logger.info(
            "Model weights file %s retrieved from local cache",
            version_match[0],
        )
        return version_match[0]
    # Otherwise try to find compatible model weights on GitHub.
    else:
        repo = github.Github().get_repo("Noble-Lab/casanovo")
        # Find the best matching release with model weights provided as asset.
        for release in repo.get_releases():
            rel_version = tuple(
                g
                for g in re.match(
                    r"v(\d+)\.(\d+)\.(\d+)", release.tag_name
                ).groups()
            )
            match = (
                sum(m)
                if (m := [i == j for i, j in zip(version, rel_version)])[0]
                else 0
            )
            if match > version_match[2]:
                for release_asset in release.get_assets():
                    fn, ext = os.path.splitext(release_asset.name)
                    if ext == ".ckpt":
                        version_match = (
                            os.path.join(
                                cache_dir,
                                f"{fn}_v{'_'.join(map(str, rel_version))}{ext}",
                            ),
                            release_asset.browser_download_url,
                            match,
                        )
                        break
        # Download the model weights if a matching release was found.
        if version_match[2] > 0:
            filename, url, _ = version_match
            logger.info(
                "Downloading model weights file %s from %s", filename, url
            )
            r = requests.get(url, stream=True, allow_redirects=True)
            r.raise_for_status()
            file_size = int(r.headers.get("Content-Length", 0))
            desc = "(Unknown total file size)" if file_size == 0 else ""
            r.raw.read = functools.partial(r.raw.read, decode_content=True)
            with tqdm.tqdm.wrapattr(
                r.raw, "read", total=file_size, desc=desc
            ) as r_raw, open(filename, "wb") as f:
                shutil.copyfileobj(r_raw, f)
            return filename
        else:
            logger.error(
                "No matching model weights for release v%s found, please "
                "specify your model weights explicitly using the `--model` "
                "parameter",
                __version__,
            )
            raise ValueError(
                f"No matching model weights for release v{__version__} found, "
                f"please specify your model weights explicitly using the "
                f"`--model` parameter"
            )


if __name__ == "__main__":
    main()
