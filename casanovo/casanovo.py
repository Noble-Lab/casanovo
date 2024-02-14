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
import depthcharge
import github
import lightning
import requests
import rich_click as click
import torch
import tqdm
from lightning.pytorch import seed_everything

from . import __version__
from . import utils
from .denovo import ModelRunner
from .config import Config

logger = logging.getLogger("casanovo")
click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""
click.rich_click.SHOW_ARGUMENTS = True


class _SharedParams(click.RichCommand):
    """Options shared between most Casanovo commands"""

    def __init__(self, *args, **kwargs) -> None:
        """Define shared options."""
        super().__init__(*args, **kwargs)
        self.params += [
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
                    ["debug", "info", "warning", "error"],
                    case_sensitive=False,
                ),
                default="info",
            ),
        ]


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main() -> None:
    """# Casanovo

    Casanovo de novo sequences peptides from tandem mass spectra using a
    Transformer model. Casanovo currently supports mzML, mzXML, and MGF files
    for de novo sequencing and annotated MGF files, such as those from
    MassIVE-KB, for training new models.

    Links:
    - Documentation: [https://casanovo.readthedocs.io]()
    - Official code repository: [https://github.com/Noble-Lab/casanovo]()

    If you use Casanovo in your work, please cite:
    - Yilmaz, M., Fondrie, W. E., Bittremieux, W., Oh, S. & Noble, W. S. De novo
    mass spectrometry peptide sequencing with a transformer model. Proceedings
    of the 39th International Conference on Machine Learning - ICML '22 (2022)
    doi:10.1101/2022.02.07.479481.

    """
    return


@main.command(cls=_SharedParams)
@click.argument(
    "peak_path",
    required=True,
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False),
)
def sequence(
    peak_path: Tuple[str],
    model: Optional[str],
    config: Optional[str],
    output: Optional[str],
    verbosity: str,
) -> None:
    """De novo sequence peptides from tandem mass spectra.

    PEAK_PATH must be one or more mzMl, mzXML, or MGF files from which
    to sequence peptides.
    """
    output = setup_logging(output, verbosity)
    config, model = setup_model(model, config, output, False)
    with ModelRunner(config, model) as runner:
        logger.info("Sequencing peptides from:")
        for peak_file in peak_path:
            logger.info("  %s", peak_file)

        runner.predict(peak_path, output)

    logger.info("DONE!")


@main.command(cls=_SharedParams)
@click.argument(
    "annotated_peak_path",
    required=True,
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False),
)
def evaluate(
    annotated_peak_path: Tuple[str],
    model: Optional[str],
    config: Optional[str],
    output: Optional[str],
    verbosity: str,
) -> None:
    """Evaluate de novo peptide sequencing performance.

    ANNOTATED_PEAK_PATH must be one or more annoated MGF files,
    such as those provided by MassIVE-KB.
    """
    output = setup_logging(output, verbosity)
    config, model = setup_model(model, config, output, False)
    with ModelRunner(config, model) as runner:
        logger.info("Sequencing and evaluating peptides from:")
        for peak_file in annotated_peak_path:
            logger.info("  %s", peak_file)

        runner.evaluate(annotated_peak_path)

    logger.info("DONE!")


@main.command(cls=_SharedParams)
@click.argument(
    "train_peak_path",
    required=True,
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "-p",
    "--validation_peak_path",
    help="""
    An annotated MGF file for validation, like from MassIVE-KB. Use this
    option multiple times to specify multiple files.
    """,
    required=True,
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
)
def train(
    train_peak_path: Tuple[str],
    validation_peak_path: Tuple[str],
    model: Optional[str],
    config: Optional[str],
    output: Optional[str],
    verbosity: str,
) -> None:
    """Train a Casanovo model on your own data.

    TRAIN_PEAK_PATH must be one or more annoated MGF files, such as those
    provided by MassIVE-KB, from which to train a new Casnovo model.
    """
    output = setup_logging(output, verbosity)
    config, model = setup_model(model, config, output, True)
    with ModelRunner(config, model) as runner:
        logger.info("Training a model from:")
        for peak_file in train_peak_path:
            logger.info("  %s", peak_file)

        logger.info("Using the following validation files:")
        for peak_file in validation_peak_path:
            logger.info("  %s", peak_file)

        runner.train(train_peak_path, validation_peak_path)

    logger.info("DONE!")


@main.command()
def version() -> None:
    """Get the Casanovo version information"""
    versions = [
        f"Casanovo: {__version__}",
        f"Depthcharge: {depthcharge.__version__}",
        f"Lightning: {lightning.__version__}",
        f"PyTorch: {torch.__version__}",
    ]
    sys.stdout.write("\n".join(versions) + "\n")


@main.command()
@click.option(
    "-o",
    "--output",
    help="The output configuration file.",
    default="casanovo.yaml",
    type=click.Path(dir_okay=False),
)
def configure(output: str) -> None:
    """Generate a Casanovo configuration file to customize.

    The casanovo configuration file is in the YAML format.
    """
    Config.copy_default(output)
    output = setup_logging(output, "info")
    logger.info(f"Wrote {output}\n")


def setup_logging(
    output: Optional[str],
    verbosity: str,
) -> Path:
    """Set up the logger.

    Logging occurs to the command-line and to the given log file.

    Parameters
    ----------
    output : Optional[str]
        The provided output file name.
    verbosity : str
        The logging level to use in the console.

    Return
    ------
    output : Path
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
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    warnings_logger = logging.getLogger("py.warnings")

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
    warnings_logger.addHandler(console_handler)
    file_handler = logging.FileHandler(output.with_suffix(".log"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    warnings_logger.addHandler(file_handler)

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

    return output


def setup_model(
    model: Optional[str],
    config: Optional[str],
    output: Optional[Path],
    is_train: bool,
) -> Config:
    """Setup Casanovo for most commands.

    Parameters
    ----------
    model : Optional[str]
        The provided model weights file.
    config : Optional[str]
        The provided configuration file.
    output : Optional[Path]
        The provided output file name.
    is_train : bool
        Are we training? If not, we need to retrieve weights when the model is
        None.

    Return
    ------
    config : Config
        The parsed configuration
    """
    # Read parameters from the config file.
    config = Config(config)
    seed_everything(seed=config["random_seed"], workers=True)

    # Download model weights if these were not specified (except when training).
    if model is None and not is_train:
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
    logger.debug("model = %s", model)
    logger.debug("config = %s", config.file)
    logger.debug("output = %s", output)
    for key, value in config.items():
        logger.debug("%s = %s", str(key), str(value))

    return config, model


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
