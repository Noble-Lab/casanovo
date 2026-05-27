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
from typing import Optional, Tuple, List

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

from . import __version__, utils
from .config import Config
from .denovo import ModelRunner

logger = logging.getLogger("casanovo")
click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""
click.rich_click.SHOW_ARGUMENTS = True


class _SharedFileIOParams(click.RichCommand):
    """File IO options shared between most Casanovo commands"""

    def __init__(self, *args, **kwargs) -> None:
        """Define shared options."""
        super().__init__(*args, **kwargs)
        self.params += [
            click.Option(
                ("-d", "--output_dir"),
                help="The destination directory for output files.",
                type=click.Path(dir_okay=True),
            ),
            click.Option(
                ("-o", "--output_root"),
                help="The root name for all output files.",
                type=str,
            ),
            click.Option(
                ("-f", "--force_overwrite"),
                help="Whether to overwrite output files.",
                is_flag=True,
                show_default=True,
                default=False,
            ),
            click.Option(
                ("-v", "--verbosity"),
                help=(
                    "Set the verbosity of console logging messages."
                    " Log files are always set to 'debug'."
                ),
                type=click.Choice(
                    ["debug", "info", "warning", "error"],
                    case_sensitive=False,
                ),
                default="info",
            ),
        ]


class _SharedParams(_SharedFileIOParams):
    """Options shared between main Casanovo commands"""

    def __init__(self, *args, **kwargs) -> None:
        """Define shared options."""
        super().__init__(*args, **kwargs)
        self.params += [
            click.Option(
                ("-m", "--model"),
                help="""Either the model weights (.ckpt file) or a URL pointing to 
                the model weights file. If not provided, Casanovo will try to 
                download the latest release automatically.""",
            ),
            click.Option(
                ("-c", "--config"),
                help="The YAML configuration file overriding the default options.",
                type=click.Path(exists=True, dir_okay=False),
            ),
        ]


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
        output_dir, output_root, force_overwrite, verbosity
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
        output_dir, output_root, force_overwrite, verbosity
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
        output_dir, output_root, force_overwrite, verbosity
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


@main.command()
def version() -> None:
    """Get the Casanovo version information."""
    _setup_output(None, None, True, "info")
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
        output_dir, output_root, force_overwrite, verbosity
    )
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
    file_handler = logging.FileHandler(log_file_path, encoding="utf8")
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


_CKPT_RE = re.compile(
    r"^casanovo_([a-z0-9][a-z0-9-]*)_v([0-9]+)-([0-9]+)-([0-9]+)\.ckpt$"
)
_DEFAULT_MODEL_ID = "orbitrap"


def _normalize(s: str) -> str:
    return re.sub(r"[-_\s]", "", s).lower()


def _parse_ckpt(filename: str) -> Optional[Tuple[str, Tuple[int, int, int]]]:
    m = _CKPT_RE.match(os.path.basename(filename))
    if not m:
        return None
    return m.group(1), (int(m.group(2)), int(m.group(3)), int(m.group(4)))


def _resolve_selector(selector: str, candidates: List[str]) -> str:
    """
    Resolve a model selector to a canonical model ID via:
      1. Exact normalized match
      2. Unique normalized prefix match
      3. Unique normalized substring match
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
                + "\n".join(f"  {m}" for m in sorted(matches))
                + f"\nPlease specify one of: {options}."
            )

    available = "\n".join(f"  {c}" for c in sorted(candidates))
    raise ValueError(
        f"Unknown model selector '{selector}'.\nAvailable models:\n{available}"
    )


def _best_ckpt(
    checkpoints: List[Tuple[Path, Tuple[int, int, int]]],
    version: Tuple[int, int, int],
) -> Optional[Path]:
    """Pick best checkpoint by: exact patch > latest same minor > latest same major."""
    maj, min_, pat = version
    for subset in (
        [x for x in checkpoints if x[1] == (maj, min_, pat)],
        [x for x in checkpoints if x[1][:2] == (maj, min_)],
        [x for x in checkpoints if x[1][0] == maj],
    ):
        if subset:
            return max(subset, key=lambda x: x[1])[0]
    return None


def setup_model(
    model: str | None,
    config: str | None,
    output_dir: Path | str,
    output_root_name: str,
    is_train: bool,
) -> Tuple["Config", Optional[Path]]:
    config = Config(config)
    seed_everything(seed=config["random_seed"], workers=True)

    cache_dir = Path(appdirs.user_cache_dir("casanovo", False, opinion=False))
    resolved_model: Optional[Path] = None

    if model and Path(model).is_file():
        resolved_model = Path(model)
    elif not is_train:
        if not model:
            logger.warning(
                "No model was specified. Using the default model '%s'. "
                "To make this choice explicit, use '--model %s'.",
                _DEFAULT_MODEL_ID,
                _DEFAULT_MODEL_ID,
            )
            model = _DEFAULT_MODEL_ID

        if _is_valid_url(model):
            resolved_model = _get_weights_from_url(model, cache_dir)
        else:
            try:
                resolved_model = _get_model_weights(
                    model, cache_dir, utils.split_version(__version__)
                )
            except github.RateLimitExceededException:
                logger.error(
                    "GitHub API rate limit exceeded. Download model weights "
                    "manually from https://github.com/Noble-Lab/casanovo "
                    "and use '--model <path>'."
                )
                raise PermissionError(
                    "GitHub API rate limit exceeded"
                ) from None

    logger.info("Casanovo version %s", str(__version__))
    logger.debug("model = %s", resolved_model)
    logger.debug("config = %s", config.file)
    logger.debug("output directory = %s", output_dir)
    logger.debug("output root name = %s", output_root_name)
    for key, value in config.items():
        logger.debug("%s = %s", str(key), str(value))

    return config, resolved_model


def _get_model_weights(
    selector: str,
    cache_dir: Path,
    casanovo_version: Tuple[int, int, int],
) -> Path:
    """
    Resolve a model selector to a checkpoint, checking the local cache
    first and falling back to GitHub releases.
    """
    casanovo_version = tuple(int(x) for x in casanovo_version)
    os.makedirs(cache_dir, exist_ok=True)

    def scan_ckpts(filenames, base_dir=None):
        """Parse filenames into (model_id, path, version) triples."""
        results = []
        for fn in filenames:
            parsed = _parse_ckpt(fn)
            if parsed:
                model_id, version = parsed
                path = (base_dir / fn) if base_dir else Path(fn)
                results.append((model_id, path, version))
        return results

    local = (
        scan_ckpts(os.listdir(cache_dir), cache_dir)
        if cache_dir.exists()
        else []
    )
    if local:
        canonical_id = _resolve_selector(
            selector, list({mid for mid, _, _ in local})
        )
        family = [(p, v) for mid, p, v in local if mid == canonical_id]
        best = _best_ckpt(family, casanovo_version)
        if best:
            logger.info(
                "Model weights file %s retrieved from local cache", best
            )
            return best

    repo = github.Github().get_repo("Noble-Lab/casanovo")

    github_ckpts = []
    for release in repo.get_releases():
        for asset in release.get_assets():
            parsed = _parse_ckpt(asset.name)
            if parsed:
                model_id, version = parsed
                github_ckpts.append(
                    (model_id, asset.name, version, asset.browser_download_url)
                )

    if not github_ckpts:
        raise ValueError(
            "No canonical model checkpoints found on GitHub. "
            "Specify weights explicitly with '--model'."
        )

    canonical_id = _resolve_selector(
        selector, list({mid for mid, _, _, _ in github_ckpts})
    )
    family_gh = [
        (Path(fn), v, url)
        for mid, fn, v, url in github_ckpts
        if mid == canonical_id
    ]
    best = _best_ckpt([(p, v) for p, v, _ in family_gh], casanovo_version)

    if best is None:
        available = "\n".join(
            f"  {p.name}" for p, _, _ in sorted(family_gh, key=lambda x: x[1])
        )
        raise ValueError(
            f"No compatible '{canonical_id}' checkpoint found for Casanovo "
            f"{'.'.join(map(str, casanovo_version))}.\n"
            f"Available {canonical_id} checkpoints:\n{available}\n"
            "Please upgrade Casanovo, choose another model, or provide a "
            "local path or URL with '--model'."
        )

    url = next(url for p, _, url in family_gh if p == best)
    dest = cache_dir / best.name
    _download_weights(url, dest)
    return dest


def _setup_output(
    output_dir: str | None,
    output_root: str | None,
    overwrite: bool,
    verbosity: str,
) -> Tuple[Path, str]:
    """
    Set up the output directory, output file root name, and logging.

    Parameters:
    -----------
    output_dir : str | None
        The path to the output directory. If `None`, the output
        directory will be resolved to the current working directory.
    output_root : str | None
        The base name for the output files. If `None` the output root
        name will be resolved to casanovo_<current date and time>
    overwrite: bool
        Whether to overwrite log file if it already exists in the output
        directory.
    verbosity : str
        The verbosity level for logging.

    Returns:
    --------
    Tuple[Path, str]
        A tuple containing the resolved output directory and root name
        for output files.
    """
    if output_root is None:
        output_root = (
            f"casanovo_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        )

    if output_dir is None:
        output_path = Path.cwd()
    else:
        output_path = Path(output_dir).expanduser().resolve()
        if not output_path.is_dir():
            output_path.mkdir(parents=True)
            logger.warning(
                "Target output directory %s does not exists, so it will be "
                "created.",
                output_path,
            )

    if not overwrite:
        utils.check_dir_file_exists(output_path, f"{output_root}.log")

    log_file_path = output_path / f"{output_root}.log"
    setup_logging(log_file_path, verbosity)
    return output_path, output_root


def _get_weights_from_url(
    file_url: str,
    cache_dir: Path,
    force_download: Optional[bool] = False,
) -> Path:
    """
    Resolve weight file from URL

    Attempt to download weight file from URL if weights are not already
    cached - otherwise use cached weights. Downloaded weight files will
    be cached.

    Parameters
    ----------
    file_url : str
        URL pointing to model weights file.
    cache_dir : Path
        Model weights cache directory path.
    force_download : Optional[bool], default=False
        If True, forces a new download of the weight file even if it
        exists in the cache.

    Returns
    -------
    Path
        Path to the cached weights file.
    """
    if not _is_valid_url(file_url):
        raise ValueError("file_url must point to a valid URL")

    os.makedirs(cache_dir, exist_ok=True)
    cache_file_name = Path(urllib.parse.urlparse(file_url).path).name
    url_hash = hashlib.shake_256(file_url.encode("utf-8")).hexdigest(5)
    cache_file_dir = cache_dir / url_hash
    cache_file_path = cache_file_dir / cache_file_name

    if cache_file_path.is_file() and not force_download:
        cache_time = cache_file_path.stat()
        url_last_modified = 0

        try:
            file_response = requests.head(file_url)
            if file_response.ok:
                if "Last-Modified" in file_response.headers:
                    url_last_modified = datetime.datetime.strptime(
                        file_response.headers["Last-Modified"],
                        "%a, %d %b %Y %H:%M:%S %Z",
                    ).timestamp()
            else:
                logger.warning(
                    "Attempted HEAD request to %s yielded non-ok status code—"
                    "using cached file",
                    file_url,
                )
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.TooManyRedirects,
        ):
            logger.warning(
                "Failed to reach %s to get remote last modified time—using "
                "cached file",
                file_url,
            )

        if cache_time.st_mtime > url_last_modified:
            logger.info(
                "Model weights %s retrieved from local cache", file_url
            )
            return cache_file_path

    _download_weights(file_url, cache_file_path)
    return cache_file_path


def _download_weights(file_url: str, download_path: Path) -> None:
    """
    Download weights file from URL.

    Download the model weights file from the specified URL and save it
    to the given path. Ensures the download directory exists, and uses a
    progress
    bar to indicate download status.

    Parameters
    ----------
    file_url : str
        URL pointing to the model weights file.
    download_path : Path
        Path where the downloaded weights file will be saved.
    """
    download_file_dir = download_path.parent
    os.makedirs(download_file_dir, exist_ok=True)
    response = requests.get(file_url, stream=True, allow_redirects=True)
    response.raise_for_status()
    file_size = int(response.headers.get("Content-Length", 0))
    desc = "(Unknown total file size)" if file_size == 0 else ""
    response.raw.read = functools.partial(
        response.raw.read, decode_content=True
    )

    with (
        tqdm.tqdm.wrapattr(
            response.raw, "read", total=file_size, desc=desc
        ) as r_raw,
        open(download_path, "wb") as file,
    ):
        shutil.copyfileobj(r_raw, file)


def _is_valid_url(file_url: str) -> bool:
    """
    Determine whether file URL is a valid URL

    Parameters
    ----------
    file_url : str
        url to verify

    Return
    ------
    is_url : bool
        whether file_url is a valid url
    """
    try:
        result = urllib.parse.urlparse(file_url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


if __name__ == "__main__":
    main()
