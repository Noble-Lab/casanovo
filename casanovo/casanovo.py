"""The command line entry point for Casanovo."""
import collections
import datetime
import functools
import logging
import os
import re
import shutil
import sys

import appdirs
import click
import github
import psutil
import pytorch_lightning as pl
import requests
import torch
import tqdm
import yaml

from . import __version__
from casanovo.denovo import model_runner


logger = logging.getLogger("casanovo")


@click.command()
@click.option(
    "--mode",
    required=True,
    default="denovo",
    help="\b\nThe mode in which to run Casanovo:\n"
    '- "denovo" will predict peptide sequences for\nunknown MS/MS spectra.\n'
    '- "train" will train a model (from scratch or by\ncontinuing training a '
    "previously trained model).\n"
    '- "eval" will evaluate the performance of a\ntrained model using '
    "previously acquired spectrum\nannotations.",
    type=click.Choice(["denovo", "train", "eval"]),
)
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
)
@click.option(
    "--peak_path_val",
    help="The file path with peak files to be used as validation data during "
    "training.",
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
    "(optionally) prediction results (extension: .csv).",
    type=click.Path(dir_okay=False),
)
def main(
    mode: str,
    model: str,
    peak_path: str,
    peak_path_val: str,
    config: str,
    output: str,
):
    """
    \b
    Casanovo: De novo mass spectrometry peptide sequencing with a transformer model.
    ================================================================================

    Yilmaz, M., Fondrie, W. E., Bittremieux, W., Oh, S. & Noble, W. S. De novo
    mass spectrometry peptide sequencing with a transformer model. Proceedings
    of the 39th International Conference on Machine Learning - ICML '22 (2022)
    doi:10.1101/2022.02.07.479481.

    Official code website: https://github.com/Noble-Lab/casanovo
    """
    if output is None:
        output = os.path.join(
            os.getcwd(),
            f"casanovo_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        )

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
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(log_formatter)
    root.addHandler(console_handler)
    file_handler = logging.FileHandler(f"{os.path.splitext(output)[0]}.log")
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
    if config is None:
        config = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "config.yaml"
        )
    with open(config) as f_in:
        config = yaml.safe_load(f_in)
    # Ensure that the config values have the correct type.
    config_types = dict(
        random_seed=int,
        n_peaks=int,
        min_mz=float,
        max_mz=float,
        min_intensity=float,
        remove_precursor_tol=float,
        dim_model=int,
        n_head=int,
        dim_feedforward=int,
        n_layers=int,
        dropout=float,
        dim_intensity=int,
        max_length=int,
        max_charge=int,
        n_log=int,
        warmup_iters=int,
        max_iters=int,
        learning_rate=float,
        weight_decay=float,
        train_batch_size=int,
        predict_batch_size=int,
        max_epochs=int,
        num_sanity_val_steps=int,
        train_from_scratch=bool,
        save_model=bool,
        model_save_folder_path=str,
        save_weights_only=bool,
        every_n_train_steps=int,
    )
    for k, t in config_types.items():
        try:
            if config[k] is not None:
                config[k] = t(config[k])
        except (TypeError, ValueError) as e:
            logger.error("Incorrect type for configuration value %s: %s", k, e)
            raise TypeError(f"Incorrect type for configuration value {k}: {e}")
    config["residues"] = {
        str(aa): float(mass) for aa, mass in config["residues"].items()
    }
    # Add extra configuration options and scale by the number of GPUs.
    n_gpus = torch.cuda.device_count()
    config["n_workers"] = len(psutil.Process().cpu_affinity()) // n_gpus
    config["train_batch_size"] = config["train_batch_size"] // n_gpus

    pl.utilities.seed.seed_everything(seed=config["random_seed"], workers=True)

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
    for key, value in config.items():
        logger.debug("%s = %s", str(key), str(value))

    # Run Casanovo in the specified mode.
    if mode == "denovo":
        logger.info("Predict peptide sequences with Casanovo.")
        model_runner.predict(peak_path, model, output, config)
    elif mode == "eval":
        logger.info("Evaluate a trained Casanovo model.")
        model_runner.evaluate(peak_path, model, config)
    elif mode == "train":
        logger.info("Train the Casanovo model.")
        model_runner.train(peak_path, peak_path_val, model, config)


def _get_model_weights() -> str:
    """
    Download model weights from GitHub.

    Model weights files (extension: .ckpt) should be provided as an asset with
    their corresponding release on GitHub.
    Model weights will be retrieved by matching release version. If no model
    weights for an identical release (major, minor, patch), alternative releases
    with matching (i) major and minor, or (ii) major versions will be used.
    If no matching release can be found, no model weights will be downloaded.
    The model weights will be downloaded to the current working directory.

    Note that the GitHub API is limited to 60 requests from the same IP per
    hour. A log message provides instructions to explicitly specify the model
    file for subsequent uses.

    Returns
    -------
    str
        The name of the model weights file.
    """
    gh = github.Github()
    repo = gh.get_repo("Noble-Lab/casanovo")
    # Collect releases with model weights provided as asset.
    assets = collections.defaultdict(lambda: collections.defaultdict(dict))
    for release in repo.get_releases():
        major, minor, patch = tuple(
            g
            for g in re.match(
                r"v(\d+)\.(\d+)\.(\d+)", release.tag_name
            ).groups()
        )
        for release_asset in release.get_assets():
            if os.path.splitext(release_asset.name)[1] == ".ckpt":
                assets[major][minor][patch] = (
                    release.tag_name,
                    release_asset.name,
                    release_asset.browser_download_url,
                )
    # Find the release with best matching version number.
    major, minor, patch = tuple(
        g
        for g in re.match(
            r"(\d+)\.(\d+)\.(\d+)(?:.dev\d+.+)?", __version__
        ).groups()
    )
    if major in assets:
        if minor in assets[major]:
            if patch in assets[major][minor]:
                asset_version, asset_name, asset_url = assets[major][minor][
                    patch
                ]
                logger.debug(
                    "Model weights matching release %s found", asset_version
                )
            else:
                asset_version, asset_name, asset_url = next(
                    iter(assets[major][minor].values())
                )
                logger.debug(
                    "Model weights matching minor release %s found; current "
                    "release v%s",
                    asset_version,
                    __version__,
                )
        else:
            asset_version, asset_name, asset_url = next(
                iter(next(iter(assets[major].values())).values())
            )
            logger.debug(
                "Model weights matching major release %s found; current "
                "release v%s",
                asset_version,
                __version__,
            )
    else:
        logger.error(
            "No matching model weights for release v%s found, please specify "
            "your model weights explicitly using the `--model` parameter",
            __version__,
        )
        raise ValueError(
            f"No matching model weights for release v{__version__} found, "
            "please specify your model weights explicitly using the `--model` "
            "parameter"
        )
    # Check whether the model weights file already exists.
    asset_name, ext = os.path.splitext(asset_name)
    asset_name = f"{asset_name}_{asset_version.replace('.', '_')}_{ext}"
    asset_path = os.path.join(
        appdirs.user_cache_dir("casanovo", False, opinion=False), asset_name
    )
    if os.path.isfile(asset_path):
        logger.warning(
            "Model weights file found at %s. Please specify this file "
            "explicitly using the `--model` parameter when running Casanovo "
            "next time.",
            asset_path,
        )
    else:
        # Download the model weights.
        logger.info(
            "Downloading model weights for release %s from URL %s",
            asset_version,
            asset_url,
        )
        response = requests.get(asset_url, stream=True, allow_redirects=True)
        response.raise_for_status()
        file_size = int(response.headers.get("Content-Length", 0))
        desc = "(Unknown total file size)" if file_size == 0 else ""
        response.raw.read = functools.partial(
            response.raw.read, decode_content=True
        )
        with tqdm.tqdm.wrapattr(
            response.raw, "read", total=file_size, desc=desc
        ) as r_raw, open(asset_path, "wb") as f:
            shutil.copyfileobj(r_raw, f)
        logger.warning(
            "Model weights file downloaded to %s. Please specify this file "
            "explicitly using the `--model` parameter when running Casanovo "
            "next time.",
            asset_path,
        )
    return asset_path


if __name__ == "__main__":
    main()
