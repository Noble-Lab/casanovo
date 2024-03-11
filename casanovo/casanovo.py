"""The command line entry point for Casanovo."""
import datetime
import functools
import logging
import os
import re
import shutil
import sys
import warnings
from typing import Optional, Tuple

warnings.filterwarnings("ignore", category=DeprecationWarning)

import appdirs
import click
import github
import requests
import torch
import tqdm
import yaml
import pandas as pd
import pyteomics.mgf as mgf
from pytorch_lightning.lite import LightningLite

from . import __version__
from . import utils
from .data import ms_io
from .denovo import model_runner
from .config import Config

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
    "previously acquired spectrum\nannotations.\n"
    '- "db" will use Casanovo-DB to score\nspectra against a database of\n'
    "candidates specified in an .mgf\ncreated with annotate mode.\n"
    '- "annotate" will use tide-search results\nto annotate an .mgf '
    "file with candidate peptides.",
    type=click.Choice(["denovo", "train", "eval", "db", "annotate"]),
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
    "training Casanovo. If mode is 'db', this should be the path to the "
    "annotated .mgf file. If mode is 'annotate', this should be the path to the "
    ".mgf file you wish to annotate.",
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
    "(optionally) prediction results (extension: .mztab).",
    type=click.Path(dir_okay=False),
)
@click.option(
    "--tide_dir_path",
    help="The directory containing the results of a successful tide search. "
    "Used in annotate mode to annotate an .mgf file with candidate peptides.",
    type=click.Path(exists=True, file_okay=False),
)
def main(
    mode: str,
    model: Optional[str],
    peak_path: str,
    peak_path_val: Optional[str],
    config: Optional[str],
    output: Optional[str],
    tide_dir_path: Optional[str],
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
    else:
        basename, ext = os.path.splitext(os.path.abspath(output))
        output = basename if ext.lower() in (".log", ".mztab") else output

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

    LightningLite.seed_everything(seed=config["random_seed"], workers=True)

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
    logger.debug("tide_dir_path = %s", tide_dir_path)
    for key, value in config.items():
        logger.debug("%s = %s", str(key), str(value))

    # Run Casanovo in the specified mode.
    if mode == "denovo":
        logger.info("Predict peptide sequences with Casanovo.")
        writer = ms_io.MztabWriter(f"{output}.mztab")
        writer.set_metadata(config, model=model, config_filename=config.file)
        model_runner.predict(peak_path, model, config, writer)
        writer.save()
    elif mode == "eval":
        logger.info("Evaluate a trained Casanovo model.")
        model_runner.evaluate(peak_path, model, config)
    elif mode == "train":
        logger.info("Train the Casanovo model.")
        model_runner.train(peak_path, peak_path_val, model, config)
    elif mode == "db":
        logger.info("Database seach with casanovo.")
        writer = ms_io.MztabWriter(f"{output}.mztab")
        model_runner.db_search(peak_path, model, config, writer)
    elif mode == "annotate":
        logger.info("Annotate .mgf file with candidate peptides.")
        create_mgf_from_tide(tide_dir_path, peak_path, output)


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


def _normalize_mods(seq: str) -> str:
    """
    Turns tide-style modifications into the format used by Casanovo-DB.

        Parameters
        ----------
        seq : str
            The peptide sequence with tide-style modifications.

        Returns
        -------
        str
            The peptide sequence with Casanovo-DB-style modifications.
    """
    seq = seq.replace("M[15.99]", "M+15.995")
    seq = seq.replace("C", "C+57.021")
    seq = seq.replace("N[0.98]", "N+0.984")
    seq = seq.replace("Q[0.98]", "Q+0.984")
    seq = re.sub(r"(.*)\[42\.01\]", r"+42.011\1", seq)
    seq = re.sub(r"(.*)\[43\.01\]", r"+43.006\1", seq)
    seq = re.sub(r"(.*)\[\-17\.03\]", r"-17.027\1", seq)
    seq = re.sub(r"(.*)\[25\.98\]", r"+43.006-17.027\1", seq)
    return seq


def create_mgf_from_tide(
    tide_dir_path: str, mgf_file: str, output_file: str
) -> None:
    """
    Accepts a directory containing the results of a successful tide search, and an .mgf file containing MS/MS spectra.
    The .mgf file is then annotated in the SEQ field with all of the candidate peptides for each spectrum, as well as their target/decoy status.
    This annotated .mgf can be given directly to Casanovo-DB to perfrom a database search.

        Parameters
        ----------
        tide_dir_path : str
            Path to the directory containing the results of a successful tide search.
        mgf_file : str
            Path to the .mgf file containing MS/MS spectra.
        output_file : str
            Path to where the annotated .mgf will be written.

    """
    # Get paths to tide search text files
    tdf_path = os.path.join(tide_dir_path, "tide-search.target.txt")
    ddf_path = os.path.join(tide_dir_path, "tide-search.decoy.txt")
    try:
        target_df = pd.read_csv(
            tdf_path, sep="\t", usecols=["scan", "sequence", "target/decoy"]
        )
        decoy_df = pd.read_csv(
            ddf_path, sep="\t", usecols=["scan", "sequence", "target/decoy"]
        )
    except FileNotFoundError as e:
        logger.error(
            "Could not find tide search results in the specified directory. "
            "Please ensure that the directory contains the following files: "
            "tide-search.target.txt and tide-search.decoy.txt"
        )
        raise e

    logger.info(
        "Successfully read tide search results from %s.", tide_dir_path
    )

    df = pd.concat([target_df, decoy_df])
    scan_groups = df.groupby("scan")[["sequence", "target/decoy"]]

    scan_map = {}

    for scan, item in scan_groups:
        target_candidate_list = list(
            map(
                _normalize_mods,
                item.groupby("target/decoy")["sequence"].apply(list)["target"],
            )
        )
        decoy_candidate_list = list(
            map(
                _normalize_mods,
                item.groupby("target/decoy")["sequence"].apply(list)["decoy"],
            )
        )
        decoy_candidate_list = list(
            map(lambda x: "decoy_" + str(x), decoy_candidate_list)
        )
        scan_map[scan] = target_candidate_list + decoy_candidate_list

    all_spec = []
    for idx, spec_dict in enumerate(
        mgf.read(mgf_file)
    ):  #! WILL NEED TO BE CHANGED FOR OTHER ENCODINGS OF SCAN
        scan = int(
            re.search(r"scan=(\d+)", spec_dict["params"]["title"]).group(1)
        )
        try:
            spec_dict["params"]["seq"] = ",".join(list(scan_map[scan]))
            all_spec.append(spec_dict)
        except KeyError as e:
            # No need to do anything if the scan is not found in the scan map
            pass
    try:
        mgf.write(all_spec, output_file)
        logger.info("Annotated .mgf file written to %s.", output_file)
    except Exception as e:
        print(
            f"Write to {output_file} failed. Check if the file path is correct."
        )
        print(e)


if __name__ == "__main__":
    main()
