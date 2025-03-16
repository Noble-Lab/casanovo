"""Small utility functions"""

import logging
import os
import pathlib
import platform
import re
import socket
import sys
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import torch

from .data.psm import PepSpecMatch

SCORE_BINS = (0.0, 0.5, 0.9, 0.95, 0.99)

logger = logging.getLogger("casanovo")


def n_workers() -> int:
    """
    Get the number of workers to use for data loading.

    This is the maximum number of CPUs allowed for the process, scaled
    for the number of GPUs being used.

    On Windows and MacOS, we only use the main process. See:
    https://discuss.pytorch.org/t/errors-when-using-num-workers-0-in-dataloader/97564/4
    https://github.com/pytorch/pytorch/issues/70344

    Returns
    -------
    int
        The number of workers.
    """
    # FIXME: remove multiprocessing Linux deadlock issue workaround when
    # deadlock issue is resolved.
    return 0

    # Windows or MacOS: no multiprocessing.
    if platform.system() in ["Windows", "Darwin"]:
        logger.warning(
            "Dataloader multiprocessing is currently not supported on Windows "
            "or MacOS; using only a single thread."
        )
        return 0
    # Linux: scale the number of workers by the number of GPUs (if present).
    try:
        n_cpu = len(psutil.Process().cpu_affinity())
    except AttributeError:
        n_cpu = os.cpu_count()
    return (
        n_cpu // n_gpu if (n_gpu := torch.cuda.device_count()) > 1 else n_cpu
    )


def split_version(version: str) -> Tuple[str, str, str]:
    """
    Split the version into its semantic versioning components.

    Parameters
    ----------
    version : str
        The version number.

    Returns
    -------
    major : str
        The major release.
    minor : str
        The minor release.
    patch : str
        The patch release.
    """
    version_regex = re.compile(r"(\d+)\.(\d+)\.*(\d*)(?:.dev\d+.+)?")
    return tuple(g for g in version_regex.match(version).groups())


def get_score_bins(
    scores: pd.Series, score_bins: Iterable[float]
) -> Dict[float, int]:
    """
    Get binned confidence scores

    From a list of confidence scores, return a dictionary mapping each
    confidence score to the number of spectra with a confidence greater
    than or equal to it.

    Parameters
    ----------
    scores: pd.Series
        Series of assigned peptide scores.
    score_bins: Iterable[float]
        Confidence scores to map.

    Returns
    -------
    score_bin_dict: Dict[float, int]
        Dictionary mapping each confidence score to the number of
        spectra with a confidence greater than or equal to it.
    """
    return {score: (scores >= score).sum() for score in score_bins}


def get_peptide_lengths(sequences: pd.Series) -> np.ndarray:
    """
    Get a numpy array containing the length of each peptide sequence

    Parameters
    ----------
    sequences: pd.Series
        Series of peptide sequences.

    Returns
    -------
    sequence_lengths: np.ndarray
        Numpy array containing the length of each sequence, listed in
        the same order that the sequences are provided in.
    """
    # Mass modifications do not contribute to sequence length
    # FIXME: If PTMs are represented in ProForma notation this filtering
    #  operation needs to be reimplemented
    return sequences.str.replace(r"[^a-zA-Z]", "", regex=True).apply(len)


def get_report_dict(
    results_table: pd.DataFrame, score_bins: Iterable[float] = SCORE_BINS
) -> Optional[Dict]:
    """
    Generate sequencing run report

    Parameters
    ----------
    results_table: pd.DataFrame
        Parsed spectrum match table.
    score_bins: Iterable[float], Optional
        Confidence scores for creating confidence CMF, see
        `get_score_bins`.

    Returns
    -------
    report_gen: Dict
        Generated report represented as a dictionary, or None if no
        sequencing predictions were logged.
    """
    if results_table.empty:
        return None

    peptide_lengths = get_peptide_lengths(results_table["sequence"])
    min_length, med_length, max_length = np.quantile(
        peptide_lengths, [0, 0.5, 1]
    )
    return {
        "num_spectra": len(results_table),
        "score_bins": get_score_bins(results_table["score"], score_bins),
        "max_sequence_length": max_length,
        "min_sequence_length": min_length,
        "median_sequence_length": med_length,
    }


def log_run_report(
    start_time: Optional[float] = None, end_time: Optional[float] = None
) -> None:
    """
    Log general run report

    Parameters
    ----------
    start_time : Optional[float], default=None
        The start time of the sequencing run in seconds since the epoch.
    end_time : Optional[float], default=None
        The end time of the sequencing run in seconds since the epoch.
    """
    logger.info("======= End of Run Report =======")
    if start_time is not None and end_time is not None:
        start_datetime = datetime.fromtimestamp(start_time)
        end_datetime = datetime.fromtimestamp(end_time)
        delta_datetime = end_datetime - start_datetime
        logger.info(
            "Run Start Time: %s",
            start_datetime.strftime("%y/%m/%d %H:%M:%S"),
        )
        logger.info(
            "Run End Time: %s", end_datetime.strftime("%y/%m/%d %H:%M:%S")
        )
        logger.info("Time Elapsed: %s", delta_datetime)

    logger.info("Executed Command: %s", " ".join(sys.argv))
    logger.info("Executed on Host Machine: %s", socket.gethostname())

    if torch.cuda.is_available():
        gpu_util = torch.cuda.max_memory_allocated()
        logger.info("Max GPU Memory Utilization: %d MiB", gpu_util >> 20)


def log_annotate_report(
    predictions: List[PepSpecMatch],
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    score_bins: Iterable[float] = SCORE_BINS,
) -> None:
    """
    Log run annotation report.

    Parameters
    ----------
    predictions: List[PepSpecMatch]
        PSM predictions.
    start_time : Optional[float], default=None
        The start time of the sequencing run in seconds since the epoch.
    end_time : Optional[float], default=None
        The end time of the sequencing run in seconds since the epoch.
    score_bins: Iterable[float], Optional
        Confidence scores for creating confidence score distribution,
        see `get_score_bins`.
    """
    log_run_report(start_time=start_time, end_time=end_time)
    run_report = get_report_dict(
        pd.DataFrame(
            {
                "sequence": [psm.sequence for psm in predictions],
                "score": [psm.peptide_score for psm in predictions],
            }
        ),
        score_bins=score_bins,
    )

    if run_report is None:
        logger.warning(
            "No predictions were logged, this may be due to an error"
        )
    else:
        num_spectra = run_report["num_spectra"]
        logger.info("Sequenced %s spectra", num_spectra)
        logger.info("Score Distribution:")
        for score, pop in sorted(run_report["score_bins"].items()):
            logger.info(
                "%s spectra (%.2f%%) scored ≥ %.2f",
                pop,
                pop / num_spectra * 100,
                score,
            )

        logger.info(
            "Min Peptide Length: %d", run_report["min_sequence_length"]
        )
        logger.info(
            "Max Peptide Length: %d", run_report["max_sequence_length"]
        )
        logger.info(
            "Median Peptide Length: %d", run_report["median_sequence_length"]
        )


def check_dir_file_exists(
    dir: pathlib.Path, file_patterns: Iterable[str] | str
) -> None:
    """
    Check that no file names in dir match any of file_patterns

    Parameters
    ----------
    dir : pathlib.Path
        The directory to check for matching file names
    file_patterns : Iterable[str] | str
        UNIX style wildcard pattern(s) to test file names against

    Raises
    ------
    FileExistsError
        If matching file name is found in dir
    """
    if isinstance(file_patterns, str):
        file_patterns = [file_patterns]

    for pattern in file_patterns:
        if next(dir.glob(pattern), None) is not None:
            raise FileExistsError(
                f"File matching wildcard pattern {pattern} already exist in"
                f"{dir} and can not be overwritten."
            )
