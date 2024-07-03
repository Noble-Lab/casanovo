"""Small utility functions"""

import logging
import os
import platform
import re
import socket
import sys
import time
from datetime import datetime
from typing import Tuple, Dict, List, Optional

import numpy as np
import psutil
import torch
from pandas import DataFrame


SCORE_BINS = [0.0, 0.5, 0.9, 0.95, 0.99]

logger = logging.getLogger("casanovo")


def n_workers() -> int:
    """
    Get the number of workers to use for data loading.

    This is the maximum number of CPUs allowed for the process, scaled for the
    number of GPUs being used.

    On Windows and MacOS, we only use the main process. See:
    https://discuss.pytorch.org/t/errors-when-using-num-workers-0-in-dataloader/97564/4
    https://github.com/pytorch/pytorch/issues/70344

    Returns
    -------
    int
        The number of workers.
    """
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
    results_table: DataFrame, score_bins: List[float]
) -> Dict[float, int]:
    """
    Get binned confidence scores

    From a list of confidence scores, return a dictionary mapping each
    confidence score to the number of spectra with a confidence greater
    than or equal to it.

    Parameters
    ----------
    results_table: DataFrame
        Parsed spectrum match table
    score_bins: List[float]
        Confidence scores to map

    Returns
    -------
    score_bin_dict: Dict[float, int]
        Dictionary mapping each confidence score to the number of spectra
        with a confidence greater than or equal to it.
    """
    return {
        score: (results_table["score"] >= score).sum() for score in score_bins
    }


def get_peptide_length_histo(peptide_lengths: np.ndarray) -> Dict[int, int]:
    """
    Get a dictionary mapping each unique peptide length to its frequency

    Parameters
    ----------
        peptide_lengths: np.ndarray
            Numpy array containing the length of each sequence

    Returns
    -------
        peptide_length_histogram: Dict[int, int]
            Dictionary mapping each unique peptide length to its frequency
    """
    bins = np.bincount(peptide_lengths)
    return {
        curr_bin: count for curr_bin, count in enumerate(bins) if count != 0
    }


def get_peptide_lengths(results_table: DataFrame) -> np.ndarray:
    """
    Get a numpy array containing the length of each peptide sequence

    Parameters
    ----------
    results_table: DataFrame
        Parsed spectrum match table

    Returns
    -------
    sequence_lengths: np.ndarray
        Numpy array containing the length of each sequence, listed in the
        same order that the sequences are provided in.
    """
    # Mass modifications do not contribute to sequence length
    # If PTMs ae represented in ProForma notation this filtering operation
    # needs to be reimplemented
    alpha_re = re.compile("[^a-zA-Z]")
    peptide_sequences = results_table["sequence"]
    return peptide_sequences.str.replace(alpha_re, "", regex=True).apply(len)


def get_report_dict(
    results_table: DataFrame, score_bins: List[float] = SCORE_BINS
) -> Optional[Dict]:
    """
    Generate sequencing run report

    Parameters
    ----------
    results_table: DataFrame
        Parsed spectrum match table
    score_bins: List[float], Optional
        Confidence scores for creating confidence CMF, see get_score_bins

    Returns
    -------
    report_gen: Dict
        Generated report represented as a dictionary, or None if no
        sequencing predictions were logged
    """
    if results_table.empty:
        return None

    peptide_lengths = get_peptide_lengths(results_table)
    return {
        "num_spectra": len(results_table),
        "score_bins": get_score_bins(results_table, score_bins),
        "max_sequence_length": int(np.max(peptide_lengths)),
        "min_sequence_length": int(np.min(peptide_lengths)),
        "median_sequence_length": int(np.median(peptide_lengths)),
        "peptide_length_histogram": get_peptide_length_histo(peptide_lengths),
    }


def log_run_report(
    start_time: Optional[int] = None, end_time: Optional[int] = None
) -> None:
    """
    Log general run report

    Parameters
    ----------
    start_time : Optional[int], default=None
        The start time of the sequencing run in seconds since the epoch.
    end_time : Optional[int], default=None
        The end time of the sequencing run in seconds since the epoch.
    """
    logger.info("======= End of Run Report =======")
    if (start_time is not None) and (end_time is not None):
        end_time = time.time()
        elapsed_time = end_time - start_time
        start_timestamp = datetime.fromtimestamp(start_time).strftime(
            "%y/%m/%d %H:%M:%S"
        )
        end_timestamp = datetime.fromtimestamp(end_time).strftime(
            "%y/%m/%d %H:%M:%S"
        )
        logger.info(f"Run Start Timestamp: {start_timestamp}")
        logger.info(f"Run End Timestamp: {end_timestamp}")
        logger.info(f"Time Elapsed: {int(elapsed_time)}s")

    logger.info(f"Executed Command: {' '.join(sys.argv)}")
    logger.info(f"Executed on Host Machine: {socket.gethostname()}")

    if torch.cuda.is_available():
        gpu_util = torch.cuda.max_memory_allocated()
        logger.info(f"Max GPU Memory Utilization: {gpu_util >> 20}MiB")


def log_sequencing_report(
    predictions: Tuple[str, Tuple[str, str], float, float, float, float, str],
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    score_bins: List[float] = SCORE_BINS,
) -> None:
    """
    Log sequencing run report

    next_prediction : Tuple[
        str, Tuple[str, str], float, float, float, float, str
    ]
        PSM predictions
    start_time : Optional[int], default=None
        The start time of the sequencing run in seconds since the epoch.
    end_time : Optional[int], default=None
        The end time of the sequencing run in seconds since the epoch.
    score_bins: List[float], Optional
        Confidence scores for creating confidence score distribution,
        see get_score_bins
    """
    log_run_report(start_time=start_time, end_time=end_time)
    run_report = get_report_dict(
        DataFrame(
            {
                "sequence": [psm[0] for psm in predictions],
                "score": [psm[2] for psm in predictions],
            }
        ),
        score_bins=score_bins,
    )

    if run_report is None:
        logger.warning(
            f"No predictions were logged, this may be due to an error"
        )
    else:
        num_spectra = run_report["num_spectra"]
        logger.info(f"Sequenced {num_spectra:,} spectra")
        logger.info("Score Distribution:")
        for score, pop in sorted(run_report["score_bins"].items()):
            logger.info(
                f"{pop:,} spectra ({pop / num_spectra:.2%}) scored >= {score}"
            )

        logger.info(f"Min Peptide Length: {run_report['min_sequence_length']}")
        logger.info(f"Max Peptide Length: {run_report['max_sequence_length']}")
