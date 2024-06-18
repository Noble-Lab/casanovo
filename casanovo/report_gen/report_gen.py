import re

from os import PathLike
from typing import Dict, List

import numpy as np

from pyteomics.mztab import MzTab
from pandas import DataFrame

SCORE_BINS = [0.0, 0.5, 0.9, 0.95, 0.99]
MZTAB_EXT = ".mztab"

def getMzTabPath(results_path: PathLike) -> PathLike:
    """
    Convert output from setup_logging (see casanovo.py) to valid mzTab file path

    Parameters
    ----------    
        results_path: PathLike
            Output from setup_logging

    Returns
    -------
        results_path: PathLike
            Valid .mztab file path
    """
    results_path = str(results_path)

    if results_path[-len(MZTAB_EXT):] != MZTAB_EXT:
        results_path += MZTAB_EXT

    return results_path

def parseMzTabResults(results_path: PathLike) -> DataFrame:
    """
    Parse the spectrum match table from a mzTab file

    Parameters
    ----------    
        results_path: PathLike
            Path to mzTab file

    Returns
    -------
        results_table: DataFrame
            Parsed spectrum match table
    """
    results_path = getMzTabPath(results_path)
    file_reader = MzTab(results_path)
    return file_reader.spectrum_match_table

def getNumSpectra(results_table: DataFrame) -> int:
    """
    Get the number of spectra in a results table

    Parameters
    ----------
        results_table: DataFrame
            Parsed spectrum match table

    Returns
    -------
        num_spectra: int
            Number of spectra in results table
    """
    return results_table.shape[0]

def getScoreBins(results_table: DataFrame, score_bins: List[float]) -> Dict[float, int]:
    """
    From a list of confidence scores, return a dictionary mapping each confidence score
    to the number of spectra with a confidence greater than or equal to it.

    Parameters
    ----------
        results_table: DataFrame
            Parsed spectrum match table
        score_bins: List[float]
            Confidence scores to map

    Returns
    -------
        score_bin_dict: Dict[float, int]
            Dictionary mapping each confidence score to the number of spectra with a confidence
            greater than or equal to it.
    """
    se_scores = results_table["search_engine_score[1]"].to_numpy()
    score_bin_dict = {score: len(se_scores[se_scores >= score]) for score in score_bins}
    return score_bin_dict

def getPeptideLengths(results_table: DataFrame) -> np.ndarray:
    """
    Get a numpy array containing the length of each peptide sequence in results_table

    Parameters
    ----------
        results_table: DataFrame
            Parsed spectrum match table

    Returns
    -------
        sequence_lengths: np.ndarray
            Numpy array containing the length of each sequence, listed in the same order
            that the sequences are provided in.
    """
    # Mass modifications do not contribute to sequence length
    alpha_re = re.compile("[^a-zA-Z]")
    filter_fun = lambda x: alpha_re.sub("", x)
    peptide_sequences = results_table["sequence"].copy()
    filtered_sequences = peptide_sequences.apply(filter_fun)
    sequence_lengths = filtered_sequences.apply(len)

    return sequence_lengths.to_numpy()

def getPeptideLengthHisto(peptide_lengths: np.ndarray) -> Dict[int, int]:
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
    lengths, counts = np.unique(peptide_lengths, return_counts=True)
    return dict(zip(lengths.tolist(), counts.tolist()))

def genReport(
    results_path: PathLike,
    score_bins: List[float] = SCORE_BINS
) -> Dict:
    """
    Generate sequencing run report

    Parameters
    ----------    
        results_path: PathLike
            Path to mzTab file
        score_bins: List[float], Optional
            Confidence scores for creating confidence CMF, see getScoreBins

    Returns:
        report_gen: Dict
            Generated report, represented as a dictionary
    """
    results_table = parseMzTabResults(results_path)
    peptide_lengths = getPeptideLengths(results_table)

    return {
        "num_spectra": getNumSpectra(results_table),
        "score_bins": getScoreBins(results_table, score_bins),
        "max_sequence_length": int(np.max(peptide_lengths)),
        "min_sequence_length": int(np.min(peptide_lengths)),
        "median_sequence_length": int(np.median(peptide_lengths)),
        "peptide_length_histogram": getPeptideLengthHisto(peptide_lengths)
    }
