import re

from os import PathLike
from typing import Dict, List

import numpy as np

from pyteomics.mztab import MzTab
from pandas import DataFrame

SCORE_BINS = [0.0, 0.5, 0.9, 0.95, 0.99]
MZTAB_EXT = ".mztab"

def getMzTabPath(results_path: PathLike) -> PathLike:
    results_path = str(results_path)

    if results_path[-len(MZTAB_EXT):] != MZTAB_EXT:
        results_path += MZTAB_EXT

    return results_path

def parseMzTabResults(results_path: PathLike) -> DataFrame:
    results_path = getMzTabPath(results_path)
    file_reader = MzTab(results_path)
    return file_reader.spectrum_match_table

def getNumSpectra(results_table: DataFrame) -> int:
    return results_table.shape[0]

def getScoreBins(results_table: DataFrame, score_bins: List[float]) -> Dict[float, int]:
    se_scores = results_table["search_engine_score[1]"].to_numpy()
    score_bin_dict = {score: len(se_scores[se_scores >= score]) for score in score_bins}
    return score_bin_dict

def getPeptideLengths(results_table: DataFrame) -> np.ndarray:
    # Mass modifications do not contribute to sequence length
    alpha_re = re.compile("[^a-zA-Z]")
    filter_fun = lambda x: alpha_re.sub("", x)
    peptide_sequences = results_table["sequence"].copy()
    filtered_sequences = peptide_sequences.apply(filter_fun)
    sequence_lengths = filtered_sequences.apply(len)

    return sequence_lengths.to_numpy()

def getPeptideLengthHisto(peptide_lengths: np.ndarray) -> Dict[int, int]:
    lengths, counts = np.unique(peptide_lengths, return_counts=True)
    return dict(zip(lengths.tolist(), counts.tolist()))

def genReport(
    results_path: PathLike,
    score_bins = SCORE_BINS
) -> Dict:
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
