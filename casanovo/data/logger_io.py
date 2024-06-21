from logging import Logger
from typing import Tuple, List, Dict
from sys import argv
from socket import gethostname
from time import time
from datetime import datetime

import re

from pandas import DataFrame
from .prediction_io import PredictionWriter

import numpy as np
import torch

SCORE_BINS = [0.0, 0.5, 0.9, 0.95, 0.99]

def get_num_spectra(results_table: DataFrame) -> int:
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

def get_score_bins(results_table: DataFrame, score_bins: List[float]) -> Dict[float, int]:
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
    se_scores = results_table["score"].to_numpy()
    score_bin_dict = {score: len(se_scores[se_scores >= score]) for score in score_bins}
    return score_bin_dict

def get_peptide_lengths(results_table: DataFrame) -> np.ndarray:
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
    lengths, counts = np.unique(peptide_lengths, return_counts=True)
    return dict(zip(lengths.tolist(), counts.tolist()))

class LogPredictionWriter(PredictionWriter):
    """
    Log predictions and use them to generate a sequencing run report
    when save is called, usually when the parent ModelRunner goes out
    of context. Sequencing run will be written at the INFO level.

    Parameters
    ----------
        logger : Logger
            logger to write sequencing run report to
        score_bins : List[float] (optional)
            Confidence score bins for generating sequence confidence score
            cmf. Defaults to [0.0, 0.5, 0.9, 0.95, 0.99].
    """
    def __init__(self, logger: Logger, score_bins: List[float] = SCORE_BINS) -> None:
        self.logger = logger
        self.score_bins = score_bins
        self.start_time = None
        self.predictions = {
            "sequence": list(),
            "score": list(),
        }

    def log_start_time(self) -> None:
        """
        Record the sequencing run start timestamp
        """
        self.start_time = time()

    def append_prediction(
        self,
        next_prediction: Tuple[
            str,
            Tuple[str, str],
            float,
            float,
            float,
            float,
            str
        ]
    ) -> None:
        """
        Add new prediction to log writer context

        Parameters
        ----------
        next_prediction : Tuple[str, Tuple[str, str], float, float, float, float, str]
            Tuple containing next prediction data. The tuple should contain the following:
                - str: next peptide prediction
                - Tuple[str, str]: sample origin file path, origin file index number ("index={i}") 
                - float: peptide prediction score (search engine score)
                - float: charge
                - float: precursor m/z
                - float: peptide mass
                - str: aa scores for each peptide in sequence, comma separated
        """
        predicted_sequence = next_prediction[0]
        prediction_score = next_prediction[2]
        self.predictions["sequence"].append(predicted_sequence)
        self.predictions["score"].append(prediction_score)

    def get_results_table(self) -> DataFrame:
        return DataFrame(self.predictions)

    def get_report_dict(self) -> Dict:
        """
        Generate sequencing run report

        Parameters
        ----------    
            score_bins: List[float], Optional
                Confidence scores for creating confidence CMF, see getScoreBins

        Returns:
            report_gen: Dict
                Generated report, represented as a dictionary
        """
        results_table = self.get_results_table()
        peptide_lengths = get_peptide_lengths(results_table)

        return {
            "num_spectra": get_num_spectra(results_table),
            "score_bins": get_score_bins(results_table, self.score_bins),
            "max_sequence_length": int(np.max(peptide_lengths)),
            "min_sequence_length": int(np.min(peptide_lengths)),
            "median_sequence_length": int(np.median(peptide_lengths)),
            "peptide_length_histogram": get_peptide_length_histo(peptide_lengths)
        }

    def save(self) -> None:
        """
        Log sequencing run report
        """
        self.logger.info("======= Sequencing Run Report =======")
        if self.start_time is not None:
            end_time = time()
            elapsed_time = end_time - self.start_time
            self.logger.info(f"Sequencing Run Start Timestamp: {int(self.start_time)}s")
            self.logger.info(f"Sequencing Run End Timestamp: {int(end_time)}s")
            self.logger.info(f"Time Elapsed: {int(elapsed_time)}s")

        run_report = self.get_report_dict()
        run_date_string = datetime.now().strftime("%m/%d/%y %H:%M:%S")
        self.logger.info(f"Executed Command: {' '.join(argv)}")
        self.logger.info(f"Executed on Host Machine: {gethostname()}")
        self.logger.info(f"Sequencing run date: {run_date_string}")
        self.logger.info(f"Sequenced {run_report['num_spectra']} spectra")
        self.logger.info(f"Sequence Score CMF: {run_report['score_bins']}")
        self.logger.info(f"Max Sequence Length: {run_report['max_sequence_length']}")
        self.logger.info(f"Min Sequence Length: {run_report['min_sequence_length']}")

        if torch.cuda.is_available():
            gpu_util = torch.cuda.max_memory_allocated() / (10 ** 6)
            self.logger.info(f"Max GPU Memory Utilization: {int(gpu_util)}mb")
