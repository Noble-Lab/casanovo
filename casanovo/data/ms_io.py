"""Mass spectrometry file type input/output operations."""

import collections
import csv
import logging
import operator
import os
import re
from datetime import datetime
from pathlib import Path
from socket import gethostname
from sys import argv
from time import time
from typing import List, Optional, Dict

import natsort
import numpy as np
import torch
from pandas import DataFrame

from .. import __version__
from ..config import Config

SCORE_BINS = [0.0, 0.5, 0.9, 0.95, 0.99]

logger = logging.getLogger("casanovo")


def get_score_bins(
    results_table: DataFrame, score_bins: List[float]
) -> Dict[float, int]:
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
    score_bin_dict = {
        score: len(se_scores[se_scores >= score]) for score in score_bins
    }
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


class MztabWriter:
    """
    Export spectrum identifications to an mzTab file.

    Parameters
    ----------
    filename : str
        The name of the mzTab file.
    score_bins : List[float] (optional)
            Confidence score bins for generating sequence confidence score
            cmf. Defaults to [0.0, 0.5, 0.9, 0.95, 0.99].
    """

    def __init__(self, filename: str, score_bins: List[float] = SCORE_BINS):
        self.filename = filename
        self.metadata = [
            ("mzTab-version", "1.0.0"),
            ("mzTab-mode", "Summary"),
            ("mzTab-type", "Identification"),
            (
                "description",
                f"Casanovo identification file "
                f"{os.path.splitext(os.path.basename(self.filename))[0]}",
            ),
            ("software[1]", f"[MS, MS:1003281, Casanovo, {__version__}]"),
            (
                "psm_search_engine_score[1]",
                "[MS, MS:1001143, search engine specific score for PSMs, ]",
            ),
        ]
        self._run_map = {}
        self.psms = []
        self.start_time = time()
        self.score_bins = score_bins

    def set_metadata(self, config: Config, **kwargs) -> None:
        """
        Specify metadata information to write to the mzTab header.

        Parameters
        ----------
        config : Config
            The active configuration options.
        kwargs
            Additional configuration options (i.e. from command-line arguments).
        """
        # Derive the fixed and variable modifications from the residue alphabet.
        known_mods = {
            "+57.021": "[UNIMOD, UNIMOD:4, Carbamidomethyl, ]",
            "+15.995": "[UNIMOD, UNIMOD:35, Oxidation, ]",
            "+0.984": "[UNIMOD, UNIMOD:7, Deamidated, ]",
            "+42.011": "[UNIMOD, UNIMOD:1, Acetyl, ]",
            "+43.006": "[UNIMOD, UNIMOD:5, Carbamyl, ]",
            "-17.027": "[UNIMOD, UNIMOD:385, Ammonia-loss, ]",
        }
        residues = collections.defaultdict(set)
        for aa, mass in config["residues"].items():
            aa_mod = re.match(r"([A-Z]?)([+-]?(?:[0-9]*[.])?[0-9]+)", aa)
            if aa_mod is None:
                residues[aa].add(None)
            else:
                residues[aa_mod[1]].add(aa_mod[2])
        fixed_mods, variable_mods = [], []
        for aa, mods in residues.items():
            if len(mods) > 1:
                for mod in mods:
                    if mod is not None:
                        variable_mods.append((aa, mod))
            elif None not in mods:
                fixed_mods.append((aa, mods.pop()))

        # Add all config values to the mzTab metadata section.
        if len(fixed_mods) == 0:
            self.metadata.append(
                (
                    "fixed_mod[1]",
                    "[MS, MS:1002453, No fixed modifications searched, ]",
                )
            )
        else:
            for i, (aa, mod) in enumerate(fixed_mods, 1):
                self.metadata.append(
                    (
                        f"fixed_mod[{i}]",
                        known_mods.get(mod, f"[CHEMMOD, CHEMMOD:{mod}, , ]"),
                    )
                )
                self.metadata.append(
                    (f"fixed_mod[{i}]-site", aa if aa else "N-term")
                )
        if len(variable_mods) == 0:
            self.metadata.append(
                (
                    "variable_mod[1]",
                    "[MS, MS:1002454, No variable modifications searched,]",
                )
            )
        else:
            for i, (aa, mod) in enumerate(variable_mods, 1):
                self.metadata.append(
                    (
                        f"variable_mod[{i}]",
                        known_mods.get(mod, f"[CHEMMOD, CHEMMOD:{mod}, , ]"),
                    )
                )
                self.metadata.append(
                    (f"variable_mod[{i}]-site", aa if aa else "N-term")
                )
        for i, (key, value) in enumerate(kwargs.items(), 1):
            self.metadata.append(
                (f"software[1]-setting[{i}]", f"{key} = {value}")
            )
        for i, (key, value) in enumerate(config.items(), len(kwargs) + 1):
            if key not in ("residues",):
                self.metadata.append(
                    (f"software[1]-setting[{i}]", f"{key} = {value}")
                )

    def set_ms_run(self, peak_filenames: List[str]) -> None:
        """
        Add input peak files to the mzTab metadata section.

        Parameters
        ----------
        peak_filenames : List[str]
            The input peak file name(s).
        """
        for i, filename in enumerate(natsort.natsorted(peak_filenames), 1):
            filename = os.path.abspath(filename)
            self.metadata.append(
                (f"ms_run[{i}]-location", Path(filename).as_uri()),
            )
            self._run_map[filename] = i

    def get_report_dict(self) -> Optional[Dict]:
        """
        Generate sequencing run report

        Parameters
        ----------
            score_bins: List[float], Optional
                Confidence scores for creating confidence CMF, see getScoreBins

        Returns:
            report_gen: Dict
                Generated report represented as a dictionary, or None if no
                sequencing predictions were logged
        """
        results_table = DataFrame(
            {
                "sequence": [psm[0] for psm in self.psms],
                "score": [psm[2] for psm in self.psms],
            }
        )

        if results_table.empty:
            return None

        peptide_lengths = get_peptide_lengths(results_table)
        return {
            "num_spectra": len(results_table),
            "score_bins": get_score_bins(results_table, self.score_bins),
            "max_sequence_length": int(np.max(peptide_lengths)),
            "min_sequence_length": int(np.min(peptide_lengths)),
            "median_sequence_length": int(np.median(peptide_lengths)),
            "peptide_length_histogram": get_peptide_length_histo(
                peptide_lengths
            ),
        }

    def log_run_report(self) -> None:
        """
        Log sequencing run report
        """
        logger.info("======= Sequencing Run Report =======")
        if self.start_time is not None:
            end_time = time()
            elapsed_time = end_time - self.start_time
            logger.info(
                f"Sequencing Run Start Timestamp: {int(self.start_time)}s"
            )
            logger.info(f"Sequencing Run End Timestamp: {int(end_time)}s")
            logger.info(f"Time Elapsed: {int(elapsed_time)}s")

        run_report = self.get_report_dict()
        run_date_string = datetime.now().strftime("%m/%d/%y %H:%M:%S")
        logger.info(f"Executed Command: {' '.join(argv)}")
        logger.info(f"Executed on Host Machine: {gethostname()}")
        logger.info(f"Sequencing run date: {run_date_string}")
        num_spectra = 0 if run_report is None else run_report["num_spectra"]

        if run_report is None:
            logger.warning(
                f"No predictions were logged, this may be due to an error"
            )
        else:
            logger.info(f"Sequenced {num_spectra} spectra")

        if run_report is not None:
            logger.info(f"Score Distribution:")
            for score, pop in sorted(run_report["score_bins"].items()):
                pop_percentage = 100 * pop / num_spectra
                logger.info(
                    f"{pop} spectra ({pop_percentage:.2f}%) scored >= {score}"
                )

            logger.info(
                f"Max Sequence Length: {run_report['max_sequence_length']}"
            )
            logger.info(
                f"Min Sequence Length: {run_report['min_sequence_length']}"
            )

        if torch.cuda.is_available():
            gpu_util = torch.cuda.max_memory_allocated() / (10**6)
            logger.info(f"Max GPU Memory Utilization: {int(gpu_util)}mb")

    def save(self) -> None:
        """
        Export the spectrum identifications to the mzTab file and
        log end of run report
        """
        self.log_run_report()
        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator=os.linesep)
            # Write metadata.
            for row in self.metadata:
                writer.writerow(["MTD", *row])
            # Write PSMs.
            writer.writerow(
                [
                    "PSH",
                    "sequence",
                    "PSM_ID",
                    "accession",
                    "unique",
                    "database",
                    "database_version",
                    "search_engine",
                    "search_engine_score[1]",
                    "modifications",
                    "retention_time",
                    "charge",
                    "exp_mass_to_charge",
                    "calc_mass_to_charge",
                    "spectra_ref",
                    "pre",
                    "post",
                    "start",
                    "end",
                    "opt_ms_run[1]_aa_scores",
                ]
            )
            for i, psm in enumerate(
                natsort.natsorted(self.psms, key=operator.itemgetter(1)), 1
            ):
                filename, idx = os.path.abspath(psm[1][0]), psm[1][1]
                writer.writerow(
                    [
                        "PSM",
                        psm[0],  # sequence
                        i,  # PSM_ID
                        "null",  # accession
                        "null",  # unique
                        "null",  # database
                        "null",  # database_version
                        f"[MS, MS:1003281, Casanovo, {__version__}]",
                        psm[2],  # search_engine_score[1]
                        # FIXME: Modifications should be specified as
                        #  controlled vocabulary terms.
                        "null",  # modifications
                        # FIXME: Can we get the retention time from the data
                        #  loader?
                        "null",  # retention_time
                        int(psm[3]),  # charge
                        psm[4],  # exp_mass_to_charge
                        psm[5],  # calc_mass_to_charge
                        f"ms_run[{self._run_map[filename]}]:{idx}",
                        "null",  # pre
                        "null",  # post
                        "null",  # start
                        "null",  # end
                        psm[6],  # opt_ms_run[1]_aa_scores
                    ]
                )
