"""Methods to evaluate peptide-spectrum predictions."""
import re
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyteomics.mgf as mgf
from spectrum_utils.utils import mass_diff

import depthcharge


def aa_match_prefix(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
) -> Tuple[np.ndarray, bool]:
    """
    Find the matching prefix amino acids between two peptide sequences.

    This is a similar evaluation criterion as used by DeepNovo.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    """
    aa_matches = np.zeros(max(len(peptide1), len(peptide2)), np.bool_)
    # Find longest mass-matching prefix.
    i1, i2, cum_mass1, cum_mass2 = 0, 0, 0.0, 0.0
    while i1 < len(peptide1) and i2 < len(peptide2):
        aa_mass1 = aa_dict.get(peptide1[i1], 0)
        aa_mass2 = aa_dict.get(peptide2[i2], 0)
        if (
            abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True))
            < cum_mass_threshold
        ):
            aa_matches[max(i1, i2)] = (
                abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            )
            i1, i2 = i1 + 1, i2 + 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2
        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 + 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 + 1, cum_mass2 + aa_mass2
    return aa_matches, aa_matches.all()


def aa_match_prefix_suffix(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
) -> Tuple[np.ndarray, bool]:
    """
    Find the matching prefix and suffix amino acids between two peptide
    sequences.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    """
    # Find longest mass-matching prefix.
    aa_matches, pep_match = aa_match_prefix(
        peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
    )
    # No need to evaluate the suffixes if the sequences already fully match.
    if pep_match:
        return aa_matches, pep_match
    # Find longest mass-matching suffix.
    i1, i2 = len(peptide1) - 1, len(peptide2) - 1
    i_stop = np.argwhere(~aa_matches)[0]
    cum_mass1, cum_mass2 = 0.0, 0.0
    while i1 >= i_stop and i2 >= i_stop:
        aa_mass1 = aa_dict.get(peptide1[i1], 0)
        aa_mass2 = aa_dict.get(peptide2[i2], 0)
        if (
            abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True))
            < cum_mass_threshold
        ):
            aa_matches[max(i1, i2)] = (
                abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            )
            i1, i2 = i1 - 1, i2 - 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2
        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 - 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 - 1, cum_mass2 + aa_mass2
    return aa_matches, aa_matches.all()


def aa_match(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
    mode: str = "best",
) -> Tuple[np.ndarray, bool]:
    """
    Find the matching amino acids between two peptide sequences.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.
    mode : {"best", "forward", "backward"}
        The direction in which to find matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    """
    if mode == "best":
        return aa_match_prefix_suffix(
            peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
        )
    elif mode == "forward":
        return aa_match_prefix(
            peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
        )
    elif mode == "backward":
        aa_matches, pep_match = aa_match_prefix(
            list(reversed(peptide1)),
            list(reversed(peptide2)),
            aa_dict,
            cum_mass_threshold,
            ind_mass_threshold,
        )
        return aa_matches[::-1], pep_match
    else:
        raise ValueError("Unknown evaluation mode")


def aa_match_batch(
    peptides1: List[str],
    peptides2: List[str],
    aa_dict: Dict[str, float],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
    mode: str = "best",
) -> Tuple[List[Tuple[np.ndarray, bool]], int, int]:
    """
    Find the matching amino acids between multiple pairs of peptide sequences.

    Parameters
    ----------
    peptides1 : List[str]
        The first list of (untokenized) peptide sequences to be compared.
    peptides2 : List[str]
        The second list of (untokenized) peptide sequences to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.
    mode : {"best", "forward", "backward"}
        The direction in which to find matching amino acids.

    Returns
    -------
    aa_matches_batch : List[Tuple[np.ndarray, bool]]
        For each pair of peptide sequences: (i) boolean flags indicating whether
        each paired-up amino acid matches across both peptide sequences, (ii)
        boolean flag to indicate whether the two peptide sequences fully match.
    n_aa1: int
        Total number of amino acids in the first list of peptide sequences.
    n_aa2: int
        Total number of amino acids in the second list of peptide sequences.
    """
    aa_matches_batch, n_aa1, n_aa2 = [], 0, 0
    for peptide1, peptide2 in zip(peptides1, peptides2):
        tokens1 = re.split(r"(?<=.)(?=[A-Z])", peptide1)
        tokens2 = re.split(r"(?<=.)(?=[A-Z])", peptide2)
        n_aa1, n_aa2 = n_aa1 + len(tokens1), n_aa2 + len(tokens2)
        aa_matches_batch.append(
            aa_match(
                tokens1,
                tokens2,
                aa_dict,
                cum_mass_threshold,
                ind_mass_threshold,
                mode,
            )
        )
    return aa_matches_batch, n_aa1, n_aa2


def aa_match_metrics(
    aa_matches_batch: List[Tuple[np.ndarray, bool]],
    n_aa_true: int,
    n_aa_pred: int,
) -> Tuple[float, float, float]:
    """
    Calculate amino acid and peptide-level evaluation metrics.

    Parameters
    ----------
    aa_matches_batch : List[Tuple[np.ndarray, bool]]
        For each pair of peptide sequences: (i) boolean flags indicating whether
        each paired-up amino acid matches across both peptide sequences, (ii)
        boolean flag to indicate whether the two peptide sequences fully match.
    n_aa_true: int
        Total number of amino acids in the true peptide sequences.
    n_aa_pred: int
        Total number of amino acids in the predicted peptide sequences.

    Returns
    -------
    aa_precision: float
        The number of correct AA predictions divided by the number of predicted
        AAs.
    aa_recall: float
        The number of correct AA predictions divided by the number of true AAs.
    pep_recall: float
        The number of correct peptide predictions divided by the number of
        peptides.
    """
    n_aa_correct = sum(
        [aa_matches[0].sum() for aa_matches in aa_matches_batch]
    )
    aa_precision = n_aa_correct / (n_aa_pred + 1e-8)
    aa_recall = n_aa_correct / (n_aa_true + 1e-8)
    pep_recall = sum([aa_matches[1] for aa_matches in aa_matches_batch]) / (
        len(aa_matches_batch) + 1e-8
    )
    return aa_precision, aa_recall, pep_recall


def aa_precision_recall(
    aa_scores_correct: List[float],
    aa_scores_all: List[float],
    n_aa_total: int,
    threshold: float,
) -> Tuple[float, float]:
    """
    Calculate amino acid level precision and recall at a given score threshold.

    Parameters
    ----------
    aa_scores_correct : List[float]
        Amino acids scores for the correct amino acids predictions.
    aa_scores_all : List[float]
        Amino acid scores for all amino acids predictions.
    n_aa_total : int
        The total number of amino acids in the predicted peptide sequences.
    threshold : float
        The amino acid score threshold.

    Returns
    -------
    aa_precision: float
        The number of correct amino acid predictions divided by the number of
        predicted amino acids.
    aa_recall: float
        The number of correct amino acid predictions divided by the total number
        of amino acids.
    """
    n_aa_correct = sum([score > threshold for score in aa_scores_correct])
    n_aa_predicted = sum([score > threshold for score in aa_scores_all])
    return n_aa_correct / n_aa_predicted, n_aa_correct / n_aa_total


def _get_true_labels(mgf_filename: str):
    """
    Extract a list of peptide labels from an annotated .mgf and make it compatible with depthcharge AA vocabulary

    Parameters:
    -----------
    mgf_filename : str
        Path to the .mgf file to extract from.

    Returns:
    --------
    mgf_df : pandas DF
        The true peptide labels with column "true_seq"
    """
    seqs = []
    with mgf.MGF(mgf_filename) as f_in:
        for spectrum_dict in f_in:
            seqs.append(spectrum_dict["params"]["seq"])
    mgf_df = pd.DataFrame({"true_seq": seqs})
    mgf_df["true_seq"] = mgf_df["true_seq"].str.replace(
        "C[57.02]", "C+57.021", regex=False
    )
    mgf_df["true_seq"] = mgf_df["true_seq"].str.replace(
        "M[15.99]", "M+15.995", regex=False
    )
    mgf_df["true_seq"] = mgf_df["true_seq"].str.replace(
        "N[0.98]", "N+0.984", regex=False
    )
    mgf_df["true_seq"] = mgf_df["true_seq"].str.replace(
        "Q[0.98]", "Q+0.984", regex=False
    )
    return mgf_df


def _calc_pc(raw_pc_df):
    # TODO: Update docstring
    """
    Calculates the precision and coverage from a pandas dataframe with columns "scan", "true_seq", "output_seq", "output_score"

    Parameters:
    -----------
    raw_pc_df : pandas DataFrame
        A pandas DataFrame with specs as listed in the docstring

    Returns:
    --------
    precision : list[float]
        List of the precision values to plot
    coverage : list[float]
        List of the coverage values to plot
    """
    aa_matches_batch = aa_match_batch(
        raw_pc_df["true_seq"],
        raw_pc_df["output_seq"],
        depthcharge.masses.PeptideMass("massivekb").masses,
    )
    peptide_matches = np.asarray(
        [aa_match[1] for aa_match in aa_matches_batch[0]]
    )
    precision = np.cumsum(peptide_matches) / np.arange(
        1, len(peptide_matches) + 1
    )
    coverage = np.arange(1, len(peptide_matches) + 1) / len(peptide_matches)

    return (
        raw_pc_df["scan"],
        raw_pc_df["true_seq"],
        raw_pc_df["output_seq"],
        raw_pc_df["output_score"],
        precision,
        coverage,
    )


def _get_preccov_mztab_mgf(
    mzt_filename: str, mgf_filename: str, excl_n_terminals=False
):
    # TODO: Update docstring
    """
    Extract the precision and coverage from an new Casanovo .mztab directory and the associated input .mgf directory

    Parameters:
    ----------
    mzt_filename: str
        Path to the .mztab file generated by Casanovo (output).
    mgf_filename : str
        Path to the input .mgf file to Casanovo.
    excl_n_terminals : bool
        If true, will mark all n-terminal peptide inputs/predictions as incorrect, defults to False

    Returns:
    --------
    precision : list[float]
        List of the precision values to plot
    coverage : list[float]
        List of the coverage values to plot
    threshold : int
        The index where, past that point, all predictions do not pass the precursor mass filter
    """
    with open(mzt_filename) as f_in:
        for skiprows, line in enumerate(f_in):
            if line.startswith("PSH"):
                break
    mzt_df = pd.read_csv(mzt_filename, skiprows=skiprows, sep="\t")
    mzt_df = mzt_df[["sequence", "PSM_ID", "search_engine_score[1]"]]
    mzt_df = mzt_df.rename(
        columns={
            "sequence": "output_seq",
            "PSM_ID": "scan",
            "search_engine_score[1]": "output_score",
        }
    )

    mgf_df = _get_true_labels(mgf_filename)
    mgf_df.index.names = ["scan"]

    raw_pc_df = pd.merge(
        mgf_df,
        mzt_df[["scan", "output_seq", "output_score"]],
        how="left",
        on="scan",
    ).sort_values(by=["output_score"], ascending=False)

    if excl_n_terminals:
        discard = ["\+42.011", "\+43.006", "\-17.027", "\+43.006\-17.027"]
        raw_pc_df = raw_pc_df[
            ~raw_pc_df.output_seq.str.contains("|".join(discard))
        ]

    scan, true_seq, output_seq, output_score, precision, coverage = _calc_pc(
        raw_pc_df
    )

    return scan, true_seq, output_seq, output_score, precision, coverage
