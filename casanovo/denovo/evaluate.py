import re
from typing import Dict, List, Tuple

import numpy as np
from spectrum_utils.utils import mass_diff


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
        Boolean flag indicating whether each paired-up amino acid matches across both
        peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    """
    aa_matches = np.zeros(max(len(peptide1), len(peptide2)), np.bool_)
    # Find longest mass-matching prefix.
    i1, i2, cum_mass1, cum_mass2 = 0, 0, 0.0, 0.0
    while i1 < len(peptide1) and i2 < len(peptide2):
        aa_mass1, aa_mass2 = aa_dict[peptide1[i1]], aa_dict[peptide2[i2]]
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
    Find the matching prefix and suffix amino acids between two peptide sequences.

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
        Boolean flag indicating whether each paired-up amino acid matches across both
        peptide sequences.
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
        aa_mass1, aa_mass2 = aa_dict[peptide1[i1]], aa_dict[peptide2[i2]]
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
        Boolean flag indicating whether each paired-up amino acid matches across both
        peptide sequences.
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
        For each pair of peptide sequences: (i) boolean flags indicating whether each
        paired-up amino acid matches across both peptide sequences, (ii) boolean flag to
        indicate whether the two peptide sequences fully match.
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


def calc_eval_metrics(
    aa_match_binary_list, orig_total_num_aa, pred_total_num_aa
):
    """
    Calculate evaluation metrics using amino acid matches

    Parameters
    ----------
    aa_match_binary_list : list of lists
        List of amino acid matches in each predicted peptide
    orig_total_num_aa : int
        Number of amino acids in the original peptide sequences
    pred_total_num_aa : int
        Number of amino acids in the predicted peptide sequences
    Returns
    -------
    aa_precision: float
        Number of correct aa predictions divided by all predicted aa
    aa_recall: float
        Number of correct aa predictions divided by all original aa
    pep_recall: float
        Number of correct peptide predictions divided by all original peptide
    """

    correct_aa_count = sum(
        [sum(pred_tuple[0]) for pred_tuple in aa_match_binary_list]
    )
    aa_recall = correct_aa_count / (orig_total_num_aa + 1e-8)
    aa_precision = correct_aa_count / (pred_total_num_aa + 1e-8)
    pep_recall = sum(
        [pred_tuple[1] for pred_tuple in aa_match_binary_list]
    ) / (len(aa_match_binary_list) + 1e-8)

    return aa_precision, aa_recall, pep_recall


def aa_precision_recall_with_threshold(
    correct_aa_confidences, all_aa_confidences, num_original_aa, threshold
):
    """
    Calculate precision and recall for the given amino acid confidence score threshold

    Parameters
    ----------
    correct_aa_confidences : list
        List of confidence scores for correct amino acids predictions
    all_aa_confidences : int
        List of confidence scores for all amino acids prediction
    num_original_aa : int
        Number of amino acids in the predicted peptide sequences
    threshold : float
        Amino acid confidence score threshold

    Returns
    -------
    aa_precision: float
        Number of correct aa predictions divided by all predicted aa
    aa_recall: float
        Number of correct aa predictions divided by all original aa
    """

    correct_aa = sum([conf >= threshold for conf in correct_aa_confidences])
    predicted_aa = sum([conf >= threshold for conf in all_aa_confidences])

    aa_precision = correct_aa / predicted_aa
    aa_recall = correct_aa / num_original_aa

    return aa_precision, aa_recall
