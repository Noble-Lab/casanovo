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
    aa_match : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across both
        peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    """
    aa_match = np.zeros(max(len(peptide1), len(peptide2)), np.bool_)
    # Find longest mass-matching prefix.
    i1, i2, cum_mass1, cum_mass2 = 0, 0, 0.0, 0.0
    while i1 < len(peptide1) and i2 < len(peptide2):
        aa_mass1, aa_mass2 = aa_dict[peptide1[i1]], aa_dict[peptide2[i2]]
        if (
            abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True))
            < cum_mass_threshold
        ):
            aa_match[max(i1, i2)] = (
                abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            )
            i1, i2 = i1 + 1, i2 + 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2
        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 + 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 + 1, cum_mass2 + aa_mass2
    return aa_match, aa_match.all()


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
    aa_match : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across both
        peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    """
    # Find longest mass-matching prefix.
    aa_match, pep_match = aa_match_prefix(
        peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
    )
    # No need to evaluate the suffixes if the sequences already fully match.
    if pep_match:
        return aa_match, pep_match
    # Find longest mass-matching suffix.
    i1, i2 = len(peptide1) - 1, len(peptide2) - 1
    i_stop = np.argwhere(~aa_match)[0]
    cum_mass1, cum_mass2 = 0.0, 0.0
    while i1 >= i_stop and i2 >= i_stop:
        aa_mass1, aa_mass2 = aa_dict[peptide1[i1]], aa_dict[peptide2[i2]]
        if (
            abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True))
            < cum_mass_threshold
        ):
            aa_match[max(i1, i2)] = (
                abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            )
            i1, i2 = i1 - 1, i2 - 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2
        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 - 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 - 1, cum_mass2 + aa_mass2
    return aa_match, aa_match.all()


def match_aa(orig_seq, pred_seq, aa_dict, eval_direction="best"):
    """
    Find the matching amino acids of an original and predicted peptide

    Parameters
    ----------
    orig_seq : list
        List of amino acids in the original peptide
    pred_seq : list
        List of amino acids in the predicted peptide
    aa_dict: dict
        Dictionary of amino acid masses
    eval_direction: str, default: 'best'
        Direction of evaluation while finding amino acid matches, e.g. 'forward', 'backward', 'best'


    Returns
    -------
    aa_match: list
        Binary list of c
    pep_match: int
        1 if all amino acid in two sequences match

    """

    if eval_direction == "best":
        aa_match, pep_match = aa_match_prefix_suffix(orig_seq, pred_seq, aa_dict)
        n_mismatch_aa = len(pred_seq) - len(aa_match)
        aa_match += n_mismatch_aa * [0]

    elif eval_direction == "forward":
        aa_match, pep_match = aa_match_prefix(
            orig_seq, pred_seq, aa_dict
        )

        n_mismatch_aa = len(pred_seq) - len(aa_match)
        aa_match += n_mismatch_aa * [0]

    elif eval_direction == "backward":
        reverse_aa_match, pep_match = aa_match_prefix(
            list(reversed(orig_seq)), list(reversed(pred_seq)), aa_dict
        )

        aa_match = list(reversed(reverse_aa_match))
        n_mismatch_aa = len(pred_seq) - len(aa_match)
        aa_match = n_mismatch_aa * [0] + aa_match

    return aa_match, pep_match


def batch_aa_match(
    pred_pep_seqs, true_pep_seqs, aa_dict, eval_direction="best"
):
    """
    Find the matching amino acids of an original and predicted peptide

    Parameters
    ----------
    pred_pep_seqs : list
        List of predicted peptides, i.e. list of amino acid sequences
    true_pep_seqs : list
        List of ground truth peptide labels
    aa_dict: dict
        Dictionary of amino acid masses
    eval_direction: str, default: 'best'
        Direction of evaluation while finding amino acid matches, e.g. 'forward', 'backward', 'best'


    Returns
    -------
    all_aa_match: list
        Binary list of lists corresponding to amino acid matches for all predicted peptides
    orig_total_num_aa: int
        Total number of amino acids in the ground truth peptide labels
    pred_total_num_aa: int
        Total number of amino acids in the predicted peptide labels

    """

    orig_total_num_aa = 0
    pred_total_num_aa = 0
    all_aa_match = []

    for pred_ind in range(len(pred_pep_seqs)):

        pred = re.split(r"(?<=.)(?=[A-Z])", pred_pep_seqs[pred_ind])
        orig = re.split(r"(?<=.)(?=[A-Z])", true_pep_seqs[pred_ind])
        orig_total_num_aa += len(orig)
        pred_total_num_aa += len(pred)

        aa_match, pep_match = match_aa(
            orig, pred, aa_dict, eval_direction=eval_direction
        )
        all_aa_match += [(aa_match, pep_match)]

    return all_aa_match, orig_total_num_aa, pred_total_num_aa


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
