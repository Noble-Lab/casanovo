import re
from spectrum_utils.utils import mass_diff
import numpy as np

from depthcharge.masses import PeptideMass

def best_aa_match(orig_seq, pred_seq, aa_dict):
    """
    Find the matching amino acids of an original and predicted peptide sequence if either their prefix or suffix match 

    :param orig_seq: List of amino acids in the original peptide
    :type orig_seq: list
    :param pred_seq: List of amino acids in the predicted peptide
    :type pred_seq: list
    :param aa_dict: Dictionary of amino acid masses
    :type aa_dict: dict
    :return: aa_match - Binary list of matches over predicted peptide
    :rtype: list
    :return: 1 if all amino acid in two sequences match
    :rtype: int    
    """    
    
    cum_aa_threshold = 0.5
    single_aa_threshold = 0.1


    aa_match = []
    pep_match = 0

    cnt_pred = 0
    cnt_orig = 0
    
    orig_aa_mass_list = [aa_dict[aa] for aa in orig_seq]
    pred_aa_mass_list = [aa_dict[aa] for aa in pred_seq]
    
    orig_prefix_mass_list = [0]+list(np.cumsum(orig_aa_mass_list))[:-1]
    orig_suffix_mass_list = list(reversed(np.cumsum(list(reversed(orig_aa_mass_list)))[:-1]))+ [0]
    
    
    pred_prefix_mass_list = [0]+list(np.cumsum(pred_aa_mass_list))[:-1]
    pred_suffix_mass_list = list(reversed(np.cumsum(list(reversed(pred_aa_mass_list)))[:-1]))+ [0]

    
    while cnt_pred < len(pred_seq) and cnt_orig < len(orig_seq):
        
        pred_aa_mass = aa_dict[pred_seq[cnt_pred]]
        orig_aa_mass = aa_dict[orig_seq[cnt_orig]]
        
        if abs(mass_diff(pred_prefix_mass_list[cnt_pred],orig_prefix_mass_list[cnt_orig], mode_is_da = True)) < cum_aa_threshold or abs(mass_diff(pred_suffix_mass_list[cnt_pred],orig_suffix_mass_list[cnt_orig], mode_is_da = True)) < cum_aa_threshold:

            if abs(mass_diff(pred_aa_mass, orig_aa_mass, mode_is_da = True)) < single_aa_threshold:
                aa_match += [1]
            else:
                aa_match += [0]

            cnt_pred += 1
            cnt_orig += 1

        
        elif mass_diff(pred_prefix_mass_list[cnt_pred],orig_prefix_mass_list[cnt_orig], mode_is_da = True) > 0:
            cnt_orig += 1
            
        else:
            cnt_pred += 1
            aa_match += [0]
            
    if sum(aa_match) == len(orig_seq) and len(pred_seq) == len(orig_seq):
        pep_match = 1

    return aa_match, pep_match

    
def find_aa_match_single_pep(orig_seq, pred_seq, aa_dict):
    """
    Find the matching amino acids of an original and predicted peptide sequence as described in DeepNovo. 

    :param orig_seq: List of amino acids in the original peptide
    :type orig_seq: list
    :param pred_seq: List of amino acids in the predicted peptide
    :type pred_seq: list
    :param aa_dict: Dictionary of amino acid masses
    :type aa_dict: dict
    :return: aa_match - Binary list of c
    :rtype: list
    :return: 1 if all amino acid in two sequences match
    :rtype: int 
    """

    cum_aa_threshold = 0.5
    single_aa_threshold = 0.1

    pred_cum_aa_mass = 0
    orig_cum_aa_mass = 0

    aa_match = []
    pep_match = 0

    cnt_pred = 0
    cnt_orig = 0
    
    while cnt_pred < len(pred_seq) and cnt_orig < len(orig_seq):
        
        pred_aa_mass = aa_dict[pred_seq[cnt_pred]]
        orig_aa_mass = aa_dict[orig_seq[cnt_orig]]
        
        if abs(mass_diff(pred_cum_aa_mass,orig_cum_aa_mass, mode_is_da = True)) < cum_aa_threshold:

            if abs(mass_diff(pred_aa_mass, orig_aa_mass, mode_is_da = True)) < single_aa_threshold:
                aa_match += [1]
            else:
                aa_match += [0]

            cnt_pred += 1
            cnt_orig += 1

            pred_cum_aa_mass += pred_aa_mass
            orig_cum_aa_mass += orig_aa_mass
        
        elif mass_diff(pred_cum_aa_mass,orig_cum_aa_mass, mode_is_da = True) > 0:
            cnt_orig += 1
            orig_cum_aa_mass += orig_aa_mass
            
        else:
            cnt_pred += 1
            pred_cum_aa_mass += pred_aa_mass
            aa_match += [0]
            
    if sum(aa_match) == len(orig_seq) and len(pred_seq) == len(orig_seq):
        pep_match = 1

    return aa_match, pep_match


def match_aa(orig_seq, pred_seq, aa_dict, eval_direction = 'best'):
    """
    Find the matching amino acids of an original and predicted peptide 

    :param orig_seq: List of amino acids in the original peptide
    :type orig_seq: list
    :param pred_seq: List of amino acids in the predicted peptide
    :type pred_seq: list
    :param aa_dict: Dictionary of amino acid masses
    :type aa_dict: dict
    :param eval_direction: Direction of evaluation while finding amino acid matches, e.g. 'forward', 'backward', 'best'
    :type eval_direction: str {'forward', 'backward', 'best'}
    :return: aa_match - Binary list of c
    :rtype: list
    :return: 1 if all amino acid in two sequences match
    :rtype: int
    """
    
    if eval_direction == 'best':
        aa_match, pep_match = best_aa_match(orig_seq, pred_seq, aa_dict)
        n_mismatch_aa = len(pred_seq) - len(aa_match)
        aa_match += n_mismatch_aa * [0]
                
    
    elif eval_direction == 'forward':
        aa_match, pep_match = find_aa_match_single_pep(orig_seq,pred_seq, aa_dict)
        
        n_mismatch_aa = len(pred_seq) - len(aa_match)
        aa_match += n_mismatch_aa * [0]
        
    elif eval_direction == 'backward':
        reverse_aa_match, pep_match = find_aa_match_single_pep(list(reversed(orig_seq)),
                                                                          list(reversed(pred_seq)), aa_dict)
        
        aa_match = list(reversed(reverse_aa_match))
        n_mismatch_aa = len(pred_seq) - len(aa_match)
        aa_match = n_mismatch_aa * [0] + aa_match
        
        
    return aa_match, pep_match

def batch_aa_match(pred_pep_seqs, true_pep_seqs, aa_dict, eval_direction = 'best'):
    """
    Find the matching amino acids of an original and predicted peptide 

    :param pred_pep_seqs: List of predicted peptides, i.e. list of amino acid sequences
    :type pred_pep_seqs: list
    :param true_pep_seqs: List of ground truth peptide labels
    :type true_pep_seqs: list
    :param aa_dict: Dictionary of amino acid masses
    :type aa_dict: dict
    :param eval_direction: Direction of evaluation while finding amino acid matches, e.g. 'forward', 'backward', 'best'
    :type eval_direction: str {'forward', 'backward', 'best'}
    :return: all_aa_match - Binary list of lists corresponding to amino acid matches for all predicted peptides
    :rtype: list
    :return: orig_total_num_aa - Total number of amino acids in the ground truth peptide labels
    :rtype: int
    :return: pred_total_num_aa - Total number of amino acids in the predicted peptide labels
    :rtype: int
    """

    orig_total_num_aa = 0
    pred_total_num_aa = 0
    all_aa_match = []

    for pred_ind in range(len(pred_pep_seqs)):
        
        pred = re.split(r"(?<=.)(?=[A-Z])", pred_pep_seqs[pred_ind])
        orig = re.split(r"(?<=.)(?=[A-Z])", true_pep_seqs[pred_ind])
        orig_total_num_aa += len(orig)
        pred_total_num_aa += len(pred)

        aa_match, pep_match = match_aa(orig, pred, aa_dict, eval_direction=eval_direction)
        all_aa_match += [(aa_match, pep_match)]
    
    return all_aa_match, orig_total_num_aa, pred_total_num_aa
    
def calc_eval_metrics(aa_match_binary_list, orig_total_num_aa, pred_total_num_aa):
    """
    Calculate evaluation metrics using amino acid matches

    :param aa_match_binary_list: List of amino acid matches in each predicted peptide
    :type aa_match_binary_list: list of lists
    :param orig_total_num_aa: Number of amino acids in the original peptide sequences
    :type orig_total_num_aa: int
    :param pred_total_num_aa: Number of amino acids in the predicted peptide sequences
    :type pred_total_num_aa: int
    :return: aa_precision - Number of correct aa predictions divided by all predicted aa
    :rtype: float
    :return: aa_recall - Number of correct aa predictions divided by all original aa  
    :rtype: float:
    :return: pep_recall - Number of correct peptide predictions divided by all original peptide
    :rtype: float 
    """ 
    
    correct_aa_count = sum([sum(pred_tuple[0]) for pred_tuple in aa_match_binary_list])
    aa_recall = correct_aa_count/(orig_total_num_aa+1e-8)
    aa_precision = correct_aa_count/(pred_total_num_aa+1e-8)
    pep_recall = sum([pred_tuple[1] for pred_tuple in aa_match_binary_list])/(len(aa_match_binary_list)+1e-8)
    
    return aa_precision, aa_recall, pep_recall

def aa_precision_recall_with_threshold(correct_aa_confidences, all_aa_confidences, num_original_aa, threshold):
    """
    Calculate precision and recall for the given amino acid confidence score threshold

    :param correct_aa_confidences: List of confidence scores for correct amino acids predictions
    :type correct_aa_confidences: list
    :param all_aa_confidences: List of confidence scores for all amino acids prediction
    :type all_aa_confidences: int
    :param num_original_aa: Number of amino acids in the predicted peptide sequences
    :type num_original_aa: int
    :param threshold: Amino acid confidence score threshold
    :type threshold: float
    :return: aa_precision - Number of correct aa predictions divided by all predicted aa
    :rtype: float
    :return: aa_recall - Number of correct aa predictions divided by all original aa
    :rtype: float    
    """    
   
    correct_aa = sum([conf>=threshold for conf in correct_aa_confidences])
    predicted_aa = sum([conf>=threshold for conf in all_aa_confidences])
    
    aa_precision = correct_aa/predicted_aa
    aa_recall = correct_aa/num_original_aa
    
    return aa_precision, aa_recall