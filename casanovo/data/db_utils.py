"""Unique methods used within db-search mode"""

import bisect
import logging
import os
from typing import List, Tuple

import depthcharge.masses
from pyteomics import fasta, parser

logger = logging.getLogger("casanovo")


# CONSTANTS
HYDROGEN = 1.007825035
OXYGEN = 15.99491463
H2O = 2 * HYDROGEN + OXYGEN
PROTON = 1.00727646677
ISOTOPE_SPACING = 1.003355

var_mods = {
    "d": ["N", "Q"],
    "ox": ["M"],
    "ace-": True,
    "carb-": True,
    "nh3x-": True,
    "carbnh3x-": True,
}
fixed_mods = {"carbm": ["C"]}


def convert_from_modx(seq: str):
    """Converts peptide sequence from modX format to Casanovo-acceptable modifications.

    Args:
        seq (str): Peptide in modX format
    """
    seq = seq.replace("carbmC", "C+57.021")  # Fixed modification
    seq = seq.replace("oxM", "M+15.995")
    seq = seq.replace("dN", "N+0.984")
    seq = seq.replace("dQ", "Q+0.984")
    seq = seq.replace("ace-", "+42.011")
    seq = seq.replace("carbnh3x-", "+43.006-17.027")
    seq = seq.replace("carb-", "+43.006")
    seq = seq.replace("nh3x-", "-17.027")
    return seq


def digest_fasta(
    fasta_filename: str,
    enzyme: str,
    digestion: str,
    missed_cleavages: int,
    max_mods: int,
    min_peptide_length: int,
    max_peptide_length: int,
):
    """
    Digests a FASTA file and returns the peptides, their masses, and associated protein.

    Parameters
    ----------
    fasta_filename : str
        Path to the FASTA file.
    enzyme : str
        The enzyme to use for digestion.
        See pyteomics.parser.expasy_rules for valid enzymes.
    digestion : str
        The type of digestion to perform. Either 'full' or 'partial'.
    missed_cleavages : int
        The number of missed cleavages to allow.
    max_mods : int
        The maximum number of modifications to allow per peptide.
    min_peptide_length : int
        The minimum length of peptides to consider.
    max_peptide_length : int
        The maximum length of peptides to consider.

    Returns
    -------
    mod_peptide_list : List[Tuple[str, float, str]]
        A list of tuples containing the peptide sequence, mass,
        and associated protein. Sorted by neutral mass in ascending order.
    """
    # Verify the existence of the file:
    if not os.path.isfile(fasta_filename):
        logger.error("File %s does not exist.", fasta_filename)
        raise FileNotFoundError(f"File {fasta_filename} does not exist.")

    fasta_data = fasta.read(fasta_filename)
    peptide_list = []
    if digestion not in ["full", "partial"]:
        logger.error("Digestion type %s not recognized.", digestion)
        raise ValueError(f"Digestion type {digestion} not recognized.")
    semi = digestion == "partial"
    for header, seq in fasta_data:
        pep_set = parser.cleave(
            seq,
            rule=parser.expasy_rules[enzyme],
            missed_cleavages=missed_cleavages,
            semi=semi,
        )
        protein = header.split()[0]
        for pep in pep_set:
            if len(pep) < min_peptide_length or len(pep) > max_peptide_length:
                continue
            if any(
                aa in pep for aa in "BJOUXZ"
            ):  # Check for incorrect AA letters
                logger.warn(
                    "Skipping peptide with ambiguous amino acids: %s", pep
                )
                continue
            peptide_list.append((pep, protein))

    # Generate modified peptides
    mass_calculator = depthcharge.masses.PeptideMass(residues="massivekb")
    mod_peptide_list = []
    for pep, prot in peptide_list:
        peptide_isoforms = parser.isoforms(
            pep,
            variable_mods=var_mods,
            fixed_mods=fixed_mods,
            max_mods=max_mods,
        )
        peptide_isoforms = list(map(convert_from_modx, peptide_isoforms))
        mod_peptide_list.extend(
            (mod_pep, mass_calculator.mass(mod_pep), prot)
            for mod_pep in peptide_isoforms
        )

    # Sort the peptides by mass and return.
    mod_peptide_list.sort(key=lambda x: x[1])
    return mod_peptide_list


def get_candidates(
    precursor_mz: float,
    charge: int,
    peptide_list: List[Tuple[str, float, str]],
    precursor_tolerance: float,
    isotope_error: str,
):
    """
    Returns a list of candidate peptides that fall within the specified mass range.

    Parameters
    ----------
    precursor_mz : float
        The precursor mass-to-charge ratio.
    charge : int
        The precursor charge.
    peptide_list : List[Tuple[str, float, str]]
        A list of tuples containing the peptide sequence, mass, and associated protein.
        Must be sorted by mass in ascending order. Uses neutral masses.
    precursor_tolerance : float
        The precursor mass tolerance in parts-per-million.
    isotope_error : str
        The isotope error levels to consider.
    """
    candidates = set()

    isotope_error = [int(x) for x in isotope_error.split(",")]
    for e in isotope_error:
        iso_shift = ISOTOPE_SPACING * e
        upper_bound = (_to_raw_mass(precursor_mz, charge) - iso_shift) * (
            1 + (precursor_tolerance / 1e6)
        )
        lower_bound = (_to_raw_mass(precursor_mz, charge) - iso_shift) * (
            1 - (precursor_tolerance / 1e6)
        )

        start, end = get_mass_indices(
            [x[1] for x in peptide_list], lower_bound, upper_bound
        )

        candidates.update(peptide_list[start:end])

    candidates = list(candidates)
    candidates.sort(key=lambda x: x[1])
    return candidates


def _to_mz(precursor_mass, charge):
    """
    Convert precursor neutral mass to m/z value.

    Parameters
    ----------
    precursor_mass : float
        The precursor neutral mass.
    charge : int
        The precursor charge.

    Returns
    -------
    mz : float
        The calculated precursor mass-to-charge ratio.
    """
    return (precursor_mass + (charge * PROTON)) / charge


def _to_raw_mass(mz_mass, charge):
    """
    Convert precursor m/z value to neutral mass.

    Parameters
    ----------
    mz_mass : float
        The precursor mass-to-charge ratio.
    charge : int
        The precursor charge.

    Returns
    -------
    mass : float
        The calculated precursor neutral mass.
    """
    return charge * (mz_mass - PROTON)


def get_mass_indices(masses, m_low, m_high):
    """Grabs mass indices that fall within a specified range.

    Pulls from masses, a list of mass values.
    Requires that the mass values are sorted in ascending order.

    Parameters
    ----------
    masses : List[int]
        List of mass values
    m_low : int
        Lower bound of mass range (inclusive)
    m_high : int
        Upper bound of mass range (inclusive)

    Return
    ------
    indices : Tuple[int, int]
        Indices of mass values that fall within the specified range
    """
    start = bisect.bisect_left(masses, m_low)
    end = bisect.bisect_right(masses, m_high)
    return start, end
