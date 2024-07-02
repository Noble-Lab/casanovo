"""Unique methods used within db-search mode"""

import os
import depthcharge.masses
from pyteomics import fasta, parser
import bisect

HYDROGEN = 1.007825035
OXYGEN = 15.99491463
H2O = 2 * HYDROGEN + OXYGEN
PROTON = 1.00727646677
ISOTOPE_SPACING = 1.003355  # - 0.00288

var_mods = {
    "d": ["N", "Q"],
    "ox": ["M"],
    "ace-": True,
    "carb-": True,
    "nh3x-": True,
    "carbnh3x-": True,
}
fixed_mods = {"carbm": ["C"]}


def convert_from_modx(seq):
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
    fasta_filename,
    enzyme,
    digestion,
    missed_cleavages,
    max_mods,
    min_length,
    max_length,
):
    """TODO: Add docstring"""

    # Verify the eistence of the file:
    if not os.path.isfile(fasta_filename):
        print(f"File {fasta_filename} does not exist.")
        raise FileNotFoundError(f"File {fasta_filename} does not exist.")

    fasta_data = fasta.read(fasta_filename)
    peptide_list = []
    if digestion in ["full", "partial"]:
        semi = True if digestion == "partial" else False
        for header, seq in fasta_data:
            pep_set = parser.cleave(
                seq,
                rule=parser.expasy_rules[enzyme],
                missed_cleavages=missed_cleavages,
                semi=semi,
            )
            protein = header.split()[0]
            peptide_list.extend([(pep, protein) for pep in pep_set])
    else:
        raise ValueError(f"Digestion type {digestion} not recognized.")

    # Generate modified peptides
    mass_calculator = depthcharge.masses.PeptideMass(residues="massivekb")
    mass_calculator.masses.update({"X": 0.0})  # TODO: REMOVE?
    mod_peptide_list = []
    for pep, prot in peptide_list:
        if len(pep) < min_length or len(pep) > max_length:
            continue
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
    precursor_mass, charge, peptide_list, precursor_tolerance, isotope_error
):
    """TODO: ADD DOCSTRING"""

    candidates = set()

    isotope_error = [int(x) for x in isotope_error.split(",")]
    for e in isotope_error:
        iso_shift = ISOTOPE_SPACING * e
        upper_bound = (_to_raw_mass(precursor_mass, charge) - iso_shift) * (
            1 + (precursor_tolerance / 1e6)
        )
        lower_bound = (_to_raw_mass(precursor_mass, charge) - iso_shift) * (
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
    """TODO: ADD DOCSTRING"""
    return (precursor_mass + (charge * PROTON)) / charge


def _to_raw_mass(mz_mass, charge):
    """TODO: ADD DOCSTRING"""
    return charge * (mz_mass - PROTON)


def get_mass_indices(masses, m_low, m_high):
    """Grabs mass indices from a list of mass values that fall within a specified range.
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
