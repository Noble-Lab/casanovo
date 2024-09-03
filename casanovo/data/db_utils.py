"""Unique methods used within db-search mode"""

import functools
import logging
import os
import re
import string
from collections import defaultdict
from typing import List, Tuple

import depthcharge.masses
import pandas as pd
import pyteomics.fasta as fasta
import pyteomics.parser as parser
from numba import njit


logger = logging.getLogger("casanovo")

# CONSTANTS
PROTON = 1.00727646677
ISOTOPE_SPACING = 1.003355


class ProteinDatabase:
    """
    Store digested .fasta data and return candidate peptides for a given precursor mass.

    Parameters
    ----------
    fasta_path : str
        Path to the FASTA file.
    enzyme : str
        The enzyme to use for digestion.
        See pyteomics.parser.expasy_rules for valid enzymes.
    digestion : str
        The type of digestion to perform. Either 'full' or 'partial'.
    missed_cleavages : int
        The number of missed cleavages to allow.
    min_peptide_len : int
        The minimum length of peptides to consider.
    max_peptide_len : int
        The maximum length of peptides to consider.
    max_mods : int
        The maximum number of modifications to allow per peptide.
    precursor_tolerance : float
        The precursor mass tolerance in ppm.
    isotope_error : Tuple[int, int]
        Isotope range [min, max] to consider when comparing predicted and observed precursor m/z's.
    allowed_fixed_mods : str
        A comma separated string of fixed modifications to consider.
    allowed_var_mods : str
        A comma separated string of variable modifications to consider.
    residues : dict
        A dictionary of amino acid masses.
    """

    def __init__(
        self,
        fasta_path: str,
        enzyme: str,
        digestion: str,
        missed_cleavages: int,
        min_peptide_len: int,
        max_peptide_len: int,
        max_mods: int,
        precursor_tolerance: float,
        isotope_error: Tuple[int, int],
        allowed_fixed_mods: str,
        allowed_var_mods: str,
        residues: dict,
    ):
        self.residues = residues
        self.fixed_mods, self.var_mods, self.swap_map = _construct_mods_dict(
            allowed_fixed_mods, allowed_var_mods
        )
        self.swap_regex = re.compile(
            "(%s)" % "|".join(map(re.escape, self.swap_map.keys()))
        )
        self.db_peptides, self.prot_map = self._digest_fasta(
            fasta_path,
            enzyme,
            digestion,
            missed_cleavages,
            max_mods,
            min_peptide_len,
            max_peptide_len,
        )
        self.precursor_tolerance = precursor_tolerance
        self.isotope_error = isotope_error

    def get_candidates(
        self,
        precursor_mz: float,
        charge: int,
    ) -> List[Tuple[str, str]]:
        """
        Returns a list of candidate peptides that fall within the specified mass range.

        Parameters
        ----------
        precursor_mz : float
            The precursor mass-to-charge ratio.
        charge : int
            The precursor charge.

        Returns
        -------
        candidates : pd.Series
            A series of candidate peptides.
        """
        candidates = []

        for e in range(self.isotope_error[0], self.isotope_error[1] + 1):
            iso_shift = ISOTOPE_SPACING * e
            shift_raw_mass = float(
                _to_raw_mass(precursor_mz, charge) - iso_shift
            )
            upper_bound = shift_raw_mass * (
                1 + (self.precursor_tolerance / 1e6)
            )
            lower_bound = shift_raw_mass * (
                1 - (self.precursor_tolerance / 1e6)
            )

            window = self.db_peptides[
                (self.db_peptides["calc_mass"] >= lower_bound)
                & (self.db_peptides["calc_mass"] <= upper_bound)
            ]
            candidates.append(window[["peptide", "calc_mass", "protein"]])

        candidates = pd.concat(candidates)
        candidates.drop_duplicates(inplace=True)
        candidates.sort_values(by=["calc_mass", "peptide"], inplace=True)
        return candidates["peptide"], candidates["protein"]

    def get_associated_protein(self, peptide: str) -> str:
        """
        Returns the associated protein for a given peptide.

        Parameters
        ----------
        peptide : str
            The peptide sequence.

        Returns
        -------
        protein : str
            The associated protein(s).
        """
        return ",".join(self.prot_map[peptide])

    def _digest_fasta(
        self,
        fasta_filename: str,
        enzyme: str,
        digestion: str,
        missed_cleavages: int,
        max_mods: int,
        min_peptide_length: int,
        max_peptide_length: int,
    ) -> pd.DataFrame:
        """
        Digests a FASTA file and returns the peptides, their masses, and associated protein.

        Parameters
        ----------
        fasta_filename : str
            Path to the FASTA file.
        enzyme : str
            The enzyme to use for digestion.
            See pyteomics.parser.expasy_rules for valid enzymes.
            Can also be a regex pattern.
        digestion : str
            The type of digestion to perform. Either 'full', 'partial' or 'non-specific'.
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
        pep_table : pd.DataFrame
            A Pandas DataFrame with peptide, mass,
            and protein columns. Sorted by neutral mass in ascending order.
        prot_map : dict
            A dictionary mapping peptides to associated proteins.
        """
        # Verify the existence of the file:
        if not os.path.isfile(fasta_filename):
            logger.error("File %s does not exist.", fasta_filename)
            raise FileNotFoundError(f"File {fasta_filename} does not exist.")

        peptide_list = []
        if digestion not in ["full", "partial", "non-specific"]:
            logger.error("Digestion type %s not recognized.", digestion)
            raise ValueError(f"Digestion type {digestion} not recognized.")
        if enzyme not in parser.expasy_rules:
            logger.info(
                "Enzyme %s not recognized. Interpreting as cleavage rule.",
                enzyme,
            )
        valid_aa = set(list(self.residues.keys()) + ["C"])
        if digestion == "non-specific":
            for header, seq in fasta.read(fasta_filename):
                pep_set = []
                # Generate all possible peptides
                for i in range(len(seq)):
                    for j in range(i + 1, len(seq) + 1):
                        pep_set.append(seq[i:j])
                protein = header.split()[0]
                for pep in pep_set:
                    if (
                        len(pep) >= min_peptide_length
                        and len(pep) <= max_peptide_length
                    ):
                        if any(aa not in valid_aa for aa in pep):
                            logger.warn(
                                "Skipping peptide with unknown amino acids: %s",
                                pep,
                            )
                        else:
                            peptide_list.append((pep, protein))
        else:
            semi = digestion == "partial"
            for header, seq in fasta.read(fasta_filename):
                pep_set = parser.cleave(
                    seq,
                    rule=enzyme,
                    missed_cleavages=missed_cleavages,
                    semi=semi,
                )
                protein = header.split()[0]
                for pep in pep_set:
                    if (
                        len(pep) >= min_peptide_length
                        and len(pep) <= max_peptide_length
                    ):
                        if any(aa not in valid_aa for aa in pep):
                            logger.warn(
                                "Skipping peptide with unknown amino acids: %s",
                                pep,
                            )
                        else:
                            peptide_list.append((pep, protein))

        # Generate modified peptides
        mass_calculator = depthcharge.masses.PeptideMass(residues="massivekb")
        peptide_isoforms = [
            (
                parser.isoforms(
                    pep,
                    variable_mods=self.var_mods,
                    fixed_mods=self.fixed_mods,
                    max_mods=max_mods,
                ),
                prot,
            )
            for pep, prot in peptide_list
        ]
        mod_peptide_list = [
            (mod_pep, mass_calculator.mass(mod_pep), prot)
            for isos, prot in peptide_isoforms
            for mod_pep in map(
                functools.partial(
                    _convert_from_modx,
                    swap_map=self.swap_map,
                    swap_regex=self.swap_regex,
                ),
                isos,
            )
        ]
        # Create a DataFrame for easy sorting and filtering
        pep_table = pd.DataFrame(
            mod_peptide_list, columns=["peptide", "calc_mass", "protein"]
        )
        pep_table.sort_values(by=["calc_mass", "peptide"], inplace=True)

        # Create a dictionary mapping for easy accession of associated proteins
        prot_map = defaultdict(list)
        for pep, _, prot in mod_peptide_list:
            prot_map[pep].append(prot)

        logger.info(
            "Digestion complete. %d peptides generated.", len(pep_table)
        )
        return pep_table, prot_map


@njit
def _to_mz(precursor_mass: float, charge: int) -> float:
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


@njit
def _to_raw_mass(mz_mass: float, charge: int) -> float:
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


def _convert_from_modx(seq: str, swap_map: dict, swap_regex: str) -> str:
    """Converts peptide sequence from modX format to Casanovo-acceptable modifications.

    Args:
        seq : str
            Peptide in modX format
        swap_map : dict
            Dictionary that allows for swapping of modX to Casanovo-acceptable modifications.
        swap_regex : str
            Regular expression to match modX format.
    """
    return swap_regex.sub(lambda x: swap_map[x.group()], seq)


def _construct_mods_dict(
    allowed_fixed_mods: str, allowed_var_mods: str
) -> Tuple[dict, dict, dict]:
    """
    Constructs dictionaries of fixed and variable modifications.

    Parameters
    ----------
    allowed_fixed_mods : str
        A comma separated string of fixed modifications to consider.
    allowed_var_mods : str
        A comma separated string of variable modifications to consider.

    Returns
    -------
    fixed_mods : dict
        A dictionary of fixed modifications.
    var_mods : dict
        A dictionary of variable modifications.
    swap_map : dict
        A dictionary that allows for swapping of modX to Casanovo-acceptable modifications.
    """
    swap_map = {}
    fixed_mods = {}
    for idx, mod in enumerate(allowed_fixed_mods.split(",")):
        aa, mod_aa = mod.split(":")
        mod_id = string.ascii_lowercase[idx]
        fixed_mods[mod_id] = [aa]
        swap_map[f"{mod_id}{aa}"] = f"{mod_aa}"

    var_mods = {}
    for idx, mod in enumerate(allowed_var_mods.split(",")):
        aa, mod_aa = mod.split(":")
        mod_id = string.ascii_lowercase[idx]
        if aa == "nterm":
            var_mods[f"{mod_id}-"] = True
            swap_map[f"{mod_id}-"] = f"{mod_aa}"
        else:
            var_mods[mod_id] = [aa]
            swap_map[f"{mod_id}{aa}"] = f"{mod_aa}"

    return fixed_mods, var_mods, swap_map
