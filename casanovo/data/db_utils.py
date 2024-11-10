"""Unique methods used within db-search mode"""

import functools
import logging
import os
import re
import string
from collections import defaultdict
from typing import DefaultDict, Dict, Iterator, Pattern, Set, Tuple

import depthcharge.masses
import numba as nb
import numpy as np
import pandas as pd
import pyteomics.fasta
import pyteomics.parser


logger = logging.getLogger("casanovo")

# CONSTANTS
PROTON = 1.00727646677
ISOTOPE_SPACING = 1.003355


class ProteinDatabase:
    """
    Store digested FASTA data and return candidate peptides for a given
    precursor mass.

    Parameters
    ----------
    fasta_path : str
        Path to the FASTA file.
    enzyme : str
        The enzyme to use for digestion.
        See pyteomics.parser.expasy_rules for valid enzymes.
    digestion : str
        The type of digestion to perform.
        Either 'full', 'partial', or 'non-specific'.
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
        Isotope range [min, max] to consider when comparing predicted
        and observed precursor m/z's.
    allowed_fixed_mods : str
        A comma-separated string of fixed modifications to consider.
    allowed_var_mods : str
        A comma-separated string of variable modifications to consider.
    residues : Dict[str, float]
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
        residues: Dict[str, float],
    ):
        self.fixed_mods, self.var_mods, self.swap_map = _construct_mods_dict(
            allowed_fixed_mods, allowed_var_mods
        )
        self.max_mods = max_mods
        self.swap_regex = re.compile(
            "(%s)" % "|".join(map(re.escape, self.swap_map.keys()))
        )
        peptide_generator = _peptide_generator(
            fasta_path,
            enzyme,
            digestion,
            missed_cleavages,
            min_peptide_len,
            max_peptide_len,
            set([aa[0] for aa in residues.keys() if aa[0].isalpha()]),
        )
        self.db_peptides, self.prot_map = self._digest_fasta(peptide_generator)
        self.precursor_tolerance = precursor_tolerance
        self.isotope_error = isotope_error

    def _digest_fasta(
        self,
        peptide_generator: Iterator[Tuple[str, str]],
    ) -> Tuple[pd.DataFrame, DefaultDict[str, Set]]:
        """
        Digests a FASTA file and returns the peptides, their masses,
        and associated protein(s).

        Parameters
        ----------
        peptide_generator : Iterator[Tuple[str, str]]
            An iterator that yields peptides and associated proteins.

        Returns
        -------
        pep_table : pd.DataFrame
            A Pandas DataFrame with peptide and mass columns.
            Sorted by neutral mass in ascending order.
        prot_map : DefaultDict[str, Set]
            A dictionary mapping peptides to associated proteins.
        """
        # Generate all possible peptide isoforms.
        mass_calculator = depthcharge.masses.PeptideMass(residues="massivekb")
        peptide_isoforms = [
            (
                pyteomics.parser.isoforms(
                    pep,
                    variable_mods=self.var_mods,
                    fixed_mods=self.fixed_mods,
                    max_mods=self.max_mods,
                ),
                prot,
            )
            for pep, prot in peptide_generator
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

        # Create a dictionary mapping for easy accession of associated
        # proteins.
        prot_map: DefaultDict[str, Set] = defaultdict(set)
        for pep, _, prot in mod_peptide_list:
            prot_map[pep].add(prot)

        # Create a DataFrame for easy sorting and filtering.
        pep_table = pd.DataFrame(
            [(pep, mass) for pep, mass, _ in mod_peptide_list],
            columns=["peptide", "calc_mass"],
        )
        pep_table.sort_values(
            by=["calc_mass", "peptide"], ascending=True, inplace=True
        )

        logger.info(
            "Digestion complete. %d peptides generated.", len(pep_table)
        )
        return pep_table, prot_map

    def get_candidates(
        self,
        precursor_mz: float,
        charge: int,
    ) -> pd.Series:
        """
        Returns candidate peptides that fall within the search
        parameter's precursor mass tolerance.

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
        # FIXME: This could potentially be sped up with only a single pass
        #  through the database.
        mask = np.zeros(len(self.db_peptides), dtype=bool)
        precursor_tol_ppm = self.precursor_tolerance / 1e6
        for e in range(self.isotope_error[0], self.isotope_error[1] + 1):
            iso_shift = ISOTOPE_SPACING * e
            shift_raw_mass = float(
                _to_neutral_mass(precursor_mz, charge) - iso_shift
            )
            upper_bound = shift_raw_mass * (1 + precursor_tol_ppm)
            lower_bound = shift_raw_mass * (1 - precursor_tol_ppm)
            mask |= (
                (self.db_peptides["calc_mass"] >= lower_bound)
                & (self.db_peptides["calc_mass"] <= upper_bound)
            )
        return self.db_peptides[mask]["peptide"]

    def get_associated_protein(self, peptide: str) -> str:
        """
        Returns the associated protein(s) for a given peptide.

        Parameters
        ----------
        peptide : str
            The peptide sequence.

        Returns
        -------
        protein : str
            The associated protein(s) identifiers, separated by commas.
        """
        return ",".join(self.prot_map[peptide])


def _construct_mods_dict(
    allowed_fixed_mods: str, allowed_var_mods: str
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Constructs dictionaries of fixed and variable modifications.

    Parameters
    ----------
    allowed_fixed_mods : str
        A comma-separated string of fixed modifications to consider.
    allowed_var_mods : str
        A comma-separated string of variable modifications to consider.

    Returns
    -------
    fixed_mods : Dict[str, str]
        A dictionary of fixed modifications.
    var_mods : Dict[str, str]
        A dictionary of variable modifications.
    swap_map : Dict[str, str]
        A dictionary that allows for swapping of modX to
        Casanovo-acceptable modifications.
    """
    swap_map, fixed_mods, var_mods = {}, {}, {}
    for mod_map, allowed_mods in zip(
        [fixed_mods, var_mods], [allowed_fixed_mods, allowed_var_mods]
    ):
        for i, mod in enumerate(allowed_mods.split(",")):
            aa, mod_aa = mod.split(":")
            mod_id = string.ascii_lowercase[i]
            if aa == "nterm":
                mod_map[f"{mod_id}-"] = True
                swap_map[f"{mod_id}-"] = f"{mod_aa}"
            else:
                mod_map[mod_id] = [aa]
                swap_map[f"{mod_id}{aa}"] = f"{mod_aa}"

    return fixed_mods, var_mods, swap_map


def _peptide_generator(
    fasta_filename: str,
    enzyme: str,
    digestion: str,
    missed_cleavages: int,
    min_peptide_len: int,
    max_peptide_len: int,
    valid_aa: Set[str],
) -> Iterator[Tuple[str, str]]:
    """
    Creates a generator that yields peptides from a FASTA file depending
    on the type of digestion specified.

    Parameters
    ----------
    fasta_filename : str
        Path to the FASTA file.
    enzyme : str
        The enzyme to use for digestion.
        See pyteomics.parser.expasy_rules for valid enzymes.
        Can also be a regex.
    digestion : str
        The type of digestion to perform.
        Either 'full', 'partial', or 'non-specific'.
    missed_cleavages : int
        The number of missed cleavages to allow.
    min_peptide_len : int
        The minimum length of peptides to consider.
    max_peptide_len : int
        The maximum length of peptides to consider.
    valid_aa : Set[str]
        A set of valid amino acids.

    Yields
    ------
    peptide : str
        A peptide sequence, unmodified.
    protein : str
        The associated protein.
    """
    # Verify the existence of the file.
    if not os.path.isfile(fasta_filename):
        logger.error("File %s does not exist.", fasta_filename)
        raise FileNotFoundError(f"File {fasta_filename} does not exist.")
    if digestion not in ("full", "partial", "non-specific"):
        logger.error("Digestion type %s not recognized.", digestion)
        raise ValueError(f"Digestion type {digestion} not recognized.")
    if enzyme not in pyteomics.parser.expasy_rules:
        logger.info(
            "Enzyme %s not recognized. Interpreting as cleavage rule.",
            enzyme,
        )
    if digestion == "non-specific":
        for header, seq in pyteomics.fasta.read(fasta_filename):
            protein = header.split()[0]
            # Generate all possible peptides.
            for i in range(len(seq)):
                for j in range(
                    i + min_peptide_len,
                    min(i + max_peptide_len + 1, len(seq) + 1),
                ):
                    peptide = seq[i:j]
                    if any(aa not in valid_aa for aa in peptide):
                        logger.warning(
                            "Skipping peptide with unknown amino acids: %s",
                            peptide,
                        )
                    else:
                        yield peptide, protein
    else:
        for header, seq in pyteomics.fasta.read(fasta_filename):
            peptides = pyteomics.parser.cleave(
                seq,
                rule=enzyme,
                missed_cleavages=missed_cleavages,
                semi=digestion == "partial",
            )
            protein = header.split()[0]
            for peptide in peptides:
                if min_peptide_len <= len(peptide) <= max_peptide_len:
                    if any(aa not in valid_aa for aa in peptide):
                        logger.warning(
                            "Skipping peptide with unknown amino acids: %s",
                            peptide,
                        )
                    else:
                        yield peptide, protein


def _convert_from_modx(
    seq: str, swap_map: dict[str, str], swap_regex: Pattern
) -> str:
    """
    Converts peptide sequence from modX format to
    Casanovo-acceptable modifications.

    Parameters:
    -----------
    seq : str
        Peptide in modX format
    swap_map : dict[str, str]
        Dictionary that allows for swapping of modX to
        Casanovo-acceptable modifications.
    swap_regex : Pattern
        Regular expression to match modX format.

    Returns:
    --------
    str
        Peptide in Casanovo-acceptable modifications.
    """
    # FIXME: This might be handled by the DepthCharge residues vocabulary
    #  instead.
    return swap_regex.sub(lambda x: swap_map[x.group()], seq)


@nb.njit
def _to_neutral_mass(mz_mass: float, charge: int) -> float:
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
