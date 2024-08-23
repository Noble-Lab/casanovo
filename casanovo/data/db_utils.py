"""Unique methods used within db-search mode"""

import logging
import os
from typing import List

import depthcharge.masses
from numba import jit
import pandas as pd
from pyteomics import fasta, parser

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
    isotope_error : List[int]
        Isotopes to consider when comparing predicted and observed precursor m/z's.
    allowed_mods : List[str]
        A list of allowed modifications to consider.
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
        isotope_error: List[int],
        allowed_mods: List[str],
    ):
        self.fixed_mods, self.var_mods = self._construct_mods_dict(
            allowed_mods
        )
        self.digest = self._digest_fasta(
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
    ):
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
        candidates : List[Tuple[str, str]]
            A list of candidate peptides and associated
            protein.
        """
        candidates = []

        for e in self.isotope_error:
            iso_shift = ISOTOPE_SPACING * e
            upper_bound = float(
                ProteinDatabase._to_raw_mass(precursor_mz, charge) - iso_shift
            ) * (1 + (self.precursor_tolerance / 1e6))
            lower_bound = float(
                ProteinDatabase._to_raw_mass(precursor_mz, charge) - iso_shift
            ) * (1 - (self.precursor_tolerance / 1e6))

            window = self.digest[
                (self.digest["calc_mass"] >= lower_bound)
                & (self.digest["calc_mass"] <= upper_bound)
            ]
            candidates.append(window[["peptide", "calc_mass", "protein"]])

        candidates = pd.concat(candidates)
        candidates.drop_duplicates(inplace=True)
        candidates.sort_values(by=["calc_mass", "peptide"], inplace=True)
        return list(candidates["peptide"]), list(candidates["protein"])

    def _digest_fasta(
        self,
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
        mod_peptide_list : pd.DataFrame
            A Pandas DataFrame with peptide, mass,
            and protein columns. Sorted by neutral mass in ascending order.
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
                if (
                    len(pep) < min_peptide_length
                    or len(pep) > max_peptide_length
                ):
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
                variable_mods=self.var_mods,
                fixed_mods=self.fixed_mods,
                max_mods=max_mods,
            )
            peptide_isoforms = list(
                map(ProteinDatabase._convert_from_modx, peptide_isoforms)
            )
            mod_peptide_list.extend(
                [mod_pep, mass_calculator.mass(mod_pep), prot]
                for mod_pep in peptide_isoforms
            )

        # Create a DataFrame for easy sorting and filtering
        pdb_df = pd.DataFrame(
            mod_peptide_list, columns=["peptide", "calc_mass", "protein"]
        )
        pdb_df.sort_values(by=["calc_mass", "peptide"], inplace=True)

        logger.info("Digestion complete. %d peptides generated.", len(pdb_df))
        return pdb_df

    def _construct_mods_dict(self, allowed_mods):
        """
        Constructs dictionaries of fixed and variable modifications.

        Parameters
        ----------
        allowed_mods : str
            A comma-separated list of allowed modifications.

        Returns
        -------
        fixed_mods : dict
            A dictionary of fixed modifications.
        var_mods : dict
            A dictionary of variable modifications.
        """
        fixed_mods = {"carbm": ["C"]}
        var_mods = {}

        if allowed_mods is "" or None:
            return fixed_mods, var_mods
        for mod in allowed_mods.split(","):
            if mod == "M+15.995":
                if "ox" not in var_mods:
                    var_mods["ox"] = []
                var_mods["ox"].append("M")
            elif mod == "N+0.984":
                if "d" not in var_mods:
                    var_mods["d"] = []
                var_mods["d"].append("N")
            elif mod == "Q+0.984":
                if "d" not in var_mods:
                    var_mods["d"] = []
                var_mods["d"].append("Q")
            elif mod == "+42.011":
                var_mods["ace-"] = True
            elif mod == "+43.006":
                var_mods["carb-"] = True
            elif mod == "-17.027":
                var_mods["nh3x-"] = True
            elif mod == "+43.006-17.027":
                var_mods["carbnh3x-"] = True
            else:
                logger.error("Modification %s not recognized.", mod)
                raise ValueError(f"Modification {mod} not recognized.")

        return fixed_mods, var_mods

    @jit
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

    @jit
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

    def _convert_from_modx(seq: str):
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
