"""Unique methods used within db-search mode"""

import functools
import logging
import os
import re
import string
from pathlib import Path
from typing import Dict, Iterator, Pattern, Set, Tuple

import depthcharge.constants
import depthcharge.tokenizers
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
    tokenizer: depthcharge.tokenizers.PeptideTokenizer
        Tokenizer to parse and process peptide sequences.
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
        tokenizer: depthcharge.tokenizers.PeptideTokenizer,
    ):
        self.fixed_mods, self.var_mods, self.swap_map = _construct_mods_dict(
            allowed_fixed_mods, allowed_var_mods
        )
        self.max_mods = max_mods
        self.swap_regex = re.compile(
            "(%s)" % "|".join(map(re.escape, self.swap_map.keys()))
        )
        aas = {r[0] for r in tokenizer.residues.keys() if r[0].isalpha()}
        if tokenizer.replace_isoleucine_with_leucine:
            aas.add("I")
        peptide_generator = _peptide_generator(
            fasta_path,
            enzyme,
            digestion,
            missed_cleavages,
            min_peptide_len,
            max_peptide_len,
            aas,
        )
        logger.info(
            "Digesting FASTA file (enzyme = %s, digestion = %s, missed "
            "cleavages = %d)...",
            enzyme,
            digestion,
            missed_cleavages,
        )
        self.tokenizer = tokenizer
        self.db_peptides = self._digest_fasta(peptide_generator)
        self.precursor_tolerance = precursor_tolerance
        self.isotope_error = isotope_error

    def _digest_fasta(
        self,
        peptide_generator: Iterator[Tuple[str, str]],
    ) -> pd.DataFrame:
        """
        Digests a FASTA file and returns the peptides, their masses,
        and associated protein(s).

        Parameters
        ----------
        peptide_generator : Iterator[Tuple[str, str]]
            An iterator that yields peptides and associated proteins.

        Returns
        -------
        peptides : pd.DataFrame
            A Pandas DataFrame with index "peptide" (the peptide
            sequence), and columns "calc_mass" (the peptide neutral
            mass) and "protein" (a list of associated protein(s)).
        """
        # Generate all possible peptide isoforms.
        peptides = pd.DataFrame(
            data=[
                (iso, prot)
                for pep, prot in peptide_generator
                for iso in pyteomics.parser.isoforms(
                    pep,
                    variable_mods=self.var_mods,
                    fixed_mods=self.fixed_mods,
                    max_mods=self.max_mods,
                )
            ],
            columns=["peptide", "protein"],
        )
        # Convert modX peptide to Casanovo format.
        peptides["peptide"] = peptides["peptide"].apply(
            functools.partial(
                _convert_from_modx,
                swap_map=self.swap_map,
                swap_regex=self.swap_regex,
            )
        )
        # Merge proteins from duplicate peptides.
        peptides = (
            peptides.groupby("peptide")["protein"]
            .apply(lambda proteins: sorted(set(proteins)))
            .reset_index()
        )
        # Calculate the mass of each peptide.
        peptides["calc_mass"] = (
            peptides["peptide"]
            .apply(self._calc_pep_mass)
            .astype(float)
            .round(5)
        )
        # Sort by peptide mass and index by peptide sequence.
        peptides.sort_values(
            by=["calc_mass", "peptide"], ascending=True, inplace=True
        )
        peptides.set_index("peptide", inplace=True)

        logger.info(
            "Digestion complete. %s peptides generated.", f"{len(peptides):,d}"
        )
        return peptides

    def export(self, output_path: Path, output_root: str) -> None:
        """
        Dumps the peptide database to a tsv file

        The file has the following columns:
        - protein (list of associated proteins to peptide)
        - peptide (peptide sequence)
        - calc_mass (calculated mass of the peptide)

        Parameters
        ----------
        output_path: Path
            Path that the tsv file will be stored
        output_root: str
           Name of the root directory of the file
        """
        self.db_peptides.to_csv(
            output_path / f"{output_root}.tsv",
            sep="\t",
            index=True,
        )

    def _calc_pep_mass(self, pep: str) -> float:
        """
        Calculates the neutral mass of a peptide sequence.

        Parameters
        ----------
        pep : str
            The peptide sequence for which the mass is to be calculated.

        Returns
        -------
        float
            The neutral mass of the peptide.
        """
        return (
            self.tokenizer.masses[self.tokenizer.tokenize(pep)]
            .sum(dim=1)
            .item()
            + depthcharge.constants.H2O
        )

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
            mask |= (self.db_peptides["calc_mass"] >= lower_bound) & (
                self.db_peptides["calc_mass"] <= upper_bound
            )
        return self.db_peptides.index[mask]

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
        return ",".join(self.db_peptides.loc[peptide, "protein"])


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
    n_skipped = 0
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
                        n_skipped += 1
                        logger.debug(
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
                        n_skipped += 1
                        logger.debug(
                            "Skipping peptide with unknown amino acids: %s",
                            peptide,
                        )
                    else:
                        yield peptide, protein
    if n_skipped > 0:
        logger.warning(
            "Skipped %d peptides with unknown amino acids", n_skipped
        )


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
