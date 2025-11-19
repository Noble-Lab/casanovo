"""Peptide spectrum match dataclass."""

import dataclasses
from typing import Iterable, Optional, Tuple

import depthcharge.tokenizers
import spectrum_utils.proforma


class PeptideParser:
    def __init__(self, tokenizer: depthcharge.tokenizers.PeptideTokenizer):
        """
        A parser for proforma peptide sequences

        Parameters
        ----------
        tokenizer : depthcharge.tokenizers.PeptideTokenizer
            A tokenizer whose residue tokens (including modified residues and
            terminal tokens) will be mapped to their amino acid and modification
            representations.
        """
        self.tokenizer: depthcharge.tokenizers.PeptideTokenizer = tokenizer
        self.sequences: dict[str, str] = {}
        self.modifications: dict[
            str, list[spectrum_utils.proforma.Modification]
        ] = {}
        n_term = []

        for curr in self.tokenizer.residues.keys():
            if curr.startswith("["):
                n_term.append(curr)
                continue

            proteoform = spectrum_utils.proforma.parse(curr)[0]
            self.sequences[curr] = proteoform.sequence
            self.modifications[curr] = proteoform.modifications or []

        # Random residue to get spectrum_utils to parse the sequences
        reference_residue = next(iter(self.sequences.keys()))
        for curr in n_term:
            proteoform = spectrum_utils.proforma.parse(
                f"{curr}{reference_residue}"
            )[0]
            self.sequences[curr] = ""
            self.modifications[curr] = proteoform.modifications

    @staticmethod
    def _get_mod_string(
        mod: spectrum_utils.proforma.Modification, residue: str, pos: int
    ) -> str:
        """
        Format a ProForma modification into an mzTab-style string.

        Parameters
        ----------
        mod : spectrum_utils.proforma.Modification
            A modification object from the parsed ProForma sequence.
        residue : str
            The residue the modification is on.
        pos : int
            The position of the modification in the peptide sequence. Should be
            0 for n-terminal modifications, and 1-based for amino acids

        Returns
        -------
        str
            The mzTab-formatted modification string.
        """
        for src in mod.source or []:
            if hasattr(src, "accession"):
                return f"{pos}-{src.name} ({residue}):{src.accession}"

        return f"{pos}-[{mod.mass:+.4f}]"

    def parse(self, sequence: str) -> Tuple[str, str]:
        """
        Parse a peptide sequence into its unmodified amino acid sequence
        and list of modifications.

        Parameters
        ----------
        sequence : str
            The peptide sequence in ProForma notation.

        Returns
        -------
        Tuple[str, str]
            A tuple containing the amino acid sequence (no modifications) and
            the mzTab-formatted modification string.
        """
        tokens = self.tokenizer.split(sequence)
        if self.tokenizer.reverse:
            tokens = tokens[::-1]

        starts_at_nterm = sequence.startswith("[")
        start_pos = 0 if starts_at_nterm else 1
        aa_sequence = []
        mztab_mods = []

        for pos, token in enumerate(tokens, start=start_pos):
            amino_acid = self.sequences[token]
            aa_sequence.append(amino_acid)
            residue_for_mod = "N-term" if pos == 0 else amino_acid

            for mod in self.modifications[token]:
                mod_str = self._get_mod_string(mod, residue_for_mod, pos)
                mztab_mods.append(mod_str)

        peptide = "".join(aa_sequence)
        mod_string = "; ".join(mztab_mods) if mztab_mods else "null"
        return peptide, mod_string


@dataclasses.dataclass
class PepSpecMatch:
    """
    Peptide Spectrum Match (PSM) dataclass

    Parameters
    ----------
    sequence : str
        The amino acid sequence of the peptide.
    spectrum_id : Tuple[str, str]
        A tuple containing the spectrum identifier in the form
        (spectrum file name, spectrum file idx).
    peptide_score : float
        Score of the match between the full peptide sequence and the
        spectrum.
    charge : int
        The precursor charge state of the peptide ion observed in the
        spectrum.
    calc_mz : float
        The calculated mass-to-charge ratio (m/z) of the peptide based
        on its sequence and charge state.
    exp_mz : float
        The observed (experimental) precursor mass-to-charge ratio (m/z)
        of the peptide as detected in the spectrum.
    aa_scores : Iterable[float]
        A list of scores for individual amino acids in the peptide
        sequence, where len(aa_scores) == len(sequence).
    protein : str
        Protein associated with the peptide sequence (for db mode).
    """

    sequence: str
    spectrum_id: Tuple[str, str]
    peptide_score: float
    charge: int
    calc_mz: float
    exp_mz: float
    aa_scores: Iterable[float]
    protein: str = "null"
    peptide_parser: Optional[PeptideParser] = None

    # Private properties to handle parsing
    _cached_sequence: Optional[str] = dataclasses.field(
        init=False, default=None, repr=False, compare=False
    )
    _cached_aa_sequence: Optional[str] = dataclasses.field(
        init=False, default=None, repr=False, compare=False
    )
    _cached_modifications: Optional[str] = dataclasses.field(
        init=False, default=None, repr=False, compare=False
    )

    def _parse(self) -> None:
        """
        Lazily parse the peptide sequence using the attached `PeptideParser`.

        If the current `sequence` differs from the last cached value (or if no
        cached value exists), the method re-parses the peptide.
        """
        if self.peptide_parser is None:
            raise ValueError(
                "peptide_parser is required to get aa_sequence or modifications."
            )

        if (
            self._cached_sequence is None
            or self._cached_sequence != self.sequence
        ):
            aa_seq, mods = self.peptide_parser.parse(self.sequence)
            self._cached_sequence = self.sequence
            self._cached_aa_sequence = aa_seq
            self._cached_modifications = mods

    @property
    def aa_sequence(self) -> str:
        """
        Plain amino-acid sequence with all ProForma modifications removed.

        Returns
        -------
        str
            The peptide sequence stripped of modification annotations.
        """
        self._parse()
        return self._cached_aa_sequence

    @property
    def modifications(self) -> str:
        """
        mzTab-formatted modification string.

        Returns
        -------
        str
            A semicolon-delimited list of modifications (e.g.,
            ``"0-Acetyl (N-term):MOD:00000; 5-[+15.9949]"``).
            Returns ``"null"`` if the peptide has no modifications.
        """
        self._parse()
        return self._cached_modifications
