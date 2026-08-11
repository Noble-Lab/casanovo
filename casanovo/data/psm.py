"""Peptide spectrum match dataclass."""

import dataclasses
from typing import Iterable, Optional, Tuple

import spectrum_utils.proforma


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

    # Private properties to handle proteoform caching
    _proteoform_sequence: Optional[str] = dataclasses.field(
        init=False, default=None
    )
    _cache_proteoform: Optional[spectrum_utils.proforma.Proteoform] = (
        dataclasses.field(init=False, default=None, repr=False, compare=False)
    )

    @staticmethod
    def _get_mod_string(
        mod: spectrum_utils.proforma.Modification,
        aa_seq: str,
    ) -> str:
        """
        Format a ProForma modification into an mzTab-style string.

        Parameters
        ----------
        mod : spectrum_utils.proforma.Modification
            A modification object from the parsed ProForma sequence.
        aa_seq : str
            The unmodified amino acid sequence with modifications stripped

        Returns
        -------
        str
            The mzTab-formatted modification string.
        """
        if mod.position == "N-term":
            pos = 0
            residue = "N-term"
        else:
            pos = mod.position + 1
            residue = aa_seq[mod.position]

        for src in mod.source or []:
            if hasattr(src, "accession"):
                return f"{pos}-{src.name} ({residue}):{src.accession}"

        return f"{pos}-[{mod.mass:+.4f}]"

    @property
    def _proteoform(self) -> spectrum_utils.proforma.Proteoform:
        """
        Parsed ProForma representation of the peptide sequence.

        Returns
        -------
        spectrum_utils.proforma.Proteoform
            The parsed ProForma object representing the current peptide sequence
        """
        if self._proteoform_sequence != self.sequence:
            self._proteoform_sequence = self.sequence
            self._cache_proteoform = spectrum_utils.proforma.parse(
                self.sequence
            )[0]

        return self._cache_proteoform

    @property
    def aa_sequence(self) -> str:
        """
        Plain amino-acid sequence with all ProForma modifications removed.

        Returns
        -------
        str
            The peptide sequence stripped of modification annotations.
        """
        return self._proteoform.sequence

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
        mods = self._proteoform.modifications or []
        if not mods:
            return "null"

        mod_strings = [
            self._get_mod_string(mod, self.aa_sequence) for mod in mods
        ]

        return "; ".join(mod_strings)
