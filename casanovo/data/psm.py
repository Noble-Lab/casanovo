"""Peptide spectrum match dataclass."""

import dataclasses
from typing import Iterable, Tuple


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
