import types
import unittest.mock

from casanovo.data import psm


def test_get_mod_string():
    # UNIMOD Oxidation on M (pos=4)
    mod = types.SimpleNamespace(
        position=3,
        source=[
            types.SimpleNamespace(name="Oxidation", accession="UNIMOD:35")
        ],
        mass=None,
    )
    s = psm.PepSpecMatch._get_mod_string(mod, "ACDMK")
    assert s == "4-Oxidation (M):UNIMOD:35"

    # Phospho on S (pos=7)
    mod = types.SimpleNamespace(
        position=6,
        source=[types.SimpleNamespace(name="Phospho", accession="MOD:00696")],
        mass=None,
    )
    s = psm.PepSpecMatch._get_mod_string(mod, "PEPTIDSX")
    assert s == "7-Phospho (S):MOD:00696"

    # Mass-only positive
    mod = types.SimpleNamespace(position=3, source=None, mass=15.994915)
    s = psm.PepSpecMatch._get_mod_string(mod, "ACDMK")
    assert s == "4-[+15.9949]"

    # Mass-only negative
    mod = types.SimpleNamespace(position=4, source=None, mass=-17.026549)
    s = psm.PepSpecMatch._get_mod_string(mod, "PEPTI")
    assert s == "5-[-17.0265]"


def test_proteoform_caching():
    mock_proteoform_1 = unittest.mock.MagicMock(sequence="ACDE")
    mock_proteoform_2 = unittest.mock.MagicMock(sequence="ACDEK")

    patch_path = f"{psm.PepSpecMatch.__module__}.spectrum_utils.proforma.parse"
    with unittest.mock.patch(patch_path) as mock_parse:
        mock_parse.side_effect = [[mock_proteoform_1], [mock_proteoform_2]]

        pep = psm.PepSpecMatch(
            sequence="ACDE",
            spectrum_id=("file", "idx"),
            peptide_score=42.0,
            charge=2,
            calc_mz=123.45,
            exp_mz=123.46,
            aa_scores=[0.1, 0.2, 0.3, 0.4],
        )

        # First access should trigger parse()
        p1 = pep._proteoform
        assert p1 is mock_proteoform_1
        mock_parse.assert_called_once_with("ACDE")

        # Second access with same sequence should use cache
        p2 = pep._proteoform
        assert p2 is p1
        mock_parse.assert_called_once()

        # Changed sequence should trigger new parse
        pep.sequence = "ACDEK"
        p3 = pep._proteoform
        assert p3 is mock_proteoform_2
        assert mock_parse.call_count == 2


def test_parse_sequence():
    # Default args for PepSpecMatch
    psm_args = [
        "spectrum_id",
        "peptide_score",
        "charge",
        "calc_mz",
        "exp_mz",
        "aa_scores",
    ]
    default_args = dict.fromkeys(psm_args)

    # No mod
    match = psm.PepSpecMatch(sequence="ACDMK", **default_args)
    assert match.aa_sequence == "ACDMK"
    assert match.modifications == "null"

    # Single internal mod
    match = psm.PepSpecMatch(sequence="ACDM[Oxidation]K", **default_args)
    assert match.aa_sequence == "ACDMK"
    assert match.modifications == "4-Oxidation (M):UNIMOD:35"

    # Multiple internal mods
    match = psm.PepSpecMatch(
        sequence="ACDM[Oxidation]KC[Carbamidomethyl]", **default_args
    )
    assert match.aa_sequence == "ACDMKC"
    assert (
        match.modifications
        == "4-Oxidation (M):UNIMOD:35; 6-Carbamidomethyl (C):UNIMOD:4"
    )

    # N-terminal mod
    match = psm.PepSpecMatch(sequence="[Acetyl]-PEPTIDE", **default_args)
    assert match.aa_sequence == "PEPTIDE"
    assert match.modifications == "0-Acetyl (N-term):UNIMOD:1"

    # N-terminal mod and multiple internal mods
    match = psm.PepSpecMatch(
        sequence="[Acetyl]-ACDM[Oxidation]KC[Carbamidomethyl]", **default_args
    )
    assert match.aa_sequence == "ACDMKC"
    assert match.modifications == (
        "0-Acetyl (N-term):UNIMOD:1; "
        "4-Oxidation (M):UNIMOD:35; "
        "6-Carbamidomethyl (C):UNIMOD:4"
    )
