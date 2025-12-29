"""Test forward function of Spec2PepTargetDecoy() model"""

import numpy as np
import torch
from unittest.mock import MagicMock, patch

from casanovo.denovo.model import Spec2PepTargetDecoy, _annotate_fragment_pairs
from casanovo.data import psm
import spectrum_utils.proforma as spf
import spectrum_utils.spectrum as sus


def make_mock_model(
    name: str,
    aa: str,
    aa_score: list[float],
    include_nterm: bool = False,
    nterm_token: int = 26,
):
    """
    Mock Spec2Pep-like model that predicts from C-terminus to N-terminus (right to left).

    Parameters
    ----------
    aa : str
        Amino acid sequence in N->C order (e.g., "PEP")
    include_nterm : bool
        If True, append N-term token at the END of prediction sequence
    """
    mock_model = MagicMock(name=name)

    residues = {"P": 97.0, "E": 129.0, "D": 115.0}
    aa_to_idx = {"P": 0, "E": 1, "D": 2}
    idx_to_aa = {0: "P", 1: "E", 2: "D", 26: "[Acetyl]"}
    STOP_TOKEN = 3

    # Tokenizer.
    mock_model.tokenizer = MagicMock()
    mock_model.tokenizer.residues = residues
    mock_model.tokenizer.reverse = (
        False  # This should actually be True for right-to-left!
    )

    def _tokenize(seq):
        tokens = []
        if include_nterm and seq.startswith("[Acetyl]-"):
            tokens.append(nterm_token)
            seq = seq[9:]
        for a in seq:
            tokens.append(aa_to_idx[a])
        return torch.tensor(tokens, dtype=torch.int64)

    mock_model.tokenizer.tokenize = MagicMock(side_effect=_tokenize)

    def _detok(tokens, join=True):
        chars = []
        has_nterm = False
        for t in tokens[0]:
            tid = int(t.item())
            if tid == STOP_TOKEN:
                continue
            if tid == nterm_token:
                has_nterm = True
                continue
            chars.append(idx_to_aa.get(tid, "?"))

        result = "".join(chars)
        if has_nterm:
            result = "[Acetyl]-" + result

        if join:
            return [result]
        else:
            return [[c for c in result]]

    mock_model.tokenizer.detokenize = MagicMock(side_effect=_detok)

    # Encoder
    def mock_encoder(mz_array: torch.Tensor, intensity_array: torch.Tensor):
        mean_mz = torch.mean(mz_array).item()
        mean_int = torch.mean(intensity_array).item()
        memory = torch.full((1, 10, 64), fill_value=mean_mz + mean_int)
        mask = torch.ones((1, 10), dtype=torch.bool)
        return memory, mask

    mock_model.encoder.side_effect = mock_encoder

    # Decoder - predicts from C-terminus to N-terminus
    def mock_decoder(
        *,
        tokens=None,
        memory=None,
        memory_key_padding_mask=None,
        precursors=None,
        **kwargs,
    ):
        cur_len = (
            0
            if (tokens is None or tokens.numel() == 0)
            else int(tokens.shape[1])
        )
        seq_len = max(1, cur_len + 1)
        vocab = mock_model.vocab_size
        logits = torch.full((1, seq_len, vocab), -1.0)

        # Predict in REVERSE order: PEP → P, E, P (C to N)
        aa_reversed = aa[
            ::-1
        ]  # "PEP" becomes "PEP" (palindrome, but concept matters)

        # Total length includes N-term at the END if requested
        total_len = len(aa) + (1 if include_nterm else 0)

        if cur_len < total_len:
            if cur_len < len(aa):
                # Regular AA position (going from C to N)
                next_idx = aa_to_idx[aa_reversed[cur_len]]
                next_score = (
                    aa_score[cur_len] if cur_len < len(aa_score) else 0.5
                )
            else:
                # Last position: N-term token
                next_idx = nterm_token
                next_score = (
                    aa_score[cur_len] if cur_len < len(aa_score) else 0.9
                )
        else:
            # Natural stop
            next_idx = STOP_TOKEN
            next_score = 1.0

        logits[0, -1, next_idx] = next_score
        return logits

    mock_model.decoder.side_effect = mock_decoder

    # attributes
    mock_model.max_peptide_len = 10
    mock_model.min_peptide_len = 1
    mock_model.precursor_mass_tol = 10
    mock_model.isotope_error_range = (0, 1)
    mock_model.n_beams = 1
    mock_model.top_match = 1
    mock_model.stop_token = STOP_TOKEN
    mock_model.vocab_size = 27
    mock_model.residues = residues
    mock_model.softmax = lambda x: x

    return mock_model


def _setup_decoder(model_t, model_d, include_nterm=False):
    """Helper to set up decoder with common configuration."""
    decoder = Spec2PepTargetDecoy(model_t, model_d)
    decoder.perturbed_aa_masses = {"P": 1.0, "E": 2.0, "D": 3.0}
    decoder.nterm_idx = torch.tensor(
        [26]
    )  # N-term tokens trigger termination (normal behavior)
    decoder.register_parameter(
        "_dummy_param", torch.nn.Parameter(torch.tensor(0.0))
    )
    return decoder


def _create_batch():
    """Helper to create standard test batch."""
    return {
        "mz_array": torch.tensor([[1.0]]),
        "intensity_array": torch.tensor([[100.0]]),
        "precursor_mz": torch.tensor([[500.0]]),
        "precursor_charge": torch.tensor([[2]]),
    }


@patch(
    "casanovo.denovo.model._perturb_spectrum",
    return_value=(torch.tensor([1.0]), torch.tensor([100.0])),
)
def test_spec2peptargetdecoy_forward(mock_perturb):
    model_t = make_mock_model(
        "MockModel_A", aa="PEP", aa_score=[0.9, 0.8, 0.7]
    )
    model_d = make_mock_model(
        "MockModel_B", aa="PED", aa_score=[0.8, 0.7, 0.8]
    )

    model = Spec2PepTargetDecoy(model_t, model_d)
    model.perturbed_aa_masses = {"P": 1.0, "E": 2.0, "D": 3.0}
    model.register_parameter(
        "_dummy_param", torch.nn.Parameter(torch.tensor(0.0))
    )

    # dummy batch (one spectrum)
    filename, scan = "file.mgf", "scan_1"
    batch = {
        "mz_array": torch.tensor([[1.0]]),
        "intensity_array": torch.tensor([[100.0]]),
        "precursor_mz": torch.tensor([[500.0]]),
        "precursor_charge": torch.tensor([[2]]),
        "seq": "",
        "peak_file": [filename],
        "scan_id": [scan],
    }

    preds = model.predict_step(batch)

    assert isinstance(preds, list)
    assert len(preds) == 1

    # Check the single prediction fit expectations
    pred = preds[0]
    exp_seq = "PEP"  # All from target model
    exp_scores = [0.9, 0.8, 0.8]  # Mixed scores: target P, target E, decoy D
    exp_mask = [True, True, False]  # Target, Target, Decoy

    assert isinstance(pred, psm.PepSpecMatch)
    assert pred.sequence == exp_seq
    assert np.allclose(pred.aa_scores, exp_scores)
    assert np.array_equal(pred.aa_mask, exp_mask)

    # peptide_score uses TARGET scores: [0.9, 0.8, 0.7] + stop token score [1.0]
    expected_peptide_score = np.exp(
        np.log(0.9) + np.log(0.8) + np.log(0.7) + np.log(1.0)
    )
    assert np.isclose(
        pred.peptide_score, expected_peptide_score
    ), f"Expected peptide_score ≈ {expected_peptide_score}, got {pred.peptide_score}"

    # Check lengths of aa_scores and aa_mask
    assert len(pred.aa_scores) == len(pred.aa_mask), (
        f"aa_scores length ({len(pred.aa_scores)}) "
        f"must match aa_mask length ({len(pred.aa_mask)})"
    )

    # Check lengths of aa_scores and tokens
    assert len(pred.aa_scores) == len(pred.sequence), (
        f"aa_scores length ({len(pred.aa_scores)}) "
        f"should match peptide length ({len(pred.sequence)})"
    )

    # Other properties remain the same as inputs.
    assert pred.spectrum_id == (filename, scan)
    assert pred.charge == 2
    assert pred.exp_mz == 500.0


@patch("casanovo.denovo.model._perturb_spectrum_batched")
def test_psm_output_lengths_three_cases(mock_perturb):
    """Test PSM output arrays have consistent lengths for three termination scenarios."""
    mock_perturb.return_value = (
        torch.tensor([[1.0]]),
        torch.tensor([[100.0]]),
    )

    # Test cases: (description, include_nterm, max_peptide_len, expected_peptide)
    cases = [
        ("Natural Stop", False, 10, "PEP"),
        ("Forced Truncation", False, 3, "PEP"),  # max_len = peptide length
        ("N-term + Stop", True, 10, "[Acetyl]-PEP"),
    ]

    for case_name, include_nterm, max_len, expected_peptide in cases:
        # Set up models
        aa_score = [0.95, 0.9, 0.8, 0.7] if include_nterm else [0.9, 0.8, 0.7]
        model_t = make_mock_model("Target", "PEP", aa_score, include_nterm)
        model_d = make_mock_model("Decoy", "PED", aa_score, include_nterm)
        model_t.max_peptide_len = max_len
        model_d.max_peptide_len = max_len

        decoder = _setup_decoder(model_t, model_d)
        psms = decoder.forward(_create_batch())

        # Validate output
        (
            peptide_score,
            mixed_aa_scores,
            target_mask,
            peptide,
        ) = psms[0]

        lengths = [
            len(mixed_aa_scores),
            len(target_mask),
        ]

        assert (
            len(set(lengths)) == 1
        ), f"{case_name} FAILED: Inconsistent lengths {lengths}"
        assert (
            peptide == expected_peptide
        ), f"{case_name}: Expected '{expected_peptide}', got '{peptide}'"


def test_complementary_ions_annotated():
    """Test that both predicted and complementary ions are annotated."""
    fragment = spf.parse("PEPTIDE")[0]

    spectrum = sus.MsmsSpectrum(
        identifier="test_spectrum",
        precursor_mz=500.0,
        precursor_charge=2,
        mz=np.array([149.0, 132.5, 239.6, 264.1, 377.2]),
        intensity=np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
    )

    result = _annotate_fragment_pairs(
        spectrum=spectrum,
        fragment=fragment,
        fragment_tolerance_da=1.0,
        pred_ion_type="y",
        comp_ion_type="b",
        max_ion_charge=2,
        max_isotope=0,
        neutral_losses=False,
    )

    ion_types = [
        annot[0].ion_type
        for annot in result._annotation
        if annot[0].ion_type != "?"
    ]

    has_y = any("y" in ion_type for ion_type in ion_types)
    has_b = any(ion_type.startswith("b:") for ion_type in ion_types)

    assert len(ion_types) > 0, "At least some peaks should be annotated."
    assert (
        has_y
    ), f"Should have y-ion annotations. Annotated ion types: {ion_types}."
    assert (
        has_b
    ), f"Should have complementary b-ion annotations. Annotated ion types: {ion_types}."
