"""Test forward function of Spec2PepTargetDecoy() model"""

import numpy as np
import torch
from unittest.mock import MagicMock, patch

from casanovo.denovo.model import Spec2PepTargetDecoy
from casanovo.data import psm


def make_mock_model(name: str, aa: str, aa_score: list[float]):
    """
    Mock Spec2Pep-like model whose:
      - encoder returns deterministic (memory, mask)
      - decoder produces predefined `aa_score` with force the next residue of `aa`
      - softmax output the same scores as input
      - tokenizer maps residues <-> indices (P=0,E=1,D=2), STOP=3
    """
    mock_model = MagicMock(name=name)

    residues = {"P": 97.0, "E": 129.0, "D": 115.0}
    aa_to_idx = {"P": 0, "E": 1, "D": 2}
    idx_to_aa = {0: "P", 1: "E", 2: "D"}
    STOP_TOKEN = 3

    # Tokenizer.
    mock_model.tokenizer = MagicMock()
    mock_model.tokenizer.residues = residues
    mock_model.tokenizer.reverse = False

    mock_model.tokenizer.tokenize = MagicMock(
        side_effect=lambda seq: torch.tensor(
            [aa_to_idx[a] for a in seq], dtype=torch.int64
        )
    )

    def _detok(
        tokens, join=True
    ):  # only detokenize the first item in the batch
        chars = []
        for t in tokens[0]:
            tid = int(t.item())
            if tid == STOP_TOKEN:
                continue
            chars.append(idx_to_aa.get(tid, "?"))
        if join:
            return ["".join(chars)]
        else:
            return [chars]

    mock_model.tokenizer.detokenize = MagicMock(side_effect=_detok)

    # Encoder (does not involved in the whole process).
    def mock_encoder(mz_array: torch.Tensor, intensity_array: torch.Tensor):
        mean_mz = torch.mean(mz_array).item()
        mean_int = torch.mean(intensity_array).item()
        memory = torch.full((1, 10, 64), fill_value=mean_mz + mean_int)
        mask = torch.ones((1, 10), dtype=torch.bool)
        return memory, mask

    mock_model.encoder.side_effect = mock_encoder

    # Decoder (output as pre-defined).
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

        if cur_len < len(aa):
            next_idx = aa_to_idx[aa[cur_len]]
            next_score = aa_score[cur_len]
        else:
            next_idx = STOP_TOKEN  # stop decoding
            next_score = 1.0

        logits[0, -1, next_idx] = next_score
        return logits

    mock_model.decoder.side_effect = mock_decoder

    # attributes
    mock_model.max_peptide_len = 5
    mock_model.min_peptide_len = 1
    mock_model.precursor_mass_tol = 10
    mock_model.isotope_error_range = (0, 1)
    mock_model.n_beams = 1
    mock_model.top_match = 1
    mock_model.stop_token = STOP_TOKEN
    mock_model.vocab_size = 4  # P, E, D, STOP
    mock_model.residues = residues
    mock_model.softmax = lambda x: x

    return mock_model


@patch(
    "casanovo.denovo.model._perturb_spectrum",
    return_value=(torch.tensor([1.0]), torch.tensor([100.0])),
)
@patch(
    "casanovo.denovo.model._aa_pep_score",
    side_effect=lambda aa_scores, *_: (aa_scores, np.nan),
)
def test_spec2peptargetdecoy_forward(mock_aa_pep, mock_perturb):
    model_t = make_mock_model(
        "MockModel_A", aa="PEP", aa_score=[0.9, 0.8, 0.7]
    )
    model_d = make_mock_model(
        "MockModel_B", aa="PED", aa_score=[0.8, 0.7, 0.8]
    )

    model = Spec2PepTargetDecoy(model_t, model_d)
    model.perturbed_aa_masses = {"P": 1.0, "E": 2.0, "D": 3.0}

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

    pred = preds[0]
    exp_seq = "PEP"
    exp_scores = [0.9, 0.8, 0.8]
    exp_mask = [True, True, False]

    assert isinstance(pred, psm.PepSpecMatch)
    assert pred.sequence == exp_seq
    assert np.allclose(pred.aa_scores, exp_scores)
    assert np.array_equal(pred.aa_mask, exp_mask)
    assert np.isnan(
        pred.peptide_score
    )  # peptide-level score is np.nan in the mock models
    assert np.isnan(pred.calc_mz)  # FIXME: np.nan is temporarily assigned.

    # Other properties remain the same as inputs.
    assert pred.spectrum_id == (filename, scan)
    assert pred.charge == 2
    assert pred.exp_mz == 500.0
