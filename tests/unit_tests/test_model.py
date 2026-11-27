import copy
import logging
import unittest.mock

import depthcharge.tokenizers.peptides
import einops
import numpy as np
import pandas as pd
import pyteomics.proforma
import pytest
import torch

from casanovo import denovo
from casanovo.config import Config
from casanovo.data import psm
from casanovo.denovo.muon import MuonWithAuxAdamW
from casanovo.denovo.model import (
    DbSpec2Pep,
    Spec2Pep,
    _calc_match_score,
    _peptide_score,
)


def test_forward_reverse():
    """Test forward and reverse peptide predictions"""
    score_A = [0.42, 1.0, 0.0, 0.0, 0.0]
    score_B = [0.42, 0.0, 1.0, 0.0, 0.0]
    score_C = [0.42, 0.0, 0.0, 1.0, 0.0]
    score_padding = [0.00, 0.0, 0.0, 0.0, 0.0]
    score_none = [0.42, 0.0, 0.0, 0.0, 0.0]

    pep1 = torch.tensor(
        [
            score_A,
            score_B,
            score_C,
            score_padding,
        ]
    )

    pep2 = torch.tensor(
        [
            score_B,
            score_A,
            score_C,
            score_padding,
        ]
    )

    pep3 = torch.tensor(
        [
            score_C,
            score_A,
            score_B,
            score_padding,
        ]
    )

    pep4 = torch.tensor(
        [
            score_C,
            score_B,
            score_A,
            score_padding,
        ]
    )

    pep5 = torch.tensor(
        [
            score_A,
            score_none,
            score_none,
            score_padding,
        ]
    )

    true_aas = torch.tensor(
        [
            [1, 2, 3],
            [2, 1, 3],
            [3, 1, 2],
            [3, 2, 1],
            [1, 0, 0],
        ],
        dtype=int,
    )

    # Forward Direction
    batch_all_aa_scores = torch.stack([pep1, pep2, pep3, pep4, pep5])
    pep_scores, aa_scores = _calc_match_score(batch_all_aa_scores, true_aas)

    assert all([pytest.approx(1.0) == x for x in pep_scores])
    assert all(
        [np.allclose(np.array([1.0, 1.0, 1.0]), x) for x in aa_scores[:4]]
    )
    assert np.allclose(np.array([1.0]), aa_scores[4])

    # Reverse Direction
    flipped_batch_all_aa_scores = torch.flip(batch_all_aa_scores, dims=[1])
    pep_scores_reversed, aa_scores_reversed = _calc_match_score(
        flipped_batch_all_aa_scores, true_aas
    )

    assert all([pytest.approx(0.0) == x for x in pep_scores_reversed])
    assert all(
        [
            np.allclose(np.array([0.0, 0.0, 0.0]), x)
            for x in aa_scores_reversed[:4]
        ]
    )
    assert np.allclose(np.array([0.0]), aa_scores_reversed[4])


@pytest.mark.parametrize(
    ("aa_scores_raw", "expected"),
    [
        (np.asarray([0.0, 0.5, 1.0]), 0.0),
        (np.asarray([1.0, 0.25]), 0.25),
    ],
)
def test_peptide_score_scalar(aa_scores_raw, expected):
    peptide_score = _peptide_score(aa_scores_raw)
    assert peptide_score == pytest.approx(expected)


def test_peptide_score_batch():
    aa_scores_batch = np.array(
        [
            [0.5, 0.8, 0.0],
            [0.9, 0.7, 0.6],
        ]
    )
    lengths_batch = np.array([2, 3])

    expected_scores = np.array(
        [
            0.5 * 0.8,
            0.9 * 0.7 * 0.6,
        ]
    )

    peptide_scores = _peptide_score(aa_scores_batch, lengths_batch)
    assert np.allclose(peptide_scores, expected_scores)


def test_peptide_score_batch_requires_lengths():
    aa_scores_batch = np.array(
        [
            [0.5, 0.8, 0.0],
            [0.9, 0.7, 0.6],
        ]
    )

    with pytest.raises(
        ValueError,
        match=r"`lengths` must be provided for batched input\.",
    ):
        _peptide_score(aa_scores_batch, None)


def test_calc_match_score():
    """Check match scores for padded batches in both sequence directions."""
    score_A = [0.42, 1.0, 0.0, 0.0, 0.0]
    score_B = [0.42, 0.0, 1.0, 0.0, 0.0]
    score_C = [0.42, 0.0, 0.0, 1.0, 0.0]
    score_padding = [0.00, 0.0, 0.0, 0.0, 0.0]
    score_none = [0.42, 0.0, 0.0, 0.0, 0.0]

    pep1 = torch.tensor(
        [
            score_A,
            score_B,
            score_C,
            score_padding,
        ]
    )

    pep2 = torch.tensor(
        [
            score_B,
            score_A,
            score_C,
            score_padding,
        ]
    )

    pep3 = torch.tensor(
        [
            score_C,
            score_A,
            score_B,
            score_padding,
        ]
    )

    pep4 = torch.tensor(
        [
            score_C,
            score_B,
            score_A,
            score_padding,
        ]
    )

    pep5 = torch.tensor(
        [
            score_A,
            score_none,
            score_none,
            score_padding,
        ]
    )

    true_aas = torch.tensor(
        [
            [1, 2, 3],
            [2, 1, 3],
            [3, 1, 2],
            [3, 2, 1],
            [1, 0, 0],
        ],
        dtype=int,
    )

    batch_all_aa_scores = torch.stack([pep1, pep2, pep3, pep4, pep5])
    pep_scores, aa_scores = _calc_match_score(batch_all_aa_scores, true_aas)

    assert all([pytest.approx(1.0) == x for x in pep_scores])
    assert all(
        [np.allclose(np.array([1.0, 1.0, 1.0]), x) for x in aa_scores[:4]]
    )
    assert np.allclose(np.array([1.0]), aa_scores[4])

    # Reverse direction.
    flipped_batch_all_aa_scores = torch.flip(batch_all_aa_scores, dims=[1])
    flipped_true_aas = torch.flip(true_aas, dims=[1])
    pep_scores_reversed, aa_scores_reversed = _calc_match_score(
        flipped_batch_all_aa_scores, flipped_true_aas
    )

    assert all([pytest.approx(0.0) == x for x in pep_scores_reversed])
    assert all(
        [
            np.allclose(np.array([0.0, 0.0, 0.0]), x)
            for x in aa_scores_reversed[:4]
        ]
    )
    assert np.allclose(np.array([0.0]), aa_scores_reversed[4])


def test_psm_batches(tiny_config):
    """Check that candidate batches preserve the matching spectrum data."""
    peptides_one = [
        "SGSGSG",
        "GSGSGT",
        "SGSGTD",
        "FSGSGS",
        "ATSIPA",
        "GASTRA",
        "LSLSPG",
        "ASQSVS",
        "GSGTDF",
        "SLSPGE",
        "AQLLFL",
        "QPEDFA",
    ]

    peptides_two = [
        "SQSVSS",
        "KPGQAP",
        "SPPTLS",
        "ASTRAT",
        "RFSGSG",
        "IYGAST",
        "APAQLL",
        "PTLSLS",
        "TLSLSP",
        "TLTISS",
        "WYQQKP",
        "TWYQQK",
    ]

    def mock_get_candidates(precursor_mz, precorsor_charge):
        if precorsor_charge == 1:
            return pd.Index(peptides_one)
        elif precorsor_charge == 2:
            return pd.Index(peptides_two)
        else:
            return pd.Index([], dtype=str)

    tokenizer = depthcharge.tokenizers.peptides.PeptideTokenizer(
        residues=Config(tiny_config).residues
    )
    db_model = DbSpec2Pep(tokenizer=tokenizer)
    db_model.protein_database = unittest.mock.MagicMock()
    db_model.protein_database.get_candidates = mock_get_candidates

    mock_batch = {
        "precursor_mz": torch.Tensor([42.0, 84.0, 126.0]),
        "precursor_charge": torch.Tensor([1, 2, 3]),
        "peak_file": ["one.mgf", "two.mgf", "three.mgf"],
        "scan_id": [1, 2, 3],
    }
    fake_cache = {
        "memory": torch.zeros(3, 1, 1),
        "mem_masks": torch.ones(3, 1, dtype=torch.bool),
        "precursors_all": torch.zeros(3, 3),
    }
    expected_batch_all = {
        "precursor_mz": torch.Tensor([42.0] * 12 + [84.0] * 12),
        "precursor_charge": torch.Tensor([1] * 12 + [2] * 12),
        "seq": tokenizer.tokenize(peptides_one + peptides_two, add_stop=True),
        "peak_file": ["one.mgf"] * 12 + ["two.mgf"] * 12,
        "scan_id": [1] * 12 + [2] * 12,
    }

    num_spectra = 0
    for psm_batch in db_model._psm_batches(mock_batch, enc_cache=fake_cache):
        batch_size = len(psm_batch["peak_file"])
        end_idx = min(
            num_spectra + batch_size, len(expected_batch_all["peak_file"])
        )
        assert torch.allclose(
            psm_batch["precursor_mz"],
            expected_batch_all["precursor_mz"][num_spectra:end_idx],
        )
        assert torch.equal(
            psm_batch["precursor_charge"],
            expected_batch_all["precursor_charge"][num_spectra:end_idx],
        )
        assert torch.equal(
            psm_batch["seq"], expected_batch_all["seq"][num_spectra:end_idx]
        )
        assert (
            psm_batch["peak_file"]
            == expected_batch_all["peak_file"][num_spectra:end_idx]
        )
        assert (
            psm_batch["scan_id"]
            == expected_batch_all["scan_id"][num_spectra:end_idx]
        )
        num_spectra += batch_size
    assert num_spectra == 24


def test_db_stop_token(tiny_config):
    """Check that the stop token contributes only to the match score."""
    peptides_one = [
        "SGSGSG",
        "GSGSGT",
        "SGSGTD",
        "FSGSGS",
        "ATSIPA",
        "GASTRA",
        "LSLSPG",
        "ASQSVS",
        "GSGTDF",
        "SLSPGE",
        "AQLLFL",
        "QPEDFA",
    ]

    peptides_two = [
        "SQSVSS",
        "KPGQAP",
        "SPPTLS",
        "ASTRAT",
        "RFSGSG",
        "IYGAST",
        "APAQLL",
        "PTLSLS",
        "TLSLSP",
        "TLTISS",
        "WYQQKP",
        "TWYQQK",
    ]

    def mock_get_candidates(precursor_mz, precorsor_charge):
        if precorsor_charge == 1:
            return pd.Index(peptides_one)
        else:
            return pd.Index(peptides_two)

    tokenizer = depthcharge.tokenizers.peptides.PeptideTokenizer(
        residues=Config(tiny_config).residues
    )
    db_model = DbSpec2Pep(tokenizer=tokenizer)
    db_model.protein_database = unittest.mock.MagicMock()
    db_model.protein_database.get_candidates = mock_get_candidates

    mock_batch = {
        "precursor_mz": torch.Tensor([42.0, 84.0]),
        "precursor_charge": torch.Tensor([1, 2]),
        "peak_file": ["one.mgf", "two.mgf"],
        "scan_id": [1, 2],
        "mz_array": torch.zeros((2, 10)),
        "intensity_array": torch.zeros((2, 10)),
    }

    predictions = db_model.predict_step(mock_batch)
    assert predictions
    for prediction in predictions:
        assert len(tokenizer.tokenize(prediction.sequence)[0]) == len(
            prediction.aa_scores
        )
        assert pytest.approx(prediction.peptide_score) != _peptide_score(
            prediction.aa_scores
        )


def test_isoleucine_match(tiny_config):
    tokenizer = depthcharge.tokenizers.peptides.PeptideTokenizer(
        residues=Config(tiny_config).residues,
        replace_isoleucine_with_leucine=True,
    )
    db_model = DbSpec2Pep(tokenizer=tokenizer)
    peptides = [["PEPTLDEK"], ["PEPTIDEK"]]

    def mock_get_candidates(_, precursor_charge) -> pd.Index:
        return pd.Index(peptides[precursor_charge - 1])

    db_model = DbSpec2Pep(tokenizer=tokenizer)
    db_model.protein_database = unittest.mock.MagicMock()
    db_model.protein_database.get_candidates = mock_get_candidates

    batch = {
        "precursor_charge": torch.tensor([1, 2]),
        "precursor_mz": torch.tensor([42.0, 42.0]),
        "mz_array": torch.zeros((2, 10)),
        "intensity_array": torch.zeros((2, 10)),
        "peak_file": ["one.mgf", "two.mgf"],
        "scan_id": [1, 2],
    }

    matches = db_model.predict_step(batch)
    assert matches[0].sequence == "PEPTLDEK"
    assert matches[1].sequence == "PEPTIDEK"


def test_psm_batches_empty_candidates(tiny_config):
    """Test that _psm_batches returns nothing when no candidates exist."""
    tokenizer = depthcharge.tokenizers.peptides.PeptideTokenizer(
        residues=Config(tiny_config).residues
    )
    db_model = DbSpec2Pep(tokenizer=tokenizer)
    db_model.protein_database = unittest.mock.MagicMock()
    db_model.protein_database.get_candidates = lambda mz, charge: pd.Index(
        [], dtype=str
    )

    mock_batch = {
        "precursor_mz": torch.Tensor([42.0]),
        "precursor_charge": torch.Tensor([2]),
        "peak_file": ["test.mgf"],
        "scan_id": [1],
    }
    fake_cache = {
        "memory": torch.zeros(1, 1, 1),
        "mem_masks": torch.ones(1, 1, dtype=torch.bool),
        "precursors_all": torch.zeros(1, 3),
    }
    batches = list(db_model._psm_batches(mock_batch, enc_cache=fake_cache))
    assert len(batches) == 0


def test_n_term_scores(tiny_config):
    out_writer = unittest.mock.MagicMock()
    out_writer.psms = list()
    model = Spec2Pep(
        out_writer=out_writer,
        tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
            residues=Config(tiny_config).residues
        ),
    )

    matches = [
        psm.PepSpecMatch(
            sequence="[Acetyl]-P",
            aa_scores=np.array([0.5, 0.8]),
            spectrum_id="",
            peptide_score=float("NaN"),
            charge=1,
            calc_mz=float("NaN"),
            exp_mz=float("NaN"),
        ),
        psm.PepSpecMatch(
            sequence="PP",
            aa_scores=np.array([0.5, 0.8]),
            spectrum_id="",
            peptide_score=float("NaN"),
            charge=1,
            calc_mz=float("NaN"),
            exp_mz=float("NaN"),
        ),
    ]
    model.on_predict_batch_end(matches)

    assert len(out_writer.psms) == 2
    assert np.allclose(out_writer.psms[0].aa_scores, np.array([0.4]))
    assert np.allclose(out_writer.psms[1].aa_scores, np.array([0.5, 0.8]))


def test_n_term_scores_db(tiny_config, monkeypatch):
    out_writer = unittest.mock.MagicMock()
    out_writer.psms = list()
    model = DbSpec2Pep(
        out_writer=out_writer,
        tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
            residues=Config(tiny_config).residues
        ),
    )

    mock_psm_batchs = unittest.mock.MagicMock()
    mock_psm_batchs.return_value = [
        {
            "peak_file": ["one.mgf", "two.mgf"],
            "scan_id": [1, 2],
            "precursor_charge": ["+1", "+1"],
            "precursor_mz": torch.tensor([42.0, 42.0]),
            "original_seq_str": ["[Acetyl]-P", "PP"],
        }
    ]

    mock_forward = unittest.mock.MagicMock()
    mock_forward.return_value = (None, None)

    mock_protein_database = unittest.mock.MagicMock()
    mock_protein_database.get_associated_protein.return_value = "UPI_FOOBAR"

    model._psm_batches = mock_psm_batchs
    model.forward = mock_forward
    model.protein_database = mock_protein_database

    with monkeypatch.context() as mnk:

        def _mock_calc_match_score(pred, truth):
            return np.array([0.4, 0.4]), [
                np.array([0.5, 0.8, 0.0]),
                np.array([0.5, 0.8, 0.0]),
            ]

        mnk.setattr(denovo.model, "_calc_match_score", _mock_calc_match_score)
        B = 2
        P = 4
        mnk.setattr(
            type(model.encoder),
            "forward",
            lambda self, mz, it: (
                torch.zeros(B, 1, 1),
                torch.ones(B, 1, dtype=torch.bool),
            ),
        )

        dummy_batch = {
            "mz_array": torch.zeros(B, P),
            "intensity_array": torch.zeros(B, P),
            "precursor_mz": torch.tensor([42.0, 42.0]),
            "precursor_charge": torch.tensor([1.0, 1.0]),
        }
        model.on_predict_batch_end(model.predict_step(dummy_batch))

    assert len(out_writer.psms) == 2
    assert np.allclose(out_writer.psms[0].aa_scores, np.array([0.4]))
    assert np.allclose(out_writer.psms[1].aa_scores, np.array([0.5, 0.8]))


def test_train_val_step_functions():
    """Test train and validation step functions operating on batches."""
    tokenizer = depthcharge.tokenizers.peptides.MskbPeptideTokenizer()
    model = Spec2Pep(
        n_beams=1,
        residues="massivekb",
        min_peptide_len=4,
        tokenizer=tokenizer,
    )

    train_batch = {
        "mz_array": torch.zeros(1, 5),
        "intensity_array": torch.zeros(1, 5),
        "precursor_mz": torch.tensor(235.63410),
        "precursor_charge": torch.tensor(2),
        "seq": tokenizer.tokenize(["PEPK"]),
    }
    val_batch = copy.deepcopy(train_batch)

    train_step_loss = model.training_step(train_batch)
    with unittest.mock.patch.object(model, "log"):
        val_step_loss = model.validation_step(val_batch, 0)

    # Check if valid loss value returned
    assert train_step_loss > 0
    assert val_step_loss > 0

    # Training and validation share the same CTC loss, with a dedicated
    # blank class as the last vocabulary index.
    assert model.ctc_loss.blank == model.blank_token == model.vocab_size - 1
    assert torch.isclose(val_step_loss, train_step_loss)


@pytest.mark.parametrize("inter_ctc_layers", [(), (1,)])
def test_frames_attend_to_each_other(inter_ctc_layers):
    """Adding decoder frames changes the logits of the earlier ones.

    If no frame attends to any other, as when every frame was inferred to
    be padding, a position's logits cannot depend on how many frames
    follow it. Both decode paths are covered, since each passes its own
    masks to the layers.
    """
    tokenizer = depthcharge.tokenizers.peptides.MskbPeptideTokenizer(
        reverse=True, start_token=None, stop_token="$"
    )
    model = Spec2Pep(
        dim_model=8,
        n_head=2,
        dim_feedforward=8,
        n_layers=2,
        decoder_frames=8,
        residues="massivekb",
        tokenizer=tokenizer,
        inter_ctc_layers=inter_ctc_layers,
    ).eval()

    torch.manual_seed(0)
    batch = {
        "mz_array": torch.linspace(100.0, 1000.0, 12).repeat(2, 1),
        "intensity_array": torch.rand(2, 12),
        "precursor_mz": torch.full((2,), 600.0),
        "precursor_charge": torch.full((2,), 2.0),
    }

    with torch.no_grad():
        short, _ = model._forward_step(batch)
        # `decoder_frames` is the frame count, so raising it adds frames.
        model.decoder_frames += 8
        long, _ = model._forward_step(batch)

    shared = short.shape[1]
    assert long.shape[1] > shared, "the second run must be longer"
    assert not torch.allclose(long[:, :shared], short), (
        "the added frames left the earlier positions untouched, so the "
        "decoder frames are not attending to one another"
    )


def test_intermediate_ctc_adds_no_parameters():
    """Scoring intermediate layers supervises them without new weights."""
    tokenizer = depthcharge.tokenizers.peptides.MskbPeptideTokenizer(
        reverse=True, start_token=None, stop_token="$"
    )
    kwargs = dict(
        dim_model=8,
        n_head=2,
        dim_feedforward=8,
        n_layers=3,
        decoder_frames=8,
        residues="massivekb",
        tokenizer=tokenizer,
    )
    torch.manual_seed(0)
    plain = Spec2Pep(**kwargs).eval()
    unfed = Spec2Pep(**kwargs, inter_ctc_layers=(1, 2)).eval()
    # Scoring reuses the output layer the model already has, so the two
    # are the same size and a checkpoint from either loads into the other.
    assert plain.state_dict().keys() == unfed.state_dict().keys()
    unfed.load_state_dict(plain.state_dict())

    batch = {
        "mz_array": torch.linspace(100.0, 1000.0, 12).repeat(2, 1),
        "intensity_array": torch.rand(2, 12),
        "precursor_mz": torch.full((2,), 600.0),
        "precursor_charge": torch.full((2,), 2.0),
        "seq": tokenizer.tokenize(["PEPK", "PEPK"]),
    }
    with torch.no_grad():
        _, _, intermediates = unfed._forward_step(
            batch, return_intermediates=True
        )
    assert len(intermediates) == 2
    assert unfed.training_step(batch) > 0


def test_pmc_decode():
    """Test precise mass control CTC decoding."""
    tokenizer = depthcharge.tokenizers.peptides.MskbPeptideTokenizer(
        reverse=True, start_token=None, stop_token="$"
    )
    model = Spec2Pep(
        dim_model=8,
        n_head=2,
        dim_feedforward=8,
        n_layers=1,
        residues="massivekb",
        tokenizer=tokenizer,
    )
    idx = tokenizer.index
    aa_k, aa_g, aa_a, aa_e = idx["K"], idx["G"], idx["A"], idx["E"]

    # Greedy decoding prefers G at the second frame, but the precursor
    # mass matches K + A + E.
    logits = torch.full((4, model.vocab_size), -10.0)
    logits[0, aa_k] = 5.0
    logits[1, aa_g] = 5.0
    logits[1, aa_a] = 4.9
    logits[2, aa_e] = 5.0
    logits[3, model.blank_token] = 5.0

    greedy = model._ctc_decode(logits.unsqueeze(0))[0][0]
    assert greedy == [aa_k, aa_g, aa_e]

    masses = model.token_masses
    precursor_mass = (
        masses[aa_k] + masses[aa_a] + masses[aa_e]
    ).item() + 18.010565

    tokens, confs, _ = model._pmc_decode(logits, precursor_mass)
    assert tokens == [aa_k, aa_a, aa_e]
    assert len(confs) == len(tokens)
    assert model._fits_precursor_mass(tokens, precursor_mass)
    assert not model._fits_precursor_mass(greedy, precursor_mass)

    # An observed precursor mass off by one isotope is still matched.
    tokens, *_ = model._pmc_decode(logits, precursor_mass + 1.00335)
    assert tokens == [aa_k, aa_a, aa_e]

    # No path can reach an infeasible precursor mass.
    assert model._pmc_decode(logits, 5_000.0) is None


def test_default_tokenizer_is_reversed():
    """A model built without a tokenizer still supports PMC.

    depthcharge's own default is a forward tokenizer, which PMC declines,
    so a bare model would quietly decode without mass control.
    """
    model = Spec2Pep(dim_model=8, n_head=2, dim_feedforward=8, n_layers=1)
    assert model.tokenizer.reverse


def test_pmc_decode_declines_forward_tokenizer(caplog):
    """PMC places the stop by flipping the frames, so it needs a flip."""
    tokenizer = depthcharge.tokenizers.peptides.MskbPeptideTokenizer(
        reverse=False, start_token=None, stop_token="$"
    )
    model = Spec2Pep(
        dim_model=8,
        n_head=2,
        dim_feedforward=8,
        n_layers=1,
        residues="massivekb",
        tokenizer=tokenizer,
    )
    idx = tokenizer.index
    aa_k, aa_a, aa_e = idx["K"], idx["A"], idx["E"]
    masses = model.token_masses
    precursor_mass = (
        masses[aa_k] + masses[aa_a] + masses[aa_e]
    ).item() + denovo.model.H2O_MASS

    logits = torch.full((4, model.vocab_size), -10.0)
    logits[0, aa_k] = 5.0
    logits[1, aa_a] = 5.0
    logits[2, aa_e] = 5.0
    logits[3, model.blank_token] = 5.0

    # Declined rather than decoded against a stop it cannot place.
    with caplog.at_level(logging.WARNING):
        assert model._pmc_decode(logits, precursor_mass) is None
    assert "reversed" in caplog.text
    # Warned once, so a full run does not repeat it per spectrum.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert model._pmc_decode(logits, precursor_mass) is None
    assert "reversed" not in caplog.text


def _pmc_model():
    """A small model with the MassIVE-KB tokenizer for PMC tests."""
    # from_massivekb carries the modification tokens, including the
    # N-terminal ones; MskbPeptideTokenizer alone is canonical-only.
    tokenizer = depthcharge.tokenizers.PeptideTokenizer.from_massivekb(
        reverse=True, start_token=None, stop_token="$"
    )
    return Spec2Pep(
        dim_model=8,
        n_head=2,
        dim_feedforward=8,
        n_layers=1,
        residues="massivekb",
        tokenizer=tokenizer,
    )


def test_pmc_decode_nterm_mod():
    """N-terminal modifications decode only at the N-terminus."""
    model = _pmc_model()
    idx = model.tokenizer.index
    aa_g, aa_k = idx["G"], idx["K"]
    nterm = next(
        i for i in model.nterm_idx.tolist() if model.token_masses[i].item() > 0
    )

    # The tokenizer is reversed, so frames run C-terminus first and the
    # modification belongs at the last frame.
    logits = torch.full((3, model.vocab_size), -10.0)
    logits[0, aa_k] = 5.0
    logits[1, aa_g] = 5.0
    logits[2, nterm] = 5.0
    masses = model.token_masses
    precursor_mass = (
        masses[aa_k] + masses[aa_g] + masses[nterm]
    ).item() + 18.010565

    tokens, confs, _ = model._pmc_decode(logits, precursor_mass)
    assert tokens == [aa_k, aa_g, nterm]
    assert len(confs) == len(tokens)
    assert model._fits_precursor_mass(tokens, precursor_mass)

    # The same evidence with the modification at the C-terminal end
    # describes an invalid peptide, so no path is returned.
    assert model._pmc_decode(logits.flip(0), precursor_mass) is None


def test_pmc_decode_all_nterm_mods():
    """Every N-terminal modification decodes, negative mass included."""
    model = _pmc_model()
    idx = model.tokenizer.index
    aa_g, aa_k = idx["G"], idx["K"]
    masses = model.token_masses

    for label in (
        "[Ammonia-loss]-",
        "[Acetyl]-",
        "[+25.980265]-",
        "[Carbamyl]-",
    ):
        nterm = idx[label]
        logits = torch.full((3, model.vocab_size), -10.0)
        logits[0, aa_k] = 5.0
        logits[1, aa_g] = 5.0
        logits[2, nterm] = 5.0
        precursor_mass = (
            masses[aa_k] + masses[aa_g] + masses[nterm]
        ).item() + 18.010565

        tokens, *_ = model._pmc_decode(logits, precursor_mass)
        assert tokens == [aa_k, aa_g, nterm], label
        assert model._fits_precursor_mass(tokens, precursor_mass), label
        # Still rejected at the wrong end, negative mass or not.
        assert model._pmc_decode(logits.flip(0), precursor_mass) is None, label

    # Ammonia loss is the one that needs bins below zero.
    assert masses[idx["[Ammonia-loss]-"]].item() < 0


def test_pmc_decode_keeps_predicted_nterm_mod():
    """A mass correction leaves the modification where it was.

    The tests above hand PMC evidence the greedy path already agrees
    with, so it only confirms them. Here the precursor forces a residue
    to change, which is the case PMC actually runs for.
    """
    model = _pmc_model()
    idx = model.tokenizer.index
    masses = model.token_masses
    aa_k, aa_g, aa_a, aa_e = idx["K"], idx["G"], idx["A"], idx["E"]

    for label in (
        "[Ammonia-loss]-",
        "[Acetyl]-",
        "[+25.980265]-",
        "[Carbamyl]-",
    ):
        nterm = idx[label]
        # Reversed tokenizer, so frames run C-terminus first and the
        # modification sits last, ahead of the blank tail.
        logits = torch.full((12, model.vocab_size), -10.0)
        logits[0, aa_k] = 5.0
        logits[1, aa_g] = 5.0
        logits[1, aa_a] = 4.9
        logits[2, aa_e] = 5.0
        logits[3, nterm] = 5.0
        logits[4:, model.blank_token] = 5.0

        precursor_mass = (
            masses[aa_k] + masses[aa_a] + masses[aa_e] + masses[nterm]
        ).item() + denovo.model.H2O_MASS
        greedy = model._ctc_decode(logits.unsqueeze(0))[0][0]
        assert greedy == [aa_k, aa_g, aa_e, nterm], label
        assert not model._fits_precursor_mass(greedy, precursor_mass), label

        tokens, *_ = model._pmc_decode(logits, precursor_mass)
        assert tokens == [aa_k, aa_a, aa_e, nterm], label
        assert model._fits_precursor_mass(tokens, precursor_mass), label

        # A precursor that accounts for no modification drops it.
        bare = (
            masses[aa_k] + masses[aa_a] + masses[aa_e]
        ).item() + denovo.model.H2O_MASS
        tokens, *_ = model._pmc_decode(logits, bare)
        assert tokens == [aa_k, aa_a, aa_e], label


def test_pmc_decode_coarse_resolution(monkeypatch):
    """A precursor too heavy for the fine mass grid still decodes."""
    model = _pmc_model()
    idx = model.tokenizer.index
    aa_k, aa_g, aa_a, aa_e = idx["K"], idx["G"], idx["A"], idx["E"]

    logits = torch.full((4, model.vocab_size), -10.0)
    logits[0, aa_k] = 5.0
    logits[1, aa_g] = 5.0
    logits[1, aa_a] = 4.9
    logits[2, aa_e] = 5.0
    logits[3, model.blank_token] = 5.0
    masses = model.token_masses
    precursor_mass = (
        masses[aa_k] + masses[aa_a] + masses[aa_e]
    ).item() + 18.010565

    # A budget this small cannot hold the 0.01 Da grid; decoding should
    # coarsen the grid instead of giving up on mass control.
    monkeypatch.setattr(denovo.model, "PMC_MAX_POINTER_BYTES", 50_000)
    tokens, *_ = model._pmc_decode(logits, precursor_mass)
    assert tokens == [aa_k, aa_a, aa_e]
    assert model._fits_precursor_mass(tokens, precursor_mass)


def test_pmc_decode_emits_one_stop():
    """The stop is scored once, however tempting a second one is.

    It carries no mass, so pinning it to the zero-mass bin left that bin
    still satisfying the pin: a blank could return there and emit
    another, counting its log-probability twice in a path score that
    reports only one stop. A continuous run is a single CTC emission, so
    the second one needs a blank-dominant frame between two
    stop-dominant ones.
    """
    model = _pmc_model()
    idx = model.tokenizer.index
    aa_k, aa_a, aa_e = idx["K"], idx["A"], idx["E"]
    stop, blank = model.stop_token, model.blank_token
    masses = model.token_masses
    precursor_mass = (
        masses[aa_k] + masses[aa_a] + masses[aa_e]
    ).item() + denovo.model.H2O_MASS

    # Built in the DP's own order, then flipped, so the model's flip for
    # a reversed tokenizer restores exactly this.
    dp = torch.full((10, model.vocab_size), -10.0)
    dp[:, blank] = 0.0
    dp[0, stop] = 3.0
    dp[1, blank] = 3.0
    dp[2, stop] = 3.0
    dp[4, aa_k] = 5.0
    dp[6, aa_a] = 5.0
    dp[8, aa_e] = 5.0

    tokens, confs, stop_conf = model._pmc_decode(dp.flip(0), precursor_mass)
    # The stop leaves separately, so it is never among the residues.
    assert tokens == [aa_e, aa_a, aa_k]
    assert len(confs) == len(tokens)
    assert stop_conf is not None
    assert model._fits_precursor_mass(tokens, precursor_mass)

    # The decoded peptide alone does not show the double emission, since
    # the stops are stripped either way, so check the rule that forbids
    # it: the stop has one legal target and it is not where it starts.
    resolution = denovo.model.PMC_RESOLUTION
    emit_idx, delta_idx, _, emit_pinned, emit_is_stop, deltas = (
        model._pmc_alphabet(model.vocab_size, 5_000.0, resolution, "cpu")
    )
    assert masses[stop].item() == 0.0, "the stop must carry no mass"
    assert deltas[stop] == 1, "yet it must still move the state"

    zero_bin, n_bins = 5, 40
    src_bins, valid = model._pmc_transitions(
        delta_idx, emit_pinned, emit_is_stop, n_bins, zero_bin, "cpu"
    )
    row = emit_idx.tolist().index(stop)
    targets = valid[row].nonzero().flatten().tolist()
    assert targets == [zero_bin + 1]
    assert int(src_bins[row, zero_bin + 1]) == zero_bin


def test_pmc_decode_near_miss_does_not_shadow_a_match():
    """A near-isobaric decoy must not shadow a mass-matching path.

    The mass axis is widened by PMC_MASS_GUARD, far beyond the 50 ppm
    tolerance. A readout testing bins against those widened windows would
    take a decoy inside the guard on score, fail the exact mass check,
    and fall back to the greedy peptide. The states carry exact masses,
    so the decoy is never a candidate.
    """
    model = _pmc_model()
    idx = model.tokenizer.index
    aa_k, aa_q, aa_a, aa_e = idx["K"], idx["Q"], idx["A"], idx["E"]

    masses = model.token_masses
    # K and Q are 0.0364 Da apart: inside PMC_MASS_GUARD (0.1 Da), well
    # outside 50 ppm of this precursor (0.017 Da).
    gap = abs((masses[aa_k] - masses[aa_q]).item())
    precursor_mass = (
        masses[aa_k] + masses[aa_a] + masses[aa_e]
    ).item() + denovo.model.H2O_MASS
    tol = model.precursor_mass_tol * precursor_mass / 1e6
    assert tol < gap < denovo.model.PMC_MASS_GUARD

    # Q outscores K, so the decoy is the highest-probability path.
    logits = torch.full((4, model.vocab_size), -10.0)
    logits[0, aa_q] = 5.0
    logits[0, aa_k] = 4.9
    logits[1, aa_a] = 5.0
    logits[2, aa_e] = 5.0
    logits[3, model.blank_token] = 5.0

    greedy = model._ctc_decode(logits.unsqueeze(0))[0][0]
    assert greedy == [aa_q, aa_a, aa_e]
    assert not model._fits_precursor_mass(greedy, precursor_mass)

    result = model._pmc_decode(logits, precursor_mass)
    assert result is not None, "the decoy shadowed a real match"
    tokens, *_ = result
    assert tokens == [aa_k, aa_a, aa_e]
    assert model._fits_precursor_mass(tokens, precursor_mass)


def test_predict_step_keeps_mass_mismatch_unpenalized(monkeypatch):
    """PMC may give up; the greedy peptide it leaves behind keeps its score.

    Upstream subtracts 1 from such a peptide. Here it is not penalized: on
    chimeric spectra it is often the co-isolated peptide, whose mass misses
    the recorded precursor.
    """
    model = _pmc_model()
    model.min_peptide_len = 0
    idx = model.tokenizer.index
    tokens = [idx["K"], idx["A"], idx["E"]]

    logits = torch.full((2, 4, model.vocab_size), -10.0)
    for frame, token in enumerate(tokens):
        logits[:, frame, token] = 5.0
    logits[:, 3, model.blank_token] = 5.0

    mass = sum(model.token_masses[t].item() for t in tokens)
    mass += denovo.model.H2O_MASS
    # The second precursor is out of reach for any path, so PMC returns
    # None and the unmatched greedy peptide survives.
    precursors = torch.tensor([[mass], [5_000.0]])
    monkeypatch.setattr(model, "_forward_step", lambda batch: (logits, None))
    monkeypatch.setattr(
        model, "_process_batch", lambda batch: (None, None, precursors, None)
    )

    matched, missed = model.predict_step(
        {
            "peak_file": ["a.mgf", "a.mgf"],
            "scan_id": [1, 2],
            "precursor_charge": torch.tensor([2, 2]),
            "precursor_mz": torch.tensor([400.0, 400.0]),
        }
    )

    assert matched.sequence == missed.sequence
    # Same peptide, same logits: the same score whether or not it fits.
    assert 0 <= missed.peptide_score == pytest.approx(matched.peptide_score)
    # Aggregated as upstream does, so the reported score stays the
    # product of the reported residue scores.
    assert matched.peptide_score == pytest.approx(
        float(np.prod(matched.aa_scores))
    )


def test_stop_token_is_scored_with_the_peptide(monkeypatch):
    """The stop is trained as an emission, so it counts in the score.

    CTC supervises it (`add_stop` puts it in the target), and upstream's
    peptide score counts it, so decoding scores it too. It is not part
    of the peptide, so it stays out of the reported residue scores.
    """
    model = _pmc_model()
    model.min_peptide_len = 0
    seq = [model.tokenizer.index[a] for a in ("K", "A", "E")]

    logits = torch.full((1, 8, model.vocab_size), -10.0)
    for frame, token in enumerate(seq):
        logits[0, frame, token] = 5.0
    # Deliberately less certain than the residues, so its contribution
    # to the product is visible.
    logits[0, 3, model.stop_token] = 2.0
    logits[0, 4:, model.blank_token] = 5.0

    mass = sum(model.token_masses[t].item() for t in seq)
    mass += denovo.model.H2O_MASS
    monkeypatch.setattr(model, "_forward_step", lambda batch: (logits, None))
    monkeypatch.setattr(
        model,
        "_process_batch",
        lambda batch: (None, None, torch.tensor([[mass]]), None),
    )

    (pred,) = model.predict_step(
        {
            "peak_file": ["a.mgf"],
            "scan_id": [1],
            "precursor_charge": torch.tensor([2]),
            "precursor_mz": torch.tensor([400.0]),
        }
    )

    _, _, (stop_conf,) = model._ctc_decode(logits)
    assert stop_conf is not None and stop_conf < 1
    # Reported residues exclude the stop; the peptide score includes it.
    assert len(pred.aa_scores) == len(seq)
    assert pred.peptide_score == pytest.approx(
        float(np.prod(pred.aa_scores)) * stop_conf
    )
    assert pred.peptide_score < float(np.prod(pred.aa_scores))


def test_predict_step_counts_missing(monkeypatch):
    """A spectrum that decodes to no peptide is counted as missing."""
    model = _pmc_model()
    model.min_peptide_len = 0
    tokens = [model.tokenizer.index[aa] for aa in ("K", "A", "E")]

    logits = torch.full((2, 4, model.vocab_size), -10.0)
    for frame, token in enumerate(tokens):
        logits[0, frame, token] = 5.0
    logits[0, 3, model.blank_token] = 5.0
    # The second spectrum is all blanks, and no path reaches its
    # precursor, so PMC cannot rescue it either.
    logits[1, :, model.blank_token] = 5.0

    mass = sum(model.token_masses[t].item() for t in tokens)
    mass += denovo.model.H2O_MASS
    precursors = torch.tensor([[mass], [5_000.0]])
    monkeypatch.setattr(model, "_forward_step", lambda batch: (logits, None))
    monkeypatch.setattr(
        model, "_process_batch", lambda batch: (None, None, precursors, None)
    )

    batch = {
        "peak_file": ["a.mgf", "a.mgf"],
        "scan_id": ["index=0", "index=1"],
        "precursor_charge": torch.tensor([2, 2]),
        "precursor_mz": torch.tensor([100.0, 200.0]),
    }
    predictions = model.predict_step(batch)
    assert len(predictions) == 1
    assert model.n_missing_predictions == 1
    model.on_predict_start()
    assert model.n_missing_predictions == 0


def test_predict_step_keeps_greedy_over_short_pmc(monkeypatch):
    """A PMC correction below min_peptide_len leaves the greedy peptide."""
    model = _pmc_model()
    model.min_peptide_len = 3
    tokens = [model.tokenizer.index[aa] for aa in ("K", "A", "E")]

    logits = torch.full((1, 4, model.vocab_size), -10.0)
    for frame, token in enumerate(tokens):
        logits[0, frame, token] = 5.0
    logits[0, 3, model.blank_token] = 5.0
    # The greedy peptide misses this precursor; the stub correction
    # matches it but has only two residues.
    precursors = torch.tensor([[5_000.0]])
    monkeypatch.setattr(model, "_forward_step", lambda batch: (logits, None))
    monkeypatch.setattr(
        model, "_process_batch", lambda batch: (None, None, precursors, None)
    )
    monkeypatch.setattr(
        model, "_pmc_decode", lambda *args: (tokens[:2], [0.9, 0.9], None)
    )

    batch = {
        "peak_file": ["a.mgf"],
        "scan_id": ["index=0"],
        "precursor_charge": torch.tensor([2]),
        "precursor_mz": torch.tensor([100.0]),
    }
    predictions = model.predict_step(batch)
    assert len(predictions) == 1
    assert predictions[0].sequence == "".join(
        model.tokenizer.detokenize(torch.tensor([tokens]), join=False)[0]
    )
    assert model.n_missing_predictions == 0


def test_pmc_alphabet_allows_heavy_residue_after_negative_mod():
    """A residue above the axis top stays emittable when a negative
    N-terminal modification can offset it."""
    model = _pmc_model()
    neg = min(model.token_masses[c].item() for c in model.neg_mass_idx)
    aa_w = model.tokenizer.index["W"]
    w_mass = model.token_masses[aa_w].item()
    hi_max = w_mass + neg + 1.0
    alphabet = model._pmc_alphabet(model.vocab_size, hi_max, 0.01, "cpu")
    assert aa_w in alphabet[0].tolist()


def test_spec2pep_rejects_bad_settings():
    """Direct construction checks what the config checks."""
    with pytest.raises(ValueError, match="inter_ctc_weight"):
        Spec2Pep(inter_ctc_weight=1.5)
    with pytest.raises(ValueError, match="inter_ctc_weight"):
        Spec2Pep(inter_ctc_weight=float("nan"))
    with pytest.raises(ValueError, match="isotope_error_range"):
        Spec2Pep(isotope_error_range=(3, 0))
    Spec2Pep(isotope_error_range=(0, 0))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ctc_log_probs_promotes_half_precision(dtype):
    """Half-precision scores reach the CTC loss as float32."""
    scores = torch.randn(2, 5, 7).to(dtype)
    log_probs = denovo.model._ctc_log_probs(scores)
    assert log_probs.dtype == torch.float32
    assert log_probs.shape == (5, 2, 7)
    assert torch.allclose(log_probs.exp().sum(-1), torch.ones(5, 2))


def test_validation_withholds_short_predictions(monkeypatch):
    """Validation precision counts a too-short peptide as missing."""
    model = _pmc_model()
    model.calculate_precision = True
    model.min_peptide_len = 4
    tokens = [model.tokenizer.index[aa] for aa in ("K", "A", "E")]
    monkeypatch.setattr(
        model,
        "_ctc_decode",
        lambda logits: ([tokens], [[0.9] * 3], [None]),
    )
    monkeypatch.setattr(model, "forward", lambda batch: (None, None))
    monkeypatch.setattr(
        model, "_ctc_losses", lambda batch: (*[torch.tensor(1.0)] * 2, None, 1)
    )
    seen = {}

    def _match(peptides_true, peptides_pred, residues):
        seen["pred"] = peptides_pred
        return [], 0, 0

    monkeypatch.setattr(denovo.model.evaluate, "aa_match_batch", _match)
    monkeypatch.setattr(
        denovo.model.evaluate, "aa_match_metrics", lambda *a: (0.0, 0.0, 0.0)
    )
    model.log = lambda *args, **kwargs: None
    batch = {"seq": model.tokenizer.tokenize(["KAE"])}
    model.validation_step(batch, 0)
    assert seen["pred"] == [None]


@pytest.mark.parametrize("option", ["maximize", "amsgrad"])
def test_muon_rejects_unsupported_adamw_options(option):
    """AdamW options this optimizer does not implement fail loudly."""
    bias = torch.nn.Parameter(torch.zeros(3))
    with pytest.raises(ValueError, match=option):
        MuonWithAuxAdamW(
            [dict(params=[bias], use_muon=False, **{option: True})]
        )


def test_ctc_warns_on_peptides_longer_than_the_frames(caplog):
    """A peptide CTC cannot align is reported once, not silently lost."""
    model = _pmc_model()
    model.decoder_frames = 2
    batch = {
        "mz_array": torch.linspace(100.0, 1000.0, 12).unsqueeze(0),
        "intensity_array": torch.rand(1, 12),
        "precursor_mz": torch.tensor([600.0]),
        "precursor_charge": torch.tensor([2.0]),
        "seq": model.tokenizer.tokenize(["PEPTIDEK"]),
    }
    with caplog.at_level(logging.WARNING):
        model._ctc_losses(batch)
    assert "need more CTC frames" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        model._ctc_losses(batch)
    assert "need more CTC frames" not in caplog.text


def test_pmc_decode_declines(monkeypatch, caplog):
    """PMC returns no match, keeping the greedy peptide, when it cannot
    search: no positive mass window, an infinite tolerance, a table over
    budget, or a vocabulary its int8 pointers cannot index."""
    model = _pmc_model()
    tokens = [model.tokenizer.index[aa] for aa in ("K", "A", "E")]
    logits = torch.full((4, model.vocab_size), -10.0)
    for frame, token in enumerate(tokens):
        logits[frame, token] = 5.0
    logits[3, model.blank_token] = 5.0
    mass = sum(model.token_masses[t].item() for t in tokens)
    mass += denovo.model.H2O_MASS
    assert model._pmc_decode(logits, mass) is not None

    assert model._pmc_decode(logits, 0.0) is None

    model.precursor_mass_tol = float("inf")
    assert model._pmc_decode(logits, mass) is None
    model.precursor_mass_tol = 50

    with monkeypatch.context() as m:
        m.setattr(denovo.model, "PMC_MAX_POINTER_BYTES", 10)
        assert model._pmc_decode(logits, mass) is None

    wide = torch.full((4, 130), -10.0)
    wide[:, : model.vocab_size] = logits
    with caplog.at_level(logging.WARNING):
        assert model._pmc_decode(wide, mass) is None
    assert "at most 127" in caplog.text


def test_muon_rejects_vectors_and_skips_missing_grads():
    """Muon groups hold matrices only, and parameters without a gradient
    are left alone by both update rules."""
    with pytest.raises(ValueError, match="only matrices"):
        MuonWithAuxAdamW(
            [dict(params=[torch.nn.Parameter(torch.zeros(3))], use_muon=True)]
        )

    matrix = torch.nn.Parameter(torch.randn(4, 3))
    filters = torch.nn.Parameter(torch.randn(4, 2, 3))
    frozen = torch.nn.Parameter(torch.randn(4, 3))
    bias = torch.nn.Parameter(torch.zeros(3))
    idle = torch.nn.Parameter(torch.zeros(3))
    opt = MuonWithAuxAdamW(
        [
            dict(params=[matrix, filters, frozen], use_muon=True),
            dict(params=[bias, idle], use_muon=False),
        ]
    )
    (matrix.sum() + filters.sum() + bias.sum()).backward()
    before = [
        p.detach().clone() for p in (matrix, filters, frozen, bias, idle)
    ]
    opt.step()
    after = [p.detach() for p in (matrix, filters, frozen, bias, idle)]
    # Updated: the matrix, the 3-D filters (flattened for Newton-Schulz),
    # and the bias. Untouched: the two without a gradient.
    assert [not torch.equal(b, a) for b, a in zip(before, after)] == [
        True,
        True,
        False,
        True,
        False,
    ]


def test_muon_accepts_iterator_params():
    """One-shot iterators of parameters are not lost to validation."""
    weight = torch.nn.Parameter(torch.randn(3, 3))
    bias = torch.nn.Parameter(torch.zeros(3))
    opt = MuonWithAuxAdamW(
        [
            dict(params=iter([weight]), use_muon=True),
            dict(params=iter([bias]), use_muon=False),
        ]
    )
    assert opt.param_groups[0]["params"] == [weight]
    assert opt.param_groups[1]["params"] == [bias]


def test_decoder_embed_without_tokens():
    """tokens=None embeds only the precursor, one row per spectrum."""
    model = _pmc_model()
    precursors = torch.tensor([[1000.0, 2.0], [1500.0, 3.0]])
    memory = torch.randn(2, 5, 8)
    out = model.decoder.embed(None, precursors, memory=memory)
    assert out.shape == (2, 1, 8)


def test_on_predict_epoch_end_aggregates(tiny_config):
    config = Config(tiny_config)
    model = Spec2Pep(
        tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
            residues=config.residues
        ),
    )
    # Local counts of 2 and 3 from two processes are summed.
    model.all_gather = lambda x: x.new_tensor([2, 3])
    model.n_missing_predictions = 2
    model.on_predict_epoch_end()
    assert model.n_missing_predictions == 5


@pytest.mark.skip(reason="No Beam Search in Non-AR Model")
def test_finish_beams(tiny_config):
    """Finish a beam at its stop token and discard it if it is too short."""
    config = Config(tiny_config)
    model = Spec2Pep(
        n_beams=4,
        min_peptide_len=4,
        tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
            residues=config.residues
        ),
    )
    model.tokenizer.reverse = False

    batch = 1
    length = model.max_peptide_len + 1
    beam = model.n_beams
    step = 3
    device = model.device

    model._batch_size = batch
    model._beam_size = beam

    vocab = len(model.tokenizer) + 1
    scores = torch.full((batch, length, vocab, beam), torch.nan, device=device)
    scores = einops.rearrange(scores, "B L V S -> (B S) L V")

    tokens = torch.zeros(
        batch * beam, length, dtype=torch.int64, device=device
    )

    peptides = [
        ("PEPK", False),
        ("PEPR", False),
        ("PEPG", False),
        ("PEP", True),
    ]

    for i, (pep, add_stop) in enumerate(peptides):
        toks = model.tokenizer.tokenize(pep, add_stop=add_stop)[0]
        tokens[i, : step + 1] = toks
        for j in range(step + 1):
            scores[i, j, toks[j]] = 1

    finished, discarded = model._finish_beams(tokens, step)

    # Last beam discarded for being too short
    assert torch.equal(
        finished, torch.tensor([False, False, False, True], device=device)
    )
    assert torch.equal(
        discarded, torch.tensor([False, False, False, True], device=device)
    )


@pytest.mark.skip(reason="No Beam Search in Non-AR Model")
@pytest.mark.parametrize("reverse", [True, False])
def test_invalid_n_term_mod(tiny_config, reverse):
    """Reject misplaced and repeated N-terminal mods in either direction."""
    config = Config(tiny_config)
    model = Spec2Pep(
        n_beams=4,
        residues="massivekb",
        tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
            residues=config.residues
        ),
    )

    model.tokenizer.reverse = reverse
    length = model.max_peptide_len + 1
    step = 4
    device = model.device
    model.min_peptide_len = 3

    # A reversed tokenizer emits the C-terminus first, so a valid N-terminal
    # modification is the last token generated; an unreversed tokenizer emits
    # it first.
    # (peptide, should_be_discarded)
    peptides = [
        (["P", "E", "P", "K", "K"], False),
        (["[Acetyl]-", "P", "E", "P", "K"], reverse),
        (["[Acetyl]-", "[Acetyl]-", "P", "E", "P"], True),
        (["[Acetyl]-", "P", "E", "P", "[Acetyl]-"], True),
        (["P", "E", "P", "K", "[Acetyl]-"], not reverse),
        (["P", "E", "P", "[Acetyl]-", "$"], not reverse),
        (["P", "E", "[Acetyl]-", "P", "K"], True),
    ]

    tokens = torch.zeros(
        len(peptides), length, dtype=torch.int64, device=device
    )

    # Stop the tokenizer from protecting against invalid n-term mods
    index = model.tokenizer.index
    index["$"] = model.tokenizer.stop_int
    for i, (pep, _) in enumerate(peptides):
        toks = torch.tensor([index[aa] for aa in pep], device=device)
        tokens[i, : step + 1] = toks

    expected_discarded = torch.tensor([c for _, c in peptides], device=device)
    expected_finished = torch.tensor(
        [pep[-1] == "$" for pep, _ in peptides], device=device
    )
    finished, discarded = model._finish_beams(tokens, step)

    assert torch.equal(discarded, expected_discarded)
    assert torch.equal(finished, expected_finished)


def _nterm_beam(tokenizer, aa_tokens, max_peptide_len, device):
    """
    Build a single beam from a list of amino acid tokens.

    Parameters
    ----------
    tokenizer : depthcharge.tokenizers.PeptideTokenizer
        The tokenizer providing the token indices.
    aa_tokens : Iterable[str]
        The amino acid tokens, in the order the decoder would emit them.
    max_peptide_len : int
        The maximum peptide length, defining the width of the beam.
    device : torch.device
        The device on which to create the beam.

    Returns
    -------
    torch.Tensor of shape (1, max_peptide_len + 1)
        The zero-padded beam.
    """
    tokens = torch.zeros(
        1, max_peptide_len + 1, dtype=torch.int64, device=device
    )
    tokens[0, : len(aa_tokens)] = torch.tensor(
        [tokenizer.index[aa] for aa in aa_tokens], device=device
    )
    return tokens


@pytest.mark.skip(reason="No Beam Search in Non-AR Model")
@pytest.mark.parametrize("reverse", [True, False])
def test_n_term_mod_orientation_matches_tokenizer(tiny_config, reverse):
    """N-term mods placed as the tokenizer emits them must be kept."""
    config = Config(tiny_config)
    tokenizer = depthcharge.tokenizers.peptides.PeptideTokenizer(
        residues=config.residues, reverse=reverse
    )
    model = Spec2Pep(n_beams=1, tokenizer=tokenizer, min_peptide_len=4)

    peptide = "[Acetyl]-PEPTIDEK"
    # Ask the tokenizer for the token order, rather than assuming one, so
    # that this test still pins down the correct behavior if the decoding
    # direction ever changes.
    aa_tokens = list(tokenizer.split(peptide))
    tokens = _nterm_beam(
        tokenizer, aa_tokens, model.max_peptide_len, model.device
    )

    _, discarded = model._finish_beams(tokens, len(aa_tokens) - 1)
    assert not discarded[0]

    # A kept beam must decode to a valid ProForma peptidoform, since
    # `on_predict_batch_end` parses it again to compute the precursor m/z.
    sequence = tokenizer.detokenize(tokens)[0]
    assert sequence == peptide
    tokenizer.calculate_precursor_ions(sequence, torch.tensor(2))


@pytest.mark.skip(reason="No Beam Search in Non-AR Model")
@pytest.mark.parametrize("reverse", [True, False])
def test_n_term_mod_wrong_end_discarded(tiny_config, reverse):
    """N-term mods at the opposite end of the beam must be discarded."""
    config = Config(tiny_config)
    tokenizer = depthcharge.tokenizers.peptides.PeptideTokenizer(
        residues=config.residues, reverse=reverse
    )
    model = Spec2Pep(n_beams=1, tokenizer=tokenizer, min_peptide_len=4)

    peptide = "[Acetyl]-PEPTIDEK"
    aa_tokens = list(tokenizer.split(peptide))[::-1]
    tokens = _nterm_beam(
        tokenizer, aa_tokens, model.max_peptide_len, model.device
    )

    _, discarded = model._finish_beams(tokens, len(aa_tokens) - 1)
    assert discarded[0]

    # Such a beam decodes to a modification trailing the sequence, which is
    # not valid ProForma; keeping it would crash `on_predict_batch_end`. The
    # exact exception type depends on the pyteomics version.
    with pytest.raises((IndexError, pyteomics.proforma.ProFormaError)):
        tokenizer.calculate_precursor_ions(
            tokenizer.detokenize(tokens)[0], torch.tensor(2)
        )


@pytest.mark.skip(reason="No Beam Search in Non-AR Model")
@pytest.mark.parametrize("reverse", [True, False])
def test_n_term_mod_short_beam_discarded(tiny_config, reverse):
    """Misplaced N-term mods are caught in two-token beams as well."""
    config = Config(tiny_config)
    tokenizer = depthcharge.tokenizers.peptides.PeptideTokenizer(
        residues=config.residues, reverse=reverse
    )
    model = Spec2Pep(n_beams=1, tokenizer=tokenizer, min_peptide_len=2)

    # Too short to be caught by the minimum peptide length check.
    aa_tokens = ["[Acetyl]-", "K"] if reverse else ["K", "[Acetyl]-"]
    tokens = _nterm_beam(
        tokenizer, aa_tokens, model.max_peptide_len, model.device
    )

    _, discarded = model._finish_beams(tokens, 1)
    assert discarded[0]


@pytest.mark.skip(reason="No Beam Search in Non-AR Model")
def test_cache_finished_beams(tiny_config):
    config = Config(tiny_config)
    model = Spec2Pep(
        n_beams=4,
        residues="massivekb",
        tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
            residues=config.residues
        ),
    )

    batch = 1
    beam = model.n_beams
    length = 10
    step = 3
    device = model.device

    model._batch_size = batch
    model._beam_size = beam

    vocab = len(model.tokenizer) + 1
    scores = torch.zeros((beam, length, vocab), device=device)
    tokens = torch.zeros((beam, length), dtype=torch.int64, device=device)

    cache_tokens = torch.full(
        (batch, beam, length, length), 0, dtype=torch.int32, device=device
    )
    cache_scores = torch.full(
        (batch, beam, length, length), 0.0, dtype=scores.dtype, device=device
    )

    true_tok = model.tokenizer.tokenize("PEPR")[0]
    tokens[1, : step + 1] = true_tok
    scores[1, : step + 1, true_tok] = 1

    finished = torch.tensor([False, True, False, False], device=device)
    discarded = torch.tensor([False, False, False, False], device=device)

    model._cache_finished_beams(
        tokens,
        scores,
        step,
        finished & ~discarded,
        cache_tokens,
        cache_scores,
    )

    # Valid slots have non-zero raw probabilities; unwritten slots are 0.
    assert cache_scores[0, 1, step].any()
    assert torch.equal(cache_tokens[0, 1, step, : step + 1], true_tok)


@pytest.mark.skip(reason="No Beam Search in Non-AR Model")
def test_get_top_peptide_ranking(tiny_config):
    """Select the highest-scoring peptide from unsorted cached beams."""
    config = Config(tiny_config)
    model = Spec2Pep(
        tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
            residues=config.residues
        )
    )

    batch = 1
    beam = 3
    length = 10
    step = 3
    device = model.device

    cache_tokens = torch.full(
        (batch, beam, length, length), 0, dtype=torch.int32, device=device
    )
    cache_scores = torch.full(
        (batch, beam, length, length), 0.0, dtype=torch.float32, device=device
    )

    seqs = ["PEPY", "PEPK", "PEPP"]
    raw_probs = [0.93, 0.95, 0.94]
    for b, (seq, raw) in enumerate(zip(seqs, raw_probs)):
        toks = model.tokenizer.tokenize(seq)[0]
        n_toks = toks.shape[0]
        cache_tokens[0, b, step, :n_toks] = toks
        cache_scores[0, b, step, :n_toks] = raw

    result = next(model._get_top_peptide(cache_tokens, cache_scores))
    assert result[0][2] == "PEPK"


@pytest.mark.skip(reason="No Beam Search in Non-AR Model")
@pytest.mark.parametrize("topk", [1, 2, 3])
def test_get_top_peptide_multiple(topk, tiny_config):
    """Return the requested number of cached peptides in score order."""
    config = Config(tiny_config)
    model = Spec2Pep(
        top_match=topk,
        tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
            residues=config.residues
        ),
    )

    batch = 1
    beam = 3
    length = 10
    step = 3
    device = model.device

    cache_tokens = torch.full(
        (batch, beam, length, length), 0, dtype=torch.int32, device=device
    )
    cache_scores = torch.full(
        (batch, beam, length, length), 0.0, dtype=torch.float32, device=device
    )

    cache_items = [(0.9, "PEPY"), (0.8, "PEPK"), (0.7, "PEPP")]
    for b, (score, seq) in enumerate(cache_items):
        toks = model.tokenizer.tokenize(seq)[0]
        n_toks = toks.shape[0]
        cache_tokens[0, b, step, :n_toks] = toks
        cache_scores[0, b, step, :n_toks] = score

    result = next(model._get_top_peptide(cache_tokens, cache_scores))
    assert len(result) == topk
    for i in range(topk):
        assert result[i][2] == cache_items[i][1]


@pytest.mark.skip(reason="No Beam Search in Non-AR Model")
@pytest.mark.parametrize("reverse", [True, False])
def test_get_top_peptide_reverse(reverse, tiny_config):
    config = Config(tiny_config)
    model = Spec2Pep(
        tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
            residues=config.residues
        )
    )

    model.tokenizer.reverse = reverse

    batch = 1
    beam = 1
    length = 10
    step = 1
    device = model.device

    cache_tokens = torch.full(
        (batch, beam, length, length), 0, dtype=torch.int32, device=device
    )
    cache_scores = torch.full(
        (batch, beam, length, length), 0.0, dtype=torch.float32, device=device
    )

    toks = model.tokenizer.tokenize("PE")[0]
    n_toks = toks.shape[0]
    cache_tokens[0, 0, step, :n_toks] = toks
    cache_scores[0, 0, step, :n_toks] = 1.0

    result = list(model._get_top_peptide(cache_tokens, cache_scores))

    assert len(result) == 1
    assert len(result[0]) == 1
    assert isinstance(result[0][0][2], str)  # string peptide


@pytest.mark.skip(reason="No Beam Search in Non-AR Model")
def test_get_topk_beams(tiny_config):
    config = Config(tiny_config)
    model = Spec2Pep(
        n_beams=1,
        tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
            residues=config.residues
        ),
    )
    batch = 2
    step = 4
    beam = model.n_beams
    device = model.device

    model._batch_size = batch
    model._beam_size = beam
    model._cumulative_masses = torch.zeros(batch * beam, device=device)

    length = model.max_peptide_len + 1
    vocab = len(model.tokenizer) + 1

    scores = torch.zeros((batch * beam, length, vocab), device=device)
    tokens = torch.zeros(
        (batch * beam, length), dtype=torch.int64, device=device
    )

    scores[:, step, :4] = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    finished = torch.tensor([True, False], device=device)
    new_tokens, _ = model._get_topk_beams(
        tokens, scores, finished, batch, step
    )

    # only second beam should extend
    assert not torch.equal(new_tokens[0], new_tokens[1])


@pytest.mark.skip(reason="No Beam Search in Non-AR Model")
def test_beam_search_decode_early_termination(tiny_config):
    config = Config(tiny_config)
    model = Spec2Pep(
        n_beams=1,
        tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
            residues=config.residues
        ),
    )

    model.max_peptide_len = 0
    mzs = ints = torch.zeros(1, 5, device=model.device)
    precursors = torch.tensor(
        [[469.25364, 2.0, 235.63410]], device=model.device
    )

    assert len(list(model.beam_search_decode(mzs, ints, precursors))[0]) == 0


@pytest.mark.skip(reason="No Beam Search in Non-AR Model")
def test_duplicate_peptide_scores(tiny_config):
    """Preserve equal peptide scores when caching finished beams."""
    config = Config(tiny_config)
    model = Spec2Pep(
        n_beams=1,
        min_peptide_len=3,
        tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
            residues=config.residues
        ),
    )

    batch = 2
    beam = model.n_beams
    device = model.device

    model._batch_size = batch
    model._beam_size = beam
    model._cumulative_masses = torch.zeros(batch * beam, device=device)

    length = model.max_peptide_len + 1
    vocab = len(model.tokenizer) + 1
    step = 4

    scores = torch.zeros((batch * beam, length, vocab), device=device)
    scores[:, : step + 1, :] = torch.rand(1, device=device)

    tokens = torch.zeros(
        (batch * beam, length), dtype=torch.int64, device=device
    )
    tokens[:, :step] = torch.randint(
        1, vocab, (batch * beam, step), device=device
    )

    cache_tokens = torch.full(
        (batch, beam, length, length), 0, dtype=torch.int32, device=device
    )
    cache_scores = torch.full(
        (batch, beam, length, length), 0.0, dtype=scores.dtype, device=device
    )

    model._cache_finished_beams(
        tokens,
        scores,
        step,
        torch.ones(batch * beam, dtype=torch.bool, device=device),
        cache_tokens,
        cache_scores,
    )

    # All beams used identical scores, so every cached candidate should have
    # the same peptide score.
    for i in range(batch):
        cached_scores = cache_scores[i, :, step, : step + 1]
        cached_mask = cached_scores.any(dim=-1)
        vals = cached_scores[cached_mask].sum(dim=-1)
        assert cached_mask.sum() == beam
        assert torch.allclose(vals, vals[0])


def test_db_spec2pep_forward_no_cache(tiny_config):
    """Scoring a candidate needs the cache, so refuse without it.

    Upstream fell back to `_forward_step`, which was equivalent there
    because it fed the candidate to the decoder. This decoder takes
    blank frames instead, so that fallback would have scored every
    candidate of a spectrum identically, against frames that never saw
    it.
    """
    tokenizer = depthcharge.tokenizers.peptides.PeptideTokenizer(
        residues=Config(tiny_config).residues
    )
    db_model = DbSpec2Pep(tokenizer=tokenizer)

    # A batch without the precomputed encoder outputs.
    mock_batch = {
        "mz_array": torch.zeros((1, 10)),
        "intensity_array": torch.zeros((1, 10)),
        "precursor_mz": torch.tensor([42.0]),
        "precursor_charge": torch.tensor([1]),
        "seq": torch.randint(1, 20, (1, 8)),
    }

    with pytest.raises(ValueError, match="cached encoder output"):
        db_model.forward(mock_batch)


def test_validation_step_logs_per_file():
    """validation_step logs per-file CTCLoss; aggregate only for main loaders."""
    tokenizer = depthcharge.tokenizers.peptides.MskbPeptideTokenizer()
    model = Spec2Pep(
        n_beams=1,
        residues="massivekb",
        min_peptide_len=4,
        tokenizer=tokenizer,
    )
    model.val_stems = ["a", "b"]
    model.n_main_loaders = 1

    batch = {
        "mz_array": torch.zeros(1, 5),
        "intensity_array": torch.zeros(1, 5),
        "precursor_mz": torch.tensor(235.63410),
        "precursor_charge": torch.tensor(2),
        "seq": tokenizer.tokenize(["PEPK"]),
    }

    # dataloader_idx=0 is a main loader: logs per-file AND aggregate.
    with unittest.mock.patch.object(model, "log") as mock_log:
        model.validation_step(batch, 0, dataloader_idx=0)
    logged = {c.args[0]: c.kwargs for c in mock_log.call_args_list}
    assert "valid_CTCLoss/a" in logged
    assert "valid_CTCLoss" in logged
    assert logged["valid_CTCLoss/a"]["add_dataloader_idx"] is False
    assert logged["valid_CTCLoss"]["add_dataloader_idx"] is False

    # dataloader_idx=1 is a tracking loader: logs per-file only.
    with unittest.mock.patch.object(model, "log") as mock_log:
        model.validation_step(batch, 0, dataloader_idx=1)
    logged = {c.args[0]: c.kwargs for c in mock_log.call_args_list}
    assert "valid_CTCLoss/b" in logged
    assert "valid_CTCLoss" not in logged


def test_optimizer_kwargs_exclude_non_adam():
    """Non-optimizer config values must not reach the Adam optimizer."""
    # `enzyme` stands in for any configuration value that is not an
    # optimizer argument, such as one loaded from a checkpoint.
    model = Spec2Pep(lr=1e-3, weight_decay=1e-5, enzyme="trypsin")
    assert model.opt_kwargs == {"lr": 1e-3, "weight_decay": 1e-5}
    optimizers, _ = model.configure_optimizers()
    assert isinstance(optimizers[0], MuonWithAuxAdamW)
    assert all("enzyme" not in group for group in optimizers[0].param_groups)
