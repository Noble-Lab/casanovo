import copy
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
        train_label_smoothing=0.1,
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

    # Check if smoothing is applied in training and not in validation
    assert model.celoss.label_smoothing == 0.1
    assert model.val_celoss.label_smoothing == 0
    assert val_step_loss != train_step_loss


def test_predict_step_counts_missing(tiny_config):
    config = Config(tiny_config)
    model = Spec2Pep(
        tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
            residues=config.residues
        ),
    )
    # One spectrum gets a prediction; the other returns no valid peptide.
    model.forward = unittest.mock.MagicMock(
        return_value=[[(0.9, np.array([0.9]), "PEPK")], []]
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
    """Test the DbSpec2Pep forward method without a cache."""
    tokenizer = depthcharge.tokenizers.peptides.PeptideTokenizer(
        residues=Config(tiny_config).residues
    )
    db_model = DbSpec2Pep(tokenizer=tokenizer)

    # Mock the _forward_step method to confirm it's called
    db_model._forward_step = unittest.mock.MagicMock(
        return_value=(torch.zeros(1, 5, 25), torch.zeros(1, 4))
    )

    # Create a batch without pre-computed encoder outputs
    mock_batch = {
        "mz_array": torch.zeros((1, 10)),
        "intensity_array": torch.zeros((1, 10)),
        "precursor_mz": torch.tensor([42.0]),
        "precursor_charge": torch.tensor([1]),
        "seq": torch.randint(1, 20, (1, 8)),
    }

    db_model.forward(mock_batch)

    # Assert that the non-cached path was taken
    db_model._forward_step.assert_called_once()


def test_validation_step_logs_per_file():
    """validation_step logs per-file CELoss; aggregate only for main loaders."""
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
    assert "valid_CELoss/a" in logged
    assert "valid_CELoss" in logged
    assert logged["valid_CELoss/a"]["add_dataloader_idx"] is False
    assert logged["valid_CELoss"]["add_dataloader_idx"] is False

    # dataloader_idx=1 is a tracking loader: logs per-file only.
    with unittest.mock.patch.object(model, "log") as mock_log:
        model.validation_step(batch, 0, dataloader_idx=1)
    logged = {c.args[0]: c.kwargs for c in mock_log.call_args_list}
    assert "valid_CELoss/b" in logged
    assert "valid_CELoss" not in logged


def test_optimizer_kwargs_exclude_non_adam():
    """Non-optimizer config values must not reach the Adam optimizer."""
    model = Spec2Pep(
        lr=1e-3,
        weight_decay=1e-5,
        precursor_mass_tol=50.0,
        isotope_error_range=(0, 1),
    )
    assert model.opt_kwargs == {"lr": 1e-3, "weight_decay": 1e-5}
    optimizers, _ = model.configure_optimizers()
    assert isinstance(optimizers[0], torch.optim.Adam)
