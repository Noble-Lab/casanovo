import os
import platform
import tempfile

import einops
import github
import numpy as np
import pytest
import torch

from casanovo import casanovo
from casanovo import utils
from casanovo.denovo.evaluate import aa_match_batch, aa_match_metrics
from casanovo.denovo.model import Spec2Pep, _aa_pep_score


def test_version():
    """Check that the version is not None."""
    assert casanovo.__version__ is not None


def test_n_workers(monkeypatch):
    """Check that n_workers is correct without a GPU."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    cpu_fun = lambda x: ["foo"] * 31

    with monkeypatch.context() as mnk:
        mnk.setattr("psutil.Process.cpu_affinity", cpu_fun, raising=False)
        expected = 0 if platform.system() in ["Windows", "Darwin"] else 31
        assert utils.n_workers() == expected

    with monkeypatch.context() as mnk:
        mnk.delattr("psutil.Process.cpu_affinity", raising=False)
        mnk.setattr("os.cpu_count", lambda: 41)
        expected = 0 if platform.system() in ["Windows", "Darwin"] else 41
        assert utils.n_workers() == expected

    with monkeypatch.context() as mnk:
        mnk.setattr("torch.cuda.device_count", lambda: 4)
        mnk.setattr("psutil.Process.cpu_affinity", cpu_fun, raising=False)
        expected = 0 if platform.system() in ["Windows", "Darwin"] else 7
        assert utils.n_workers() == expected

    with monkeypatch.context() as mnk:
        mnk.delattr("psutil.Process.cpu_affinity", raising=False)
        mnk.delattr("os.cpu_count")
        if platform.system() not in ["Windows", "Darwin"]:
            with pytest.raises(AttributeError):
                utils.n_workers()
        else:
            assert utils.n_workers() == 0


def test_split_version():
    """Test that splitting the version number works as expected."""
    version = utils.split_version("2.0.1")
    assert version == ("2", "0", "1")

    version = utils.split_version("0.1.dev1+g39f8c53")
    assert version == ("0", "1", "")

    version = utils.split_version("3.0.1.dev10282blah")
    assert version == ("3", "0", "1")


def test_get_model_weights(monkeypatch):
    """
    Test that model weights can be downloaded from GitHub or used from the
    cache.
    """
    # Model weights for fully matching version, minor matching version, major
    # matching version.
    for version in ["3.0.0", "3.0.999", "3.999.999"]:
        with monkeypatch.context() as mnk, tempfile.TemporaryDirectory() as tmp_dir:
            mnk.setattr(casanovo, "__version__", version)
            mnk.setattr(
                "appdirs.user_cache_dir", lambda n, a, opinion: tmp_dir
            )

            filename = os.path.join(tmp_dir, "casanovo_massivekb_v3_0_0.ckpt")
            assert not os.path.isfile(filename)
            assert casanovo._get_model_weights() == filename
            assert os.path.isfile(filename)
            assert casanovo._get_model_weights() == filename

    # Impossible to find model weights for (i) full version mismatch and (ii)
    # major version mismatch.
    for version in ["999.999.999", "999.0.0"]:
        with monkeypatch.context() as mnk:
            mnk.setattr(casanovo, "__version__", version)
            with pytest.raises(ValueError):
                casanovo._get_model_weights()

    # Test GitHub API rate limit.
    def request(self, *args, **kwargs):
        raise github.RateLimitExceededException(
            403, "API rate limit exceeded", None
        )

    with monkeypatch.context() as mnk, tempfile.TemporaryDirectory() as tmp_dir:
        mnk.setattr("appdirs.user_cache_dir", lambda n, a, opinion: tmp_dir)
        mnk.setattr("github.Requester.Requester.requestJsonAndCheck", request)
        with pytest.raises(github.RateLimitExceededException):
            casanovo._get_model_weights()


def test_tensorboard():
    """
    Test tensorboard.SummaryWriter object created only when folder path passed
    """
    model = Spec2Pep(tb_summarywriter="test_path")
    assert model.tb_summarywriter is not None

    model = Spec2Pep()
    assert model.tb_summarywriter is None


def test_aa_pep_score():
    """
    Test the calculation of amino acid and peptide scores from the raw amino
    acid scores.
    """
    aa_scores_raw = np.asarray([0.0, 0.5, 1.0])

    aa_scores, peptide_score = _aa_pep_score(aa_scores_raw, True)
    np.testing.assert_array_equal(aa_scores, np.asarray([0.25, 0.5, 0.75]))
    assert peptide_score == pytest.approx(0.5)

    aa_scores, peptide_score = _aa_pep_score(aa_scores_raw, False)
    np.testing.assert_array_equal(aa_scores, np.asarray([0.25, 0.5, 0.75]))
    assert peptide_score == pytest.approx(-0.5)


def test_beam_search_decode():
    """
    Test beam search decoding and its sub-functions.
    """
    model = Spec2Pep(n_beams=4, residues="massivekb")
    model.decoder.reverse = False  # For simplicity.
    aa2idx = model.decoder._aa2idx

    # Sizes.
    batch = 1  # B
    length = model.max_length + 1  # L
    vocab = model.decoder.vocab_size + 1  # V
    beam = model.n_beams  # S
    step = 3

    # Initialize scores and tokens.
    scores = torch.full(
        size=(batch, length, vocab, beam), fill_value=torch.nan
    )
    scores = einops.rearrange(scores, "B L V S -> (B S) L V")
    tokens = torch.zeros(batch * beam, length, dtype=torch.int64)
    # Create cache for decoded beams.
    pred_cache = {i: [] for i in range(batch)}
    finished_beams = -torch.ones(batch * beam, dtype=torch.int8)
    beam_fits_precursor = torch.zeros(batch * beam, dtype=torch.bool)

    # Ground truth peptide is "PEPK".
    true_peptide = "PEPK"
    precursors = torch.tensor([469.25364, 2.0, 235.63410]).repeat(
        beam * batch, 1
    )
    # Fill scores and tokens with relevant predictions.
    scores[:, : step + 1, :] = 0
    for i, peptide in enumerate(["PEPK", "PEPR", "PEPG", "PEP$"]):
        tokens[i, : step + 1] = torch.tensor([aa2idx[aa] for aa in peptide])
        for j in range(step + 1):
            scores[i, j, tokens[1, j]] = 1

    # Test _finish_beams().
    model._finish_beams(
        tokens, precursors, step, finished_beams, beam_fits_precursor
    )
    # First two beams finished due to the precursor m/z filter, final beam
    # finished due to predicted stop token, third beam unfinished.
    assert torch.equal(finished_beams, torch.tensor([step, step, -1, step]))
    assert torch.equal(
        beam_fits_precursor, torch.tensor([True, False, False, False])
    )

    # Test _cache_finished_beams().
    model._cache_finished_beams(
        tokens, scores, step, finished_beams, beam_fits_precursor, pred_cache
    )
    # Verify that the correct peptides have been cached.
    correct_cached = 0
    for _, _, pep in pred_cache[0]:
        if torch.equal(pep, torch.tensor([4, 14, 4, 13])):
            correct_cached += 1
        elif torch.equal(pep, torch.tensor([4, 14, 4, 18])):
            correct_cached += 1
        elif torch.equal(pep, torch.tensor([4, 14, 4])):
            correct_cached += 1
        else:
            pytest.fail(
                "Unexpected peptide tensor in the finished beams cache"
            )
    assert correct_cached == 3

    # Test _get_top_peptide().
    assert model._get_top_peptide(pred_cache)[0][-1] == true_peptide
    # Test that an empty predictions is returned when no beams have been
    # finished.
    empty_cache = {i: [] for i in range(batch)}
    assert model._get_top_peptide(empty_cache)[0][-1] == ""

    # Test _get_topk_beams().
    # FIXME
    # # Set scores to proceed generating the unfinished beam.
    # scores[2, step + 1, :] = 0
    # scores[2, step + 1, range(4)] = torch.tensor([1., 2., 3., 4.])
    # new_tokens, new_scores = model._get_topk_beams(tokens, scores, batch, step)
    # expected_tokens = torch.tensor(
    #     [
    #         [4, 14, 4, 1, 4],
    #         [4, 14, 4, 1, 3],
    #         [4, 14, 4, 1, 2],
    #         [4, 14, 4, 1, 1],
    #     ]
    # )
    # # Only the expected scores of the final step.
    # expected_scores = torch.zeros(beam, vocab)
    # expected_scores[:, range(4)] = torch.tensor([1., 2., 3., 4.])
    #
    # assert torch.equal(new_tokens[:, :step + 1], expected_tokens)
    # assert torch.equal(new_scores[:, step + 1, :], expected_scores)

    # Test output if decoding loop isn't stopped with termination of all beams.
    model.max_length = 0
    # 1 spectrum with 5 peaks (2 values: m/z and intensity).
    spectra = torch.zeros(1, 5, 2)
    precursors = torch.tensor([[469.25364, 2.0, 235.63410]])
    assert model.beam_search_decode(spectra, precursors)[0][-1] == ""
    model.max_length = 100

    step = 4

    # Re-initialize scores and tokens to further test caching functionality.
    scores = torch.full(
        size=(batch, length, vocab, beam), fill_value=torch.nan
    )
    scores = einops.rearrange(scores, "B L V S -> (B S) L V")
    tokens = torch.zeros(batch * beam, length, dtype=torch.int64)

    scores[:, : step + 1, :] = 0
    for i, peptide in enumerate(["PKKP$", "EPPK$", "PEPK$", "PMKP$"]):
        tokens[i, : step + 1] = torch.tensor([aa2idx[aa] for aa in peptide])
    i, j, s = np.arange(step), np.arange(4), torch.Tensor([4, 0.5, 3, 0.4])
    scores[:, i, :] = 1
    scores[j, i, tokens[j, i]] = s

    pred_cache = {i: [] for i in range(batch)}
    finished_beams = torch.tensor([step, step, step, step])
    beam_fits_precursor = torch.BoolTensor([False, True, True, False])

    model._cache_finished_beams(
        tokens, scores, step, finished_beams, beam_fits_precursor, pred_cache
    )
    # Verify predictions with matching/non-matching precursor m/z.
    positive_score = negative_score = 0
    for peptide_score, _, _ in pred_cache[0]:
        positive_score += peptide_score >= 0
        negative_score += peptide_score < 0
    assert positive_score == 2
    assert negative_score == 2

    # Test using a single beam only.
    model = Spec2Pep(n_beams=1)
    beam = model.n_beams  # S
    step = 3

    # Initialize scores and tokens.
    scores = torch.full(
        size=(batch, length, vocab, beam), fill_value=torch.nan
    )
    scores = einops.rearrange(scores, "B L V S -> (B S) L V")
    tokens = torch.zeros(batch * beam, length, dtype=torch.int64)

    pred_cache = {i: [] for i in range(batch)}
    finished_beams = -torch.ones(batch * beam, dtype=torch.int8)
    beam_fits_precursor = torch.zeros(batch * beam, dtype=torch.bool)

    # Ground truth peptide is "PEPK".
    true_peptide = "PEPK"
    precursors = torch.tensor([469.25364, 2.0, 235.63410]).repeat(
        beam * batch, 1
    )
    scores[:, range(step), :] = 1
    tokens[0, : step + 1] = torch.tensor([aa2idx[aa] for aa in true_peptide])

    # Test _finish_beams().
    model._finish_beams(
        tokens, precursors, step, finished_beams, beam_fits_precursor
    )
    assert torch.equal(finished_beams, torch.tensor([step]))

    # Test _cache_finished_beams().
    model._cache_finished_beams(
        tokens, scores, step, finished_beams, beam_fits_precursor, pred_cache
    )
    assert torch.equal(pred_cache[0][0][-1], torch.tensor([4, 14, 4, 13]))

    # Test _finish_beams() for tokens with a negative mass.
    model = Spec2Pep(n_beams=2, residues="massivekb")
    beam = model.n_beams  # S
    step = 1
    finished_beams = -torch.ones(batch * beam, dtype=torch.int8)
    beam_fits_precursor = torch.zeros(batch * beam, dtype=torch.bool)

    # Ground truth peptide is "-17.027GK".
    precursors = torch.tensor([186.10044, 2.0, 94.05750]).repeat(
        beam * batch, 1
    )
    tokens = torch.zeros(batch * beam, length, dtype=torch.int64)
    for i, peptide in enumerate(["GK", "AK"]):
        tokens[i, : step + 1] = torch.tensor([aa2idx[aa] for aa in peptide])

    # Test _finish_beams().
    model._finish_beams(
        tokens, precursors, step, finished_beams, beam_fits_precursor
    )
    assert torch.equal(finished_beams, torch.tensor([-1, step]))


def test_eval_metrics():
    """
    Test peptide and amino acid-level evaluation metrics.
    Predicted AAs are considered correct if they are <0.1Da from the
    corresponding ground truth (GT) AA with either a suffix or prefix <0.5Da
    from GT. A peptide prediction is correct if all its AA are correct matches.
    """
    model = Spec2Pep()

    preds = [
        "SPEIK",
        "SPAEL",
        "SPAEKL",
        "ASPEKL",
        "SPEKL",
        "SPELQ",
        "PSEKL",
        "SPEK",
    ]
    gt = len(preds) * ["SPELK"]

    aa_matches, n_pred_aa, n_gt_aa = aa_match_batch(
        peptides1=preds,
        peptides2=gt,
        aa_dict=model.decoder._peptide_mass.masses,
        mode="best",
    )

    assert n_pred_aa == 41
    assert n_gt_aa == 40

    aa_precision, aa_recall, pep_precision = aa_match_metrics(
        aa_matches, n_gt_aa, n_pred_aa
    )

    assert 2 / 8 == pytest.approx(pep_precision)
    assert 26 / 40 == pytest.approx(aa_recall)
    assert 26 / 41 == pytest.approx(aa_precision)
