import os
import platform
import tempfile

import github
import numpy as np
import pytest
import torch

from casanovo import casanovo
from casanovo import utils
from casanovo.denovo.evaluate import aa_match_batch, aa_match_metrics
from casanovo.denovo.model import Spec2Pep, _aa_to_pep_score


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


def test_aa_to_pep_score():
    """
    Test how peptide confidence scores are derived from amino acid scores.
    Currently, AA scores are just averaged.
    """
    assert (
        _aa_to_pep_score(
            [
                0.0,
                0.5,
                1.0,
            ]
        )
        == 0.5
    )


def test_beam_search_decode():
    """
    Test beam search decoding and its sub-functions
    """
    model = Spec2Pep(n_beams=4, residues="massivekb")

    # Sizes.
    batch = 1  # B
    length = model.max_length + 1  # L
    vocab = model.decoder.vocab_size + 1  # V
    beam = model.n_beams  # S
    idx = 4

    # Initialize scores and tokens.
    scores = torch.full(
        size=(batch, length, vocab, beam), fill_value=torch.nan
    )
    is_beam_prec_fit = torch.zeros(batch * beam, dtype=torch.bool)

    # Ground truth peptide is "PEPK".
    precursors = torch.tensor([469.2536487, 2.0, 235.63410081688]).repeat(
        beam * batch, 1
    )
    tokens = torch.zeros(batch * beam, length).long()

    tokens[0, :idx] = torch.tensor(
        [
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["E"],
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["K"],
        ]
    )
    tokens[1, :idx] = torch.tensor(
        [
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["E"],
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["R"],
        ]
    )
    tokens[2, :idx] = torch.tensor(
        [
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["E"],
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["G"],
        ]
    )
    tokens[3, :idx] = torch.tensor(
        [
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["E"],
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["$"],
        ]
    )

    # Test _terminate_finished_beams().
    finished_beams_idx, updated_tokens = model._terminate_finished_beams(
        tokens=tokens,
        precursors=precursors,
        is_beam_prec_fit=is_beam_prec_fit,
        idx=idx,
    )

    assert torch.equal(finished_beams_idx, torch.tensor([0, 1, 3]))
    assert torch.equal(
        updated_tokens[:, idx],
        torch.tensor([model.stop_token, model.stop_token, 0, 0]),
    )

    # Test _create_beamsearch_cache() and _cache_finished_beams().
    tokens = torch.zeros(batch, length, beam).long()

    (
        cache_scores,
        cache_tokens,
        cache_next_idx,
        cache_pred_seq,
        cache_pred_score,
    ) = model._create_beamsearch_cache(scores, tokens)

    scores = cache_scores.clone()
    for i in range(idx):
        scores[:, i, :] = 1
        scores[1, i, updated_tokens[1, i].item()] = 2

    model._cache_finished_beams(
        finished_beams_idx,
        cache_next_idx,
        cache_pred_seq,
        cache_pred_score,
        cache_tokens,
        cache_scores,
        updated_tokens,
        scores,
        is_beam_prec_fit,
        idx,
    )

    assert cache_next_idx[0] == 3
    # Keep track of peptides that should be in cache.
    correct_pep = 0
    for pep in cache_pred_seq[0]:
        correct_pep += (
            torch.equal(pep, torch.tensor([4, 14, 4, 13]))
            or torch.equal(pep, torch.tensor([4, 14, 4, 18]))
            or torch.equal(pep, torch.tensor([4, 14, 4]))
        )
    assert correct_pep == 3
    # Check if precursor fitting and non-fitting peptides cached correctly.
    assert len(cache_pred_score[0][0]) == 1
    assert len(cache_pred_score[0][1]) == 2

    # Test _get_top_peptide().
    output_tokens, output_scores = model._get_top_peptide(
        cache_pred_score, cache_tokens, cache_scores, batch
    )

    # Check if output equivalent to "PEPK".
    assert torch.equal(output_tokens[0], cache_tokens[0])

    # If no peptides are finished
    dummy_cache_pred_score = {0: [[], []]}

    dummy_output_tokens, dummy_output_scores = model._get_top_peptide(
        dummy_cache_pred_score, cache_tokens, cache_scores, batch
    )

    # Check if output equivalent to zero tensor
    assert sum(dummy_output_tokens[0]).item() == 0

    # Test _get_topk_beams()
    # Generate scores for the non-terminated beam
    scores[2, idx, :] = 1

    for i in range(1, 5):
        scores[2, idx, i] = i + 1

    new_scores, new_tokens = model._get_topk_beams(
        scores=scores, tokens=updated_tokens, batch=batch, idx=idx
    )

    expected_tokens = torch.tensor(
        [
            [4, 14, 4, 1, 4],
            [4, 14, 4, 1, 3],
            [4, 14, 4, 1, 2],
            [4, 14, 4, 1, 1],
        ]
    )

    expected_scores = torch.ones(beam, vocab)

    for i in range(1, 5):
        expected_scores[:, i] = i + 1

    assert torch.equal(new_tokens[:, : idx + 1], expected_tokens)
    assert torch.equal(new_scores[:, idx, :], expected_scores)

    # Test beam_search_decode().
    spectra = torch.zeros(1, 5, 2)
    precursors = torch.tensor([[469.2536487, 2.0, 235.63410081688]])
    model_scores, model_tokens = model.beam_search_decode(spectra, precursors)

    assert model_tokens.shape[0] == 1
    assert model.stop_token in model_tokens

    # Test output if decoding loop isn't stopped with termination of all beams
    model.max_length = 0
    model_scores, model_tokens = model.beam_search_decode(spectra, precursors)
    assert torch.equal(model_tokens, torch.tensor([[0]]))
    model.max_length = 100

    # Re-initialize scores and tokens to further test caching functionality.
    scores_v2 = torch.full(
        size=(batch * beam, length, vocab), fill_value=torch.nan
    )
    tokens_v2 = torch.zeros(batch * beam, length).long()

    tokens_v2[0, : idx + 1] = torch.tensor(
        [
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["K"],
            model.decoder._aa2idx["K"],
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["$"],
        ]
    )
    tokens_v2[1, : idx + 1] = torch.tensor(
        [
            model.decoder._aa2idx["E"],
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["K"],
            model.decoder._aa2idx["$"],
        ]
    )
    tokens_v2[2, : idx + 1] = torch.tensor(
        [
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["E"],
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["R"],
            model.decoder._aa2idx["$"],
        ]
    )
    tokens_v2[3, : idx + 1] = torch.tensor(
        [
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["M"],
            model.decoder._aa2idx["K"],
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["$"],
        ]
    )

    # Test if fitting replaces non-fitting in the cache and only higher scoring
    # non-fitting replaces non-fitting.
    for i in range(idx + 1):
        scores_v2[:, i, :] = 1
        scores_v2[0, i, tokens_v2[0, i].item()] = 4
        scores_v2[1, i, tokens_v2[1, i].item()] = 0.5
        scores_v2[2, i, tokens_v2[2, i].item()] = 3
        scores_v2[3, i, tokens_v2[3, i].item()] = 0.4

    finished_beams_idx_v2 = torch.tensor([0, 1, 2, 3])
    is_beam_prec_fit_v2 = torch.BoolTensor([False, True, False, False])

    model._cache_finished_beams(
        finished_beams_idx_v2,
        cache_next_idx,
        cache_pred_seq,
        cache_pred_score,
        cache_tokens,
        cache_scores,
        tokens_v2,
        scores_v2,
        is_beam_prec_fit_v2,
        idx + 1,
    )

    assert cache_next_idx[0] == 4
    # Check if precursor fitting and non-fitting peptides cached correctly.
    assert len(cache_pred_score[0][0]) == 2
    assert len(cache_pred_score[0][1]) == 2

    # Keep track of peptides that should (not) be in cache.
    correct_pep = 0
    wrong_pep = 0

    for pep in cache_pred_seq[0]:
        if (
            torch.equal(pep, torch.tensor([4, 13, 13, 4]))
            or torch.equal(pep, torch.tensor([14, 4, 4, 13]))
            or torch.equal(pep, torch.tensor([4, 14, 4, 18]))
        ):
            correct_pep += 1
        elif torch.equal(pep, torch.tensor([4, 15, 13, 4])):
            wrong_pep += 1
    assert correct_pep == 3
    assert wrong_pep == 0

    # Test for a single beam.
    model = Spec2Pep(n_beams=1)

    # Sizes.
    batch = 1  # B
    length = model.max_length + 1  # L
    vocab = model.decoder.vocab_size + 1  # V
    beam = model.n_beams  # S
    idx = 4

    # Initialize scores and tokens.
    scores = torch.full(
        size=(batch, length, vocab, beam), fill_value=torch.nan
    )
    is_beam_prec_fit = torch.zeros(batch * beam, dtype=torch.bool)

    # Ground truth peptide is "PEPK"
    precursors = torch.tensor([469.2536487, 2.0, 235.63410081688]).repeat(
        beam * batch, 1
    )
    tokens = torch.zeros(batch * beam, length).long()

    tokens[0, :idx] = torch.tensor(
        [
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["E"],
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["K"],
        ]
    )

    # Test _terminate_finished_beams().
    finished_beams_idx, updated_tokens = model._terminate_finished_beams(
        tokens=tokens,
        precursors=precursors,
        is_beam_prec_fit=is_beam_prec_fit,
        idx=idx,
    )

    assert torch.equal(finished_beams_idx, torch.tensor([0]))
    assert torch.equal(
        updated_tokens[:, idx], torch.tensor([model.stop_token])
    )

    # Test _create_beamsearch_cache() and _cache_finished_beams().
    tokens = torch.zeros(batch, length, beam).long()

    (
        cache_scores,
        cache_tokens,
        cache_next_idx,
        cache_pred_seq,
        cache_pred_score,
    ) = model._create_beamsearch_cache(scores, tokens)

    scores = cache_scores.clone()
    for i in range(idx):
        scores[:, i, :] = 1

    model._cache_finished_beams(
        finished_beams_idx,
        cache_next_idx,
        cache_pred_seq,
        cache_pred_score,
        cache_tokens,
        cache_scores,
        updated_tokens,
        scores,
        is_beam_prec_fit,
        idx,
    )

    assert cache_next_idx[0] == 1
    # Keep track of peptides that should be in cache.
    correct_pep = 0
    for pep in cache_pred_seq[0]:
        correct_pep += torch.equal(pep, torch.tensor([4, 14, 4, 13]))
    assert correct_pep == 1
    # Check if precursor fitting and non-fitting peptides cached correctly.
    assert len(cache_pred_score[0][0]) == 1
    assert len(cache_pred_score[0][1]) == 0

    # Test _get_top_peptide().
    output_tokens, output_scores = model._get_top_peptide(
        cache_pred_score, cache_tokens, cache_scores, batch
    )

    # Check if output equivalent to "PEPK".
    assert torch.equal(output_tokens[0], cache_tokens[0])

    # Test _terminate_finished_beams for tokens with negative mass
    model = Spec2Pep(n_beams=2, residues="massivekb")
    # Sizes:
    batch = 1  # B
    length = model.max_length + 1  # L
    vocab = model.decoder.vocab_size + 1  # V
    beam = model.n_beams  # S
    idx = 2
    # Initialize scores and tokens:
    scores = torch.full(
        size=(batch, length, vocab, beam), fill_value=torch.nan
    )
    is_beam_prec_fit = (batch * beam) * [False]
    # Ground truth peptide is "-17.027GK"
    precursors = torch.tensor([186.100442485, 2.0, 94.05749770938]).repeat(
        beam * batch, 1
    )
    tokens = torch.zeros(batch * beam, length).long()
    tokens[0, :idx] = torch.tensor(
        [
            model.decoder._aa2idx["G"],
            model.decoder._aa2idx["K"],
        ]
    )

    tokens[1, :idx] = torch.tensor(
        [
            model.decoder._aa2idx["A"],
            model.decoder._aa2idx["K"],
        ]
    )
    # Test _terminate_finished_beams()
    finished_beams_idx, updated_tokens = model._terminate_finished_beams(
        tokens=tokens,
        precursors=precursors,
        is_beam_prec_fit=is_beam_prec_fit,
        idx=idx,
    )
    assert torch.equal(finished_beams_idx, torch.tensor([1]))
    assert torch.equal(
        updated_tokens[:, idx],
        torch.tensor([0, model.stop_token]),
    )


def test_get_output_peptide_and_scores():
    """
    Test output peptides and amino acid/peptide-level scores have correct format.
    """
    # Test a common case with reverse decoding (C- to N-terminus)
    model = Spec2Pep()
    aa_tokens = [model.decoder._idx2aa[model.stop_token], "G", "K"]
    aa_scores = torch.zeros(model.max_length, model.decoder.vocab_size + 1)
    aa_scores[0][model.decoder._aa2idx["K"]] = 1
    aa_scores[1][model.decoder._aa2idx["G"]] = 1

    (
        peptide,
        aa_tokens,
        peptide_score,
        aa_scores,
    ) = model._get_output_peptide_and_scores(aa_tokens, aa_scores)
    assert peptide == "GK"
    assert peptide_score == 1
    assert aa_scores == "1.00000,1.00000"

    # Test a case with straigth decoding (N- to C-terminus)
    model.decoder.reverse = False
    aa_tokens = ["G", "K", model.decoder._idx2aa[model.stop_token]]
    aa_scores = torch.zeros(model.max_length, model.decoder.vocab_size + 1)
    aa_scores[0][model.decoder._aa2idx["G"]] = 1
    aa_scores[1][model.decoder._aa2idx["K"]] = 1

    (
        peptide,
        aa_tokens,
        peptide_score,
        aa_scores,
    ) = model._get_output_peptide_and_scores(aa_tokens, aa_scores)
    assert peptide == "GK"
    assert peptide_score == 1
    assert aa_scores == "1.00000,1.00000"

    # Test when predicted peptide is empty
    aa_tokens = ["", ""]

    (
        peptide,
        aa_tokens,
        peptide_score,
        aa_scores,
    ) = model._get_output_peptide_and_scores(aa_tokens, aa_scores)
    assert peptide == ""
    assert np.isnan(peptide_score)
    assert aa_scores == ""


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
