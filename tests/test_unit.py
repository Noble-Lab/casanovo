import os
import platform
import tempfile

import github
import pytest
import torch

from casanovo import casanovo
from casanovo import utils
from casanovo.denovo.model import Spec2Pep


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

    # Impossible to find model weights for non-matching version.
    with monkeypatch.context() as mnk:
        mnk.setattr(casanovo, "__version__", "999.999.999")
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
    Test that tensorboard.SummaryWriter object is created only when folder path is passed
    """
    model = Spec2Pep(tb_summarywriter="test_path")
    assert model.tb_summarywriter is not None

    model = Spec2Pep()
    assert model.tb_summarywriter is None


def test_beam_search_decode():
    """
    Test that beam search decoding and its sub-functions as intended during inference
    """
    model = Spec2Pep(n_beams=4)

    # Sizes:
    batch = 1  # B
    length = model.max_length + 1  # L
    vocab = model.decoder.vocab_size + 1  # V
    beam = model.n_beams  # S
    idx = 4

    # Initialize scores and tokens:
    scores = torch.zeros(batch, length, vocab, beam)
    scores[scores == 0] = torch.nan

    # Ground truth peptide is "PEPK"
    precursors = torch.tensor([469.2536487, 2.0, 235.63410081688]).repeat(
        beam * batch, 1
    )
    tokens = torch.zeros(batch * beam, length).long()

    tokens[0, :4] = torch.tensor(
        [
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["E"],
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["K"],
        ]
    )
    tokens[1, :4] = torch.tensor(
        [
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["E"],
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["R"],
        ]
    )
    tokens[2, :4] = torch.tensor(
        [
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["E"],
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["G"],
        ]
    )
    tokens[3, :4] = torch.tensor(
        [
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["E"],
            model.decoder._aa2idx["P"],
            model.decoder._aa2idx["$"],
        ]
    )

    # Test _terminate_finished_beams()
    finished_beams_idx, updated_tokens = model._terminate_finished_beams(
        tokens=tokens, precursors=precursors, idx=idx
    )

    assert finished_beams_idx == [0, 1, 3]
    assert torch.equal(
        updated_tokens[:, 4],
        torch.tensor([model.stop_token, model.stop_token, 0, 0]),
    )

    # Test _create_beamsearch_cache() and _cache_finished_beams()
    tokens = torch.zeros(batch, length, beam).long()

    (
        cache_scores,
        cache_tokens,
        cache_idx_dict,
        cache_seq_dict,
        cache_score_dict,
    ) = model._create_beamsearch_cache(scores, tokens)

    scores = cache_scores.clone()
    for i in range(idx):
        scores[:, i, :] = 1
        scores[1, i, updated_tokens[1, i].item()] = 2

    (
        cache_idx_dict,
        cache_seq_dict,
        cache_score_dict,
    ) = model._cache_finished_beams(
        finished_beams_idx,
        cache_idx_dict,
        cache_seq_dict,
        cache_score_dict,
        cache_tokens,
        cache_scores,
        updated_tokens,
        precursors,
        scores,
        idx,
    )

    assert cache_idx_dict[0] == 3
    assert cache_seq_dict[0] == set(["PEPK", "PEPR", "PEP"])
    # Check if precursor fitting and non-fitting peptides cached correctly
    assert len(cache_score_dict[0][0]) == 1
    assert len(cache_score_dict[0][1]) == 2

    # Test _get_top_peptide()
    output_tokens, output_scores = model._get_top_peptide(
        cache_score_dict, cache_tokens, cache_scores, batch
    )

    # Check if output equivalent to "PEPK"
    assert torch.equal(output_tokens[0], cache_tokens[0])

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
            [4, 4, 4, 4],
            [14, 14, 14, 14],
            [4, 4, 4, 4],
            [1, 1, 1, 1],
            [4, 3, 2, 1],
        ]
    )

    expected_scores = torch.ones(vocab, beam)

    for i in range(1, 5):
        expected_scores[i] = i + 1

    assert torch.equal(new_tokens[0][: idx + 1, :], expected_tokens)
    assert torch.equal(new_scores[0][idx, :], expected_scores)

    # Test beam_search_decode()
    spectra = torch.zeros(1, 5, 2)
    precursors = torch.tensor([[469.2536487, 2.0, 235.63410081688]])
    scores, tokens = model.beam_search_decode(spectra, precursors)

    assert tokens.shape[0] == 1
    assert model.stop_token in tokens
