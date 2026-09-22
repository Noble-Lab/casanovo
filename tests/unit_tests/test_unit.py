import collections
import copy
import datetime
import functools
import hashlib
import heapq
import io
import logging
import math
import os
import pathlib
import platform
import re
import shutil
import tempfile
import types
import unittest
import unittest.mock

import depthcharge
import depthcharge.data
import depthcharge.tokenizers.peptides
import einops
import github
import numpy as np
import pandas as pd
import pytest
import requests
import torch

from casanovo import casanovo, denovo, utils
from casanovo.config import Config
from casanovo.data import db_utils, ms_io, psm
from casanovo.denovo.dataloaders import DeNovoDataModule
from casanovo.denovo.evaluate import aa_match, aa_match_batch, aa_match_metrics
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


def test_export(tiny_fasta_file, tmp_path):
    """Test for export feature in db search"""
    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="partial",
        missed_cleavages=0,
        min_peptide_len=6,
        max_peptide_len=100,
        max_mods=0,
        precursor_tolerance=20,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )

    pdb.export(tmp_path, "output")
    output_file = tmp_path / "output.tsv"
    ground_truth = pdb.db_peptides

    loaded = pd.read_csv(output_file, sep="\t", index_col=0)
    ground_truth["protein"] = ground_truth["protein"].apply(str)
    loaded["protein"] = loaded["protein"].astype(str)

    pd.testing.assert_frame_equal(ground_truth, loaded)


@pytest.mark.parametrize(
    "enzyme, specificity, expected, use_sort",
    [
        (
            "N/A",
            "non-specific",
            [
                ("", "foo"),
                ("M", "foo"),
                ("ME", "foo"),
                ("", "foo"),
                ("E", "foo"),
                ("", "corrupted"),
                ("M", "corrupted"),
                ("ME", "corrupted"),
                ("", "corrupted"),
                ("E", "corrupted"),
                ("", "corrupted"),
            ],
            False,
        ),
        (
            "trypsin",
            "partial",
            [
                ("M", "foo"),
                ("ME", "foo"),
                ("E", "foo"),
                ("ME", "corrupted"),
                ("M", "corrupted"),
            ],
            True,
        ),
        (
            ".Q.",
            "partial",
            [
                ("M", "foo"),
                ("ME", "foo"),
                ("E", "foo"),
                ("ME", "corrupted"),
                ("M", "corrupted"),
            ],
            True,
        ),
        (
            "trypsin",
            "full",
            [("ME", "foo")],
            True,
        ),
        (
            ".Q.",
            "full",
            [("ME", "foo")],
            True,
        ),
    ],
)
def test_digestion_with_unknown_amino_acids(
    enzyme, specificity, expected, use_sort, tmp_path
):
    """Test digestion with unknown amino acids"""
    fasta_path = tmp_path / "tiny_fasta.fasta"
    fasta_path.write_text(">foo\nME\n>corrupted\nMEX\n", encoding="utf-8")

    valid_aa = list("ARNDCEQGHILKMFPSTWYV")
    min_len = 0
    max_len = 18

    results = list(
        db_utils._peptide_generator(
            str(fasta_path),
            enzyme,
            specificity,
            0,
            min_len,
            max_len,
            valid_aa,
        )
    )

    result_peptides = [p for p, _ in results]
    expected_peptides = [p for p, _ in expected]

    if use_sort:
        assert sorted(result_peptides) == sorted(
            expected_peptides
        ), f"Failed for enzyme={enzyme}, specificity={specificity}"
    else:
        assert (
            result_peptides == expected_peptides
        ), f"Failed for enzyme={enzyme}, specificity={specificity}"


def test_version():
    """Check that the version is not None."""
    assert casanovo.__version__ is not None


@pytest.mark.skip(reason="Skipping due to Linux deadlock issue")
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


class MockAsset:
    def __init__(self, file_name):
        self.name = file_name
        self.browser_download_url = f"http://example.com/{file_name}"


class MockGithub:
    def __init__(self, releases):
        self.releases = releases

    def get_repo(self, repo_name):
        return MockRepo()


class MockRelease:
    def __init__(self, tag_name, assets):
        self.tag_name = tag_name
        self.assets = [MockAsset(asset) for asset in assets]

    def get_assets(self):
        return self.assets


class MockRepo:
    def __init__(
        self,
        release_dict={
            "v3.0.0": [
                "casanovo_massivekb.ckpt",
                "casanovo_non-enzy.checkpt",
                "v3.0.0.zip",
                "v3.0.0.tar.gz",
            ],
            "v3.1.0": ["v3.1.0.zip", "v3.1.0.tar.gz"],
            "v3.2.0": ["v3.2.0.zip", "v3.2.0.tar.gz"],
            "v3.3.0": ["v3.3.0.zip", "v3.3.0.tar.gz"],
            "v4.0.0": [
                "casanovo_massivekb.ckpt",
                "casanovo_nontryptic.ckpt",
                "v4.0.0.zip",
                "v4.0.0.tar.gz",
            ],
        },
    ):
        self.releases = [
            MockRelease(tag_name, assets)
            for tag_name, assets in release_dict.items()
        ]

    def get_releases(self):
        return self.releases


class MockResponseGet:
    file_content = b"fake model weights content"

    class MockRaw(io.BytesIO):
        def read(self, *args, **kwargs):
            return super().read(*args)

    def __init__(self):
        self.request_counter = 0
        self.is_ok = True

    def raise_for_status(self):
        if not self.is_ok:
            raise requests.HTTPError

    def __call__(self, url, stream=True, allow_redirects=True):
        self.request_counter += 1
        response = unittest.mock.MagicMock()
        response.raise_for_status = self.raise_for_status
        response.headers = {"Content-Length": str(len(self.file_content))}
        response.raw = MockResponseGet.MockRaw(self.file_content)
        return response


class MockResponseHead:
    def __init__(self):
        self.last_modified = None
        self.is_ok = True
        self.fail = False

    def __call__(self, url):
        if self.fail:
            raise requests.ConnectionError

        response = unittest.mock.MagicMock()
        response.headers = dict()
        response.ok = self.is_ok
        if self.last_modified is not None:
            response.headers["Last-Modified"] = self.last_modified

        return response


def test_setup_model(monkeypatch):
    test_releases = ["3.0.0", "3.0.999", "3.999.999"]
    mock_get = MockResponseGet()
    mock_github = functools.partial(MockGithub, test_releases)
    version = "3.0.0"

    # Test model is none when not training
    with monkeypatch.context() as mnk, tempfile.TemporaryDirectory() as tmp_dir:
        mnk.setattr(casanovo, "__version__", version)
        mnk.setattr("appdirs.user_cache_dir", lambda n, a, opinion: tmp_dir)
        mnk.setattr(github, "Github", mock_github)
        mnk.setattr(requests, "get", mock_get)
        filename = pathlib.Path(tmp_dir) / "casanovo_massivekb_v3_0_0.ckpt"

        assert not filename.is_file()
        _, result_path = casanovo.setup_model(None, None, None, None, False)
        assert result_path.resolve() == filename.resolve()
        assert filename.is_file()
        assert mock_get.request_counter == 1
        os.remove(result_path)

        assert not filename.is_file()
        _, result = casanovo.setup_model(None, None, None, None, True)
        assert result is None
        assert not filename.is_file()
        assert mock_get.request_counter == 1

    with monkeypatch.context() as mnk, tempfile.TemporaryDirectory() as tmp_dir:
        mnk.setattr(casanovo, "__version__", version)
        mnk.setattr("appdirs.user_cache_dir", lambda n, a, opinion: tmp_dir)
        mnk.setattr(github, "Github", mock_github)
        mnk.setattr(requests, "get", mock_get)

        cache_file_name = "model_weights.ckpt"
        file_url = f"http://www.example.com/{cache_file_name}"
        url_hash = hashlib.shake_256(file_url.encode("utf-8")).hexdigest(5)
        cache_dir = pathlib.Path(tmp_dir)
        cache_file_dir = cache_dir / url_hash
        cache_file_path = cache_file_dir / cache_file_name

        assert not cache_file_path.is_file()
        _, result_path = casanovo.setup_model(
            file_url, None, None, None, False
        )
        assert cache_file_path.is_file()
        assert result_path.resolve() == cache_file_path.resolve()
        assert mock_get.request_counter == 2
        os.remove(result_path)

        assert not cache_file_path.is_file()
        _, result_path = casanovo.setup_model(
            file_url, None, None, None, False
        )
        assert cache_file_path.is_file()
        assert result_path.resolve() == cache_file_path.resolve()
        assert mock_get.request_counter == 3

    # Test model is file
    with monkeypatch.context() as mnk, tempfile.NamedTemporaryFile(
        suffix=".ckpt"
    ) as temp_file, tempfile.TemporaryDirectory() as tmp_dir:
        mnk.setattr(casanovo, "__version__", version)
        mnk.setattr("appdirs.user_cache_dir", lambda n, a, opinion: tmp_dir)
        mnk.setattr(github, "Github", mock_github)
        mnk.setattr(requests, "get", mock_get)

        temp_file_path = temp_file.name
        _, result = casanovo.setup_model(
            temp_file_path, None, None, None, False
        )
        assert mock_get.request_counter == 3
        assert result == temp_file_path

        _, result = casanovo.setup_model(
            temp_file_path, None, None, None, True
        )
        assert mock_get.request_counter == 3
        assert result == temp_file_path

    # Test model is neither a URL or File
    with monkeypatch.context() as mnk, tempfile.TemporaryDirectory() as tmp_dir:
        mnk.setattr(casanovo, "__version__", version)
        mnk.setattr("appdirs.user_cache_dir", lambda n, a, opinion: tmp_dir)
        mnk.setattr(github, "Github", mock_github)
        mnk.setattr(requests, "get", mock_get)

        with pytest.raises(ValueError):
            casanovo.setup_model("FooBar", None, None, None, False)

        assert mock_get.request_counter == 3

        with pytest.raises(ValueError):
            casanovo.setup_model("FooBar", None, None, None, False)

        assert mock_get.request_counter == 3


def test_get_model_weights(monkeypatch):
    """
    Test that model weights can be downloaded from GitHub or used from the
    cache.
    """
    # Model weights for fully matching version, minor matching version, major
    # matching version.
    test_releases = ["3.0.0", "3.0.999", "3.999.999"]
    mock_get = MockResponseGet()
    mock_github = functools.partial(MockGithub, test_releases)

    for version in test_releases:
        with monkeypatch.context() as mnk, tempfile.TemporaryDirectory() as tmp_dir:
            mnk.setattr(casanovo, "__version__", version)
            mnk.setattr(
                "appdirs.user_cache_dir", lambda n, a, opinion: tmp_dir
            )
            mnk.setattr(github, "Github", mock_github)
            mnk.setattr(requests, "get", mock_get)

            tmp_path = pathlib.Path(tmp_dir)
            filename = tmp_path / "casanovo_massivekb_v3_0_0.ckpt"
            assert not filename.is_file()
            result_path = casanovo._get_model_weights(tmp_path)
            assert result_path == filename
            assert filename.is_file()
            result_path = casanovo._get_model_weights(tmp_path)
            assert result_path == filename

    # Impossible to find model weights for (i) full version mismatch and (ii)
    # major version mismatch.
    for version in ["999.999.999", "999.0.0"]:
        with monkeypatch.context() as mnk, tempfile.TemporaryDirectory() as tmp_dir:
            mnk.setattr(casanovo, "__version__", version)
            mnk.setattr(github, "Github", mock_github)
            mnk.setattr(requests, "get", mock_get)
            with pytest.raises(ValueError):
                casanovo._get_model_weights(pathlib.Path(tmp_dir))

    # Test GitHub API rate limit.
    def request(self, *args, **kwargs):
        raise github.RateLimitExceededException(
            403, "API rate limit exceeded", None
        )

    with monkeypatch.context() as mnk, tempfile.TemporaryDirectory() as tmp_dir:
        mnk.setattr("appdirs.user_cache_dir", lambda n, a, opinion: tmp_dir)
        mnk.setattr("github.Requester.Requester.requestJsonAndCheck", request)
        mnk.setattr(requests, "get", mock_get)
        mock_get.request_counter = 0
        with pytest.raises(github.RateLimitExceededException):
            casanovo._get_model_weights(pathlib.Path(tmp_dir))

        assert mock_get.request_counter == 0


def test_get_weights_from_url(monkeypatch):
    file_url = "http://example.com/model_weights.ckpt"

    with monkeypatch.context() as mnk, tempfile.TemporaryDirectory() as tmp_dir:
        mock_get = MockResponseGet()
        mock_head = MockResponseHead()
        mnk.setattr(requests, "get", mock_get)
        mnk.setattr(requests, "head", mock_head)

        cache_dir = pathlib.Path(tmp_dir)
        url_hash = hashlib.shake_256(file_url.encode("utf-8")).hexdigest(5)
        cache_file_name = "model_weights.ckpt"
        cache_file_dir = cache_dir / url_hash
        cache_file_path = cache_file_dir / cache_file_name

        # Test downloading and caching the file
        assert not cache_file_path.is_file()
        result_path = casanovo._get_weights_from_url(file_url, cache_dir)
        assert cache_file_path.is_file()
        assert result_path.resolve() == cache_file_path.resolve()
        assert mock_get.request_counter == 1

        # Test that cached file is used
        result_path = casanovo._get_weights_from_url(file_url, cache_dir)
        assert result_path.resolve() == cache_file_path.resolve()
        assert mock_get.request_counter == 1

        # Test force downloading the file
        result_path = casanovo._get_weights_from_url(
            file_url, cache_dir, force_download=True
        )
        assert result_path.resolve() == cache_file_path.resolve()
        assert mock_get.request_counter == 2

        # Test that file is re-downloaded if last modified is newer than
        # file last modified
        # NOTE: Assuming test takes < 1 year to run
        curr_utc = datetime.datetime.now().astimezone(datetime.timezone.utc)
        mock_head.last_modified = (
            curr_utc + datetime.timedelta(days=365.0)
        ).strftime("%a, %d %b %Y %H:%M:%S GMT")
        result_path = casanovo._get_weights_from_url(file_url, cache_dir)
        assert result_path.resolve() == cache_file_path.resolve()
        assert mock_get.request_counter == 3

        # Test file is not redownloaded if its newer than upstream file
        mock_head.last_modified = (
            curr_utc - datetime.timedelta(days=365.0)
        ).strftime("%a, %d %b %Y %H:%M:%S GMT")
        result_path = casanovo._get_weights_from_url(file_url, cache_dir)
        assert result_path.resolve() == cache_file_path.resolve()
        assert mock_get.request_counter == 3

        # Test that error is raised if file get response is not OK
        mock_get.is_ok = False
        with pytest.raises(requests.HTTPError):
            casanovo._get_weights_from_url(
                file_url, cache_dir, force_download=True
            )
        mock_get.is_ok = True
        assert mock_get.request_counter == 4

        # Test that cached file is used if head requests yields non-ok status
        # code, even if upstream file is newer
        mock_head.is_ok = False
        mock_head.last_modified = (
            curr_utc + datetime.timedelta(days=365.0)
        ).strftime("%a, %d %b %Y %H:%M:%S GMT")
        result_path = casanovo._get_weights_from_url(file_url, cache_dir)
        assert result_path.resolve() == cache_file_path.resolve()
        assert mock_get.request_counter == 4
        mock_head.is_ok = True

        # Test that cached file is used if head request fails
        mock_head.fail = True
        result_path = casanovo._get_weights_from_url(file_url, cache_dir)
        assert result_path.resolve() == cache_file_path.resolve()
        assert mock_get.request_counter == 4
        mock_head.fail = False

        # Test invalid URL
        with pytest.raises(ValueError):
            bad_url = "foobar"
            casanovo._get_weights_from_url(bad_url, cache_dir)


def test_is_valid_url():
    assert casanovo._is_valid_url("https://www.washington.edu/")
    assert not casanovo._is_valid_url("foobar")


def test_peptide_score():
    """
    Test the calculation of amino acid and peptide scores from the raw amino
    acid scores.
    """
    aa_scores_raw = np.asarray([0.0, 0.5, 1.0])

    peptide_score = _peptide_score(aa_scores_raw, True)
    assert peptide_score == pytest.approx(0.0)

    peptide_score = _peptide_score(aa_scores_raw, False)
    assert peptide_score == pytest.approx(-1.0)

    aa_scores_raw = np.asarray([1.0, 0.25])
    peptide_score = _peptide_score(aa_scores_raw, True)
    assert peptide_score == pytest.approx(0.25)

    aa_scores_batch = np.array([[0.5, 0.8, 0.0], [0.9, 0.7, 0.6]])
    lengths_batch = np.array([2, 3])

    peptide_scores = _peptide_score(aa_scores_batch, True, lengths_batch)
    expected_scores = np.array([0.5 * 0.8, 0.9 * 0.7 * 0.6])
    assert np.allclose(peptide_scores, expected_scores)

    peptide_scores_no_fit = _peptide_score(
        aa_scores_batch, False, lengths_batch
    )
    assert np.allclose(peptide_scores_no_fit, expected_scores - 1)

    fits_array = np.array([True, False])
    peptide_scores_mixed_fit = _peptide_score(
        aa_scores_batch, fits_array, lengths_batch
    )
    expected_mixed_scores = expected_scores.copy()
    expected_mixed_scores[~fits_array] -= 1
    assert np.allclose(peptide_scores_mixed_fit, expected_mixed_scores)

    with pytest.raises(
        ValueError, match="`lengths` must be provided for batched input."
    ):
        _peptide_score(aa_scores_batch, True, None)


def test_peptide_generator_errors(tiny_fasta_file):
    residues_dict = (
        depthcharge.tokenizers.PeptideTokenizer.from_massivekb().residues
    )
    with pytest.raises(FileNotFoundError):
        [
            (a, b)
            for a, b in db_utils._peptide_generator(
                "fail.fasta", "trypsin", "full", 0, 5, 10, residues_dict
            )
        ]
    with pytest.raises(ValueError):
        [
            (a, b)
            for a, b in db_utils._peptide_generator(
                tiny_fasta_file, "trypsin", "fail", 0, 5, 10, residues_dict
            )
        ]


def test_to_neutral_mass():
    mz = 500
    charge = 2
    neutral_mass = db_utils._to_neutral_mass(mz, charge)
    assert neutral_mass == 997.98544706646

    mz = 500
    charge = 1
    neutral_mass = db_utils._to_neutral_mass(mz, charge)
    assert neutral_mass == 498.99272353323


def test_calc_match_score():
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


def test_digest_fasta_cleave(tiny_fasta_file):
    # No missed cleavages
    expected_normal = [
        "ATSIPAR",
        "VTLSC[Carbamidomethyl]R",
        "LLIYGASTR",
        "EIVMTQSPPTLSLSPGER",
        "MEAPAQLLFLLLLWLPDTTR",
        "ASQSVSSSYLTWYQQKPGQAPR",
        "FSGSGSGTDFTLTISSLQPEDFAVYYC[Carbamidomethyl]QQDYNLP",
    ]

    # 1 missed cleavage
    expected_1missedcleavage = [
        "ATSIPAR",
        "VTLSC[Carbamidomethyl]R",
        "LLIYGASTR",
        "LLIYGASTRATSIPAR",
        "EIVMTQSPPTLSLSPGER",
        "MEAPAQLLFLLLLWLPDTTR",
        "ASQSVSSSYLTWYQQKPGQAPR",
        "EIVMTQSPPTLSLSPGERVTLSC[Carbamidomethyl]R",
        "VTLSC[Carbamidomethyl]RASQSVSSSYLTWYQQKPGQAPR",
        "ASQSVSSSYLTWYQQKPGQAPRLLIYGASTR",
        "FSGSGSGTDFTLTISSLQPEDFAVYYC[Carbamidomethyl]QQDYNLP",
        "MEAPAQLLFLLLLWLPDTTREIVMTQSPPTLSLSPGER",
        "ATSIPARFSGSGSGTDFTLTISSLQPEDFAVYYC[Carbamidomethyl]QQDYNLP",
    ]

    # 3 missed cleavages
    expected_3missedcleavage = [
        "ATSIPAR",
        "VTLSC[Carbamidomethyl]R",
        "LLIYGASTR",
        "LLIYGASTRATSIPAR",
        "EIVMTQSPPTLSLSPGER",
        "MEAPAQLLFLLLLWLPDTTR",
        "ASQSVSSSYLTWYQQKPGQAPR",
        "EIVMTQSPPTLSLSPGERVTLSC[Carbamidomethyl]R",
        "VTLSC[Carbamidomethyl]RASQSVSSSYLTWYQQKPGQAPR",
        "ASQSVSSSYLTWYQQKPGQAPRLLIYGASTR",
        "FSGSGSGTDFTLTISSLQPEDFAVYYC[Carbamidomethyl]QQDYNLP",
        "ASQSVSSSYLTWYQQKPGQAPRLLIYGASTRATSIPAR",
        "VTLSC[Carbamidomethyl]RASQSVSSSYLTWYQQKPGQAPRLLIYGASTR",
        "MEAPAQLLFLLLLWLPDTTREIVMTQSPPTLSLSPGER",
        "ATSIPARFSGSGSGTDFTLTISSLQPEDFAVYYC[Carbamidomethyl]QQDYNLP",
        "VTLSC[Carbamidomethyl]RASQSVSSSYLTWYQQKPGQAPRLLIYGASTRATSIPAR",
        "MEAPAQLLFLLLLWLPDTTREIVMTQSPPTLSLSPGERVTLSC[Carbamidomethyl]R",
        "EIVMTQSPPTLSLSPGERVTLSC[Carbamidomethyl]RASQSVSSSYLTWYQQKPGQAPR",
        "LLIYGASTRATSIPARFSGSGSGTDFTLTISSLQPEDFAVYYC[Carbamidomethyl]QQDYNLP",
    ]
    for missed_cleavages, expected in zip(
        (0, 1, 3),
        (expected_normal, expected_1missedcleavage, expected_3missedcleavage),
    ):
        pdb = db_utils.ProteinDatabase(
            fasta_path=str(tiny_fasta_file),
            enzyme="trypsin",
            digestion="full",
            missed_cleavages=missed_cleavages,
            min_peptide_len=6,
            max_peptide_len=50,
            max_mods=0,
            precursor_tolerance=20,
            isotope_error=[0, 0],
            allowed_fixed_mods="C:C[Carbamidomethyl]",
            allowed_var_mods=(
                "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
                "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
            ),
            tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
        )
        assert pdb.db_peptides.index.to_list() == expected


def test_digest_fasta_mods(tiny_fasta_file):
    # 1 modification allowed
    # fixed: C+57.02146
    # variable: 1M+15.994915,1N+0.984016,1Q+0.984016
    # nterm: 1X+42.010565,1X+43.005814,1X-17.026549,1X+25.980265
    expected = np.array(
        [
            697.37586,  # -17.027ATSIPAR
            714.40241,  # ATSIPAR
            717.34797,  # -17.027VTLSC+57.021R
            734.37452,  # VTLSC+57.021R
            740.3827,  # +43.006-17.027ATSIPAR
            756.41297,  # +42.011ATSIPAR
            757.40821,  # +43.006ATSIPAR
            760.35474,  # +43.006-17.027VTLSC+57.021R
            776.38502,  # +42.011VTLSC+57.021R
            777.38032,  # +43.006VTLSC+57.021R
            975.53889,  # -17.027LLIYGASTR
            992.5655,  # LLIYGASTR
            1018.54572,  # +43.006-17.027LLIYGASTR
            1034.57599,  # +42.011LLIYGASTR
            1035.57123,  # +43.006LLIYGASTR
            1923.96125,  # -17.027EIVMTQSPPTLSLSPGER
            1940.98774,  # EIVMTQSPPTLSLSPGER
            1941.97175,  # EIVMTQ+0.984SPPTLSLSPGER
            1956.98261,  # EIVM+15.995TQSPPTLSLSPGER
            1966.96808,  # +43.006-17.027EIVMTQSPPTLSLSPGER
            1982.99836,  # +42.011EIVMTQSPPTLSLSPGER
            1983.99372,  # +43.006EIVMTQSPPTLSLSPGER
            2323.26496,  # -17.027MEAPAQLLFLLLLWLPDTTR
            2339.25983,  # -17.027M+15.995EAPAQLLFLLLLWLPDTTR
            2340.29157,  # MEAPAQLLFLLLLWLPDTTR
            2341.27546,  # MEAPAQ+0.984LLFLLLLWLPDTTR
            2356.28644,  # M+15.995EAPAQLLFLLLLWLPDTTR
            2366.2718,  # +43.006-17.027MEAPAQLLFLLLLWLPDTTR
            2382.26667,  # +43.006-17.027M+15.995EAPAQLLFLLLLWLPDTTR
            2382.30207,  # +42.011MEAPAQLLFLLLLWLPDTTR
            2383.29743,  # +43.006MEAPAQLLFLLLLWLPDTTR
            2398.29694,  # +42.011M+15.995EAPAQLLFLLLLWLPDTTR
            2399.2923,  # +43.006M+15.995EAPAQLLFLLLLWLPDTTR
            2451.18171,  # -17.027ASQSVSSSYLTWYQQKPGQAPR
            2468.20832,  # ASQSVSSSYLTWYQQKPGQAPR
            2469.19221,  # ASQSVSSSYLTWYQ+0.984QKPGQAPR
            2469.19245,  # ASQSVSSSYLTWYQQ+0.984KPGQAPR
            2469.19245,  # ASQ+0.984SVSSSYLTWYQQKPGQAPR
            2469.19245,  # ASQSVSSSYLTWYQQKPGQ+0.984APR
            2494.18879,  # +43.006-17.027ASQSVSSSYLTWYQQKPGQAPR
            2510.21906,  # +42.011ASQSVSSSYLTWYQQKPGQAPR
            2511.21418,  # +43.006ASQSVSSSYLTWYQQKPGQAPR
            3789.66681,  # -17.027FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP
            3806.69318,  # FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP
            3807.67707,  # FSGSGSGTDFTLTISSLQ+0.984PEDFAVYYC+57.021QQDYNLP
            3807.67707,  # FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYN+0.984LP
            3807.67731,  # FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021Q+0.984QDYNLP
            3807.67731,  # FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQ+0.984DYNLP
            3832.67365,  # +43.006-17.027FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP
            3848.70392,  # +42.011FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP
            3849.69904,  # +43.006FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP
        ]
    )

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="full",
        missed_cleavages=0,
        min_peptide_len=6,
        max_peptide_len=50,
        max_mods=1,
        precursor_tolerance=20,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )

    assert np.allclose(pdb.db_peptides["calc_mass"].to_numpy(), expected)


def test_length_restrictions(tiny_fasta_file):
    # length between 20 and 50
    expected_long = [
        "MEAPAQLLFLLLLWLPDTTR",
        "ASQSVSSSYLTWYQQKPGQAPR",
        "FSGSGSGTDFTLTISSLQPEDFAVYYC[Carbamidomethyl]QQDYNLP",
    ]

    # length between 6 and 8
    expected_short = ["ATSIPAR", "VTLSC[Carbamidomethyl]R"]

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="full",
        missed_cleavages=0,
        min_peptide_len=20,
        max_peptide_len=50,
        max_mods=0,
        precursor_tolerance=20,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    assert pdb.db_peptides.index.to_list() == expected_long

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="full",
        missed_cleavages=0,
        min_peptide_len=6,
        max_peptide_len=8,
        max_mods=0,
        precursor_tolerance=20,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    assert pdb.db_peptides.index.to_list() == expected_short


@pytest.mark.parametrize(
    "enzyme,digestion,expected",
    [
        pytest.param(
            "arg-c",
            "full",
            np.array(
                [
                    714.40241,  # ATSIPAR
                    734.37452,  # VTLSC+57.021R
                    992.5655,  # LLIYGASTR
                    1940.98774,  # EIVMTQSPPTLSLSPGER
                    2340.29157,  # MEAPAQLLFLLLLWLPDTTR
                    2468.20832,  # ASQSVSSSYLTWYQQKPGQAPR
                    3806.69318,  # FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP
                ]
            ),
            id="arg-c",
        ),
        pytest.param(
            "asp-n",
            "full",
            np.array(
                [
                    1192.48591,  # DFAVYYC+57.021QQ
                    1349.67145,  # DFTLTISSLQPE
                    1867.06818,  # MEAPAQLLFLLLLWLP
                ]
            ),
            id="asp-n",
        ),
        pytest.param(
            "R",
            "full",
            np.array(
                [
                    714.40241,  # ATSIPAR
                    734.37452,  # VTLSC+57.021R
                    992.5655,  # LLIYGASTR
                    1940.98774,  # EIVMTQSPPTLSLSPGER
                    2340.29157,  # MEAPAQLLFLLLLWLPDTTR
                    2468.20832,  # ASQSVSSSYLTWYQQKPGQAPR
                    3806.69318,  # FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP
                ]
            ),
            id="regex-R",
        ),
        pytest.param(
            "trypsin",
            "partial",
            np.array(
                [
                    540.21796,  # FSGSGS
                    558.30128,  # ATSIPA
                    577.27076,  # ASQSVS
                    624.33436,  # PGQAPR
                    643.3653,  # TSIPAR
                    645.27924,  # MEAPAQ
                    648.38459,  # LLIYGA
                    653.3133,  # YGASTR
                    657.34455,  # LSPGER
                    701.37079,  # LPDTTR
                    719.35236,  # EIVMTQ
                    734.37452,  # VTLSC+57.021R
                    748.33912,  # QDYNLP
                ]
            ),
            id="semispecific",
        ),
        pytest.param(
            "trypsin",
            "non-specific",
            np.array(
                [
                    450.17106,  # SGSGSG
                    464.18665,  # GSGSGT
                    522.19211,  # SGSGTD
                    540.21796,  # FSGSGS
                    558.30128,  # ATSIPA
                    561.28705,  # GASTRA
                    572.31696,  # LSLSPG
                    577.27076,  # ASQSVS
                    582.22858,  # GSGTDF
                    588.27546,  # SLSPGE
                    593.26569,  # QSVSSS
                    593.26569,  # SQSVSS
                    596.32819,  # KPGQAP
                    600.3119,  # SPPTLS
                    605.3133,  # ASTRAT
                    609.28711,  # RFSGSG
                    610.29621,  # IYGAST
                    611.36426,  # APAQLL
                    616.34321,  # PTLSLS
                    616.34321,  # TLSLSP
                    620.33808,  # TLTISS
                    621.30817,  # STRATS
                    622.33259,  # LIYGAS
                    623.30274,  # ARFSGS
                    624.33436,  # PGQAPR
                    626.25477,  # SGTDFT
                    626.3639,  # PPTLSL
                    627.32282,  # EAPAQL
                    627.33405,  # QKPGQA
                    628.27045,  # SVSSSY
                    629.30207,  # TQSPPT
                    632.37446,  # LTISSL
                    633.32343,  # PARFSG
                    640.36567,  # GQAPRL
                    641.33845,  # QSPPTL
                    643.32892,  # SPGERV
                    643.35407,  # ISSLQP
                    643.3653,  # TSIPAR
                    643.3653,  # RATSIP
                    645.27924,  # MEAPAQ
                    646.33985,  # RASQSV
                    647.349,  # TISSLQ
                    647.36024,  # TRATSI
                    648.38459,  # LLIYGA
                    652.30683,  # GTDFTL
                    653.3133,  # YGASTR
                    654.32252,  # VSSSYL
                    656.3017,  # SSSYLT
                    657.34455,  # LSPGER
                    657.34455,  # PGERVT
                    659.29487,  # MTQSPP
                    659.31263,  # SSLQPE
                    661.31049,  # VMTQSP
                    673.37586,  # GERVTL
                    676.30683,  # PEDFAV
                    677.3418,  # IVMTQS
                    680.37446,  # FTLTIS
                    681.45374,  # APRLLI
                    684.35554,  # QQKPGQ
                    687.30756,  # SLQPED
                    687.39551,  # PAQLLF
                    689.38605,  # IPARFS
                    689.38605,  # SIPARF
                    692.32758,  # LSC+57.021RAS
                    696.33301,  # TDFTLT
                    696.42817,  # QAPRLL
                    701.37079,  # LPDTTR
                    703.38642,  # ERVTLS
                    703.42682,  # AQLLFL
                    705.29694,  # QPEDFA
                    706.34321,  # TLSC+57.021RA
                    707.30207,  # SC+57.021RASQ
                    707.30213,  # C+57.021RASQS
                    708.36939,  # DFTLTI
                    717.32929,  # PDTTRE
                    717.40204,  # TTREIV
                    719.35236,  # EIVMTQ
                    719.3603,  # YQQKPG
                    730.49927,  # LFLLLL
                    730.49927,  # LLFLLL
                    731.349,  # WLPDTT
                    733.36066,  # DTTREI
                    733.44861,  # RLLIYG
                    734.37452,  # RVTLSC+57.021
                    734.37452,  # VTLSC+57.021R
                    742.31733,  # EDFAVY
                    743.38544,  # LWLPDT
                    745.47376,  # QLLFLL
                    747.34394,  # LQPEDF
                    747.39484,  # TREIVM
                    747.3949,  # REIVMT
                    748.33912,  # QDYNLP
                    753.47883,  # LLLWLP
                    755.349,  # SSYLTW
                    755.42176,  # LLWLPD
                    769.51014,  # LLLLWL
                    773.47993,  # PRLLIY
                    776.33814,  # DFAVYY
                    779.34492,  # QQDYNL
                    802.33198,  # AVYYC+57.021Q
                    803.49451,  # FLLLLW
                    821.34186,  # FAVYYC+57.021
                    826.29157,  # C+57.021QQDYN
                    831.38032,  # SYLTWY
                    837.40217,  # LTWYQQ
                    848.41816,  # WYQQKP
                    852.41303,  # TWYQQK
                    859.35346,  # VYYC+57.021QQ
                    872.4068,  # YLTWYQ
                    875.31196,  # YYC+57.021QQD
                    875.31202,  # YC+57.021QQDY
                ]
            ),
            id="nonspecific",
        ),
    ],
)
def test_digest_fasta_enzyme(tiny_fasta_file, enzyme, digestion, expected):
    """Test peptide digestion behavior for various enzymes and digest modes."""
    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme=enzyme,
        digestion=digestion,
        missed_cleavages=0,
        min_peptide_len=6,
        max_peptide_len=50 if digestion == "full" else 6,
        max_mods=0,
        precursor_tolerance=20 if digestion == "full" else 10000,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )

    assert np.allclose(pdb.db_peptides["calc_mass"].to_numpy(), expected)


def test_psm_batches(tiny_config):
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
            return pd.Series(peptides_one)
        elif precorsor_charge == 2:
            return pd.Series(peptides_two)
        else:
            return pd.Series()

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
            return pd.Series(peptides_one)
        else:
            return pd.Series(peptides_two)

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

    for predction in db_model.predict_step(mock_batch):
        # make sure the stop token score is not inlcuded in reported AA scores
        # but is included in match score calculation
        assert len(tokenizer.tokenize(predction.sequence)[0]) == len(
            predction.aa_scores
        )
        assert pytest.approx(predction.peptide_score) != _peptide_score(
            predction.aa_scores, True
        )


def test_isoleucine_match(tiny_config):
    tokenizer = depthcharge.tokenizers.peptides.PeptideTokenizer(
        residues=Config(tiny_config).residues,
        replace_isoleucine_with_leucine=True,
    )
    db_model = DbSpec2Pep(tokenizer=tokenizer)
    peptides = [["PEPTLDEK"], ["PEPTIDEK"]]
    mock_get_candidates = lambda _, precursor_charge: pd.Series(
        peptides[precursor_charge - 1]
    )

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


def test_get_candidates(tiny_fasta_file):
    # precursor_window is 10000
    expected_smallwindow = ["LLIYGASTR"]

    # precursor window is 150000
    expected_midwindow = ["LLIYGASTR"]

    # precursor window is 600000
    expected_widewindow = ["ATSIPAR", "VTLSC[Carbamidomethyl]R", "LLIYGASTR"]

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="full",
        missed_cleavages=1,
        min_peptide_len=6,
        max_peptide_len=50,
        max_mods=0,
        precursor_tolerance=10000,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    candidates = pdb.get_candidates(precursor_mz=496.2, charge=2)
    assert expected_smallwindow == list(candidates)

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="full",
        missed_cleavages=1,
        min_peptide_len=6,
        max_peptide_len=50,
        max_mods=0,
        precursor_tolerance=150000,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    candidates = pdb.get_candidates(precursor_mz=496.2, charge=2)
    assert expected_midwindow == list(candidates)

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="full",
        missed_cleavages=1,
        min_peptide_len=6,
        max_peptide_len=50,
        max_mods=0,
        precursor_tolerance=600000,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    candidates = pdb.get_candidates(precursor_mz=496.2, charge=2)
    assert expected_widewindow == list(candidates)


def test_get_candidates_isotope_error(tiny_fasta_file):
    # Tide isotope error windows for 496.2, 2+:
    # 0: [980.481617, 1000.289326]
    # 1: [979.491114, 999.278813]
    # 2: [978.500611, 998.268300]
    # 3: [977.510108, 997.257787]

    peptide_list = [
        ("A", 1001, "foo"),
        ("B", 1000, "foo"),
        ("C", 999, "foo"),
        ("D", 998, "foo"),
        ("E", 997, "foo"),
        ("F", 996, "foo"),
        ("G", 995, "foo"),
        ("H", 994, "foo"),
        ("I", 993, "foo"),
        ("J", 992, "foo"),
        ("K", 991, "foo"),
        ("L", 990, "foo"),
        ("M", 989, "foo"),
        ("N", 988, "foo"),
        ("O", 987, "foo"),
        ("P", 986, "foo"),
        ("Q", 985, "foo"),
        ("R", 984, "foo"),
        ("S", 983, "foo"),
        ("T", 982, "foo"),
        ("U", 981, "foo"),
        ("V", 980, "foo"),
        ("W", 979, "foo"),
        ("X", 978, "foo"),
        ("Y", 977, "foo"),
        ("Z", 976, "foo"),
    ]

    peptide_list = pd.DataFrame(
        peptide_list, columns=["peptide", "calc_mass", "protein"]
    ).set_index("peptide")
    peptide_list.sort_values("calc_mass", inplace=True)

    expected_isotope0 = list("UTSRQPONMLKJIHGFEDCB")
    expected_isotope01 = list("VUTSRQPONMLKJIHGFEDCB")
    expected_isotope012 = list("WVUTSRQPONMLKJIHGFEDCB")
    expected_isotope0123 = list("XWVUTSRQPONMLKJIHGFEDCB")

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="full",
        missed_cleavages=0,
        min_peptide_len=0,
        max_peptide_len=0,
        max_mods=0,
        precursor_tolerance=10000,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    pdb.db_peptides = peptide_list
    candidates = pdb.get_candidates(precursor_mz=496.2, charge=2)
    assert expected_isotope0 == list(candidates)

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="full",
        missed_cleavages=0,
        min_peptide_len=0,
        max_peptide_len=0,
        max_mods=0,
        precursor_tolerance=10000,
        isotope_error=[0, 1],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    pdb.db_peptides = peptide_list
    candidates = pdb.get_candidates(precursor_mz=496.2, charge=2)
    assert expected_isotope01 == list(candidates)

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="full",
        missed_cleavages=0,
        min_peptide_len=0,
        max_peptide_len=0,
        max_mods=0,
        precursor_tolerance=10000,
        isotope_error=[0, 2],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    pdb.db_peptides = peptide_list
    candidates = pdb.get_candidates(precursor_mz=496.2, charge=2)
    assert expected_isotope012 == list(candidates)

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="full",
        missed_cleavages=0,
        min_peptide_len=0,
        max_peptide_len=0,
        max_mods=0,
        precursor_tolerance=10000,
        isotope_error=[0, 3],
        allowed_fixed_mods="C:C[Carbamidomethyl]",
        allowed_var_mods=(
            "M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],"
            "nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    pdb.db_peptides = peptide_list
    candidates = pdb.get_candidates(precursor_mz=496.2, charge=2)
    assert expected_isotope0123 == list(candidates)


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


def test_eval_metrics():
    """
    Test peptide and amino acid-level evaluation metrics.
    Predicted AAs are considered correct if they are <0.1Da from the
    corresponding ground truth AA with either a suffix or prefix <0.5Da from
    the ground truth. A peptide prediction is correct if all its AA are correct
    matches.
    """
    tokenizer = depthcharge.tokenizers.peptides.MskbPeptideTokenizer()

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
        aa_dict=tokenizer.residues,
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

    aa_matches, pep_match = aa_match(None, None, tokenizer.residues)

    assert aa_matches.shape == (0,)
    assert not pep_match

    aa_matches, pep_match = aa_match("PEPTIDE", None, tokenizer.residues)

    assert np.array_equal(aa_matches, np.zeros(len("PEPTIDE"), dtype=bool))
    assert not pep_match


def test_spectrum_id_mgf(mgf_small, tmp_path):
    """Test that spectra from MGF files are specified by their index."""
    mgf_small2 = tmp_path / "mgf_small2.mgf"
    shutil.copy(mgf_small, mgf_small2)
    data_module = DeNovoDataModule(
        lance_dir=tmp_path,
        train_paths=[mgf_small, mgf_small2],
        valid_paths=[mgf_small, mgf_small2],
        test_paths=[mgf_small, mgf_small2],
        min_peaks=0,
        shuffle=False,
    )
    data_module.setup()

    for dataset in [
        data_module.train_dataset,
        data_module.valid_dataset,
        data_module.test_dataset,
    ]:
        # depthcharge's MgfParser labels spectra "index=<n>", not "<n>"
        # (parsers.py, MgfParser._parse_spectrum). mzTab output is
        # unaffected either way: ms_io adds the "index=" prefix itself
        # when the id is bare, and leaves it alone when it is not.
        for i, (filename, scan_id) in enumerate(
            [
                (mgf_small, "index=0"),
                (mgf_small, "index=1"),
                (mgf_small2, "index=0"),
                (mgf_small2, "index=1"),
            ]
        ):
            assert dataset[i]["peak_file"][0] == filename.name
            assert dataset[i]["scan_id"][0] == scan_id


def test_spectrum_id_mzml(mzml_small, tmp_path):
    """Test that spectra from mzML files are specified by their scan id."""
    mzml_small2 = tmp_path / "mzml_small2.mzml"
    shutil.copy(mzml_small, mzml_small2)
    data_module = DeNovoDataModule(
        lance_dir=tmp_path,
        test_paths=[mzml_small, mzml_small2],
        min_peaks=0,
        shuffle=False,
    )
    data_module.setup(stage="test", annotated=False)

    dataset = data_module.test_dataset
    for i, (filename, scan_id) in enumerate(
        [
            (mzml_small, "scan=17"),
            (mzml_small, "merged=11 frame=12 scanStart=763 scanEnd=787"),
            (mzml_small2, "scan=17"),
            (mzml_small2, "merged=11 frame=12 scanStart=763 scanEnd=787"),
        ]
    ):
        assert dataset[i]["peak_file"][0] == filename.name
        assert dataset[i]["scan_id"][0] == scan_id


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
    val_step_loss = model.validation_step(val_batch)

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


def test_predict_step_penalizes_mass_mismatch(monkeypatch):
    """PMC may give up, and what it leaves behind must sort last."""
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
    assert missed.peptide_score < 0 <= matched.peptide_score
    assert matched.peptide_score == pytest.approx(missed.peptide_score + 1)
    # Aggregated as upstream does, so the reported score stays the
    # product of the reported residue scores.
    assert matched.peptide_score == pytest.approx(
        float(np.prod(matched.aa_scores))
    )


def test_run_map(mgf_small):
    out_writer = ms_io.MztabWriter("dummy.mztab")
    # Set peak file by base file name only.
    out_writer.set_ms_run([os.path.basename(mgf_small.name)])
    assert mgf_small.name in out_writer._run_map
    assert os.path.abspath(mgf_small.name) not in out_writer._run_map


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
    default_args = {arg: None for arg in psm_args}

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


def test_check_dir(tmp_path):
    exists_path = tmp_path / "exists-1234.ckpt"
    exists_pattern = "exists-*.ckpt"
    exists_path.touch()
    dne_pattern = "dne-*.ckpt"

    with pytest.raises(FileExistsError):
        utils.check_dir_file_exists(tmp_path, [exists_pattern, dne_pattern])

    with pytest.raises(FileExistsError):
        utils.check_dir_file_exists(tmp_path, exists_pattern)

    utils.check_dir_file_exists(tmp_path, [dne_pattern])
    utils.check_dir_file_exists(tmp_path, dne_pattern)


def test_setup_output(tmp_path, monkeypatch):
    with monkeypatch.context() as mnk:
        mnk.setattr(pathlib.Path, "cwd", lambda: tmp_path)
        output_path, output_root = casanovo._setup_output(
            None, None, False, "info"
        )
        assert output_path.resolve() == tmp_path.resolve()
        assert re.fullmatch(r"casanovo_\d+", output_root) is not None

        target_path = tmp_path / "foo"
        output_path, output_root = casanovo._setup_output(
            str(target_path), "bar", False, "info"
        )

        assert output_path.resolve() == target_path.resolve()
        assert output_root == "bar"


def test_spectrum_preprocessing(tmp_path, mgf_small):
    """
    Test the spectrum preprocessing function.
    """
    min_peaks, max_peaks = 0, 100
    min_mz, max_mz = 0, 2000
    min_intensity = 0
    remove_precursor_tol = 0
    max_charge = 4

    # Test number of peaks filtering.
    min_peaks, max_peaks = 20, 50
    # One spectrum removed with too few peaks.
    total_spectra = 1
    dataloader = DeNovoDataModule(
        tmp_path,
        test_paths=str(mgf_small),
        min_peaks=min_peaks,
        max_peaks=max_peaks,
        min_mz=min_mz,
        max_mz=max_mz,
        min_intensity=min_intensity,
        remove_precursor_tol=remove_precursor_tol,
        max_charge=max_charge,
    )
    dataloader.setup("test", annotated=False)
    assert dataloader.test_dataset.n_spectra == total_spectra
    for spec in dataloader.test_dataset:
        mz = (mz := spec["mz_array"][0])[torch.nonzero(mz)]
        assert min_peaks <= mz.shape[0] <= max_peaks
    min_peaks, max_peaks = 0, 100

    # Test m/z range filtering.
    min_mz, max_mz = 200, 600
    # All spectra retained.
    total_spectra = 2
    dataloader = DeNovoDataModule(
        tmp_path,
        test_paths=str(mgf_small),
        min_peaks=min_peaks,
        max_peaks=max_peaks,
        min_mz=min_mz,
        max_mz=max_mz,
        min_intensity=min_intensity,
        remove_precursor_tol=remove_precursor_tol,
        max_charge=max_charge,
    )
    dataloader.setup("test", annotated=False)
    assert dataloader.test_dataset.n_spectra == total_spectra
    for spec in dataloader.test_dataset:
        mz = (mz := spec["mz_array"][0])[torch.nonzero(mz)]
        assert mz.min() >= min_mz
        assert mz.max() <= max_mz
    min_mz, max_mz = 0, 2000

    # Test charge filtering.
    max_charge = 2
    # One spectrum removed with too high charge.
    total_spectra = 1
    dataloader = DeNovoDataModule(
        tmp_path,
        test_paths=str(mgf_small),
        min_peaks=min_peaks,
        max_peaks=max_peaks,
        min_mz=min_mz,
        max_mz=max_mz,
        min_intensity=min_intensity,
        remove_precursor_tol=remove_precursor_tol,
        max_charge=max_charge,
    )
    dataloader.setup("test", annotated=False)
    assert dataloader.test_dataset.n_spectra == total_spectra
    for spec in dataloader.test_dataset:
        assert spec["precursor_charge"] <= max_charge
    max_charge = 4


@pytest.mark.skip(reason="No Beam Search in Non-AR Model")
def test_beam_search_decode(tiny_config):
    """
    Test beam search decoding and its sub-functions.
    """
    config = casanovo.Config(tiny_config)
    model = Spec2Pep(
        n_beams=4,
        min_peptide_len=4,
        tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
            residues=config.residues
        ),
    )
    model.tokenizer.reverse = False

    batch = 1
    length = model.decoder_frames + 1
    beam = model.n_beams
    step = 3
    device = model.device

    model._batch_size = batch
    model._beam_size = beam
    model._cumulative_masses = torch.zeros(batch * beam, device=device)

    vocab = len(model.tokenizer) + 1
    scores = torch.full((batch, length, vocab, beam), torch.nan, device=device)
    scores = einops.rearrange(scores, "B L V S -> (B S) L V")

    tokens = torch.zeros(
        batch * beam, length, dtype=torch.int64, device=device
    )
    precursors = torch.tensor(
        [469.25364, 2.0, 235.63410], device=device
    ).repeat(beam, 1)

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

    # update masses
    for i in range(batch * beam):
        mass = sum(
            model.token_masses[tokens[i, j].item()]
            for j in range(step + 1)
            if 0 < tokens[i, j].item() < len(model.token_masses)
        )
        model._cumulative_masses[i] = mass

    finished, fits, discarded = model._finish_beams(tokens, precursors, step)

    assert torch.equal(
        finished, torch.tensor([False, True, False, True], device=device)
    )
    assert torch.equal(
        fits, torch.tensor([False, False, False, False], device=device)
    )
    assert torch.equal(
        discarded, torch.tensor([False, False, False, True], device=device)
    )


@pytest.mark.skip(reason="No Beam Search in Non-AR Model")
def test_peptide_too_short_too_heavy(tiny_config):
    config = Config(tiny_config)
    model = Spec2Pep(
        n_beams=1,
        min_peptide_len=20,
        tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
            residues=config.residues
        ),
    )

    # Mass of "peptid"
    precursors = torch.tensor([[670.31737, 2.0, 336.16596]])
    peptide = "PEPTIDE"
    tokens = model.tokenizer.tokenize(peptide)
    step = len(peptide) - 1
    # This needs to be initialized, but is recalculated anyway so the value
    # doesn't matter
    model._cumulative_masses = torch.tensor([42.424242])

    finished_beams, beam_fits_precursor, discarded_beams = model._finish_beams(
        tokens, precursors, step
    )

    assert finished_beams.item()
    assert discarded_beams.item()
    assert not beam_fits_precursor.item()


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

    scores = torch.zeros(
        (beam, length, len(model.tokenizer) + 1), device=device
    )
    tokens = torch.zeros((beam, length), dtype=torch.int64, device=device)

    pred_cache = collections.OrderedDict((i, []) for i in range(batch))

    true_tok = model.tokenizer.tokenize("PEPR")[0]
    tokens[1, : step + 1] = true_tok
    scores[1, : step + 1, true_tok] = 1

    finished = torch.tensor([False, True, False, False], device=device)
    discarded = torch.tensor([False, False, False, False], device=device)
    fits = torch.tensor([False, False, False, False], device=device)

    model._cache_finished_beams(
        tokens, scores, step, finished & ~discarded, fits, pred_cache
    )

    cached = [
        pep for (_, _, _, pep) in pred_cache[0] if torch.equal(pep, true_tok)
    ]
    assert len(cached) == 1


@pytest.mark.skip(reason="No Beam Search in Non-AR Model")
def test_get_top_peptide_ranking(tiny_config):
    config = Config(tiny_config)
    model = Spec2Pep(
        tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
            residues=config.residues
        )
    )

    cache = collections.OrderedDict({0: []})

    heapq.heappush(
        cache[0], (0.93, 0.1, [0.93] * 4, model.tokenizer.tokenize("PEPY")[0])
    )
    heapq.heappush(
        cache[0], (0.95, 0.2, [0.95] * 4, model.tokenizer.tokenize("PEPK")[0])
    )
    heapq.heappush(
        cache[0], (0.94, 0.3, [0.94] * 4, model.tokenizer.tokenize("PEPP")[0])
    )

    result = next(model._get_top_peptide(cache))
    assert result[0][-1] == "PEPK"


@pytest.mark.skip(reason="No Beam Search in Non-AR Model")
@pytest.mark.parametrize("topk", [1, 2, 3])
def test_get_top_peptide_multiple(topk, tiny_config):
    config = Config(tiny_config)
    model = Spec2Pep(
        top_match=topk,
        tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
            residues=config.residues
        ),
    )

    cache = collections.OrderedDict({0: []})
    cache_items = [(0.9, "PEPY"), (0.8, "PEPK"), (0.7, "PEPP")]
    for score, seq in cache_items:
        heapq.heappush(
            cache[0],
            (score, 0.0, [score] * 3, model.tokenizer.tokenize(seq)[0]),
        )

    result = next(model._get_top_peptide(cache))
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

    cache = {
        0: [
            (
                1.0,
                0.42,
                np.array([1.0, 0.0]),
                model.tokenizer.tokenize("PE")[0],
            )
        ]
    }

    result = list(model._get_top_peptide(cache))

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

    length = model.decoder_frames + 1
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
def test_finish_beams_negative_mods(tiny_config):
    config = Config(tiny_config)
    model = Spec2Pep(
        n_beams=2,
        tokenizer=depthcharge.tokenizers.peptides.MskbPeptideTokenizer(
            residues=config.residues
        ),
        residues=config.residues,
        min_peptide_len=0,
    )
    batch = 1
    beam = 2
    device = model.device

    model._batch_size = batch
    model._beam_size = beam
    model._cumulative_masses = torch.zeros(batch * beam, device=device)

    length = model.decoder_frames + 1
    step = 1
    tokens = torch.zeros((beam, length), dtype=torch.int64, device=device)
    tokens[:, : step + 1] = model.tokenizer.tokenize(["GK", "AK"])
    precursors = torch.tensor(
        [186.10044, 2.0, 94.05750], device=device
    ).repeat(beam, 1)
    finished, fits, disc = model._finish_beams(tokens, precursors, step)

    # Ground truth peptide is "-17.027GK".
    assert torch.equal(finished, torch.tensor([False, True], device=device))
    assert torch.equal(fits, torch.tensor([False, False], device=device))
    assert torch.equal(disc, torch.tensor([False, False], device=device))


@pytest.mark.skip(reason="No Beam Search in Non-AR Model")
def test_beam_search_decode_early_termination(tiny_config):
    config = Config(tiny_config)
    model = Spec2Pep(
        n_beams=1,
        tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
            residues=config.residues
        ),
    )

    model.decoder_frames = 0
    mzs = ints = torch.zeros(1, 5, device=model.device)
    precursors = torch.tensor(
        [[469.25364, 2.0, 235.63410]], device=model.device
    )

    assert len(list(model.beam_search_decode(mzs, ints, precursors))[0]) == 0


@pytest.mark.skip(reason="No Beam Search in Non-AR Model")
def test_duplicate_peptide_scores(tiny_config):
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

    length = model.decoder_frames + 1
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

    pred_cache = collections.OrderedDict((i, []) for i in range(batch))

    model._cache_finished_beams(
        tokens,
        scores,
        step,
        torch.ones(batch * beam, dtype=torch.bool, device=device),
        torch.ones(batch * beam, dtype=torch.bool, device=device),
        pred_cache,
    )

    for _, preds in pred_cache.items():
        assert len(preds) == beam
        vals = [p[0] for p in preds]
        assert np.allclose(vals, vals[0])


class MiniTok:
    residues = {"P": 97.05276, "NLoss1": -10.0, "NLoss2": -30.0}
    index = {"<pad>": 0, "<stop>": 1, "P": 2, "NLoss1": 3, "NLoss2": 4}
    padding_int, stop_int, start_int = 0, 1, -1

    def __init__(self):
        # Mass lookup table
        self.masses = torch.zeros(len(self.index))
        for k, m in self.residues.items():
            self.masses[self.index[k]] = m

        self.stop_token = self.index["<stop>"]
        self.reverse = False
        self.n_term = []

    def __len__(self):
        return len(self.index)

    def tokenize(self, seq, add_stop: bool = False):
        toks = [self.index[a] for a in seq]
        if add_stop:
            toks.append(self.stop_int)
        return torch.tensor([toks])

    def calculate_precursor_ions(self, toks, charges):
        mass = self.masses[toks].sum(1) + 18.01056
        return mass / charges + 1.007276


def _build_model(tok, cls=Spec2Pep, ppm_tol=20):
    model = cls(
        tokenizer=tok,
        precursor_mass_tol=ppm_tol,
        isotope_error_range=(0, 0),
        n_beams=1,
        min_peptide_len=1,
    )
    model.register_buffer(
        "neg_mass_idx",
        torch.tensor(
            [tok.index["NLoss1"], tok.index["NLoss2"]], dtype=torch.int
        ),
    )
    model.register_buffer("token_masses", tok.masses.double())
    model.register_buffer("nterm_idx", torch.tensor([], dtype=torch.int))
    model._cumulative_masses = torch.tensor(
        [tok.residues["P"]], dtype=torch.float64
    )
    return model


@pytest.mark.skip(reason="No Beam Search in Non-AR Model")
def test_precursor_rescue():
    """
    Verifies that the current Spec2Pep keeps a rescuable beam alive,
    while the legacy-style logic still terminates it.
    """
    tok = MiniTok()
    charge = 2
    target_mass = tok.residues["P"] + tok.residues["NLoss2"] + 18.01056
    precursor_mz = target_mass / charge + 1.007276
    # Sequence “P” + padding
    TOKENS = torch.tensor([[tok.index["P"], 0, 0]])
    PRECURSOR = torch.tensor(
        [[target_mass, charge, precursor_mz]], dtype=torch.float64
    )
    STEP = 0

    new_model = _build_model(tok)
    finished, _, _ = new_model._finish_beams(TOKENS, PRECURSOR, STEP)
    assert (
        not finished.item()
    ), "New logic failed and beam terminated unexpectedly!"

    class PrematureSpec2Pep(Spec2Pep):
        """
        Mimic the legacy behaviour, abort immediately if ppm > tol,
        without trying negative-mass residues.
        """

        def _finish_beams(self, tokens, precursors, step):
            theo = self.tokenizer.calculate_precursor_ions(
                tokens[:, : step + 1], precursors[:, 1]
            )
            delta = (theo - precursors[:, 2]) / precursors[:, 2] * 1e6
            over = delta.abs() > self.precursor_mass_tol
            # Terminate all over-tol beams
            finished = over.clone()
            beam_fits = ~over
            discarded = torch.zeros_like(over)
            return finished, beam_fits, discarded

    old_like = _build_model(tok, PrematureSpec2Pep)
    finished_old, _, _ = old_like._finish_beams(TOKENS, PRECURSOR, STEP)
    assert (
        finished_old.item()
    ), "Old logic was expected to terminate the beam but did not"


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
