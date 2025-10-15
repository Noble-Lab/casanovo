import collections
import copy
import datetime
import functools
import hashlib
import heapq
import io
import math
import os
import pathlib
import platform
import re
import shutil
import tempfile
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

from casanovo import casanovo, utils, denovo
from casanovo.config import Config
from casanovo.data import db_utils, ms_io, psm
from casanovo.denovo.dataloaders import DeNovoDataModule
from casanovo.denovo.evaluate import aa_match, aa_match_batch, aa_match_metrics
from casanovo.denovo.model import (
    DbSpec2Pep,
    Spec2Pep,
    _peptide_score,
    _calc_match_score,
)


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
        "VTLSC+57.021R",
        "LLIYGASTR",
        "EIVMTQSPPTLSLSPGER",
        "MEAPAQLLFLLLLWLPDTTR",
        "ASQSVSSSYLTWYQQKPGQAPR",
        "FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP",
    ]

    # 1 missed cleavage
    expected_1missedcleavage = [
        "ATSIPAR",
        "VTLSC+57.021R",
        "LLIYGASTR",
        "LLIYGASTRATSIPAR",
        "EIVMTQSPPTLSLSPGER",
        "MEAPAQLLFLLLLWLPDTTR",
        "ASQSVSSSYLTWYQQKPGQAPR",
        "EIVMTQSPPTLSLSPGERVTLSC+57.021R",
        "VTLSC+57.021RASQSVSSSYLTWYQQKPGQAPR",
        "ASQSVSSSYLTWYQQKPGQAPRLLIYGASTR",
        "FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP",
        "MEAPAQLLFLLLLWLPDTTREIVMTQSPPTLSLSPGER",
        "ATSIPARFSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP",
    ]

    # 3 missed cleavages
    expected_3missedcleavage = [
        "ATSIPAR",
        "VTLSC+57.021R",
        "LLIYGASTR",
        "LLIYGASTRATSIPAR",
        "EIVMTQSPPTLSLSPGER",
        "MEAPAQLLFLLLLWLPDTTR",
        "ASQSVSSSYLTWYQQKPGQAPR",
        "EIVMTQSPPTLSLSPGERVTLSC+57.021R",
        "VTLSC+57.021RASQSVSSSYLTWYQQKPGQAPR",
        "ASQSVSSSYLTWYQQKPGQAPRLLIYGASTR",
        "FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP",
        "ASQSVSSSYLTWYQQKPGQAPRLLIYGASTRATSIPAR",
        "VTLSC+57.021RASQSVSSSYLTWYQQKPGQAPRLLIYGASTR",
        "MEAPAQLLFLLLLWLPDTTREIVMTQSPPTLSLSPGER",
        "ATSIPARFSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP",
        "VTLSC+57.021RASQSVSSSYLTWYQQKPGQAPRLLIYGASTRATSIPAR",
        "MEAPAQLLFLLLLWLPDTTREIVMTQSPPTLSLSPGERVTLSC+57.021R",
        "EIVMTQSPPTLSLSPGERVTLSC+57.021RASQSVSSSYLTWYQQKPGQAPR",
        "LLIYGASTRATSIPARFSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP",
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
            allowed_fixed_mods="C:C+57.021",
            allowed_var_mods=(
                "M:M+15.995,N:N+0.984,Q:Q+0.984,"
                "nterm:+42.011,nterm:+43.006,nterm:-17.027,nterm:+43.006-17.027"
            ),
            tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
        )
        assert pdb.db_peptides.index.to_list() == expected


def test_digest_fasta_mods(tiny_fasta_file):
    # 1 modification allowed
    # fixed: C+57.02146
    # variable: 1M+15.994915,1N+0.984016,1Q+0.984016
    # nterm: 1X+42.010565,1X+43.005814,1X-17.026549,1X+25.980265
    expected_1mod = [
        "-17.027ATSIPAR",
        "ATSIPAR",
        "-17.027VTLSC+57.021R",
        "VTLSC+57.021R",
        "+43.006-17.027ATSIPAR",
        "+42.011ATSIPAR",
        "+43.006ATSIPAR",
        "+43.006-17.027VTLSC+57.021R",
        "+42.011VTLSC+57.021R",
        "+43.006VTLSC+57.021R",
        "-17.027LLIYGASTR",
        "LLIYGASTR",
        "+43.006-17.027LLIYGASTR",
        "+42.011LLIYGASTR",
        "+43.006LLIYGASTR",
        "-17.027EIVMTQSPPTLSLSPGER",
        "EIVMTQSPPTLSLSPGER",
        "EIVMTQ+0.984SPPTLSLSPGER",
        "EIVM+15.995TQSPPTLSLSPGER",
        "+43.006-17.027EIVMTQSPPTLSLSPGER",
        "+42.011EIVMTQSPPTLSLSPGER",
        "+43.006EIVMTQSPPTLSLSPGER",
        "-17.027MEAPAQLLFLLLLWLPDTTR",
        "-17.027M+15.995EAPAQLLFLLLLWLPDTTR",
        "MEAPAQLLFLLLLWLPDTTR",
        "MEAPAQ+0.984LLFLLLLWLPDTTR",
        "M+15.995EAPAQLLFLLLLWLPDTTR",
        "+43.006-17.027MEAPAQLLFLLLLWLPDTTR",
        "+43.006-17.027M+15.995EAPAQLLFLLLLWLPDTTR",
        "+42.011MEAPAQLLFLLLLWLPDTTR",
        "+43.006MEAPAQLLFLLLLWLPDTTR",
        "+42.011M+15.995EAPAQLLFLLLLWLPDTTR",
        "+43.006M+15.995EAPAQLLFLLLLWLPDTTR",
        "-17.027ASQSVSSSYLTWYQQKPGQAPR",
        "ASQSVSSSYLTWYQQKPGQAPR",
        "ASQSVSSSYLTWYQ+0.984QKPGQAPR",
        "ASQSVSSSYLTWYQQ+0.984KPGQAPR",
        "ASQ+0.984SVSSSYLTWYQQKPGQAPR",
        "ASQSVSSSYLTWYQQKPGQ+0.984APR",
        "+43.006-17.027ASQSVSSSYLTWYQQKPGQAPR",
        "+42.011ASQSVSSSYLTWYQQKPGQAPR",
        "+43.006ASQSVSSSYLTWYQQKPGQAPR",
        "-17.027FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP",
        "FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP",
        "FSGSGSGTDFTLTISSLQ+0.984PEDFAVYYC+57.021QQDYNLP",
        "FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYN+0.984LP",
        "FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021Q+0.984QDYNLP",
        "FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQ+0.984DYNLP",
        "+43.006-17.027FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP",
        "+42.011FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP",
        "+43.006FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP",
    ]
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
        allowed_fixed_mods="C:C+57.021",
        allowed_var_mods=(
            "M:M+15.995,N:N+0.984,Q:Q+0.984,"
            "nterm:+42.011,nterm:+43.006,nterm:-17.027,nterm:+43.006-17.027"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    assert pdb.db_peptides.index.to_list() == expected_1mod


def test_length_restrictions(tiny_fasta_file):
    # length between 20 and 50
    expected_long = [
        "MEAPAQLLFLLLLWLPDTTR",
        "ASQSVSSSYLTWYQQKPGQAPR",
        "FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP",
    ]

    # length between 6 and 8
    expected_short = ["ATSIPAR", "VTLSC+57.021R"]

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
        allowed_fixed_mods="C:C+57.021",
        allowed_var_mods=(
            "M:M+15.995,N:N+0.984,Q:Q+0.984,"
            "nterm:+42.011,nterm:+43.006,nterm:-17.027,nterm:+43.006-17.027"
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
        allowed_fixed_mods="C:C+57.021",
        allowed_var_mods=(
            "M:M+15.995,N:N+0.984,Q:Q+0.984,"
            "nterm:+42.011,nterm:+43.006,nterm:-17.027,nterm:+43.006-17.027"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    assert pdb.db_peptides.index.to_list() == expected_short


def test_digest_fasta_enzyme(tiny_fasta_file):
    # arg-c enzyme
    expected_argc = [
        "ATSIPAR",
        "VTLSC+57.021R",
        "LLIYGASTR",
        "EIVMTQSPPTLSLSPGER",
        "MEAPAQLLFLLLLWLPDTTR",
        "ASQSVSSSYLTWYQQKPGQAPR",
        "FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP",
    ]

    # asp-n enzyme
    expected_aspn = ["DFAVYYC+57.021QQ", "DFTLTISSLQPE", "MEAPAQLLFLLLLWLP"]

    expected_semispecific = [
        "FSGSGS",
        "ATSIPA",
        "ASQSVS",
        "PGQAPR",
        "TSIPAR",
        "MEAPAQ",
        "LLIYGA",
        "YGASTR",
        "LSPGER",
        "LPDTTR",
        "EIVMTQ",
        "VTLSC+57.021R",
        "QDYNLP",
    ]

    expected_nonspecific = [
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
        "QSVSSS",
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
        "STRATS",
        "LIYGAS",
        "ARFSGS",
        "PGQAPR",
        "SGTDFT",
        "PPTLSL",
        "EAPAQL",
        "QKPGQA",
        "SVSSSY",
        "TQSPPT",
        "LTISSL",
        "PARFSG",
        "GQAPRL",
        "QSPPTL",
        "SPGERV",
        "ISSLQP",
        "TSIPAR",
        "RATSIP",
        "MEAPAQ",
        "RASQSV",
        "TISSLQ",
        "TRATSI",
        "LLIYGA",
        "GTDFTL",
        "YGASTR",
        "VSSSYL",
        "SSSYLT",
        "LSPGER",
        "PGERVT",
        "MTQSPP",
        "SSLQPE",
        "VMTQSP",
        "GERVTL",
        "PEDFAV",
        "IVMTQS",
        "FTLTIS",
        "APRLLI",
        "QQKPGQ",
        "SLQPED",
        "PAQLLF",
        "IPARFS",
        "SIPARF",
        "LSC+57.021RAS",
        "TDFTLT",
        "QAPRLL",
        "LPDTTR",
        "ERVTLS",
        "AQLLFL",
        "QPEDFA",
        "TLSC+57.021RA",
        "SC+57.021RASQ",
        "C+57.021RASQS",
        "DFTLTI",
        "PDTTRE",
        "TTREIV",
        "EIVMTQ",
        "YQQKPG",
        "LFLLLL",
        "LLFLLL",
        "WLPDTT",
        "DTTREI",
        "RLLIYG",
        "RVTLSC+57.021",
        "VTLSC+57.021R",
        "EDFAVY",
        "LWLPDT",
        "QLLFLL",
        "LQPEDF",
        "TREIVM",
        "REIVMT",
        "QDYNLP",
        "LLLWLP",
        "SSYLTW",
        "LLWLPD",
        "LLLLWL",
        "PRLLIY",
        "DFAVYY",
        "QQDYNL",
        "AVYYC+57.021Q",
        "FLLLLW",
        "FAVYYC+57.021",
        "C+57.021QQDYN",
        "SYLTWY",
        "LTWYQQ",
        "WYQQKP",
        "TWYQQK",
        "VYYC+57.021QQ",
        "YLTWYQ",
        "YYC+57.021QQD",
        "YC+57.021QQDY",
    ]

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="arg-c",
        digestion="full",
        missed_cleavages=0,
        min_peptide_len=6,
        max_peptide_len=50,
        max_mods=0,
        precursor_tolerance=20,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C+57.021",
        allowed_var_mods=(
            "M:M+15.995,N:N+0.984,Q:Q+0.984,"
            "nterm:+42.011,nterm:+43.006,nterm:-17.027,nterm:+43.006-17.027"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    assert pdb.db_peptides.index.to_list() == expected_argc

    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="asp-n",
        digestion="full",
        missed_cleavages=0,
        min_peptide_len=6,
        max_peptide_len=50,
        max_mods=0,
        precursor_tolerance=20,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C+57.021",
        allowed_var_mods=(
            "M:M+15.995,N:N+0.984,Q:Q+0.984,"
            "nterm:+42.011,nterm:+43.006,nterm:-17.027,nterm:+43.006-17.027"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    assert pdb.db_peptides.index.to_list() == expected_aspn

    # Test regex rule instead of named enzyme
    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="R",
        digestion="full",
        missed_cleavages=0,
        min_peptide_len=6,
        max_peptide_len=50,
        max_mods=0,
        precursor_tolerance=20,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C+57.021",
        allowed_var_mods=(
            "M:M+15.995,N:N+0.984,Q:Q+0.984,"
            "nterm:+42.011,nterm:+43.006,nterm:-17.027,nterm:+43.006-17.027"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    assert pdb.db_peptides.index.to_list() == expected_argc

    # Test semispecific digest
    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="partial",
        missed_cleavages=0,
        min_peptide_len=6,
        max_peptide_len=6,
        max_mods=0,
        precursor_tolerance=10000,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C+57.021",
        allowed_var_mods=(
            "M:M+15.995,N:N+0.984,Q:Q+0.984,"
            "nterm:+42.011,nterm:+43.006,nterm:-17.027,nterm:+43.006-17.027"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    assert pdb.db_peptides.index.to_list() == expected_semispecific

    # Test nonspecific digest
    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="non-specific",
        missed_cleavages=0,
        min_peptide_len=6,
        max_peptide_len=6,
        max_mods=0,
        precursor_tolerance=10000,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C+57.021",
        allowed_var_mods=(
            "M:M+15.995,N:N+0.984,Q:Q+0.984,"
            "nterm:+42.011,nterm:+43.006,nterm:-17.027,nterm:+43.006-17.027"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(),
    )
    assert pdb.db_peptides.index.to_list() == expected_nonspecific

    # Test replace_isoleucine_with_leucine == True
    pdb = db_utils.ProteinDatabase(
        fasta_path=str(tiny_fasta_file),
        enzyme="trypsin",
        digestion="non-specific",
        missed_cleavages=0,
        min_peptide_len=6,
        max_peptide_len=6,
        max_mods=0,
        precursor_tolerance=10000,
        isotope_error=[0, 0],
        allowed_fixed_mods="C:C+57.021",
        allowed_var_mods=(
            "M:M+15.995,N:N+0.984,Q:Q+0.984,"
            "nterm:+42.011,nterm:+43.006,nterm:-17.027,nterm:+43.006-17.027"
        ),
        tokenizer=depthcharge.tokenizers.PeptideTokenizer.from_massivekb(
            replace_isoleucine_with_leucine=True
        ),
    )
    assert pdb.db_peptides.index.to_list() == expected_nonspecific


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

    expected_batch_all = {
        "precursor_mz": torch.Tensor([42.0] * 12 + [84.0] * 12),
        "precursor_charge": torch.Tensor([1] * 12 + [2] * 12),
        "seq": tokenizer.tokenize(peptides_one + peptides_two, add_stop=True),
        "peak_file": ["one.mgf"] * 12 + ["two.mgf"] * 12,
        "scan_id": [1] * 12 + [2] * 12,
    }

    num_spectra = 0
    for psm_batch in db_model._psm_batches(mock_batch):
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
    expected_widewindow = ["ATSIPAR", "VTLSC+57.021R", "LLIYGASTR"]

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
        allowed_fixed_mods="C:C+57.021",
        allowed_var_mods=(
            "M:M+15.995,N:N+0.984,Q:Q+0.984,"
            "nterm:+42.011,nterm:+43.006,nterm:-17.027,nterm:+43.006-17.027"
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
        allowed_fixed_mods="C:C+57.021",
        allowed_var_mods=(
            "M:M+15.995,N:N+0.984,Q:Q+0.984,"
            "nterm:+42.011,nterm:+43.006,nterm:-17.027,nterm:+43.006-17.027"
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
        allowed_fixed_mods="C:C+57.021",
        allowed_var_mods=(
            "M:M+15.995,N:N+0.984,Q:Q+0.984,"
            "nterm:+42.011,nterm:+43.006,nterm:-17.027,nterm:+43.006-17.027"
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
        allowed_fixed_mods="C:C+57.021",
        allowed_var_mods=(
            "M:M+15.995,N:N+0.984,Q:Q+0.984,"
            "nterm:+42.011,nterm:+43.006,nterm:-17.027,nterm:+43.006-17.027"
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
        allowed_fixed_mods="C:C+57.021",
        allowed_var_mods=(
            "M:M+15.995,N:N+0.984,Q:Q+0.984,"
            "nterm:+42.011,nterm:+43.006,nterm:-17.027,nterm:+43.006-17.027"
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
        allowed_fixed_mods="C:C+57.021",
        allowed_var_mods=(
            "M:M+15.995,N:N+0.984,Q:Q+0.984,"
            "nterm:+42.011,nterm:+43.006,nterm:-17.027,nterm:+43.006-17.027"
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
        allowed_fixed_mods="C:C+57.021",
        allowed_var_mods=(
            "M:M+15.995,N:N+0.984,Q:Q+0.984,"
            "nterm:+42.011,nterm:+43.006,nterm:-17.027,nterm:+43.006-17.027"
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
        model.on_predict_batch_end(model.predict_step(None))

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
        lance_dir=tmp_path.name,
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
        for i, (filename, scan_id) in enumerate(
            [
                (mgf_small, "0"),
                (mgf_small, "1"),
                (mgf_small2, "0"),
                (mgf_small2, "1"),
            ]
        ):
            assert dataset[i]["peak_file"][0] == filename.name
            assert dataset[i]["scan_id"][0] == scan_id


def test_spectrum_id_mzml(mzml_small, tmp_path):
    """Test that spectra from mzML files are specified by their scan id."""
    mzml_small2 = tmp_path / "mzml_small2.mzml"
    shutil.copy(mzml_small, mzml_small2)
    data_module = DeNovoDataModule(
        lance_dir=tmp_path.name,
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

    # Check if smoothing is applied in training and not in validation
    assert model.celoss.label_smoothing == 0.1
    assert model.val_celoss.label_smoothing == 0
    assert val_step_loss != train_step_loss


def test_run_map(mgf_small):
    out_writer = ms_io.MztabWriter("dummy.mztab")
    # Set peak file by base file name only.
    out_writer.set_ms_run([os.path.basename(mgf_small.name)])
    assert mgf_small.name in out_writer._run_map
    assert os.path.abspath(mgf_small.name) not in out_writer._run_map


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
        tmp_path.name,
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
        tmp_path.name,
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
        tmp_path.name,
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


def test_beam_search_decode(tiny_config):
    """
    Test beam search decoding and its sub-functions.
    """
    config = casanovo.Config(tiny_config)
    model = Spec2Pep(
        n_beams=4,
        residues="massivekb",
        min_peptide_len=4,
        tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
            residues=config.residues
        ),
    )
    model.tokenizer.reverse = False  # For simplicity

    # Sizes
    batch = 1  # B
    length = model.max_peptide_len + 1  # L
    vocab = len(model.tokenizer) + 1  # V
    beam = model.n_beams  # S
    step = 3
    device = model.device

    # Initialize required attributes
    model._batch_size = batch
    model._beam_size = beam
    model._cumulative_masses = torch.zeros(batch * beam, device=device)

    # Initialize scores and tokens
    scores = torch.full(
        size=(batch, length, vocab, beam), fill_value=torch.nan, device=device
    )
    scores = einops.rearrange(scores, "B L V S -> (B S) L V")
    tokens = torch.zeros(
        batch * beam, length, dtype=torch.int64, device=device
    )

    # Create cache for decoded beams
    pred_cache = collections.OrderedDict((i, []) for i in range(batch))

    # Ground truth peptide is "PEPK"
    true_peptide = "PEPK"
    precursors = torch.tensor(
        [469.25364, 2.0, 235.63410], device=device
    ).repeat(beam * batch, 1)

    # Fill scores and tokens with relevant predictions
    scores[:, : step + 1, :] = 0
    peptides = [
        ("PEPK", False),
        ("PEPR", False),
        ("PEPG", False),
        ("PEP", True),
    ]
    for i, (peptide, add_stop) in enumerate(peptides):
        tokens[i, : step + 1] = model.tokenizer.tokenize(
            peptide, add_stop=add_stop
        )[0]
        for j in range(step + 1):
            scores[i, j, tokens[i, j]] = 1

    # Set cumulative masses
    for i in range(batch * beam):
        mass = 0
        for j in range(step + 1):
            token = tokens[i, j].item()
            if token != 0 and token < len(model.token_masses):
                mass += model.token_masses[token]
        model._cumulative_masses[i] = mass

    try:
        # Test _finish_beams()
        (
            finished_beams,
            beam_fits_precursor,
            discarded_beams,
        ) = model._finish_beams(tokens, precursors, step)

        # Second beam finished due to the precursor m/z filter, final beam
        # finished due to predicted stop token, first and third beam
        # unfinished. Final beam discarded due to length.
        assert torch.equal(
            finished_beams,
            torch.tensor([False, True, False, True], device=device),
        )
        assert torch.equal(
            beam_fits_precursor,
            torch.tensor([False, False, False, False], device=device),
        )
        assert torch.equal(
            discarded_beams,
            torch.tensor([False, False, False, True], device=device),
        )

        # Test _cache_finished_beams()
        model._cache_finished_beams(
            tokens,
            scores,
            step,
            finished_beams & ~discarded_beams,
            beam_fits_precursor,
            pred_cache,
        )

        # Verify that the correct peptides have been cached
        correct_cached = 0
        for _, _, _, pep in pred_cache[0]:
            if torch.equal(pep, model.tokenizer.tokenize("PEPR")[0]):
                correct_cached += 1
        assert correct_cached == 1

        # Test _get_top_peptide()
        # Create test cache
        test_cache = collections.OrderedDict((i, []) for i in range(batch))
        heapq.heappush(
            test_cache[0],
            (0.93, 0.1, 4 * [0.93], model.tokenizer.tokenize("PEPY")[0]),
        )
        heapq.heappush(
            test_cache[0],
            (0.95, 0.2, 4 * [0.95], model.tokenizer.tokenize("PEPK")[0]),
        )
        heapq.heappush(
            test_cache[0],
            (0.94, 0.3, 4 * [0.94], model.tokenizer.tokenize("PEPP")[0]),
        )

        # Verify that the highest scoring peptide is returned
        assert next(model._get_top_peptide(test_cache))[0][-1] == "PEPK"

        # Test empty predictions case
        empty_cache = collections.OrderedDict((i, []) for i in range(batch))
        assert len(list(model._get_top_peptide(empty_cache))[0]) == 0

        # Test multiple PSMs per spectrum
        model.top_match = 2
        top_results = next(model._get_top_peptide(test_cache))
        assert set([pep[-1] for pep in top_results]) == {"PEPK", "PEPP"}

        # Test reverse AA scores when decoder is reversed.
        pred_cache = {
            0: [
                (
                    1.0,
                    0.42,
                    np.array([1.0, 0.0]),
                    model.tokenizer.tokenize("PE")[0],
                )
            ]
        }

        # Test when tokenizer is reversed
        model.tokenizer.reverse = True
        top_peptides = list(model._get_top_peptide(pred_cache))
        assert len(top_peptides) == 1
        assert len(top_peptides[0]) == 1
        assert np.allclose(top_peptides[0][0][1], np.array([0.0, 1.0]))
        assert top_peptides[0][0][2] == "EP"

        # Test when tokenizer is not reversed
        model.tokenizer.reverse = False
        top_peptides = list(model._get_top_peptide(pred_cache))
        assert len(top_peptides) == 1
        assert len(top_peptides[0]) == 1
        assert np.allclose(top_peptides[0][0][1], np.array([1.0, 0.0]))
        assert top_peptides[0][0][2] == "PE"

        # Test _get_topk_beams()
        # Reset test state
        step = 4
        scores[2, step, :] = 0
        next_tokens = model.tokenizer.tokenize(["P", "S", "A", "G"]).flatten()
        scores[2, step, next_tokens] = torch.tensor(
            [4.0, 3.0, 2.0, 1.0], device=device
        )
        test_finished_beams = torch.tensor(
            [True, True, False, True], device=device
        )

        # Update cumulative masses
        token_indices = [
            model.tokenizer.index[aa] for aa in ["P", "S", "A", "G"]
        ]
        token_masses = [model.token_masses[idx] for idx in token_indices]
        base_mass = model._cumulative_masses[2]

        new_tokens, new_scores = model._get_topk_beams(
            tokens, scores, test_finished_beams, batch, step
        )

        # Verify generated tokens and scores
        expected_tokens = model.tokenizer.tokenize(
            ["PEPGP", "PEPGS", "PEPGA", "PEPGG"]
        )
        assert torch.equal(new_tokens[:, : step + 1], expected_tokens)

        # Verify cumulative masses are correctly updated
        for i, token in enumerate(token_indices):
            expected_mass = base_mass + token_masses[i]
            assert torch.isclose(model._cumulative_masses[i], expected_mass)

        # Test _finish_beams with negative mass tokens
        # Reset model and test conditions
        model = Spec2Pep(
            n_beams=2,
            tokenizer=depthcharge.tokenizers.peptides.MskbPeptideTokenizer(
                residues=config.residues
            ),
        )

        # Initialize attributes
        model._batch_size = batch
        model._beam_size = 2
        model._cumulative_masses = torch.zeros(batch * 2, device=device)

        step = 1
        beam = 2

        # Ground truth peptide is "-17.027GK".
        precursors = torch.tensor(
            [186.10044, 2.0, 94.05750], device=device
        ).repeat(beam * batch, 1)
        tokens = torch.zeros(
            batch * beam, length, dtype=torch.int64, device=device
        )
        tokens[:, : step + 1] = model.tokenizer.tokenize(["GK", "AK"])

        # Set cumulative masses
        for i in range(batch * beam):
            mass = 0
            for j in range(step + 1):
                token = tokens[i, j].item()
                if token != 0 and token < len(model.token_masses):
                    mass += model.token_masses[token]
            model._cumulative_masses[i] = mass

        # Test _finish_beams
        (
            finished_beams,
            beam_fits_precursor,
            discarded_beams,
        ) = model._finish_beams(tokens, precursors, step)
        assert torch.equal(
            finished_beams, torch.tensor([False, True], device=device)
        )
        assert torch.equal(
            beam_fits_precursor, torch.tensor([False, False], device=device)
        )
        assert torch.equal(
            discarded_beams, torch.tensor([False, False], device=device)
        )

        # Test _finish_beams with multiple/internal N-mods and dummy predictions
        model = Spec2Pep(
            n_beams=3,
            min_peptide_len=3,
            tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
                residues=config.residues
            ),
        )

        # Initialize attributes
        model._batch_size = batch
        model._beam_size = 3
        model._cumulative_masses = torch.zeros(batch * 3, device=device)

        beam = 3
        step = 4

        # Precursor m/z irrelevant for this test
        precursors = torch.tensor(
            [1861.0044, 2.0, 940.5750], device=device
        ).repeat(beam * batch, 1)

        # Use sequences with invalid mass modifications
        tokens = torch.zeros(
            batch * beam, length, dtype=torch.int64, device=device
        )
        sequences = [
            ["K", "A", "A", "A", "[+25.980265]-"],
            ["K", "A", "A", "[Acetyl]-", "A"],
            ["K", "A", "A", "[Carbamyl]-", "[Ammonia-loss]-"],
        ]

        for i, seq in enumerate(sequences):
            tokens[i, : step + 1] = torch.tensor(
                [model.tokenizer.index[aa] for aa in seq], device=device
            )

        # Set cumulative masses
        for i in range(batch * beam):
            mass = 0
            for j in range(step + 1):
                token = tokens[i, j].item()
                if token != 0 and token < len(model.token_masses):
                    mass += model.token_masses[token]
            model._cumulative_masses[i] = mass

        # Test _finish_beams - all should be discarded
        (
            finished_beams,
            beam_fits_precursor,
            discarded_beams,
        ) = model._finish_beams(tokens, precursors, step)
        assert torch.equal(
            finished_beams, torch.tensor([False, False, False], device=device)
        )
        assert torch.equal(
            beam_fits_precursor,
            torch.tensor([False, False, False], device=device),
        )
        assert torch.equal(
            discarded_beams, torch.tensor([False, True, True], device=device)
        )

        # Test _get_topk_beams with finished beams in the batch
        model = Spec2Pep(
            n_beams=1,
            min_peptide_len=3,
            tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
                residues=config.residues
            ),
        )

        # Sizes and other variables
        batch = 2  # B
        beam = model.n_beams  # S

        # Initialize attributes
        model._batch_size = batch
        model._beam_size = beam
        model._cumulative_masses = torch.zeros(batch * beam, device=device)

        length = model.max_peptide_len + 1  # L
        vocab = len(model.tokenizer) + 1  # V
        step = 4

        # Initialize dummy scores and tokens
        scores = torch.full(
            size=(batch, length, vocab, beam),
            fill_value=torch.nan,
            device=device,
        )
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")
        tokens = torch.zeros(
            batch * beam, length, dtype=torch.int64, device=device
        )

        # Simulate non-zero amino acid-level probability scores
        scores[:, : step + 1, :] = torch.rand(
            batch * beam, step + 1, vocab, device=device
        )
        scores[:, step, range(1, 4)] = torch.tensor(
            [1.0, 2.0, 3.0], device=device
        )

        # Simulate one finished and one unfinished beam in the same batch
        tokens[0, :step] = model.tokenizer.tokenize("PEP", add_stop=True)[
            0
        ].to(device)
        tokens[1, :step] = model.tokenizer.tokenize("PEPG")[0].to(device)

        # Set cumulative masses
        for i in range(batch * beam):
            mass = 0
            for j in range(step):
                token = tokens[i, j].item()
                if token != 0 and token < len(model.token_masses):
                    mass += model.token_masses[token]
            model._cumulative_masses[i] = mass

        # Set finished beams array to allow decoding from only one beam
        test_finished_beams = torch.tensor([True, False], device=device)

        new_tokens, new_scores = model._get_topk_beams(
            tokens, scores, test_finished_beams, batch, step
        )

        # Only the second peptide should have a new token predicted
        expected_tokens = tokens.clone()
        expected_tokens[1, len("PEPG")] = 3

        assert torch.equal(new_tokens, expected_tokens)

        # Test full functionality of beam search decode
        # Test max peptide length and early stopping
        model.max_peptide_len = 0
        # 1 spectrum with 5 peaks (2 values: m/z and intensity)
        mzs = ints = torch.zeros(1, 5, device=device)
        precursors = torch.tensor([[469.25364, 2.0, 235.63410]], device=device)

        # Expected to stop before loop begins
        assert (
            len(list(model.beam_search_decode(mzs, ints, precursors))[0]) == 0
        )

        # Reset max peptide length
        model.max_peptide_len = 100

        # Test that duplicate peptide scores don't lead to a conflict in the cache.
        model = Spec2Pep(
            n_beams=1,
            min_peptide_len=3,
            tokenizer=depthcharge.tokenizers.peptides.PeptideTokenizer(
                residues=config.residues
            ),
        )

        # Initialize attributes
        model._batch_size = 2
        model._beam_size = 1
        model._cumulative_masses = torch.zeros(2 * 1, device=device)

        batch = 2  # B
        beam = model.n_beams  # S
        length = model.max_peptide_len + 1  # L
        vocab = len(model.tokenizer) + 1  # V
        step = 4

        # Simulate beams with identical amino acid scores but different tokens
        scores = torch.zeros(size=(batch * beam, length, vocab), device=device)
        scores[: batch * beam, : step + 1, :] = torch.rand(1, device=device)
        tokens = torch.zeros(
            batch * beam, length, dtype=torch.int64, device=device
        )
        tokens[: batch * beam, :step] = torch.randint(
            1, vocab, (batch * beam, step), device=device
        )

        # Set cumulative masses
        for i in range(batch * beam):
            mass = 0
            for j in range(step):
                token = tokens[i, j].item()
                if token != 0 and token < len(model.token_masses):
                    mass += model.token_masses[token]
            model._cumulative_masses[i] = mass

        pred_cache = collections.OrderedDict((i, []) for i in range(batch))
        model._cache_finished_beams(
            tokens,
            scores,
            step,
            torch.ones(batch * beam, dtype=torch.bool, device=device),
            torch.ones(batch * beam, dtype=torch.bool, device=device),
            pred_cache,
        )

        for beam_i, preds in pred_cache.items():
            assert len(preds) == beam
            peptide_scores = [pep[0] for pep in preds]
            assert np.allclose(peptide_scores, peptide_scores[0])

    finally:
        # Clean up temporary attributes
        for attr in ["_cumulative_masses", "_batch_size", "_beam_size"]:
            if hasattr(model, attr):
                delattr(model, attr)


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

    # New model should stay alive
    new_model = _build_model(tok)
    finished, _, _ = new_model._finish_beams(TOKENS, PRECURSOR, STEP)
    assert (
        not finished.item()
    ), "New logic failed and beam terminated unexpectedly!"

    # Old-logic sentinel should kill
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
