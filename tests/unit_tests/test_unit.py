import collections
import datetime
import functools
import hashlib
import heapq
import io
import itertools
import os
import pathlib
import platform
import re
import requests
import shutil
import tempfile
import unittest
import unittest.mock

import depthcharge
import depthcharge.tokenizers.peptides
import einops
import github
import numpy as np
import pandas as pd
import pytest
import torch

from casanovo import casanovo
from casanovo import utils
from casanovo.data import db_utils, ms_io
from casanovo.data.datasets import SpectrumDataset, AnnotatedSpectrumDataset
from casanovo.denovo.evaluate import aa_match, aa_match_batch, aa_match_metrics
from casanovo.denovo.model import Spec2Pep, _aa_pep_score, _calc_match_score
from casanovo.data import ms_io
from casanovo.denovo.dataloaders import DeNovoDataModule
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


class MockAsset:
    def __init__(self, file_name):
        self.name = file_name
        self.browser_download_url = f"http://example.com/{file_name}"


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


class MockGithub:
    def __init__(self, releases):
        self.releases = releases

    def get_repo(self, repo_name):
        return MockRepo()


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


def test_aa_pep_score():
    """
    Test the calculation of amino acid and peptide scores from the raw amino
    acid scores.
    """
    aa_scores_raw = np.asarray([0.0, 0.5, 1.0])

    aa_scores, peptide_score = _aa_pep_score(aa_scores_raw, True)
    np.testing.assert_array_equal(aa_scores, np.asarray([0.0, 0.25, 0.5]))
    assert peptide_score == pytest.approx(0.0)

    aa_scores, peptide_score = _aa_pep_score(aa_scores_raw, False)
    np.testing.assert_array_equal(aa_scores, np.asarray([0.0, 0.25, 0.5]))
    assert peptide_score == pytest.approx(-1.0)

    aa_scores_raw = np.asarray([1.0, 0.25])
    aa_scores, peptide_score = _aa_pep_score(aa_scores_raw, True)
    np.testing.assert_array_equal(aa_scores, np.asarray([0.75, 0.375]))
    assert peptide_score == pytest.approx(0.5)


def test_peptide_generator_errors(residues_dict, tiny_fasta_file):
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
    """
    Test the calculation of geometric scores using teacher-forced
    decoder output probabilities and ground truth amino acid sequences.
    """
    first_slot_prob = torch.zeros(29)
    first_slot_prob[1] = 1.0  # A
    second_slot_prob = torch.zeros(29)
    second_slot_prob[2] = 1.0  # B
    third_slot_prob = torch.zeros(29)
    third_slot_prob[3] = 1.0  # C
    stop_slot_prob = torch.zeros(29)
    stop_slot_prob[28] = 1.0  # $
    blank_slot_prob = torch.zeros(29)
    blank_slot_prob[0] = 0.42  # Should never come into play
    fourth_slot_prob = torch.zeros(29)
    fourth_slot_prob[4] = 0.5  # D
    fifth_slot_prob = torch.zeros(29)
    fifth_slot_prob[5] = 0.5  # E

    pep_1_aa = torch.stack(
        [
            first_slot_prob,
            second_slot_prob,
            third_slot_prob,
            stop_slot_prob,
            blank_slot_prob,
        ]
    )
    pep_2_aa = torch.stack(
        [
            third_slot_prob,
            second_slot_prob,
            stop_slot_prob,
            blank_slot_prob,
            blank_slot_prob,
        ]
    )
    pep_3_aa = torch.stack(
        [
            fourth_slot_prob,
            fifth_slot_prob,
            first_slot_prob,
            stop_slot_prob,
            blank_slot_prob,
        ]
    )
    pep_4_aa = torch.stack(
        [
            first_slot_prob,
            second_slot_prob,
            third_slot_prob,
            stop_slot_prob,
            blank_slot_prob,
        ]
    )
    batch_all_aa_scores = torch.stack([pep_1_aa, pep_2_aa, pep_3_aa, pep_4_aa])
    truth_aa_indices = torch.tensor(
        [[1, 2, 3, 28], [3, 2, 28, 0], [4, 5, 1, 28], [2, 2, 3, 28]]
    )

    all_scores, masked_per_aa_scores = _calc_match_score(
        batch_all_aa_scores, truth_aa_indices, True
    )

    assert all_scores[0] == np.exp(0)
    assert all_scores[1] == np.exp(0)
    assert all_scores[2] == pytest.approx(
        np.exp(np.log(0.5 * 0.5 * 1 * 1) / 4)
    )
    assert all_scores[3] == pytest.approx(
        np.exp(np.log(1e-10 * 1 * 1 * 1) / 4)
    )

    aa_scores = np.array([1, 1, 1, 1])
    assert np.allclose(masked_per_aa_scores[0], (aa_scores + 1) / 2)
    aa_scores = np.array([1, 1, 1])
    assert np.allclose(masked_per_aa_scores[1], (aa_scores + 1) / 2)
    aa_scores = np.array([0.5, 0.5, 1, 1])
    assert np.allclose(
        masked_per_aa_scores[2],
        (aa_scores + np.exp(np.log(0.5 * 0.5 * 1 * 1) / 4)) / 2,
    )
    aa_scores = np.array([1e-10, 1, 1, 1])
    assert np.allclose(
        masked_per_aa_scores[3],
        (aa_scores + np.exp(np.log(1e-10 * 1 * 1 * 1) / 4)) / 2,
    )


def test_digest_fasta_cleave(tiny_fasta_file, residues_dict):

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
            residues=residues_dict,
        )
        assert pdb.db_peptides.index.to_list() == expected


def test_digest_fasta_mods(tiny_fasta_file, residues_dict):
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
        "-17.027M+15.995EAPAQLLFLLLLWLPDTTR",  #
        "MEAPAQLLFLLLLWLPDTTR",
        "MEAPAQ+0.984LLFLLLLWLPDTTR",
        "M+15.995EAPAQLLFLLLLWLPDTTR",
        "+43.006-17.027MEAPAQLLFLLLLWLPDTTR",
        "+43.006-17.027M+15.995EAPAQLLFLLLLWLPDTTR",  #
        "+42.011MEAPAQLLFLLLLWLPDTTR",
        "+43.006MEAPAQLLFLLLLWLPDTTR",
        "+42.011M+15.995EAPAQLLFLLLLWLPDTTR",  #
        "+43.006M+15.995EAPAQLLFLLLLWLPDTTR",  #
        "-17.027ASQSVSSSYLTWYQQKPGQAPR",
        "ASQSVSSSYLTWYQQKPGQAPR",
        "ASQ+0.984SVSSSYLTWYQQKPGQAPR",
        "ASQSVSSSYLTWYQ+0.984QKPGQAPR",
        "ASQSVSSSYLTWYQQ+0.984KPGQAPR",
        "ASQSVSSSYLTWYQQKPGQ+0.984APR",
        "+43.006-17.027ASQSVSSSYLTWYQQKPGQAPR",
        "+42.011ASQSVSSSYLTWYQQKPGQAPR",
        "+43.006ASQSVSSSYLTWYQQKPGQAPR",
        "-17.027FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP",
        "FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYNLP",
        "FSGSGSGTDFTLTISSLQ+0.984PEDFAVYYC+57.021QQDYNLP",
        "FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021Q+0.984QDYNLP",
        "FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQ+0.984DYNLP",
        "FSGSGSGTDFTLTISSLQPEDFAVYYC+57.021QQDYN+0.984LP",
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
        residues=residues_dict,
    )
    assert pdb.db_peptides.index.to_list() == expected_1mod


def test_length_restrictions(tiny_fasta_file, residues_dict):
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
        residues=residues_dict,
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
        residues=residues_dict,
    )
    assert pdb.db_peptides.index.to_list() == expected_short


def test_digest_fasta_enzyme(tiny_fasta_file, residues_dict):
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
        "RATSIP",
        "TSIPAR",
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
        "C+57.021RASQS",
        "SC+57.021RASQ",
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
        "REIVMT",
        "TREIVM",
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
        "YC+57.021QQDY",
        "YYC+57.021QQD",
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
        residues=residues_dict,
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
        residues=residues_dict,
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
        residues=residues_dict,
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
        residues=residues_dict,
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
        residues=residues_dict,
    )
    assert pdb.db_peptides.index.to_list() == expected_nonspecific


def test_get_candidates(tiny_fasta_file, residues_dict):
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
        residues=residues_dict,
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
        residues=residues_dict,
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
        residues=residues_dict,
    )
    candidates = pdb.get_candidates(precursor_mz=496.2, charge=2)
    assert expected_widewindow == list(candidates)


def test_get_candidates_isotope_error(tiny_fasta_file, residues_dict):

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
        residues=residues_dict,
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
        residues=residues_dict,
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
        residues=residues_dict,
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
        residues=residues_dict,
    )
    pdb.db_peptides = peptide_list
    candidates = pdb.get_candidates(precursor_mz=496.2, charge=2)
    assert expected_isotope0123 == list(candidates)


def test_beam_search_decode():
    """
    Test beam search decoding and its sub-functions.
    """
    model = Spec2Pep(n_beams=4, residues="massivekb", min_peptide_len=4)
    model.decoder.reverse = False  # For simplicity.

    # Sizes.
    batch = 1  # B
    length = model.max_length + 1  # L
    vocab = len(model.tokenizer) + 1  # V
    beam = model.n_beams  # S
    step = 3

    # Initialize scores and tokens.
    scores = torch.full(
        size=(batch, length, vocab, beam), fill_value=torch.nan
    )
    scores = einops.rearrange(scores, "B L V S -> (B S) L V")
    tokens = torch.zeros(batch * beam, length, dtype=torch.int64)
    # Create cache for decoded beams.
    pred_cache = collections.OrderedDict((i, []) for i in range(batch))

    # Ground truth peptide is "PEPK".
    true_peptide = "PEPK"
    precursors = torch.tensor([469.25364, 2.0, 235.63410]).repeat(
        beam * batch, 1
    )
    # Fill scores and tokens with relevant predictions.
    scores[:, : step + 1, :] = 0
    for i, peptide in enumerate(["PEPK", "PEPR", "PEPG", "PEP$"]):
        tokens[i, : step + 1] = model.tokenizer.tokenize(peptide)[0]
        for j in range(step + 1):
            scores[i, j, tokens[1, j]] = 1

    # Test _finish_beams().
    finished_beams, beam_fits_precursor, discarded_beams = model._finish_beams(
        tokens, precursors, step
    )
    # Second beam finished due to the precursor m/z filter, final beam finished
    # due to predicted stop token, first and third beam unfinished. Final beam
    # discarded due to length.
    assert torch.equal(
        finished_beams, torch.tensor([False, True, False, True])
    )
    assert torch.equal(
        beam_fits_precursor, torch.tensor([False, False, False, False])
    )
    assert torch.equal(
        discarded_beams, torch.tensor([False, False, False, True])
    )

    # Test _cache_finished_beams().
    model._cache_finished_beams(
        tokens,
        scores,
        step,
        finished_beams & ~discarded_beams,
        beam_fits_precursor,
        pred_cache,
    )
    # Verify that the correct peptides have been cached.
    correct_cached = 0
    for _, _, _, pep in pred_cache[0]:
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
    assert correct_cached == 1

    # Test _get_top_peptide().
    # Return the candidate peptide with the highest score
    test_cache = collections.OrderedDict((i, []) for i in range(batch))
    heapq.heappush(
        test_cache[0], (0.93, 0.1, 4 * [0.93], torch.tensor([4, 14, 4, 19]))
    )
    heapq.heappush(
        test_cache[0], (0.95, 0.2, 4 * [0.95], torch.tensor([4, 14, 4, 13]))
    )
    heapq.heappush(
        test_cache[0], (0.94, 0.3, 4 * [0.94], torch.tensor([4, 14, 4, 4]))
    )

    assert list(model._get_top_peptide(test_cache))[0][0][-1] == "PEPK"
    # Test that an empty predictions is returned when no beams have been
    # finished.
    empty_cache = collections.OrderedDict((i, []) for i in range(batch))
    assert len(list(model._get_top_peptide(empty_cache))[0]) == 0
    # Test multiple PSM per spectrum and if it's highest scoring peptides
    model.top_match = 2
    assert set(
        [pep[-1] for pep in list(model._get_top_peptide(test_cache))[0]]
    ) == {"PEPK", "PEPP"}

    # Test _get_topk_beams().
    # Set scores to proceed generating the unfinished beam.
    step = 4
    scores[2, step, :] = 0
    scores[2, step, range(1, 5)] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    # Modify finished beams array to allow decoding from only one beam
    test_finished_beams = torch.tensor([True, True, False, True])
    new_tokens, new_scores = model._get_topk_beams(
        tokens, scores, test_finished_beams, batch, step
    )
    expected_tokens = torch.tensor(
        [
            [4, 14, 4, 1, 4],
            [4, 14, 4, 1, 3],
            [4, 14, 4, 1, 2],
            [4, 14, 4, 1, 1],
        ]
    )
    # Only the expected scores of the final step.
    expected_scores = torch.zeros(beam, vocab)
    expected_scores[:, range(1, 5)] = torch.tensor([1.0, 2.0, 3.0, 4.0])

    assert torch.equal(new_tokens[:, : step + 1], expected_tokens)
    assert torch.equal(new_scores[:, step, :], expected_scores)

    # Test output if decoding loop isn't stopped with termination of all beams.
    model.max_peptide_len = 0
    # 1 spectrum with 5 peaks (2 values: m/z and intensity).
    spectra = torch.zeros(1, 5, 2)
    precursors = torch.tensor([[469.25364, 2.0, 235.63410]])
    assert len(list(model.beam_search_decode(spectra, precursors))[0]) == 0
    model.max_peptide_len = 100

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

    pred_cache = collections.OrderedDict((i, []) for i in range(batch))
    finished_beams = torch.tensor([True, True, True, True])
    beam_fits_precursor = torch.BoolTensor([False, True, True, False])

    model._cache_finished_beams(
        tokens, scores, step, finished_beams, beam_fits_precursor, pred_cache
    )
    # Verify predictions with matching/non-matching precursor m/z.
    positive_score = negative_score = 0
    for peptide_score, _, _, _ in pred_cache[0]:
        positive_score += peptide_score >= 0
        negative_score += peptide_score < 0
    assert positive_score == 2
    assert negative_score == 2

    # Test using a single beam only.
    model = Spec2Pep(n_beams=1, residues="massivekb", min_peptide_len=2)
    beam = model.n_beams  # S
    model.decoder.reverse = False  # For simplicity.
    aa2idx = model.decoder._aa2idx
    step = 4

    # Initialize scores and tokens.
    scores = torch.full(
        size=(batch, length, vocab, beam), fill_value=torch.nan
    )
    scores = einops.rearrange(scores, "B L V S -> (B S) L V")
    tokens = torch.zeros(batch * beam, length, dtype=torch.int64)

    pred_cache = collections.OrderedDict((i, []) for i in range(batch))

    # Ground truth peptide is "PEPK".
    true_peptide = "PEPK$"
    precursors = torch.tensor([469.25364, 2.0, 235.63410]).repeat(
        beam * batch, 1
    )
    scores[:, range(step), :] = 1
    tokens[0, : step + 1] = torch.tensor([aa2idx[aa] for aa in true_peptide])

    # Test _finish_beams().
    finished_beams, beam_fits_precursor, discarded_beams = model._finish_beams(
        tokens, precursors, step
    )

    assert torch.equal(finished_beams, torch.tensor([True]))
    assert torch.equal(beam_fits_precursor, torch.tensor([True]))
    assert torch.equal(discarded_beams, torch.tensor([False]))

    # Test _cache_finished_beams().
    model._cache_finished_beams(
        tokens, scores, step, finished_beams, beam_fits_precursor, pred_cache
    )

    assert torch.equal(pred_cache[0][0][-1], torch.tensor([4, 14, 4, 13]))

    # Test _get_topk_beams().
    step = 1
    scores = torch.full(
        size=(batch, length, vocab, beam), fill_value=torch.nan
    )
    scores = einops.rearrange(scores, "B L V S -> (B S) L V")
    tokens = torch.zeros(batch * beam, length, dtype=torch.int64)
    tokens[0, 0] = 4
    scores[0, step, :] = 0
    scores[0, step, 14] = torch.tensor([1])
    test_finished_beams = torch.tensor([False])

    new_tokens, new_scores = model._get_topk_beams(
        tokens, scores, test_finished_beams, batch, step
    )

    expected_tokens = torch.tensor(
        [
            [4, 14],
        ]
    )

    expected_scores = torch.zeros(beam, vocab)
    expected_scores[:, 14] = torch.tensor([1])

    assert torch.equal(new_scores[:, step, :], expected_scores)
    assert torch.equal(new_tokens[:, : step + 1], expected_tokens)

    # Test _finish_beams() for tokens with a negative mass.
    model = Spec2Pep(n_beams=2, residues="massivekb")
    beam = model.n_beams  # S
    aa2idx = model.decoder._aa2idx
    step = 1

    # Ground truth peptide is "-17.027GK".
    precursors = torch.tensor([186.10044, 2.0, 94.05750]).repeat(
        beam * batch, 1
    )
    tokens = torch.zeros(batch * beam, length, dtype=torch.int64)
    for i, peptide in enumerate(["GK", "AK"]):
        tokens[i, : step + 1] = torch.tensor([aa2idx[aa] for aa in peptide])

    # Test _finish_beams().
    finished_beams, beam_fits_precursor, discarded_beams = model._finish_beams(
        tokens, precursors, step
    )
    assert torch.equal(finished_beams, torch.tensor([False, True]))
    assert torch.equal(beam_fits_precursor, torch.tensor([False, False]))
    assert torch.equal(discarded_beams, torch.tensor([False, False]))

    # Test _finish_beams() for multiple/internal N-mods and dummy predictions.
    model = Spec2Pep(n_beams=3, residues="massivekb", min_peptide_len=3)
    beam = model.n_beams  # S
    model.decoder.reverse = True
    aa2idx = model.decoder._aa2idx
    step = 4

    # Ground truth peptide is irrelevant for this test.
    precursors = torch.tensor([1861.0044, 2.0, 940.5750]).repeat(
        beam * batch, 1
    )
    tokens = torch.zeros(batch * beam, length, dtype=torch.int64)
    # Reverse decoding
    for i, peptide in enumerate(
        [
            ["K", "A", "A", "A", "+43.006-17.027"],
            ["K", "A", "A", "+42.011", "A"],
            ["K", "A", "A", "+43.006", "+42.011"],
        ]
    ):
        tokens[i, : step + 1] = torch.tensor([aa2idx[aa] for aa in peptide])

    # Test _finish_beams(). All should be discarded
    finished_beams, beam_fits_precursor, discarded_beams = model._finish_beams(
        tokens, precursors, step
    )
    assert torch.equal(finished_beams, torch.tensor([False, False, False]))
    assert torch.equal(
        beam_fits_precursor, torch.tensor([False, False, False])
    )
    assert torch.equal(discarded_beams, torch.tensor([False, True, True]))

    # Test _get_topk_beams() with finished beams in the batch.
    model = Spec2Pep(n_beams=1, residues="massivekb", min_peptide_len=3)

    # Sizes and other variables.
    batch = 2  # B
    beam = model.n_beams  # S
    model.decoder.reverse = True
    length = model.max_peptide_len + 1  # L
    vocab = model.decoder.vocab_size + 1  # V
    step = 4

    # Initialize dummy scores and tokens.
    scores = torch.full(
        size=(batch, length, vocab, beam), fill_value=torch.nan
    )
    scores = einops.rearrange(scores, "B L V S -> (B S) L V")
    tokens = torch.zeros(batch * beam, length, dtype=torch.int64)

    # Simulate non-zero amino acid-level probability scores.
    scores[:, : step + 1, :] = torch.rand(batch, step + 1, vocab)
    scores[:, step, range(1, 4)] = torch.tensor([1.0, 2.0, 3.0])

    # Simulate one finished and one unfinished beam in the same batch.
    tokens[0, :step] = torch.tensor([4, 14, 4, 28])
    tokens[1, :step] = torch.tensor([4, 14, 4, 1])

    # Set finished beams array to allow decoding from only one beam.
    test_finished_beams = torch.tensor([True, False])

    new_tokens, new_scores = model._get_topk_beams(
        tokens, scores, test_finished_beams, batch, step
    )

    # Only the second peptide should have a new token predicted.
    expected_tokens = torch.tensor(
        [
            [4, 14, 4, 28, 0],
            [4, 14, 4, 1, 3],
        ]
    )

    assert torch.equal(new_tokens[:, : step + 1], expected_tokens)

    # Test that duplicate peptide scores don't lead to a conflict in the cache.
    model = Spec2Pep(n_beams=5, residues="massivekb", min_peptide_len=3)
    batch = 2  # B
    beam = model.n_beams  # S
    model.decoder.reverse = True
    length = model.max_peptide_len + 1  # L
    vocab = model.decoder.vocab_size + 1  # V
    step = 4

    # Simulate beams with identical amino acid scores but different tokens.
    scores = torch.zeros(size=(batch * beam, length, vocab))
    scores[: batch * beam, : step + 1, :] = torch.rand(1)
    tokens = torch.zeros(batch * beam, length, dtype=torch.int64)
    tokens[: batch * beam, :step] = torch.randint(
        1, vocab, (batch * beam, step)
    )

    pred_cache = collections.OrderedDict((i, []) for i in range(batch))
    model._cache_finished_beams(
        tokens,
        scores,
        step,
        torch.ones(batch * beam, dtype=torch.bool),
        torch.ones(batch * beam, dtype=torch.bool),
        pred_cache,
    )
    for beam_i, preds in pred_cache.items():
        assert len(preds) == beam
        peptide_scores = [pep[0] for pep in preds]
        assert np.allclose(peptide_scores, peptide_scores[0])


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
        train_paths=[mgf_small, mgf_small2],
        valid_paths=[mgf_small, mgf_small2],
        test_paths=[mgf_small, mgf_small2],
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
    """Test that spectra from mzML files are specified by their scan number."""
    mzml_small2 = tmp_path / "mzml_small2.mzml"
    shutil.copy(mzml_small, mzml_small2)

    index = SpectrumIndex(
        tmp_path / "index.hdf5", [mzml_small, mzml_small2], overwrite=True
    )
    dataset = SpectrumDataset(index)
    for i, (filename, scan_nr) in enumerate(
        [
            (mzml_small, 17),
            (mzml_small, 111),
            (mzml_small2, 17),
            (mzml_small2, 111),
        ]
    ):
        spectrum_id = str(filename), f"scan={scan_nr}"
        assert dataset.get_spectrum_id(i) == spectrum_id


def test_train_val_step_functions():
    """Test train and validation step functions operating on batches."""
    model = Spec2Pep(
        n_beams=1,
        residues="massivekb",
        min_peptide_len=4,
        train_label_smoothing=0.1,
    )
    spectra = torch.zeros(1, 5, 2)
    precursors = torch.tensor([[469.25364, 2.0, 235.63410]])
    peptides = ["PEPK"]
    batch = (spectra, precursors, peptides)

    train_step_loss = model.training_step(batch)
    val_step_loss = model.validation_step(batch)

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
    assert os.path.basename(mgf_small.name) not in out_writer._run_map
    assert os.path.abspath(mgf_small.name) in out_writer._run_map
    # Set peak file by full path.
    out_writer.set_ms_run([os.path.abspath(mgf_small.name)])
    assert os.path.basename(mgf_small.name) not in out_writer._run_map
    assert os.path.abspath(mgf_small.name) in out_writer._run_map


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
