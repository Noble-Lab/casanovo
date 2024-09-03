import collections
import heapq
import os
import platform
import re
import shutil
import tempfile

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
from casanovo.denovo.evaluate import aa_match_batch, aa_match_metrics
from casanovo.denovo.model import Spec2Pep, _aa_pep_score, _calc_match_score
from depthcharge.data import SpectrumIndex, AnnotatedSpectrumIndex


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


@pytest.mark.skip(reason="Hit rate limit during CI/CD")
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
    Test that the tensorboard.SummaryWriter object is only created when a folder
    path is passed.
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
        batch_all_aa_scores, truth_aa_indices
    )

    assert all_scores.numpy()[0] == 0
    assert all_scores.numpy()[1] == 0
    assert all_scores.numpy()[2] == pytest.approx(
        np.log(0.5 * 0.5 * 1 * 1) / 4
    )
    assert all_scores.numpy()[3] == pytest.approx(
        np.log(1e-10 * 1 * 1 * 1) / 4
    )

    assert np.sum(masked_per_aa_scores.numpy()[0]) == 4
    assert np.sum(masked_per_aa_scores.numpy()[1]) == 3
    assert np.sum(masked_per_aa_scores.numpy()[2]) == 3
    assert np.sum(masked_per_aa_scores.numpy()[3]) == 3


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
        peptide_list = list(pdb.db_peptides["peptide"])
        assert peptide_list == expected


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
        "MEAPAQLLFLLLLWLPDTTR",
        "MEAPAQ+0.984LLFLLLLWLPDTTR",
        "M+15.995EAPAQLLFLLLLWLPDTTR",
        "+43.006-17.027MEAPAQLLFLLLLWLPDTTR",
        "+42.011MEAPAQLLFLLLLWLPDTTR",
        "+43.006MEAPAQLLFLLLLWLPDTTR",
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
    peptide_list = list(pdb.db_peptides["peptide"])
    peptide_list = [
        x
        for x in peptide_list
        if not re.search(
            r"(\+42\.011|\+43\.006|\-17\.027|\+43\.006\-17\.027)+[A-Z]\+", x
        )
    ]
    assert peptide_list == expected_1mod


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
    peptide_list = list(pdb.db_peptides["peptide"])
    assert peptide_list == expected_long

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
    peptide_list = list(pdb.db_peptides["peptide"])
    assert peptide_list == expected_short


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
    peptide_list = list(pdb.db_peptides["peptide"])
    assert peptide_list == expected_argc

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
    peptide_list = list(pdb.db_peptides["peptide"])
    assert peptide_list == expected_aspn

    # Tesr regex rule instead of named enzyme
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
    peptide_list = list(pdb.db_peptides["peptide"])
    assert peptide_list == expected_argc


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
    candidates, _ = pdb.get_candidates(precursor_mz=496.2, charge=2)
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
    candidates, _ = pdb.get_candidates(precursor_mz=496.2, charge=2)
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
    candidates, _ = pdb.get_candidates(precursor_mz=496.2, charge=2)
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
    )
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
    candidates, _ = pdb.get_candidates(precursor_mz=496.2, charge=2)
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
    candidates, _ = pdb.get_candidates(precursor_mz=496.2, charge=2)
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
    candidates, _ = pdb.get_candidates(precursor_mz=496.2, charge=2)
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
    candidates, _ = pdb.get_candidates(precursor_mz=496.2, charge=2)
    assert expected_isotope0123 == list(candidates)


def test_beam_search_decode():
    """
    Test beam search decoding and its sub-functions.
    """
    model = Spec2Pep(n_beams=4, residues="massivekb", min_peptide_len=4)
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
    pred_cache = collections.OrderedDict((i, []) for i in range(batch))

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
    model.max_length = 0
    # 1 spectrum with 5 peaks (2 values: m/z and intensity).
    spectra = torch.zeros(1, 5, 2)
    precursors = torch.tensor([[469.25364, 2.0, 235.63410]])
    assert len(list(model.beam_search_decode(spectra, precursors))[0]) == 0
    model.max_length = 100

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
    length = model.max_length + 1  # L
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
    length = model.max_length + 1  # L
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


def test_spectrum_id_mgf(mgf_small, tmp_path):
    """Test that spectra from MGF files are specified by their index."""
    mgf_small2 = tmp_path / "mgf_small2.mgf"
    shutil.copy(mgf_small, mgf_small2)

    for index_func, dataset_func in [
        (SpectrumIndex, SpectrumDataset),
        (AnnotatedSpectrumIndex, AnnotatedSpectrumDataset),
    ]:
        index = index_func(
            tmp_path / "index.hdf5", [mgf_small, mgf_small2], overwrite=True
        )
        dataset = dataset_func(index)
        for i, (filename, mgf_i) in enumerate(
            [
                (mgf_small, 0),
                (mgf_small, 1),
                (mgf_small2, 0),
                (mgf_small2, 1),
            ]
        ):
            spectrum_id = str(filename), f"index={mgf_i}"
            assert dataset.get_spectrum_id(i) == spectrum_id


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
