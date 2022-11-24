import os
import platform
import tempfile

import pandas as pd
import github
import pytest
from pyteomics import mgf

from casanovo import casanovo
from casanovo import utils
from casanovo.denovo.model import Spec2Pep
import casanovo.denovo.model_runner as model_runner
import casanovo.denovo.evaluate as evaluate


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
    model = Spec2Pep(tb_summarywriter="test_path")
    assert model.tb_summarywriter is not None

    model = Spec2Pep()
    assert model.tb_summarywriter is None


def test_eval(tmp_path):
    data_len = 99

    d = tmp_path / "eval_files"
    d.mkdir()

    mgf_file_path = str(d / "input.mgf")
    mzt_file_path = str(d / "casanovo_evaluate_output.mztab")

    mgf_dict = []
    # Will generate .mgf with only "VARN" peptide seqs
    for i in range(data_len):
        scan = i
        seq = "VARN"
        params_dict = {
            "TITLE": "irrelevant" + "_" + str(i),
            "PEPMASS": 314.1592653,
            "CHARGE": "2+",
            "SCANS": scan,
            "RTINSECONDS": 825.62256,
            "SEQ": seq,
        }
        td = {
            "m/z array": [1, 2],
            "intensity array": [1, 2],
            "params": params_dict,
        }

        mgf_dict.append(td)
    mgf.write(mgf_dict, mgf_file_path, file_mode="w")  # Write .mgf to temp dir

    psm_ids = []
    all_scores = []
    for i in range(data_len):
        psm_ids.append(i)
        all_scores.append(
            0.99 if i < int(0.75 * data_len) else -0.99
        )  # Generate 25% negative and 75% positive scores, threshold should be 0.75*data_len
    mzt_dict = {
        "PSH": ["PSM"] * data_len,
        "sequence": ["VARN"] * data_len,
        "PSM_ID": psm_ids,
        "accession": ["null"] * data_len,
        "unique": ["null"] * data_len,
        "database": ["null"] * data_len,
        "database_version": ["null"] * data_len,
        "search_engine": ["null"] * data_len,
        "search_engine_score[1]": all_scores,
        "modifications": ["null"] * data_len,
        "retention_time": ["null"] * data_len,
        "charge": [2] * data_len,
        "exp_mass_to_charge": ["null"] * data_len,
        "calc_mass_to_charge": ["null"] * data_len,
        "spectra_ref": ["null"] * data_len,
        "pre": ["null"] * data_len,
        "post": ["null"] * data_len,
        "start": ["null"] * data_len,
        "end": ["null"] * data_len,
        "opt_ms_run[1]_aa_scores": ["null"] * data_len,
    }
    mzt_df = pd.DataFrame.from_dict(mzt_dict)
    mzt_df.to_csv(
        mzt_file_path, index=False, sep="\t"
    )  # Write .mztab to temp dir

    (
        scan,
        true_seq,
        output_seq,
        output_score,
        precision,
        coverage,
    ) = evaluate._get_preccov_mztab_mgf(mzt_file_path, mgf_file_path)

    # Ensure benchmark values consistent
    for prec in precision:
        assert prec == 1
    assert coverage[-1] == 1

    # Ensure graph generation does not crash
    model_runner.generate_pc_graph(mgf_file_path, mzt_file_path)


def test_eval_metrics():
    """
    Test that peptide and amino acid-level evaluation metrics.
    Predicted AAs are considered correct match if they're <0.1Da from
    the corresponding ground truth (GT) AA with either a suffix or
    prefix <0.5Da from GT. A peptide prediction is correct if all
    its AA are correct matches.
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

    aa_matches, n_pred_aa, n_gt_aa = evaluate.aa_match_batch(
        peptides1=preds,
        peptides2=gt,
        aa_dict=model.decoder._peptide_mass.masses,
        mode="best",
    )

    assert n_pred_aa == 41
    assert n_gt_aa == 40

    aa_precision, aa_recall, pep_precision = evaluate.aa_match_metrics(
        aa_matches, n_gt_aa, n_pred_aa
    )

    assert round(2 / 8, 3) == round(pep_precision, 3)
    assert round(26 / 40, 3) == round(aa_recall, 3)
    assert round(26 / 41, 3) == round(aa_precision, 3)
