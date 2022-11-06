import os
import platform
import tempfile
import pandas as pd

import pytest

import casanovo.denovo.model_runner as model_runner
import casanovo.denovo.evaluate as evaluate
from casanovo import casanovo
from casanovo.denovo.model import Spec2Pep

from pyteomics import mgf

"""Test that setuptools-scm is working correctly"""
import casanovo


def test_version():
    """Check that the version is not None"""
    assert casanovo.__version__ is not None


def test_graph_gen(tmp_path):
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

    precision, coverage, threshold = evaluate._get_preccov_mztab_mgf(
        mzt_file_path, mgf_file_path
    )

    # Ensure benchmark values consistent
    assert threshold == int(0.75 * data_len)
    for prec in precision:
        assert prec == 1
    assert coverage[-1] == 1

    # Ensure graph generation does not crash
    model_runner.generate_pc_graph(mgf_file_path, mzt_file_path)
