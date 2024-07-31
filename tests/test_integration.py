import functools
import subprocess
from pathlib import Path

import pyteomics.mztab
from click.testing import CliRunner

from casanovo import casanovo


TEST_DIR = Path(__file__).resolve().parent


def test_train_and_run(
    mgf_small, mzml_small, tiny_config, tmp_path, monkeypatch
):
    # We can use this to explicitly test different versions.
    monkeypatch.setattr(casanovo, "__version__", "3.0.1")

    # Run a command:
    run = functools.partial(
        CliRunner().invoke, casanovo.main, catch_exceptions=False
    )

    # Train a tiny model:
    train_args = [
        "train",
        "--validation_peak_path",
        str(mgf_small),
        "--config",
        tiny_config,
        "--output",
        str(tmp_path / "train"),
        str(mgf_small),  # The training files.
    ]

    result = run(train_args)
    model_file = tmp_path / "epoch=19-step=20.ckpt"
    best_model = tmp_path / "best.ckpt"
    assert result.exit_code == 0
    assert model_file.exists()
    assert best_model.exists()

    # Try predicting:
    output_filename = tmp_path / "test.mztab"
    predict_args = [
        "sequence",
        "--model",
        str(model_file),
        "--config",
        tiny_config,
        "--output",
        str(output_filename),
        str(mgf_small),
        str(mzml_small),
    ]

    result = run(predict_args)
    assert result.exit_code == 0
    assert output_filename.is_file()

    mztab = pyteomics.mztab.MzTab(str(output_filename))
    # Verify that both input peak files are listed in the metadata.
    for i, filename in enumerate(["small.mgf", "small.mzml"], 1):
        assert f"ms_run[{i}]-location" in mztab.metadata
        assert mztab.metadata[f"ms_run[{i}]-location"].endswith(filename)

    # Verify that the spectrum predictions are correct
    # and indexed according to the peak input file type.
    psms = mztab.spectrum_match_table
    assert psms.loc[1, "sequence"] == "LESLLEK"
    assert psms.loc[1, "spectra_ref"] == "ms_run[1]:index=0"
    assert psms.loc[2, "sequence"] == "PEPTLDEK"
    assert psms.loc[2, "spectra_ref"] == "ms_run[1]:index=1"
    assert psms.loc[3, "sequence"] == "LESLLEK"
    assert psms.loc[3, "spectra_ref"] == "ms_run[2]:scan=17"
    assert psms.loc[4, "sequence"] == "PEPTLDEK"
    assert psms.loc[4, "spectra_ref"] == "ms_run[2]:scan=111"

    # Finally, try evaluating:
    output_filename = tmp_path / "test-eval.mztab"
    eval_args = [
        "sequence",
        "--model",
        str(model_file),
        "--config",
        tiny_config,
        "--output",
        str(output_filename),
        str(mgf_small),
        str(mzml_small),
        "--evaluate",
    ]

    result = run(eval_args)
    assert result.exit_code == 0
    assert output_filename.is_file()

    mztab = pyteomics.mztab.MzTab(str(output_filename))
    # Verify that both input peak files are listed in the metadata.
    for i, filename in enumerate(["small.mgf"], 1):
        assert f"ms_run[{i}]-location" in mztab.metadata
        assert mztab.metadata[f"ms_run[{i}]-location"].endswith(filename)

    # Verify that the spectrum predictions are correct
    # and indexed according to the peak input file type.
    psms = mztab.spectrum_match_table
    assert psms.loc[1, "sequence"] == "LESLLEK"
    assert psms.loc[1, "spectra_ref"] == "ms_run[1]:index=0"
    assert psms.loc[2, "sequence"] == "PEPTLDEK"
    assert psms.loc[2, "spectra_ref"] == "ms_run[1]:index=1"

    # Validate mztab output
    validate_args = [
        "java",
        "-jar",
        f"{TEST_DIR}/jmzTabValidator.jar",
        "--check",
        f"inFile={output_filename}",
    ]

    validate_result = subprocess.run(
        validate_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert validate_result.returncode == 0
    assert not any(
        [
            line.startswith("[Error-")
            for line in validate_result.stdout.splitlines()
        ]
    )


def test_auxilliary_cli(tmp_path, monkeypatch):
    """Test the secondary CLI commands"""
    run = functools.partial(
        CliRunner().invoke, casanovo.main, catch_exceptions=False
    )

    monkeypatch.chdir(tmp_path)
    run("configure")
    assert Path("casanovo.yaml").exists()

    run(["configure", "-o", "test.yaml"])
    assert Path("test.yaml").exists()

    res = run("version")
    assert res.output
