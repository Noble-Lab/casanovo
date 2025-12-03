import functools
import subprocess
from pathlib import Path

import pyteomics.mztab
import pytest
import yaml
from click.testing import CliRunner

from casanovo import casanovo

TEST_DIR = Path(__file__).resolve().parent


def test_train_and_run(
    mgf_small,
    mzml_small,
    tiny_config,
    tiny_config_db,
    tmp_path,
    monkeypatch,
    mgf_medium,
    tiny_fasta_file,
):
    # We can use this to explicitly test different versions.
    monkeypatch.setattr(casanovo, "__version__", "3.0.1")

    # Run a command.
    run = functools.partial(
        CliRunner().invoke, casanovo.main, catch_exceptions=False
    )

    # Run Casanovo to train a tiny model.
    train_args = [
        "train",
        str(mgf_small),
        "--config",
        tiny_config,
        "--output_dir",
        str(tmp_path),
        "--output_root",
        "train",
    ]

    result = run(train_args)
    model_file = tmp_path / "train.epoch=19-step=20.ckpt"
    best_model = tmp_path / "train.best.ckpt"
    assert result.exit_code == 0
    assert model_file.exists()
    assert best_model.exists()

    # Run Casanovo in de novo prediction mode.
    output_rootname = "test"
    output_filename = (tmp_path / output_rootname).with_suffix(".mztab")
    predict_args = [
        "sequence",
        "--model",
        str(model_file),
        "--config",
        tiny_config,
        "--output_dir",
        str(tmp_path),
        "--output_root",
        output_rootname,
        str(mgf_small),
        str(mzml_small),
    ]

    result = run(predict_args)
    assert result.exit_code == 0
    assert output_filename.is_file()

    # Verify that the output file is correct.
    mztab = pyteomics.mztab.MzTab(str(output_filename))
    # Verify that both input peak files are listed in the metadata.
    for i, filename in enumerate(["small.mgf", "small.mzml"], 1):
        assert f"ms_run[{i}]-location" in mztab.metadata
        assert mztab.metadata[f"ms_run[{i}]-location"].endswith(filename)

    # Verify that the spectrum predictions are correct and indexed
    # according to the peak input file type.
    psms = mztab.spectrum_match_table
    assert psms.loc[1, "sequence"] == "LESLLEK"
    assert psms.loc[1, "spectra_ref"] == "ms_run[1]:index=0"
    assert psms.loc[2, "sequence"] == "PEPTLDEK"
    assert psms.loc[2, "spectra_ref"] == "ms_run[1]:index=1"
    assert psms.loc[3, "sequence"] == "PEPTLDEK"
    assert (
        psms.loc[3, "spectra_ref"]
        == "ms_run[2]:merged=11 frame=12 scanStart=763 scanEnd=787"
    )
    assert psms.loc[4, "sequence"] == "LESLLEK"
    assert psms.loc[4, "spectra_ref"] == "ms_run[2]:scan=17"

    # Run Casanovo in de novo evaluation mode.
    output_rootname = "test-eval"
    output_filename = (tmp_path / output_rootname).with_suffix(".mztab")
    eval_args = [
        "sequence",
        "--model",
        str(model_file),
        "--config",
        tiny_config,
        "--output_dir",
        str(tmp_path),
        "--output_root",
        output_rootname,
        str(mgf_small),
        "--evaluate",
    ]

    result = run(eval_args)
    assert result.exit_code == 0
    assert output_filename.is_file()

    # Verify that the output file is correct.
    mztab = pyteomics.mztab.MzTab(str(output_filename))
    filename = "small.mgf"
    # Verify that the input annotated peak file is listed in the metadata.
    assert "ms_run[1]-location" in mztab.metadata
    assert mztab.metadata["ms_run[1]-location"].endswith(filename)

    # Verify that the spectrum predictions are correct and indexed
    # according to the peak input file type.
    psms = mztab.spectrum_match_table
    assert psms.loc[1, "sequence"] == "LESLLEK"
    assert psms.loc[1, "spectra_ref"] == "ms_run[1]:index=0"
    assert psms.loc[2, "sequence"] == "PEPTLDEK"
    assert psms.loc[2, "spectra_ref"] == "ms_run[1]:index=1"

    # Validate the mzTab output file.
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

    assert output_filename.is_file()

    # Run Casanovo in database prediction mode.
    output_rootname = "db"
    output_filename = (tmp_path / output_rootname).with_suffix(".mztab")
    output_db_file = (tmp_path / output_rootname).with_suffix(".tsv")

    search_args = [
        "db-search",
        "--model",
        str(model_file),
        "--config",
        tiny_config_db,
        "--output_dir",
        str(tmp_path),
        "--output_root",
        output_rootname,
        "--export",
        str(mgf_medium),
        str(tiny_fasta_file),
    ]

    result = run(search_args)

    assert result.exit_code == 0
    assert output_filename.exists()
    assert output_db_file.exists()

    # Verify that the output file is correct.
    mztab = pyteomics.mztab.MzTab(str(output_filename))

    psms = mztab.spectrum_match_table
    assert list(psms.sequence) == [
        "ATSIPAR",
        "VTLSCR",
        "LLIYGASTR",
        "EIVMTQSPPTLSLSPGER",
        "MEAPAQLLFLLLLWLPDTTR",
        "ASQSVSSSYLTWYQQKPGQAPR",
        "FSGSGSGTDFTLTISSLQPEDFAVYYCQQDYNLP",
    ]

    mods = psms["modifications"].to_list()
    assert mods == [
        None,
        "5-Carbamidomethyl (C):UNIMOD:4",
        None,
        None,
        None,
        None,
        "27-Carbamidomethyl (C):UNIMOD:4",
    ]

    # Validate the mzTab output file.
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


def test_auxilliary_cli(tmp_path, mgf_small, monkeypatch):
    """Test the secondary CLI commands."""
    run = functools.partial(
        CliRunner().invoke, casanovo.main, catch_exceptions=False
    )

    monkeypatch.chdir(tmp_path)
    run("configure")
    assert Path("casanovo.yaml").exists()

    run(["configure", "-o", "test.yaml"])
    assert Path("test.yaml").exists()

    with pytest.raises(FileExistsError):
        run(["configure", "-o", "test.yaml"])

    with open("casanovo.yaml") as f_in, open("small.yaml", "w") as f_out:
        config = yaml.safe_load(f_in)
        config["max_epochs"] = 1
        config["n_layers"] = 1
        yaml.dump(config, f_out)

    train_args = [
        "train",
        "--validation_peak_path",
        str(mgf_small),
        "--config",
        "small.yaml",
        "--output_dir",
        str(tmp_path),
        "--output_root",
        "train",
        str(mgf_small),
    ]

    result = run(train_args)
    assert result.exit_code == 0

    res = run("version")
    assert res.output
