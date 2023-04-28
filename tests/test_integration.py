import functools
import pyteomics.mztab

from click.testing import CliRunner
from casanovo import casanovo


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
        "--mode",
        "train",
        "--peak_path",
        mgf_small,
        "--peak_path_val",
        mgf_small,
        "--config",
        tiny_config,
        "--output",
        str(tmp_path / "train"),
    ]

    result = run(train_args)
    model_file = tmp_path / "epoch=9-step=10.ckpt"
    assert result.exit_code == 0
    assert model_file.exists()

    # Try evaluating:
    eval_args = [
        "--mode",
        "eval",
        "--peak_path",
        mgf_small,
        "--model",
        model_file,
        "--config",
        tiny_config,
        "--output",
        str(tmp_path / "eval"),
    ]

    result = run(eval_args)
    assert result.exit_code == 0

    # Finally try predicting:
    output_filename = tmp_path / "test.mztab"
    predict_args = [
        "--mode",
        "denovo",
        "--peak_path",
        mgf_small,
        "--peak_path",
        mzml_small,
        "--model",
        model_file,
        "--config",
        tiny_config,
        "--output",
        str(output_filename),
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
    # and indexed according to
    # the peak input file type.
    psms = mztab.spectrum_match_table
    assert psms.loc[1, "sequence"] == "LESLLEK"
    assert psms.loc[1, "spectra_ref"] == "ms_run[1]:index=0"
    assert psms.loc[2, "sequence"] == "PEPTLDEK"
    assert psms.loc[2, "spectra_ref"] == "ms_run[1]:index=1"
    assert psms.loc[3, "sequence"] == "LESLLEK"
    assert psms.loc[3, "spectra_ref"] == "ms_run[2]:scan=17"
    assert psms.loc[4, "sequence"] == "PEPTLDEK"
    assert psms.loc[4, "spectra_ref"] == "ms_run[2]:scan=111"
