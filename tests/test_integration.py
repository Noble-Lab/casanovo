import pandas as pd

from casanovo import casanovo


def test_denovo(mgf_small, tmp_path, monkeypatch):
    # We can use this to explicitly test different versions.
    monkeypatch.setattr(casanovo, "__version__", "3.0.1")

    # Predict on a small MGF file and verify that the output file exists.
    output_filename = tmp_path / "test.mztab"
    casanovo.main(
        [
            "--mode",
            "denovo",
            "--peak_path",
            str(mgf_small),
            "--output",
            str(output_filename),
        ],
        standalone_mode=False,
    )
    assert output_filename.is_file()

    # Verify that the spectrum predictions are correct.
    with open(output_filename) as f_in:
        for skiprows, line in enumerate(f_in):
            if line.startswith("PSH"):
                break

    psms = pd.read_csv(output_filename, skiprows=skiprows, sep="\t")

    # Because we're searching from an MGF file, the PSMs are identified using
    # their index.
    assert psms.PSM_ID[0] == "index=0"
    assert psms.sequence[0] == "LESLLEK"
    assert psms.PSM_ID[1] == "index=1"
    assert psms.sequence[1] == "PEPTLDEK"
