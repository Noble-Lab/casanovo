import pandas as pd

import casanovo
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

    with open(output_filename) as f_in:
        for skiprows, line in enumerate(f_in):
            if line.startswith("PSH"):
                break
                
    df = pd.read_csv(output_filename, skiprows=skiprows, sep="\t")

    assert df.PSM_ID[0] == "index=0"
    assert df.PSM_ID[1] == "index=1"
