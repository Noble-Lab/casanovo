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
