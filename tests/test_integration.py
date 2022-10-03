"""Test that denovo mode will run"""

from casanovo import casanovo


def test_denovo(mgf_small, tmp_path):
    """Predict on a small MGF file and verify that the output file exists."""
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
