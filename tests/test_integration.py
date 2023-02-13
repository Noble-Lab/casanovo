import pyteomics.mztab

from casanovo import casanovo


def test_denovo(mgf_small, mzml_small, tmp_path, monkeypatch):
    # We can use this to explicitly test different versions.
    monkeypatch.setattr(casanovo, "__version__", "3.0.1")

    # Predict on small files (MGF and mzML) and verify that the output mzTab
    # file exists.
    output_filename = tmp_path / "test.mztab"
    casanovo.main(
        [
            "--mode",
            "denovo",
            "--peak_path",
            str(mgf_small).replace(".mgf", ".m*"),
            "--output",
            str(output_filename),
        ],
        standalone_mode=False,
    )
    assert output_filename.is_file()

    mztab = pyteomics.mztab.MzTab(str(output_filename))
    # Verify that both input peak files are listed in the metadata.
    for i, filename in enumerate(["small.mgf", "small.mzml"], 1):
        assert f"ms_run[{i}]-location" in mztab.metadata
        assert mztab.metadata[f"ms_run[{i}]-location"].endswith(filename)

    # Verify that the spectrum predictions are correct and indexed according to
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
