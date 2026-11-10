import os

import pandas as pd
import pyteomics.mztab

from casanovo.config import Config
from casanovo.data import ms_io, psm


def test_mztab_save(tiny_config, tmp_path):
    file = tmp_path / "test.mztab"
    writer = ms_io.MztabWriter(file)
    tiny_config = Config(tiny_config)
    writer.set_metadata(tiny_config)
    writer.set_ms_run(["test.mgf"])

    psm_test = psm.PepSpecMatch(
        sequence="AAAA",
        spectrum_id=("test.mgf", "0"),
        peptide_score=1.0,
        charge=3,
        calc_mz=100.0,
        exp_mz=100.0,
        aa_scores=[0.5, 0.5, 0.5, 0.5],
    )

    writer.psms = [psm_test]
    writer.save()

    assert file.is_file()


def test_mztab_save_eval_columns(tiny_config, tmp_path):
    """Evaluation results are exported as additional mzTab columns."""
    file = tmp_path / "test.mztab"
    writer = ms_io.MztabWriter(file)
    writer.set_metadata(Config(tiny_config))
    writer.set_ms_run(["test.mgf"])
    writer.psms = [
        psm.PepSpecMatch(
            sequence="AAAA",
            spectrum_id=("test.mgf", "0"),
            peptide_score=1.0,
            charge=3,
            calc_mz=100.0,
            exp_mz=100.0,
            aa_scores=[0.5, 0.5, 0.5, 0.5],
            ground_truth_sequence="AAAK",
            precision=1.0,
            coverage=0.5,
        )
    ]
    writer.save()

    lines = file.read_text(encoding="utf-8").splitlines()
    header = next(ln for ln in lines if ln.startswith("PSH")).split("\t")
    fields = next(ln for ln in lines if ln.startswith("PSM")).split("\t")
    assert fields[header.index("opt_global_ground_truth_sequence")] == "AAAK"
    assert fields[header.index("opt_global_precision")] == "1.0"
    assert fields[header.index("opt_global_coverage")] == "0.5"


def test_run_map(mgf_small):
    out_writer = ms_io.MztabWriter("dummy.mztab")
    # Set peak file by base file name only.
    out_writer.set_ms_run([os.path.basename(mgf_small.name)])
    assert mgf_small.name in out_writer._run_map
    assert os.path.abspath(mgf_small.name) not in out_writer._run_map


def test_set_database(tmp_path):
    """Test that set_database populates PSM database columns."""
    fasta = tmp_path / "test.fasta"
    fasta.touch()
    out_writer = ms_io.MztabWriter(str(tmp_path / "test.mztab"))
    assert out_writer.database == "null"
    assert out_writer.database_version == "null"
    out_writer.set_database(str(fasta))
    assert out_writer.database == "test"
    assert out_writer.database_version == "null"


def test_mgf_scan_index_built_correctly(mgf_small_with_scans):
    """
    _build_mgf_scan_index must yield `(spectrum_ref_id, scan_number)`
    pairs for SCANS values written into the MGF file.
    """
    result = dict(ms_io._build_mgf_scan_index(str(mgf_small_with_scans)))
    assert result == {"index=0": "17", "index=1": "42"}


def test_mgf_without_scans_gives_empty_index(mgf_small):
    """
    An MGF file without any SCANS fields must produce an empty index
    (not crash and not map anything).
    """
    result = dict(ms_io._build_mgf_scan_index(str(mgf_small)))
    assert result == {}


def test_mgf_scan_alias_scan_field(tmp_path):
    """
    _build_mgf_scan_index must recognise the SCAN= header alias
    in addition to SCANS=.
    """
    mgf_file = tmp_path / "alias_scan.mgf"
    mgf_file.write_text(
        "BEGIN IONS\nSCAN=99\nPEPMASS=400.2\nCHARGE=2+\n200.0 1.0\nEND IONS\n",
        encoding="utf-8",
    )
    result = dict(ms_io._build_mgf_scan_index(str(mgf_file)))
    assert result == {"index=0": "99"}


def test_mgf_scan_alias_scan_id_field(tmp_path):
    """
    _build_mgf_scan_index must recognise the SCAN ID= header alias
    in addition to SCANS=.
    """
    mgf_file = tmp_path / "alias_scan_id.mgf"
    mgf_file.write_text(
        "BEGIN IONS\nSCAN ID=42\nPEPMASS=400.2\nCHARGE=2+\n200.0 1.0\nEND IONS\n",
        encoding="utf-8",
    )
    result = dict(ms_io._build_mgf_scan_index(str(mgf_file)))
    assert result == {"index=0": "42"}


def test_scan_num_in_mztab_spectra_ref(tmp_path):
    """
    After set_ms_run is called, MztabWriter must embed the scan
    number in a new opt_global_cv_MS:1003057_scan_number column.
    """
    # Create a minimal MGF with SCANS=17 for the first spectrum.
    mgf_file = tmp_path / "test.mgf"
    mgf_file.write_text(
        "BEGIN IONS\nSCANS=17\nPEPMASS=400.2\nCHARGE=2+\n200.0 1.0\nEND IONS\n",
        encoding="utf-8",
    )

    results_path = str(tmp_path / "results.mztab")
    writer = ms_io.MztabWriter(results_path)
    writer.set_ms_run([str(mgf_file)])
    writer.psms = [
        psm.PepSpecMatch(
            sequence="PEPTIDE",
            spectrum_id=("test.mgf", "index=0"),
            peptide_score=0.9,
            charge=2,
            calc_mz=400.2,
            exp_mz=400.2,
            aa_scores=[0.9] * 7,
        )
    ]
    writer.save()

    mztab = pyteomics.mztab.MzTab(results_path)
    psms = mztab.spectrum_match_table
    assert psms.loc[1, "spectra_ref"] == "ms_run[1]:index=0"
    assert (
        psms.loc[1, "opt_global_cv_MS:1003057_scan_number"]
        == "ms_run[1]:scan=17"
    )


def test_no_scan_num_keeps_index_only_spectra_ref(tmp_path):
    """
    An MGF without SCANS must produce 'ms_run[1]:index=0' only,
    preserving backward compatibility.
    """
    mgf_file = tmp_path / "test_no_scan.mgf"
    mgf_file.write_text(
        "BEGIN IONS\nPEPMASS=400.2\nCHARGE=2+\n200.0 1.0\nEND IONS\n"
    )

    results_path = str(tmp_path / "results_no_scan.mztab")
    writer = ms_io.MztabWriter(results_path)
    writer.set_ms_run([str(mgf_file)])
    writer.psms = [
        psm.PepSpecMatch(
            sequence="PEPTIDE",
            spectrum_id=("test_no_scan.mgf", "index=0"),
            peptide_score=0.9,
            charge=2,
            calc_mz=400.2,
            exp_mz=400.2,
            aa_scores=[0.9] * 7,
        )
    ]
    writer.save()

    mztab = pyteomics.mztab.MzTab(results_path)
    psms = mztab.spectrum_match_table
    assert psms.loc[1, "spectra_ref"] == "ms_run[1]:index=0"
    assert "opt_global_cv_MS:1003057_scan_number" not in psms.columns


def test_mzml_spectra_ref_unaffected_by_scan_num_feature(mzml_small, tmp_path):
    """
    mzML spectra use their native scan ID and must be unchanged even
    when set_ms_run is called (mzML files are silently ignored).
    """
    results_path = str(tmp_path / "results_mzml.mztab")
    writer = ms_io.MztabWriter(results_path)
    writer.set_ms_run([str(mzml_small)])
    writer.psms = [
        psm.PepSpecMatch(
            sequence="PEPTIDE",
            spectrum_id=("small.mzml", "scan=17"),
            peptide_score=0.9,
            charge=2,
            calc_mz=400.2,
            exp_mz=400.2,
            aa_scores=[0.9] * 7,
        )
    ]
    writer.save()

    mztab = pyteomics.mztab.MzTab(results_path)
    psms = mztab.spectrum_match_table
    assert psms.loc[1, "spectra_ref"] == "ms_run[1]:scan=17"
    assert "opt_global_cv_MS:1003057_scan_number" not in psms.columns


def test_mixed_mgf_mzml_scan_number_column(mzml_small, tmp_path):
    """
    When MGF and mzML files are processed together and the MGF contains
    scan numbers, the opt_global_cv_MS:1003057_scan_number column must be
    present for the whole mzTab. The MGF row must carry the scan value and
    the mzML row must be null.
    """
    mgf_file = tmp_path / "mixed.mgf"
    mgf_file.write_text(
        "BEGIN IONS\nSCANS=99\nPEPMASS=400.2\nCHARGE=2+\n200.0 1.0\nEND IONS\n"
    )

    results_path = str(tmp_path / "results_mixed.mztab")
    writer = ms_io.MztabWriter(results_path)
    writer.set_ms_run([str(mgf_file), str(mzml_small)])
    writer.psms = [
        psm.PepSpecMatch(
            sequence="PEPTIDE",
            spectrum_id=("mixed.mgf", "index=0"),
            peptide_score=0.9,
            charge=2,
            calc_mz=400.2,
            exp_mz=400.2,
            aa_scores=[0.9] * 7,
        ),
        psm.PepSpecMatch(
            sequence="PEPTIDE",
            spectrum_id=("small.mzml", "scan=17"),
            peptide_score=0.9,
            charge=2,
            calc_mz=400.2,
            exp_mz=400.2,
            aa_scores=[0.9] * 7,
        ),
    ]
    writer.save()

    mztab = pyteomics.mztab.MzTab(results_path)
    psms = mztab.spectrum_match_table
    # Column must be present because the MGF file has scan numbers.
    assert "opt_global_cv_MS:1003057_scan_number" in psms.columns
    # MGF row must carry the scan value.
    mgf_row = psms[psms["spectra_ref"].str.contains("index=0")].iloc[0]
    assert (
        mgf_row["opt_global_cv_MS:1003057_scan_number"] == "ms_run[1]:scan=99"
    )
    # mzML row must be null.
    mzml_row = psms[psms["spectra_ref"].str.contains("scan=17")].iloc[0]
    assert pd.isna(mzml_row["opt_global_cv_MS:1003057_scan_number"])
