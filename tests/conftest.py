"""Fixtures used for testing."""

import depthcharge
import numpy as np
import psims
import pytest
import yaml
from pyteomics.mass import calculate_mass, fast_mass, std_aa_mass


@pytest.fixture
def mgf_small(tmp_path):
    """An MGF file with 2 annotated spectra."""
    peptides = ["LESLIEK", "PEPTIDEK"]
    mgf_file = tmp_path / "small.mgf"
    return _create_mgf(peptides, mgf_file)


@pytest.fixture
def tiny_fasta_file(tmp_path):
    fasta_file = tmp_path / "tiny_fasta.fasta"
    with fasta_file.open("w+") as fasta_ref:
        fasta_ref.write(
            (
                ">foo\nMEAPAQLLFLLLLWLPDTTREIVMTQSPPTLSLSPGERVTLSCRASQSVSSSYLTWYQ"
                "QKPGQAPRLLIYGASTRATSIPARFSGSGSGTDFTLTISSLQPEDFAVYYCQQDYNLP"
            )
        )
    return fasta_file


@pytest.fixture
def mgf_medium(tmp_path):
    """An MGF file with 7 spectra and scan numbers,
    C+57.021 mass modification considered"""
    peptides = [
        "ATSIPAR",
        "VTLSCR",
        "LLIYGASTR",
        "EIVMTQSPPTLSLSPGER",
        "MEAPAQLLFLLLLWLPDTTR",
        "ASQSVSSSYLTWYQQKPGQAPR",
        "FSGSGSGTDFTLTISSLQPEDFAVYYCQQDYNLP",
    ]
    mgf_file = tmp_path / "db_search.mgf"
    return _create_mgf(peptides, mgf_file, mod_aa_mass={"C": 160.030649})


@pytest.fixture
def mgf_small_unannotated(tmp_path):
    """An MGF file with 2 unannotated spectra."""
    peptides = ["LESLIEK", "PEPTIDEK"]
    mgf_file = tmp_path / "small_unannotated.mgf"
    return _create_mgf(peptides, mgf_file, annotate=False)


def _create_mgf(
    peptides, mgf_file, random_state=42, mod_aa_mass=None, annotate=True
):
    """
    Create a fake MGF file from one or more peptides.

    Parameters
    ----------
    peptides : str or list of str
        The peptides for which to create spectra.
    mgf_file : Path
        The MGF file to create.
    random_state : int or numpy.random.Generator, optional
        The random seed. The charge states are chosen to be 2 or 3 randomly.
    mod_aa_mass : dict, optional
        A dictionary that specifies the modified masses of amino acids.
        e.g. {"C": 160.030649} for carbamidomethylated C.
    annotate: bool, optional
        Whether to add peptide annotations to mgf file

    Returns
    -------
    mgf_file : Path
    """
    rng = np.random.default_rng(random_state)
    entries = [
        _create_mgf_entry(
            p, rng.choice([2, 3]), mod_aa_mass=mod_aa_mass, annotate=annotate
        )
        for p in peptides
    ]
    with mgf_file.open("w+") as mgf_ref:
        mgf_ref.write("\n".join(entries))

    return mgf_file


def _create_mgf_entry(peptide, charge=2, mod_aa_mass=None, annotate=True):
    """
    Create a MassIVE-KB style MGF entry for a single PSM.

    Parameters
    ----------
    peptide : str
        A peptide sequence.
    charge : int, optional
        The peptide charge state.
    mod_aa_mass : dict, optional
        A dictionary that specifies the modified masses of amino acids.
    annotate: bool, optional
        Whether to add peptide annotation to entry

    Returns
    -------
    str
        The PSM entry in an MGF file format.
    """
    if mod_aa_mass is None:
        precursor_mz = fast_mass(peptide, charge=int(charge))
    else:
        aa_mass = std_aa_mass.copy()
        aa_mass.update(mod_aa_mass)
        precursor_mz = fast_mass(peptide, charge=int(charge), aa_mass=aa_mass)
    mzs, intensities = _peptide_to_peaks(peptide, charge)
    frags = "\n".join([f"{m} {i}" for m, i in zip(mzs, intensities)])

    mgf = [
        "BEGIN IONS",
        f"TITLE={title}",
        f"SEQ={peptide}",
        f"PEPMASS={precursor_mz}",
        f"CHARGE={charge}+",
        f"SCANS=F1:{2470 + title}",
        f"{frags}",
        "END IONS",
    ]
    return "\n".join(mgf)


def _peptide_to_peaks(peptide, charge):
    """
    Generate a simulated spectrum for the given peptide.

    All canonical b and y fragments will occur with intensity 1.

    Parameters
    ----------
    peptide : str
        A peptide sequence.
    charge : int
        The peptide charge state.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Linked arrays with the fragment m/z and intensity values.
    """
    mzs = []
    for i in range(len(peptide)):
        for zstate in range(1, charge):
            b_pep, y_pep = peptide[: i + 1], peptide[i:]
            mzs.append(calculate_mass(b_pep, charge=zstate, ion_type="b"))
            mzs.append(calculate_mass(y_pep, charge=zstate, ion_type="y"))
    mzs = sorted(mzs)
    intensities = [1.0] * len(mzs)
    return np.asarray(mzs), np.asarray(intensities)


@pytest.fixture
def mzml_small(tmp_path):
    """An mzML file with 2 annotated spectra."""
    peptides = ["LESLIEK", "PEPTIDEK"]
    mzml_file = tmp_path / "small.mzml"
    return _create_mzml(peptides, mzml_file)


def _create_mzml(peptides, mzml_file, random_state=42):
    """
    Create a fake mzML file from one or more peptides.

    Parameters
    ----------
    peptides : str or list of str
        The peptides for which to create spectra.
    mzml_file : Path
        The mzML file to create.
    random_state : int or numpy.random.Generator, optional
        The random seed. The charge states are chosen to be 2 or 3 randomly.

    Returns
    -------
    mzml_file : Path
    """
    rng = np.random.default_rng(random_state)

    with psims.mzml.MzMLWriter(str(mzml_file)) as writer:
        writer.controlled_vocabularies()
        writer.file_description(["MSn spectrum"])
        writer.software_list(
            [
                {
                    "id": "psims-writer",
                    "version": psims.__version__,
                    "params": ["python-psims"],
                }
            ]
        )
        writer.instrument_configuration_list(
            [
                writer.InstrumentConfiguration(
                    "ic",
                    [
                        writer.Source(1, ["ionization type"]),
                        writer.Analyzer(2, ["mass analyzer type"]),
                        writer.Detector(3, ["detector type"]),
                    ],
                    ["instrument model"],
                )
            ]
        )
        writer.data_processing_list(
            [
                writer.DataProcessing(
                    [writer.ProcessingMethod(1, "psims-writer")], id="dp"
                )
            ]
        )
        with writer.run(id=1, instrument_configuration="ic"):
            with writer.spectrum_list(len(peptides)):
                for scan_nr, peptide in zip([17, 111], peptides):
                    charge = rng.choice([2, 3])

                    precursor = writer.precursor_builder()
                    precursor.selected_ion(
                        mz=calculate_mass(peptide, charge=charge),
                        charge=charge,
                    )
                    precursor.activation({"params": ["HCD"]})

                    mzs, intensities = _peptide_to_peaks(peptide, charge)
                    writer.write_spectrum(
                        mzs,
                        intensities,
                        id=f"scan={scan_nr}",
                        centroided=True,
                        params=[{"ms level": 2}],
                        precursor_information=precursor,
                    )

    return mzml_file


@pytest.fixture
def tiny_config(tmp_path):
    """A config file for a tiny model."""
    cfg = {
        "n_head": 2,
        "dim_feedforward": 10,
        "n_layers": 1,
        "train_label_smoothing": 0.01,
        "warmup_iters": 1,
        "cosine_schedule_period_iters": 1,
        "max_epochs": 20,
        "val_check_interval": 1,
        "accelerator": "cpu",
        "precursor_mass_tol": 5,
        "isotope_error_range": [0, 1],
        "min_peptide_len": 6,
        "max_peptide_len": 100,
        "enzyme": "trypsin",
        "digestion": "full",
        "missed_cleavages": 0,
        "max_mods": None,
        "predict_batch_size": 1024,
        "n_beams": 1,
        "top_match": 1,
        "devices": None,
        "random_seed": 454,
        "n_log": 1,
        "tb_summarywriter": False,
        "log_metrics": False,
        "log_every_n_steps": 50,
        "n_peaks": 150,
        "min_mz": 50.0,
        "max_mz": 2500.0,
        "min_intensity": 0.01,
        "remove_precursor_tol": 2.0,
        "max_charge": 10,
        "dim_model": 512,
        "dropout": 0.0,
        "dim_intensity": None,
        "learning_rate": 5e-4,
        "weight_decay": 1e-5,
        "train_batch_size": 32,
        "num_sanity_val_steps": 0,
        "calculate_precision": False,
        "lance_dir": None,
        "shuffle": False,
        "buffer_size": 64,
        "accumulate_grad_batches": 1,
        "gradient_clip_val": None,
        "gradient_clip_algorithm": None,
        "precision": "32-true",
        "replace_isoleucine_with_leucine": True,
        "reverse_peptides": False,
        "mskb_tokenizer": True,
        "residues": {
            "G": 57.021464,
            "A": 71.037114,
            "S": 87.032028,
            "P": 97.052764,
            "V": 99.068414,
            "T": 101.047670,
            "C+57.021": 160.030649,
            "L": 113.084064,
            "I": 113.084064,
            "N": 114.042927,
            "D": 115.026943,
            "Q": 128.058578,
            "K": 128.094963,
            "E": 129.042593,
            "M": 131.040485,
            "H": 137.058912,
            "F": 147.068414,
            "R": 156.101111,
            "Y": 163.063329,
            "W": 186.079313,
            "M+15.995": 147.035400,
            "N+0.984": 115.026943,
            "Q+0.984": 129.042594,
            "+42.011": 42.010565,
            "+43.006": 43.005814,
            "-17.027": -17.026549,
            "+43.006-17.027": 25.980265,
        },
        "allowed_fixed_mods": "C:C+57.021",
        "allowed_var_mods": (
            "M:M+15.995,N:N+0.984,Q:Q+0.984,"
            "nterm:+42.011,nterm:+43.006,nterm:-17.027,nterm:+43.006-17.027"
        ),
    }

    cfg_file = tmp_path / "config.yml"
    with cfg_file.open("w+") as out_file:
        yaml.dump(cfg, out_file)

    return cfg_file


@pytest.fixture
def residues_dict():
    return depthcharge.masses.PeptideMass("massivekb").masses
