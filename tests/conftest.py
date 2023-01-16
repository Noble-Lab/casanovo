"""Fixtures used for testing."""
import numpy as np
import psims
import pytest
from pyteomics.mass import calculate_mass


@pytest.fixture
def mgf_small(tmp_path):
    """An MGF file with 2 annotated spectra."""
    peptides = ["LESLIEK", "PEPTIDEK"]
    mgf_file = tmp_path / "small.mgf"
    return _create_mgf(peptides, mgf_file)


def _create_mgf(peptides, mgf_file, random_state=42):
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

    Returns
    -------
    mgf_file : Path
    """
    rng = np.random.default_rng(random_state)
    entries = [_create_mgf_entry(p, rng.choice([2, 3])) for p in peptides]
    with mgf_file.open("w+") as mgf_ref:
        mgf_ref.write("\n".join(entries))

    return mgf_file


def _create_mgf_entry(peptide, charge=2):
    """
    Create a MassIVE-KB style MGF entry for a single PSM.

    Parameters
    ----------
    peptide : str
        A peptide sequence.
    charge : int, optional
        The peptide charge state.

    Returns
    -------
    str
        The PSM entry in an MGF file format.
    """
    precursor_mz = calculate_mass(peptide, charge=int(charge))
    mzs, intensities = _peptide_to_peaks(peptide, charge)
    frags = "\n".join([f"{m} {i}" for m, i in zip(mzs, intensities)])

    mgf = [
        "BEGIN IONS",
        f"SEQ={peptide}",
        f"PEPMASS={precursor_mz}",
        f"CHARGE={charge}+",
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
