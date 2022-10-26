"""Fixtures used for testing."""
import pytest
import numpy as np
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
    mz = calculate_mass(peptide, charge=int(charge))
    frags = []
    for idx in range(len(peptide)):
        for zstate in range(1, charge):
            b_pep = peptide[: idx + 1]
            frags.append(
                str(calculate_mass(b_pep, charge=zstate, ion_type="b"))
            )
            y_pep = peptide[idx:]
            frags.append(
                str(calculate_mass(y_pep, charge=zstate, ion_type="y"))
            )
    frag_string = " 1\n".join(frags) + " 1"

    mgf = [
        "BEGIN IONS",
        f"SEQ={peptide}",
        f"PEPMASS={mz}",
        f"CHARGE={charge}+",
        f"{frag_string}",
        "END IONS",
    ]
    return "\n".join(mgf)
