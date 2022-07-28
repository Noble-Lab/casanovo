"""Fixtures used for testing

We need:
- To set the PPX_DATA_DIR environment variable to a temporary directory
- A mock GET response from PRIDE
- A mock GET response from ProteomeXchange
- A mock FTP server response from PRIDE
- A mock FTP server response from MassIVE
"""
import json
import socket
import ftplib

import pytest
import requests
import ppx

from pyteomics.mass import calculate_mass
import numpy as np

# Pytest fixtures added here can be used as parameters for testing methods


@pytest.fixture
def mgf_small(tmp_path):
    """An MGF file with 2 annotated spectra."""
    peptides = ["LESLIEK", "PEPTIDEK"]
    mgf_file = tmp_path / "small.mgf"
    return _create_mgf(peptides, mgf_file)


# Utility Funcs
def _create_mgf(peptides, mgf_file, random_state=42):
    """Create a fake MGF file from one or more peptides.
    Parameters
    ----------
    peptides : str or list of str
        The peptides for which to create spectra.
    mgf_file : Path
        The MGF file to create.
    random_state : int or numpy.random.Generator, optional
        The random seed. The charge states are chosen to be
        2 or 3 randomly.
    Returns
    -------
    """
    rng = np.random.default_rng(random_state)
    entries = [_create_mgf_entry(p, rng.choice([2, 3])) for p in peptides]
    with mgf_file.open("w+") as mgf_ref:
        mgf_ref.write("\n".join(entries))

    return mgf_file


def _create_mgf_entry(peptide, charge=2):
    """Create a MassIVE-KB style MGF entry for a single PSM.
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


# Set the PPX_DATA_DIRECTORY --------------------------------------------------
@pytest.fixture(autouse=True)
def ppx_data_dir(monkeypatch, tmp_path):
    """Set the PPX_DATA_DIR environment variable"""
    monkeypatch.setenv("PPX_DATA_DIR", str(tmp_path))
    ppx.set_data_dir()


# PRIDE projects/<accession>/files endpoint -----------------------------------
class MockPrideFilesResponse:
    """A mock of the PRIDE files REST response"""

    status_code = 200

    @staticmethod
    def json():
        with open("tests/data/pride_files_response.json") as ref:
            out = json.load(ref)

        return out


@pytest.fixture
def mock_pride_files_response(monkeypatch):
    """Patch requests.get() to use a local file."""

    def mock_get(*args, **kwargs):
        return MockPrideFilesResponse()

    monkeypatch.setattr(requests, "get", mock_get)


# PRIDE projects/<accession? endpoint -----------------------------------------
class MockPrideProjectResponse:
    """A mock of the PRIDE projects REST response"""

    status_code = 200

    @staticmethod
    def json():
        with open("tests/data/pride_project_response.json") as ref:
            out = json.load(ref)

        return out


@pytest.fixture
def mock_pride_project_response(monkeypatch):
    """Patch requests.get() to use a local file."""

    def mock_get(*args, **kwargs):
        return MockPrideProjectResponse()

    monkeypatch.setattr(requests, "get", mock_get)


# MassIVE FTP server ----------------------------------------------------------
class MockMassiveFtpResponse:
    """A mock of the MassIVE FTP server response"""

    @staticmethod
    def dir(fun):
        with open("tests/data/massive_ftp_response.txt") as ref:
            [fun(line) for line in ref]


def mock_massive_ftp_response(monkeypatch):
    """Patch ftplib to use a local file as a response"""

    def null(*args, **kwargs):
        pass

    def mock_dir(fun):
        with open("tests/data/massive_ftp_response.txt") as ref:
            [fun(line) for line in ref]

    monkeypatch.setattr(ftplib.FTP, "login", null)
    monkeypatch.setattr(ftplib.FTP, "cwd", null)
    monkeypatch.setattr(ftplib.FTP, "dir", mock_dir)


# Mock up local files ---------------------------------------------------------
@pytest.fixture
def local_files(tmp_path):
    """Create some files to test local file detection"""
    local_dirs = [tmp_path / f"test_dir{i}" for i in range(10)]
    local_files = []
    for local_dir in local_dirs:
        local_dir.mkdir()
        files = [
            local_dir / "test_file.mzML",
            local_dir / "test_file.txt",
        ]
        local_files += files
        for local_file in files:
            local_file.touch()

    local_files.append(tmp_path / "test_file.mzML")
    local_files[-1].touch()
    return local_files, local_dirs


# Block internet --------------------------------------------------------------
@pytest.fixture
def block_internet(monkeypatch):
    """Turn off internet access"""

    class Blocker(socket.socket):
        def __init__(self, *args, **kwargs):
            raise OSError("Network call blocked")

    monkeypatch.setattr(socket, "socket", Blocker)
