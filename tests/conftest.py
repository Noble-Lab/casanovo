"""Fixtures used for testing."""
import os
from contextlib import closing

import pytest
import numpy as np
from pyteomics.mass import calculate_mass
from torch.distributed.elastic.agent.server.api import (
    SimpleElasticAgent,
    _get_socket_with_port,
    _get_fq_hostname,
)


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


@pytest.fixture(autouse=True)
def windows_patch(monkeypatch):
    """Needed for multiprocessing in Windows on the GitHub actions runners.

    See here:
    https://github.com/pytorch/pytorch/issues/74824
    """
    if not os.name == "nt":
        return

    monkeypatch.setattr(
        SimpleElasticAgent,
        "_set_master_addr_port",
        staticmethod(_hook_set_master_addr_port),
    )


def _hook_set_master_addr_port(store, master_addr, master_port):
    """From: https://github.com/pytorch/pytorch/issues/74824"""
    if master_port is None:
        sock = _get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]

        if master_addr is None:
            hostname = _get_fq_hostname()
            # use IP address as master_addr
            master_addr = sock.gethostbyname(hostname)

    store.set("MASTER_ADDR", master_addr.encode(encoding="UTF-8"))
    store.set("MASTER_PORT", str(master_port).encode(encoding="UTF-8"))
