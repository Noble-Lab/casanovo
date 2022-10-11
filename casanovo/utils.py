"""Small utility functions"""
import os
import re
from typing import Tuple

import torch
import psutil

from casanovo import __version__


def n_workers() -> int:
    """Get the number of workers to use for data loading.

    This is the maximum number of CPUs allowed for the process,
    scaled for the number of GPUs being used.

    On MacOS, we need to use all CPUs. See:
    https://stackoverflow.com/a/42658430

    Returns
    -------
    int
        The number of workers.
    """
    try:
        n_cpu = len(psutil.Process().cpu_affinity())
    except AttributeError:
        n_cpu = os.cpu_count()

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        return n_cpu // torch.cuda.device_count()

    return n_cpu


def split_version(version: str) -> Tuple[str, str, str, str]:
    """Split the version into its semantic versioning components.

    Parameters
    ----------
    version : str
        The version number.

    Returns
    -------
    major : str
        The major release.
    minor : str
        The minor release.
    patch : str
        The patch release.
    """
    version_regex = re.compile(r"(\d+)\.(\d+)\.*(\d*)(?:.dev\d+.+)?")
    return tuple(g for g in version_regex.match(version).groups())
