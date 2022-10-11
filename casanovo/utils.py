"""Small utility functions"""
import os

import torch
import psutil


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
