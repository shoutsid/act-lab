"""Suite-wide fixtures and worker setup.

The suite runs one pytest-xdist worker per core (``addopts = -n auto`` in
``pyproject.toml``). Torch and the BLAS libraries each default to one thread per
core *per process*, so twenty workers would spawn four hundred threads and spend
the run contending for the cores rather than using them - the same lesson
``IMPLEMENTATION_STATUS.md`` records for the study runners (``xargs -P 18`` with
``OMP_NUM_THREADS=1`` per process). Under a worker each process is pinned to one
thread; a serial run (``-n 0``) is left at the defaults so a single heavy test
still gets the whole machine.
"""

from __future__ import annotations

import os

if os.environ.get("PYTEST_XDIST_WORKER"):
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):  # torch absent, or interop already started
        pass
