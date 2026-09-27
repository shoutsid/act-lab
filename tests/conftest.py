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

import contextlib
import os

import pytest


@pytest.fixture(autouse=True)
def _no_live_log_tailer(monkeypatch, tmp_path):
    """Keep the driver's live log tailer offline for every test.

    ``kaggle_queue.capture_live_log`` spawns ``kaggle kernels logs -f`` as a real
    process, which is the one part of the driver that would reach Kaggle from a
    test: the rest goes through ``_run``, which every driver test replaces. The
    tailer is neutralised here rather than per-test so that a test which forgets
    to think about it cannot quietly start talking to the platform; the tests
    that *do* exercise the capture patch this attribute themselves.
    """
    try:
        import kaggle_queue
    except ImportError:  # a test module that does not touch the driver at all
        return

    @contextlib.contextmanager
    def inert(*_args, **_kwargs):
        yield

    monkeypatch.setattr(kaggle_queue, "capture_live_log", inert, raising=False)
    # The same shape of guard, for the same reason: the session-evidence writer
    # copies a finished session's records into the repository's own ``artifacts/``
    # unless it is pointed elsewhere, and the driver tests drive ``drive()`` for
    # real - so a test that did not think about it left a fake session's records
    # in the project's own evidence trail (measured 2026-09-27: the first run of
    # that code left five files there).
    monkeypatch.setattr(
        kaggle_queue, "EVIDENCE_ROOT", tmp_path / "session-evidence", raising=False
    )

if os.environ.get("PYTEST_XDIST_WORKER"):
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):  # torch absent, or interop already started
        pass
