"""Kaggle T4 driver for the section 5.1 run (``T4_RUNBOOK.md`` sections 0-2c).

The runbook describes the commands an operator runs *inside* a Kaggle GPU
session. This module is the thing that is actually pushed: a single script
kernel whose behaviour is selected by ``ACT_T4_MODE``, so the four session
kinds cannot drift apart from each other or from the registered commands.

Modes (``ACT_T4_MODE=<mode>``):

- ``metadata``   write ``kernel-metadata.json`` for the selected queue;
- ``probe``      is the Kaggle dataset API authenticated inside a kernel?
                 (measured: yes, from the kernel's own identity - see below);
- ``preflight``  runbook 0.4 + 1: environment check, declared-parameter
                 assertion, and the session-1 throughput benchmark;
- ``session``    runbook 2: the primary pair queue;
- ``control``    runbook 2b: the pool-matched (loss-blind) control queue;
- ``null``       runbook 2c: the do-nothing null queue.

Push each mode as a script kernel with the metadata this module writes, then
re-push the *same* kernel until its queue reports complete: restore latest,
skip seeds with ``complete`` versions, train the next incomplete pair, exit
publish. Nothing here re-tunes a protocol knob; the modes only arrange for
the registered command to run on a T4 with its input artifact present.

Two gotchas this module exists to absorb, both learned the hard way:

1. **Mount layout.** Kaggle attaches a dataset at ``/kaggle/input/<slug>``,
   at ``/kaggle/input/<owner>/<slug>`` when the slug needs disambiguation,
   or (current runtime) at ``/kaggle/input/datasets/<owner>/<slug>``. The
   first run assumed the flat layout, found nothing, and exited before
   training. :func:`find_code_root` searches the tree instead.
2. **Accelerator pinning.** ``kaggle kernels push`` honours
   ``machine_shape`` (``NvidiaTeslaT4``), not ``gpu_type``; an unknown key
   is silently ignored and the run lands on whatever Kaggle picks.

The exit publish needs no User Secrets: ``KaggleApi().authenticate()``
succeeds inside a Kaggle kernel with no environment variables and no
``kaggle.json`` (probed 2026-09-15), but internet access must be enabled for
the API calls to leave the session.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

MODE_ENV = "ACT_T4_MODE"
HOURS_ENV = "ACT_T4_HOURS"
OWNER_ENV = "ACT_T4_OWNER"
SLUG_ENV = "ACT_T4_SLUG"

DEFAULT_OWNER = "sid123123234"
DEFAULT_SLUG = "act-v0-5-1"
# Section 5.2 session model: a 12 h Kaggle clock less boot, restore and the
# 45 min publish reserve leaves ~9 h of usable work.
DEFAULT_HOURS = 9.0
BENCHMARK_STEPS = 50
# The adopted section 5.5 revision: queue moved off the spent 9000-9011
# block, primary candidate policy is ``spread``.
SEED_BLOCK = "15000-15011"
PRIMARY_POLICY = "spread"

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/act-lab")


@dataclass(frozen=True)
class Queue:
    """One declared queue: its Kaggle dataset, its roots, its arm flags."""

    mode: str
    title: str
    extra: tuple[str, ...] = ()

    @property
    def dataset(self) -> str:
        slug = os.environ.get(SLUG_ENV, DEFAULT_SLUG)
        suffix = {"session": "", "control": "-control", "null": "-null"}[self.mode]
        return f"{self.owner}/{slug}{suffix}"

    @property
    def kernel_id(self) -> str:
        return f"{self.owner}/act-t4-{self.mode}"

    @property
    def owner(self) -> str:
        return os.environ.get(OWNER_ENV, DEFAULT_OWNER)

    @property
    def dataset_root(self) -> str:
        suffix = {"session": "", "control": "-control", "null": "-null"}[self.mode]
        return f"artifacts/datasets{suffix}"

    @property
    def session_dir(self) -> str:
        suffix = {"session": "", "control": "-control", "null": "-null"}[self.mode]
        return f"artifacts/v0-sessions{suffix}"


QUEUES: dict[str, Queue] = {
    "session": Queue("session", "act-t4-session"),
    "control": Queue("control", "act-t4-control"),
    "null": Queue("null", "act-t4-null"),
}


def find_code_root(input_root: str | Path = INPUT_ROOT) -> Path | None:
    """Locate the attached copy of this repo anywhere in the mount tree.

    The repo copy is identified by ``pyproject.toml`` with a sibling
    ``src/act``, so an unrelated attached dataset cannot be mistaken for it.
    Returns ``None`` when nothing matches (the caller fails closed).
    """
    root = Path(input_root)
    if not root.is_dir():
        return None
    for marker in sorted(root.rglob("pyproject.toml")):
        if (marker.parent / "src" / "act").is_dir():
            return marker.parent
    return None


def benchmark_argv(python: str, steps: int = BENCHMARK_STEPS) -> list[str]:
    """Runbook section 1: the throughput benchmark that fixes dtype."""
    return [python, "scripts/v0_throughput_benchmark.py", "--steps", str(steps)]


def session_argv(queue: Queue, hours: float = DEFAULT_HOURS) -> list[str]:
    """The registered launch command for one queue (runbook 2 / 2b / 2c).

    The primary queue restates the code defaults (``--seeds`` and
    ``--probe-candidate-policy``) rather than changing them; the control adds
    only the loss-blind ``--probe-selection random``; the null queue is
    policy-free by construction, so it carries no probe flag at all.
    """
    argv = [
        sys.executable,
        "-m",
        "act.training.cli",
        "--mode",
        "pair-session",
        "--device",
        "cuda",
        "--hours",
        f"{hours:g}",
        "--seeds",
        SEED_BLOCK,
    ]
    if queue.mode in ("session", "control"):
        argv += ["--probe-candidate-policy", PRIMARY_POLICY]
    if queue.mode == "control":
        argv += ["--probe-selection", "random"]
    if queue.mode == "null":
        argv += ["--session-arms", "null"]
    argv += [
        "--dataset-root",
        queue.dataset_root,
        "--session-dir",
        queue.session_dir,
        "--kaggle-dataset",
        queue.dataset,
    ]
    return argv


def kernel_metadata(queue: Queue, code_file: str = "kaggle_t4.py") -> dict[str, object]:
    """Metadata for pushing one queue's kernel.

    ``machine_shape`` is the only honoured accelerator key; ``gpu_type`` is
    silently ignored by the CLI. Internet stays on for the dataset API.
    """
    return {
        "id": queue.kernel_id,
        "title": queue.title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": [f"{queue.owner}/act-lab-code"],
        "competition_sources": [],
        "kernel_sources": [],
    }


def _log(message: str) -> None:
    print(message, flush=True)


def _run(argv: list[str], env: dict[str, str]) -> None:
    _log(f"$ {' '.join(argv)}")
    proc = subprocess.run(argv, env=env, capture_output=True, text=True)
    _log(proc.stdout[-8000:])
    if proc.returncode != 0:
        _log("--- stderr ---")
        _log(proc.stderr[-6000:])
        raise SystemExit(f"command failed with exit {proc.returncode}: {argv[1:3]}")


def _bootstrap() -> tuple[Path, dict[str, str]]:
    """Copy the mounted repo into the writable working dir and ready the env."""
    source = find_code_root()
    if source is None:
        raise SystemExit(
            "no code dataset under /kaggle/input (expected a directory with "
            "pyproject.toml + src/act anywhere in the mount tree)"
        )
    _log(f"code dataset: {source}")
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    shutil.copytree(source, WORK_ROOT)
    os.chdir(WORK_ROOT)
    return WORK_ROOT, dict(os.environ, PYTHONPATH=str(WORK_ROOT / "src"))


def mode_probe() -> None:
    """Diagnostic: is the Kaggle dataset API usable inside a kernel?"""
    _log("=== Kaggle-in-kernel credential probe ===")
    for key in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_CONFIG_DIR", "HOME"):
        value = os.environ.get(key)
        if value and key == "KAGGLE_KEY":
            value = f"<set, {len(value)} chars>"
        _log(f"env {key}: {value if value else '<unset>'}")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        _log("KaggleApi.authenticate(): OK")
        datasets = api.dataset_list(mine=True)
        _log(f"dataset_list(mine=True): {len(datasets)} dataset(s)")
        for dataset in datasets[:10]:
            _log(f"  {dataset.ref}")
        _log("VERDICT: in-kernel dataset API usable without attached secrets")
    except Exception as error:  # noqa: BLE001 - a probe reports, never raises
        _log(f"KaggleApi.authenticate(): FAILED - {type(error).__name__}: {error}")
        _log("VERDICT: attach User Secrets before launching a session")


def mode_preflight() -> None:
    """Runbook 0.4 + 1: environment check and the session-1 benchmark."""
    import torch

    _log("=== ACT lab: T4 pre-flight ===")
    work, env = _bootstrap()

    nvidia = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    _log(nvidia.stdout[-800:] if nvidia.returncode == 0 else "nvidia-smi unavailable")
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device on this kernel - enable a GPU accelerator")
    _log(f"torch {torch.__version__} | device {torch.cuda.get_device_name(0)}")

    from act.model.v0_moe import estimated_parameter_count
    from act.training.session_plan import (
        expected_probe_cost_percent,
        tolerance_band,
    )
    from act.training.v0_session import V0SessionConfig

    # Runbook pre-flight 4: the declared config must reproduce the counts the
    # tolerance band derives from.
    total, active = estimated_parameter_count(V0SessionConfig().model_config())
    expected_total, expected_active = 285_336_576, 134_341_632
    _log(f"declared parameters: {total:,} total / {active:,} active")
    assert total == expected_total, f"total {total} != {expected_total}"
    assert active == expected_active, f"active {active} != {expected_active}"
    _log("parameter-count assertion: OK")

    _log(f"--- runbook 1: throughput benchmark (--steps {BENCHMARK_STEPS}) ---")
    _run(benchmark_argv(sys.executable), env)
    artifact_path = work / "artifacts" / "v0-benchmark.json"
    data = json.loads(artifact_path.read_text())
    shutil.copy(artifact_path, "/kaggle/working/v0-benchmark.json")
    _log(f"measured_tokens_per_second: {data['measured_tokens_per_second']}")
    _log(f"dtype_decision: {data['dtype_decision']}")
    for key in (
        "pair_time_hours",
        "full_pairs_per_session",
        "sessions_for_primary",
        "sessions_total",
        "working_dir_headroom_gb",
    ):
        _log(f"{key}: {data['schedule'][key]}")
    _log(f"predicted probe cost (cap4/1-batch): {expected_probe_cost_percent(1)}%"
         f" band {tolerance_band(1)}")
    _log("=== PRE-FLIGHT COMPLETE: /kaggle/working/v0-benchmark.json ===")


def mode_session(queue: Queue) -> None:
    """Runbook 2 / 2b / 2c: one session's worth of one declared queue."""
    import torch

    hours = float(os.environ.get(HOURS_ENV, str(DEFAULT_HOURS)))
    started = time.time()
    _log(f"=== ACT lab: {queue.mode} session ===")
    _log(f"dataset={queue.dataset} hours={hours:g} seeds={SEED_BLOCK}")
    work, env = _bootstrap()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device on this kernel - enable a GPU accelerator")
    _log(f"torch {torch.__version__} | device {torch.cuda.get_device_name(0)}")

    # The launch command reads artifacts/v0-benchmark.json and refuses a stale
    # or hand-edited one; a fresh session has no artifacts/ directory, so the
    # benchmark runs first in every session.
    _log("--- throughput benchmark (the launch command's input) ---")
    _run(benchmark_argv(sys.executable), env)
    artifact_path = work / "artifacts" / "v0-benchmark.json"
    data = json.loads(artifact_path.read_text())
    _log(
        f"measured {data['measured_tokens_per_second']:.1f} tokens/s | "
        f"dtype {data['dtype_decision']} | "
        f"pair_time {data['schedule']['pair_time_hours']:.2f} h"
    )
    shutil.copy(artifact_path, "/kaggle/working/v0-benchmark.json")

    _log(f"--- {queue.mode} queue (--hours {hours:g}) ---")
    _run(session_argv(queue, hours), env)

    # Runbook 6: keep the audit trail in the kernel output, so the per-pair
    # actions and the publish outcome survive without the dataset API.
    for relative in (
        f"{queue.session_dir}/session-log.json",
        f"{queue.dataset_root}/remote-publish-log.json",
    ):
        path = work / relative
        if path.is_file():
            target = Path("/kaggle/working") / Path(relative).name
            shutil.copy(path, target)
            _log(f"kept {target}")

    _log(f"=== {queue.mode} session finished after {(time.time() - started) / 3600:.2f} h ===")


def main() -> int:
    mode = os.environ.get(MODE_ENV, "")
    if mode == "metadata":
        queue_mode = os.environ.get("ACT_T4_QUEUE", "session")
        path = Path("kernel-metadata.json")
        path.write_text(json.dumps(kernel_metadata(QUEUES[queue_mode]), indent=2) + "\n")
        _log(f"wrote {path} for queue {queue_mode!r}")
        return 0
    if mode == "probe":
        mode_probe()
        return 0
    if mode == "preflight":
        mode_preflight()
        return 0
    if mode in QUEUES:
        mode_session(QUEUES[mode])
        return 0
    raise SystemExit(
        f"set {MODE_ENV} to one of: metadata, probe, preflight, "
        + ", ".join(QUEUES)
    )


if __name__ == "__main__":
    raise SystemExit(main())
