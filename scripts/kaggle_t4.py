"""Kaggle T4 driver for the section 5.1 run (``T4_RUNBOOK.md`` sections 0-4).

The runbook describes the commands an operator runs *inside* a Kaggle
session. This module is the thing that is actually pushed: a single script
kernel whose behaviour is selected by ``ACT_T4_MODE``, so the session kinds
cannot drift apart from each other or from the registered commands.

Modes (``ACT_T4_MODE=<mode>``):

- ``metadata``  write ``kernel-metadata.json`` for the selected queue
                (``ACT_T4_QUEUE`` = session / control / null / verdict);
- ``probe``     is the Kaggle dataset API authenticated inside a kernel?
                (measured: yes, from the kernel's own identity);
- ``preflight`` runbook 0.4 + 1: environment check, declared-parameter
                assertion, and the session-1 throughput benchmark;
- ``selfcheck`` the V0 session battery (``scripts/v0_session_selfcheck.py``),
                including the per-arm placement check that needs the real
                two-T4 shape to mean anything;
- ``session``   runbook 2: the primary pair queue;
- ``control``   runbook 2b: the pool-matched (loss-blind) control queue;
- ``null``      runbook 2c: the do-nothing null queue;
- ``verdict``   runbook 4: restore all three queues from Kaggle and compute
                the pre-registered verdict;
- ``archzoo``   the architecture-search horizon arm
                (``artifacts/arch-zoo-100m-v3-crossed-horizon10x/NOTES.md``):
                one ``scripts/arch_zoo_screen.py`` sweep per T4, one seed each,
                the ledgers copied into the kernel output on exit.

Push each mode as a script kernel with the metadata this module writes, then
re-push the *same* kernel until its queue reports complete: restore latest,
skip seeds with ``complete`` versions, train the next incomplete pair, exit
publish. Nothing here re-tunes a protocol knob; the modes only arrange for
the registered command to run on a T4 with its input artifact present.

The verdict mode exists because the runbook's section 4 command reads *local*
dataset roots, and after a Kaggle run the queues live in published datasets:
each session pushes the whole handoff directory, so the newest published
version of each queue is cumulative and restoring it reconstructs the queue.
Its 0/1/2 exit codes are outcomes rather than failures - 1 (cost FAIL) and 2
(a withholding clause fired) are both legitimate final answers.

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

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only: the kernel imports these lazily
    import torch

    from act.training.session_plan import PairSchedule

MODE_ENV = "ACT_T4_MODE"
QUEUE_ENV = "ACT_T4_QUEUE"
HOURS_ENV = "ACT_T4_HOURS"
OWNER_ENV = "ACT_T4_OWNER"
SLUG_ENV = "ACT_T4_SLUG"

DEFAULT_OWNER = "sid123123234"
DEFAULT_SLUG = "act-v0-5-1"
# The declared session clock. Section 5.2's model assumes a 12 h Kaggle clock
# (~9 h of usable work once boot, restore and the publish reserve are paid for),
# but the T4 x2 shape measured 8 h: session 1, pushed 2026-09-20 10:39:54Z, was
# killed with ERROR exactly 8 h later, before its own 9 h deadline, with no
# kernel log and nothing published. `session_plan.DECLARED_SESSION_HOURS` is
# that cap less the reserve and the prep allowance (7.0 h); at the measured
# 1.443 h pair time it packs 4 pairs instead of 5 and still needs three sessions
# for the 12-seed block, so the correction costs the campaign no session. Tested
# equal to the constant it is derived from
# (`tests/test_kaggle_t4.py::test_the_declared_window_is_the_measured_cap_less_the_reserve`).
DEFAULT_HOURS = 7.0
BENCHMARK_STEPS = 50
# The adopted section 5.5 revision: queue moved off the spent 9000-9011
# block, primary candidate policy is ``spread``.
SEED_BLOCK = "15000-15011"
PRIMARY_POLICY = "spread"
# v0_verdict.py exit codes are the pre-registered verdict, not failures.
VERDICT_EXIT_CODES = {0: "PASS", 1: "FAIL", 2: "PASS with qualification"}

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/act-lab")

# --- The architecture-search horizon arm (ARCH_SEARCH_FINDINGS.md §3, §11) ---
# The crossed depth x hidden_size factorial of ``arch-zoo-100m-v3-crossed``, run
# 10x longer. Every design flag below is the one that arm's ledgers record, so
# the candidate set is byte-identical (checked by ``--dry-run`` before the push
# and asserted by ``verify_record.py`` once the ledgers land); only the training
# length, the precision and the host differ - which is the arm's whole question.
ARCHZOO_EXPERIMENT = "arch-zoo-100m-v3-crossed-horizon10x"
ARCHZOO_KERNEL = "act-t4-archzoo"
TOKENS_DATASET_SLUG = "act-lab-tinystories-tokens"
# One seed per T4: the sweep is single-process, so the second card is only used
# by running a second seed beside the first. Seeds 42/43 are the crossed arm's
# own first two, so every 12,000-step row has a same-seed 1200-step fp32 row in
# the committed record to pair with.
ARCHZOO_SEEDS = (42, 43)
# 1200 is the same-precision anchor: fp16 losses are not comparable to the
# record's fp32 ones, so the decay 1200 -> 12000 is measured inside this arm and
# the anchor-vs-record gap measures the precision offset directly.
ARCHZOO_FLAT_STEPS = "1200,12000"
ARCHZOO_PRECISION = "fp16"
# Conservative for fp16 on a T4 at ~78M parameters: the dtype probe measured
# 5,433 tok/s at the 285M declared shape, 2.56x its fp32 rate. The token budget
# guard is sized so it never binds - the wall clock below is the binding guard.
ARCHZOO_ASSUMED_TOKENS_PER_S = 9_000.0
ARCHZOO_BUDGET_HOURS = 12.0
# Kaggle's session cap is 12 h; the kernel output is only saved on a clean exit,
# so the sweeps are stopped with margin and whatever ledger rows exist are kept.
ARCHZOO_DEADLINE_HOURS = 11.0
ARCHZOO_POLL_SECONDS = 300.0
# Written beside the ledgers and read back by ``kaggle_queue.py --queue archzoo``:
# the kernel exits 0 whatever happened (a non-zero exit puts the output at
# risk), so the outcome travels as data rather than as an exit code.
ARCHZOO_STATUS_NAME = "archzoo-status.json"


CONTAINER_NAME_ENV = "KAGGLE_CONTAINER_NAME"


def run_token() -> str:
    """A filesystem-safe token unique to this kernel run, or ``""`` outside one.

    ``KAGGLE_CONTAINER_NAME`` carries the run identity (verified 2026-09-15:
    ``kaggle_<hash>-<number>-webtier``), which is what makes the overlay work
    root collision-proof: the queues are driven one session at a time, and two
    kernels of the *same* queue must never share a directory.
    """
    name = os.environ.get(CONTAINER_NAME_ENV, "")
    if not name:
        return ""
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]


def workspace_root(mode: str = "") -> Path:
    """Where a run's handoff roots live.

    Not the kernel's working directory. In a Kaggle kernel ``/kaggle/working``
    is a 20 GB loopback mount (measured: 19.5 GiB on ``/dev/loop1``) while the
    container's root overlay has ~1 TB free (measured: 7.9 TB total, 1.1 TB
    available), and a completed seed publishes its final checkpoint per arm
    *including the AdamW moments* - ~6.4 GiB per seed, ~77 GiB over the 12-seed
    block - so the roots cannot live in ``/kaggle/working``. Nothing here needs
    to survive the session: the published dataset is restored at session start
    and pushed again at exit.

    Scoped by mode *and* run token, because the overlay is host-shared. Locally
    the roots stay repo-relative, which is what the runbook prints.
    """
    if not Path("/kaggle/working").is_dir():
        return Path(".")
    token = run_token()
    scope = f"act-lab-{mode}" if mode else "act-lab"
    return Path("/tmp") / (f"{scope}-{token}" if token else scope)


@dataclass(frozen=True)
class Queue:
    """One declared queue: its Kaggle dataset and its handoff roots."""

    mode: str
    title: str
    suffix: str = ""

    @property
    def owner(self) -> str:
        return os.environ.get(OWNER_ENV, DEFAULT_OWNER)

    @property
    def dataset(self) -> str:
        slug = os.environ.get(SLUG_ENV, DEFAULT_SLUG)
        return f"{self.owner}/{slug}{self.suffix}"

    @property
    def kernel_id(self) -> str:
        return f"{self.owner}/act-t4-{self.mode}"

    @property
    def dataset_root(self) -> str:
        root = workspace_root(self.mode)
        return str(root / "artifacts" / f"datasets{self.suffix}")

    @property
    def session_dir(self) -> str:
        root = workspace_root(self.mode)
        return str(root / "artifacts" / f"v0-sessions{self.suffix}")


QUEUES: dict[str, Queue] = {
    "session": Queue("session", "act-t4-session", ""),
    "control": Queue("control", "act-t4-control", "-control"),
    "null": Queue("null", "act-t4-null", "-null"),
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


def verdict_argv(
    python: str,
    *,
    primary: str,
    control: str,
    null: str,
    probe_batch_count: int = 1,
) -> list[str]:
    """Runbook section 4: the pre-registered verdict over all three queues."""
    return [
        python,
        "scripts/v0_verdict.py",
        "--dataset-root",
        primary,
        "--control-root",
        control,
        "--null-root",
        null,
        "--probe-batch-count",
        str(probe_batch_count),
    ]


def find_tokens_root(input_root: str | Path = INPUT_ROOT) -> Path | None:
    """Locate the attached TinyStories token stream anywhere in the mount tree.

    Same layout hazard as :func:`find_code_root`: the dataset may mount at any
    depth, so the pair ``tokens.npy`` + ``val-tokens.npy`` is searched for rather
    than assumed. Both files are required - a stream without its held-out file
    would make the sweep carve validation off the training tail, which is a
    different measurement from the record's.
    """
    root = Path(input_root)
    if not root.is_dir():
        return None
    for candidate in sorted(root.rglob("tokens.npy")):
        if (candidate.parent / "val-tokens.npy").is_file():
            return candidate.parent
    return None


def archzoo_argv(
    python: str,
    seed: int,
    *,
    data_dir: str | Path,
    artifacts_root: str | Path,
) -> list[str]:
    """The horizon arm's launch line for one seed.

    The design half restates ``arch-zoo-100m-v3-crossed`` flag for flag (crossed
    ``depth x hidden_size``, 2 replicates, 78M +/-5%, context 1024, batch 2, zoo
    seed 42); the arm half is what its NOTES.md pre-registers: flat at 1200 and
    12,000 steps, fp16, one seed.
    """
    data_dir = Path(data_dir)
    return [
        python,
        "scripts/arch_zoo_screen.py",
        "--experiment-id",
        ARCHZOO_EXPERIMENT,
        "--design",
        "crossed",
        "--cross-factors",
        "depth,hidden_size",
        "--cross-replicates",
        "2",
        "--match-target",
        "78000000",
        "--match-tolerance",
        "0.05",
        "--zoo-seed",
        "42",
        "--context-size",
        "1024",
        "--batch-size",
        "2",
        "--plan",
        "flat",
        "--flat-steps",
        ARCHZOO_FLAT_STEPS,
        "--finalist-seeds",
        str(seed),
        "--mixed-precision",
        ARCHZOO_PRECISION,
        "--device",
        "cuda",
        "--data-path",
        str(data_dir / "tokens.npy"),
        "--val-data-path",
        str(data_dir / "val-tokens.npy"),
        "--artifacts-root",
        str(artifacts_root),
        "--budget-hours",
        f"{ARCHZOO_BUDGET_HOURS:g}",
        "--assumed-tokens-per-s",
        f"{ARCHZOO_ASSUMED_TOKENS_PER_S:g}",
    ]


def kernel_metadata(
    kernel_id: str,
    title: str,
    *,
    gpu: bool = True,
    code_file: str = "kaggle_t4.py",
    extra_datasets: tuple[str, ...] = (),
) -> dict[str, object]:
    """Metadata for pushing one mode's kernel.

    ``machine_shape`` is the only honoured accelerator key; ``gpu_type`` is
    silently ignored by the CLI. Internet stays on for the dataset API.
    ``extra_datasets`` are attached after the code dataset, for a mode whose
    input is a dataset the code does not carry (the token stream).
    """
    owner = os.environ.get(OWNER_ENV, DEFAULT_OWNER)
    return {
        "id": kernel_id,
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_tpu": False,
        "machine_shape": "NvidiaTeslaT4" if gpu else "",
        "enable_internet": True,
        "dataset_sources": [f"{owner}/act-lab-code", *extra_datasets],
        "competition_sources": [],
        "kernel_sources": [],
    }


# Modes that are diagnostics rather than queues: (title, wants a GPU).
# ``preflight`` measures throughput and so needs the T4; ``probe`` only asks
# who it is on the Kaggle API, so it must not queue behind one.
#: The startup-restore measurement's artifact name, shared with the driver so a
#: drifted name cannot make the probe report a number nothing reads.
RESTORE_PROBE_NAME = "restore-probe.json"

DIAGNOSTIC_MODES = {
    "probe": ("act-t4-probe", False),
    "dtypeprobe": ("act-t4-dtypeprobe", True),
    "preflight": ("act-t4-preflight", True),
    # The battery trains tiny non-declared pairs, so it wants a GPU (the
    # placement check is about devices) but costs a few minutes, not a window.
    "selfcheck": ("act-t4-selfcheck", True),
    # The startup restore is a download and a local merge, so measuring it on a
    # GPU would bill the week's scarcest resource for a number the accelerator
    # cannot influence. It is the one measurement that answers whether a session
    # can start at all, and it is free.
    "restoreprobe": ("act-t4-restoreprobe", False),
}
PUSH_MODES = (*DIAGNOSTIC_MODES, "verdict", "archzoo", *QUEUES)


def metadata_for(mode: str) -> dict[str, object]:
    """The push metadata for one mode (queues, ``verdict``, diagnostics)."""
    owner = os.environ.get(OWNER_ENV, DEFAULT_OWNER)
    if mode in QUEUES:
        queue = QUEUES[mode]
        return kernel_metadata(queue.kernel_id, queue.title)
    if mode == "archzoo":
        # The sweep reads the token stream from a second attached dataset: the
        # code dataset ships only tracked files and the stream is local scratch.
        return kernel_metadata(
            f"{owner}/{ARCHZOO_KERNEL}",
            ARCHZOO_KERNEL,
            extra_datasets=(f"{owner}/{TOKENS_DATASET_SLUG}",),
        )
    if mode == "verdict":
        # The verdict only downloads datasets and does arithmetic - no
        # accelerator, so it never queues behind a GPU and costs no quota.
        return kernel_metadata(f"{owner}/act-t4-verdict", "act-t4-verdict", gpu=False)
    if mode in DIAGNOSTIC_MODES:
        title, gpu = DIAGNOSTIC_MODES[mode]
        return kernel_metadata(f"{owner}/{title}", title, gpu=gpu)
    raise KeyError(mode)


def _log(message: str) -> None:
    print(message, flush=True)


def _run(argv: list[str], env: dict[str, str], allowed: tuple[int, ...] = (0,)) -> int:
    _log(f"$ {' '.join(argv)}")
    proc = subprocess.run(argv, env=env, capture_output=True, text=True)
    _log(proc.stdout[-8000:])
    if proc.returncode not in allowed:
        _log("--- stderr ---")
        _log(proc.stderr[-6000:])
        raise SystemExit(f"command failed with exit {proc.returncode}: {argv[1:3]}")
    return proc.returncode


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
    src = str(WORK_ROOT / "src")
    # The env dict configures the *subprocesses* the modes launch, but the
    # pre-flight, the dtype probe and the verdict import ``act`` in this very
    # process straight after bootstrapping. Exporting PYTHONPATH never reached
    # them, so they died on ``ModuleNotFoundError: No module named 'act'`` -
    # measured 2026-09-15 on the dtype probe, which is also how the session and
    # the verdict would have failed.
    if src not in sys.path:
        sys.path.insert(0, src)
    return WORK_ROOT, dict(os.environ, PYTHONPATH=src)


def _keep(path: Path) -> None:
    """Copy an audit-trail file into the kernel output."""
    if path.is_file():
        target = Path("/kaggle/working") / path.name
        shutil.copy(path, target)
        _log(f"kept {target}")


def _keep_tree(path: Path) -> None:
    """Copy a whole experiment directory into the kernel output.

    The zoo ledgers are kilobytes, so unlike the checkpoint handoff they can
    ride the kernel output rather than a dataset publish - one fewer API call
    on the exit path, and ``kaggle kernels output`` fetches them.
    """
    if not path.is_dir():
        _log(f"nothing to keep: {path} is not a directory")
        return
    target = Path("/kaggle/working") / path.name
    if target.resolve() == path.resolve():
        return
    shutil.copytree(path, target, dirs_exist_ok=True)
    files = sorted(p.name for p in target.iterdir() if p.is_file())
    _log(f"kept {target} ({len(files)} files: {', '.join(files)})")


def _keep_seed_summaries(root: Path, output: Path | None = None) -> int:
    """Copy every published seed's summary into the kernel output, uniquely named.

    The dataset push is the durable copy, but it is also the one step that can fail
    *after* hours of training - and when it does, the session's work is otherwise gone
    with the kernel. Every number the verdict reads per seed (each arm's final loss,
    search FLOPs, mutation count) is in this kilobyte file, so keeping them makes a
    failed publish recoverable instead of a re-run. Checkpoints stay dataset-only: they
    are gigabytes, and the kernel output is not a handoff channel for them.

    The target name carries the seed because every summary is called
    ``seed-summary.json`` - copying them under their own names would keep one and
    silently drop the rest.
    """
    target_root = output if output is not None else Path("/kaggle/working")
    kept = 0
    for summary in sorted(root.rglob("seed-summary.json")):
        name = summary.parent.name  # seed-<n>
        # The publisher writes the same payload twice (once under the seed, once
        # under its reports/ directory), so the walk sees both. Keeping the second
        # would file eleven seeds' data under one `reports-summary.json`, which is
        # worse than useless on the exit path - it looks like a seed's ledger and
        # belongs to whichever seed happened to sort last.
        if not name.startswith("seed-"):
            continue
        shutil.copy(summary, target_root / f"{name}-summary.json")
        kept += 1
    if kept:
        _log(f"kept {kept} seed summaries in the kernel output (publish insurance)")
    return kept


def report_resources() -> None:
    """Log disk headroom, because the dataset root grows with every seed.

    A completed seed publishes its final checkpoint per arm including the
    AdamW moments (~3.19 GB each, the constant the working-dir ledger uses),
    so ``<dataset-root>/act-checkpoints/v0-5-1/`` reaches roughly 6.4 GB x
    12 seeds. That root lives in the kernel's working directory, which is why
    this is logged rather than assumed.
    """
    for label, path in (
        ("working", Path("/kaggle/working")),
        ("input", INPUT_ROOT),
        ("tmp", Path("/tmp")),
    ):
        try:
            usage = shutil.disk_usage(path)
        except OSError as error:
            _log(f"disk {label}: unavailable ({error})")
            continue
        _log(
            f"disk {label}: total {usage.total / 2**30:.1f} GiB | "
            f"free {usage.free / 2**30:.1f} GiB"
        )
    mounts = subprocess.run(
        ["df", "-h", "/kaggle/working", "/kaggle/temp", "/tmp", "/"],
        capture_output=True,
        text=True,
    )
    _log("--- df -h ---")
    _log((mounts.stdout or mounts.stderr).strip())
    for candidate in ("/kaggle/working", "/kaggle/temp", "/tmp"):
        path = Path(candidate)
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".act-write-probe"
            probe.write_bytes(b"x" * 1024)
            probe.unlink()
            _log(f"writable {candidate}: yes")
        except OSError as error:
            _log(f"writable {candidate}: no ({error})")
    for queue in QUEUES.values():
        root = Path(queue.dataset_root)
        if root.is_dir():
            total = sum(
                path.stat().st_size for path in root.rglob("*") if path.is_file()
            )
            _log(f"handoff root {root}: {total / 2**30:.2f} GiB")


def _log_environ() -> None:
    """Log the KAGGLE_* environment, values masked.

    Two purposes: confirm which credential variables are absent (the publish
    path relies on the kernel's own identity), and look for a run-unique
    identifier to scope the overlay work root by, so two kernels of the same
    queue can never share a directory.
    """
    secret = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    keys = sorted(k for k in os.environ if k.startswith("KAGGLE"))
    _log(f"KAGGLE_* variables: {len(keys)}")
    for key in keys:
        value = os.environ[key]
        if any(word in key.upper() for word in secret):
            value = f"<set, {len(value)} chars>"
        elif len(value) > 80:
            value = value[:77] + "..."
        _log(f"  {key}={value}")


def _reclaim(publish_log: Path | None = None) -> None:
    """Delete this run's overlay work root once its state is elsewhere.

    The overlay is shared with the host, so leaving tens of GB per session
    behind would be antisocial; the durable copy is the published dataset. The
    root is kept when the publish failed, so a failed session can be
    inspected - and because the publish outcome is already in the kernel
    output, keeping it costs nothing but space.
    """
    root = workspace_root()
    if root == Path(".") or not root.is_dir():
        return
    if publish_log is not None and publish_log.is_file():
        try:
            if json.loads(publish_log.read_text()).get("status") == "failed":
                _log(f"keeping {root}: the publish failed, state is only local")
                return
        except (OSError, ValueError):
            pass
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    shutil.rmtree(root, ignore_errors=True)
    _log(f"reclaimed {root} ({size / 2**30:.2f} GiB)")


def mode_probe() -> None:
    """Diagnostic: is the Kaggle dataset API usable inside a kernel?"""
    _log("=== Kaggle-in-kernel credential probe ===")
    report_resources()
    _log_environ()
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

    from act.training.runtime import gpu_inventory

    # The head of ``nvidia-smi`` is the GPU inventory table; its tail is the
    # process table. Logging only the tail is why the measured rate could not
    # be tied to a device count afterwards, so log the device list explicitly.
    nvidia = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    _log(nvidia.stdout if nvidia.returncode == 0 else "nvidia-smi unavailable")
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device on this kernel - enable a GPU accelerator")
    _log(f"torch {torch.__version__} | device {torch.cuda.get_device_name(0)}")
    _log(f"gpu inventory: {gpu_inventory()}")

    from act.model.v0_moe import estimated_parameter_count
    from act.training.session_plan import (
        expected_probe_cost_percent,
        tolerance_band,
    )
    from act.training.v0_session import V0SessionConfig, declared_session_config

    # Runbook pre-flight 4: the declared config must reproduce the counts the
    # tolerance band derives from.
    total, active = estimated_parameter_count(declared_session_config().model_config())
    expected_total, expected_active = 285_336_576, 134_341_632
    _log(f"declared parameters: {total:,} total / {active:,} active")
    assert total == expected_total, f"total {total} != {expected_total}"
    assert active == expected_active, f"active {active} != {expected_active}"
    _log("parameter-count assertion: OK")

    _log(f"--- runbook 1: throughput benchmark (--steps {BENCHMARK_STEPS}) ---")
    _run(benchmark_argv(sys.executable), env)
    artifact_path = work / "artifacts" / "v0-benchmark.json"
    data = json.loads(artifact_path.read_text())
    _keep(artifact_path)
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


def mode_selfcheck() -> None:
    """Run the V0 session battery on the target host.

    Its last check trains a pair's arms in their own processes and reads back
    the device each used, so on the two-T4 shape this is what confirms section
    5.2's "one arm per GPU" is what a session will actually do - before a
    session spends a window finding out. A one-device host passes too (that is
    the sequential fallback), which is why the device count is logged.
    """
    import torch

    _log("=== ACT lab: V0 session self-check ===")
    work, env = _bootstrap()

    from act.training.runtime import gpu_inventory

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device on this kernel - enable a GPU accelerator")
    _log(f"torch {torch.__version__} | device {torch.cuda.get_device_name(0)}")
    _log(f"gpu inventory: {gpu_inventory()}")

    _run(
        [sys.executable, "scripts/v0_session_selfcheck.py", "--device", "cuda"],
        env,
    )
    _log("=== SELF-CHECK COMPLETE (all checks passed) ===")


def _free_gib(path: Path) -> float:
    """Free space at the nearest existing ancestor of ``path``, in GiB."""
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free / 2**30
    except OSError:
        return float("nan")


def mode_restoreprobe() -> None:
    """Price a session's startup restore, which no session has ever reported.

    Every session restores the *whole* published dataset before it trains a
    token, because a queue republishes its entire handoff at exit and so the
    roots have to be merged locally first - which makes the cost of starting a
    session grow with every seed the queue publishes (~6.4 GiB per seed).
    Measured 2026-09-21: a session ended ERROR after 1.32 h with no log and no
    output, and this startup cost is one of the two things known to occupy a
    session for that long. It had been priced nowhere because the only logs that
    carry it are the sessions' own, and those are the logs the platform discards.

    CPU-only by design (see ``DIAGNOSTIC_MODES``): the accelerator cannot change
    a download time or a copy time. Nothing is published - the queue's dataset
    is read, never written - so this cannot disturb the campaign's progress,
    which is read from the published seed set.
    """
    import time

    queue = QUEUES["session"]
    _log("=== ACT lab: startup-restore probe (CPU, no quota) ===")
    work, env = _bootstrap()
    # Imported *after* the bootstrap, which is what puts the mounted ``src/`` on
    # sys.path. This process imports act directly instead of only configuring the
    # subprocesses, so an import above that line dies with ModuleNotFoundError -
    # the failure mode `_bootstrap` records, measured 2026-09-15 on the dtype
    # probe and again on this probe's first push.
    from act.ops.kaggle_remote import default_client, restore_latest
    from act.training.session_plan import working_dir_headroom_gb

    report_resources()
    root = Path(queue.dataset_root)
    _log(f"dataset root: {root}")
    _log(f"remote dataset: {queue.dataset}")
    _log(f"registered working-dir headroom: {working_dir_headroom_gb():.2f} GiB")

    free_before = _free_gib(root.parent)
    started = time.time()
    restored = restore_latest(
        root, dataset=queue.dataset, client=default_client(), log=_log
    )
    minutes = (time.time() - started) / 60.0
    free_after = _free_gib(root)
    _log(
        f"=== restore probe: {len(restored)} seed version(s) in {minutes:.1f} min, "
        f"{free_before - free_after:.2f} GiB written, "
        f"{free_after:.2f} GiB free after ==="
    )
    _log(f"seeds restored: {restored}")
    _log(
        "A session pays this before training. The registered session window is "
        f"{DEFAULT_HOURS:g} h and a pair is ~1.5 h, so a restore that costs a "
        "meaningful fraction of a window is a scheduling input, not an overhead."
    )
    # Written out as well as logged, because a kernel log is not durable: the
    # platform keeps none at all for a run it kills, and this number is the one
    # that decides whether a session can start - so it belongs with the other
    # measured artifacts, where it can be read and audited later.
    record = {
        "dataset": queue.dataset,
        "dataset_root": str(root),
        "seeds_restored": restored,
        "minutes": minutes,
        "gib_written": free_before - free_after,
        "gib_free_before": free_before,
        "gib_free_after": free_after,
        "registered_session_hours": DEFAULT_HOURS,
        "accelerator": "cpu",
    }
    artifact = work / "artifacts" / RESTORE_PROBE_NAME
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    _keep(artifact)


def mode_session(queue: Queue) -> None:
    """Runbook 2 / 2b / 2c: one session's worth of one declared queue."""
    import torch

    hours = float(os.environ.get(HOURS_ENV, str(DEFAULT_HOURS)))
    started = time.time()
    _log(f"=== ACT lab: {queue.mode} session ===")
    _log(f"dataset={queue.dataset} hours={hours:g} seeds={SEED_BLOCK}")
    work, env = _bootstrap()

    from act.training.runtime import gpu_inventory

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device on this kernel - enable a GPU accelerator")
    _log(f"torch {torch.__version__} | device {torch.cuda.get_device_name(0)}")
    _log(f"gpu inventory: {gpu_inventory()}")
    report_resources()

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
    _keep(artifact_path)

    _log(f"--- {queue.mode} queue (--hours {hours:g}) ---")
    _run(session_argv(queue, hours), env)

    # Runbook 6: keep the audit trail in the kernel output, so the per-pair
    # actions and the publish outcome survive without the dataset API.
    _keep(work / queue.session_dir / "session-log.json")
    publish_log = work / queue.dataset_root / "remote-publish-log.json"
    _keep(publish_log)
    # And the per-seed payloads the verdict reads, because the publish above is the
    # step that can fail after the hours are already spent.
    _keep_seed_summaries(work / queue.dataset_root)
    report_resources()
    _reclaim(publish_log)

    _log(f"=== {queue.mode} session finished after {(time.time() - started) / 3600:.2f} h ===")


def _missing_archzoo_runs(
    rows: dict[str, dict[str, int]],
    *,
    seeds: tuple[int, ...],
    requested: list[int],
    expected: int,
) -> list[str]:
    """Requested ``(length, seed)`` runs with no completed row, named for a log.

    ``expected`` is the design's candidate count; zero means the design could not
    be read, and the bar falls back to one row per requested length rather than
    treating an unknown grid as served.
    """
    return [
        f"seed {seed} at {step} steps"
        for seed in seeds
        for step in requested
        if rows.get(str(seed), {}).get(str(step), 0) < max(1, expected)
    ]


def _expected_candidates(experiment_dir: Path) -> int:
    """How many candidates the design declares, or 0 when it cannot be read.

    Zero means "unknown", which makes the completeness check fall back to
    requiring at least one row per requested length - it never silently counts an
    unreadable design as a served grid.
    """
    definition = experiment_dir / "zoo_definition.json"
    if not definition.is_file():
        return 0
    try:
        return len(json.loads(definition.read_text()).get("candidates", []))
    except (OSError, ValueError, AttributeError):
        return 0


def _ledger_rows(experiment_dir: Path, seed: int) -> dict[str, int]:
    """Completed runs per training length for one seed, read off its ledgers."""
    rows: dict[str, int] = {}
    for ledger in sorted(experiment_dir.glob(f"flat*_seed{seed}_results.jsonl")):
        length = ledger.name.split("_seed")[0].removeprefix("flat")
        try:
            rows[length] = sum(1 for line in ledger.read_text().splitlines() if line.strip())
        except OSError:
            rows[length] = 0
    return rows


def mode_archzoo() -> None:
    """The horizon arm: one seed per T4, ledgers into the kernel output.

    ``scripts/arch_zoo_screen.py`` is single-process and the two members of a
    T4x2 host are otherwise idle, so parallelism is one sweep per device with
    ``CUDA_VISIBLE_DEVICES`` set, not a distributed run - the same shape as the
    session runner's one-arm-per-GPU placement. Each sweep owns its own seed's
    ledgers (``flat<steps>_seed<seed>_results.jsonl``); the shared design files
    are written identically by both. The wall clock is the binding guard: the
    kernel output only survives a clean exit, so at the deadline the sweeps are
    stopped and whatever rows exist are kept rather than lost with the session.
    """
    import torch

    started = time.time()
    _log("=== ACT lab: architecture-zoo horizon arm ===")
    _log(
        f"experiment={ARCHZOO_EXPERIMENT} seeds={ARCHZOO_SEEDS} "
        f"steps={ARCHZOO_FLAT_STEPS} precision={ARCHZOO_PRECISION} "
        f"deadline={ARCHZOO_DEADLINE_HOURS:g}h"
    )
    work, env = _bootstrap()

    from act.training.runtime import gpu_inventory

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device on this kernel - enable a GPU accelerator")
    devices = torch.cuda.device_count()
    _log(f"torch {torch.__version__} | {devices} device(s): {gpu_inventory()}")
    report_resources()

    data_dir = find_tokens_root()
    if data_dir is None:
        raise SystemExit(
            "no token stream under /kaggle/input (expected tokens.npy beside "
            f"val-tokens.npy; attach the {TOKENS_DATASET_SLUG} dataset)"
        )
    _log(f"token stream: {data_dir}")

    artifacts_root = work / "artifacts"
    experiment_dir = artifacts_root / ARCHZOO_EXPERIMENT
    experiment_dir.mkdir(parents=True, exist_ok=True)

    sweeps: list[tuple[int, subprocess.Popen[bytes], object]] = []
    for index, seed in enumerate(ARCHZOO_SEEDS):
        device = index % devices
        argv = archzoo_argv(
            sys.executable, seed, data_dir=data_dir, artifacts_root=artifacts_root
        )
        log_path = experiment_dir / f"run_seed{seed}.log"
        handle = open(log_path, "wb")  # noqa: SIM115 - closed after the wait loop
        _log(f"$ CUDA_VISIBLE_DEVICES={device} {' '.join(argv)}  > {log_path.name}")
        proc = subprocess.Popen(
            argv,
            env=dict(env, CUDA_VISIBLE_DEVICES=str(device), PYTHONUNBUFFERED="1"),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        sweeps.append((seed, proc, handle))

    deadline = started + ARCHZOO_DEADLINE_HOURS * 3600
    stopped_at_deadline = False
    while any(proc.poll() is None for _, proc, _ in sweeps):
        if time.time() >= deadline:
            stopped_at_deadline = True
            _log(f"deadline {ARCHZOO_DEADLINE_HOURS:g} h reached: stopping the sweeps")
            for _, proc, _ in sweeps:
                if proc.poll() is None:
                    proc.terminate()
            for _, proc, _ in sweeps:
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()
            break
        time.sleep(min(ARCHZOO_POLL_SECONDS, max(1.0, deadline - time.time())))
        elapsed = (time.time() - started) / 3600
        progress = ", ".join(
            f"seed {seed}: " + (
                " ".join(f"{k}x{v}" for k, v in sorted(_ledger_rows(experiment_dir, seed).items()))
                or "0 rows"
            )
            + ("" if proc.poll() is None else f" (exit {proc.returncode})")
            for seed, proc, _ in sweeps
        )
        _log(f"[{elapsed:5.2f} h] {progress}")

    for _, _, handle in sweeps:
        handle.close()  # type: ignore[attr-defined]

    for seed, proc, _ in sweeps:
        log_path = experiment_dir / f"run_seed{seed}.log"
        tail = log_path.read_bytes()[-3000:].decode("utf-8", "replace") if log_path.is_file() else ""
        _log(f"--- seed {seed}: exit {proc.returncode}; log tail ---")
        _log(tail)

    # Like the verdict kernel, this one exits 0 whatever happened: the ledgers
    # travel in the kernel output, which a non-zero exit would put at risk, so
    # the outcome travels beside them and the driver reads it back.
    hours = (time.time() - started) / 3600
    rows = {str(seed): _ledger_rows(experiment_dir, seed) for seed, _, _ in sweeps}
    # Completeness is read off the ledgers, not inferred from a clean exit. The
    # screen catches a per-candidate exception and keeps going, so a sweep exits 0
    # even when every run in it failed - the horizon arm's first attempt trained
    # none of its 12,000-step runs and still reported complete=True, because exit
    # codes were the only evidence consulted. What the arm *owes* is its requested
    # (length, seed) grid, so that grid is what is checked.
    requested = [int(part) for part in ARCHZOO_FLAT_STEPS.split(",") if part.strip()]
    expected = _expected_candidates(experiment_dir)
    missing = _missing_archzoo_runs(
        rows, seeds=ARCHZOO_SEEDS, requested=requested, expected=expected
    )
    status = {
        "experiment": ARCHZOO_EXPERIMENT,
        "seeds": list(ARCHZOO_SEEDS),
        "flat_steps": ARCHZOO_FLAT_STEPS,
        "mixed_precision": ARCHZOO_PRECISION,
        "devices": devices,
        "hours": hours,
        "stopped_at_deadline": stopped_at_deadline,
        "exit_codes": {str(seed): proc.returncode for seed, proc, _ in sweeps},
        "ledger_rows": rows,
        "expected_runs_per_length": expected,
        "missing": missing,
        "complete": not stopped_at_deadline
        and all(proc.returncode == 0 for _, proc, _ in sweeps)
        and not missing,
    }
    (experiment_dir / ARCHZOO_STATUS_NAME).write_text(json.dumps(status, indent=2) + "\n")
    _keep_tree(experiment_dir)
    report_resources()
    _log(json.dumps(status, indent=2))
    _log(f"=== archzoo finished after {hours:.2f} h (complete={status['complete']}) ===")


def probe_pair_schedule(
    tokens_per_second: float, arms: int, device: torch.device
) -> PairSchedule:
    """The probe's plan for *this* host, priced at its arm placement.

    The benchmark prices the launch input at the placement it measures
    (``arm_slots(...)``), but the probe built its schedule with the dataclass
    default - one device - so on Kaggle's two-T4 kernels the plan a human reads
    to decide (pairs per session, sessions for the primary queue) was twice the
    run the launch would actually execute. Measured on the same host
    2026-09-20: the self-check reports 2 devices and one arm per device, so the
    sequential arithmetic disagreed with the runner's own. Defaulting to one
    device stays honest for a host that really has one.
    """
    from act.training.session_plan import PairSchedule
    from act.training.v0_session import arm_slots

    return PairSchedule(
        tokens_per_second=tokens_per_second,
        arms=arms,
        arm_slots=arm_slots(arms, device),
    )


def dtypeprobe_report(
    rates: dict[str, float], arms: int, device: torch.device
) -> str:
    """The probe's plan lines, priced at the placement the runner will use.

    One spelling of the probe's arithmetic, so the number a human reads and the
    number the launch plans from cannot come apart - the defect this replaced.
    Both session counts are printed: the registered section 5.2 arithmetic and
    the one the runner's own start gate will honour, because a reader planning
    from the registered field alone expects one session too few whenever the
    45-minute publish reserve binds. The rule-1 numbers are priced at the window
    a session is *declared* (the measured 8 h platform cap less its reserve, not
    the documented 12 h clock section 5.2 was written for), so the line states
    the plan the launch will actually run.
    """
    import math

    from act.training.session_plan import (
        DECLARED_SESSION_HOURS,
        PairSchedule,
        session_pair_capacity,
        three_queue_plan,
    )

    lines: list[str] = []
    for dtype, rate in rates.items():
        schedule = probe_pair_schedule(rate, arms, device)
        capacity = session_pair_capacity(
            schedule.pair_time_hours, DECLARED_SESSION_HOURS
        )
        # The declared null seed is a single arm, so its pair time is one wave;
        # the campaign total is what the driver's re-push loop and the quota
        # gate care about, and it prices the null queue at full rate.
        null_schedule = PairSchedule(tokens_per_second=rate, arms=1, arm_slots=1)
        campaign = three_queue_plan(
            schedule.pair_time_hours,
            null_pair_time_hours=null_schedule.pair_time_hours,
            hours=DECLARED_SESSION_HOURS,
        )
        lines.append(
            f"{dtype}: {rate:.1f} tokens/s -> pair "
            f"{schedule.pair_time_hours:.3f} h, "
            f"{capacity} pairs/session by rule 1 at the declared "
            f"{DECLARED_SESSION_HOURS:g} h window "
            f"({schedule.full_pairs_per_session} registered), "
            f"{math.ceil(12 / capacity) if capacity else 0} sessions for the "
            f"primary queue ({schedule.sessions_for_primary} registered); "
            f"three queues {campaign.sessions} sessions / "
            f"{campaign.training_hours:.1f} h training floor"
        )
    return "\n".join(lines)


def mode_dtypeprobe() -> None:
    """Diagnostic: the declared step loop timed in fp32 and in fp16.

    The section 5.2 dtype rule reads the *fp32* rate and, below 2,500
    tokens/s, selects fp16 - but the schedule is derived from that same fp32
    number. On a T4 fp16 runs on tensor cores (65 vs 8.1 TFLOP/s), so if the
    two rates differ the plan and the configuration that actually runs have
    come apart, and the run buys more sessions than it needs.
    """
    import math

    import torch

    _bootstrap()
    from act.model.v0_moe import V0MoECausalLM
    from act.training.runtime import resolve_device
    from act.training.v0_session import (
        V0SessionConfig,
        arm_slots,
        declared_session_config,
    )

    config = declared_session_config()
    arms = len(config.arms)
    device = resolve_device(config.device)
    slots = arm_slots(arms, device)
    steps, warmup = 10, 3
    _log("=== ACT lab: dtype throughput probe ===")
    _log(f"declared config: batch {config.batch_size} x context {config.context_size}")
    _log(
        f"arm placement: {arms} arm(s) in {math.ceil(arms / slots)} wave(s) "
        f"of slots={slots} on {device}"
    )

    def measure(dtype: str) -> float:
        torch.manual_seed(15012)  # outside the reserved 15000-15011 block
        model = V0MoECausalLM(config.model_config()).to("cuda")
        # The declared config trains with checkpointed activations, so the
        # comparison must too - otherwise it measures a loop nobody runs.
        model.activation_checkpointing = True
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        scaler = torch.amp.GradScaler("cpu", enabled=dtype == "fp16")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(15012)
        shape = (config.batch_size, config.context_size)

        def one_step() -> None:
            tokens = torch.randint(
                0, config.vocab_size, shape, generator=generator
            ).to("cuda")
            optimizer.zero_grad(set_to_none=True)
            if dtype == "fp16":
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model.loss(tokens)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model.loss(tokens)
                loss.backward()
                optimizer.step()

        for _ in range(warmup):
            one_step()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(steps):
            one_step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        del model, optimizer
        torch.cuda.empty_cache()
        return steps * config.batch_size * config.context_size / elapsed

    _log(dtypeprobe_report({dtype: measure(dtype) for dtype in ("fp32", "fp16")}, arms, device))
    _log("=== DTYPE PROBE COMPLETE ===")


def mode_verdict() -> None:
    """Runbook 4: restore all three queues from Kaggle and compute the verdict."""
    work, env = _bootstrap()
    from act.ops.kaggle_remote import default_client, restore_latest

    client = default_client()
    for queue in QUEUES.values():
        restored = restore_latest(
            queue.dataset_root, dataset=queue.dataset, client=client, log=_log
        )
        _log(f"{queue.mode}: {len(restored)} seed version(s) restored")

    argv = verdict_argv(
        sys.executable,
        primary=QUEUES["session"].dataset_root,
        control=QUEUES["control"].dataset_root,
        null=QUEUES["null"].dataset_root,
    )
    code = _run(argv, env, allowed=tuple(VERDICT_EXIT_CODES))
    _log(f"verdict exit code {code}: {VERDICT_EXIT_CODES[code]}")

    verdict_path = work / QUEUES["session"].dataset_root / "v0-5-1-verdict.json"
    _keep(verdict_path)
    if verdict_path.is_file():
        _log(json.dumps(json.loads(verdict_path.read_text()), indent=2, sort_keys=True))
    _reclaim()
    _log("=== VERDICT COMPLETE ===")


def main() -> int:
    mode = os.environ.get(MODE_ENV, "")
    if mode == "metadata":
        queue_mode = os.environ.get(QUEUE_ENV, "session")
        path = Path("kernel-metadata.json")
        path.write_text(json.dumps(metadata_for(queue_mode), indent=2) + "\n")
        _log(f"wrote {path} for mode {queue_mode!r}")
        return 0
    if mode == "probe":
        mode_probe()
        return 0
    if mode == "preflight":
        mode_preflight()
        return 0
    if mode == "dtypeprobe":
        mode_dtypeprobe()
        return 0
    if mode == "selfcheck":
        mode_selfcheck()
        return 0
    if mode == "restoreprobe":
        mode_restoreprobe()
        return 0
    if mode in QUEUES:
        mode_session(QUEUES[mode])
        return 0
    if mode == "verdict":
        mode_verdict()
        return 0
    if mode == "archzoo":
        mode_archzoo()
        return 0
    raise SystemExit(
        f"set {MODE_ENV} to one of: metadata, " + ", ".join(PUSH_MODES)
    )


if __name__ == "__main__":
    raise SystemExit(main())
