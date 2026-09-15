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
- ``session``   runbook 2: the primary pair queue;
- ``control``   runbook 2b: the pool-matched (loss-blind) control queue;
- ``null``      runbook 2c: the do-nothing null queue;
- ``verdict``   runbook 4: restore all three queues from Kaggle and compute
                the pre-registered verdict.

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

MODE_ENV = "ACT_T4_MODE"
QUEUE_ENV = "ACT_T4_QUEUE"
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
# v0_verdict.py exit codes are the pre-registered verdict, not failures.
VERDICT_EXIT_CODES = {0: "PASS", 1: "FAIL", 2: "PASS with qualification"}

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/act-lab")


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


def kernel_metadata(
    kernel_id: str,
    title: str,
    *,
    gpu: bool = True,
    code_file: str = "kaggle_t4.py",
) -> dict[str, object]:
    """Metadata for pushing one mode's kernel.

    ``machine_shape`` is the only honoured accelerator key; ``gpu_type`` is
    silently ignored by the CLI. Internet stays on for the dataset API.
    """
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
        "dataset_sources": [f"{os.environ.get(OWNER_ENV, DEFAULT_OWNER)}/act-lab-code"],
        "competition_sources": [],
        "kernel_sources": [],
    }


# Modes that are diagnostics rather than queues: (title, wants a GPU).
# ``preflight`` measures throughput and so needs the T4; ``probe`` only asks
# who it is on the Kaggle API, so it must not queue behind one.
DIAGNOSTIC_MODES = {
    "probe": ("act-t4-probe", False),
    "dtypeprobe": ("act-t4-dtypeprobe", True),
    "preflight": ("act-t4-preflight", True),
}
PUSH_MODES = (*DIAGNOSTIC_MODES, "verdict", *QUEUES)


def metadata_for(mode: str) -> dict[str, object]:
    """The push metadata for one mode (queues, ``verdict``, diagnostics)."""
    owner = os.environ.get(OWNER_ENV, DEFAULT_OWNER)
    if mode in QUEUES:
        queue = QUEUES[mode]
        return kernel_metadata(queue.kernel_id, queue.title)
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
    report_resources()
    _reclaim(publish_log)

    _log(f"=== {queue.mode} session finished after {(time.time() - started) / 3600:.2f} h ===")


def mode_dtypeprobe() -> None:
    """Diagnostic: the declared step loop timed in fp32 and in fp16.

    The section 5.2 dtype rule reads the *fp32* rate and, below 2,500
    tokens/s, selects fp16 - but the schedule is derived from that same fp32
    number. On a T4 fp16 runs on tensor cores (65 vs 8.1 TFLOP/s), so if the
    two rates differ the plan and the configuration that actually runs have
    come apart, and the run buys more sessions than it needs.
    """
    import torch

    _bootstrap()
    from act.model.v0_moe import V0MoECausalLM
    from act.training.session_plan import PairSchedule
    from act.training.v0_session import V0SessionConfig

    config = V0SessionConfig()
    steps, warmup = 10, 3
    _log("=== ACT lab: dtype throughput probe ===")
    _log(f"declared config: batch {config.batch_size} x context {config.context_size}")

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

    for dtype in ("fp32", "fp16"):
        rate = measure(dtype)
        schedule = PairSchedule(tokens_per_second=rate)
        _log(
            f"{dtype}: {rate:.1f} tokens/s -> pair {schedule.pair_time_hours:.3f} h, "
            f"{schedule.full_pairs_per_session} pairs/session, "
            f"{schedule.sessions_for_primary} sessions for the primary queue"
        )
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
    if mode in QUEUES:
        mode_session(QUEUES[mode])
        return 0
    if mode == "verdict":
        mode_verdict()
        return 0
    raise SystemExit(
        f"set {MODE_ENV} to one of: metadata, verdict, " + ", ".join(PUSH_MODES)
    )


if __name__ == "__main__":
    raise SystemExit(main())
