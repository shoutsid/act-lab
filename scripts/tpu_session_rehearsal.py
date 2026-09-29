"""Rehearse the registered V0 *session* path on a TPU, before any queue moves there.

The Gate-4 campaign is GPU-quota-gated: the three queues need roughly 52 GPU-hours against a
30 GPU-hour week, so the block's calendar is two weeks whatever else happens. Kaggle's TPU
quota is **separate** (20 TPU-hours/week beside the 30 GPU-hours) and was measured at
**19.86/20 hours idle** on 2026-09-27 while the GPU week stood at 1.65/30. The priced route
into that quota is section 5.14's option 2: the **null queue** is twelve single-arm seeds,
bills **16.97 h** of T4 wall clock, and would take **8.06 h of TPU time at the port rule's
own 2.0x bar** - leaving both pair queues on the GPU.

What that option was gated on is now measured: `tpu_probe.py` v6 returned **PORT** (15,034.2
tokens/s on eight bound ranks, 2.96x the T4 pair, recompile 2.07% of a training step). What
it did **not** measure is a session. The probe timed a step loop around a hand-built model;
the registered path adds activation checkpointing, an optimizer with AdamW state, the
per-arm RNG/seed contract, checkpoint save **and resume**, the seed handoff publish, and the
two-arm placement - and none of it has ever run on XLA. Moving a queue onto that is a real
change, and the honest way to price it is to run the path rather than assume it.

So this runs the real `run_v0_session` at **development fidelity** on whatever device the
caller names (``xla`` on a TPU host, ``cpu`` for a dry run) in four contained phases:

1. ``environment`` - torch / torch-xla versions, the platform's TPU topology block, the
   device count and a bound device, recorded with unset keys kept as ``None``.
2. ``session`` - a single-arm **null** session (the queue the amendment would move), at the
   dev shape, with a revision recorded on every checkpoint; the arm's own result is kept,
   and the file it wrote is read back off the disk to say which revision is *in* it and
   what device its tensors are on (a checkpoint is only evidence if it loads as itself).
3. ``resume`` - the same session again from the checkpoint it just wrote. The resume contract
   is what makes a deadline stop survivable, and it is the first thing XLA can break.
4. ``handoff`` - publish that seed through ``publish_seed`` into a temporary dataset root and
   read the summary back, because the verdict's null clause consumes exactly that file
   (``arms.null.search_flops == 0`` and ``mutation_count == 0``).

Every phase is contained (``tpu_probe._run_phase``): one that raises is recorded with its
exception type and text and the phases after it still run, because the run is queued once and
the interesting failure is usually the first one. The record is saved after each phase, and
each phase files its own entry - and its log - *before* the work starts, so a phase that dies
in the middle of a training step still leaves the lines it printed rather than only the
exception that ended it.

v1 (2026-09-27) was queued once and came back with nothing to read, so v2 is built around the
failure it actually had. The session phase never returned: the log carries ``--- phase
session``, then three hours of silence, then ``Killed``, and no checkpoint was ever written -
where the same twenty development-fidelity steps take 24 s on a CPU. Everything the run could
have said was in a record that only gets written when a phase returns, and the session's own
log lines go into that record's list. So the session's lines are now **printed as they
arrive**, and every phase runs under a :class:`_Watch` that samples the host's resident set and
the device's bytes in use on a timer, printing each sample and rewriting a sidecar file beside
the record. A run that is killed rather than raised then still leaves a trace with a shape -
flat means the phase is working, climbing means it is accumulating - and a per-phase
``faulthandler`` deadline dumps every thread's stack before the platform's own kill, which
names the call that hung instead of leaving it to be guessed at.

The training path itself was the other half. XLA is lazy: the ops of a step are traced into a
graph that does not execute until a step is marked, and ``_train_arm`` never marked one, so the
loop traced the whole run and executed it as a single program. ``runtime.step_marker`` now
marks the step after the optimizer's update - the point the probe's own measured loop marks,
and a no-op on the CPU and on CUDA, so the registered campaign's numbers are untouched.

**What this is not.** It is not a Gate-4 result, not a port, and not evidence about the
hypothesis: it trains at development fidelity on synthetic tokens and reports whether the
*path* runs on this device and what it booked. It does not move any queue, and it does not
touch the registered block. Per-core parallelism (eight seeds at once, one per chip) is
deliberately **not** measured here - the probe measured that pattern for the step loop, and
one session on one device is the new question. ``not_measured`` says so in the record.

    PYTHONPATH=src python scripts/tpu_session_rehearsal.py --device cpu --steps 4   # dry run
    PYTHONPATH=src python scripts/tpu_session_rehearsal.py --device xla            # on a TPU

``--device cpu`` cannot reach the branches that only run on XLA, which is where every failure
on this path has been. ``torch_xla`` also runs XLA on a CPU host (``PJRT_DEVICE=CPU``), so the
same torch/torch_xla pair the platform runs can be rehearsed locally at development fidelity -
see the runbook's "Before firing at a TPU" section for the throwaway env, and add
``--checkpoint-activations`` to take the declared config's activation-checkpointing field with
it (the field whose absence from the dev config let the 2026-09-29 defect reach a queue).
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Reused rather than re-implemented: the mount search (a script kernel receives only its own
# code file, so ``act`` arrives through the mounted code dataset), the per-phase containment,
# the record writer, and the TPU/hardware reports. Two copies of the mount search is exactly
# how a run ends up importing a different tree than the one it was staged from.
import tpu_probe  # noqa: E402

RECORD_NAME = "tpu-session-rehearsal.json"
OUTPUT = Path(os.environ.get("TPU_SESSION_OUTPUT", f"/kaggle/working/{RECORD_NAME}"))

#: The phases, in the order a failure would cost the reading. The session comes first because
#: everything else depends on it; the resume second because it is the first thing XLA can
#: break; the handoff last because it needs a result to publish.
PHASES = ("environment", "session", "resume", "handoff")

#: How long one phase may run before the run dumps every thread's stack and stops itself.
#: The session gets the longest leash because it is the phase that trains; the others are
#: bookkeeping around it. Bounded rather than unlimited, because a rehearsal whose failure
#: mode is "three hours then killed" costs the whole queued slot and reports nothing.
PHASE_DEADLINE_SECONDS = {
    "environment": 600.0,
    "session": 1800.0,
    "resume": 900.0,
    "handoff": 600.0,
}

#: How often :class:`_Watch` samples the process while a phase runs.
WATCH_SECONDS = 30.0

#: What this rehearsal deliberately leaves alone, stated in the record so a reader does not
#: read a partial rehearsal as a complete port study.
NOT_MEASURED = (
    "per-core parallelism (eight seeds at once, one per chip): the probe measured that "
    "pattern for the step loop, and this rehearsal's question is one session on one device",
    "declared-fidelity throughput: the shape here is the development one, so its tokens/s "
    "is a path measurement and not a schedule input",
    "the two-arm pair placement: the null queue is single-arm by construction, and the pair "
    "queues are not the ones this option moves",
)


def phase_deadline_seconds(
    name: str, deadlines: Mapping[str, float] | None = None
) -> float:
    """The deadline for one phase, overridable for a whole run.

    ``TPU_SESSION_DEADLINE`` sets every phase to the same value, which is what a
    shortened re-fire wants; the per-phase defaults are what an ordinary run uses.
    ``deadlines`` names another table, for a caller whose phases cost more than these
    do - the declared shape is ~19x the dev model per step, so the session phase's
    1800 s leash here is not a leash there.
    """
    override = os.environ.get("TPU_SESSION_DEADLINE")
    if override:
        return float(override)
    table = deadlines or PHASE_DEADLINE_SECONDS
    return table.get(name, table["session"])


def watch_seconds() -> float:
    """How often the watch samples, overridable through ``TPU_SESSION_WATCH``."""
    return float(os.environ.get("TPU_SESSION_WATCH") or WATCH_SECONDS)


def _rss_kb() -> int | None:
    """This process's resident set, from the kernel's own accounting.

    ``/proc/self/status`` rather than a dependency: the sample has to be cheap enough to run
    on a timer for the length of a training phase, and ``None`` is the honest reading on a
    host that does not have the file rather than a zero that would read as flat.
    """
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _device_bytes_used() -> int | None:
    """TPU memory in use, or ``None`` where the runtime does not report it.

    The second growth signal beside the host's resident set. A traced-but-unexecuted graph
    grows on *both* - the HLO in the host's heap and the buffers the device allocator is
    holding - so a trace that watched only one of them could call an accumulating phase flat.
    """
    try:  # pragma: no cover - only a TPU host reaches the body
        import torch_xla
        import torch_xla.runtime as xr

        return int(xr.memory_info(torch_xla.device()).get("bytes_used") or 0)
    except Exception:  # noqa: BLE001 - a diagnostic, never a failure
        return None


class _Watch:
    """What the process looked like while a phase ran, printed and kept.

    Every phase in v1 could hang and say nothing, and the session phase did. The session's
    log lines live in the record's list, and the record is only written when a phase returns,
    so a killed run's whole account was the phase boundary and, three hours later, ``Killed``.
    This samples the host's resident set and the device's bytes in use on a timer, prints each
    sample into the kernel log (the one channel a killed run still has) and rewrites a sidecar
    file beside the record (the other). Climbing samples are the accumulating-graph failure
    this rehearsal exists to catch; flat ones are a phase that is simply working.
    """

    def __init__(self, name: str, *, seconds: float, journal: Path) -> None:
        self.name = name
        self.samples: list[dict[str, Any]] = []
        self._seconds = seconds
        self._journal = journal
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = time.time()

    def _sample(self) -> dict[str, Any]:
        sample = {
            "seconds": round(time.time() - self._started, 1),
            "rss_kb": _rss_kb(),
            "device_bytes_used": _device_bytes_used(),
        }
        self.samples.append(sample)
        print(f"[watch {self.name}] {json.dumps(sample, sort_keys=True)}", flush=True)
        self._write_journal()
        return sample

    def _write_journal(self) -> None:
        """Rewrite the sidecar through a rename, so a killed process leaves a whole file."""
        temporary = self._journal.with_name(self._journal.name + ".part")
        temporary.write_text(
            json.dumps(self.samples, indent=1) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self._journal)

    def _run(self) -> None:
        while not self._stopped.wait(self._seconds):
            self._sample()

    def start(self) -> "_Watch":
        self._sample()
        self._thread = threading.Thread(
            target=self._run, name=f"watch-{self.name}", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=self._seconds + 1)


def _session_log(lines: list[str], label: str) -> Callable[[str], None]:
    """The session's log callback: printed as it arrives *and* kept in the record.

    Both, because they fail differently. The record's copy is what a completed run is read
    through; the printed copy is what a killed run still has, which is the run that needs it.
    """

    def record_line(line: str) -> None:
        print(f"[{label}] {line}", flush=True)
        lines.append(line)

    return record_line


def _revision() -> str:
    """The revision this code came from, by the same reader the published records use.

    Imported rather than re-read: ``act.training.cli._git_revision`` already falls back to the
    mounted code dataset's ``revision.txt``, and a rehearsal that reported a revision the
    sessions would not report would be measuring a different question.
    """
    from act.training.cli import _git_revision

    return _git_revision()


def _dev_null_config(
    device: str, *, steps: int, seed: int, activation_checkpointing: bool = False
):
    """The dev shape with one null arm: the queue section 5.14's option would move.

    The dev shape (``declared=False``) rather than the declared 285M config, because this is
    a rehearsal of the *path*: the probe already measured the declared shape's throughput, and
    a portfolio of fresh XLA failures is cheaper to find at 20 steps than at 6,000.

    ``activation_checkpointing`` is carried as a flag because it is the one field where the
    development and declared configs disagree in a way that matters: the dev default is off
    (memory is not the constraint) and ``declared_session_config`` sets it on. Measured
    2026-09-29: a rehearsal that ran the path at dev fidelity with this at its dev value went
    green, and the declared-shape probe fired at the same path died in its first step because
    ``torch.utils.checkpoint`` cannot run on XLA at all. A rehearsal of the *path* is then
    only as good as the fields it turns on with it.
    """
    from act.training.v0_session import V0SessionConfig

    return V0SessionConfig(
        seed=seed,
        arms=("null",),
        device=device,
        total_steps=steps,
        checkpoint_interval=max(1, steps // 2),
        max_checkpoints=2,
        activation_checkpointing=activation_checkpointing,
    )


def _arm_reading(arm: Any, checkpoints: list[Path]) -> dict[str, Any]:
    """What one arm's own result says, plus the checkpoint files it left behind."""
    return {
        "mode": arm.mode,
        "validation_loss": arm.validation_loss,
        "final_loss": arm.final_loss,
        "mutation_count": arm.mutation_count,
        "search_flops": arm.search_flops,
        "training_flops": arm.training_flops,
        "parameter_count": arm.parameter_count,
        "active_parameter_count": arm.active_parameter_count,
        "architecture_hash": arm.architecture_hash,
        "device": arm.device,
        "wall_time_seconds": arm.wall_time_seconds,
        "tokens_per_second": arm.tokens_per_second,
        "checkpoints": [
            {"name": path.name, "bytes": path.stat().st_size} for path in checkpoints
        ],
    }


def _checkpoint_files(root: Path, mode: str | None = None) -> list[Path]:
    """Checkpoint files under a session root, optionally only one arm's.

    The arm's own directory rather than every checkpoint found: a pair's two arms write to
    ``arm-<mode>`` side by side, and a reading that mixed them would report one arm's file
    count under the other's name.
    """
    pattern = f"**/arm-{mode}/checkpoint-*.pt" if mode else "**/checkpoint-*.pt"
    return sorted(root.glob(pattern))


def phase_environment(record: dict[str, Any], work: Path) -> None:
    """What the runtime gives us, before anything is built on it."""
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    device = torch_xla.device()
    record["runtime"] = {
        "pytorch": torch.__version__,
        "torch_xla": getattr(torch_xla, "__version__", "unknown"),
        **tpu_probe._device_report(torch, xm, xr),
        "bound_device": str(device),
    }
    # A tensor actually moving to the device is the cheapest statement that the backend is
    # live: `torch.device("xla")` alone would be a string.
    probe = (torch.ones(4) * 2).to(device)
    record["runtime"]["sum"] = float(probe.sum().cpu())


def phase_session(
    record: dict[str, Any],
    work: Path,
    *,
    device: str,
    steps: int,
    seed: int,
    build: Callable[..., Any] | None = None,
) -> None:
    """Run the real session path once, at development fidelity, on this device.

    The record entry goes in **before** the session runs, and its log is the very list the
    session appends to. ``_run_phase`` saves the record after the phase, so a session that
    dies inside a training step still lands the lines it printed - whereas writing the
    entry after the call, which is the obvious shape, leaves an exception text and nothing
    else from a run that is queued once.

    ``build`` names the config the phase trains, so the declared-shape price probe runs this
    same phase - the checkpoint host-payload assertions below are the reason it wants to.
    """
    from act.training.v0_session import run_v0_session

    config = (build or _dev_null_config)(device, steps=steps, seed=seed)
    revision = _revision()
    lines: list[str] = []
    record["session"] = {
        "config": asdict(config),
        "code_revision": revision,
        "log": lines,
    }
    print(
        f"[session] {config.arms[0]} arm, {config.total_steps} steps on {config.device}, "
        f"checkpoint every {config.checkpoint_interval}",
        flush=True,
    )
    started = time.time()
    result = run_v0_session(
        config,
        checkpoint_dir=work / "session",
        log=_session_log(lines, "session"),
        code_revision=revision,
    )
    root = work / "session"
    checkpoints = _checkpoint_files(root)
    record["session"].update(
        {
            "wall_seconds": time.time() - started,
            "arms": {
                arm.mode: _arm_reading(arm, _checkpoint_files(root, arm.mode))
                for arm in result.arms
            },
            "checkpoint_files": [
                str(path.relative_to(work)) for path in checkpoints
            ],
        }
    )
    # The revision has to be *in* the payload, not only in this record: that is the property
    # the campaign was missing (every published checkpoint had nothing dated in it), and the
    # reader here is the same `torch.load` every other reader uses.
    import torch

    payload = (
        torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
        if checkpoints
        else None
    )
    record["session"]["checkpoint_revision"] = (
        payload["training"].get("code_revision") if payload is not None else None
    )
    # And the payload has to be a *host* payload. `save_v0_checkpoint` writes one through
    # `_cpu_payload` deliberately, because `torch.save` on a lazy accelerator tensor writes
    # the tensor rather than the training - a checkpoint that can load and not be the run it
    # names. This is the only place that sees a file an XLA session wrote, so it is where
    # that property is measurable rather than argued: the *devices of the tensors in the
    # file*, read back off the disk this device just filled.
    record["session"]["checkpoint_tensor_devices"] = (
        sorted({str(value.device) for value in payload["model"].values()})
        if payload is not None
        else None
    )
    if record["session"]["checkpoint_tensor_devices"] != ["cpu"]:
        raise RuntimeError(
            "the checkpoint holds tensors on "
            f"{record['session']['checkpoint_tensor_devices']} rather than the host: "
            "a payload like that is not the training it was written from"
        )
    if record["session"]["checkpoint_revision"] != revision:
        raise RuntimeError(
            "the checkpoint does not carry the revision this run recorded "
            f"({record['session']['checkpoint_revision']!r} != {revision!r})"
        )


def phase_resume(
    record: dict[str, Any],
    work: Path,
    *,
    device: str,
    steps: int,
    seed: int,
    build: Callable[..., Any] | None = None,
) -> None:
    """Run it again from the checkpoint: the resume contract, on this device."""
    from act.training.v0_session import run_v0_session

    config = (build or _dev_null_config)(device, steps=steps, seed=seed)
    lines: list[str] = []
    record["resume"] = {"log": lines}
    started = time.time()
    result = run_v0_session(
        config,
        checkpoint_dir=work / "session",
        log=_session_log(lines, "resume"),
        code_revision=_revision(),
    )
    arm = result.arms[0]
    first = record["session"]["arms"][arm.mode]
    record["resume"].update(
        {
            "wall_seconds": time.time() - started,
            "resumed_from_a_checkpoint": any("resumed from" in line for line in lines),
            "validation_loss": arm.validation_loss,
            "same_loss_as_the_first_run": arm.validation_loss == first["validation_loss"],
            "checkpoint_files": [
                str(path.relative_to(work))
                for path in _checkpoint_files(work / "session")
            ],
        }
    )


def null_clause_defect(
    summary: dict[str, Any], manifest: dict[str, Any], revision: str
) -> str | None:
    """What section 5.14's null clause would refuse about a published seed, or ``None``.

    Recomputed from the artifacts rather than asserted from this record: ``search_flops``
    and ``mutation_count`` are exactly what ``v0_verdict._null_block`` refuses a seed for,
    so a rehearsal that published a mutating "null" arm would be reporting a queue the
    verdict cannot read - and a rehearsal whose own guards ran on its own variables would
    be reporting nothing. Naming the defect rather than the fact of one is for the same
    reason the phases are contained: the run is queued once, and the artifact has to say
    what was wrong with it.
    """
    if summary.get("status") != "complete":
        return (
            f"the published summary says {summary.get('status')!r}, not 'complete'"
        )
    null_arm = summary.get("arms", {}).get("null")
    if null_arm is None:
        return "the published summary carries no arms.null block, which the clause reads"
    if null_arm["search_flops"] != 0 or null_arm["mutation_count"] != 0:
        return (
            "the published null arm books search/mutations, which section 5.14's clause "
            f"refuses: search_flops={null_arm['search_flops']} "
            f"mutation_count={null_arm['mutation_count']}"
        )
    if manifest.get("code_revision") != revision:
        return (
            "the published manifest does not carry the revision this run recorded: "
            f"{manifest.get('code_revision')!r} != {revision!r}"
        )
    return None


def phase_handoff(
    record: dict[str, Any],
    work: Path,
    *,
    device: str,
    steps: int,
    seed: int,
    build: Callable[..., Any] | None = None,
) -> None:
    """Publish the seed the way a session does, and read back what the verdict reads.

    The null clause pairs a *published* record against the primary's adaptive loss, so the
    question this phase answers is not "did training run" but "does the artifact the verdict
    reads come out of a TPU-written checkpoint with the shape that clause requires":
    ``arms.null.search_flops == 0`` and ``mutation_count == 0``. A path that trains and
    cannot publish is not a candidate for the queue, which is why this is a phase and not a
    footnote.

    The session is re-run rather than carried over from the phase above: it resumes from the
    checkpoint the earlier phases left, so the loop does not execute again - which is itself
    a reading (a no-op resume on this device).
    """
    from act.training.v0_handoff import load_seed_summary, publish_seed, seed_directory
    from act.training.v0_session import run_v0_session

    revision = _revision()
    config = (build or _dev_null_config)(device, steps=steps, seed=seed)
    lines: list[str] = []
    record["handoff"] = {"log": lines}
    result = run_v0_session(
        config,
        checkpoint_dir=work / "session",
        log=_session_log(lines, "handoff"),
        code_revision=revision,
    )
    dataset_root = work / "dataset"
    publish_seed(
        dataset_root,
        result,
        session_dir=work / "session",
        code_revision=revision,
    )
    summary = load_seed_summary(dataset_root, seed)
    layout = seed_directory(dataset_root, seed)
    manifest = json.loads(
        (layout / "manifests" / "manifest.json").read_text(encoding="utf-8")
    )
    null_arm = summary.get("arms", {}).get("null") or {}
    record["handoff"].update(
        {
            "dataset_root": str(dataset_root),
            "summary_status": summary.get("status"),
            "manifest_code_revision": manifest.get("code_revision"),
            "null_arm": {
                "search_flops": null_arm.get("search_flops"),
                "mutation_count": null_arm.get("mutation_count"),
                "final_loss": null_arm.get("final_loss"),
                "validation_loss": null_arm.get("validation_loss"),
            },
            "seed_files": sorted(
                str(path.relative_to(dataset_root))
                for path in layout.rglob("*")
                if path.is_file()
            ),
        }
    )
    defect = null_clause_defect(summary, manifest, revision)
    if defect is not None:
        raise RuntimeError(defect)


def run_phases(
    record: dict[str, Any],
    output: Path,
    phases: dict[str, Callable[[], None]],
    *,
    deadlines: Mapping[str, float] | None = None,
) -> None:
    """Run the phases under a watch and a deadline each, saving as they land.

    Extracted so a *second* artifact can use the same containment without a second copy of
    it: a watch or a leash that only one of two kernels has is the defect this mechanism was
    written for, one fingerprint further along (2026-09-28, v1's three hours of silence).
    """
    record["watch"] = {}
    for name in record["phase_order"]:
        print(f"--- phase {name}", flush=True)
        started = time.time()
        deadline = phase_deadline_seconds(name, deadlines)
        print(f"    {name}: deadline {deadline:.0f} s", flush=True)
        watch = _Watch(
            name,
            seconds=watch_seconds(),
            journal=Path(f"{output}.watch.json"),
        ).start()
        # A phase stuck inside an XLA call cannot be interrupted from Python, so the deadline
        # does not cancel the work - it dumps every thread's stack into the log and stops the
        # process. That *is* the reading this exists for: the stack names the call that hung.
        faulthandler.dump_traceback_later(deadline, exit=True)
        try:
            tpu_probe._run_phase(record, name, phases[name], output)
        finally:
            faulthandler.cancel_dump_traceback_later()
            watch.stop()
        # Folded in and saved again, because `_run_phase` saved the record before this. The
        # sidecar above covers the case where the process never reaches this line.
        record["watch"][name] = watch.samples
        tpu_probe._save(record, output)
        print(f"    {name}: {'ok' if name in record['completed'] else 'failed'}"
              f" ({time.time() - started:.1f} s)", flush=True)


def base_record(args: argparse.Namespace) -> dict[str, Any]:
    """The record's labels, before any phase has run.

    A function rather than a literal inside ``main`` so the fields a reader has to trust -
    that this is a rehearsal, that nothing here is a declared-scale result, and that the
    probe's per-core pattern is *not* what it measures - can be pinned without a device.
    They are what keeps a partial rehearsal from being read as a port study.
    """
    return {
        "record_type": "session-rehearsal",
        "kind": "tpu-session-rehearsal",
        "rehearsal": True,
        "fidelity": "development",
        "declared_scale": False,
        "device_requested": args.device,
        "steps": args.steps,
        "seed": args.seed,
        # Whether this run took the field that the declared config sets and the development
        # one does not. A reader comparing two rehearsals has to be able to tell whether the
        # checkpointed path was in the one they are reading - see `_dev_null_config`.
        "activation_checkpointing": args.activation_checkpointing,
        # What the phase loop walks, in the record rather than in the caller: `run_phases`
        # reads it from here, so a second artifact's phases and this one's cannot be
        # interleaved by a caller that passed them in a different order.
        "phase_order": list(PHASES),
        "completed": [],
        "errors": [],
        # What a phase that hangs was given, and how often it was sampled: a reading
        # through a deadline is a different reading from one a phase finished under, so the
        # leash the run was on is part of the record rather than a property of the runner.
        "phase_deadlines": {
            name: phase_deadline_seconds(name) for name in PHASES
        },
        "watch_seconds": watch_seconds(),
        "not_measured": list(NOT_MEASURED),
        "what_this_is": (
            "A rehearsal of the registered V0 session path on the device named in "
            "`device_requested`, at development fidelity. It reports whether the path runs "
            "and what it booked - it is not a Gate-4 result, not a port, and not evidence "
            "about the hypothesis. No queue moves on this record."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="xla", help="xla (a TPU host) or cpu (a dry run)")
    parser.add_argument("--steps", type=int, default=20, help="training steps per arm")
    parser.add_argument("--seed", type=int, default=51, help="dev-fidelity session seed")
    parser.add_argument(
        "--checkpoint-activations",
        action="store_true",
        # Named for the config field it sets rather than for the option string argparse would
        # derive (`checkpoint_activations`): the record and the phase configs both read this
        # name, and one flag spelled two ways is how a field gets turned on in one place and
        # read as off in another.
        dest="activation_checkpointing",
        help="take the declared config's activation-checkpointing field too; see "
        "_dev_null_config for why a rehearsal of the path wants it",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    record = base_record(args)
    work = Path(args.output).parent / f"tpu-session-work-{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)

    def build(device: str, *, steps: int, seed: int):
        return _dev_null_config(
            device,
            steps=steps,
            seed=seed,
            activation_checkpointing=args.activation_checkpointing,
        )

    phases: dict[str, Callable[[], None]] = {
        "environment": lambda: phase_environment(record, work),
        "session": lambda: phase_session(
            record, work, device=args.device, steps=args.steps, seed=args.seed, build=build
        ),
        "resume": lambda: phase_resume(
            record, work, device=args.device, steps=args.steps, seed=args.seed, build=build
        ),
        "handoff": lambda: phase_handoff(
            record, work, device=args.device, steps=args.steps, seed=args.seed, build=build
        ),
    }
    run_phases(record, Path(args.output), phases)

    print(json.dumps(record, indent=1, sort_keys=True, default=str), flush=True)
    # The exit status is a signal, not the result: the record is the result, and it was
    # written as each phase landed - so a nonzero exit here still leaves a fetchable record
    # (the fetcher downloads the output *and* the log on both branches).
    return 0 if len(record["completed"]) == len(PHASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
