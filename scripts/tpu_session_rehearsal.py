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

**What this is not.** It is not a Gate-4 result, not a port, and not evidence about the
hypothesis: it trains at development fidelity on synthetic tokens and reports whether the
*path* runs on this device and what it booked. It does not move any queue, and it does not
touch the registered block. Per-core parallelism (eight seeds at once, one per chip) is
deliberately **not** measured here - the probe measured that pattern for the step loop, and
one session on one device is the new question. ``not_measured`` says so in the record.

    PYTHONPATH=src python scripts/tpu_session_rehearsal.py --device cpu --steps 4   # dry run
    PYTHONPATH=src python scripts/tpu_session_rehearsal.py --device xla            # on a TPU
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

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


def _revision() -> str:
    """The revision this code came from, by the same reader the published records use.

    Imported rather than re-read: ``act.training.cli._git_revision`` already falls back to the
    mounted code dataset's ``revision.txt``, and a rehearsal that reported a revision the
    sessions would not report would be measuring a different question.
    """
    from act.training.cli import _git_revision

    return _git_revision()


def _dev_null_config(device: str, *, steps: int, seed: int):
    """The dev shape with one null arm: the queue section 5.14's option would move.

    The dev shape (``declared=False``) rather than the declared 285M config, because this is
    a rehearsal of the *path*: the probe already measured the declared shape's throughput, and
    a portfolio of fresh XLA failures is cheaper to find at 20 steps than at 6,000.
    """
    from act.training.v0_session import V0SessionConfig

    return V0SessionConfig(
        seed=seed,
        arms=("null",),
        device=device,
        total_steps=steps,
        checkpoint_interval=max(1, steps // 2),
        max_checkpoints=2,
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
    record: dict[str, Any], work: Path, *, device: str, steps: int, seed: int
) -> None:
    """Run the real session path once, at development fidelity, on this device.

    The record entry goes in **before** the session runs, and its log is the very list the
    session appends to. ``_run_phase`` saves the record after the phase, so a session that
    dies inside a training step still lands the lines it printed - whereas writing the
    entry after the call, which is the obvious shape, leaves an exception text and nothing
    else from a run that is queued once.
    """
    from act.training.v0_session import run_v0_session

    config = _dev_null_config(device, steps=steps, seed=seed)
    revision = _revision()
    lines: list[str] = []
    record["session"] = {
        "config": asdict(config),
        "code_revision": revision,
        "log": lines,
    }
    started = time.time()
    result = run_v0_session(
        config,
        checkpoint_dir=work / "session",
        log=lines.append,
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
    record: dict[str, Any], work: Path, *, device: str, steps: int, seed: int
) -> None:
    """Run it again from the checkpoint: the resume contract, on this device."""
    from act.training.v0_session import run_v0_session

    config = _dev_null_config(device, steps=steps, seed=seed)
    lines: list[str] = []
    record["resume"] = {"log": lines}
    started = time.time()
    result = run_v0_session(
        config,
        checkpoint_dir=work / "session",
        log=lines.append,
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
    record: dict[str, Any], work: Path, *, device: str, steps: int, seed: int
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
    config = _dev_null_config(device, steps=steps, seed=seed)
    lines: list[str] = []
    record["handoff"] = {"log": lines}
    result = run_v0_session(
        config,
        checkpoint_dir=work / "session",
        log=lines.append,
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
        "completed": [],
        "errors": [],
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
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    record = base_record(args)
    work = Path(args.output).parent / f"tpu-session-work-{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)

    phases: dict[str, Callable[[], None]] = {
        "environment": lambda: phase_environment(record, work),
        "session": lambda: phase_session(
            record, work, device=args.device, steps=args.steps, seed=args.seed
        ),
        "resume": lambda: phase_resume(
            record, work, device=args.device, steps=args.steps, seed=args.seed
        ),
        "handoff": lambda: phase_handoff(
            record, work, device=args.device, steps=args.steps, seed=args.seed
        ),
    }
    for name in PHASES:
        print(f"--- phase {name}", flush=True)
        started = time.time()
        tpu_probe._run_phase(record, name, phases[name], Path(args.output))
        print(f"    {name}: {'ok' if name in record['completed'] else 'failed'}"
              f" ({time.time() - started:.1f} s)", flush=True)

    print(json.dumps(record, indent=1, sort_keys=True, default=str), flush=True)
    # The exit status is a signal, not the result: the record is the result, and it was
    # written as each phase landed - so a nonzero exit here still leaves a fetchable record
    # (the fetcher downloads the output *and* the log on both branches).
    return 0 if len(record["completed"]) == len(PHASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
