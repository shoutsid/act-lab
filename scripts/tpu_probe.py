"""Measure whether the declared V0 session can use a TPU efficiently, before porting it.

The Gate-4 campaign is GPU-quota-gated: three queues need ~52 GPU-hours of training
against a 30 GPU-hour week, so ~2 weeks of wall clock, and nothing advances faster than
that. A TPU is a *separate* quota (Kaggle's v3-8 is 20 TPU-hours/week beside the 30
GPU-hours) and it is free, so the question is not whether TPU time exists - it does.
The question is whether this workload can use it, and the honest way to answer that is to
measure rather than assume: the session code is a hand-written MoE with no XLA anywhere in
it, and the two things most likely to sink a TPU port are (1) the throughput per core,
which decides whether 8 cores beat the two T4s, and (2) the cost of a *rebuild*, because a
mutation event and every probed candidate construct a new model - a new XLA graph - so a
port pays a recompile each time.

Four measurements, all at the declared 285M shape:

1. **Single-core throughput**, fp32 and bf16, with the first step's compile cost separated
   from the steady-state steps. This is the number that matters - a T4 pair runs the
   declared pair at a measured 5,083 tokens/s (2026-09-20), so a core has to clear
   ~640 tokens/s for 8 of them to be worth the port, and the interesting question is by
   how much.
2. **Rebuild cost** - the same model built from a mutated genome and compiled again. If a
   rebuild costs minutes, the bounded probe path (4 candidates x 3 events per adaptive
   arm) is affordable; if it costs tens of minutes, it is not.
3. **Per-core memory headroom** - the declared model plus AdamW state has to fit in one
   core's HBM, and 8 independent seeds is only possible if it does.
4. **8 independent seeds in parallel** (one per core, no collectives), which is the pattern
   the campaign actually wants: the arms and seeds are independent, so the TPU can be used
   embarrassingly parallel instead of data-parallel, which avoids collectives entirely.

This is a probe, not a port. It does not train a session, does not claim a TPU result, and
does not compare against the registered numbers - it reports what the hardware does with
this model's shape, so the decision to port (or not) rests on measurements. The record is
written to ``/kaggle/working/tpu-probe.json`` and printed.

What a verdict *buys* is declared beside it (``week_capacity``), because the two answer
different questions from one measurement and only the second one is about the campaign's
clock. The rule's bar moves the port go/no-go; the GPU week moves at
``ONE_GPU_WEEK_MULTIPLE``, where the TPU's separate 20-hour week carries two of the three
queues. A reading between them is a real outcome - a port worth making that still leaves the
block two weeks - so it is stated with the rule rather than discovered afterwards.

Three properties of the run are deliberate, and all three were paid for. **The measurements
run in the order a failure would cost the decision** (``MEASUREMENT_ORDER``): bf16 is what
``evaluate`` reads and what the session trains in, so it goes first, then the eight-worker
aggregate, and the fp32 diagnostic last. **Each measurement is contained**
(``_run_phase``): one that cannot run is recorded as an error, stays out of ``completed``,
and does not stop the measurements after it. A probe is queued for hours and runs once, so
"the record holds whatever the hardware gave up before it broke" is the requirement, not a
nicety. **The eight-worker aggregate runs in its own process** (``seeds_in_child``), because
torch_xla refuses ``xmp.spawn`` once the XLA runtime is initialized -
``torch_xla._internal.pjrt.run_multiprocess`` raises ``RuntimeError('Runtime is already
initialized. Do not use the XLA device before calling xmp.spawn.')`` when
``_XLAC._xla_runtime_is_initialized()`` is true - and every other measurement here, plus
``_device_report`` before them, calls ``xm.xla_device()``.

The costs of ignoring them are committed history, one run each: the 2026-09-21 run raised
148 s in and wrote no record at all (6.5 h of queue, no number). The 2026-09-22 run saved
its first measurement, then died inside **fp32's** rebuild - a dtype the decision never
reads - losing bf16, the aggregate, and its own exit status. The *next* run (kernel v4,
fetched 2026-09-25) completed four of five phases and lost the aggregate to that runtime
check: bf16, both rebuilds and fp32 all measured, ``independent_seeds`` raised, and with it
the only measurement the port headline is read from. The phase that decides is the phase
that needs a process nothing has touched yet.

    PJRT_DEVICE=TPU python scripts/tpu_probe.py --steps 12 --workers 8

``--seeds-only`` is that untouched process, run by the parent rather than by hand: it takes
the aggregate alone and writes a record carrying just that block. It is not a mode to
invoke on a fresh machine, because what makes it work is that *nothing before it ran*.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent


def _resolve_sources() -> Path:
    """Make ``act`` importable, locally or inside a Kaggle kernel.

    A script kernel receives only its own code file, so the repository's ``src/`` arrives
    as the mounted code dataset rather than beside the script. Kaggle has attached that
    dataset at three different paths over time (``/kaggle/input/<slug>``,
    ``/kaggle/input/<owner>/<slug>``, and now ``/kaggle/input/datasets/<owner>/<slug>``),
    so the mount is found rather than assumed.
    """
    if (ROOT / "src" / "act" / "__init__.py").is_file():
        sys.path.insert(0, str(ROOT / "src"))
        return ROOT
    mounts = sorted(Path("/kaggle/input").glob("**/src/act/__init__.py"))
    if not mounts:
        raise SystemExit(
            "no act package: neither beside this script nor under /kaggle/input"
        )
    source = mounts[0].parent.parent.parent
    sys.path.insert(0, str(source / "src"))
    return source


_resolve_sources()

OUTPUT = Path(os.environ.get("TPU_PROBE_OUTPUT", "/kaggle/working/tpu-probe.json"))

#: The rate the declared pair trains at on the current host (two T4s, fp16, one arm per
#: device), from the launch artifact's own derived schedule. Quoted so the probe's output
#: can be read against the thing it would have to beat without re-reading the runbook.
T4_PAIR_TOKENS_PER_SECOND = 5_083.4

#: The pre-registered decision rule, declared *before* the probe was pushed so the go/no-go
#: cannot be fitted to whatever the hardware returned - the same discipline the study
#: family applies to every arm. Two conditions, both required:
#:
#: 1. **Throughput.** The 8 independent seeds must run at least `PORT_THRESHOLD_MULTIPLE`
#:    times the T4 pair aggregate. A port is only worth its cost if the TPU is not merely
#:    comparable but decisively faster per week, since it also carries a bf16 protocol
#:    amendment and a re-verified resume contract. At 2x, the three-queue campaign's ~2
#:    weeks of GPU wall clock becomes ~1 week of TPU work *beside* the GPU campaign rather
#:    than instead of it.
#: 2. **Recompiles.** A mutation event rebuilds the model and each probed candidate is a
#:    separate graph, so each is a recompile. The overhead is computed per core (seeds
#:    spread across cores), against the per-core training time of the primary queue, and
#:    must stay under `RECOMPILE_OVERHEAD_LIMIT` of it - above that the port is trading
#:    training for compilation and the headline throughput is not the rate a
#:    session would actually see.
PORT_THRESHOLD_MULTIPLE = 2.0
RECOMPILE_OVERHEAD_LIMIT = 0.10

#: The order the measurements run in, decided by what a *failure* would cost the decision
#: rather than by the dtypes' names. ``evaluate`` reads ``bf16`` (the dtype the session
#: trains in, fp16 being its nearest neighbour) and the eight-worker aggregate; fp32 is a
#: diagnostic that decides nothing. On 2026-09-22 the probe ran fp32 first, died inside its
#: rebuild, and its one queued run produced no bf16 measurement and no aggregate - so the
#: decision was INCONCLUSIVE on a record that had never reached the shape it needed.
MEASUREMENT_ORDER = ("bf16", "fp32")

#: The declared primary queue, for the overhead arithmetic: 12 seeds x 2 arms x 24,576,000
#: tokens, with 3 mutation events per arm and a candidate cap of 4, so each arm pays
#: 3 + 3x4 = 15 rebuilds.
DECLARED_SEEDS_PER_QUEUE = 12
DECLARED_ARMS_PER_SEED = 2
DECLARED_TOKENS_PER_ARM = 24_576_000
MUTATION_EVENTS_PER_ARM = 3
PROBED_CANDIDATES_PER_EVENT = 4
REBUILDS_PER_ARM = MUTATION_EVENTS_PER_ARM * (1 + PROBED_CANDIDATES_PER_EVENT)

#: The campaign's own bill, for the port's *schedule* reading. ``evaluate`` answers whether
#: the TPU is worth porting to; this answers the question an operator then has to decide
#: with - how much of the campaign its separate week carries, and whether the block's GPU
#: side still needs two weeks. Both readings come from the same measurement, which is why
#: they are declared together rather than in a document after the numbers arrive.
#:
#: The two clocks differ, and that is the whole reason a separate quota matters: a queue is
#: ``DECLARED_SEEDS_PER_QUEUE`` pairs at :data:`PAIR_HOURS_ON_T4` each, and the same queue's
#: tokens over the TPU's eight-core aggregate. Kaggle's TPU line is 20.00 h/week beside the
#: GPU's 30.00 h/week and is billed per hour a v3-8 is *running* - read 2026-09-22, when the
#: week's TPU usage stood at 0.10 h after two probe runs of 148 s and 180 s with 4.3 h
#: queued between them billed not at all.
TPU_QUOTA_HOURS = 20.0
GPU_QUOTA_HOURS = 30.0
QUEUES_PER_CAMPAIGN = 3
#: Section 5.2's per-pair event/rebuild overhead. Pinned against
#: ``act.training.session_plan.PAIR_OVERHEAD_HOURS`` by a test, so the two cannot drift.
PAIR_OVERHEAD_HOURS = 0.1
#: One pair's GPU wall time at the measured rate, and a queue's twelve of them.
PAIR_HOURS_ON_T4 = (
    DECLARED_TOKENS_PER_ARM / T4_PAIR_TOKENS_PER_SECOND / 3600.0 + PAIR_OVERHEAD_HOURS
)
GPU_HOURS_PER_QUEUE = DECLARED_SEEDS_PER_QUEUE * PAIR_HOURS_ON_T4


def queue_tpu_hours(multiple: float, *, arms: int = DECLARED_ARMS_PER_SEED) -> float:
    """One queue's wall hours on eight TPU cores at a measured multiple of a T4 pair.

    Total tokens over the eight-core aggregate, so this is the wall time the queue occupies
    a v3-8 - the unit the quota line bills in, and the reason it can be compared with the
    GPU's own hours at all.
    """
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    tokens = DECLARED_SEEDS_PER_QUEUE * arms * DECLARED_TOKENS_PER_ARM
    return tokens / (T4_PAIR_TOKENS_PER_SECOND * multiple) / 3600.0


#: The two cheapest of the campaign's three queues: the null queue (one arm per seed) and
#: one pair queue (two).
_TWO_QUEUES_ARMS = (1, DECLARED_ARMS_PER_SEED)

#: The status journal's registered amendment Option 1: two seeds per null pair, one arm each.
#: It is the only one-arm queue, so this halves its GPU hours (6 pairs instead of 12 at the
#: same pair time) without touching a seed's tokens - and it is what makes the one-week block
#: reachable with a *single* queue on the TPU. See :func:`week_capacity`.
NULL_SEEDS_PER_PAIR_IF_AMENDED = 2


def _multiple_that_fits(arms: tuple[int, ...]) -> float:
    """The multiple at which this set of queues fits the TPU's week."""
    tokens = sum(
        DECLARED_SEEDS_PER_QUEUE * arm_count * DECLARED_TOKENS_PER_ARM
        for arm_count in arms
    )
    return tokens / (T4_PAIR_TOKENS_PER_SECOND * TPU_QUOTA_HOURS * 3600.0)


#: The multiple at which the TPU's week carries **two** of the campaign's three queues, which
#: is what it takes for the block's GPU side to fit a single 30-hour week. Below it the port
#: buys a two-week block however clean the verdict: at the rule's own 2.0x pass mark the week
#: carries one pair queue in 16.1 h and the GPU side is still 34.6 h.
ONE_GPU_WEEK_MULTIPLE = _multiple_that_fits(_TWO_QUEUES_ARMS)


def week_capacity(multiple: float) -> dict[str, Any]:
    """The schedule reading of a verdict: what the TPU's week carries, and what is left.

    Declared with the rule rather than derived after the result, because the number a port
    is *for* is the campaign's wall clock and that number does not move where the rule's bar
    does. A reading between the two is a real outcome - a port worth making that still
    leaves the GPU side two weeks - not a rounding of the bar. ``gpu_hours_left`` is a
    floor: a session also pays boot, restore and its exit publish.

    Two routes reach one week and they compose, which is why both are reported: two queues
    on the TPU at ``ONE_GPU_WEEK_MULTIPLE``, or **one**    queue on the TPU plus the recorded
    null-queue amendment (two seeds per pair, halving that queue's GPU hours), which lands at
    the rule's own bar instead. The second is reported as
    ``gpu_hours_left_with_one_queue_and_the_amendment`` and needs no new measurement - only
    the registration the journal already prices.
    """
    pair = queue_tpu_hours(multiple)
    null = queue_tpu_hours(multiple, arms=1)
    # Inclusive at the boundary, for the same reason the rule's own two conditions are: the
    # pivot is *defined* as the multiple where these two queues fit, so a float comparison
    # cannot be left to land on the wrong side of its own definition.
    carry_two = pair + null <= TPU_QUOTA_HOURS + 1e-9
    gpu_left = (QUEUES_PER_CAMPAIGN - (2 if carry_two else 1)) * GPU_HOURS_PER_QUEUE
    amended_null = GPU_HOURS_PER_QUEUE / NULL_SEEDS_PER_PAIR_IF_AMENDED
    one_queue_amended = GPU_HOURS_PER_QUEUE + amended_null
    return {
        "multiple": round(multiple, 3),
        "pair_queue_hours": round(pair, 2),
        "null_queue_hours": round(null, 2),
        "null_and_pair_queue_hours": round(pair + null, 2),
        "carries_two_queues": carry_two,
        "one_gpu_week_multiple": round(ONE_GPU_WEEK_MULTIPLE, 3),
        "gpu_hours_left": round(gpu_left, 2),
        "gpu_weeks_left": math.ceil(gpu_left / GPU_QUOTA_HOURS),
        "gpu_null_queue_hours_if_amended": round(amended_null, 2),
        "gpu_hours_left_with_one_queue_and_the_amendment": round(one_queue_amended, 2),
        "one_week_with_one_queue_and_the_amendment": (
            pair <= TPU_QUOTA_HOURS + 1e-9 and one_queue_amended <= GPU_QUOTA_HOURS
        ),
    }


def evaluate(record: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered rule to a probe record, and say what it decides.

    Returns the verdict, both measured quantities and both thresholds, so the report of the
    decision is the arithmetic rather than a summary of it. A record missing either
    measurement is `INCONCLUSIVE` - never a pass, because the fields a port depends on are
    exactly the ones whose absence would otherwise read as "no problem found".
    """
    parallel = record.get("independent_seeds") or {}
    aggregate = parallel.get("aggregate_tokens_per_second")
    workers = parallel.get("workers") or 0
    bf16 = record.get("bf16") or {}
    rebuild = bf16.get("rebuild") or {}
    first = rebuild.get("first_step_seconds_including_compile")
    steady = rebuild.get("steady_step_seconds_mean")

    result: dict[str, Any] = {
        "thresholds": {
            "aggregate_tokens_per_second": PORT_THRESHOLD_MULTIPLE
            * T4_PAIR_TOKENS_PER_SECOND,
            "recompile_overhead_fraction": RECOMPILE_OVERHEAD_LIMIT,
        },
        "measured": {},
        "capacity": None,
        "verdict": "INCONCLUSIVE",
        "why": "",
    }
    missing = [
        name
        for name, value in (
            ("independent_seeds.aggregate_tokens_per_second", aggregate),
            ("independent_seeds.workers", workers),
            ("bf16.rebuild.first_step_seconds_including_compile", first),
            ("bf16.rebuild.steady_step_seconds_mean", steady),
        )
        if value in (None, 0)
    ]
    if missing:
        result["why"] = f"the record is missing {', '.join(missing)}"
        return result

    per_core = aggregate / workers
    # The primary queue's tokens per core, and the training seconds that buys at the rate
    # one core sustains - the recompiles are paid per core, not in aggregate.
    queue_tokens = DECLARED_SEEDS_PER_QUEUE * DECLARED_ARMS_PER_SEED * DECLARED_TOKENS_PER_ARM
    per_core_tokens = queue_tokens / workers
    training_seconds = per_core_tokens / per_core
    compile_seconds = max(0.0, first - steady)
    rebuilds_per_core = (DECLARED_SEEDS_PER_QUEUE / workers) * DECLARED_ARMS_PER_SEED * REBUILDS_PER_ARM
    overhead = rebuilds_per_core * compile_seconds / training_seconds

    # What this verdict *buys* the campaign, from the same measurement: the rule's bar moves
    # the port go/no-go, and the campaign's GPU wall clock moves at ONE_GPU_WEEK_MULTIPLE.
    result["capacity"] = week_capacity(aggregate / T4_PAIR_TOKENS_PER_SECOND)

    throughput_ok = aggregate >= result["thresholds"]["aggregate_tokens_per_second"]
    overhead_ok = overhead <= RECOMPILE_OVERHEAD_LIMIT
    result["measured"] = {
        "aggregate_tokens_per_second": aggregate,
        "per_core_tokens_per_second": round(per_core, 1),
        "multiple_of_t4_pair": round(aggregate / T4_PAIR_TOKENS_PER_SECOND, 2),
        "rebuild_compile_seconds": round(compile_seconds, 2),
        "rebuilds_per_core": round(rebuilds_per_core, 1),
        "per_core_training_hours": round(training_seconds / 3600, 2),
        "recompile_overhead_fraction": round(overhead, 4),
    }
    result["throughput_ok"] = throughput_ok
    result["recompile_overhead_ok"] = overhead_ok
    if throughput_ok and overhead_ok:
        result["verdict"] = "PORT"
        result["why"] = (
            f"{aggregate:,.0f} tokens/s is {aggregate / T4_PAIR_TOKENS_PER_SECOND:.2f}x the "
            f"T4 pair, and recompiles cost {overhead:.1%} of the per-core training time"
        )
    elif throughput_ok:
        # Recompiles are the *only* blocker, and they are a property of the session path:
        # a mutation event and a probed candidate each rebuild the model. A fixed-graph arm
        # - the architecture-zoo sweeps, which train one sampled architecture to a horizon
        # - has no mid-run rebuild, so the accelerator's throughput is what it would get.
        # This branch is declared here, before the numbers, so it cannot be discovered
        # afterwards as a consolation.
        result["verdict"] = "PORT_FIXED_GRAPH_ONLY"
        result["why"] = (
            f"throughput clears the bar ({aggregate / T4_PAIR_TOKENS_PER_SECOND:.2f}x the "
            f"T4 pair) but recompiles would cost {overhead:.1%} of the per-core training "
            f"time: the session path is out, fixed-graph arms are not"
        )
    else:
        result["verdict"] = "DO-NOT-PORT"
        failed = []
        if not throughput_ok:
            failed.append(
                f"throughput is {aggregate / T4_PAIR_TOKENS_PER_SECOND:.2f}x the T4 pair, "
                f"below the declared {PORT_THRESHOLD_MULTIPLE:.1f}x"
            )
        if not overhead_ok:
            failed.append(
                f"recompiles would cost {overhead:.1%} of the per-core training time, "
                f"above the declared {RECOMPILE_OVERHEAD_LIMIT:.0%}"
            )
        result["why"] = "; ".join(failed)
    return result


def _save(record: dict[str, Any], path: Path) -> None:
    """Write the record as it stands, rather than only once every measurement is in.

    The probe's first run measured nothing: it raised 148 s in, and the record was written
    after the last measurement, so 6.5 h of Kaggle queue produced no number at all. Every
    completed measurement now lands on disk, so a later failure still leaves what the
    hardware had already given up - and the record carries a ``completed`` list, because
    a partial record must not read like a finished one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def phase_names(dtypes: tuple[str, ...], *, workers: int) -> tuple[str, ...]:
    """The measurement order, decided by what a failure would *cost the decision*.

    ``evaluate`` reads ``bf16`` and the eight-worker aggregate, so those run first and
    fp32's diagnostic runs last: a phase that cannot run then loses the reading nobody
    decides on. A pure function so the property is testable without a TPU - the order is
    the sort of thing a refactor flips silently, and on 2026-09-22 it cost the run.
    """
    names: list[str] = []
    for index, dtype in enumerate(dtypes):
        names.append(f"{dtype}.throughput")
        names.append(f"{dtype}.rebuild")
        if index == 0 and workers:
            names.append("independent_seeds")
    return tuple(names)


def _binder(function: Callable[..., None], *arguments: Any) -> Callable[[], None]:
    """Bind a phase's arguments **now**, not when the loop variable has moved on.

    The phase table is built in a loop, so a closure that read ``dtype`` or ``state`` at
    call time would run every phase against the last dtype's state and write the same
    measurements under two names - a record that reads as complete and is wrong. That is
    the failure this project treats as worse than a crash, so the binding is explicit and
    `phase_table` is a function a test can watch rather than an idiom to be noticed.
    """
    return lambda: function(*arguments)


def phase_table(
    dtypes: tuple[str, ...],
    build_and_time: Callable[[str, dict[str, Any]], None],
    rebuild_of: Callable[[str, dict[str, Any]], None],
) -> tuple[dict[str, dict[str, Any]], dict[str, Callable[[], None]]]:
    """The per-dtype state and the phase name -> callable table, each properly bound.

    Returns the states as well as the callables because what makes a rebuild measurable is
    the genome its own dtype's throughput pass put there - the sharing is the point, and a
    table that gave each phase a fresh dict would silently make every rebuild an error.
    """
    states: dict[str, dict[str, Any]] = {dtype: {} for dtype in dtypes}
    work: dict[str, Callable[[], None]] = {}
    for dtype in dtypes:
        state = states[dtype]
        work[f"{dtype}.throughput"] = _binder(build_and_time, dtype, state)
        work[f"{dtype}.rebuild"] = _binder(rebuild_of, dtype, state)
    return states, work


def _run_phase(
    record: dict[str, Any], name: str, work: Callable[[], None], path: Path
) -> bool:
    """Run one measurement, keeping a failure in the record rather than ending the run.

    Returns whether it completed. A phase that raises is recorded under ``errors`` and stays
    out of ``completed``: the record is then honestly partial, and the phases after it still
    run. ``evaluate`` reads the fields it needs rather than the exit status, so a contained
    failure still decides nothing on its own - a missing measurement is INCONCLUSIVE, never
    a pass (which is the property ``_save`` exists for as well).

    Both halves of that were bought. A probe run that dies does not just lose its number:
    it ends the kernel ERROR after hours in a queue, and a reader who has only the record
    cannot tell a finished one from a partial one unless the record says so.
    """
    try:
        work()
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        record.setdefault("errors", []).append(
            {"phase": name, "error": f"{type(exc).__name__}: {exc}"}
        )
    else:
        record["completed"].append(name)
    _save(record, path)
    return name in record["completed"]


def _device_report(torch, xm, xr) -> dict[str, Any]:
    """What the runtime actually gave us: device count, type, and per-core memory."""
    devices = xm.get_xla_supported_devices()
    device = xm.xla_device()
    report: dict[str, Any] = {
        "pytorch": torch.__version__,
        "xla_devices": len(devices),
        "device_kind": str(device),
    }
    try:
        import torch_xla  # noqa: PLC0415

        report["torch_xla"] = torch_xla.__version__
    except Exception:  # pragma: no cover - version metadata only
        report["torch_xla"] = "unknown"
    try:
        info = xr.memory_info(device)
        report["memory_bytes_used"] = info.get("bytes_used")
        report["memory_bytes_limit"] = int(info.get("bytes_limit", 0)) or None
    except Exception:  # pragma: no cover - not every runtime exposes it
        report["memory_bytes_used"] = None
        report["memory_bytes_limit"] = None
    return report


def _build_declared(torch, xm, dtype: str):
    """The pre-registered 285M model and an optimizer over it, on the XLA device."""
    from act.model.factory import v0_moe_model_from_genome
    from act.training.v0_session import _genome_from_config, declared_session_config

    config = declared_session_config()
    genome = _genome_from_config(config)
    model = v0_moe_model_from_genome(genome)
    if dtype == "bf16":
        model = model.to(torch.bfloat16)
    elif dtype == "fp32":
        model = model.to(torch.float32)
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(dtype)
    device = xm.xla_device()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total = sum(p.numel() for p in model.parameters())
    return config, genome, model, optimizer, total


def _measure(torch, xm, model, optimizer, config, *, steps: int) -> dict[str, Any]:
    """Time one short step loop, with the first (compiling) step reported separately."""
    generator = torch.Generator().manual_seed(config.seed)
    batch = (config.batch_size, config.context_size)
    tokens_per_step = config.batch_size * config.context_size
    step_seconds: list[float] = []
    for step in range(steps):
        inputs = torch.randint(
            0, config.vocab_size, batch, generator=generator, dtype=torch.long
        ).to(xm.xla_device())
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(inputs)
        loss.backward()
        optimizer.step()
        xm.mark_step()
        elapsed = time.perf_counter() - start
        step_seconds.append(elapsed)
        del loss, inputs
    steady = step_seconds[1:]
    steady_mean = statistics.fmean(steady) if steady else float("nan")
    return {
        "steps": steps,
        "tokens_per_step": tokens_per_step,
        "first_step_seconds_including_compile": round(step_seconds[0], 4),
        "steady_step_seconds_mean": round(steady_mean, 6),
        "steady_step_seconds_min": round(min(steady), 6) if steady else None,
        "tokens_per_second": round(tokens_per_step / steady_mean, 1) if steady else None,
    }


def _rebuild_genome(genome) -> tuple[Any, str, str]:
    """The genome a rebuild builds from: one expert grown by the project's own operator.

    Returns the mutated copy, the new expert's id, and the layer it landed on. The copy
    matters because the caller's genome is the declared one and is reused for the other
    dtype.

    This uses ``act.topology.mutations.grow_expert`` rather than a hand-written edit. That
    is not tidiness: measured 2026-09-21, a hand-written version (append an expert *and*
    raise that layer's ``active_experts``) produced a genome that passes validation and
    still cannot be built, because ``v0_moe_config_from_genome`` requires one uniform
    active-expert count while its config holds a single scalar for every layer. The probe
    raised ``ValueError: V0 MoE reconstruction requires uniform active expert counts``
    148 s in and wrote no record at all - so it measured nothing, after 6.5 h of queue.
    Using the operator the session actually applies also makes "the shape a real event
    produces" literally true rather than approximate.
    """
    from copy import deepcopy

    from act.topology.mutations import grow_expert

    rebuilt = deepcopy(genome)
    target = rebuilt.layers[-1]
    expert_id = grow_expert(rebuilt, target.id, 0)
    if expert_id is None:  # pragma: no cover - the declared layer holds 8 of 16
        raise SystemExit(f"cannot measure a rebuild: grow_expert declined {target.id}")
    return rebuilt, expert_id, target.id


def _rebuild_cost(torch, xm, config, genome, *, dtype: str, steps: int) -> dict[str, Any]:
    """What a mutation event (or a probed candidate) costs: a new graph, compiled again.

    The candidate path builds a model from a *different* genome, so this rebuilds from a
    genome with one expert added - the shape a real event produces - rather than reusing
    the model above, whose graph XLA already has.

    The caller has already released the declared model and its optimizer (see
    ``build_and_time`` in ``main``), so this measures one rebuild rather than two sessions'
    worth of state. The coexistence that costs a *session* per-core memory is narrower than
    that: ``v0_session._adaptive_event`` holds the previous model while it rebuilds the
    winner, but only one optimizer's state is ever stepped, so pricing an event's peak is a
    previous model's parameters and grads on top of this one - which the recorded parameter
    count gives a reader directly.

    2026-09-22: this is where the run died. The fp32 pair of models plus two stepped
    optimizers needed 7.96G with 5.33G free, and the probe was holding both because it
    measured the rebuild before releasing the declared model. bf16 is wider of the limit,
    but the order is what made the loss total: see ``MEASUREMENT_ORDER``.
    """
    from act.model.factory import v0_moe_model_from_genome

    rebuilt_genome, expert_id, layer_id = _rebuild_genome(genome)
    model = v0_moe_model_from_genome(rebuilt_genome)
    if dtype == "bf16":
        model = model.to(torch.bfloat16)
    model = model.to(xm.xla_device())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    measured = _measure(torch, xm, model, optimizer, config, steps=steps)
    measured["experts_added"] = 1
    measured["expert_id"] = expert_id
    measured["rebuilt_layer"] = layer_id
    measured["expert_count_after"] = next(
        len(layer.experts) for layer in rebuilt_genome.layers if layer.id == layer_id
    )
    del model, optimizer
    return measured


def _worker(rank: int, steps: int, dtype: str, directory: str) -> None:
    """One independent seed on one core: no collectives, no shared state.

    This is the pattern that makes a TPU useful here. The campaign's arms and seeds are
    independent, so 8 cores become 8 concurrent *seeds* rather than one data-parallel run -
    which needs no AllReduce, keeps each core's graph sequential, and multiplies the
    weekly throughput by the core count instead of by some scaling efficiency.

    ``args`` is positional because ``xmp.spawn`` takes no keyword arguments.
    """
    import torch
    import torch_xla.core.xla_model as xm

    from act.training.v0_session import declared_session_config

    results_path = Path(directory) / f"rank-{rank}.json"
    torch.manual_seed(10_000 + rank)
    config = declared_session_config(seed=15_000 + rank)
    _, _, model, optimizer, _ = _build_declared(torch, xm, dtype)
    started = time.perf_counter()
    measured = _measure(torch, xm, model, optimizer, config, steps=steps)
    measured["rank"] = rank
    measured["seconds_wall"] = round(time.perf_counter() - started, 3)
    measured["seed"] = 15_000 + rank
    results_path.write_text(
        json.dumps(measured, sort_keys=True) + "\n", encoding="utf-8"
    )


def spawn_arguments(
    workers: int, steps: int, dtype: str, directory: str
) -> dict[str, Any]:
    """The arguments for ``xmp.spawn``, as data so the property is testable.

    ``nprocs`` is **None**, not the worker count. Measured 2026-09-22: torch_xla 2.8
    refuses ``nprocs=8`` with ``ValueError: Unsupported nprocs (8). Please use nprocs=1 or
    None (default)``, and the acceptance takes ``None`` as "one process per available
    device". The declared worker count still governs the measurement - the ranks that ran
    are checked against it and it is carried in the record - but it is not what the spawn
    call is told.
    """
    return {
        "args": (steps, dtype, directory),
        "nprocs": None,
        "start_method": "fork",
        "expected_workers": workers,
    }


def seeds_child_arguments(
    *, workers: int, steps: int, dtype: str, output: Path, probe: Path | None = None
) -> list[str]:
    """The argv of the process that takes the aggregate, as data so it is testable.

    A *fresh* interpreter, and that is the whole point rather than an implementation
    detail: ``xmp.spawn`` refuses to run once the XLA runtime is initialized, and this
    probe initializes it before the aggregate's turn by design (``_device_report`` calls
    ``xm.xla_device()`` to record what the hardware gave us, and every dtype's throughput
    pass calls it too). So the one measurement the decision reads is taken by a process
    that has done nothing else - the same interpreter, the same mounted probe, and an XLA
    runtime that nothing has touched.

    ``probe`` exists so the path is a parameter: run normally, ``__file__`` is this script
    (or its mounted copy under ``/kaggle/input``, since the kernel entry runs it with
    ``runpy.run_path``); the child needs that same path, and a test needs to name one.
    """
    return [
        sys.executable,
        str(probe or Path(__file__).resolve()),
        "--seeds-only",
        "--workers",
        str(workers),
        "--steps",
        str(steps),
        "--dtype",
        dtype,
        "--output",
        str(output),
    ]


def seeds_in_child(
    *,
    workers: int,
    steps: int,
    dtype: str,
    timeout: float | None = None,
    probe: Path | None = None,
) -> dict[str, Any]:
    """Run the aggregate in a child process and return its ``independent_seeds`` block.

    The child's whole record is read and its ``independent_seeds`` block returned, so the
    parent records the same object the child measured rather than a re-derived summary.

    A child that fails raises with its own stderr attached: this phase's failure has twice
    been the run's failure, and a ``RuntimeError`` naming the child's exit code and nothing
    else would have made this session's diagnosis a guess. The child is given a timeout
    (``TPU_PROBE_SEEDS_TIMEOUT``, an hour by default) because eight compilations are slow
    and a hung child would otherwise hold the kernel until the platform killed it.
    """
    if workers < 1:
        raise ValueError(f"the aggregate needs at least one worker, not {workers}")
    budget = timeout
    if budget is None:
        budget = float(os.environ.get("TPU_PROBE_SEEDS_TIMEOUT", 3600.0))
    with tempfile.TemporaryDirectory(prefix="tpu-probe-seeds-") as temporary:
        output = Path(temporary) / "seeds.json"
        argv = seeds_child_arguments(
            workers=workers, steps=steps, dtype=dtype, output=output, probe=probe
        )
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=budget)
        except subprocess.TimeoutExpired as expired:
            raise RuntimeError(
                f"the seed-parallel child did not finish in {budget:.0f}s; raise "
                "TPU_PROBE_SEEDS_TIMEOUT if the hardware is merely slow"
            ) from expired
        if proc.returncode != 0 or not output.is_file():
            detail = (proc.stdout + proc.stderr)[-2000:]
            raise RuntimeError(
                f"the seed-parallel child exited {proc.returncode} without a record: "
                f"\n{detail}"
            )
        payload = json.loads(output.read_text(encoding="utf-8"))
    block = payload.get("independent_seeds")
    if not isinstance(block, dict):
        raise RuntimeError(
            "the seed-parallel child wrote a record with no independent_seeds block, so "
            "the aggregate the decision divides by the cores was not measured"
        )
    # Recorded beside the numbers, not in a log: a reader of the record has to be able to
    # tell an aggregate taken in the child process from one taken after the runtime was
    # already initialized - because the second is the measurement that cannot be trusted to
    # have spawned at all, and it is what three runs produced before this existed.
    return {**block, "taken_in": "child-process"}


def aggregate_dtype(dtype: str) -> str:
    """The dtype the eight-worker aggregate is taken at.

    bf16 when the run measures both, because bf16 is what ``evaluate`` reads and what the
    session trains in - the aggregate is divided into per-core *training* hours, so taking
    it at the diagnostic dtype would price a port nobody would run. One definition, used by
    both the parent's request and the child that fulfils it, because two copies of this
    choice would be two different measurements under one name.
    """
    return dtype if dtype != "both" else MEASUREMENT_ORDER[0]


def seeds_only_request(args: argparse.Namespace) -> dict[str, Any]:
    """What the child measures - and the device count it must **not** ask for.

    ``devices`` is ``None`` here on purpose and is the mechanism of the whole child
    process: the host's device count is normally taken with
    ``xm.get_xla_supported_devices()`` before spawning, which initializes the runtime
    ``xmp.spawn`` requires untouched. This session's fetched v4 record is what that costs:
    four phases measured, ``independent_seeds`` raised ``Runtime is already initialized``,
    and the one measurement the decision reads was lost. The child is checked against the
    ranks that ran instead (see ``measure_independent_seeds``).
    """
    workers = int(args.workers)
    if workers < 1:
        raise ValueError(
            f"--seeds-only measures the aggregate, so --workers must be at least 1, not {workers}"
        )
    return {
        "devices": None,
        "workers": workers,
        "steps": int(args.steps),
        "dtype": aggregate_dtype(args.dtype),
    }


def run_seeds_only(args: argparse.Namespace) -> int:
    """The child mode: take the aggregate and nothing else, in an untouched process."""
    request = seeds_only_request(args)
    import torch_xla.distributed.xla_multiprocessing as xmp

    with tempfile.TemporaryDirectory(prefix="tpu-probe-ranks-") as directory:
        block = measure_independent_seeds(xmp=xmp, directory=directory, **request)
    record: dict[str, Any] = {
        "record_type": "hardware-probe-seeds",
        "kind": "tpu-v0-session-probe",
        "what_this_is": (
            "The eight-worker aggregate alone, taken by the child process the main probe "
            "starts. Nothing here is a Gate-4 number: it is the rate at which one "
            "independent seed per core runs the declared session shape."
        ),
        "independent_seeds": block,
    }
    _save(record, args.output)
    print(json.dumps(record, indent=1, sort_keys=True))
    print(f"\nwrote {args.output}")
    return 0


def measure_independent_seeds(
    *,
    xmp: Any,
    devices: int | None,
    workers: int,
    steps: int,
    dtype: str,
    directory: str,
) -> dict[str, Any]:
    """One single-arm seed per core, and the aggregate the decision divides by the cores.

    The aggregate is the **sum of the ranks' steady rates**, not total tokens over the
    spawn's wall clock. Two reasons, both structural rather than conveniences: the decision
    compares it with the T4 pair's *steady* rate (5,083.4 tokens/s) and divides it by the
    worker count to get per-core training hours, and the wall clock carries one-off costs
    the session pays once - eight process startups and eight XLA compilations - which
    amortize over a 24.6M-token arm and are otherwise double-counted, since the rule models
    recompiles separately from the rebuild measurement. The wall-based figure is recorded
    beside it rather than dropped, so the difference between the two is visible in the
    record instead of being an assumption.

    A rank that did not finish raises: a partial parallel measurement is not the
    measurement the decision reads, and it is contained as a phase error rather than
    averaged into a plausible number.

    ``devices`` is the host's core count for a caller that already has it; **no
    production caller does**, and that is the finding this argument carries: asking for it
    needs ``xm.get_xla_supported_devices()``, which initializes the very runtime
    ``xmp.spawn`` requires untouched. So the child passes ``None`` and is checked against
    the ranks instead - with ``nprocs=None`` torch_xla derives the process count from the
    TPU's own topology, so the number of rank reports *is* the device count that mattered,
    and requiring all ``workers`` of them is the same property with no pre-spawn device
    query at all. The two checks are kept distinct because the mismatch case has a
    different fix (a wrong ``--workers``) from the missing-rank case (a core that failed).
    """
    if devices is not None and devices != workers:
        raise RuntimeError(
            f"the host exposes {devices} XLA device(s) but the measurement declares "
            f"{workers} worker(s); the aggregate is divided by the worker count, so a "
            "mismatch is not a measurement"
        )
    plan = spawn_arguments(workers, steps, dtype, directory)
    paths = [Path(directory) / f"rank-{rank}.json" for rank in range(workers)]
    started = time.perf_counter()
    xmp.spawn(
        _worker,
        args=plan["args"],
        nprocs=plan["nprocs"],
        start_method=plan["start_method"],
        join=True,
    )
    wall = time.perf_counter() - started
    per_rank = [json.loads(path.read_text()) for path in paths if path.exists()]
    if len(per_rank) != workers:
        raise RuntimeError(
            f"{len(per_rank)} of {workers} ranks wrote a report; the aggregate needs all "
            "of them"
        )
    steady = sum(float(row["tokens_per_second"]) for row in per_rank)
    tokens = sum(row["tokens_per_step"] * steps for row in per_rank)
    return {
        "workers": workers,
        "completed": len(per_rank),
        "wall_seconds": round(wall, 3),
        "aggregate_tokens_per_second": round(steady, 1),
        "wall_tokens_per_second_including_startup": round(tokens / wall, 1) if wall else None,
        "aggregate_definition": (
            "sum of the ranks' steady tokens/s (compile excluded), which is the quantity "
            "comparable to the T4 pair rate and to the per-core training hours the rule "
            "divides by the worker count"
        ),
        "per_rank": per_rank,
        "ratio_to_t4_pair": round(steady / T4_PAIR_TOKENS_PER_SECOND, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12, help="steady-state steps to time")
    parser.add_argument(
        "--workers", type=int, default=0,
        help="cores for the independent-seed measurement (0 = skip, 8 = a full v3-8)",
    )
    parser.add_argument("--dtype", choices=("fp32", "bf16", "both"), default="both")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--seeds-only", action="store_true",
        help="take the eight-worker aggregate alone and exit - the child mode the main run "
        "starts, and only correct on a machine where nothing has used an XLA device yet",
    )
    parser.add_argument(
        "--evaluate", type=Path, default=None,
        help="apply the pre-registered rule to a fetched record (needs no TPU); "
        "exits 0 PORT, 2 PORT_FIXED_GRAPH_ONLY, 1 DO-NOT-PORT, 3 INCONCLUSIVE",
    )
    args = parser.parse_args()

    if args.evaluate is not None:
        # The decision runs where the record is, not where the hardware is: this mode is
        # deliberately before the torch import so it can be applied on a laptop, by CI, or
        # by anyone re-reading the record later. The exit codes are distinct so a script can
        # tell "port" from "port only the fixed-graph work" from "neither" from "the probe
        # did not measure what the decision needs".
        decision = evaluate(json.loads(args.evaluate.read_text(encoding="utf-8")))
        print(json.dumps(decision, indent=1, sort_keys=True))
        return {
            "PORT": 0,
            "PORT_FIXED_GRAPH_ONLY": 2,
            "DO-NOT-PORT": 1,
        }.get(decision["verdict"], 3)

    if args.seeds_only:
        # Before the torch import below, and before ``_device_report``: this mode's whole
        # requirement is that no XLA device has been used in this process yet.
        return run_seeds_only(args)

    import torch
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    record: dict[str, Any] = {
        "record_type": "hardware-probe",
        "kind": "tpu-v0-session-probe",
        "what_this_is": (
            "A hardware probe of the declared 285M session shape on a TPU. No session "
            "trained and nothing here is a Gate-4 number: it reports throughput per core, "
            "the cost of a rebuild (a mutation or probed candidate), per-core memory, and "
            "the aggregate rate of one independent seed per core."
        ),
        "declared_pair_tokens_per_second_on_t4": T4_PAIR_TOKENS_PER_SECOND,
        "device": _device_report(torch, xm, xr),
        "steps": args.steps,
    }

    dtypes = tuple(dtype for dtype in MEASUREMENT_ORDER if args.dtype in ("both", dtype))
    record["completed"] = []
    record["errors"] = []
    _save(record, args.output)

    def build_and_time(dtype: str, state: dict[str, Any]) -> None:
        """The declared model at one dtype, timed, then released before any rebuild.

        The model and its optimizer are dropped here rather than after the rebuild: a
        rebuild's cost is the new graph, and holding a second fully-stepped optimizer's
        state while measuring one is not a shape any event produces. The genome survives
        (it is device-free), which is what the rebuild builds from.
        """
        config, genome, model, optimizer, total = _build_declared(torch, xm, dtype)
        state["config"], state["genome"] = config, genome
        record[dtype] = {
            "parameters": total,
            "throughput": _measure(torch, xm, model, optimizer, config, steps=args.steps),
        }
        del model, optimizer

    def rebuild_of(dtype: str, state: dict[str, Any]) -> None:
        if "genome" not in state:
            raise RuntimeError(
                f"{dtype}.throughput did not complete, so there is no genome to rebuild from"
            )
        record[dtype]["rebuild"] = _rebuild_cost(
            torch, xm, state["config"], state["genome"], dtype=dtype,
            steps=max(3, args.steps // 3),
        )

    def independent_seeds() -> None:
        """Eight cores, one independent seed each: the aggregate the decision reads first.

        Measured by a **child process**, never here: this parent has already initialized the
        XLA runtime (``_device_report`` ran before the first phase), and ``xmp.spawn``
        refuses to run against an initialized runtime. See ``seeds_in_child``.
        """
        record["independent_seeds"] = seeds_in_child(
            workers=args.workers,
            steps=args.steps,
            dtype=aggregate_dtype(args.dtype),
        )

    # The order is the decision's, not the dtypes': every dtype's own measurements, then
    # the aggregate the port headline is read from - ahead of the last dtype's diagnostic.
    # Only the callables are needed here: each already holds its own dtype's state dict,
    # which is what makes a rebuild buildable from the genome its own pass stored.
    _, work = phase_table(dtypes, build_and_time, rebuild_of)
    work["independent_seeds"] = independent_seeds
    for name in phase_names(dtypes, workers=args.workers):
        _run_phase(record, name, work[name], args.output)

    _save(record, args.output)
    print(json.dumps(record, indent=1, sort_keys=True))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
