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
nicety. **The eight-worker aggregate rides the single-process path** (``_seed_rank``): one
thread per device in the process that is already running, with an explicit ``xla:<rank>``
per rank, because that is the only path this hardware has ever been measured on - and the
process-based one has now failed three times for three different reasons.

The costs of ignoring them are committed history, one run each: the 2026-09-21 run raised
148 s in and wrote no record at all (6.5 h of queue, no number). The 2026-09-22 run saved
its first measurement, then died inside **fp32's** rebuild - a dtype the decision never
reads - losing bf16, the aggregate, and its own exit status. The phase that decides is the
one that has cost the most attempts, and each attempt died one layer further in than the
last: ``nprocs=8`` was refused by torch_xla 2.8 (the aggregate never spawned); then
``xmp.spawn`` refused the process it was called from, because
``torch_xla._internal.pjrt.run_multiprocess`` opens with ``RuntimeError('Runtime is already
initialized. Do not use the XLA device before calling xmp.spawn.')`` when
``_XLAC._xla_runtime_is_initialized()`` is true, and ``_device_report`` initializes it
before the first phase runs (kernel v4, fetched 2026-09-25). Kernel **v5** (fetched
2026-09-26) fixed exactly that by handing the aggregate to a child process that had touched
nothing; the child ran, and the failure moved one layer down into libtpu's topology -
``RuntimeError: Bad StatusOr access: UNKNOWN: TPU initialization failed: Invalid
--2a886c8_slice_builder_worker_addresses specified. Expected 8 worker addresses, got 1.``
The parent's own client initializes with ``tpu_process_addresses="local"`` (the kernel
log's E0000 line), while ``tpu.configure_topology``'s ``os.environ.setdefault`` calls handed
the spawned ranks a process grid that wanted eight worker addresses and supplied one - and
neither input to that arithmetic, the platform's topology environment nor the PCI chip
count, was in any record the probe had written. They are recorded now
(``TOPOLOGY_ENV_KEYS``, ``_tpu_report``), which is what a fourth attempt would need to be
diagnosed from the artifact rather than from a kernel log.

So the aggregate is taken where the hardware has already been measured doing this work: one
process, one thread per device, one independent seed per device, and a barrier that releases
only once every rank has compiled. Each rank's thread also binds itself to its own chip
before it builds anything (``bind_thread_device``), because a step marker is not a device
argument: ``xm.mark_step()`` is ``torch_xla.sync()`` in torch_xla 2.8, and that marks a step
on the **calling thread's** current device - which a fresh thread has never chosen, so it
falls back to the process default that every earlier phase left at ``xla:0``. Without the
binding all eight ranks would mark chip 0's steps, seven of the eight chips would compile
graphs nobody executes, and the "steady step" would be graph construction. It is the call
torch_xla's own one-process/thread-per-device launcher makes at the top of each thread
(``_run_thread_per_device``), and the ranks' reports carry both sides of it, so the parent
checks the binding rather than trusting it. The number is then the tokens the ranks pushed
inside a window in which all of them were stepping, and the record carries each rank's
device, seed, window and compile cost - so the claim "one seed per chip" is checkable from
the record rather than assumed from the spawn's behaviour. ``xmp.spawn``, its child process and the
whole process boundary (``--seeds-only``, ``seeds_in_child``) are retired with it: the
thread-per-device shape needs no untouched runtime, which is the only reason the boundary
existed.

    PJRT_DEVICE=TPU python scripts/tpu_probe.py --steps 12 --workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import json
import math
import os
import statistics
import sys
import tempfile
import threading
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


#: The environment keys recorded beside the hardware: what the *platform* told the runtime
#: about the TPU's topology. The process-based aggregate's v5 failure is what puts them here
#: - ``configure_topology`` builds its process grid and its worker-address list out of these
#: values with ``os.environ.setdefault``, so a value the platform already set wins over the
#: one torch_xla computes, and "Expected 8 worker addresses, got 1" is that disagreement.
#: Neither the values nor the PCI chip count (``_tpu_report``) was in any earlier record.
TOPOLOGY_ENV_KEYS = (
    "PJRT_DEVICE",
    "PJRT_LOCAL_PROCESS_RANK",
    "PJRT_LOCAL_PROCESS_COUNT",
    "PJRT_SELECT_DEFAULT_DEVICE",
    "TPU_ACCELERATOR_TYPE",
    "TPU_PROCESS_BOUNDS",
    "TPU_CHIPS_PER_PROCESS_BOUNDS",
    "TPU_CHIPS_PER_HOST_BOUNDS",
    "TPU_HOST_BOUNDS",
    "TPU_PROCESS_ADDRESSES",
    "TPU_PROCESS_PORT",
    "TPU_WORKER_HOSTNAMES",
    "TPU_WORKER_ID",
    "TPU_NAME",
    "TPU_VISIBLE_CHIPS",
    "TPU_SKIP_MDS_QUERY",
    "CLOUD_TPU_TASK_ID",
    "XRT_TPU_CONFIG",
    "XLA_TPU_MAX_INFLIGHT_COMPUTATIONS",
    "LIBTPU_INIT_ARGS",
    "ALLOW_MULTIPLE_LIBTPU_LOAD",
    "TPU_LIBRARY_PATH",
    "X_NUM_DEVICES",
)


def environment_report() -> dict[str, str | None]:
    """The platform's TPU topology environment, with unset keys kept as ``None``.

    ``None`` rather than an omitted key on purpose: a reader has to be able to tell "the
    platform did not set it" from "this probe did not collect it", and the difference is
    exactly what a ``setdefault`` that never fired hides.
    """
    return {key: os.environ.get(key) for key in TOPOLOGY_ENV_KEYS}


def _tpu_report() -> dict[str, Any]:
    """What the platform says this TPU *is*: chip count, and the metadata block.

    Both numbers the process-based aggregate's failure is read through, and neither was
    recorded before v5: ``tpu.num_local_processes()`` is ``min(pci chips, process bounds)``,
    so a PCI scan that finds no chip (or a bounds value the platform set) is the difference
    between eight spawned ranks and one - and ``configure_topology`` reads the metadata
    block for the accelerator type it derives the process grid from. Failures are recorded
    as their own names rather than raised: this is diagnostics for a decision, and a probe
    that cannot read the metadata should still measure the hardware.
    """
    try:
        from torch_xla._internal import tpu as tpu_internal  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - not every runtime has the module
        return {"unavailable": f"{type(exc).__name__}: {exc}"}
    report: dict[str, Any] = {}
    try:
        report["pci_chips"] = tpu_internal.num_available_chips()
    except Exception as exc:  # pragma: no cover - the scan is platform-specific
        report["pci_chips"] = f"{type(exc).__name__}: {exc}"
    try:
        report["metadata"] = {str(k): str(v) for k, v in tpu_internal.get_tpu_env().items()}
    except Exception as exc:  # pragma: no cover - the metadata server is platform-specific
        report["metadata"] = f"{type(exc).__name__}: {exc}"
    return report


def _device_report(torch, xm, xr) -> dict[str, Any]:
    """What the runtime actually gave us: device count, type, per-core memory, environment."""
    devices = xm.get_xla_supported_devices()
    device = xm.xla_device()
    report: dict[str, Any] = {
        "pytorch": torch.__version__,
        "xla_devices": len(devices),
        "device_kind": str(device),
        "environment": environment_report(),
    }
    try:
        import torch_xla  # noqa: PLC0415

        report["torch_xla"] = torch_xla.__version__
        report["tpu"] = _tpu_report()
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


def _build_declared(torch, xm, dtype: str, *, device=None, config=None):
    """The pre-registered 285M model and an optimizer over it, on an XLA device.

    ``device`` is explicit for a caller that has several in play - the aggregate's ranks -
    and defaults to the process's device, which is what the single-device phases use.
    ``config`` is the declared session unless a caller names another one: the aggregate's
    ranks each carry their own seed, and that is the one place a caller does.
    """
    from act.model.factory import v0_moe_model_from_genome
    from act.training.v0_session import _genome_from_config, declared_session_config

    config = config or declared_session_config()
    genome = _genome_from_config(config)
    model = v0_moe_model_from_genome(genome)
    if dtype == "bf16":
        model = model.to(torch.bfloat16)
    elif dtype == "fp32":
        model = model.to(torch.float32)
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(dtype)
    if device is None:
        device = xm.xla_device()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total = sum(p.numel() for p in model.parameters())
    return config, genome, model, optimizer, total


def _measure(
    torch,
    xm,
    model,
    optimizer,
    config,
    *,
    steps: int,
    device=None,
    after_first: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Time one short step loop, with the first (compiling) step reported separately.

    ``after_first`` runs once, after the first step has been executed - the hook the
    aggregate's ranks use to meet at their barrier, which is what keeps every rank's compile
    out of the window the aggregate is measured over. The step *times* are not affected by
    it: they are read from the same clock either way.
    """
    if device is None:
        device = xm.xla_device()
    generator = torch.Generator().manual_seed(config.seed)
    batch = (config.batch_size, config.context_size)
    tokens_per_step = config.batch_size * config.context_size
    step_seconds: list[float] = []
    for step in range(steps):
        inputs = torch.randint(
            0, config.vocab_size, batch, generator=generator, dtype=torch.long
        ).to(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(inputs)
        loss.backward()
        optimizer.step()
        xm.mark_step()
        elapsed = time.perf_counter() - start
        step_seconds.append(elapsed)
        del loss, inputs
        if step == 0 and after_first is not None:
            after_first()
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


def _seed_rank(
    *,
    rank: int,
    device,
    config: Any,
    steps: int,
    dtype: str,
    directory: str | Path,
    ready: threading.Barrier,
    torch: Any,
    torch_xla: Any,
    xm: Any,
) -> None:
    """One independent seed on one chip: build, compile, then step with the other ranks.

    The pattern that makes a TPU useful here is unchanged - the campaign's arms and seeds
    are independent, so eight chips become eight concurrent *seeds* rather than one
    data-parallel run, which needs no AllReduce, keeps each chip's graph sequential, and
    multiplies the weekly throughput by the chip count instead of by some scaling
    efficiency. What changed on 2026-09-26 is who runs the ranks: this process, one thread
    each, instead of ``xmp.spawn``'s child processes. The rank's device is *passed in* and
    reported back, so "one seed per chip" is a value the parent checks rather than a
    behaviour it trusts the spawn to have had - which is the property the three failed
    process-based attempts at this phase could not state.

    The compile is deliberately outside the measured window. ``_wait_for_ranks`` is called
    from ``_measure``'s ``after_first`` hook, so every rank compiles its own graph first and
    the barrier releases only once the last one has: the steady steps the aggregate counts
    then begin for all ranks together. Without that a slow compile would sit inside the
    fast ranks' window and the aggregate would be an artifact of which compile the host
    scheduled first.

    Before any of that, the thread binds itself to the chip this rank was given
    (:func:`bind_thread_device`). That is not bookkeeping: the step markers this rank issues
    inside ``_measure`` are ``torch_xla.sync()``, which acts on the calling thread's current
    device - so a rank that skipped the binding would mark steps on the process default
    while its tensors sat on another chip, and the window would time graph construction.
    """
    thread_device_before, thread_device_after = bind_thread_device(torch_xla, device)
    _, _, model, optimizer, _ = _build_declared(
        torch, xm, dtype, device=device, config=config
    )
    stamps: dict[str, float] = {}

    def start_steady() -> None:
        _wait_for_ranks(ready, rank=rank)
        stamps["steady_started"] = time.perf_counter()

    started = time.perf_counter()
    measured = _measure(
        torch,
        xm,
        model,
        optimizer,
        config,
        steps=steps,
        device=device,
        after_first=start_steady,
    )
    stamps["steady_finished"] = time.perf_counter()
    measured.update(
        {
            "rank": rank,
            "seed": config.seed,
            "device": str(device),
            # Both sides of the binding, recorded rather than assumed: the process default a
            # fresh thread inherits, and the chip its step markers actually act on. The
            # parent compares the second with ``device``, so "this rank's steps went to its
            # own chip" is a value in the record instead of a property of the runtime.
            "thread_device_before": thread_device_before,
            "thread_device_after": thread_device_after,
            "seconds_wall": round(time.perf_counter() - started, 3),
            # The compile step is not part of the window, so the tokens the window counts
            # are the steps after it.
            "steady_steps": max(0, steps - 1),
            "steady_started": round(stamps["steady_started"], 6),
            "steady_finished": round(stamps["steady_finished"], 6),
        }
    )
    rank_report_path(directory, rank).write_text(
        json.dumps(measured, sort_keys=True) + "\n", encoding="utf-8"
    )


def parallel_plan(
    workers: int, steps: int, dtype: str, directory: str | Path
) -> dict[str, Any]:
    """The aggregate's shape as data, so what each rank was told is testable.

    One thread per device in *this* process, and the ordinal is the rank: the device is
    ``xla:<rank>`` rather than a default the runtime picked, which is what lets the record
    carry the device each seed ran on. A plan rather than a spawn call because the shape is
    the property - eight ranks, one per device, all of them required - and a test should be
    able to hold it without a TPU.
    """
    if workers < 1:
        raise ValueError(f"the aggregate needs at least one rank, not {workers}")
    return {
        "workers": workers,
        "steps": steps,
        "dtype": dtype,
        "directory": str(directory),
    }


def rank_report_path(directory: str | Path, rank: int) -> Path:
    """Where one rank's report goes: one file per rank, so a missing rank is visible."""
    return Path(directory) / f"rank-{rank}.json"


def rank_device(torch: Any, rank: int):
    """The device one rank's seed runs on. Explicit, because the record has to say so."""
    return torch.device("xla", rank)


def bind_thread_device(torch_xla: Any, device) -> tuple[str, str]:
    """Point the calling thread's XLA current device at the rank's device, and say both.

    A step marker is not a device argument. In torch_xla 2.8 ``xm.mark_step()`` is
    ``torch_xla.sync()``, and ``sync`` calls ``_XLAC._xla_step_marker(_xla_get_default_device(), ...)``
    - i.e. it acts on the **calling thread's** current device. That value is a C++
    ``thread_local`` (``g_current_device`` in ``torch_xla/csrc/aten_xla_bridge.cpp``) whose
    fallback, when a thread has never set it, is the process default: the device every
    earlier phase of this probe has left at ``xla:0``. So in a one-process, eight-thread
    aggregate, seven ranks would issue step markers for chip 0, ``GetLiveTensors(device)``
    would collect their tensors on the wrong chip, and their graphs would never execute -
    the "steady step" would be graph construction, with a rate to match. Binding each
    thread to its chip first is what torch_xla's own single-process launcher does
    (``_run_thread_per_device`` sets the current device before running the body);
    ``torch_xla.device(index)`` is the public spelling of that call and returns the device
    it bound, so a rank's report can carry both what it was told to use and what its thread
    ended up on.

    Returns ``(before, after)``. ``before`` is normally ``xla:0`` - the process default a
    fresh thread inherits - and it is recorded rather than assumed, because "this thread had
    never chosen a device" is the state the binding exists to change. A device without an
    ordinal is refused: the binding would then be a read of whatever the process default
    happens to be, which is the failure it is here to prevent.
    """
    if device.index is None:
        raise RuntimeError(
            f"cannot bind a rank's thread to {device}: it names no device ordinal, so the "
            "binding would be a read of the process default rather than a choice"
        )
    before = str(torch_xla.device())
    return before, str(torch_xla.device(device.index))


def seeds_timeout() -> float:
    """How long a rank waits for every rank to compile (``TPU_PROBE_SEEDS_TIMEOUT``).

    The phase's clock, and the trade inside its default. Eight compiles of the declared
    shape take minutes, so an hour leaves a slow host room - and a rank that failed before
    the barrier spends that hour of the TPU's **billed** running time before the phase
    records the failure. That is the cheaper half of the alternative: a probe queued for
    hours whose kernel ends with no record at all, which is how three attempts at this one
    measurement ended.
    """
    return float(os.environ.get("TPU_PROBE_SEEDS_TIMEOUT", 3600.0))


class RankWaitTimeout(RuntimeError):
    """A rank stopped at the barrier because some other rank never reached it.

    Its own type so the parent can tell collateral from cause when it reports a failed
    aggregate: seven ranks that stopped waiting are not seven failures, and the record has
    to be able to say which one was the cause.
    """


def _wait_for_ranks(ready: threading.Barrier, *, rank: int) -> None:
    """Wait until every rank has compiled, and name the wait when it breaks.

    A rank that fails before it reaches this point takes the others with it: they wait, the
    barrier breaks (by budget, or by the parent's ``abort`` the moment it sees a rank fail),
    and each of them raises *this* - naming the rank and the budget rather than a bare
    ``BrokenBarrierError``. A probe is queued for hours and runs once, and the four failed
    attempts at this measurement each had to be diagnosed from a line that did not name
    itself.
    """
    budget = seeds_timeout()
    try:
        ready.wait(timeout=budget)
    except threading.BrokenBarrierError as broken:
        raise RankWaitTimeout(
            f"rank {rank} waited {budget:.0f}s for every rank to compile and the barrier "
            "broke - a rank that failed before compiling is the usual cause; raise "
            "TPU_PROBE_SEEDS_TIMEOUT if the compiles are merely slow"
        ) from broken


def aggregate_dtype(dtype: str) -> str:
    """The dtype the eight-worker aggregate is taken at.

    bf16 when the run measures both, because bf16 is what ``evaluate`` reads and what the
    session trains in - the aggregate is divided into per-core *training* hours, so taking
    it at the diagnostic dtype would price a port nobody would run. One definition, in one
    place, because two copies of this choice would be two different measurements under one
    name.
    """
    return dtype if dtype != "both" else MEASUREMENT_ORDER[0]


def measure_independent_seeds(
    *,
    torch: Any,
    xm: Any,
    torch_xla: Any | None = None,
    devices: int | None,
    workers: int,
    steps: int,
    dtype: str,
    directory: str | Path,
    config_for: Callable[[int], Any] | None = None,
    device_for: Callable[[int], Any] | None = None,
    run_rank: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """One independent seed per device, all of them stepping, and the rate they sustain.

    The aggregate is the **tokens the ranks pushed inside the window during which all of
    them were stepping**, divided by that window - not a sum of rates each rank measured
    for itself. That is the difference the process-based version could not see and this one
    does not need to: the barrier in ``_wait_for_ranks`` releases once every rank has
    compiled, so the window opens for all of them together, and a step marker that turns
    out to sync more than the rank that called it then stretches the *window* for everyone
    instead of inflating one rank's rate, which a sum would count twice.

    A rank that did not finish refuses the whole aggregate: a partial parallel measurement is
    not the measurement the decision reads, and it is contained as a phase error rather than
    averaged into a plausible number. The refusal is also **prompt**: the ranks are awaited
    with ``FIRST_EXCEPTION`` and the barrier is aborted the moment one of them fails, so a
    dead rank costs the phase seconds and names its own error, instead of the ranks behind it
    spending the whole barrier budget (an hour of billed TPU time) and reporting a break. So does a set of reports that does not name one distinct device per
    rank, one that does not agree on the step's token count or the number of steady steps,
    and one whose window holds no steady step - each of those makes the arithmetic below a
    different measurement than the one the rule reads.

    ``devices`` is the host's device count when the caller has it. ``main`` does, because it
    has already initialized the runtime and recorded that count in the hardware report, so
    requiring ``devices == workers`` is a real check here rather than the one the child
    process had to skip. The two refusal cases stay distinct because a mismatch has a
    different fix (a wrong ``--workers``) from a rank that failed (a core that is not
    there).

    Each rank's thread has to *name* its own device before it steps (see
    :func:`bind_thread_device`), so the rank reports are checked for that too: a report
    whose thread ended up on another chip is refused by name rather than averaged in, since
    its steps would have been compile-and-queue time rather than training time.

    ``torch_xla`` is the module the default rank binds with; ``config_for``, ``device_for``
    and ``run_rank`` exist so the arithmetic and every refusal can be tested without a TPU:
    their defaults are the declared session's seeds, ``xla:<rank>``, and :func:`_seed_rank`.
    A caller that supplies its own ``run_rank`` need not supply the module, but the default
    path refuses to run without it.
    """
    plan = parallel_plan(workers, steps, dtype, directory)
    if run_rank is None and torch_xla is None:
        raise RuntimeError(
            "the aggregate needs the torch_xla module (or an explicit run_rank): every "
            "rank's thread has to be bound to its own device before it steps, or its step "
            "markers land on the process default and seven of the eight chips compile "
            "graphs nobody executes"
        )
    if devices is not None and devices != workers:
        raise RuntimeError(
            f"the host exposes {devices} XLA device(s) but the measurement declares "
            f"{workers} worker(s); the aggregate is divided by the worker count, so a "
            "mismatch is not a measurement"
        )
    directory = Path(plan["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    if config_for is None:
        from act.training.v0_session import declared_session_config  # noqa: PLC0415

        config_for = lambda rank: declared_session_config(seed=15_000 + rank)  # noqa: E731
    build_device = device_for or (lambda rank: rank_device(torch, rank))
    body = run_rank or _seed_rank
    ready = threading.Barrier(workers)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                functools.partial(
                    body,
                    torch=torch,
                    torch_xla=torch_xla,
                    xm=xm,
                    ready=ready,
                    steps=plan["steps"],
                    dtype=plan["dtype"],
                    directory=directory,
                ),
                rank=rank,
                device=build_device(rank),
                config=config_for(rank),
            )
            for rank in range(workers)
        ]
        # A rank that raised is reported by its own exception, never averaged away - and
        # reported *when it raises*. Waiting on the ranks in order would let one that failed
        # early leave the others sitting at the barrier until the budget ran out: the phase
        # would burn that hour of billed TPU time and then report a broken barrier instead of
        # the error that caused it. ``abort`` releases every waiting rank at once, and the
        # summary below keeps cause and collateral apart by exception type.
        done, _ = concurrent.futures.wait(
            futures, return_when=concurrent.futures.FIRST_EXCEPTION
        )
        if any(future.exception() is not None for future in done):
            ready.abort()
        failures = [
            (rank, future.exception())
            for rank, future in enumerate(futures)
            if future.exception() is not None
        ]
        if failures:
            raised = [
                f"rank {rank} raised {type(error).__name__}: {error}"
                for rank, error in failures
                if not isinstance(error, RankWaitTimeout)
            ]
            waiting = [
                rank for rank, error in failures if isinstance(error, RankWaitTimeout)
            ]
            raise RuntimeError(
                "the aggregate did not complete: "
                + ("; ".join(raised) or "no rank reported a cause")
                + (
                    f"; rank(s) {waiting} stopped at the barrier waiting for it"
                    if waiting
                    else ""
                )
            )
        # Every rank completed: the reports below are then the whole measurement.
        for future in futures:
            future.result()
    wall = time.perf_counter() - started

    rows: list[dict[str, Any]] = []
    for rank in range(workers):
        path = rank_report_path(directory, rank)
        if not path.is_file():
            raise RuntimeError(
                f"rank {rank} of {workers} wrote no report; the aggregate needs all of them"
            )
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    named = {str(row["device"]) for row in rows}
    if len(named) != workers:
        raise RuntimeError(
            f"the {workers} ranks named {len(named)} device(s) - {sorted(named)} - and the "
            "aggregate's claim is one independent seed per device"
        )
    stray = [
        (int(row["rank"]), str(row["device"]), str(row["thread_device_after"]))
        for row in rows
        if str(row["thread_device_after"]) != str(row["device"])
    ]
    if stray:
        raise RuntimeError(
            "a rank's thread is not bound to its own device: "
            + "; ".join(
                f"rank {rank} was given {device} but runs with {thread}"
                for rank, device, thread in stray
            )
            + " - a step marker acts on the calling thread's device, so those steps "
            "measured graph construction, not training"
        )
    shapes = {int(row["tokens_per_step"]) for row in rows}
    steady_steps = {int(row["steady_steps"]) for row in rows}
    if len(shapes) != 1 or len(steady_steps) != 1:
        raise RuntimeError(
            "the ranks did not measure one shape: tokens per step "
            f"{sorted(shapes)}, steady steps {sorted(steady_steps)}"
        )
    per_step = shapes.pop()
    counted = steady_steps.pop()
    if counted < 1:
        raise RuntimeError(
            f"the ranks reported {counted} steady step(s); a window with no steady step "
            "measures the compile and nothing else"
        )
    window = max(float(row["steady_finished"]) for row in rows) - min(
        float(row["steady_started"]) for row in rows
    )
    if window <= 0:  # pragma: no cover - two readings of one clock cannot invert
        raise RuntimeError(f"the steady window measured {window:.6f}s")
    tokens = sum(int(row["steps"]) * int(row["tokens_per_step"]) for row in rows)
    aggregate = workers * counted * per_step / window
    seeds = sorted(int(row["seed"]) for row in rows)
    return {
        "workers": workers,
        "completed": len(rows),
        "wall_seconds": round(wall, 3),
        "steady_window_seconds": round(window, 3),
        "steady_steps_per_rank": counted,
        "steady_tokens": workers * counted * per_step,
        "aggregate_tokens_per_second": round(aggregate, 1),
        "wall_tokens_per_second_including_startup": round(tokens / wall, 1) if wall else None,
        "aggregate_definition": (
            "the tokens the ranks pushed inside the window during which all of them were "
            "stepping (each rank's compile is outside it), divided by that window: one "
            "independent seed per device, one process, one thread per device"
        ),
        "seed_note": (
            f"each rank's seed ({seeds[0]}-{seeds[-1]}) is its input generator's; the model "
            "is constructed fresh in the rank's own thread, so the ranks are independent "
            "runs but a rank's initialisation is not reproducible from this seed (the "
            "process RNG is shared)"
        ),
        "per_rank": rows,
        # The property the step markers depend on, recorded beside the number: each rank's
        # thread named its own chip before it built, and the parent checked that it did.
        "rank_threads": (
            "each rank's thread bound to its own device before building "
            "(per_rank[i].thread_device_after; the process default it replaced is in "
            "thread_device_before)"
        ),
        "ratio_to_t4_pair": round(aggregate / T4_PAIR_TOKENS_PER_SECOND, 2),
        # The mechanism, named beside the number: a reader has to be able to tell this
        # aggregate from the process-based one the three earlier attempts tried, because
        # what it means is "eight seeds, one process".
        "taken_in": "single-process-thread-per-device",
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

    import torch
    import torch_xla
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
        """Eight devices, one independent seed each: the aggregate the decision reads first.

        Taken in *this* process, and that is the fix rather than a detail. ``xmp.spawn``
        refuses a process whose XLA runtime is initialized, and ``_device_report``
        initialized this one before the first phase ran - so this measurement was handed to
        a child process for three attempts, and the last of them died in libtpu's topology.
        The single-process path needs no such precondition: it is the path every other
        phase in this file has already measured the declared shape through, and
        ``measure_independent_seeds`` pins one rank per device explicitly, so the record
        says which chip each seed ran on instead of trusting a spawn to have spread them.
        """
        with tempfile.TemporaryDirectory(prefix="tpu-probe-seeds-") as directory:
            record["independent_seeds"] = measure_independent_seeds(
                torch=torch,
                xm=xm,
                # The module, not just ``xm``: each rank's thread binds its chip through
                # ``torch_xla.device(rank)`` before it steps, which is the whole reason a
                # one-process aggregate can say which chip each seed ran on.
                torch_xla=torch_xla,
                # Known here, and recorded: ``_device_report`` already asked for it, and the
                # child process's version of this check had to skip it precisely because
                # asking would have initialized the runtime its spawn required untouched.
                devices=len(xm.get_xla_supported_devices()),
                workers=args.workers,
                steps=args.steps,
                dtype=aggregate_dtype(args.dtype),
                directory=directory,
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
