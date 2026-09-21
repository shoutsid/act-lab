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

    PJRT_DEVICE=TPU python scripts/tpu_probe.py --steps 12 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

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

#: The declared primary queue, for the overhead arithmetic: 12 seeds x 2 arms x 24,576,000
#: tokens, with 3 mutation events per arm and a candidate cap of 4, so each arm pays
#: 3 + 3x4 = 15 rebuilds.
DECLARED_SEEDS_PER_QUEUE = 12
DECLARED_ARMS_PER_SEED = 2
DECLARED_TOKENS_PER_ARM = 24_576_000
MUTATION_EVENTS_PER_ARM = 3
PROBED_CANDIDATES_PER_EVENT = 4
REBUILDS_PER_ARM = MUTATION_EVENTS_PER_ARM * (1 + PROBED_CANDIDATES_PER_EVENT)


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

    dtypes = ("fp32", "bf16") if args.dtype == "both" else (args.dtype,)
    record["completed"] = []
    _save(record, args.output)
    for dtype in dtypes:
        config, genome, model, optimizer, total = _build_declared(torch, xm, dtype)
        record[dtype] = {
            "parameters": total,
            "throughput": _measure(torch, xm, model, optimizer, config, steps=args.steps),
        }
        record["completed"].append(f"{dtype}.throughput")
        _save(record, args.output)
        record[dtype]["rebuild"] = _rebuild_cost(
            torch, xm, config, genome, dtype=dtype, steps=max(3, args.steps // 3)
        )
        record["completed"].append(f"{dtype}.rebuild")
        _save(record, args.output)
        del model, optimizer

    if args.workers:
        import torch_xla.distributed.xla_multiprocessing as xmp

        with __import__("tempfile").TemporaryDirectory(prefix="tpu-probe-") as temporary:
            paths = [Path(temporary) / f"rank-{rank}.json" for rank in range(args.workers)]
            started = time.perf_counter()
            xmp.spawn(
                _worker,
                args=(args.steps, args.dtype if args.dtype != "both" else "bf16", temporary),
                nprocs=args.workers,
                start_method="fork",
                join=True,
            )
            wall = time.perf_counter() - started
            # xmp.spawn shares one process tree, so the per-rank files are the report.
            per_rank = [
                json.loads(path.read_text()) for path in paths if path.exists()
            ]
        tokens = sum(row["tokens_per_step"] * args.steps for row in per_rank)
        record["independent_seeds"] = {
            "workers": args.workers,
            "completed": len(per_rank),
            "wall_seconds": round(wall, 3),
            "aggregate_tokens_per_second": round(tokens / wall, 1) if wall else None,
            "per_rank": per_rank,
            "ratio_to_t4_pair": (
                round((tokens / wall) / T4_PAIR_TOKENS_PER_SECOND, 2) if wall else None
            ),
        }
        record["completed"].append("independent_seeds")

    _save(record, args.output)
    print(json.dumps(record, indent=1, sort_keys=True))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
