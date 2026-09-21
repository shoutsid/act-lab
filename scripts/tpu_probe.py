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


def _rebuild_cost(torch, xm, config, genome, *, dtype: str, steps: int) -> dict[str, Any]:
    """What a mutation event (or a probed candidate) costs: a new graph, compiled again.

    The candidate path builds a model from a *different* genome, so this rebuilds from a
    genome with one expert added on the last layer - the shape a real event produces -
    rather than reusing the model above, whose graph XLA already has.
    """
    from act.model.factory import v0_moe_model_from_genome

    mutated = genome.to_dict()
    last = mutated["layers"][-1]
    experts = list(last.get("experts", []))
    experts.append(
        {
            "id": f"{last['id']}-probe{len(experts)}",
            "width": config.expert_width,
            "birth_step": 0,
        }
    )
    last["experts"] = experts
    last["active_experts"] = min(
        int(last.get("active_experts", 0)) + 1, len(experts)
    )
    from act.topology.genome import ArchitectureGenome

    rebuilt_genome = ArchitectureGenome.from_dict(mutated)
    model = v0_moe_model_from_genome(rebuilt_genome)
    if dtype == "bf16":
        model = model.to(torch.bfloat16)
    model = model.to(xm.xla_device())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    measured = _measure(torch, xm, model, optimizer, config, steps=steps)
    measured["experts_added"] = 1
    measured["expert_count_after"] = len(experts)
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
    args = parser.parse_args()

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
    for dtype in dtypes:
        config, genome, model, optimizer, total = _build_declared(torch, xm, dtype)
        record[dtype] = {
            "parameters": total,
            "throughput": _measure(torch, xm, model, optimizer, config, steps=args.steps),
        }
        record[dtype]["rebuild"] = _rebuild_cost(
            torch, xm, config, genome, dtype=dtype, steps=max(3, args.steps // 3)
        )
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps(record, indent=1, sort_keys=True))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
