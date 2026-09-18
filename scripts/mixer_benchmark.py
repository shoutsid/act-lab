#!/usr/bin/env python
"""Measure the sequence-mixer cost of each module pattern at a fixed geometry.

Brief section 14 makes ``layer.type`` a first-class search gene. That is only
usable if the mixer layouts are comparable in cost, or if the difference is
measured and paid for deliberately. Nothing here assumes the Qwen 3:1 schedule
is cheap; the schedule is benchmarked as one layout among nine, next to its
inverse as an explicit control.

The measurement uses the same data path, batch size and precision as
``architecture_eval.train_architecture``, because a throughput number taken
under different conditions is not a budget.

Results are appended to the output JSON after each pattern, so an interrupted
benchmark still yields the patterns it finished.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from act.model.factory import v0_moe_config_from_genome
from act.model.v0_moe import V0MoECausalLM, estimated_parameter_count
from act.search.architecture_eval import (
    RunConfig,
    _autocast_context,
    _prepare_streams,
    _training_batches,
)
from act.search.architecture_zoo import MODULE_PATTERNS, generate_module_pattern_zoo


def benchmark_pattern(
    candidate,
    *,
    run: RunConfig,
    device: torch.device,
    warmup: int,
    steps: int,
) -> dict:
    """Time forward+backward+step for one fixed architecture."""
    torch.manual_seed(run.seed)
    config = v0_moe_config_from_genome(candidate.genome, honor_module_types=True)
    model = V0MoECausalLM(config).to(device)

    total_params, active_params = estimated_parameter_count(config)
    train_data, _, _ = _prepare_streams(config.vocab_size, run)
    batches = _training_batches(train_data, run)
    if len(batches) < warmup + steps:
        # Short training streams would otherwise silently benchmark fewer steps
        # than requested, which reads as a faster machine rather than a shorter
        # run. Repeat the batch list instead of reporting a wrong number.
        repeats = (warmup + steps) // max(len(batches), 1) + 1
        batches = (batches * repeats)[: warmup + steps]

    optimizer = torch.optim.AdamW(model.parameters(), lr=run.learning_rate)
    module_types = "/".join(m[0] for m in config.module_types) if config.module_types else "att"

    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    model.train()
    losses: list[float] = []
    step_times: list[float] = []

    for index in range(warmup + steps):
        batch = batches[index].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, run.mixed_precision):
            loss = model.loss(batch)
        loss.backward()
        if run.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), run.grad_clip)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        if index >= warmup:
            step_times.append(elapsed)
            losses.append(float(loss.detach()))

    tokens = steps * run.batch_size * run.context_size
    total = sum(step_times)
    peak = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
    return {
        "pattern": candidate.name,
        "module_types": module_types,
        "total_params": total_params,
        "active_params": active_params,
        "steps": steps,
        "tokens": tokens,
        "median_step_s": statistics.median(step_times),
        "total_step_s": total,
        "tokens_per_s": tokens / total if total > 0 else 0.0,
        "peak_memory_mb": peak,
        "final_loss": losses[-1] if losses else float("nan"),
        "mixed_precision": run.mixed_precision,
        "context_size": run.context_size,
        "batch_size": run.batch_size,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--expert-count", type=int, default=8)
    parser.add_argument("--active-experts", type=int, default=2)
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=12, help="timed steps after warmup")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--patterns",
        default=",".join(MODULE_PATTERNS),
        help="comma-separated subset of module patterns",
    )
    parser.add_argument(
        "--out",
        default="artifacts/mixer-benchmark/mixer_throughput.json",
    )
    args = parser.parse_args(argv)

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    run = RunConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        context_size=args.context_size,
        seed=args.seed,
        # Match the training harness exactly: fp32 (fp16 measured 2.3x slower on
        # this GPU) so the numbers are directly usable as a budget.
        mixed_precision="off",
    )

    candidates = generate_module_pattern_zoo(
        depth=args.depth,
        hidden_size=args.hidden_size,
        expert_count=args.expert_count,
        active_experts=args.active_experts,
        seed=args.seed,
        context_size=args.context_size,
        patterns=[p for p in args.patterns.split(",") if p],
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    print(f"device={device} context={args.context_size} batch={args.batch_size} "
          f"steps={args.steps} warmup={args.warmup}")
    print(f"{'pattern':24s} {'params':>10s} {'tok/s':>10s} {'s/step':>9s} {'peakMB':>8s} {'loss':>8s}")
    for candidate in candidates:
        pattern = candidate.name.rsplit("_", 1)[0].replace("mixer_", "")
        try:
            record = benchmark_pattern(
                candidate, run=run, device=device, warmup=args.warmup, steps=args.steps
            )
        except torch.cuda.OutOfMemoryError:
            print(f"{pattern:24s} {'OOM - skipped':>50s}")
            records.append({"pattern": candidate.name, "error": "out_of_memory"})
            torch.cuda.empty_cache()
            continue
        record["pattern"] = pattern
        records.append(record)
        print(
            f"{pattern:24s} {record['total_params']:10,d} {record['tokens_per_s']:10.0f} "
            f"{record['median_step_s']:9.3f} {record['peak_memory_mb']:8.0f} "
            f"{record['final_loss']:8.4f}"
        )
        out_path.write_text(json.dumps({"config": vars(args), "records": records}, indent=2))
        del record
        torch.cuda.empty_cache()

    usable = [r for r in records if "tokens_per_s" in r]
    baseline = next((r for r in usable if r["pattern"] == "all_attention"), None)
    if baseline:
        print("\nCost relative to all_attention (higher = slower than attention):")
        for record in sorted(usable, key=lambda r: r["tokens_per_s"]):
            ratio = baseline["tokens_per_s"] / record["tokens_per_s"] if record["tokens_per_s"] else float("inf")
            print(f"  {record['pattern']:24s} {ratio:6.2f}x  ({record['tokens_per_s']:7.0f} tok/s)")

    out_path.write_text(json.dumps({"config": vars(args), "records": records}, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
