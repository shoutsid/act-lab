#!/usr/bin/env python
"""Time the three ways to execute one routing contract, at the shape that matters.

``_MoEFeedForward`` implements top-k expert routing in a **dense** form: every expert is
applied to every token by one batched matmul, and the top-k rows are selected afterwards.
What that costs is visible in the two estimators - ``estimate_dense_runtime_flops_per_token``
against ``estimate_flops_per_token`` - and at the code-LM shape the ratio is **1.934**, so
the dense form executes about twice the arithmetic the routing contract books. The module
already carries the sparse reference (``eager_forward``), and the dev-shape measurement that
chose dense over eager (51.8 vs 42.5 ms/step, +18% for dense) implies FLOP rates above the
card's physical peak and is 11.0x the throughput benchmark's own timing of the same geometry.
See ``docs/training-throughput.md``; the crossover therefore has to be measured rather than
inherited.

Three forms are timed at one shape, one dtype, one device:

* ``dense``   - the module's ``forward``, what the training loop executes today;
* ``eager``   - the module's ``eager_forward``, the reference dispatch;
* ``grouped`` - ``act.model.dispatch.grouped_forward``: route once, sort the ``(token, slot)``
  rows by expert, run one contiguous matmul pair per expert, and scatter the gate-weighted rows
  back with a single ``index_add_``. It does the contracted arithmetic with a bounded number of
  launches, instead of the dense form's 2x arithmetic or the eager form's per-expert masking and
  ``nonzero`` passes. It lives in the library rather than here because the switch it implies has
  to be available to a caller that has measured it, not only to this script.

Equivalence is checked **before** anything is timed: all three forms are run in float64 on a
small batch and compared to each other, and a set of forms that disagrees is refused rather
than timed. Each form's reported rate is computed from **its own** executed arithmetic, so
the table cannot credit a form with FLOPs it did not do.

Usage (never while a training job holds the card - this allocates on the device it is given):

    PYTHONPATH=src python scripts/moe_dispatch_benchmark.py --shape match --steps 20
    PYTHONPATH=src python scripts/moe_dispatch_benchmark.py --check-only --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from act.model.dispatch import grouped_forward
from act.model.v0_moe import V0MoEConfig, _MoEFeedForward
from act.search.architecture_eval import (
    estimate_dense_runtime_flops_per_token,
    estimate_flops_per_token,
)

ROOT = Path(__file__).resolve().parents[1]

#: The two layer geometries worth deciding on. ``dev`` is the shape the inherited +18%
#: measurement was taken at; ``match`` is the code-LM student's, where the arithmetic says
#: the crossover should sit the other way. Only ``hidden``, ``experts``, ``active`` and
#: ``width`` enter a single feed-forward layer; the vocabulary enters the model, not this.
SHAPES: dict[str, dict] = {
    "dev": dict(hidden_size=320, depth=6, attention_heads=5, expert_count=8, active_experts=2, expert_width=320),
    "match": dict(hidden_size=768, depth=12, attention_heads=12, expert_count=8, active_experts=2, expert_width=768),
}

#: GTX 1070 fp32: 1,920 lanes at 1.683 GHz, one FMA each. The T4's fp32 peak is 8.1 TFLOP/s
#: and its fp16 tensor peak 65, which is why the number is an argument rather than a constant.
PEAK_TFLOPS = {"dev-1070": 6.46, "t4": 8.1}

FORMS = ("dense", "eager", "grouped")
#: Which estimator prices which form: the dense form's expert term uses total parameters.
FORM_ARITHMETIC = {
    "dense": "dense_runtime",
    "eager": "contracted",
    "grouped": "contracted",
}


def form_forward(feed_forward: _MoEFeedForward, hidden: torch.Tensor, form: str) -> torch.Tensor:
    if form == "dense":
        return feed_forward(hidden)
    if form == "eager":
        return feed_forward.eager_forward(hidden)
    if form == "grouped":
        return grouped_forward(feed_forward, hidden)
    raise ValueError(f"unknown dispatch form {form!r}")


def build(config: V0MoEConfig, device: torch.device, dtype: torch.dtype, seed: int) -> _MoEFeedForward:
    torch.manual_seed(seed)
    expert_ids = tuple(f"E{index}" for index in range(config.experts_per_layer[0]))
    return _MoEFeedForward(config, expert_ids).to(device=device, dtype=dtype)


def check_equivalence(
    config: V0MoEConfig,
    *,
    batch: int = 2,
    sequence: int = 10,
    atol: float = 1e-12,
) -> dict:
    """Compare the three forms in float64 on the CPU before timing any of them.

    float64 on purpose: the two existing forms are pinned equal at ``1e-12`` by
    ``tests/test_v0_session.py``, so the same tolerance can decide a third. A form that
    disagrees is a different function, and timing it would measure something else. The gate
    runs on the CPU regardless of ``--device``, so a failing gate costs no device memory and
    cannot disturb whatever else is on the card.
    """
    feed_forward = build(config, torch.device("cpu"), torch.float64, seed=5)
    hidden = torch.randn(batch, sequence, config.hidden_size, dtype=torch.float64)
    with torch.no_grad():
        outputs = {form: form_forward(feed_forward, hidden, form) for form in FORMS}
        loads = {form: feed_forward.last_load.clone() for form in FORMS}
    reference = outputs["dense"]
    return {
        "atol": atol,
        "max_abs_diff": {
            form: float((outputs[form] - reference).abs().max().item())
            for form in FORMS
            if form != "dense"
        },
        "loads_agree": all(
            torch.equal(loads[form], loads["dense"]) for form in FORMS
        ),
        "ok": all(
            float((outputs[form] - reference).abs().max().item()) <= atol for form in FORMS
        )
        and all(torch.equal(loads[form], loads["dense"]) for form in FORMS),
    }


def time_form(
    config: V0MoEConfig,
    form: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    batch: int,
    sequence: int,
    steps: int,
    warmup: int = 3,
) -> dict:
    """Time a forward+backward pass of one form, which is what a training step pays.

    Backward is included because the question is a *training* step: the dense form's extra
    arithmetic is paid in both directions. The optimizer is left out deliberately - it sees
    the same parameter count either way, so it would only add a constant to all three.
    """
    feed_forward = build(config, device, dtype, seed=11)
    hidden = torch.randn(batch, sequence, config.hidden_size, device=device, dtype=dtype)

    def one_step() -> None:
        feed_forward.zero_grad(set_to_none=True)
        form_forward(feed_forward, hidden, form).sum().backward()

    for _ in range(warmup):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(steps):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens = batch * sequence * steps
    peak_mb = (
        torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
    )
    return {
        "form": form,
        "elapsed_s": elapsed,
        "ms_per_step": elapsed / steps * 1000.0,
        "tokens_per_s": tokens / elapsed,
        "peak_memory_mb": peak_mb,
    }


def flops_per_token(config: V0MoEConfig, form: str) -> float:
    """The arithmetic that form actually executes, per token, at this shape."""
    which = FORM_ARITHMETIC[form]
    if which == "dense_runtime":
        return estimate_dense_runtime_flops_per_token(config, context_size=config.context_size)
    return estimate_flops_per_token(config, context_size=config.context_size)


def format_table(rows: list[dict]) -> str:
    """The timing table. ``of peak`` is blank off an accelerator, not zero.

    A CPU run has no meaningful fraction of a GPU's peak, and printing one would be the
    same category error as quoting a dev-shape rate for a code-LM shape - so the column
    shows ``n/a`` and the recorded field is ``None``.
    """
    lines = [
        f"{'form':>8} {'ms/step':>9} {'tokens/s':>10} {'MFLOP/token':>12} {'TFLOP/s':>8} {'of peak':>8} {'peak MB':>8}",
    ]
    for row in rows:
        share = "     n/a" if row["of_peak"] is None else f"{row['of_peak'] * 100:>7.1f}%"
        lines.append(
            f"{row['form']:>8} {row['ms_per_step']:>9.2f} {row['tokens_per_s']:>10,.0f} "
            f"{row['flops_per_token'] / 1e6:>12.1f} {row['tflops_per_s']:>8.2f} "
            f"{share} {row['peak_memory_mb']:>8.0f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shape", choices=sorted(SHAPES), default="match")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--peak", choices=sorted(PEAK_TFLOPS), default="dev-1070")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="run the equivalence gate and stop, without allocating for timing",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="time without the gate - for a shape whose equivalence is already pinned",
    )
    args = parser.parse_args(argv)

    spec = SHAPES[args.shape]
    config = V0MoEConfig(
        vocab_size=32_256,
        context_size=args.sequence,
        depth=spec["depth"],
        hidden_size=spec["hidden_size"],
        attention_heads=spec["attention_heads"],
        experts_per_layer=(spec["expert_count"],) * spec["depth"],
        active_experts=spec["active_experts"],
        expert_width=spec["expert_width"],
    )
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    peak_tflops = PEAK_TFLOPS[args.peak]

    report: dict = {
        "shape": args.shape,
        "geometry": spec,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "dtype": args.dtype,
        "batch": args.batch,
        "sequence": args.sequence,
        "steps": args.steps,
        "peak_tflops": peak_tflops,
        "forms": {},
    }

    gate = check_equivalence(config)
    report["equivalence"] = gate
    print(
        f"equivalence gate (float64, {gate['atol']:g}): "
        + ", ".join(f"{form} {diff:.3g}" for form, diff in gate["max_abs_diff"].items())
        + f", routing load identical: {gate['loads_agree']}"
    )
    if not gate["ok"]:
        print("REFUSED: the forms are not the same function; timing them would measure a different one")
        return 1
    if args.check_only:
        return 0

    for form in FORMS:
        row = time_form(
            config,
            form,
            device,
            dtype,
            batch=args.batch,
            sequence=args.sequence,
            steps=args.steps,
        )
        row["flops_per_token"] = flops_per_token(config, form)
        row["arithmetic"] = FORM_ARITHMETIC[form]
        row["tflops_per_s"] = row["flops_per_token"] * row["tokens_per_s"] / 1e12
        # Only an accelerator has a peak to be a fraction of; off one this stays None.
        row["of_peak"] = row["tflops_per_s"] / peak_tflops if device.type == "cuda" else None
        report["forms"][form] = row
        print(f"  timed {form}")

    dense = report["forms"]["dense"]
    for form in FORMS:
        if form != "dense":
            report[f"speedup_vs_dense_{form}"] = dense["ms_per_step"] / report["forms"][form]["ms_per_step"]

    print()
    print(f"{args.shape} geometry, {args.dtype}, {device} ({report['device_name']}), batch {args.batch} x {args.sequence}")
    print(format_table([report["forms"][form] for form in FORMS]))
    for form in FORMS:
        if form == "dense":
            continue
        share = ""
        if dense["of_peak"] is not None:
            share = (
                f" ({report['forms'][form]['of_peak'] * 100:.1f}% of peak against "
                f"{dense['of_peak'] * 100:.1f}%)"
            )
        print(f"  {form} is {report[f'speedup_vs_dense_{form}']:.2f}x the dense form{share}")
    best = max(FORMS, key=lambda form: report["forms"][form]["tokens_per_s"])
    print(
        f"  fastest: {best} - and it is the speedup over the dense form the training loop uses "
        "that decides, not the ordering"
    )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {os.path.relpath(args.json, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
