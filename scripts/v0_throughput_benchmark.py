"""Session-1 throughput benchmark and schedule report (section 5.2, pre-launch item 2).

Times the declared V0 model's real training step loop (fwd + bwd + optimizer,
identical op mix to the session runner) for a configurable number of steps,
then applies the pre-committed dtype rule and prints the derived schedule:

    PYTHONPATH=src python scripts/v0_throughput_benchmark.py --steps 50

Pre-launch checklist (section 5.2):
  2. benchmark completes, measured tokens/s recorded, dtype decision logged;
  3. max_checkpoints=2 and checkpoint_interval=500 confirmed;
  4. declared seed block untouched, probe budgets as pre-registered.

The measured JSON is the artifact; the dtype decision rides on it.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from act.model import V0MoECausalLM, estimated_parameter_count
from act.training.runtime import _autocast_context, gpu_inventory, resolve_device
from act.training.session_plan import (
    FP32_FLOOR_TOKENS_PER_SECOND,
    MAX_CHECKPOINTS,
    PairSchedule,
    decide_dtype,
    deadline_decision,
    expected_probe_cost_percent,
    tolerance_band,
    working_dir_headroom_gb,
)
from act.training.v0_session import V0SessionConfig

DECLARED_CHECKPOINT_INTERVAL = 500


def build_declared_model(device: torch.device) -> tuple[V0MoECausalLM, int, int]:
    """Construct the declared 285M model and return it with (total, active) counts."""
    config = V0SessionConfig().model_config()
    total, active = estimated_parameter_count(config)
    torch.manual_seed(15012)  # outside the reserved 15000-15011 seed block
    model = V0MoECausalLM(config).to(device)
    return model, total, active


def measure_tokens_per_second(
    model: V0MoECausalLM,
    config: V0SessionConfig,
    device: torch.device,
    steps: int,
    warmup: int = 5,
    dtype: str = "fp32",
) -> tuple[float, dict[str, float]]:
    """Time the true step loop: loss fwd + backward + AdamW, like _train_arm.

    ``dtype`` selects the loop the session will actually run: ``fp16`` adds the
    autocast context and the GradScaler over fp32 master weights that
    ``V0SessionConfig(dtype="fp16")`` trains with. Returns (tokens/s over the
    timed steps, timing detail dict).
    """
    if dtype not in ("fp32", "fp16"):
        raise ValueError(f"dtype must be fp32 or fp16, got {dtype!r}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(15012)  # benchmark data, not a protocol seed stream
    batch_shape = (config.batch_size, config.context_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # The declared config trains with checkpointed activations (section 5.1);
    # the measured throughput must reflect the config that will actually run.
    model.activation_checkpointing = True
    model.train()
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == "fp16")

    def one_step(tokens: torch.Tensor) -> None:
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, "fp16" if dtype == "fp16" else "off"):
            loss = model.loss(tokens)
        if dtype == "fp16":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    for _ in range(warmup):
        tokens = torch.randint(
            0, config.vocab_size, batch_shape, generator=generator
        ).to(device)
        one_step(tokens)
    if device.type == "cuda":
        torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(steps):
        tokens = torch.randint(
            0, config.vocab_size, batch_shape, generator=generator
        ).to(device)
        one_step(tokens)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    tokens_seen = steps * config.batch_size * config.context_size
    rate = tokens_seen / elapsed if elapsed > 0 else 0.0
    detail = {
        "steps": float(steps),
        "warmup_steps": float(warmup),
        "elapsed_seconds": elapsed,
        "tokens_seen": float(tokens_seen),
    }
    return rate, detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps", type=int, default=50, help="timed steps after warmup (section 5.2 uses 50)"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="auto resolves to CUDA when available (the declared decision context)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="where to write the benchmark artifact (default: artifacts/v0-benchmark.json)",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="run a tiny smoke config instead of the declared model (for local checks)",
    )
    args = parser.parse_args()

    if args.steps < 1 or args.warmup < 0:
        raise SystemExit("--steps must be >= 1 and --warmup must be >= 0")

    session_config = V0SessionConfig(
        declared=not args.profile,
        device=args.device,
        **(
            {}
            if not args.profile
            else dict(
                vocab_size=64,
                context_size=256,
                hidden_size=64,
                depth=2,
                attention_heads=4,
                expert_count=4,
                expert_width=32,
            )
        ),
    )
    device = resolve_device(args.device)

    def measure(dtype: str) -> tuple[float, dict[str, float]]:
        """One dtype's rate, on a freshly built model.

        A fresh model per dtype is what makes the comparison a comparison:
        reusing one would time the second pass through the first pass's AdamW
        state and warmed allocator.
        """
        if args.profile:
            fresh = V0MoECausalLM(session_config.model_config()).to(device)
        else:
            fresh, _, _ = build_declared_model(device)
        return measure_tokens_per_second(
            fresh, session_config, device, args.steps, args.warmup, dtype=dtype
        )

    try:
        rate, timing = measure("fp32")
        fp16_rate = (
            measure("fp16")[0]
            if decide_dtype(rate) == "fp16" and device.type == "cuda"
            else None
        )
    except torch.OutOfMemoryError as error:
        raise SystemExit(
            "the declared 285M model does not fit this device in fp32 "
            f"({error}); the section 5.2 decision context is a 16 GB T4 - "
            "run there, or use --profile for a local smoke check"
        ) from error

    total, active = estimated_parameter_count(session_config.model_config())
    dtype = decide_dtype(rate)
    # The pre-committed rule reads the fp32 rate, but the schedule is arithmetic
    # about the session that will run - and the session trains the *decided*
    # dtype. Scheduling from the fp32 number measured 2,066 tok/s on the T4
    # where the config that runs does 4,695, i.e. 2.27x too pessimistic: every
    # session stopped with usable clock left and the run bought twice the
    # sessions it needed. Dtype is an execution detail that touches no
    # pre-registered criterion (section 5.2), so only the packing changes.
    rates = {"fp32": rate}
    if fp16_rate is not None:
        rates["fp16"] = fp16_rate
    schedule_rate = fp16_rate if fp16_rate is not None else rate
    schedule = PairSchedule(tokens_per_second=schedule_rate)

    report = {
        "benchmark": "v0-throughput-5.2",
        "profile": "declared-285M" if not args.profile else "smoke-profile",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        # The rate above is single-device (resolve_device -> cuda:0); how many
        # devices the host actually offered is the difference between a
        # measured rate and a measured rate with idle hardware next to it.
        "gpu_inventory": gpu_inventory(),
        "torch_version": torch.__version__,
        "steps": args.steps,
        "warmup_steps": args.warmup,
        # The dtype rule's input (fp32, section 5.2). ``schedule_*`` are the
        # rate the pair-time arithmetic used, in the dtype the session trains.
        "measured_tokens_per_second": rate,
        "schedule_tokens_per_second": schedule_rate,
        "schedule_dtype": dtype,
        "tokens_per_second_by_dtype": rates,
        "fp32_floor_tokens_per_second": FP32_FLOOR_TOKENS_PER_SECOND,
        "dtype_decision": dtype,
        "declared_parameter_count": total,
        "declared_active_parameter_count": active,
        "timing": timing,
        "schedule": {
            "pair_time_hours": schedule.pair_time_hours,
            "full_pairs_per_session": schedule.full_pairs_per_session,
            "sessions_for_primary": schedule.sessions_for_primary,
            "sessions_total": schedule.sessions_total,
            "working_dir_steady_state_gb": schedule.steady_state_gb(),
            "working_dir_headroom_gb": working_dir_headroom_gb(),
            "max_checkpoints": MAX_CHECKPOINTS,
            "checkpoint_interval": DECLARED_CHECKPOINT_INTERVAL,
        },
        "predicted_probe_cost_percent": {
            "cap4_1batch": expected_probe_cost_percent(1),
            "cap4_2batch": expected_probe_cost_percent(2),
        },
        "tolerance_band_primary": tolerance_band(1),
        "example_deadline_check": deadline_decision(2.0, schedule.pair_time_hours).reason,
        "seed_block": "15000-15011 (reserved, untouched by this benchmark)",
    }
    output = args.output or Path("artifacts/v0-benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "timing"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
