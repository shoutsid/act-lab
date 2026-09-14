"""Local proof run: the section 5.1 protocol machinery on a small real LLM.

This is the local, GPU-executable rehearsal of the declared T4 protocol
(``RESEARCH_REPORT.md`` section 5.1, operator-scripted in ``T4_RUNBOOK.md``).
Everything about the *machinery* is the production path; everything about the
*scale* is reduced so a local GPU can train a real language model:

- the declared architecture shape (static top-k MoE, 8 experts/layer,
  2 active, checkpointed activations, AdamW) at reduced dims;
- the declared controller (cap-4 candidates, 1 probe batch - the validated
  cost-saving primary) and mutation budget (E=3);
- the declared queue discipline: the 12-seed queue 15000-15011, atomic pairs,
  skip-completed idempotence, deadline gating, publish-to-dataset;
- the machine-checked verdict from ``scripts/v0_verdict.py``.

Deliberately NOT done here (the pre-registration forbids it): no criterion is
loosened and no knob is tuned. A reduced-dims run cannot pass the declared
tolerance band - the band guards the declared 285M model, so a FAIL or
PASS_WITH_QUALIFICATION verdict is the *expected* machine-checked outcome for
this rehearsal. The proof value is in the loop closing: train -> publish ->
verdict, with every number computed by the registered machinery.

Scratch roots (gitignored, disposable):
  artifacts/local-proof/sessions   working dir (max_checkpoints=2 pin)
  artifacts/local-proof/datasets   the handoff dataset root

Usage:
  PYTHONPATH=src python scripts/local_proof_run.py            # full queue
  PYTHONPATH=src python scripts/local_proof_run.py --dry-run  # plan only
  PYTHONPATH=src python scripts/local_proof_run.py --steps N   # shorter horizon
  PYTHONPATH=src python scripts/local_proof_run.py --real-data  # TinyStories streams

With ``--real-data`` the sessions train on the real TinyStories token stream
built by ``scripts/train_real_lm.py`` (vocab 8192) instead of uniform tokens:
same protocol, same identity, but the arm losses are real language losses and
the harm guard becomes semantically meaningful. Uses the dedicated scratch
roots ``artifacts/real-proof/{sessions,datasets}``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from act.training import V0SessionConfig  # noqa: E402
from act.training.session_plan import MAX_CHECKPOINTS  # noqa: E402
from act.training.v0_runner import DECLARED_SEEDS, run_pair_session  # noqa: E402
from act.training.runtime import resolve_device  # noqa: E402

# The reduced-fidelity model. Every dim is explicit: V0SessionConfig defaults
# are the DECLARED protocol dimensions, and a partial config would silently
# instantiate the 285M model (the documented operational hazard). declared=False
# marks this as a non-protocol config everywhere it is recorded.
PROOF_CONFIG = V0SessionConfig(
    arms=("random", "adaptive"),
    seed=15000,  # placeholder; the runner substitutes each queue seed
    declared=False,
    device="auto",
    vocab_size=8_000,
    context_size=512,
    hidden_size=256,
    depth=8,
    attention_heads=8,
    expert_count=8,
    active_experts=2,
    expert_width=256,
    batch_size=8,
    total_steps=600,
    mutation_interval=200,
    mutation_budget=3,
    probe_candidate_cap=4,
    probe_batch_count=1,
    validation_batches=4,
    activation_checkpointing=True,
    checkpoint_interval=150,
    max_checkpoints=MAX_CHECKPOINTS,  # the registered section 5.2 pin (2)
)

SESSION_DIR = ROOT / "artifacts" / "local-proof" / "sessions"
DATASET_ROOT = ROOT / "artifacts" / "local-proof" / "datasets"
REAL_SESSION_DIR = ROOT / "artifacts" / "real-proof" / "sessions"
REAL_DATASET_ROOT = ROOT / "artifacts" / "real-proof" / "datasets"
REAL_VOCAB_SIZE = 8_192  # the TinyStories stream's tokenizer vocabulary


def _measure_tokens_per_second(config: V0SessionConfig, steps: int = 20) -> float:
    """Benchmark the exact proof model with the session-1 production function."""
    import torch

    from act.model import V0MoECausalLM, estimated_parameter_count
    from v0_throughput_benchmark import measure_tokens_per_second

    device = resolve_device(config.device)
    model_config = config.model_config()
    total, active = estimated_parameter_count(model_config)
    torch.manual_seed(15012)  # benchmark data seed, outside the 15000-15011 queue
    model = V0MoECausalLM(model_config).to(device)
    rate, _detail = measure_tokens_per_second(model, config, device, steps=steps)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"benchmark: {total:,} total / {active:,} active parameters, "
        f"{rate:,.0f} tokens/s over {steps} steps on {device}"
    )
    return rate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps", type=int, default=PROOF_CONFIG.total_steps,
        help="training steps per arm (default 600; shorten for a faster rehearsal)",
    )
    parser.add_argument(
        "--hours", type=float, default=12.0,
        help="session clock at entry (the 12-hour Kaggle limit is the protocol value)",
    )
    parser.add_argument(
        "--benchmark-steps", type=int, default=20,
        help="timed steps for the throughput measurement",
    )
    parser.add_argument(
        "--real-data", action="store_true",
        help="train on the real TinyStories token stream (train_real_lm.py artifact)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the benchmark, schedule, and queue plan, then exit",
    )
    args = parser.parse_args()

    overrides: dict = {}
    session_dir, dataset_root = SESSION_DIR, DATASET_ROOT
    if args.real_data:
        data_path, val_path = _ensure_real_streams()
        overrides.update(
            vocab_size=REAL_VOCAB_SIZE,
            data_path=str(data_path),
            val_data_path=str(val_path),
        )
        session_dir, dataset_root = REAL_SESSION_DIR, REAL_DATASET_ROOT
    config = V0SessionConfig(
        **{
            **PROOF_CONFIG.__dict__,
            "total_steps": args.steps,
            **overrides,
        }
    )
    if args.real_data:
        print(
            f"REAL-DATA PROOF: training on TinyStories streams\n"
            f"  train: {config.data_path}\n  val:   {config.val_data_path}"
        )
    tokens_per_arm = config.training_tokens()
    print(f"model dims: {config.__dict__}")
    print(f"queue: {list(DECLARED_SEEDS)}; tokens per arm: {tokens_per_arm:,}")

    tokens_per_second = _measure_tokens_per_second(config, steps=args.benchmark_steps)

    # Session arithmetic uses the production scheduler, not hand math.
    from act.training.session_plan import PairSchedule, deadline_decision

    schedule = PairSchedule(tokens_per_second=tokens_per_second, tokens_per_arm=tokens_per_arm)
    print(
        f"pair time: {schedule.pair_time_hours:.3f} h; "
        f"full pairs per 12 h session: {schedule.full_pairs_per_session}"
    )
    decision = deadline_decision(args.hours, schedule.pair_time_hours)
    print(f"deadline decision at --hours {args.hours}: {decision.action} ({decision.reason})")

    if args.dry_run:
        return 0

    session_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    summary = run_pair_session(
        seeds=DECLARED_SEEDS,
        config=config,
        session_dir=session_dir,
        dataset_root=dataset_root,
        code_revision=_git_revision(),
        remaining_hours=args.hours,
        measured_tokens_per_second=tokens_per_second,
        output=dataset_root / "local-proof-session.json",
        log=_log,
    )
    print(json.dumps({k: summary[k] for k in ("pairs_completed", "latest_published_seed")}, sort_keys=True))

    # The machine-checked verdict closes the loop. Reduced dims cannot land in
    # the declared band (it guards the 285M model), so exit 1 or 2 is the
    # expected rehearsal outcome; what matters is that every number came from
    # the registered machinery.
    verdict_script = ROOT / "scripts" / "v0_verdict.py"
    result = subprocess.run(
        [sys.executable, str(verdict_script), "--dataset-root", str(dataset_root)],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    print(f"verdict exit code: {result.returncode} (0 PASS, 1 FAIL, 2 qualification)")
    print(f"full verdict: {dataset_root / 'v0-5-1-verdict.json'}")
    return 0


def _ensure_real_streams() -> tuple[Path, Path]:
    """Guarantee the TinyStories train/val token streams exist (cached)."""
    import train_real_lm

    train_path, valid_path = train_real_lm.stage_corpus()
    tokens_path = train_real_lm.ART / "tokens.npy"
    if not tokens_path.is_file():
        train_real_lm.stage_tokens(train_path, 40_000_000)
    val_path = train_real_lm.ART / "val-tokens.npy"
    if not val_path.is_file():
        train_real_lm.load_val_tokens(valid_path)
    return tokens_path, val_path


def _log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
