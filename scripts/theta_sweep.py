"""Section 5.11 study: the intermediate-theta sweep against the null arm.

Runs the pre-registered report section 5.11 protocol on the fresh block
9008-9017 (10 seeds, V0 diagnostic path, steps 600 / interval 200 /
budget 3):

- ``null``  : the do-nothing control (arms=("null",)) - the bar that matters
  since section 5.8 (``doing nothing`` beats random churn on real text);
- ``t002`` / ``t05`` / ``t01`` : gated sessions (draw-then-gate, report
  5.4 semantics) at theta in {0.02, 0.05, 0.1} - the family declared in
  section 5.11 from the recorded 5.4 contrasts (4/11/21 of 24 observed
  draws would have been accepted).

Primary: for each theta, a one-tailed sign test on per-seed gated-minus-
null deltas against Bonferroni alpha/3 = 0.0167 (n = 10 is the smallest
block whose bar, 9/10, can clear it). Family verdict: at least one theta
clears -> the gate axis reopens; none clears -> the probe-on-real-text
program is concluded as falsified in all of its usable forms.

Usage:

    PYTHONPATH=src python scripts/theta_sweep.py run            # sessions + verdict
    PYTHONPATH=src python scripts/theta_sweep.py verdict        # aggregate only
    PYTHONPATH=src python scripts/theta_sweep.py run --dry-run

OUTCOME (2026-09-14, revision 9bd9323): NO theta clears - 0/10 wins at
every theta (p 1.0, bar 9/10), mean gated-minus-null delta monotonically
harmful in acceptance rate (+0.0027 / +0.0236 / +0.0713): a dose-response
of churn harm. The pre-registered falsification branch fires; the
probe-on-real-text program is concluded as falsified in all of its usable
forms. Seeds accepting nothing are bit-identical to the null arm, so the
sweep measured exactly the cost of accepted churn.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from act.evolution.study import _binomial_survival, _gate_min_wins  # noqa: E402
from act.training import V0SessionConfig  # noqa: E402
from act.training.v0_runner import run_pair_session  # noqa: E402

SWEEP_ROOT = ROOT / "artifacts" / "real-diag" / "theta-sweep"
BASELINE_ROOT = ROOT / "artifacts" / "real-proof" / "datasets"
SEEDS = tuple(range(9008, 9018))
STEPS = 600
ALPHA = 0.05
ARMS: dict[str, dict] = {
    "null": {"arms": ("null",)},
    "t002": {"arms": ("gated",), "gate_margin": 0.02},
    "t05": {"arms": ("gated",), "gate_margin": 0.05},
    "t01": {"arms": ("gated",), "gate_margin": 0.1},
}
THETAS = {"t002": 0.02, "t05": 0.05, "t01": 0.1}


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _make_config(overrides: dict) -> V0SessionConfig:
    from local_proof_run import PROOF_CONFIG, REAL_VOCAB_SIZE, _ensure_real_streams

    data_path, val_path = _ensure_real_streams()
    return V0SessionConfig(
        **{
            **PROOF_CONFIG.__dict__,
            "total_steps": STEPS,
            "vocab_size": REAL_VOCAB_SIZE,
            "data_path": str(data_path),
            "val_data_path": str(val_path),
            **overrides,
        }
    )


def run_sessions(measured_tokens_per_second: float) -> None:
    for name, overrides in ARMS.items():
        config = _make_config(overrides)
        _log(
            f"session {name}: arms={config.arms} gate_margin={config.gate_margin} "
            f"seeds={list(SEEDS)}"
        )
        run_pair_session(
            seeds=SEEDS,
            config=config,
            session_dir=SWEEP_ROOT / name / "sessions",
            dataset_root=SWEEP_ROOT / name / "datasets",
            code_revision=_git_revision(),
            remaining_hours=12.0,
            measured_tokens_per_second=measured_tokens_per_second,
            output=SWEEP_ROOT / name / "datasets" / "session.json",
            log=_log,
        )


def _losses(dataset_root: Path, arm: str) -> dict[int, float]:
    losses: dict[int, float] = {}
    for seed in SEEDS:
        path = (
            dataset_root / "act-checkpoints" / "v0-5-1" / f"seed-{seed:04d}"
            / "reports" / "seed-summary.json"
        )
        if not path.is_file():
            continue
        summary = json.loads(path.read_text())
        if summary.get("status") != "complete":
            continue
        losses[seed] = float(summary["arms"][arm]["final_loss"])
    return losses


def _gate_observables(dataset_root: Path, seeds: tuple[int, ...]) -> dict:
    decisions: dict[str, int] = {}
    contrasts: list[float] = []
    for seed in seeds:
        path = (
            dataset_root / "act-checkpoints" / "v0-5-1" / f"seed-{seed:04d}"
            / "reports" / "seed-summary.json"
        )
        if not path.is_file():
            continue
        summary = json.loads(path.read_text())
        for event in summary["arms"].get("gated", {}).get("mutation_history", []):
            decision = event.get("gate_decision")
            if decision is None:
                continue
            decisions[decision] = decisions.get(decision, 0) + 1
            if event.get("draw_probe_loss") is not None and event.get("reference_probe_loss") is not None:
                contrasts.append(
                    float(event["draw_probe_loss"]) - float(event["reference_probe_loss"])
                )
    total = sum(decisions.values())
    return {
        "decision_counts": dict(sorted(decisions.items())),
        "accept_rate": decisions.get("accept", 0) / total if total else None,
        "mean_draw_minus_reference": sum(contrasts) / len(contrasts) if contrasts else None,
    }


def verdict() -> int:
    null_losses = _losses(SWEEP_ROOT / "null" / "datasets", "null")
    if len(null_losses) != len(SEEDS):
        print(f"null arm incomplete ({len(null_losses)}/{len(SEEDS)}); verdict withheld")
        return 1
    results: dict[str, dict] = {}
    family_pass = False
    for name, theta in THETAS.items():
        gated_losses = _losses(SWEEP_ROOT / name / "datasets", "gated")
        common = sorted(set(null_losses) & set(gated_losses))
        if len(common) != len(SEEDS):
            print(f"{name} incomplete ({len(common)}/{len(SEEDS)}); verdict withheld")
            return 1
        deltas = [gated_losses[seed] - null_losses[seed] for seed in common]
        wins = sum(delta < 0 for delta in deltas)
        p = _binomial_survival(wins, len(deltas))
        bar = _gate_min_wins(len(deltas), ALPHA / 3)
        passed = p <= ALPHA / 3
        family_pass = family_pass or passed
        observables = _gate_observables(SWEEP_ROOT / name / "datasets", SEEDS)
        results[name] = {
            "theta": theta,
            "wins": wins,
            "n": len(deltas),
            "mean_delta": sum(deltas) / len(deltas),
            "per_seed_deltas": {str(seed): gated_losses[seed] - null_losses[seed] for seed in common},
            "binomial_p": p,
            "min_wins_bonferroni": bar,
            "passes_bonferroni": passed,
            "gate_observables": observables,
        }
        _log(
            f"{name} (theta={theta}): wins {wins}/{len(deltas)}, "
            f"mean {results[name]['mean_delta']:+.4f}, p {p:.4f} "
            f"(bar {bar}, alpha/3={ALPHA / 3}) -> {'PASS' if passed else 'FAIL'}; "
            f"accept {observables['accept_rate']}"
        )
    # Secondary: the published random arms at 9008-9011 (4 pairs only).
    random_losses = _losses(BASELINE_ROOT, "random")
    secondary = {
        str(seed): null_losses[seed] - random_losses[seed]
        for seed in sorted(set(random_losses) & set(null_losses))
    }
    payload = {
        "protocol": "theta-sweep-vs-null",
        "report_section": "5.11",
        "code_revision": _git_revision(),
        "seeds": list(SEEDS),
        "alpha": ALPHA,
        "bonferroni_alpha": ALPHA / 3,
        "thetas": THETAS,
        "results": results,
        "secondary_null_vs_published_random": secondary,
        "family_verdict": (
            "at least one theta clears alpha/3: the gate axis reopens"
            if family_pass
            else "no theta clears: the probe-on-real-text program is concluded as falsified"
        ),
    }
    out = SWEEP_ROOT / "verdict.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    _log(f"verdict written: {out}")
    _log(f"FAMILY VERDICT: {payload['family_verdict']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verdict"))
    parser.add_argument("--benchmark-steps", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "verdict":
        return verdict()
    from local_proof_run import _measure_tokens_per_second

    probe_config = _make_config({"arms": ("null",)})
    tokens_per_second = _measure_tokens_per_second(probe_config, steps=args.benchmark_steps)
    _log(f"plan: {len(ARMS)} sessions x {len(SEEDS)} seeds at {tokens_per_second:.0f} tokens/s")
    if args.dry_run:
        return 0
    run_sessions(tokens_per_second)
    return verdict()


if __name__ == "__main__":
    raise SystemExit(main())
