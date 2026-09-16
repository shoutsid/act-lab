"""Section 5.38: the router/clone treatment on the real path, at dev fidelity.

Section 5.36 ran the treatment on the real token stream as a dev-scale smoke and
found every arm's loss improved on three spent seeds. Sections 5.24-5.28 measured the
mechanism on the toy substrate. What is missing is a *registered* real-path
measurement, which is what this block is: pre-registered in ``RESEARCH_REPORT.md``
section 5.38 before the seeds below were observed.

Per seed, two sessions that differ in exactly one thing - the mid-training rebuild:

    base    ``V0SessionConfig`` defaults (the historical rebuild)
    clone   ``preserve_router_on_rebuild=True``, ``new_expert_initialization="clone"``

Both run the dev shape (~14.9M total / 7.6M active) on the real TinyStories stream,
400 steps, three mutation events at 100/200/300, both arms (``random`` and
``adaptive``) in the same session, same seed and same data schedule. The primary is
the **random** arm: it mutates on a seeded schedule and never probes, so its
treatment effect cannot be an artifact of how the probe ranks anything.

**This block is not bit-reproducible in the section-6 sense.** It runs on a GPU, and
section 6's fixed-thread-count contract is a CPU statement. Reproduction here means
re-running the block and checking that the verdict is stable, not that the losses are
bit-identical.

Because a session of this shape takes minutes, the runner is resumable and
budget-aware: completed ``(seed, setting)`` runs are recorded in a scratch JSONL and
skipped, an interrupted session resumes from its own checkpoints, and an invocation
stops cleanly once its wall-clock budget is spent. Re-run the same command to finish.

    PYTHONPATH=src python scripts/real_path_block.py --run
    PYTHONPATH=src python scripts/real_path_block.py --run --budget-seconds 450
    PYTHONPATH=src python scripts/real_path_block.py --summarize
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

from act.training import (
    V0SessionConfig,
    V0SessionInterrupted,
    run_v0_session,
)

ROOT = Path(__file__).resolve().parent.parent
REAL_TRAIN = ROOT / "artifacts" / "real-lm" / "tokens.npy"
REAL_VAL = ROOT / "artifacts" / "real-lm" / "val-tokens.npy"
SCRATCH = ROOT / "artifacts" / "real-path-block"
RUNS = SCRATCH / "runs.jsonl"
SESSIONS = SCRATCH / "sessions"
SUMMARY = ROOT / "artifacts" / "study-summaries" / "real-path-clone-dev.json"

REPORT_SECTION = "5.38"
ALPHA = 0.05
# Two declared primaries (the random arm and the adaptive arm), so the per-primary
# bar is alpha/2 = 0.025; the sign tests are reported beside the paired mean tests.
PRIMARIES = 2
N_PERMUTATIONS = 20_000
PERMUTATION_SEED = 5_380_001

REPORT_DEPTH = "dev (14.9M total / 7.6M active)"

# Fresh, never-observed seeds. The section 5.36 smoke spent 15201-15203 and is
# disclosed as such rather than reused.
SEEDS = (15210, 15211, 15212, 15213, 15214, 15215)

SETTINGS = {
    "base": {},
    "clone": {
        "preserve_router_on_rebuild": True,
        "new_expert_initialization": "clone",
    },
}
ARMS = ("random", "adaptive")


def session_config(seed: int, setting: str) -> V0SessionConfig:
    """The registered configuration for one (seed, setting) cell."""
    if not REAL_TRAIN.exists():
        raise SystemExit(f"real token stream missing: {REAL_TRAIN}")
    return V0SessionConfig(
        seed=seed,
        arms=ARMS,
        data_path=str(REAL_TRAIN),
        val_data_path=str(REAL_VAL),
        **SETTINGS[setting],
    )


def config_record() -> dict:
    """The registered knobs, taken from the base config so it cannot go stale."""
    config = session_config(SEEDS[0], "base")
    fields = (
        "arms", "vocab_size", "context_size", "hidden_size", "depth",
        "attention_heads", "expert_count", "active_experts", "expert_width",
        "batch_size", "total_steps", "learning_rate", "mutation_interval",
        "mutation_budget", "probe_candidate_cap", "probe_batch_count",
        "probe_candidate_policy", "probe_selection", "dtype",
        "activation_checkpointing",
    )
    record = {name: getattr(config, name) for name in fields}
    record["treatments"] = {name: dict(values) for name, values in SETTINGS.items()}
    record["report_section"] = REPORT_SECTION
    record["depth_label"] = REPORT_DEPTH
    return record


def config_hash(record: dict) -> str:
    blob = json.dumps(record, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_runs() -> dict[tuple[int, str], dict]:
    done: dict[tuple[int, str], dict] = {}
    if RUNS.exists():
        for line in RUNS.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row["seed"], row["setting"])] = row
    return done


def record_run(seed: int, setting: str) -> dict:
    """Run (or resume) one cell and return its record."""
    config = session_config(seed, setting)
    tag = f"{setting}/seed-{seed}"
    checkpoint_dir = SESSIONS / setting
    started = time.perf_counter()
    interrupted = False
    try:
        result = run_v0_session(config, checkpoint_dir=checkpoint_dir)
    except V0SessionInterrupted:
        interrupted = True
        result = None
    wall = time.perf_counter() - started
    if interrupted or result is None:
        return {"seed": seed, "setting": setting, "interrupted": True, "wall": wall}
    row: dict = {
        "seed": seed,
        "setting": setting,
        "interrupted": False,
        "wall_seconds": wall,
        "checkpoint_tag": tag,
        "arms": {},
    }
    for arm in result.arms:
        row["arms"][arm.mode] = {
            "final_loss": arm.final_loss,
            "validation_loss": arm.validation_loss,
            "mutation_count": arm.mutation_count,
            "architecture_hash": arm.architecture_hash,
            "parameter_count": arm.parameter_count,
            "active_parameter_count": arm.active_parameter_count,
            "search_flops": arm.search_flops,
            "training_flops": arm.training_flops,
            "tokens_per_second": arm.tokens_per_second,
            "events": [
                {
                    "step": event.get("step"),
                    "operation": event.get("operation"),
                    "layer_id": event.get("layer_id"),
                    "expert_id": event.get("expert_id"),
                }
                for event in arm.mutation_history
            ],
        }
    return row


def run_block(budget_seconds: float) -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    SESSIONS.mkdir(parents=True, exist_ok=True)
    done = load_runs()
    deadline = time.perf_counter() + budget_seconds
    finished_before = len(done)
    for seed in SEEDS:
        for setting in SETTINGS:
            if (seed, setting) in done:
                continue
            if time.perf_counter() >= deadline:
                print(
                    f"budget spent after {len(load_runs()) - finished_before} new run(s); "
                    "re-run the same command to continue"
                )
                return 0
            print(f"[{time.strftime('%H:%M:%S')}] {setting} seed {seed} ...", flush=True)
            row = record_run(seed, setting)
            if row["interrupted"]:
                print(
                    f"  interrupted after {row['wall']:.0f}s (checkpoints kept); "
                    "re-run to resume"
                )
                return 0
            with RUNS.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            losses = "  ".join(
                f"{mode}={arm['final_loss']:.4f}"
                for mode, arm in sorted(row["arms"].items())
            )
            print(f"  {row['wall_seconds']:.0f}s   {losses}", flush=True)
    print(f"block complete: {len(load_runs())} run(s) recorded in {RUNS}")
    return 0


# --- statistics --------------------------------------------------------------


def _binomial_survival(wins: int, trials: int) -> float:
    """Exact one-tailed sign-test p: P(X >= wins) under p = 0.5."""
    total = 0
    for k in range(wins, trials + 1):
        total += math.comb(trials, k)
    return total / 2**trials


def _min_wins(trials: int, alpha: float) -> int:
    for wins in range(trials + 1):
        if _binomial_survival(wins, trials) <= alpha:
            return wins
    return trials + 1


def _permutation_p(deltas: list[float], rng: random.Random) -> float:
    """One-sided paired sign-flip permutation p on the mean (helpful if positive).

    Deltas are ``base - clone``, so the tail counted is the upper one.
    """
    observed = statistics.fmean(deltas)
    extreme = 0
    for _ in range(N_PERMUTATIONS):
        flipped = statistics.fmean(
            delta if rng.random() < 0.5 else -delta for delta in deltas
        )
        if flipped >= observed:
            extreme += 1
    return (extreme + 1) / (N_PERMUTATIONS + 1)


def _t_p_one_tailed(t: float, degrees: float) -> float:
    try:
        from scipy import stats  # noqa: PLC0415

        return float(stats.t.sf(t, degrees))
    except Exception:  # pragma: no cover - scipy is a study-script dependency
        return float("nan")


def compare(
    label: str, base: list[float], clone: list[float], alpha: float
) -> dict:
    """Paired comparison, helpful direction = clone lower.

    The deltas are ``base - clone``, so a *positive* mean is the helpful direction and
    ``passes`` requires it. (An earlier revision of this function tested ``mean < 0`` -
    a sign inherited from the section 5.34 studies, where the delta was treatment minus
    reference. The direction error was found after the block ran, is corrected here,
    and changes only the pass predicate: the deltas, wins, sign p and permutation p are
    computed before it and are unaffected.)
    """
    if len(base) != len(clone) or not base:
        raise SystemExit(f"{label}: unpaired or empty")
    deltas = [b - c for b, c in zip(base, clone)]
    n = len(deltas)
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    t = mean / se if se and se == se and se != 0 else float("nan")
    wins = sum(1 for delta in deltas if delta > 0)
    sign_p = _binomial_survival(wins, n)
    perm_p = _permutation_p(deltas, random.Random(PERMUTATION_SEED))
    bar = _min_wins(n, alpha)
    sign_passes = wins >= bar
    # Section 5.31.1: the sign test and the mean test are different statistics over the
    # same runs, and where they disagree the effect is small relative to the block, so
    # the disagreement is recorded rather than resolved quietly.
    return {
        "the_two_statistics_disagree": sign_passes != (perm_p <= alpha),
        "n": n,
        "mean_delta": mean,
        "standard_error": se,
        "t": t,
        "t_p_one_tailed": _t_p_one_tailed(t, n - 1),
        "wins": wins,
        "sign_test_min_wins": bar,
        "sign_test_p": sign_p,
        "sign_test_passes": sign_passes,
        "permutation_p": perm_p,
        "passes": bool(mean > 0 and perm_p <= alpha),
        "per_seed_deltas": deltas,
    }


def summarize() -> int:
    rows = load_runs()
    missing = [
        (seed, setting) for seed in SEEDS for setting in SETTINGS
        if (seed, setting) not in rows
    ]
    if missing:
        raise SystemExit(f"incomplete block, missing {missing}")

    def series(arm: str, key: str, setting: str) -> list[float]:
        return [rows[(seed, setting)]["arms"][arm][key] for seed in SEEDS]

    per_seed = [
        {
            "seed": seed,
            "random_base": rows[(seed, "base")]["arms"]["random"]["final_loss"],
            "random_clone": rows[(seed, "clone")]["arms"]["random"]["final_loss"],
            "adaptive_base": rows[(seed, "base")]["arms"]["adaptive"]["final_loss"],
            "adaptive_clone": rows[(seed, "clone")]["arms"]["adaptive"]["final_loss"],
            "random_events_base": len(rows[(seed, "base")]["arms"]["random"]["events"]),
            "random_events_clone": len(rows[(seed, "clone")]["arms"]["random"]["events"]),
        }
        for seed in SEEDS
    ]

    per_primary_alpha = ALPHA / PRIMARIES
    comparisons = {
        "random_clone_vs_base": compare(
            "random_clone_vs_base",
            series("random", "final_loss", "base"),
            series("random", "final_loss", "clone"),
            per_primary_alpha,
        ),
        "adaptive_clone_vs_base": compare(
            "adaptive_clone_vs_base",
            series("adaptive", "final_loss", "base"),
            series("adaptive", "final_loss", "clone"),
            per_primary_alpha,
        ),
    }
    gap_base = [
        row["adaptive_base"] - row["random_base"] for row in per_seed
    ]
    gap_clone = [
        row["adaptive_clone"] - row["random_clone"] for row in per_seed
    ]
    comparisons["adaptive_minus_random_gap"] = compare(
        "adaptive_minus_random_gap", gap_base, gap_clone, ALPHA
    )

    events_match = all(
        row["random_events_base"] == row["random_events_clone"] == 3
        for row in per_seed
    )

    def sequences(arm: str, setting: str) -> list[tuple]:
        return [
            tuple(
                (event["operation"], event["layer_id"], event["expert_id"])
                for event in rows[(seed, setting)]["arms"][arm]["events"]
            )
            for seed in SEEDS
        ]

    # Added after the block ran: structural diagnostics, not verdict-bearing. They are
    # what makes the two primaries readable, because they say whether the treatment
    # changed the *rebuild* alone or also *which mutations were selected*.
    sequence_identical = {
        arm: sum(
            1
            for seed in SEEDS
            if sequences(arm, "base")[SEEDS.index(seed)]
            == sequences(arm, "clone")[SEEDS.index(seed)]
        )
        for arm in ARMS
    }
    events_changed_by_arm = {
        arm: len(SEEDS) - sequence_identical[arm] for arm in ARMS
    }

    def grows(arm: str, setting: str) -> int:
        return sum(
            1
            for sequence in sequences(arm, setting)
            for event in sequence
            if event[0] == "grow"
        )

    grows_by_arm_and_setting = {
        arm: {setting: grows(arm, setting) for setting in SETTINGS} for arm in ARMS
    }
    settings_differ = all(
        rows[(seed, "base")]["arms"]["random"]["architecture_hash"]
        != rows[(seed, "clone")]["arms"]["random"]["architecture_hash"]
        or rows[(seed, "base")]["arms"]["random"]["final_loss"]
        != rows[(seed, "clone")]["arms"]["random"]["final_loss"]
        for seed in SEEDS
    )
    treatment_applied = all(
        rows[(seed, "clone")]["arms"][arm]["mutation_count"] == 3
        for seed in SEEDS
        for arm in ARMS
    )
    built_in_checks = {
        "block_matches_declared_n": len(rows) == len(SEEDS) * len(SETTINGS),
        "the_two_settings_are_not_redundant": settings_differ,
        "both_settings_ran_three_events": events_match,
        "the_treatment_reached_both_arms": treatment_applied,
        "every_seed_is_paired": all(
            (seed, setting) in rows for seed in SEEDS for setting in SETTINGS
        ),
        # The clean-comparison guarantee: on the arm that never probes, only the
        # rebuild differs, so its contrast cannot be a selection effect.
        "the_random_arms_events_are_setting_independent": (
            events_changed_by_arm["random"] == 0
        ),
    }

    primaries_pass = all(
        comparisons[name]["passes"]
        for name in ("random_clone_vs_base", "adaptive_clone_vs_base")
    )
    outcome = {
        "branch_a_all_primaries_pass": primaries_pass,
        "branch_b_random_only": (
            comparisons["random_clone_vs_base"]["passes"]
            and not comparisons["adaptive_clone_vs_base"]["passes"]
        ),
        "branch_c_adaptive_only": (
            comparisons["adaptive_clone_vs_base"]["passes"]
            and not comparisons["random_clone_vs_base"]["passes"]
        ),
        "branch_d_neither": not any(
            comparisons[name]["passes"]
            for name in ("random_clone_vs_base", "adaptive_clone_vs_base")
        ),
    }
    record = config_record()
    payload = {
        "record_type": "study-summary",
        "kind": "real-path-clone-dev",
        "report_section": REPORT_SECTION,
        "protocol": (
            "paired per seed: base vs clone on the dev shape and the real TinyStories "
            "stream, 400 steps, three events, both arms in one session"
        ),
        "config": record,
        "config_hash": config_hash(record),
        "declared_n": len(SEEDS),
        "seeds": list(SEEDS),
        "spent_seeds_disclosed": [15201, 15202, 15203],
        "alpha": ALPHA,
        "per_primary_alpha": per_primary_alpha,
        "n_permutations": N_PERMUTATIONS,
        "bit_reproducible": False,
        "reproduction_note": (
            "GPU block: section 6's fixed-thread-count contract does not apply. "
            "Reproduction means re-running and checking verdict stability."
        ),
        "per_seed": per_seed,
        "comparisons": comparisons,
        "built_in_checks": built_in_checks,
        "event_sequences_identical_on": sequence_identical,
        "events_changed_by_arm": events_changed_by_arm,
        "grows_by_arm_and_setting": grows_by_arm_and_setting,
        "outcome": outcome,
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {SUMMARY}")
    for name, row in comparisons.items():
        print(
            f"  {name:28s} mean {row['mean_delta']:+.5f}  t {row['t']:+.2f}  "
            f"wins {row['wins']}/{row['n']} (bar {row['sign_test_min_wins']})  "
            f"sign p {row['sign_test_p']:.4f}  perm p {row['permutation_p']:.5f}  "
            f"{'PASS' if row['passes'] else 'FAIL'}"
        )
    for name, value in built_in_checks.items():
        print(f"  check {name:38s} {value}")
    print(f"  outcome {[k for k, v in outcome.items() if v]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run the block (resumable)")
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=450.0,
        help="wall-clock budget for this invocation before stopping cleanly",
    )
    parser.add_argument("--summarize", action="store_true", help="write the summary")
    args = parser.parse_args()
    if args.run:
        return run_block(args.budget_seconds)
    if args.summarize:
        return summarize()
    parser.error("pass --run or --summarize")


if __name__ == "__main__":
    raise SystemExit(main())
