"""Section 5.39: is real-text churn harm a property of churn, or of the rebuild?

Registered in ``RESEARCH_REPORT.md`` section 5.39 before the null arm was observed.
The question it settles: section 5.7 measured that an arm performing **zero**
mutations beat the random arms by a mean `-0.1217` nats, and section 5.3's
five-variant verdict reads as a corollary - every selection variant spent its full
mutation budget, so all of them inherited that churn harm. But those runs executed
on a path that re-drew the mutated layer's learned router on every event (the
section 5.35 defect), and sections 5.24-5.28 / 5.38.1 show repairing it makes
mutation cheaper. So the measured harm is a *level* taken on an unrepaired rebuild.

Three points, the same 18 seeds, the same dev fidelity and real stream as sections
5.38/5.38.1:

    null           zero mutations, budget-matched no-op events, zero probe FLOPs  (this script)
    random_base    seeded churn, historical rebuild                              (stored by 5.38/5.38.1)
    random_clone   seeded churn, preserve_router_on_rebuild + clone              (stored by 5.38/5.38.1)

Primaries are ``null - churn``, so **positive means churn is harmful**:

    P1  null - random_base    untreated churn harm   (registered: positive)
    P2  null - random_clone   treated churn harm     (registered: smaller than P1)

The contrast between them is not a third primary: per seed
``(null - base) - (null - clone) == clone - base``, which is section 5.38's own
registered contrast, so it is reported only as a replication check that the stored
rows reproduce their committed summary's arithmetic.

The two churn arms are already observed (their rows and verdicts are committed), so
this block is registered in advance **of the null arm's observations only** - said
plainly in the registration rather than glossed. No arm in either block has ever
been run with zero mutations, and the *level* of churn harm has not been asked of
these seeds or any other on this path.

Not bit-reproducible in the section 6 sense: this is a GPU block, and section 6's
fixed-thread-count contract is a CPU statement. Reproduction here means re-running
and checking verdict stability.

    PYTHONPATH=src python scripts/churn_harm_study.py --smoke
    PYTHONPATH=src python scripts/churn_harm_study.py --run
    PYTHONPATH=src python scripts/churn_harm_study.py --summarize
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
# The stored churn rows and the sections 5.38/5.38.1 summaries this block reads.
BLOCK_SCRATCH = ROOT / "artifacts" / "real-path-block"
CHURN_RUNS = BLOCK_SCRATCH / "runs.jsonl"
SUMMARIES = ROOT / "artifacts" / "study-summaries"
# This block's own scratch, so it cannot write into the sections' records.
SCRATCH = ROOT / "artifacts" / "churn-harm-study"
NULL_RUNS = SCRATCH / "null-runs.jsonl"
SESSIONS = SCRATCH / "sessions"

REPORT_SECTION = "5.39"
ALPHA = 0.05
# Two primaries, so alpha/2 each (registered).
PER_PRIMARY_ALPHA = ALPHA / 2
N_PERMUTATIONS = 20_000
PERMUTATION_SEED = 5_390_001
REPORT_DEPTH = "dev (14.9M total / 7.6M active)"

# The two registered blocks' own queues. Neither has ever run a zero-mutation arm.
SEED_BLOCKS = {
    "5.38": (15210, 15211, 15212, 15213, 15214, 15215),
    "5.38.1": tuple(range(15220, 15232)),
}
SEEDS = tuple(seed for block in SEED_BLOCKS.values() for seed in block)


def null_config(seed: int) -> V0SessionConfig:
    """The registered null-arm cell: one arm, no draw, no probe, no rebuild."""
    if not REAL_TRAIN.exists():
        raise SystemExit(f"real token stream missing: {REAL_TRAIN}")
    return V0SessionConfig(
        seed=seed,
        arms=("null",),
        data_path=str(REAL_TRAIN),
        val_data_path=str(REAL_VAL),
    )


def config_record() -> dict:
    """The registered knobs, read from the live config so they cannot go stale."""
    config = null_config(SEEDS[0])
    fields = (
        "arms", "vocab_size", "context_size", "hidden_size", "depth",
        "attention_heads", "expert_count", "active_experts", "expert_width",
        "batch_size", "total_steps", "learning_rate", "mutation_interval",
        "mutation_budget", "probe_candidate_cap", "probe_batch_count",
        "probe_candidate_policy", "probe_selection", "dtype",
        "activation_checkpointing",
    )
    record = {name: getattr(config, name) for name in fields}
    # This block runs under the current default (False, section 5.37.1). The stored
    # churn rows ran while it was still True; the flag is trajectory-inert and
    # bit-identical under the section 5.37 liveness contract, so it cannot move a
    # churn loss either way. Disclosed because the record has to say what ran.
    record["churn_rows_ran_under_activation_checkpointing"] = True
    record["null_rows_ran_under_activation_checkpointing"] = bool(
        record["activation_checkpointing"]
    )
    record["report_section"] = REPORT_SECTION
    record["depth_label"] = REPORT_DEPTH
    record["churn_points"] = {
        "random_base": "V0SessionConfig defaults (the historical rebuild)",
        "random_clone": {
            "preserve_router_on_rebuild": True,
            "new_expert_initialization": "clone",
        },
    }
    return record


def config_hash(record: dict) -> str:
    blob = json.dumps(record, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_null_runs() -> dict[int, dict]:
    done: dict[int, dict] = {}
    if NULL_RUNS.exists():
        for line in NULL_RUNS.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["seed"]] = row
    return done


def load_churn_rows() -> dict[tuple[int, str], dict]:
    """The stored section 5.38/5.38.1 rows, keyed by (seed, setting)."""
    if not CHURN_RUNS.exists():
        raise SystemExit(f"section 5.38 runs missing: {CHURN_RUNS}")
    rows: dict[tuple[int, str], dict] = {}
    for line in CHURN_RUNS.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("interrupted"):
            continue
        if row["seed"] in SEEDS:
            rows[(row["seed"], row["setting"])] = row
    return rows


def _arm(row: dict) -> dict:
    """The row's random arm - the only arm this block's churn points use."""
    arms = row.get("arms") or {}
    if "random" not in arms:
        raise SystemExit(f"seed {row['seed']} {row['setting']}: no random arm stored")
    return arms["random"]


def record_null_run(seed: int) -> dict:
    config = null_config(seed)
    checkpoint_dir = SESSIONS / f"seed-{seed}"
    started = time.perf_counter()
    try:
        result = run_v0_session(config, checkpoint_dir=checkpoint_dir)
    except V0SessionInterrupted:
        return {"seed": seed, "interrupted": True, "wall": time.perf_counter() - started}
    wall = time.perf_counter() - started
    row: dict = {
        "seed": seed,
        "interrupted": False,
        "wall_seconds": wall,
        "checkpoint_dir": str(checkpoint_dir),
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
                    "policy": event.get("policy"),
                    "gate_decision": event.get("gate_decision"),
                    "architecture_hash_before": event.get("architecture_hash_before"),
                    "architecture_hash_after": event.get("architecture_hash_after"),
                }
                for event in arm.mutation_history
            ],
        }
    return row


def run_block(budget_seconds: float, limit: int | None) -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    SESSIONS.mkdir(parents=True, exist_ok=True)
    done = load_null_runs()
    deadline = time.perf_counter() + budget_seconds
    started_with = len(done)
    ran = 0
    for seed in SEEDS:
        if seed in done:
            continue
        if limit is not None and ran >= limit:
            break
        if time.perf_counter() >= deadline:
            break
        print(f"[{time.strftime('%H:%M:%S')}] null seed {seed} ...", flush=True)
        row = record_null_run(seed)
        if row["interrupted"]:
            print(
                f"  interrupted after {row['wall']:.0f}s (checkpoints kept); "
                "re-run to resume",
                flush=True,
            )
            return 0
        with NULL_RUNS.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        ran += 1
        loss = row["arms"]["null"]["final_loss"]
        print(f"  {row['wall_seconds']:.0f}s   null={loss:.4f}", flush=True)
    total = len(load_null_runs())
    if total < len(SEEDS):
        print(
            f"budget spent after {total - started_with} new run(s); "
            f"{total}/{len(SEEDS)} recorded; re-run the same command to continue"
        )
    else:
        print(f"block complete: {total} null run(s) recorded in {NULL_RUNS}")
    return 0


# --- statistics (the definitions section 5.38's runner uses, unchanged) -------


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
    """One-sided paired sign-flip permutation p on the mean (upper tail counted)."""
    observed = statistics.fmean(deltas)
    extreme = 0
    for _ in range(N_PERMUTATIONS):
        flipped = statistics.fmean(
            delta if rng.random() < 0.5 else -delta for delta in deltas
        )
        if flipped >= observed:
            extreme += 1
    return (extreme + 1) / (N_PERMUTATIONS + 1)


def compare(label: str, left: list[float], right: list[float], alpha: float) -> dict:
    """Paired comparison on ``left - right``.

    Here the registered direction is "the harm is real", i.e. a *positive* mean, so
    ``positive_established`` is the analogue of section 5.38's ``passes``.
    """
    if len(left) != len(right) or not left:
        raise SystemExit(f"{label}: unpaired or empty")
    deltas = [a - b for a, b in zip(left, right)]
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
    return {
        "label": label,
        "n": n,
        "mean_delta": mean,
        "sample_sd": sd,
        "standard_error": se,
        # Descriptive only: the declared statistics are the exact sign test and the
        # sign-flip permutation test, and no t p-value is recorded anywhere else in
        # this report, so none is invented here.
        "t": t,
        "wins": wins,
        "sign_test_min_wins": bar,
        "sign_test_p": sign_p,
        "sign_test_establishes_positive": sign_passes,
        "permutation_p": perm_p,
        "positive_established": bool(mean > 0 and perm_p <= alpha),
        "the_two_statistics_disagree": sign_passes != (perm_p <= alpha),
        "per_seed_deltas": deltas,
    }


def summarize() -> int:
    nulls = load_null_runs()
    churn = load_churn_rows()
    missing_null = [seed for seed in SEEDS if seed not in nulls]
    missing_churn = [
        (seed, setting)
        for seed in SEEDS
        for setting in ("base", "clone")
        if (seed, setting) not in churn
    ]
    if missing_null or missing_churn:
        raise SystemExit(
            f"incomplete block: missing null {missing_null}, missing churn {missing_churn}"
        )

    base = [_arm(churn[(seed, "base")])["final_loss"] for seed in SEEDS]
    clone = [_arm(churn[(seed, "clone")])["final_loss"] for seed in SEEDS]
    null = [nulls[seed]["arms"]["null"]["final_loss"] for seed in SEEDS]

    comparisons = {
        "null_minus_random_base": compare(
            "null_minus_random_base", null, base, PER_PRIMARY_ALPHA
        ),
        "null_minus_random_clone": compare(
            "null_minus_random_clone", null, clone, PER_PRIMARY_ALPHA
        ),
    }
    # Not a third primary: per seed this is exactly ``clone - base``, section 5.38's
    # own registered contrast, recomputed from the stored rows as a replication check.
    replication = compare(
        "clone_minus_base_replication", clone, base, ALPHA
    )

    p1 = comparisons["null_minus_random_base"]["positive_established"]
    p2 = comparisons["null_minus_random_clone"]["positive_established"]
    treated_harm = comparisons["null_minus_random_clone"]["mean_delta"]
    untreated_harm = comparisons["null_minus_random_base"]["mean_delta"]
    if p1 and not p2:
        branch = "a_harm_is_a_defect_artifact"
    elif p1 and p2:
        branch = "b_harm_survives_the_repair"
    elif not p1 and not p2:
        branch = "c_not_harmful_at_this_fidelity"
    else:
        # P1 fails and P2 passes, or the treatment made churn more costly.
        branch = (
            "d_red_flag_treated_harm_larger"
            if p2 and treated_harm > untreated_harm
            else "unregistered_combination"
        )

    per_seed = [
        {
            "seed": seed,
            "null_final_loss": null[i],
            "random_base_final_loss": base[i],
            "random_clone_final_loss": clone[i],
            "null_minus_random_base": null[i] - base[i],
            "null_minus_random_clone": null[i] - clone[i],
            "clone_minus_base": clone[i] - base[i],
            "null_train_flops": nulls[seed]["arms"]["null"]["training_flops"],
            "null_events": len(nulls[seed]["arms"]["null"]["events"]),
        }
        for i, seed in enumerate(SEEDS)
    ]

    committed = json.loads(
        (SUMMARIES / "real-path-clone-dev-n12.json").read_text()
    )["comparisons"]["random_clone_vs_base"]
    # The 5.38.1 rows recomputed from this block's join, against that block's own
    # committed mean.
    n12 = [row for row in per_seed if row["seed"] in SEED_BLOCKS["5.38.1"]]
    pooled_mean = statistics.fmean(row["clone_minus_base"] for row in n12)
    built_in_checks = {
        "every_seed_has_a_null_run": all(seed in nulls for seed in SEEDS),
        "every_seed_has_both_churn_rows": all(
            (seed, setting) in churn for seed in SEEDS for setting in ("base", "clone")
        ),
        "the_two_registered_blocks_are_disjoint": not (
            set(SEED_BLOCKS["5.38"]) & set(SEED_BLOCKS["5.38.1"])
        ),
        "no_null_arm_mutated": all(
            nulls[seed]["arms"]["null"]["mutation_count"] == 0 for seed in SEEDS
        ),
        "no_null_arm_booked_probe_flops": all(
            nulls[seed]["arms"]["null"]["search_flops"] == 0 for seed in SEEDS
        ),
        "every_null_arm_consumed_the_budget": all(
            len(nulls[seed]["arms"]["null"]["events"]) == 3 for seed in SEEDS
        ),
        "no_null_event_changed_the_architecture": all(
            event["architecture_hash_before"] == event["architecture_hash_after"]
            for seed in SEEDS
            for event in nulls[seed]["arms"]["null"]["events"]
        ),
        "null_events_are_all_recorded_no_ops": all(
            event["operation"] == "null" and event["policy"] == "null"
            for seed in SEEDS
            for event in nulls[seed]["arms"]["null"]["events"]
        ),
        # The stored rows must reproduce section 5.38.1's own committed arithmetic,
        # otherwise this block is joining to numbers nobody can account for.
        "stored_rows_reproduce_the_committed_5_38_1_mean": (
            abs(pooled_mean - committed["mean_delta"]) < 1e-12
        ),
    }

    record = config_record()
    payload = {
        "record_type": "study-summary",
        "kind": "real-path-churn-harm",
        "report_section": REPORT_SECTION,
        "protocol": (
            "paired per seed, three points on the dev shape and the real TinyStories "
            "stream, 400 steps, three events: null (zero mutations) vs the stored "
            "random arms under the historical rebuild (base) and the repair (clone)"
        ),
        "config": record,
        "config_hash": config_hash(record),
        "declared_n": len(SEEDS),
        "seeds": list(SEEDS),
        "seed_blocks": {name: list(seeds) for name, seeds in SEED_BLOCKS.items()},
        "spent_seeds_disclosed": [15201, 15202, 15203],
        "alpha": ALPHA,
        "per_primary_alpha": PER_PRIMARY_ALPHA,
        "n_permutations": N_PERMUTATIONS,
        "bit_reproducible": False,
        "reproduction_note": (
            "GPU block: section 6's fixed-thread-count contract does not apply. "
            "Reproduction means re-running and checking verdict stability."
        ),
        "blindness_note": (
            "Registered in advance of the null arm's observations only: the two churn "
            "arms are the stored sections 5.38/5.38.1 rows, whose values and verdicts "
            "are committed. No arm in either block had ever run with zero mutations, "
            "and the level of churn harm had not been asked of these seeds or any "
            "other on this path."
        ),
        "direction": "deltas are null minus the churn arm; positive means churn is harmful",
        "per_seed": per_seed,
        "comparisons": comparisons,
        "replication_check": replication,
        "built_in_checks": built_in_checks,
        "branch": branch,
        "pooled_5_38_1_replication_mean": pooled_mean,
    }
    target = SUMMARIES / "real-path-churn-harm.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {target}")
    for name, row in comparisons.items():
        print(
            f"  {name:26s} mean {row['mean_delta']:+.5f}  sd {row['sample_sd']:.4f}  "
            f"t {row['t']:+.2f}  wins {row['wins']}/{row['n']} "
            f"(bar {row['sign_test_min_wins']})  sign p {row['sign_test_p']:.4f}  "
            f"perm p {row['permutation_p']:.5f}  "
            f"{'HARM ESTABLISHED' if row['positive_established'] else 'not established'}"
        )
    for name, value in built_in_checks.items():
        print(f"  check {name:44s} {value}")
    print(f"  branch {branch}")
    return 0


def smoke() -> int:
    """A 4-step null session, to check the arm is wired before a block is spent."""
    if not REAL_TRAIN.exists():
        raise SystemExit(f"real token stream missing: {REAL_TRAIN}")
    config = V0SessionConfig(
        seed=SEEDS[0],
        arms=("null",),
        data_path=str(REAL_TRAIN),
        val_data_path=str(REAL_VAL),
        total_steps=4,
        mutation_interval=2,
    )
    result = run_v0_session(config, checkpoint_dir=SCRATCH / "smoke")
    for arm in result.arms:
        print(
            f"  arm {arm.mode}: loss {arm.final_loss:.4f}  mutations {arm.mutation_count}  "
            f"events {len(arm.mutation_history)}  search_flops {arm.search_flops}  "
            f"params {arm.parameter_count:,}/{arm.active_parameter_count:,}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run the null arm (resumable)")
    parser.add_argument("--summarize", action="store_true", help="write the summary")
    parser.add_argument("--smoke", action="store_true", help="force a tiny null session")
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=450.0,
        help="wall-clock budget for this invocation before stopping cleanly",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="stop after this many new runs"
    )
    args = parser.parse_args()
    if args.smoke:
        return smoke()
    if args.run:
        return run_block(args.budget_seconds, args.limit)
    if args.summarize:
        return summarize()
    parser.error("pass --run, --summarize or --smoke")


if __name__ == "__main__":
    raise SystemExit(main())
