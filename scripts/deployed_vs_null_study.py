"""Section 5.40: the deployed configuration against the do-nothing arm, fresh seeds.

Registered in ``RESEARCH_REPORT.md`` section 5.40 before this block ran, with the
post-hoc six-seed diagnostic that sourced its prediction disclosed in the registration
itself. The question: every vehicle in section 5.13's real-text ladder - probe-ordered,
probe-gated, load-directed - lost to doing nothing, and **all of them ran on a path that
re-drew the mutated layer's learned router on every event**. The repair exists now
(section 5.35) and the deployed configuration for any future run is the treatment on
(section 5.38.3), so 5.13's closure is a statement about a configuration the project no
longer ships. Section 5.39 supplied the level (churn costs 0.11279 nats with the repair);
what was never measured is the comparison the project's own null-arm contract defines as
the bar - **adaptive against doing nothing** (sections 5.9/5.12 primary 2) - on the real
path, with the treatment on, on unobserved seeds.

Per seed, two sessions:

    pair    arms=("random","adaptive") in one session (section 5.38's shape), both with
            preserve_router_on_rebuild=True and new_expert_initialization="clone"
    null    arms=("null",) alone, exactly as section 5.39 ran it

Primaries are ``null - arm``, so **negative = the mutating arm is worse than doing
nothing** (the null arm has the lower loss) - the section 5.39 convention, applied
uniformly. Note that section 5.40's registration sentence states the opposite ("negative
= that arm beats doing nothing"); every other use of the convention in the program, and
this block's own diagnostic narrative, read it as below. The two primaries are therefore
tested on **opposite tails**:

    P1  null - adaptive   adaptive beats doing nothing   tail "upper", prediction FAILS
    P2  null - random     churn harm under the treatment tail "lower", prediction PASSES

The registered prediction is stated in the report before the block ran, sourced from the
six-seed post-hoc diagnostic on the spent 15210-15215 rows, so that a passing P1 reads as
a genuine surprise. That diagnostic is recomputed from the stored rows here and checked,
so the registration's table cannot drift from the data behind it.

Not bit-reproducible in the section 6 sense: this is a GPU block and section 6's
fixed-thread-count contract is a CPU statement.

    PYTHONPATH=src python scripts/deployed_vs_null_study.py --smoke
    PYTHONPATH=src python scripts/deployed_vs_null_study.py --run
    PYTHONPATH=src python scripts/deployed_vs_null_study.py --summarize
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from act.training import (
    V0SessionConfig,
    V0SessionInterrupted,
    run_v0_session,
)

# The sign test, the sign-flip permutation test and the interval arithmetic live in one
# place. Two independent copies of this code is how a sign error survives review, which
# section 5.39.1 demonstrates at its own expense.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from churn_harm_study import PERMUTATION_SEED, compare  # noqa: E402
from churn_harm_study import CHURN_RUNS as SECTION_5_38_RUNS  # noqa: E402
from churn_harm_study import NULL_RUNS as SECTION_5_39_RUNS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REAL_TRAIN = ROOT / "artifacts" / "real-lm" / "tokens.npy"
REAL_VAL = ROOT / "artifacts" / "real-lm" / "val-tokens.npy"
SCRATCH = ROOT / "artifacts" / "real-path-null-study"
RUNS = SCRATCH / "runs.jsonl"
SESSIONS = SCRATCH / "sessions"
SUMMARIES = ROOT / "artifacts" / "study-summaries"
SUMMARY_PATH = SUMMARIES / "real-path-deployed-vs-null.json"

REPORT_SECTION = "5.40"
ALPHA = 0.05
PER_PRIMARY_ALPHA = ALPHA / 2
REPORT_DEPTH = "dev (14.9M total / 7.6M active)"

# The registered block: fresh, and verified unused across src/, scripts/, tests/ and
# every documented block before the first session ran.
SEEDS = tuple(range(15240, 15252))

# The deployed configuration. Section 5.39 measured churn harm with it in place; this
# block asks whether selection beats doing nothing with it in place.
TREATMENT = {
    "preserve_router_on_rebuild": True,
    "new_expert_initialization": "clone",
}

KINDS = {
    "pair": ("random", "adaptive"),
    "null": ("null",),
}

# The post-hoc diagnostic the registered prediction was sourced from: seeds 15210-15215
# hold section 5.38's both-arm rows and section 5.39's null rows, so the three-way read
# costs no compute. Recorded as registered - post-hoc, n=6, spent seeds, not evidence.
DIAGNOSTIC_SEEDS = (15210, 15211, 15212, 15213, 15214, 15215)
DIAGNOSTIC_EXPECTED = {
    "null_minus_adaptive_base": -0.24844,
    "null_minus_random_base": -0.19041,
    "null_minus_adaptive_clone": -0.11686,
    "null_minus_random_clone": -0.11618,
    "adaptive_clone_minus_random_clone": 0.00068,
}
DIAGNOSTIC_TOLERANCE = 5e-05  # the registration round the table to five decimals


def session_config(seed: int, kind: str) -> V0SessionConfig:
    if not REAL_TRAIN.exists():
        raise SystemExit(f"real token stream missing: {REAL_TRAIN}")
    overrides = dict(TREATMENT) if kind == "pair" else {}
    return V0SessionConfig(
        seed=seed,
        arms=KINDS[kind],
        data_path=str(REAL_TRAIN),
        val_data_path=str(REAL_VAL),
        **overrides,
    )


def config_record() -> dict:
    config = session_config(SEEDS[0], "pair")
    fields = (
        "vocab_size", "context_size", "hidden_size", "depth", "attention_heads",
        "expert_count", "active_experts", "expert_width", "batch_size", "total_steps",
        "learning_rate", "mutation_interval", "mutation_budget", "probe_candidate_cap",
        "probe_batch_count", "probe_candidate_policy", "probe_selection", "dtype",
        "activation_checkpointing",
    )
    record = {name: getattr(config, name) for name in fields}
    record["sessions"] = {kind: list(arms) for kind, arms in KINDS.items()}
    record["treatment"] = dict(TREATMENT)
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
                done[(row["seed"], row["kind"])] = row
    return done


def record_run(seed: int, kind: str) -> dict:
    config = session_config(seed, kind)
    checkpoint_dir = SESSIONS / kind / f"seed-{seed}"
    started = time.perf_counter()
    try:
        result = run_v0_session(config, checkpoint_dir=checkpoint_dir)
    except V0SessionInterrupted:
        return {
            "seed": seed,
            "kind": kind,
            "interrupted": True,
            "wall": time.perf_counter() - started,
        }
    wall = time.perf_counter() - started
    row: dict = {
        "seed": seed,
        "kind": kind,
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
                    "architecture_hash_before": event.get("architecture_hash_before"),
                    "architecture_hash_after": event.get("architecture_hash_after"),
                }
                for event in arm.mutation_history
            ],
        }
    return row


def run_block(budget_seconds: float, limit: int | None) -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    done = load_runs()
    deadline = time.perf_counter() + budget_seconds
    ran = 0
    for seed in SEEDS:
        for kind in ("pair", "null"):
            if (seed, kind) in done:
                continue
            if limit is not None and ran >= limit:
                print(f"stopping after {ran} new run(s) as asked")
                return 0
            if time.perf_counter() >= deadline:
                print(
                    f"budget spent after {ran} new run(s); "
                    f"{len(load_runs())}/{len(SEEDS) * 2} recorded; re-run to continue"
                )
                return 0
            print(f"[{time.strftime('%H:%M:%S')}] {kind} seed {seed} ...", flush=True)
            row = record_run(seed, kind)
            if row["interrupted"]:
                print(
                    f"  interrupted after {row['wall']:.0f}s (checkpoints kept); "
                    "re-run to resume",
                    flush=True,
                )
                return 0
            with RUNS.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            ran += 1
            losses = "  ".join(
                f"{mode}={arm['final_loss']:.4f}"
                for mode, arm in sorted(row["arms"].items())
            )
            print(f"  {row['wall_seconds']:.0f}s   {losses}", flush=True)
    total = len(load_runs())
    if total < len(SEEDS) * 2:
        print(f"budget spent after {ran} new run(s); {total}/{len(SEEDS) * 2} recorded")
    else:
        print(f"block complete: {total} run(s) recorded in {RUNS}")
    return 0


def diagnostic() -> dict:
    """Recompute the registration's post-hoc six-seed table from the stored rows.

    Section 5.38's runs carry both arms for seeds 15210-15215; section 5.39's runs carry
    the null arm for the same seeds. This is the evidence the registered prediction was
    sourced from, so it is re-derived here rather than trusted as prose.
    """
    churn: dict[tuple[int, str], dict] = {}
    for line in SECTION_5_38_RUNS.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row["seed"] in DIAGNOSTIC_SEEDS:
                churn[(row["seed"], row["setting"])] = row
    nulls: dict[int, float] = {}
    for line in SECTION_5_39_RUNS.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row["seed"] in DIAGNOSTIC_SEEDS:
                nulls[row["seed"]] = row["arms"]["null"]["final_loss"]

    def series(setting: str, mode: str) -> list[float]:
        return [
            churn[(seed, setting)]["arms"][mode]["final_loss"]
            for seed in DIAGNOSTIC_SEEDS
        ]

    null = [nulls[seed] for seed in DIAGNOSTIC_SEEDS]
    values = {
        "null_minus_adaptive_base": compare(
            "d_ab", null, series("base", "adaptive"), ALPHA, "lower"
        )["mean_delta"],
        "null_minus_random_base": compare(
            "d_rb", null, series("base", "random"), ALPHA, "lower"
        )["mean_delta"],
        "null_minus_adaptive_clone": compare(
            "d_ac", null, series("clone", "adaptive"), ALPHA, "lower"
        )["mean_delta"],
        "null_minus_random_clone": compare(
            "d_rc", null, series("clone", "random"), ALPHA, "lower"
        )["mean_delta"],
        "adaptive_clone_minus_random_clone": compare(
            "d_ar", series("clone", "adaptive"), series("clone", "random"), ALPHA, "lower"
        )["mean_delta"],
    }
    agreement = {
        name: abs(values[name] - expected) <= DIAGNOSTIC_TOLERANCE
        for name, expected in DIAGNOSTIC_EXPECTED.items()
    }
    return {
        "post_hoc": True,
        "n": len(DIAGNOSTIC_SEEDS),
        "seeds": list(DIAGNOSTIC_SEEDS),
        "spent": True,
        "caveat": (
            "Post-hoc, n=6, on seeds sections 5.38/5.38.1 already spent. Not evidence: "
            "it is the disclosed source of section 5.40's registered prediction."
        ),
        "measured": values,
        "registered_table": DIAGNOSTIC_EXPECTED,
        "tolerance": DIAGNOSTIC_TOLERANCE,
        "matches_registration": agreement,
    }


def branch_from_primaries(p1: bool, p2: bool, red_flag: bool) -> str:
    """The registered branch, from the two primaries and the red-flag check.

    Extracted so the derivation is testable independently of the block: the
    registered branch is a function of *which primary passed*, and the failure
    mode this guards is a primary tested on the wrong tail, which changes the
    boolean without changing anything a reader can see in the numbers. Both
    flags mean "the registered primary passed" - P1 by ``null - adaptive`` being
    significantly positive, P2 by ``null - random`` being significantly negative.
    """
    if p1:
        return "a_real_text_leg_reopens"
    if red_flag:
        return "d_red_flag_selection_worse_than_churn"
    if p2:
        return "b_closure_holds_under_the_deployed_configuration"
    return "c_churn_harm_does_not_reproduce_on_fresh_seeds"


def summarize() -> int:
    rows = load_runs()
    missing = [
        (seed, kind) for seed in SEEDS for kind in ("pair", "null")
        if (seed, kind) not in rows
    ]
    if missing:
        raise SystemExit(f"incomplete block, missing {missing}")

    def losses(kind: str, mode: str) -> list[float]:
        return [rows[(seed, kind)]["arms"][mode]["final_loss"] for seed in SEEDS]

    null = losses("null", "null")
    random = losses("pair", "random")
    adaptive = losses("pair", "adaptive")

    # The two primaries are tested on opposite tails, because they are opposite
    # claims about the same sign convention, and getting this wrong prints the
    # inverted branch rather than an error. The convention is section 5.39's, and
    # section 5.40's own diagnostic table states it: the delta is ``null - arm``,
    # so **negative means the mutating arm is worse than doing nothing** (the null
    # arm has the lower loss).
    #
    #   P1  adaptive beats doing nothing  <=>  ``null - adaptive`` significantly
    #                                          *positive*  -> tail "upper".
    #   P2  churn harm under the treatment <=>  ``null - random`` significantly
    #                                          *negative*  -> tail "lower".
    #
    # The first run of this block tested P1 on "lower" as well, which made the
    # p-value mean "the null is significantly better than adaptive" - the exact
    # opposite of P1 passing - and reported branch (a), the real-text leg
    # reopening, on a block where the null arm had the lowest loss on 12/12
    # seeds. The registered prose has the same inversion in one sentence
    # ("negative means that arm beats doing nothing"), which is how it survived
    # review; the signed quantity is what the tests pin.
    comparisons = {
        "null_minus_adaptive": compare(
            "null_minus_adaptive", null, adaptive, PER_PRIMARY_ALPHA, "upper"
        ),
        "null_minus_random": compare(
            "null_minus_random", null, random, PER_PRIMARY_ALPHA, "lower"
        ),
        # Secondary, and the registered red-flag check: positive means selection is
        # worse than churn under the deployed treatment.
        "adaptive_minus_random": compare(
            "adaptive_minus_random", adaptive, random, ALPHA, "upper"
        ),
    }

    p1 = comparisons["null_minus_adaptive"]["established"]
    p2 = comparisons["null_minus_random"]["established"]
    red_flag = (not p1) and comparisons["adaptive_minus_random"]["established"]
    branch = branch_from_primaries(p1, p2, red_flag)

    per_seed = [
        {
            "seed": seed,
            "null_final_loss": null[i],
            "random_final_loss": random[i],
            "adaptive_final_loss": adaptive[i],
            "null_minus_random": null[i] - random[i],
            "null_minus_adaptive": null[i] - adaptive[i],
            "adaptive_minus_random": adaptive[i] - random[i],
            "null_events": len(rows[(seed, "null")]["arms"]["null"]["events"]),
            "random_mutations": rows[(seed, "pair")]["arms"]["random"]["mutation_count"],
            "adaptive_mutations": rows[(seed, "pair")]["arms"]["adaptive"]["mutation_count"],
            # P1 prices the controller's search: the adaptive arm's probe FLOPs are
            # booked, not deducted.
            "adaptive_search_flops": rows[(seed, "pair")]["arms"]["adaptive"]["search_flops"],
            "random_search_flops": rows[(seed, "pair")]["arms"]["random"]["search_flops"],
            "null_search_flops": rows[(seed, "null")]["arms"]["null"]["search_flops"],
        }
        for i, seed in enumerate(SEEDS)
    ]

    diag = diagnostic()
    built_in_checks = {
        "every_seed_has_both_sessions": all(
            (seed, kind) in rows for seed in SEEDS for kind in ("pair", "null")
        ),
        "the_pair_session_really_carries_both_arms": all(
            set(rows[(seed, "pair")]["arms"]) == {"random", "adaptive"}
            for seed in SEEDS
        ),
        "the_null_session_carries_only_the_null_arm": all(
            set(rows[(seed, "null")]["arms"]) == {"null"} for seed in SEEDS
        ),
        "no_null_arm_mutated": all(
            rows[(seed, "null")]["arms"]["null"]["mutation_count"] == 0
            for seed in SEEDS
        ),
        "no_null_arm_booked_probe_flops": all(
            row["null_search_flops"] == 0 for row in per_seed
        ),
        "no_null_event_changed_the_architecture": all(
            event["architecture_hash_before"] == event["architecture_hash_after"]
            for seed in SEEDS
            for event in rows[(seed, "null")]["arms"]["null"]["events"]
        ),
        "every_null_arm_recorded_all_four_cadence_slots": all(
            row["null_events"] == 4 for row in per_seed
        ),
        "both_mutating_arms_applied_their_three_mutations": all(
            row["random_mutations"] == 3 and row["adaptive_mutations"] == 3
            for row in per_seed
        ),
        "the_adaptive_arm_booked_probe_flops": all(
            row["adaptive_search_flops"] > row["random_search_flops"] for row in per_seed
        ),
        "the_pair_session_config_carries_the_deployed_treatment": (
            config_record()["treatment"] == TREATMENT
        ),
        # The registration's own table has to be reproducible from the data behind it.
        "the_registered_diagnostic_recomputes": all(
            diag["matches_registration"].values()
        ),
    }

    record = config_record()
    payload = {
        "record_type": "study-summary",
        "kind": "real-path-deployed-vs-null",
        "report_section": REPORT_SECTION,
        "protocol": (
            "paired per seed on fresh seeds: one pair session (random, adaptive) and one "
            "null session, dev shape, real TinyStories stream, 400 steps, three events, "
            "with preserve_router_on_rebuild and clone on both mutating arms"
        ),
        "config": record,
        "config_hash": config_hash(record),
        "declared_n": len(SEEDS),
        "seeds": list(SEEDS),
        "alpha": ALPHA,
        "per_primary_alpha": PER_PRIMARY_ALPHA,
        "permutation_seed": PERMUTATION_SEED,
        "direction": (
            "deltas are null minus the arm; **negative means the mutating arm is "
            "worse than doing nothing** (the null arm has the lower loss) - the "
            "section 5.39 convention, restated because section 5.40's registration "
            "sentence inverts it"
        ),
        # Which tail each primary is tested on. P1 and P2 are opposite claims about
        # the same delta, so a single tail for both cannot be right; the first run
        # used "lower" for P1 and reported the inverted branch.
        "primary_tails": {
            "null_minus_adaptive": "upper (P1: adaptive beats doing nothing)",
            "null_minus_random": "lower (P2: churn harm under the treatment)",
        },
        "registered_prediction": {
            "null_minus_adaptive": "fails",
            "null_minus_random": "passes (negative at the bar)",
            "source": "the disclosed post-hoc six-seed diagnostic below",
        },
        "prediction_met": {
            "null_minus_adaptive": "passes" if p1 else "fails",
            "null_minus_random": (
                "passes (negative at the bar)" if p2 else "fails (not negative at the bar)"
            ),
        },
        "bit_reproducible": False,
        "reproduction_note": (
            "GPU block: section 6's fixed-thread-count contract does not apply. "
            "Reproduction means re-running and checking verdict stability."
        ),
        "per_seed": per_seed,
        "comparisons": comparisons,
        "mechanism_diagnostic": diag,
        "built_in_checks": built_in_checks,
        "branch": branch,
        "red_flag_adaptive_worse_than_churn": red_flag,
    }
    target = SUMMARY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {target}")
    for name, row in comparisons.items():
        print(
            f"  {name:26s} mean {row['mean_delta']:+.5f}  sd {row['sample_sd']:.4f}  "
            f"t {row['t']:+.2f}  wins {row['wins_in_tested_direction']}/{row['n']} "
            f"(bar {row['sign_test_min_wins']})  sign p {row['sign_test_p']:.4f}  "
            f"perm p {row['permutation_p']:.5f}  "
            f"{'ESTABLISHED' if row['established'] else 'not established'}"
        )
    for name, value in built_in_checks.items():
        print(f"  check {name:50s} {value}")
    print(f"  branch {branch}")
    return 0


def smoke() -> int:
    if not REAL_TRAIN.exists():
        raise SystemExit(f"real token stream missing: {REAL_TRAIN}")
    config = replace(
        session_config(SEEDS[0], "pair"), total_steps=4, mutation_interval=2
    )
    result = run_v0_session(config, checkpoint_dir=SCRATCH / "smoke-pair")
    for arm in result.arms:
        print(
            f"  pair arm {arm.mode}: loss {arm.final_loss:.4f}  mutations "
            f"{arm.mutation_count}  events {len(arm.mutation_history)}  search_flops "
            f"{arm.search_flops:,}"
        )
    print(f"  treatment: preserve_router={config.preserve_router_on_rebuild} "
          f"new_expert={config.new_expert_initialization}")
    null_config = replace(
        session_config(SEEDS[0], "null"), total_steps=4, mutation_interval=2
    )
    result = run_v0_session(null_config, checkpoint_dir=SCRATCH / "smoke-null")
    for arm in result.arms:
        print(
            f"  null arm {arm.mode}: loss {arm.final_loss:.4f}  mutations "
            f"{arm.mutation_count}  events {len(arm.mutation_history)}  search_flops "
            f"{arm.search_flops:,}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run the block (resumable)")
    parser.add_argument("--summarize", action="store_true", help="write the summary")
    parser.add_argument("--smoke", action="store_true", help="force tiny sessions")
    parser.add_argument(
        "--budget-seconds", type=float, default=450.0,
        help="wall-clock budget for this invocation before stopping cleanly",
    )
    parser.add_argument("--limit", type=int, default=None)
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
