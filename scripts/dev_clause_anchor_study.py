"""Section 5.66: dev-fidelity anchors for all three Gate-4 withholding clauses.

The verdict that decides Gate 4 withholds on three clauses - the harm guard, the
section 5.5 pool-matched control and the section 5.14 do-nothing clause - and
``scripts/null_clause_power.py`` reads all three before the declared block lands.
That reading is anchored on the only real-text effect this program has measured with
the treatment in place: section 5.40's null-versus-adaptive contrast (0.13554 nats at
sd 0.1129, z = 1.203). Two clauses have no anchor, and section 5.40's own entry
records the reason for one of them - the pool-matched loss-blind arm "has never been
run beside them".

This block is registered in ``RESEARCH_REPORT.md`` section 5.66 **before it runs**: 12
fresh seeds (15260-15271), four sessions per seed at the dev shape on the real
TinyStories stream with the deployed treatment on all four:

    pair_prefix     arms=("random","adaptive"), probe policy prefix, selection argmin
                    (section 5.40's pair session, fresh seeds - the policy replication)
    pair_spread     arms=("random","adaptive"), probe policy spread, selection argmin
                    (the declared primary's policy)
    control_spread  arms=("random","adaptive"), probe policy spread, selection random
                    (the declared pool-matched loss-blind arm)
    null            arms=("null",)  (the declared null queue's arm)

The clause statistics are the verdict's own, with its limits **imported** from
``scripts/v0_verdict.py`` rather than restated, and each clause's anchor is reported as
mean / sample sd / z with the firing rate at that z computed through
``null_clause_power.clause_reading`` - the same function that produced the existing
null-clause reading, so the two cannot be two different calculations.

Two checks gate the anchors rather than decorating them:

    A4  on pair_prefix the null arm is the lowest of the three arms on >= 10 of 12
        seeds (section 5.40's ordering, fresh seeds). If it is not, the deployment is
        not stable across fresh seeds and that is reported before any anchor is quoted.
    A5  for each contrast the prefix and spread policies agree in sign. If they do not,
        the anchor for the declared block is the spread root's (the declared primary
        runs spread) and the disagreement is reported.

Nothing here adjudicates a verdict branch. It produces the missing anchor column of
the clause table, and it changes no criterion, gates nothing and adds nothing to the
declared block.

    PYTHONPATH=src python scripts/dev_clause_anchor_study.py --smoke
    PYTHONPATH=src python scripts/dev_clause_anchor_study.py --run
    PYTHONPATH=src python scripts/dev_clause_anchor_study.py --summarize
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

from act.training import (
    V0SessionConfig,
    V0SessionInterrupted,
    run_v0_session,
)

# One place for the paired statistics and the firing-rate arithmetic: two independent
# copies of this code is how a sign error survives review, which section 5.39.1
# demonstrates at its own expense.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from churn_harm_study import compare  # noqa: E402
from null_clause_power import clause_reading  # noqa: E402

# The verdict's own constants, imported rather than restated: a drifted limit or block
# length here would silently describe a different rule than the verdict applies. The
# three limits are asserted equal by ``null_clause_power`` itself (and again below),
# because every withholding clause is the same 10-of-12 sign test.
from v0_verdict import (  # noqa: E402
    HARM_GUARD_LOSS_LIMIT,
    NULL_WIN_LIMIT,
    POOL_CONTROL_WIN_LIMIT,
)

ROOT = Path(__file__).resolve().parent.parent
REAL_TRAIN = ROOT / "artifacts" / "real-lm" / "tokens.npy"
REAL_VAL = ROOT / "artifacts" / "real-lm" / "val-tokens.npy"
SCRATCH = ROOT / "artifacts" / "dev-clause-anchor-study"
RUNS = SCRATCH / "runs.jsonl"
SESSIONS = SCRATCH / "sessions"
SUMMARIES = ROOT / "artifacts" / "study-summaries"
SUMMARY_PATH = SUMMARIES / "dev-clause-anchor.json"

REPORT_SECTION = "5.66"

ALPHA = 0.05

#: 12 fresh seeds, verified unused across src/, scripts/, tests/, docs/ and every
#: committed block before the first session ran.
SEEDS = tuple(range(15260, 15272))

#: The deployed configuration (sections 5.39/5.40): the treatment every future run uses.
TREATMENT = {
    "preserve_router_on_rebuild": True,
    "new_expert_initialization": "clone",
}

#: The four sessions per seed. ``null`` keeps the dev defaults for the two probe fields
#: because a single null arm never probes - they are inert and recorded as such.
KINDS: dict[str, dict[str, object]] = {
    "pair_prefix": {
        "arms": ("random", "adaptive"),
        "probe_candidate_policy": "prefix",
        "probe_selection": "argmin",
    },
    "pair_spread": {
        "arms": ("random", "adaptive"),
        "probe_candidate_policy": "spread",
        "probe_selection": "argmin",
    },
    "control_spread": {
        "arms": ("random", "adaptive"),
        "probe_candidate_policy": "spread",
        "probe_selection": "random",
    },
    "null": {
        "arms": ("null",),
        "probe_candidate_policy": "prefix",
        "probe_selection": "argmin",
    },
}

#: The clauses, as (name, root, comparison arm). The adaptive series is always the
#: root's ``adaptive`` arm - which in ``control_spread`` is the loss-blind controller.
CLAUSES: tuple[tuple[str, str, str], ...] = (
    ("harm_guard_spread", "pair_spread", "random"),
    ("harm_guard_prefix", "pair_prefix", "random"),
    ("pool_matched_control", "control_spread", "random"),
    ("null_clause_spread", "pair_spread", "null"),
    ("null_clause_prefix", "pair_prefix", "null"),
)

#: The clause each PRE-REGISTERED anchor belongs to, and the root it is read from.
ANCHOR_CLAUSES = ("harm_guard_spread", "pool_matched_control", "null_clause_spread")

#: A4: section 5.40's ordering must reproduce on fresh seeds before anchors are read.
REPLICATION_BAR = 10

CONFIG_FIELDS = (
    "vocab_size", "context_size", "hidden_size", "depth", "attention_heads",
    "expert_count", "active_experts", "expert_width", "batch_size", "total_steps",
    "learning_rate", "mutation_interval", "mutation_budget", "probe_candidate_cap",
    "probe_batch_count", "probe_candidate_policy", "probe_selection", "dtype",
    "activation_checkpointing", "validation_batches",
    "preserve_router_on_rebuild", "new_expert_initialization",
)


def session_config(seed: int, kind: str) -> V0SessionConfig:
    if not REAL_TRAIN.exists():
        raise SystemExit(f"real token stream missing: {REAL_TRAIN}")
    spec = KINDS[kind]
    return V0SessionConfig(
        seed=seed,
        arms=spec["arms"],
        probe_candidate_policy=spec["probe_candidate_policy"],
        probe_selection=spec["probe_selection"],
        data_path=str(REAL_TRAIN),
        val_data_path=str(REAL_VAL),
        **TREATMENT,
    )


def config_record() -> dict:
    record: dict = {"sessions": {}, "treatment": dict(TREATMENT)}
    for kind in KINDS:
        config = session_config(SEEDS[0], kind)
        fields = {name: getattr(config, name) for name in CONFIG_FIELDS}
        fields["arms"] = list(config.arms)
        record["sessions"][kind] = fields
    record["report_section"] = REPORT_SECTION
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
    total_runs = len(SEEDS) * len(KINDS)
    ran = 0
    # Seeds outermost so a budget stop leaves whole seeds rather than a seed with three
    # of its four sessions, which is what the paired analysis needs.
    for seed in SEEDS:
        for kind in KINDS:
            if (seed, kind) in done:
                continue
            if limit is not None and ran >= limit:
                print(f"stopping after {ran} new run(s) as asked")
                return 0
            if time.perf_counter() >= deadline:
                print(
                    f"budget spent after {ran} new run(s); "
                    f"{len(load_runs())}/{total_runs} recorded; re-run to continue"
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
    if total < total_runs:
        print(f"budget spent after {ran} new run(s); {total}/{total_runs} recorded")
    else:
        print(f"block complete: {total} run(s) recorded in {RUNS}")
    return 0


def clause_row(
    name: str,
    root: str,
    comparison_label: str,
    adaptive: list[float],
    comparison: list[float],
    limit: int,
) -> dict:
    """One clause's dev-fidelity reading, computed as the verdict computes it.

    ``compare`` is called on ``(comparison - adaptive)`` with the lower tail, so a
    negative mean delta means the comparison arm has the lower loss - the direction in
    which that clause fires. The anchor ``z`` is that mean over its own sample sd,
    signed so that positive means the comparison arm tends to win, which is the
    convention ``null_clause_power.clause_reading`` takes.
    """
    stats = compare(name, comparison, adaptive, ALPHA, "lower")
    mean = stats["mean_delta"]
    sd = stats["sample_sd"]
    z = -mean / sd if sd else float("nan")
    wins = sum(1 for a, c in zip(adaptive, comparison) if c < a)
    fires = wins >= limit
    return {
        "clause": name,
        "root": root,
        "comparison_arm": comparison_label,
        "limit": limit,
        "fires_at_dev_fidelity": fires,
        "comparison_wins": wins,
        "n": len(adaptive),
        "mean_delta_comparison_minus_adaptive": mean,
        "sample_sd": sd,
        "anchor_z": z,
        "firing_rate_at_anchor_z": clause_reading(z)["p_clause_fires"],
        "level_without_effect": clause_reading(0.0)["p_clause_fires"],
        "paired_statistics": stats,
    }


def summarize() -> int:
    rows = load_runs()
    missing = [
        (seed, kind) for seed in SEEDS for kind in KINDS if (seed, kind) not in rows
    ]
    if missing:
        raise SystemExit(f"incomplete block, missing {missing}")

    def losses(kind: str, mode: str) -> list[float]:
        return [rows[(seed, kind)]["arms"][mode]["final_loss"] for seed in SEEDS]

    adaptive_prefix = losses("pair_prefix", "adaptive")
    random_prefix = losses("pair_prefix", "random")
    adaptive_spread = losses("pair_spread", "adaptive")
    random_spread = losses("pair_spread", "random")
    loss_blind = losses("control_spread", "adaptive")
    random_control = losses("control_spread", "random")
    null = losses("null", "null")

    series: dict[str, dict[str, list[float]]] = {
        "pair_prefix": {"adaptive": adaptive_prefix, "random": random_prefix, "null": null},
        "pair_spread": {"adaptive": adaptive_spread, "random": random_spread, "null": null},
        "control_spread": {"adaptive": loss_blind, "random": random_control, "null": null},
    }
    limits = {
        "harm_guard_spread": HARM_GUARD_LOSS_LIMIT,
        "harm_guard_prefix": HARM_GUARD_LOSS_LIMIT,
        "pool_matched_control": POOL_CONTROL_WIN_LIMIT,
        "null_clause_spread": NULL_WIN_LIMIT,
        "null_clause_prefix": NULL_WIN_LIMIT,
    }
    clauses = {
        name: clause_row(
            name,
            root,
            "null (do nothing)" if comparison == "null" else "churn (random)",
            series[root]["adaptive"],
            series[root][comparison],
            limits[name],
        )
        for name, root, comparison in CLAUSES
    }

    # A4: section 5.40's ordering on fresh seeds, under its own policy.
    null_lowest = sum(
        1
        for i in range(len(SEEDS))
        if null[i] < min(random_prefix[i], adaptive_prefix[i])
    )
    replication = {
        "seeds_with_null_lowest": null_lowest,
        "declared_n": len(SEEDS),
        "bar": REPLICATION_BAR,
        "reproduces_section_5_40_ordering": null_lowest >= REPLICATION_BAR,
        "section_5_40_measurement": {
            "seeds_with_null_lowest": 12,
            "null_minus_adaptive_mean": -0.1355433166027069,
            "null_minus_random_mean": -0.1457,
        },
        "this_block": {
            "null_minus_adaptive_mean": statistics.fmean(
                [n - a for n, a in zip(null, adaptive_prefix)]
            ),
            "null_minus_random_mean": statistics.fmean(
                [n - r for n, r in zip(null, random_prefix)]
            ),
        },
        "note": (
            "fresh seeds and the same policy as section 5.40; the ordering is the "
            "replication claim, the means are quoted beside it because a fresh-seed "
            "replication is not bit-equality"
        ),
    }

    # A5: the two policies must agree in sign on each contrast, reported not averaged.
    policy_rows = {}
    contrasts = {
        "null_minus_adaptive": (
            [n - a for n, a in zip(null, adaptive_prefix)],
            [n - a for n, a in zip(null, adaptive_spread)],
        ),
        "null_minus_random": (
            [n - r for n, r in zip(null, random_prefix)],
            [n - r for n, r in zip(null, random_spread)],
        ),
        "adaptive_minus_random": (
            [a - r for a, r in zip(adaptive_prefix, random_prefix)],
            [a - r for a, r in zip(adaptive_spread, random_spread)],
        ),
    }
    for label, (prefix_deltas, spread_deltas) in contrasts.items():
        prefix_mean = statistics.fmean(prefix_deltas)
        spread_mean = statistics.fmean(spread_deltas)
        policy_rows[label] = {
            "prefix_mean": prefix_mean,
            "spread_mean": spread_mean,
            "signs_agree": (prefix_mean >= 0) == (spread_mean >= 0),
        }

    per_seed = [
        {
            "seed": seed,
            "null_final_loss": null[i],
            "prefix_random_final_loss": random_prefix[i],
            "prefix_adaptive_final_loss": adaptive_prefix[i],
            "spread_random_final_loss": random_spread[i],
            "spread_adaptive_final_loss": adaptive_spread[i],
            "control_random_final_loss": random_control[i],
            "control_loss_blind_final_loss": loss_blind[i],
            "null_lowest_of_three": null[i]
            < min(random_prefix[i], adaptive_prefix[i]),
        }
        for i, seed in enumerate(SEEDS)
    ]

    def events(kind: str, mode: str) -> list[dict]:
        return [
            event for seed in SEEDS for event in rows[(seed, kind)]["arms"][mode]["events"]
        ]

    built_in_checks = {
        "every_seed_has_all_four_sessions": all(
            (seed, kind) in rows for seed in SEEDS for kind in KINDS
        ),
        "the_pair_sessions_carry_both_arms": all(
            set(rows[(seed, kind)]["arms"]) == {"random", "adaptive"}
            for seed in SEEDS
            for kind in ("pair_prefix", "pair_spread", "control_spread")
        ),
        "the_null_session_carries_only_the_null_arm": all(
            set(rows[(seed, "null")]["arms"]) == {"null"} for seed in SEEDS
        ),
        "every_pair_session_config_carries_the_deployed_treatment": all(
            config_record()["sessions"][kind]["preserve_router_on_rebuild"] is True
            and config_record()["sessions"][kind]["new_expert_initialization"] == "clone"
            for kind in ("pair_prefix", "pair_spread", "control_spread")
        ),
        "the_policies_and_selections_are_the_registered_ones": (
            config_record()["sessions"]["pair_prefix"]["probe_candidate_policy"] == "prefix"
            and config_record()["sessions"]["pair_prefix"]["probe_selection"] == "argmin"
            and config_record()["sessions"]["pair_spread"]["probe_candidate_policy"]
            == "spread"
            and config_record()["sessions"]["pair_spread"]["probe_selection"] == "argmin"
            and config_record()["sessions"]["control_spread"]["probe_selection"] == "random"
        ),
        "no_null_arm_mutated_or_booked_probe_flops": all(
            rows[(seed, "null")]["arms"]["null"]["mutation_count"] == 0
            and rows[(seed, "null")]["arms"]["null"]["search_flops"] == 0
            for seed in SEEDS
        ),
        "no_null_event_changed_the_architecture": all(
            event["architecture_hash_before"] == event["architecture_hash_after"]
            for event in events("null", "null")
        ),
        "every_null_arm_recorded_all_four_cadence_slots": all(
            len(rows[(seed, "null")]["arms"]["null"]["events"]) == 4 for seed in SEEDS
        ),
        "every_mutating_arm_applied_its_three_mutations": all(
            rows[(seed, kind)]["arms"][mode]["mutation_count"] == 3
            for seed in SEEDS
            for kind in ("pair_prefix", "pair_spread", "control_spread")
            for mode in ("random", "adaptive")
        ),
        "the_adaptive_arms_booked_probe_flops": all(
            rows[(seed, kind)]["arms"]["adaptive"]["search_flops"]
            > rows[(seed, kind)]["arms"]["random"]["search_flops"]
            for seed in SEEDS
            for kind in ("pair_prefix", "pair_spread", "control_spread")
        ),
        "the_anchors_are_read_only_from_a_reproduced_ordering": (
            replication["reproduces_section_5_40_ordering"]
        ),
    }

    payload = {
        "record_type": "study-summary",
        "kind": "dev-clause-anchor",
        "report_section": REPORT_SECTION,
        "protocol": (
            "four sessions per seed on fresh seeds 15260-15271: pair_prefix, "
            "pair_spread, control_spread and null, dev shape on the real TinyStories "
            "stream, 400 steps, three events, deployed treatment on every session"
        ),
        "config": config_record(),
        "config_hash": config_hash(config_record()),
        "declared_n": len(SEEDS),
        "seeds": list(SEEDS),
        "alpha": ALPHA,
        "limits": limits,
        "clauses": clauses,
        "anchor_clauses": list(ANCHOR_CLAUSES),
        "replication_check": replication,
        "policy_check": policy_rows,
        "per_seed": per_seed,
        "built_in_checks": built_in_checks,
        "bit_reproducible": False,
        "reproduction_note": (
            "GPU block: section 6's fixed-thread-count contract does not apply. "
            "Reproduction means re-running and checking the readings' stability."
        ),
        "what_this_is_not": (
            "Not a Gate 4 result and not a verdict: no branch of the declared verdict "
            "is adjudicated here. It produces the dev-fidelity anchor column for the "
            "three withholding clauses."
        ),
    }
    target = SUMMARY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {target}")
    for name in ("harm_guard_spread", "pool_matched_control", "null_clause_spread"):
        row = clauses[name]
        print(
            f"  anchor {name:22s} {row['comparison_wins']}/{row['n']} wins "
            f"(bar {row['limit']})  mean {row['mean_delta_comparison_minus_adaptive']:+.5f}  "
            f"sd {row['sample_sd']:.4f}  z {row['anchor_z']:+.3f}  "
            f"fires at dev {row['fires_at_dev_fidelity']}  "
            f"rate at z {row['firing_rate_at_anchor_z']:.3f}  "
            f"level {row['level_without_effect']:.4f}"
        )
    print(
        f"  replication: null lowest on {replication['seeds_with_null_lowest']}/"
        f"{replication['declared_n']} seeds "
        f"(bar {replication['bar']}) -> "
        f"{replication['reproduces_section_5_40_ordering']}"
    )
    for name, value in built_in_checks.items():
        print(f"  check {name:52s} {value}")
    return 0


def smoke() -> int:
    from dataclasses import replace

    if not REAL_TRAIN.exists():
        raise SystemExit(f"real token stream missing: {REAL_TRAIN}")
    for kind in KINDS:
        config = replace(
            session_config(SEEDS[0], kind), total_steps=4, mutation_interval=2
        )
        result = run_v0_session(config, checkpoint_dir=SCRATCH / "smoke" / kind)
        arms = ", ".join(
            f"{arm.mode} loss {arm.final_loss:.4f} mut {arm.mutation_count} "
            f"probe {arm.search_flops:,}"
            for arm in result.arms
        )
        print(
            f"  {kind:15s} policy {config.probe_candidate_policy:6s} "
            f"selection {config.probe_selection:7s} :: {arms}"
        )
    print(
        f"  treatment: preserve_router={TREATMENT['preserve_router_on_rebuild']} "
        f"new_expert={TREATMENT['new_expert_initialization']}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run the block (resumable)")
    parser.add_argument("--summarize", action="store_true", help="write the summary")
    parser.add_argument("--smoke", action="store_true", help="force tiny sessions")
    parser.add_argument(
        "--budget-seconds", type=float, default=9000.0,
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
