"""Section 5.38 / 5.38.1: the router-and-clone treatment on the real path, at dev fidelity.

Section 5.36 ran the treatment on the real token stream as a dev-scale smoke and found
every arm's loss improved on three spent seeds. Sections 5.24-5.28 measured the mechanism
on the toy substrate. What was missing is a *registered* real-path measurement. Section
5.38 provided one and landed on its pre-registered branch (c): the probe-guided arm
improved decisively while the clean seeded arm did not clear its bar. Section 5.38.1 is
the follow-up, sized from the dispersion 5.38 measured rather than from the pilot's.

Two profiles share this runner:

    5.38    seeds 15210-15215, both arms, summary real-path-clone-dev.json
    5.38.1  seeds 15220-15231, random arm only, summary real-path-clone-dev-n12.json

Per seed, two sessions that differ in exactly one thing - the mid-training rebuild:

    base    ``V0SessionConfig`` defaults (the historical rebuild)
    clone   ``preserve_router_on_rebuild=True``, ``new_expert_initialization="clone"``

Both run the dev shape (~14.9M total / 7.6M active) on the real TinyStories stream,
400 steps, three mutation events at 100/200/300, same seed and same data schedule. The
primary is the **random** arm: it mutates on a seeded schedule and never probes, so its
treatment effect cannot be an artifact of how the probe ranks anything.

**This block is not bit-reproducible in the section-6 sense.** It runs on a GPU, and
section 6's fixed-thread-count contract is a CPU statement. Reproduction here means
re-running the block and checking that the verdict is stable, not that the losses are
bit-identical.

Because a session of this shape takes a minute or more, the runner is resumable and
budget-aware: completed ``(profile, seed, setting)`` runs are recorded in a scratch JSONL
and skipped, an interrupted session resumes from its own checkpoints, and an invocation
stops cleanly once its wall-clock budget is spent. Re-run the same command to finish.

    PYTHONPATH=src python scripts/real_path_block.py --run --profile 5.38.1
    PYTHONPATH=src python scripts/real_path_block.py --summarize --profile 5.38.1
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
SUMMARIES = ROOT / "artifacts" / "study-summaries"

ALPHA = 0.05
# The sign-flip permutation test is the declared statistic for every primary; the exact
# sign test is reported beside it (section 5.31.1).
N_PERMUTATIONS = 20_000
PERMUTATION_SEED = 5_380_001

REPORT_DEPTH = "dev (14.9M total / 7.6M active)"

SETTINGS = {
    "base": {},
    "clone": {
        "preserve_router_on_rebuild": True,
        "new_expert_initialization": "clone",
    },
}

# Two registered blocks share this runner. Seeds are disjoint between them (and per
# profile the run key includes the profile, so a seed can never be reused by accident),
# and the section 5.36 smoke's 15201-15203 are disclosed as spent rather than reused.
PROFILES = {
    "5.38": {
        "seeds": (15210, 15211, 15212, 15213, 15214, 15215),
        "arms": ("random", "adaptive"),
        "label": "both arms",
        "summary": "real-path-clone-dev.json",
        "primaries": ("random_clone_vs_base", "adaptive_clone_vs_base"),
        # Recorded as executed, not as the live default: these sessions ran while the
        # field default was still True, before section 5.37.1 flipped it. The flag is
        # trajectory-inert (same section), so it cannot affect the contrast either way,
        # but the record has to say what actually ran.
        "activation_checkpointing_used": True,
    },
    # Section 5.38.1: the clean arm only, sized from the dispersion 5.38 measured
    # (sd 0.0884 against a planning effect of 0.0742), because 5.38's n=6 was sized
    # from the three-seed pilot's 0.023 and could not resolve an effect that size.
    "5.38.1": {
        "seeds": tuple(range(15220, 15232)),
        "arms": ("random",),
        "label": "random arm only",
        "summary": "real-path-clone-dev-n12.json",
        "primaries": ("random_clone_vs_base",),
        "activation_checkpointing_used": False,
    },
}
DEFAULT_PROFILE = "5.38"


def profile(name: str) -> dict:
    if name not in PROFILES:
        raise SystemExit(f"unknown profile {name!r}; have {sorted(PROFILES)}")
    return PROFILES[name]


def session_config(seed: int, setting: str, profile_name: str) -> V0SessionConfig:
    """The registered configuration for one (seed, setting) cell."""
    if not REAL_TRAIN.exists():
        raise SystemExit(f"real token stream missing: {REAL_TRAIN}")
    return V0SessionConfig(
        seed=seed,
        arms=profile(profile_name)["arms"],
        data_path=str(REAL_TRAIN),
        val_data_path=str(REAL_VAL),
        **SETTINGS[setting],
    )


def config_record(profile_name: str) -> dict:
    """The registered knobs, taken from the base config so it cannot go stale."""
    config = session_config(profile(profile_name)["seeds"][0], "base", profile_name)
    fields = (
        "arms", "vocab_size", "context_size", "hidden_size", "depth",
        "attention_heads", "expert_count", "active_experts", "expert_width",
        "batch_size", "total_steps", "learning_rate", "mutation_interval",
        "mutation_budget", "probe_candidate_cap", "probe_batch_count",
        "probe_candidate_policy", "probe_selection", "dtype",
        "activation_checkpointing",
    )
    record = {name: getattr(config, name) for name in fields}
    # What the block's sessions actually ran under, which for a flag whose default has
    # since moved is not the same as reading it live.
    record["activation_checkpointing"] = profile(profile_name)[
        "activation_checkpointing_used"
    ]
    record["treatments"] = {name: dict(values) for name, values in SETTINGS.items()}
    record["report_section"] = profile_name
    record["depth_label"] = REPORT_DEPTH
    return record


def config_hash(record: dict) -> str:
    blob = json.dumps(record, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_runs(profile_name: str) -> dict[tuple[int, str], dict]:
    done: dict[tuple[int, str], dict] = {}
    if RUNS.exists():
        for line in RUNS.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                # Rows written before the runner gained profiles are 5.38's: it was
                # the only block when they were recorded, and its seeds are disjoint
                # from the later profile's.
                if row.get("profile", "5.38") != profile_name:
                    continue
                done[(row["seed"], row["setting"])] = row
    return done


def record_run(seed: int, setting: str, profile_name: str) -> dict:
    """Run (or resume) one cell and return its record."""
    config = session_config(seed, setting, profile_name)
    checkpoint_dir = SESSIONS / profile_name / setting
    started = time.perf_counter()
    interrupted = False
    try:
        result = run_v0_session(config, checkpoint_dir=checkpoint_dir)
    except V0SessionInterrupted:
        interrupted = True
        result = None
    wall = time.perf_counter() - started
    if interrupted or result is None:
        return {
            "profile": profile_name,
            "seed": seed,
            "setting": setting,
            "interrupted": True,
            "wall": wall,
        }
    row: dict = {
        "profile": profile_name,
        "seed": seed,
        "setting": setting,
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
                    "layer_id": event.get("layer_id"),
                    "expert_id": event.get("expert_id"),
                }
                for event in arm.mutation_history
            ],
        }
    return row


def run_block(budget_seconds: float, profile_name: str) -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SESSIONS / profile_name).mkdir(parents=True, exist_ok=True)
    done = load_runs(profile_name)
    deadline = time.perf_counter() + budget_seconds
    finished_before = len(done)
    seeds = profile(profile_name)["seeds"]
    for seed in seeds:
        for setting in SETTINGS:
            if (seed, setting) in done:
                continue
            if time.perf_counter() >= deadline:
                print(
                    f"budget spent after {len(load_runs(profile_name)) - finished_before} "
                    "new run(s); re-run the same command to continue"
                )
                return 0
            print(f"[{time.strftime('%H:%M:%S')}] {setting} seed {seed} ...", flush=True)
            row = record_run(seed, setting, profile_name)
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
    print(f"block complete: {len(load_runs(profile_name))} run(s) recorded in {RUNS}")
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


def compare(label: str, base: list[float], clone: list[float], alpha: float) -> dict:
    """Paired comparison, helpful direction = clone lower.

    The deltas are ``base - clone``, so a *positive* mean is the helpful direction and
    ``passes`` requires it. (An earlier revision of this function tested ``mean < 0`` -
    a sign inherited from the section 5.34 studies, where the delta was treatment minus
    reference. The direction error was found after the section 5.38 block ran, is
    corrected here, and changes only the pass predicate: the deltas, wins, sign p and
    permutation p are computed before it and are unaffected.)
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
        "sample_sd": sd,
        "standard_error": se,
        # Descriptive only. The project's declared statistics are the exact sign test and
        # the sign-flip permutation test; no t p-value is recorded, so none is invented
        # here (and nothing depends on scipy being installed).
        "t": t,
        "wins": wins,
        "sign_test_min_wins": bar,
        "sign_test_p": sign_p,
        "sign_test_passes": sign_passes,
        "permutation_p": perm_p,
        "passes": bool(mean > 0 and perm_p <= alpha),
        "per_seed_deltas": deltas,
    }


def summarize(profile_name: str) -> int:
    spec = profile(profile_name)
    seeds, arms, primaries = spec["seeds"], spec["arms"], spec["primaries"]
    rows = load_runs(profile_name)
    missing = [
        (seed, setting) for seed in seeds for setting in SETTINGS
        if (seed, setting) not in rows
    ]
    if missing:
        raise SystemExit(f"incomplete block, missing {missing}")

    def series(arm: str, key: str, setting: str) -> list[float]:
        return [rows[(seed, setting)]["arms"][arm][key] for seed in seeds]

    per_seed = []
    for seed in seeds:
        row: dict = {"seed": seed}
        for arm in arms:
            row[f"{arm}_base"] = rows[(seed, "base")]["arms"][arm]["final_loss"]
            row[f"{arm}_clone"] = rows[(seed, "clone")]["arms"][arm]["final_loss"]
            row[f"{arm}_events_base"] = len(rows[(seed, "base")]["arms"][arm]["events"])
            row[f"{arm}_events_clone"] = len(rows[(seed, "clone")]["arms"][arm]["events"])
        per_seed.append(row)

    per_primary_alpha = ALPHA / len(primaries)
    comparisons = {}
    for arm in arms:
        name = f"{arm}_clone_vs_base"
        comparisons[name] = compare(
            name,
            series(arm, "final_loss", "base"),
            series(arm, "final_loss", "clone"),
            per_primary_alpha,
        )
    if "adaptive" in arms and "random" in arms:
        gap_base = [row["adaptive_base"] - row["random_base"] for row in per_seed]
        gap_clone = [row["adaptive_clone"] - row["random_clone"] for row in per_seed]
        comparisons["adaptive_minus_random_gap"] = compare(
            "adaptive_minus_random_gap", gap_base, gap_clone, ALPHA
        )

    events_match = all(
        row[f"{arm}_events_base"] == row[f"{arm}_events_clone"] == 3
        for row in per_seed
        for arm in arms
    )

    def sequences(arm: str, setting: str) -> list[tuple]:
        return [
            tuple(
                (event["operation"], event["layer_id"], event["expert_id"])
                for event in rows[(seed, setting)]["arms"][arm]["events"]
            )
            for seed in seeds
        ]

    settings_differ = all(
        rows[(seed, "base")]["arms"][arm]["architecture_hash"]
        != rows[(seed, "clone")]["arms"][arm]["architecture_hash"]
        or rows[(seed, "base")]["arms"][arm]["final_loss"]
        != rows[(seed, "clone")]["arms"][arm]["final_loss"]
        for seed in seeds
        for arm in arms
    )
    treatment_applied = all(
        rows[(seed, "clone")]["arms"][arm]["mutation_count"] == 3
        for seed in seeds
        for arm in arms
    )
    # Added after the section 5.38 block ran: structural diagnostics, not
    # verdict-bearing. They are what makes the two primaries readable, because they say
    # whether the treatment changed the *rebuild* alone or also *which mutations were
    # selected*.
    sequence_identical = {
        arm: sum(
            1
            for index, seed in enumerate(seeds)
            if sequences(arm, "base")[index] == sequences(arm, "clone")[index]
        )
        for arm in arms
    }
    events_changed_by_arm = {arm: len(seeds) - sequence_identical[arm] for arm in arms}

    def grows(arm: str, setting: str) -> int:
        return sum(
            1
            for sequence in sequences(arm, setting)
            for event in sequence
            if event[0] == "grow"
        )

    grows_by_arm_and_setting = {
        arm: {setting: grows(arm, setting) for setting in SETTINGS} for arm in arms
    }

    built_in_checks = {
        "block_matches_declared_n": len(rows) == len(seeds) * len(SETTINGS),
        "the_two_settings_are_not_redundant": settings_differ,
        "both_settings_ran_three_events": events_match,
        "the_treatment_reached_both_arms": treatment_applied,
        "every_seed_is_paired": all(
            (seed, setting) in rows for seed in seeds for setting in SETTINGS
        ),
        # The clean-comparison guarantee: on the arm that never probes, only the
        # rebuild differs, so its contrast cannot be a selection effect.
        "the_random_arms_events_are_setting_independent": (
            events_changed_by_arm["random"] == 0
        ),
    }

    if len(primaries) == 1:
        outcome = {"primary_passes": comparisons[primaries[0]]["passes"]}
    else:
        all_pass = all(comparisons[name]["passes"] for name in primaries)
        outcome = {
            "branch_a_all_primaries_pass": all_pass,
            "branch_b_random_only": (
                comparisons["random_clone_vs_base"]["passes"]
                and not comparisons["adaptive_clone_vs_base"]["passes"]
            ),
            "branch_c_adaptive_only": (
                comparisons["adaptive_clone_vs_base"]["passes"]
                and not comparisons["random_clone_vs_base"]["passes"]
            ),
            "branch_d_neither": not any(
                comparisons[name]["passes"] for name in primaries
            ),
        }
    record = config_record(profile_name)
    payload = {
        "record_type": "study-summary",
        "kind": f"real-path-clone-dev-{profile_name}",
        "report_section": profile_name,
        "protocol": (
            "paired per seed: base vs clone on the dev shape and the real TinyStories "
            f"stream, 400 steps, three events, {spec['label']} in one session"
        ),
        "config": record,
        "config_hash": config_hash(record),
        "declared_n": len(seeds),
        "seeds": list(seeds),
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
    target = SUMMARIES / spec["summary"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {target}")
    for name, row in comparisons.items():
        print(
            f"  {name:28s} mean {row['mean_delta']:+.5f}  sd {row['sample_sd']:.4f}  "
            f"t {row['t']:+.2f}  wins {row['wins']}/{row['n']} "
            f"(bar {row['sign_test_min_wins']})  sign p {row['sign_test_p']:.4f}  "
            f"perm p {row['permutation_p']:.5f}  "
            f"{'PASS' if row['passes'] else 'FAIL'}"
        )
    for name, value in built_in_checks.items():
        print(f"  check {name:38s} {value}")
    print(f"  outcome {outcome}")
    return 0


def pool(profile_names: tuple[str, ...] = ("5.38", "5.38.1")) -> int:
    """Report the two blocks together, post hoc, as a secondary rather than a verdict.

    Both blocks ran the same prescription on disjoint fresh seeds, so their per-seed
    deltas can be read together and that makes the effect estimate much tighter than
    either block alone. It is **not** a pre-registered test: section 5.38's seeds sized
    section 5.38.1, so the second block was designed with knowledge of the first. It is
    recorded as a declared secondary, with both blocks' own verdicts left standing.
    """
    deltas: list[float] = []
    per_block = {}
    for name in profile_names:
        for seed in profile(name)["seeds"]:
            row = load_runs(name)[(seed, "base")]["arms"]["random"]["final_loss"]
            clone = load_runs(name)[(seed, "clone")]["arms"]["random"]["final_loss"]
            deltas.append(row - clone)
        per_block[name] = len(profile(name)["seeds"])
    # ``compare`` takes paired vectors; the pooled statistic is the same arithmetic on a
    # single sample, so it is computed here rather than faked as a paired comparison.
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas)
    se = sd / math.sqrt(len(deltas))
    wins = sum(1 for delta in deltas if delta > 0)
    sign_p = _binomial_survival(wins, len(deltas))
    perm_p = _permutation_p(deltas, random.Random(PERMUTATION_SEED))
    t = mean / se
    payload = {
        "record_type": "study-summary",
        "kind": "real-path-clone-dev-pooled",
        "report_section": "5.38.2",
        "post_hoc": True,
        "caveat": (
            "Declared secondary, not a verdict: 5.38's seeds sized 5.38.1, so this is a "
            "post-hoc meta-analysis of two registered blocks over disjoint fresh seeds. "
            "Both blocks' own pre-registered verdicts stand as recorded."
        ),
        "blocks": per_block,
        "n": len(deltas),
        "per_seed_deltas": deltas,
        "mean_delta": mean,
        "sample_sd": sd,
        "standard_error": se,
        "t": t,
        "wins": wins,
        "sign_test_min_wins": _min_wins(len(deltas), ALPHA),
        "sign_test_p": sign_p,
        "permutation_p": perm_p,
        "sign_test_passes": wins >= _min_wins(len(deltas), ALPHA),
        "mean_test_passes": bool(mean > 0 and perm_p <= ALPHA),
        "the_two_statistics_disagree": (wins >= _min_wins(len(deltas), ALPHA))
        != (perm_p <= ALPHA),
    }
    target = SUMMARIES / "real-path-clone-dev-pooled.json"
    target.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {target}")
    print(
        f"  pooled n={len(deltas)}  mean {mean:+.5f}  sd {sd:.4f}  se {se:.4f}  t {t:+.2f}  "
        f"mean-test p {perm_p:.5f}  wins {wins}/{len(deltas)}  sign p {sign_p:.4f}"
    )
    print(f"  disagreeing statistics: {payload['the_two_statistics_disagree']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run the block (resumable)")
    parser.add_argument(
        "--pool",
        action="store_true",
        help="write the post-hoc pooled secondary across both profiles",
    )
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=450.0,
        help="wall-clock budget for this invocation before stopping cleanly",
    )
    parser.add_argument("--summarize", action="store_true", help="write the summary")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="5.38 | 5.38.1")
    args = parser.parse_args()
    if args.run:
        return run_block(args.budget_seconds, args.profile)
    if args.summarize:
        return summarize(args.profile)
    if args.pool:
        return pool()
    parser.error("pass --run, --summarize or --pool")


if __name__ == "__main__":
    raise SystemExit(main())
