"""Section 5.9 study: does the toy adaptive edge clear the do-nothing bar?

Runs the registered study-3 config as a THREE-arm block per seed
(``arms="random,adaptive,null"``, ``independent_arm_rng=True``, ``cpu``) on
the fresh block 70000-70383, sharded for parallel execution. Because arm
order keeps indices 0/1 on random/adaptive and the null arm consumes no
global torch RNG, the adaptive-vs-random comparison is by construction
identical to the two-arm block (pinned by test).

Usage (one shard per worker, then the verdict):

    PYTHONPATH=src python scripts/null_toy_study.py run --shard 0 --shards 20
    ...
    PYTHONPATH=src python scripts/null_toy_study.py verdict --shards 20

The verdict computes both pre-registered primaries from the same fresh run:

1. null-vs-random  (churn harm at toy scale)      - one-tailed sign test
2. adaptive-vs-null (the new bar for Gate 3)      - one-tailed sign test

each against the Bonferroni alpha/2 = 0.0025 bar, using the exact binomial
survival from ``act.evolution.study``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from act.evolution.experiment import ExperimentConfig, run_experiment  # noqa: E402
from act.evolution.registry import _stable_hash  # noqa: E402
from act.evolution.study import _binomial_survival, _gate_min_wins  # noqa: E402

SHARD_DIR = ROOT / "artifacts" / "toy-null-study"
# Section 5.9: study 3's registered depth-1 config, fresh block 70000-70383.
# Section 5.10: the section-5.6 depth-4 confirmation's config (study-3
# defaults with depth 4 and the spread candidate policy), fresh block
# 71000-71383.
PROFILES: dict[str, dict] = {
    "depth1": {
        "seeds": tuple(range(70000, 70384)),
        "overrides": {},
        "run_revision": "78186e4",
        "report_section": "5.9",
        "summary": "artifacts/study-summaries/null-control-depth1-384.json",
    },
    "depth4": {
        "seeds": tuple(range(71000, 71384)),
        "overrides": {"depth": 4, "probe_candidate_policy": "spread"},
        "run_revision": "a9632a6",
        "report_section": "5.10",
        "summary": "artifacts/study-summaries/null-control-depth4-384.json",
    },
    # Section 5.12: the missing 2x2 cells. Sections 5.9 (depth 1 + prefix)
    # and 5.10 (depth 4 + spread) confound depth with the candidate policy;
    # these two cells complete the factorial design on fresh blocks.
    "depth1-spread": {
        "seeds": tuple(range(72000, 72384)),
        "overrides": {"probe_candidate_policy": "spread"},
        "run_revision": "c086685",
        "report_section": "5.12",
        "summary": "artifacts/study-summaries/null-control-depth1-spread-384.json",
    },
    "depth4-prefix": {
        "seeds": tuple(range(73000, 73384)),
        "overrides": {"depth": 4},
        "run_revision": "c086685",
        "report_section": "5.12",
        "summary": "artifacts/study-summaries/null-control-depth4-prefix-384.json",
    },
    # Section 5.15: the depth dose-response. The 2x2 established that DEPTH
    # separates churn suppression (depth 1) from value creation (depth 4);
    # these two cells place depths 2 and 3 on the same axis (prefix policy,
    # the registered default), locating where the onset begins. Fresh blocks.
    "depth2": {
        "seeds": tuple(range(74000, 74384)),
        "overrides": {"depth": 2},
        "run_revision": "bbad439",
        "report_section": "5.15",
        "summary": "artifacts/study-summaries/null-control-depth2-384.json",
    },
    "depth3": {
        "seeds": tuple(range(75000, 75384)),
        "overrides": {"depth": 3},
        "run_revision": "bbad439",
        "report_section": "5.15",
        "summary": "artifacts/study-summaries/null-control-depth3-384.json",
    },
    # Section 5.48: the width covariate at the onset. The dose-response (5.15)
    # brackets the onset to (3, 4] at hidden_size 24; these two fresh cells run
    # the same registered config at **width 28**, the closest integer width
    # whose parameter count equals depth 4 x width 24 (31,192 vs 30,672 total,
    # +1.7%; +2.0% active; pinned by a test). Width-d3-equal is therefore the
    # equal-parameter partner of the committed above-onset cell, and the arm's
    # primaries are the pre-registered per-cell bars - see `width_verdict` for
    # the four fixed branches, and report section 5.48 for the design.
    "width-d3-equal": {
        "seeds": tuple(range(76000, 76576)),
        "overrides": {"depth": 3, "hidden_size": 28},
        "run_revision": None,  # set to the pre-registration commit before the run
        "report_section": "5.48",
        "summary": "artifacts/study-summaries/width-d3-equal-576.json",
    },
    "width-d4-equal": {
        "seeds": tuple(range(77000, 77576)),
        "overrides": {"depth": 4, "hidden_size": 28},
        "run_revision": None,  # set to the pre-registration commit before the run
        "report_section": "5.48",
        "summary": "artifacts/study-summaries/width-d4-equal-576.json",
    },
}
REGISTERED = dict(
    hidden_size=24,
    vocab_size=32,
    depth=1,
    expert_count=4,
    active_experts=2,
    batch_size=2,
    sequence_length=12,
    steps=48,
    mutation_interval=12,
    mutation_budget=3,
    learning_rate=0.0003,
    validation_batches=4,
    adaptive_policy="probe_loss",
    adaptive_expert_metric="random",
    probe_candidate_cap=4,
    probe_batch_count=2,
    independent_arm_rng=True,
    device="cpu",
)
ALPHA = 0.05
# The width arm's cells (section 5.48), in the order the branches read them:
# the equal-parameter partner first, the above-onset depth second.
WIDTH_PROFILES = ("width-d3-equal", "width-d4-equal")


def run_shard(shard: int, shards: int, profile: str) -> Path:
    if shards < 1 or not 0 <= shard < shards:
        raise SystemExit("--shard must be in [0, --shards)")
    if profile not in PROFILES:
        raise SystemExit(f"--profile must be one of {sorted(PROFILES)}")
    spec = PROFILES[profile]
    directory = SHARD_DIR / profile
    path = directory / f"shard{shard:02d}.json"
    if path.exists():
        print(f"shard {shard}: already complete ({path.name})")
        return path
    experiments = []
    for seed in spec["seeds"][shard::shards]:
        config = ExperimentConfig(
            **{**REGISTERED, **spec["overrides"], "arms": "random,adaptive,null"}
        )
        result = run_experiment(replace(config, seed=seed))
        experiments.append(
            {
                "seed": seed,
                "losses": {arm.mode: arm.final_loss for arm in result.arms},
                "mutations": {arm.mode: arm.mutation_count for arm in result.arms},
                "search_flops": {arm.mode: arm.search_flops for arm in result.arms},
            }
        )
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"shard": shard, "shards": shards, "profile": profile, "experiments": experiments}
    path.write_text(json.dumps(payload) + "\n")
    print(f"shard {shard}: wrote {len(experiments)} seeds -> {path.name}")
    return path


def load_shards(shards: int, profile: str) -> list[dict]:
    experiments: list[dict] = []
    seeds = PROFILES[profile]["seeds"]
    for shard in range(shards):
        path = SHARD_DIR / profile / f"shard{shard:02d}.json"
        if not path.exists():
            raise SystemExit(f"missing shard: {path}")
        experiments.extend(json.loads(path.read_text())["experiments"])
    seen = {exp["seed"] for exp in experiments}
    if seen != set(seeds):
        raise SystemExit(f"shard set covers {len(seen)}/{len(seeds)} seeds; verdict withheld")
    return sorted(experiments, key=lambda exp: exp["seed"])


def _check_built_ins(experiments: list[dict]) -> None:
    """The registered per-seed checks every profile shares (5.9)."""
    ok = all(
        exp["mutations"] == {"random": 3, "adaptive": 3, "null": 0}
        and exp["search_flops"]["null"] == 0
        for exp in experiments
    )
    if not ok:
        raise SystemExit("built-in check failed: unexpected mutation counts or null search FLOPs")
    print(
        f"built-in check: mutations 3/3/0 and zero null search FLOPs in "
        f"{len(experiments)}/{len(experiments)} seeds: OK"
    )


def _sign_test(deltas: list[float]) -> tuple[int, float, float]:
    """One-sided exact sign test: (wins, mean delta, binomial p-value)."""
    wins = sum(delta < 0 for delta in deltas)
    p = _binomial_survival(wins, len(deltas))
    return wins, sum(deltas) / len(deltas), p


def width_verdict(shards: int, alpha: float) -> int:
    """Section 5.48's pre-registered primaries and four fixed branches.

    Two fresh cells at width 28: depth 3 (the equal-parameter partner of the
    committed depth-4 x width-24 cell) and depth 4. Both primaries are the
    dose-response's own test - the one-sided exact sign test of adaptive-vs-null
    per-seed wins at the Bonferroni bar alpha/2 - and a cell counts as
    *creating value* only when the wins clear the bar **and** the mean delta is
    negative. The conjunction is part of the pre-registration because a
    win-rate crossing can be a tail effect (depth 1 is exactly that: 117/384
    wins while the adaptive arm is +0.0508 worse on average), and the six
    committed cells never disagreed on the sign. The branches are fixed in
    advance so no outcome can be re-narrated after the fact:

    - P1 and P2: the onset is a **capacity** onset - a below-onset depth
      creates value at the above-onset cell's parameter count, and the wider
      above-onset cell keeps its own.
    - P1 only: widening creates value below the onset and destroys it above -
      a depth x width interaction, not a monotone capacity story.
    - P2 only: the onset is depth-like at this width; widening does not rescue
      a below-onset substrate.
    - neither: width is harmful at both depths here, and the depth axis is not
      capacity - the width covariate is closed for this search space.
    """
    cells: dict[str, dict[str, float | bool]] = {}
    for profile in WIDTH_PROFILES:
        experiments = load_shards(shards, profile)
        _check_built_ins(experiments)
        deltas = [
            exp["losses"]["adaptive"] - exp["losses"]["null"]
            for exp in experiments
        ]
        wins, mean, p = _sign_test(deltas)
        bar = _gate_min_wins(len(deltas), alpha / 2)
        creates = wins >= bar and mean < 0
        cells[profile] = {
            "wins": wins,
            "n": len(deltas),
            "mean": mean,
            "binomial_p": p,
            "min_wins_bonferroni": bar,
            "creates_value": creates,
        }
        print(
            f"{profile}: adaptive-vs-null wins {wins}/{len(deltas)} (bar {bar}), "
            f"mean {mean:+.6f}, p {p:.2e} -> creates value: {creates}"
        )
    p1 = bool(cells[WIDTH_PROFILES[0]]["creates_value"])
    p2 = bool(cells[WIDTH_PROFILES[1]]["creates_value"])
    branches = {
        (True, True): (
            "P1+P2 PASS: the onset is a capacity onset - depth and width trade "
            "off at matched parameters"
        ),
        (True, False): (
            "P1 only: widening creates value below the onset and destroys it "
            "above - a depth x width interaction"
        ),
        (False, True): (
            "P2 only: the onset is depth-like at this width; widening does not "
            "rescue a below-onset substrate"
        ),
        (False, False): (
            "both FAIL: width is harmful at both depths here; the depth axis "
            "is not capacity"
        ),
    }
    print(f"\nOUTCOME (alpha={alpha}, Bonferroni alpha/2={alpha / 2}): {branches[(p1, p2)]}")
    return 0


def verdict(shards: int, profile: str) -> int:
    experiments = load_shards(shards, profile)
    n = len(experiments)
    _check_built_ins(experiments)

    null_deltas = [exp["losses"]["null"] - exp["losses"]["random"] for exp in experiments]
    bar_deltas = [exp["losses"]["adaptive"] - exp["losses"]["null"] for exp in experiments]
    rand_deltas = [exp["losses"]["adaptive"] - exp["losses"]["random"] for exp in experiments]

    sign_test = _sign_test
    n_wins, n_mean, n_p = sign_test(null_deltas)
    b_wins, b_mean, b_p = sign_test(bar_deltas)
    a_wins, a_mean, a_p = sign_test(rand_deltas)
    n_min = _gate_min_wins(n, ALPHA / 2)
    b_min = _gate_min_wins(n, ALPHA / 2)

    print(f"\nprimary 1  null-vs-random:   wins {n_wins}/{n}, mean {n_mean:+.6f}, p {n_p:.2e} (bar {n_min})")
    print(f"primary 2  adaptive-vs-null: wins {b_wins}/{n}, mean {b_mean:+.6f}, p {b_p:.2e} (bar {b_min})")
    print(f"secondary  adaptive-vs-random: wins {a_wins}/{n}, mean {a_mean:+.6f}, p {a_p:.2e}")
    print(f"per-arm mean losses: random {sum(e['losses']['random'] for e in experiments)/n:.6f}, "
          f"adaptive {sum(e['losses']['adaptive'] for e in experiments)/n:.6f}, "
          f"null {sum(e['losses']['null'] for e in experiments)/n:.6f}")

    p1 = n_wins >= n_min
    p2 = b_wins >= b_min
    if p1 and p2:
        outcome = "1+2 PASS: the toy edge is real beyond the pool effect AND churn suppression"
    elif p1:
        outcome = "1 PASS / 2 FAIL: toy Gate 3 passes are substantially churn suppression"
    elif p2:
        outcome = "1 FAIL / 2 PASS: no toy churn harm; random-churn comparisons remain meaningful"
    else:
        outcome = "both FAIL: the adaptive arm does not clear the do-nothing bar"
    print(f"\nOUTCOME (alpha={ALPHA}, Bonferroni alpha/2={ALPHA / 2}): {outcome}")
    return 0


def _comparison(deltas: list[float], alpha: float) -> dict:
    """One pre-registered primary: sign-test statistics at the Bonferroni bar."""
    wins = sum(delta < 0 for delta in deltas)
    n = len(deltas)
    mean = sum(deltas) / n
    stddev = (
        (sum((value - mean) ** 2 for value in deltas) / n) ** 0.5 if n > 1 else 0.0
    )
    p = _binomial_survival(wins, n)
    return {
        "wins": wins,
        "n": n,
        "mean_delta": mean,
        "delta_stddev": stddev,
        "binomial_p": p,
        "min_wins_bonferroni": _gate_min_wins(n, alpha / 2),
        "passes_bonferroni": p <= alpha / 2,
    }


def summarize(profile: str, alpha: float) -> dict:
    """Build the committed study summary from this profile's shard payloads.

    The summary is self-contained: the per-seed loss triples let every
    quoted statistic be recomputed exactly, so the report numbers stay
    machine-checked against ``artifacts/study-summaries/`` without
    committing the (large, gitignored) shard payloads.
    """
    spec = PROFILES[profile]
    experiments = load_shards(20 if profile == "depth1" else 20, profile)
    n = len(experiments)
    per_seed = [
        {"seed": exp["seed"], "losses": exp["losses"]} for exp in experiments
    ]
    comparisons = {
        "null_vs_random": _comparison(
            [exp["losses"]["null"] - exp["losses"]["random"] for exp in experiments],
            alpha,
        ),
        "adaptive_vs_null": _comparison(
            [exp["losses"]["adaptive"] - exp["losses"]["null"] for exp in experiments],
            alpha,
        ),
        "adaptive_vs_random": _comparison(
            [
                exp["losses"]["adaptive"] - exp["losses"]["random"]
                for exp in experiments
            ],
            alpha,
        ),
    }
    arm_means = {
        arm: sum(exp["losses"][arm] for exp in experiments) / n
        for arm in ("random", "adaptive", "null")
    }
    built_in = {
        "mutations_random_3": all(exp["mutations"]["random"] == 3 for exp in experiments),
        "mutations_adaptive_3": all(
            exp["mutations"]["adaptive"] == 3 for exp in experiments
        ),
        "mutations_null_0": all(exp["mutations"]["null"] == 0 for exp in experiments),
        "null_search_flops_zero": all(
            exp["search_flops"]["null"] == 0 for exp in experiments
        ),
    }
    config = {**REGISTERED, **spec["overrides"], "arms": "random,adaptive,null"}
    if not all(built_in.values()):
        raise SystemExit(f"built-in checks failed: {built_in}")
    p1 = comparisons["null_vs_random"]["passes_bonferroni"]
    p2 = comparisons["adaptive_vs_null"]["passes_bonferroni"]
    if p1 and p2:
        outcome = (
            "branch 1 (1+2 PASS): the edge is real beyond the pool effect and "
            "churn suppression"
        )
    elif p1:
        outcome = (
            "branch 2 (1 PASS / 2 FAIL): the pass is substantially churn suppression"
        )
    elif p2:
        outcome = "branch 3 (1 FAIL / 2 PASS): no churn harm on this substrate"
    else:
        outcome = "branch 4 (both FAIL): the adaptive arm does not clear the do-nothing bar"
    payload = {
        "record_type": "null_control_study",
        "protocol": "three-arm-null-control",
        "profile": profile,
        "report_section": spec["report_section"],
        "code_revision": spec["run_revision"],
        "alpha": alpha,
        "bonferroni_alpha": alpha / 2,
        "arms": ["random", "adaptive", "null"],
        "independent_arm_rng": True,
        "seeds": list(spec["seeds"]),
        "config": config,
        "per_seed": per_seed,
        "comparisons": comparisons,
        "arm_means": arm_means,
        "built_in_checks": built_in,
        "outcome": outcome,
        "summary_note": (
            "Do-nothing control at matched budget (report sections 5.9-5.10): "
            "null books zero search FLOPs and performs zero mutations; all three "
            "comparisons are paired on the same fresh run. Statistics recompute "
            "exactly from per_seed; p-values are exact one-tailed binomial."
        ),
    }
    payload["study_hash"] = _stable_hash(
        {
            "config": payload["config"],
            "seeds": payload["seeds"],
            "per_seed": per_seed,
            "alpha": alpha,
        }
    )
    payload["study_id"] = f"study_{payload['study_hash']}"
    payload["config_hash"] = _stable_hash(payload["config"])
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run one shard")
    run_p.add_argument("--shard", type=int, required=True)
    run_p.add_argument("--shards", type=int, required=True)
    run_p.add_argument("--profile", default="depth1", choices=sorted(PROFILES))
    verdict_p = sub.add_parser("verdict", help="aggregate shards and score the contract")
    verdict_p.add_argument("--shards", type=int, required=True)
    verdict_p.add_argument("--profile", default="depth1", choices=sorted(PROFILES))
    width_p = sub.add_parser(
        "width-verdict",
        help="section 5.48: score both width cells and print the pre-registered branch",
    )
    width_p.add_argument("--shards", type=int, required=True)
    width_p.add_argument("--alpha", type=float, default=ALPHA)
    sum_p = sub.add_parser("summary", help="write the committed study summary")
    sum_p.add_argument("--profile", default="depth1", choices=sorted(PROFILES))
    sum_p.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args()
    if args.command == "run":
        run_shard(args.shard, args.shards, args.profile)
        return 0
    if args.command == "summary":
        payload = summarize(args.profile, args.alpha)
        out = ROOT / PROFILES[args.profile]["summary"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"summary written: {out} ({payload['study_id']})")
        return 0
    if args.command == "width-verdict":
        return width_verdict(args.shards, args.alpha)
    return verdict(args.shards, args.profile)


if __name__ == "__main__":
    sys.exit(main())
