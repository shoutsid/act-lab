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
import math
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
        "bonferroni_divisor": 2.0,
        "summary": "artifacts/study-summaries/null-control-depth1-384.json",
    },
    "depth4": {
        "seeds": tuple(range(71000, 71384)),
        "overrides": {"depth": 4, "probe_candidate_policy": "spread"},
        "run_revision": "a9632a6",
        "report_section": "5.10",
        "bonferroni_divisor": 2.0,
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
        "bonferroni_divisor": 2.0,
        "summary": "artifacts/study-summaries/null-control-depth1-spread-384.json",
    },
    "depth4-prefix": {
        "seeds": tuple(range(73000, 73384)),
        "overrides": {"depth": 4},
        "run_revision": "c086685",
        "report_section": "5.12",
        "bonferroni_divisor": 2.0,
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
        "bonferroni_divisor": 2.0,
        "summary": "artifacts/study-summaries/null-control-depth2-384.json",
    },
    "depth3": {
        "seeds": tuple(range(75000, 75384)),
        "overrides": {"depth": 3},
        "run_revision": "bbad439",
        "report_section": "5.15",
        "bonferroni_divisor": 2.0,
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
        "run_revision": "38f3921",
        "report_section": "5.48",
        "per_cell_primary": True,
        "bonferroni_divisor": 2.0,
        "summary": "artifacts/study-summaries/width-d3-equal-576.json",
    },
    "width-d4-equal": {
        "seeds": tuple(range(77000, 77576)),
        "overrides": {"depth": 4, "hidden_size": 28},
        "run_revision": "38f3921",
        "report_section": "5.48",
        "per_cell_primary": True,
        "bonferroni_divisor": 2.0,
        "summary": "artifacts/study-summaries/width-d4-equal-576.json",
    },
    # Section 5.49: the **dose** form the 5.48 outcome left open. Width 48 at
    # both depths is deliberately *not* parameter-matched - 88,032 / 116,640
    # total parameters, 2.87x the above-onset cell - because the question is
    # whether enough capacity can buy value at a below-onset depth, which the
    # +1.7% contrast could not answer. See `dose_verdict` for the branches.
    "width-d3-dose": {
        "seeds": tuple(range(78000, 78576)),
        "overrides": {"depth": 3, "hidden_size": 48},
        "run_revision": "e5dcbf6",
        "report_section": "5.49",
        "per_cell_primary": True,
        "bonferroni_divisor": 2.0,
        "summary": "artifacts/study-summaries/width-d3-dose-576.json",
    },
    "width-d4-dose": {
        "seeds": tuple(range(79000, 79576)),
        "overrides": {"depth": 4, "hidden_size": 48},
        "run_revision": "e5dcbf6",
        "report_section": "5.49",
        "per_cell_primary": True,
        "bonferroni_divisor": 2.0,
        "summary": "artifacts/study-summaries/width-d4-dose-576.json",
    },
    # Section 5.50: the amplifier's **second** dose. 5.49 showed that above the
    # onset a 3.80x dose takes the adaptive-vs-null edge from 0.586 to 0.828
    # (mean -0.2097) while below the onset the same dose makes value creation
    # *worse* - but one dose cannot say whether capacity keeps paying or
    # plateaus, the shape 10.5(B)'s movable-width policy would need. These two
    # fresh cells run width 96 (11.15x / 14.82x the above-onset cell, 3.90x the
    # 5.49 dose above it) and add a third pre-registered primary - a one-sided
    # rank-sum of the depth-4 cells' per-seed deltas, the study family's one
    # asymptotic test. See `dose2_verdict` for the branches.
    "width-d3-dose2": {
        "seeds": tuple(range(80000, 80576)),
        "overrides": {"depth": 3, "hidden_size": 96},
        "run_revision": "05efa7b",
        "report_section": "5.50",
        "per_cell_primary": True,
        # Three primaries in 5.50 (two per-cell bars plus the rank-sum), so this
        # arm's cells are read at alpha/3. Declared here rather than left to the
        # command line: regenerating one of these summaries with the sibling
        # arms' alpha/2 silently writes bar 313 where the registration states
        # 315 - which is exactly what happened the first time this refactor was
        # exercised, and is the defect 5.50's own verdict run already recorded.
        "bonferroni_divisor": 3.0,
        "summary": "artifacts/study-summaries/width-d3-dose2-576.json",
    },
    # The depth-4 block was first drafted at 81000; the freshness test caught
    # that 81000-81383 is the section-5.17 reusability study's (the depth-3
    # block's 80000-80575 was free). Moved to the first free 576-window.
    "width-d4-dose2": {
        "seeds": tuple(range(84000, 84576)),
        "overrides": {"depth": 4, "hidden_size": 96},
        "run_revision": "05efa7b",
        "report_section": "5.50",
        "per_cell_primary": True,
        "bonferroni_divisor": 3.0,
        "summary": "artifacts/study-summaries/width-d4-dose2-576.json",
    },
    # Section 5.51: the limit 5.50 registered. At 14.8x the do-nothing arm
    # collapses, and at the fixed 48-step budget that is inseparable from
    # "this substrate needs more training than the budget gives it". This cell
    # runs the identical config at 4x the steps (192) and 4x the mutation
    # interval (48), which keeps the three mutational events (asserted by the
    # built-in checks) - so only the training amount and the events' timing
    # change. See `dose2_long_verdict` for the branches.
    "width-d4-dose2-long": {
        "seeds": tuple(range(87000, 87576)),
        "overrides": {
            "depth": 4,
            "hidden_size": 96,
            "steps": 192,
            "mutation_interval": 48,
        },
        "run_revision": "a590a7f",
        "report_section": "5.51",
        "per_cell_primary": True,
        "bonferroni_divisor": 2.0,
        "summary": "artifacts/study-summaries/width-d4-dose2-long-576.json",
    },
    # Section 5.52: the control 5.51 named as absent, in the same breath as its
    # result - "4x training moves the arms from 15.1-16.9 to 3.8-4.5 ... with no
    # 192-step width-24 control, no cross-width absolute claim is made". One
    # fresh cell runs the identical config with only the width changed (96 ->
    # 24), completing the 2x2 the collapse has to be read in - and every corner
    # of it is prefix-policy, so the widths are comparable at both budgets:
    #
    #   w24 @  48 steps  depth4-prefix         (5.12, 384 seeds)
    #   w96 @  48 steps  width-d4-dose2        (5.50, 576 seeds)
    #   w96 @ 192 steps  width-d4-dose2-long   (5.51, 576 seeds)
    #   w24 @ 192 steps  this cell             (576 seeds)
    #
    # See `long_w24_verdict` for the branches. Block 87600: the first 576-window
    # at or after 87600 that no other profile and no committed summary claims.
    "width-d4-long-w24": {
        "seeds": tuple(range(87600, 88176)),
        "overrides": {
            "depth": 4,
            "hidden_size": 24,
            "steps": 192,
            "mutation_interval": 48,
        },
        "run_revision": "f636990",
        "report_section": "5.52",
        "per_cell_primary": True,
        "bonferroni_divisor": 2.0,
        "summary": "artifacts/study-summaries/width-d4-long-w24-576.json",
    },
    # Section 5.53: the **expert-count** covariate at the onset - the second of
    # the two candidates 10.3(ii) names ("expert count, total compute"), and the
    # one that matters most here because expert grow/prune is the ONLY operation
    # the V0 search space can reach. The dose-response brackets the onset to
    # (3, 4] at width 24 and expert_count 4; these two fresh cells raise the
    # expert count to 6 at each depth, holding width and active_experts.
    #
    # The depth-3 cell is the family's tightest equal-parameter partner yet:
    # 30,624 total parameters against the committed above-onset cell's 30,672
    # (-0.156%, where the width-28 partner was +1.695%) - and it buys *no*
    # compute, because a token still routes to two experts. Its active
    # (per-token) parameter count is 9,024, 21.3% BELOW the above-onset cell's
    # 11,472, where the width-28 partner matched both (+1.7% total, +2.0%
    # active). So this cell separates stored capacity from per-token compute for
    # the first time in the family.
    #
    # The probe's candidate pool does not move with the cell: at cap 4 with the
    # prefix policy the pool is [grow L0, prune L0E0, prune L0E1, prune L0E2] at
    # expert count 4 and at 6 alike (the cap binds in both - 12 and 18 feasible
    # candidates), so the treatment is the substrate's expert count and not the
    # candidate set the controller sees. Pinned by a test.
    #
    # See `experts_verdict` for the four fixed branches. Blocks 88176 and 89384:
    # the first free 576-windows at or after 88000 that no other profile and no
    # committed summary claims.
    "experts-d3-dose": {
        "seeds": tuple(range(88176, 88752)),
        "overrides": {"depth": 3, "expert_count": 6},
        "run_revision": "76d2134",
        "report_section": "5.53",
        "per_cell_primary": True,
        "bonferroni_divisor": 2.0,
        "summary": "artifacts/study-summaries/experts-d3-dose-576.json",
    },
    "experts-d4-dose": {
        "seeds": tuple(range(89384, 89960)),
        "overrides": {"depth": 4, "expert_count": 6},
        "run_revision": "76d2134",
        "report_section": "5.53",
        "per_cell_primary": True,
        "bonferroni_divisor": 2.0,
        "summary": "artifacts/study-summaries/experts-d4-dose-576.json",
    },
    # Section 5.54: the **total compute** covariate, the last of the three 10.3(ii)
    # names ("parameter count, layer width, expert count" were the first two; the
    # third is the budget). One cell, because three corners of the depth x budget
    # 2x2 at width 24 are already committed:
    #
    #   d3 @  48  depth3              (5.15, 384 seeds)  fails the bar
    #   d3 @ 192  this cell           (576 seeds)        the missing corner
    #   d4 @  48  depth4-prefix       (5.12, 384 seeds)  creates value
    #   d4 @ 192  width-d4-long-w24   (5.52, 576 seeds)  creates value
    #
    # The config is 5.52's with only the depth changed (4 -> 3): 192 steps and a
    # 48-step mutation interval, which keeps the three mutational events the
    # built-in checks assert. 4x training at depth 4 *flipped the churn regime*
    # (collapse -0.0701 -> +0.1638), so the arm asks what it does one depth below
    # the gate - the question "is the gate a compute threshold" in its sharpest
    # available form.
    #
    # See `long_d3_verdict` for the four fixed branches. Block 90000: the first
    # free 576-window at or after 90000 (5.52 holds 87600-88175 and 5.53 holds
    # 88176-88751 and 89384-89959).
    "depth3-long": {
        "seeds": tuple(range(90000, 90576)),
        "overrides": {"depth": 3, "steps": 192, "mutation_interval": 48},
        "run_revision": "64046da",
        "report_section": "5.54",
        "per_cell_primary": True,
        "bonferroni_divisor": 2.0,
        "summary": "artifacts/study-summaries/depth3-long-576.json",
    },
    # Section 5.55: the depth dose above the onset. One gene changed and nothing
    # else - the family's cleanest dose, and the only covariate that has never
    # been dosed on the axis that moves the gate.
    "depth5-dose": {
        "seeds": tuple(range(91384, 91960)),
        "overrides": {"depth": 5},
        "run_revision": "5fb7bd3",
        "report_section": "5.55",
        "per_cell_primary": True,
        "bonferroni_divisor": 2.0,
        "summary": "artifacts/study-summaries/depth5-dose-576.json",
    },
    # Section 5.56: the confirmation rung. Same one-gene dose as 5.55, one layer
    # further up - so a plateau or a reversal both have to be read as bounds on
    # the monotone-ladder claim rather than as noise.
    "depth6-dose": {
        "seeds": tuple(range(92384, 92960)),
        "overrides": {"depth": 6},
        "run_revision": "9f31495",
        "report_section": "5.56",
        "per_cell_primary": True,
        "bonferroni_divisor": 2.0,
        "summary": "artifacts/study-summaries/depth6-dose-576.json",
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
# The dose cells (section 5.49), same order and same reading.
DOSE_PROFILES = ("width-d3-dose", "width-d4-dose")
# The second-dose cells (section 5.50), same order and same reading. The
# continuation primary compares DOSE2_PROFILES[1] (the 14.8x cell) against the
# committed 3.8x rung, so the reference is named rather than spelled inline.
DOSE2_PROFILES = ("width-d3-dose2", "width-d4-dose2")
DOSE2_REFERENCE = "width-d4-dose"
# The long-budget cell (section 5.51) and the 48-step cell it is read against.
DOSE2_LONG_PROFILE = "width-d4-dose2-long"
DOSE2_LONG_REFERENCE = "width-d4-dose2"
# The long-budget width-24 control (section 5.52), plus the two corners of its
# 2x2 it is read against: the same-width shorter-budget cell (5.12's depth-4
# prefix cell, the family's 48-step width-24 prefix reference) and the
# same-budget wider cell (5.51's).
LONG_W24_PROFILE = "width-d4-long-w24"
LONG_W24_SAME_WIDTH = "depth4-prefix"
LONG_W24_SAME_BUDGET = DOSE2_LONG_PROFILE
# The expert-count arm (section 5.53): the second candidate covariate 10.3(ii)
# names, and the only gene the V0 search space can actually move
# (`feasible_expert_mutations` yields grow/prune and nothing else, section
# 10.4). The two cells and the committed cells they are read against.
EXPERTS_PROFILES = ("experts-d3-dose", "experts-d4-dose")
EXPERTS_BELOW_BASELINE = "depth3"
EXPERTS_ABOVE_BASELINE = "depth4-prefix"
EXPERTS_WIDTH_PARTNER = "width-d3-equal"
# The *total compute* covariate (section 5.54): 10.3(ii)'s last candidate, and
# the only corner of the depth x budget 2x2 at width 24 that is not yet measured.
# 4x training flips the churn regime at depth 4 (5.52 measured the collapse go
# -0.0701 -> +0.1638); this arm asks what it does one depth *below* the gate.
LONG_D3_PROFILE = "depth3-long"
LONG_D3_SAME_DEPTH = "depth3"
LONG_D3_ABOVE_LONG = "width-d4-long-w24"
LONG_D3_ABOVE_SHORT = "depth4-prefix"
# The *depth dose* above the onset (section 5.55): the one covariate that moves
# the gate, and the only one this family has never dosed. Width (24 -> 28 -> 96),
# expert count (4 -> 6) and budget (48 -> 192) each got a dose; depth has only
# its bracket - `depth2` (161/384, no gate) and `depth3` (188/384, no gate)
# below, `depth4-prefix` (225/384, gate) at the top - so "the gate is depth per
# se" currently rests on a single step with nothing measured above it.
DEPTH_DOSE_PROFILE = "depth5-dose"
DEPTH_DOSE_REFERENCE = "depth4-prefix"
# The *second* depth dose (section 5.56): one rung further up the ladder, asked
# as a confirmation rather than as an extension. 5.55 found the depth axis
# monotone with the whole 4 -> 5 gain in the churn term, but "the axis keeps
# paying" is a two-point slope above the gate; this arm makes it a three-point
# one, and its failure modes (a plateau, or a reversal) both bound the claim.
DEPTH6_PROFILE = "depth6-dose"
DEPTH6_REFERENCE = DEPTH_DOSE_PROFILE


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


def _cell_deltas(
    profile: str, shards: int, left: str = "adaptive", right: str = "null"
) -> list[float]:
    """One cell's per-seed deltas for a named pair of arms, from its shards."""
    return [
        exp["losses"][left] - exp["losses"][right]
        for exp in load_shards(shards, profile)
    ]


def _rank_sum_one_sided(a: list[float], b: list[float]) -> tuple[float, float]:
    """One-sided Mann-Whitney U, tie-corrected normal approximation.

    Section 5.50's third primary compares two *independent* cells (different
    fresh seed blocks), which is the one place in this study family no paired
    exact test can reach. The statistic tests whether ``a``'s deltas are
    stochastically smaller than ``b``'s (more negative = more value): average
    ranks for ties, the standard tie correction to the variance, and a
    continuity correction added toward the null in the lower tail. The
    registration states the level and the rehearsed sensitivity; this function
    is the one spelling both the verdict and the rehearsal call.
    """

    def _ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = average
            i = j + 1
        return ranks

    n_a, n_b = len(a), len(b)
    n = n_a + n_b
    ranks = _ranks(a + b)
    u = sum(ranks[:n_a]) - n_a * (n_a + 1) / 2
    mean = n_a * n_b / 2
    tie_sum = sum(rank * rank for rank in ranks)
    variance = (n_a * n_b / (n * (n - 1))) * (tie_sum - n * (n + 1) ** 2 / 4)
    if variance <= 0:  # every observation identical: no ordering information
        return u, 1.0
    z = (u + 0.5 - mean) / variance ** 0.5
    return u, 0.5 * math.erfc(-z / math.sqrt(2))


def _cell_stats(profile: str, shards: int, level: float) -> dict[str, float | bool]:
    """One width-arm cell's pre-registered statistic, from its shards.

    One spelling for every arm (5.48's matched contrast, 5.49's dose, 5.50's
    second dose): the built-in checks, the one-sided sign test, the bar at the
    caller's **per-primary level** and the conjunction that a cell creates
    value only when the wins clear the bar *and* the mean delta is negative.

    The caller passes the level its own registration states - alpha/2 for the
    two-primary arms, alpha/3 for 5.50's three - so the printed bar can never
    disagree with the declared family. (5.50's first verdict run passed alpha/3
    into an earlier version that halved again, printing bar 318 where the
    registration said 315: stricter than declared, and still the defect 5.16.1
    adjudicates, because a declaration whose bar and alpha disagree cannot be
    applied as written.)
    """
    experiments = load_shards(shards, profile)
    _check_built_ins(experiments)
    deltas = [
        exp["losses"]["adaptive"] - exp["losses"]["null"]
        for exp in experiments
    ]
    wins, mean, p = _sign_test(deltas)
    bar = _gate_min_wins(len(deltas), level)
    creates = wins >= bar and mean < 0
    print(
        f"{profile}: adaptive-vs-null wins {wins}/{len(deltas)} (bar {bar}), "
        f"mean {mean:+.6f}, p {p:.2e} -> creates value: {creates}"
    )
    return {
        "wins": wins,
        "n": len(deltas),
        "mean": mean,
        "binomial_p": p,
        "min_wins_bonferroni": bar,
        "creates_value": creates,
    }


def dose_verdict(shards: int, alpha: float) -> int:
    """Section 5.49's primaries and branches: the dose form 5.48 left open.

    Two fresh cells at width 48 (88,032 and 116,640 total parameters, 2.87x the
    above-onset cell) - deliberately *not* parameter-matched, because the
    question the +1.7% contrast could not answer is whether enough capacity can
    buy value at a below-onset depth at all. Same test, same conjunction, same
    bar as 5.48 (the two-primary level). Branches, fixed before the run:

    - P1 and P2: a dose of capacity buys value below the onset and preserves it
      above - the onset is capacity-driven above a threshold that lies between
      +1.7% and +280% parameters.
    - P1 only: a dose buys value below the onset and destroys it above - a
      depth x width interaction in dose form.
    - P2 only: depth stands even at 2.87x width - with 5.48, the width
      covariate is closed at this search space's reachable widths.
    - neither: width is harmful above the onset and useless below; the depth
      axis is not capacity in dose form.
    """
    cells = {profile: _cell_stats(profile, shards, alpha / 2) for profile in DOSE_PROFILES}
    p1 = bool(cells[DOSE_PROFILES[0]]["creates_value"])
    p2 = bool(cells[DOSE_PROFILES[1]]["creates_value"])
    branches = {
        (True, True): (
            "P1+P2 PASS: a dose of capacity buys value at a below-onset depth - "
            "the onset is capacity-driven above a threshold between +1.7% and "
            "+280% parameters"
        ),
        (True, False): (
            "P1 only: a dose buys value below the onset and destroys it above - "
            "a depth x width interaction in dose form"
        ),
        (False, True): (
            "P2 only: depth stands even at 2.87x width; with 5.48, the width "
            "covariate is closed at this space's reachable widths"
        ),
        (False, False): (
            "both FAIL: width is harmful above the onset and useless below; the "
            "depth axis is not capacity in dose form"
        ),
    }
    print(f"\nOUTCOME (alpha={alpha}, Bonferroni alpha/2={alpha / 2}): {branches[(p1, p2)]}")
    return 0


def dose2_verdict(shards: int, alpha: float) -> int:
    """Section 5.50's primaries and branches: the amplifier's second dose.

    Two fresh cells at width 96 (341,952 / 454,464 total parameters - 11.15x
    and 14.82x the above-onset cell, 3.90x the 5.49 dose above it). P1 and P2
    are the sibling cells' own test at this arm's Bonferroni level (three
    primaries: alpha/3, bar 315/576); P3 is the arm's new test - a one-sided
    rank-sum of the depth-4 cells' adaptive-vs-null deltas against the
    committed 3.80x cell, stochastic superiority meaning the bigger dose's
    deltas tend more negative. P3 was rehearsed on the committed adjacent rung
    before this arm ran: the same test on the 3.80x cell versus the width-28
    cell (a 2.84x parameter step) gives p = 4.5e-44, and on the +1.7% step
    gives p = 0.50, so it detects the measured dose step and is a dead null on
    the measured wash. Branches, fixed before the run:

    - P1 and P2: at 14.8x, capacity buys value at a below-onset depth - the
      onset is capacity-driven above a threshold between 3.8x and 14.8x.
    - P1 only: the largest dose buys value below the onset and destroys it
      above - the depth x width interaction inverts at high dose.
    - P2 only: depth stands at every reachable dose; value creation survives
      14.8x width. The dose form then reads P3.
    - neither: at 14.8x nothing creates value - capacity above the gate
      reverses at the fixed training budget.

    P3 is a rider on the P2-only branch, printed separately and never folded
    into the cell labels: a continuation of the amplifier is a claim about two
    cells, and the pre-registered test is the only thing allowed to carry it.
    The churn reading is secondary and stated as such (as in 5.49).
    """
    alpha_3 = alpha / 3
    cells = {profile: _cell_stats(profile, shards, alpha_3) for profile in DOSE2_PROFILES}
    p1 = bool(cells[DOSE2_PROFILES[0]]["creates_value"])
    p2 = bool(cells[DOSE2_PROFILES[1]]["creates_value"])
    u, p3 = _rank_sum_one_sided(
        _cell_deltas(DOSE2_PROFILES[1], shards),
        _cell_deltas(DOSE2_REFERENCE, shards),
    )
    continues = p3 <= alpha_3
    print(
        f"\nP3 (the amplifier continues): rank-sum {DOSE2_PROFILES[1]} vs "
        f"{DOSE2_REFERENCE}: U {u:.1f}, one-sided p {p3:.2e} "
        f"(bar alpha/3 = {alpha_3:.4f}) -> continues: {continues}"
    )
    for profile in DOSE2_PROFILES:
        churn = [
            exp["losses"]["random"] - exp["losses"]["null"]
            for exp in load_shards(shards, profile)
        ]
        wins, mean, p = _sign_test(churn)
        bar = _gate_min_wins(len(churn), alpha_3)
        print(
            f"churn (secondary, {profile}): random-vs-null wins {wins}/{len(churn)} "
            f"(bar {bar}), mean {mean:+.6f}, p {p:.2e} -> churn pays: {wins >= bar}"
        )
    branches = {
        (True, True): (
            "P1+P2 PASS: at 14.8x, capacity buys value at a below-onset depth - "
            "the onset is capacity-driven above a threshold between 3.8x and 14.8x"
        ),
        (True, False): (
            "P1 only: the largest dose buys value below the onset and destroys "
            "it above - the depth x width interaction inverts at high dose"
        ),
        (False, True): (
            "P2 only: depth stands at every reachable dose; value creation "
            "survives 14.8x width"
        ),
        (False, False): (
            "both FAIL: at 14.8x nothing creates value - capacity above the "
            "gate reverses at the fixed training budget"
        ),
    }
    print(f"\nOUTCOME (alpha={alpha}, Bonferroni alpha/3={alpha_3}): {branches[(p1, p2)]}")
    if (p1, p2) == (False, True):
        print(
            "dose form (P3, pre-registered rider): "
            + (
                "the amplifier continues - the 14.8x cell's deltas are "
                "stochastically smaller than the 3.8x cell's"
                if continues
                else "not distinguishable from a plateau at this level"
            )
        )
    return 0


def dose2_long_verdict(shards: int, alpha: float) -> int:
    """Section 5.51's primaries and branches: 5.50's collapse against the budget.

    5.50 registered a limit rather than only a result: at 14.8x the crossing
    arrives with the do-nothing arm collapsing, and at a fixed 48-step budget
    "capacity is a liability unless mutated at all" cannot be separated from
    "this substrate needs more training than the budget gives it". One fresh
    cell runs the identical config at 4x the steps and 4x the mutation interval,
    keeping the three mutational events. Two primaries at the sibling arms'
    alpha/2 (bar 313/576):

    - P1 (the collapse is the budget's): the longer cell's *collapse* - its
      null-vs-random deltas, the quantity 5.50 measured at `+1.574` - is
      stochastically *smaller* than the 48-step cell's: a one-sided rank-sum
      against the committed 5.50 cell, the same non-paired test 5.50 introduced
      and rehearsed on committed rungs.
    - P2 (value creation survives the budget): adaptive-vs-null clears the bar
      and the mean delta is negative (5.50's own per-cell conjunction).

    Branches, fixed before the run:

    - P1 and P2: the collapse was the budget and value creation survives it -
      5.50's crossing is not a fixed-budget artifact.
    - P1 only: the collapse was the budget and the advantage goes with it -
      5.50's crossing WAS the fixed-budget collapse, and the honest form of the
      width story is about trainability, not selection value.
    - P2 only: the collapse is structural at width 96 - 4x training does not
      repair the unmutated arm - and value creation rides it.
    - neither: the longer budget neither relieves the collapse nor keeps the
      advantage.

    The controller's own edge and the absolute levels are reported
    descriptively, as every cross-cell reading in this family is.
    """
    level = alpha / 2
    cell = _cell_stats(DOSE2_LONG_PROFILE, shards, level)
    p2 = bool(cell["creates_value"])
    # The collapse is measured in the same direction 5.50 quoted it: null minus
    # random, positive when churning beats doing nothing. "Relieved" means the
    # longer cell's collapse is the smaller one, so it is the first argument.
    collapse_long = _cell_deltas(DOSE2_LONG_PROFILE, shards, "null", "random")
    collapse_reference = _cell_deltas(DOSE2_LONG_REFERENCE, shards, "null", "random")
    u, p1 = _rank_sum_one_sided(collapse_long, collapse_reference)
    relief = p1 <= level
    print(
        f"\nP1 (the collapse is the budget's): rank-sum {DOSE2_LONG_PROFILE} vs "
        f"{DOSE2_LONG_REFERENCE} on null-vs-random: U {u:.1f}, one-sided p {p1:.2e} "
        f"(bar alpha/2 = {level:.4f}; means {sum(collapse_long) / len(collapse_long):+.6f} "
        f"vs {sum(collapse_reference) / len(collapse_reference):+.6f}) "
        f"-> collapse relieved: {relief}"
    )
    edge_long = _cell_deltas(DOSE2_LONG_PROFILE, shards, "adaptive", "random")
    edge_reference = _cell_deltas(DOSE2_LONG_REFERENCE, shards, "adaptive", "random")
    u_edge, p_edge = _rank_sum_one_sided(edge_long, edge_reference)
    print(
        f"edge (descriptive): adaptive-vs-random rank-sum vs the 48-step cell: "
        f"U {u_edge:.1f}, one-sided p {p_edge:.2e} "
        f"(mean {sum(edge_long) / len(edge_long):+.6f} vs "
        f"{sum(edge_reference) / len(edge_reference):+.6f})"
    )
    branches = {
        (True, True): (
            "P1+P2 PASS: the collapse was the budget and value creation survives "
            "it - 5.50's crossing is not a fixed-budget artifact"
        ),
        (True, False): (
            "P1 only: the collapse was the budget and the advantage goes with it - "
            "5.50's crossing WAS the fixed-budget collapse"
        ),
        (False, True): (
            "P2 only: the collapse is structural at width 96 - 4x training does "
            "not repair the unmutated arm - and value creation rides it"
        ),
        (False, False): (
            "both FAIL: the longer budget neither relieves the collapse nor keeps "
            "the advantage"
        ),
    }
    print(f"\nOUTCOME (alpha={alpha}, Bonferroni alpha/2={level}): {branches[(relief, p2)]}")
    return 0


def long_w24_verdict(shards: int, alpha: float) -> int:
    """Section 5.52's primaries and branches: 5.51's missing width control.

    5.51 stated its limit beside its result: at 14.8x the collapse halves at 4x
    the budget, but there is no 192-step *width-24* control, so its absolute
    levels (3.8-4.5 against 15.1-16.9) support no cross-width claim. One fresh
    cell runs the identical config with only the width changed (96 -> 24), which
    completes the prefix-lineage 2x2 - w24 and w96 at 48 and at 192 steps - and
    gives its cross-budget reference for free. Two primaries at the sibling
    arms' alpha/2 (bar 313/576):

    - P1 (the residual collapse is the wide substrate's): the width-24 cell's
      *collapse* - its null-vs-random deltas, the quantity 5.50 measured at
      +1.574 and 5.51 at +0.708 - is stochastically *smaller* than the width-96
      long cell's: a one-sided rank-sum between two independent fresh cells, the
      non-paired test 5.50 introduced and rehearsed on committed rungs.
    - P2 (value creation survives at width 24): adaptive-vs-null clears the bar
      and the mean delta is negative (the family's per-cell conjunction).

    Branches, fixed before the run:

    - P1 and P2: the collapse that survives 4x training is the wide substrate's
      and the narrow one creates value - so 5.51's relief is a *capacity*
      effect, and 5.50's dose ladder is a liability of capacity itself.
    - P1 only: the collapse is the wide substrate's, but width 24 does not
      create value at the longer budget - the budget is not what the narrow
      substrate was short of.
    - P2 only: the collapse is general across widths at 4x (both substrates
      churn away the unmutated arm), and the narrow cell still creates value.
    - neither: the longer budget neither relieves the narrow cell's collapse nor
      keeps its value.

    The 2x2 levels, the within-width relief and the controller's own edge are
    reported descriptively, as every cross-cell reading in this family is.
    """
    level = alpha / 2
    cell = _cell_stats(LONG_W24_PROFILE, shards, level)
    p2 = bool(cell["creates_value"])
    collapse_w24 = _cell_deltas(LONG_W24_PROFILE, shards, "null", "random")
    collapse_w96 = _cell_deltas(LONG_W24_SAME_BUDGET, shards, "null", "random")
    u, p1 = _rank_sum_one_sided(collapse_w24, collapse_w96)
    confined = p1 <= level
    print(
        f"\nP1 (the residual collapse is the wide substrate's): rank-sum "
        f"{LONG_W24_PROFILE} vs {LONG_W24_SAME_BUDGET} on null-vs-random: "
        f"U {u:.1f}, one-sided p {p1:.2e} (bar alpha/2 = {level:.4f}; means "
        f"{sum(collapse_w24) / len(collapse_w24):+.6f} vs "
        f"{sum(collapse_w96) / len(collapse_w96):+.6f}) "
        f"-> collapse confined to width 96: {confined}"
    )
    # Descriptive: the 2x2 those two cells are two corners of. The other two
    # corners are committed, and all four are prefix-policy, so the widths are
    # comparable at both budgets without a confound.
    short_w24 = _cell_deltas(LONG_W24_SAME_WIDTH, shards, "null", "random")
    short_w96 = _cell_deltas(DOSE2_LONG_REFERENCE, shards, "null", "random")
    for label, values in (
        (f"w24 @  48 ({LONG_W24_SAME_WIDTH})", short_w24),
        (f"w96 @  48 ({DOSE2_LONG_REFERENCE})", short_w96),
        (f"w24 @ 192 ({LONG_W24_PROFILE})", collapse_w24),
        (f"w96 @ 192 ({LONG_W24_SAME_BUDGET})", collapse_w96),
    ):
        print(
            f"  collapse {label}: mean {sum(values) / len(values):+.6f} "
            f"over {len(values)} seeds"
        )
    print(
        "  within-width relief (descriptive, the difference in differences): "
        f"w24 {sum(collapse_w24) / len(collapse_w24) - sum(short_w24) / len(short_w24):+.6f}, "
        f"w96 {sum(collapse_w96) / len(collapse_w96) - sum(short_w96) / len(short_w96):+.6f}"
    )
    edge_w24 = _cell_deltas(LONG_W24_PROFILE, shards, "adaptive", "random")
    edge_w96 = _cell_deltas(LONG_W24_SAME_BUDGET, shards, "adaptive", "random")
    u_edge, p_edge = _rank_sum_one_sided(edge_w24, edge_w96)
    print(
        f"edge (descriptive): adaptive-vs-random rank-sum vs the width-96 long "
        f"cell: U {u_edge:.1f}, one-sided p {p_edge:.2e} "
        f"(mean {sum(edge_w24) / len(edge_w24):+.6f} vs "
        f"{sum(edge_w96) / len(edge_w96):+.6f})"
    )
    branches = {
        (True, True): (
            "P1+P2 PASS: the collapse that survives 4x training is the wide "
            "substrate's, and width 24 creates value - 5.51's relief is a "
            "capacity effect"
        ),
        (True, False): (
            "P1 only: the collapse is the wide substrate's, but width 24 does "
            "not create value at the longer budget - the budget is not what the "
            "narrow substrate was short of"
        ),
        (False, True): (
            "P2 only: the collapse is general across widths at 4x, and the "
            "narrow cell still creates value"
        ),
        (False, False): (
            "both FAIL: the longer budget neither relieves the narrow cell's "
            "collapse nor keeps its value"
        ),
    }
    print(
        f"\nOUTCOME (alpha={alpha}, Bonferroni alpha/2={level}): "
        f"{branches[(confined, p2)]}"
    )
    return 0


def long_d3_verdict(shards: int, alpha: float) -> int:
    """Section 5.54's primaries and branches: the total-compute covariate.

    One fresh cell at width 24 and depth 3 with 192 steps and a 48-step mutation
    interval - 5.52's config with only the depth changed - completing the depth x
    budget 2x2 at the registered width. Three corners are committed: `depth3`
    (48 steps, fails the bar), `depth4-prefix` (48, creates value) and
    `width-d4-long-w24` (192, creates value). Two primaries at the sibling
    arms' alpha/2 (bar 313/576):

    - P1 (the below-onset cell creates value at 4x the budget): the family's
      per-cell conjunction. This is the question 10.3(ii)'s third covariate
      asks, and the one that decides whether the onset is a compute threshold.
    - P2 (4x training moves the churn *regime* below the onset too): a one-sided
      rank-sum on null-vs-random deltas against the committed 48-step cell of
      the same depth, larger meaning the longer budget has created the churn
      harm. Rehearsed on the one measured instance before this arm ran: at depth
      4 the same comparison gives U 28795, p 1.9e-84 on the measured flip
      (`-0.0701` -> `+0.1638`), so the test detects the regime change where the
      records say one exists.

    Branches, fixed before the run:

    - P1 and P2: **total compute is the covariate** - 4x training creates value
      below the onset and brings the churn regime with it. That would make the
      below-onset cells' failures a fixed-budget statement, and the onset a
      compute threshold rather than depth per se.
    - P1 only: 4x training creates value below the onset without creating a churn
      harm - the gate is compute, and the regime change belongs to depth 4.
    - P2 only: the budget reaches the *regime* below the onset (a churn harm
      appears, as it did at depth 4) but not the *gate* (no value) - so the budget
      is not what the below-onset cell is short of, and the gate is not compute.
    - neither: 4x training changes nothing below the onset - no churn harm, no
      value. 10.3(ii)'s last candidate is then closed too, and the onset is depth
      per se at every covariate this search space can measure.

    The 2x2's four corners and the controller's own edge are reported
    descriptively, as every cross-cell reading in this family is.
    """
    level = alpha / 2
    cell = _cell_stats(LONG_D3_PROFILE, shards, level)
    p1 = bool(cell["creates_value"])
    collapse_long = _cell_deltas(LONG_D3_PROFILE, shards, "null", "random")
    collapse_short = _cell_deltas(LONG_D3_SAME_DEPTH, shards, "null", "random")
    u, p2 = _rank_sum_one_sided(collapse_short, collapse_long)
    p2_passes = p2 <= level
    print(
        f"\nP2 (4x training moves the churn regime below the onset): rank-sum "
        f"{LONG_D3_PROFILE} vs {LONG_D3_SAME_DEPTH} on null-vs-random: "
        f"U {u:.1f}, one-sided p {p2:.2e} (bar alpha/2 = {level:.4f}; means "
        f"{sum(collapse_long) / len(collapse_long):+.6f} vs "
        f"{sum(collapse_short) / len(collapse_short):+.6f}) "
        f"-> regime moved: {p2_passes}"
    )
    branches = {
        (True, True): (
            "P1+P2 PASS: total compute is the covariate - 4x training creates "
            "value below the onset and brings the churn regime with it, so the "
            "below-onset failures are a fixed-budget statement and the onset is "
            "a compute threshold rather than depth per se"
        ),
        (True, False): (
            "P1 only: 4x training creates value below the onset without creating "
            "a churn harm - the gate is compute, and the regime change belongs "
            "to depth 4"
        ),
        (False, True): (
            "P2 only: the budget reaches the regime below the onset but not the "
            "gate - a churn harm appears where none was, and no value is "
            "created, so the budget is not what the below-onset cell is short of"
        ),
        (False, False): (
            "both FAIL: 4x training changes nothing below the onset; the "
            "compute covariate is closed too and the onset is depth per se at "
            "every covariate this search space can measure"
        ),
    }
    print(f"BRANCH: {branches[(p1, p2_passes)]}")
    # Descriptive: the 2x2 those four cells are, and the controller's edge.
    for label, profile in (
        ("depth 3 @  48 (committed)", LONG_D3_SAME_DEPTH),
        ("depth 3 @ 192 (this arm)", LONG_D3_PROFILE),
        ("depth 4 @  48 (committed)", LONG_D3_ABOVE_SHORT),
        ("depth 4 @ 192 (committed)", LONG_D3_ABOVE_LONG),
    ):
        collapse = _cell_deltas(profile, shards, "null", "random")
        edge = _cell_deltas(profile, shards, "adaptive", "random")
        value = _cell_deltas(profile, shards, "adaptive", "null")
        print(
            f"  {label:<26} collapse {sum(collapse) / len(collapse):+.6f}  "
            f"value {sum(value) / len(value):+.6f}  "
            f"edge {sum(edge) / len(edge):+.6f}  ({profile})"
        )
    return 0


def depth6_verdict(shards: int, alpha: float) -> int:
    """Section 5.56's primaries and branches: the confirmation rung at depth 6.

    One fresh cell of the registered config with ``depth`` 5 -> 6 and nothing
    else changed - 45,456 total / 16,656 active parameters. Same one-gene dose
    as 5.55, one layer further up, and the same two primaries at alpha/2 (bar
    313/576):

    - P1 (the gate is still open): the family's per-cell conjunction. Its
      failure at the *third* rung above the onset is the strongest fragility
      reading this family can produce.
    - P2 (a sixth layer creates more value than a fifth): a one-sided rank-sum
      of the depth-5 cell's adaptive-vs-null deltas against this cell's. 5.55
      established the step from 4 to 5; this asks whether the slope continues
      or flattens, and a failure is a real bound either way - a plateau means
      5.55's "the axis pays" is a one-step statement, and a reversal means the
      ladder is not monotone past 5.

    P2's ruler is 5.55's, rehearsed on a measured instance of the same
    comparison shape before that arm ran (`depth4-prefix` ->
    `width-d4-long-w24`, U 41829, p 2.6e-60).

    Branches, fixed before the run:

    - P1 and P2: **the slope continues** - the ladder is monotone across all
      three rungs above the onset (4, 5 and 6), so the axis keeps paying and
      the depth-4 gate is a point on a rising curve rather than a first step
      that flattens.
    - P1 only: **the axis saturates at depth 5** - depth 6 keeps the gate open
      but buys nothing more than depth 5, so the rising part of the curve is
      the gate itself and one step past it.
    - P2 only: a sixth layer creates more value than a fifth and still does not
      clear the bar on its own - recorded as a fragile upper reading rather
      than as either of the two above.
    - neither: **the ladder is not monotone past depth 5** - a sixth layer fails
      where a fifth passes, bounding the monotone claim to depths 2..5.

    The ladder's levels (depths 2, 3, 4, 5 and this cell), the arm that moves
    between rungs and the controller's own edge are reported descriptively, as
    every cross-cell reading in this family is.
    """
    level = alpha / 2
    cell = _cell_stats(DEPTH6_PROFILE, shards, level)
    p1 = bool(cell["creates_value"])
    new = _cell_deltas(DEPTH6_PROFILE, shards, "adaptive", "null")
    reference = _cell_deltas(DEPTH6_REFERENCE, shards, "adaptive", "null")
    # `_rank_sum_one_sided(a, b)` answers whether b is stochastically larger, so
    # the new cell comes first for the claim "deeper creates more value".
    u, p2 = _rank_sum_one_sided(new, reference)
    p2_passes = p2 <= level
    print(
        f"\nP2 (deeper still creates more value): rank-sum "
        f"{DEPTH6_PROFILE} vs {DEPTH6_REFERENCE} on adaptive-vs-null: "
        f"U {u:.1f}, one-sided p {p2:.2e} (bar alpha/2 = {level:.4f}; means "
        f"{sum(new) / len(new):+.6f} vs {sum(reference) / len(reference):+.6f}) "
        f"-> deeper is better: {p2_passes}"
    )
    branches = {
        (True, True): (
            "P1+P2 PASS: the slope continues - the ladder is monotone across all "
            "three rungs above the onset, so a depth-4 pass is a point on a "
            "rising curve rather than a first step that flattens"
        ),
        (True, False): (
            "P1 only: the axis saturates at depth 5 - a sixth layer keeps the "
            "gate open but buys nothing more than a fifth, so the rising part of "
            "the curve is the gate itself and one step past it"
        ),
        (False, True): (
            "P2 only: a sixth layer creates more value than a fifth and still "
            "does not clear the bar on its own - a fragile upper reading"
        ),
        (False, False): (
            "both FAIL: the ladder is not monotone past depth 5 - a sixth layer "
            "fails where a fifth passes, bounding the monotone claim to depths "
            "2..5"
        ),
    }
    print(f"BRANCH: {branches[(p1, p2_passes)]}")
    # Descriptive: the ladder to depth 6, and which arm moves between the two
    # top rungs (5.55 found the whole 4 -> 5 gain in the churn term).
    for label, profile in (
        ("depth 2 (committed)", "depth2"),
        ("depth 3 (committed)", "depth3"),
        ("depth 4 (committed)", DEPTH_DOSE_REFERENCE),
        ("depth 5 (committed)", DEPTH6_REFERENCE),
        ("depth 6 (this arm)", DEPTH6_PROFILE),
    ):
        value = _cell_deltas(profile, shards, "adaptive", "null")
        collapse = _cell_deltas(profile, shards, "null", "random")
        edge = _cell_deltas(profile, shards, "adaptive", "random")
        print(
            f"  {label:<22} value {sum(value) / len(value):+.6f}  "
            f"collapse {sum(collapse) / len(collapse):+.6f}  "
            f"edge {sum(edge) / len(edge):+.6f}  ({profile})"
        )
    # Which term moves between the two top rungs - the decomposition 5.55 used,
    # where the whole 4 -> 5 gain was the churn term and the controller's edge
    # was flat. `adaptive - null = (adaptive - random) - (null - random)`.
    print("  rungs 5 -> 6, decomposed (descriptive; negative = better arm):")
    for label, left, right in (
        ("controller's edge (adaptive-vs-random)", "adaptive", "random"),
        ("churn term (null-vs-random)", "null", "random"),
    ):
        short = _cell_deltas(DEPTH6_REFERENCE, shards, left, right)
        long_ = _cell_deltas(DEPTH6_PROFILE, shards, left, right)
        ua, pa = _rank_sum_one_sided(short, long_)
        _, reverse = _rank_sum_one_sided(long_, short)
        print(
            f"    {label:<38} {sum(short) / len(short):+.6f} -> "
            f"{sum(long_) / len(long_):+.6f}   U {ua:.1f}, p {pa:.3e} "
            f"(reversed {reverse:.3e})"
        )
    return 0


def depth_dose_verdict(shards: int, alpha: float) -> int:
    """Section 5.55's primaries and branches: the depth dose above the onset.

    One fresh cell of the registered config with ``depth`` 4 -> 5 and nothing
    else changed - 38,064 total / 14,064 active parameters against the committed
    above-onset cell's 30,672 / 11,472. That is deliberately *not* a
    matched-parameter contrast: a fifth layer carries its parameters
    irreducibly, and 5.48 and 5.53 already measured that neither buying
    parameters with width nor buying them with experts moves the gate. So this
    arm reads the depth axis as a package, which is the reading "the gate is
    depth per se" needs - not a third matched-parameter probe. Two primaries at
    the sibling arms' alpha/2 (bar 313/576):

    - P1 (the gate stays open above the onset): the family's per-cell
      conjunction. Its failure would mean the gate is a narrow window at depth 4
      rather than a threshold.
    - P2 (a fifth layer creates *more* value than a fourth): a one-sided
      rank-sum of `depth4-prefix`'s adaptive-vs-null deltas against this cell's,
      smaller meaning deeper is better - so P1 without P2 is a threshold (a step,
      not a slope) and P1 with P2 is an axis that pays.

    P2's ruler was rehearsed on a measured instance of the same comparison shape
    before this arm ran: two independent cells at the same depth of the same
    family, one substrate strictly better - `depth4-prefix` against
    `width-d4-long-w24` (the same width and depth at 4x the budget) gives U
    41829, p 2.6e-60 in the direction this primary registers. So the test detects
    a substrate that creates more value where the records say one exists.

    Branches, fixed before the run:

    - P1 and P2: **the gate stays open above the onset and the axis pays** - a
      fifth layer clears the bar and creates more value than a fourth, so the
      depth axis is a slope below the gate and above it, and a Gate 4 pass at
      depth 4 is a point on that axis rather than a knife-edge.
    - P1 only: **the onset is a threshold at depth 4** - deeper keeps the gate
      open but buys nothing more, so the step is real and the slope is not.
    - P2 only: the fifth layer creates more value than the fourth and still does
      not clear the bar on its own, so the upper edge of the gate is fragile.
    - neither: **the gate is not monotone in depth** - a fifth layer fails where
      a fourth passes, and the onset is a narrow window at depth 4. That is the
      strongest statement this family could make about Gate 4's fragility, and
      it is recorded as such.

    The depth ladder's levels (depths 2, 3, 4 and this cell) and the controller's
    own edge are reported descriptively, as every cross-cell reading here is.
    """
    level = alpha / 2
    cell = _cell_stats(DEPTH_DOSE_PROFILE, shards, level)
    p1 = bool(cell["creates_value"])
    new = _cell_deltas(DEPTH_DOSE_PROFILE, shards, "adaptive", "null")
    reference = _cell_deltas(DEPTH_DOSE_REFERENCE, shards, "adaptive", "null")
    # `_rank_sum_one_sided(a, b)` answers whether b is stochastically *larger*
    # than a, so the new cell comes first for the claim "deeper creates more
    # value" (a smaller delta is a better one).
    u, p2 = _rank_sum_one_sided(new, reference)
    p2_passes = p2 <= level
    print(
        f"\nP2 (deeper creates more value than the onset cell): rank-sum "
        f"{DEPTH_DOSE_PROFILE} vs {DEPTH_DOSE_REFERENCE} on adaptive-vs-null: "
        f"U {u:.1f}, one-sided p {p2:.2e} (bar alpha/2 = {level:.4f}; means "
        f"{sum(new) / len(new):+.6f} vs {sum(reference) / len(reference):+.6f}) "
        f"-> deeper is better: {p2_passes}"
    )
    branches = {
        (True, True): (
            "P1+P2 PASS: the gate stays open above the onset and the depth axis "
            "pays - a fifth layer clears the bar and creates more value than a "
            "fourth, so the onset is a slope on both sides and a depth-4 pass is "
            "a point on that axis rather than a knife-edge"
        ),
        (True, False): (
            "P1 only: the onset is a threshold at depth 4 - deeper keeps the gate "
            "open but buys nothing more, so the step is real and the slope is not"
        ),
        (False, True): (
            "P2 only: a fifth layer creates more value than a fourth and still "
            "does not clear the bar on its own - the upper edge of the gate is "
            "fragile"
        ),
        (False, False): (
            "both FAIL: the gate is not monotone in depth - a fifth layer fails "
            "where a fourth passes, so the onset is a narrow window at depth 4"
        ),
    }
    print(f"BRANCH: {branches[(p1, p2_passes)]}")
    # Descriptive: the depth ladder at the registered width and budget.
    for label, profile in (
        ("depth 2 (committed)", "depth2"),
        ("depth 3 (committed)", "depth3"),
        ("depth 4 (committed)", DEPTH_DOSE_REFERENCE),
        ("depth 5 (this arm)", DEPTH_DOSE_PROFILE),
    ):
        value = _cell_deltas(profile, shards, "adaptive", "null")
        collapse = _cell_deltas(profile, shards, "null", "random")
        edge = _cell_deltas(profile, shards, "adaptive", "random")
        print(
            f"  {label:<22} value {sum(value) / len(value):+.6f}  "
            f"collapse {sum(collapse) / len(collapse):+.6f}  "
            f"edge {sum(edge) / len(edge):+.6f}  ({profile})"
        )
    return 0


def experts_verdict(shards: int, alpha: float) -> int:
    """Section 5.53's primaries and branches: the expert-count covariate.

    Two fresh cells at width 24 with ``expert_count`` 4 -> 6, holding
    ``active_experts`` at 2: depth 3 is 30,624 total / 9,024 active parameters
    (-0.156% total, -21.3% active against the committed above-onset cell) and
    depth 4 is 40,464 / 11,664 (+31.9%, +1.7%). The depth-3 cell is therefore
    the family's tightest equal-parameter partner *and* the first that buys its
    parameters without buying any per-token compute - a token still routes to
    two experts. Same test, same conjunction, same bar as the sibling
    two-cell arms (alpha/2, bar 313/576).

    Why this covariate: 10.3(ii) names "expert count, total compute" as what is
    left after 5.48 closed the matched-parameter width form, and expert
    grow/prune is the only operation the V0 search space can reach, so whether
    expert count moves the onset decides whether the reachable gene is the one
    the ontology reads. Branches, fixed before the run:

    - P1 and P2: capacity is the covariate even when bought with experts, and it
      is *stored* capacity rather than compute - the gate falls at an
      equal-parameter partner whose active parameter count is 21.3% lower, which
      no width cell could separate. Read with 5.48, the matched-parameter form
      of (ii) is then answered in favour of capacity rather than depth per se.
    - P1 only: the expert dose buys value below the onset and destroys it above
      - a depth x expert-capacity interaction, and the reachable gene is
      capacity-like only where there was no value to lose.
    - P2 only: the gate does not move for expert count either, so at this scale
      the onset is depth per se against two measured covariates and the
      remaining candidate is total compute. Recorded beside the fact that the
      search space's only reachable gene is a covariate that does not read here.
    - neither: expert capacity is harmful at both depths; the reachable axis is
      closed as a covariate too.

    The four cells' levels and their churn readings (null-vs-random, the axis
    the width line showed flipping sign) are reported descriptively, as every
    cross-cell reading in this family is.
    """
    level = alpha / 2
    cells = {profile: _cell_stats(profile, shards, level) for profile in EXPERTS_PROFILES}
    p1 = bool(cells[EXPERTS_PROFILES[0]]["creates_value"])
    p2 = bool(cells[EXPERTS_PROFILES[1]]["creates_value"])
    print(
        f"\nOUTCOME (alpha={alpha}, Bonferroni alpha/2={level}): "
        f"P1 (depth 3 x expert_count 6, the equal-parameter partner) "
        f"{'PASS' if p1 else 'FAIL'}, P2 (depth 4 x expert_count 6, the same "
        f"dose above the gate) {'PASS' if p2 else 'FAIL'}"
    )
    branches = {
        (True, True): (
            "P1+P2 PASS: capacity is the covariate when bought with experts, "
            "and it is stored capacity rather than per-token compute - the gate "
            "falls at an equal-parameter partner whose active parameter count "
            "is 21.3% lower, where 5.48's width partner matched both"
        ),
        (True, False): (
            "P1 only: the expert dose buys value below the onset and destroys "
            "it above - a depth x expert-capacity interaction"
        ),
        (False, True): (
            "P2 only: the gate does not move for expert count either - with "
            "5.48 the onset is depth per se against two measured covariates, "
            "and the search space's only reachable gene is one that does not "
            "read here"
        ),
        (False, False): (
            "both FAIL: expert capacity is harmful at both depths; the "
            "reachable axis is closed as a covariate too"
        ),
    }
    print(f"BRANCH: {branches[(p1, p2)]}")
    # Descriptive: the four cells of the covariate, and the churn reading at
    # each (the axis that flipped sign with width in 5.49/5.50).
    for label, profile in (
        ("depth 3, E=4 (committed below-onset)", EXPERTS_BELOW_BASELINE),
        ("depth 3, E=6 (this arm, equal parameters)", EXPERTS_PROFILES[0]),
        ("depth 4, E=4 (committed above-onset)", EXPERTS_ABOVE_BASELINE),
        ("depth 4, E=6 (this arm, same dose above)", EXPERTS_PROFILES[1]),
    ):
        collapse = _cell_deltas(profile, shards, "null", "random")
        edge = _cell_deltas(profile, shards, "adaptive", "random")
        print(
            f"  {label:<40} collapse {sum(collapse) / len(collapse):+.6f}  "
            f"controller's edge {sum(edge) / len(edge):+.6f}  ({profile})"
        )
    print(
        "  width-28 partner for reference (5.48, depth 3): "
        f"{EXPERTS_WIDTH_PARTNER}"
    )
    return 0


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
    cells = {profile: _cell_stats(profile, shards, alpha / 2) for profile in WIDTH_PROFILES}
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


def _comparison(deltas: list[float], level: float) -> dict:
    """One pre-registered primary: sign-test statistics at the declared level.

    ``level`` is the per-primary Bonferroni level the arm's registration
    states - alpha/2 for the two-primary arms (5.9-5.49), alpha/3 for 5.50 -
    and the bar and the pass flag are both computed at it, so the summary and
    the verdict cannot disagree about the level either.
    """
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
        "min_wins_bonferroni": _gate_min_wins(n, level),
        "passes_bonferroni": p <= level,
    }


def summarize(profile: str, alpha: float, divisor: float | None = None) -> dict:
    """Build the committed study summary from this profile's shard payloads.

    The summary is self-contained: the per-seed loss triples let every
    quoted statistic be recomputed exactly, so the report numbers stay
    machine-checked against ``artifacts/study-summaries/`` without
    committing the (large, gitignored) shard payloads.

    The Bonferroni divisor is the **profile's own declaration**
    (``bonferroni_divisor``), not an argument: two primaries from 5.9 to 5.49
    and in 5.51/5.52, three in 5.50. A caller may still pass ``divisor``, but
    only to confirm the declared value - passing anything else is refused
    rather than honoured, so a summary can never be written at a level its
    registration does not state. (The CLI used to default it to 2.0 for every
    profile, which quietly rewrote the 5.50 cells at bar 313 where the
    registration states 315 - the same disagreement between a bar and its
    alpha that 5.16.1 adjudicates, and one a default value cannot prevent.)

    A cell's ``outcome`` label is the arm's **own** per-cell primary whenever the
    profile declares one (``per_cell_primary``), and the 5.9 vocabulary only for
    the arms that registered the 5.9 pair. The vocabulary's two primaries are
    null-vs-random and adaptive-vs-null *of the same cell*, so a cell whose
    arm registered something else - a per-cell conjunction (5.48-5.52) or a
    rank-sum between cells (5.50-5.52) - cannot be described by it: 5.48's
    below-onset cell was labelled 5.9's "branch 2 (1 PASS / 2 FAIL)", which
    names a *pass* for a cell the arm recorded as a failure, and 5.52's was
    labelled "no churn harm on this substrate" while its recorded finding is a
    collapse. The flag makes the choice data rather than a section list, so a
    new arm cannot silently inherit the wrong family's wording - the same
    defect class 5.16.1 adjudicates (a declaration whose label and content
    disagree cannot be read as written).
    """
    spec = PROFILES[profile]
    declared = float(spec["bonferroni_divisor"])
    if divisor is not None and float(divisor) != declared:
        raise SystemExit(
            f"profile '{profile}' declares Bonferroni divisor {declared:g}; "
            f"refusing to write its summary at {float(divisor):g} "
            f"(the arm's bar would disagree with its registration)"
        )
    divisor = declared
    level = alpha / divisor
    experiments = load_shards(20 if profile == "depth1" else 20, profile)
    n = len(experiments)
    per_seed = [
        {"seed": exp["seed"], "losses": exp["losses"]} for exp in experiments
    ]
    comparisons = {
        "null_vs_random": _comparison(
            [exp["losses"]["null"] - exp["losses"]["random"] for exp in experiments],
            level,
        ),
        "adaptive_vs_null": _comparison(
            [exp["losses"]["adaptive"] - exp["losses"]["null"] for exp in experiments],
            level,
        ),
        "adaptive_vs_random": _comparison(
            [
                exp["losses"]["adaptive"] - exp["losses"]["random"]
                for exp in experiments
            ],
            level,
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
    if spec.get("per_cell_primary"):
        # This arm registered a per-cell primary, so the 5.9 vocabulary (whose
        # pair of primaries is a different pair, on two values this arm never
        # declared) would mislabel the cell.
        creates = p2 and comparisons["adaptive_vs_null"]["mean_delta"] < 0
        section = spec["report_section"]
        outcome = (
            f"cell creates value ({section}'s per-cell primary: adaptive-vs-null clears "
            f"the alpha/{divisor:g} bar and the mean delta is negative)"
            if creates
            else f"cell does not create value ({section}'s per-cell primary: the conjunction fails)"
        )
    elif p1 and p2:
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
        "bonferroni_alpha": level,
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
    dose_p = sub.add_parser(
        "dose-verdict",
        help="section 5.49: score both dose cells and print the pre-registered branch",
    )
    dose_p.add_argument("--shards", type=int, required=True)
    dose_p.add_argument("--alpha", type=float, default=ALPHA)
    dose2_p = sub.add_parser(
        "dose2-verdict",
        help="section 5.50: score the second-dose cells, the rank-sum primary and the branch",
    )
    dose2_p.add_argument("--shards", type=int, required=True)
    dose2_p.add_argument("--alpha", type=float, default=ALPHA)
    long_p = sub.add_parser(
        "dose2-long-verdict",
        help="section 5.51: the 14.8x collapse against the training budget",
    )
    long_p.add_argument("--shards", type=int, required=True)
    long_p.add_argument("--alpha", type=float, default=ALPHA)
    w24_p = sub.add_parser(
        "long-w24-verdict",
        help="section 5.52: the 192-step width-24 control 5.51 named as absent",
    )
    w24_p.add_argument("--shards", type=int, required=True)
    w24_p.add_argument("--alpha", type=float, default=ALPHA)
    experts_p = sub.add_parser(
        "experts-verdict",
        help="section 5.53: the expert-count covariate at the onset",
    )
    experts_p.add_argument("--shards", type=int, required=True)
    experts_p.add_argument("--alpha", type=float, default=ALPHA)
    d3long_p = sub.add_parser(
        "depth3-long-verdict",
        help="section 5.54: the total-compute covariate at the below-onset depth",
    )
    d3long_p.add_argument("--shards", type=int, required=True)
    d3long_p.add_argument("--alpha", type=float, default=ALPHA)
    d5_p = sub.add_parser(
        "depth5-verdict",
        help="section 5.55: the depth dose above the onset",
    )
    d5_p.add_argument("--shards", type=int, required=True)
    d5_p.add_argument("--alpha", type=float, default=ALPHA)
    d6_p = sub.add_parser(
        "depth6-verdict",
        help="section 5.56: the confirmation rung above the depth dose",
    )
    d6_p.add_argument("--shards", type=int, required=True)
    d6_p.add_argument("--alpha", type=float, default=ALPHA)
    sum_p = sub.add_parser("summary", help="write the committed study summary")
    sum_p.add_argument("--profile", default="depth1", choices=sorted(PROFILES))
    sum_p.add_argument("--alpha", type=float, default=ALPHA)
    sum_p.add_argument(
        "--divisor",
        type=float,
        default=None,
        help=(
            "confirm the profile's declared Bonferroni divisor; omitted means "
            "the declaration (2 from 5.9 to 5.49 and 5.51-5.52, 3 for 5.50), "
            "and a different value is refused"
        ),
    )
    args = parser.parse_args()
    if args.command == "run":
        run_shard(args.shard, args.shards, args.profile)
        return 0
    if args.command == "summary":
        payload = summarize(args.profile, args.alpha, args.divisor)
        out = ROOT / PROFILES[args.profile]["summary"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"summary written: {out} ({payload['study_id']})")
        return 0
    if args.command == "width-verdict":
        return width_verdict(args.shards, args.alpha)
    if args.command == "dose-verdict":
        return dose_verdict(args.shards, args.alpha)
    if args.command == "dose2-verdict":
        return dose2_verdict(args.shards, args.alpha)
    if args.command == "dose2-long-verdict":
        return dose2_long_verdict(args.shards, args.alpha)
    if args.command == "long-w24-verdict":
        return long_w24_verdict(args.shards, args.alpha)
    if args.command == "experts-verdict":
        return experts_verdict(args.shards, args.alpha)
    if args.command == "depth3-long-verdict":
        return long_d3_verdict(args.shards, args.alpha)
    if args.command == "depth5-verdict":
        return depth_dose_verdict(args.shards, args.alpha)
    if args.command == "depth6-verdict":
        return depth6_verdict(args.shards, args.alpha)
    return verdict(args.shards, args.profile)


if __name__ == "__main__":
    sys.exit(main())
