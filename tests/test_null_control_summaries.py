"""The committed null-control study summaries stay machine-checked.

These tests pin the versioned records produced by
``scripts/null_toy_study.py summary`` (report sections 5.9, 5.10 and
5.12): the
quoted statistics must recompute exactly from the stored per-seed triples,
the built-in checks must hold, and the stored verdicts must keep their
labels. They also pin the committed schema: per-seed payloads are kept
(triples per seed, not per-arm blobs), so the files stay self-contained
and the shard payloads remain gitignored.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from act.evolution.figures_data import summaries_dir
from act.evolution.study import _binomial_survival, _gate_min_wins

SUMMARIES = {
    "depth1": "null-control-depth1-384.json",
    "depth4": "null-control-depth4-384.json",
    "depth1-spread": "null-control-depth1-spread-384.json",
    "depth4-prefix": "null-control-depth4-prefix-384.json",
    "depth2": "null-control-depth2-384.json",
    "depth3": "null-control-depth3-384.json",
}
QUOTED = {
    # profile -> comparison -> (wins, mean, p): the numbers quoted in the
    # report's section 5.9 / 5.10 / 5.15 outcome paragraphs.
    "depth1": {
        "null_vs_random": (280, -0.079528, 4.69e-20),
        "adaptive_vs_null": (117, +0.050757, 1.0),
        "adaptive_vs_random": (225, -0.028771, 4.43e-04),
    },
    "depth4": {
        "null_vs_random": (254, -0.083303, 1.22e-10),
        "adaptive_vs_null": (218, -0.029696, 4.58e-03),
        "adaptive_vs_random": (263, -0.112999, 1.68e-13),
    },
    "depth1-spread": {
        "null_vs_random": (289, -0.076105, 3.54e-24),
        "adaptive_vs_null": (131, +0.047507, 1.0),
        "adaptive_vs_random": (231, -0.028598, 4.04e-05),
    },
    "depth4-prefix": {
        "null_vs_random": (266, -0.070072, 1.52e-14),
        "adaptive_vs_null": (225, -0.037147, 4.43e-04),
        "adaptive_vs_random": (270, -0.107220, 5.19e-16),
    },
    "depth2": {
        "null_vs_random": (290, -0.099659, 1.15e-24),
        "adaptive_vs_null": (161, +0.020811, 9.99e-01),
        "adaptive_vs_random": (254, -0.078848, 1.22e-10),
    },
    "depth3": {
        "null_vs_random": (286, -0.106794, 9.45e-23),
        "adaptive_vs_null": (188, +0.004643, 6.77e-01),
        "adaptive_vs_random": (267, -0.102151, 6.65e-15),
    },
}
OUTCOME_LABEL = {
    "depth1": "branch 2 (1 PASS / 2 FAIL)",
    "depth4": "branch 1 (1+2 PASS)",
    "depth1-spread": "branch 2 (1 PASS / 2 FAIL)",
    "depth4-prefix": "branch 1 (1+2 PASS)",
    "depth2": "branch 2 (1 PASS / 2 FAIL)",
    "depth3": "branch 2 (1 PASS / 2 FAIL)",
}


def _load(name: str) -> dict:
    path = summaries_dir() / name
    if not path.exists():
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("profile,filename", sorted(SUMMARIES.items()))
def test_null_control_summary_recomputes_from_per_seed(profile: str, filename: str) -> None:
    summary = _load(filename)
    alpha = summary["alpha"]
    assert summary["record_type"] == "null_control_study"
    assert summary["seeds"] == [row["seed"] for row in summary["per_seed"]]
    assert len(summary["per_seed"]) == 384
    for name, quoted in QUOTED[profile].items():
        treatment, control = name.split("_vs_")
        comparison = summary["comparisons"][name]
        deltas = [
            row["losses"][treatment] - row["losses"][control]
            for row in summary["per_seed"]
        ]
        wins = sum(delta < 0 for delta in deltas)
        mean = sum(deltas) / len(deltas)
        p = _binomial_survival(wins, len(deltas))
        # The stored statistics recompute exactly from the stored triples.
        assert comparison["wins"] == wins == quoted[0]
        assert math.isclose(comparison["mean_delta"], mean, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(mean, quoted[1], rel_tol=0, abs_tol=5e-7)
        assert math.isclose(comparison["binomial_p"], p, rel_tol=1e-6, abs_tol=0)
        assert math.isclose(p, quoted[2], rel_tol=5e-3)
        # The Bonferroni bar and its verdict recompute too.
        assert comparison["min_wins_bonferroni"] == _gate_min_wins(len(deltas), alpha / 2)
        assert comparison["passes_bonferroni"] == (p <= alpha / 2)


@pytest.mark.parametrize("profile,filename", sorted(SUMMARIES.items()))
def test_null_control_summary_built_in_checks_and_labels(profile: str, filename: str) -> None:
    summary = _load(filename)
    assert summary["arms"] == ["random", "adaptive", "null"]
    assert summary["independent_arm_rng"] is True
    assert summary["config"]["arms"] == "random,adaptive,null"
    assert all(summary["built_in_checks"].values())
    # The p-values quoting the bar are exact: wins >= min_wins <=> p <= alpha/2.
    for comparison in summary["comparisons"].values():
        assert (comparison["wins"] >= comparison["min_wins_bonferroni"]) == (
            comparison["passes_bonferroni"]
        )
    assert summary["outcome"].startswith(OUTCOME_LABEL[profile])
    # Seed blocks are disjoint and never observed elsewhere.
    seeds = set(summary["seeds"])
    assert len(seeds) == 384
    assert seeds.isdisjoint(set(json.loads(
        (summaries_dir() / "gate3-probe-loss-384.json").read_text()
    )["seeds"]))


@pytest.mark.parametrize("profile,filename", sorted(SUMMARIES.items()))
def test_null_control_summary_arm_means_and_hash(profile: str, filename: str) -> None:
    summary = _load(filename)
    for arm, mean in summary["arm_means"].items():
        recomputed = sum(row["losses"][arm] for row in summary["per_seed"]) / 384
        assert math.isclose(mean, recomputed, rel_tol=0, abs_tol=1e-12)
    # The depth-1/2/3 orderings are null < adaptive < random; at depth 4
    # the ordering flips, regardless of the candidate policy (the
    # section-5.12 2x2), pinning the section-5.15 dose-response: the sign
    # of adaptive-minus-null flips just above depth 3.
    d1 = summary["arm_means"]
    if profile in ("depth1", "depth1-spread", "depth2", "depth3"):
        assert d1["null"] < d1["adaptive"] < d1["random"]
    else:
        assert d1["adaptive"] < d1["null"] < d1["random"]


def test_depth_dose_response_is_monotone_across_the_four_cells() -> None:
    """Section 5.15: the depth axis is monotone in wins and mean harm.

    adaptive-vs-null wins rise with depth (117 < 161 < 188 < 225) and the
    mean delta falls through zero between depths 3 and 4: the value-creation
    onset sits between depth 3 and depth 4, the pre-registered branch (b).
    Churn harm (null-vs-random) clears at every depth.
    """
    wins = []
    means = []
    churn_p = []
    for profile in ("depth1", "depth2", "depth3", "depth4"):
        summary = _load(SUMMARIES[profile])
        comparisons = summary["comparisons"]
        wins.append(comparisons["adaptive_vs_null"]["wins"])
        means.append(comparisons["adaptive_vs_null"]["mean_delta"])
        churn = comparisons["null_vs_random"]
        assert churn["passes_bonferroni"] is True  # churn harm at every depth
        churn_p.append(churn["binomial_p"])
    assert wins == sorted(wins)
    assert means == sorted(means, reverse=True)
    assert means[2] > 0 > means[3]  # the sign flip sits between depth 3 and 4
    assert all(p < 1e-9 for p in churn_p)
