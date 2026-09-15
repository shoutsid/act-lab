"""Section 5.21's cross-study observation stays tied to the artifacts behind it.

Section 5.21 is an **unregistered** reanalysis of three blocks that were run for
other reasons: it measures, per seed, each mutating arm's own fused run (mutations
applied in place) against a frozen retrain of that arm's final genome from the
same initialization and the same data schedule. The claim is that the fused run
is dominated at every depth and for *both* mutating arms, i.e. the penalty is a
property of the process rather than of the topology the controller found.

Two things need holding down, because either could quietly evaporate:

- the numbers must still recompute from the committed per-seed rows, and
- the observation must still be labelled unregistered in the report - it is a
  reanalysis, and the moment it reads like a pre-registered finding it is being
  oversold. Section 5.22 is the registered test that has to actually run.
"""

from __future__ import annotations

import json
import math
import re

from act.evolution.figures_data import summaries_dir
from act.evolution.study import _binomial_survival

REPO_ROOT = summaries_dir().parents[1]
#: profile -> (summary file, adaptive transport, random transport, fixed transport)
RECORDS = {
    "depth2": ("specificity-depth2-384.json", "A_own", "R_own", "F_own"),
    "depth3": ("specificity-depth3-384.json", "A_own", "R_own", "F_own"),
    "depth4": ("initialization-depth4-384.json", "B_adaptive", "B_random", "B_fixed"),
}
#: (depth, arm) -> (frozen wins, mean fused-minus-frozen, p). Positive mean = fused lost.
QUOTED = {
    ("depth2", "adaptive"): (314, 0.1027, 2.66e-38),
    ("depth2", "random"): (292, 0.1050, 1.17e-25),
    ("depth3", "adaptive"): (288, 0.0913, 1.07e-23),
    ("depth3", "random"): (293, 0.1248, 3.65e-26),
    ("depth4", "adaptive"): (271, 0.0961, 2.16e-16),
    ("depth4", "random"): (258, 0.0803, 7.33e-12),
}


def _load(depth: str) -> dict:
    name = RECORDS[depth][0]
    path = summaries_dir() / name
    if not path.exists():
        import pytest

        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


def _contrast(summary: dict, transport: str, arm: str) -> list[float]:
    """Per-seed (fused loss - frozen loss); positive means the fused run lost."""
    return [
        row["discovery_losses"][arm] - row["transport_losses"][transport]
        for row in summary["per_seed"]
    ]


def test_section_5_21_numbers_recompute_from_committed_rows() -> None:
    for (depth, arm), (wins, mean, p) in QUOTED.items():
        _, adaptive, random, _fixed = RECORDS[depth]
        transport = adaptive if arm == "adaptive" else random
        deltas = _contrast(_load(depth), transport, arm)
        assert sum(delta > 0 for delta in deltas) == wins, (depth, arm)
        assert math.isclose(sum(deltas) / len(deltas), mean, rel_tol=0, abs_tol=5e-5)
        assert math.isclose(_binomial_survival(wins, len(deltas)), p, rel_tol=5e-2)


def test_section_5_21_transport_validity_is_exact() -> None:
    """The unmutated genome must reproduce its own run exactly, at every depth.

    Max |delta| exactly 0.0 is what makes a topology effect separable from a
    transport artifact, so a nonzero value here invalidates the whole comparison
    rather than merely adding noise.
    """
    for depth, (_name, _adaptive, _random, fixed) in RECORDS.items():
        summary = _load(depth)
        deltas = _contrast(summary, fixed, "fixed")
        assert max(abs(delta) for delta in deltas) == 0.0, depth


def test_section_5_21_penalty_is_arm_generic() -> None:
    """The reading depends on the RANDOM arm losing too, not just the adaptive one.

    Section 5.17.2 saw this only for the adaptive arm and called it an observation
    about the discovered topology. If the random arm's frozen retrain stopped
    winning, the arm-generic (process) reading recorded in 5.21 would be wrong.
    """
    for depth, (_name, adaptive, random, _fixed) in RECORDS.items():
        summary = _load(depth)
        for transport, arm in ((adaptive, "adaptive"), (random, "random")):
            wins = sum(delta > 0 for delta in _contrast(summary, transport, arm))
            assert wins > 0.5 * len(summary["per_seed"]), (depth, arm, wins)


def test_section_5_21_is_still_labelled_unregistered() -> None:
    report = (REPO_ROOT / "RESEARCH_REPORT.md").read_text()
    match = re.search(r"^### 5\.21 (.*)$", report, re.M)
    assert match, "section 5.21 is missing from the report"
    assert "unregistered" in match.group(1).lower(), (
        "section 5.21 is a reanalysis of blocks run for other reasons; its header "
        "must keep the 'unregistered' label"
    )
    assert "Prediction" not in match.group(1)


#: Section 5.22.1: block -> (summary file, churned transport, fused arm, fixed transport)
CAPACITY_BLOCKS = {
    "5.20-d2": ("specificity-depth2-384.json", "R_own", "random", "F_own"),
    "5.20-d3": ("specificity-depth3-384.json", "R_own", "random", "F_own"),
    "5.17-d4": ("initialization-depth4-384.json", "B_random", "random", "B_fixed"),
    "5.22-d2": ("frozen-delivery-depth2-384.json", "R_own", "random", "F_own"),
    "5.22-d3": ("frozen-delivery-depth3-384.json", "R_own", "random", "F_own"),
}
#: Quoted correlation between the fused-run penalty and the topology's size change.
QUOTED_CAPACITY_CORRELATION = {
    "5.20-d2": -0.039, "5.20-d3": -0.051, "5.17-d4": -0.002,
    "5.22-d2": 0.013, "5.22-d3": 0.046,
}
#: Quoted frozen-churned vs frozen-unpruned win counts.
QUOTED_FROZEN_VS_UNPRUNED = {
    "5.20-d2": 213, "5.20-d3": 197, "5.17-d4": 205, "5.22-d2": 195, "5.22-d3": 214,
}


def _capacity_rows(block: str) -> dict:
    name = CAPACITY_BLOCKS[block][0]
    return json.loads((summaries_dir() / name).read_text())


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy)


def test_section_5_22_1_penalty_is_independent_of_size_change() -> None:
    """The adopted consequence needs the penalty to be a process effect.

    If the penalty tracked how much the topology shrank, "the frozen retrain just
    trains a smaller model" would explain it, and the lesson would be to prune
    early instead of to freeze and retrain. Section 5.22.1 reports near-zero
    correlations; this pins that they stay near zero.
    """
    for block, quoted in QUOTED_CAPACITY_CORRELATION.items():
        _name, transport, arm, fixed = CAPACITY_BLOCKS[block]
        rows = _capacity_rows(block)["per_seed"]
        penalty = [r["discovery_losses"][arm] - r["transport_losses"][transport] for r in rows]
        shrink = [
            r["transport_active_parameter_count"][fixed]
            - r["transport_active_parameter_count"][transport]
            for r in rows
        ]
        assert abs(_pearson(penalty, shrink)) < 0.1, (block, _pearson(penalty, shrink))
        assert abs(_pearson(penalty, shrink) - quoted) < 5e-3, block


def test_section_5_22_1_size_buys_nothing_when_frozen() -> None:
    """With the process removed, the churned topology must NOT be better.

    Same frozen treatment on both sides, so any win would be a pure capacity
    effect. None of the five blocks shows one - that is what makes "smaller
    models train better at this scale" unavailable as an explanation.
    """
    for block, quoted in QUOTED_FROZEN_VS_UNPRUNED.items():
        _name, transport, _arm, fixed = CAPACITY_BLOCKS[block]
        rows = _capacity_rows(block)["per_seed"]
        deltas = [r["transport_losses"][transport] - r["transport_losses"][fixed] for r in rows]
        wins = sum(delta < 0 for delta in deltas)
        assert wins == quoted, (block, wins)
        assert wins <= 0.6 * len(deltas), (block, wins)
