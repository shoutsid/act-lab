"""Pins for the section 5.30 stored records (pre-registered grow stratum).

Section 5.29's registered verdicts were artifacts of its block: the probe-guided
adaptive arm grows in only 18% of events, so 53% of seeds contain no grow and
cloning is then a literal no-op. A full-block sign test scores such a seed as a
non-win for P1 and hands P2 the full historical penalty. Section 5.30 re-tested it
on fresh blocks with the stratum declared in advance, and both the truth and the
artefact replicated.

Pinned here, because nothing else re-derives them:

1. **The stratum boundary is the no-op boundary.** In the no-grow stratum the two
   settings must be bit-identical, so `C_own - C` is exactly `R_own - R`. This is
   the positive control that makes the stratification credible rather than
   convenient, and it is asserted on every no-grow seed.
2. **The two statistics disagree on the same runs.** The full-block sign test
   fails while the pre-registered stratified test passes at p ~1e-11. A test that
   kept only the passing number would lose the entire methodological point.
"""

from __future__ import annotations

import json

import pytest

from act.evolution.figures_data import summaries_dir
from act.evolution.study import _binomial_survival, _gate_min_wins

PER_PRIMARY_ALPHA = 0.05 / 4  # two primaries, two depths

QUOTED: dict[str, dict] = {
    "adaptive-clone-stratified-depth2-384.json": {
        "study_id": "study_54cff7dc8d8b6f11",
        "seeds": (111000, 111384),
        "strata": {"grow": 195, "no_grow": 189},
        "comparisons": {
            # name -> (wins, mean, p, passes)
            "C_vs_R_grow_stratum": (142, -0.054215, 7.06e-11, True),
            "C_own_vs_C_grow_stratum": (98, -0.006417, 5.00e-01, False),
            "C_vs_R_full_block": (142, -0.027531, 1.00, False),
            "C_own_vs_C_no_grow_stratum": (134, -0.055302, 4.28e-09, True),
            "C_vs_F_full_block": (263, -0.050806, 1.68e-13, True),
        },
        "means": {
            "R_own": 14.23677122530838,
            "C_own": 14.238320591549078,
            "C": 14.268798170611262,
            "R": 14.296329386532307,
            "F": 14.319604308033982,
        },
    },
    "adaptive-clone-stratified-depth3-384.json": {
        "study_id": "study_da528e81c3204901",
        "seeds": (112000, 112384),
        "strata": {"grow": 179, "no_grow": 205},
        "comparisons": {
            "C_vs_R_grow_stratum": (130, -0.046553, 5.86e-10, True),
            "C_own_vs_C_grow_stratum": (77, +0.014380, 9.74e-01, False),
            "C_vs_R_full_block": (130, -0.021700, 1.00, False),
            "C_own_vs_C_no_grow_stratum": (151, -0.074135, 4.09e-12, True),
            "C_vs_F_full_block": (260, -0.057474, 1.68e-12, True),
        },
        "means": {
            "R_own": 13.31768164349099,
            "C_own": 13.319985394676527,
            "C": 13.352859519422054,
            "R": 13.374559786170721,
            "F": 13.410333127404252,
        },
    },
}

# comparison name -> (first loss label, second loss label, subset)
SPEC: dict[str, tuple[str, str, str]] = {
    "C_vs_R_grow_stratum": ("C", "R", "grow"),
    "C_own_vs_C_grow_stratum": ("C_own", "C", "grow"),
    "C_vs_R_full_block": ("C", "R", "all"),
    "C_own_vs_C_no_grow_stratum": ("C_own", "C", "no_grow"),
    "C_vs_F_full_block": ("C", "F", "all"),
}


def _load(name: str) -> dict:
    path = summaries_dir() / name
    if not path.exists():
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


def _subset(summary: dict, stratum: str) -> list[dict]:
    if stratum == "all":
        return summary["per_seed"]
    return [row for row in summary["per_seed"] if row["stratum"] == stratum]


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_30_numbers_recompute_from_per_seed(name: str) -> None:
    quoted = QUOTED[name]
    summary = _load(name)
    assert summary["record_type"] == "adaptive_clone_stratified_study"
    assert summary["study_id"] == quoted["study_id"]
    assert summary["per_primary_alpha"] == PER_PRIMARY_ALPHA
    start, stop = quoted["seeds"]
    assert summary["seeds"] == list(range(start, stop))
    assert summary["seeds"] == [row["seed"] for row in summary["per_seed"]]
    assert all(summary["built_in_checks"].values()), summary["built_in_checks"]
    assert summary["stratum_sizes"] == quoted["strata"]

    for comparison_name, (wins, mean, p, passes) in quoted["comparisons"].items():
        first, second, stratum = SPEC[comparison_name]
        subset = _subset(summary, stratum)
        deltas = [row["losses"][first] - row["losses"][second] for row in subset]
        stored = summary["comparisons"][comparison_name]
        assert sum(delta < 0 for delta in deltas) == wins, comparison_name
        assert sum(deltas) / len(deltas) == pytest.approx(mean, abs=5e-6)
        assert _binomial_survival(wins, len(deltas)) == pytest.approx(p, rel=1e-2)
        # The stored block must agree with its own per_seed rows, and the reported
        # bar must be the one for the realized stratum size.
        assert stored["wins"] == wins, comparison_name
        assert stored["mean_delta"] == pytest.approx(mean, abs=5e-6)
        assert stored["n"] == len(deltas)
        assert stored["passes"] is passes
        assert stored["min_wins"] == _gate_min_wins(len(deltas), PER_PRIMARY_ALPHA)
        assert stored["bar_and_p_agree"] is True

    for label, value in quoted["means"].items():
        assert summary["loss_means"][label] == pytest.approx(value, abs=1e-6)

    assert summary["comparisons"]["C_vs_F_full_block"]["passes"] is True


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_30_the_stratum_is_the_historical_runs_mix(name: str) -> None:
    quoted = QUOTED[name]
    summary = _load(name)
    grow = [row for row in summary["per_seed"] if row["stratum"] == "grow"]
    no_grow = [row for row in summary["per_seed"] if row["stratum"] == "no_grow"]
    assert len(grow) == quoted["strata"]["grow"]
    assert len(no_grow) == quoted["strata"]["no_grow"]
    # The stratum is defined by the HISTORICAL run's drawn operations, never by the
    # clone run's - that is what keeps it a pre-treatment covariate.
    assert all(row["historical_grows"] >= 1 for row in grow)
    assert all(row["historical_grows"] == 0 for row in no_grow)


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_30_the_no_grow_stratum_is_a_true_no_op(name: str) -> None:
    """The positive control the stratification's credibility rests on.

    Where the historical run drew no grow there is no new expert, so the clone
    setting must change nothing at all - the two fused losses identical to the
    last bit, and therefore `C_own - C` exactly `R_own - R`. If this ever fails,
    the stratum boundary is not the no-op boundary and the split is not measuring
    what section 5.30 claims.
    """
    summary = _load(name)
    no_grow = [row for row in summary["per_seed"] if row["stratum"] == "no_grow"]
    assert no_grow, "an empty no-grow stratum makes the control vacuous"
    assert all(row["losses"]["C"] == row["losses"]["R"] for row in no_grow)
    assert all(
        row["losses"]["C_own"] - row["losses"]["C"]
        == row["losses"]["R_own"] - row["losses"]["R"]
        for row in no_grow
    )


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_30_the_full_block_statistic_fails_on_the_same_runs(name: str) -> None:
    """The methodological point: same runs, same evidence, opposite verdicts.

    Section 5.29's registered statistic (a full-block sign test) fails here while
    the pre-registered stratified test on the same runs passes decisively. A test
    that asserted only the passing number would lose the whole finding.
    """
    summary = _load(name)
    stratified = summary["comparisons"]["C_vs_R_grow_stratum"]
    full_block = summary["comparisons"]["C_vs_R_full_block"]
    assert stratified["passes"] is True
    assert full_block["passes"] is False
    assert stratified["mean_delta"] < full_block["mean_delta"] < 0, (
        "the stratified effect must be larger in magnitude than the diluted one"
    )
    # And the dilution is large: the grow stratum's win rate is over 70% while the
    # full block's is under 40%, which is what the no-op seeds do to a sign test.
    assert stratified["wins"] / stratified["n"] > 0.70
    assert full_block["wins"] / full_block["n"] < 0.40
