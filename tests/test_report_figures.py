"""Tests for the report-figure data assembly (act.evolution.figures_data).

The figures in RESEARCH_REPORT.md section 8 must be plotted from
machine-checked records only; these tests pin the loading, the verdict
re-checks, and the stdlib statistics helpers to independently computed
reference values (scipy for the beta quantiles).
"""

from __future__ import annotations

import copy
import json
import math

import pytest

from act.evolution.figures_data import (
    DECOUPLED_STUDIES,
    PASS_THRESHOLDS,
    STUDIES,
    arm_totals,
    beta_cdf,
    beta_quantile,
    check_summary,
    clopper_pearson_interval,
    joint_rank_fractions,
    load_summaries,
    paired_deltas,
    probe_concordance,
    probe_cost_pct,
    summaries_dir,
)


def test_load_summaries_returns_all_nine_studies() -> None:
    summaries = load_summaries()
    assert set(summaries) == {key for key, _, _ in STUDIES}
    assert len(summaries) == 9
    for key, _, filename in STUDIES:
        summary = summaries[key]
        assert summary["record_type"] == "gate_study"
        assert isinstance(summary["experiments"], list) and summary["experiments"]


def test_missing_summary_file_raises_helpful_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="missing study summary"):
        load_summaries(tmp_path)


def test_decoupled_summaries_load_and_recompute() -> None:
    """The corrected figure set must be machine-checked like the registered one.

    These summaries come from the section 9.10 recheck (`factor_map_recheck.py
    --summaries`); each is validated by ``check_summary`` on the way out, and
    this pins that the committed set still recomputes.
    """
    directory = summaries_dir() / "decoupled"
    summaries = load_summaries(directory, studies=DECOUPLED_STUDIES)

    assert len(DECOUPLED_STUDIES) == len(STUDIES) - 1
    assert set(summaries) == {key for key, _, _ in DECOUPLED_STUDIES}
    # Study 1 is not reconstructible on the decoupled path (section 9.10).
    assert "legacy" not in summaries
    for key, _, filename in DECOUPLED_STUDIES:
        summary = summaries[key]
        assert summary["provenance"] == "decoupled-independent-arm-rng"
        assert summary["record_type"] == "gate_study"
        check_summary(summary, filename)


def test_decoupled_recheck_matches_the_report_section_9_11_table() -> None:
    """Pin the committed numbers section 3 and 9.11 quote.

    The report tells the reader to quote the depth-1 factor map from the
    decoupled column, so the corrected win counts and the one label flip are
    pinned here rather than left as prose.
    """
    summaries = load_summaries(summaries_dir() / "decoupled", studies=DECOUPLED_STUDIES)

    expected_wins = {
        "guided": 62,
        "probe-loss": 222,
        "cheap-probe": 188,
        "cadence": 148,
        "variance": 209,
        "horizon": 250,
        "rich-a": 229,
        "rich-b": 198,
    }
    for key, wins in expected_wins.items():
        assert summaries[key]["adaptive_wins"] == wins, key

    # The single label flip: variance was a FAIL on the coupled path and is a
    # threshold-level PASS here (209/384 is exactly the alpha=0.05 minimum).
    assert summaries["variance"]["gate_min_wins"] == 209
    assert summaries["variance"]["gate_passed"] is True
    assert load_summaries()["variance"]["gate_passed"] is False

    # Every recheckable study moved toward the treatment.
    coupled = load_summaries()
    for key in expected_wins:
        assert expected_wins[key] > coupled[key]["adaptive_wins"], key


def test_paired_deltas_reproduce_report_means() -> None:
    # Mean deltas quoted in RESEARCH_REPORT.md section 3's table.
    report_means = {
        "legacy": 0.006766,
        "probe-loss": -0.025955,
        "cheap-probe": 0.011747,
        "cadence": 0.047014,
        "variance": -0.003907,
        "horizon": -0.052402,
        "rich-a": -0.036982,
        "rich-b": 0.007156,
    }
    summaries = load_summaries()
    for key, expected in report_means.items():
        deltas = paired_deltas(summaries[key])
        assert math.isclose(sum(deltas) / len(deltas), expected, rel_tol=2e-4), key


def test_paired_deltas_respect_treatment_arm() -> None:
    summaries = load_summaries()
    guided = summaries["guided"]
    assert guided["treatment"] == "guided"
    deltas = paired_deltas(guided, treatment="guided")
    assert math.isclose(sum(deltas) / len(deltas), -0.041325, rel_tol=2e-4)
    with pytest.raises(KeyError):
        paired_deltas(guided, treatment="adaptive")  # guided study has no adaptive arm


def test_check_summary_passes_on_all_versioned_summaries() -> None:
    for key, _, filename in STUDIES:
        summary = load_summaries()[key]
        check_summary(summary, filename)  # must not raise


def test_check_summary_detects_tampering() -> None:
    summary = copy.deepcopy(load_summaries()["probe-loss"])
    summary["adaptive_wins"] += 1
    with pytest.raises(AssertionError, match="binomial p"):
        check_summary(summary)


def test_check_summary_detects_mean_delta_tampering() -> None:
    summary = copy.deepcopy(load_summaries()["horizon"])
    summary["mean_adaptive_delta"] += 0.01
    with pytest.raises(AssertionError, match="mean delta"):
        check_summary(summary)


def test_check_summary_detects_gate_pass_flag_tampering() -> None:
    summary = copy.deepcopy(load_summaries()["legacy"])
    summary["gate_passed"] = True
    with pytest.raises(AssertionError, match="gate_passed"):
        check_summary(summary)


def test_probe_cost_identity_reproduces_report_percentages() -> None:
    # Probe-cost percentages quoted in RESEARCH_REPORT.md sections 3-4.
    report_costs = {
        "probe-loss": 41.9,
        "horizon": 14.0,
        "rich-a": 7.3,
        "cheap-probe": 4.13,
        "rich-b": 8.1,
        "cadence": 8.41,
        "variance": 16.0,
    }
    summaries = load_summaries()
    for key, expected in report_costs.items():
        assert math.isclose(probe_cost_pct(summaries[key]), expected, rel_tol=0.01), key
    # The guided diagnostic arm books its own probe cost (~95%), far above the bar.
    assert probe_cost_pct(summaries["guided"]) > 50.0


def test_arm_totals_aggregate_per_arm() -> None:
    summary = load_summaries()["probe-loss"]
    search, training = arm_totals(summary, "adaptive")
    assert search > 0 and training > 0
    random_search, random_training = arm_totals(summary, "random")
    # Mutation budgets are matched, but not architectures: a grown expert
    # raises that seed's training FLOPs, so per-arm training totals differ.
    # The probe-cost identity is therefore a per-arm ratio, not a shared one.
    assert random_training != training
    assert random_search < search  # the random arm runs no probe search


def test_beta_cdf_matches_scipy_reference_values() -> None:
    # Reference values from scipy.stats.beta (independent implementation).
    references = [
        (0.5, 3.0, 7.0, 0.91015625),
        (0.3, 2.5, 4.2, 0.3741155180265067),
        (0.7, 5.0, 2.0, 0.4201749999999999),
        (0.999, 1.0, 1.0, 0.999),
        (1e-9, 1.0, 1.0, 1e-9),
        (1e-10, 3.0, 500.0, 2.095849921562822e-23),
    ]
    for x, a, b, expected in references:
        assert math.isclose(beta_cdf(x, a, b), expected, rel_tol=1e-6), (x, a, b)


def test_beta_quantile_matches_scipy_reference_values() -> None:
    # Reference values from scipy.stats.beta.ppf (independent implementation).
    references = [
        (0.3, 2.5, 4.2, 0.2643244243398563),
        (0.7, 5.0, 2.0, 0.8181965286810509),
        (0.5, 1.0, 1.0, 0.5),
        (0.05, 1.0, 1.0, 0.05),
    ]
    for prob, a, b, expected in references:
        assert math.isclose(beta_quantile(prob, a, b), expected, rel_tol=1e-6), (prob, a, b)


@pytest.mark.parametrize(
    "wins,n,lo,hi",
    [
        (57, 96, 0.488683114686, 0.692871106153),
        (59, 96, 0.509718629854, 0.712176970194),
        (209, 384, 0.492977692067, 0.594876415255),
        (231, 384, 0.550664630164, 0.650882145417),
        (228, 384, 0.542758310653, 0.643285018672),
        (195, 384, 0.456613409514, 0.558890271440),
        (137, 384, 0.308821969658, 0.406947007153),
        (0, 10, 0.0, 0.308497107819),
        (10, 10, 0.691502892181, 1.0),
        (5, 10, 0.187086028447, 0.812913971553),
        (1, 10, 0.002528578544, 0.445016117028),
    ],
)
def test_clopper_pearson_matches_scipy_reference(wins, n, lo, hi) -> None:
    lower, upper = clopper_pearson_interval(wins, n)
    assert math.isclose(lower, lo, abs_tol=1e-9)
    assert math.isclose(upper, hi, abs_tol=1e-9)


def test_pass_thresholds_match_pre_registration() -> None:
    assert PASS_THRESHOLDS == {96: 57, 380: 207, 384: 209}


def test_probe_concordance_matches_stored_result() -> None:
    payload = json.loads((summaries_dir() / "probe-validity-analysis.json").read_text())
    concordance, total = probe_concordance(payload["per_seed"])
    assert total == payload["results"]["n_seeds"] == 384
    # The stored result rounds to 3 decimals (108/384 = 0.28125 -> 0.281).
    assert math.isclose(concordance, payload["results"]["argmin_concordance"], abs_tol=5e-4)
    assert math.isclose(concordance, 108 / 384, abs_tol=1e-12)


def test_joint_rank_fractions_sum_to_one_and_recover_concordance() -> None:
    payload = json.loads((summaries_dir() / "probe-validity-analysis.json").read_text())
    joint = joint_rank_fractions(payload["per_seed"])
    assert len(joint) == 4 and all(len(row) == 4 for row in joint)
    # Each row is the conditional distribution of the final rank given the
    # probe rank (every seed contributes one candidate per probe rank).
    for row in joint:
        assert math.isclose(sum(row), 1.0, rel_tol=1e-9)
    # The (probe 1, final 1) cell is exactly the argmin concordance (108/384).
    assert math.isclose(joint[0][0], 108 / 384, abs_tol=1e-12)
    assert math.isclose(joint[0][0], payload["results"]["argmin_concordance"], abs_tol=5e-4)
