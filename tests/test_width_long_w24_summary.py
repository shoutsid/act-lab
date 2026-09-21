"""Section 5.52's recorded result: every number the report quotes, pinned.

The design pins live in `test_width_long_w24.py`; these read the committed
summary and the report against each other, so a quoted figure cannot drift from
the artifact it came from (the defect class this repository has hit most often).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import null_toy_study as study  # noqa: E402

SUMMARY = ROOT / "artifacts" / "study-summaries" / "width-d4-long-w24-576.json"
REPORT = ROOT / "docs" / "RESEARCH_REPORT.md"


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads(SUMMARY.read_text())


@pytest.fixture(scope="module")
def report() -> str:
    return REPORT.read_text()


def test_the_summary_is_the_pre_registered_cell(summary: dict) -> None:
    """The cell, its frozen revision and its block are the registered ones."""
    assert summary["profile"] == study.LONG_W24_PROFILE
    assert summary["report_section"] == "5.52"
    assert summary["code_revision"] == study.PROFILES[study.LONG_W24_PROFILE]["run_revision"]
    assert summary["code_revision"] == "f636990"
    assert summary["alpha"] == 0.05
    assert len(summary["seeds"]) == 576
    assert summary["seeds"][0] == 87600
    assert summary["seeds"][-1] == 88175
    config = summary["config"]
    assert (config["depth"], config["hidden_size"]) == (4, 24)
    assert (config["steps"], config["mutation_interval"]) == (192, 48)


def test_the_two_primaries_read_as_registered(summary: dict) -> None:
    """P2's own conjunction: 550/576 against the bar of 313, mean negative."""
    cell = summary["comparisons"]["adaptive_vs_null"]
    assert cell["n"] == 576
    assert cell["wins"] == 550
    assert cell["min_wins_bonferroni"] == 313
    assert cell["passes_bonferroni"] is True
    assert cell["mean_delta"] == pytest.approx(-0.204292, abs=5e-6)
    assert cell["binomial_p"] == pytest.approx(3.50e-129, rel=0.05)


def test_the_arms_order_as_the_report_quotes_them(summary: dict) -> None:
    """adaptive < random < null - value creation and a controller edge together."""
    means = summary["arm_means"]
    assert means["adaptive"] == pytest.approx(5.4605, abs=5e-4)
    assert means["random"] == pytest.approx(5.5010, abs=5e-4)
    assert means["null"] == pytest.approx(5.6648, abs=5e-4)
    assert means["adaptive"] < means["random"] < means["null"]


def test_the_controller_edge_is_the_narrow_cells_larger_one(summary: dict) -> None:
    """The descriptive reading the report leans on: the edge is 390/576 here.

    Against the width-96 long cell's 335/576 - the comparison the verdict's
    rank-sum reports at p 3.85e-16.
    """
    edge = summary["comparisons"]["adaptive_vs_random"]
    assert edge["wins"] == 390
    assert edge["mean_delta"] == pytest.approx(-0.040483, abs=5e-6)
    wide = json.loads(
        (ROOT / "artifacts" / "study-summaries" / "width-d4-dose2-long-576.json").read_text()
    )["comparisons"]["adaptive_vs_random"]
    assert wide["wins"] == 335
    assert edge["mean_delta"] < wide["mean_delta"]  # more negative = more edge
    # And both are real on their own cells' terms.
    assert edge["passes_bonferroni"] is True
    assert wide["passes_bonferroni"] is True


def test_the_churn_reading_is_not_a_collapse(summary: dict) -> None:
    """At width 24 the longer budget has no churn harm to relieve (62/576)."""
    churn = summary["comparisons"]["null_vs_random"]
    assert churn["wins"] == 62
    assert churn["mean_delta"] == pytest.approx(0.163809, abs=5e-6)
    assert churn["passes_bonferroni"] is False


def test_the_built_in_checks_held(summary: dict) -> None:
    checks = summary["built_in_checks"]
    assert checks == {
        "mutations_adaptive_3": True,
        "mutations_null_0": True,
        "mutations_random_3": True,
        "null_search_flops_zero": True,
    }


def test_the_summary_carries_this_arms_label_not_the_5_9_vocabulary(
    summary: dict,
) -> None:
    """The outcome names 5.52's own per-cell primary, not 5.9's branch 3.

    The committed summary read "branch 3 (1 FAIL / 2 PASS): no churn harm on this
    substrate" - the 5.9 vocabulary, whose primaries are a different pair, and
    which reads *backwards* for this cell: 5.52 recorded a collapse (+0.163809),
    not its absence. Rewritten at the source (post-run correction 2026-09-21);
    no measured number moved.
    """
    assert summary["outcome"].startswith("cell creates value (5.52's per-cell primary")
    assert summary["bonferroni_alpha"] == pytest.approx(0.025)
    assert "no churn harm" not in summary["outcome"]


def test_the_report_quotes_the_recorded_branch(report: str) -> None:
    """The outcome names the branch, and the 2x2's four corners, as recorded."""
    assert "Registered branch 1: P1+P2 PASS." in report
    for value in ("+0.163809", "+0.708276", "5.73e-190", "550/576", "-0.204292"):
        assert value in report, f"the report does not quote {value}"
    # Every corner of the 2x2 is named, so the branch can be checked.
    for profile in (
        study.LONG_W24_SAME_WIDTH,
        study.DOSE2_LONG_REFERENCE,
        study.LONG_W24_PROFILE,
        study.LONG_W24_SAME_BUDGET,
    ):
        assert f"`{profile}`" in report
    # And the levels reversal, which is the claim the control exists for.
    assert "the absolute levels reverse with the budget" in report
    assert "12.4413 / 12.5485 / 12.4785" in report
    assert "3.8333 / 3.8399 / 4.5482" in report


def test_the_verdict_still_reports_this_branch(
    capsys: pytest.CaptureFixture,
) -> None:
    """Running the pre-registered verdict on the committed shards reproduces it.

    The strongest form of the pin: not the summary's stored numbers but the
    verdict function itself, on the shards on disk.
    """
    assert study.long_w24_verdict(20, study.ALPHA) == 0
    out = capsys.readouterr().out
    assert "-> collapse confined to width 96: True" in out
    assert "-> creates value: True" in out
    assert "5.51's relief is a capacity effect" in out
