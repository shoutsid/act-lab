"""Section 5.69's recorded result: the gate opens inside the ratio window.

The registration declared a ratio band (1.15-1.35) and a shut gate. The cell came
back 338/576 (bar 315), mean -0.069389, edge -0.338858, pool +0.269470: ratio
1.2575, inside the band, gate open. So the ratio window narrows to [1.10, 1.26]
and the width price to (123,006, 140,422], while the declared shut half is the
miss - the informative one, since it places the threshold below every cell that
bracketed it from above.

The summary is read from the artifact rather than transcribed: a hand-edited
summary or a drifted record set fails here instead of leaving a plausible
paragraph.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import null_toy_study as study  # noqa: E402

SUMMARY = ROOT / "artifacts" / "study-summaries" / "width-d3-dose5-576.json"


def _record() -> dict:
    return json.loads(SUMMARY.read_text())


def test_the_summary_is_the_frozen_revision_and_the_registered_block() -> None:
    summary = _record()
    assert summary["code_revision"] == "2920671"
    assert summary["report_section"] == "5.69"
    assert len(summary["seeds"]) == 576
    assert summary["seeds"][0] == 101000 and summary["seeds"][-1] == 101575
    assert summary["bonferroni_alpha"] == pytest.approx(study.ALPHA / 3)
    assert summary["comparisons"]["adaptive_vs_null"]["min_wins_bonferroni"] == 315


def test_the_cell_creates_value_at_the_bar() -> None:
    cell = _record()["comparisons"]["adaptive_vs_null"]
    assert cell["wins"] == 338
    assert cell["wins"] - cell["min_wins_bonferroni"] == 23
    assert cell["passes_bonferroni"] is True
    assert cell["mean_delta"] == pytest.approx(-0.069389, abs=1e-6)
    assert cell["binomial_p"] == pytest.approx(1.78e-05, rel=1e-2)
    assert "cell creates value" in _record()["outcome"]


def test_the_ratio_is_inside_the_declared_band_and_the_gate_is_open() -> None:
    """The ratio is the cell's own decomposition, not a restatement of the bar."""
    comparisons = _record()["comparisons"]
    edge = comparisons["adaptive_vs_random"]["mean_delta"]
    pool = -comparisons["null_vs_random"]["mean_delta"]
    assert edge == pytest.approx(-0.338858, abs=1e-6)
    assert pool == pytest.approx(+0.269470, abs=1e-6)
    ratio = abs(edge) / pool
    assert ratio == pytest.approx(1.2575, abs=1e-4)
    low, high = study.DOSE5_RATIO_BAND
    assert low <= ratio <= high, "the measured ratio is outside the declared band"
    # The declared gate half was the miss: it said shut, the cell opened.
    assert study.DOSE5_GATE_OPEN_DECLARED is False
    assert comparisons["adaptive_vs_null"]["passes_bonferroni"] is True
    # The window the branch prints is the one §10.1 and the report quote.
    assert f"[1.10, {ratio:.2f}]" == "[1.10, 1.26]"


def test_the_two_decomposed_terms_are_still_on_the_edge_dominated_side() -> None:
    """The pool term is positive here, which is what makes this the edge stage."""
    comparisons = _record()["comparisons"]
    pool = -comparisons["null_vs_random"]["mean_delta"]
    assert pool > 0
    assert abs(comparisons["adaptive_vs_random"]["mean_delta"]) > pool
