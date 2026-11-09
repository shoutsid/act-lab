"""Section 5.70's recorded result: the declared-shape real-text block.

The registration predicted the development-fidelity negative would replicate -
adaptive at most 3 of 12 - on the real TinyStories stream at the declared
285M/134M shape. The block read 9 of 12: the prediction missed in direction and
the campaign's 10-of-12 bar did not clear, so the registered branch is C, the
same withholding as the synthetic block.

These tests read the committed summary rather than transcribing it, and they
recompute every statistic the report quotes from the summary's own per-seed
rows. A hand-edited summary or a drifted reading fails here.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "artifacts" / "study-summaries" / "real-text-declared-12.json"
NOTES = ROOT / "artifacts" / "real-text-declared" / "NOTES.md"


def _record() -> dict:
    return json.loads(SUMMARY.read_text())


def test_the_summary_is_the_registered_block_and_revision() -> None:
    summary = _record()
    assert summary["report_section"] == "5.70"
    assert summary["kind"] == "real-text-declared-5.70"
    assert summary["code_revision"] == "8fe7bbff1b3f1e10e08260e6c434e43a21085a83"
    assert summary["seeds"] == list(range(15036, 15048))
    assert [row["seed"] for row in summary["per_seed"]] == summary["seeds"]
    assert summary["declared"]["shape"] == "285M total / 134M active"
    assert summary["declared"]["tokens_per_arm"] == 24_576_000
    assert summary["declared_bar_wins"] == 10
    assert summary["alpha"] == 0.05


def test_every_record_is_a_complete_paired_row() -> None:
    rows = _record()["per_seed"]
    assert len(rows) == 12
    for row in rows:
        assert row["adaptive_mutations"] == row["random_mutations"] == 3
        assert math.isfinite(row["adaptive_loss"]) and math.isfinite(row["random_loss"])
        assert row["adaptive_loss"] != row["random_loss"]
        assert row["delta"] == pytest.approx(
            row["adaptive_loss"] - row["random_loss"], abs=1e-12
        )
        assert row["wall_seconds"] > 0
        assert row["adaptive_active_parameters"] > 100_000_000


def test_the_reading_is_nine_of_twelve_and_short_of_the_bar() -> None:
    summary = _record()
    rows = summary["per_seed"]
    deltas = [row["delta"] for row in rows]
    wins = sum(delta < 0 for delta in deltas)
    ties = sum(delta == 0 for delta in deltas)
    assert (wins, len(deltas) - wins - ties, ties) == (9, 3, 0)
    primary = summary["primary"]
    assert primary["wins"] == wins
    assert primary["pairs"] == len(deltas)
    assert primary["ties"] == ties
    assert primary["bar"] == 10
    assert primary["passes"] is False
    assert primary["branch"] == "C"
    assert primary["reading"] == "not demonstrated"
    # The exact one-sided tail at >= 9 wins, which is the number the report quotes.
    p_at_nine = sum(math.comb(12, k) for k in range(9, 13)) / 2**12
    assert p_at_nine == 0.072998046875
    assert primary["binomial_p"] == pytest.approx(p_at_nine)
    assert "inconclusive" in primary["consequence"]


def test_the_quoted_mean_and_spread_recompute_from_the_rows() -> None:
    summary = _record()
    deltas = [row["delta"] for row in summary["per_seed"]]
    assert statistics.mean(deltas) == pytest.approx(-0.013136, abs=1e-6)
    assert statistics.stdev(deltas) == pytest.approx(0.022574, abs=1e-6)
    assert summary["primary"]["mean_delta"] == pytest.approx(statistics.mean(deltas))
    assert summary["primary"]["sample_sd"] == pytest.approx(statistics.stdev(deltas))
    assert summary["primary"]["min_delta"] == pytest.approx(min(deltas))
    assert summary["primary"]["max_delta"] == pytest.approx(max(deltas))
    # The three losses are all in wave c, which the report records as descriptive.
    losses = [row["seed"] for row in summary["per_seed"] if row["delta"] > 0]
    assert losses == [15044, 15045, 15046]


def test_the_builder_checks_are_all_satisfied() -> None:
    checks = _record()["built_in_checks"]
    assert checks, "the reader recorded no built-in checks"
    assert all(checks.values()), checks
    assert checks["final_loss_is_the_held_out_validation_loss"] is True
    assert checks["all_twelve_seeds_present"] is True
    assert checks["one_code_revision"] is True
    assert checks["three_events_on_the_registered_steps"] is True


def test_the_caveats_are_the_ones_the_registration_declared() -> None:
    caveats = " ".join(_record()["caveats"]).lower()
    assert "40.04m-token stream" in caveats
    assert "24.576m per arm" in caveats
    assert "32768" in caveats and "8191" in caveats
    assert "24576 embedding rows" in caveats
    notes = NOTES.read_text()
    assert "9 of 12" in notes
    assert "two versions" in notes
    assert "~25.7 h" in notes


def test_the_measured_cost_is_the_block_total() -> None:
    summary = _record()
    hours = summary["observed"]["wall_hours"]
    assert hours[0] == pytest.approx(1.467674, abs=1e-6)
    assert hours[1] == pytest.approx(1.783803, abs=1e-6)
    total = sum(row["wall_seconds"] for row in summary["per_seed"]) / 3600
    assert total == pytest.approx(19.404858, abs=1e-6)
    assert hours[2] == pytest.approx(total)
