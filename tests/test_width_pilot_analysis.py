"""The width pilot's reader computes the registered statistics and guards them.

The registration fixes 12 paired seeds, three events at 100/200/300, and a
positive edge at 10 or more wins. These tests pin the exact sign test, the
paired mean and standard deviation, the final-width/accepted-event bookkeeping,
and the two conditions that make a reading impossible: an incomplete block and
an extra event outside the registered schedule.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_width_pilot as analysis  # noqa: E402

SEEDS = analysis.SEEDS
WIDTH = "width"
RANDOM = "width_random"
NULL = analysis.NULL_ARM


def _record(
    seed: int,
    arm: str,
    loss: float,
    events: list[tuple[int, bool]],
) -> dict[str, object]:
    """One synthetic record with the registration's fixed config."""

    accepted = sum(1 for _, kept in events if kept)
    hidden = 320 + 128 * accepted
    heads = 5 + 2 * accepted
    history = [
        {
            "step": step,
            "operation": "widen" if kept else "widen_declined",
            "policy": arm,
        }
        for step, kept in events
    ]
    return {
        "kind": "width-pilot",
        "seed": seed,
        "arm": arm,
        "config": {
            "arms": [arm],
            "total_steps": 400,
            "mutation_interval": 100,
            "mutation_budget": 3,
            "hidden_size": 320,
            "attention_heads": 5,
            "width_heads_added": 2,
            "width_accept_margin": 0.0,
            "width_random_probability": 0.5,
        },
        "final_hidden_size": hidden,
        "final_heads": heads,
        "arm_result": {
            "validation_loss": loss,
            "mutation_count": accepted,
            "mutation_history": history,
        },
    }


def _block(
    wins: int,
    random_events: list[tuple[int, bool]] | None = None,
) -> dict[tuple[int, str], dict[str, object]]:
    """A 12-pair block whose adaptive arm wins the first ``wins`` seeds."""

    random_events = random_events or [(100, True), (200, False), (300, True)]
    records: dict[tuple[int, str], dict[str, object]] = {}
    for index, seed in enumerate(SEEDS):
        width_loss, random_loss = (3.0, 3.1) if index < wins else (3.1, 3.0)
        records[(seed, WIDTH)] = _record(
            seed, WIDTH, width_loss, [(100, True), (200, True), (300, True)]
        )
        records[(seed, RANDOM)] = _record(seed, RANDOM, random_loss, random_events)
    return records


def _null_record(seed: int, loss: float) -> dict[str, object]:
    """One do-nothing record: three consumed events, no graft, width unchanged."""

    record = _record(seed, NULL, loss, [])
    record["config"]["arms"] = [NULL]  # type: ignore[index]
    record["final_hidden_size"] = 320
    record["final_heads"] = 5
    record["arm_result"]["mutation_count"] = 0  # type: ignore[index]
    record["arm_result"]["mutation_history"] = [  # type: ignore[index]
        {"step": step, "operation": "null", "policy": "null"}
        for step in (100, 200, 300)
    ]
    return record


def test_the_pool_term_is_not_run_without_null_records() -> None:
    pool = analysis.summarize(_block(12))["pool"]

    assert pool["status"] == "not_run"
    assert pool["n"] == 0
    assert pool["missing_jobs"] == [f"{seed}:null" for seed in SEEDS]


def test_a_complete_null_arm_produces_a_positive_pool_term() -> None:
    records = _block(1)
    for seed in SEEDS:
        records[(seed, NULL)] = _null_record(seed, 3.5)

    pool = analysis.summarize(records)["pool"]

    assert pool["status"] == "complete"
    assert pool["n"] == 12
    assert pool["wins"] == 12
    assert pool["losses"] == 0
    assert pool["mean_delta"] < 0
    assert pool["sign_test_p"] == pytest.approx(1 / 4096)
    assert pool["outcome"].startswith("pool positive")


def test_a_pool_below_the_bar_is_flat_and_closes_the_gene() -> None:
    records = _block(1)
    for seed in SEEDS:
        records[(seed, NULL)] = _null_record(seed, 2.0)

    pool = analysis.summarize(records)["pool"]

    assert pool["status"] == "complete"
    assert pool["wins"] == 0
    assert pool["outcome"].startswith("pool flat or negative")


def test_a_null_record_that_widened_is_refused() -> None:
    records = _block(6)
    record = _null_record(71000, 3.5)
    record["arm_result"]["mutation_history"][0]["operation"] = "widen"  # type: ignore[index]
    records[(71000, NULL)] = record

    with pytest.raises(analysis.ReadingError, match="null arm"):
        analysis.summarize(records)


def test_a_null_arm_that_booked_an_extra_event_deviates_the_pool_not_the_edge() -> None:
    # Measured 2026-11-07: the pilot's null records booked a fourth, inert no-op
    # at step 400. The edge reading never used the null arm, so it stands; the
    # pool term cannot be called registered, and the direction is still reported.
    records = _block(1)
    for seed in SEEDS:
        record = _null_record(seed, 3.5)
        record["arm_result"]["mutation_history"].append(  # type: ignore[index]
            {"step": 400, "operation": "null", "policy": "null"}
        )
        records[(seed, NULL)] = record

    summary = analysis.summarize(records)

    assert summary["status"] == "complete"
    assert summary["registered_reading"] is True
    assert summary["protocol"]["ok"] is True
    pool = summary["pool"]
    assert pool["status"] == "protocol_deviation"
    assert pool["registered_reading"] is False
    assert len(pool["protocol"]["deviations"]) == 12
    assert all(
        deviation["extra_steps"] == [400]
        for deviation in pool["protocol"]["deviations"]
    )
    assert pool["wins"] == 12
    assert "descriptive term: pool positive" in pool["outcome"]
    assert "pool protocol: FAILED" in analysis.render(summary)


def test_a_partial_null_block_leaves_the_pool_undecided() -> None:
    records = _block(12)
    records[(71000, NULL)] = _null_record(71000, 3.5)

    pool = analysis.summarize(records)["pool"]

    assert pool["status"] == "incomplete"
    assert "71011:null" in pool["missing_jobs"]
    assert pool["outcome"].startswith("UNDECIDABLE")


def test_a_complete_ten_win_block_is_positive_at_the_registered_bar() -> None:
    summary = analysis.summarize(_block(10))

    assert summary["status"] == "complete"
    assert summary["registered_reading"] is True
    assert summary["protocol"]["ok"] is True
    comparison = summary["comparison"]
    assert comparison["n"] == 12
    assert comparison["wins"] == 10
    assert comparison["losses"] == 2
    assert comparison["ties"] == 0
    # 79/4096 is the registration's exact one-sided level at 10 of 12.
    assert comparison["sign_test_p"] == pytest.approx(79 / 4096)
    deltas = [-0.1] * 10 + [0.1] * 2
    assert comparison["mean_delta"] == pytest.approx(statistics.fmean(deltas))
    assert comparison["paired_sd"] == pytest.approx(statistics.stdev(deltas))
    assert "positive and clear" in summary["outcome"]

    schedule = summary["arm_schedule"]
    assert schedule[WIDTH] == {
        "pairs": 12,
        "accepted": 36,
        "events": 36,
        "mean_accepted": 3.0,
        "final_widths": {"704": 12},
    }
    assert schedule[RANDOM]["accepted"] == 24
    assert schedule[RANDOM]["events"] == 36
    assert schedule[RANDOM]["final_widths"] == {"576": 12}


def test_nine_wins_is_flat_at_the_registered_bar() -> None:
    summary = analysis.summarize(_block(9))

    assert summary["status"] == "complete"
    assert summary["registered_reading"] is True
    assert summary["comparison"]["wins"] == 9
    assert summary["comparison"]["sign_test_p"] == pytest.approx(299 / 4096)
    assert "flat or negative" in summary["outcome"]


def test_an_incomplete_block_is_undecidable_even_when_every_pair_wins() -> None:
    records = _block(12)
    # Mirror the fetched state: 71007's width arm is missing, and both arms of
    # 71008-71011 have not run; 71007's coin arm is present.
    del records[(71007, WIDTH)]
    for seed in SEEDS[8:]:
        del records[(seed, WIDTH)]
        del records[(seed, RANDOM)]

    summary = analysis.summarize(records)

    assert summary["status"] == "incomplete"
    assert summary["registered_reading"] is False
    assert summary["complete_pairs"] == list(SEEDS[:7])
    assert summary["missing_jobs"] == [
        "71007:width",
        "71008:width",
        "71008:width_random",
        "71009:width",
        "71009:width_random",
        "71010:width",
        "71010:width_random",
        "71011:width",
        "71011:width_random",
    ]
    assert summary["comparison"]["n"] == 7
    assert summary["comparison"]["wins"] == 7
    assert summary["outcome"].startswith("UNDECIDABLE - block incomplete")


def test_an_extra_event_invalidates_the_registered_reading() -> None:
    # The session budget counts accepted mutations, so a declining coin arm can
    # reach step 400 - after training but before validation. The script must
    # report that rather than call the resulting deltas the registered reading.
    summary = analysis.summarize(
        _block(12, random_events=[(100, True), (200, False), (300, True), (400, True)])
    )

    assert summary["status"] == "protocol_deviation"
    assert summary["registered_reading"] is False
    assert summary["protocol"]["ok"] is False
    assert len(summary["protocol"]["deviations"]) == 12
    assert all(
        deviation["extra_steps"] == [400]
        for deviation in summary["protocol"]["deviations"]
    )
    assert "did not run the registered three events" in summary["outcome"]
    # The descriptive statistics are still computed, and still 12/12 wins.
    assert summary["comparison"]["wins"] == 12


def test_a_missing_registered_event_also_deviates() -> None:
    records = _block(12)
    records[(71000, WIDTH)] = _record(
        71000, WIDTH, 3.0, [(100, True), (200, True)]
    )

    summary = analysis.summarize(records)

    assert summary["status"] == "protocol_deviation"
    assert summary["protocol"]["deviations"] == [
        {
            "seed": 71000,
            "arm": WIDTH,
            "extra_steps": [],
            "missing_steps": [300],
        }
    ]


def test_a_config_mismatch_is_refused_before_it_enters_a_delta() -> None:
    records = _block(12)
    records[(71000, RANDOM)]["config"]["total_steps"] = 20  # type: ignore[index]

    with pytest.raises(analysis.ReadingError, match="total_steps"):
        analysis.summarize(records)


def test_a_final_width_that_disagrees_with_its_events_is_refused() -> None:
    records = _block(12)
    records[(71000, WIDTH)]["final_hidden_size"] = 576  # type: ignore[index]

    with pytest.raises(analysis.ReadingError, match="final width"):
        analysis.summarize(records)


def test_the_loader_ignores_the_kernel_summary_and_reads_records(tmp_path: Path) -> None:
    record = _record(71000, WIDTH, 3.0, [(100, True), (200, True), (300, True)])
    (tmp_path / "width-seed71000.json").write_text(json.dumps(record))
    (tmp_path / "summary.json").write_text(json.dumps({"kind": "width-pilot-summary"}))

    loaded = analysis.load_records(tmp_path)

    assert list(loaded) == [(71000, WIDTH)]


def test_the_rendered_report_names_the_bar_and_the_arm_schedule() -> None:
    text = analysis.render(analysis.summarize(_block(10)))

    assert "wins: 10/12" in text
    assert "704/11/3/3" in text
    assert "OUTCOME: edge positive and clear" in text
