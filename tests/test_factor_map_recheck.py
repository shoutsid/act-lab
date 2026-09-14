"""Tests for the factor-map recheck (scripts/factor_map_recheck.py).

The recheck is a reproduction check with a declared rule: a published study
**reproduces** when its alpha=0.05 verdict label is unchanged on the decoupled
arm-RNG path with the published config, seed block and treatment arm, and
**flips** otherwise. These tests pin the parts that decide that call - which
studies are in the recheck set, that only the RNG regime is changed, that shard
records merge into one seed list, that the threshold is recomputed from n (so a
truncated shard set cannot be scored against the full block's bar), and that an
incomplete seed set never reports agreement.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import factor_map_recheck as recheck  # noqa: E402


def _published(
    *,
    seeds: list[int],
    wins: int,
    trials: int | None = None,
    treatment: str = "adaptive",
) -> dict:
    trials = len(seeds) if trials is None else trials
    return {
        "config": {"steps": 48, "device": "auto", "arms": "random,adaptive"},
        "seeds": seeds,
        "treatment": treatment,
        "adaptive_wins": wins,
        "gate_min_wins": recheck._gate_min_wins(trials, 0.05),
        "gate_passed": wins >= recheck._gate_min_wins(trials, 0.05),
        "gate_p_value": recheck._binomial_survival(wins, trials),
        "mean_adaptive_delta": -0.01,
    }


def _record(study: str, rows: list[tuple[int, float, float]]) -> dict:
    return {
        "study": study,
        "shard": 0,
        "shards": 1,
        "independent_arm_rng": True,
        "treatment": "adaptive",
        "experiments": [
            {
                "config": {"seed": seed},
                "arms": [
                    {
                        "mode": "adaptive",
                        "final_loss": treatment_loss,
                        "search_flops": 11,
                        "training_flops": 100,
                    },
                    {
                        "mode": "random",
                        "final_loss": random_loss,
                        "search_flops": 0,
                        "training_flops": 100,
                    },
                ],
            }
            for seed, treatment_loss, random_loss in rows
        ],
    }


def test_recheck_set_covers_the_reconstructible_studies() -> None:
    available = recheck.recheck_set()
    # The eight probe-loss gate studies plus the guided upper bound.
    assert "gate3-probe-loss-384" in available
    assert "gate3-horizon-384" in available
    assert "guided-upper-bound-96" in available
    # The signal-era study cannot be reconstructed from its recorded config.
    assert "gate3-legacy-signal-384" not in available
    assert "gate3-legacy-signal-384" in recheck.EXCLUDED
    # The analysis-only artifact carries no config and is not a study, and the
    # recheck's own comparison artifact (a list of rows) must not be read as one.
    assert "probe-validity-analysis" not in available
    assert "factor-map-recheck" not in available
    assert "factor-map-recheck" not in recheck.published_summaries()


def test_build_config_changes_only_the_rng_regime() -> None:
    published = {
        "config": {
            "steps": 144,
            "device": "auto",
            "probe_candidate_cap": 2,
            "probe_batch_count": 1,
            "mutation_interval": 12,
        },
        "seeds": [1, 2],
        "treatment": "adaptive",
        "adaptive_wins": 1,
        "gate_min_wins": 2,
        "gate_passed": False,
        "gate_p_value": 0.5,
        "mean_adaptive_delta": 0.0,
    }

    config = recheck.build_config(published)

    assert config.independent_arm_rng is True
    assert config.device == "cpu"
    # Everything the claim depends on is carried over verbatim.
    assert config.steps == 144
    assert config.probe_candidate_cap == 2
    assert config.probe_batch_count == 1
    assert config.mutation_interval == 12


def test_run_shard_rejects_a_non_reconstructible_study() -> None:
    import pytest

    with pytest.raises(SystemExit, match="not in the recheck set"):
        recheck.run_shard("gate3-legacy-signal-384", shard=0, shards=1)
    with pytest.raises(SystemExit, match="not in the recheck set"):
        recheck.run_shard("probe-validity-analysis", shard=0, shards=1)


def test_aggregate_merges_shards_in_seed_order() -> None:
    records = [
        _record("s", [(3001, 1.0, 1.1), (3003, 1.0, 1.1)]),
        _record("s", [(3000, 1.0, 1.1), (3002, 1.0, 1.1)]),
    ]

    merged = recheck.aggregate(records)

    assert [row[0] for row in merged["s"]] == [3000, 3001, 3002, 3003]


def test_compare_reports_agreement_when_the_label_is_unchanged() -> None:
    published = {"s": _published(seeds=list(range(20)), wins=15)}
    # 16 of 20 -> still a PASS (needs 15).
    rows = recheck.compare(
        [_record("s", [(seed, 1.0 if seed < 16 else 1.2, 1.1) for seed in range(20)])],
        published,
    )

    row = rows[0]
    assert row["recheck_wins"] == 16
    assert row["recheck_passed"] is True
    assert row["published_passed"] is True
    assert row["seeds_complete"] is True
    assert row["label_agreement"] is True
    assert row["win_delta"] == 1


def test_compare_flags_a_label_flip() -> None:
    published = {"s": _published(seeds=list(range(20)), wins=15)}
    # 14 of 20 loses the published PASS.
    rows = recheck.compare(
        [_record("s", [(seed, 1.0 if seed < 14 else 1.2, 1.1) for seed in range(20)])],
        published,
    )

    row = rows[0]
    assert row["published_passed"] is True
    assert row["recheck_passed"] is False
    assert row["label_agreement"] is False


def test_compare_recomputes_the_threshold_from_n() -> None:
    # A truncated shard set (10 of the 20 seeds) must not be judged against the
    # full block's bar: 9/10 is a PASS at n=10 (needs 9), though it would fail
    # the n=20 threshold (15).
    published = {"s": _published(seeds=list(range(20)), wins=15)}
    rows = recheck.compare(
        [_record("s", [(seed, 1.0 if seed < 9 else 1.2, 1.1) for seed in range(10)])],
        published,
    )

    row = rows[0]
    assert row["trials"] == 10
    assert row["recheck_min_wins"] == 9
    assert row["recheck_passed"] is True
    # ...but the block is not complete, so it is never reported as agreement.
    assert row["seeds_complete"] is False
    assert row["label_agreement"] is False


def test_compare_marks_an_incomplete_seed_set() -> None:
    published = {"s": _published(seeds=[1, 2, 3, 4], wins=3)}
    rows = recheck.compare([_record("s", [(1, 1.0, 1.1), (2, 1.0, 1.1)])], published)
    assert rows[0]["seeds_complete"] is False
    assert rows[0]["trials"] == 2


# --- emitting decoupled summaries for the figures ---------------------------


def test_summaries_from_records_emits_the_committed_schema() -> None:
    """The figures need full summaries, not just win counts.

    Each emitted summary is put through ``figures_data.check_summary`` inside
    ``summaries_from_records``, so a summary that does not recompute cannot
    reach a figure; this pins the schema and the FLOPs aggregation that the
    probe-cost figure depends on.
    """
    from act.evolution.figures_data import check_summary

    rows = [(seed, 1.0 if seed < 15 else 1.2, 1.1) for seed in range(20)]

    summary = recheck.summaries_from_records([_record("s", rows)])["s"]

    assert summary["adaptive_wins"] == 15
    assert summary["gate_min_wins"] == recheck._gate_min_wins(20, 0.05)
    assert summary["gate_passed"] is True
    assert summary["provenance"] == "decoupled-independent-arm-rng"
    assert summary["treatment"] == "adaptive"
    assert [item["config"]["seed"] for item in summary["experiments"]] == list(
        range(20)
    )
    assert summary["search_flops_by_arm"] == {"adaptive": 220, "random": 0}
    assert summary["training_flops_by_arm"] == {"adaptive": 2000, "random": 2000}
    # The stored verdict must recompute - this is the figure pipeline's own check.
    check_summary(summary, "s")


def test_summaries_from_records_merges_shards_in_seed_order() -> None:
    summaries = recheck.summaries_from_records(
        [
            _record("s", [(3001, 1.0, 1.1)]),
            _record("s", [(3000, 1.0, 1.1)]),
        ]
    )
    assert [item["config"]["seed"] for item in summaries["s"]["experiments"]] == [
        3000,
        3001,
    ]
    assert summaries["s"]["adaptive_wins"] == 2
