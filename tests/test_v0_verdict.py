"""Tests for the machine-checked section 5.1 verdict (scripts/v0_verdict.py).

The verdict script applies the pre-registered decision rules of
RESEARCH_REPORT.md section 5.1 to the published seed dataset - aggregate
probe-search cost vs the 5% criterion, the [0.5, 2]x tolerance band around
the declared prediction, and the 10-of-12 anti-adaptivity harm guard. No
verdict number may be hand-computed and no rule may drift from the
registration, so these tests pin every branch of T4_RUNBOOK.md section 4's
exit-code contract: 0 = PASS (Gate 4 authorized only on a complete queue),
1 = FAIL (primary cost miss), 2 = PASS with qualification (band miss) or
the harm-guard rejection. Stray complete seeds (e.g. smoke-test leftovers)
must be excluded from the aggregate, and missing declared seeds must keep
``gate_4_authorized`` false.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import v0_verdict  # noqa: E402

from act.training import V0SessionConfig, run_v0_session  # noqa: E402
from act.training.session_plan import (  # noqa: E402
    expected_probe_cost_percent,
    tolerance_band,
)
from act.training.v0_handoff import (  # noqa: E402
    HANDOFF_DATASET,
    load_seed_summary,
    publish_seed,
    seed_directory,
)

DECLARED = list(v0_verdict.DECLARED_SEEDS)
TRAINING_FLOPS_PER_SEED = 10_000_000


def _write_seed(
    dataset_root: Path,
    seed: int,
    *,
    adaptive_loss: float,
    random_loss: float,
    search_flops: int,
    training_flops: int = TRAINING_FLOPS_PER_SEED,
    mutation_count: int = 3,
    adaptive_mutation_count: int | None = None,
    status: str = "complete",
) -> Path:
    """Materialize one published seed summary in the handoff layout."""

    def arm(mode: str, loss: float, mutations: int) -> dict[str, object]:
        return {
            "mode": mode,
            "final_loss": loss,
            "validation_loss": loss,
            "quality": 0.0,
            "mutation_count": mutations,
            "parameter_count": 1_000,
            "active_parameter_count": 800,
            "training_flops": training_flops,
            "search_flops": search_flops if mode == "adaptive" else 0,
            "architecture_hash": "0" * 64,
            "mutation_history": [],
            "wall_time_seconds": 1.0,
            "tokens_per_second": 1.0,
        }

    destination = seed_directory(dataset_root, seed)
    reports = destination / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": seed,
        "status": status,
        "protocol": "v0-compute-efficiency-5.1",
        "interrupted_step": None,
        "predicted_cost_ratio": expected_probe_cost_percent(1) / 100.0,
        "arms": {
            "random": arm("random", random_loss, mutation_count),
            "adaptive": arm(
                "adaptive",
                adaptive_loss,
                mutation_count if adaptive_mutation_count is None else adaptive_mutation_count,
            ),
        },
        "config": {},
    }
    (reports / "seed-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return destination


def _in_band_search_flops(probe_batch_count: int = 1, factor: float = 1.0) -> int:
    """Search FLOPs per seed measuring exactly ``factor`` x the prediction."""
    ratio = factor * expected_probe_cost_percent(probe_batch_count) / 100.0
    return round(ratio * TRAINING_FLOPS_PER_SEED)


def _populate(
    dataset_root: Path,
    *,
    search_flops: int,
    adaptive_wins: int | None = None,
) -> None:
    """Write all 12 declared seeds; ``adaptive_wins`` caps the win count."""
    for index, seed in enumerate(DECLARED):
        adaptive_won = adaptive_wins is None or index < adaptive_wins
        _write_seed(
            dataset_root,
            seed,
            adaptive_loss=1.0 if adaptive_won else 1.5,
            random_loss=1.5 if adaptive_won else 1.0,
            search_flops=search_flops,
        )


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    dataset_root: Path,
    *,
    probe_batch_count: int = 1,
    output: Path | None = None,
) -> int:
    argv = [
        "v0-verdict",
        "--dataset-root",
        str(dataset_root),
        "--probe-batch-count",
        str(probe_batch_count),
    ]
    if output is not None:
        argv += ["--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    return v0_verdict.main()


def _summary_at(dataset_root: Path, seed: int) -> dict:
    path = seed_directory(dataset_root, seed) / "reports" / "seed-summary.json"
    return json.loads(path.read_text())


def _rewrite_summary(dataset_root: Path, seed: int, summary: dict) -> None:
    path = seed_directory(dataset_root, seed) / "reports" / "seed-summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


# --- the registered PASS branch -------------------------------------------


def test_full_in_band_queue_passes_and_authorizes_gate_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _populate(tmp_path, search_flops=_in_band_search_flops())
    exit_code = _run_main(monkeypatch, tmp_path)  # default output beside the root

    assert exit_code == 0
    verdict = json.loads((tmp_path / "v0-5-1-verdict.json").read_text())
    assert verdict["verdict"] == "PASS"
    assert verdict["protocol"] == "v0-compute-efficiency-5.1"
    assert verdict["queue_complete"] is True
    assert verdict["seeds_complete"] == DECLARED
    assert verdict["seeds_missing"] == []
    assert verdict["gate_4_authorized"] is True
    assert verdict["primary_pass"] is True
    assert verdict["in_tolerance_band"] is True
    # Per-seed search FLOPs are integers, so the measured cost carries the
    # rounding of one seed's payload, not an exact identity.
    assert verdict["measured_cost_percent"] == pytest.approx(
        100.0 * _in_band_search_flops() / TRAINING_FLOPS_PER_SEED, rel=1e-12
    )
    assert verdict["harm_guard"] == {
        "adaptive_wins": 12,
        "adaptive_losses": 0,
        "pairs": 12,
        "loss_limit": v0_verdict.HARM_GUARD_LOSS_LIMIT,
        "clear": True,
    }
    assert [pair["seed"] for pair in verdict["per_seed"]] == DECLARED


def test_aggregate_cost_matches_the_toy_accounting_over_summaries(tmp_path: Path) -> None:
    factor = 1.3
    _populate(tmp_path, search_flops=_in_band_search_flops(factor=factor))

    verdict = v0_verdict.compute_verdict(tmp_path, probe_batch_count=1)
    summaries = [_summary_at(tmp_path, seed) for seed in DECLARED]
    total_search = sum(int(s["arms"]["adaptive"]["search_flops"]) for s in summaries)
    total_training = sum(int(s["arms"]["adaptive"]["training_flops"]) for s in summaries)

    assert math.isclose(
        verdict["measured_cost_percent"],
        100.0 * total_search / total_training,
        rel_tol=1e-12,
    )
    assert math.isclose(
        verdict["measured_cost_percent"],
        factor * expected_probe_cost_percent(1),
        rel_tol=1e-4,  # one integer-rounded search-FLOPs payload per seed
    )
    assert verdict["tolerance_band_percent"] == list(tolerance_band(1))


def test_probe_batch_count_two_rebases_the_band_without_retuning(tmp_path: Path) -> None:
    # A measurement at 0.7x the primary prediction: in the primary band, out
    # of the secondary's. The runbook's --probe-batch-count 2 re-bases the
    # band on the declared prediction; it never re-tunes the measurement.
    _populate(tmp_path, search_flops=_in_band_search_flops(factor=0.7))

    primary = v0_verdict.compute_verdict(tmp_path, probe_batch_count=1)
    secondary = v0_verdict.compute_verdict(tmp_path, probe_batch_count=2)

    assert primary["verdict"] == "PASS"
    assert secondary["verdict"] == "PASS_WITH_QUALIFICATION"
    assert primary["measured_cost_percent"] == secondary["measured_cost_percent"]
    assert secondary["predicted_cost_percent"] == pytest.approx(
        expected_probe_cost_percent(2), rel=1e-9
    )


def test_exactly_five_percent_still_passes_the_primary_criterion(tmp_path: Path) -> None:
    # The registered rule is "cost <= 5%": the boundary itself passes the
    # primary criterion (then qualifies on the band, as it must).
    _populate(
        tmp_path,
        search_flops=_in_band_search_flops(factor=5.0 / expected_probe_cost_percent(1)),
    )

    verdict = v0_verdict.compute_verdict(tmp_path, probe_batch_count=1)

    assert verdict["measured_cost_percent"] == pytest.approx(5.0, rel=1e-9)
    assert verdict["primary_pass"] is True
    assert verdict["verdict"] == "PASS_WITH_QUALIFICATION"


# --- queue discipline ------------------------------------------------------


def test_incomplete_queue_withholds_gate_4_but_still_passes_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _populate(tmp_path, search_flops=_in_band_search_flops())
    # Seed 9011 has only its deadline-stop snapshot: forensic, never judged.
    summary = _summary_at(tmp_path, 9011)
    summary["status"] = "in-progress"
    _rewrite_summary(tmp_path, 9011, summary)

    exit_code = _run_main(monkeypatch, tmp_path)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "WARNING: 1 declared seed(s) missing: [9011]" in out
    verdict = json.loads((tmp_path / "v0-5-1-verdict.json").read_text())
    assert verdict["queue_complete"] is False
    assert verdict["seeds_missing"] == [9011]
    assert verdict["verdict"] == "PASS"
    assert verdict["gate_4_authorized"] is False


def test_stray_complete_seeds_are_excluded_from_the_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _populate(tmp_path, search_flops=_in_band_search_flops())
    # A smoke-test leftover whose cost would explode the aggregate if it were
    # (wrongly) pooled, plus a non-numeric directory ignored outright.
    _write_seed(tmp_path, 20, adaptive_loss=0.5, random_loss=2.0, search_flops=10**15)
    junk = tmp_path / HANDOFF_DATASET / "seed-junk" / "reports"
    junk.mkdir(parents=True)
    (junk / "seed-summary.json").write_text(
        json.dumps({"seed": 1, "status": "complete"}) + "\n"
    )

    exit_code = _run_main(monkeypatch, tmp_path)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "outside the declared queue" in out
    assert "seed-junk" not in out
    verdict = json.loads((tmp_path / "v0-5-1-verdict.json").read_text())
    assert verdict["seeds_stray"] == [20]
    assert verdict["measured_cost_percent"] == pytest.approx(
        expected_probe_cost_percent(1), rel=1e-4
    )
    assert verdict["gate_4_authorized"] is True


# --- the FAIL branch -------------------------------------------------------


def test_primary_cost_over_criterion_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _populate(tmp_path, search_flops=_in_band_search_flops(factor=100.0))

    exit_code = _run_main(monkeypatch, tmp_path)

    assert exit_code == 1
    verdict = json.loads((tmp_path / "v0-5-1-verdict.json").read_text())
    assert verdict["verdict"] == "FAIL"
    assert verdict["primary_pass"] is False
    assert verdict["primary_criterion_percent"] == 5.0
    assert verdict["measured_cost_percent"] == pytest.approx(
        100.0 * expected_probe_cost_percent(1), rel=1e-4
    )
    assert verdict["gate_4_authorized"] is False


@pytest.mark.parametrize("factor", [3.0, 0.3])
def test_band_miss_qualifies_the_pass_in_both_directions(factor: float, tmp_path: Path) -> None:
    _populate(tmp_path, search_flops=_in_band_search_flops(factor=factor))

    verdict = v0_verdict.compute_verdict(tmp_path, probe_batch_count=1)

    assert verdict["primary_pass"] is True
    assert verdict["in_tolerance_band"] is False
    assert verdict["verdict"] == "PASS_WITH_QUALIFICATION"


# --- the harm guard --------------------------------------------------------


@pytest.mark.parametrize(
    "adaptive_wins,expected_verdict",
    [(3, "PASS"), (2, "PASS_COST_ONLY_HARM_GUARD_REJECTS")],
)
def test_harm_guard_rejects_at_ten_losses(
    adaptive_wins: int, expected_verdict: str, tmp_path: Path
) -> None:
    # The registered guard: adaptive must not lose in >= 10 of 12 pairs
    # (exact one-tailed sign test p = 0.0193), i.e. fewer than 3 wins rejects.
    _populate(tmp_path, search_flops=_in_band_search_flops(), adaptive_wins=adaptive_wins)

    verdict = v0_verdict.compute_verdict(tmp_path, probe_batch_count=1)

    assert verdict["verdict"] == expected_verdict
    assert verdict["harm_guard"]["adaptive_wins"] == adaptive_wins
    assert verdict["harm_guard"]["adaptive_losses"] == 12 - adaptive_wins
    assert verdict["harm_guard"]["clear"] is (adaptive_wins == 3)
    assert verdict["gate_4_authorized"] is (adaptive_wins == 3)
    assert sum(1 for pair in verdict["per_seed"] if pair["adaptive_wins"]) == adaptive_wins


# --- fail-fast guards ------------------------------------------------------


def test_absent_handoff_dataset_aborts(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no handoff dataset"):
        v0_verdict.compute_verdict(tmp_path, probe_batch_count=1)


def test_dataset_without_complete_versions_aborts(tmp_path: Path) -> None:
    _write_seed(
        tmp_path, 9000, adaptive_loss=1.0, random_loss=1.0, search_flops=1,
        status="in-progress",
    )
    with pytest.raises(SystemExit, match="no complete seed versions"):
        v0_verdict.compute_verdict(tmp_path, probe_batch_count=1)


def test_unmatched_mutation_budgets_abort(tmp_path: Path) -> None:
    _write_seed(
        tmp_path, 9000, adaptive_loss=1.0, random_loss=1.0, search_flops=1,
        adaptive_mutation_count=4,
    )
    with pytest.raises(SystemExit, match="unmatched mutation budgets"):
        v0_verdict.compute_verdict(tmp_path, probe_batch_count=1)


def test_seed_missing_an_arm_aborts(tmp_path: Path) -> None:
    _write_seed(tmp_path, 9000, adaptive_loss=1.0, random_loss=1.0, search_flops=1)
    summary = _summary_at(tmp_path, 9000)
    del summary["arms"]["adaptive"]
    _rewrite_summary(tmp_path, 9000, summary)

    with pytest.raises(SystemExit, match="random/adaptive pair"):
        v0_verdict.compute_verdict(tmp_path, probe_batch_count=1)


# --- integration with the real publish pipeline ----------------------------


def _session_config(seed: int) -> V0SessionConfig:
    return V0SessionConfig(
        seed=seed,
        arms=("random", "adaptive"),
        declared=False,
        device="cpu",
        batch_size=2,
        vocab_size=64,
        context_size=256,
        hidden_size=64,
        depth=2,
        attention_heads=4,
        expert_count=4,
        expert_width=32,
        total_steps=12,
        mutation_interval=6,
        mutation_budget=2,
        probe_candidate_cap=4,
        probe_batch_count=1,
        validation_batches=2,
        checkpoint_interval=4,
        max_checkpoints=2,
    )


def test_verdict_reads_the_real_publish_pipeline(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions"
    dataset_root = tmp_path / "dataset"
    result = run_v0_session(_session_config(9000), checkpoint_dir=session_dir)
    publish_seed(
        dataset_root, result, session_dir=session_dir,
        code_revision="test-rev", clean_working_dir=True,
    )

    verdict = v0_verdict.compute_verdict(dataset_root, probe_batch_count=1)

    assert verdict["seeds_complete"] == [9000]
    assert verdict["seeds_missing"] == list(range(9001, 9012))
    assert verdict["queue_complete"] is False
    assert verdict["gate_4_authorized"] is False
    summary = load_seed_summary(dataset_root, 9000)
    pair = verdict["per_seed"][0]
    assert pair["adaptive_loss"] == summary["arms"]["adaptive"]["final_loss"]
    assert pair["random_loss"] == summary["arms"]["random"]["final_loss"]
    assert pair["adaptive_wins"] == (
        summary["arms"]["adaptive"]["final_loss"] < summary["arms"]["random"]["final_loss"]
    )
    assert int(summary["arms"]["adaptive"]["search_flops"]) > 0
    assert int(summary["arms"]["adaptive"]["training_flops"]) > 0


def test_completed_toy_queue_is_judged_out_of_the_declared_band(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete toy-scale queue cannot pass the declared-scale band.

    The band exists to catch measuring the wrong model (section 5.1): the
    0.0942% prediction holds only for the declared 285M V0 config, so a full
    12-seed toy queue must land outside the band and stay unauthorized - the
    machine-check refuses to bless a substituted model.
    """
    session_dir = tmp_path / "sessions"
    dataset_root = tmp_path / "dataset"
    for seed in v0_verdict.DECLARED_SEEDS:
        result = run_v0_session(_session_config(seed), checkpoint_dir=session_dir)
        publish_seed(
            dataset_root, result, session_dir=session_dir,
            code_revision="test-rev", clean_working_dir=True,
        )

    exit_code = _run_main(monkeypatch, dataset_root)

    assert exit_code in (1, 2)  # toy-scale cost misses the declared band or the bar
    verdict = json.loads((dataset_root / "v0-5-1-verdict.json").read_text())
    assert verdict["queue_complete"] is True
    assert verdict["seeds_complete"] == DECLARED
    assert verdict["gate_4_authorized"] is False
    assert verdict["in_tolerance_band"] is False
    assert verdict["harm_guard"]["pairs"] == 12
    assert verdict["harm_guard"]["adaptive_wins"] == sum(
        1 for pair in verdict["per_seed"] if pair["adaptive_wins"]
    )
