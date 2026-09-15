"""Tests for the machine-checked section 5.1 verdict (scripts/v0_verdict.py).

The verdict script applies the pre-registered decision rules of
RESEARCH_REPORT.md section 5.1 to the published seed dataset - aggregate
probe-search cost vs the 5% criterion, the [0.5, 2]x tolerance band around
the declared prediction, and the 10-of-12 anti-adaptivity harm guard. No
verdict number may be hand-computed and no rule may drift from the
registration, so these tests pin every branch of T4_RUNBOOK.md section 4's
exit-code contract: 0 = PASS (Gate 4 authorized only on a complete queue),
1 = FAIL (primary cost miss), 2 = PASS with qualification (band miss) or
the harm-guard rejection, 3 = no verdict (malformed invocation - kept
distinct from 1 so an operator error is never read as a cost FAIL). Stray
complete seeds (e.g. smoke-test leftovers)
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
    control_root: Path | None = None,
    control_wins: int = 0,
    null_root: Path | None = None,
    null_wins: int | None = None,
) -> None:
    """Write all 12 declared seeds; ``adaptive_wins`` caps the win count.

    With ``control_root`` the same pairs are also written as the adopted
    section 5.5 pool-matched loss-blind arm, ``control_wins`` of them won.
    With ``null_root`` a section 5.14 do-nothing queue is written as
    single-arm null sessions; ``null_wins`` counts the seeds where the
    adaptive arm beats the null arm (``None`` = adaptive always wins, the
    authorization-friendly default, mirroring ``adaptive_wins``).
    """
    for index, seed in enumerate(DECLARED):
        adaptive_won = adaptive_wins is None or index < adaptive_wins
        _write_seed(
            dataset_root,
            seed,
            adaptive_loss=1.0 if adaptive_won else 1.5,
            random_loss=1.5 if adaptive_won else 1.0,
            search_flops=search_flops,
        )
    if control_root is not None:
        for index, seed in enumerate(DECLARED):
            control_won = index < control_wins
            _write_seed(
                control_root,
                seed,
                adaptive_loss=1.0 if control_won else 1.5,
                random_loss=1.5 if control_won else 1.0,
                search_flops=search_flops,
            )
    if null_root is not None:
        for index, seed in enumerate(DECLARED):
            adaptive_beats = null_wins is None or index < null_wins
            _write_null_seed(
                null_root,
                seed,
                null_loss=2.0 if adaptive_beats else 0.5,
            )


def _write_null_seed(dataset_root: Path, seed: int, *, null_loss: float) -> None:
    """Materialize one single-arm null session summary (section 5.14 layout)."""
    destination = seed_directory(dataset_root, seed)
    reports = destination / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": seed,
        "status": "complete",
        "protocol": "v0-compute-efficiency-5.1",
        "interrupted_step": None,
        "predicted_cost_ratio": 0.0,
        "arms": {
            "null": {
                "mode": "null",
                "final_loss": null_loss,
                "validation_loss": null_loss,
                "quality": 0.0,
                "mutation_count": 0,
                "parameter_count": 1_000,
                "active_parameter_count": 800,
                "training_flops": TRAINING_FLOPS_PER_SEED,
                "search_flops": 0,
                "architecture_hash": "0" * 64,
                "mutation_history": [],
                "wall_time_seconds": 1.0,
                "tokens_per_second": 1.0,
            }
        },
        "config": {},
    }
    (reports / "seed-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    dataset_root: Path,
    *,
    probe_batch_count: int = 1,
    output: Path | None = None,
    control_root: Path | None = None,
    null_root: Path | None = None,
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
    if control_root is not None:
        argv += ["--control-root", str(control_root)]
    if null_root is not None:
        argv += ["--null-root", str(null_root)]
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
    control = tmp_path / "control"
    null = tmp_path / "null"
    _populate(
        tmp_path,
        search_flops=_in_band_search_flops(),
        control_root=control,
        null_root=null,
    )
    exit_code = _run_main(  # default output beside the root
        monkeypatch, tmp_path, control_root=control, null_root=null
    )

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
    # The adopted section 5.5 clause was evaluated and did not fire: the
    # loss-blind arm lost its own pairs, so the win demonstrates selection.
    assert verdict["pool_matched_control"]["evaluated"] is True
    assert verdict["pool_matched_control"]["adaptive_wins"] == 0
    assert verdict["pool_matched_control"]["control_passes_gate"] is False
    # The section 5.14 clause was evaluated and did not fire: the adaptive
    # arm beat the do-nothing arm on every seed, and the built-in checks held.
    assert verdict["null_control"]["evaluated"] is True
    assert verdict["null_control"]["adaptive_wins"] == 12
    assert verdict["null_control"]["clause_rejects"] is False
    assert all(
        pair["null_search_flops"] == 0 and pair["null_mutation_count"] == 0
        for pair in verdict["null_control"]["per_seed"]
    )


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
    control = tmp_path / "control"
    null = tmp_path / "null"
    _populate(
        tmp_path,
        search_flops=_in_band_search_flops(factor=0.7),
        control_root=control,
        null_root=null,
    )

    primary = v0_verdict.compute_verdict(
        tmp_path, probe_batch_count=1, control_root=control, null_root=null
    )
    secondary = v0_verdict.compute_verdict(
        tmp_path, probe_batch_count=2, control_root=control, null_root=null
    )

    assert primary["verdict"] == "PASS"
    assert secondary["verdict"] == "PASS_WITH_QUALIFICATION"
    assert primary["measured_cost_percent"] == secondary["measured_cost_percent"]
    assert secondary["predicted_cost_percent"] == pytest.approx(
        expected_probe_cost_percent(2), rel=1e-9
    )


def test_exactly_five_percent_still_passes_the_primary_criterion(tmp_path: Path) -> None:
    # The registered rule is "cost <= 5%": the boundary itself passes the
    # primary criterion (then qualifies on the band, as it must).
    control = tmp_path / "control"
    null = tmp_path / "null"
    _populate(
        tmp_path,
        search_flops=_in_band_search_flops(factor=5.0 / expected_probe_cost_percent(1)),
        control_root=control,
        null_root=null,
    )

    verdict = v0_verdict.compute_verdict(
        tmp_path, probe_batch_count=1, control_root=control, null_root=null
    )

    assert verdict["measured_cost_percent"] == pytest.approx(5.0, rel=1e-9)
    assert verdict["primary_pass"] is True
    assert verdict["verdict"] == "PASS_WITH_QUALIFICATION"


# --- queue discipline ------------------------------------------------------


def test_incomplete_queue_withholds_gate_4_but_still_passes_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    last = DECLARED[-1]
    control = tmp_path / "control"
    null = tmp_path / "null"
    _populate(
        tmp_path,
        search_flops=_in_band_search_flops(),
        control_root=control,
        null_root=null,
    )
    # The final seed has only its deadline-stop snapshot: forensic, never judged.
    summary = _summary_at(tmp_path, last)
    summary["status"] = "in-progress"
    _rewrite_summary(tmp_path, last, summary)

    exit_code = _run_main(
        monkeypatch, tmp_path, control_root=control, null_root=null
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert f"WARNING: 1 declared seed(s) missing: [{last}]" in out
    verdict = json.loads((tmp_path / "v0-5-1-verdict.json").read_text())
    assert verdict["queue_complete"] is False
    assert verdict["seeds_missing"] == [last]
    assert verdict["verdict"] == "PASS"
    assert verdict["gate_4_authorized"] is False


def test_stray_complete_seeds_are_excluded_from_the_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    control = tmp_path / "control"
    null = tmp_path / "null"
    _populate(
        tmp_path,
        search_flops=_in_band_search_flops(),
        control_root=control,
        null_root=null,
    )
    # A smoke-test leftover whose cost would explode the aggregate if it were
    # (wrongly) pooled, plus a non-numeric directory ignored outright.
    _write_seed(tmp_path, 20, adaptive_loss=0.5, random_loss=2.0, search_flops=10**15)
    junk = tmp_path / HANDOFF_DATASET / "seed-junk" / "reports" 
    junk.mkdir(parents=True)
    (junk / "seed-summary.json").write_text(
        json.dumps({"seed": 1, "status": "complete"}) + "\n"
    )

    exit_code = _run_main(
        monkeypatch, tmp_path, control_root=control, null_root=null
    )

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
    control = tmp_path / "control"
    null = tmp_path / "null"
    _populate(
        tmp_path,
        search_flops=_in_band_search_flops(factor=factor),
        control_root=control,
        null_root=null,
    )

    verdict = v0_verdict.compute_verdict(
        tmp_path, probe_batch_count=1, control_root=control, null_root=null
    )

    assert verdict["primary_pass"] is True
    assert verdict["in_tolerance_band"] is False
    assert verdict["verdict"] == "PASS_WITH_QUALIFICATION"


def test_harm_guard_takes_label_precedence_over_band_miss(tmp_path: Path) -> None:
    """Cost passes, band misses, and the harm guard rejects: the verdict
    string must surface the stronger blocker (anti-adaptivity), not the
    bookkeeping qualification. Both exit 2; the label is decision-relevant.
    """
    _populate(
        tmp_path,
        search_flops=_in_band_search_flops(factor=2.5),  # outside [0.5, 2]x, under 5%
        adaptive_wins=2,  # 10 losses -> harm guard rejects
    )

    verdict = v0_verdict.compute_verdict(tmp_path, probe_batch_count=1)

    assert verdict["primary_pass"] is True
    assert verdict["in_tolerance_band"] is False
    assert verdict["harm_guard"]["clear"] is False
    assert verdict["verdict"] == "PASS_COST_ONLY_HARM_GUARD_REJECTS"


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
    control = tmp_path / "control"
    null = tmp_path / "null"
    _populate(
        tmp_path,
        search_flops=_in_band_search_flops(),
        adaptive_wins=adaptive_wins,
        control_root=control,
        null_root=null,
    )

    verdict = v0_verdict.compute_verdict(
        tmp_path, probe_batch_count=1, control_root=control, null_root=null
    )

    assert verdict["verdict"] == expected_verdict
    assert verdict["harm_guard"]["adaptive_wins"] == adaptive_wins
    assert verdict["harm_guard"]["adaptive_losses"] == 12 - adaptive_wins
    assert verdict["harm_guard"]["clear"] is (adaptive_wins == 3)
    assert verdict["gate_4_authorized"] is (adaptive_wins == 3)
    assert sum(1 for pair in verdict["per_seed"] if pair["adaptive_wins"]) == adaptive_wins


# --- fail-fast guards ------------------------------------------------------


def test_malformed_inputs_exit_3_and_write_no_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator error must never be readable as the pre-registered FAIL.

    Exit 1 is reserved for ``FAIL`` (primary cost above the criterion); a
    wrong ``--dataset-root`` is a different event and gets its own code, so an
    unattended multi-session run cannot record "the amortization hypothesis is
    false" when the command simply pointed somewhere else.
    """
    exit_code = _run_main(monkeypatch, tmp_path)  # no handoff dataset at all

    assert exit_code == v0_verdict.INPUT_ERROR_EXIT_CODE
    assert exit_code != 1
    assert not (tmp_path / "v0-5-1-verdict.json").exists()


def test_input_error_code_is_distinct_from_every_verdict_branch() -> None:
    assert v0_verdict.INPUT_ERROR_EXIT_CODE == 3
    assert v0_verdict.INPUT_ERROR_EXIT_CODE not in (0, 1, 2)


def test_absent_handoff_dataset_aborts(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no handoff dataset"):
        v0_verdict.compute_verdict(tmp_path, probe_batch_count=1)


def test_dataset_without_complete_versions_aborts(tmp_path: Path) -> None:
    _write_seed(
        tmp_path, DECLARED[0], adaptive_loss=1.0, random_loss=1.0, search_flops=1,
        status="in-progress",
    )
    with pytest.raises(SystemExit, match="no complete seed versions"):
        v0_verdict.compute_verdict(tmp_path, probe_batch_count=1)


def test_unmatched_mutation_budgets_abort(tmp_path: Path) -> None:
    _write_seed(
        tmp_path, DECLARED[0], adaptive_loss=1.0, random_loss=1.0, search_flops=1,
        adaptive_mutation_count=4,
    )
    with pytest.raises(SystemExit, match="unmatched mutation budgets"):
        v0_verdict.compute_verdict(tmp_path, probe_batch_count=1)


def test_seed_missing_an_arm_aborts(tmp_path: Path) -> None:
    _write_seed(tmp_path, DECLARED[0], adaptive_loss=1.0, random_loss=1.0, search_flops=1)
    summary = _summary_at(tmp_path, DECLARED[0])
    del summary["arms"]["adaptive"]
    _rewrite_summary(tmp_path, DECLARED[0], summary)

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
    first = DECLARED[0]
    result = run_v0_session(_session_config(first), checkpoint_dir=session_dir)
    publish_seed(
        dataset_root, result, session_dir=session_dir,
        code_revision="test-rev", clean_working_dir=True,
    )

    verdict = v0_verdict.compute_verdict(dataset_root, probe_batch_count=1)

    assert verdict["seeds_complete"] == [first]
    assert verdict["seeds_missing"] == list(range(first + 1, DECLARED[-1] + 1))
    assert verdict["queue_complete"] is False
    assert verdict["gate_4_authorized"] is False
    summary = load_seed_summary(dataset_root, first)
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


# --- re-reserving the declared block (section 5.5 change 4) ----------------


def test_parse_seeds_accepts_ranges_and_lists() -> None:
    assert v0_verdict.parse_seeds("12000-12003") == (12000, 12001, 12002, 12003)
    assert v0_verdict.parse_seeds(" 5, 7 ,9 ") == (5, 7, 9)
    with pytest.raises(SystemExit):
        v0_verdict.parse_seeds("5,5")
    with pytest.raises(SystemExit):
        v0_verdict.parse_seeds("  ")


def test_declared_block_can_be_re_reserved_without_loosening_a_rule(
    tmp_path: Path,
) -> None:
    """A re-reserved queue is a flag, not a code edit, and changes no rule.

    Section 5.5 change 4 moves the declared block off the seeds the real-data
    diagnostic already spent. The same 5% criterion, tolerance band, harm
    guard and queue-completeness requirement must apply to whatever block is
    declared, and the registered block must be untouched when no override is
    passed.
    """
    fresh = tuple(range(12000, 12012))
    control = tmp_path / "control"
    null = tmp_path / "null"
    for index, seed in enumerate(fresh):
        won = index < 6
        for root in (tmp_path, control):
            _write_seed(
                root,
                seed,
                adaptive_loss=1.0 if won else 1.5,
                random_loss=1.5 if won else 1.0,
                search_flops=_in_band_search_flops(),
            )
        _write_null_seed(null, seed, null_loss=2.0 if won else 2.5)

    # Default: the registered queue is still the analysis block, so the fresh
    # block reads as strays and the registered seeds as missing.
    default = v0_verdict.compute_verdict(tmp_path, probe_batch_count=1)
    assert default["declared_seeds"] == list(DECLARED)
    assert default["seeds_missing"] == list(DECLARED)
    assert default["seeds_stray"] == list(fresh)
    assert default["gate_4_authorized"] is False

    # Overridden: the block is judged on exactly the same rules.
    re_declared = v0_verdict.compute_verdict(
        tmp_path,
        probe_batch_count=1,
        declared_seeds=fresh,
        control_root=control,
        null_root=null,
    )
    assert re_declared["declared_seeds"] == list(fresh)
    assert re_declared["seeds_complete"] == list(fresh)
    assert re_declared["queue_complete"] is True
    assert re_declared["seeds_stray"] == []
    assert re_declared["measured_cost_percent"] == pytest.approx(
        expected_probe_cost_percent(1), rel=1e-4
    )
    assert re_declared["in_tolerance_band"] is True
    assert re_declared["harm_guard"]["adaptive_losses"] == 6
    assert re_declared["verdict"] == "PASS"
    assert re_declared["gate_4_authorized"] is True


# --- the adopted pool-matched control clause (section 5.5 changes 2 and 3) --


def test_pool_matched_control_that_also_passes_withholds_gate_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A loss-blind arm clearing the same bar means the pool alone suffices.

    Section 5.5 change 3: the cost PASS stands (exit 2, not 1), but Gate 4 is
    withheld because the adaptive win no longer demonstrates selection value
    over a candidate-placement effect.
    """
    control = tmp_path / "control"
    _populate(
        tmp_path,
        search_flops=_in_band_search_flops(),
        control_root=control,
        control_wins=v0_verdict.POOL_CONTROL_WIN_LIMIT,
    )

    exit_code = _run_main(monkeypatch, tmp_path, control_root=control)

    assert exit_code == 2
    out = capsys.readouterr().out
    assert "does not demonstrate selection value" in out
    verdict = json.loads((tmp_path / "v0-5-1-verdict.json").read_text())
    assert verdict["primary_pass"] is True
    assert verdict["in_tolerance_band"] is True
    assert verdict["harm_guard"]["clear"] is True
    assert verdict["verdict"] == "PASS_COST_ONLY_POOL_MATCHED_CONTROL_PASSES"
    assert verdict["gate_4_authorized"] is False
    assert verdict["pool_matched_control"]["control_passes_gate"] is True
    assert verdict["pool_matched_control"]["adaptive_wins"] == (
        v0_verdict.POOL_CONTROL_WIN_LIMIT
    )


def test_control_just_under_the_limit_keeps_gate_4_authorized(tmp_path: Path) -> None:
    # The clause is a threshold, not a gradient: one win below the limit the
    # loss-blind arm does not pass its own gate and authorization stands.
    control = tmp_path / "control"
    null = tmp_path / "null"
    _populate(
        tmp_path,
        search_flops=_in_band_search_flops(),
        control_root=control,
        control_wins=v0_verdict.POOL_CONTROL_WIN_LIMIT - 1,
        null_root=null,
    )

    verdict = v0_verdict.compute_verdict(
        tmp_path, probe_batch_count=1, control_root=control, null_root=null
    )

    assert verdict["verdict"] == "PASS"
    assert verdict["gate_4_authorized"] is True


def test_missing_control_withholds_gate_4_rather_than_skipping_the_clause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The adopted protocol requires the control; absence is disqualifying.

    Not running the control must never read as "no pool effect found".
    """
    _populate(tmp_path, search_flops=_in_band_search_flops())

    exit_code = _run_main(monkeypatch, tmp_path)

    assert exit_code == 2
    out = capsys.readouterr().out
    assert "no pool-matched control supplied" in out
    verdict = json.loads((tmp_path / "v0-5-1-verdict.json").read_text())
    assert verdict["primary_pass"] is True
    assert verdict["in_tolerance_band"] is True
    assert verdict["verdict"] == "PASS_COST_ONLY_CONTROL_MISSING"
    assert verdict["pool_matched_control"]["evaluated"] is False
    assert verdict["gate_4_authorized"] is False


def test_control_missing_an_arm_aborts(tmp_path: Path) -> None:
    control = tmp_path / "control"
    _populate(tmp_path, search_flops=_in_band_search_flops(), control_root=control)
    summary = _summary_at(control, DECLARED[0])
    del summary["arms"]["adaptive"]
    _rewrite_summary(control, DECLARED[0], summary)

    with pytest.raises(SystemExit, match="random/adaptive pair"):
        v0_verdict.compute_verdict(
            tmp_path, probe_batch_count=1, control_root=control
        )


# --- the section 5.14 do-nothing clause -------------------------------------


def test_null_clause_rejects_when_adaptive_fails_to_beat_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Adaptive losing-or-tying >= 10/12 vs null withholds Gate 4.

    Section 5.14: the registered harm guard only measures against churn arms;
    the do-nothing clause adds the bar the churn-harm results (5.8, 5.11,
    5.13) demand. The cost PASS stands (exit 2, not 1).
    """
    control = tmp_path / "control"
    null = tmp_path / "null"
    _populate(
        tmp_path,
        search_flops=_in_band_search_flops(),
        control_root=control,
        null_root=null,
        null_wins=2,  # adaptive beats null on only 2 of 12 seeds
    )

    exit_code = _run_main(
        monkeypatch, tmp_path, control_root=control, null_root=null
    )

    assert exit_code == 2
    out = capsys.readouterr().out
    assert "value over doing nothing" in out
    verdict = json.loads((tmp_path / "v0-5-1-verdict.json").read_text())
    assert verdict["primary_pass"] is True
    assert verdict["in_tolerance_band"] is True
    assert verdict["harm_guard"]["clear"] is True
    assert verdict["pool_matched_control"]["control_passes_gate"] is False
    assert verdict["verdict"] == "PASS_COST_ONLY_NULL_REJECTS"
    assert verdict["gate_4_authorized"] is False
    assert verdict["null_control"]["clause_rejects"] is True
    assert verdict["null_control"]["adaptive_losses_or_ties"] == (
        v0_verdict.NULL_WIN_LIMIT
    )


def test_null_clause_just_under_the_limit_keeps_gate_4_authorized(
    tmp_path: Path,
) -> None:
    # Threshold semantics, mirroring the harm guard: 9 adaptive losses vs
    # null sits one below the limit, the clause does not fire, and the
    # full authorization stands.
    control = tmp_path / "control"
    null = tmp_path / "null"
    _populate(
        tmp_path,
        search_flops=_in_band_search_flops(),
        control_root=control,
        null_root=null,
        null_wins=3,  # 12 - 3 = 9 losses-or-ties: just under the bar
    )

    verdict = v0_verdict.compute_verdict(
        tmp_path, probe_batch_count=1, control_root=control, null_root=null
    )

    assert verdict["verdict"] == "PASS"
    assert verdict["gate_4_authorized"] is True
    assert verdict["null_control"]["clause_rejects"] is False


def test_missing_null_queue_withholds_gate_4_rather_than_skipping_the_clause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The pre-registered protocol requires the do-nothing queue; absence is
    disqualifying, and must never read as "no null effect found"."""
    control = tmp_path / "control"
    _populate(tmp_path, search_flops=_in_band_search_flops(), control_root=control)

    exit_code = _run_main(monkeypatch, tmp_path, control_root=control)

    assert exit_code == 2
    out = capsys.readouterr().out
    assert "do-nothing queue was not supplied" in out
    verdict = json.loads((tmp_path / "v0-5-1-verdict.json").read_text())
    assert verdict["verdict"] == "PASS_COST_ONLY_NULL_MISSING"
    assert verdict["null_control"]["evaluated"] is False
    assert verdict["gate_4_authorized"] is False


def test_incomplete_null_queue_withholds_gate_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A partial null queue cannot authorize a full-block claim: the clause
    # must not silently evaluate over whatever seeds happen to exist.
    control = tmp_path / "control"
    null = tmp_path / "null"
    _populate(
        tmp_path,
        search_flops=_in_band_search_flops(),
        control_root=control,
        null_root=null,
    )
    _rewrite_summary(null, DECLARED[0], {"status": "in-progress", "arms": {}})

    exit_code = _run_main(
        monkeypatch, tmp_path, control_root=control, null_root=null
    )

    assert exit_code == 2
    verdict = json.loads((tmp_path / "v0-5-1-verdict.json").read_text())
    assert verdict["verdict"] == "PASS_COST_ONLY_NULL_MISSING"
    assert verdict["null_control"]["evaluated"] is False
    assert str(DECLARED[0]) in verdict["null_control"]["reason"]


def test_null_label_precedence_over_control_missing_and_band(
    tmp_path: Path,
) -> None:
    """The 5.14 clause outranks the missing-control and band labels."""
    null = tmp_path / "null"
    _populate(
        tmp_path,
        search_flops=_in_band_search_flops(factor=3.0),  # also a band miss
        null_root=null,
        null_wins=2,
    )

    verdict = v0_verdict.compute_verdict(tmp_path, probe_batch_count=1, null_root=null)

    assert verdict["verdict"] == "PASS_COST_ONLY_NULL_REJECTS"
    assert verdict["in_tolerance_band"] is False
    assert verdict["pool_matched_control"]["evaluated"] is False


def test_null_root_pointed_at_the_primary_dataset_aborts(tmp_path: Path) -> None:
    # Pointing --null-root at a pair-session dataset is an operator error the
    # machine-check must refuse, not misread: those summaries carry no null
    # arm at all.
    control = tmp_path / "control"
    _populate(
        tmp_path,
        search_flops=_in_band_search_flops(),
        control_root=control,
    )
    with pytest.raises(SystemExit, match="missing the null arm"):
        v0_verdict.compute_verdict(
            tmp_path,
            probe_batch_count=1,
            control_root=control,
            null_root=tmp_path,  # the primary root: pair arms live here
        )
