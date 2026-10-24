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
from act.training.v0_runner import run_pair_session  # noqa: E402
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
    config: dict | None = None,
) -> Path:
    """Materialize one published seed summary in the handoff layout.

    ``config`` is written verbatim; the default empty dict is the shape the committed
    real-path artifacts have (they predate the fields the section 5.46 conformance
    reading looks for), which is why it must stay the default here.
    """

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
        "config": {} if config is None else config,
    }
    (reports / "seed-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return destination


def test_run_scale_reading_reduces_to_the_registered_band_at_declared_scale() -> None:
    """At declared scale the run-scale reading IS the registered comparison.

    The point of the reading is that it generalises the band rather than adding a
    second criterion, so at declared scale its ratio has to equal the registered
    measured/predicted ratio and its verdict has to equal ``in_tolerance_band``.
    """
    from act.training.session_plan import (
        DECLARED_ACTIVE_PARAMETERS,
        DECLARED_TOTAL_PARAMETERS,
        SEED_TOKENS_PER_ARM,
    )

    pairs = [
        {"seed": seed, "adaptive_active_parameters": DECLARED_ACTIVE_PARAMETERS}
        for seed in DECLARED
    ]
    declared_total = 2 * DECLARED_TOTAL_PARAMETERS * SEED_TOKENS_PER_ARM * len(pairs)
    predicted = expected_probe_cost_percent(1)
    for factor in (0.5, 1.0, 2.0, 3.0):
        measured = factor * predicted
        reading = v0_verdict._run_scale_reading(measured, predicted, pairs, declared_total)
        assert reading["scale_factor_measured_to_declared"] == pytest.approx(1.0, rel=1e-12)
        assert reading["run_is_at_declared_scale"] is True
        assert reading["predicted_cost_percent_at_run_scale"] == pytest.approx(
            predicted, rel=1e-12
        )
        assert reading["measured_over_predicted_at_run_scale"] == pytest.approx(
            factor, rel=1e-12
        )
        # The registered band, expressed through the reading's own inputs.
        assert reading["in_tolerance_band_at_run_scale"] == (
            0.5 <= factor <= 2.0
        ) == (tolerance_band(1)[0] <= measured <= tolerance_band(1)[1])
        assert "declared scale" in reading["reading"]


def test_run_scale_reading_is_descriptive_and_moves_no_registered_field(
    tmp_path: Path,
) -> None:
    """A reduced-scale block reports the identity at its own scale, not a false miss.

    The registered reading is pinned to the value the block produced before this
    reading existed, so the addition cannot have moved a criterion.
    """
    dataset_root = tmp_path / "datasets"
    # A block at 1/1000 of the declared training budget, measuring the declared
    # prediction's worth of search FLOPs - by construction a band miss.
    training = TRAINING_FLOPS_PER_SEED
    for seed in DECLARED:
        _write_seed(
            dataset_root,
            seed,
            adaptive_loss=1.0,
            random_loss=1.1,
            search_flops=_in_band_search_flops(1),
            training_flops=training,
        )
    verdict = v0_verdict.compute_verdict(dataset_root, probe_batch_count=1)
    # Registered fields, unchanged by the new reading.
    assert verdict["in_tolerance_band"] is True
    assert verdict["predicted_cost_percent"] == expected_probe_cost_percent(1)
    assert verdict["tolerance_band_percent"] == list(tolerance_band(1))
    reading = verdict["run_scale"]
    assert reading["descriptive_only"] is True
    assert reading["run_is_at_declared_scale"] is False
    assert reading["measured_over_predicted_at_run_scale"] != pytest.approx(1.0, rel=0.5)
    assert reading["in_tolerance_band_at_run_scale"] is False
    assert not reading["reading"].startswith("declared scale")
    # Internally consistent: the ratio is the two percentages it names, and the
    # in-band flag is that ratio against the registered [0.5, 2] band.
    assert reading["measured_over_predicted_at_run_scale"] == pytest.approx(
        reading["measured_cost_percent"]
        / reading["predicted_cost_percent_at_run_scale"]
    )
    assert verdict["band_reading"] == reading["reading"]


def test_the_committed_real_path_block_confirms_the_identity_at_its_own_scale() -> None:
    """The only real-path execution of the protocol, read at the scale it ran at.

    Section 5.3 ran the section 5.1 machinery over seeds 9000-9011 at reduced
    fidelity. Its committed verdict reports ``in_tolerance_band`` false because the
    registered band is derived at the declared model size and token budget, which that
    run is ~10.7x away from - a mismatch that no accounting error could ever fix. Read
    at the scale it executed at, the same numbers are a confirmation of the identity.
    """
    dataset_root = ROOT / "artifacts" / "real-proof" / "datasets"
    if not dataset_root.exists():  # pragma: no cover - artifacts are committed
        pytest.skip("committed section 5.3 dataset is absent")
    verdict = v0_verdict.compute_verdict(
        dataset_root, probe_batch_count=1, declared_seeds=tuple(range(9000, 9012))
    )
    # The registered reading stands exactly as the committed artifact records it.
    assert verdict["in_tolerance_band"] is False
    assert verdict["verdict"] == "PASS_COST_ONLY_HARM_GUARD_REJECTS"
    reading = verdict["run_scale"]
    assert reading["run_is_at_declared_scale"] is False
    assert reading["scale_factor_measured_to_declared"] == pytest.approx(10.6983, rel=2e-4)
    assert reading["measured_over_predicted_at_run_scale"] == pytest.approx(
        1.000286, rel=2e-4
    )
    assert reading["in_tolerance_band_at_run_scale"] is True


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


def test_the_verdict_reads_a_summary_the_runner_actually_wrote(tmp_path: Path) -> None:
    """The reader/writer coupling, which every other test here bypasses.

    ``_write_seed`` hand-builds its summaries, so nothing checked that the
    verdict script can consume what ``run_pair_session`` publishes - and that
    coupling *is* the section 4 path: the runner writes it on Kaggle, the
    verdict reads it back weeks later. A mismatch would only surface once the
    quota was already spent.
    """
    config = V0SessionConfig(
        seed=15000,
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
        total_steps=4,
        mutation_interval=2,
        mutation_budget=1,
        probe_candidate_cap=4,
        probe_batch_count=1,
        validation_batches=2,
        checkpoint_interval=2,
        max_checkpoints=2,
    )
    dataset_root = tmp_path / "datasets"
    run_pair_session(
        config=config,
        session_dir=tmp_path / "sessions",
        dataset_root=dataset_root,
        code_revision="rev",
        remaining_hours=1.0,
        measured_tokens_per_second=1e9,  # instantaneous tiny pair
        seeds=(15000,),
    )
    assert load_seed_summary(dataset_root, 15000)["status"] == "complete"

    verdict = v0_verdict.compute_verdict(
        dataset_root, probe_batch_count=1, declared_seeds=[15000]
    )

    assert verdict["seeds_complete"] == [15000]
    assert verdict["seeds_missing"] == []
    assert verdict["queue_complete"] is True
    assert isinstance(verdict["exit_code"], int)


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
    # The artifact carries the registered code, not just the string: the
    # Kaggle-side runner exits 0 whatever the decision, so a reader (the queue
    # driver) can only report PASS/FAIL by reading it back from here.
    assert verdict["exit_code"] == exit_code == 0
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
    [
        # Clears the pre-committed bar, so the quality premise is supported and the
        # loss-blind arm (0 wins here) does not clear it either: authorized.
        (12, "PASS"),
        # Exactly at the bar: authorized, because the clause is about the two arms being
        # on the same side of it, not about clearing it by a margin.
        (10, "PASS"),
        # Below the bar and the loss-blind arm below it too: same side, indistinguishable.
        (9, "PASS_COST_ONLY_NO_DEMONSTRATED_EDGE"),
        # The registered guard is still clear at 3 wins (9 losses < 10) - but the section
        # 5.44 quality clause withholds, where before the amendment this authored Gate 4.
        (3, "PASS_COST_ONLY_NO_DEMONSTRATED_EDGE"),
        (2, "PASS_COST_ONLY_HARM_GUARD_REJECTS"),
    ],
)
def test_harm_guard_rejects_at_ten_losses(
    adaptive_wins: int, expected_verdict: str, tmp_path: Path
) -> None:
    # The registered guard: adaptive must not lose in >= 10 of 12 pairs
    # (exact one-tailed sign test p = 0.0193), i.e. fewer than 3 wins rejects. The
    # guard's own boundary is unchanged by section 5.44; what changed is that clearing
    # the guard is no longer sufficient on its own, because clause 3 now also withholds
    # when neither arm clears the pre-committed bar.
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
    assert verdict["harm_guard"]["clear"] is (adaptive_wins >= 3)
    # And the clause that did the withholding in the same-side-of-the-bar rows is the
    # quality one, recorded rather than inferred from the label.
    premise = verdict["quality_premise"]
    assert premise["adaptive_wins"] == adaptive_wins
    assert premise["adaptive_clears_bar"] is (
        adaptive_wins >= v0_verdict.POOL_CONTROL_WIN_LIMIT
    )
    assert premise["control_clears_bar"] is False
    assert premise["indistinguishable"] is (not premise["adaptive_clears_bar"])
    assert verdict["gate_4_authorized"] is (expected_verdict == "PASS")
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
        # The primary arm clears the pre-committed quality bar (10 of 12) while the
        # loss-blind arm wins 3, so this block is authorized on the amended rule too:
        # what is being tested is that the override judges the fresh block by the same
        # rules, not a weakened version of them.
        won = index < 10
        lost = index < 3
        _write_seed(
            tmp_path,
            seed,
            adaptive_loss=1.0 if won else 1.5,
            random_loss=1.5 if won else 1.0,
            search_flops=_in_band_search_flops(),
        )
        _write_seed(
            control,
            seed,
            adaptive_loss=1.0 if lost else 1.5,
            random_loss=1.5 if lost else 1.0,
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
    assert re_declared["harm_guard"]["adaptive_losses"] == 2
    assert re_declared["quality_premise"] == {
        "adaptive_wins": 10,
        "adaptive_clears_bar": True,
        "control_clears_bar": False,
        "win_limit": v0_verdict.POOL_CONTROL_WIN_LIMIT,
        "indistinguishable": False,
        "reading": "adaptive clears the bar and the loss-blind arm does not",
    }
    assert re_declared["verdict"] == "PASS"
    assert re_declared["gate_4_authorized"] is True


# --- section 5.46: the recorded configuration against the registered one ------ #


def _declared_fixture_with_config(tmp_path: Path, config: dict):
    """An authorizing declared block whose primary seeds all record ``config``.

    The primary arm wins all 12 pairs (clearing the section 5.44 quality bar), the
    loss-blind control wins none, and adaptive beats the null arm everywhere - so every
    withholding clause is clear and the block authorizes unless something else moves it.
    """
    control = tmp_path / "control"
    null = tmp_path / "null"
    for seed in DECLARED:
        _write_seed(
            tmp_path,
            seed,
            adaptive_loss=1.0,
            random_loss=1.5,
            search_flops=_in_band_search_flops(),
            config=config,
        )
        _write_seed(
            control,
            seed,
            adaptive_loss=1.5,
            random_loss=1.0,
            search_flops=_in_band_search_flops(),
            config=config,
        )
        _write_null_seed(null, seed, null_loss=2.0)
    return control, null


def test_config_conformance_reads_a_registered_run_as_conforming(tmp_path: Path) -> None:
    """The conforming case, field by field, at declared scale."""
    registered = dict(v0_verdict.REGISTERED_RUN_CONFIG, probe_batch_count=1)
    control, null = _declared_fixture_with_config(tmp_path, registered)

    verdict = v0_verdict.compute_verdict(
        tmp_path, probe_batch_count=1, control_root=control, null_root=null
    )

    conformance = verdict["config_conformance"]
    assert conformance["conforms"] is True
    assert conformance["fields_differing"] == []
    assert conformance["fields_not_recorded"] == []
    assert conformance["seeds_with_a_recorded_config"] == len(DECLARED)
    assert conformance["descriptive_only"] is True
    assert all(row["status"] == "matches" for row in conformance["fields"].values())
    assert verdict["verdict"] == "PASS"
    assert verdict["gate_4_authorized"] is True


def test_config_conformance_flags_the_untreated_rebuild_without_changing_a_rule(
    tmp_path: Path,
) -> None:
    """Section 5.46's scenario: a run whose recorded config is not the registered one.

    The two must differ in the conformance block and *nowhere else* - same label, same
    exit code, same authorization - because a block that changed a decision would be a
    new criterion rather than a reading, and this one is registered as descriptive.
    """
    registered = dict(v0_verdict.REGISTERED_RUN_CONFIG, probe_batch_count=1)
    untreated = dict(
        registered,
        preserve_router_on_rebuild=False,
        new_expert_initialization="random",
    )
    conforming_root = tmp_path / "conforming"
    untreated_root = tmp_path / "untreated"
    conforming_root.mkdir()
    untreated_root.mkdir()
    c1, n1 = _declared_fixture_with_config(conforming_root, registered)
    c2, n2 = _declared_fixture_with_config(untreated_root, untreated)

    good = v0_verdict.compute_verdict(
        conforming_root, probe_batch_count=1, control_root=c1, null_root=n1
    )
    bad = v0_verdict.compute_verdict(
        untreated_root, probe_batch_count=1, control_root=c2, null_root=n2
    )

    assert good["config_conformance"]["conforms"] is True
    assert bad["config_conformance"]["conforms"] is False
    assert bad["config_conformance"]["fields_differing"] == [
        "new_expert_initialization",
        "preserve_router_on_rebuild",
    ]
    assert "is not the registered configuration" in bad["config_conformance"]["reading"]

    for key in ("verdict", "exit_code", "gate_4_authorized", "primary_pass", "quality_premise"):
        assert bad[key] == good[key], f"the conformance block moved {key}"


def test_config_conformance_reads_a_pre_field_artifact_as_unrecorded(
    tmp_path: Path,
) -> None:
    """No spent block may be re-read as non-conforming for not having the field."""
    registered = dict(v0_verdict.REGISTERED_RUN_CONFIG, probe_batch_count=1)
    old_root = tmp_path / "legacy"
    old_root.mkdir()
    control, null = _declared_fixture_with_config(old_root, {})

    verdict = v0_verdict.compute_verdict(
        old_root, probe_batch_count=1, control_root=control, null_root=null
    )

    conformance = verdict["config_conformance"]
    assert conformance["fields_differing"] == []
    assert conformance["seeds_with_a_recorded_config"] == 0
    assert len(conformance["fields_not_recorded"]) == len(registered)
    assert conformance["conforms"] is False
    assert "predates" in conformance["reading"]
    # Descriptive, so the legacy fixture still authorizes exactly as it always did.
    assert verdict["verdict"] == "PASS"
    assert verdict["gate_4_authorized"] is True


def test_the_registered_run_config_carries_the_adopted_protocol() -> None:
    """The registered mapping is the protocol, so it is pinned to it (section 5.47)."""
    registered = v0_verdict.REGISTERED_RUN_CONFIG

    # Section 5.5 changes 1 and 2: spread candidates, argmin selection.
    assert registered["probe_candidate_policy"] == "spread"
    assert registered["probe_selection"] == "argmin"
    assert registered["arms"] == ["random", "adaptive"]
    # Section 5.47: the rebuild treatment the declared run is registered on.
    assert registered["preserve_router_on_rebuild"] is True
    assert registered["new_expert_initialization"] == "clone"
    # The declared shape and budget the cost prediction is derived from.
    assert registered["total_steps"] == 6000
    assert registered["mutation_budget"] == 3
    assert registered["mutation_interval"] == 1500
    assert registered["depth"] == 12
    assert registered["hidden_size"] == 1024
    assert registered["batch_size"] == 2
    assert registered["context_size"] == 2048
    assert "dtype" not in registered  # the section 5.2 rule decides it, not a constant


def test_the_registered_run_config_matches_what_the_launch_command_builds() -> None:
    """The mapping and the declared session config must not drift apart.

    A conformance block compared against a stale constant would be worse than none: it
    would certify a run nothing builds. The registered launch command passes
    ``--probe-candidate-policy spread --preserve-router-on-rebuild --new-expert clone``
    (section 5.47), so the declared session config must be able to produce exactly that.
    """
    from act.training.v0_session import declared_session_config

    config = declared_session_config(
        probe_candidate_policy="spread",
        preserve_router_on_rebuild=True,
        new_expert_initialization="clone",
    )

    for field, expected in v0_verdict.REGISTERED_RUN_CONFIG.items():
        recorded = getattr(config, field, "<absent>")
        assert recorded == expected or list(recorded) == expected, (
            f"the launch command cannot produce registered {field}={expected!r} "
            f"(it builds {recorded!r})"
        )
    assert config.probe_batch_count == 1  # the invocation's registered value


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


def test_a_short_control_withholds_gate_4_rather_than_leaving_the_clause_unfireable(
    tmp_path: Path,
) -> None:
    """The clause needs 10 pairs, so an incomplete control cannot fire it at all.

    A control run on 4 seeds - a deadline-stopped session's queue, say - leaves
    ``wins >= 10`` unreachable while the loss-blind arm is winning every seed it did
    run. Before this guard existed that state authorized Gate 4 outright: verdict
    ``PASS``, ``gate_4_authorized`` True, exit 0. The do-nothing clause always refused
    to evaluate an incomplete queue; this makes the pool-matched clause agree with it.
    """
    primary = tmp_path / "datasets"
    control = tmp_path / "control"
    null_root = tmp_path / "null"
    in_band = _in_band_search_flops(1)
    _populate(primary, search_flops=in_band, null_root=null_root)
    for seed in DECLARED[:4]:
        # Every pair the control did run shows the pool effect winning.
        _write_seed(
            control, seed, adaptive_loss=1.0, random_loss=1.5, search_flops=in_band
        )

    verdict = v0_verdict.compute_verdict(
        primary, probe_batch_count=1, control_root=control, null_root=null_root
    )
    block = verdict["pool_matched_control"]
    assert block["evaluated"] is False
    assert block["pairs"] == 4
    assert block["seeds_missing"] == list(DECLARED[4:])
    assert "cannot be evaluated" in block["reason"]
    assert str(v0_verdict.POOL_CONTROL_WIN_LIMIT) in block["reason"]
    assert verdict["verdict"] == "PASS_COST_ONLY_CONTROL_MISSING"
    assert verdict["gate_4_authorized"] is False
    assert verdict["exit_code"] == 2


def test_a_control_one_seed_short_is_withheld_even_when_it_would_not_have_fired(
    tmp_path: Path,
) -> None:
    """Eleven innocent pairs still cannot evaluate a clause that needs the twelfth.

    11 of 12 is the case an honest operator is most likely to hit, and the clause
    reads the same either way when it *could* fire. Withholding is what the missing
    control, the incomplete null queue and the incomplete primary queue all do.
    """
    primary = tmp_path / "datasets"
    control = tmp_path / "control"
    null_root = tmp_path / "null"
    in_band = _in_band_search_flops(1)
    _populate(primary, search_flops=in_band, null_root=null_root)
    for seed in DECLARED[:-1]:
        # The loss-blind arm loses every pair it ran, so nothing would have fired.
        _write_seed(
            control, seed, adaptive_loss=1.5, random_loss=1.0, search_flops=in_band
        )

    verdict = v0_verdict.compute_verdict(
        primary, probe_batch_count=1, control_root=control, null_root=null_root
    )
    block = verdict["pool_matched_control"]
    assert block["evaluated"] is False
    assert block["pairs"] == len(DECLARED) - 1
    assert block["seeds_missing"] == [DECLARED[-1]]
    assert verdict["verdict"] == "PASS_COST_ONLY_CONTROL_MISSING"
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


def test_the_process_exit_code_is_the_artifact_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented command's exit code *is* the decision, so it cannot drift.

    Measured 2026-10-24 on the live datasets: the verdict produced
    ``PASS_COST_ONLY_NO_DEMONSTRATED_EDGE`` with ``exit_code: 2`` in the artifact, and
    ``main`` returned 0, because section 5.44 added the label to the artifact but not to
    a restated label->code map. The Kaggle path reads the artifact's ``exit_code`` and was
    unaffected; anything scripting the runbook's own command would have read a withheld
    verdict as a PASS. The fix is one source of truth, and this pins it.
    """
    control = tmp_path / "control"
    null = tmp_path / "null"
    _populate(
        tmp_path,
        search_flops=_in_band_search_flops(),
        adaptive_wins=9,  # below the bar, and so is the loss-blind arm
        control_wins=3,
        control_root=control,
        null_root=null,
    )

    exit_code = _run_main(monkeypatch, tmp_path, control_root=control, null_root=null)

    verdict = json.loads((tmp_path / "v0-5-1-verdict.json").read_text())
    assert verdict["verdict"] == "PASS_COST_ONLY_NO_DEMONSTRATED_EDGE"
    assert verdict["gate_4_authorized"] is False
    assert verdict["exit_code"] == 2
    assert exit_code == verdict["exit_code"]
