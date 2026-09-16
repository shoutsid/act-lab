"""Section 5.43: rehearse the declared Gate-4 verdict path, with no GPU.

The compute-efficiency protocol's verdict (`scripts/v0_verdict.py`) decides whether Gate 4
is authorized, and it is applied exactly once, at the end of a T4/Kaggle spend that takes
hours. Every clause in it had been exercised by a unit test at *some* scale, but never all
together, at the declared scale, in the configuration the real run will be in - so the
first time the whole path runs end to end would have been the run that matters.

This builds synthetic **declared-scale** fixtures (285,336,576 total / 134,341,632 active
parameters, 24,576,000 tokens per arm, three complete queues) and applies the real
`compute_verdict` to each variant: every withholding clause in isolation, both absent-queue
cases, both short-queue cases, and the combinations whose label precedence is documented in
`compute_verdict`'s own comment. The output is a truth table, written to
`artifacts/study-summaries/gate4-verdict-rehearsal.json` and pinned by
`tests/test_gate4_rehearsal.py`.

What this is not: training. No session ran, every loss is synthetic, and the record says so.
The rehearsal validates the **decision machinery**, not the hypothesis. Its use is as a
pre-flight for the declared run - `--check` exits non-zero if any variant's label, exit
code or authorization ever stops matching the registered rule, so the verdict path can be
shown to be intact before the GPU time is spent rather than after.

    PYTHONPATH=src python scripts/verdict_rehearsal.py            # write the record
    PYTHONPATH=src python scripts/verdict_rehearsal.py --check    # verify, non-zero on drift
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import v0_verdict  # noqa: E402

from act.training.session_plan import (  # noqa: E402
    DECLARED_ACTIVE_PARAMETERS,
    DECLARED_TOTAL_PARAMETERS,
    SEED_TOKENS_PER_ARM,
    expected_probe_cost_percent,
)
from act.training.v0_handoff import seed_directory  # noqa: E402

REPORT_SECTION = "5.43"
SUMMARIES = ROOT / "artifacts" / "study-summaries"
RECORD = SUMMARIES / "gate4-verdict-rehearsal.json"

DECLARED = list(v0_verdict.DECLARED_SEEDS)
# The declared run's own budget: 2 x P x D forward-equivalent FLOPs per arm.
DECLARED_TRAINING_FLOPS = 2 * DECLARED_TOTAL_PARAMETERS * SEED_TOKENS_PER_ARM
DECLARED_PREDICTION_PERCENT = expected_probe_cost_percent(1)

# The registered rule, restated here as the rehearsal's expectations. Every entry is the
# label `compute_verdict`'s own decision chain must produce, plus whether it may authorize
# Gate 4. `gate_4_authorized` is `PASS` **and** a complete primary queue.
TRUTH_TABLE = {
    "all_clear": ("PASS", True),
    "cost_over_criterion": ("FAIL", False),
    "harm_guard_rejects": ("PASS_COST_ONLY_HARM_GUARD_REJECTS", False),
    "pool_control_passes": ("PASS_COST_ONLY_POOL_MATCHED_CONTROL_PASSES", False),
    "null_clause_rejects": ("PASS_COST_ONLY_NULL_REJECTS", False),
    "band_miss_at_declared_scale": ("PASS_WITH_QUALIFICATION", False),
    "control_absent": ("PASS_COST_ONLY_CONTROL_MISSING", False),
    "control_short": ("PASS_COST_ONLY_CONTROL_MISSING", False),
    "null_queue_absent": ("PASS_COST_ONLY_NULL_MISSING", False),
    "null_queue_short": ("PASS_COST_ONLY_NULL_MISSING", False),
    # Complete queues, clean clauses, one seed missing from the primary: the label is an
    # ordinary PASS but authorization is withheld by the completeness requirement.
    "primary_queue_short": ("PASS", False),
    # Label precedence, documented in compute_verdict's comment and previously untested.
    "cost_fail_beats_every_other_clause": ("FAIL", False),
    "harm_guard_beats_pool_control_and_null": (
        "PASS_COST_ONLY_HARM_GUARD_REJECTS",
        False,
    ),
    "pool_control_beats_null_clause": (
        "PASS_COST_ONLY_POOL_MATCHED_CONTROL_PASSES",
        False,
    ),
}

# Clause-triggering knobs per variant. `cost_multiple_of_prediction` is a multiple of the
# **declared prediction**, which is itself a percentage (0.0942%) - so crossing the 5%
# criterion takes a multiple above 5 / 0.0942 = 53.1, and a multiple of 6 lands at 0.565%,
# comfortably inside it. This rehearsal found that the hard way: a first pass used 6.0 for
# both cost variants, the machinery correctly returned PASS_WITH_QUALIFICATION and
# HARM_GUARD_REJECTS instead of FAIL, and the check below reported the drift. `adaptive_wins`
# caps how many of the 12 pairs the adaptive arm wins; `control_wins` and `null_wins` are
# wins for the loss-blind arm and for the adaptive arm (`None` = all, the
# authorization-friendly default).
VARIANTS: dict[str, dict] = {
    "all_clear": {},
    "cost_over_criterion": {"cost_multiple_of_prediction": 60.0},
    "harm_guard_rejects": {"adaptive_wins": 2},
    "pool_control_passes": {"control_wins": 12},
    "null_clause_rejects": {"null_wins": 2},
    # 3x the prediction is 0.28% - inside the 5% criterion, outside the [0.5, 2] band.
    "band_miss_at_declared_scale": {"cost_multiple_of_prediction": 3.0},
    "control_absent": {"with_control": False},
    "control_short": {"with_control": True, "control_seeds": 4, "control_wins": 12},
    "null_queue_absent": {"with_null": False},
    "null_queue_short": {"with_null": True, "null_seeds": 8},
    "primary_queue_short": {"primary_seeds": 11},
    "cost_fail_beats_every_other_clause": {
        "cost_multiple_of_prediction": 60.0,
        "adaptive_wins": 0,
        "control_wins": 12,
        "null_wins": 0,
    },
    "harm_guard_beats_pool_control_and_null": {
        "adaptive_wins": 2,
        "control_wins": 12,
        "null_wins": 0,
    },
    "pool_control_beats_null_clause": {"control_wins": 12, "null_wins": 0},
}


def _write_seed(
    dataset_root: Path,
    seed: int,
    *,
    adaptive_loss: float,
    random_loss: float,
    search_flops: int,
) -> None:
    """One synthetic published seed summary in the handoff layout.

    Declared scale on both sides of the identity: the parameter counts are the declared
    model's and the training FLOPs are the declared per-arm budget, so the band check is
    evaluated exactly as the real run will evaluate it.
    """

    def arm(mode: str, loss: float) -> dict[str, object]:
        return {
            "mode": mode,
            "final_loss": loss,
            "validation_loss": loss,
            "quality": -loss,
            "mutation_count": 3,
            "parameter_count": DECLARED_TOTAL_PARAMETERS,
            "active_parameter_count": DECLARED_ACTIVE_PARAMETERS,
            "training_flops": DECLARED_TRAINING_FLOPS,
            "search_flops": search_flops if mode == "adaptive" else 0,
            "architecture_hash": "0" * 64,
            "mutation_history": [],
            "wall_time_seconds": 1.0,
            "tokens_per_second": 1.0,
        }

    reports = seed_directory(dataset_root, seed) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "seed-summary.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "status": "complete",
                "protocol": "v0-compute-efficiency-5.1",
                "interrupted_step": None,
                "predicted_cost_ratio": DECLARED_PREDICTION_PERCENT / 100.0,
                "arms": {
                    "random": arm("random", random_loss),
                    "adaptive": arm("adaptive", adaptive_loss),
                },
                "config": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _write_null_seed(dataset_root: Path, seed: int, *, null_loss: float) -> None:
    reports = seed_directory(dataset_root, seed) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "seed-summary.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "status": "complete",
                "protocol": "v0-compute-efficiency-5.1",
                "interrupted_step": None,
                "arms": {
                    "null": {
                        "mode": "null",
                        "final_loss": null_loss,
                        "validation_loss": null_loss,
                        "quality": -null_loss,
                        "mutation_count": 0,
                        "parameter_count": DECLARED_TOTAL_PARAMETERS,
                        "active_parameter_count": DECLARED_ACTIVE_PARAMETERS,
                        "training_flops": DECLARED_TRAINING_FLOPS,
                        "search_flops": 0,
                        "architecture_hash": "0" * 64,
                        "mutation_history": [],
                        "wall_time_seconds": 1.0,
                        "tokens_per_second": 1.0,
                    }
                },
                "config": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def build(root: Path, variant: dict) -> tuple[Path, Path | None, Path | None]:
    """Materialize one variant's three queues and return their roots."""
    search_flops = round(
        variant.get("cost_multiple_of_prediction", 1.0)
        * DECLARED_PREDICTION_PERCENT
        / 100.0
        * DECLARED_TRAINING_FLOPS
    )
    primary = root / "datasets"
    for index, seed in enumerate(DECLARED[: variant.get("primary_seeds", len(DECLARED))]):
        # A complete, authorization-friendly primary queue unless the variant says not.
        wins = variant.get("adaptive_wins")
        adaptive_won = wins is None or index < wins
        _write_seed(
            primary,
            seed,
            adaptive_loss=1.0 if adaptive_won else 1.5,
            random_loss=1.5 if adaptive_won else 1.0,
            search_flops=search_flops,
        )

    control: Path | None = None
    if variant.get("with_control", True):
        control = root / "control"
        for index, seed in enumerate(DECLARED[: variant.get("control_seeds", len(DECLARED))]):
            # In the control root the arm recorded as `adaptive` IS the loss-blind arm.
            blind_won = index < variant.get("control_wins", 0)
            _write_seed(
                control,
                seed,
                adaptive_loss=1.0 if blind_won else 1.5,
                random_loss=1.5 if blind_won else 1.0,
                search_flops=search_flops,
            )

    null_root: Path | None = None
    if variant.get("with_null", True):
        null_root = root / "null"
        for index, seed in enumerate(DECLARED[: variant.get("null_seeds", len(DECLARED))]):
            # The primary adaptive loss is 1.0 throughout, so 2.0 is beaten and 0.5 beats.
            beats = variant.get("null_wins")
            adaptive_beats = beats is None or index < beats
            _write_null_seed(null_root, seed, null_loss=2.0 if adaptive_beats else 0.5)
    return primary, control, null_root


def rehearse() -> dict:
    rows = []
    for name, variant in VARIANTS.items():
        root = Path(tempfile.mkdtemp())
        try:
            primary, control, null_root = build(root, variant)
            verdict = v0_verdict.compute_verdict(
                primary,
                probe_batch_count=1,
                control_root=control,
                null_root=null_root,
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)
        expected_label, expected_authorized = TRUTH_TABLE[name]
        reading = verdict["run_scale"]
        rows.append(
            {
                "variant": name,
                "variant_knobs": variant,
                "expected_label": expected_label,
                "expected_gate_4_authorized": expected_authorized,
                "observed_label": verdict["verdict"],
                "observed_exit_code": verdict["exit_code"],
                "observed_gate_4_authorized": verdict["gate_4_authorized"],
                "matches_registered_rule": (
                    verdict["verdict"] == expected_label
                    and verdict["gate_4_authorized"] is expected_authorized
                ),
                "run_is_at_declared_scale": reading["run_is_at_declared_scale"],
                "in_tolerance_band": verdict["in_tolerance_band"],
                "in_tolerance_band_at_run_scale": reading[
                    "in_tolerance_band_at_run_scale"
                ],
                "cost_percent": verdict["measured_cost_percent"],
                "harm_guard_clear": verdict["harm_guard"]["clear"],
                "pool_control_evaluated": verdict["pool_matched_control"]["evaluated"],
                "pool_control_passes_gate": (
                    verdict["pool_matched_control"].get("control_passes_gate")
                ),
                "null_evaluated": verdict["null_control"]["evaluated"],
                "null_clause_rejects": (
                    verdict["null_control"].get("clause_rejects")
                ),
            }
        )

    checks = {
        "every_variant_matches_the_registered_rule": all(
            row["matches_registered_rule"] for row in rows
        ),
        "exactly_one_variant_authorizes_gate_4": sum(
            1 for row in rows if row["observed_gate_4_authorized"]
        )
        == 1,
        "the_authorizing_variant_is_the_all_clear_case": next(
            row["variant"] for row in rows if row["observed_gate_4_authorized"]
        )
        == "all_clear",
        # Exit 1 is the cost FAIL and nothing else: exactly the variants whose measured
        # cost crosses the 5% criterion, and no others.
        "cost_failures_exit_1_and_nothing_else_does": all(
            (row["observed_exit_code"] == 1)
            == (row["cost_percent"] > v0_verdict.PRIMARY_COST_PERCENT)
            for row in rows
        ),
        # The knob is a multiple of the prediction, so the measured cost has to be that
        # multiple of it - the unit slip this rehearsal was written to survive.
        "each_variant_costs_the_multiple_its_knob_says": all(
            abs(
                row["cost_percent"]
                - row["variant_knobs"].get("cost_multiple_of_prediction", 1.0)
                * DECLARED_PREDICTION_PERCENT
            )
            <= 1e-9 * max(1.0, row["cost_percent"])
            for row in rows
        ),
        "the_cost_over_criterion_variant_clears_the_5_percent_bar": next(
            row["cost_percent"]
            for row in rows
            if row["variant"] == "cost_over_criterion"
        )
        > v0_verdict.PRIMARY_COST_PERCENT,
        "the_band_miss_variant_stays_inside_the_5_percent_bar": next(
            row["cost_percent"]
            for row in rows
            if row["variant"] == "band_miss_at_declared_scale"
        )
        <= v0_verdict.PRIMARY_COST_PERCENT,
        "every_fixture_is_at_declared_scale": all(
            row["run_is_at_declared_scale"] for row in rows
        ),
        # At declared scale the run-scale reading has to agree with the registered band.
        "the_band_readings_agree_at_declared_scale": all(
            row["in_tolerance_band"] == row["in_tolerance_band_at_run_scale"]
            for row in rows
        ),
    }
    return {
        "record_type": "verdict-rehearsal",
        "kind": "gate4-verdict-rehearsal",
        "report_section": REPORT_SECTION,
        "synthetic": True,
        "what_this_is": (
            "A rehearsal of the declared Gate-4 verdict machinery on synthetic "
            "declared-scale fixtures. No session ran and every loss is fabricated: this "
            "validates the decision path, not the hypothesis. Intended as a pre-flight "
            "for the declared run, so the whole verdict can be shown intact before the "
            "GPU time is spent."
        ),
        "declared_parameters": {
            "total": DECLARED_TOTAL_PARAMETERS,
            "active": DECLARED_ACTIVE_PARAMETERS,
            "tokens_per_arm": SEED_TOKENS_PER_ARM,
            "training_flops_per_arm": DECLARED_TRAINING_FLOPS,
            "predicted_cost_percent": DECLARED_PREDICTION_PERCENT,
        },
        "declared_seeds": DECLARED,
        "variants": rows,
        "built_in_checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the written record against the registered rule; exit 1 on drift",
    )
    args = parser.parse_args()
    payload = rehearse()
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {RECORD}")
    for row in payload["variants"]:
        mark = "ok " if row["matches_registered_rule"] else "DRIFT"
        flag = "AUTHORIZES" if row["observed_gate_4_authorized"] else ""
        print(
            f"  {mark} {row['variant']:38s} {row['observed_label']:44s} "
            f"exit {row['observed_exit_code']} {flag}"
        )
    for name, value in payload["built_in_checks"].items():
        print(f"  check {name:46s} {value}")
    if args.check and not all(payload["built_in_checks"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
