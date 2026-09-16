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
    # The state the dev-fidelity evidence predicts: adaptive 6/12 against its own churn
    # arm (a tie, no edge), the loss-blind control 6/12 likewise, and the do-nothing
    # clause satisfied so nothing else can catch it. Before section 5.44 this AUTHORIZED
    # Gate 4; the completed clause 3 withholds it.
    "control_ties_adaptive_nobody_demonstrates_an_edge": (
        "PASS_COST_ONLY_NO_DEMONSTRATED_EDGE",
        False,
    ),
    # A distinguishable state: adaptive clears the bar, the loss-blind arm does not. This
    # still authorizes, so the amendment gates the quality premise rather than requiring
    # the bar to be cleared unconditionally.
    "adaptive_clears_bar_loss_blind_does_not": ("PASS", True),
    # The boundary of the proxy, and the amendment's most consequential reading: adaptive
    # is clearly better than the loss-blind arm (9 vs 3) but neither clears the bar, so
    # under "same side of the bar = indistinguishable" this withholds. Flagged as the
    # reading a reader may want to contest - section 5.44 records it as a boundary rather
    # than deriving it from the wording alone.
    "adaptive_just_short_of_bar_loss_blind_far_below": (
        "PASS_COST_ONLY_NO_DEMONSTRATED_EDGE",
        False,
    ),
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
    # adaptive 6/12 vs churn (a tie, no demonstrated edge), the loss-blind control also
    # 6/12, and the do-nothing clause satisfied so nothing else can catch it.
    "control_ties_adaptive_nobody_demonstrates_an_edge": {
        "adaptive_wins": 6,
        "control_wins": 6,
        "null_wins": 12,
    },
    # Distinguishable: adaptive clears the 10-of-12 bar, the loss-blind arm does not.
    "adaptive_clears_bar_loss_blind_does_not": {"adaptive_wins": 12, "control_wins": 3},
    # Adaptive 9 of 12, loss-blind 3: distinguishable by arm counts, same side of the bar.
    "adaptive_just_short_of_bar_loss_blind_far_below": {
        "adaptive_wins": 9,
        "control_wins": 3,
    },
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


# Clause amendments: where a clause's registered text and the shipped rule disagreed, the
# disagreement is recorded, measured, and then resolved as an explicit amendment (report
# section 5.44) rather than left as a code/report drift.
REGISTERED_TEXT_AMENDMENTS = [
    {
        "section": "5.5 change 3",
        "registered_text": (
            "if the pool-matched arm's quality is indistinguishable from the adaptive "
            "arm's, record the cost PASS but withhold the Gate 4 recommendation, since "
            "the quality premise would then be unsupported"
        ),
        "shipped_rule_before": (
            "only one direction of the clause was implemented: it fired when the "
            "loss-blind arm beat its own churn arm on >= 10 of 12 pairs "
            "(POOL_CONTROL_WIN_LIMIT). No adaptive-vs-control comparison existed "
            "anywhere in v0_verdict.py."
        ),
        "demonstrated_before": {
            "variant": "control_ties_adaptive_nobody_demonstrates_an_edge",
            "observed_then": "adaptive 6/12, control 6/12 -> PASS, exit 0, Gate 4 AUTHORIZED",
            "consequence": (
                "the state the registered wording says must withhold was the state the "
                "shipped rule authorized, and it is the state the dev-fidelity evidence "
                "predicts for the real path. The harm guard cannot catch it (it only "
                "rejects adaptive losing >= 10 of 12) and neither can the section 5.14 "
                "clause (churn against do-nothing, not adaptive against the loss-blind "
                "arm)."
            ),
        },
        "amendment": (
            "the clause is completed in the missing direction, using the same "
            "pre-committed bar rather than a new threshold: it fires when NEITHER arm "
            "clears POOL_CONTROL_WIN_LIMIT against its own churn arm, since the two are "
            "then indistinguishable and no edge was shown to recommend Gate 4 on "
            "(label PASS_COST_ONLY_NO_DEMONSTRATED_EDGE, exit 2)."
        ),
        "direction": (
            "strictly tightening: it can only convert an authorization into a "
            "withholding, never the reverse, so no spent block is reinterpreted. "
            "Verified on the two committed real-path verdicts after the change (see "
            "section 5.44)."
        ),
        "demonstrated_after": [
            {
                "variant": "pool_control_passes",
                "label": "PASS_COST_ONLY_POOL_MATCHED_CONTROL_PASSES",
                "reads": "both arms clear the bar: a pool/location effect alone would have passed the gate",
            },
            {
                "variant": "control_ties_adaptive_nobody_demonstrates_an_edge",
                "label": "PASS_COST_ONLY_NO_DEMONSTRATED_EDGE",
                "reads": "neither arm clears the bar: indistinguishable, no quality premise",
            },
        ],
        "still_not_a_test": (
            "'indistinguishable' is operationalised as both arms sitting on the same "
            "side of the bar, not as a statistical test of equality; recorded as a proxy."
        ),
        "boundary_reading": (
            "Because 'has an edge' is binary at POOL_CONTROL_WIN_LIMIT, the clause also "
            "withholds when the adaptive arm is clearly better than the loss-blind arm "
            "(9 of 12 against 3 of 12) but neither clears the bar - variant "
            "adaptive_just_short_of_bar_loss_blind_far_below. Consequence of the "
            "amendment: authorizing Gate 4 now requires the adaptive arm to clear the "
            "bar AND the loss-blind arm not to, so the primary run's quality has to be "
            "demonstrated at the pre-committed bar rather than merely not-failed (the "
            "harm guard only rejects adaptive losing 10 of 12). This is the reading to "
            "contest if a sub-bar but real edge should authorize; recorded here rather "
            "than resolved silently."
        ),
    }
]


def _row_by_name(rows: list[dict], name: str) -> dict:
    """Look a variant's row up by name, loudly - a typo must not pass as a check."""
    for row in rows:
        if row["variant"] == name:
            return row
    raise KeyError(f"no rehearsed variant named {name!r}")


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
        # Exactly the variants that should authorize under the amended rule: the all-clear
        # case and the two distinguishable states. The tie case that used to authorize is
        # now withheld (section 5.44, pinned separately below).
        "the_authorizing_variants_are_the_clear_and_distinguishable_cases": [
            row["variant"] for row in rows if row["observed_gate_4_authorized"]
        ]
        == [
            "all_clear",
            "adaptive_clears_bar_loss_blind_does_not",
        ],
        # The amendment is only closed if both directions of clause 3 are observed: the
        # pool-driven one and the neither-arm-has-an-edge one. The expectations live in
        # REGISTERED_TEXT_AMENDMENTS, so the record and the check cannot drift apart.
        "the_clause_3_amendment_withholds_in_both_directions": all(
            _row_by_name(rows, item["variant"])["observed_label"] == item["label"]
            and not _row_by_name(rows, item["variant"])["observed_gate_4_authorized"]
            for amendment in REGISTERED_TEXT_AMENDMENTS
            for item in amendment["demonstrated_after"]
        ),
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
        "registered_text_amendments": REGISTERED_TEXT_AMENDMENTS,
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
