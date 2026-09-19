"""Tests for the promotion-margin report on the halving plan.

The findings measure that a screening run ranks the matched set well but does not
separate candidates inside the seed noise (section 4), and that a promotion
decided on a short-run proxy is biased (section 4a). The plan therefore has to
*say* when its cut falls inside that noise, because the candidate it drops is
never trained again - the search cannot revisit the decision.

These tests pin the report, not the training: a margin above the noise reads as a
decision, a margin below it reads as elimination, and the set named as
unresolved is every candidate the cut could have chosen between.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from arch_zoo_screen import (  # noqa: E402
    SCREEN_SEED_STD_BY_STEPS,
    SCREEN_SEED_STD_NATS,
    comparison_noise_nats,
    seed_std_for_steps,
    seed_std_measured_at_steps,
    promoted_names,
    promotion_margins,
    report_promotion,
)


#: A synthetic threshold for the unit tests below, which test the *reporting*
#: rules rather than the measured value: every assertion is written against it, so
#: the derived default can change without silently changing what they assert.
NOISE = 0.03


def ranking(*losses: float) -> list[tuple[str, float, int]]:
    return [(f"c{index}", loss, 2) for index, loss in enumerate(losses)]


def test_a_cut_wider_than_the_noise_is_reported_as_a_decision():
    # Gap at the cut is 0.100 nats, well clear of the synthetic threshold.
    margins = promotion_margins(ranking(2.90, 2.95, 3.05, 3.20), keep=2, noise=NOISE)
    assert margins["available"]
    assert margins["cut_margin"] == pytest.approx(0.10)
    assert margins["resolvable"] is True
    assert margins["kept"] == ["c0", "c1"]
    assert margins["dropped"] == ["c2", "c3"]
    assert [row["name"] for row in margins["cut_pair"]] == ["c1", "c2"]
    assert margins["borderline"] == []


def test_a_cut_inside_the_noise_names_every_candidate_it_could_have_taken():
    """The tie-break's candidates are the kept *and* dropped ones near the cut.

    Reporting only the kept ones would understate it: the question a reader has is
    which candidates the cut could not separate, and that set spans the boundary.
    """
    margins = promotion_margins(ranking(2.90, 2.94, 2.95, 3.30), keep=2, noise=NOISE)
    assert margins["cut_margin"] == pytest.approx(0.01)
    assert margins["resolvable"] is False
    assert [row["name"] for row in margins["cut_pair"]] == ["c1", "c2"]
    # c1 is the other half of the cut pair, so it is named there rather than twice;
    # c2 is excluded because its own gap to the cut is zero by construction.
    assert [row["name"] for row in margins["borderline"]] == ["c1"]
    assert all(row["gap_to_cut"] < NOISE for row in margins["borderline"])


def test_the_borderline_set_reaches_past_the_cut_pair():
    """A third candidate inside the noise belongs in the report too.

    A cut can be unresolvable across more than the two candidates it lands
    between, and the log has to name all of them: promoting 'the borderline set'
    is the remedy, so the set has to be complete.
    """
    margins = promotion_margins(
        ranking(2.90, 2.94, 2.95, 2.952, 3.30), keep=2, noise=NOISE
    )
    assert [row["name"] for row in margins["cut_pair"]] == ["c1", "c2"]
    assert [row["name"] for row in margins["borderline"]] == ["c1", "c3"]


def test_the_margin_at_the_cut_is_what_is_tested_not_the_stage_spread():
    """A wide overall spread can hide an unresolvable cut, and must not.

    This is the failure the plan is guarding against: a design whose spread is
    obvious can still have its *promotion boundary* decided by noise, and a report
    that only printed the spread would read as if the cut were safe.
    """
    margins = promotion_margins(ranking(2.00, 2.01, 4.00, 4.01), keep=2, noise=NOISE)
    assert margins["cut_margin"] == pytest.approx(1.99)  # spread is huge...
    assert margins["resolvable"] is True
    # ...but the pair that matters is the boundary, so a tight cut elsewhere still
    # shows up when the grouping is different.
    tight = promotion_margins(ranking(2.00, 2.01, 2.02, 4.01), keep=2, noise=NOISE)
    assert tight["resolvable"] is False
    assert tight["cut_margin"] == pytest.approx(0.01)


def test_a_cut_at_an_end_decides_nothing():
    assert promotion_margins(ranking(2.9, 3.0), keep=0, noise=NOISE)["available"] is False
    assert promotion_margins(ranking(2.9, 3.0), keep=2, noise=NOISE)["available"] is False
    assert promotion_margins([], keep=1, noise=NOISE)["available"] is False


def test_the_report_prints_the_verdict_and_the_borderline_set(capsys):
    report_promotion("Stage 1", ranking(2.90, 2.94, 2.95, 3.30), 2, steps=400, noise=NOISE)
    text = capsys.readouterr().out
    assert "INSIDE" in text
    assert "does not adjudicate" in text
    # Both halves of the cut pair are named, so the reader can see which two runs
    # the stage failed to separate rather than only that it failed.
    assert "`c1` (2.9400)" in text and "`c2` (2.9500)" in text
    # The consequence is stated, because it is what makes the flag actionable.
    assert "never trained again" in text

    report_promotion("Stage 1", ranking(2.90, 2.95, 3.05, 3.20), 2, steps=400, noise=NOISE)
    resolved = capsys.readouterr().out
    assert "clears" in resolved
    assert "INSIDE" not in resolved


def test_promotion_takes_the_top_k_when_the_cut_is_clean():
    ranked = ranking(2.90, 2.95, 3.05, 3.20)
    promotion = promoted_names(ranked, 2, noise=NOISE, promote_ties=True)
    assert promotion["names"] == ["c0", "c1"]
    assert promotion["promoted_ties"] == []


def test_promote_ties_advances_the_borderline_set_in_ranking_order():
    """An unresolvable cut hands the tie to confirmation instead of deciding it.

    The remedy the report prints has to be an option the driver actually offers,
    otherwise a reader is told to do something the code cannot do.
    """
    ranked = ranking(2.90, 2.94, 2.95, 2.952, 3.30)
    assert promoted_names(ranked, 2, noise=NOISE, promote_ties=False)["names"] == [
        "c0",
        "c1",
    ]
    promotion = promoted_names(ranked, 2, noise=NOISE, promote_ties=True)
    assert promotion["names"] == ["c0", "c1", "c2", "c3"]
    assert promotion["promoted_ties"] == ["c2", "c3"]
    # c0 is the far side of a clean gap, so it is never added as a tie.
    assert "c0" not in promotion["promoted_ties"]


def test_promote_ties_never_drops_a_candidate_it_would_have_kept():
    ranked = ranking(2.90, 2.94, 2.95, 2.952, 3.30)
    plain = set(promoted_names(ranked, 2, noise=NOISE)["names"])
    tied = set(promoted_names(ranked, 2, noise=NOISE, promote_ties=True)["names"])
    assert plain <= tied


def test_promote_ties_is_inert_when_the_cut_decides_nothing():
    # A cut at an end of the ranking is not a decision, so there is no tie to hand on.
    promotion = promoted_names(ranking(2.9, 3.0), 2, noise=NOISE, promote_ties=True)
    assert promotion["names"] == ["c0", "c1"]
    assert promotion["margins"]["available"] is False


def test_the_driver_writes_its_promotion_verdicts_into_the_decision(tmp_path, monkeypatch):
    """End to end through `main`, with the trainer stubbed out.

    The unit test above renders `outcome["promotions"]`; this one checks the wiring
    that fills it - a halving plan's two cuts, each costed at its own stage length and
    seed count - because a report assembled correctly but never passed to the decision
    writer would still lose the verdicts it exists to preserve.
    """
    import arch_zoo_screen as screen
    from act.search.architecture_eval import EvalResult

    def fake_evaluate_batch(candidates, *, run, device, output_dir, tag):
        return [
            EvalResult(
                name=candidate.name,
                family=candidate.family,
                genome_hash=f"hash-{candidate.name}",
                seed=run.seed,
                total_params=candidate.total_params,
                active_params=candidate.active_params,
                contracted_flops_per_token=2.5e8,
                dense_runtime_flops_per_token=4.0e8,
                training_flops=2.5e8 * run.train_tokens,
                validation_flops=2.5e8 * 16 * 2 * 1024,
                final_train_loss=3.1 + 0.02 * index,
                mean_train_loss=3.15 + 0.02 * index,
                # A clear spread, so stage 1's cut is a decision and stage 2's is not.
                validation_loss=3.0 + (0.08 if index == 0 else 0.0002 * index),
                steps=run.steps,
                train_tokens=run.train_tokens,
                val_tokens=16 * 2 * 1024,
                elapsed_s=100.0,
                measured_step_s=0.25,
                measured_tokens_per_s=5000.0,
                peak_memory_mb=3000.0,
                device="cpu",
                description="",
                val_source="holdout:2000000",
            )
            for index, candidate in enumerate(candidates)
        ]

    monkeypatch.setattr(screen, "evaluate_batch", fake_evaluate_batch)
    directory = tmp_path / "half-arm"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "arch_zoo_screen.py",
            "--experiment-id", str(directory),
            "--artifacts-root", str(tmp_path),
            "--design", "matched-total",
            "--candidates", "4",
            "--plan", "halving",
            "--stage1-steps", "400",
            "--stage2-steps", "800",
            "--stage3-steps", "1200",
            "--screen-seeds", "42,43",
            "--finalist-seeds", "42,43,44",
            "--keep-stage2", "2",
            "--keep-stage3", "1",
            "--budget-hours", "100",
        ],
    )
    screen.main()

    text = (directory / "decision.md").read_text()
    assert "## Promotion at each screening cut" in text
    # Stage 1 is a 400-step screen on two seeds; stage 2 an 800-step one, so the two
    # rows must not share a threshold.
    assert "| Stage 1 | 400 | 2 | 0.0286 | 0.0286 |" in text
    assert "| Stage 2 | 800 | 2 | 0.0165 | 0.0165 |" in text
    assert "decided by the data" in text


def test_a_flat_arm_does_not_claim_the_screening_seeds_it_never_ran(tmp_path, monkeypatch):
    """The record's seed list has to be the seeds it was actually read over.

    A flat plan trains no screening stage, so `--screen-seeds` names seeds that never
    produce a row. Taking the *plan's* union wrote a record whose `seeds` read
    `[42, 43, 45, 46, 47]` against three seeds of ledgers - contradicting its own
    `seeds_per_architecture`, and therefore the rendered decision, which lists the
    rows. `arch_zoo_analyze.py` derives the set from the results, so the two writers
    disagreed about what the same arm had been read over; this is the arm the
    freshness check caught it on.
    """
    import arch_zoo_screen as screen
    from act.search.architecture_eval import EvalResult

    def fake_evaluate_batch(candidates, *, run, device, output_dir, tag):
        return [
            EvalResult(
                name=candidate.name,
                family=candidate.family,
                genome_hash=f"hash-{candidate.name}",
                seed=run.seed,
                total_params=candidate.total_params,
                active_params=candidate.active_params,
                contracted_flops_per_token=2.5e8,
                dense_runtime_flops_per_token=4.0e8,
                training_flops=2.5e8 * run.train_tokens,
                validation_flops=2.5e8 * 16 * 2 * 1024,
                final_train_loss=3.1 + 0.02 * index,
                mean_train_loss=3.15 + 0.02 * index,
                validation_loss=3.0 + 0.02 * index,
                steps=run.steps,
                train_tokens=run.train_tokens,
                val_tokens=16 * 2 * 1024,
                elapsed_s=100.0,
                measured_step_s=0.25,
                measured_tokens_per_s=5000.0,
                peak_memory_mb=3000.0,
                device="cpu",
                description="",
                val_source="holdout:2000000",
            )
            for index, candidate in enumerate(candidates)
        ]

    monkeypatch.setattr(screen, "evaluate_batch", fake_evaluate_batch)
    directory = tmp_path / "flat-arm"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "arch_zoo_screen.py",
            "--experiment-id", str(directory),
            "--artifacts-root", str(tmp_path),
            "--design", "matched-total",
            "--candidates", "3",
            "--plan", "flat",
            "--flat-steps", "1200",
            "--screen-seeds", "42,43",
            "--finalist-seeds", "45,46,47",
            "--budget-hours", "100",
        ],
    )
    screen.main()

    record = json.loads((directory / "ranked.json").read_text())
    assert record["seeds"] == [45, 46, 47]
    # And that is the same set every ranking row was read over, so the record cannot
    # disagree with itself about which seeds exist.
    assert {frozenset(seeds) for seeds in record["seeds_per_architecture"].values()} == {
        frozenset({45, 46, 47})
    }


def test_the_screen_plan_is_executable_and_restarts_only_its_own_ledgers(tmp_path):
    """The recommended loop has to run end to end, not just be written down.

    A budget that cannot cover a single candidate batch makes the run stop at the
    first screen without training, which is enough to prove the plan is wired to
    the two-stage branch, that it clears exactly its own ledgers and that it still
    writes a report from what it has (nothing) rather than crashing.
    """
    import subprocess

    output_dir = tmp_path / "screen-arm"
    stale = output_dir / "stage2_seed42_results.jsonl"
    output_dir.mkdir()
    stale.write_text("{}\n")

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "scripts" / "arch_zoo_screen.py"),
            "--experiment-id",
            str(output_dir),
            "--plan",
            "screen",
            "--promote-ties",
            "--stage1-steps",
            "400",
            "--stage3-steps",
            "1200",
            "--screen-seeds",
            "42,43",
            "--finalist-seeds",
            "42,43,44",
            "--budget-hours",
            "1e-9",
        ],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    assert "budget reached before Screen completed" in result.stdout
    assert "screen produced no results; stopping" in result.stdout
    # The screen plan owns stage1 and stage3 ledgers; an unrelated stage2 ledger
    # is not this invocation's to overwrite.
    assert stale.exists()
    assert not (output_dir / "stage1_seed42_results.jsonl").exists()
    # A report is still written, so an aborted arm is readable rather than absent.
    assert (output_dir / "decision.md").exists()


def test_the_default_threshold_is_derived_from_the_measured_std_and_the_seed_count():
    """The threshold is a standard error of a *difference*, not a pinned number.

    The phase's screening threshold used to be 0.018 nats, documented as the matched
    horizon set's standard error at 400 steps. That set does not produce 0.018 under
    any definition: its 400-step per-architecture std is 0.0286 nats and its
    single-mean standard error at three seeds is 0.0165. The quantity a promotion
    needs is neither - it is the standard error of the difference between two
    candidates' means, `std * sqrt(2 / seeds)` - so the threshold is computed from the
    measured std and the design's seed count, and these are the numbers it yields.
    """
    # The measured stds, which are findings section 4's `seed std` column.
    assert SCREEN_SEED_STD_BY_STEPS == {400: 0.0286, 800: 0.0165, 1200: 0.0148}
    assert SCREEN_SEED_STD_NATS == pytest.approx(0.0286)

    # Two seeds: the threshold is the std itself, because sqrt(2/2) = 1.
    assert comparison_noise_nats(2, 400) == pytest.approx(0.0286)
    # Three seeds: 0.82x the std, which is why the seed count is printed beside it.
    assert comparison_noise_nats(3, 400) == pytest.approx(0.0286 * (2 / 3) ** 0.5)
    assert comparison_noise_nats(3, 400) == pytest.approx(0.02335, abs=1e-5)
    # Later stages are quieter, so the same design's threshold is smaller there.
    assert comparison_noise_nats(2, 800) == pytest.approx(0.0165)
    assert comparison_noise_nats(2, 1200) == pytest.approx(0.0148)
    # A threshold below the std only ever makes the report stricter, never looser.
    assert comparison_noise_nats(2, 400) > 0.018

    with pytest.raises(ValueError):
        comparison_noise_nats(0, 400)


def test_an_unmeasured_length_borrows_the_nearest_measured_std_below_it():
    """Borrowing downward is the conservative direction: std falls with training.

    Taking a longer stage's smaller std would understate the noise at a shorter stage,
    which is the one way this helper could make the report claim resolution it does not
    have.
    """
    assert seed_std_for_steps(400) == pytest.approx(0.0286)
    assert seed_std_for_steps(600) == pytest.approx(0.0286)
    assert seed_std_for_steps(800) == pytest.approx(0.0165)
    assert seed_std_for_steps(1500) == pytest.approx(0.0148)
    # Shorter than anything measured: fall back to the noisiest measurement, not to a
    # guess about a length nobody has run.
    assert seed_std_for_steps(100) == pytest.approx(SCREEN_SEED_STD_NATS)


def test_a_borrowed_length_is_named_beside_the_value_it_came_from():
    """Borrowing downward protects a short stage; borrowing upward is the other half.

    A confirmation longer than every measured length takes the longest measurement,
    which *overstates* the noise at that stage. That is the safe direction - a larger
    std means a larger promotion margin, so fewer candidates are promoted on noise -
    but it is not free and it is not visible if the record prints the value alone: a
    12,000-step cut would show 0.0148 as though it had been measured at 12,000.

    The horizon arm measured what that costs. Its 12,000-step rows give a
    per-architecture seed std of **0.0056** against the 0.0148 borrowed from 1,200, so
    the borrowed figure is a **2.6x over-estimate** there and the promotion margin is
    2.6x wider than the data requires - costing seeds, not buying safety. The value is
    deliberately *not* substituted, because the two numbers are not the same
    measurement: different candidate sets (crossed vs the matched v2 set), different
    precision (fp16 vs fp32), and two seeds against three. The arm's own 1,200-step
    anchor reads 0.0118 where this table reads 0.0148 at the same length, which is
    how far apart the two designs' noise levels are.
    """
    # The horizon this phase measured, and the length its std is actually borrowed from.
    assert seed_std_measured_at_steps(12000) == 1200
    assert seed_std_for_steps(12000) == pytest.approx(SCREEN_SEED_STD_BY_STEPS[1200])
    assert seed_std_for_steps(12000) / 0.0056 == pytest.approx(2.64, abs=0.01)
    # A length in the table reports itself, so the annotation appears only where it is
    # true and the table does not carry a caveat on every row.
    for steps in SCREEN_SEED_STD_BY_STEPS:
        assert seed_std_measured_at_steps(steps) == steps
