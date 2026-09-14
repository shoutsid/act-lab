"""Tests for the probe-validity counterfactual analysis generator.

That analysis (report section 3.2, figure 4) had no generator in the repository,
so its numbers were reproducible only in principle. These tests pin the parts
that make the new generator trustworthy: the recovered statistic definitions
(which are not the obvious ones), the merge/completeness behaviour, and - the
important one - that it reproduces the committed file's own results block.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import probe_validity_analysis as pv  # noqa: E402


def test_verify_reproduces_every_committed_field() -> None:
    """The committed results block must recompute from its own per_seed block.

    This is the check that recovered the definitions; if it ever fails, the
    committed file and the generator have drifted apart.
    """
    assert pv.verify_against_committed() == 0


def test_one_tailed_p_is_a_normal_tail_not_a_student_t_tail() -> None:
    """The committed file stores the *normal* tail of the t statistic.

    Reading it as a Student t tail would overstate p by ~15%, so the choice is
    pinned against both: the normal tail reproduces the stored 1.374e-04, and
    the df=383 t tail is different (heavier) rather than approximately equal.
    """
    from act.evolution.figures_data import beta_cdf

    t_stat, df = 3.638, 383
    normal = pv.normal_one_tailed_p(t_stat)
    student = 0.5 * beta_cdf(df / (df + t_stat * t_stat), df / 2, 0.5)

    assert normal == pytest.approx(0.0001374, abs=5e-7)
    assert student > normal
    assert not math.isclose(student, normal, rel_tol=1e-3)


def test_pooled_pearson_z_scores_within_seeds() -> None:
    """Within-seed z-scoring, not global: the two differ sharply by construction.

    Two seeds whose probe/final losses are shifted apart will correlate weakly
    globally (the between-seed offset dominates) but perfectly within seed.
    """
    per_seed = {
        "1": {"probe_losses": [1.0, 2.0, 3.0, 4.0], "finals": [10.0, 20.0, 30.0, 40.0]},
        "2": {"probe_losses": [5.0, 6.0, 7.0, 8.0], "finals": [50.0, 60.0, 70.0, 80.0]},
    }

    assert pv.pooled_zscored_pearson(per_seed) == pytest.approx(1.0)

    # The global (between-seed) version would also be 1.0 here, so use a case
    # where the second seed's association is reversed to separate them.
    mixed = {
        "1": {"probe_losses": [1.0, 2.0, 3.0, 4.0], "finals": [10.0, 20.0, 30.0, 40.0]},
        "2": {"probe_losses": [1.0, 2.0, 3.0, 4.0], "finals": [40.0, 30.0, 20.0, 10.0]},
    }
    assert pv.pooled_zscored_pearson(mixed) == pytest.approx(0.0, abs=1e-12)


def test_spearman_matches_known_value_and_breaks_ties_by_order() -> None:
    # Perfectly monotone -> +1; perfectly reversed -> -1.
    assert pv.spearman_rho([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]) == pytest.approx(1.0)
    assert pv.spearman_rho([1.0, 2.0, 3.0, 4.0], [8.0, 7.0, 6.0, 5.0]) == pytest.approx(-1.0)
    # A tied pair is broken by candidate order, so the result is finite and
    # deterministic rather than undefined.
    value = pv.spearman_rho([1.0, 1.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    assert -1.0 <= value <= 1.0


def test_merge_is_seed_ordered_and_complete() -> None:
    merged = pv.merge(
        [
            {"per_seed": {"3001": {"probe_losses": [0], "finals": [0]}}},
            {"per_seed": {"3000": {"probe_losses": [0], "finals": [0]}}},
        ]
    )
    assert list(merged) == ["3000", "3001"]


def test_build_document_records_the_rng_regime() -> None:
    # Two seeds, not one: the t statistic the results block stores needs a
    # sample standard deviation, so a single-seed document is undefined by
    # construction and never arises from a complete sweep.
    per_seed = {
        "3000": {"probe_losses": [1.0, 2.0, 3.0, 4.0], "finals": [1.0, 2.0, 3.0, 4.0]},
        "3001": {"probe_losses": [1.0, 3.0, 2.0, 4.0], "finals": [1.0, 3.0, 2.0, 4.0]},
    }
    decoupled = pv.build_document(per_seed, independent_arm_rng=True)
    coupled = pv.build_document(per_seed, independent_arm_rng=False)

    assert decoupled["rng_regime"] == "decoupled"
    assert coupled["rng_regime"] == "coupled"
    assert decoupled["independent_arm_rng"] is True
    assert coupled["independent_arm_rng"] is False
    # The frozen design string travels with both, so the regimes are comparable.
    assert decoupled["design"] == coupled["design"] == pv.DESIGN
    assert decoupled["results"]["n_seeds"] == 2
    assert decoupled["results"]["pooled_pairs"] == 8


def test_committed_decoupled_analysis_matches_the_report() -> None:
    """Pin the corrected numbers section 3.2 and 9.11 quote."""
    document = json.loads(
        (ROOT / "artifacts/study-summaries/decoupled/probe-validity-analysis.json").read_text()
    )
    results = document["results"]

    assert document["rng_regime"] == "decoupled"
    assert results["mean_per_seed_spearman_rho"] == pytest.approx(0.0776, abs=5e-4)
    assert results["pooled_zscored_pearson_r"] == pytest.approx(0.0809, abs=5e-4)
    assert results["argmin_concordance"] == pytest.approx(0.284, abs=1e-3)
    assert results["n_seeds"] == 384
    assert results["pooled_pairs"] == 1536
    # Still a real signal, weaker than the coupled 0.1057.
    assert results["h1_supported"] is True
    assert results["mean_per_seed_spearman_rho"] < 0.1057
    # The avoidance statistics are essentially unmoved.
    assert results["mean_worst_branch_excess_avoided"] == pytest.approx(0.0839, abs=1e-3)
    assert results["mean_loss_left_on_table_probe_best_vs_best_of_4"] == pytest.approx(
        0.0592, abs=1e-3
    )
