"""Pins for the section 5.33 stored records (probe selection vs a load rule).

Section 5.33 asks whether the probe's own prune-target selection beats the cheap load
rule that section 5.32 validated on the non-probe paths. It matters because section
5.32's consequence - "the discovery controller should prefer the quietest expert" - is
wrong in the place a reader would act on it: the setting is unreachable on the probe
path, and following the load rule instead costs 0.031-0.034 nats.

Four things are pinned, and each is a way the result could quietly stop meaning what
the report says it means:

1. **The verdict and its direction.** Both primaries pass at both depths, and the
   three-deep ordering (`A` < `Q` < `M`) is the interpretable part - an `A`-vs-`Q`
   failure would have said the opposite, and a test asserting only "selection helps"
   would let that rot.
2. **The validity conditions the study declared.** The fixed arm identical across all
   four selection rules, the full mutation budget used, and the treatment actually
   binding - prunes present in >= 95% of seed-selection pairs. A block where the rule
   rarely fires would answer nothing.
3. **The re-derivation.** Every stored comparison recomputes from the stored per-seed
   rows, so a summary edited after the fact fails here rather than reading as the
   original result.
4. **The cross-check that would catch a regenerated, drifted summary:** the stored
   per-seed rows must reproduce the stored loss means, and the fixed arm must be one
   value per block (the `F` column is shared by all four selections, so if that ever
   varied the contrast would not be paired).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import probe_selection_study as sel  # noqa: E402

from act.evolution.figures_data import summaries_dir  # noqa: E402

QUOTED: dict[str, dict] = {
    "probe-selection-depth2-1536.json": {
        "study_id": "study_dbd63572517fd575",
        "depth": 2,
        "means": {
            "F": 14.322995707082251,
            "A": 14.277621073027452,
            "Q": 14.30881832834954,
            "M": 14.318801315811774,
            "G": 14.360936088332286,
        },
        "primaries": {
            "fused_A_vs_M": (-0.04118, 899, 5.00e-05, True),
            "fused_A_vs_Q": (-0.03120, 818, 5.00e-05, True),
        },
        "load_rule_share_of_probe_advantage": 0.242,
        "own_A_vs_Q": -0.03571,
        "pairs_with_a_prune": (6110, 6144),
        "four_distinct": (1253, 1536),
    },
    "probe-selection-depth3-1536.json": {
        "study_id": "study_a24a62de8ce82d39",
        "depth": 3,
        "means": {
            "F": 13.444358552573249,
            "A": 13.387507734975467,
            "Q": 13.42194030690007,
            "M": 13.435984409802282,
            "G": 13.477511267488202,
        },
        "primaries": {
            "fused_A_vs_M": (-0.04848, 947, 5.00e-05, True),
            "fused_A_vs_Q": (-0.03443, 845, 5.00e-05, True),
        },
        "load_rule_share_of_probe_advantage": 0.290,
        "own_A_vs_Q": -0.05414,
        "pairs_with_a_prune": (6091, 6144),
        "four_distinct": (1300, 1536),
    },
}


def _load(name: str) -> dict:
    path = summaries_dir() / name
    if not path.exists():
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_the_committed_record_matches_the_published_numbers(name: str) -> None:
    summary = _load(name)
    quoted = QUOTED[name]
    assert summary["protocol"] == (
        "probe-argmin-vs-quietest-load-rule-vs-pool-matched-control"
    )
    assert summary["study_id"] == quoted["study_id"]
    assert summary["depth"] == quoted["depth"]
    assert len(summary["per_seed"]) == summary["declared_n"]
    for label, value in quoted["means"].items():
        assert summary["loss_means"][label] == pytest.approx(value, rel=1e-12)


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_the_three_deep_ordering_holds(name: str) -> None:
    """`A` < `Q` < `M` is the result; `A` < `M` alone would not distinguish the rules.

    An `A`-vs-`Q` failure is section 5.33's branch (c) - the load rule is no worse and
    the probe's prune decision can be replaced by a cheap one - so a test that only
    asserted "the probe beats random" would let the actionable half rot.
    """
    summary = _load(name)
    means = summary["loss_means"]
    assert means["A"] < means["Q"] < means["M"], (
        "the ordering that carries the interpretation no longer holds"
    )
    assert means["A"] < means["F"], "mutation must pay in this configuration"
    quoted = QUOTED[name]
    for key, (mean, wins, p, passes) in quoted["primaries"].items():
        comparison = summary["comparisons"][key]
        assert comparison["passes"] is passes
        assert comparison["wins"] == wins
        assert comparison["mean_delta"] == pytest.approx(mean, abs=5e-6)
        # Several primaries sit at the permutation floor 1/(N+1) = 4.9998e-05, so
        # the quoted 5e-05 is a rounded reading of "no permutation was as extreme"
        # rather than a measured value: compare in absolute terms.
        assert comparison["permutation_p"] == pytest.approx(p, abs=1e-7)
        assert comparison["permutation_p"] >= 1.0 / (sel.N_PERMUTATIONS + 1) * 0.999
    # The load rule is not empty - it beats the loss-blind control - and it is not
    # enough. Both halves are asserted because either alone misreads the result.
    assert summary["comparisons"]["fused_Q_vs_M"]["passes"] is True
    share = summary["comparisons"]["fused_Q_vs_M"]["mean_delta"] / (
        summary["comparisons"]["fused_A_vs_M"]["mean_delta"]
    )
    assert share == pytest.approx(quoted["load_rule_share_of_probe_advantage"], abs=0.01)
    assert summary["comparisons"]["own_A_vs_Q"]["mean_delta"] == pytest.approx(
        quoted["own_A_vs_Q"], abs=5e-6
    )


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_the_declared_validity_conditions_hold(name: str) -> None:
    summary = _load(name)
    checks = summary["built_in_checks"]
    assert all(checks.values()), checks
    assert checks["the_fixed_arm_is_identical_across_all_four_selections"]
    assert checks["all_selections_used_the_full_budget"]
    pairs, total = QUOTED[name]["pairs_with_a_prune"]
    assert summary["seed_selection_pairs_with_a_prune"] == pairs
    assert summary["seed_selection_pairs"] == total
    distinct, seeds = QUOTED[name]["four_distinct"]
    assert summary["seeds_with_four_distinct_genomes"] == distinct
    assert len(summary["seeds"]) == seeds
    # Two primaries per depth, so the Bonferroni bar the study registered.
    assert summary["per_primary_alpha"] == 0.05 / 4


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_the_stored_comparisons_recompute_from_the_stored_rows(name: str) -> None:
    """A summary edited after the fact must fail here, not read as the result."""
    summary = _load(name)
    rows = summary["per_seed"]

    def losses(label: str) -> list[float]:
        return [row["losses"][label] for row in rows]

    recomputed = sel._comparison(
        [a - b for a, b in zip(losses("A"), losses("Q"))]
    )
    stored = summary["comparisons"]["fused_A_vs_Q"]
    assert stored["n"] == recomputed["n"]
    assert stored["wins"] == recomputed["wins"]
    assert stored["mean_delta"] == pytest.approx(recomputed["mean_delta"], rel=1e-12)
    assert stored["permutation_p"] == pytest.approx(recomputed["permutation_p"], rel=1e-12)
    assert stored["passes"] is recomputed["passes"]
