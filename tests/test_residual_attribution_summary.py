"""The committed section 5.25 summaries stay machine-checked.

Section 5.25 attributes the residual mid-training penalty by forcing every
mutation event to a single operation, so each candidate gets its own run rather
than a subgroup of a mix. The registered verdict is branch (a) - both a prune and
a grow cost - and the finding worth holding down is the comparison between them:
the *grow* is the more expensive operation, which is the opposite of what the
pre-registration predicted.

The forced-operation built-in checks matter more than usual here. If a forced
prune the genome could not perform had ever been silently replaced by the
load-based grow fallback, the prune block would contain grows and its number
would be measuring the wrong operation - so those checks are asserted directly.
"""

from __future__ import annotations

import json

import pytest

from act.evolution.figures_data import summaries_dir
from act.evolution.study import _binomial_survival, _gate_min_wins

PER_PRIMARY_ALPHA = 0.025
DECLARED_BAR = 212
#: file -> (study id, operation, seed block, (wins, mean, p, passes))
QUOTED = {
    "residual-prune-384.json": (
        "study_68591d55f3a56db7", "prune", (98000, 98384), (241, -0.029680, 3.25e-07, True)
    ),
    "residual-grow-384.json": (
        "study_c7ec09294da043fc", "grow", (99000, 99384), (241, -0.047590, 3.25e-07, True)
    ),
}
#: Section 5.24's depth-2 mixed residual with the routing head preserved.
MIXED_RESIDUAL_5_24 = 0.044130


def _load(name: str) -> dict:
    path = summaries_dir() / name
    if not path.exists():
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_25_numbers_recompute_from_per_seed(name: str) -> None:
    study_id, operation, block, (wins, mean, p, passes) = QUOTED[name]
    summary = _load(name)
    assert summary["record_type"] == "residual_attribution_study"
    assert summary["protocol"] == "forced-operation-residual-attribution"
    assert summary["study_id"] == study_id
    assert summary["operation"] == operation
    assert summary["per_primary_alpha"] == PER_PRIMARY_ALPHA
    assert summary["declared_bar"] == DECLARED_BAR
    start, stop = block
    assert summary["seeds"] == list(range(start, stop))
    assert summary["seeds"] == [row["seed"] for row in summary["per_seed"]]

    deltas = [
        row["losses"]["Rp_own"] - row["losses"]["Rp"] for row in summary["per_seed"]
    ]
    assert sum(delta < 0 for delta in deltas) == wins
    assert sum(deltas) / len(deltas) == pytest.approx(mean, abs=5e-6)
    assert _binomial_survival(wins, len(deltas)) == pytest.approx(p, rel=1e-2)
    comparison = summary["comparisons"]["Rp_own_vs_Rp"]
    assert comparison["passes"] is passes
    assert comparison["min_wins"] == _gate_min_wins(len(deltas), PER_PRIMARY_ALPHA)


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_25_the_operation_really_was_forced(name: str) -> None:
    """No event may be anything but the forced operation."""
    summary = _load(name)
    checks = summary["built_in_checks"]
    assert all(checks.values()), checks
    for key in (
        "every_event_was_the_forced_operation",
        "every_seed_used_the_full_budget",
        "no_layer_went_below_two_experts",
        "fixed_transport_reproduces_the_fixed_arm",
    ):
        assert checks[key] is True, key
    sign = -1 if summary["operation"] == "prune" else 1
    start = summary["config"]["expert_count"]
    depth = summary["config"]["depth"]
    for row in summary["per_seed"]:
        assert sum(row["experts"]["final"]) == start * depth + sign * 3
        assert min(row["experts"]["final"]) >= 2
        assert row["mutations"] == 3


def test_section_5_25_grow_costs_more_than_prune() -> None:
    """Branch (a) plus the comparison that the prediction got backwards.

    The pre-registration declared the prune would be the costly operation and the
    grow would not cost at all. Both cost, and the grow costs more. Pinned here so
    the report's claim is tied to the numbers it was read from.
    """
    prune = abs(_load("residual-prune-384.json")["residual_penalty"])
    grow = abs(_load("residual-grow-384.json")["residual_penalty"])
    assert prune > 0 and grow > 0
    assert grow > prune, "the grow is the more expensive operation"
    assert 1.3 <= grow / prune <= 1.9, (grow / prune)


def test_section_5_25_additive_check_against_section_5_24() -> None:
    """Per-event costs, added up, must roughly reproduce 5.24's mixed residual.

    This is the study's consistency test: two pure-operation blocks and one mixed
    block, on different substrates, have to tell the same story. It is asserted
    loosely (20%) because the mix was assumed at 50/50 rather than recorded.
    """
    per_prune = abs(_load("residual-prune-384.json")["residual_penalty"]) / 3
    per_grow = abs(_load("residual-grow-384.json")["residual_penalty"]) / 3
    predicted_mix = 1.5 * per_prune + 1.5 * per_grow
    assert predicted_mix == pytest.approx(MIXED_RESIDUAL_5_24, rel=0.2), (
        predicted_mix,
        MIXED_RESIDUAL_5_24,
    )
