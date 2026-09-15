"""Section 5.24: the discarded routing head, pinned at the code and study level.

Two distinct things have to hold, and they fail in different ways.

The **code** claim is that a mutated layer's learned router is re-drawn at random
by the historical rebuild - because the router's weight shape depends on the
expert count and the rebuild carries only same-shape keys - while the experts'
own weights survive by id. If that stopped being true, the study's premise would
be gone and section 5.24's numbers would be measuring nothing.

The **study** claim is branch (b): preservation removes over half the penalty and
a residual survives. Its cleanest support is the pairing - both settings make the
same architectural choices, so the final genomes and the frozen references are
identical in every seed. That is pinned here, because if the pairing ever broke
the residual penalty would become a topology comparison instead of a paired one.
"""

from __future__ import annotations

import json

import pytest
import torch

from act.evolution.figures_data import summaries_dir
from act.evolution.study import _binomial_survival, _gate_min_wins
from act.model import model_from_genome, rebuild_model_from_genome
from act.topology import create_genome, prune_expert

PER_PRIMARY_ALPHA = 0.0125
DECLARED_BAR = 215
#: file -> (study id, seed block, {comparison: (wins, mean, p, passes)})
QUOTED = {
    "router-preservation-depth2-384.json": (
        "study_18c56d8bf539e2f3",
        (96000, 96384),
        {
            "Rp_vs_R": (264, -0.054204, 7.64e-14, True),
            "Rp_own_vs_Rp": (254, -0.044130, 1.22e-10, True),
            "R_own_vs_R": (304, -0.098334, 3.78e-32, True),
        },
    ),
    "router-preservation-depth3-384.json": (
        "study_65103c230c610378",
        (97000, 97384),
        {
            "Rp_vs_R": (270, -0.069380, 5.19e-16, True),
            "Rp_own_vs_Rp": (258, -0.058341, 7.33e-12, True),
            "R_own_vs_R": (301, -0.127720, 1.94e-30, True),
        },
    ),
}
#: Share of the historical penalty that keeping the router removes, per depth.
QUOTED_SHARE_REMOVED = {
    "router-preservation-depth2-384.json": 0.551,
    "router-preservation-depth3-384.json": 0.543,
}


def _load(name: str) -> dict:
    path = summaries_dir() / name
    if not path.exists():
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


def test_a_mutated_layers_router_is_discarded_without_the_flag() -> None:
    """The defect section 5.24 exists to measure, pinned directly.

    Experts survive by id; the mutated layer's routing head does not, because its
    shape changes with the expert count. The unmutated layer's router is copied,
    which is what shows the copy loop is working and the loss is specific.
    """
    genome = create_genome(
        7, hidden_size=24, vocab_size=32, depth=2, ffn="moe",
        expert_count=4, active_experts=2,
    )
    model = model_from_genome(genome, context_size=32, num_heads=4)
    with torch.no_grad():
        for block in model.blocks:
            block.router.weight.fill_(0.25)
            block.experts[0][0].weight.fill_(0.5)

    target = create_genome(
        7, hidden_size=24, vocab_size=32, depth=2, ffn="moe",
        expert_count=4, active_experts=2,
    )
    prune_expert(target, target.layers[0].id, target.layers[0].experts[-1].id)

    rebuilt = rebuild_model_from_genome(model, target, context_size=32, num_heads=4)
    mutated = rebuilt.blocks[0]
    assert bool((mutated.experts[0][0].weight == 0.5).all()), "experts survive by id"
    assert not bool((mutated.router.weight == 0.25).all()), "router was re-drawn"
    assert bool((rebuilt.blocks[1].router.weight == 0.25).all()), "unmutated layer kept"


def test_preserve_router_keeps_the_surviving_rows() -> None:
    """The flag must carry exactly the surviving experts' rows, no more."""
    genome = create_genome(
        7, hidden_size=24, vocab_size=32, depth=2, ffn="moe",
        expert_count=4, active_experts=2,
    )
    model = model_from_genome(genome, context_size=32, num_heads=4)
    with torch.no_grad():
        for block in model.blocks:
            block.router.weight.fill_(0.25)

    target = create_genome(
        7, hidden_size=24, vocab_size=32, depth=2, ffn="moe",
        expert_count=4, active_experts=2,
    )
    prune_expert(target, target.layers[0].id, target.layers[0].experts[-1].id)
    rebuilt = rebuild_model_from_genome(
        model, target, context_size=32, num_heads=4, preserve_router=True
    )
    rows = rebuilt.blocks[0].router.weight
    assert rows.shape[0] == 3
    assert bool((rows == 0.25).all()), "every surviving row must be carried"


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_24_numbers_recompute_from_per_seed(name: str) -> None:
    study_id, block, quoted = QUOTED[name]
    summary = _load(name)
    assert summary["record_type"] == "router_preservation_study"
    assert summary["study_id"] == study_id
    assert summary["per_primary_alpha"] == PER_PRIMARY_ALPHA
    assert summary["declared_bar"] == DECLARED_BAR
    start, stop = block
    assert summary["seeds"] == list(range(start, stop))
    assert summary["seeds"] == [row["seed"] for row in summary["per_seed"]]
    assert all(summary["built_in_checks"].values()), summary["built_in_checks"]

    for comparison_name, (wins, mean, p, passes) in quoted.items():
        first, second = comparison_name.split("_vs_")
        deltas = [
            row["losses"][first] - row["losses"][second] for row in summary["per_seed"]
        ]
        assert sum(delta < 0 for delta in deltas) == wins, comparison_name
        assert sum(deltas) / len(deltas) == pytest.approx(mean, abs=5e-6)
        assert _binomial_survival(wins, len(deltas)) == pytest.approx(p, rel=1e-2)
        assert summary["comparisons"][comparison_name]["passes"] is passes
        assert summary["comparisons"][comparison_name]["min_wins"] == _gate_min_wins(
            len(deltas), PER_PRIMARY_ALPHA
        )


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_24_pairing_is_exact(name: str) -> None:
    """Both settings must make the same architectural choices.

    This is what makes the residual penalty a within-seed paired contrast on one
    final topology rather than a comparison between two topologies. It is also
    the property the study text leans on hardest, so it is asserted rather than
    reasoned about.
    """
    summary = _load(name)
    rows = summary["per_seed"]
    assert all(row["mutations"]["R"] == row["mutations"]["Rp"] == 3 for row in rows)
    assert all(row["routers_differ"] is False for row in rows), (
        "the two settings ended on different final genomes; the residual penalty "
        "would no longer be a paired comparison"
    )
    assert all(row["losses"]["R_own"] == row["losses"]["Rp_own"] for row in rows)


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_24_branch_b_and_the_share_removed(name: str) -> None:
    """Branch (b): preservation helps AND a residual penalty survives."""
    summary = _load(name)
    comparisons = summary["comparisons"]
    assert comparisons["Rp_vs_R"]["passes"] is True, "P1 must pass for branch (b)"
    assert comparisons["Rp_own_vs_Rp"]["passes"] is True, "P2 must pass for branch (b)"

    discarded = abs(comparisons["R_own_vs_R"]["mean_delta"])
    preserved = abs(comparisons["Rp_own_vs_Rp"]["mean_delta"])
    share = (discarded - preserved) / discarded
    assert share == pytest.approx(QUOTED_SHARE_REMOVED[name], abs=2e-3), (
        "the share of the penalty that keeping the router removes is the study's "
        "headline quantity"
    )
    assert 0.4 < share < 0.7, "over half, but not all - the branch (b) claim"
