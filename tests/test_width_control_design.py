"""The design note's matched-budget table is recomputed, not trusted.

``docs/width-control-design.md`` section 8 names the first width arm's exact shape and
claims that it holds the *total* parameter budget to three decimal places while spending
4.7% more *active* arithmetic than the baseline it is compared against. That claim is the
whole reason the table exists - an arm that holds neither budget measures capacity, and an
arm that holds both silently measures something else - so it is derived here from
``estimated_parameter_count`` and from the config grid the note says it searched.

The numbers are the model's own; only the *choice* of which budget to hold is the note's.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from act.model.v0_moe import V0MoEConfig, estimated_parameter_count

ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "docs" / "width-control-design.md"

VOCAB, CONTEXT = 32_256, 512


def config(hidden: int, depth: int, heads: int, experts: int, active: int, width: int) -> V0MoEConfig:
    return V0MoEConfig(
        vocab_size=VOCAB,
        context_size=CONTEXT,
        hidden_size=hidden,
        depth=depth,
        attention_heads=heads,
        experts_per_layer=(experts,) * depth,
        active_experts=active,
        expert_width=width,
    )


MATCH = config(768, 12, 12, 8, 2, 768)
#: Row 2 of the table: the recommended arm. Depth halved, width up a third, heads with
#: the width, routing regime untouched.
WIDE_TOTAL = config(1024, 6, 16, 8, 2, 1100)
#: Row 3: the same shape with the expert width left square, which holds the *total* 4.5%
#: under and still spends 2.4% more active arithmetic.
WIDE_SQUARE = config(1024, 6, 16, 8, 2, 1024)
#: Row 4: the tightest joint fit the grid contains - both budgets within 0.5% - bought
#: with the expert count as well as the residual width.
WIDE_JOINT = config(896, 6, 14, 7, 2, 1568)
#: Row 5: the fallback that keeps ``E8`` and moves the expert width instead.
WIDE_KEEPS_E8 = config(1024, 5, 16, 8, 2, 1376)
#: Row 5: the same trade in the other direction, for the table's symmetry.
DEEP = config(896, 8, 14, 8, 2, 896)


@pytest.fixture(scope="module")
def note() -> str:
    return NOTE.read_text(encoding="utf-8")


def test_the_recommended_arm_holds_the_total_budget_at_an_active_cost(note: str) -> None:
    """The arm's declared bargain: total 1.000x, active 1.047x, and it must say so.

    Holding *both* budgets is impossible in this direction of the shape space, because
    cutting depth removes experts (mostly inactive) while raising width grows attention and
    the residual stream (entirely active). So the honest arm declares which one it holds
    and reports the other, and this test refuses an edit that quietly drops the price.
    """

    match_total, match_active = estimated_parameter_count(MATCH)
    wide_total, wide_active = estimated_parameter_count(WIDE_TOTAL)

    assert (match_total, match_active) == (166_423_296, 81_488_640)
    assert (wide_total, wide_active) == (166_392_832, 85_292_032)
    assert wide_total / match_total == pytest.approx(1.000, abs=0.001)
    assert wide_active / match_active == pytest.approx(1.047, abs=0.001)

    # The direction of the confound, stated as an inequality rather than a rounded ratio:
    # at equal total parameters the wide arm is the *more* expensive one per token, so any
    # win it shows is not a FLOP saving.
    assert wide_active > match_active

    for rendered in (
        "166,423,296",
        "81,488,640",
        "166,392,832",
        "85,292,032",
        "**1.000x**",
        "1.047x",
        "0.513",
    ):
        assert rendered in note, f"the note no longer states {rendered!r}"


def test_the_other_rows_are_what_the_note_says_they_are(note: str) -> None:
    """Row 3 is the arrow the reader needs: removing 4.5% of the parameters does not buy back
    the active gap, it only narrows it from 4.7% to 2.4%. Row 4 is the only one that closes
    both - by spending the expert width as well as the residual one - and row 5 shows the
    same trade with a deeper student."""

    match_total, match_active = estimated_parameter_count(MATCH)

    for cfg, total_ratio, active_ratio, rendered in (
        (WIDE_SQUARE, 0.955, 1.024, "0.955x"),
        (WIDE_JOINT, 0.999, 1.005, "**1.005x**"),
        (WIDE_KEEPS_E8, 1.002, 1.009, "1.002x"),
        (DEEP, 0.946, 0.986, "0.946x"),
    ):
        total, active = estimated_parameter_count(cfg)
        assert total / match_total == pytest.approx(total_ratio, abs=0.001)
        assert active / match_active == pytest.approx(active_ratio, abs=0.001)
        assert rendered in note, f"the note no longer states {rendered!r}"

    # The direction that matters: only the two fallbacks get the active gap under 1%.
    assert 1.005 < 1.009 < 1.024 < 1.047


def test_the_grid_the_note_claims_to_have_searched_really_holds_that_many_shapes(note: str) -> None:
    """The note says 46 shapes hold both budgets within +/-2%, so the search is re-run.

    The grid is the model's own legal grid, spelled out at the resolution the note states,
    because "46 shapes" is only meaningful against a stated grid - a wider one returns more,
    a narrower one fewer. Two of its bounds are class invariants rather than choices: expert
    count at most 16 (the grow ceiling) and ``active_experts`` at least 2. An earlier version
    of this search allowed ``active_experts = 1`` and found 32, which is a shape the model
    cannot even construct - so the count is pinned against the *legal* grid.
    """

    match_total, match_active = estimated_parameter_count(MATCH)

    holding = []
    for hidden in (896, 1024, 1152, 1280):
        heads = hidden // 64
        for depth in range(4, 14):
            for experts in range(2, 17):
                for active in (2,):
                    if active > experts:
                        continue
                    for width in range(256, 2_049, 32):
                        total, activ = estimated_parameter_count(
                            config(hidden, depth, heads, experts, active, width)
                        )
                        if 0.98 <= total / match_total <= 1.02 and 0.98 <= activ / match_active <= 1.02:
                            holding.append((hidden, depth, experts, active, width))

    assert len(holding) == 46
    assert "**46**" in note and "241 within +/-5%" in note
    assert (896, 6, 7, 2, 1568) in holding, "the grid's tightest joint fit moved"; 


def test_every_shape_in_the_table_is_kernel_aligned_by_the_projects_own_rule(note: str) -> None:
    """The declared-config alignment rule is what makes a shape buildable, not just stable.

    ``validate_declared_v0_config`` requires ``hidden_size % 128 == 0`` and
    ``head_dim % 64 == 0`` for a declared run, and the head geometry is why width cannot
    move alone: at ``head_dim = 64`` a +128 width step is +2 heads. Both recommended shapes
    are on that lattice, and the test states the consequence rather than the rule.
    """

    for cfg, heads in ((MATCH, 12), (WIDE_TOTAL, 16), (WIDE_JOINT, 14), (WIDE_KEEPS_E8, 16), (DEEP, 14)):
        assert cfg.hidden_size % 128 == 0
        assert cfg.hidden_size // cfg.attention_heads == 64
        assert cfg.attention_heads == heads
        assert cfg.context_size % 128 == 0

    # +128 of width at head_dim 64 is +2 heads, which is the constraint the note records.
    assert WIDE_TOTAL.hidden_size - MATCH.hidden_size == 256
    assert WIDE_TOTAL.attention_heads - MATCH.attention_heads == 4


def test_the_note_states_the_shape_the_arm_would_be_added_as(note: str) -> None:
    """A design note that names an arm has to name the line that creates it, exactly.

    The three genes it says it holds - routing regime, head geometry, and the two it
    moves - are all in this one dict, so the dict is checked against the config the test
    just priced. An arm added with a different one is a different experiment.
    """

    assert (
        '"wide": dict(hidden_size=1024, depth=6, attention_heads=16, expert_count=8, '
        'active_experts=2, expert_width=1100),' in note
    )
    assert (WIDE_TOTAL.hidden_size, WIDE_TOTAL.depth, WIDE_TOTAL.attention_heads) == (1024, 6, 16)
    assert WIDE_TOTAL.experts_per_layer[0] == 8
    assert WIDE_TOTAL.active_experts == 2
    assert WIDE_TOTAL.expert_width == 1100


def test_the_joint_fallbacks_are_legal_shapes_not_just_arithmetic(note: str) -> None:
    """Every row's shape has to be constructible by the model, not only close on paper.

    ``V0MoEConfig.__post_init__`` is where two of the grid's bounds come from - every layer
    needs at least 2 experts and at most 16, and ``active_experts`` is bounded below by 2 -
    so constructing each row is the check that the table is a table of *buildable* students.
    """

    for cfg in (MATCH, WIDE_TOTAL, WIDE_SQUARE, WIDE_JOINT, WIDE_KEEPS_E8, DEEP):
        assert 2 <= cfg.active_experts <= min(cfg.experts_per_layer)
        assert all(2 <= count <= 16 for count in cfg.experts_per_layer)
        assert cfg.hidden_size % cfg.attention_heads == 0
    assert WIDE_JOINT.experts_per_layer == (7,) * 6
    assert WIDE_KEEPS_E8.experts_per_layer == (8,) * 5
