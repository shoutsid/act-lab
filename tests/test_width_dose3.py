"""Section 5.60's width bisection: the design pins, before any new seed.

The covariate programme closed on "only the depth axis moves the gate", and
that closure carried a number it never closed. 5.49 measured width 48 (2.87x
the above-onset cell, 88,032 parameters at depth 3) as still below the gate;
5.50 measured width 96 (11.15x, 341,952) as above it. So the width-axis gate
sits somewhere in a **3.9x window** - the widest unresolved quantity left in
the programme, and the one that decides what "only depth moves the gate" is
worth, since the same programme says capacity *does* move it above 3.8x.

This arm bisects that window at its geometric midpoint in parameter space:
width 68, 173,672 parameters at depth 3 (5.66x) and 230,520 at depth 4 (7.52x).
One dose halves the window from either side, whichever way the gate falls.

These tests pin the two cells, the one gene each changes, the block freshness,
the bisection arithmetic, the bar, the declared mechanism prediction and the
branch table - so the registration cannot drift between now and the run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import null_toy_study as study  # noqa: E402

from act.evolution.study import _gate_min_wins  # noqa: E402
from act.model import model_from_genome  # noqa: E402
from act.topology import create_genome  # noqa: E402

SUMMARIES = ROOT / "artifacts" / "study-summaries"

# The window this arm bisects, as the two committed doses and their measured
# parameter counts at depth 3.
CLOSED_PARAMS = 88_032  # width 48, 5.49: value +0.0615, no gate
OPEN_PARAMS = 341_952  # width 96, 5.50: value -0.8966, gate


def _genome(depth: int, width: int):
    return create_genome(
        0,
        hidden_size=width,
        vocab_size=study.REGISTERED["vocab_size"],
        depth=depth,
        ffn="moe",
        expert_count=study.REGISTERED["expert_count"],
        active_experts=study.REGISTERED["active_experts"],
    )


def _parameters(depth: int, width: int) -> int:
    model = model_from_genome(
        _genome(depth, width), context_size=study.REGISTERED["sequence_length"]
    )
    return sum(p.numel() for p in model.parameters())


def test_the_two_cells_change_exactly_one_gene_each() -> None:
    assert study.DOSE3_PROFILES == ("width-d3-dose3", "width-d4-dose3")
    assert study.PROFILES[study.DOSE3_BELOW]["overrides"] == {
        "depth": 3,
        "hidden_size": 68,
    }
    assert study.PROFILES[study.DOSE3_ABOVE]["overrides"] == {
        "depth": 4,
        "hidden_size": 68,
    }
    # Every other registered knob is the family's, unchanged: this is a width
    # dose, not a retune.
    assert study.REGISTERED["steps"] == 48
    assert study.REGISTERED["mutation_interval"] == 12
    assert study.REGISTERED["expert_count"] == 4
    assert study.REGISTERED["adaptive_policy"] == "probe_loss"


def test_the_bisection_lands_on_the_windows_geometric_midpoint() -> None:
    """Width 68 is chosen by the window's geometry, not by what it might show."""
    below = _parameters(3, 68)
    above = _parameters(4, 68)
    assert (below, above) == (173_672, 230_520)
    midpoint = (CLOSED_PARAMS * OPEN_PARAMS) ** 0.5
    assert below / midpoint == pytest.approx(1.0, abs=0.01), (below, midpoint)
    # And the dose is a real step on both depths' ladders, not a rounding of one
    # of them: it sits between the two measured doses in parameter space.
    assert CLOSED_PARAMS < below < OPEN_PARAMS
    assert _parameters(4, 48) < above < _parameters(4, 96)
    # The multipliers the report quotes, against the above-onset cell.
    onset = _parameters(4, 24)
    assert below / onset == pytest.approx(5.66, abs=0.01)
    assert above / onset == pytest.approx(7.52, abs=0.01)


def test_the_reports_window_table_is_the_profiles_own_arithmetic() -> None:
    """Every row of §5.60's table, derived from the profiles rather than transcribed.

    This exists because the first draft of that table carried two invented
    parameter counts: `width-d3-equal` was written as 27,456 / 0.90x when the
    committed cell is 31,192 / 1.02x and 5.48's own table says so. A number that
    is quoted from the record can drift silently; one that is *computed from the
    profile* cannot, so the whole column is pinned here.
    """
    onset = _parameters(4, 24)
    rows = {
        ("depth3", 3, 24): (23_280, 0.76),
        ("width-d3-equal", 3, 28): (31_192, 1.02),
        ("width-d3-dose", 3, 48): (88_032, 2.87),
        (study.DOSE3_BELOW, 3, 68): (173_672, 5.66),
        ("width-d3-dose2", 3, 96): (341_952, 11.15),
        ("depth4-prefix", 4, 24): (30_672, 1.00),
        ("width-d4-equal", 4, 28): (41_160, 1.34),
        ("width-d4-dose", 4, 48): (116_640, 3.80),
        (study.DOSE3_ABOVE, 4, 68): (230_520, 7.52),
        ("width-d4-dose2", 4, 96): (454_464, 14.82),
    }
    for (name, depth, width), (params, multiple) in rows.items():
        # Width 24 is the registered default, so those two profiles override only
        # the depth - which is the point: they are the un-dosed reference cells.
        expected: dict[str, int] = {"depth": depth}
        if width != study.REGISTERED["hidden_size"]:
            expected["hidden_size"] = width
        assert study.PROFILES[name]["overrides"] == expected, name
        assert _parameters(depth, width) == params, name
        assert params / onset == pytest.approx(multiple, abs=0.005), name
    # And the two committed doses that define the window are the ones the
    # registration quotes, so the bisection's bracket is not a restatement.
    assert rows[("width-d3-dose", 3, 48)][0] == CLOSED_PARAMS
    assert rows[("width-d3-dose2", 3, 96)][0] == OPEN_PARAMS


def test_the_blocks_are_fresh_against_every_other_claim() -> None:
    """A block is fresh against other profiles *and* against committed summaries."""
    used: set[int] = set()
    for name, spec in study.PROFILES.items():
        if name in study.DOSE3_PROFILES:
            continue
        used |= set(spec["seeds"])
    for path in SUMMARIES.glob("*.json"):
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            used |= set(payload.get("seeds") or [])
    for name in study.DOSE3_PROFILES:
        seeds = study.PROFILES[name]["seeds"]
        assert len(seeds) == 576
        assert not (set(seeds) & used), name
        assert seeds == tuple(range(min(seeds), min(seeds) + 576))


def test_the_bar_is_the_three_primary_level() -> None:
    """Three primaries, so alpha/3 - stated in the profile, not passed in."""
    for name in study.DOSE3_PROFILES:
        assert study.PROFILES[name]["bonferroni_divisor"] == 3.0
    assert study.PROFILES[study.DOSE3_BELOW]["report_section"] == "5.60"
    assert study.PROFILES[study.DOSE3_ABOVE]["report_section"] == "5.60"
    assert _gate_min_wins(576, study.ALPHA / 3) == 315
    assert _gate_min_wins(576, study.ALPHA / 2) == 313


def test_the_declared_prediction_is_derived_from_the_committed_ladder() -> None:
    """The prediction is a direction the committed cells already point in.

    Below the onset the pool term (random-vs-null) rises with width and above it
    is negative, which is 5.50's mechanism for the gate. Declaring that the flip
    is above the bisection is a claim about *that mechanism*, read off the
    committed cells - so the assertion here is that the series it is derived from
    really is monotone on the way up and negative once the gate opens. A
    prediction whose premise has drifted fails here rather than silently becoming
    a guess.
    """

    def pool(name: str) -> float:
        rows = json.loads((SUMMARIES / name).read_text())["per_seed"]
        return sum(r["losses"]["random"] - r["losses"]["null"] for r in rows) / len(rows)

    below_series = [
        pool("null-control-depth3-384.json"),
        pool("width-d3-equal-576.json"),
        pool("width-d3-dose-576.json"),
    ]
    assert below_series == pytest.approx([0.1068, 0.1484, 0.3053], abs=1e-3)
    assert below_series == sorted(below_series), below_series
    assert pool("width-d3-dose2-576.json") < 0, "the flip is not above the gate"
    assert study.DOSE3_POOL_GROWS is True
    # The above-onset series the dose continues is monotone in value, which is
    # what P3 is registered against.
    assert study.DOSE3_BELOW_REFERENCE == "width-d3-dose"
    assert study.DOSE3_ABOVE_REFERENCE == "width-d4-dose"


def test_the_branch_table_covers_all_three_primaries() -> None:
    """Eight branches, and the two the report names as readings are among them."""
    import inspect

    source = inspect.getsource(study.dose3_verdict)
    assert "(False, True, True)" in source
    assert "(True, True, True)" in source
    assert "(True, False, True)" in source
    assert "(False, False, True)" in source
    assert "(False, True, False)" in source
    assert "(True, True, False)" in source
    assert "(True, False, False)" in source
    assert "(False, False, False)" in source
    assert "the window narrows to (5.66x, 11.15x)" in source
    assert "the window narrows to (2.87x, 5.66x)" in source
    # A miss on the mechanism prediction has to say what state it landed in
    # rather than only reporting False.
    assert "PREDICTION miss names the state" in source


def test_the_run_revision_is_declared_before_the_first_shard() -> None:
    """A registration is only a registration if the revision is frozen first.

    ``None`` is the state a registration passes through between the code landing
    and the freeze commit - the state the summary-label machinery treats as
    "registered but not yet run" rather than as a missing record. The run may not
    start until this is a revision hash.
    """
    import re

    for name in study.DOSE3_PROFILES:
        revision = study.PROFILES[name]["run_revision"]
        assert revision is not None, f"{name} was not frozen before the run"
        assert re.fullmatch(r"[0-9a-f]{7}", revision), (name, revision)
