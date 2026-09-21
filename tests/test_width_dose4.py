"""Section 5.61's second width bisection: the design pins, before any new seed.

5.60 halved the width axis's gate window from 3.9x to (2.87x, 5.66x) - 88,032 to
173,672 parameters at the below-onset depth - and, more usefully, found that the
crossing coincides with the controller's edge reaching its **ladder maximum**
while the pool term is still positive.

This arm sits at the *new* window's geometric midpoint: width 57, 123,006
parameters at depth 3 (4.01x) and 163,134 at depth 4 (5.32x). It halves the window
again, and it asks the question the previous dose could not - whether the edge's
maximum is approached gradually (still rising at 4.01x) or reached abruptly.

Two declared brackets, fixed before the run: the depth-3 edge lies strictly
between its 2.87x and 5.66x readings, and the depth-3 pool term is still positive
but below its 2.87x peak. These tests pin the cells, the bisection arithmetic, the
fresh blocks, the bar, the brackets' *premises* (recomputed from the committed
neighbours, so a drifting ladder invalidates the prediction rather than quietly
becoming a guess) and the branch table.
"""

from __future__ import annotations

import json
import math
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


def _mean(name: str, left: str, right: str) -> float:
    rows = json.loads((SUMMARIES / name).read_text())["per_seed"]
    values = [r["losses"][left] - r["losses"][right] for r in rows]
    return sum(values) / len(values)


def test_the_two_cells_change_exactly_one_gene_each() -> None:
    assert study.DOSE4_PROFILES == ("width-d3-dose4", "width-d4-dose4")
    assert study.PROFILES[study.DOSE4_BELOW]["overrides"] == {
        "depth": 3,
        "hidden_size": 57,
    }
    assert study.PROFILES[study.DOSE4_ABOVE]["overrides"] == {
        "depth": 4,
        "hidden_size": 57,
    }
    assert study.REGISTERED["steps"] == 48
    assert study.REGISTERED["mutation_interval"] == 12
    assert study.REGISTERED["adaptive_policy"] == "probe_loss"


def test_the_second_bisection_lands_on_the_narrower_windows_midpoint() -> None:
    """Width 57 is the new window's midpoint, not a round number picked by hand."""
    onset = _parameters(4, 24)
    below = _parameters(3, 57)
    above = _parameters(4, 57)
    assert (below, above) == (123_006, 163_134)
    assert below / onset == pytest.approx(4.01, abs=0.01)
    assert above / onset == pytest.approx(5.32, abs=0.01)
    # The window this bisects is the one 5.60 left: 88,032 (closed) to 173,672.
    midpoint = math.sqrt(88_032 * 173_672)
    assert below / midpoint == pytest.approx(1.0, abs=0.01), (below, midpoint)
    # Its multiplier column is what the report's table quotes.
    rows = {
        ("depth3", 3, 24): 0.76,
        ("width-d3-equal", 3, 28): 1.02,
        ("width-d3-dose", 3, 48): 2.87,
        (study.DOSE4_BELOW, 3, 57): 4.01,
        (study.DOSE4_BELOW_UPPER, 3, 68): 5.66,
        ("width-d3-dose2", 3, 96): 11.15,
        ("depth4-prefix", 4, 24): 1.00,
        ("width-d4-equal", 4, 28): 1.34,
        ("width-d4-dose", 4, 48): 3.80,
        (study.DOSE4_ABOVE, 4, 57): 5.32,
        (study.DOSE3_ABOVE, 4, 68): 7.52,
        ("width-d4-dose2", 4, 96): 14.82,
    }
    for (name, depth, width), multiple in rows.items():
        expected: dict[str, int] = {"depth": depth}
        if width != study.REGISTERED["hidden_size"]:
            expected["hidden_size"] = width
        assert study.PROFILES[name]["overrides"] == expected, name
        assert _parameters(depth, width) / onset == pytest.approx(multiple, abs=0.005), name


def test_the_blocks_are_fresh_against_every_other_claim() -> None:
    used: set[int] = set()
    own = {Path(study.PROFILES[name]["summary"]).name for name in study.DOSE4_PROFILES}
    for name, spec in study.PROFILES.items():
        if name in study.DOSE4_PROFILES:
            continue
        used |= set(spec["seeds"])
    for path in SUMMARIES.glob("*.json"):
        if path.name in own:
            continue
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            used |= set(payload.get("seeds") or [])
    for name, first in ((study.DOSE4_BELOW, 98384), (study.DOSE4_ABOVE, 99384)):
        seeds = study.PROFILES[name]["seeds"]
        assert seeds == tuple(range(first, first + 576))
        assert not (set(seeds) & used), name


def test_the_bar_is_the_three_primary_level() -> None:
    for name in study.DOSE4_PROFILES:
        assert study.PROFILES[name]["bonferroni_divisor"] == 3.0
        assert study.PROFILES[name]["report_section"] == "5.61"
    assert _gate_min_wins(576, study.ALPHA / 3) == 315


def test_the_declared_brackets_are_their_neighbours_own_readings() -> None:
    """The brackets' premises, recomputed - so a drifting ladder cannot hide.

    Neither bracket is a free parameter: the edge bracket is its two neighbours'
    readings in order, and the pool bracket is zero to the 2.87x peak. If either
    committed number moves, the assertion fails here rather than turning the
    prediction into a guess.
    """
    edge_low, edge_high = study.DOSE4_EDGE_BRACKET
    assert edge_low == pytest.approx(_mean("width-d3-dose3-576.json", "adaptive", "random"), abs=1e-6)
    assert edge_high == pytest.approx(_mean("width-d3-dose-576.json", "adaptive", "random"), abs=1e-6)
    assert edge_low < edge_high, "the bracket is not the interval the ladder rises through"
    pool_low, pool_high = study.DOSE4_POOL_BRACKET
    assert pool_low == 0.0
    assert pool_high == pytest.approx(_mean("width-d3-dose-576.json", "random", "null"), abs=1e-6)
    # And the neighbours really do sandwich this dose, in the committed readings.
    assert _mean("width-d3-dose-576.json", "adaptive", "random") == pytest.approx(
        -0.243801, abs=1e-6
    )
    assert _mean("width-d3-dose3-576.json", "adaptive", "random") == pytest.approx(
        -0.359770, abs=1e-6
    )
    assert _mean("width-d3-dose-576.json", "random", "null") == pytest.approx(
        +0.305267, abs=1e-6
    )
    # The declaration that the gate is still shut at 4.01x is on the record.
    assert study.PROFILES[study.DOSE4_BELOW]["per_cell_primary"] is True


def test_the_branch_table_covers_all_three_primaries() -> None:
    import inspect

    source = inspect.getsource(study.dose4_verdict)
    for branch in (
        "(False, True, True)",
        "(True, True, True)",
        "(True, False, True)",
        "(False, False, True)",
        "(False, True, False)",
        "(True, True, False)",
        "(True, False, False)",
        "(False, False, False)",
    ):
        assert branch in source
    assert "the window narrows to (123,006, 173,672]" in source
    assert "the window narrows to (88,032, 123,006]" in source
    # A miss on either bracket has to name the regime, not only report False.
    assert "PREDICTION miss names the regime" in source
    assert "PREDICTION missed: the gate is open at this dose" in source


def test_the_run_revision_is_declared_before_the_first_shard() -> None:
    import re

    for name in study.DOSE4_PROFILES:
        revision = study.PROFILES[name]["run_revision"]
        assert revision is not None, f"{name} was not frozen before the run"
        assert re.fullmatch(r"[0-9a-f]{7}", revision), (name, revision)
