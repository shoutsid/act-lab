"""Section 5.69's registration: a dose inside the ratio window, before any seed.

Section 10.1's crossing rule is a ratio - ``|adaptive - random| / (random -
null)`` - and it was read off committed cells as an *interval* rather than a
threshold: every cell at 1.37 or above opens the gate (1.37, 1.53, 1.55, 1.64),
every cell below 1.10 leaves it shut (0.96, 0.88, 0.80, and 1.10 at 4.01x), and
nothing has ever been placed between them. This arm places one cell there -
depth 3, width 61, 140,422 parameters (4.58x) - interpolated between the 4.01x
and 5.66x rungs so the measured ratio lands near 1.25.

These tests pin the cell, the arithmetic behind both declared predictions, the
fresh block, the bar, and the four fixed branches, so the registration cannot
drift between now and the run.
"""

from __future__ import annotations

import json
import re
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
    return model.parameter_count


def _mean(name: str, left: str, right: str) -> float:
    rows = json.loads((SUMMARIES / name).read_text())["per_seed"]
    return sum(row["losses"][left] - row["losses"][right] for row in rows) / len(rows)


def _ratio(name: str) -> float:
    edge = abs(_mean(name, "adaptive", "random"))
    pool = _mean(name, "random", "null")
    return edge / pool


def test_the_cell_changes_only_width_at_the_below_onset_depth() -> None:
    assert study.DOSE5_PROFILE == "width-d3-dose5"
    spec = study.PROFILES[study.DOSE5_PROFILE]
    assert spec["overrides"] == {"depth": 3, "hidden_size": 61}
    assert spec["report_section"] == "5.69"
    assert spec["per_cell_primary"] is True
    # Every other registered knob is the family's, unchanged: this is a width
    # dose, not a retune.
    assert study.REGISTERED["steps"] == 48
    assert study.REGISTERED["mutation_interval"] == 12
    assert study.REGISTERED["mutation_budget"] == 3
    assert study.REGISTERED["adaptive_policy"] == "probe_loss"


def test_the_dose_is_the_interpolation_between_the_bracketing_rungs() -> None:
    """Width 61 sits between the two rungs, chosen by their geometry."""
    low = _parameters(3, 57)
    high = _parameters(3, 68)
    cell = _parameters(3, 61)
    assert (low, high, cell) == (123_006, 173_672, 140_422)
    assert low < cell < high
    onset = _parameters(4, 24)
    assert cell / onset == pytest.approx(4.58, abs=0.005)
    # The profile's neighbours are the committed cells the window is read off,
    # named rather than inlined, so a ladder drift cannot quietly move them.
    assert study.DOSE5_CLOSED == "width-d3-dose4"
    assert study.DOSE5_OPEN == "width-d3-dose3"
    assert study.DOSE5_BELOW == "width-d3-dose"


def test_the_ratio_prediction_is_the_neighbours_own_readings() -> None:
    """Both declared numbers are recomputed from the committed neighbours.

    The prediction is not a free parameter: the endpoints are the two committed
    rungs' measured ratios (1.096 at 123,006 and 1.552 at 173,672), the
    interpolation is linear in parameter count, and the band is that value plus
    or minus 0.10. If any committed number moves, this fails here rather than
    letting the prediction become a guess.
    """
    low_ratio = _ratio("width-d3-dose4-576.json")
    high_ratio = _ratio("width-d3-dose3-576.json")
    assert low_ratio == pytest.approx(1.0959, abs=1e-3)
    assert high_ratio == pytest.approx(1.5522, abs=1e-3)
    low_ratio < high_ratio

    fraction = (_parameters(3, 61) - _parameters(3, 57)) / (
        _parameters(3, 68) - _parameters(3, 57)
    )
    predicted = low_ratio + fraction * (high_ratio - low_ratio)
    assert predicted == pytest.approx(1.2528, abs=1e-3)
    low, high = study.DOSE5_RATIO_BAND
    assert (low, high) == (1.15, 1.35)
    assert low < predicted < high, "the declared band does not contain its own interpolation"
    # The window the dose is placed in is the one §10.1 left empty, and the
    # committed ladder really does bracket it: 1.10 shut, 1.37 open.
    assert _ratio("width-d3-dose-576.json") == pytest.approx(0.7988, abs=1e-3)
    assert study.DOSE5_GATE_OPEN_DECLARED is False


def test_the_block_is_fresh_against_every_other_claim() -> None:
    seeds = study.PROFILES[study.DOSE5_PROFILE]["seeds"]
    assert seeds == tuple(range(101000, 101576))
    used: set[int] = set()
    for name, spec in study.PROFILES.items():
        if name == study.DOSE5_PROFILE:
            continue
        used |= set(spec["seeds"])
    for path in SUMMARIES.glob("*.json"):
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and isinstance(payload.get("seeds"), list):
            used |= {seed for seed in payload["seeds"] if isinstance(seed, int)}
    assert not (set(seeds) & used), sorted(set(seeds) & used)[:5]


def test_the_bar_is_the_neighbours_level() -> None:
    """One primary, but the bar is held at 315/576, not weakened for a single test.

    A single-comparison level would be alpha = 0.05 (bar ~310) and would move
    the cell's bar below the 4.01x rung's - making the two incomparable. The
    registration states the level instead: the neighbours' alpha/3.
    """
    spec = study.PROFILES[study.DOSE5_PROFILE]
    assert spec["bonferroni_divisor"] == 3.0
    assert _gate_min_wins(576, study.ALPHA / 3) == 315
    assert _gate_min_wins(576, study.ALPHA) < 315


def test_the_four_branches_and_the_predictions_are_fixed() -> None:
    import inspect

    source = inspect.getsource(study.dose5_verdict)
    for branch in ("(False, True)", "(True, True)", "(False, False)", "(True, False)"):
        assert branch in source
    assert "gate shut inside the window" in source
    assert "gate open inside the window" in source
    assert "outside its declared band" in source
    assert "PREDICTION missed" in source


def test_the_run_revision_is_declared_before_the_first_shard() -> None:
    revision = study.PROFILES[study.DOSE5_PROFILE]["run_revision"]
    # ``None`` is the state a registration passes through between the code
    # landing and the freeze commit; the run may not start until it is a hash.
    assert re.fullmatch(r"[0-9a-f]{7}", revision), revision
