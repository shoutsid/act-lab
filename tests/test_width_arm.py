"""Section 5.48's width arm: the design pins, before any fresh seed is touched.

The arm asks open question (ii) - is the depth onset *depth*, or *capacity* -
by extending the 5.15 dose-response with the width covariate: two fresh cells
at width 28, the closest integer width whose parameter count equals the
committed above-onset cell's. These tests pin the design (cells, blocks,
width choice), the test's bar and power, and the four pre-registered branches -
including the conjunction that a win-rate crossing alone cannot count as value
creation. The results are pinned separately, from the committed summaries.
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

REGISTERED_CELLS = {"width-d3-equal": 3, "width-d4-equal": 4}


def _parameters(depth: int, hidden: int) -> tuple[int, int]:
    genome = create_genome(
        0,
        hidden_size=hidden,
        vocab_size=study.REGISTERED["vocab_size"],
        depth=depth,
        ffn="moe",
        expert_count=study.REGISTERED["expert_count"],
        active_experts=study.REGISTERED["active_experts"],
    )
    model = model_from_genome(genome, context_size=study.REGISTERED["sequence_length"])
    return model.parameter_count, model.active_parameter_count


def test_the_width_cells_are_the_pre_registered_ones() -> None:
    """Fresh blocks 76000/77000, width 28, the dose-response's own config."""
    for profile, depth in REGISTERED_CELLS.items():
        spec = study.PROFILES[profile]
        assert spec["overrides"] == {"depth": depth, "hidden_size": 28}
        assert spec["report_section"] == "5.48"
        assert len(spec["seeds"]) == 576
    assert study.PROFILES["width-d3-equal"]["seeds"][0] == 76000
    assert study.PROFILES["width-d4-equal"]["seeds"][0] == 77000
    # And the arm's two cells are the ones the branch logic reads, in order.
    assert study.WIDTH_PROFILES == ("width-d3-equal", "width-d4-equal")
    # The registered config is otherwise untouched: same width (24) for every
    # committed cell, same policy, same arms, same events. The treatment is
    # the substrate's width and nothing else - so if either cell's override
    # loses its width, this fails rather than silently re-running the
    # committed experiment under a new report section.
    assert study.REGISTERED["hidden_size"] == 24
    assert study.REGISTERED["adaptive_policy"] == "probe_loss"
    assert study.REGISTERED["mutation_interval"] == 12
    assert study.REGISTERED["mutation_budget"] == 3
    assert study.REGISTERED["probe_candidate_cap"] == 4


def test_width_28_is_the_equal_parameter_partner_of_depth_4_width_24() -> None:
    """The width is solved from the model's parameter count, not tuned.

    `params(4, 24)` is the committed above-onset cell; width 28 at depth 3 is
    the closest integer width to its total parameter count (the arm's
    equal-parameter contrast). If the model's arithmetic or the registered
    width changes, the design no longer tests what it says it tests.
    """
    target_total, target_active = _parameters(4, 24)
    assert (target_total, target_active) == (30_672, 11_472)
    chosen_total, chosen_active = _parameters(3, 28)
    assert (chosen_total, chosen_active) == (31_192, 11_704)
    # Within 2% of the above-onset cell on both axes, and the closest integer
    # width to it in the count that matters:
    assert abs(chosen_total - target_total) / target_total < 0.02
    assert abs(chosen_active - target_active) / target_active < 0.021
    for width in range(20, 65):
        if width == 28:
            continue
        total, _ = _parameters(3, width)
        assert abs(total - target_total) > abs(chosen_total - target_total)


def test_the_width_arm_bar_is_the_dose_responses_bar() -> None:
    """576 seeds per cell: the committed onset rate is detectable at 0.98.

    The primary is the dose-response's own test - a one-sided exact sign test
    of adaptive-vs-null per-seed wins at the Bonferroni bar alpha/2 = `0.025`
    (alpha 0.05 with two primaries), which is the level every executed sibling
    cell applied and the level the committed summaries record. Section 5.16.1
    is the reason the number is statement *once* here and read from the same
    `_gate_min_wins` the verdict calls: a declaration whose bar and alpha
disagree cannot be applied as written.

    The seed count was sized to the *committed* above-onset rate (225/384 =
    0.5859): at n = 576 the bar is 313/576 and the exact power to detect that
    rate is 0.982 (0.801 at 0.56, 0.641 at 0.55, 0.13 at 0.52). A smaller true
    effect will read as a null *by design*, which is why the number is pinned.
    """
    assert _gate_min_wins(576, 0.025) == 313
    assert _gate_min_wins(384, 0.025) == 212  # the committed cells' applied bar
    # The committed above-onset cell (225/384 = 0.5859) would land ~337 wins
    # at n = 576, above the bar - and the committed below-onset cell
    # (188/384 = 0.4896) would land ~282, below it.
    assert 0.5859 * 576 > 313
    assert 0.4896 * 576 < 313


def _experiment(seed: int, delta: float) -> dict:
    """One synthetic per-seed payload in the study's committed schema."""
    return {
        "seed": seed,
        "losses": {"random": 10.0, "adaptive": 10.0 + delta, "null": 10.0},
        "mutations": {"random": 3, "adaptive": 3, "null": 0},
        "search_flops": {"random": 1.0, "adaptive": 1.0, "null": 0},
    }


def _write_cells(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deltas: dict) -> None:
    """Install synthetic shards for both cells under a monkeypatched root."""
    seeds = tuple(range(90000, 90100))
    for profile in study.WIDTH_PROFILES:
        monkeypatch.setitem(
            study.PROFILES,
            profile,
            {**study.PROFILES[profile], "seeds": seeds},
        )
        directory = tmp_path / profile
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "shard": 0,
            "shards": 1,
            "profile": profile,
            "experiments": [
                _experiment(seed, delta)
                for seed, delta in zip(seeds, deltas[profile])
            ],
        }
        (directory / "shard00.json").write_text(json.dumps(payload) + "\n")
    monkeypatch.setattr(study, "SHARD_DIR", tmp_path)


def _all(delta: float) -> list[float]:
    return [delta] * 100


def test_a_win_rate_crossing_alone_is_not_value_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The pre-registered conjunction: wins clear the bar AND the mean is negative.

    Constructed failure for the tail guard: 100 seeds with exactly the bar's
    61 wins, each a small improvement, and 39 seeds with a large degradation
    in the adaptive arm's disfavour. The win rate clears the bar while the
    mean delta is positive - the depth-1 pattern at larger n (117/384 wins,
    +0.0508 mean) - and it must not read as value creation.
    """
    tau = [(-0.01)] * 61 + [(+0.5)] * 39
    _write_cells(
        tmp_path,
        monkeypatch,
        {
            "width-d3-equal": tau,
            "width-d4-equal": _all(-0.05),
        },
    )
    assert study.width_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    assert "width-d3-equal: adaptive-vs-null wins 61/100 (bar 61)" in out
    assert "creates value: False" in out
    assert "P2 only" in out


@pytest.mark.parametrize(
    ("first", "second", "branch"),
    [
        ("better", "better", "the onset is a capacity onset"),
        (
            "better",
            "worse",
            "widening creates value below the onset and destroys it above",
        ),
        ("worse", "better", "the onset is depth-like at this width"),
        ("worse", "worse", "width is harmful at both depths here"),
    ],
)
def test_the_four_branches_are_fixed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    first: str,
    second: str,
    branch: str,
) -> None:
    cells = {"better": _all(-0.05), "worse": _all(+0.05)}
    _write_cells(
        tmp_path,
        monkeypatch,
        {"width-d3-equal": cells[first], "width-d4-equal": cells[second]},
    )
    assert study.width_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    assert branch in out
