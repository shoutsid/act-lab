"""Section 5.57's fourth rung: the design pins, before any new seed.

5.55-5.56 turned the depth axis from a direction into a model: three points
above the gate with increments of `-0.0418`, `-0.0382`, `-0.0422`, and a
controller edge flat across all of them. A model that predicts a *number* is
testable in a way a direction is not, so this arm registers the prediction
before it runs and reports it either way - a forecast that is checked and misses
is more informative than one that is never asked.

These tests pin the cell, its block freshness, the one gene it changes, the pool
identity across the whole ladder, the bar, the bands themselves (so the
prediction cannot be widened after the fact), and the four branches.
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
from act.topology import create_genome, feasible_expert_mutations  # noqa: E402

SUMMARIES = ROOT / "artifacts" / "study-summaries"


def _genome(depth: int):
    return create_genome(
        0,
        hidden_size=study.REGISTERED["hidden_size"],
        vocab_size=study.REGISTERED["vocab_size"],
        depth=depth,
        ffn="moe",
        expert_count=study.REGISTERED["expert_count"],
        active_experts=study.REGISTERED["active_experts"],
    )


def _parameters(depth: int) -> tuple[int, int]:
    model = model_from_genome(_genome(depth), context_size=study.REGISTERED["sequence_length"])
    return model.parameter_count, model.active_parameter_count


def test_the_cell_changes_one_gene_and_only_one() -> None:
    spec = study.PROFILES[study.DEPTH7_PROFILE]
    assert spec["overrides"] == {"depth": 7}
    assert spec["report_section"] == "5.57"
    assert spec["per_cell_primary"] is True
    assert spec["bonferroni_divisor"] == 2.0
    assert len(spec["seeds"]) == 576
    assert spec["seeds"][0] == 93384
    assert spec["seeds"][-1] == 93959
    assert study.DEPTH7_REFERENCE == study.DEPTH6_PROFILE == "depth6-dose"
    reference = study.PROFILES[study.DEPTH7_REFERENCE]
    assert reference["overrides"] == {"depth": 6}
    assert set(spec["overrides"]) ^ set(reference["overrides"]) == set()


def test_the_prediction_bands_are_the_models_own_numbers() -> None:
    """The bands are the measured increments' own range plus a flat margin.

    Pinning them is the point: a band that can be widened after the measurement
    is not a prediction. The three measured increments above the gate are
    `-0.0418`, `-0.0382`, `-0.0422`, so the model's forecast is ~`0.04` per layer
    and the declared band is that range widened by exactly `0.005` at each end.
    """
    assert study.DEPTH7_GRADIENT_BAND == (-0.047, -0.033)
    assert study.DEPTH7_EDGE_BAND == 0.020
    low, high = study.DEPTH7_GRADIENT_BAND
    increments = (-0.0418, -0.0382, -0.0422)
    # The band is the increments' range plus the declared margin, and no wider.
    assert low == pytest.approx(min(increments) - 0.005, abs=5e-4)
    assert high == pytest.approx(max(increments) + 0.005, abs=5e-4)
    assert low < min(increments) and max(increments) < high
    # The edge band is declared against the depth-6 reading, not as a free tolerance.
    assert study.DEPTH7_EDGE_BAND == pytest.approx(0.020)


def test_the_candidate_pool_is_identical_across_the_whole_ladder() -> None:
    """Depths 3-7 see the same four candidates, so only the substrate changes."""
    cap = study.REGISTERED["probe_candidate_cap"]
    heads = {}
    for depth in (3, 4, 5, 6, 7):
        candidates = feasible_expert_mutations(_genome(depth))
        assert len(candidates) > cap, depth
        heads[depth] = [str(c) for c in candidates[:cap]]
    assert len({tuple(v) for v in heads.values()}) == 1, heads


def test_the_ladder_it_extends_is_the_one_the_model_was_fitted_to() -> None:
    rungs = (
        ("depth2", 2),
        ("depth3", 3),
        ("depth4-prefix", 4),
        (study.DEPTH_DOSE_PROFILE, 5),
        (study.DEPTH6_PROFILE, 6),
        (study.DEPTH7_PROFILE, 7),
    )
    for profile, depth in rungs:
        overrides = study.PROFILES[profile]["overrides"]
        assert overrides == {"depth": depth}, profile
    # Every rung adds exactly 7,392 total / 2,592 active parameters.
    ladder = [_parameters(d) for _, d in rungs]
    assert ladder == [
        (15_888, 6_288),
        (23_280, 8_880),
        (30_672, 11_472),
        (38_064, 14_064),
        (45_456, 16_656),
        (52_848, 19_248),
    ]
    for before, after in zip(ladder, ladder[1:]):
        assert after[0] - before[0] == 7_392
        assert after[1] - before[1] == 2_592


def test_the_block_is_fresh() -> None:
    block = set(study.PROFILES[study.DEPTH7_PROFILE]["seeds"])
    assert block == set(range(93384, 93960))
    used: set[int] = set()
    for profile, spec in study.PROFILES.items():
        if profile == study.DEPTH7_PROFILE:
            continue
        used |= set(spec["seeds"])
    own = Path(study.PROFILES[study.DEPTH7_PROFILE]["summary"]).name
    for path in sorted(SUMMARIES.glob("*.json")):
        if path.name == own:
            continue
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            continue
        seeds = payload.get("seeds")
        if isinstance(seeds, list):
            used |= {seed for seed in seeds if isinstance(seed, int)}
    assert not (block & used), f"the block overlaps an earlier study: {sorted(block & used)[:5]}"


def test_the_bar_is_the_two_primary_level() -> None:
    assert _gate_min_wins(576, study.ALPHA / 2) == 313
    assert float(study.PROFILES[study.DEPTH7_PROFILE]["bonferroni_divisor"]) == 2.0


def test_the_gradient_the_model_predicts_is_what_five_56_measured() -> None:
    """The three increments behind the forecast, recomputed from the summaries."""
    def value(name: str) -> float:
        rows = json.loads((SUMMARIES / name).read_text())["per_seed"]
        return sum(r["losses"]["adaptive"] - r["losses"]["null"] for r in rows) / len(rows)

    ladder = [
        value(name)
        for name in (
            "null-control-depth4-prefix-384.json",
            "depth5-dose-576.json",
            "depth6-dose-576.json",
        )
    ]
    # The two increments this ladder covers, plus the 3 -> 4 step from 5.55's
    # record; the band was declared from all three.
    increments = [b - a for a, b in zip(ladder, ladder[1:])] + [-0.0418]
    assert increments[:2] == pytest.approx([-0.0382, -0.0422], abs=1e-3)
    low, high = study.DEPTH7_GRADIENT_BAND
    assert all(low <= step <= high for step in increments), increments


def _experiment(seed: int, value: float, collapse: float) -> dict:
    return {
        "seed": seed,
        "losses": {"random": 10.0 - collapse, "adaptive": 10.0 + value, "null": 10.0},
        "mutations": {"random": 3, "adaptive": 3, "null": 0},
        "search_flops": {"random": 1.0, "adaptive": 1.0, "null": 0.0},
    }


def _all(value: float, collapse: float = 0.0) -> list[tuple[float, float]]:
    return [(value, collapse)] * 100


def _write_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    per_seed: dict[str, list[tuple[float, float]]],
) -> None:
    seeds = tuple(range(94000, 94100))
    for profile in (
        study.DEPTH7_PROFILE,
        study.DEPTH7_REFERENCE,
        "depth2",
        "depth3",
        study.DEPTH_DOSE_REFERENCE,
        study.DEPTH6_REFERENCE,
    ):
        per_seed.setdefault(profile, _all(0.0))
        monkeypatch.setitem(
            study.PROFILES, profile, {**study.PROFILES[profile], "seeds": seeds}
        )
        directory = tmp_path / profile
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "shard": 0,
            "shards": 1,
            "profile": profile,
            "experiments": [
                _experiment(seed, value, collapse)
                for seed, (value, collapse) in zip(seeds, per_seed[profile])
            ],
        }
        (directory / "shard00.json").write_text(json.dumps(payload) + "\n")
    monkeypatch.setattr(study, "SHARD_DIR", tmp_path)


@pytest.mark.parametrize(
    ("value", "deeper", "branch"),
    [
        ("better", "better", "the ladder is monotone across all four rungs above the onset"),
        ("better", "same", "the axis saturates at depth 6"),
        ("worse", "better", "a fragile upper reading"),
        ("worse", "same", "the ladder is not monotone past depth 6"),
    ],
)
def test_the_four_branches_are_fixed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    value: str,
    deeper: str,
    branch: str,
) -> None:
    value_delta = -0.05 if value == "better" else +0.05
    reference_delta = (value_delta + 0.1) if deeper == "better" else value_delta
    _write_cells(
        tmp_path,
        monkeypatch,
        {
            study.DEPTH7_PROFILE: _all(value_delta),
            study.DEPTH7_REFERENCE: _all(reference_delta),
        },
    )
    assert study.depth7_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    assert branch in out
    # The prediction is reported whichever way the primaries land, hit or miss.
    assert "PREDICTION (declared before the run, descriptive): constant-gradient" in out
    assert "hit: " in out


def test_the_prediction_is_reported_as_a_miss_when_the_increment_is_outside_the_band(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """A forecast that cannot miss is not a forecast."""
    _write_cells(
        tmp_path,
        monkeypatch,
        {
            study.DEPTH7_PROFILE: _all(-0.05),
            study.DEPTH7_REFERENCE: _all(0.0),
        },
    )
    assert study.depth7_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    line = next(item for item in out.splitlines() if item.startswith("PREDICTION (declared"))
    assert line.endswith("hit: False"), line


def test_the_verdict_withholds_a_cell_whose_shards_are_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_cells(tmp_path, monkeypatch, {study.DEPTH7_PROFILE: _all(-0.05)})
    (tmp_path / study.DEPTH7_PROFILE / "shard00.json").unlink()
    with pytest.raises(SystemExit) as excinfo:
        study.depth7_verdict(1, study.ALPHA)
    assert "missing shard" in str(excinfo.value)
