"""Section 5.58's fifth rung: the design pins, before any new seed.

5.57 asked the constant-gradient model for a number and the number came back
**1.5x too small** - `-0.063548` against a declared `(-0.047, -0.033)`. One point
past a flat series cannot separate an acceleration from a wobble, so this arm
asks the same question with the fourth increment itself as the model. That makes
its declaration a *choice between two readings* rather than a fresh guess: the
band below is the fourth increment with the same flat margin, and the flat-`0.04`
reading it replaces sits a full band away.

These tests pin the cell, its block freshness, the one gene it changes, the pool
identity across the whole ladder, the bar, the two bands (so a prediction cannot
be widened after the fact), the measured increment the band is declared from, and
the four branches.
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

LADDER = (
    ("depth2", 2),
    ("depth3", 3),
    ("depth4-prefix", 4),
    (study.DEPTH_DOSE_PROFILE, 5),
    (study.DEPTH6_PROFILE, 6),
    (study.DEPTH7_PROFILE, 7),
    (study.DEPTH8_PROFILE, 8),
)


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


def _value(name: str) -> float:
    rows = json.loads((SUMMARIES / name).read_text())["per_seed"]
    return sum(r["losses"]["adaptive"] - r["losses"]["null"] for r in rows) / len(rows)


def test_the_cell_changes_one_gene_and_only_one() -> None:
    spec = study.PROFILES[study.DEPTH8_PROFILE]
    assert spec["overrides"] == {"depth": 8}
    assert spec["report_section"] == "5.58"
    # Registered but not yet frozen: the revision is set from the commit the
    # shards actually run at, and ``None`` is what keeps the summary tests honest
    # while the arm is only declared.
    assert spec["run_revision"] is None
    assert spec["per_cell_primary"] is True
    assert spec["bonferroni_divisor"] == 2.0
    assert len(spec["seeds"]) == 576
    assert spec["seeds"][0] == 94384
    assert spec["seeds"][-1] == 94959
    assert study.DEPTH8_REFERENCE == study.DEPTH7_PROFILE == "depth7-dose"
    reference = study.PROFILES[study.DEPTH8_REFERENCE]
    assert reference["overrides"] == {"depth": 7}
    assert set(spec["overrides"]) ^ set(reference["overrides"]) == set()


def test_the_prediction_is_declared_from_the_measured_increment() -> None:
    """The band is the fourth increment plus a flat margin - and a real choice.

    `-0.0635` is what 5.57 measured, so this arm's forecast is that increment
    repeated rather than the flat `0.04` the model predicted first. Both readings
    are checked here: the band contains the fourth increment and **excludes** the
    three increments before it, which is what makes this a test of acceleration
    rather than another restatement of "deeper is better".
    """
    increment = _value("depth7-dose-576.json") - _value("depth6-dose-576.json")
    assert increment == pytest.approx(-0.0635, abs=5e-4)
    low, high = study.DEPTH8_GRADIENT_BAND
    assert study.DEPTH8_GRADIENT_BAND == (-0.069, -0.059)
    # The declared band is the measured increment widened by a flat 0.005 and
    # rounded to the milli-nat the declaration states, so the check is exact
    # rather than approximate: a band that moved by one unit fails here.
    assert low == round(increment - 0.005, 3)
    assert high == round(increment + 0.005, 3)
    # The flat series the first model predicted, recomputed, falls outside.
    flat = (
        _value("depth5-dose-576.json") - _value("null-control-depth4-prefix-384.json"),
        _value("depth6-dose-576.json") - _value("depth5-dose-576.json"),
        increment,
    )
    assert flat[0] == pytest.approx(-0.0382, abs=1e-3)
    assert flat[1] == pytest.approx(-0.0422, abs=1e-3)
    assert not any(low <= step <= high for step in flat[:2]), flat
    assert study.DEPTH8_EDGE_BAND == 0.020


def test_the_candidate_pool_is_identical_across_the_whole_ladder() -> None:
    """Depths 3-8 see the same four candidates, so only the substrate changes."""
    cap = study.REGISTERED["probe_candidate_cap"]
    heads = {}
    for depth in range(3, 9):
        candidates = feasible_expert_mutations(_genome(depth))
        assert len(candidates) > cap, depth
        heads[depth] = [str(c) for c in candidates[:cap]]
    assert len({tuple(v) for v in heads.values()}) == 1, heads
    assert sum(len(feasible_expert_mutations(_genome(d))) for d in range(3, 9)) == 165


def test_the_ladder_it_extends_is_the_one_the_fourth_rung_measured() -> None:
    for profile, depth in LADDER:
        assert study.PROFILES[profile]["overrides"] == {"depth": depth}, profile
    assert study.REGISTERED["steps"] == 48
    assert study.REGISTERED["mutation_interval"] == 12
    assert study.REGISTERED["hidden_size"] == 24
    assert study.REGISTERED["expert_count"] == 4
    ladder = [_parameters(depth) for _, depth in LADDER]
    assert ladder == [
        (15_888, 6_288),
        (23_280, 8_880),
        (30_672, 11_472),
        (38_064, 14_064),
        (45_456, 16_656),
        (52_848, 19_248),
        (60_240, 21_840),
    ]
    for before, after in zip(ladder, ladder[1:]):
        assert after[0] - before[0] == 7_392
        assert after[1] - before[1] == 2_592


def test_the_block_is_fresh() -> None:
    block = set(study.PROFILES[study.DEPTH8_PROFILE]["seeds"])
    assert block == set(range(94384, 94960))
    used: set[int] = set()
    for profile, spec in study.PROFILES.items():
        if profile == study.DEPTH8_PROFILE:
            continue
        used |= set(spec["seeds"])
    own = Path(study.PROFILES[study.DEPTH8_PROFILE]["summary"]).name
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
    assert float(study.PROFILES[study.DEPTH8_PROFILE]["bonferroni_divisor"]) == 2.0


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
    seeds = tuple(range(94384, 94484))
    for profile in (
        study.DEPTH8_PROFILE,
        study.DEPTH8_REFERENCE,
        "depth2",
        "depth3",
        study.DEPTH_DOSE_REFERENCE,
        study.DEPTH6_REFERENCE,
        study.DEPTH7_REFERENCE,
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
        ("better", "better", "the ladder is monotone across all five rungs above the onset"),
        ("better", "same", "the axis saturates at depth 7"),
        ("worse", "better", "a fragile upper reading"),
        ("worse", "same", "the ladder is not monotone past depth 7"),
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
            study.DEPTH8_PROFILE: _all(value_delta),
            study.DEPTH8_REFERENCE: _all(reference_delta),
        },
    )
    assert study.depth8_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    assert branch in out
    # The prediction is reported whichever way the primaries land, hit or miss.
    assert "PREDICTION (declared before the run, descriptive): accelerating" in out
    assert "hit: " in out


def test_the_prediction_is_reported_as_a_miss_when_the_increment_is_flat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """The reading this band was declared *against* has to be a miss.

    A band that swallowed the flat `0.04` increment too would not distinguish the
    two models at all, so the flat case is asserted to report `hit: False`.
    """
    _write_cells(
        tmp_path,
        monkeypatch,
        {
            study.DEPTH8_PROFILE: _all(-0.04),
            study.DEPTH8_REFERENCE: _all(0.0),
        },
    )
    assert study.depth8_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    line = next(item for item in out.splitlines() if item.startswith("PREDICTION (declared"))
    assert line.endswith("hit: False"), line


def test_the_verdict_withholds_a_cell_whose_shards_are_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_cells(tmp_path, monkeypatch, {study.DEPTH8_PROFILE: _all(-0.05)})
    (tmp_path / study.DEPTH8_PROFILE / "shard00.json").unlink()
    with pytest.raises(SystemExit) as excinfo:
        study.depth8_verdict(1, study.ALPHA)
    assert "missing shard" in str(excinfo.value)
