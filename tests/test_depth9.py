"""Section 5.59's sixth rung: the design pins, before any new seed.

Two registered forecasts have now missed from opposite sides of the same series -
the flat model at the fourth increment (`-0.063548` against `(-0.047, -0.033)`) and
the accelerating model at the fifth (`-0.054124` against `(-0.069, -0.059)`). That
leaves **three** readings rather than two: flat, the mild rise the last two
increments show, and something steeper than the fourth. This arm declares the
middle one, and the band it declares is constructed so both excluded readings are a
miss - which is what makes the prediction a regime check rather than a restatement.

These tests pin the cell, its block freshness, the one gene it changes, the pool
identity across the whole ladder, the bar, the three-way band, and the four branches
plus the regime sentence a miss prints.
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
    (study.DEPTH9_PROFILE, 9),
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
    spec = study.PROFILES[study.DEPTH9_PROFILE]
    assert spec["overrides"] == {"depth": 9}
    assert spec["report_section"] == "5.59"
    # The revision the summary cites: the registration commit, frozen before the
    # shards run.
    assert spec["run_revision"] == "1816e34"
    assert spec["per_cell_primary"] is True
    assert spec["bonferroni_divisor"] == 2.0
    assert len(spec["seeds"]) == 576
    assert spec["seeds"][0] == 95384
    assert spec["seeds"][-1] == 95959
    assert study.DEPTH9_REFERENCE == study.DEPTH8_PROFILE == "depth8-dose"
    reference = study.PROFILES[study.DEPTH9_REFERENCE]
    assert reference["overrides"] == {"depth": 8}
    assert set(spec["overrides"]) ^ set(reference["overrides"]) == set()


def test_the_regime_band_excludes_both_refuted_readings() -> None:
    """The band is the last two increments, and both alternatives fall outside.

    A three-way declaration is only a declaration if the readings it excludes
    would miss: the flat `0.04` level sits above the band and a step at or below
    the fourth (`0.0635`) sits below it, so either miss names a regime.
    """
    fourth = _value("depth7-dose-576.json") - _value("depth6-dose-576.json")
    fifth = _value("depth8-dose-576.json") - _value("depth7-dose-576.json")
    assert fourth == pytest.approx(-0.063548, abs=1e-6)
    assert fifth == pytest.approx(-0.054124, abs=1e-6)
    low, high = study.DEPTH9_GRADIENT_BAND
    assert study.DEPTH9_GRADIENT_BAND == (-0.069, -0.049)
    assert low == round(fourth - 0.005, 3)
    assert high == round(fifth + 0.005, 3)
    # The three readings, as numbers rather than prose. The flat level sits above
    # the band, a step steeper than the fourth by more than the declared margin sits
    # below it, and the fourth itself sits inside - which is what the margin means.
    flat = -0.0407
    assert flat > high, "the flat level is inside the band, so a hit would not exclude it"
    assert fourth - 0.02 < low, "a steeper step cannot miss on the low side"
    assert low < fourth < high
    assert study.DEPTH9_EDGE_BAND == 0.020


def test_the_candidate_pool_is_identical_across_the_whole_ladder() -> None:
    """Depths 3-9 see the same four candidates, so only the substrate changes."""
    cap = study.REGISTERED["probe_candidate_cap"]
    heads = {}
    for depth in range(3, 10):
        candidates = feasible_expert_mutations(_genome(depth))
        assert len(candidates) > cap, depth
        heads[depth] = [str(c) for c in candidates[:cap]]
    assert len({tuple(v) for v in heads.values()}) == 1, heads
    assert sum(len(feasible_expert_mutations(_genome(d))) for d in range(3, 10)) == 210


def test_the_ladder_it_extends_is_the_one_the_fifth_rung_measured() -> None:
    for profile, depth in LADDER:
        assert study.PROFILES[profile]["overrides"] == {"depth": depth}, profile
    assert study.REGISTERED["steps"] == 48
    assert study.REGISTERED["mutation_interval"] == 12
    assert study.REGISTERED["hidden_size"] == 24
    assert study.REGISTERED["expert_count"] == 4
    ladder = [_parameters(depth) for _, depth in LADDER]
    assert ladder[-2:] == [(60_240, 21_840), (67_632, 24_432)]
    for before, after in zip(ladder, ladder[1:]):
        assert after[0] - before[0] == 7_392
        assert after[1] - before[1] == 2_592


def test_the_block_is_fresh() -> None:
    block = set(study.PROFILES[study.DEPTH9_PROFILE]["seeds"])
    assert block == set(range(95384, 95960))
    used: set[int] = set()
    for profile, spec in study.PROFILES.items():
        if profile == study.DEPTH9_PROFILE:
            continue
        used |= set(spec["seeds"])
    own = Path(study.PROFILES[study.DEPTH9_PROFILE]["summary"]).name
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
    assert float(study.PROFILES[study.DEPTH9_PROFILE]["bonferroni_divisor"]) == 2.0


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
    seeds = tuple(range(95384, 95484))
    for profile in (
        study.DEPTH9_PROFILE,
        study.DEPTH9_REFERENCE,
        "depth2",
        "depth3",
        study.DEPTH_DOSE_REFERENCE,
        study.DEPTH6_REFERENCE,
        study.DEPTH7_REFERENCE,
        study.DEPTH8_REFERENCE,
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
        ("better", "better", "the ladder is monotone across all six rungs above the onset"),
        ("better", "same", "the axis saturates at depth 8"),
        ("worse", "better", "a fragile upper reading"),
        ("worse", "same", "the ladder is not monotone past depth 8"),
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
            study.DEPTH9_PROFILE: _all(value_delta),
            study.DEPTH9_REFERENCE: _all(reference_delta),
        },
    )
    assert study.depth9_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    assert branch in out
    assert "PREDICTION (declared before the run, descriptive): mild-rise" in out


@pytest.mark.parametrize(
    ("increment", "regime"),
    [
        (-0.04, "flat, or shallower than the flat reading"),
        (-0.08, "steeper than the fourth increment"),
        # The crack between the two declared bands: steeper than the flat model's
        # own band allows (`-0.047`) and shallower than this arm's requires
        # (`-0.049`). Calling that "flat" would be the verdict claiming more than
        # the number says, and the arm's own measurement landed here.
        (-0.0487, "between the flat reading's band and the mild-rise band"),
    ],
)
def test_a_miss_names_the_regime_it_landed_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    increment: float,
    regime: str,
) -> None:
    """Both excluded readings are live, and each miss says which one it is."""
    _write_cells(
        tmp_path,
        monkeypatch,
        {
            study.DEPTH9_PROFILE: _all(increment),
            study.DEPTH9_REFERENCE: _all(0.0),
        },
    )
    assert study.depth9_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    line = next(item for item in out.splitlines() if item.startswith("PREDICTION (declared"))
    assert line.endswith("hit: False"), line
    assert f"PREDICTION miss names the regime: the increment is {regime}" in out


def test_the_verdict_withholds_a_cell_whose_shards_are_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_cells(tmp_path, monkeypatch, {study.DEPTH9_PROFILE: _all(-0.05)})
    (tmp_path / study.DEPTH9_PROFILE / "shard00.json").unlink()
    with pytest.raises(SystemExit) as excinfo:
        study.depth9_verdict(1, study.ALPHA)
    assert "missing shard" in str(excinfo.value)
