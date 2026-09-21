"""Section 5.54's total-compute arm: the design pins, before any new seed.

Section 10.3(ii) named "parameter count, layer width, expert count" as the
covariates that could separate depth from capacity, and 5.48/5.53 measured the
last two. The remaining candidate in that list is the budget itself, and this arm
runs it in the only corner of the depth x budget 2x2 at width 24 that is not yet
measured: depth 3 at 192 steps. Three corners are committed (`depth3` at 48 fails
the bar, `depth4-prefix` at 48 and `width-d4-long-w24` at 192 both create value).

These tests pin the cell, its block freshness, the fact that only the budget
changes from the committed same-depth cell, the bar, the rank-sum primary's
correctness, its rehearsed sensitivity on the one measured regime flip, and the
four fixed branches. Results are pinned separately, from the committed summary.
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
# The committed corner of the 2x2 this arm is read against, and the measured
# regime flip the rank-sum primary was rehearsed on.
SAME_DEPTH = "null-control-depth3-384.json"
ABOVE_LONG = "width-d4-long-w24-576.json"
ABOVE_SHORT = "null-control-depth4-prefix-384.json"


def _parameters(depth: int) -> tuple[int, int]:
    genome = create_genome(
        0,
        hidden_size=study.REGISTERED["hidden_size"],
        vocab_size=study.REGISTERED["vocab_size"],
        depth=depth,
        ffn="moe",
        expert_count=study.REGISTERED["expert_count"],
        active_experts=study.REGISTERED["active_experts"],
    )
    model = model_from_genome(genome, context_size=study.REGISTERED["sequence_length"])
    return model.parameter_count, model.active_parameter_count


def test_the_cell_is_the_pre_registered_one() -> None:
    """192 steps, a 48-step interval, depth 3 - 5.52's config with depth changed."""
    spec = study.PROFILES[study.LONG_D3_PROFILE]
    assert spec["overrides"] == {"depth": 3, "steps": 192, "mutation_interval": 48}
    assert spec["report_section"] == "5.54"
    # The registration commit is the revision the summary cites, frozen before
    # the shards run so it cannot be chosen after the numbers are known.
    assert spec["run_revision"] == "64046da"
    assert spec["per_cell_primary"] is True
    assert spec["bonferroni_divisor"] == 2.0
    assert len(spec["seeds"]) == 576
    assert spec["seeds"][0] == 90000
    # Compared against the three committed corners and nothing else.
    assert study.LONG_D3_SAME_DEPTH == "depth3"
    assert study.LONG_D3_ABOVE_LONG == "width-d4-long-w24"
    assert study.LONG_D3_ABOVE_SHORT == "depth4-prefix"
    # The committed same-depth cell is the 48-step one, so the *only* difference
    # is the budget - which is what makes this the compute covariate.
    base = study.PROFILES[study.LONG_D3_SAME_DEPTH]
    assert base["overrides"] == {"depth": 3}
    assert study.REGISTERED["steps"] == 48
    assert study.REGISTERED["mutation_interval"] == 12
    # Four times the steps and four times the interval is the same three events.
    assert spec["overrides"]["steps"] == 4 * study.REGISTERED["steps"]
    assert spec["overrides"]["mutation_interval"] == 4 * study.REGISTERED["mutation_interval"]
    assert study.REGISTERED["mutation_budget"] == 3
    assert study.REGISTERED["hidden_size"] == 24
    # Depth 3 and depth 4 keep their own parameter counts: this arm moves only
    # the budget, unlike 5.53 which moved the expert count by 31.9% total.
    assert _parameters(3) == (23_280, 8_880)
    assert _parameters(4) == (30_672, 11_472)


def test_the_block_is_fresh() -> None:
    """90000-90575 is free; the report calls it fresh, so check it."""
    block = set(study.PROFILES[study.LONG_D3_PROFILE]["seeds"])
    assert block == set(range(90000, 90576))
    used: set[int] = set()
    for profile, spec in study.PROFILES.items():
        if profile == study.LONG_D3_PROFILE:
            continue
        used |= set(spec["seeds"])
    own = Path(study.PROFILES[study.LONG_D3_PROFILE]["summary"]).name
    for path in sorted(SUMMARIES.glob("*.json")):
        if path.name == own:
            continue
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):  # some files are panels, not records
            continue
        seeds = payload.get("seeds")
        if isinstance(seeds, list):
            used |= {seed for seed in seeds if isinstance(seed, int)}
    assert not (block & used), f"the block overlaps an earlier study: {sorted(block & used)[:5]}"
    # And the three corners of the 2x2 are the ones the outcome table names.
    for name in (SAME_DEPTH, ABOVE_SHORT, ABOVE_LONG):
        assert (SUMMARIES / name).is_file(), name


def test_the_bar_is_the_two_primary_level() -> None:
    assert _gate_min_wins(576, study.ALPHA / 2) == 313
    assert float(study.PROFILES[study.LONG_D3_PROFILE]["bonferroni_divisor"]) == 2.0


def _deltas(name: str, left: str, right: str) -> list[float]:
    payload = json.loads((SUMMARIES / name).read_text())
    return [row["losses"][left] - row["losses"][right] for row in payload["per_seed"]]


def test_the_rank_sum_primary_was_rehearsed_on_the_measured_flip() -> None:
    """The regime test detects the budget effect where the records say one exists.

    At depth 4 the same 48 -> 192 comparison moves the collapse from `-0.0701` to
    `+0.1638`, and this is the test that has to see it - the arm's P2 is only
    informative if the ruler is not blind to the thing it is measuring. The
    direction is fixed here too: ``_rank_sum_one_sided(a, b)`` asks whether ``a``
    is stochastically smaller, so the *short* cell comes first for the claim
    "4x training makes the collapse larger".
    """
    short = _deltas(ABOVE_SHORT, "null", "random")
    long_ = _deltas(ABOVE_LONG, "null", "random")
    assert sum(short) / len(short) < sum(long_) / len(long_)
    _, p = study._rank_sum_one_sided(short, long_)
    assert p < 1e-30, "the regime test cannot see the one measured flip"
    # And the direction convention is what it is claimed to be: the same two
    # lists the other way round must read as a dead null (p near 1), not as a
    # second detection.
    _, reverse = study._rank_sum_one_sided(long_, short)
    assert reverse > 0.999


def _experiment(seed: int, value: float, collapse: float) -> dict:
    """One synthetic per-seed payload driving both primaries independently.

    ``value`` is ``adaptive - null`` and ``collapse`` is ``null - random``, so the
    two primaries can be set to any combination instead of co-varying with one
    another through a single loss triple.
    """
    return {
        "seed": seed,
        "losses": {"random": 10.0 - collapse, "adaptive": 10.0 + value, "null": 10.0},
        "mutations": {"random": 3, "adaptive": 3, "null": 0},
        "search_flops": {"random": 1.0, "adaptive": 1.0, "null": 0.0},
    }


def _write_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    per_seed: dict[str, list[tuple[float, float]]],
) -> None:
    """Install synthetic shards for the new cell and the three corners it prints."""
    seeds = tuple(range(92000, 92100))
    for profile in (
        study.LONG_D3_PROFILE,
        study.LONG_D3_SAME_DEPTH,
        study.LONG_D3_ABOVE_SHORT,
        study.LONG_D3_ABOVE_LONG,
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


def _all(value: float, collapse: float = 0.0) -> list[tuple[float, float]]:
    return [(value, collapse)] * 100


@pytest.mark.parametrize(
    ("value", "regime", "branch"),
    [
        ("better", "moved", "total compute is the covariate"),
        ("better", "still", "creates value below the onset without creating a churn harm"),
        ("worse", "moved", "the budget reaches the regime below the onset but not the gate"),
        ("worse", "still", "4x training changes nothing below the onset"),
    ],
)
def test_the_four_branches_are_fixed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    value: str,
    regime: str,
    branch: str,
) -> None:
    """Each of the four pre-registered readings is reachable and printed.

    ``value`` is the new cell's adaptive-vs-null mean; ``regime`` is whether its
    null-vs-random deltas are stochastically larger than the committed 48-step
    cell's - the two primaries, driven independently so every combination is
    exercised rather than only the one the real data happens to land on.
    """
    # adaptive - null: negative is "better" (the new cell creates value).
    value_delta = -0.05 if value == "better" else +0.05
    # null - random: the committed cell sits at 0.0 for every seed, so any
    # strictly positive shift on the new cell's side makes its deltas
    # stochastically larger - which is the direction P2 registers.
    shift = 0.4 if regime == "moved" else 0.0
    _write_cells(
        tmp_path,
        monkeypatch,
        {
            study.LONG_D3_PROFILE: _all(value_delta, shift),
            study.LONG_D3_SAME_DEPTH: _all(0.0, 0.0),
        },
    )
    assert study.long_d3_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    assert branch in out
    line = next(item for item in out.splitlines() if item.startswith(f"{study.LONG_D3_PROFILE}: "))
    assert f"creates value: {value == 'better'}" in line


def test_the_verdict_withholds_a_cell_whose_shards_are_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial shard set is refused, not scored on what happens to be there."""
    _write_cells(
        tmp_path,
        monkeypatch,
        {study.LONG_D3_PROFILE: _all(-0.05)},
    )
    (tmp_path / study.LONG_D3_PROFILE / "shard00.json").unlink()
    with pytest.raises(SystemExit) as excinfo:
        study.long_d3_verdict(1, study.ALPHA)
    assert "missing shard" in str(excinfo.value)
