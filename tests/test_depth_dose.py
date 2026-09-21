"""Section 5.55's depth-dose arm: the design pins, before any new seed.

The covariate programme closed with 5.54, and it closed on the axis it started
from: width (24 -> 28 -> 96), expert count (4 -> 6) and budget (48 -> 192) each
got a dose, and depth - the one covariate that actually moves the gate - has only
ever had its bracket measured (depths 2 and 3 fail, depth 4 passes). So "the gate
is depth per se" rests on a single step with nothing above its upper cell, and
this arm is that dose.

These tests pin the cell, its block freshness, the one gene it changes, the bar,
the rank-sum primary's correctness and its rehearsed sensitivity on a measured
instance of the same comparison shape, and the four fixed branches. Results are
pinned separately, from the committed summary.
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
REFERENCE = "null-control-depth4-prefix-384.json"
# The measured instance of P2's comparison shape: two independent cells at the
# same depth and width, one given 4x the budget.
SHAPE_SHORT = "null-control-depth4-prefix-384.json"
SHAPE_LONG = "width-d4-long-w24-576.json"


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


def test_the_cell_changes_one_gene_and_only_one() -> None:
    """`depth` 4 -> 5 against the registered config, and nothing else at all."""
    spec = study.PROFILES[study.DEPTH_DOSE_PROFILE]
    assert spec["overrides"] == {"depth": 5}
    assert spec["report_section"] == "5.55"
    assert spec["per_cell_primary"] is True
    assert spec["bonferroni_divisor"] == 2.0
    assert len(spec["seeds"]) == 576
    assert spec["seeds"][0] == 91384
    assert spec["seeds"][-1] == 91959
    assert study.DEPTH_DOSE_REFERENCE == "depth4-prefix"
    # The reference is the committed above-onset cell, and its overrides differ
    # from this arm's in exactly the one key.
    reference = study.PROFILES[study.DEPTH_DOSE_REFERENCE]
    assert reference["overrides"] == {"depth": 4}
    assert set(spec["overrides"]) ^ set(reference["overrides"]) == set()
    # Everything else is the registered config: this arm moves no width, no
    # expert count, no budget and no policy - the family's cleanest dose.
    assert reference["seeds"] == tuple(range(73000, 73384))


def test_the_depth_ladder_it_sits_on_is_the_bracket_being_dosed() -> None:
    """Below the onset at two depths, above it at one, and nothing above that."""
    ladder = {}
    for profile in ("depth2", "depth3", study.DEPTH_DOSE_REFERENCE):
        spec = study.PROFILES[profile]
        ladder[spec["overrides"]["depth"]] = profile
    assert ladder == {2: "depth2", 3: "depth3", 4: "depth4-prefix"}
    for profile in ladder.values():
        assert study.PROFILES[profile]["overrides"]["depth"] in (2, 3, 4)
        assert "steps" not in study.PROFILES[profile]["overrides"]
    # The committed ladder is 48 steps at the registered width, which is what
    # makes depth the only gene this arm moves.
    assert study.REGISTERED["steps"] == 48
    assert study.REGISTERED["hidden_size"] == 24
    assert study.REGISTERED["expert_count"] == 4
    # A fifth layer is 38,064 total / 14,064 active parameters - deliberately
    # unmatched: a layer's parameters are irreducible, and 5.48/5.53 already
    # measured that buying parameters with width or experts does not move the
    # gate, so this reads the depth axis as a package.
    assert _parameters(5) == (38_064, 14_064)
    assert _parameters(4) == (30_672, 11_472)
    assert _parameters(5)[0] > _parameters(4)[0]


def test_the_block_is_fresh() -> None:
    """91384-91959 is free; the report calls it fresh, so check it."""
    block = set(study.PROFILES[study.DEPTH_DOSE_PROFILE]["seeds"])
    assert block == set(range(91384, 91960))
    used: set[int] = set()
    for profile, spec in study.PROFILES.items():
        if profile == study.DEPTH_DOSE_PROFILE:
            continue
        used |= set(spec["seeds"])
    own = Path(study.PROFILES[study.DEPTH_DOSE_PROFILE]["summary"]).name
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
    assert (SUMMARIES / REFERENCE).is_file()


def test_the_bar_is_the_two_primary_level() -> None:
    assert _gate_min_wins(576, study.ALPHA / 2) == 313
    assert float(study.PROFILES[study.DEPTH_DOSE_PROFILE]["bonferroni_divisor"]) == 2.0


def _deltas(name: str, left: str, right: str) -> list[float]:
    payload = json.loads((SUMMARIES / name).read_text())
    return [row["losses"][left] - row["losses"][right] for row in payload["per_seed"]]


def test_the_rank_sum_primary_was_rehearsed_on_a_measured_shaped_instance() -> None:
    """The ruler sees a substrate that creates more value where one is measured to.

    P2's shape is "two independent cells of this family, one substrate strictly
    better, is the better one stochastically better on the value deltas". The
    measured instance of that shape is same-depth, same-width and 4x the budget:
    `depth4-prefix` -> `width-d4-long-w24` moves the value reading from
    `-0.037147` to `-0.204292`, and the test has to see it - a primary that
    cannot detect that is not measuring the thing it is registered for. The
    direction is fixed too: ``_rank_sum_one_sided(a, b)`` answers whether ``b``
    is stochastically *larger* than ``a``, so the better cell comes first for a
    claim about it being smaller.
    """
    short = _deltas(SHAPE_SHORT, "adaptive", "null")
    long_ = _deltas(SHAPE_LONG, "adaptive", "null")
    assert sum(short) / len(short) > sum(long_) / len(long_)
    u, p = study._rank_sum_one_sided(long_, short)
    assert u == pytest.approx(41829.0)
    assert p < 1e-50
    # The same two lists the other way round must read as a dead null.
    _, reverse = study._rank_sum_one_sided(short, long_)
    assert reverse > 0.999


def _experiment(seed: int, value: float, collapse: float) -> dict:
    """One synthetic per-seed payload driving both primaries independently.

    ``value`` is ``adaptive - null`` (negative is better) and ``collapse`` is
    ``null - random``.
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
    """Install synthetic shards for the new cell, its reference and the ladder."""
    seeds = tuple(range(92000, 92100))
    for profile in (
        study.DEPTH_DOSE_PROFILE,
        study.DEPTH_DOSE_REFERENCE,
        "depth2",
        "depth3",
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
    ("value", "deeper", "branch"),
    [
        ("better", "better", "the gate stays open above the onset and the depth axis pays"),
        ("better", "same", "the onset is a threshold at depth 4"),
        ("worse", "better", "the upper edge of the gate is fragile"),
        ("worse", "same", "the gate is not monotone in depth"),
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
    """Each of the four pre-registered readings is reachable and printed.

    ``value`` is the new cell's own adaptive-vs-null direction (P1) and ``deeper``
    whether its deltas are stochastically better than the reference's (P2), driven
    independently so every combination is exercised rather than only the one the
    real data lands on.
    """
    value_delta = -0.05 if value == "better" else +0.05
    # P2 passes only when the new cell's deltas are strictly smaller than the
    # reference's, so "better" needs a gap and "same" needs none.
    reference_delta = (value_delta + 0.1) if deeper == "better" else value_delta
    _write_cells(
        tmp_path,
        monkeypatch,
        {
            study.DEPTH_DOSE_PROFILE: _all(value_delta),
            study.DEPTH_DOSE_REFERENCE: _all(reference_delta),
        },
    )
    assert study.depth_dose_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    assert branch in out
    line = next(
        item for item in out.splitlines() if item.startswith(f"{study.DEPTH_DOSE_PROFILE}: ")
    )
    assert f"creates value: {value == 'better'}" in line


def test_the_verdict_withholds_a_cell_whose_shards_are_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial shard set is refused, not scored on what happens to be there."""
    _write_cells(tmp_path, monkeypatch, {study.DEPTH_DOSE_PROFILE: _all(-0.05)})
    (tmp_path / study.DEPTH_DOSE_PROFILE / "shard00.json").unlink()
    with pytest.raises(SystemExit) as excinfo:
        study.depth_dose_verdict(1, study.ALPHA)
    assert "missing shard" in str(excinfo.value)
