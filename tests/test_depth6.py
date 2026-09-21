"""Section 5.56's confirmation rung: the design pins, before any new seed.

5.55 found the depth axis monotone, with the whole 4 -> 5 gain in the churn term
rather than the controller's edge. "The axis keeps paying" is then a *two-point*
slope above the gate, and both of its failure modes bound the claim: a plateau
makes it a one-step statement, a reversal bounds monotonicity to depths 2..5.
This arm is the third point - the same one-gene dose, one layer further up.

These tests pin the cell, its block freshness, the one gene it changes, the fact
that the candidate pool does **not** move with depth (so only the substrate
changes), the bar, the rank-sum primary's direction against 5.55's rehearsed
ruler, and the four fixed branches. Results are pinned separately.
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
    """`depth` 5 -> 6 against the registered config, and nothing else at all."""
    spec = study.PROFILES[study.DEPTH6_PROFILE]
    assert spec["overrides"] == {"depth": 6}
    assert spec["report_section"] == "5.56"
    assert spec["per_cell_primary"] is True
    assert spec["bonferroni_divisor"] == 2.0
    assert len(spec["seeds"]) == 576
    assert spec["seeds"][0] == 92384
    assert spec["seeds"][-1] == 92959
    # 5.55 is the reference, and the two differ in exactly the one key.
    assert study.DEPTH6_REFERENCE == study.DEPTH_DOSE_PROFILE == "depth5-dose"
    reference = study.PROFILES[study.DEPTH6_REFERENCE]
    assert reference["overrides"] == {"depth": 5}
    assert set(spec["overrides"]) ^ set(reference["overrides"]) == set()
    # The ladder now has three rungs above the onset and two below.
    assert [study.PROFILES[study.DEPTH6_REFERENCE]["overrides"]["depth"]] == [5]
    assert study.REGISTERED["depth"] == 1


def test_the_candidate_pool_does_not_move_with_depth() -> None:
    """The controller sees the same four candidates at every rung of the ladder.

    Without this the arm would move two things at once - the substrate *and* the
    set the policy chooses from - which is the check 5.53 had to make for the
    expert dose too. The pool bind is the cap (4 of 15-30 feasible candidates),
    so the same four survive truncation at every depth.
    """
    cap = study.REGISTERED["probe_candidate_cap"]
    assert cap == 4
    heads = {}
    for depth in (3, 4, 5, 6):
        candidates = feasible_expert_mutations(_genome(depth))
        assert len(candidates) > cap, depth  # the cap really does bind
        heads[depth] = [str(c) for c in candidates[:cap]]
    assert len(set(tuple(v) for v in heads.values())) == 1, heads
    assert heads[6][0].startswith("('grow', 'L0'")
    assert heads[6][1].startswith("('prune', 'L0', 'L0E0'")


def test_the_ladder_it_extends_is_the_one_being_confirmed() -> None:
    """Two rungs below the onset, two above, and one gene between them all."""
    for profile, depth in (
        ("depth2", 2),
        ("depth3", 3),
        ("depth4-prefix", 4),
        (study.DEPTH_DOSE_PROFILE, 5),
        (study.DEPTH6_PROFILE, 6),
    ):
        overrides = study.PROFILES[profile]["overrides"]
        assert overrides["depth"] == depth
        # Every rung is the registered config otherwise: no width, expert count,
        # budget or policy change anywhere on the ladder.
        assert set(overrides) == {"depth"}, profile
    assert study.REGISTERED["steps"] == 48
    assert study.REGISTERED["mutation_interval"] == 12
    assert study.REGISTERED["hidden_size"] == 24
    assert study.REGISTERED["expert_count"] == 4
    # Every rung adds exactly 7,392 total / 2,592 active parameters, which is
    # why this is the depth axis as a package and not a matched-parameter arm.
    ladder = [_parameters(d) for d in (2, 3, 4, 5, 6)]
    assert ladder == [
        (15_888, 6_288),
        (23_280, 8_880),
        (30_672, 11_472),
        (38_064, 14_064),
        (45_456, 16_656),
    ]
    for before, after in zip(ladder, ladder[1:]):
        assert after[0] - before[0] == 7_392
        assert after[1] - before[1] == 2_592


def test_the_block_is_fresh() -> None:
    """92384-92959 is free; the report calls it fresh, so check it."""
    block = set(study.PROFILES[study.DEPTH6_PROFILE]["seeds"])
    assert block == set(range(92384, 92960))
    used: set[int] = set()
    for profile, spec in study.PROFILES.items():
        if profile == study.DEPTH6_PROFILE:
            continue
        used |= set(spec["seeds"])
    own = Path(study.PROFILES[study.DEPTH6_PROFILE]["summary"]).name
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
    # And the rung it must be read against is committed.
    assert (SUMMARIES / Path(study.PROFILES[study.DEPTH6_REFERENCE]["summary"]).name).is_file()


def test_the_bar_is_the_two_primary_level() -> None:
    assert _gate_min_wins(576, study.ALPHA / 2) == 313
    assert float(study.PROFILES[study.DEPTH6_PROFILE]["bonferroni_divisor"]) == 2.0


def test_the_slope_claim_is_the_one_5_55_measured_the_previous_step_of() -> None:
    """The 4 -> 5 step 5.55 recorded, which this arm asks about one rung up.

    Recomputing it from the committed summaries is what makes this arm a
    *confirmation*: the previous step is a known detection, so P2's failure here
    would be a real plateau rather than a ruler that cannot see a step.
    """
    def deltas(name: str) -> list[float]:
        payload = json.loads((SUMMARIES / name).read_text())
        return [
            row["losses"]["adaptive"] - row["losses"]["null"] for row in payload["per_seed"]
        ]

    four = deltas("null-control-depth4-prefix-384.json")
    five = deltas("depth5-dose-576.json")
    u, p = study._rank_sum_one_sided(five, four)
    assert p < 1e-3, "the 4 -> 5 step 5.55 recorded is not visible"
    assert sum(five) / len(five) < sum(four) / len(four)
    _, reverse = study._rank_sum_one_sided(four, five)
    assert reverse > 0.999


def _experiment(seed: int, value: float, collapse: float) -> dict:
    """One synthetic per-seed payload; ``value`` is adaptive-null and negative is better."""
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
    """Install synthetic shards for the new cell, its reference and the ladder."""
    seeds = tuple(range(93000, 93100))
    for profile in (
        study.DEPTH6_PROFILE,
        study.DEPTH6_REFERENCE,
        "depth2",
        "depth3",
        study.DEPTH_DOSE_REFERENCE,
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
        ("better", "better", "the ladder is monotone across all three rungs above the onset"),
        ("better", "same", "the axis saturates at depth 5"),
        ("worse", "better", "a fragile upper reading"),
        ("worse", "same", "the ladder is not monotone past depth 5"),
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
    """Each of the four pre-registered readings is reachable and printed."""
    value_delta = -0.05 if value == "better" else +0.05
    # P2 passes only when the new cell's deltas are strictly smaller than the
    # reference's, so "better" needs a gap and "same" needs none.
    reference_delta = (value_delta + 0.1) if deeper == "better" else value_delta
    _write_cells(
        tmp_path,
        monkeypatch,
        {
            study.DEPTH6_PROFILE: _all(value_delta),
            study.DEPTH6_REFERENCE: _all(reference_delta),
        },
    )
    assert study.depth6_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    assert branch in out
    line = next(
        item for item in out.splitlines() if item.startswith(f"{study.DEPTH6_PROFILE}: ")
    )
    assert f"creates value: {value == 'better'}" in line


def test_the_verdict_withholds_a_cell_whose_shards_are_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial shard set is refused, not scored on what happens to be there."""
    _write_cells(tmp_path, monkeypatch, {study.DEPTH6_PROFILE: _all(-0.05)})
    (tmp_path / study.DEPTH6_PROFILE / "shard00.json").unlink()
    with pytest.raises(SystemExit) as excinfo:
        study.depth6_verdict(1, study.ALPHA)
    assert "missing shard" in str(excinfo.value)
