"""Section 5.53's expert-count covariate: the design pins, before any new seed.

Section 10.3(ii) named the covariate candidates left after 5.48 closed the
matched-parameter width form -- "expert count, total compute" -- and this arm
runs the first of them. It matters more than the other remaining candidate
because expert grow/prune is the *only* operation the V0 search space can reach
(section 10.4), so whether expert count moves the onset decides whether the
reachable gene is the one the ontology reads.

These tests pin the design, the equal-parameter partner's two axes, the fact
that the probe's candidate pool does not move with the cell, the bar, and the
four fixed branches. Results are pinned separately, from the committed
summaries.
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
from act.topology.mutations import feasible_expert_mutations  # noqa: E402

REGISTERED_CELLS = {"experts-d3-dose": 3, "experts-d4-dose": 4}
SUMMARIES = ROOT / "artifacts" / "study-summaries"
PROBE_POOL = ("grow", "prune", "prune", "prune")


def _parameters(depth: int, hidden: int, experts: int) -> tuple[int, int]:
    genome = create_genome(
        0,
        hidden_size=hidden,
        vocab_size=study.REGISTERED["vocab_size"],
        depth=depth,
        ffn="moe",
        expert_count=experts,
        active_experts=study.REGISTERED["active_experts"],
    )
    model = model_from_genome(genome, context_size=study.REGISTERED["sequence_length"])
    return model.parameter_count, model.active_parameter_count


def test_the_expert_cells_are_the_pre_registered_ones() -> None:
    """Expert count 4 -> 6 at both depths, width and active_experts held."""
    for profile, depth in REGISTERED_CELLS.items():
        spec = study.PROFILES[profile]
        assert spec["overrides"] == {"depth": depth, "expert_count": 6}
        assert spec["report_section"] == "5.53"
        assert spec["per_cell_primary"] is True
        assert spec["bonferroni_divisor"] == 2.0
        assert len(spec["seeds"]) == 576
    assert study.PROFILES["experts-d3-dose"]["seeds"][0] == 88176
    assert study.PROFILES["experts-d4-dose"]["seeds"][0] == 89384
    # The arm's cells are the ones the branch logic reads, in order.
    assert study.EXPERTS_PROFILES == ("experts-d3-dose", "experts-d4-dose")
    # And the treatment is the expert count and nothing else: if a cell loses
    # its override this fails rather than silently re-running a committed
    # experiment under a new report section.
    assert study.REGISTERED["hidden_size"] == 24
    assert study.REGISTERED["expert_count"] == 4
    assert study.REGISTERED["active_experts"] == 2
    assert study.REGISTERED["adaptive_policy"] == "probe_loss"
    assert study.REGISTERED["mutation_interval"] == 12
    assert study.REGISTERED["mutation_budget"] == 3
    assert study.REGISTERED["probe_candidate_cap"] == 4


def test_the_expert_blocks_are_fresh() -> None:
    """88176-88751 and 89384-89959 are fresh blocks; the report says so.

    Against two lists: every other profile's registered block in the script
    itself, and every committed study summary in the repository. 5.52 holds
    87600-88175, so the depth-3 cell starts at the first free window at or after
    88000 rather than at a round number.
    """
    blocks = {profile: set(study.PROFILES[profile]["seeds"]) for profile in REGISTERED_CELLS}
    assert blocks["experts-d3-dose"] == set(range(88176, 88752))
    assert blocks["experts-d4-dose"] == set(range(89384, 89960))
    used: set[int] = set()
    for profile in study.PROFILES:
        if profile in REGISTERED_CELLS:
            continue
        used |= set(study.PROFILES[profile]["seeds"])
    own = {Path(study.PROFILES[profile]["summary"]).name for profile in REGISTERED_CELLS}
    for path in sorted(SUMMARIES.glob("*.json")):
        if path.name in own:
            continue
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):  # some files are panels, not records
            continue
        seeds = payload.get("seeds")
        if isinstance(seeds, list):
            used |= {seed for seed in seeds if isinstance(seed, int)}
    for profile, block in blocks.items():
        assert not (block & used), f"{profile} overlaps an earlier study: {sorted(block & used)[:5]}"


def test_the_dose_is_the_tightest_equal_parameter_partner_and_buys_no_compute() -> None:
    """30,624 total (-0.156%) but 9,024 active (-21.3%) - the branch's claim.

    The branch text rests on both numbers, so both are pinned. 5.48's width
    partner matched *both* axes (+1.695% total, +2.0% active); this cell matches
    the one it is read on and is 21.3% below on the other, which is what makes
    it the first cell in the family that can separate stored capacity from
    per-token compute.
    """
    above_total, above_active = _parameters(4, 24, 4)
    assert (above_total, above_active) == (30_672, 11_472)
    d3_total, d3_active = _parameters(3, 24, 6)
    assert (d3_total, d3_active) == (30_624, 9_024)
    d4_total, d4_active = _parameters(4, 24, 6)
    assert (d4_total, d4_active) == (40_464, 11_664)
    assert (d3_total - above_total) / above_total == pytest.approx(-0.001565, abs=1e-6)
    assert (d3_active - above_active) / above_active == pytest.approx(-0.21339, abs=1e-4)
    assert (d4_total - above_total) / above_total == pytest.approx(0.31925, abs=1e-4)
    assert (d4_active - above_active) / above_active == pytest.approx(0.01674, abs=1e-4)
    # The reference the branch contrasts it with: width 28 at depth 3 matched
    # both axes of the same target.
    w28_total, w28_active = _parameters(3, 28, 4)
    assert w28_total / above_total == pytest.approx(1.01695, abs=1e-4)
    assert w28_active / above_active == pytest.approx(1.02022, abs=1e-4)
    # And the committed baselines the four cells are read against, so the
    # descriptive levels cannot drift either.
    below_total, below_active = _parameters(3, 24, 4)
    assert (below_total, below_active) == (23_280, 8_880)


def test_the_probe_pool_does_not_move_with_the_expert_count() -> None:
    """Cap 4 with the prefix policy sees the same four operations at E=4 and E=6.

    This is what makes the cells comparable at all: if the pool changed, the
    arm would move the substrate and the controller's candidate set at once.
    """
    pools = {}
    for experts in (4, 6):
        genome = create_genome(
            0,
            hidden_size=study.REGISTERED["hidden_size"],
            vocab_size=study.REGISTERED["vocab_size"],
            depth=3,
            ffn="moe",
            expert_count=experts,
            active_experts=study.REGISTERED["active_experts"],
        )
        feasible = feasible_expert_mutations(genome)
        cap = study.REGISTERED["probe_candidate_cap"]
        assert len(feasible) > cap  # the cap binds at both counts
        pools[experts] = tuple(operation for operation, _, _ in feasible[:cap])
    assert pools[4] == pools[6] == PROBE_POOL


def test_the_bar_is_the_two_primary_level() -> None:
    """Two primaries: alpha/2 = 0.025, bar 313/576 - the sibling arms'."""
    assert _gate_min_wins(576, study.ALPHA / 2) == 313
    for profile in REGISTERED_CELLS:
        assert float(study.PROFILES[profile]["bonferroni_divisor"]) == 2.0


def _experiment(seed: int, delta: float) -> dict:
    """One synthetic per-seed payload in the study's committed schema."""
    return {
        "seed": seed,
        "losses": {"random": 10.0, "adaptive": 10.0 + delta, "null": 10.0},
        "mutations": {"random": 3, "adaptive": 3, "null": 0},
        "search_flops": {"random": 1.0, "adaptive": 1.0, "null": 0.0},
    }


def _write_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deltas: dict[str, list[float]],
) -> None:
    """Install synthetic shards for both cells and the references they print.

    The verdict's descriptive block reads three *committed* cells as well, so
    they are installed too: a missing shard is a withheld verdict by design, and
    a test that skipped the block would not exercise the branch logic at all.
    """
    seeds = tuple(range(91000, 91100))
    references = (
        study.EXPERTS_BELOW_BASELINE,
        study.EXPERTS_ABOVE_BASELINE,
        study.EXPERTS_WIDTH_PARTNER,
    )
    for profile in references:
        deltas.setdefault(profile, _all(0.0))
    for profile in study.EXPERTS_PROFILES + references:
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
                _experiment(seed, delta) for seed, delta in zip(seeds, deltas[profile])
            ],
        }
        (directory / "shard00.json").write_text(json.dumps(payload) + "\n")
    monkeypatch.setattr(study, "SHARD_DIR", tmp_path)
    return seeds


def _all(delta: float) -> list[float]:
    return [delta] * 100


@pytest.mark.parametrize(
    ("first", "second", "branch"),
    [
        ("better", "better", "capacity is the covariate when bought with experts"),
        ("better", "worse", "buys value below the onset and destroys"),
        ("worse", "better", "the gate does not move for expert count either"),
        ("worse", "worse", "expert capacity is harmful at both depths"),
    ],
)
def test_the_four_expert_branches_are_fixed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    first: str,
    second: str,
    branch: str,
) -> None:
    """Each of the four pre-registered readings is reachable and printed."""
    values = {"better": -0.05, "worse": +0.05}
    _write_cells(
        tmp_path,
        monkeypatch,
        {
            study.EXPERTS_PROFILES[0]: _all(values[first]),
            study.EXPERTS_PROFILES[1]: _all(values[second]),
        },
    )
    assert study.experts_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    assert branch in out
    expected = {
        study.EXPERTS_PROFILES[0]: values[first] < 0,
        study.EXPERTS_PROFILES[1]: values[second] < 0,
    }
    for profile, creates in expected.items():
        assert f"{profile}: " in out
        line = next(item for item in out.splitlines() if item.startswith(f"{profile}: "))
        assert f"creates value: {creates}" in line


def test_the_verdict_withholds_a_cell_whose_shards_are_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial shard set is refused, not scored on what happens to be there."""
    _write_cells(
        tmp_path,
        monkeypatch,
        {profile: _all(-0.05) for profile in study.EXPERTS_PROFILES},
    )
    (tmp_path / study.EXPERTS_PROFILES[1] / "shard00.json").unlink()
    with pytest.raises(SystemExit) as excinfo:
        study.experts_verdict(1, study.ALPHA)
    assert "missing shard" in str(excinfo.value)
