"""Section 5.50's second dose: the design pins, before any fresh seed is touched.

5.49 established what one dose does - above the onset a 3.80x dose amplifies
value creation (0.586 -> 0.828, mean -0.2097) and below it the same dose makes
value creation *worse* - but one dose cannot say whether capacity keeps paying
or plateaus, the shape 10.5(B)'s movable-width policy would need. This arm adds
a second, larger rung (width 96, 3.90x the 5.49 dose above the gate) and, for
the first time in this family, a primary that compares *two independent cells*:
a one-sided rank-sum between the committed 3.80x cell and the new 14.8x cell,
so the continuation claim is carried by a pre-registered test rather than by a
bigger number. These tests pin the design, this arm's Bonferroni bar, the
rank-sum's correctness and its rehearsed sensitivity on committed data, and the
four fixed branches plus the rider only the P2-only branch may print. Results
are pinned separately, from the committed summaries.
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

REGISTERED_CELLS = {"width-d3-dose2": 3, "width-d4-dose2": 4}
SUMMARIES = ROOT / "artifacts" / "study-summaries"


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


def test_the_second_dose_cells_are_the_pre_registered_ones() -> None:
    """Fresh blocks 80000/84000, width 96, the dose-response's own config."""
    for profile, depth in REGISTERED_CELLS.items():
        spec = study.PROFILES[profile]
        assert spec["overrides"] == {"depth": depth, "hidden_size": 96}
        assert spec["report_section"] == "5.50"
        assert spec["run_revision"] is None  # frozen by the registration commit
        assert len(spec["seeds"]) == 576
    assert study.PROFILES["width-d3-dose2"]["seeds"][0] == 80000
    assert study.PROFILES["width-d4-dose2"]["seeds"][0] == 84000
    # The arm's cells are the ones the branch logic reads, in order, and the
    # continuation primary reads the committed 3.80x rung - named, not inline.
    assert study.DOSE2_PROFILES == ("width-d3-dose2", "width-d4-dose2")
    assert study.DOSE2_REFERENCE == "width-d4-dose"
    # The treatment is the substrate's width and nothing else: if a cell loses
    # its width override this fails rather than silently re-running a
    # committed experiment under a new report section.
    assert study.REGISTERED["hidden_size"] == 24
    assert study.REGISTERED["adaptive_policy"] == "probe_loss"
    assert study.REGISTERED["mutation_interval"] == 12
    assert study.REGISTERED["mutation_budget"] == 3
    assert study.REGISTERED["probe_candidate_cap"] == 4


def test_the_second_dose_blocks_are_fresh() -> None:
    """The report calls 80000-80575 and 84000-84575 fresh blocks; check it.

    Against two lists: every other profile's registered block in the script
    itself (where an unrun collision would show up first) and every committed
    study summary in the repository. The depth-4 block's first draft (81000)
    overlapped the section-5.17 reusability study's block and was caught here,
    which is why the pinned base is 84000 rather than the next round number.
    """
    blocks = {profile: set(study.PROFILES[profile]["seeds"]) for profile in REGISTERED_CELLS}
    assert blocks["width-d3-dose2"] == set(range(80000, 80576))
    assert blocks["width-d4-dose2"] == set(range(84000, 84576))
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


def test_the_second_dose_is_the_measured_multiple_of_the_rungs() -> None:
    """Width 96: 11.15x / 14.82x the above-onset cell, 3.89x the 5.49 dose.

    The branch text and the registration quote these factors, so the counts are
    pinned: if the model's arithmetic changes, the quoted factors must be
    revisited rather than silently invalidated.
    """
    above_onset_total = _parameters(4, 24)[0]
    dose3_total = _parameters(3, 48)[0]
    dose4_total = _parameters(4, 48)[0]
    new3, new3_active = _parameters(3, 96)
    new4, new4_active = _parameters(4, 96)
    assert (new3, new3_active) == (341_952, 118_464)
    assert (new4, new4_active) == (454_464, 156_480)
    assert new3 / above_onset_total == pytest.approx(11.149, abs=5e-3)
    assert new4 / above_onset_total == pytest.approx(14.817, abs=5e-3)
    assert new3 / dose3_total == pytest.approx(3.884, abs=5e-3)
    assert new4 / dose4_total == pytest.approx(3.896, abs=5e-3)


def test_the_second_dose_bar_is_the_three_primary_level() -> None:
    """Three primaries: alpha/3 = 0.01667, bar 315/576 - stricter than 5.49's.

    The sibling cells applied alpha/2 = 0.025 (bar 313/576) because they posed
    two primaries each. This arm poses three (two cells plus the continuation
    test), so the registration is explicit that the cell bar moved with the
    family size, computed by the same function the verdict calls. At n = 576,
    the committed above-onset rate (225/384 = 0.5859) still lands ~338 wins,
    above the bar; the committed below-onset rate (188/384 = 0.4896) lands
    ~282, below it.
    """
    assert _gate_min_wins(576, 0.05 / 3) == 315
    assert _gate_min_wins(576, 0.025) == 313  # the sibling cells' applied bar
    assert 0.5859 * 576 > 315
    assert 0.4896 * 576 < 315


def test_the_rank_sum_is_one_sided_and_tie_corrected() -> None:
    """The one asymptotic test in the family, checked against hand values.

    a = [1, 2, 2], b = [3, 4, 5]: average ranks put both 2s at 2.5, so
    U = 0, the tie-corrected sd is sqrt(5.1), and the continuity-corrected
    lower tail gives p = 0.0383. Reversed, U is at its maximum and the
    one-sided p is near 1 (that direction is the null side). Identical samples
    carry no ordering information at all: the variance is zero and the
    function returns p = 1 rather than dividing by it.
    """
    u, p = study._rank_sum_one_sided([1.0, 2.0, 2.0], [3.0, 4.0, 5.0])
    assert u == pytest.approx(0.0)
    assert p == pytest.approx(0.0383, abs=1e-4)
    u_big, p_big = study._rank_sum_one_sided([3.0, 4.0, 5.0], [1.0, 2.0, 2.0])
    assert u_big == pytest.approx(9.0)
    assert p_big > 0.95
    u_flat, p_flat = study._rank_sum_one_sided([0.0] * 5, [0.0] * 5)
    assert u_flat == pytest.approx(12.5)  # the mean U: no ordering information
    assert p_flat == 1.0


def _summary_deltas(filename: str) -> list[float]:
    payload = json.loads((SUMMARIES / filename).read_text())
    return [
        row["losses"]["adaptive"] - row["losses"]["null"] for row in payload["per_seed"]
    ]


def test_the_rank_sum_was_rehearsed_on_the_committed_rungs() -> None:
    """The continuation test is priced on real data *before* the run.

    The registration quotes two rehearsals, and both are recomputed here from
    the committed summaries so the quoted p-values cannot drift: the measured
    dose step (3.80x versus width 28 - a 2.84x parameter step) clears this
    arm's alpha/3 at p = 4.5e-44, and the measured wash (+1.7%, width 28
    versus 24) is a dead null at p = 0.50. A test that separated the step but
    not the wash would be a ruler with the wrong zero; one that separated
    neither could not carry the continuation claim at all.
    """
    dose = _summary_deltas("width-d4-dose-576.json")
    equal = _summary_deltas("width-d4-equal-576.json")
    width24 = _summary_deltas("null-control-depth4-384.json")
    _, p_step = study._rank_sum_one_sided(dose, equal)
    _, p_wash = study._rank_sum_one_sided(equal, width24)
    assert p_step < 1e-30
    assert p_wash > 0.3


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
    """Install synthetic shards for both cells and the reference under a temp root."""
    seeds = tuple(range(90000, 90100))
    for profile in study.DOSE2_PROFILES + (study.DOSE2_REFERENCE,):
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


def _all(delta: float) -> list[float]:
    return [delta] * 100


@pytest.mark.parametrize(
    ("first", "second", "rider", "branch"),
    [
        (
            "better",
            "better",
            False,
            "at 14.8x, capacity buys value at a below-onset depth",
        ),
        (
            "better",
            "worse",
            False,
            "the largest dose buys value below the onset and destroys",
        ),
        ("worse", "better", True, "depth stands at every reachable dose"),
        ("worse", "worse", False, "at 14.8x nothing creates value"),
    ],
)
def test_the_four_second_dose_branches_are_fixed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    first: str,
    second: str,
    rider: bool,
    branch: str,
) -> None:
    """Every branch text is fixed in advance, and the rider is gated to one."""
    cells = {"better": _all(-0.05), "worse": _all(+0.05)}
    _write_cells(
        tmp_path,
        monkeypatch,
        {
            "width-d3-dose2": cells[first],
            "width-d4-dose2": cells[second],
            study.DOSE2_REFERENCE: _all(+0.05),
        },
    )
    assert study.dose2_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    assert branch in out
    # The raw P3 statistic is a primary: its line always prints. The dose-form
    # *rider* is a claim about the P2-only branch and may not leak into others.
    assert "P3 (the amplifier continues): rank-sum" in out
    assert ("dose form (P3, pre-registered rider)" in out) == rider


def test_a_bigger_win_rate_is_not_a_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The rider reads the test, not the win rate - the trap case, constructed.

    90 of 100 seeds improve (a win rate *above* the committed 3.80x cell's
    0.828) with the mean negative enough to create value, yet the typical
    delta sits *above* the reference cell's bulk: the pre-registered rank-sum
    sees no stochastic superiority, and the rider must say plateau. The
    converse construction - a typical delta clearly below the reference's -
    prints the continuation. Same verdict code, opposite rider lines: the
    label is the test's, not the win rate's.
    """
    _write_cells(
        tmp_path,
        monkeypatch,
        {
            "width-d3-dose2": _all(+0.05),
            # 0.90 wins, mean -0.40: value creation, and no stochastic
            # superiority over a reference whose bulk sits lower.
            "width-d4-dose2": [(-1.0)] * 90 + [(+5.0)] * 10,
            study.DOSE2_REFERENCE: [(-1.5)] * 50 + [(-1.2)] * 50,
        },
    )
    assert study.dose2_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    assert "width-d4-dose2: adaptive-vs-null wins 90/100" in out
    assert "P2 only" in out
    assert "dose form (P3, pre-registered rider): not distinguishable from a plateau" in out
    assert "rider): the amplifier continues" not in out

    _write_cells(
        tmp_path,
        monkeypatch,
        {
            "width-d3-dose2": _all(+0.05),
            "width-d4-dose2": [(-1.0)] * 90 + [(+0.5)] * 10,
            study.DOSE2_REFERENCE: [(-0.2)] * 50 + [(+0.5)] * 50,
        },
    )
    assert study.dose2_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    assert "P2 only" in out
    assert "dose form (P3, pre-registered rider): the amplifier continues" in out
