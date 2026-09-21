"""Section 5.52's long-budget width-24 control: the design pins, before a seed runs.

5.51 stated its limit in the same breath as its result - at 14.8x the collapse
halves at 4x the budget, but with no 192-step *width-24* control its absolute
levels (3.8-4.5 against 15.1-16.9) support no cross-width claim. This arm is
that control: the identical config with only the width changed (96 -> 24), which
completes the prefix-lineage 2x2 - w24 and w96 at 48 and at 192 steps. These
tests pin the design, the two primaries and their directions, the choice of the
shorter-budget corner (prefix, not spread, or the contrast would move width and
policy at once), and the four fixed branches. Results are pinned separately, from
the committed summary.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import null_toy_study as study  # noqa: E402

from act.evolution.experiment import ExperimentConfig  # noqa: E402
from act.evolution.study import _gate_min_wins  # noqa: E402

SUMMARIES = ROOT / "artifacts" / "study-summaries"
# The four corners of the 2x2, in the order the verdict prints them. Every one
# is prefix-policy, which is what makes the two widths comparable at both
# budgets; the cells this arm is a control for are all in that same lineage.
CELLS_2X2 = (
    study.LONG_W24_SAME_WIDTH,  # w24 @  48
    study.DOSE2_LONG_REFERENCE,  # w96 @  48
    study.LONG_W24_PROFILE,  # w24 @ 192 (this arm)
    study.LONG_W24_SAME_BUDGET,  # w96 @ 192
)


def test_the_control_changes_only_the_width() -> None:
    """5.51's cell at hidden_size 24 - the one difference this arm exists to make.

    If the steps, the interval or anything else drifted, the cell would no
    longer be the budget-scaling of 5.51's and the 2x2's width axis would carry
    the drift instead of the width.
    """
    wide = study.PROFILES[study.LONG_W24_SAME_BUDGET]["overrides"]
    narrow = study.PROFILES[study.LONG_W24_PROFILE]["overrides"]
    assert wide["hidden_size"] == 96
    assert narrow["hidden_size"] == 24
    assert {k: v for k, v in wide.items() if k != "hidden_size"} == {
        k: v for k, v in narrow.items() if k != "hidden_size"
    }
    # Which is 4x the registered budget, with the events' count preserved.
    assert narrow["steps"] == 4 * study.REGISTERED["steps"] == 192
    assert narrow["mutation_interval"] == 4 * study.REGISTERED["mutation_interval"]
    assert narrow["depth"] == study.REGISTERED["depth"] + 3  # the above-onset depth


def test_the_cell_is_the_pre_registered_one() -> None:
    spec = study.PROFILES[study.LONG_W24_PROFILE]
    assert spec["report_section"] == "5.52"
    # The frozen revision is the *registration* commit - it cannot be named
    # before it exists, so it is set by the freeze commit that follows. The
    # summary's `code_revision` is checked against it once the shards run.
    assert spec["run_revision"] == "f636990"
    assert len(spec["seeds"]) == 576
    assert spec["seeds"][0] == 87600
    assert spec["summary"].endswith("width-d4-long-w24-576.json")


def test_the_control_block_is_fresh() -> None:
    """The report calls 87600-88175 a fresh block; check it against both lists.

    Against every other profile's registered block in the script itself and
    every committed study summary in the repository - the check that caught the
    section-5.17 collision when 5.50's depth-4 block was first drafted at 81000.
    """
    block = set(study.PROFILES[study.LONG_W24_PROFILE]["seeds"])
    assert block == set(range(87600, 88176))
    used: set[int] = set()
    for profile in study.PROFILES:
        if profile == study.LONG_W24_PROFILE:
            continue
        used |= set(study.PROFILES[profile]["seeds"])
    own = Path(study.PROFILES[study.LONG_W24_PROFILE]["summary"]).name
    for path in sorted(SUMMARIES.glob("*.json")):
        if path.name == own:
            continue
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):  # some files are panels, not records
            continue
        seeds = payload.get("seeds")
        if isinstance(seeds, list):
            used |= {seed for seed in seeds if isinstance(seed, int)}
    assert not (block & used), f"block overlaps an earlier study: {sorted(block & used)[:5]}"


def test_the_shorter_budget_corner_is_the_prefix_cell_not_the_spread_one() -> None:
    """A 2x2 compares width only if the candidate policy is held fixed.

    Section 5.10's depth-4 cell is the same width at the same budget but runs the
    *spread* candidate policy, while every 5.48-5.52 cell runs the registered
    default. Ranking the new cell against that one would move width and policy
    together, so the corner is section 5.12's depth-4 **prefix** cell instead -
    the reason the 2x2 can be read at all.
    """
    assert study.LONG_W24_SAME_WIDTH == "depth4-prefix"
    assert study.PROFILES[study.LONG_W24_SAME_WIDTH]["overrides"] == {"depth": 4}
    assert study.PROFILES["depth4"]["overrides"] == {
        "depth": 4,
        "probe_candidate_policy": "spread",
    }
    defaults = {field.name: field.default for field in dataclasses.fields(ExperimentConfig)}
    assert defaults["probe_candidate_policy"] == "prefix"
    # And no cell from 5.48 on overrides it: width (and the budget) are the only
    # axes any of these arms moves.
    for profile in (
        *study.WIDTH_PROFILES,
        *study.DOSE_PROFILES,
        *study.DOSE2_PROFILES,
        study.DOSE2_LONG_PROFILE,
        study.LONG_W24_PROFILE,
    ):
        assert "probe_candidate_policy" not in study.PROFILES[profile]["overrides"]


def test_the_primaries_are_the_two_primary_level() -> None:
    """Two primaries: alpha/2 = 0.025, bar 313/576 on the cell's own test."""
    assert _gate_min_wins(576, 0.025) == 313
    # The committed width-24 above-onset rate still clears it, so P2 is not
    # asking the cell to beat a bar the short-budget sibling misses.
    assert 225 / 384 > 313 / 576


def _experiment(seed: int, *, collapse: float, adaptive: float) -> dict:
    """One synthetic per-seed payload in the study's committed schema.

    ``collapse`` is the cell's null-vs-random delta (positive when churning
    beats doing nothing) and ``adaptive`` its adaptive-vs-null delta.
    """
    return {
        "seed": seed,
        "losses": {
            "random": 10.0 - collapse,
            "adaptive": 10.0 + adaptive,
            "null": 10.0,
        },
        "mutations": {"random": 3, "adaptive": 3, "null": 0},
        "search_flops": {"random": 1.0, "adaptive": 1.0, "null": 0.0},
    }


def _write_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    *,
    collapse: float,
    adaptive: float,
) -> None:
    seeds = tuple(range(90000, 90100))
    monkeypatch.setitem(
        study.PROFILES, profile, {**study.PROFILES[profile], "seeds": seeds}
    )
    directory = tmp_path / profile
    directory.mkdir(parents=True, exist_ok=True)
    rows = [_experiment(seed, collapse=collapse, adaptive=adaptive) for seed in seeds]
    payload = {"shard": 0, "shards": 1, "profile": profile, "experiments": rows}
    (directory / "shard00.json").write_text(json.dumps(payload) + "\n")


def _write_2x2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    narrow_collapse: float,
    narrow_adaptive: float,
    wide_collapse: float,
    short_narrow_collapse: float = 0.1,
    short_wide_collapse: float = 1.6,
    wide_adaptive: float = -0.01,
) -> None:
    """Install synthetic shards for all four corners of the 2x2."""
    for profile, collapse, adaptive in (
        (study.LONG_W24_SAME_WIDTH, short_narrow_collapse, -0.03),
        (study.DOSE2_LONG_REFERENCE, short_wide_collapse, -0.12),
        (study.LONG_W24_PROFILE, narrow_collapse, narrow_adaptive),
        (study.LONG_W24_SAME_BUDGET, wide_collapse, wide_adaptive),
    ):
        _write_cell(tmp_path, monkeypatch, profile, collapse=collapse, adaptive=adaptive)
    monkeypatch.setattr(study, "SHARD_DIR", tmp_path)


@pytest.mark.parametrize(
    ("narrow_collapse", "narrow_adaptive", "branch"),
    [
        (0.05, -0.05, "5.51's relief is a capacity effect"),
        (0.05, +0.05, "the budget is not what the narrow substrate was short of"),
        (1.5, -0.05, "the collapse is general across widths at 4x"),
        (1.5, +0.05, "neither relieves the narrow cell's collapse nor keeps its value"),
    ],
)
def test_the_four_branches_are_fixed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    narrow_collapse: float,
    narrow_adaptive: float,
    branch: str,
) -> None:
    """Every branch text is fixed in advance, and each one is reachable."""
    _write_2x2(
        tmp_path,
        monkeypatch,
        narrow_collapse=narrow_collapse,
        narrow_adaptive=narrow_adaptive,
        wide_collapse=1.5,
    )
    assert study.long_w24_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    assert branch in out
    assert "P1 (the residual collapse is the wide substrate's): rank-sum" in out


def test_p1_reads_the_pre_registered_direction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """P1 asks whether the *narrow* cell's collapse is the smaller one.

    The two cells carry the same collapse (1.5) in the second case, so the
    one-sided test cannot separate them and the branch must not claim the
    collapse is the wide substrate's - which is the claim the arm's value rests
    on, so a test that fired on any difference would be the wrong test.
    """
    _write_2x2(
        tmp_path,
        monkeypatch,
        narrow_collapse=0.05,
        narrow_adaptive=-0.05,
        wide_collapse=1.5,
    )
    assert study.long_w24_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    assert "-> collapse confined to width 96: True" in out

    _write_2x2(
        tmp_path,
        monkeypatch,
        narrow_collapse=1.5,
        narrow_adaptive=-0.05,
        wide_collapse=1.5,
    )
    assert study.long_w24_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    assert "-> collapse confined to width 96: False" in out


def test_the_verdict_reports_the_whole_2x2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The branch is read in a 2x2, so the verdict prints all four corners.

    Every number a cross-width reading needs is in the output: the four
    collapses with their seed counts, the within-width relief (the difference in
    differences, descriptive) and the controller's own edge beside it.
    """
    _write_2x2(
        tmp_path,
        monkeypatch,
        narrow_collapse=0.05,
        narrow_adaptive=-0.05,
        wide_collapse=1.5,
    )
    assert study.long_w24_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    for label in CELLS_2X2:
        assert f"({label})" in out
    assert "over 100 seeds" in out
    assert "within-width relief (descriptive, the difference in differences)" in out
    assert "edge (descriptive): adaptive-vs-random rank-sum" in out
