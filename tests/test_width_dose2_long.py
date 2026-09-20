"""Section 5.51's long-budget cell: the design pins, before any fresh seed runs.

5.50 registered its own limit rather than only a result: at 14.8x the do-nothing
arm collapses, and at a fixed 48-step budget "a wide substrate is unusable
unless mutated at all" cannot be separated from "it needs more training than
the budget gives it". This arm runs the identical config at 4x the steps and 4x
the mutation interval - keeping the three mutational events - and asks which of
the two the collapse is. These tests pin the design, the two primaries and
their directions, and the four fixed branches, including the independence the
branch table turns on (a cell can create value with or without the pool
relief). Results are pinned separately, from the committed summary.
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

SUMMARIES = ROOT / "artifacts" / "study-summaries"


def test_the_long_cell_is_the_pre_registered_one() -> None:
    """Fresh block 87000-87575: width 96 at 4x the steps, 4x the interval."""
    spec = study.PROFILES[study.DOSE2_LONG_PROFILE]
    assert spec["overrides"] == {
        "depth": 4,
        "hidden_size": 96,
        "steps": 192,
        "mutation_interval": 48,
    }
    assert spec["report_section"] == "5.51"
    assert len(spec["seeds"]) == 576
    assert spec["seeds"][0] == 87000
    assert study.DOSE2_LONG_REFERENCE == "width-d4-dose2"
    # The registered config is otherwise untouched: the treatment is the budget
    # and nothing else.
    assert study.REGISTERED["steps"] == 48
    assert study.REGISTERED["mutation_interval"] == 12
    assert study.REGISTERED["hidden_size"] == 24
    assert study.REGISTERED["mutation_budget"] == 3


def test_the_budget_scaling_keeps_the_mutational_schedule() -> None:
    """4x the steps and 4x the interval is still three events per run.

    If either factor drifts, the arm silently becomes a different experiment -
    more training *with more mutations* - and the branch table no longer reads
    the budget. The built-in checks assert 3/3/0 at verdict time; this asserts
    the arithmetic that produces it.
    """
    registered = study.REGISTERED["steps"] // study.REGISTERED["mutation_interval"]
    longer = (
        study.PROFILES[study.DOSE2_LONG_PROFILE]["overrides"]["steps"]
        // study.PROFILES[study.DOSE2_LONG_PROFILE]["overrides"]["mutation_interval"]
    )
    assert registered == longer == 4
    assert 4 * study.REGISTERED["steps"] == 192


def test_the_long_cell_block_is_fresh() -> None:
    """The report calls 87000-87575 a fresh block; check it against both lists."""
    block = set(study.PROFILES[study.DOSE2_LONG_PROFILE]["seeds"])
    assert block == set(range(87000, 87576))
    used: set[int] = set()
    for profile in study.PROFILES:
        if profile == study.DOSE2_LONG_PROFILE:
            continue
        used |= set(study.PROFILES[profile]["seeds"])
    own = Path(study.PROFILES[study.DOSE2_LONG_PROFILE]["summary"]).name
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


def test_the_primaries_are_the_two_primary_level() -> None:
    """Two primaries: alpha/2 = 0.025, bar 313/576 on the cell's own test."""
    assert _gate_min_wins(576, 0.025) == 313
    assert 0.5859 * 576 > 313  # the committed above-onset rate still clears it


def _experiment(seed: int, delta: float) -> dict:
    """One synthetic per-seed payload in the study's committed schema.

    ``delta`` moves the adaptive arm; the random arm sits at ``random_delta``.
    """
    return {
        "seed": seed,
        "losses": {"random": 10.0, "adaptive": 10.0 + delta, "null": 10.0},
        "mutations": {"random": 3, "adaptive": 3, "null": 0},
        "search_flops": {"random": 1.0, "adaptive": 1.0, "null": 0.0},
    }


def _write_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    adaptive: float,
    pool: float,
) -> None:
    """Install synthetic shards for the long cell and its 48-step reference.

    ``adaptive`` is the long cell's adaptive-vs-null delta per seed; ``pool``
    is its random-vs-null delta. The reference cell is written flat.
    """
    seeds = tuple(range(90000, 90100))
    for profile, payload_delta, payload_pool in (
        (study.DOSE2_LONG_PROFILE, adaptive, pool),
        (study.DOSE2_LONG_REFERENCE, +0.05, -2.0),
    ):
        monkeypatch.setitem(
            study.PROFILES,
            profile,
            {**study.PROFILES[profile], "seeds": seeds},
        )
        directory = tmp_path / profile
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for seed in seeds:
            row = _experiment(seed, payload_delta)
            row["losses"]["random"] = 10.0 + payload_pool
            rows.append(row)
        payload = {"shard": 0, "shards": 1, "profile": profile, "experiments": rows}
        (directory / "shard00.json").write_text(json.dumps(payload) + "\n")
    monkeypatch.setattr(study, "SHARD_DIR", tmp_path)


@pytest.mark.parametrize(
    ("adaptive", "pool", "branch"),
    [
        (-0.05, -0.1, "5.50's crossing is not a fixed-budget artifact"),
        (+0.05, -0.1, "5.50's crossing WAS the fixed-budget collapse"),
        (-0.05, -3.0, "value creation rides it"),
        (+0.05, -3.0, "neither relieves the collapse nor keeps the advantage"),
    ],
)
def test_the_four_long_budget_branches_are_fixed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    adaptive: float,
    pool: float,
    branch: str,
) -> None:
    """Every branch text is fixed in advance, and each is reachable."""
    _write_cells(tmp_path, monkeypatch, adaptive=adaptive, pool=pool)
    assert study.dose2_long_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    assert branch in out
    assert "P1 (the collapse is the budget's): rank-sum" in out


def test_the_pool_primary_reads_the_pre_registered_direction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """P1 asks whether the longer budget *relieved* the collapse, not reversed it.

    The reference's pool effect is deeply negative (-2.0); a long cell at -0.1
    is a relief and must clear the bar, while a long cell at -3.0 is worse than
    the reference and must not - the test is one-sided on purpose, because "the
    collapse persists" is the claim the arm's limit rests on.
    """
    _write_cells(tmp_path, monkeypatch, adaptive=-0.05, pool=-0.1)
    assert study.dose2_long_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    assert "-> collapse relieved: True" in out

    _write_cells(tmp_path, monkeypatch, adaptive=-0.05, pool=-3.0)
    assert study.dose2_long_verdict(1, study.ALPHA) == 0
    out = capsys.readouterr().out
    assert "-> collapse relieved: False" in out
