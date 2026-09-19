"""Tests for the capacity-threshold sweep (scripts/capacity_sweep.py).

Two defects, both of the same shape: a step that did something other than what it
said. The script read no arguments at all, so ``--help`` started a full sweep and a
misspelled flag was silently dropped while the default run went ahead - which is how
the unguarded run was discovered, by running ``--help`` to see the flags. And the
reset that opens the sweep named ``<scale>-seed-<seed>`` while every run writes
``<scale>-seed-<seed>-<arm>``, so it reset nothing: ``run_v0_session`` *resumes* from
an existing checkpoint directory, so each re-run continued from whatever the previous
one had left behind.

These are tested rather than only fixed because the sweep is not covered by a study
summary, so nothing else reads it.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import capacity_sweep  # noqa: E402


@dataclass
class _Arm:
    """The arm fields ``run_one`` reads off a session result."""

    final_loss: float = 1.0
    mutation_count: int = 0
    search_flops: float = 0.0
    training_flops: float = 0.0


@dataclass
class _Session:
    arms: list[_Arm] = field(default_factory=lambda: [_Arm()])


def _patch_session(monkeypatch, seen: list[str]) -> None:
    """Record the checkpoint directory each run is handed, without training."""

    def fake_session(config, *, checkpoint_dir, **kwargs):
        seen.append(str(checkpoint_dir))
        return _Session()

    monkeypatch.setattr(capacity_sweep, "run_v0_session", fake_session)


def _patch_reset(monkeypatch, reset: list[str]) -> None:
    monkeypatch.setattr(shutil, "rmtree", lambda path, **kwargs: reset.append(str(path)))


def _run(monkeypatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["capacity_sweep.py", *argv])


def test_help_prints_the_flags_instead_of_running_a_sweep(tmp_path, capsys, monkeypatch):
    """``--help`` used to begin a multi-hour sweep; that is how this was found."""
    reset: list[str] = []
    _patch_reset(monkeypatch, reset)
    _run(monkeypatch, ["--help", "--out", str(tmp_path)])

    with pytest.raises(SystemExit) as exit_info:
        capacity_sweep.main()

    assert exit_info.value.code == 0
    printed = capsys.readouterr().out
    assert "usage: capacity_sweep.py" in printed
    assert "--scales" in printed and "--steps" in printed
    # Nothing was reset and nothing was run: help is help.
    assert reset == []


def test_a_flag_it_does_not_understand_is_refused(tmp_path, monkeypatch):
    """A dropped flag is worse than a rejected one - it runs the default silently."""
    _run(monkeypatch, ["--bogus", "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as exit_info:
        capacity_sweep.main()
    assert exit_info.value.code != 0


def test_an_unknown_scale_or_arm_is_named_not_dropped(tmp_path, monkeypatch):
    _patch_reset(monkeypatch, [])
    for flag, value, choices in (
        ("--scales", "500M", "20M"),
        ("--arms", "sideways", "adaptive"),
    ):
        _run(monkeypatch, [flag, value, "--out", str(tmp_path)])
        with pytest.raises(SystemExit) as exit_info:
            capacity_sweep.main()
        message = str(exit_info.value)
        assert value in message and choices in message


def test_an_empty_selection_is_refused_rather_than_run_as_nothing(tmp_path, monkeypatch):
    _patch_reset(monkeypatch, [])
    _run(monkeypatch, ["--seeds", "", "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as exit_info:
        capacity_sweep.main()
    assert "no seed given" in str(exit_info.value)


def test_the_reset_covers_the_directory_the_runs_actually_write(tmp_path, monkeypatch):
    """The defect: reset named ``<scale>-seed-<seed>``, runs wrote ``...-<arm>``.

    ``run_v0_session`` resolves its checkpoint directory to
    ``<checkpoint_dir>/seed-<seed>`` and resumes from what it finds there, so the old
    name reset nothing and every re-run continued from the previous one's weights.
    The reset target therefore has to be the directory the run is handed, and this
    pins the two together rather than pinning either string alone.
    """
    reset: list[str] = []
    seen: list[str] = []
    _patch_reset(monkeypatch, reset)
    _patch_session(monkeypatch, seen)
    _run(
        monkeypatch,
        ["--scales", "20M", "--arms", "null", "--seeds", "42", "--steps", "1",
         "--out", str(tmp_path)],
    )

    capacity_sweep.main()

    assert seen == ["artifacts/v0-sessions/20M-seed-42-null"]
    assert reset == seen, "the reset must name the directory the run writes"

    written = json.loads((tmp_path / "threshold_sweep.json").read_text())
    assert [(row["scale"], row["seed"], row["arm"]) for row in written] == [
        ("20M", 42, "null")
    ]


def test_only_the_selected_runs_are_reset_and_reported(tmp_path, monkeypatch):
    """A selection must not reset, run, or report the axis it did not ask for."""
    reset: list[str] = []
    seen: list[str] = []
    _patch_reset(monkeypatch, reset)
    _patch_session(monkeypatch, seen)
    _run(
        monkeypatch,
        ["--scales", "20M", "--arms", "null,random", "--seeds", "42,43", "--steps", "1",
         "--out", str(tmp_path)],
    )

    capacity_sweep.main()

    assert set(reset) == set(seen)
    assert set(reset) == {
        f"artifacts/v0-sessions/20M-seed-{seed}-{arm}"
        for seed in (42, 43)
        for arm in ("null", "random")
    }
