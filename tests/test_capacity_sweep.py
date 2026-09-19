"""Tests for the capacity-threshold sweep (scripts/capacity_sweep.py).

Three defects, all of the same shape: a step that did something other than what it
said. The script read no arguments at all, so ``--help`` started a full sweep and a
misspelled flag was silently dropped while the default run went ahead - which is how
the unguarded run was discovered, by running ``--help`` to see the flags. The reset
that opens the sweep named ``<scale>-seed-<seed>`` while every run writes
``<scale>-seed-<seed>-<arm>``, so it reset nothing: ``run_v0_session`` *resumes* from
an existing checkpoint directory, so each re-run continued from whatever the previous
one had left behind. And the recorded arm was read as ``result.arms[0]`` while the
three probe-selection arms are all configured as ``("random", "adaptive")``, so the
``adaptive`` and ``anti`` rows carried the ``random`` arm's numbers.

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


# A loss that identifies which arm a row came from, so a row recording the wrong arm
# is readable rather than needing two runs to compare.
_MODE_LOSS = {"null": 1.0, "random": 2.0, "adaptive": 3.0}


@dataclass
class _Arm:
    """The arm fields ``run_one`` reads off a session result."""

    mode: str = "null"
    final_loss: float = 1.0
    mutation_count: int = 0
    search_flops: float = 0.0
    training_flops: float = 0.0


@dataclass
class _Session:
    arms: list[_Arm] = field(default_factory=lambda: [_Arm()])


def _patch_session(monkeypatch, seen: list[str], modes: list[str] | None = None) -> None:
    """Record the checkpoint directory each run is handed, without training.

    The fake mirrors the real contract in the way that matters here: it returns one
    arm per ``config.arms``, in that order, each carrying its own ``mode``.
    """

    def fake_session(config, *, checkpoint_dir, **kwargs):
        seen.append(str(checkpoint_dir))
        wanted = modes if modes is not None else list(config.arms)
        return _Session(
            arms=[_Arm(mode=mode, final_loss=_MODE_LOSS[mode]) for mode in wanted]
        )

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


def test_each_sweep_arm_records_the_arm_it_configures(tmp_path, monkeypatch):
    """The defect: ``adaptive`` and ``anti`` recorded the random arm.

    Every probe-selection arm is configured as ``("random", "adaptive")``, so
    ``result.arms[0]`` is the plain random-churn arm for all of them and the
    ``probe_selection`` that distinguishes them governs ``arms[1]``. The sweep's own
    scratch output showed the signature - ``random``, ``adaptive`` and ``anti``
    bit-identical at every scale.
    """
    _patch_reset(monkeypatch, [])
    _patch_session(monkeypatch, [])
    _run(
        monkeypatch,
        ["--scales", "20M", "--arms", "null,random,adaptive,anti", "--seeds", "42",
         "--steps", "1", "--out", str(tmp_path)],
    )

    capacity_sweep.main()

    written = json.loads((tmp_path / "threshold_sweep.json").read_text())
    recorded = {row["arm"]: row["loss"] for row in written}
    assert recorded == {
        "null": _MODE_LOSS["null"],
        "random": _MODE_LOSS["random"],
        # Both report the adaptive arm - the one probe_selection governs.
        "adaptive": _MODE_LOSS["adaptive"],
        "anti": _MODE_LOSS["adaptive"],
    }


def test_the_arm_mode_map_covers_every_configured_arm() -> None:
    """A new arm cannot be added without saying which session arm it reports."""
    assert set(capacity_sweep.ARM_MODE) == set(capacity_sweep.ARM_CONFIGS)


def test_an_arm_the_session_did_not_return_is_an_error(tmp_path, monkeypatch, capsys):
    """A missing mode must fail loudly, not fall back to whatever arm is first.

    ``main`` reports a failed run and carries on, so the run-level error is printed;
    what must not happen is a results file written from nothing, which would read as a
    sweep whose arm was merely quiet. An empty set of results is refused outright.
    """
    _patch_reset(monkeypatch, [])
    # The session returns only the plain random arm, so the adaptive arm the sweep
    # asked for is absent.
    _patch_session(monkeypatch, [], modes=["random"])
    _run(
        monkeypatch,
        ["--scales", "20M", "--arms", "adaptive", "--seeds", "42", "--steps", "1",
         "--out", str(tmp_path)],
    )

    with pytest.raises(SystemExit) as exit_info:
        capacity_sweep.main()

    assert "nothing to write" in str(exit_info.value)
    reported = capsys.readouterr().out
    assert "FAILED" in reported and "'adaptive'" in reported and "'random'" in reported
    assert not (tmp_path / "threshold_sweep.json").exists()


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
