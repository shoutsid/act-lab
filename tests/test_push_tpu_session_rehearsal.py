"""Pin the rehearsal kernel's push: its accelerator, its record, and its guard.

The rehearsal runs the registered session path on a TPU before any queue moves there, so the
ways this push could go wrong are all quiet ones. ``enable_gpu`` would answer about the wrong
accelerator. A record name that the entry and the fetch do not agree on is a fetch that keeps
a log and no reading. And publishing the code dataset under a live session is the defect that
cost the campaign its 2026-09-21 session - so the publish goes through the probe's own guard,
and the tests below hold that it is *that* function being called rather than a copy of its
logic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import push_tpu_probe  # noqa: E402
import push_tpu_session_rehearsal as pusher  # noqa: E402
import tpu_session_rehearsal as rehearsal  # noqa: E402


def test_the_metadata_turns_the_tpu_on_and_the_gpu_off() -> None:
    """The rehearsal names ``xla`` as its device; the two accelerators are exclusive."""
    meta = pusher.metadata(owner="someone", steps=20, seed=51)

    assert meta["id"] == "someone/act-tpu-session-rehearsal"
    assert meta["kernel_type"] == "script"
    assert meta["enable_tpu"] is True
    assert meta["enable_gpu"] is False
    assert meta["enable_internet"] is True
    assert meta["dataset_sources"] == ["someone/act-lab-code"]
    assert meta["is_private"] is True


def test_the_entry_runs_the_mounted_rehearsal_and_writes_the_record_the_fetch_looks_for() -> None:
    """A script kernel gets only its own text, so these are three copies of one name.

    The rehearsal's ``RECORD_NAME``, the path the entry passes as ``--output``, and the file
    the fetch opens after downloading - a rename on any one of them alone means a fetch that
    keeps a log and no reading, which is exactly how the probe's v4 went unread for days.
    """
    assert pusher.RECORD_NAME == rehearsal.RECORD_NAME
    assert pusher.RECORD_NAME in pusher.ENTRY
    assert f"/kaggle/working/{pusher.RECORD_NAME}" in pusher.ENTRY
    assert "scripts/tpu_session_rehearsal.py" in pusher.ENTRY
    assert '"--device", "xla"' in pusher.ENTRY


def test_the_entry_bootstraps_torch_xla_the_same_way_the_probe_does() -> None:
    """One answer to "is torch_xla here?", shared rather than written twice.

    Both kernels are queued on the same platform and the bootstrap is what puts Torch-XLA on
    an image that may not carry it, so a second copy is a second way for the run to die of an
    ImportError an hour into a queue.
    """
    assert push_tpu_probe.XLA_BOOTSTRAP in pusher.ENTRY
    assert push_tpu_probe.XLA_BOOTSTRAP in push_tpu_probe.ENTRY
    assert "status = ensure_xla()" in pusher.ENTRY


def test_stage_substitutes_the_steps_and_the_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The entry is generated text with no runtime of its own, so the substitution is it."""
    monkeypatch.setattr(pusher, "WORK", tmp_path)

    folder = pusher.stage(owner="someone", steps=4, seed=77)

    assert folder == tmp_path
    entry = (tmp_path / pusher.ENTRY_NAME).read_text(encoding="utf-8")
    assert "STEPS" not in entry and "SEED" not in entry
    assert 'str(4)' in entry and 'str(77)' in entry
    written = json.loads((tmp_path / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert written == pusher.metadata(owner="someone", steps=4, seed=77)


@pytest.mark.parametrize(
    "record, expected",
    [
        (
            {
                "device_requested": "xla",
                "completed": [],
                "errors": [{"phase": "environment", "error": "ModuleNotFoundError: no torch_xla"}],
            },
            "nothing ran (environment: ModuleNotFoundError: no torch_xla)",
        ),
        (
            {
                "device_requested": "xla",
                "completed": ["environment", "session"],
                "errors": [{"phase": "resume", "error": "RuntimeError: the step loop exploded"}],
            },
            "partial rehearsal: ran environment, session; resume, handoff did not run "
            "(resume: RuntimeError: the step loop exploded)",
        ),
        (
            {"device_requested": "xla", "completed": list(rehearsal.PHASES), "errors": []},
            "all four phases ran",
        ),
    ],
)
def test_the_verdict_keeps_nothing_ran_apart_from_a_partial_rehearsal(
    record: dict[str, object], expected: str
) -> None:
    """A run that died in the environment phase has said nothing about the session path.

    That distinction is the whole reason the phases are contained, so the one line a fetch
    prints has to carry it - reading "the hardware works" out of a record whose session never
    started would be the kind of conclusion this project treats as worse than no conclusion.
    """
    assert expected in pusher.verdict(record)


def test_the_publish_is_the_probes_guard_and_its_refusal_stops_the_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 5, the probe's own function, and no ``kaggle kernels push`` behind it.

    Measured 2026-09-21: the code dataset was republished at 12:31:51 while a Gate-4 session
    was mounted on it, and that session then died with no log and no output. The rehearsal is
    not the critical path, so it waits - and it waits on the guard rather than on a copy of
    the guard's logic, because two copies of "is a session live?" is the failure mode.
    """
    calls: list[list[str]] = []
    monkeypatch.setattr(pusher, "stage", lambda **kwargs: tmp_path)
    monkeypatch.setattr(pusher.kaggle_queue, "code_dataset_id", lambda: "someone/act-lab-code")
    monkeypatch.setattr(
        pusher.push_tpu_probe, "publish_code_dataset", lambda: 5
    )
    monkeypatch.setattr(
        pusher.push_tpu_probe, "_kaggle", lambda argv: calls.append(argv) or 0
    )
    monkeypatch.setattr(sys, "argv", ["push_tpu_session_rehearsal.py"])

    assert pusher.main() == 5
    assert calls == [], "a refused publish must not be followed by a push"


def test_the_fetch_keeps_the_record_the_log_and_the_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The half the probe's fetch was missing until 2026-09-25: keep what you downloaded.

    ``kernels output`` writes into a temporary directory the caller then deletes, so a fetch
    that only printed its files left nothing behind - and the prints die with the terminal.
    The rehearsal's record is the reading, so this pins the three things the real fetch has to
    leave: the file, the log beside it, and the verdict for the record that just landed.
    """
    save = tmp_path / "output"
    record = {
        "record_type": "session-rehearsal",
        "device_requested": "xla",
        "completed": list(rehearsal.PHASES),
        "errors": [],
    }

    def fake_kaggle(argv: list[str]) -> int:
        directory = Path(argv[argv.index("-p") + 1])
        (directory / pusher.RECORD_NAME).write_text(json.dumps(record))
        return 0

    monkeypatch.setattr(pusher, "stage", lambda **kwargs: tmp_path)
    monkeypatch.setattr(pusher.kaggle_queue, "code_dataset_id", lambda: "someone/act-lab-code")
    monkeypatch.setattr(pusher.push_tpu_probe, "_kaggle", fake_kaggle)
    monkeypatch.setattr(pusher.kaggle_queue, "fetch_kernel_log", lambda kernel: "the log, 12 chars")
    monkeypatch.setattr(
        sys, "argv", ["push_tpu_session_rehearsal.py", "--fetch", "--save", str(save)]
    )

    assert pusher.main() == 0
    kept = save / pusher.RECORD_NAME
    assert json.loads(kept.read_text()) == record
    assert (save / "act-tpu-session-rehearsal.log").read_text() == "the log, 12 chars"
    out = capsys.readouterr().out
    assert str(kept) in out
    assert "all four phases ran" in out


def test_the_watcher_fetches_on_both_terminal_states() -> None:
    """An ERROR run is not an empty one, so the watcher must not only fetch on COMPLETE.

    The rehearsal writes its record after every phase, so a run that died inside the session
    still carries the environment phase's reading and the exception that stopped the rest -
    which is the whole reason the phases are contained, and the reason this branch exists.
    """
    text = (ROOT / "scripts" / "fetch_tpu_session_rehearsal_when_done.sh").read_text(
        encoding="utf-8"
    )
    assert "--fetch --save" in text
    assert "*COMPLETE*" in text
    assert "*ERROR* | *CANCELLED*" in text
    assert pusher.KERNEL_SLUG in text
