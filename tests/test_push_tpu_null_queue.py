"""Pin the null-queue kernel's push: its accelerator, its command, its record and its guard.

This is the only kernel the campaign pushes that runs a **queue** rather than a measurement, so
the ways its push could go wrong are the expensive ones: ``enable_gpu`` would answer about the
wrong accelerator with twelve real-looking seeds, a dataset or a step budget that never reached
the entry text is a placement a reader of the queued run cannot check, and republishing the code
dataset under a live session is the defect that cost the campaign its 2026-09-21 session - so the
publish goes through the probe's own guard, and the tests below hold that it is *that* function
being called rather than a copy of its logic.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import push_tpu_probe  # noqa: E402
import push_tpu_null_queue as pusher  # noqa: E402
import tpu_null_queue  # noqa: E402


def test_the_metadata_turns_the_tpu_on_and_the_gpu_off() -> None:
    """The runner names ``xla`` as its device; the two accelerators are exclusive on Kaggle."""
    meta = pusher.metadata(owner="someone")

    assert meta["id"] == "someone/act-tpu-null-queue"
    assert meta["kernel_type"] == "script"
    assert meta["enable_tpu"] is True
    assert meta["enable_gpu"] is False
    assert meta["enable_internet"] is True
    assert meta["dataset_sources"] == ["someone/act-lab-code"]
    assert meta["is_private"] is True


def test_the_entry_runs_the_mounted_runner_and_writes_the_record_the_fetch_looks_for() -> None:
    """A script kernel gets only its own text, so these are three copies of one name.

    The runner's ``RECORD_NAME``, the path the entry passes as ``--output``, and the file the
    fetch opens after downloading - a rename on any one of them alone means a fetch that keeps a
    log and no queue, which is how the probe's v4 went unread for three days.
    """
    assert pusher.RECORD_NAME == tpu_null_queue.RECORD_NAME
    assert pusher.RECORD_NAME in pusher.ENTRY
    assert f"/kaggle/working/{pusher.RECORD_NAME}" in pusher.ENTRY
    assert "scripts/tpu_null_queue.py" in pusher.ENTRY


def test_the_entry_runs_the_registered_command_at_the_declared_fidelity() -> None:
    """What the kernel substitutes is the registration, and the entry text is where it is read."""
    assert '"--device", "xla"' in pusher.ENTRY
    assert '"--fidelity", "declared"' in pusher.ENTRY
    assert '"--seeds", "SEED_RANGE"' in pusher.ENTRY
    assert '"--dataset", "DATASET"' in pusher.ENTRY
    # The placement is on the queued command line: eight seeds, one per chip, is the whole
    # reason this queue fits a week's clock at all.
    assert '"--workers", str(WORKERS)' in pusher.ENTRY


def test_the_entry_bootstraps_torch_xla_the_same_way_the_other_three_do() -> None:
    """One answer to "is torch_xla here?", shared rather than written a fourth time."""
    assert push_tpu_probe.XLA_BOOTSTRAP in pusher.ENTRY
    assert "status = ensure_xla()" in pusher.ENTRY


def test_the_entry_installs_a_working_kaggle_client_before_the_runner_asks_for_one() -> None:
    """The startup restore and the exit publish are dataset API calls, and the image's client is
    not importable as shipped.

    Measured 2026-09-30: the queue's first fire reached the runner with `find_spec("kaggle")`
    true, then died at restore (`errors: ["restore"]`, `seeds_completed: 0`) on
    `ModuleNotFoundError: No module named 'kaggle.api'` - 59 s in, no seed trained. A script
    kernel is staged from its own text, so the fix has to be in that text, and it has to run
    before `runpy.run_path`, because the runner imports the client at restore time.
    """
    assert push_tpu_probe.KAGGLE_BOOTSTRAP in pusher.ENTRY
    call = "kaggle_status = ensure_kaggle()"
    assert call in pusher.ENTRY
    assert pusher.ENTRY.index(call) < pusher.ENTRY.index("runpy.run_path")


def test_the_kaggle_bootstrap_probes_the_import_rather_than_the_name() -> None:
    """A present-but-broken release and a working one share the name `kaggle`, so the name is not
    the check - only the import is.

    And it upgrades rather than installs: measured 2026-09-30, `pip install kaggle` on an image
    that already carries a broken version reports "Requirement already satisfied" and changes
    nothing, so a plain install would have left this failure exactly where it was.
    """
    assert "from kaggle.api.kaggle_api_extended import KaggleApi" in push_tpu_probe.KAGGLE_BOOTSTRAP
    assert '"-U", "kaggle"' in push_tpu_probe.KAGGLE_BOOTSTRAP


def test_stage_substitutes_every_field_the_queued_run_is_read_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The entry is generated text with no runtime of its own, so the substitution is it."""
    monkeypatch.setattr(pusher, "WORK", tmp_path)

    folder = pusher.stage(
        owner="someone",
        seeds="15024-15035",
        steps=6000,
        workers=8,
        dataset="someone/act-v0-5-2-null",
    )

    assert folder == tmp_path
    entry = (tmp_path / pusher.ENTRY_NAME).read_text(encoding="utf-8")
    for placeholder in ("SEED_RANGE", "STEPS", "WORKERS", "DATASET"):
        assert placeholder not in entry
    assert '"15024-15035"' in entry
    assert "str(6000)" in entry and "str(8)" in entry
    assert '"someone/act-v0-5-2-null"' in entry
    written = json.loads((tmp_path / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert written == pusher.metadata(owner="someone")


def test_the_default_dataset_is_the_registered_null_queue_s_own(tmp_path: Path) -> None:
    """The block has to land where the campaign's null clause already looks for it."""
    assert tpu_null_queue.DATASET_ROOT == "act-v0-5-2-null"


def test_the_publish_is_the_probes_guard_and_its_refusal_stops_the_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 5, the probe's own function, and no ``kaggle kernels push`` behind it.

    Measured 2026-09-21: the code dataset was republished at 12:31:51 while a Gate-4 session was
    mounted on it, and that session then died with no log and no output. A queue push is not the
    critical path, so it waits - and it waits on the guard rather than on a copy of the guard's
    logic, because two copies of "is a session live?" is the failure mode.
    """
    calls: list[list[str]] = []
    monkeypatch.setattr(pusher, "stage", lambda **kwargs: tmp_path)
    monkeypatch.setattr(pusher.kaggle_queue, "code_dataset_id", lambda: "someone/act-lab-code")
    monkeypatch.setattr(pusher.push_tpu_probe, "publish_code_dataset", lambda: 5)
    monkeypatch.setattr(pusher.push_tpu_probe, "_kaggle", lambda argv: calls.append(argv) or 0)
    monkeypatch.setattr(sys, "argv", ["push_tpu_null_queue.py"])

    assert pusher.main() == 5
    assert calls == [], "a refused publish must not be followed by a push"


def test_a_push_that_never_left_is_not_reported_as_fired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI exits **0** on this failure, so the fired/not-fired answer is its own line.

    Measured 2026-09-29: with another TPU kernel holding the account's single TPU session,
    `kaggle kernels push` printed `Kernel push error: Maximum batch TPU session count of 1
    reached.` and returned 0. A caller that read the exit code - which is what this script did
    first, and what the slot waiter did too - announced a queue that never left, and armed a
    fetch watcher against a kernel that did not exist.
    """
    refused = "Kernel push error: Maximum batch TPU session count of 1 reached.\n"
    monkeypatch.setattr(
        pusher.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, refused, ""),
    )

    assert pusher.push_kernel(tmp_path) == 6
    assert "nothing was queued" in capsys.readouterr().err

    fired = "Kernel version 1 successfully pushed.  Please check progress at https://x\n"
    monkeypatch.setattr(
        pusher.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, fired, ""),
    )

    assert pusher.push_kernel(tmp_path) == 0


def test_the_slot_waiter_pushes_what_was_staged_and_publishes_nothing_itself() -> None:
    """Kaggle allows one TPU session at a time, so the queue's push can be refused by a slot.

    The refusal is a platform event rather than a mistake, so it is waited out - and the wait
    pushes the folder the pusher already staged instead of re-running it: re-running would
    republish the code dataset at HEAD, which would let a later commit become the revision the
    kernel mounts and the record names.
    """
    text = (ROOT / "scripts" / "push_tpu_null_queue_when_free.sh").read_text(encoding="utf-8")

    assert f"{pusher.KERNEL_SLUG}" in text
    assert "kaggle kernels push -p" in text
    assert "fetch_tpu_session_rehearsal_when_done.sh" in text
    # The pusher is named exactly once, as the fetch watcher's PUSHER: it is never *run* here,
    # because running it would republish the code dataset at HEAD on the way to the push.
    assert text.count("push_tpu_null_queue.py") == 1
    assert "PUSHER=scripts/push_tpu_null_queue.py" in text
    assert "ensure_code_dataset" not in text
    assert "fetch_tpu_session_rehearsal_when_done.sh" in text
    # And it cannot be fooled by a refused push, because the CLI exits 0 on one: the waiter reads
    # the success line, the same answer `push_kernel` reads.
    assert 'successfully pushed' in text
    # Bounded, and the slot's holder is named in the log rather than left to be guessed.
    assert "MAX_POLLS" in text and "gave up" in text
    assert "BLOCKER" in text and "tpu-session-price" in text
    # And the fetch watcher is armed with a *longer* bound than the slot waiter's own. Measured
    # 2026-10-01: the kernel was still `QUEUED` 25 h after its push, past the 24 h the fetch
    # watcher was armed with, so it gave up on a run that had not started and the fetch would have
    # had to be noticed by hand. The two bounds wait on different things - a slot that frees
    # quickly, a queue that is a platform-capacity event - so they are held apart by parsing both
    # off the shell rather than by pinning either to a number.
    assert 'POLL_SECONDS="$FETCH_POLL_SECONDS" MAX_POLLS="$FETCH_MAX_POLLS"' in text

    def declared_bound(seconds_name: str, polls_name: str) -> int:
        """One poll loop's bound in seconds, read off the shell that declares it."""

        def declared(name: str) -> int:
            found = re.search(rf"{name}=\$\{{{name}:-([0-9]+)\}}", text)
            assert found is not None, f"{name} is not declared"
            return int(found.group(1))

        return declared(seconds_name) * declared(polls_name)

    assert declared_bound("POLL_SECONDS", "MAX_POLLS") < declared_bound(
        "FETCH_POLL_SECONDS", "FETCH_MAX_POLLS"
    )


def test_the_fetch_keeps_the_record_the_log_and_the_queue_s_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reading is why the record is fetched, so the fetch has to leave the file *and* say it.

    ``kernels output`` writes into a temporary directory the caller then deletes, so a fetch that
    only printed its files left nothing behind - and the prints die with the terminal.
    """
    save = tmp_path / "output"
    record = {
        "record_type": "tpu-null-queue",
        "fidelity": "declared",
        "dtype": "fp32",
        "seeds_requested": [15024],
        "seeds_completed": 1,
        "errors": [],
        "seeds": {
            "15024": {"arm": {"final_loss": 10.3972, "search_flops": 0, "mutation_count": 0,
                              "tokens_per_second": 1879.0}}
        },
    }

    def fake_kaggle(argv: list[str]) -> int:
        directory = Path(argv[argv.index("-p") + 1])
        (directory / pusher.RECORD_NAME).write_text(json.dumps(record))
        return 0

    monkeypatch.setattr(pusher, "stage", lambda **kwargs: tmp_path)
    monkeypatch.setattr(pusher.kaggle_queue, "code_dataset_id", lambda: "someone/act-lab-code")
    monkeypatch.setattr(pusher.push_tpu_probe, "_kaggle", fake_kaggle)
    monkeypatch.setattr(
        pusher.kaggle_queue, "fetch_kernel_log", lambda kernel: "the log, 12 chars"
    )
    monkeypatch.setattr(
        sys, "argv", ["push_tpu_null_queue.py", "--fetch", "--save", str(save)]
    )

    assert pusher.main() == 0
    kept = save / pusher.RECORD_NAME
    assert json.loads(kept.read_text()) == record
    assert (save / "act-tpu-null-queue.log").read_text() == "the log, 12 chars"
    out = capsys.readouterr().out
    assert str(kept) in out
    assert "1 of 1 seed(s) landed in fp32 on XLA" in out
