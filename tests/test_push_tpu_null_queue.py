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
