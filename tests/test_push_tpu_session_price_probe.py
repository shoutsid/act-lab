"""Pin the price-probe kernel's push: its accelerator, its record, and its guard.

The price probe measures what the registered session path sustains at the declared shape on a
TPU, which is the number section 5.14's option 2 is priced on. The ways this push could go
wrong are all quiet ones: ``enable_gpu`` would answer about the wrong accelerator, a record
name the entry and the fetch disagree on is a fetch that keeps a log and no price, and
republishing the code dataset under a live session is the defect that cost the campaign its
2026-09-21 session - so the publish goes through the probe's own guard, and the tests below
hold that it is *that* function being called rather than a copy of its logic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import push_tpu_probe  # noqa: E402
import push_tpu_session_price_probe as pusher  # noqa: E402
import tpu_session_price_probe as price_probe  # noqa: E402


def test_the_metadata_turns_the_tpu_on_and_the_gpu_off() -> None:
    """The probe names ``xla`` as its device; the two accelerators are exclusive on Kaggle."""
    meta = pusher.metadata(owner="someone")

    assert meta["id"] == "someone/act-tpu-session-price"
    assert meta["kernel_type"] == "script"
    assert meta["enable_tpu"] is True
    assert meta["enable_gpu"] is False
    assert meta["enable_internet"] is True
    assert meta["dataset_sources"] == ["someone/act-lab-code"]
    assert meta["is_private"] is True


def test_the_entry_runs_the_mounted_probe_and_writes_the_record_the_fetch_looks_for() -> None:
    """A script kernel gets only its own text, so these are three copies of one name.

    The probe's ``RECORD_NAME``, the path the entry passes as ``--output``, and the file the
    fetch opens after downloading - a rename on any one of them alone means a fetch that keeps
    a log and no price, which is exactly how the probe's v4 went unread for days.
    """
    assert pusher.RECORD_NAME == price_probe.RECORD_NAME
    assert pusher.RECORD_NAME in pusher.ENTRY
    assert f"/kaggle/working/{pusher.RECORD_NAME}" in pusher.ENTRY
    assert "scripts/tpu_session_price_probe.py" in pusher.ENTRY
    assert '\"--device\", \"xla\"' in pusher.ENTRY
    # The dtype is on the queued command line, not only in the probe's default: which dtype
    # a price was taken in is the first thing a reader of the kernel text needs (see DTYPE).
    assert '\"--dtype\", \"DTYPE\"' in pusher.ENTRY


def test_the_entry_bootstraps_torch_xla_the_same_way_the_other_two_do() -> None:
    """One answer to \"is torch_xla here?\", shared rather than written a third time.

    All three kernels are queued on the same platform and the bootstrap is what puts Torch-XLA
    on an image that may not carry it, so a third copy is a third way for the run to die of an
    ImportError an hour into a queue.
    """
    assert push_tpu_probe.XLA_BOOTSTRAP in pusher.ENTRY
    assert "status = ensure_xla()" in pusher.ENTRY


def test_stage_substitutes_the_steps_the_seed_and_the_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The entry is generated text with no runtime of its own, so the substitution is it."""
    monkeypatch.setattr(pusher, "WORK", tmp_path)

    folder = pusher.stage(owner="someone", steps=240, seed=price_probe.SEED)

    assert folder == tmp_path
    entry = (tmp_path / pusher.ENTRY_NAME).read_text(encoding="utf-8")
    assert "STEPS" not in entry and "SEED" not in entry and "DTYPE" not in entry
    assert "str(240)" in entry and f"str({price_probe.SEED})" in entry
    assert f'\"{price_probe.DTYPE}\"' in entry
    written = json.loads((tmp_path / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert written == pusher.metadata(owner="someone")


def test_the_publish_is_the_probes_guard_and_its_refusal_stops_the_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 5, the probe's own function, and no ``kaggle kernels push`` behind it.

    Measured 2026-09-21: the code dataset was republished at 12:31:51 while a Gate-4 session
    was mounted on it, and that session then died with no log and no output. A price
    measurement is not the critical path, so it waits - and it waits on the guard rather than
    on a copy of the guard's logic, because two copies of \"is a session live?\" is the
    failure mode.
    """
    calls: list[list[str]] = []
    monkeypatch.setattr(pusher, "stage", lambda **kwargs: tmp_path)
    monkeypatch.setattr(pusher.kaggle_queue, "code_dataset_id", lambda: "someone/act-lab-code")
    monkeypatch.setattr(pusher.push_tpu_probe, "publish_code_dataset", lambda: 5)
    monkeypatch.setattr(pusher.push_tpu_probe, "_kaggle", lambda argv: calls.append(argv) or 0)
    monkeypatch.setattr(sys, "argv", ["push_tpu_session_price_probe.py"])

    assert pusher.main() == 5
    assert calls == [], "a refused publish must not be followed by a push"


def test_the_fetch_keeps_the_record_the_log_and_the_price(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The price is the reading, so the fetch has to leave the file *and* say what it was.

    ``kernels output`` writes into a temporary directory the caller then deletes, so a fetch
    that only printed its files left nothing behind - and the prints die with the terminal.
    """
    save = tmp_path / "output"
    record = {
        "record_type": "session-price-probe",
        "device_requested": "xla",
        "completed": list(price_probe.PHASES),
        "errors": [],
        "price": price_probe.price_block({"arms": {"null": {"tokens_per_second": 1879.0}}}),
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
        sys, "argv", ["push_tpu_session_price_probe.py", "--fetch", "--save", str(save)]
    )

    assert pusher.main() == 0
    kept = save / pusher.RECORD_NAME
    assert json.loads(kept.read_text()) == record
    assert (save / "act-tpu-session-price.log").read_text() == "the log, 12 chars"
    out = capsys.readouterr().out
    assert str(kept) in out
    assert "tokens/s on one chip" in out
