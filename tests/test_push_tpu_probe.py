"""Pin the probe's rules about the campaign and about its own record.

On 2026-09-21 the code dataset was republished at 12:31:51 while a Gate-4 session was
mounted on it, and that session then died with no log and no output. The timing is not
proof of cause - the log the session would have written does not exist - but the hazard is
real and the probe is the tool that publishes: a session kernel's body and the `src/` it
imports have to be one revision, and the dataset is replaced in place. So the probe waits
instead of racing, because it is a measurement and the campaign is the critical path.

And the fetch keeps what it fetches. On 2026-09-25 the probe's v4 kernel was found to have
finished `COMPLETE` three days earlier with nobody reading it: `--fetch` printed the record
into a temporary directory and deleted it, so the copy on disk stayed v3's, the rule
written for that record was never applied, and the run had to be diagnosed from a kernel log
instead. Both halves of that - the keep, and the applying - are pinned below.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import kaggle_queue  # noqa: E402
import kaggle_t4  # noqa: E402
import push_tpu_probe  # noqa: E402


def test_no_live_session_means_no_objection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kaggle_queue, "active_status", lambda kernel: "COMPLETE")
    assert push_tpu_probe.live_campaign_kernel() is None


@pytest.mark.parametrize("status", ["RUNNING", "QUEUED", "NEW_SCRIPT"])
def test_a_live_session_is_named(monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    """The refusal has to name the session, because the fix is to wait for that one.

    All three campaign queues are checked, so this pins the session by name rather than
    assuming it is the one that answers first.
    """
    session = str(kaggle_t4.metadata_for("session")["id"])
    monkeypatch.setattr(
        kaggle_queue,
        "active_status",
        lambda kernel: status if kernel == session else "COMPLETE",
    )
    live = push_tpu_probe.live_campaign_kernel()
    assert live is not None
    assert "is " + status in live
    assert session in live


def test_the_publish_refuses_and_never_reaches_the_dataset_api(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 5, and ``ensure_code_dataset`` is never called - the publish is the hazard."""
    published: list[object] = []
    monkeypatch.setattr(kaggle_queue, "active_status", lambda kernel: "RUNNING")
    monkeypatch.setattr(
        kaggle_queue, "ensure_code_dataset", lambda *a, **k: published.append(a) or "rev"
    )

    assert push_tpu_probe.publish_code_dataset() == 5
    assert published == [], "republishing is the hazard; the guard must not reach it"
    assert "refusing to publish" in capsys.readouterr().err


def test_the_publish_proceeds_when_nothing_is_live(monkeypatch: pytest.MonkeyPatch) -> None:
    published: list[object] = []
    monkeypatch.setattr(kaggle_queue, "active_status", lambda kernel: "COMPLETE")
    monkeypatch.setattr(
        kaggle_queue, "ensure_code_dataset", lambda *a, **k: published.append(a) or "abc123"
    )
    assert push_tpu_probe.publish_code_dataset() is None
    assert published, "with no session live there is nothing to wait for"


def test_the_fetch_keeps_the_record_it_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The loss this was fixed for: a `COMPLETE` run whose record was printed and deleted.

    The fake platform is the smallest thing that reproduces it - ``kernels output`` writing
    a record into the directory the fetch then removes - and the assertions are the three
    the real fetch has to leave behind: the file, the log beside it, and the rule's verdict
    for the record that just landed.
    """
    save = tmp_path / "output"
    record = {
        "bf16": {
            "rebuild": {
                "first_step_seconds_including_compile": 29.2,
                "steady_step_seconds_mean": 10.8,
            }
        },
        "independent_seeds": {"workers": 8, "aggregate_tokens_per_second": 12_000.0},
    }

    def fake_kaggle(argv: list[str]) -> int:
        directory = Path(argv[argv.index("-p") + 1])
        (directory / push_tpu_probe.RECORD_NAME).write_text(json.dumps(record))
        return 0

    monkeypatch.setattr(push_tpu_probe, "stage", lambda **kwargs: tmp_path)
    monkeypatch.setattr(push_tpu_probe, "_kaggle", fake_kaggle)
    monkeypatch.setattr(kaggle_queue, "fetch_kernel_log", lambda kernel: "the log, 12 chars")
    monkeypatch.setattr(kaggle_queue, "code_dataset_id", lambda: "someone/act-lab-code")
    monkeypatch.setattr(
        sys, "argv", ["push_tpu_probe.py", "--fetch", "--save", str(save)]
    )

    assert push_tpu_probe.main() == 0
    kept = save / push_tpu_probe.RECORD_NAME
    assert json.loads(kept.read_text()) == record
    assert (save / "act-tpu-probe.log").read_text() == "the log, 12 chars"
    out = capsys.readouterr().out
    assert str(kept) in out
    # The rule, applied where the record landed. The record above is a pass on both
    # conditions the pre-registration names, so this pins the verdict rather than a call.
    assert "PORT" in out


def test_the_record_name_is_the_one_the_kernel_writes() -> None:
    """A script kernel gets only its entry text, so the path is a literal there and a
    constant here; a rename on one side alone is a fetch that finds a log and no measurement."""
    assert push_tpu_probe.RECORD_NAME in push_tpu_probe.ENTRY
    assert f"/kaggle/working/{push_tpu_probe.RECORD_NAME}" in push_tpu_probe.ENTRY
