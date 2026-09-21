"""Pin the probe's one rule about the campaign: never republish under a live session.

On 2026-09-21 the code dataset was republished at 12:31:51 while a Gate-4 session was
mounted on it, and that session then died with no log and no output. The timing is not
proof of cause - the log the session would have written does not exist - but the hazard is
real and the probe is the tool that publishes: a session kernel's body and the `src/` it
imports have to be one revision, and the dataset is replaced in place. So the probe waits
instead of racing, because it is a measurement and the campaign is the critical path.
"""

from __future__ import annotations

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
