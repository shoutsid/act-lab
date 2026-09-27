"""Pin the TPU session rehearsal's labels, its guards, and its record shape.

``scripts/tpu_session_rehearsal.py`` runs the registered V0 session path on the device the
caller names - ``xla`` on a TPU host, ``cpu`` for a dry run - before any queue moves there.
It is queued once and can wait hours, so two properties are load bearing and both are
pinned here. **A phase files its own entry and log before it starts**, because the run is
saved after each phase and a phase that dies mid-step would otherwise leave an exception
text and nothing else. And **the record is labelled**, because a rehearsal that reads as a
port study would be worse than no rehearsal: the labels below are what keeps ``declared_scale``
false, ``fidelity`` development, and the probe's per-core pattern named as *not measured
here*.

The end-to-end case runs the real path at one step on the CPU, which is the only dry run
that can exist away from a TPU - it is the shape a fired run produces, minus the hardware.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import tpu_session_rehearsal as rehearsal  # noqa: E402


def _args(**overrides) -> argparse.Namespace:
    settings = {"device": "cpu", "steps": 2, "seed": 51}
    settings.update(overrides)
    return argparse.Namespace(**settings)


def test_the_labels_say_rehearsal_and_name_what_is_not_measured() -> None:
    """``declared_scale`` false is the field that stops a pass reading as a port study.

    Measured 2026-09-27, when the rehearsal was written: the probe had already returned
    PORT on the declared shape, and the thing left unmeasured was the *session path*, so a
    record from this script has to say which question it answers and which it does not.
    """
    record = rehearsal.base_record(_args(device="xla"))

    assert record["record_type"] == "session-rehearsal"
    assert record["kind"] == "tpu-session-rehearsal"
    assert record["rehearsal"] is True
    assert record["fidelity"] == "development"
    assert record["declared_scale"] is False
    assert record["device_requested"] == "xla"
    assert record["completed"] == [] and record["errors"] == []
    assert "not a Gate-4 result" in record["what_this_is"]

    not_measured = " ".join(record["not_measured"])
    assert len(record["not_measured"]) == len(rehearsal.NOT_MEASURED)
    # The probe's own headline (eight seeds, one per chip) is explicitly *not* this
    # measurement, so a reader cannot take the rehearsal as a second reading of it.
    assert "per-core parallelism" in not_measured
    assert "two-arm pair placement" in not_measured


def test_the_dev_config_is_the_single_arm_null_queue_at_dev_fidelity() -> None:
    """The queue section 5.14's amendment moves is the one-arm null queue.

    ``declared=False`` is deliberate rather than incidental: the probe already measured the
    declared shape's throughput, and this is a rehearsal of the *path*, where twenty steps
    find a fresh XLA failure as well as six thousand do.
    """
    config = rehearsal._dev_null_config("xla", steps=20, seed=51)

    assert config.declared is False
    assert config.arms == ("null",)
    assert config.device == "xla"
    assert config.total_steps == 20
    assert 1 <= config.checkpoint_interval <= 20
    assert config.max_checkpoints >= 2


def test_the_phases_run_in_the_order_a_failure_would_cost() -> None:
    """Session first, resume second, handoff last, and nothing after the environment.

    The session is what everything else needs, the resume is the first contract XLA can
    break, and the handoff needs a result to publish - so the order is the cost of a
    failure, not the order the names read in.
    """
    assert rehearsal.PHASES == ("environment", "session", "resume", "handoff")


def test_a_phase_files_its_entry_and_log_before_the_work_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure this shape was fixed for: a run that dies mid-step and says nothing.

    ``tpu_probe._run_phase`` saves the record *after* the phase returns, so a phase that
    filled its entry at the end would leave a queued run's whole account as one exception
    string. The log is the list the session writes into, and it is the record's list - which
    is the property this holds: the entry and the list are already in place when the first
    step runs.
    """
    import act.training.v0_session as session_module

    record = rehearsal.base_record(_args(seed=5))
    observed: dict[str, object] = {}

    def failing_session(config, **kwargs):
        observed["entry"] = dict(record["session"])
        kwargs["log"]("a line the session printed before it died")
        raise RuntimeError("the step loop exploded")

    monkeypatch.setattr(session_module, "run_v0_session", failing_session)
    with pytest.raises(RuntimeError, match="exploded"):
        rehearsal.phase_session(record, tmp_path, device="cpu", steps=2, seed=5)

    entry = observed["entry"]
    assert tuple(entry["config"]["arms"]) == ("null",)
    assert entry["config"]["device"] == "cpu"
    assert entry["code_revision"] == rehearsal._revision()
    # The line the session printed before it died is in the record `_run_phase` will save.
    assert record["session"]["log"] == ["a line the session printed before it died"]


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda summary, manifest: None, None),
        (lambda summary, manifest: summary.update(status="in_progress"), "not 'complete'"),
        (lambda summary, manifest: summary["arms"]["null"].update(search_flops=4), "search_flops=4"),
        (
            lambda summary, manifest: summary["arms"]["null"].update(mutation_count=2),
            "mutation_count=2",
        ),
        (lambda summary, manifest: summary["arms"].pop("null"), "no arms.null block"),
        (lambda summary, manifest: manifest.update(code_revision="other"), "does not carry"),
    ],
)
def test_the_null_clause_is_recomputed_from_the_published_artifacts(mutate, expected) -> None:
    """These are exactly what ``v0_verdict._null_block`` refuses a seed for.

    A rehearsal that published a mutating "null" arm would be reporting a queue the verdict
    cannot read, so the check runs on the artifacts rather than on the rehearsal's own
    variables - and it *names* the defect, because the run that produces it is queued once.
    """
    summary = {
        "status": "complete",
        "arms": {
            "null": {
                "search_flops": 0,
                "mutation_count": 0,
                "final_loss": 10.4,
                "validation_loss": 10.4,
            }
        },
    }
    manifest = {"code_revision": "rev"}
    mutate(summary, manifest)

    defect = rehearsal.null_clause_defect(summary, manifest, "rev")
    if expected is None:
        assert defect is None
    else:
        assert defect is not None and expected in defect


def test_a_cpu_dry_run_produces_the_record_a_fired_run_would(tmp_path: Path) -> None:
    """The end-to-end shape, on the only device this host has.

    The environment phase cannot run without ``torch_xla`` - that is the one phase whose
    outcome depends on the host - so the exit status is asserted *conditionally* rather
    than fixed. Everything the later phases prove is not: the session trains and writes a
    **host** payload carrying this revision, the resume restores it, and the handoff
    publishes a summary whose null arm books no search and no mutation.
    """
    output = tmp_path / "tpu-session-rehearsal.json"
    code = rehearsal.main(["--device", "cpu", "--steps", "1", "--seed", "7",
                           "--output", str(output)])

    record = json.loads(output.read_text(encoding="utf-8"))
    try:
        import torch_xla  # noqa: F401
    except ImportError:
        assert code == 1, "a phase that cannot run must not read as a complete rehearsal"
        assert [entry["phase"] for entry in record["errors"]] == ["environment"]
        assert "torch_xla" in record["errors"][0]["error"]
    else:  # pragma: no cover - a TPU host would take this branch
        assert code == 0, record["errors"]
        assert record["errors"] == []

    assert record["completed"] == ["session", "resume", "handoff"]
    assert record["declared_scale"] is False
    assert record["session"]["checkpoint_tensor_devices"] == ["cpu"]
    assert record["session"]["checkpoint_revision"] == rehearsal._revision()
    assert record["resume"]["resumed_from_a_checkpoint"] is True
    assert record["resume"]["same_loss_as_the_first_run"] is True
    assert record["handoff"]["summary_status"] == "complete"
    assert record["handoff"]["manifest_code_revision"] == rehearsal._revision()
    assert record["handoff"]["null_arm"]["search_flops"] == 0
    assert record["handoff"]["null_arm"]["mutation_count"] == 0
    # The published seed is a real one under a dataset root, not a name in the record.
    assert any(
        name.endswith("seed-summary.json") for name in record["handoff"]["seed_files"]
    )
