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
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import tpu_session_rehearsal as rehearsal  # noqa: E402


def _args(**overrides) -> argparse.Namespace:
    settings = {
        "device": "cpu",
        "steps": 2,
        "seed": 51,
        "activation_checkpointing": False,
    }
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


def test_the_checkpointed_path_can_be_rehearsed_too(tmp_path: Path) -> None:
    """The declared config's activation checkpointing, put back into a dev-fidelity run.

    Measured 2026-09-29: the rehearsal ran the path at dev fidelity and went green, while the
    declared-shape probe on the same path died in its first step - because
    ``declared_session_config`` sets ``activation_checkpointing`` and the dev config does not,
    so no rehearsal had ever entered a checkpointed block on XLA. The flag exists so that gap
    can be closed without a TPU (XLA:CPU runs this in about two minutes), and the record has
    to say which way it was run.
    """
    output = tmp_path / "tpu-session-rehearsal.json"

    rehearsal.main(
        [
            "--device", "cpu", "--steps", "1", "--seed", "9",
            "--checkpoint-activations", "--output", str(output),
        ]
    )

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["activation_checkpointing"] is True
    assert record["session"]["config"]["activation_checkpointing"] is True
    assert "session" in record["completed"]

    # And the unfixed default is still the developmental one, which is what the rehearsal's
    # own record has always meant by `fidelity: development`.
    assert rehearsal._dev_null_config("cpu", steps=1, seed=9).activation_checkpointing is False
    assert rehearsal.base_record(_args())["activation_checkpointing"] is False


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


def test_the_session_log_is_printed_as_it_arrives(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Both copies, because they fail differently.

    v1's session phase was killed three hours in and its whole account was the phase
    boundary: the session's lines go into the *record's* list, and the record is only
    written when the phase returns. The printed copy is what a killed run still has.
    """
    lines: list[str] = []
    log = rehearsal._session_log(lines, "session")
    log("resumed from checkpoint-000002.pt at step 2")

    assert lines == ["resumed from checkpoint-000002.pt at step 2"]
    assert "[session] resumed from checkpoint-000002.pt at step 2" in capsys.readouterr().out


def test_the_watch_prints_and_keeps_what_the_process_looked_like(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The trace a killed run never had, in both places a killed run can still be read.

    The sample at ``seconds: 0`` is the baseline a reader compares the rest against: flat
    samples are a phase that is working, climbing ones are a phase that is accumulating,
    which is the failure mode this rehearsal exists to catch. The sidecar is written through
    a rename so a process killed mid-write leaves a whole file rather than half of one.
    """
    journal = tmp_path / "tpu-session-rehearsal.json.watch.json"
    watch = rehearsal._Watch("session", seconds=0.01, journal=journal).start()
    deadline = time.time() + 5
    while len(watch.samples) < 3 and time.time() < deadline:
        time.sleep(0.01)
    watch.stop()

    assert len(watch.samples) >= 3, "one at the start, then on the timer"
    assert [sample["seconds"] for sample in watch.samples] == sorted(
        sample["seconds"] for sample in watch.samples
    )
    assert set(watch.samples[0]) == {"seconds", "rss_kb", "device_bytes_used"}
    assert watch.samples[0]["seconds"] == 0.0
    # A host without /proc (or a runtime without a memory report) says so with a null,
    # never with a zero that would read as a perfectly flat run.
    for sample in watch.samples:
        assert sample["rss_kb"] is None or sample["rss_kb"] > 0

    assert json.loads(journal.read_text(encoding="utf-8")) == watch.samples
    assert not journal.with_name(journal.name + ".part").exists()
    assert "[watch session]" in capsys.readouterr().out


def test_the_deadline_and_the_watch_are_part_of_the_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reading through a deadline is a different reading, so the leash is recorded.

    v1 had none, which is how a hang became a three-hour slot with no artifact and no
    explanation. The session gets the longest leash because it is the phase that trains;
    ``TPU_SESSION_DEADLINE`` overrides every phase at once, which is what a shortened re-fire
    wants - and it is read at call time rather than at import, so a test can set it.
    """
    record = rehearsal.base_record(_args(device="xla"))

    assert record["phase_deadlines"]["session"] == 1800.0
    assert record["phase_deadlines"]["environment"] == 600.0
    assert set(record["phase_deadlines"]) == set(rehearsal.PHASES)
    assert record["watch_seconds"] == 30.0

    monkeypatch.setenv("TPU_SESSION_DEADLINE", "45")
    monkeypatch.setenv("TPU_SESSION_WATCH", "2.5")
    shortened = rehearsal.base_record(_args(device="xla"))

    assert set(shortened["phase_deadlines"].values()) == {45.0}
    assert shortened["watch_seconds"] == 2.5


def test_a_phase_is_deadlined_and_watched_around_its_own_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wiring: every phase arms a deadline, cancels it, and lands its samples.

    Asserted for all four phases on a CPU dry run, because the property that failed in v1
    belongs to the runner rather than to the set of phases that happened to be reached.
    """
    armed: list[float] = []
    cancelled: list[int] = []

    def fake_deadline(seconds: float, **kwargs: object) -> None:
        armed.append(seconds)

    monkeypatch.setattr(rehearsal.faulthandler, "dump_traceback_later", fake_deadline)
    monkeypatch.setattr(
        rehearsal.faulthandler,
        "cancel_dump_traceback_later",
        lambda: cancelled.append(1),
    )
    monkeypatch.setenv("TPU_SESSION_WATCH", "0.01")
    output = tmp_path / "tpu-session-rehearsal.json"

    rehearsal.main(["--device", "cpu", "--steps", "1", "--seed", "8", "--output", str(output)])

    record = json.loads(output.read_text(encoding="utf-8"))
    assert armed == [rehearsal.phase_deadline_seconds(name) for name in rehearsal.PHASES]
    assert len(cancelled) == len(rehearsal.PHASES), "cancelled even when the phase failed"
    assert sorted(record["watch"]) == sorted(rehearsal.PHASES)
    assert all(samples for samples in record["watch"].values())
    # Beside the record, under a name the fetch keeps: `save_fetched` keeps `*.json`.
    assert (tmp_path / f"{output.name}.watch.json").is_file()
