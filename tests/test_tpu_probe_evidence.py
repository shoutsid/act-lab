"""The TPU probe's record: what it measured, what it could not, and what the rule says.

The kernel `act-tpu-probe` v4 finished `COMPLETE` on 2026-09-22 and nobody read it - the
fetch unpacked the record into a temporary directory, printed it, and deleted it, so the copy
on disk stayed the previous run's. The record is committed now, and this is what keeps it from
becoming a claim nobody can re-read: the numbers the journal quotes are read back out of the
artifact here, and the decision the rule reaches on it is recomputed rather than asserted.

Three properties are load-bearing rather than incidental:

* **four of the five phases ran.** The port decision reads bf16 and the eight-worker
  aggregate; what arrived is bf16 and fp32 throughput plus both rebuilds, so the record is
  `INCONCLUSIVE` - never a pass, which is what `evaluate` exists to enforce.
* **the fifth failed for a named reason**, and it is the reason a child process now takes it:
  `Runtime is already initialized. Do not use the XLA device before calling xmp.spawn.`
* **the per-core rates sit below the bar.** Eight cores at the measured bf16 rate would be
  1.73x the T4 pair against a declared 2.0x - a proxy for the aggregate, not the aggregate,
  which is exactly why the measurement is being taken rather than inferred.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import tpu_probe  # noqa: E402

RECORD = ROOT / "artifacts" / "tpu-probe" / "output" / "tpu-probe.json"

#: The four measurements that ran, as the journal quotes them.
DOCUMENTED = {"bf16": (1098.2, 379.4), "fp32": (1121.2, 370.2)}

#: The rate one core would have to sustain for eight of them to clear the rule's own bar.
PER_CORE_FOR_THE_BAR = 2.0 * tpu_probe.T4_PAIR_TOKENS_PER_SECOND / 8


def record() -> dict:
    assert RECORD.is_file(), (
        f"{RECORD} is committed evidence; without it the journal entry quotes nothing"
    )
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_the_four_phases_that_ran_are_the_numbers_the_journal_quotes() -> None:
    rows = record()
    for dtype, (throughput, rebuild) in DOCUMENTED.items():
        assert rows[dtype]["throughput"]["tokens_per_second"] == throughput
        assert rows[dtype]["rebuild"]["tokens_per_second"] == rebuild
    assert rows["device"]["xla_devices"] == 8
    assert rows["declared_pair_tokens_per_second_on_t4"] == tpu_probe.T4_PAIR_TOKENS_PER_SECOND
    assert sorted(rows["completed"]) == [
        "bf16.rebuild",
        "bf16.throughput",
        "fp32.rebuild",
        "fp32.throughput",
    ]


def test_the_missing_measurement_names_its_own_cause() -> None:
    """The kernel-log line and the record agree, and the record is what a reader has.

    Three days of this run's life were spent unread, and the diagnosis that followed was done
    from the log rather than from the record it was written for. If a later run fails the same
    way, this is the line that says why without leaving the artifact.
    """
    rows = record()
    assert rows["errors"] == [
        {
            "phase": "independent_seeds",
            "error": "RuntimeError: Runtime is already initialized. Do not use the XLA "
            "device before calling xmp.spawn.",
        }
    ]
    assert "independent_seeds" not in rows
    assert "independent_seeds" not in rows["completed"]


def test_the_rule_refuses_this_record_rather_than_passing_it() -> None:
    """`INCONCLUSIVE`, with the two fields it is missing named - the rule, recomputed.

    Read through `evaluate` rather than restated here: a second implementation of "is this
    record complete" is a second answer to the question the probe is for.
    """
    decision = tpu_probe.evaluate(record())
    assert decision["verdict"] == "INCONCLUSIVE"
    assert decision["capacity"] is None
    assert decision["measured"] == {}
    assert "independent_seeds.aggregate_tokens_per_second" in decision["why"]
    assert "independent_seeds.workers" in decision["why"]


def test_the_per_core_reading_the_journal_states_follows_from_the_record() -> None:
    """Eight cores at this rate do not clear the bar, and the journal says so before the
    aggregate is measured rather than after it.

    The aggregate is the sum of eight ranks' steady rates and each rank has its own chip and
    HBM, so eight times the single-core rate is the closest thing to a forecast available -
    and it is a *forecast*, which is why it is stated as one and the measurement is still
    being taken.
    """
    rows = record()
    multiple = 8 * rows["bf16"]["throughput"]["tokens_per_second"] / tpu_probe.T4_PAIR_TOKENS_PER_SECOND
    assert round(multiple, 2) == 1.73
    assert multiple < tpu_probe.PORT_THRESHOLD_MULTIPLE
    for dtype in DOCUMENTED:
        assert rows[dtype]["throughput"]["tokens_per_second"] < PER_CORE_FOR_THE_BAR
    assert round(PER_CORE_FOR_THE_BAR, 1) == 1270.8
    # And the bf16 amendment the rule assumes buys nothing here: fp32 measured *faster*.
    assert rows["fp32"]["throughput"]["tokens_per_second"] > rows["bf16"]["throughput"]["tokens_per_second"]
