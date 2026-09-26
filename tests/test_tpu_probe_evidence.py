"""The TPU probe's records: what each measured, what it could not, and what the rule says.

The kernel `act-tpu-probe` has finished twice and both records are committed here, because
each one *is* a finding and the second is only readable against the first:

* **v4** (2026-09-22, fetched 2026-09-25) completed four of its five phases and lost the
  aggregate to `xmp.spawn`'s precondition - `Runtime is already initialized. Do not use the
  XLA device before calling xmp.spawn.`
* **v5** (2026-09-25, fetched 2026-09-26) fixed exactly that by handing the aggregate to a
  child process; the child ran, and the failure moved one layer down into libtpu's topology -
  `TPU initialization failed: Invalid --2a886c8_slice_builder_worker_addresses specified.
  Expected 8 worker addresses, got 1.`

Neither record can decide the port, and that is the property `evaluate` exists to enforce:
the rule reads `independent_seeds.aggregate_tokens_per_second`, so a record that does not
carry it is `INCONCLUSIVE` rather than a pass. What the two records *do* carry is the
per-core reading they agree on, and it is why the aggregate is now taken in one process, one
thread per device (`scripts/tpu_probe.py`).

Four further properties are load-bearing rather than incidental:

* **the four phases that ran are the numbers the journal quotes**, per record, recomputed
  here rather than asserted;
* **the records disagree about the dtype**: v4 measured fp32 2% faster than bf16, v5 measured
  bf16 6% faster than fp32 - so the reading that "the bf16 amendment buys nothing on this
  hardware" is not reproducible at this spread, and the honest statement is that the two
  dtypes sit within run-to-run noise of each other;
* **both per-core rates sit below the bar** either way: eight cores at v5's bf16 rate would
  be 1.78x the T4 pair against a declared 2.0x, and 1.73x at v4's;
* **a rebuild's steady mean is not a steady step** - its fastest step matches the throughput
  pass's while its mean is 10x that - which is recorded because the rule's recompile-overhead
  arithmetic subtracts that mean from the first step.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import tpu_probe  # noqa: E402

OUTPUT = ROOT / "artifacts" / "tpu-probe" / "output"

#: The records, named by the run that produced them rather than "the latest": a fetch lands
#: in this directory as ``tpu-probe.json`` and would otherwise overwrite the evidence the
#: journal quotes. Each is added here when the journal cites its numbers.
RECORDS = ("tpu-probe-v4.json", "tpu-probe-v5.json")

#: The four measurements that ran in each record, as the journal quotes them: throughput and
#: rebuild, bf16 and fp32, in tokens/s.
DOCUMENTED = {
    "tpu-probe-v4.json": {"bf16": (1098.2, 379.4), "fp32": (1121.2, 370.2)},
    "tpu-probe-v5.json": {"bf16": (1133.0, 394.7), "fp32": (1068.7, 358.2)},
}

#: What each record says went wrong with ``independent_seeds``, and how far the attempt got:
#: v4 died on the spawn's own precondition, v5's child reached libtpu's topology check.
FAILURES = {
    "tpu-probe-v4.json": (
        "Runtime is already initialized. Do not use the XLA device before calling xmp.spawn."
    ),
    "tpu-probe-v5.json": (
        "TPU initialization failed: Invalid --2a886c8_slice_builder_worker_addresses "
        "specified. Expected 8 worker addresses, got 1."
    ),
}

#: The rate one core would have to sustain for eight of them to clear the rule's own bar.
PER_CORE_FOR_THE_BAR = 2.0 * tpu_probe.T4_PAIR_TOKENS_PER_SECOND / 8


def record(name: str) -> dict:
    path = OUTPUT / name
    assert path.is_file(), (
        f"{path} is committed evidence; without it the journal entry quotes nothing"
    )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", RECORDS)
def test_the_four_phases_that_ran_are_the_numbers_the_journal_quotes(name: str) -> None:
    rows = record(name)
    for dtype, (throughput, rebuild) in DOCUMENTED[name].items():
        assert rows[dtype]["throughput"]["tokens_per_second"] == throughput
        assert rows[dtype]["rebuild"]["tokens_per_second"] == rebuild
    assert rows["bf16"]["parameters"] == rows["fp32"]["parameters"] == 285_336_576
    assert rows["device"]["xla_devices"] == 8
    assert rows["declared_pair_tokens_per_second_on_t4"] == tpu_probe.T4_PAIR_TOKENS_PER_SECOND
    assert sorted(rows["completed"]) == [
        "bf16.rebuild",
        "bf16.throughput",
        "fp32.rebuild",
        "fp32.throughput",
    ]


@pytest.mark.parametrize("name", RECORDS)
def test_the_missing_measurement_names_its_own_cause(name: str) -> None:
    """The failure is in the record, not only in a kernel log, and it says which one it was.

    Two of these runs were diagnosed from the log rather than from the artifact they were
    written for; the second record's line is also what shows the *first* fix working, since
    the failure it carries is one layer further in than v4's.
    """
    rows = record(name)
    assert len(rows["errors"]) == 1
    assert rows["errors"][0]["phase"] == "independent_seeds"
    assert FAILURES[name] in rows["errors"][0]["error"]
    assert "independent_seeds" not in rows
    assert "independent_seeds" not in rows["completed"]


@pytest.mark.parametrize("name", RECORDS)
def test_the_rule_refuses_these_records_rather_than_passing_them(name: str) -> None:
    """`INCONCLUSIVE`, with the two fields it is missing named - the rule, recomputed.

    Read through `evaluate` rather than restated here: a second implementation of "is this
    record complete" is a second answer to the question the probe is for.
    """
    decision = tpu_probe.evaluate(record(name))
    assert decision["verdict"] == "INCONCLUSIVE"
    assert decision["capacity"] is None
    assert decision["measured"] == {}
    assert "independent_seeds.aggregate_tokens_per_second" in decision["why"]
    assert "independent_seeds.workers" in decision["why"]


@pytest.mark.parametrize(
    "name,multiple",
    [("tpu-probe-v4.json", 1.73), ("tpu-probe-v5.json", 1.78)],
)
def test_the_per_core_reading_the_journal_states_follows_from_each_record(
    name: str, multiple: float
) -> None:
    """Eight cores at this rate do not clear the bar, and the journal says so before the
    aggregate is measured rather than after it.

    The aggregate is eight chips' worth of rate and each chip has its own HBM, so eight times
    the single-core rate is the closest thing to a forecast available - and it is a
    *forecast*, which is why it is stated as one and the measurement is still being taken.
    """
    rows = record(name)
    bf16 = rows["bf16"]["throughput"]["tokens_per_second"]
    assert round(8 * bf16 / tpu_probe.T4_PAIR_TOKENS_PER_SECOND, 2) == multiple
    assert multiple < tpu_probe.PORT_THRESHOLD_MULTIPLE
    for dtype in DOCUMENTED[name]:
        assert rows[dtype]["throughput"]["tokens_per_second"] < PER_CORE_FOR_THE_BAR
    assert round(PER_CORE_FOR_THE_BAR, 1) == 1270.8


def test_the_two_records_disagree_about_the_dtype_and_that_is_the_reading() -> None:
    """v4: fp32 2% faster. v5: bf16 6% faster. The delta is run-to-run spread, not a dtype.

    The entry that read v4's ordering as "the bf16 amendment the rule assumes buys nothing on
    this hardware" is superseded by the later record as it stands: the same measurement on the
    same shape moved the other way, so what the two runs support is that the dtypes are within
    a few percent of each other - not which of them is faster.
    """

    def spread(rows: dict) -> float:
        bf16 = rows["bf16"]["throughput"]["tokens_per_second"]
        return (rows["fp32"]["throughput"]["tokens_per_second"] - bf16) / bf16

    v4, v5 = record(RECORDS[0]), record(RECORDS[1])
    assert spread(v4) > 0 and spread(v5) < 0
    assert abs(spread(v4)) < 0.05 and abs(spread(v5)) < 0.10
    # v4 does not even agree with itself: its rebuild block puts bf16 ahead (379.4 against
    # 370.2) while its throughput block puts fp32 ahead - so neither ordering is a dtype
    # effect to build on, and the rebuild block is the shakier of the two (see below).
    assert (
        v4["fp32"]["throughput"]["tokens_per_second"]
        > v4["bf16"]["throughput"]["tokens_per_second"]
    )
    assert (
        v4["fp32"]["rebuild"]["tokens_per_second"]
        < v4["bf16"]["rebuild"]["tokens_per_second"]
    )
    for dtype in ("bf16", "fp32"):
        assert (
            v5[dtype]["rebuild"]["tokens_per_second"]
            < v5[dtype]["throughput"]["tokens_per_second"]
        )


@pytest.mark.parametrize("name", RECORDS)
def test_the_rebuilds_steady_mean_is_not_a_steady_step(name: str) -> None:
    """A record-level caution, so the journal does not read it as one.

    In both records a rebuild's *fastest* step is the same ~1.0 s as the throughput pass's,
    while its mean is 10x that: ``_measure`` separates the first (compiling) step, and in a
    four-step loop later samples still carry cost. The rule's recompile-overhead arithmetic
    subtracts ``steady_step_seconds_mean`` from the first step, so it under-states the compile
    - which is the *un*conservative direction for that condition. It is recorded here rather
    than smoothed over: the overhead condition is the one to re-derive before it is relied on.
    """
    rows = record(name)
    for dtype in ("bf16", "fp32"):
        steady = rows[dtype]["rebuild"]["steady_step_seconds_mean"]
        fastest = rows[dtype]["rebuild"]["steady_step_seconds_min"]
        assert steady > 8 * fastest
        assert fastest == pytest.approx(
            rows[dtype]["throughput"]["steady_step_seconds_min"], abs=0.05
        )
