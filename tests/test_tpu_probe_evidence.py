"""The TPU probe's records: what each measured, what it could not, and what the rule says.

The kernel `act-tpu-probe` has finished three times and all three records are committed
here, because each one *is* a finding and the later result is only readable against the
failures it resolved:

* **v4** (2026-09-22, fetched 2026-09-25) completed four of its five phases and lost the
  aggregate to `xmp.spawn`'s precondition - `Runtime is already initialized. Do not use the
  XLA device before calling xmp.spawn.`
* **v5** (2026-09-25, fetched 2026-09-26) fixed exactly that by handing the aggregate to a
  child process; the child ran, and the failure moved one layer down into libtpu's topology -
  `TPU initialization failed: Invalid --2a886c8_slice_builder_worker_addresses specified.
  Expected 8 worker addresses, got 1.`
* **v6** (2026-09-26) took the aggregate in one process, one bound thread per device, and
  completed all five phases on a v5litepod-8. The measured aggregate was 15,034.2 tokens/s;
  the pre-registered rule returns **PORT** (2.96x the T4 pair; 2.07% recompile overhead).

The v4/v5 records could not decide the port, and that is the property `evaluate` enforces:
the rule reads `independent_seeds.aggregate_tokens_per_second`, so records that do not carry
it are `INCONCLUSIVE`, rather than passes. v6 now supplies the missing aggregate, and its
verdict is pinned below against the committed run artifact.

Five further properties are load-bearing rather than incidental:

* **the four measurements are the numbers the journal quotes**, recomputed here rather than
  asserted; the v4/v5 single-device diagnostic rates do not substitute for v6's aggregate;
* **v6's aggregate comes from all eight recorded ranks**, with each rank bound to its own
  chip; the aggregate remains the rule's direct measurement, not a sum of per-core rates;
* **the records disagree about the dtype**: v4 measured fp32 2% faster than bf16, v5 measured
  bf16 6% faster than fp32 - so the reading that "the bf16 amendment buys nothing on this
  hardware" is not reproducible at this spread, and the honest statement is that the two
  dtypes sit within run-to-run noise of each other;
* **both v4/v5 per-core rates sit below the bar**: eight cores at v5's bf16 rate would be
  1.78x the T4 pair against a declared 2.0x, and 1.73x at v4's;
* **a rebuild's steady mean is not a steady step** - its fastest step matches the throughput
  pass's while its mean is 10x that - which is recorded because the rule's recompile-overhead
  arithmetic subtracts that mean from the first step. Quantified below rather than left as a
  caution: the caveat moves that condition by ~1.5x and both readings sit ~5x under its limit,
  and the condition would only bind at a compile cost ~146-149 s.
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
ALL_RECORDS = (*RECORDS, "tpu-probe-v6.json")
PORT_RECORD = "tpu-probe-v6.json"

#: The four single-device measurements that ran in each record, as the journal quotes them:
#: throughput and rebuild, bf16 and fp32, in tokens/s.
DOCUMENTED = {
    "tpu-probe-v4.json": {"bf16": (1098.2, 379.4), "fp32": (1121.2, 370.2)},
    "tpu-probe-v5.json": {"bf16": (1133.0, 394.7), "fp32": (1068.7, 358.2)},
    "tpu-probe-v6.json": {"bf16": (1110.1, 390.0), "fp32": (1121.9, 372.0)},
}

#: What each inconclusive record says went wrong with ``independent_seeds``.
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


@pytest.mark.parametrize("name", ALL_RECORDS)
def test_the_four_phases_that_ran_are_the_numbers_the_journal_quotes(name: str) -> None:
    rows = record(name)
    for dtype, (throughput, rebuild) in DOCUMENTED[name].items():
        assert rows[dtype]["throughput"]["tokens_per_second"] == throughput
        assert rows[dtype]["rebuild"]["tokens_per_second"] == rebuild
    assert rows["bf16"]["parameters"] == rows["fp32"]["parameters"] == 285_336_576
    assert rows["device"]["xla_devices"] == 8
    assert rows["declared_pair_tokens_per_second_on_t4"] == tpu_probe.T4_PAIR_TOKENS_PER_SECOND
    expected_phases = [
        "bf16.rebuild",
        "bf16.throughput",
        "fp32.rebuild",
        "fp32.throughput",
    ]
    if name == PORT_RECORD:
        expected_phases.append("independent_seeds")
    assert sorted(rows["completed"]) == sorted(expected_phases)


@pytest.mark.parametrize("name", RECORDS)
def test_the_missing_measurement_names_its_own_cause(name: str) -> None:
    """The failure is in the record, not only in the log, and says which one it was.

    The v5 record's line also shows the first fix working, since its failure is one layer
    further in than v4's.
    """
    rows = record(name)
    assert len(rows["errors"]) == 1
    assert rows["errors"][0]["phase"] == "independent_seeds"
    assert FAILURES[name] in rows["errors"][0]["error"]
    assert "independent_seeds" not in rows
    assert "independent_seeds" not in rows["completed"]


@pytest.mark.parametrize("name", RECORDS)
def test_the_rule_refuses_incomplete_records_rather_than_passing_them(name: str) -> None:
    """`INCONCLUSIVE`, with the fields it needs named - the rule, recomputed."""
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
def test_the_inconclusive_records_per_core_readings_stay_below_the_bar(
    name: str, multiple: float
) -> None:
    """Eight cores at these historical single-core rates do not clear the rule's bar."""
    rows = record(name)
    bf16 = rows["bf16"]["throughput"]["tokens_per_second"]
    assert round(8 * bf16 / tpu_probe.T4_PAIR_TOKENS_PER_SECOND, 2) == multiple
    assert multiple < tpu_probe.PORT_THRESHOLD_MULTIPLE
    for dtype in DOCUMENTED[name]:
        assert rows[dtype]["throughput"]["tokens_per_second"] < PER_CORE_FOR_THE_BAR
    assert round(PER_CORE_FOR_THE_BAR, 1) == 1270.8


def test_the_two_inconclusive_records_disagree_about_the_dtype() -> None:
    """v4: fp32 2% faster. v5: bf16 6% faster. This is run-to-run spread, not a dtype."""

    def spread(rows: dict) -> float:
        bf16 = rows["bf16"]["throughput"]["tokens_per_second"]
        return (rows["fp32"]["throughput"]["tokens_per_second"] - bf16) / bf16

    v4, v5 = record(RECORDS[0]), record(RECORDS[1])
    assert spread(v4) > 0 and spread(v5) < 0
    assert abs(spread(v4)) < 0.05 and abs(spread(v5)) < 0.10
    # v4 does not even agree with itself: its rebuild block puts bf16 ahead (379.4 against
    # 370.2) while its throughput block puts fp32 ahead.
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


def test_v6_completes_the_eight_rank_aggregate_and_clears_the_registered_rule() -> None:
    rows = record(PORT_RECORD)
    aggregate = rows["independent_seeds"]

    assert rows["errors"] == []
    assert "independent_seeds" in rows["completed"]
    assert aggregate["workers"] == aggregate["completed"] == 8
    assert aggregate["aggregate_tokens_per_second"] == 15034.2
    assert len(aggregate["per_rank"]) == 8
    assert [rank["device"] for rank in aggregate["per_rank"]] == [
        f"xla:{i}" for i in range(8)
    ]
    assert [rank["thread_device_after"] for rank in aggregate["per_rank"]] == [
        f"xla:{i}" for i in range(8)
    ]
    assert aggregate["taken_in"] == "single-process-thread-per-device"
    assert aggregate["ratio_to_t4_pair"] == 2.96

    decision = tpu_probe.evaluate(rows)
    assert decision["verdict"] == "PORT"
    assert decision["throughput_ok"] is True
    assert decision["recompile_overhead_ok"] is True
    assert decision["measured"]["multiple_of_t4_pair"] == 2.96
    assert decision["measured"]["recompile_overhead_fraction"] == pytest.approx(0.0207)
    assert decision["capacity"]["gpu_weeks_left"] == 1


def _overhead_forecast(rows: dict) -> tuple[float, float, float]:
    """The rule's historical recompile forecast, using v4/v5's single-core rate.

    Their aggregate was unavailable, so that is the rate used to forecast the full queue.
    This helper is only called for those records, not the v6 measurement.
    """
    rebuild = rows["bf16"]["rebuild"]
    first = rebuild["first_step_seconds_including_compile"]
    per_core = rows["bf16"]["throughput"]["tokens_per_second"]
    queue_tokens = (
        tpu_probe.DECLARED_SEEDS_PER_QUEUE
        * tpu_probe.DECLARED_ARMS_PER_SEED
        * tpu_probe.DECLARED_TOKENS_PER_ARM
    )
    training_seconds = (queue_tokens / rows["device"]["xla_devices"]) / per_core
    rebuilds_per_core = (
        (tpu_probe.DECLARED_SEEDS_PER_QUEUE / rows["device"]["xla_devices"])
        * tpu_probe.DECLARED_ARMS_PER_SEED
        * tpu_probe.REBUILDS_PER_ARM
    )
    binding = tpu_probe.RECOMPILE_OVERHEAD_LIMIT * training_seconds / rebuilds_per_core

    def fraction(steady: float) -> float:
        return rebuilds_per_core * max(0.0, first - steady) / training_seconds

    return (
        fraction(rebuild["steady_step_seconds_mean"]),
        fraction(rebuild["steady_step_seconds_min"]),
        binding,
    )


@pytest.mark.parametrize("name", RECORDS)
def test_the_rebuilds_steady_mean_is_not_a_steady_step(name: str) -> None:
    """A record-level caution, so the journal does not read it as one.

    In both records a rebuild's *fastest* step is the same ~1.0 s as the throughput pass's,
    while its mean is 10x that: ``_measure`` separates the first (compiling) step, and in a
    four-step loop later samples still carry cost. The rule subtracts the mean from the first
    step, so it under-states the compile cost. The caveat moves that condition by ~1.5x and
    both readings stay below a 10% limit.
    """
    rows = record(name)
    for dtype in ("bf16", "fp32"):
        steady = rows[dtype]["rebuild"]["steady_step_seconds_mean"]
        fastest = rows[dtype]["rebuild"]["steady_step_seconds_min"]
        assert steady > 8 * fastest
        assert fastest == pytest.approx(
            rows[dtype]["throughput"]["steady_step_seconds_min"], abs=0.05
        )

    as_rule_reads_it, conservative, binding = _overhead_forecast(rows)
    assert as_rule_reads_it < tpu_probe.RECOMPILE_OVERHEAD_LIMIT / 4
    assert conservative < tpu_probe.RECOMPILE_OVERHEAD_LIMIT / 4
    assert conservative > 1.4 * as_rule_reads_it
    assert binding > 100
    assert binding / rows["bf16"]["rebuild"]["steady_step_seconds_mean"] > 10
