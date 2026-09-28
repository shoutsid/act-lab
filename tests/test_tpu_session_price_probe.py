"""Pin the TPU session *price* probe: the shape it trains, its labels, and its arithmetic.

``scripts/tpu_session_price_probe.py`` measures the one number section 5.14's option 2 is
priced on and that nothing has measured on the path that would run it - what the registered
session path sustains at the **declared** 285M shape on a TPU. Three properties are load
bearing and all three are pinned here, because the run is queued once and can wait hours:

* the config is the *declared* shape with only the step budget overridden, since a price
  measured on a shape nobody will run is not a price;
* the labels say ``declared_shape`` **without** claiming a declared seed, so a rate
  measurement cannot be read as a queue record;
* the price is ``tpu_probe.queue_tpu_hours``, the campaign's own arithmetic, so this figure
  and the port rule's 8.06 h bar cannot be computed two different ways.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import tpu_probe  # noqa: E402
import tpu_session_price_probe as price_probe  # noqa: E402
import tpu_session_rehearsal as rehearsal  # noqa: E402


def _args(**overrides) -> argparse.Namespace:
    settings = {"device": "cpu", "steps": 240, "seed": price_probe.SEED}
    settings.update(overrides)
    return argparse.Namespace(**settings)


def test_the_labels_price_a_declared_shape_without_claiming_a_declared_seed() -> None:
    """``declared_shape`` true beside ``declared_seed`` false is the distinction that matters.

    The rehearsal's labels would be wrong here in the dangerous direction - it says the
    fidelity is development and its tokens/s is not a schedule input - while a record that
    read as a *declared seed* would be a queue record this run did not produce.
    """
    record = price_probe.base_record(_args())

    assert record["record_type"] == "session-price-probe"
    assert record["kind"] == "tpu-session-price-probe"
    assert record["rehearsal"] is False
    assert record["declared_shape"] is True
    assert record["declared_seed"] is False
    assert record["declared_scale"] is False
    assert record["phase_order"] == list(price_probe.PHASES)
    assert record["completed"] == [] and record["errors"] == []
    assert "not a declared seed" in record["what_this_is"]
    assert "not a Gate-4 result" in record["what_this_is"]
    # Nothing here publishes to a campaign dataset, and the field says so rather than leaving
    # a reader to infer it from a scratch path.
    assert "not into any campaign dataset" in record["scratch_root"]
    assert len(record["not_measured"]) == len(price_probe.NOT_MEASURED)
    joined = " ".join(record["not_measured"])
    assert "not the protocol's" in joined
    assert "eight-chips-at-once" in joined


def test_the_declared_shape_needs_a_longer_leash_than_the_rehearsal_s() -> None:
    """The declared model is ~19x the dev one per step, so the phases may not share a leash.

    A deadline is what turns a hung phase into a stack instead of a killed slot, and a leash
    borrowed from the development shape would cut a legitimate declared run short.
    """
    record = price_probe.base_record(_args())

    for phase in price_probe.PHASES:
        assert (
            record["phase_deadlines"][phase]
            >= rehearsal.phase_deadline_seconds(phase)
        )
    assert (
        price_probe.PHASE_DEADLINE_SECONDS["session"]
        > rehearsal.PHASE_DEADLINE_SECONDS["session"]
    )
    # And the override still reaches every phase, so a shortened re-fire stays possible.
    assert rehearsal.phase_deadline_seconds(
        "session", price_probe.PHASE_DEADLINE_SECONDS
    ) == price_probe.PHASE_DEADLINE_SECONDS["session"]


def test_the_config_is_the_declared_shape_with_only_the_run_s_budget_overridden() -> None:
    """Every field that defines the declared shape survives; only the budget is this run's."""
    from act.training.v0_session import declared_session_config

    declared = price_probe.price_config("cpu", steps=240, seed=price_probe.SEED)

    reference = declared_session_config()
    assert declared.model_config() == reference.model_config()
    assert declared.batch_size == 2 and declared.context_size == 2048
    assert declared.dtype == "fp16"
    assert declared.activation_checkpointing is True
    assert declared.declared is True
    assert declared.arms == ("null",)
    # The overrides, and nothing else.
    assert declared.total_steps == 240
    assert declared.checkpoint_interval == 120
    assert declared.seed == price_probe.SEED
    assert declared.max_checkpoints == 2
    # A rate does not need the protocol's 24,576,000 tokens, and the difference is the point:
    # this prices them without producing them.
    assert declared.training_tokens() < reference.training_tokens()


def test_the_price_is_the_campaign_s_own_arithmetic() -> None:
    """The hours come from ``queue_tpu_hours``, not from a second formula for the same thing."""
    rate = 1879.0
    price = price_probe.price_block({"arms": {"null": {"tokens_per_second": rate}}})

    multiple = rate * price_probe.XLA_CHIPS / tpu_probe.T4_PAIR_TOKENS_PER_SECOND
    assert price["per_session_tokens_per_second"] == round(rate, 1)
    assert price["implied_aggregate_tokens_per_second"] == round(rate * 8, 1)
    assert price["multiple_of_t4_pair"] == pytest.approx(multiple, abs=1e-3)
    assert price["null_queue_tokens"] == 12 * tpu_probe.DECLARED_TOKENS_PER_ARM
    assert price["null_queue_hours_at_this_rate"] == pytest.approx(
        tpu_probe.queue_tpu_hours(multiple, arms=1), abs=1e-3
    )
    # The bar the amendment is held to: twelve single-arm seeds at the rule's own 2.0x.
    assert price["null_queue_hours_at_the_2_0x_bar"] == pytest.approx(8.06, abs=0.01)
    # The same price without taking the eight-chip scaling on trust.
    assert price["null_queue_hours_if_the_chips_do_not_scale"] == pytest.approx(
        price["null_queue_tokens"] / rate / 3600.0, abs=1e-3
    )
    assert price["fits_the_week"] == (
        price["null_queue_hours_at_this_rate"] <= tpu_probe.TPU_QUOTA_HOURS
    )


def test_the_price_refuses_a_session_that_left_no_rate() -> None:
    """A price computed from nothing would be a number with no measurement behind it."""
    with pytest.raises(RuntimeError, match="no throughput"):
        price_probe.price_block({"arms": {"null": {"tokens_per_second": 0.0}}})
    with pytest.raises(RuntimeError, match="no throughput"):
        price_probe.price_block({"arms": {}})


def test_the_verdict_names_the_rate_and_whether_the_queue_fits() -> None:
    """One line, in the order the amendment reads it: rate, multiple, hours, the bar."""
    line = price_probe.verdict(
        {"price": price_probe.price_block({"arms": {"null": {"tokens_per_second": 1879.0}}})}
    )
    assert "tokens/s on one chip" in line
    assert "x the T4 pair" in line
    assert "8.06 h at the rule's 2.0x bar" in line
    assert "fits" in line

    # Slower than the bar: the same line, saying so rather than rounding it away.
    slow = price_probe.verdict(
        {"price": price_probe.price_block({"arms": {"null": {"tokens_per_second": 300.0}}})}
    )
    assert "does not fit" in slow

    assert price_probe.verdict({}) == (
        "no price: the session phase did not leave a throughput to read"
    )


def test_a_cpu_dry_run_prices_the_queue_from_a_real_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wiring end to end: a real session path, a real price, one record.

    The declared shape is deliberately replaced here - 285M parameters is not a unit test -
    with the development one, which is the only dry run that can exist away from a TPU. What
    this pins is that the phases run through the *rehearsal's* own code (so the probe cannot
    drift into measuring a different path than the one that was rehearsed) and that the price
    lands in the record the fetch reads, even though the environment phase cannot run on a
    host with no Torch-XLA.
    """
    monkeypatch.setattr(price_probe, "price_config", rehearsal._dev_null_config)
    monkeypatch.setenv("TPU_SESSION_WATCH", "0.01")
    output = tmp_path / price_probe.RECORD_NAME

    price_probe.main(["--device", "cpu", "--steps", "2", "--output", str(output)])

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["kind"] == "tpu-session-price-probe"
    assert "session" in record["completed"]
    assert record["price"]["per_session_tokens_per_second"] > 0
    assert record["price"]["null_queue_hours_at_this_rate"] > 0
    # The watch's sidecar, under a name the fetch keeps: `save_fetched` keeps `*.json`.
    assert (tmp_path / f"{output.name}.watch.json").is_file()
    assert sorted(record["watch"]) == sorted(price_probe.PHASES)
    assert price_probe.verdict(record).startswith("the declared session path sustains")
