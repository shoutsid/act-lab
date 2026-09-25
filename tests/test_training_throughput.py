"""The throughput note's arithmetic is recomputed rather than trusted.

``docs/training-throughput.md`` answers three questions with numbers - how fast the card
is actually running, how much arithmetic the dense expert form executes that the routing
contract does not require, and how much work a completion costs without a KV cache - and
each of those is *derived*, so each of them can be re-derived here. A note that quotes a
rate is prose; a note whose rate is recomputed from the project's own estimators is a
claim, and these tests make it one.

Two of them are stronger than arithmetic and are the reason the module exists:

- the dense-vs-eager measurement the model's docstring rests on implies a FLOP rate
  **above the card's physical peak**, at the config it names and at the batch it names.
  That is not a disagreement between two documents; it is a number that cannot be true,
  and it is the number a GPU-hour budget would have been sized from.
- the ratio of executed to booked arithmetic is not a property of one shape: it holds at
  the code-LM shape *and* at the development shape, so the conclusion did not need
  re-arguing, only re-measuring.

These are derivation checks. The numbers behind the *results* are owned by
``scripts/verify_record.py`` and by the per-study pinning tests; this note deliberately
carries no ledger claim, and ``test_the_note_is_not_a_ledger_claim`` keeps it that way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from act.model.v0_moe import DEV_V0_CONFIG, V0MoEConfig, estimated_parameter_count
from act.search.architecture_eval import (
    estimate_dense_runtime_flops_per_token,
    estimate_flops_per_token,
)

ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "docs" / "training-throughput.md"

#: GTX 1070 (GP104, sm_61): 1,920 fp32 lanes at 1.683 GHz boost, one FMA per lane per
#: cycle. Stated here as a declared constant rather than queried, because the card is
#: running the campaign while this test runs and the peak is a spec, not a measurement.
PEAK_FLOPS = 1920 * 2 * 1.683e9

#: The leg's student: the campaign's ``match`` scale, at the teacher's own vocabulary.
MATCH = V0MoEConfig(
    vocab_size=32_256,
    context_size=512,
    hidden_size=768,
    depth=12,
    attention_heads=12,
    experts_per_layer=(8,) * 12,
    active_experts=2,
    expert_width=768,
)

#: Batches and rates exactly as the note quotes them: 8 x 512 tokens per step, and the
#: two live measurements (the 2026-09-19 dev cycle and this leg's arm 1 at 20:10).
STEP_TOKENS = 8 * 512
MATCH_RATE = 2_731.0
DEV_RATE = 24_299.0
#: The throughput benchmark's own fp32 batch-8 figure for the dev shape, and the s/step it
#: corresponds to - the measurement the docstring's 42.5 ms/step contradicts by 11x.
BENCH_RATE = 8_747.0
BENCH_S_PER_STEP = 0.468
#: The docstring figures the note refutes, in ms per step.
QUOTED_DENSE_MS = 42.5
QUOTED_EAGER_MS = 51.8


@pytest.fixture(scope="module")
def note() -> str:
    return NOTE.read_text(encoding="utf-8")


def _booked_and_executed(config: V0MoEConfig) -> tuple[float, float]:
    return (
        estimate_flops_per_token(config),
        estimate_dense_runtime_flops_per_token(config),
    )


def test_the_dense_expert_form_executes_twice_the_booked_arithmetic(note: str) -> None:
    """The lever: 1.93x of the step is arithmetic the routing contract does not require.

    ``active/total`` is ~0.49 at both shapes, so this is a property of the dense form -
    every expert applied to every token, then the top-k rows selected - rather than of
    one configuration. The test pins the ratio at both shapes so that "the code-LM shape
    is special" cannot be asserted later without new numbers.
    """

    booked, executed = _booked_and_executed(MATCH)
    waste = executed / booked

    total, active = estimated_parameter_count(MATCH)
    assert (total, active) == (166_423_296, 81_488_640)
    assert active / total == pytest.approx(0.490, abs=0.001)
    assert waste == pytest.approx(1.934, abs=0.001)

    _, dev_executed = _booked_and_executed(DEV_V0_CONFIG)
    dev_booked = estimate_flops_per_token(DEV_V0_CONFIG)
    assert dev_executed / dev_booked == pytest.approx(1.774, abs=0.001)

    # The parameter counts themselves are pinned against the design note that tabulates
    # them (``tests/test_width_control_design.py``); this note quotes the ratio and the
    # per-token cost, so those are what it has to keep saying.
    for rendered in (
        "545.6 MFLOP/token",
        "1,055.2 MFLOP/token",
        "**1.934x**",
        "57.1 MFLOP/token",
        "101.4 MFLOP",
        "1.774x",
        "0.490",
    ):
        assert rendered in note, f"the note no longer states {rendered!r}"


def test_the_live_rates_leave_little_room_in_the_step_loop(note: str) -> None:
    """Half the card is where an eager fp32 loop runs, so the loop is not the problem.

    Both of the project's own live measurements - the dev cycle and this leg's arm - land
    between a third and a half of the card's fp32 peak. The test's job is to keep the note
    from claiming large headroom here: if a future edit quotes a rate that would put the
    loop far below this band, one of the two numbers is wrong.
    """

    _, executed = _booked_and_executed(MATCH)
    match_pct = executed * MATCH_RATE / PEAK_FLOPS
    assert match_pct == pytest.approx(0.45, abs=0.01)

    _, dev_executed = _booked_and_executed(DEV_V0_CONFIG)
    dev_pct = dev_executed * DEV_RATE / PEAK_FLOPS
    assert dev_pct == pytest.approx(0.38, abs=0.01)

    # The small shape is launch-bound and converts to a much lower rate - which is why the
    # code-LM shape is the one worth quoting an efficiency for.
    bench_pct = dev_executed * BENCH_RATE / PEAK_FLOPS
    assert bench_pct == pytest.approx(0.137, abs=0.002)

    for rendered in (
        f"{executed * MATCH_RATE / 1e12:.2f} TFLOP/s",
        f"{match_pct * 100:.0f}%",
        f"{dev_executed * DEV_RATE / 1e12:.2f} TFLOP/s",
        f"{dev_pct * 100:.0f}%",
        f"{dev_executed * BENCH_RATE / 1e12:.2f} TFLOP/s",
        f"{bench_pct * 100:.1f}%",
    ):
        assert rendered in note, f"the note no longer states {rendered!r}"


def test_the_quoted_dense_and_eager_figures_exceed_the_cards_peak(note: str) -> None:
    """The measurement the dense form was chosen on cannot have been taken as described.

    At ``DEV_V0_CONFIG`` and batch 8 x context 512, the dense form's arithmetic per step
    is fixed by the config, so each quoted ms/step converts to exactly one FLOP rate. Both
    conversions land *above* the card's fp32 peak - 151% and 124% - which is why the note
    calls the docstring pair internally impossible rather than merely inconsistent with the
    performance table. That distinction is what makes re-measuring mandatory rather than
    tidy, and it is what this test pins.
    """

    _, executed = _booked_and_executed(DEV_V0_CONFIG)
    per_step = executed * STEP_TOKENS

    dense = per_step / (QUOTED_DENSE_MS / 1000)
    eager = per_step / (QUOTED_EAGER_MS / 1000)

    assert dense / PEAK_FLOPS == pytest.approx(1.51, abs=0.01)
    assert eager / PEAK_FLOPS == pytest.approx(1.24, abs=0.01)
    assert dense > PEAK_FLOPS and eager > PEAK_FLOPS

    # The same shape is measured in the benchmark at 0.468 s/step, so the docstring's figure
    # is not merely above the hardware's peak but 11x off the project's own timing of it.
    assert BENCH_S_PER_STEP / (QUOTED_DENSE_MS / 1000) == pytest.approx(11.0, abs=0.1)

    for rendered in (
        f"{dense / 1e12:.2f} TFLOP/s",
        f"{eager / 1e12:.2f} TFLOP/s",
        "151% of peak",
        "124% of peak",
        "0.468 s/step",
        "11.0x faster than the same shape was measured at",
    ):
        assert rendered in note, f"the note no longer states {rendered!r}"


def test_generation_without_a_cache_costs_about_three_hundred_times_more(note: str) -> None:
    """The second lever, and the one that decides whether a benchmark can be run at all.

    One token per full-prefix forward, with the prefix capped at the context, is what both
    samplers do today. The test counts the token-forwards directly from that definition -
    it is not quoting a benchmark - and pins the two counts and the ratio.
    """

    context, prompt, new_tokens = 512, 185, 512
    without = sum(min(prompt + step, context) for step in range(new_tokens))
    cached = prompt + new_tokens

    assert without == 208_516
    assert cached == 697
    assert without / cached == pytest.approx(299, abs=1)

    attention = sum(min(prompt + step, context) ** 2 for step in range(new_tokens)) / 1e6
    cached_attention = prompt * prompt + new_tokens * prompt
    assert attention == pytest.approx(91.0, abs=0.1)
    assert cached_attention == 128_945
    assert attention * 1e6 / cached_attention == pytest.approx(706, abs=1)

    for rendered in ("208,516", "697", "**299x**", "91.0M", "128,945", "**706x**"):
        assert rendered in note, f"the note no longer states {rendered!r}"


def test_the_token_budget_is_orders_of_magnitude_below_the_comparator(note: str) -> None:
    """The part no speedup fixes, and the reason this test exists at all.

    The leg trains 46,080,000 tokens on a 166.4M model - 0.28 tokens per parameter. The
    published comparator's 7.84% HumanEval came from 100B tokens on a 164.1M model, ~609
    tokens per parameter. Keeping both figures in the note is what stops a 1.93x step
    speedup from being read as the road to a working model.
    """

    # The steps already carry the epochs: 11,250 steps x 8 windows is 90,000 windows,
    # which is 3 epochs of the 30,000 the targets stage scored. Multiplying by epochs
    # again is the mistake this comment exists to prevent.
    steps, batch, context = 11_250, 8, 512
    windows, epochs = 30_000, 3
    tokens = steps * batch * context
    assert tokens == windows * context * epochs == 46_080_000

    total, _ = estimated_parameter_count(MATCH)
    assert tokens / total == pytest.approx(0.28, abs=0.005)

    baseline_params, baseline_tokens = 164_144_128, 100_000_000_000
    assert baseline_tokens / baseline_params == pytest.approx(609, abs=1)
    assert baseline_tokens / tokens == pytest.approx(2_170, abs=1)

    for rendered in ("46,080,000", "**0.28 tokens per parameter**", "**609 tokens per parameter**", "**2,170x**"):
        assert rendered in note, f"the note no longer states {rendered!r}"


def test_the_note_is_not_a_ledger_claim(note: str) -> None:
    """A note that quotes live measurements must not become a document claims rest on.

    The rates in it are observations with timestamps, not stored artifacts, so a ledger
    claim drawn from it could not be recomputed from a row. It declares itself a
    design/measurement note for that reason, and this keeps the declaration true.
    """

    assert "design and measurement note, not a record" in note
    ledger = (ROOT / "scripts" / "verify_record.py").read_text(encoding="utf-8")
    assert "training-throughput" not in ledger, (
        "the throughput note is a dated observation, not a record - a claim recomputed "
        "from it would not be recomputable from a stored row"
    )
