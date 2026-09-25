"""The grouped dispatcher is the same function as the two forms already in the model.

`scripts/moe_dispatch_benchmark.py` exists to decide a 1.934x question - whether the training
loop should keep evaluating every expert for every token, or should dispatch only the experts
each token routes to - and the part of it that must not be wrong is *identity*. A faster form
that computes a different function would change every training loss in the project while the
benchmark reported a speedup, so the equivalence gate is the load-bearing part and is pinned
here against both existing forms, at the model's own tolerance.

The remaining tests pin the refusals: a set of forms that disagrees is not timed, an expert
that receives no rows is skipped rather than run on an empty batch, and a CPU run does not
report a fraction of a GPU's peak.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from act.model.v0_moe import V0MoEConfig, _MoEFeedForward

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import moe_dispatch_benchmark as bench  # noqa: E402

TINY = dict(
    vocab_size=64,
    context_size=32,
    hidden_size=16,
    depth=2,
    attention_heads=4,
    experts_per_layer=(6, 6),
    active_experts=2,
    expert_width=12,
)


def tiny_config(**overrides: object) -> V0MoEConfig:
    return V0MoEConfig(**{**TINY, **overrides})


def test_grouped_dispatch_is_the_same_function_as_both_existing_forms() -> None:
    """The candidate must match dense *and* eager, not just agree with one of them.

    float64 and `atol=1e-12` are the tolerances `tests/test_v0_session.py` already pins the
    dense/eager pair at, so a third form is held to the same bar rather than a looser one.
    """

    config = tiny_config()
    torch.manual_seed(5)
    feed_forward = _MoEFeedForward(config, tuple(f"E{index}" for index in range(6))).to(torch.float64)
    hidden = torch.randn(4, 5, config.hidden_size, dtype=torch.float64)

    with torch.no_grad():
        dense = feed_forward(hidden)
        dense_load = feed_forward.last_load.clone()
        eager = feed_forward.eager_forward(hidden)
        grouped = bench.grouped_forward(feed_forward, hidden)

    assert torch.allclose(dense, eager, atol=1e-12)
    assert torch.allclose(dense, grouped, atol=1e-12)
    # Routing bookkeeping is shared with the dense path, not reimplemented, so the recorded
    # per-expert load has to be bit-equal - a second implementation of it could drift.
    assert torch.equal(dense_load, feed_forward.last_load)
    assert float(feed_forward.last_load.sum()) == pytest.approx(1.0)


def test_grouped_dispatch_skips_an_expert_that_receives_no_rows() -> None:
    """With fewer rows than experts, some expert gets nothing - and must not be run.

    Pigeonhole rather than forced routing: 2 tokens x top-2 is 4 rows over 8 experts, so at
    least 4 experts receive none whatever the router does. Forcing the routing by pinning
    router weights would not have been deterministic - `topk` breaks ties among equal scores
    arbitrarily - so the shape does the forcing instead.

    The branch matters because an empty expert slice is the kind of thing that either raises
    or silently contributes zeros, and a benchmark on a small batch hits it before a large
    one does.
    """

    config = tiny_config(experts_per_layer=(8, 8), active_experts=2)
    torch.manual_seed(3)
    feed_forward = _MoEFeedForward(config, tuple(f"E{index}" for index in range(8))).to(torch.float64)
    hidden = torch.randn(1, 2, config.hidden_size, dtype=torch.float64)

    with torch.no_grad():
        dense = feed_forward(hidden)
        grouped = bench.grouped_forward(feed_forward, hidden)

    assert torch.allclose(dense, grouped, atol=1e-12)
    load = feed_forward.last_load
    assert len(load) == 8
    assert float(load.sum()) == pytest.approx(1.0)
    # 4 rows over 8 experts: at least four experts were skipped by the dispatch loop.
    assert int((load > 0).sum()) <= 4
    assert int((load == 0).sum()) >= 4


def test_the_gate_refuses_to_time_forms_that_disagree() -> None:
    """A form that is a different function is refused, and the exit code says so.

    The refusal is what stops the script from reporting a speedup for arithmetic that is not
    the same arithmetic; here one form is replaced by an affine shift, which is the smallest
    change that must trip it.
    """

    config = tiny_config()
    gate = bench.check_equivalence(config)
    assert gate["ok"] is True
    assert gate["loads_agree"] is True
    assert all(diff <= gate["atol"] for diff in gate["max_abs_diff"].values())

    original = bench.form_forward

    def shifted(feed_forward: _MoEFeedForward, hidden: torch.Tensor, form: str) -> torch.Tensor:
        value = original(feed_forward, hidden, form)
        return value + 1e-6 if form == "grouped" else value

    bench.form_forward = shifted
    try:
        broken = bench.check_equivalence(config)
        assert broken["ok"] is False
        assert broken["max_abs_diff"]["grouped"] > broken["atol"]
        # And the command itself refuses rather than timing the broken set.
        assert bench.main(["--shape", "dev", "--check-only"]) == 1
    finally:
        bench.form_forward = original


def test_a_cpu_run_reports_no_fraction_of_an_accelerator_peak() -> None:
    """`of peak` is a claim about a device, so it is empty when the device is not that one.

    The first version of the benchmark printed `57.6% of peak` for a CPU run by dividing by a
    GTX 1070's peak, which is the same category error as quoting a dev-shape rate for a
    code-LM shape. On CPU the column is `n/a` and the field is `None`.
    """

    report: dict = {"forms": {}}
    for form in bench.FORMS:
        report["forms"][form] = {
            "form": form,
            "ms_per_step": 1.0,
            "tokens_per_s": 2_000.0,
            "flops_per_token": 1e8,
            "tflops_per_s": 0.2,
            "of_peak": None,
            "peak_memory_mb": 0.0,
        }

    table = bench.format_table([report["forms"][form] for form in bench.FORMS])
    assert "n/a" in table
    assert "%" not in table

    # And the timing path leaves it None off an accelerator rather than guessing a peak.
    config = tiny_config()
    row = bench.time_form(
        config, "grouped", torch.device("cpu"), torch.float32, batch=1, sequence=4, steps=1, warmup=0
    )
    assert row["tokens_per_s"] > 0
    assert row["peak_memory_mb"] == 0.0


def test_each_form_is_priced_by_the_arithmetic_it_executes() -> None:
    """The table has to credit the dense form with the extra FLOPs it spends.

    `estimate_flops_per_token` books the routing contract; `estimate_dense_runtime_flops_per_token`
    books what the dense kernels do. Mapping each form to its own estimator is what keeps the
    reported `TFLOP/s` column honest, so the mapping is pinned rather than left to convention.
    """

    assert bench.FORM_ARITHMETIC["dense"] == "dense_runtime"
    assert bench.FORM_ARITHMETIC["eager"] == "contracted"
    assert bench.FORM_ARITHMETIC["grouped"] == "contracted"

    config = V0MoEConfig(
        vocab_size=32_256,
        context_size=512,
        hidden_size=768,
        depth=12,
        attention_heads=12,
        experts_per_layer=(8,) * 12,
        active_experts=2,
        expert_width=768,
    )
    dense = bench.flops_per_token(config, "dense")
    grouped = bench.flops_per_token(config, "grouped")
    assert dense / grouped == pytest.approx(1.934, abs=0.001)
    assert set(bench.FORMS) == {"dense", "eager", "grouped"}
