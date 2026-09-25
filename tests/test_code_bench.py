"""The benchmark harness's units and its execution path, pinned on CPU.

A benchmark is only as good as its arithmetic. Three things here are easy to get
wrong in a way that still produces a plausible number:

* **bits per byte** divides by the bytes of the *text*, not by a token count. A
  token count is a property of the tokenizer, so dividing by it would make a 32k
  student and a 49k baseline incomparable while looking like a fair table.
* **the window scan** scores every token but the first of each window. That
  omission is fine because every model carries it, and it is a bug the moment one
  model is scored by another code path.
* **completion truncation**. Without it the harness measures how long a model
  keeps writing rather than whether its answer was right.
"""

from __future__ import annotations

import gzip
import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import code_bench  # noqa: E402

HUMANEVAL_PRESENT = code_bench.HUMANEVAL.is_file()


def test_bits_per_byte_divides_nats_by_bytes_not_tokens():
    assert code_bench.bits_per_byte(math.log(2), 10) == pytest.approx(0.1)
    # One nat per byte is log2(e) bits per byte, a useful fixed point.
    assert code_bench.bits_per_byte(1.0, 1) == pytest.approx(1 / math.log(2))


class _Uniform:
    """A model that always predicts a flat distribution over ``vocab`` tokens.

    Its loss is ln(V) for every scored token, so any error in how many tokens are
    scored shows up in the total directly instead of hiding in a rate.
    """

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, tokens):
        return torch.zeros(tokens.shape[0], tokens.shape[1], self.vocab)


def test_window_scan_scores_every_token_but_the_first_of_each_window():
    """A uniform model's loss is ln(V) per scored token, so the count is visible.

    Three windows of the harness's own width, plus slack: the omission is one
    token per window, and the assertion is on the total rather than on a rate so
    the omission cannot hide inside a division.
    """
    vocab = 12
    total_tokens = code_bench.WINDOW * 3 + 1
    ids = np.arange(total_tokens, dtype=np.int32) % vocab

    nats, per_window = code_bench.window_nlls(_Uniform(vocab), ids, torch.device("cpu"), batch=2)
    expected_scored = 3 * code_bench.WINDOW
    assert nats == pytest.approx(expected_scored * math.log(vocab), rel=1e-6)
    assert len(per_window) == 3
    assert sum(per_window) == pytest.approx(nats, rel=1e-6)
    assert all(value == pytest.approx(code_bench.WINDOW * math.log(vocab), rel=1e-6) for value in per_window)


def test_the_per_window_nlls_are_a_paired_test_the_table_can_use():
    """The returned list is the part that makes a comparison testable.

    Two models that share a tokenizer see identical window boundaries, so their
    difference is one number per window and can be tested paired. That only works
    if the list is per window, in order, and its length is the window count the
    report advertises - a shuffled or batched list would still sum to the total
    and would silently break the pairing.
    """
    vocab = 12
    total_tokens = code_bench.WINDOW * 4 + 1
    ids = np.arange(total_tokens, dtype=np.int32) % vocab

    wider = 33  # ln(33) - ln(12) nats per token worse than the first model
    base_nats, base = code_bench.window_nlls(_Uniform(vocab), ids, torch.device("cpu"), batch=3)
    worse_nats, worse = code_bench.window_nlls(_Uniform(wider), ids, torch.device("cpu"), batch=3)
    assert base == sorted(base)  # a uniform model is flat, so every window is equal
    assert len(base) == len(worse) == 4
    # Identical windows, so the per-window difference is a constant, window by window.
    shift = code_bench.WINDOW * (math.log(wider) - math.log(vocab))
    for low, high in zip(base, worse):
        assert high - low == pytest.approx(shift, rel=1e-6)
    assert worse_nats - base_nats == pytest.approx(4 * shift, rel=1e-6)


def test_a_window_scan_reports_one_value_per_non_overlapping_window():
    """Windows tile the stream: the count follows from the length, not from the batch.

    A window is one token longer than ``WINDOW`` because the last token is a target
    and the first is its context, so three windows need ``3 * WINDOW + 1`` tokens.
    """
    vocab = 5
    ids = np.arange(code_bench.WINDOW * 3 + 1, dtype=np.int32) % vocab

    _, per_window = code_bench.window_nlls(_Uniform(vocab), ids, torch.device("cpu"), batch=1)
    assert len(per_window) == 3


def test_a_tail_shorter_than_a_window_is_scored_rather_than_crashing_the_scan():
    """The regression the Kaggle byte pass died of, on the real shape of the data.

    The held-out text never divides evenly into windows: the baseline's tokenizer made
    1,106,839 tokens of the 4 MB prefix, so the last window is 407 tokens and the batch
    holding it is ragged - ``np.stack`` refuses that with "all input arrays must have
    the same shape", after the run has downloaded 2.7 GB of teacher. The tail is a
    window like any other, just a short one, and it is scored.
    """
    vocab = 7
    tail = 407
    ids = np.arange(code_bench.WINDOW * 3 + tail, dtype=np.int32) % vocab

    total, per_window = code_bench.window_nlls(_Uniform(vocab), ids, torch.device("cpu"), batch=4)

    assert len(per_window) == 4
    assert per_window[-1] == pytest.approx((tail - 1) * math.log(vocab), rel=1e-6)
    # Every token except the first is scored exactly once, tail included.
    assert total == pytest.approx((len(ids) - 1) * math.log(vocab), rel=1e-6)
    # The batch is a performance choice, not part of the measurement.
    for batch in (1, 2, 8):
        again, windows = code_bench.window_nlls(_Uniform(vocab), ids, torch.device("cpu"), batch=batch)
        assert again == pytest.approx(total, rel=1e-6)
        assert windows == per_window


def test_a_trailing_fragment_too_short_to_carry_context_is_dropped():
    """A handful of leftover tokens is not a window: there is too little context
    for the loss on it to mean anything, and counting it would weight it as equal
    to a full window. The last full window is kept and the fragment is not."""
    vocab = 5
    ids = np.arange(code_bench.WINDOW + 3, dtype=np.int32) % vocab

    total, per_window = code_bench.window_nlls(_Uniform(vocab), ids, torch.device("cpu"), batch=1)
    assert len(per_window) == 1
    assert total == pytest.approx(code_bench.WINDOW * math.log(vocab), rel=1e-6)


def test_a_stream_too_short_to_score_is_refused_rather_than_answered_with_nan():
    ids = np.zeros(4, dtype=np.int32)
    with pytest.raises(SystemExit, match="too short"):
        code_bench.window_nlls(_Uniform(3), ids, torch.device("cpu"))


def test_a_completion_is_cut_at_the_first_top_level_boundary():
    assert code_bench.truncate_at_stop("    return 1\ndef other():") == "    return 1"
    assert code_bench.truncate_at_stop("    return 1\n\n\nimport os") == "    return 1"
    assert code_bench.truncate_at_stop("    return 1\n") == "    return 1\n"


def test_the_held_out_prefix_is_cut_on_a_character_boundary(tmp_path, monkeypatch):
    """The cut lands on a byte count, and a byte count can fall inside a character."""
    text = "a" * 8 + "é" + "b" * 8  # 17 characters, 18 bytes
    path = tmp_path / "val-text.txt"
    path.write_text(text)
    monkeypatch.setattr(code_bench, "VAL_TEXT", path)

    assert code_bench.held_out_text(0) == text
    cut = code_bench.held_out_text(9)
    assert cut == "a" * 8  # the half of the two-byte character is dropped, not mangled
    assert len(cut.encode()) <= 9
    assert len(code_bench.held_out_text(100).encode()) == 18


def test_a_missing_held_out_text_asks_for_the_stage_that_writes_it(tmp_path, monkeypatch):
    monkeypatch.setattr(code_bench, "VAL_TEXT", tmp_path / "absent.txt")
    with pytest.raises(SystemExit, match="train_code_lm"):
        code_bench.held_out_text(0)


def test_a_checkpoint_path_is_reported_in_a_form_both_callers_can_write(tmp_path, monkeypatch):
    """The campaign passes a repository-relative path, a reader passes an absolute one.

    ``Path.relative_to`` on the argument as given raises for the first - a relative path
    against an absolute root - and it raised *after* the model had loaded and the whole
    scan had run, which is how it survived the tests and reached a smoke run.
    """
    monkeypatch.chdir(code_bench.ROOT)
    relative = Path("artifacts/code-lm/checkpoints/match-distill.pt")
    assert code_bench.display_path(relative) == str(relative)
    assert code_bench.display_path(code_bench.ROOT / relative) == str(relative)
    # A checkpoint outside the repository is named where it is, not forced into a shape
    # it does not have.
    outside = tmp_path / "elsewhere.pt"
    assert code_bench.display_path(outside) == str(outside)


def test_model_logits_accepts_both_shapes_a_scorer_meets():
    tensor = torch.zeros(1, 2, 3)

    class Output:
        logits = tensor

    class ReturnsOutput:
        def __call__(self, tokens):
            return Output()

    class ReturnsTensor:
        def __call__(self, tokens):
            return tensor

    assert code_bench.model_logits(ReturnsOutput(), torch.zeros(1, 2)) is tensor
    assert code_bench.model_logits(ReturnsTensor(), torch.zeros(1, 2)) is tensor


def test_the_decoding_defaults_are_the_published_harness_protocol():
    """512 tokens, temperature 0.2, top_p 0.95 - what the baseline's 7.84% was drawn under.

    The number is pinned here rather than only in a docstring because the comparison it
    protects is against an external published figure: a stricter protocol of our own
    (256 tokens, no nucleus cut) reads as the model underperforming its own paper. And
    the generators have to default to the same value the CLI does, or one row is drawn
    under a protocol the report does not state.
    """
    assert (code_bench.DEFAULT_MAX_NEW_TOKENS, code_bench.DEFAULT_TEMPERATURE) == (512, 0.2)
    assert code_bench.DEFAULT_TOP_P == 0.95
    for factory in (code_bench.make_hf_generator, code_bench.make_ours_generator):
        default = inspect.signature(factory).parameters["top_p"].default
        assert default == code_bench.DEFAULT_TOP_P, f"{factory.__name__} would sample its own way"


def test_greedy_decoding_is_the_argmax():
    logits = torch.tensor([0.1, 2.0, 0.5])
    generator = torch.Generator().manual_seed(0)
    assert code_bench.sample_token(logits, 0.0, 0.95, generator) == 1
    assert code_bench.sample_token(logits, -1.0, 0.95, generator) == 1, "0 disables sampling, not a cut"


def test_a_token_outside_the_nucleus_is_never_drawn():
    """The cut is a filter on the distribution, not a preference."""
    logits = torch.tensor([1.0, 1.0, -20.0])
    generator = torch.Generator().manual_seed(0)
    drawn = {code_bench.sample_token(logits, 1.0, 0.95, generator) for _ in range(200)}
    assert drawn == {0, 1}, "the 20-nat tail is outside a 0.95 nucleus and must not appear"


def test_a_tight_cut_on_a_peaked_distribution_collapses_to_one_token():
    logits = torch.tensor([8.0, 0.0, 0.0])
    generator = torch.Generator().manual_seed(0)
    assert {code_bench.sample_token(logits, 1.0, 0.5, generator) for _ in range(50)} == {0}


def test_the_sampler_always_returns_a_usable_token_index():
    """The generators feed the result straight back in as an id."""
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(4096)
    for temperature in (0.0, 0.2, 1.0):
        drawn = code_bench.sample_token(logits, temperature, 0.95, generator)
        assert isinstance(drawn, int) and 0 <= drawn < 4096


# --- the execution path -----------------------------------------------------


PROBLEM = {
    "task_id": "HumanEval/0",
    "prompt": 'def add(a, b):\n    """Add two numbers."""\n',
    "entry_point": "add",
    "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n",
}


def test_the_exit_code_is_the_verdict():
    """Pass means exit zero, which is how the published problem tests signal success.

    The test bodies are ``assert`` statements with no return value, so there is no
    value to inspect: the only channel is whether the process came back clean.
    """
    assert code_bench.run_program("pass", timeout=5)
    assert code_bench.run_program("raise SystemExit(0)", timeout=5)
    assert not code_bench.run_program("raise ValueError('nope')", timeout=5)
    assert not code_bench.run_program("raise SystemExit(1)", timeout=5)


def test_a_program_that_never_terminates_is_not_a_pass():
    """The model writes the code, so the timeout is the safety mechanism."""
    assert not code_bench.run_program("while True:\n    pass", timeout=1.0)


def test_a_correct_completion_passes_and_a_wrong_one_does_not():
    """End to end through the harness: assembly, truncation, execution, counting.

    The completion is the canonical solution, so a pass is the only correct
    outcome - if the prompt or the test were concatenated wrongly this fails. The
    second generator returns a plausible-looking wrong answer, which must not pass;
    a harness that reports a pass for both would be measuring nothing.
    """
    def correct(prompt, seed):
        return "    return a + b\n"

    def wrong(prompt, seed):
        return "    return a - b\n"

    passed = code_bench.score_humaneval(correct, [PROBLEM], timeout=5, samples=1, seed=0)
    assert passed["pass_at_1"] == 1.0
    assert passed["records"][0]["passed"] is True

    failed = code_bench.score_humaneval(wrong, [PROBLEM], timeout=5, samples=1, seed=0)
    assert failed["pass_at_1"] == 0.0
    assert failed["records"][0]["samples"] == [False]


def test_pass_at_1_averages_over_problems_rather_than_over_samples():
    """One pass out of two problems is 0.5, and the count must not be carried by
    the first problem alone."""
    def first_wrong_then_right(prompt, seed):
        index = first_wrong_then_right.calls
        first_wrong_then_right.calls = index + 1
        return "    return a + b\n" if index else "    return a - b\n"

    first_wrong_then_right.calls = 0
    problems = [PROBLEM, dict(PROBLEM, task_id="HumanEval/1")]
    result = code_bench.score_humaneval(first_wrong_then_right, problems, timeout=5, samples=1, seed=0)
    assert result["passed"] == 1
    assert result["pass_at_1"] == pytest.approx(0.5)
    assert [record["passed"] for record in result["records"]] == [False, True]


@pytest.mark.skipif(not HUMANEVAL_PRESENT, reason="HumanEval not downloaded (it is fetched, not committed)")
def test_the_real_humaneval_has_the_shape_the_harness_assumes():
    problems = code_bench.load_humaneval()
    assert len(problems) == 164
    for problem in problems[:5]:
        assert set(problem) >= {"task_id", "prompt", "entry_point", "test", "canonical_solution"}
        assert f"check({problem['entry_point']})" not in problem["test"]


@pytest.mark.skipif(not HUMANEVAL_PRESENT, reason="HumanEval not downloaded (it is fetched, not committed)")
def test_the_canonical_solution_passes_the_published_test():
    """The harness's claim to be executing the real benchmark.

    Runs the reference solution for the first few problems through exactly the
    assembly used for model output. If the tests or the truncation disagreed with
    the published data, this is where it shows.
    """
    for problem in code_bench.load_humaneval()[:5]:
        program = (
            problem["prompt"]
            + code_bench.truncate_at_stop(problem["canonical_solution"])
            + "\n"
            + problem["test"]
            + f"\ncheck({problem['entry_point']})"
        )
        assert code_bench.run_program(program, timeout=10), problem["task_id"]
