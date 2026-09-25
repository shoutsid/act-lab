"""The code-LM leg's sharp edges, pinned without a GPU, a network, or the teacher.

Everything here runs on CPU in milliseconds. The teacher weights are 2.7 GB and
deliberately not committed, so the tests that need its tokenizer skip by name when
it is absent - the same form ``tests/test_architecture_report.py`` uses for the
study shards.

Three of these exist for mistakes this leg actually made:

* the **tokenizer swap**. The teacher was changed mid-build, and a token stream
  encoded by one BPE is not stale under the other - it is well-formed and
  meaningless, which trains happily and reports nonsense.
* the **residual bucket**. The teacher's distribution is cached as a top-32 plus a
  single number holding everything else. Collapsing that tail and renormalising the
  top-32 would leave the student's own tail free, and nothing would look wrong.
* the **return type**. ``transformers`` causal LMs return a ``CausalLMOutputWithPast``
  and this project's model returns a tensor; code that assumes one dies on the other.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import train_code_lm as code_lm  # noqa: E402

TEACHER_PRESENT = (code_lm.TEACHER_DIR / "tokenizer.json").is_file()
needs_teacher = pytest.mark.skipif(
    not TEACHER_PRESENT,
    reason=f"teacher tokenizer not present at {code_lm.TEACHER_DIR} (it is downloaded, not committed)",
)


# --- the tokenizer swap -----------------------------------------------------


def test_a_stream_encoded_by_another_tokenizer_is_never_reused():
    """Length is the tempting check and it is the wrong one."""
    manifest = {"tokenizer_id": "someone-else@deadbeef", "complete": True}
    assert not code_lm.stream_cache_usable(160_000_000, 162_411_637, manifest, "deepseek@cafe")

    # Same manifest, same tokenizer: the complete flag carries a short stream.
    manifest = {"tokenizer_id": "deepseek@cafe", "complete": True}
    assert code_lm.stream_cache_usable(160_000_000, 162_411_637, manifest, "deepseek@cafe")


def test_an_incomplete_short_stream_is_rebuilt_but_a_long_one_is_not():
    manifest = {"tokenizer_id": "x", "complete": False}
    assert not code_lm.stream_cache_usable(160_000_000, 100_000_000, manifest, "x")
    assert code_lm.stream_cache_usable(100_000_000, 160_000_000, manifest, "x")


def test_a_manifest_with_no_tokenizer_id_is_rebuilt():
    """Absent is not equal: a manifest from before the field existed cannot vouch."""
    assert not code_lm.stream_cache_usable(10, 10, {}, "x")
    assert not code_lm.stream_cache_usable(10, 10, {"complete": True}, "x")


@needs_teacher
def test_the_tokenizer_id_follows_the_tokenizer_bytes(tmp_path, monkeypatch):
    original = code_lm.tokenizer_id()
    assert "deepseek-ai/deepseek-coder-1.3b-base@" in original

    other = tmp_path / "tokenizer.json"
    other.write_bytes(code_lm.teacher_tokenizer_path().read_bytes() + b"\n")
    monkeypatch.setattr(code_lm, "teacher_tokenizer_path", lambda: other)
    assert code_lm.tokenizer_id() != original


# --- the residual bucket ----------------------------------------------------


def _targets(ids, delta, top1, residual) -> dict:
    return {
        "ids": np.asarray(ids, dtype=np.int32),
        "delta": np.asarray(delta, dtype=np.float32),
        "top1": np.asarray(top1, dtype=np.float32),
        "residual": np.asarray(residual, dtype=np.float32),
    }


def test_a_certain_teacher_against_a_uniform_student_costs_log_vocab():
    """The KL of a point mass against a uniform distribution is exactly ln(V).

    A closed form, so this catches a wrong normalisation rather than merely a
    changed one.
    """
    vocab = 64
    targets = _targets([[[3, 9]]], [[[0.0, -50.0]]], [[0.0]], [[0.0]])
    student = torch.zeros(1, 1, vocab)
    kl, _, _ = code_lm.distillation_terms(student, targets, np.array([0]), torch.device("cpu"))
    assert kl.item() == pytest.approx(math.log(vocab), abs=1e-3)


def test_a_student_that_ignores_the_teachers_tail_is_penalised_for_it():
    """Half the teacher's mass is outside the top-32; a student certain of one
    token must pay for missing it. With the tail collapsed and renormalised this
    penalty would not exist."""
    targets = _targets([[[7]]], [[[0.0]]], [[math.log(0.5)]], [[0.5]])
    student = torch.full((1, 1, 64), -30.0)
    student[..., 7] = 30.0
    kl, _, _ = code_lm.distillation_terms(student, targets, np.array([0]), torch.device("cpu"))
    expected = 0.5 * math.log(0.5) + 0.5 * (math.log(0.5) - math.log(1e-6))
    assert kl.item() == pytest.approx(expected, abs=0.05)


def test_a_student_that_matches_the_teachers_tail_scores_zero():
    """The bucket must be *matchable*, not merely penalising.

    A student holding 0.5 on the stored token and 0.5 elsewhere has a residual of
    exactly 0.5, the teacher's residual, so the divergence is zero. A cache that
    stored the tail as a constant would fail this in the other direction.
    """
    targets = _targets([[[7]]], [[[0.0]]], [[math.log(0.5)]], [[0.5]])
    student = torch.zeros(1, 1, 64)
    student[..., 7] = math.log(63.0)
    kl, _, _ = code_lm.distillation_terms(student, targets, np.array([0]), torch.device("cpu"))
    assert kl.item() == pytest.approx(0.0, abs=1e-5)


def test_the_stored_delta_is_relative_to_the_top_token():
    """``delta`` is ``log p_i - log p_top1``, with ``log p_top1`` stored separately.

    Float16 resolves a difference near zero to ~1e-3 and a log-probability near -10
    to ~1e-2, which is why the cache stores the difference; this checks the
    reconstruction undoes that exactly.
    """
    top1 = -2.5
    deltas = [0.0, -1.0, -4.0]
    targets = _targets([[[1, 2, 3]]], [[deltas]], [[top1]], [[0.1]])
    student = torch.zeros(1, 1, 8)
    _, teacher_p, _ = code_lm.distillation_terms(student, targets, np.array([0]), torch.device("cpu"))
    expected = [math.exp(top1 + d) for d in deltas]
    assert teacher_p[0, 0].tolist() == pytest.approx(expected, abs=1e-9)


# --- the return type --------------------------------------------------------


def test_forward_logits_unwraps_a_transformers_output_and_passes_a_tensor_through():
    tensor = torch.zeros(1, 2, 3)

    class Output:
        logits = tensor

    class ReturnsTensor:
        def __call__(self, tokens):
            return tensor

    class ReturnsOutput:
        def __call__(self, tokens):
            return Output()

    assert code_lm.forward_logits(ReturnsTensor(), torch.zeros(1, 2)) is tensor
    assert code_lm.forward_logits(ReturnsOutput(), torch.zeros(1, 2)) is tensor


# --- geometry and the cache directory ---------------------------------------


def test_window_counts_ignore_a_partial_trailing_window():
    assert code_lm.window_count(1024) == 2
    assert code_lm.window_count(1023) == 1
    (windows, positions, k), flat = code_lm.targets_shapes(1536)
    assert (windows, positions) == (3, code_lm.CONTEXT_SIZE - 1)
    assert k == code_lm.TOP_K
    assert flat == (3, code_lm.CONTEXT_SIZE - 1)


def test_a_directory_without_write_permission_is_not_usable(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        assert not code_lm._nearest_writable(locked)
        assert not code_lm._nearest_writable(locked / "does" / "not" / "exist")
    finally:
        locked.chmod(0o700)
    assert code_lm._nearest_writable(locked)


# --- the teacher's identity, when it is on disk -----------------------------


@needs_teacher
def test_the_document_separator_is_a_real_token_of_the_teachers_tokenizer():
    separator = code_lm.boundary_token_id()
    from tokenizers import Tokenizer

    tokenizer = code_lm.shared_tokenizer()
    assert 0 <= separator < code_lm.teacher_vocab_size()
    # It must be encodable: a separator the BPE cannot produce is not in the stream.
    assert tokenizer.id_to_token(separator) is not None


@needs_teacher
def test_the_student_is_sized_to_the_models_output_width_not_the_bpes():
    """The two numbers differ (32,256 rows against 32,022 emitted ids).

    Sized to the smaller one, a teacher top-k landing on a reserved row would index
    past the student's logits. The student must be built on the model's number.
    """
    assert code_lm.teacher_vocab_size() == 32_256
    assert code_lm.vocab_size() < code_lm.teacher_vocab_size()
    assert code_lm.student_config("match").vocab_size == code_lm.teacher_vocab_size()


@needs_teacher
def test_the_match_student_is_in_the_baselines_parameter_class():
    """``match`` is named for the comparison it exists to make.

    The baseline (``bigcode/tiny_starcoder_py``) is 164M dense. The student is
    allowed to differ in total parameters - it is a different architecture - but not
    so far that "same class" stops meaning anything. It activates far fewer, which
    is the project's claim rather than a caveat.
    """
    from act.model.v0_moe import V0MoEConfig, estimated_parameter_count

    config = code_lm.student_config("match")
    assert isinstance(config, V0MoEConfig)
    total, active = estimated_parameter_count(config)
    assert 120_000_000 < total < 210_000_000
    assert active < total / 1.5
