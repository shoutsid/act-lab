"""The code-LM leg's first two measured numbers, checked against what produced them.

The Kaggle half of this leg spent GPU quota and, until the driver grew ``--save``,
produced nothing but scrollback: ``kaggle kernels output`` writes into a temporary
directory that the fetch then deleted. Those two reports are committed now, and this is
what keeps them from becoming a claim nobody can re-read - the numbers in
``docs/IMPLEMENTATION_STATUS.md`` are read back out of the artifacts here, through the
same reader the comparison table uses.

Two things about them are load-bearing rather than incidental:

* **the ragged tail is real and is recorded.** The teacher's tokenizer makes 1,289,028
  tokens of the 4 MB prefix and the baseline's makes 1,106,839, neither a multiple of
  512: every scoring window is 513 tokens except the last, which was 407 tokens for the
  baseline. Stacking that batch is a ``ValueError``, which is how the first Kaggle run
  died after a 2.7 GB download, and this artifact is the case that failed.
* **the two rows are on the same text.** Bits per byte is only comparable at one
  ``bytes_limit``, and these are the reference rows the student is measured against.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import code_bench  # noqa: E402
import code_lm_table  # noqa: E402

KAGGLE = ROOT / "artifacts" / "code-lm" / "bench" / "kaggle"

#: The two numbers the journal quotes, as the journal quotes them. Rounded to four
#: decimals here rather than compared as floats, because that is the precision the
#: measurement supports and the precision the record is written at.
DOCUMENTED = {"teacher": 0.4686, "baseline": 0.5372}

#: The held-out prefix the whole leg is scored on, including the campaign's local rows.
BYTES_LIMIT = 4_000_000


def report(name: str) -> dict:
    path = KAGGLE / f"{name}.json"
    assert path.is_file(), f"{path} is committed evidence; a missing file means the record quotes nothing"
    return json.loads(path.read_text())


def test_the_two_committed_rows_are_the_numbers_the_record_quotes() -> None:
    for name, expected in DOCUMENTED.items():
        bits = report(name)["bytes"]["bits_per_byte"]
        assert round(bits, 4) == expected, (
            f"{name} now scores {bits:.6f} ({bits:.4f}); the journal quotes {expected}. "
            "A re-scored row means the journal entry has to be updated, not that this "
            "assertion is in the way."
        )


def test_both_rows_were_scored_on_the_same_held_out_text() -> None:
    limits = {name: report(name)["bytes_limit"] for name in DOCUMENTED}
    assert set(limits.values()) == {BYTES_LIMIT}, f"rows disagree on the text: {limits}"
    assert {report(name)["bytes"]["held_out_bytes"] for name in DOCUMENTED} == {BYTES_LIMIT}
    # Whatever the limit is, it is the one the local campaign and the driver default to.
    assert code_bench.DEFAULT_BYTES_LIMIT == BYTES_LIMIT


def test_the_table_reader_accepts_them() -> None:
    """The consumer's own reader, not a second implementation of the same check."""
    for name in DOCUMENTED:
        block = report(name)["bytes"]
        losses = code_lm_table.window_losses(name, report(name))
        assert len(losses) == block["windows"]
        assert sum(losses) == block["nats"] or math.isclose(sum(losses), block["nats"], rel_tol=1e-4)
        assert code_lm_table.report_tokenizer(name, report(name))


def test_the_recorded_windows_end_in_a_short_one() -> None:
    """The failing case of the first Kaggle run, now an artifact rather than an anecdote.

    512 tokens per window means the last window is whatever is left over, and for both of
    these tokenizers that is a partial window of a few hundred tokens.
    """
    for name in DOCUMENTED:
        block = report(name)["bytes"]
        # Windows tile the stream, so every token is scored once and one token per window
        # has no context: the scored count is the token count minus one, and the windows
        # before the last are full.
        scored = block["tokens"] - 1
        tail = scored - (block["windows"] - 1) * code_bench.WINDOW
        assert 0 < tail < code_bench.WINDOW, f"{name}'s last window is {tail} scored tokens"
        losses = code_lm_table.window_losses(name, report(name))
        assert len(losses) == block["windows"]
        assert losses[-1] > 0, "the tail is scored, not padded to a full window's cost"


def test_no_stage_of_the_kernel_run_failed() -> None:
    """The kernel reports this itself, and the fetch prints it - an empty summary with
    ``failed: []`` and an empty summary with ``failed: [teacher]`` look the same in a log."""
    summary = json.loads((KAGGLE / "code-bench-summary.json").read_text())
    assert summary["failed"] == []
    assert set(summary["summary"]) == {"teacher.json", "baseline.json"}
    assert summary["suite"] == "bytes", "HumanEval is left local on purpose; see the driver's docstring"
