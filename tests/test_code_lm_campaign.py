"""The campaign's two fairness invariants, which live in a shell script.

Nothing in the Python tests touches ``run_code_lm_campaign.sh``, and the two
properties that make its output a comparison rather than a curiosity are both
properties of how it *invokes* things:

* **the two arms train on the same data for the same number of steps.** They are
  cross-entropy and distillation at matched budget, so every flag that bounds the
  work has to be equal and ``--mode`` has to be the only difference. This was
  already wrong once in the other direction: the control arm read the full 162M-token
  stream while the distilled arm could only see the scored 15.36M, which would have
  credited distillation with a tenfold difference in data.
* **every benchmark row is scored on the same held-out text.** Bits per byte divides
  by the bytes of one text; two rows at different ``--bytes-limit`` values are two
  experiments. ``code_lm_table.py`` refuses such a table, but by then the hours are
  spent - the campaign is where it has to be right.

Parsing a shell script is an odd thing to do in a test, and it is the only place these
can be checked: the numbers the invariants protect are produced by running it, so the
check has to read the thing that runs.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
CAMPAIGN = ROOT / "scripts" / "run_code_lm_campaign.sh"

import code_bench  # noqa: E402
import code_lm_table  # noqa: E402
import push_code_bench  # noqa: E402


def invocations(script: str, needle: str) -> list[list[str]]:
    """Every command line mentioning ``needle``, with continuations joined."""
    joined = script.replace("\\\n", " ")
    return [line.split() for line in joined.splitlines() if needle in line]


def unquoted(value: str) -> str:
    """Shell quoting stripped, so a literal can be compared to one of the script's."""
    return value.strip().strip('"').strip("'")


def flags(argv: list[str]) -> dict[str, str]:
    """``--flag value`` pairs; a bare flag maps to ``"true"``."""
    out: dict[str, str] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token.startswith("--"):
            if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                out[token] = argv[index + 1]
                index += 1
            else:
                out[token] = "true"
        index += 1
    return out


@pytest.fixture(scope="module")
def script() -> str:
    return CAMPAIGN.read_text()


def test_the_two_arms_differ_only_in_their_objective(script: str) -> None:
    """Distillation against plain cross-entropy, at matched data and matched steps."""
    arms = invocations(script, "--stage train")
    assert len(arms) == 2, f"expected two training arms, found {len(arms)}"
    modes = [flags(arm)["--mode"] for arm in arms]
    assert sorted(modes) == ["ce", "distill"]

    first, second = flags(arms[0]), flags(arms[1])
    assert first.pop("--mode") != second.pop("--mode")
    assert first == second, (
        "the arms must bound their work identically or the difference between them is "
        f"not attributable to the objective: {first} vs {second}"
    )
    for required in ("--max-steps", "--batch-size", "--epochs", "--score-tokens", "--scale"):
        assert required in first, f"{required} must be pinned in the campaign, not left to a default"


def test_the_step_count_is_three_passes_over_what_the_teacher_labelled(script: str) -> None:
    """Evaluated by bash, because the arithmetic is bash's.

    Reimplementing ``$(( ))`` in Python would test the reimplementation.
    """
    line = next(
        (line for line in script.splitlines() if line.startswith("STEPS=")),
        None,
    )
    assert line is not None, "the campaign must derive STEPS rather than hardcode one"
    defaults = {"SCORE_TOKENS": "15360000", "EPOCHS": "3", "BATCH": "8"}
    program = "; ".join(f"{name}={value}" for name, value in defaults.items()) + f"; {line}; echo $STEPS"
    steps = int(subprocess.run(["bash", "-c", program], capture_output=True, text=True, check=True).stdout)
    tokens, epochs, batch = (int(defaults[name]) for name in ("SCORE_TOKENS", "EPOCHS", "BATCH"))
    assert steps == tokens // (batch * 512) * epochs
    assert steps * batch * 512 == tokens * epochs, "the arms must see EPOCHS passes over the labelled prefix"

    # And the arms use that derived value rather than a literal of their own.
    for arm in invocations(script, "--stage train"):
        assert flags(arm)["--max-steps"] == '"$STEPS"'


def test_every_benchmark_row_is_scored_on_the_same_held_out_text(script: str) -> None:
    rows = invocations(script, "code_bench.py --model")
    assert len(rows) == 4, f"four models are compared; found {len(rows)}"
    limits = {unquoted(flags(row)["--bytes-limit"]) for row in rows}
    assert limits == {"$BENCH_BYTES"}, f"rows disagree on the held-out text: {limits}"
    tags = [unquoted(flags(row)["--tag"]) for row in rows]
    assert len(set(tags)) == 4, f"tags must be distinct or rows overwrite each other: {tags}"


def test_the_benchmarked_models_are_the_ones_compared(script: str) -> None:
    """The teacher, the published baseline, and both arms - or the table is partial."""
    rows = {unquoted(flags(row)["--tag"]) for row in invocations(script, "code_bench.py --model")}
    assert rows == {"ours-$SCALE-distill", "ours-$SCALE-ce", "teacher", "baseline"}


def test_targets_are_scored_over_exactly_the_tokens_the_arms_train_on(script: str) -> None:
    """The scored prefix and the training window range must be the same number.

    Training on windows the teacher never scored would leave those positions with a
    zero KL term, which pulls the student toward the teacher's distribution on no
    evidence - and it looks like a smaller distillation loss, not like a bug.
    """
    targets = invocations(script, "--stage targets")
    assert len(targets) == 1
    scored = unquoted(flags(targets[0])["--score-tokens"])
    for arm in invocations(script, "--stage train"):
        assert unquoted(flags(arm)["--score-tokens"]) == scored


def test_the_tags_the_campaign_writes_are_the_roles_the_table_resolves(script: str) -> None:
    """A tag renamed in the shell script breaks the table, thirteen hours later.

    ``code_lm_table`` matches student roles as ``ours-<scale>-<arm>`` and the fixed two
    by name, and it refuses a missing role rather than printing a short table. That
    refusal is correct and, at the end of this campaign, also expensive - so the two
    spellings are checked against each other here, against the table's own resolver
    rather than against a copy of its rule.
    """
    tags = {unquoted(flags(row)["--tag"]).replace("$SCALE", "match")
            for row in invocations(script, "code_bench.py --model")}
    resolved = code_lm_table.resolve_roles({tag: {"bytes": {}} for tag in tags}, "match")
    assert set(resolved) == set(code_lm_table.ROLES)
    assert {tag for tag, _ in resolved.values()} == tags, "a tag no role claims is a row nobody reads"


def test_the_local_and_kaggle_rows_default_to_the_same_held_out_text(script: str) -> None:
    """One limit in three files, or the two halves of the evidence are not one experiment.

    The shell default is evaluated by bash rather than parsed, because it is written as
    ``${BENCH_BYTES:-...}`` and the arithmetic-free way to know what it resolves to is to
    ask the shell that resolves it.
    """
    line = next((line for line in script.splitlines() if line.startswith("BENCH_BYTES=")), None)
    assert line is not None
    resolved = subprocess.run(
        ["bash", "-c", f"unset BENCH_BYTES; {line}; echo $BENCH_BYTES"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert int(resolved) == code_bench.DEFAULT_BYTES_LIMIT
    assert push_code_bench.DEFAULT_BYTES_LIMIT == code_bench.DEFAULT_BYTES_LIMIT


def test_the_script_fails_fast_and_echoes_its_budget(script: str) -> None:
    assert "set -euo pipefail" in script, "a failed stage must stop the campaign, not be skipped"
    assert "log \"campaign:" in script, "the resolved budget has to be in the log, not only in the code"
    assert re.search(r'PY="?\.venv/bin/python"?', script), "the interpreter must be the project's"
