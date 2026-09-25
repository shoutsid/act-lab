"""Tests for the arm-parity check.

The check exists because the leg's central claim - both arms see the same data, in the same
order, for the same budget, from the same seed, differing only in the objective - was checked
against the shell script's flags and not against what the arms did. It was already false once
on disk: a 2 GB checkpoint from a 20-step smoke run sat where the control arm's checkpoint
goes, and adopting it would have started that arm from another run's weights at step 21 while
still reporting the full budget.

These pin the two halves that matter: that a genuine asymmetry is named with its path, and
that the fields which are *supposed* to differ are not reported as defects - a check that
cries wolf on its own result field is a check nobody runs twice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import check_code_lm_arms as check  # noqa: E402


def arm(mode: str = "distill", **overrides) -> dict:
    """A report shaped exactly like the one ``train_code_lm.py --stage train`` writes."""
    training = {
        "steps": 11_250,
        "batch_size": 8,
        "batch_tokens": 4_096,
        "epochs_over_stream": 3,
        "tokens_budget": 46_080_000,
        "tokens_seen": 46_080_000,
        "steps_trained": 11_250,
        "resumed_from_step": 0,
        "run": {
            "scale": "match",
            "mode": mode,
            "seed": 11,
            "batch_size": 8,
            "max_steps": 11_250,
            "windows": 30_000,
        },
        "train_windows": 30_000,
        "train_tokens": 15_360_000,
        "stream_tokens": 162_411_637,
        "optimizer": "AdamW warmup+cosine",
        "optimizer_state": "fresh (no checkpoint)",
        "seed": 11,
        "device": "cuda",
    }
    training.update(overrides.pop("training", {}))
    report = {
        "leg": "code-lm",
        "corpus": "codeparrot/codeparrot-clean-valid",
        "corpus_source": "https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid",
        "tokenizer": "deepseek-ai/deepseek-coder-1.3b-base@ef48ebdc8546c2d8",
        "teacher": {"repo": "deepseek-ai/deepseek-coder-1.3b-base"},
        "baseline_same_class": {"repo": "bigcode/tiny_starcoder_py"},
        "mode": mode,
        "distill_weight": 0.5 if mode == "distill" else 0.0,
        "student": {"scale": "match", "total_parameters": 166_423_296},
        "training": training,
        "validation": {
            "final_val_loss": 1.9000,
            "final_perplexity": 6.69,
            "bits_per_byte": 0.6100 if mode == "distill" else 0.6400,
            "bits_per_byte_measured_on": "the whole held-out stream as the student tokenizes it",
            "uniform_floor_loss": 10.3766,
            "held_out_bytes": 28_682_299,
            "history": [],
        },
        "references": {"uniform": {"bits_per_byte": 4.8154}},
        "samples": {mode: "generated text"},
    }
    report.update(overrides)
    return report


def run(monkeypatch, tmp_path: Path, distill: dict, control: dict, *argv: str) -> int:
    """Call the checker the way the runbook does: on files, with its own argv."""
    (tmp_path / "report-match-distill.json").write_text(json.dumps(distill))
    (tmp_path / "report-match-ce.json").write_text(json.dumps(control))
    monkeypatch.setattr(check, "ART", tmp_path)
    monkeypatch.setattr(sys, "argv", ["check_code_lm_arms.py", *argv])
    return check.main()


def test_the_two_arms_differing_only_in_their_objective_pass(tmp_path, monkeypatch, capsys):
    """The happy path, including everything the objective legitimately changes."""
    assert run(monkeypatch, tmp_path, arm("distill"), arm("ce")) == 0
    printed = capsys.readouterr().out
    assert "differ in their objective and nothing else" in printed
    assert "gain +0.0300 bits/byte" in printed, "the direction is control minus distilled"
    assert "not the table's" in printed, "two bits-per-byte numbers exist and this is not that one"


def test_an_asymmetry_is_named_with_its_path(tmp_path, monkeypatch, capsys):
    """A seed is one number in a report of a hundred, so the check has to say where to look."""
    assert run(monkeypatch, tmp_path, arm("distill"), arm("ce", training={"seed": 12})) == 1
    printed = capsys.readouterr().out
    assert "not one experiment" in printed
    assert "training.seed: 11 vs 12" in printed


def test_the_fields_that_are_allowed_to_differ_are_not_defects():
    """Every one of these differs in a real campaign and none of them is a finding."""
    assert check.differences(arm("distill"), arm("ce")) == []


def test_a_data_difference_is_a_defect_even_when_the_budget_looks_the_same():
    """The first defect this leg had: one arm on 162M tokens, the other on the scored 15.36M.

    Both reports say the same ``steps``, so pinning the step count is not enough - the window
    range is what the arms actually saw, and it has to be compared.
    """
    control = arm("ce", training={"train_windows": 317_210, "train_tokens": 162_411_637})
    found = check.differences(arm("distill"), control)
    assert "training.train_windows: 30000 vs 317210" in found
    assert "training.train_tokens: 15360000 vs 162411637" in found


def test_a_resumed_arm_is_printed_rather_than_refused(tmp_path, monkeypatch, capsys):
    """A resume is faithful now - weights, optimiser moments and the draw sequence all come
    back - so it is bookkeeping to show, not a reason to refuse the comparison."""
    control = arm(
        "ce",
        training={
            "tokens_seen": 11_230 * 8 * 512,
            "steps_trained": 11_230,
            "resumed_from_step": 20,
            "optimizer_state": "restored by name (110 parameters)",
        },
    )
    assert run(monkeypatch, tmp_path, arm("distill"), control) == 0
    printed = capsys.readouterr().out
    assert "of a 46,080,000 token budget after a resume" in printed
    assert "restored by name" in printed


def test_a_missing_arm_says_which_command_writes_it(tmp_path, monkeypatch):
    (tmp_path / "report-match-distill.json").write_text(json.dumps(arm("distill")))
    monkeypatch.setattr(check, "ART", tmp_path)
    monkeypatch.setattr(sys, "argv", ["check_code_lm_arms.py"])
    with pytest.raises(SystemExit, match="run_code_lm_campaign"):
        check.main()


def test_a_report_from_another_leg_is_refused(tmp_path, monkeypatch):
    """The artifacts directory holds other legs' reports; one of them is not this comparison."""
    monkeypatch.setattr(sys, "argv", ["check_code_lm_arms.py"])
    with pytest.raises(SystemExit, match="not a report of this leg"):
        run(monkeypatch, tmp_path, arm("distill"), arm("ce", leg="tiny-stories"))


def test_the_json_form_records_what_was_compared_and_what_was_found(tmp_path, monkeypatch):
    """So that "the arms agreed" is a claim with an artifact behind it, not a printed line."""
    out = tmp_path / "arms.json"
    control = arm("ce", training={"batch_size": 4})
    assert run(monkeypatch, tmp_path, arm("distill"), control, "--json", str(out)) == 1
    saved = json.loads(out.read_text())
    assert saved["differences"] == ["training.batch_size: 8 vs 4"]
    assert "validation" in saved["result_fields"], "the result fields are recorded, not implied"
    assert sorted(saved["agreed"]) == saved["agreed"]
