"""Tests for what the analyzer's ``--write`` leaves behind, and how it is labelled.

A pooled analysis (``--also``) reads two arms' ledgers but writes into the primary
one's directory, so its ``search_cost.json`` holds a total larger than the arm it
sits beside: the crossed arm's file reads 6.53 GPU-hours while its own ledgers hold
3.28, the other half being the confirmation arm's. An auditor summing those files -
or reading one as that arm's cost - double-counts an arm, so a pooled write has to
carry the experiments it spans *and* their split, and a single-arm write has to stay
free of pooling fields so the two are distinguishable without reading the numbers.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from act.search.architecture_eval import EvalResult  # noqa: E402
from arch_zoo_analyze import cost_by_experiment, main  # noqa: E402

STEPS = 400
TOKENS = STEPS * 2 * 1024


def result(name: str, validation_loss: float, seed: int = 42) -> EvalResult:
    return EvalResult(
        name=name,
        family="test",
        genome_hash=f"hash-{name}",
        seed=seed,
        total_params=100_000_000,
        active_params=40_000_000,
        contracted_flops_per_token=2.5e8,
        dense_runtime_flops_per_token=4.0e8,
        training_flops=2.5e8 * TOKENS,
        validation_flops=2.5e8 * 16 * 2 * 1024,
        final_train_loss=validation_loss + 0.2,
        mean_train_loss=validation_loss + 0.25,
        validation_loss=validation_loss,
        steps=STEPS,
        train_tokens=TOKENS,
        val_tokens=16 * 2 * 1024,
        elapsed_s=170.0,
        measured_step_s=0.44,
        measured_tokens_per_s=4700.0,
        peak_memory_mb=3400.0,
        device="cuda",
        description="",
        val_source="holdout:2000000",
    )


def arm(root: Path, name: str, seeds: tuple[int, ...], names: tuple[str, ...]) -> Path:
    """A minimal arm directory: a definition and one ledger per seed."""
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "zoo_definition.json").write_text(
        json.dumps({"candidates": [{"name": candidate} for candidate in names]})
    )
    for index, seed in enumerate(seeds):
        rows = [
            asdict(result(candidate, 3.5 + 0.05 * position, seed=seed))
            for position, candidate in enumerate(names)
        ]
        ledger = directory / f"flat1200_seed{seed}_results.jsonl"
        ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return directory


def test_the_split_is_per_experiment_not_an_aggregate(tmp_path):
    """Each arm's own runs and hours, which is what a cost table quotes."""
    primary = arm(tmp_path, "crossed", (42, 43), ("a", "b"))
    other = arm(tmp_path, "confirm", (45,), ("a", "b"))

    split = cost_by_experiment([primary, other], candidates=2)
    assert set(split) == {"crossed", "confirm"}
    # Keys match the cost dict's own, so a share can be compared against the total
    # without translating names.
    assert set(split["crossed"]) == {"runs_completed", "tokens_trained", "wall_clock_hours"}
    assert split["crossed"]["runs_completed"] == 4
    assert split["confirm"]["runs_completed"] == 2
    # Hours follow the tokens each arm trained, so they add to the pooled total.
    assert split["crossed"]["wall_clock_hours"] > split["confirm"]["wall_clock_hours"]
    assert split["crossed"]["tokens_trained"] == 4 * TOKENS
    assert split["confirm"]["tokens_trained"] == 2 * TOKENS


def test_a_pooled_write_labels_its_cost_file_with_what_it_spans(tmp_path, monkeypatch, capsys):
    """The written file is the artifact an auditor reads, not the return value.

    This is the end-to-end version of the split: the file that lands beside the
    primary arm must name both experiments and decompose its own total, because the
    total alone is unattributable from the directory it sits in.
    """
    primary = arm(tmp_path, "crossed", (42, 43), ("a", "b", "c"))
    arm(tmp_path, "confirm", (45,), ("a", "b", "c"))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "arch_zoo_analyze.py",
            "--artifacts-root", str(tmp_path),
            "--experiment-id", "crossed",
            "--also", "confirm",
            "--write",
        ],
    )
    main()

    cost = json.loads((primary / "search_cost.json").read_text())
    assert cost["pooled_experiments"] == ["crossed", "confirm"]
    assert set(cost["by_experiment"]) == {"crossed", "confirm"}
    assert cost["by_experiment"]["crossed"]["runs_completed"] == 6
    assert cost["by_experiment"]["confirm"]["runs_completed"] == 3
    # The pooled total is the sum of the split, which is the property that makes
    # double-counting detectable rather than invisible.
    assert cost["runs_completed"] == 9
    assert cost["tokens_trained"] == sum(
        row["tokens_trained"] for row in cost["by_experiment"].values()
    )
    assert cost["wall_clock_hours"] == pytest.approx(
        sum(row["wall_clock_hours"] for row in cost["by_experiment"].values())
    )
    assert (primary / "decision.md").exists()
    assert "Pooled analysis" in cost["pooled_note"]
    assert "Cost by experiment" in capsys.readouterr().out


def test_a_single_arm_write_carries_no_pooling_fields(tmp_path, monkeypatch):
    """The absence of the field is meaningful: it says the file is one arm's own."""
    primary = arm(tmp_path, "solo", (42, 43), ("a", "b", "c"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "arch_zoo_analyze.py",
            "--artifacts-root", str(tmp_path),
            "--experiment-id", "solo",
            "--write",
        ],
    )
    main()
    cost = json.loads((primary / "search_cost.json").read_text())
    assert "pooled_experiments" not in cost
    assert "by_experiment" not in cost
    assert cost["runs_completed"] == 6


def test_a_directory_with_no_ledgers_is_skipped_rather_than_reported_as_zero(tmp_path):
    """A zero would read as an arm that ran for free."""
    primary = arm(tmp_path, "crossed", (42,), ("a",))
    empty = tmp_path / "not-run"
    empty.mkdir()
    split = cost_by_experiment([primary, empty], candidates=1)
    assert set(split) == {"crossed"}
