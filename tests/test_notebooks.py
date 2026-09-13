"""Tests for the reproducible Kaggle notebooks (plan deliverable 2).

The notebooks are generated deterministically by ``scripts/build_notebooks.py``
so they can be versioned and reviewed like code: the builder must render
byte-identical files on every run, and every code cell must wrap real
code-path contracts (plan sections 39 and 104) that stay importable. The
cells' full execution is deliberately NOT part of the pytest suite (some run
training loops); it lives in the same manual cell-runner pattern as
``scripts/v0_session_selfcheck.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_notebooks  # noqa: E402

NOTEBOOK_NAMES = (
    "00_environment.ipynb",
    "01_dense_baseline.ipynb",
    "02_random_topology_churn.ipynb",
    "03_adaptive_topology_v0.ipynb",
    "04_importance.ipynb",
    "05_adaptive.ipynb",
    "06_search.ipynb",
    "07_extract.ipynb",
    "08_retrain.ipynb",
    "09_compare.ipynb",
)


def _cells(name: str) -> list[dict]:
    return build_notebooks.NOTEBOOKS[name]()["cells"]


def test_builder_registry_covers_the_plan_notebooks() -> None:
    assert tuple(build_notebooks.NOTEBOOKS) == NOTEBOOK_NAMES


def test_rendered_notebooks_are_valid_nbformat(tmp_path: Path) -> None:
    for name, payload in build_notebooks.build_notebooks(tmp_path).items():
        notebook = json.loads(payload.decode("utf-8"))
        assert notebook["nbformat"] == build_notebooks.NBFORMAT
        assert notebook["cells"], name
        for cell in notebook["cells"]:
            assert cell["cell_type"] in {"code", "markdown"}
            if cell["cell_type"] == "code":
                # nbformat 4.x: no execution count and no outputs on a
                # never-executed notebook.
                assert cell["execution_count"] is None
                assert cell["outputs"] == []


def test_builder_output_is_byte_identical_across_runs(tmp_path: Path) -> None:
    first = build_notebooks.build_notebooks(tmp_path)
    second = build_notebooks.build_notebooks(tmp_path)
    assert first == second


def test_written_notebooks_round_trip(tmp_path: Path) -> None:
    build_notebooks.write_notebooks(tmp_path)
    assert build_notebooks.notebooks_match(tmp_path)


def test_written_notebooks_detect_drift(tmp_path: Path) -> None:
    build_notebooks.write_notebooks(tmp_path)
    victim = tmp_path / "09_compare.ipynb"
    payload = json.loads(victim.read_text())
    payload["cells"][0]["source"] = ["tampered\n"]
    victim.write_text(json.dumps(payload))
    assert not build_notebooks.notebooks_match(tmp_path)


def test_committed_notebooks_match_builder() -> None:
    # The versioned .ipynb files must never drift from the builder output.
    assert build_notebooks.notebooks_match(build_notebooks.NOTEBOOK_DIR), (
        "notebooks/ drifted from scripts/build_notebooks.py; "
        "re-run `PYTHONPATH=src python scripts/build_notebooks.py`"
    )


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_notebook_opens_with_the_session_contract(name: str) -> None:
    cells = _cells(name)
    assert cells[0]["cell_type"] == "markdown"
    header = "".join(cells[0]["source"])
    assert "plan section 40" in header
    # Second cell: the bootstrap that inserts src/ onto sys.path.
    assert cells[1]["cell_type"] == "code"
    bootstrap = "".join(cells[1]["source"])
    assert "sys.path.insert" in bootstrap
    assert "src" in bootstrap


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_every_code_cell_compiles(name: str) -> None:
    for index, cell in enumerate(_cells(name)):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(source, f"{name}[{index}]", "exec")


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_notebook_ends_with_a_summary_record(name: str) -> None:
    """The section 40 end-of-session summary must be present and keyed."""
    last = _cells(name)[-1]
    assert last["cell_type"] == "code"
    source = "".join(last["source"])
    assert build_notebooks.RECORD_DIR in source
    assert "payload =" in source and "json.dumps(payload, sort_keys=True" in source
    assert f"'{name.removesuffix('.ipynb')}'" in source  # keyed by notebook name


def test_environment_notebook_pins_declared_parameter_counts() -> None:
    source = "\n".join(
        "".join(cell["source"])
        for cell in _cells("00_environment.ipynb")
        if cell["cell_type"] == "code"
    )
    assert "285_336_576" in source
    assert "134_341_632" in source
    assert "decide_dtype" in source


def test_dense_notebook_proves_all_three_section_104_claims() -> None:
    source = "\n".join(
        "".join(cell["source"])
        for cell in _cells("01_dense_baseline.ipynb")
        if cell["cell_type"] == "code"
    )
    # trains, resumes, reproduces
    assert "train_steps(" in source
    assert "save_checkpoint(" in source and "load_checkpoint(" in source
    assert "fresh_losses == losses" in source


def test_churn_notebook_uses_real_topology_operators() -> None:
    source = "\n".join(
        "".join(cell["source"])
        for cell in _cells("02_random_topology_churn.ipynb")
        if cell["cell_type"] == "code"
    )
    assert "grow_expert(" in source and "prune_expert(" in source
    assert "genome_hash(" in source
    assert "rebuild_model_from_genome(" in source
    assert "append_result(" in source


def test_adaptive_notebook_runs_a_gate_study_and_records_the_verdict() -> None:
    source = "\n".join(
        "".join(cell["source"])
        for cell in _cells("03_adaptive_topology_v0.ipynb")
        if cell["cell_type"] == "code"
    )
    assert "run_gate_study(" in source
    assert "append_study(" in source
    assert "gate_passed" in source


def test_importance_notebook_extracts_real_signals_and_fossils() -> None:
    source = "\n".join(
        "".join(cell["source"])
        for cell in _cells("04_importance.ipynb")
        if cell["cell_type"] == "code"
    )
    assert "measure_expert_signals(" in source
    assert "expert_ablation_delta(" in source
    assert "LifecycleLedger" in source
    assert "loss.backward()" in source  # signals require a real backward pass


def test_adaptive_controller_notebook_reports_comparables() -> None:
    source = "\n".join(
        "".join(cell["source"])
        for cell in _cells("05_adaptive.ipynb")
        if cell["cell_type"] == "code"
    )
    assert "efficiency_report(" in source and "ablation_report(" in source
    assert "'fixed,random,adaptive'" in source  # all three arms, matched budgets
    assert "mutation_history" in source  # decisions are inspectable


def test_search_notebook_covers_the_full_search_stack() -> None:
    source = "\n".join(
        "".join(cell["source"])
        for cell in _cells("06_search.ipynb")
        if cell["cell_type"] == "code"
    )
    assert "enumerate_module_depth_variants(" in source
    assert "rank_module_candidates(" in source
    assert "run_frontier(" in source
    assert "pareto_frontier(" in source and "select_population(" in source


def test_extract_notebook_freezes_and_publishes() -> None:
    source = "\n".join(
        "".join(cell["source"])
        for cell in _cells("07_extract.ipynb")
        if cell["cell_type"] == "code"
    )
    assert "save_checkpoint(" in source
    assert "genome.to_dict()" in source and "ArchitectureGenome.from_dict(" in source
    assert "publish_artifacts(" in source and "restore_artifacts(" in source


def test_retrain_notebook_compares_all_init_policies() -> None:
    source = "\n".join(
        "".join(cell["source"])
        for cell in _cells("08_retrain.ipynb")
        if cell["cell_type"] == "code"
    )
    for mode in ("fresh", "rewound", "teacher"):
        assert f"'{mode}'" in source
    assert "copied_parameters" in source and "skipped_parameters" in source


def test_compare_notebook_reports_stability_and_equivalence() -> None:
    source = "\n".join(
        "".join(cell["source"])
        for cell in _cells("09_compare.ipynb")
        if cell["cell_type"] == "code"
    )
    assert "summarize_stability(" in source
    assert "architecture_equivalence(" in source
    assert "gate_p_value" in source
