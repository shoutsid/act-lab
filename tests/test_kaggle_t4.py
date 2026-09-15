"""Tests for the Kaggle T4 driver (``scripts/kaggle_t4.py``).

The driver is what actually gets pushed for the section 5.1 run, so the two
things worth pinning are (a) that the mount-layout search survives every
Kaggle attachment layout - the first run died on the assumption that it did
not - and (b) that each queue's launch command still says exactly what the
runbook says, since a drifted flag is a protocol change, not a typo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import kaggle_t4  # noqa: E402


@pytest.fixture(autouse=True)
def _default_mount_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assertions below quote the default owner/slug, so pin them."""
    monkeypatch.delenv(kaggle_t4.OWNER_ENV, raising=False)
    monkeypatch.delenv(kaggle_t4.SLUG_ENV, raising=False)


def _make_repo(root: Path, relative: str) -> Path:
    code = root / relative
    (code / "src" / "act").mkdir(parents=True)
    (code / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return code


@pytest.mark.parametrize(
    "layout",
    [
        "act-lab-code",
        "sid123123234/act-lab-code",
        "datasets/sid123123234/act-lab-code",
    ],
)
def test_find_code_root_handles_every_attachment_layout(
    tmp_path: Path, layout: str
) -> None:
    root = tmp_path / "input"
    code = _make_repo(root, layout)
    assert kaggle_t4.find_code_root(root) == code


def test_find_code_root_ignores_a_decoy_and_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "input"
    (root / "unrelated-dataset").mkdir(parents=True)
    (root / "unrelated-dataset" / "pyproject.toml").write_text("[project]\n")
    assert kaggle_t4.find_code_root(root) is None
    assert kaggle_t4.find_code_root(tmp_path / "does-not-exist") is None


def test_primary_launch_command_is_the_registered_one() -> None:
    queue = kaggle_t4.QUEUES["session"]
    assert kaggle_t4.session_argv(queue, 9.0) == [
        sys.executable,
        "-m",
        "act.training.cli",
        "--mode",
        "pair-session",
        "--device",
        "cuda",
        "--hours",
        "9",
        "--seeds",
        "15000-15011",
        "--probe-candidate-policy",
        "spread",
        "--dataset-root",
        "artifacts/datasets",
        "--session-dir",
        "artifacts/v0-sessions",
        "--kaggle-dataset",
        "sid123123234/act-v0-5-1",
    ]


def test_control_queue_isolates_the_decision_rule() -> None:
    argv = kaggle_t4.session_argv(kaggle_t4.QUEUES["control"])
    assert argv[-1] == "sid123123234/act-v0-5-1-control"
    # Same candidate pool as the primary, so the only difference is the rule.
    assert argv[argv.index("--probe-candidate-policy") + 1] == "spread"
    assert argv[argv.index("--probe-selection") + 1] == "random"
    assert argv[argv.index("--dataset-root") + 1] == "artifacts/datasets-control"
    assert argv[argv.index("--session-dir") + 1] == "artifacts/v0-sessions-control"
    assert "--session-arms" not in argv


def test_null_queue_is_policy_free_and_never_shares_roots() -> None:
    argv = kaggle_t4.session_argv(kaggle_t4.QUEUES["null"])
    assert argv[argv.index("--session-arms") + 1] == "null"
    assert "--probe-candidate-policy" not in argv
    assert "--probe-selection" not in argv
    assert argv[argv.index("--dataset-root") + 1] == "artifacts/datasets-null"
    assert argv[-1] == "sid123123234/act-v0-5-1-null"


def test_queues_own_disjoint_datasets_and_roots() -> None:
    queues = list(kaggle_t4.QUEUES.values())
    assert len({q.dataset for q in queues}) == len(queues)
    assert len({q.dataset_root for q in queues}) == len(queues)
    assert len({q.session_dir for q in queues}) == len(queues)


def test_kernel_metadata_pins_the_t4_and_keeps_internet_on() -> None:
    metadata = kaggle_t4.kernel_metadata(kaggle_t4.QUEUES["session"])
    # ``gpu_type`` is silently ignored by the CLI; only machine_shape counts.
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert "gpu_type" not in metadata
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is True
    assert metadata["kernel_type"] == "script"
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == ["sid123123234/act-lab-code"]


def test_benchmark_argv_is_the_registered_step_count() -> None:
    assert kaggle_t4.benchmark_argv("python") == [
        "python",
        "scripts/v0_throughput_benchmark.py",
        "--steps",
        "50",
    ]


def test_unknown_mode_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(kaggle_t4.MODE_ENV, "not-a-mode")
    with pytest.raises(SystemExit):
        kaggle_t4.main()
