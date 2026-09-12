from pathlib import Path

import pytest

from act.ops import KaggleManifest, kaggle_layout


def test_kaggle_layout_has_stable_artifact_directories() -> None:
    layout = kaggle_layout("/tmp/act-kaggle")
    assert layout.required_directories() == (
        Path("/tmp/act-kaggle/checkpoints"),
        Path("/tmp/act-kaggle/evaluations"),
        Path("/tmp/act-kaggle/reports"),
        Path("/tmp/act-kaggle/manifests"),
    )


def test_kaggle_manifest_is_sorted_json() -> None:
    manifest = KaggleManifest("synthetic-v1", "abc", "def", ("report.json",))
    assert '"dataset_id": "synthetic-v1"' in manifest.to_json()
    with pytest.raises(ValueError, match="current"):
        kaggle_layout(".")
