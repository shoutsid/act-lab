from pathlib import Path

import pytest

from act.ops import (
    KaggleManifest,
    kaggle_layout,
    publish_artifacts,
    restore_artifacts,
)


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


def test_publish_and_restore_verify_checksums(tmp_path: Path) -> None:
    source = tmp_path / "result.json"
    source.write_text("{}", encoding="utf-8")
    layout = kaggle_layout(tmp_path / "artifact")
    manifest = publish_artifacts(
        layout,
        (source,),
        dataset_id="synthetic-v1",
        code_revision="abc",
        config_hash="def",
    )
    assert manifest.artifacts == ("reports/result.json",)
    assert restore_artifacts(layout) == (layout.reports / "result.json",)
    (layout.reports / "result.json").write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        restore_artifacts(layout)
