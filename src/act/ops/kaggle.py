"""Deterministic Kaggle artifact layout and manifest contracts."""

from dataclasses import asdict, dataclass
import hashlib
import json
import shutil
from pathlib import Path


@dataclass(frozen=True)
class KaggleLayout:
    root: Path

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def evaluations(self) -> Path:
        return self.root / "evaluations"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    def required_directories(self) -> tuple[Path, ...]:
        return (self.checkpoints, self.evaluations, self.reports, self.manifests)


@dataclass(frozen=True)
class KaggleManifest:
    dataset_id: str
    code_revision: str
    config_hash: str
    artifacts: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2) + "\n"


def publish_artifacts(
    layout: KaggleLayout,
    sources: tuple[str | Path, ...],
    *,
    dataset_id: str,
    code_revision: str,
    config_hash: str,
) -> KaggleManifest:
    """Copy named artifacts into the layout and write a checksum manifest."""
    for directory in layout.required_directories():
        directory.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    checksums: dict[str, str] = {}
    for source_value in sources:
        source = Path(source_value)
        if not source.is_file():
            raise FileNotFoundError(source)
        target = layout.reports / source.name
        shutil.copy2(source, target)
        relative = str(target.relative_to(layout.root))
        copied.append(relative)
        checksums[relative] = _sha256(target)
    manifest = KaggleManifest(dataset_id, code_revision, config_hash, tuple(sorted(copied)))
    payload = {**asdict(manifest), "checksums": checksums}
    (layout.manifests / "manifest.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def restore_artifacts(layout: KaggleLayout) -> tuple[Path, ...]:
    """Verify and return artifacts listed by the local manifest."""
    manifest_path = layout.manifests / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    restored: list[Path] = []
    for relative, expected in payload.get("checksums", {}).items():
        path = layout.root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {relative}")
        restored.append(path)
    return tuple(sorted(restored))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kaggle_layout(root: str | Path = "artifacts/kaggle") -> KaggleLayout:
    path = Path(root)
    if path == Path("."):
        raise ValueError("Kaggle artifact root must not be the current directory")
    return KaggleLayout(path)
