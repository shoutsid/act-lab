"""Deterministic Kaggle artifact layout and manifest contracts."""

from dataclasses import asdict, dataclass
import json
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


def kaggle_layout(root: str | Path = "artifacts/kaggle") -> KaggleLayout:
    path = Path(root)
    if path == Path("."):
        raise ValueError("Kaggle artifact root must not be the current directory")
    return KaggleLayout(path)
