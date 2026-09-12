"""Bounded checkpoint-session policies for long-running experiments."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SessionCheckpointPolicy:
    interval: int = 100
    max_checkpoints: int = 3
    max_bytes: int = 2**30

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError("checkpoint interval must be positive")
        if self.max_checkpoints < 1:
            raise ValueError("max_checkpoints must be positive")
        if self.max_bytes < 1:
            raise ValueError("max_bytes must be positive")

    def should_save(self, step: int) -> bool:
        if step < 0:
            raise ValueError("step must be non-negative")
        return step > 0 and step % self.interval == 0

    def accept(self, path: str | Path) -> None:
        size = Path(path).stat().st_size
        if size > self.max_bytes:
            raise ValueError(
                f"checkpoint {path} exceeds max_bytes ({size} > {self.max_bytes})"
            )


def retain_checkpoints(
    directory: str | Path,
    *,
    pattern: str = "checkpoint-*.pt",
    max_checkpoints: int = 3,
) -> tuple[Path, ...]:
    """Keep newest files by modification time and return retained paths."""
    if max_checkpoints < 1:
        raise ValueError("max_checkpoints must be positive")
    paths = sorted(
        Path(directory).glob(pattern),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    retained = tuple(paths[:max_checkpoints])
    retained_set = set(retained)
    for path in paths[max_checkpoints:]:
        if path not in retained_set:
            path.unlink()
    return retained
