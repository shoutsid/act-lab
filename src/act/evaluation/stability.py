"""Cross-seed stability summaries for comparable experiment measurements."""

from dataclasses import dataclass
from statistics import mean, pstdev


@dataclass(frozen=True)
class StabilitySummary:
    values: tuple[float, ...]
    mean: float
    stddev: float
    minimum: float
    maximum: float


def summarize_stability(values: tuple[float, ...] | list[float]) -> StabilitySummary:
    samples = tuple(float(value) for value in values)
    if not samples:
        raise ValueError("stability requires at least one value")
    return StabilitySummary(
        samples,
        mean(samples),
        pstdev(samples) if len(samples) > 1 else 0.0,
        min(samples),
        max(samples),
    )


def architecture_equivalence(hashes: tuple[str, ...] | list[str]) -> bool:
    values = tuple(hashes)
    if not values:
        raise ValueError("architecture hashes must not be empty")
    return len(set(values)) == 1
