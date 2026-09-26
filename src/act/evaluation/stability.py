"""Cross-seed stability summaries for comparable experiment measurements."""

from dataclasses import dataclass
import math
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
    if any(not math.isfinite(value) for value in samples):
        raise ValueError("stability values must be finite")
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
    if any(not value for value in values):
        raise ValueError("architecture hashes must not contain empty values")
    return len(set(values)) == 1
