"""Aggregate compute-efficiency measurements across protocol sessions."""

from collections.abc import Iterable

from .v0_session import V0SessionResult


def aggregate_arm_cost_ratio(
    results: Iterable[V0SessionResult],
    arm_mode: str = "adaptive",
) -> float:
    """Aggregate search/training FLOPs ratio for one arm across sessions.

    This is the pre-registered primary metric: aggregate per-arm search FLOPs
    over aggregate per-arm training FLOPs (the same accounting that produced
    every toy cost figure), summed over all measured seeds.
    """
    search = 0
    training = 0
    for result in results:
        for arm in result.arms:
            if arm.mode == arm_mode:
                search += arm.search_flops
                training += arm.training_flops
    if training == 0:
        raise ValueError(f"no training FLOPs recorded for arm {arm_mode}")
    return search / training


def sign_pairs(
    results: Iterable[V0SessionResult],
    treatment: str = "adaptive",
    baseline: str = "random",
) -> tuple[tuple[float, float], ...]:
    """Return paired ``(treatment, baseline)`` final losses per session seed.

    Used by the pre-registered harm guard: adaptive worse than random in
    >= 10 of 12 pairs withholds the Gate 4 recommendation.
    """
    pairs: list[tuple[float, float]] = []
    for result in results:
        arms = {arm.mode: arm.final_loss for arm in result.arms}
        if treatment not in arms or baseline not in arms:
            raise ValueError("session result is missing an arm for pairing")
        pairs.append((arms[treatment], arms[baseline]))
    return tuple(pairs)


def harm_guard_rejects(pairs: tuple[tuple[float, float], ...]) -> bool:
    """True when the treatment loses in >= 10 of 12 pairs (p = 0.0193)."""
    losses = sum(treatment > baseline for treatment, baseline in pairs)
    return losses >= 10
