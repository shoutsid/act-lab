import math

import pytest

from act.evolution import ExperimentConfig, run_frontier


def test_frontier_runner_reports_matched_families() -> None:
    points = run_frontier(
        ExperimentConfig(steps=2, validation_batches=1),
        seeds=(2, 3),
    )
    assert len(points) == 4
    assert {point.family for point in points} == {"static-moe-control", "adaptive-moe"}
    assert all(math.isfinite(point.validation_loss) for point in points)


def test_frontier_requests_its_reported_arms_independent_of_config() -> None:
    points = run_frontier(
        ExperimentConfig(steps=2, validation_batches=1, arms="random"),
        seeds=(5,),
    )
    assert {point.family for point in points} == {"static-moe-control", "adaptive-moe"}


def test_frontier_requires_seed() -> None:
    with pytest.raises(ValueError, match="at least one"):
        run_frontier(ExperimentConfig(steps=1), seeds=())
