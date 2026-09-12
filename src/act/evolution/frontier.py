"""Small matched-budget frontier experiments for static and adaptive MoE."""

from dataclasses import asdict, dataclass

from .experiment import ExperimentConfig, run_experiment


@dataclass(frozen=True)
class FrontierPoint:
    family: str
    seed: int
    validation_loss: float
    parameter_count: int
    active_parameter_count: int
    validation_flops: int


def run_frontier(
    config: ExperimentConfig,
    *,
    seeds: tuple[int, ...] = (1,),
) -> tuple[FrontierPoint, ...]:
    if not seeds:
        raise ValueError("frontier requires at least one seed")
    points: list[FrontierPoint] = []
    for seed in seeds:
        result = run_experiment(
            ExperimentConfig(**{**asdict(config), "seed": seed})
        )
        arms = {arm.mode: arm for arm in result.arms}
        for family, mode in (("static-moe-control", "fixed"), ("adaptive-moe", "adaptive")):
            arm = arms[mode]
            points.append(
                FrontierPoint(
                    family,
                    seed,
                    arm.validation_loss,
                    arm.parameter_count,
                    arm.active_parameter_count,
                    arm.validation_flops,
                )
            )
    return tuple(points)
