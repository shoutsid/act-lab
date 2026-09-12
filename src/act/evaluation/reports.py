"""Comparable ablation and efficiency reports for experiment results."""

from act.evolution.experiment import ExperimentResult


def efficiency_report(result: ExperimentResult) -> dict[str, object]:
    arms = {arm.mode: arm for arm in result.arms}
    rows = []
    for arm in sorted(result.arms, key=lambda item: item.mode):
        total_flops = arm.training_flops + arm.validation_flops + arm.search_flops
        rows.append(
            {
                "mode": arm.mode,
                "validation_loss": arm.validation_loss,
                "quality_per_million_flops": (
                    arm.quality / (total_flops / 1_000_000)
                    if total_flops
                    else 0.0
                ),
                "total_flops": total_flops,
                "parameter_count": arm.parameter_count,
                "active_parameter_count": arm.active_parameter_count,
                "mutation_count": arm.mutation_count,
            }
        )
    return {
        "rows": rows,
        "adaptive_minus_random": (
            arms["adaptive"].validation_loss - arms["random"].validation_loss
        ),
    }


def ablation_report(
    result: ExperimentResult,
    *,
    reference_mode: str = "fixed",
) -> dict[str, object]:
    arms = {arm.mode: arm for arm in result.arms}
    if reference_mode not in arms:
        raise ValueError(f"unknown reference mode: {reference_mode}")
    reference = arms[reference_mode]
    return {
        "reference_mode": reference_mode,
        "reference_validation_loss": reference.validation_loss,
        "comparisons": [
            {
                "mode": arm.mode,
                "validation_loss_delta": arm.validation_loss - reference.validation_loss,
                "parameter_delta": arm.parameter_count - reference.parameter_count,
                "search_flops": arm.search_flops,
            }
            for arm in sorted(result.arms, key=lambda item: item.mode)
            if arm.mode != reference_mode
        ],
    }
