"""Small, deterministic summaries for first-gate experiment results."""

from statistics import mean, pstdev

from act.evolution.experiment import ExperimentResult
from act.evolution.study import GateStudyResult
from .perplexity import PerplexityResult, evaluate_perplexity
from .probes import CapabilityProbeResult, evaluate_capability_probes
from .reports import ablation_report, efficiency_report
from .stability import StabilitySummary, architecture_equivalence, summarize_stability


def summarize_experiment(result: ExperimentResult) -> dict[str, object]:
    arms = {arm.mode: arm for arm in result.arms}
    return {
        "config": result.config,
        "arms": {
            mode: {
                "validation_loss": arm.validation_loss,
                "parameter_count": arm.parameter_count,
                "active_parameter_count": arm.active_parameter_count,
                "mutation_count": arm.mutation_count,
                "training_flops": arm.training_flops,
                "validation_flops": arm.validation_flops,
                "search_flops": arm.search_flops,
                "architecture_distance": arm.architecture_distance,
                "architecture_hash": arm.architecture_hash,
            }
            for mode, arm in arms.items()
        },
        "adaptive_minus_random": (
            arms["adaptive"].validation_loss - arms["random"].validation_loss
        ),
    }


def summarize_study(study: GateStudyResult) -> dict[str, object]:
    deltas = []
    for experiment in study.experiments:
        arms = {arm.mode: arm for arm in experiment.arms}
        deltas.append(arms["adaptive"].validation_loss - arms["random"].validation_loss)
    return {
        "seeds": list(study.seeds),
        "adaptive_wins": study.adaptive_wins,
        "win_rate": study.win_rate,
        "mean_adaptive_delta": mean(deltas) if deltas else 0.0,
        "delta_population_stddev": pstdev(deltas) if len(deltas) > 1 else 0.0,
        "delta_stddev": study.delta_stddev,
        "matched_mutation_budget": study.matched_mutation_budget,
        "experiments": [summarize_experiment(experiment) for experiment in study.experiments],
    }


__all__ = [
    "PerplexityResult",
    "CapabilityProbeResult",
    "evaluate_capability_probes",
    "ablation_report",
    "efficiency_report",
    "StabilitySummary",
    "architecture_equivalence",
    "summarize_stability",
    "evaluate_perplexity",
    "summarize_experiment",
    "summarize_study",
]
