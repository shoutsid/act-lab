"""Reproducible multi-seed studies for the low-fidelity ACT gate."""

from dataclasses import dataclass, replace

from .experiment import ExperimentConfig, ExperimentResult, run_experiment


@dataclass(frozen=True)
class GateStudyResult:
    config: dict[str, object]
    seeds: tuple[int, ...]
    experiments: tuple[ExperimentResult, ...]
    adaptive_wins: int
    mean_adaptive_delta: float
    matched_mutation_budget: bool

    @property
    def win_rate(self) -> float:
        return self.adaptive_wins / len(self.seeds) if self.seeds else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config,
            "seeds": list(self.seeds),
            "adaptive_wins": self.adaptive_wins,
            "win_rate": self.win_rate,
            "mean_adaptive_delta": self.mean_adaptive_delta,
            "matched_mutation_budget": self.matched_mutation_budget,
            "experiments": [experiment.to_dict() for experiment in self.experiments],
        }


def run_gate_study(
    config: ExperimentConfig = ExperimentConfig(),
    seeds: tuple[int, ...] = (1, 2, 3, 42),
) -> GateStudyResult:
    if len(set(seeds)) != len(seeds):
        raise ValueError("study seeds must be unique")
    experiments = tuple(
        run_experiment(replace(config, seed=seed))
        for seed in seeds
    )
    deltas = tuple(_adaptive_delta(experiment) for experiment in experiments)
    matched_budget = all(_has_matched_budget(experiment) for experiment in experiments)
    if not matched_budget:
        raise RuntimeError("adaptive and random arms used different mutation budgets")
    return GateStudyResult(
        config={key: value for key, value in config.__dict__.items() if key != "seed"},
        seeds=seeds,
        experiments=experiments,
        adaptive_wins=sum(delta < 0.0 for delta in deltas),
        mean_adaptive_delta=sum(deltas) / len(deltas) if deltas else 0.0,
        matched_mutation_budget=matched_budget,
    )


def _adaptive_delta(result: ExperimentResult) -> float:
    arms = {arm.mode: arm for arm in result.arms}
    return arms["adaptive"].final_loss - arms["random"].final_loss


def _has_matched_budget(result: ExperimentResult) -> bool:
    arms = {arm.mode: arm for arm in result.arms}
    return arms["random"].mutation_count == arms["adaptive"].mutation_count