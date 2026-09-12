"""Reproducible multi-seed studies for the low-fidelity ACT gate."""

from dataclasses import dataclass, replace
import math
from statistics import NormalDist

from .experiment import ExperimentConfig, ExperimentResult, run_experiment


@dataclass(frozen=True)
class GateStudyResult:
    config: dict[str, object]
    seeds: tuple[int, ...]
    experiments: tuple[ExperimentResult, ...]
    adaptive_wins: int
    mean_adaptive_delta: float
    delta_stddev: float
    matched_mutation_budget: bool
    gate_alpha: float = 0.05
    treatment: str = "adaptive"

    @property
    def search_flops_by_arm(self) -> dict[str, int]:
        """Total search FLOPs per arm so selection cost stays comparable."""
        totals: dict[str, int] = {}
        for experiment in self.experiments:
            for arm in experiment.arms:
                totals[arm.mode] = totals.get(arm.mode, 0) + arm.search_flops
        return dict(sorted(totals.items()))

    @property
    def treatment_search_overhead_ratio(self) -> float:
        """Treatment search FLOPs divided by random-arm search FLOPs."""
        totals = self.search_flops_by_arm
        random_flops = totals.get("random", 0)
        if not random_flops:
            return 0.0
        return totals.get(self.treatment, 0) / random_flops

    @property
    def win_rate(self) -> float:
        return self.adaptive_wins / len(self.seeds) if self.seeds else 0.0

    @property
    def gate_p_value(self) -> float:
        """Exact one-tailed binomial p-value for the observed adaptive wins."""
        return _binomial_survival(self.adaptive_wins, len(self.seeds))

    @property
    def gate_min_wins(self) -> int:
        """Smallest adaptive win count that passes the gate at ``gate_alpha``."""
        return _gate_min_wins(len(self.seeds), self.gate_alpha)

    @property
    def gate_passed(self) -> bool:
        return self.adaptive_wins >= self.gate_min_wins

    @property
    def t_statistic(self) -> float:
        """One-sample t statistic for the paired adaptive-minus-random deltas."""
        if len(self.seeds) < 2:
            return 0.0
        sample_stddev = self.delta_stddev * math.sqrt(len(self.seeds) / (len(self.seeds) - 1))
        if sample_stddev == 0.0:
            return 0.0
        return self.mean_adaptive_delta / (sample_stddev / math.sqrt(len(self.seeds)))

    @property
    def t_p_value_one_tailed(self) -> float:
        """Normal-approximation one-tailed p-value that the treatment helps.

        The paired delta is treatment-minus-random loss, so "treatment is
        better" means a negative mean delta and the one-tailed p-value is
        P(T <= t). The binomial sign test above is the pre-registered primary
        verdict; this continuous-outcome approximation is reported as secondary
        context and is reliable only for larger studies (roughly N >= 30 seeds).
        """
        if len(self.seeds) < 2 or self.delta_stddev == 0.0:
            return 0.5
        return NormalDist().cdf(self.t_statistic)

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config,
            "seeds": list(self.seeds),
            "adaptive_wins": self.adaptive_wins,
            "win_rate": self.win_rate,
            "mean_adaptive_delta": self.mean_adaptive_delta,
            "delta_stddev": self.delta_stddev,
            "matched_mutation_budget": self.matched_mutation_budget,
            "treatment": self.treatment,
            "search_flops_by_arm": self.search_flops_by_arm,
            "treatment_search_overhead_ratio": self.treatment_search_overhead_ratio,
            "gate_alpha": self.gate_alpha,
            "gate_p_value": self.gate_p_value,
            "gate_min_wins": self.gate_min_wins,
            "gate_passed": self.gate_passed,
            "t_statistic": self.t_statistic,
            "t_p_value_one_tailed": self.t_p_value_one_tailed,
            "experiments": [experiment.to_dict() for experiment in self.experiments],
        }


def run_gate_study(
    config: ExperimentConfig = ExperimentConfig(),
    seeds: tuple[int, ...] = (1, 2, 3, 42),
    treatment: str = "adaptive",
) -> GateStudyResult:
    """Run a paired multi-seed study of a treatment arm versus random churn.

    ``treatment`` selects the arm scored against ``random``: the budget-aware
    ``adaptive`` controller (default, the Gate 3 comparison) or the probe-loss
    greedy ``guided`` diagnostic upper bound.
    """
    if not seeds:
        raise ValueError("study requires at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("study seeds must be unique")
    if treatment not in {"adaptive", "guided"}:
        raise ValueError("unsupported treatment arm")
    experiments = tuple(
        run_experiment(replace(config, seed=seed))
        for seed in seeds
    )
    deltas = tuple(_treatment_delta(experiment, treatment) for experiment in experiments)
    matched_budget = all(
        _has_matched_budget(experiment, treatment) for experiment in experiments
    )
    if not matched_budget:
        raise RuntimeError(f"{treatment} and random arms used different mutation budgets")
    return GateStudyResult(
        config={key: value for key, value in config.__dict__.items() if key != "seed"},
        seeds=seeds,
        experiments=experiments,
        adaptive_wins=sum(delta < 0.0 for delta in deltas),
        mean_adaptive_delta=sum(deltas) / len(deltas) if deltas else 0.0,
        delta_stddev=_population_stddev(deltas),
        matched_mutation_budget=matched_budget,
        treatment=treatment,
    )


def _binomial_survival(wins: int, trials: int) -> float:
    """Exact P(X >= wins) for X ~ Binomial(trials, 0.5)."""
    if trials < 0:
        raise ValueError("trials must be non-negative")
    wins = max(0, min(int(wins), trials))
    favorable = sum(math.comb(trials, k) for k in range(wins, trials + 1))
    return favorable / (2**trials)


def _gate_min_wins(trials: int, alpha: float) -> int:
    """Smallest win count whose one-tailed binomial p-value is <= alpha.

    Returns ``trials + 1`` when no achievable win count can pass, which keeps
    ``gate_passed`` honestly false for underpowered study sizes.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    required = trials + 1
    for wins in range(trials, -1, -1):
        if _binomial_survival(wins, trials) <= alpha:
            required = wins
        else:
            break
    return required


def _treatment_delta(result: ExperimentResult, treatment: str) -> float:
    arms = {arm.mode: arm for arm in result.arms}
    return arms[treatment].final_loss - arms["random"].final_loss


def _has_matched_budget(result: ExperimentResult, treatment: str) -> bool:
    arms = {arm.mode: arm for arm in result.arms}
    return arms["random"].mutation_count == arms[treatment].mutation_count


def _population_stddev(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5