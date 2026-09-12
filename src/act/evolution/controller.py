"""Deterministic, budget-aware topology controller policies."""

from dataclasses import dataclass
from typing import Mapping, Sequence

from act.topology import ArchitectureGenome


@dataclass(frozen=True)
class ControllerDecision:
    operation: str
    layer_id: str
    expert_id: str | None = None
    reason: str = ""


class AdaptiveController:
    """Select a topology action from the latest measurable expert signals."""

    def __init__(
        self,
        *,
        load_threshold: float = 0.5,
        metric: str = "routing_load",
        max_experts: int = 8,
    ) -> None:
        if not 0.0 < load_threshold < 1.0:
            raise ValueError("load_threshold must be between 0 and 1")
        if metric not in {
            "routing_load",
            "gradient_norm",
            "fisher_ema",
            "parameter_norm",
            "temporal_stability",
        }:
            raise ValueError("unsupported controller metric")
        if max_experts < 2:
            raise ValueError("max_experts must be at least 2")
        self.load_threshold = load_threshold
        self.metric = metric
        self.max_experts = max_experts

    def decide(
        self,
        genome: ArchitectureGenome,
        *,
        loads: Sequence[Sequence[float]],
        latest: Mapping[str, Mapping[str, object]],
    ) -> ControllerDecision:
        if len(loads) != len(genome.layers):
            raise ValueError("loads must contain one row per genome layer")
        busiest = max(range(len(genome.layers)), key=lambda index: max(loads[index]))
        layer = genome.layers[busiest]
        if max(loads[busiest]) > self.load_threshold and len(layer.experts) < self.max_experts:
            return ControllerDecision(
                "grow",
                layer.id,
                reason="routing load exceeds threshold",
            )
        candidates = [
            (float(latest.get(f"{item.id}:{expert.id}", {}).get(self.metric, 0.0)), item.id, expert.id)
            for item in genome.layers
            for expert in item.experts
            if len(item.experts) > 2
        ]
        if not candidates:
            return ControllerDecision("grow", layer.id, reason="no safe prune candidate")
        _, layer_id, expert_id = min(candidates, key=lambda value: (value[0], value[1], value[2]))
        return ControllerDecision(
            "prune",
            layer_id,
            expert_id,
            reason=f"lowest {self.metric}",
        )
