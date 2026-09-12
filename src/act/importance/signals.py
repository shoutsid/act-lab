"""Device-safe, deterministic importance signals for static MoE experts."""

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor

from act.model.moe import StaticMoECausalLM


@dataclass(frozen=True)
class ExpertSignal:
    layer_id: str
    expert_id: str
    routing_load: float
    gradient_norm: float
    parameter_norm: float
    temporal_stability: float
    functional_similarity: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def cosine_similarity(left: Tensor, right: Tensor) -> float:
    """Return a bounded cosine similarity for flattened expert representations."""
    if left.shape != right.shape:
        raise ValueError("similarity inputs must have identical shapes")
    left_flat = left.detach().reshape(-1).float()
    right_flat = right.detach().reshape(-1).float()
    denominator = left_flat.norm() * right_flat.norm()
    if float(denominator) == 0.0:
        return 0.0
    return float(torch.clamp(torch.dot(left_flat, right_flat) / denominator, -1.0, 1.0))


def measure_expert_signals(
    model: StaticMoECausalLM,
    *,
    previous_loads: dict[str, float] | None = None,
) -> tuple[ExpertSignal, ...]:
    """Extract routing, gradient, parameter, and stability signals after a step."""
    previous = previous_loads or {}
    records: list[ExpertSignal] = []
    for layer_index, block in enumerate(model.blocks):
        loads = model.routing_load[layer_index]
        for expert_index, expert_id in enumerate(block.expert_ids):
            expert = block.experts[expert_index]
            gradient_squared = 0.0
            parameter_squared = 0.0
            for parameter in expert.parameters():
                parameter_squared += float(parameter.detach().pow(2).sum())
                if parameter.grad is not None:
                    gradient_squared += float(parameter.grad.detach().pow(2).sum())
            key = f"L{layer_index}:{expert_id}"
            routing_load = float(loads[expert_index])
            stability = 1.0 - min(abs(routing_load - previous.get(key, routing_load)), 1.0)
            records.append(
                ExpertSignal(
                    layer_id=f"L{layer_index}",
                    expert_id=expert_id,
                    routing_load=routing_load,
                    gradient_norm=math.sqrt(gradient_squared),
                    parameter_norm=math.sqrt(parameter_squared),
                    temporal_stability=stability,
                )
            )
    return tuple(records)
