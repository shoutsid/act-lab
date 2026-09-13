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


def aggregate_expert_signals(
    populations: list[tuple[ExpertSignal, ...]] | tuple[tuple[ExpertSignal, ...], ...],
) -> tuple[ExpertSignal, ...]:
    """Average signals for matching experts across repeated evaluations."""
    if not populations:
        return ()
    grouped: dict[tuple[str, str], list[ExpertSignal]] = {}
    for population in populations:
        for signal in population:
            grouped.setdefault((signal.layer_id, signal.expert_id), []).append(signal)
    aggregated: list[ExpertSignal] = []
    for layer_id, expert_id in sorted(grouped):
        signals = grouped[(layer_id, expert_id)]
        average = lambda value: sum(value(signal) for signal in signals) / len(signals)
        similarity_values = [
            signal.functional_similarity
            for signal in signals
            if signal.functional_similarity is not None
        ]
        aggregated.append(
            ExpertSignal(
                layer_id=layer_id,
                expert_id=expert_id,
                routing_load=average(lambda signal: signal.routing_load),
                gradient_norm=average(lambda signal: signal.gradient_norm),
                parameter_norm=average(lambda signal: signal.parameter_norm),
                temporal_stability=average(lambda signal: signal.temporal_stability),
                functional_similarity=(
                    sum(similarity_values) / len(similarity_values)
                    if similarity_values
                    else None
                ),
            )
        )
    return tuple(aggregated)


def fisher_ema(
    previous: float,
    gradient_norm: float,
    *,
    decay: float = 0.9,
) -> float:
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be between 0 and 1")
    if not math.isfinite(previous) or not math.isfinite(gradient_norm):
        raise ValueError("importance values must be finite")
    return decay * previous + (1.0 - decay) * gradient_norm**2


def parameter_magnitude(model: torch.nn.Module) -> float:
    total = sum(float(parameter.detach().pow(2).sum()) for parameter in model.parameters())
    return math.sqrt(total)


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


def expert_functional_sketches(
    model: StaticMoECausalLM,
    hidden: Tensor,
) -> dict[str, Tensor]:
    """Return detached output sketches for each expert on the same hidden batch."""
    if hidden.ndim != 2 or hidden.shape[-1] != model.config.hidden_size:
        raise ValueError("hidden must have shape [tokens, hidden_size]")
    sketches: dict[str, Tensor] = {}
    with torch.no_grad():
        for block in model.blocks:
            for expert_id, expert in zip(block.expert_ids, block.experts):
                sketches[f"{expert_id}"] = expert(hidden).detach().flatten()
    return sketches


def expert_ablation_delta(
    model: StaticMoECausalLM,
    tokens: Tensor,
    expert_id: str,
) -> float:
    """Measure validation-loss increase when one expert is temporarily zeroed."""
    target = None
    for block in model.blocks:
        if expert_id in block.expert_ids:
            target = block.experts[block.expert_ids.index(expert_id)]
            break
    if target is None:
        raise ValueError(f"unknown expert: {expert_id}")
    original = {
        name: parameter.detach().clone()
        for name, parameter in target.named_parameters()
    }
    model.eval()
    with torch.no_grad():
        baseline = float(model.loss(tokens).detach())
        for parameter in target.parameters():
            parameter.zero_()
        ablated = float(model.loss(tokens).detach())
    with torch.no_grad():
        for name, parameter in target.named_parameters():
            parameter.copy_(original[name])
    return ablated - baseline
