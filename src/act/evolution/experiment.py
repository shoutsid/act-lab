"""Matched-budget Experiment 0 for the Python training path."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import random

import torch

from act.model import StaticMoECausalLM, model_from_genome, rebuild_model_from_genome
from act.evolution.controller import AdaptiveController
from act.topology import ArchitectureGenome, create_genome, genome_hash, grow_expert, prune_expert
from act.training.runtime import resolve_device


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    steps: int = 48
    batch_size: int = 2
    sequence_length: int = 12
    validation_batches: int = 4
    mutation_interval: int = 12
    mutation_budget: int = 3
    hidden_size: int = 24
    vocab_size: int = 32
    depth: int = 1
    expert_count: int = 4
    active_experts: int = 2
    learning_rate: float = 3e-4
    dataset_id: str = "synthetic-v1"
    tokenizer_id: str = "synthetic-tokenizer-v1"
    code_revision: str = "working-tree"
    adaptive_load_threshold: float = 0.50
    adaptive_expert_metric: str = "random"
    device: str = "cpu"


@dataclass(frozen=True)
class ArmResult:
    mode: str
    final_loss: float
    validation_loss: float
    quality: float
    mutation_count: int
    parameter_count: int
    active_parameter_count: int
    training_flops: int
    validation_flops: int
    search_flops: int
    architecture_distance: int
    architecture_hash: str
    mutation_history: tuple[dict[str, object], ...]
    importance_records: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class ExperimentResult:
    config: dict[str, object]
    arms: tuple[ArmResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {"config": self.config, "arms": [asdict(arm) for arm in self.arms]}


@dataclass
class _Arm:
    mode: str
    genome: ArchitectureGenome
    model: StaticMoECausalLM
    optimizer: torch.optim.Optimizer
    initial_genome: ArchitectureGenome
    mutations: int = 0
    training_flops: int = 0
    search_flops: int = 0
    importance_records: list[dict[str, object]] | None = None
    fisher_ema: dict[str, float] | None = None
    previous_load: dict[str, float] | None = None
    mutation_history: list[dict[str, object]] | None = None
    device: torch.device = torch.device("cpu")


def run_experiment(config: ExperimentConfig = ExperimentConfig()) -> ExperimentResult:
    if config.steps < 1 or config.mutation_interval < 1:
        raise ValueError("steps and mutation_interval must be positive")
    if config.validation_batches < 1:
        raise ValueError("validation_batches must be positive")
    device = resolve_device(config.device)
    if not 0.0 < config.adaptive_load_threshold < 1.0:
        raise ValueError("adaptive_load_threshold must be between 0 and 1")
    if config.adaptive_expert_metric not in {
        "random",
        "routing_load",
        "gradient_norm",
        "fisher_ema",
        "parameter_norm",
    }:
        raise ValueError("unsupported adaptive_expert_metric")
    torch.manual_seed(config.seed)
    initial_genome = create_genome(
        config.seed,
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        depth=config.depth,
        ffn="moe",
        expert_count=config.expert_count,
        active_experts=config.active_experts,
    )
    initial_model = model_from_genome(initial_genome, context_size=config.sequence_length)
    if not isinstance(initial_model, StaticMoECausalLM):
        raise TypeError("Experiment 0 requires a static MoE genome")
    initial_model.to(device)
    arms = tuple(
        _make_arm(mode, initial_genome, initial_model, config.learning_rate, device)
        for mode in ("fixed", "random", "adaptive")
    )
    data = torch.Generator(device="cpu")
    data.manual_seed(config.seed ^ 0x5EED)
    for step in range(1, config.steps + 1):
        tokens = torch.randint(
            0,
            config.vocab_size,
            (config.batch_size, config.sequence_length),
            generator=data,
        ).to(device)
        for arm in arms:
            arm.optimizer.zero_grad(set_to_none=True)
            loss = arm.model.loss(tokens)
            loss.backward()
            arm.optimizer.step()
            arm.training_flops += arm.model.parameter_count * tokens.numel() * 2
            if (
                arm.mode != "fixed"
                and step % config.mutation_interval == 0
                and arm.mutations < config.mutation_budget
            ):
                _record_importance(arm, step)
                _mutate_arm(
                    arm,
                    step,
                    config.seed + step,
                    config.learning_rate,
                    config.adaptive_load_threshold,
                    config.adaptive_expert_metric,
                )
    validation_tokens = _validation_batches(config, device)
    results = tuple(
        _finish_arm(arm, validation_tokens) for arm in arms
    )
    return ExperimentResult(config=asdict(config), arms=results)


def _make_arm(
    mode: str,
    genome: ArchitectureGenome,
    model: StaticMoECausalLM,
    learning_rate: float,
    device: torch.device,
) -> _Arm:
    cloned_model = deepcopy(model)
    cloned_genome = deepcopy(genome)
    optimizer = torch.optim.AdamW(cloned_model.parameters(), lr=learning_rate)
    return _Arm(
        mode,
        cloned_genome,
        cloned_model,
        optimizer,
        deepcopy(genome),
        device=device,
        importance_records=[],
        fisher_ema={},
        previous_load={},
        mutation_history=[],
    )


def _record_importance(arm: _Arm, step: int) -> None:
    if arm.importance_records is None or arm.fisher_ema is None or arm.previous_load is None:
        raise RuntimeError("importance state is not initialized")
    architecture_hash = genome_hash(arm.genome)
    for layer_index, layer in enumerate(arm.genome.layers):
        loads = arm.model.routing_load[layer_index]
        block = arm.model.blocks[layer_index]
        for expert_index, expert_gene in enumerate(layer.experts):
            expert = block.experts[expert_index]
            gradient_squared = 0.0
            parameter_squared = 0.0
            for parameter in expert.parameters():
                parameter_squared += float(parameter.detach().pow(2).sum())
                if parameter.grad is not None:
                    gradient_squared += float(parameter.grad.detach().pow(2).sum())
            key = f"{layer.id}:{expert_gene.id}"
            routing_load = float(loads[expert_index])
            previous = arm.previous_load.get(key, routing_load)
            stability = 1.0 - min(abs(routing_load - previous), 1.0)
            fisher = 0.9 * arm.fisher_ema.get(key, 0.0) + 0.1 * gradient_squared
            arm.fisher_ema[key] = fisher
            arm.previous_load[key] = routing_load
            arm.importance_records.append(
                {
                    "step": step,
                    "mutation_count": arm.mutations,
                    "architecture_hash": architecture_hash,
                    "layer_id": layer.id,
                    "expert_id": expert_gene.id,
                    "routing_load": routing_load,
                    "gradient_norm": math.sqrt(gradient_squared),
                    "parameter_norm": math.sqrt(parameter_squared),
                    "fisher_ema": fisher,
                    "temporal_stability": stability,
                }
            )


def _mutate_arm(
    arm: _Arm,
    step: int,
    seed: int,
    learning_rate: float,
    adaptive_load_threshold: float,
    adaptive_expert_metric: str,
) -> None:
    chooser = random.Random(seed)
    before_hash = genome_hash(arm.genome)
    layer = arm.genome.layers[chooser.randrange(len(arm.genome.layers))]
    if arm.mode == "adaptive":
        records = _latest_importance(arm)
        decision = AdaptiveController(
            load_threshold=adaptive_load_threshold,
            metric=adaptive_expert_metric if adaptive_expert_metric != "random" else "routing_load",
        ).decide(
            arm.genome,
            loads=arm.model.routing_load,
            latest=records,
        )
        operation = decision.operation
        layer = next(item for item in arm.genome.layers if item.id == decision.layer_id)
    else:
        operation = chooser.choice(("grow", "prune"))
    changed = False
    if operation == "grow":
        changed = grow_expert(arm.genome, layer.id, step) is not None
    elif len(layer.experts) > 2:
        if arm.mode == "adaptive" and adaptive_expert_metric != "random":
            records = _latest_importance(arm)
            decision = AdaptiveController(
                load_threshold=adaptive_load_threshold,
                metric=adaptive_expert_metric,
            ).decide(arm.genome, loads=arm.model.routing_load, latest=records)
            expert = next(
                candidate for candidate in layer.experts if candidate.id == decision.expert_id
            )
        else:
            expert = chooser.choice(layer.experts)
        changed = prune_expert(arm.genome, layer.id, expert.id)
    if not changed:
        fallback = max(
            arm.genome.layers,
            key=lambda candidate: max(arm.model.routing_load[candidate.index]),
        )
        if len(fallback.experts) < 8:
            changed = grow_expert(arm.genome, fallback.id, step) is not None
    if not changed:
        return
    rebuilt = rebuild_model_from_genome(
        arm.model,
        arm.genome,
        context_size=arm.model.config.context_size,
        num_heads=arm.model.config.num_heads,
    )
    if not isinstance(rebuilt, StaticMoECausalLM):
        raise TypeError("mutation changed the baseline model family")
    arm.model = rebuilt.to(arm.device)
    arm.optimizer = torch.optim.AdamW(arm.model.parameters(), lr=learning_rate)
    arm.mutations += 1
    arm.search_flops += arm.model.parameter_count * 4
    if arm.mutation_history is None:
        raise RuntimeError("mutation history is not initialized")
    arm.mutation_history.append(
        {
            "step": step,
            "operation": operation,
            "layer_id": layer.id,
            "architecture_hash_before": before_hash,
            "architecture_hash_after": genome_hash(arm.genome),
            "mutation_count": arm.mutations,
        }
    )


def _latest_importance(arm: _Arm) -> dict[str, dict[str, object]]:
    if not arm.importance_records:
        return {}
    latest: dict[str, dict[str, object]] = {}
    for record in arm.importance_records:
        key = f"{record['layer_id']}:{record['expert_id']}"
        latest[key] = record
    return latest


def _validation_batches(
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed ^ 0xA11CE)
    batches = tuple(
        torch.randint(
            0,
            config.vocab_size,
            (config.batch_size, config.sequence_length),
            generator=generator,
        )
        for _ in range(config.validation_batches)
    )
    return tuple(batch.to(device) for batch in batches)


def _finish_arm(arm: _Arm, batches: tuple[torch.Tensor, ...]) -> ArmResult:
    arm.model.eval()
    with torch.no_grad():
        loss = sum(float(arm.model.loss(tokens)) for tokens in batches) / len(batches)
    active_parameter_count = arm.model.active_parameter_count
    return ArmResult(
        mode=arm.mode,
        final_loss=loss,
        validation_loss=loss,
        quality=-loss,
        mutation_count=arm.mutations,
        parameter_count=arm.model.parameter_count,
        active_parameter_count=active_parameter_count,
        training_flops=arm.training_flops,
        validation_flops=sum(
            active_parameter_count * tokens.numel() * 2 for tokens in batches
        ),
        search_flops=arm.search_flops,
        architecture_distance=_architecture_distance(arm.initial_genome, arm.genome),
        architecture_hash=genome_hash(arm.genome),
        mutation_history=tuple(arm.mutation_history or ()),
        importance_records=tuple(arm.importance_records or ()),
    )


def _architecture_distance(
    initial: ArchitectureGenome,
    current: ArchitectureGenome,
) -> int:
    distance = 0
    for initial_layer, current_layer in zip(initial.layers, current.layers):
        initial_ids = {expert.id for expert in initial_layer.experts}
        current_ids = {expert.id for expert in current_layer.experts}
        distance += len(initial_ids.symmetric_difference(current_ids))
        distance += abs(initial_layer.active_experts - current_layer.active_experts)
    return distance