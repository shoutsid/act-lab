"""Matched-budget Experiment 0 for the Python training path."""

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import random

import torch

from act.model import StaticMoECausalLM, model_from_genome, rebuild_model_from_genome
from act.evolution.controller import AdaptiveController
from act.topology import ArchitectureGenome, create_genome, feasible_expert_mutations, genome_hash, grow_expert, prune_expert
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
    adaptive_policy: str = "probe_loss"
    probe_candidate_cap: int = 4
    probe_batch_count: int = 2
    # Which candidates a cap-bounded probe sees. "prefix" (default, the
    # historical behaviour every published study used) takes the first
    # ``probe_candidate_cap`` entries of the layer-major enumeration, which at
    # depth 1 is nearly the whole space but at depth > 1 confines the search
    # to the earliest layer(s). "spread" draws a deterministic per-event
    # random subset of the same size from the WHOLE feasible set, so the cap
    # is a genuine search budget over layers and operations rather than a
    # prefix.
    probe_candidate_policy: str = "prefix"
    # Decision rule applied to the probed candidates. "argmin" (default, the
    # registered controller) takes the lowest probe loss, "argmax" the highest
    # (the inverted-signal control), and "random" picks uniformly among the
    # SAME probed candidates at the SAME probe cost - the pool-matched random
    # control that isolates selection quality from the candidate pool's shape.
    # Mirrors `V0SessionConfig.probe_selection` so both paths share semantics.
    probe_selection: str = "argmin"
    # Give every arm its own rebuild-initialisation stream. The historical
    # default (False) shares the global torch RNG across arms, and because a
    # grow draws fresh expert weights while a prune draws nothing, two runs
    # whose treatments choose differently perturb each other's *random* arm.
    # True seeds the global RNG per (arm, event) before each mutation, so the
    # random baseline is exactly the same control regardless of the treatment
    # (report section 9.8). Off by default so published numbers reproduce.
    independent_arm_rng: bool = False
    probe_landscape: bool = False
    probe_force_candidate: int | None = None
    probe_force_events: int | None = None
    # Steps at which every arm's optimizer is re-created from scratch, with no
    # change to the genome or the weights - Adam's moment estimates are simply
    # discarded, which is what the mutation path does as a side effect of
    # rebuilding the model (see ``_mutate_by_probe`` / the random path: both
    # assign a fresh AdamW after rebuilding). It exists so the mid-training
    # process penalty of report section 5.22 can be decomposed: a run whose only
    # difference from its twin is these resets isolates the optimizer cost from
    # the architecture-change cost. Empty (the default) leaves every published
    # number untouched.
    optimizer_reset_steps: tuple[int, ...] = ()
    # Whether a mid-training rebuild carries the mutated layer's learned router
    # (the ``hidden x experts`` routing head) across the mutation. Off by default
    # because its shape changes with the expert count, so the historical rebuild
    # re-draws it at random whenever that layer's expert count changes - the
    # behaviour every published study measured. On, the surviving experts keep
    # their learned routing rows by id (`rebuild_model_from_genome
    # (preserve_router=True)`). Report section 5.24 tests whether the
    # mid-training penalty is this discard.
    preserve_router_on_rebuild: bool = False
    arms: str = "fixed,random,adaptive"
    device: str = "cpu"


def _mutate_null(arm, step, mutations, budget):
    """Record one do-nothing structural event for the null arm (report 5.9).

    The toy twin of the V0 path's null arm (report section 5.8): no draw, no
    probe, no rebuild, no fallback - and, unlike the V0 diagnostic schedule
    (whose 600 steps / interval 200 give exactly 3 slots), the toy loop fires
    at every interval slot, so the arm self-limits its recorded events to
    ``mutation_budget`` (the random arm stops mutating once its count reaches
    the budget, so alignment with the random arm's event count is exact).
    """
    if arm.mutation_history is None:
        raise RuntimeError("mutation history is not initialized")
    # ``mutations`` stays 0 forever on this arm, so the recorded history is
    # the slot counter: stop once the budget is consumed.
    if len(arm.mutation_history) >= budget:
        return
    mutations = len(arm.mutation_history)
    arm.mutation_history.append(
        {
            "step": step,
            "policy": "null",
            "operation": "null",
            "layer_id": None,
            "architecture_hash_before": genome_hash(arm.genome),
            "architecture_hash_after": genome_hash(arm.genome),
            "mutation_count": mutations,
        }
    )


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
    # Plan section 11 (weight/topology separation): the discovered topology as
    # a portable object, so the genome - not the weights - can be extracted,
    # transported and retrained. ``ArchitectureGenome.from_dict`` inverts it.
    genome: dict[str, object]


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
    probe_batches: tuple[torch.Tensor, ...] = ()
    preserve_router: bool = False


def _parse_arms(spec: str) -> tuple[str, ...]:
    """Parse and validate the comma-separated arm selection."""
    modes = tuple(mode.strip() for mode in spec.split(",") if mode.strip())
    supported = {"fixed", "random", "adaptive", "guided", "null"}
    if not modes:
        raise ValueError("arms must name at least one arm")
    if len(set(modes)) != len(modes):
        raise ValueError("arms must be unique")
    unknown = [mode for mode in modes if mode not in supported]
    if unknown:
        raise ValueError(f"unsupported arms: {unknown}")
    return modes


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
    if config.adaptive_policy not in {"probe_loss", "signal"}:
        raise ValueError("unsupported adaptive_policy")
    if config.probe_candidate_cap < 1:
        raise ValueError("probe_candidate_cap must be at least 1")
    if config.probe_candidate_policy not in {"prefix", "spread"}:
        raise ValueError("unsupported probe_candidate_policy")
    if config.probe_selection not in {"argmin", "argmax", "random"}:
        raise ValueError("unsupported probe_selection")
    if config.probe_batch_count < 1:
        raise ValueError("probe_batch_count must be at least 1")
    if config.probe_force_candidate is not None and config.probe_force_candidate < 0:
        raise ValueError("probe_force_candidate must be non-negative when set")
    if config.probe_force_events is not None and config.probe_force_events < 1:
        raise ValueError("probe_force_events must be at least 1 when set")
    reset_steps = config.optimizer_reset_steps
    if reset_steps and (
        any(step < 1 or step > config.steps for step in reset_steps)
        or tuple(sorted(set(reset_steps))) != reset_steps
    ):
        raise ValueError(
            "optimizer_reset_steps must be strictly increasing positive steps "
            "within the run"
        )
    arm_modes = _parse_arms(config.arms)
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
    probe_generator = torch.Generator(device="cpu")
    probe_generator.manual_seed(config.seed ^ 0xB0B)
    probe_batches = tuple(
        batch.to(device)
        for batch in (
            torch.randint(
                0,
                config.vocab_size,
                (config.batch_size, config.sequence_length),
                generator=probe_generator,
            )
            for _ in range(config.validation_batches)
        )
    )
    arms = tuple(
        _make_arm(
            mode,
            initial_genome,
            initial_model,
            config.learning_rate,
            device,
            probe_batches,
            config.preserve_router_on_rebuild,
        )
        for mode in arm_modes
    )
    reset_step_set = set(config.optimizer_reset_steps)
    for step, tokens in enumerate(_training_batches(config, device), start=1):
        for arm_index, arm in enumerate(arms):
            arm.optimizer.zero_grad(set_to_none=True)
            loss = arm.model.loss(tokens)
            loss.backward()
            arm.optimizer.step()
            arm.training_flops += arm.model.parameter_count * tokens.numel() * 2
            if step in reset_step_set:
                # Section 5.23's control: discard Adam's moments without
                # touching the genome or the weights.
                arm.optimizer = torch.optim.AdamW(
                    arm.model.parameters(), lr=config.learning_rate
                )
            if (
                arm.mode != "fixed"
                and step % config.mutation_interval == 0
                and arm.mutations < config.mutation_budget
            ):
                if arm.mode == "null":
                    _mutate_null(
                        arm,
                        step,
                        arm.mutations,
                        config.mutation_budget,
                    )
                    continue
                if config.independent_arm_rng:
                    # Rebuild initialisation is the only consumer of the global
                    # torch RNG here (data and validation use dedicated CPU
                    # generators), so re-seeding per (arm, event) decouples the
                    # arms without touching the shared init identity.
                    torch.manual_seed(
                        config.seed * 1_000_003 + arm_index * 7_919 + step
                    )
                _record_importance(arm, step)
                _mutate_arm(
                    arm,
                    step,
                    config.seed + step,
                    config.learning_rate,
                    config.adaptive_load_threshold,
                    config.adaptive_expert_metric,
                    config.adaptive_policy,
                    config.probe_candidate_cap,
                    config.probe_batch_count,
                    config.probe_candidate_policy,
                    config.probe_selection,
                    config.probe_landscape,
                    config.probe_force_candidate,
                    config.probe_force_events,
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
    probe_batches: tuple[torch.Tensor, ...],
    preserve_router: bool = False,
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
        probe_batches=probe_batches,
        preserve_router=preserve_router,
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


def _mutation_candidates(
    genome: ArchitectureGenome,
) -> tuple[tuple[str, str, str | None], ...]:
    """Backward-compatible alias for the shared candidate enumeration."""
    return feasible_expert_mutations(genome)


def _mutate_by_probe(
    arm: _Arm,
    step: int,
    learning_rate: float,
    candidate_cap: int | None,
    probe_batch_count: int,
    policy_label: str,
    candidate_policy: str = "prefix",
    probe_selection: str = "argmin",
    sampler_seed: int = 0,
    record_landscape: bool = False,
    force_candidate: int | None = None,
    force_events: int | None = None,
) -> None:
    """Probe feasible mutations and apply the one with the lowest probe loss.

    ``candidate_cap`` bounds the number of candidates evaluated (``None``
    evaluates all of them, the exact guided upper bound) and
    ``probe_batch_count`` bounds the probe batch budget. Probe forward cost is
    booked into ``search_flops`` so selection strategies stay comparable.
    ``record_landscape`` stores every candidate's probe loss in the mutation
    history and ``force_candidate`` deterministically applies the candidate at
    that evaluation index instead of the probe-best one (counterfactual runs).
    """
    if not arm.probe_batches:
        raise RuntimeError("probe-guided mutation requires probe batches")
    candidates = _mutation_candidates(arm.genome)
    if not candidates:
        return
    if candidate_cap is not None:
        if candidate_policy == "spread" and len(candidates) > candidate_cap:
            # Deterministic per-event subset from the WHOLE feasible set, so a
            # bounded probe can still reach any layer/operation. Seeded from
            # the event's seed (config.seed + step) so re-runs and resumes
            # redraw the identical subset.
            candidates = tuple(
                random.Random(sampler_seed).sample(list(candidates), candidate_cap)
            )
        else:
            candidates = candidates[:candidate_cap]
    probe_batches = arm.probe_batches[:probe_batch_count]
    probe_tokens = sum(batch.numel() for batch in probe_batches)
    evaluated: list[tuple[float, int, str, str, str | None, ArchitectureGenome]] = []
    for order, (operation, layer_id, expert_id) in enumerate(candidates):
        trial_genome = deepcopy(arm.genome)
        if operation == "grow":
            if grow_expert(trial_genome, layer_id, step) is None:
                continue
        elif not prune_expert(trial_genome, layer_id, expert_id):
            continue
        probe_model = rebuild_model_from_genome(
            deepcopy(arm.model),
            trial_genome,
            context_size=arm.model.config.context_size,
            num_heads=arm.model.config.num_heads,
        )
        if not isinstance(probe_model, StaticMoECausalLM):
            raise TypeError("probe mutation changed the baseline model family")
        probe_model.to(arm.device)
        probe_model.eval()
        with torch.no_grad():
            loss = (
                sum(float(probe_model.loss(batch)) for batch in probe_batches)
                / len(probe_batches)
            )
        arm.search_flops += (
            probe_model.active_parameter_count * probe_tokens * 2
        )
        evaluated.append((loss, order, operation, layer_id, expert_id, trial_genome))
    if not evaluated:
        return
    evaluated.sort(key=lambda item: (item[0], item[1]))
    forced_here = (
        force_candidate is not None
        and (force_events is None or arm.mutations < force_events)
    )
    if forced_here and force_candidate >= len(evaluated):
        raise ValueError(
            f"probe_force_candidate {force_candidate} out of range "
            f"(only {len(evaluated)} candidates evaluated)"
        )
    if forced_here:
        selected_index = force_candidate
    elif probe_selection == "argmax":
        selected_index = len(evaluated) - 1
    elif probe_selection == "random":
        # Pool-matched random control: the same probed candidates at the same
        # probe cost, chosen without reference to probe loss. Seeded from the
        # event's seed so re-runs redraw the identical choice.
        selected_index = random.Random(sampler_seed).randrange(len(evaluated))
    else:
        selected_index = 0
    probe_loss, _, operation, layer_id, expert_id, winner = evaluated[selected_index]
    before_hash = genome_hash(arm.genome)
    arm.genome = winner
    rebuilt = rebuild_model_from_genome(
        arm.model,
        arm.genome,
        context_size=arm.model.config.context_size,
        num_heads=arm.model.config.num_heads,
        preserve_router=arm.preserve_router,
    )
    if not isinstance(rebuilt, StaticMoECausalLM):
        raise TypeError("mutation changed the baseline model family")
    arm.model = rebuilt.to(arm.device)
    arm.optimizer = torch.optim.AdamW(arm.model.parameters(), lr=learning_rate)
    arm.mutations += 1
    arm.search_flops += arm.model.parameter_count * 4
    if arm.mutation_history is None:
        raise RuntimeError("mutation history is not initialized")
    entry: dict[str, object] = {
        "step": step,
        "policy": policy_label,
        "operation": operation,
        "layer_id": layer_id,
        "expert_id": expert_id,
        "architecture_hash_before": before_hash,
        "architecture_hash_after": genome_hash(arm.genome),
        "mutation_count": arm.mutations,
        "probe_loss": probe_loss,
        "probe_candidates": len(evaluated),
        "probe_batch_count": len(probe_batches),
        "probe_candidate_policy": candidate_policy,
        "probe_selection": probe_selection,
    }
    if record_landscape:
        entry["probe_selected_index"] = selected_index
        # evaluated holds (loss, order, operation, layer_id, expert_id, genome)
        # sorted by (loss, order); landscape is reported in that sorted order,
        # so index k in the landscape is the k-th best candidate.
        entry["probe_landscape"] = [
            {
                "operation": item[2],
                "layer_id": item[3],
                "expert_id": item[4],
                "probe_loss": item[0],
            }
            for item in evaluated
        ]
    if forced_here:
        entry["forced_candidate"] = force_candidate
    arm.mutation_history.append(entry)


def _mutate_arm(
    arm: _Arm,
    step: int,
    seed: int,
    learning_rate: float,
    adaptive_load_threshold: float,
    adaptive_expert_metric: str,
    adaptive_policy: str,
    probe_candidate_cap: int,
    probe_batch_count: int,
    probe_candidate_policy: str = "prefix",
    probe_selection: str = "argmin",
    record_landscape: bool = False,
    force_candidate: int | None = None,
    force_events: int | None = None,
) -> None:
    if arm.mode == "guided":
        _mutate_by_probe(
            arm,
            step,
            learning_rate,
            None,
            len(arm.probe_batches),
            "guided",
            record_landscape=record_landscape,
            force_candidate=force_candidate,
            force_events=force_events,
        )
        return
    if arm.mode == "adaptive" and adaptive_policy == "probe_loss":
        _mutate_by_probe(
            arm,
            step,
            learning_rate,
            probe_candidate_cap,
            probe_batch_count,
            "adaptive_probe_loss",
            candidate_policy=probe_candidate_policy,
            probe_selection=probe_selection,
            sampler_seed=seed,
            record_landscape=record_landscape,
            force_candidate=force_candidate,
            force_events=force_events,
        )
        return
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
        preserve_router=arm.preserve_router,
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


def _training_batches(
    config: ExperimentConfig,
    device: torch.device,
) -> Iterator[torch.Tensor]:
    """The per-seed training data schedule: one batch per step, in order.

    Extracted so that a study which retrains a discovered topology on the
    *same* schedule (plan section 52) consumes the same draws rather than a
    duplicated copy of the seed constant. The draws depend only on the
    config, never on the arm, so every arm of a block sees identical data.
    """
    data = torch.Generator(device="cpu")
    data.manual_seed(config.seed ^ 0x5EED)
    for _ in range(config.steps):
        yield torch.randint(
            0,
            config.vocab_size,
            (config.batch_size, config.sequence_length),
            generator=data,
        ).to(device)


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
        genome=arm.genome.to_dict(),
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