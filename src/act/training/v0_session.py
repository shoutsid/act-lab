"""Resumable V0-scale training sessions for the compute-efficiency protocol.

Executes the pre-registered section 5.1 measurement design: the adaptive arm
runs probe-loss selection events over a depth-12 sparse MoE while the random
baseline arm trains under matched budgets. Everything needed to cross the
12-hour Kaggle session boundary is persisted in the checkpoint:

- model and optimizer state (the version-3 checkpoint contract, extended to
  version 4 with the ``v0_session`` block below),
- the architecture genome and mutation history,
- the training/search FLOPs counters the cost criterion is computed from,
- the CPU data-generator state, so the token stream resumes exactly,
- global torch RNG state.

Determinism contract: training tokens come from a single CPU generator
seeded ``seed ^ 0x5EED`` whose state is checkpointed; probe and validation
batches come from dedicated generators (``seed ^ 0xB0B`` and
``seed ^ 0xA11CE``) so resuming never shifts the token stream.

Selection semantics mirror the toy evolution path (candidate enumeration
order, probe-loss argmin with order tie-break, fresh AdamW after each
mutation, and the ``parameter_count * 4`` search-FLOPs bookkeeping), so the
cost identity calibrated on the toy studies applies unchanged.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import random
import time
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

from act.model import (
    V0MoECausalLM,
    V0MoEConfig,
    estimated_parameter_count,
    rebuild_v0_moe_from_genome,
    v0_moe_model_from_genome,
    validate_declared_v0_config,
)
from act.topology import (
    ArchitectureGenome,
    create_genome,
    feasible_expert_mutations,
    genome_hash,
    grow_expert,
    prune_expert,
)
from act.training.runtime import move_optimizer_state, resolve_device
from act.training.session import SessionCheckpointPolicy, retain_checkpoints
from act.training.v0_span import WindowSampler


@dataclass(frozen=True)
class V0SessionConfig:
    """Declared protocol configuration by default (section 5.1)."""

    seed: int = 9000
    arms: tuple[str, ...] = ("random", "adaptive")
    device: str = "auto"
    # Declared V0 model (plan section 8).
    vocab_size: int = 32_768
    context_size: int = 2_048
    hidden_size: int = 1_024
    depth: int = 12
    attention_heads: int = 16
    expert_count: int = 8
    active_experts: int = 2
    expert_width: int = 1_024
    declared: bool = True
    # Declared budget: batch 2 x context 2048 x 6000 steps = 24,576,000 tokens.
    batch_size: int = 2
    total_steps: int = 6_000
    learning_rate: float = 3e-4
    # Declared events: E=3 at steps 1500/3000/4500.
    mutation_interval: int = 1_500
    mutation_budget: int = 3
    # Selection: cap-4/1-batch primary (set probe_batch_count=2 for secondary).
    probe_candidate_cap: int = 4
    probe_batch_count: int = 1
    validation_batches: int = 4
    # Session mechanics. The declared config trains with checkpointed
    # activations (section 5.1); the T4 fit arithmetic in section 5.2 assumes
    # it. Blocks carry no dropout, so the recompute is deterministic and the
    # bit-identical resume contract is unaffected.
    activation_checkpointing: bool = True
    # Execution dtype (section 5.2's throughput-gated rule): "fp32", or
    # "fp16" = autocast forward/backward with a GradScaler over fp32 master
    # weights. Dtype touches no pre-registered criterion: FLOPs counters
    # book operations, not time, so both dtypes produce identical counters.
    dtype: str = "fp32"
    checkpoint_interval: int = 500
    max_checkpoints: int = 3
    checkpoint_max_bytes: int | None = None
    # Real-data token streams (section 5.1 machinery on real text). When set,
    # every draw is a random window into a flat uint16 token stream instead of
    # a uniform-random token tensor; shapes, generators, checkpointed state,
    # and FLOPs booking are identical to the synthetic path, so the cost
    # identity and the bit-identical resume contract apply unchanged.
    # ``data_path`` feeds training and probe batches; ``val_data_path`` feeds
    # the held-out validation loss (defaulting to the training stream).
    data_path: str | None = None
    val_data_path: str | None = None

    def __post_init__(self) -> None:
        if self.total_steps < 1 or self.batch_size < 1:
            raise ValueError("total_steps and batch_size must be positive")
        if self.mutation_interval < 1 or self.mutation_budget < 1:
            raise ValueError("mutation_interval and mutation_budget must be positive")
        if self.probe_candidate_cap < 1 or self.probe_batch_count < 1:
            raise ValueError("probe budgets must be positive")
        if self.dtype not in {"fp32", "fp16"}:
            raise ValueError(f"unsupported session dtype: {self.dtype}")
        if self.validation_batches < 1:
            raise ValueError("validation_batches must be positive")
        unknown = [arm for arm in self.arms if arm not in {"random", "adaptive"}]
        if unknown:
            raise ValueError(f"unsupported V0 session arms: {unknown}")
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("arms must be unique")
        for path in (self.data_path, self.val_data_path):
            if path is not None and not Path(path).is_file():
                raise ValueError(f"data stream not found: {path}")
        if self.val_data_path is not None and self.data_path is None:
            raise ValueError("val_data_path requires data_path (synthetic sessions are fully synthetic)")

    def model_config(self) -> V0MoEConfig:
        return V0MoEConfig(
            vocab_size=self.vocab_size,
            context_size=self.context_size,
            hidden_size=self.hidden_size,
            depth=self.depth,
            attention_heads=self.attention_heads,
            experts_per_layer=(self.expert_count,) * self.depth,
            active_experts=self.active_experts,
            expert_width=self.expert_width,
        )

    def training_tokens(self) -> int:
        return self.batch_size * self.context_size * self.total_steps


def predicted_cost_ratio(config: V0SessionConfig) -> float:
    """Predicted search/training FLOPs ratio from the section 5.1 identity.

    ``S = E * k * Bp * 2 * N_active`` (probe forwards) and ``T = 2 * P * D``
    (training forwards), the same forward-equivalent convention the toy
    studies and ``run_experiment`` book with.
    """
    total, active = estimated_parameter_count(config.model_config())
    events = min(config.mutation_budget, config.total_steps // config.mutation_interval)
    probe_tokens = config.batch_size * config.context_size * config.probe_batch_count
    search = events * config.probe_candidate_cap * probe_tokens * 2 * active
    training = 2 * total * config.training_tokens()
    return search / training


@dataclass(frozen=True)
class V0ArmResult:
    mode: str
    final_loss: float
    validation_loss: float
    quality: float
    mutation_count: int
    parameter_count: int
    active_parameter_count: int
    training_flops: int
    search_flops: int
    architecture_hash: str
    mutation_history: tuple[dict[str, object], ...]
    wall_time_seconds: float
    tokens_per_second: float


@dataclass(frozen=True)
class V0SessionResult:
    config: dict[str, object]
    predicted_cost_ratio: float
    arms: tuple[V0ArmResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config,
            "predicted_cost_ratio": self.predicted_cost_ratio,
            "arms": [asdict(arm) for arm in self.arms],
        }


class V0SessionInterrupted(RuntimeError):
    """Raised when a run stops cooperatively at a checkpoint boundary.

    Carries the step it stopped at so the caller can distinguish a clean
    deadline stop from a crash: state up to ``step`` is saved and valid.
    """

    def __init__(self, message: str, step: int) -> None:
        super().__init__(message)
        self.step = step


@dataclass
class _SessionState:
    step: int = 0
    tokens: int = 0
    training_flops: int = 0
    search_flops: int = 0
    mutations: int = 0
    generator_state: Tensor | None = None
    genome: ArchitectureGenome | None = None
    mutation_history: list[dict[str, object]] = field(default_factory=list)


def _genome_from_config(config: V0SessionConfig) -> ArchitectureGenome:
    genome = create_genome(
        config.seed,
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        depth=config.depth,
        ffn="moe",
        expert_count=config.expert_count,
        active_experts=config.active_experts,
    )
    genome.context_size = config.context_size
    for layer in genome.layers:
        layer.heads = config.attention_heads
        layer.kv_heads = config.attention_heads
        for expert in layer.experts:
            expert.width = config.expert_width
    return genome


def _checkpoint_path(directory: Path, step: int) -> Path:
    return directory / f"checkpoint-{step:06d}.pt"


def _latest_checkpoint(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    candidates = []
    for path in directory.glob("checkpoint-*.pt"):
        stem = path.stem.split("-")[-1]
        if stem.isdigit():
            candidates.append((int(stem), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def save_v0_checkpoint(
    path: str | Path,
    model: V0MoECausalLM,
    optimizer: torch.optim.Optimizer,
    state: _SessionState,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    """Persist resumable session state in the version-5 checkpoint contract.

    Version 5 adds the GradScaler state for fp16 sessions (``scaler`` is
    None in fp32 mode). Version-4 payloads remain loadable: they simply have
    no scaler entry, which a resumed fp32 session never needs.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 5,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "training": {
                "step": state.step,
                "tokens": state.tokens,
                "generator_state": state.generator_state,
                "genome": state.genome.to_dict() if state.genome is not None else None,
            },
            "v0_session": {
                "training_flops": state.training_flops,
                "search_flops": state.search_flops,
                "mutations": state.mutations,
                "mutation_history": state.mutation_history,
            },
        },
        target,
    )


def _state_from_payload(
    payload: dict[str, object],
    model: V0MoECausalLM,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> _SessionState:
    """Apply model/optimizer/RNG state from a loaded payload and build state.

    With ``scaler`` given (fp16 resume), the payload's scaler state is
    restored so loss-scale growth resumes exactly where the stop left it.
    """
    model.load_state_dict(payload["model"])  # type: ignore[arg-type]
    optimizer.load_state_dict(payload["optimizer"])  # type: ignore[arg-type]
    move_optimizer_state(optimizer, device)
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"])  # type: ignore[arg-type]
    scaler_payload = payload.get("scaler")
    if scaler is not None and scaler_payload is not None:
        scaler.load_state_dict(scaler_payload)
    training = payload["training"]
    session = payload.get("v0_session", {})
    genome_dict = training.get("genome")
    generator_state = training.get("generator_state")
    return _SessionState(
        step=int(training["step"]),
        tokens=int(training["tokens"]),
        training_flops=int(session.get("training_flops", 0)),
        search_flops=int(session.get("search_flops", 0)),
        mutations=int(session.get("mutations", 0)),
        generator_state=generator_state.clone() if generator_state is not None else None,
        genome=ArchitectureGenome.from_dict(genome_dict) if genome_dict else None,
        mutation_history=list(session.get("mutation_history", [])),
    )


def _prepare_model(model: V0MoECausalLM, config: V0SessionConfig) -> V0MoECausalLM:
    """Apply the declared session mode to a (fresh or rebuilt) model."""
    model.activation_checkpointing = config.activation_checkpointing
    model.train()
    return model


def _probe_loss(
    model: V0MoECausalLM,
    probe_batches: tuple[Tensor, ...],
) -> float:
    model.eval()
    with torch.no_grad():
        return sum(float(model.loss(batch)) for batch in probe_batches) / len(probe_batches)


def _apply_candidate(
    genome: ArchitectureGenome,
    operation: str,
    layer_id: str,
    expert_id: str | None,
    step: int,
) -> bool:
    if operation == "grow":
        return grow_expert(genome, layer_id, step) is not None
    if expert_id is None:
        return False
    return prune_expert(genome, layer_id, expert_id)


def _adaptive_event(
    arm_state: _SessionState,
    model: V0MoECausalLM,
    optimizer: torch.optim.Optimizer,
    config: V0SessionConfig,
    step: int,
    device: torch.device,
    probe_batches: tuple[Tensor, ...],
) -> tuple[V0MoECausalLM, torch.optim.Optimizer]:
    """Probe candidate mutations and apply the lowest-probe-loss one.

    Mirrors the toy ``_mutate_by_probe``: probe models are built one at a time
    (never retained), search FLOPs are booked per evaluated candidate, and the
    winner is rebuilt once from the current model with by-id expert transfer.
    A no-op event (no feasible candidates) preserves the existing optimizer
    exactly as the toy path does.
    """
    candidates = feasible_expert_mutations(arm_state.genome)[: config.probe_candidate_cap]
    probe_tokens = sum(batch.numel() for batch in probe_batches)
    evaluated: list[tuple[float, int, str, str, str | None, ArchitectureGenome]] = []
    for order, (operation, layer_id, expert_id) in enumerate(candidates):
        trial_genome = deepcopy(arm_state.genome)
        if not _apply_candidate(trial_genome, operation, layer_id, expert_id, step):
            continue
        probe_model = rebuild_v0_moe_from_genome(model, trial_genome).to(device)
        loss = _probe_loss(probe_model, probe_batches)
        arm_state.search_flops += probe_model.active_parameter_count * probe_tokens * 2
        evaluated.append((loss, order, operation, layer_id, expert_id, trial_genome))
        del probe_model
    if not evaluated:
        return model, optimizer
    evaluated.sort(key=lambda item: (item[0], item[1]))
    probe_loss, _, operation, layer_id, expert_id, winner_genome = evaluated[0]
    before_hash = genome_hash(arm_state.genome)
    arm_state.genome = winner_genome
    arm_state.mutations += 1
    winner = rebuild_v0_moe_from_genome(model, arm_state.genome).to(device)
    arm_state.search_flops += winner.parameter_count * 4
    _prepare_model(winner, config)
    new_optimizer = torch.optim.AdamW(winner.parameters(), lr=config.learning_rate)
    arm_state.mutation_history.append(
        {
            "step": step,
            "policy": "adaptive_probe_loss",
            "operation": operation,
            "layer_id": layer_id,
            "expert_id": expert_id,
            "architecture_hash_before": before_hash,
            "architecture_hash_after": genome_hash(arm_state.genome),
            "mutation_count": arm_state.mutations,
            "probe_loss": probe_loss,
            "probe_candidates": len(evaluated),
            "probe_batch_count": len(probe_batches),
        }
    )
    return winner, new_optimizer


def _random_event(
    arm_state: _SessionState,
    model: V0MoECausalLM,
    optimizer: torch.optim.Optimizer,
    config: V0SessionConfig,
    step: int,
    device: torch.device,
) -> tuple[V0MoECausalLM, torch.optim.Optimizer]:
    """Apply one random mutation with the toy baseline's exact semantics.

    Matches ``_mutate_arm`` for the random mode: chooser call order (layer,
    operation, expert), in-place genome mutation with the operators' internal
    rollback, and the load-based fallback grow. Only the operator choice
    distribution differs from a probe arm; probe cost stays zero.
    """
    chooser = random.Random(config.seed + step)
    before_hash = genome_hash(arm_state.genome)
    layer = arm_state.genome.layers[chooser.randrange(len(arm_state.genome.layers))]
    operation = chooser.choice(("grow", "prune"))
    event_layer_id = layer.id
    applied_expert: str | None = None
    changed = False
    if operation == "grow":
        changed = grow_expert(arm_state.genome, layer.id, step) is not None
    elif len(layer.experts) > 2:
        expert = chooser.choice(layer.experts)
        applied_expert = expert.id
        changed = prune_expert(arm_state.genome, layer.id, expert.id)
    if not changed:
        fallback = max(
            arm_state.genome.layers,
            key=lambda candidate: max(model.routing_load[candidate.index]),
        )
        if len(fallback.experts) < 8:
            changed = grow_expert(arm_state.genome, fallback.id, step) is not None
            if changed:
                operation, event_layer_id, applied_expert = "grow", fallback.id, None
    if not changed:
        return model, optimizer
    arm_state.mutations += 1
    rebuilt = rebuild_v0_moe_from_genome(model, arm_state.genome).to(device)
    arm_state.search_flops += rebuilt.parameter_count * 4
    _prepare_model(rebuilt, config)
    new_optimizer = torch.optim.AdamW(rebuilt.parameters(), lr=config.learning_rate)
    arm_state.mutation_history.append(
        {
            "step": step,
            "policy": "random",
            "operation": operation,
            "layer_id": event_layer_id,
            "expert_id": applied_expert,
            "architecture_hash_before": before_hash,
            "architecture_hash_after": genome_hash(arm_state.genome),
            "mutation_count": arm_state.mutations,
        }
    )
    return rebuilt, new_optimizer


def _train_arm(
    mode: str,
    config: V0SessionConfig,
    session_root: Path,
    device: torch.device,
    log: Callable[[str], None] | None,
    deadline_check: Callable[[], bool] | None = None,
) -> V0ArmResult:
    """Train one arm with checkpoint/resume, returning its final result."""
    arm_dir = session_root / f"arm-{mode}"
    policy = SessionCheckpointPolicy(
        interval=config.checkpoint_interval,
        max_checkpoints=config.max_checkpoints,
        max_bytes=(
            config.checkpoint_max_bytes
            if config.checkpoint_max_bytes is not None
            else 2**62
        ),
    )
    latest = _latest_checkpoint(arm_dir)
    resumed_from: str | None = None
    # Model init consumes global torch RNG, so it must be seeded per arm (not
    # per session): two fresh runs, or a resumed run, must rebuild identical
    # weights regardless of execution order or process history. Python's
    # str hash is process-salted, so the offset is an explicit map.
    arm_seed = config.seed * 10 + {"random": 1, "adaptive": 2}[mode]
    # fp16 sessions train under autocast with a GradScaler over fp32 master
    # weights; fp32 sessions keep the plain path. Both are created here so a
    # resume restores into the same object graph either way.
    use_amp = config.dtype == "fp16"
    scaler = torch.amp.GradScaler("cpu", enabled=use_amp) if use_amp else None
    if latest is not None:
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        version = payload.get("version") if isinstance(payload, dict) else None
        if version == 4 and use_amp:
            raise ValueError(
                f"checkpoint {latest} is a version-4 payload with no scaler state; "
                "fp16 sessions require version-5 checkpoints - rerun this arm in fp32"
            )
        if version not in {4, 5}:
            raise ValueError(
                f"checkpoint {latest} is not a version-4/5 V0 session payload"
            )
        genome = ArchitectureGenome.from_dict(payload["training"]["genome"])
        model = _prepare_model(v0_moe_model_from_genome(genome).to(device), config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        arm_state = _state_from_payload(payload, model, optimizer, device, scaler)
        resumed_from = str(latest)
        if log is not None:
            log(f"[{mode}] resumed from {latest} at step {arm_state.step}")
    else:
        torch.manual_seed(arm_seed)
        genome = _genome_from_config(config)
        model = _prepare_model(v0_moe_model_from_genome(genome).to(device), config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        arm_state = _SessionState(genome=genome)
    if config.declared:
        validate_declared_v0_config(config.model_config())

    # Training token stream: single generator, checkpointed state, so a
    # resumed session continues the identical sequence of batches. With a
    # real-data stream the same generator picks window positions instead of
    # raw tokens; RNG consumption and checkpointed state are identical either
    # way, so the resume contract is untouched.
    train_sampler = (
        WindowSampler(config.data_path, config.vocab_size)
        if config.data_path is not None
        else None
    )
    generator = torch.Generator(device="cpu")
    if arm_state.generator_state is None:
        generator.manual_seed(config.seed ^ 0x5EED)
    else:
        generator.set_state(arm_state.generator_state)

    # Probe batches: dedicated generator (never touches the token stream), one
    # training-batch shape per probe batch, as declared in section 5.1.
    probe_generator = torch.Generator(device="cpu")
    probe_generator.manual_seed(config.seed ^ 0xB0B)
    if train_sampler is not None:
        probe_batches = tuple(
            train_sampler.draw(probe_generator, config.batch_size, config.context_size).to(device)
            for _ in range(config.probe_batch_count)
        )
    else:
        probe_batches = tuple(
            torch.randint(
                0,
                config.vocab_size,
                (config.batch_size, config.context_size),
                generator=probe_generator,
            ).to(device)
            for _ in range(config.probe_batch_count)
        )

    _prepare_model(model, config)
    wall_start = time.time()
    for step in range(arm_state.step + 1, config.total_steps + 1):
        if train_sampler is not None:
            tokens = train_sampler.draw(
                generator, config.batch_size, config.context_size
            ).to(device)
        else:
            tokens = torch.randint(
                0,
                config.vocab_size,
                (config.batch_size, config.context_size),
                generator=generator,
            ).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            loss = model.loss(tokens)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{mode} arm produced non-finite loss at step {step}")
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        arm_state.training_flops += model.parameter_count * tokens.numel() * 2
        arm_state.tokens += tokens.numel()
        arm_state.step = step
        # Snapshot the generator after each consumed batch so every checkpoint
        # carries the exact token-stream position it was saved at.
        arm_state.generator_state = generator.get_state()
        if (
            step % config.mutation_interval == 0
            and arm_state.mutations < config.mutation_budget
        ):
            if mode == "adaptive":
                model, optimizer = _adaptive_event(
                    arm_state, model, optimizer, config, step, device, probe_batches
                )
            else:
                model, optimizer = _random_event(
                    arm_state, model, optimizer, config, step, device
                )
        if policy.should_save(step):
            path = _checkpoint_path(arm_dir, step)
            save_v0_checkpoint(path, model, optimizer, arm_state, scaler)
            policy.accept(path)
            retain_checkpoints(arm_dir, max_checkpoints=config.max_checkpoints)
            # Cooperative deadline stop: only at a saved checkpoint boundary,
            # only when the caller's clock says stop. State up to here is on
            # disk, so a snapshot publish loses nothing.
            if deadline_check is not None and deadline_check():
                if log is not None:
                    log(f"[{mode}] deadline stop at step {step}")
                raise V0SessionInterrupted(
                    f"{mode} arm stopped at checkpoint boundary step {step}", step
                )
    wall_time = time.time() - wall_start
    arm_state.generator_state = generator.get_state()

    # Final checkpoint so a completed arm never needs recomputation.
    final_path = _checkpoint_path(arm_dir, config.total_steps)
    save_v0_checkpoint(final_path, model, optimizer, arm_state, scaler)
    policy.accept(final_path)
    retain_checkpoints(arm_dir, max_checkpoints=config.max_checkpoints)
    if resumed_from is not None and log is not None:
        log(f"[{mode}] completed after resume from {resumed_from}")

    validation_generator = torch.Generator(device="cpu")
    validation_generator.manual_seed(config.seed ^ 0xA11CE)
    val_sampler = (
        WindowSampler(config.val_data_path or config.data_path, config.vocab_size)
        if (config.val_data_path or config.data_path) is not None
        else None
    )
    if val_sampler is not None:
        validation_batches = tuple(
            val_sampler.draw(validation_generator, config.batch_size, config.context_size).to(device)
            for _ in range(config.validation_batches)
        )
    else:
        validation_batches = tuple(
            torch.randint(
                0,
                config.vocab_size,
                (config.batch_size, config.context_size),
                generator=validation_generator,
            ).to(device)
            for _ in range(config.validation_batches)
        )
    model.eval()
    with torch.no_grad():
        validation_loss = (
            sum(float(model.loss(batch)) for batch in validation_batches)
            / len(validation_batches)
        )
    return V0ArmResult(
        mode=mode,
        final_loss=validation_loss,
        validation_loss=validation_loss,
        quality=-validation_loss,
        mutation_count=arm_state.mutations,
        parameter_count=model.parameter_count,
        active_parameter_count=model.active_parameter_count,
        training_flops=arm_state.training_flops,
        search_flops=arm_state.search_flops,
        architecture_hash=genome_hash(arm_state.genome),
        mutation_history=tuple(arm_state.mutation_history),
        wall_time_seconds=wall_time,
        tokens_per_second=arm_state.tokens / wall_time if wall_time > 0 else 0.0,
    )


def run_v0_session(
    config: V0SessionConfig = V0SessionConfig(),
    *,
    checkpoint_dir: str | Path = "artifacts/v0-sessions",
    log: Callable[[str], None] | None = None,
    deadline_check: Callable[[], bool] | None = None,
) -> V0SessionResult:
    """Run (or resume) the declared protocol session for one seed.

    ``deadline_check`` is consulted only at checkpoint boundaries; when it
    returns True the run stops cooperatively by raising
    ``V0SessionInterrupted`` with every checkpoint up to the stop step saved.
    """
    if config.declared:
        validate_declared_v0_config(config.model_config())
    device = resolve_device(config.device)
    root = Path(checkpoint_dir) / f"seed-{config.seed}"
    root.mkdir(parents=True, exist_ok=True)
    arms = tuple(
        _train_arm(mode, config, root, device, log, deadline_check)
        for mode in config.arms
    )
    return V0SessionResult(
        config=asdict(config),
        predicted_cost_ratio=predicted_cost_ratio(config),
        arms=arms,
    )
