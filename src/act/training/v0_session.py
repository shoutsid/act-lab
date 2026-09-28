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
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

from act.model import (
    DECLARED_V0_CONFIG,
    DEV_V0_CONFIG,
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
    quietest_prune_index,
)
from act.training.runtime import move_optimizer_state, resolve_device, step_marker
from act.training.session import SessionCheckpointPolicy, retain_checkpoints
from act.training.v0_span import WindowSampler


@dataclass(frozen=True)
class V0SessionConfig:
    """Fast-iteration dev configuration by default (see ``declared_session_config``).

    The default is deliberately NOT the pre-registered section 5.1 target. That
    target is a 285M-parameter, 24.6M-token-per-arm protocol run whose cycle time
    on a single development GPU is an hour or more, and using it as the default
    made every iteration cost a run. The default here is the measured-fastest
    small shape (``DEV_V0_CONFIG``, ~14.9M total / 7.6M    active) with a 400-step
    budget: 1.64M tokens per arm, about 1.1 minutes per arm on a GTX 1070
    (measured end to end on the real TinyStories stream by ``scripts/dev_cycle.py``,
    with the dev default's activation checkpointing off - about 1.4 minutes with it
    on, which is the recompute the dev shape has no memory reason to pay).
    The
    declared target is built explicitly by ``declared_session_config()`` and
    validated by ``validate_declared_v0_config`` whenever ``declared`` is True.
    """

    # Default seed for a lone session. 15000 is the start of the declared
    # section 5.1 queue (re-reserved from 9000 by the adopted section 5.5
    # change 4, since 9000-9011 was spent by the section 5.3 diagnostic).
    seed: int = 15000
    arms: tuple[str, ...] = ("random", "adaptive")
    device: str = "auto"
    # Dev V0 model (plan section 8's target is reached via
    # ``declared_session_config()``).
    vocab_size: int = 8_192
    context_size: int = 512
    hidden_size: int = 320
    depth: int = 6
    attention_heads: int = 5
    expert_count: int = 8
    active_experts: int = 2
    expert_width: int = 320
    declared: bool = False
    # Dev budget: batch 8 x context 512 x 400 steps = 1,638,400 tokens per arm.
    batch_size: int = 8
    total_steps: int = 400
    learning_rate: float = 3e-4
    # Dev events: E=3 at steps 100/200/300 (the declared cadence compressed to
    # the dev horizon; the declared run keeps 1500/3000/4500).
    mutation_interval: int = 100
    mutation_budget: int = 3
    # Selection: cap-4/1-batch primary (set probe_batch_count=2 for secondary).
    probe_candidate_cap: int = 4
    probe_batch_count: int = 1
    # Which candidates a capped probe sees. "prefix" is the registered
    # section 5.1 default: the first ``probe_candidate_cap`` entries of the
    # layer-major enumeration, which at the declared depth (12 layers x 8
    # experts = 108 candidates) confines a cap of 4 to the earliest layer.
    # "spread" draws a deterministic per-event subset of the whole feasible
    # set, so the cap is a search budget over layers and operations rather
    # than a prefix. Added by the substrate audit (`RESEARCH_REPORT.md`
    # section 9); the registered protocol is unchanged at the default.
    probe_candidate_policy: str = "prefix"
    # Decision rule applied to the probed candidates (real-data diagnostic,
    # report section 5.3): "argmin" is the registered controller (lowest probe
    # loss wins); "argmax" deliberately selects the probe-WORST candidate as
    # the inverted-signal control. Both probe the same candidates with the
    # same batches and book the same search FLOPs - only the decision differs.
    probe_selection: str = "argmin"
    # Report section 5.34/5.24: what the mid-training rebuild does with the
    # mutated layer's learned router (a ``hidden x experts_in_layer`` head whose
    # shape moves with the expert count, so the generic same-shape copy cannot
    # carry it and the historical rebuild re-draws it at random on every event).
    # Off by default because it is the behaviour every committed number was
    # measured under; on, surviving experts keep their learned rows by id and a
    # grown expert is initialised per `new_expert_initialization` below.
    preserve_router_on_rebuild: bool = False
    # "random" (the historical draw), "clone" (a copy of the busiest surviving
    # expert in the layer, router row included) or "clone_body" (the same body
    # with a fresh row). Requires `preserve_router_on_rebuild`.
    new_expert_initialization: str = "random"
    # Landscape instrumentation (report section 5.3): when enabled, each
    # adaptive mutation-history event records every probed candidate with its
    # probe loss, sorted best-first so list index = probe rank, plus the
    # selected candidate's index. Bookkeeping only - it changes no decision,
    # no FLOPs counter, and no RNG consumption.
    probe_landscape: bool = False
    validation_batches: int = 4
    # Session mechanics. Activation checkpointing trades recompute for memory, so
    # it belongs where memory is the constraint: the *declared* target (285M) sets
    # it explicitly for that reason, and section 5.2's T4 fit arithmetic assumes it.
    # The dev default is OFF, because at the dev shape memory is not a constraint
    # and the recompute is pure cost - measured on the GTX 1070 at the dev shape,
    # 60 steps on the real stream, one arm: 13.3 s with it on against 10.1 s off
    # (1.32x), with **bit-identical losses**. The flag is trajectory-inert by
    # construction, which report section 5.37 pins as a measured contract entry, and
    # blocks carry no dropout, so the bit-identical resume contract is unaffected.
    activation_checkpointing: bool = False
    # Execution dtype (section 5.2's throughput-gated rule): "fp32", or
    # "fp16" = autocast forward/backward with a GradScaler over fp32 master
    # weights. Dtype touches no pre-registered criterion: FLOPs counters
    # book operations, not time, so both dtypes produce identical counters.
    dtype: str = "fp32"
    checkpoint_interval: int = 100
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
    # Probe data source (report section 5.3's post-argmax contract): "train"
    # (default) probes on training-stream windows - the registered controller;
    # "heldout" probes on windows from the held-out validation stream, testing
    # whether the argmin ranking's anti-predictive component comes from probe
    # windows sitting in-corpus. Same candidates, same probe batch count, same
    # booked probe FLOPs - only the probe windows' source stream differs.
    probe_data_source: str = "train"
    # Long-horizon probe (report section 5.3's contingency fix candidate):
    # 0 (default) keeps the registered snapshot probe - probe loss on fresh
    # batches, no adaptation. R > 0 turns the probe into an R-step lookahead:
    # each probed candidate is fine-tuned for R AdamW steps (learning_rate,
    # fresh optimizer, no momentum carried in) on R fresh training-stream
    # batches before its probe loss is measured on the SAME probe batches, so
    # selection scores post-mutation trainability instead of the instantaneous
    # loss dip (a freshly initialized expert looks bad before any adaptation,
    # which biases argmin against grows). Cost adds ~2R% of training FLOPs at
    # the proof configuration (R=1 lands ~3%, under the 5% criterion); the
    # decision rule, candidate cap, and probe-batch yardstick are unchanged.
    # Fine-tune batches are event/candidate/step-addressed from a dedicated
    # throwaway generator (never checkpointed), so resume stays bit-identical.
    probe_finetune_steps: int = 0
    # Accept/reject gating (report section 5.4's pre-registered fix
    # candidate): the diagnostic verdict showed the probe's candidate
    # ORDERING is anti-correlated with downstream value, so the gated arm
    # abandons ordering entirely and uses only the probe's ABSOLUTE signal -
    # "does this single mutation hurt the current model on the probe
    # batches?" - as an accept/reject gate on otherwise-random draws.
    # Mechanics: draw a mutation exactly as the random arm does (same seeded
    # chooser, same call order, same fallback), probe (i) the drawn mutation
    # and (ii) the no-op reference (the current model itself, so the gate is
    # a per-event two-point contrast, not a fixed absolute threshold), and
    # accept when the drawn mutation does not lose to the reference by more
    # than ``gate_margin`` nats; otherwise fall back to the draw's own
    # load-based grow (the random arm's already-exercised fallback path), so
    # the mutation budget stays matched (publish refuses unmatched budgets).
    # Probe cost: 2 evaluations per event (draw + reference), ~2x baseline.
    # ``gate_margin = inf`` reduces the arm to the random arm plus idle
    # probe FLOPs - the built-in budget-matched random control.
    gate_margin: float = 0.0
    # Do-nothing control (report section 5.8, motivated by the gated
    # outcome's ``reject_noop`` sweep: the only real-text configuration that
    # beat the random arms was one that performed zero mutations). The null
    # arm is that configuration made explicit: a single-arm session that
    # shares the random arm's init identity (offset 1), training stream, and
    # chooser-seed discipline, but performs NO mutation at any structural
    # event - each event appends one recorded no-op entry (operation "null",
    # gate_decision "null") and books no probe FLOPs, so search_flops stays
    # exactly 0 and mutation_count stays 0. It is the cleanest churn-harm
    # baseline the matched-budget protocol supports: budget-matched (every
    # event slot is consumed), single-arm like gated, and paired against the
    # baseline block's stored random arms. Because the gated arm at theta=0
    # rejected every draw and every fallback on the diagnostic block, the
    # null arm should reproduce the gated arm's losses bit-for-bit on the
    # same seeds - a built-in replication check of the section-5.4 run.
    # Routing-statistics heuristic (report section 5.13, the section-5.11
    # contract's named direction off probe-derived selection): a deterministic,
    # probe-free policy - grow one expert at the argmax-routing-load layer
    # (the same load statistic the random arm's fallback reads), every event,
    # no chooser draw, no probes, no rollback. It is the minimal non-probe
    # selection signal: the only information used is the load imbalance the
    # training forwards already produce. Single-arm like gated/null, sharing
    # the random arm's init identity; paired against the published null arms
    # to answer "can DIRECTED churn beat doing nothing?" after probe-ordered
    # and gated churn both lost to it.

    def __post_init__(self) -> None:
        if self.total_steps < 1 or self.batch_size < 1:
            raise ValueError("total_steps and batch_size must be positive")
        if self.mutation_interval < 1 or self.mutation_budget < 1:
            raise ValueError("mutation_interval and mutation_budget must be positive")
        if self.probe_candidate_cap < 1 or self.probe_batch_count < 1:
            raise ValueError("probe budgets must be positive")
        if self.probe_selection not in {"argmin", "argmax", "random", "quietest"}:
            raise ValueError(f"unsupported probe_selection: {self.probe_selection}")
        if self.new_expert_initialization not in {"random", "clone", "clone_body"}:
            raise ValueError(
                "unsupported new_expert_initialization: "
                f"{self.new_expert_initialization}"
            )
        if (
            self.new_expert_initialization in {"clone", "clone_body"}
            and not self.preserve_router_on_rebuild
        ):
            # Cloning needs the surviving routers in place to clone from, so this
            # combination is rejected rather than silently downgraded.
            raise ValueError(
                f"new_expert_initialization={self.new_expert_initialization!r} "
                "requires preserve_router_on_rebuild=True"
            )
        if self.probe_candidate_policy not in {"prefix", "spread"}:
            raise ValueError(
                f"unsupported probe_candidate_policy: {self.probe_candidate_policy}"
            )
        if self.probe_data_source not in {"train", "heldout"}:
            raise ValueError(f"unsupported probe_data_source: {self.probe_data_source}")
        if self.probe_data_source == "heldout" and self.val_data_path is None:
            raise ValueError("probe_data_source='heldout' requires val_data_path")
        if self.probe_finetune_steps < 0:
            raise ValueError("probe_finetune_steps must be non-negative")
        if self.gate_margin < 0:
            raise ValueError("gate_margin must be non-negative")
        if "gated" in self.arms and self.arms != ("gated",):
            raise ValueError("the gated arm runs as its own single-arm session (arms=('gated',))")
        if "null" in self.arms and self.arms != ("null",):
            raise ValueError("the null arm runs as its own single-arm session (arms=('null',))")
        if "heuristic" in self.arms and self.arms != ("heuristic",):
            raise ValueError(
                "the heuristic arm runs as its own single-arm session (arms=('heuristic',))"
            )
        if self.dtype not in {"fp32", "fp16"}:
            raise ValueError(f"unsupported session dtype: {self.dtype}")
        if self.validation_batches < 1:
            raise ValueError("validation_batches must be positive")
        unknown = [arm for arm in self.arms if arm not in {"random", "adaptive", "gated", "null", "heuristic"}]
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


def declared_session_config(**overrides: object) -> V0SessionConfig:
    """The pre-registered section 5.1 / plan-section-8 target session.

    285M total / 134M active parameters, batch 2 x context 2048 x 6000 steps =
    24,576,000 tokens per arm, events at steps 1500/3000/4500. This is what a
    declared run uses; the plain ``V0SessionConfig()`` default is the fast dev
    shape, so anything that means "the protocol configuration" must say so by
    calling this rather than by using the defaults.
    """
    fields: dict[str, object] = {
        "vocab_size": DECLARED_V0_CONFIG.vocab_size,
        "context_size": DECLARED_V0_CONFIG.context_size,
        "hidden_size": DECLARED_V0_CONFIG.hidden_size,
        "depth": DECLARED_V0_CONFIG.depth,
        "attention_heads": DECLARED_V0_CONFIG.attention_heads,
        "expert_count": DECLARED_V0_CONFIG.experts_per_layer[0],
        "active_experts": DECLARED_V0_CONFIG.active_experts,
        "expert_width": DECLARED_V0_CONFIG.expert_width,
        "declared": True,
        "batch_size": 2,
        "total_steps": 6_000,
        "mutation_interval": 1_500,
        "mutation_budget": 3,
        "checkpoint_interval": 500,
        "dtype": "fp16",
        # Explicit rather than inherited: section 5.2's T4 fit arithmetic assumes
        # checkpointed activations, and the field default is the (fast) dev value.
        "activation_checkpointing": True,
    }
    fields.update(overrides)
    config = V0SessionConfig(**fields)  # type: ignore[arg-type]
    validate_declared_v0_config(config.model_config())
    return config


def predicted_cost_ratio(config: V0SessionConfig) -> float:
    """Predicted search/training FLOPs ratio from the section 5.1 identity.

    ``S = E * k * Bp * 2 * N_active`` (probe forwards) and ``T = 2 * P * D``
    (training forwards), the same forward-equivalent convention the toy
    studies and ``run_experiment`` book with. With the long-horizon probe
    (``probe_finetune_steps = R > 0``) each candidate additionally books R
    full training steps on the whole probed model (``R * Bp * 2 * P``).
    """
    total, active = estimated_parameter_count(config.model_config())
    events = min(config.mutation_budget, config.total_steps // config.mutation_interval)
    probe_tokens = config.batch_size * config.context_size * config.probe_batch_count
    if config.arms == ("gated",):
        # Accept/reject gating probes TWO things per event: the drawn
        # mutation and the no-op reference (the current model). No candidate
        # cap applies - the gate is a two-point contrast, not a ranking.
        search = events * 2 * probe_tokens * 2 * active
        training = 2 * total * config.training_tokens()
        return search / training
    if config.arms == ("null",):
        # The do-nothing control books nothing: no probes, no rebuilds.
        return 0.0
    if config.arms == ("heuristic",):
        # The routing-statistics heuristic books no probes; its only extra
        # cost is the post-grow rebuild every mutating arm books anyway.
        return 0.0
    per_candidate = probe_tokens * 2 * active
    if config.probe_finetune_steps > 0:
        per_candidate += (
            config.probe_finetune_steps
            * config.batch_size
            * config.context_size
            * 2
            * total
        )
    search = events * config.probe_candidate_cap * per_candidate
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
    # Where this arm trained. Section 5.2 prices a pair with the arms running
    # concurrently, one per device, so the published summary is the durable
    # evidence that the parallelism actually engaged - without it a reader
    # cannot tell a two-device pair from a two-arm sequential one. Placement
    # metadata, like the timing fields beside it: it is not part of the result.
    device: str = ""


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


def _cpu_payload(value: object) -> object:
    """A checkpoint payload with every tensor on the host, containers rebuilt.

    **The version-5 payload is CPU-canonical**, and that is a property of the
    format rather than of any device: every reader of these files loads with
    ``map_location="cpu"`` and then hands the optimizer state to
    ``move_optimizer_state``, so a CPU payload is the shape both ends already
    assume. So the write converts, and the conflict this settles is one this
    project has already been bitten by in its other form - a checkpoint that
    loads and means something else.

    The conversion is not a precaution for the CPU or CUDA paths. There it is
    the same host copy ``torch.save`` makes while serializing, and on the CPU
    it is literally a no-op (``Tensor.cpu()`` returns ``self``). It is a *fix*
    for XLA, where ``torch.save`` writes the **XLA tensors themselves** rather
    than copying them out: torch_xla ships ``xm.save`` for exactly this reason
    (it moves everything to the host first), and the two ways the alternative
    fails are both unacceptable here. A payload of XLA-tagged tensors is
    unreadable on a host with a different device count (pytorch-lightning
    #2700, and pytorch/xla #2364 before it), and a lazy tensor whose storage was
    never materialized can serialize as the **wrong bytes** - a checkpoint that
    loads, checksums, publishes, and is not the training it claims. This
    campaign cannot tell that one from a good checkpoint, which is the failure
    this project treats as worse than a crash.

    Device-free values (the genome, the FLOP counters, the generator state,
    which is already a CPU tensor) pass through unchanged.
    """
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, dict):
        return {key: _cpu_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_payload(item) for item in value)
    return value


def save_v0_checkpoint(
    path: str | Path,
    model: V0MoECausalLM,
    optimizer: torch.optim.Optimizer,
    state: _SessionState,
    scaler: torch.amp.GradScaler | None = None,
    *,
    code_revision: str | None = None,
) -> None:
    """Persist resumable session state in the version-5 checkpoint contract.

    Version 5 adds the GradScaler state for fp16 sessions (``scaler`` is
    None in fp32 mode). Version-4 payloads remain loadable: they simply have
    no scaler entry, which a resumed fp32 session never needs.

    The payload is written through :func:`_cpu_payload`, so the file holds host
    tensors whichever device trained them - the shape every reader assumes, and
    the only shape a lazy accelerator's tensor can be written without the risk
    of a checkpoint that loads and is not the training it names.

    ``code_revision`` is provenance carried *on the artifact*: the published
    manifest records it too (2026-09-27), but a checkpoint separated from its
    manifest carried nothing at all - the measured state of every checkpoint
    this campaign published, since the payload had no revision and no timestamp
    anywhere in it. It is written only when the caller has one, and a payload
    without the key loads exactly as before.

    Written beside the target and renamed into place. A declared checkpoint is
    ~3.4 GB, so the write takes seconds, and a process killed inside it (the
    OOM killer, a kernel timeout, a deadline) would otherwise leave a
    *truncated* file at the final name - where the retention policy takes it
    for the newest checkpoint, ``publish_seed`` checksums it into a `complete`
    version, and restore's checksum verification cannot tell the difference.
    A partial write is invisible from every reader; a rename is atomic.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    training: dict[str, object] = {
        "step": state.step,
        "tokens": state.tokens,
        "generator_state": state.generator_state,
        "genome": state.genome.to_dict() if state.genome is not None else None,
    }
    if code_revision is not None:
        training["code_revision"] = code_revision
    torch.save(
        _cpu_payload(
            {
                "version": 5,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "training": training,
                "v0_session": {
                    "training_flops": state.training_flops,
                    "search_flops": state.search_flops,
                    "mutations": state.mutations,
                    "mutation_history": state.mutation_history,
                },
            }
        ),
        temporary,
    )
    os.replace(temporary, target)


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


def genome_from_checkpoint(path: str | Path) -> ArchitectureGenome:
    """Read a V0 session checkpoint's architecture genome, and nothing else.

    Gate 5 (plan section 99) reads *fresh architecture remains competitive*: the
    claim is that the topology a search discovered is worth carrying, not the
    training it accumulated. Every checkpoint carries both - the payload holds
    ``training.genome`` beside ``model`` - but before this reader the only code
    that consumed a checkpoint's genome was :func:`_train_arms`' resume branch,
    which rebuilds a model from the genome and then immediately restores
    ``model``, ``optimizer``, RNG and scaler state through
    :func:`_state_from_payload`. So an architecture could only ever be
    transferred *with* its training, and a Gate 5 run had no path to take. This
    is that path's first half: it returns the genome and touches nothing else.

    Raises on a payload that is not a version-4/5 V0 session checkpoint, and on
    one whose ``training.genome`` is absent or ``None`` - a payload written
    before the field existed records ``None``, and ``ArchitectureGenome.from_dict``
    would otherwise be handed an empty mapping.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path} is not a V0 session payload")
    if payload.get("version") not in {4, 5}:
        raise ValueError(
            f"checkpoint {path} is not a version-4/5 V0 session payload"
        )
    training = payload.get("training")
    genome_dict = training.get("genome") if isinstance(training, dict) else None
    if not genome_dict:
        raise ValueError(
            f"checkpoint {path} carries no genome; architecture transfer needs "
            "the training.genome field"
        )
    return ArchitectureGenome.from_dict(genome_dict)


def fresh_init_from_checkpoint(
    path: str | Path,
    config: V0SessionConfig,
    *,
    device: str | torch.device = "cpu",
) -> tuple[V0MoECausalLM, ArchitectureGenome]:
    """Build a *freshly initialized* model from a checkpoint's genome.

    The counterpart to :func:`_train_arms`' resume branch, and the reason the
    two are distinguishable: resume applies the payload's ``model``,
    ``optimizer``, RNG and scaler state over a model rebuilt from the same
    genome, so a resumed arm's parameters *equal* the ones it stopped with,
    while this seeds a fresh initialization at ``config.seed`` and leaves the
    optimizer and the token generator untouched. The returned model therefore
    matches the published arm in topology - same parameter names, same shapes,
    same parameter count - and differs in every trained value. That pair is
    what Gate 5 has to read, and is what a rehearsal pins.

    ``config`` supplies the run shape (context, batch, learning rate, cadence);
    the genome supplies the architecture, and wins where the two disagree,
    since the architecture is the thing being carried.
    """
    genome = genome_from_checkpoint(path)
    torch.manual_seed(config.seed)
    model = _prepare_model(v0_moe_model_from_genome(genome).to(torch.device(device)), config)
    return model, genome


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


def _probe_loss_lookahead(
    model: V0MoECausalLM,
    probe_batches: tuple[Tensor, ...],
    config: V0SessionConfig,
    next_finetune_batch: Callable[[], Tensor],
) -> float:
    """Score a candidate by its post-adaptation loss: the long-horizon probe.

    Fine-tunes the probed model for ``config.probe_finetune_steps`` AdamW
    steps (the session learning rate, a fresh optimizer - no momentum is
    carried in or out) on freshly drawn training batches, then re-scores the
    SAME probe batches the snapshot probe would use. The per-candidate
    protocol is identical for every candidate of an event, so the ranking is
    a trainability comparison rather than an instantaneous-loss comparison: a
    freshly initialized expert (bad before adaptation, potentially good
    after) is no longer structurally penalized. The caller discards the
    probed model, so no adapted state can leak into the arm.
    """
    steps = config.probe_finetune_steps
    if steps > 0:
        finetune_optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        model.train()
        for _ in range(steps):
            batch = next_finetune_batch()
            finetune_optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch)
            if not torch.isfinite(loss):
                return float("inf")  # untrainable candidate: sorts last
            loss.backward()
            finetune_optimizer.step()
        del finetune_optimizer
    return _probe_loss(model, probe_batches)


def _rebuild_mutated(
    model: V0MoECausalLM,
    genome: ArchitectureGenome,
    config: V0SessionConfig,
) -> V0MoECausalLM:
    """One place where the applied rebuild's settings are read (section 5.34).

    Every applied rebuild in this module goes through here so the treatment cannot
    be wired into some paths and forgotten in others - the failure mode that made
    section 5.33's ``prune_expert_policy`` and ``forced_operation`` inert on the
    probe path. Note that the rebuild's model construction still draws the same
    amount of global torch RNG whatever the settings, so the gated arm's lockstep
    with the random arm is unaffected.
    """
    return rebuild_v0_moe_from_genome(
        model,
        genome,
        preserve_router=config.preserve_router_on_rebuild,
        new_expert_initialization=config.new_expert_initialization,
    )


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
    next_finetune_batch: Callable[[], Tensor] | None = None,
) -> tuple[V0MoECausalLM, torch.optim.Optimizer]:
    """Probe candidate mutations and apply the probed-selection winner.

    Mirrors the toy ``_mutate_by_probe``: probe models are built one at a time
    (never retained), search FLOPs are booked per evaluated candidate, and the
    winner is rebuilt once from the current model with by-id expert transfer.
    The decision rule is ``config.probe_selection``: "argmin" (the registered
    controller) takes the lowest probe loss; "argmax" (the diagnostic control)
    takes the highest. With ``probe_finetune_steps = R > 0`` (the long-horizon
    probe) each candidate is scored after R adaptation steps on freshly drawn
    batches (``next_finetune_batch``), so the ranking measures post-mutation
    trainability, and the FLOPs booking covers those steps on the full probed
    model. A no-op event (no feasible candidates) preserves the existing
    optimizer exactly as the toy path does.
    """
    candidates = feasible_expert_mutations(arm_state.genome)
    if len(candidates) > config.probe_candidate_cap:
        if config.probe_candidate_policy == "spread":
            # Deterministic per-event subset of the WHOLE feasible set, seeded
            # from (seed, step) so any re-executed event - a fresh run or a
            # mid-run resume - redraws the identical candidates.
            candidates = tuple(
                random.Random(config.seed * 1_000_003 + step).sample(
                    list(candidates), config.probe_candidate_cap
                )
            )
        else:
            candidates = candidates[: config.probe_candidate_cap]
    probe_tokens = sum(batch.numel() for batch in probe_batches)
    evaluated: list[tuple[float, int, str, str, str | None, ArchitectureGenome]] = []
    for order, (operation, layer_id, expert_id) in enumerate(candidates):
        trial_genome = deepcopy(arm_state.genome)
        if not _apply_candidate(trial_genome, operation, layer_id, expert_id, step):
            continue
        probe_model = _rebuild_mutated(model, trial_genome, config).to(device)
        if config.probe_finetune_steps > 0:
            if next_finetune_batch is None:
                raise RuntimeError(
                    "probe_finetune_steps > 0 requires a fine-tune batch source"
                )
            loss = _probe_loss_lookahead(
                probe_model, probe_batches, config, next_finetune_batch
            )
        else:
            loss = _probe_loss(probe_model, probe_batches)
        arm_state.search_flops += probe_model.active_parameter_count * probe_tokens * 2
        if config.probe_finetune_steps > 0:
            # The adaptation steps run on the FULL probed model (forward +
            # backward over its whole parameter count), not just the active
            # routing - booked per candidate so the cost identity holds.
            arm_state.search_flops += (
                config.probe_finetune_steps
                * probe_batches[0].numel()
                * 2
                * probe_model.parameter_count
            )
        evaluated.append((loss, order, operation, layer_id, expert_id, trial_genome))
        del probe_model
    if not evaluated:
        return model, optimizer
    evaluated.sort(key=lambda item: (item[0], item[1]))
    if config.probe_selection == "argmax":
        selected_index = len(evaluated) - 1
    elif config.probe_selection == "random":
        # Pool-matched random control (substrate audit, report section 9):
        # the same probed candidates at the same probe cost, chosen without
        # reference to probe loss. Seeded from (seed, step) for resume
        # determinism.
        selected_index = random.Random(config.seed * 1_000_003 + step).randrange(
            len(evaluated)
        )
    elif config.probe_selection == "quietest":
        # The section 5.32 lever applied to the probe pool, isolating the TARGET;
        # mirrors the toy evolution path so the two cannot drift. The rule fires
        # only when the probe's own pick is a prune, and then takes the least-loaded
        # expert instead - so the contrast against "argmin" differs in the prune
        # target alone, not in which operation happens.
        pool = [(item[2], item[3], item[4]) for item in evaluated]
        selected_index = 0
        if pool and pool[0][0] == "prune":
            selected_index = quietest_prune_index(
                pool,
                {
                    layer.id: [expert.id for expert in layer.experts]
                    for layer in arm_state.genome.layers
                },
                {
                    layer.id: model.routing_load[layer.index]
                    for layer in arm_state.genome.layers
                },
                fallback=0,
            )
    else:
        selected_index = 0
    probe_loss, _, operation, layer_id, expert_id, winner_genome = evaluated[
        selected_index
    ]
    before_hash = genome_hash(arm_state.genome)
    arm_state.genome = winner_genome
    arm_state.mutations += 1
    winner = _rebuild_mutated(model, arm_state.genome, config).to(device)
    arm_state.search_flops += winner.parameter_count * 4
    _prepare_model(winner, config)
    new_optimizer = torch.optim.AdamW(winner.parameters(), lr=config.learning_rate)
    entry = {
        "step": step,
        "policy": f"adaptive_probe_{config.probe_selection}",
        "operation": operation,
        "layer_id": layer_id,
        "expert_id": expert_id,
        "architecture_hash_before": before_hash,
        "architecture_hash_after": genome_hash(arm_state.genome),
        "mutation_count": arm_state.mutations,
        "probe_loss": probe_loss,
        "probe_candidates": len(evaluated),
        "probe_batch_count": len(probe_batches),
        "probe_finetune_steps": config.probe_finetune_steps,
        "probe_candidate_policy": config.probe_candidate_policy,
        "probe_selection": config.probe_selection,
    }
    if config.probe_landscape:
        # ``evaluated`` is sorted by (loss, order) ascending, so the landscape
        # is reported lowest-probe-loss first and index k is the k-th best
        # candidate under that loss ordering. ``probe_selected_index`` points
        # into it (normalized from the argmax rule's -1 tail pick), so an
        # argmin event selects index 0 and an argmax event selects the last.
        entry["probe_landscape"] = [
            {
                "operation": item[2],
                "layer_id": item[3],
                "expert_id": item[4],
                "probe_loss": item[0],
            }
            for item in evaluated
        ]
        entry["probe_selected_index"] = selected_index
    arm_state.mutation_history.append(entry)
    return winner, new_optimizer


def _random_draw(
    arm_state: _SessionState,
    config: V0SessionConfig,
    step: int,
    model: V0MoECausalLM,
) -> tuple[str, str, str | None, bool]:
    """Draw one random mutation with the random arm's exact semantics.

    Reproduces ``_random_event``'s chooser call order (layer, operation,
    expert), in-place genome mutation with the operators' internal rollback,
    and the load-based fallback grow, but does NOT append to the mutation
    history, count the mutation, or rebuild the model. Returns the applied
    (operation, layer_id, expert_id) and whether anything changed. The
    draw's changes, if any, are left IN ``arm_state.genome``; the caller
    decides to keep them (accept path) or roll them back (reject path) via
    its saved pre-draw genome. The chooser is freshly seeded
    ``config.seed + step`` exactly as the random arm's, so an accepted
    draw's chooser call history is bit-identical to the random arm's at the
    same (seed, step) whenever the feasible sets agree.
    """
    chooser = random.Random(config.seed + step)
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
    return operation, event_layer_id, applied_expert, changed


def _gated_event(
    arm_state: _SessionState,
    model: V0MoECausalLM,
    optimizer: torch.optim.Optimizer,
    config: V0SessionConfig,
    step: int,
    device: torch.device,
    probe_batches: tuple[Tensor, ...],
) -> tuple[V0MoECausalLM, torch.optim.Optimizer]:
    """Accept/reject gating of a random draw by its absolute probe harm.

    Section 5.4's fix candidate: the diagnostic showed the probe's candidate
    ORDERING is anti-correlated with downstream value on real text, so this
    event abandons ordering and uses only the probe's ABSOLUTE signal as a
    two-point contrast per event - (i) the drawn mutation probed on the
    probe batches, (ii) the no-op reference (the current model itself) probed
    on the same batches - accepting the draw unless it loses to the
    reference by more than ``config.gate_margin`` nats. The draw happens
    FIRST, while ``model.routing_load`` still reflects the latest training
    forwards (what the random arm's load-fallback sees); a draw that changes
    nothing is a recorded no-op with no probes booked. On rejection the
    draw is rolled back and the draw's own load-based grow fallback is
    applied instead (the random arm's already-exercised path, genome
    permitting; when it cannot change anything the event is a recorded
    no-op and the budget is consumed), so the mutation budget stays matched
    with the random arm and publish's budget guard never trips. Both probe
    evaluations book search FLOPs (draw: active parameters; reference: the
    current model's), matching the adaptive arm's per-evaluation booking.
    """
    before_hash = genome_hash(arm_state.genome)
    before_genome = deepcopy(arm_state.genome)
    # Draw first: the chooser and its internal load-fallback then see exactly
    # the routing statistics the random arm's draw sees (post-training-step
    # loads). Probing first would taint the loads with eval forwards.
    operation, layer_id, expert_id, changed = _random_draw(
        arm_state, config, step, model
    )
    if not changed:
        # The draw itself could not change anything; nothing to gate, and no
        # probe cost (the random arm books nothing on its no-op events).
        arm_state.mutation_history.append(
            {
                "step": step,
                "policy": "gated",
                "operation": operation,
                "layer_id": layer_id,
                "expert_id": expert_id,
                "architecture_hash_before": before_hash,
                "architecture_hash_after": genome_hash(arm_state.genome),
                "mutation_count": arm_state.mutations,
                "gate_decision": "noop",
                "gate_margin": config.gate_margin,
                "reference_probe_loss": None,
                "draw_probe_loss": None,
                "probe_candidates": 0,
                "probe_batch_count": len(probe_batches),
            }
        )
        return model, optimizer
    # With a live draw in hand, score the two-point contrast: the no-op
    # reference (the unmutated current model) and the drawn mutation, both on
    # the same probe batches. Routing loads are captured before any probe
    # forward so a rejection's fallback reads the same statistics the random
    # arm's fallback would have.
    loads_at_draw = model.routing_load
    reference_loss = _probe_loss(model, probe_batches)
    arm_state.search_flops += model.active_parameter_count * sum(
        batch.numel() for batch in probe_batches
    ) * 2
    # Probe the DRAWN mutation: rebuild ONCE from the mutated genome (the
    # in-place draw changed the genome, not the model) and score it on the
    # same probe batches as the no-op reference. The rebuilt model IS the
    # accepted winner: re-rebuilding on the accept path would consume global
    # torch RNG a second time (fresh-expert init draws from the global
    # stream), and at gate_margin=inf - where every draw is accepted - the
    # gated arm would drift off the random arm's RNG stream and break the
    # built-in control. One rebuild per accepted event keeps the streams in
    # lockstep.
    draw_model = _rebuild_mutated(model, arm_state.genome, config).to(device)
    arm_state.search_flops += draw_model.active_parameter_count * sum(
        batch.numel() for batch in probe_batches
    ) * 2
    draw_probe_loss = _probe_loss(draw_model, probe_batches)
    accepted = draw_probe_loss <= reference_loss + config.gate_margin
    if accepted:
        # The draw stays in place on arm_state.genome; the rebuilt model is
        # the event's winner (booked like every post-mutation rebuild).
        arm_state.mutations += 1
        arm_state.search_flops += draw_model.parameter_count * 4
        _prepare_model(draw_model, config)
        new_optimizer = torch.optim.AdamW(
            draw_model.parameters(), lr=config.learning_rate
        )
        arm_state.mutation_history.append(
            {
                "step": step,
                "policy": "gated",
                "operation": operation,
                "layer_id": layer_id,
                "expert_id": expert_id,
                "architecture_hash_before": before_hash,
                "architecture_hash_after": genome_hash(arm_state.genome),
                "mutation_count": arm_state.mutations,
                "gate_decision": "accept",
                "gate_margin": config.gate_margin,
                "reference_probe_loss": reference_loss,
                "draw_probe_loss": draw_probe_loss,
                "probe_candidates": 2,
                "probe_batch_count": len(probe_batches),
            }
        )
        return draw_model, new_optimizer
    # Rejected: roll the draw back and apply the draw's own load-based grow
    # fallback so the mutation budget stays matched (publish refuses
    # unmatched budgets). The fallback lives in _random_event's semantics;
    # reproduce it here against the restored genome.
    del draw_model  # the rejected candidate is discarded (memory hygiene)
    arm_state.genome = before_genome
    fallback = max(
        arm_state.genome.layers,
        key=lambda candidate: max(loads_at_draw[candidate.index]),
    )
    fallback_applied = False
    fallback_operation, fallback_layer, fallback_expert = (
        operation,
        layer_id,
        expert_id,
    )
    if len(fallback.experts) < 8:
        fallback_applied = grow_expert(arm_state.genome, fallback.id, step) is not None
        if fallback_applied:
            fallback_operation, fallback_layer, fallback_expert = (
                "grow",
                fallback.id,
                None,
            )
    if fallback_applied:
        arm_state.mutations += 1
        winner = _rebuild_mutated(model, arm_state.genome, config).to(device)
        arm_state.search_flops += winner.parameter_count * 4
        _prepare_model(winner, config)
        new_optimizer = torch.optim.AdamW(winner.parameters(), lr=config.learning_rate)
        arm_state.mutation_history.append(
            {
                "step": step,
                "policy": "gated",
                "operation": fallback_operation,
                "layer_id": fallback_layer,
                "expert_id": fallback_expert,
                "architecture_hash_before": before_hash,
                "architecture_hash_after": genome_hash(arm_state.genome),
                "mutation_count": arm_state.mutations,
                "gate_decision": "reject_fallback",
                "gate_margin": config.gate_margin,
                "reference_probe_loss": reference_loss,
                "draw_probe_loss": draw_probe_loss,
                "draw_operation": operation,
                "draw_layer_id": layer_id,
                "draw_expert_id": expert_id,
                "probe_candidates": 2,
                "probe_batch_count": len(probe_batches),
            }
        )
        return winner, new_optimizer
    # The fallback could not change anything either: record the no-op and
    # consume the budget (the random arm's chooser path would have hit the
    # same wall; its event would also be a no-op).
    arm_state.mutation_history.append(
        {
            "step": step,
            "policy": "gated",
            "operation": operation,
            "layer_id": layer_id,
            "expert_id": expert_id,
            "architecture_hash_before": before_hash,
            "architecture_hash_after": genome_hash(arm_state.genome),
            "mutation_count": arm_state.mutations,
            "gate_decision": "reject_noop",
            "gate_margin": config.gate_margin,
            "reference_probe_loss": reference_loss,
            "draw_probe_loss": draw_probe_loss,
            "draw_operation": operation,
            "draw_layer_id": layer_id,
            "draw_expert_id": expert_id,
            "probe_candidates": 2,
            "probe_batch_count": len(probe_batches),
        }
    )
    _prepare_model(model, config)  # undo _probe_loss's eval() switch
    return model, optimizer


def _heuristic_event(
    arm_state: _SessionState,
    model: V0MoECausalLM,
    optimizer: torch.optim.Optimizer,
    config: V0SessionConfig,
    step: int,
    device: torch.device,
) -> tuple[V0MoECausalLM, torch.optim.Optimizer]:
    """Apply the routing-statistics heuristic (report section 5.13).

    The section-5.11 contract's named non-probe direction: grow one expert
    at the argmax-routing-load layer, every event, deterministically. The
    only information used is the load imbalance the training forwards
    already produce - the same per-layer key (max over the layer's expert
    loads) the random arm's load-fallback reads - so the policy is fully
    determined by the training stream and consumes no chooser RNG. Among
    layers that can still grow (the operators cap layers at 16 experts),
    the highest-load layer wins; ties resolve to the lowest layer index
    (``max`` semantics, as in the fallback). If no layer can grow, the
    event is a recorded no-op. No probes are booked: search_flops only
    carries the post-grow rebuild every mutating arm books.
    """
    before_hash = genome_hash(arm_state.genome)
    growable = [
        layer
        for layer in arm_state.genome.layers
        if len(layer.experts) < 16
    ]
    if not growable:
        arm_state.mutation_history.append(
            {
                "step": step,
                "policy": "heuristic",
                "operation": "noop",
                "layer_id": None,
                "expert_id": None,
                "gate_decision": "heuristic_noop",
                "gate_margin": None,
                "architecture_hash_before": before_hash,
                "architecture_hash_after": before_hash,
                "mutation_count": arm_state.mutations,
                "probe_candidates": 0,
            }
        )
        return model, optimizer
    target = max(growable, key=lambda layer: max(model.routing_load[layer.index]))
    expert_id = grow_expert(arm_state.genome, target.id, step)
    if expert_id is None:
        # Unreachable while ``growable`` filters the operator cap, but a
        # validation rollback would land here: record the no-op.
        arm_state.mutation_history.append(
            {
                "step": step,
                "policy": "heuristic",
                "operation": "noop",
                "layer_id": target.id,
                "expert_id": None,
                "gate_decision": "heuristic_noop",
                "gate_margin": None,
                "architecture_hash_before": before_hash,
                "architecture_hash_after": genome_hash(arm_state.genome),
                "mutation_count": arm_state.mutations,
                "probe_candidates": 0,
            }
        )
        return model, optimizer
    arm_state.mutations += 1
    rebuilt = _rebuild_mutated(model, arm_state.genome, config).to(device)
    arm_state.search_flops += rebuilt.parameter_count * 4
    _prepare_model(rebuilt, config)
    new_optimizer = torch.optim.AdamW(rebuilt.parameters(), lr=config.learning_rate)
    arm_state.mutation_history.append(
        {
            "step": step,
            "policy": "heuristic",
            "operation": "grow",
            "layer_id": target.id,
            "expert_id": expert_id,
            "gate_decision": "heuristic_grow",
            "gate_margin": None,
            "architecture_hash_before": before_hash,
            "architecture_hash_after": genome_hash(arm_state.genome),
            "mutation_count": arm_state.mutations,
            "probe_candidates": 0,
        }
    )
    return rebuilt, new_optimizer


def _null_event(
    arm_state: _SessionState,
    config: V0SessionConfig,
    step: int,
) -> None:
    """Record one do-nothing structural event (report section 5.8).

    The null arm consumes its event slot but performs no mutation: no draw,
    no probe, no rebuild, no fallback - not even the random arm's chooser
    RNG draw, whose only observable effect here would be bookkeeping noise.
    The entry exists so the arm's history shows the events were consumed
    deliberately (budget-matched), and so publish and downstream analysis
    see a well-formed mutation_history.
    """
    arm_state.mutation_history.append(
        {
            "step": step,
            "policy": "null",
            "operation": "null",
            "layer_id": None,
            "expert_id": None,
            "gate_decision": "null",
            "gate_margin": None,
            "architecture_hash_before": genome_hash(arm_state.genome),
            "architecture_hash_after": genome_hash(arm_state.genome),
            "mutation_count": arm_state.mutations,
            "probe_candidates": 0,
        }
    )


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
    rebuilt = _rebuild_mutated(model, arm_state.genome, config).to(device)
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
    code_revision: str | None = None,
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
    # The gated arm shares the random arm's init identity (offset 1) by
    # design: same init seed, same training stream, same chooser seeds, so a
    # gated session at gate_margin=inf reproduces the random arm EXACTLY -
    # the built-in budget-matched random control - and at any finite margin
    # the trajectory diverges only where the gate intervenes.
    arm_seed = config.seed * 10 + {"random": 1, "adaptive": 2, "gated": 1, "null": 1, "heuristic": 1}[mode]
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
    # training-batch shape per probe batch, as declared in section 5.1. The
    # probe stream defaults to the training stream; the section-5.3 heldout
    # diagnostic probes the validation stream instead. Same dedicated
    # generator and draw discipline either way, so resume determinism and the
    # bit-identical contract are untouched - the probe batches are rebuilt
    # identically from the seed on resume.
    probe_sampler = train_sampler
    if config.data_path is not None and config.probe_data_source == "heldout":
        probe_sampler = WindowSampler(config.val_data_path, config.vocab_size)
    probe_generator = torch.Generator(device="cpu")
    probe_generator.manual_seed(config.seed ^ 0xB0B)
    if probe_sampler is not None:
        probe_batches = tuple(
            probe_sampler.draw(probe_generator, config.batch_size, config.context_size).to(device)
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

    # Long-horizon probe (R > 0): fresh training batches for candidate
    # adaptation, drawn from a dedicated throwaway generator seeded PER EVENT
    # (step-addressed) so any launch re-executing an event - a fresh run or a
    # mid-run resume - redraws that event's fine-tune batches exactly. The
    # batches never touch checkpointed state, so the bit-identical resume
    # contract holds; candidates consume their R batches in fixed enumeration
    # order, keeping the per-event sequence deterministic.
    def draw_training_batch(source: torch.Generator) -> Tensor:
        if train_sampler is not None:
            return train_sampler.draw(
                source, config.batch_size, config.context_size
            ).to(device)
        return torch.randint(
            0,
            config.vocab_size,
            (config.batch_size, config.context_size),
            generator=source,
        ).to(device)

    def finetune_batches_for(event_step: int) -> Callable[[], Tensor] | None:
        if config.probe_finetune_steps <= 0:
            return None
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed ^ 0xF1A7 ^ (event_step * 0x9E3779B9))
        return lambda: draw_training_batch(generator)

    _prepare_model(model, config)
    wall_start = time.time()
    # The backend's step boundary, where it has one. On XLA the ops of a step are
    # traced into a graph that does not run until a step is marked, so a loop that
    # never marks one traces the whole run and executes it as a single program -
    # not the loop the port rule was priced on, and on 2026-09-28 not a loop that
    # finished a development-fidelity rehearsal. A no-op on the CPU and on CUDA.
    mark_step = step_marker(device)
    for step in range(arm_state.step + 1, config.total_steps + 1):
        tokens = draw_training_batch(generator)
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
        # Marked where the probe's own measured loop marks: after the optimizer's
        # update, so the graph executes one step wide.
        mark_step()
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
                    arm_state,
                    model,
                    optimizer,
                    config,
                    step,
                    device,
                    probe_batches,
                    finetune_batches_for(step),
                )
            elif mode == "gated":
                model, optimizer = _gated_event(
                    arm_state, model, optimizer, config, step, device, probe_batches
                )
            elif mode == "null":
                _null_event(arm_state, config, step)
            elif mode == "heuristic":
                model, optimizer = _heuristic_event(
                    arm_state, model, optimizer, config, step, device
                )
            else:
                model, optimizer = _random_event(
                    arm_state, model, optimizer, config, step, device
                )
        if policy.should_save(step):
            path = _checkpoint_path(arm_dir, step)
            save_v0_checkpoint(
                path,
                model,
                optimizer,
                arm_state,
                scaler,
                code_revision=code_revision,
            )
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
    save_v0_checkpoint(
        final_path,
        model,
        optimizer,
        arm_state,
        scaler,
        code_revision=code_revision,
    )
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
        device=str(device),
    )


def arm_device_indices(arms: int, device: torch.device) -> tuple[int, ...]:
    """The CUDA index each arm should run on, one per arm when the host has them.

    Section 5.2's pair time is defined with the arms running concurrently, one
    per GPU. This returns device 0 for every arm when the host is not CUDA,
    when a pair has a single arm, or when fewer devices exist than arms - and
    the caller then trains them sequentially.
    """
    if device.type != "cuda" or arms < 2:
        return tuple(0 for _ in range(arms))
    available = torch.cuda.device_count()
    if available < arms:
        return tuple(0 for _ in range(arms))
    return tuple(range(arms))


def arm_devices(arms: int, device: torch.device) -> tuple[torch.device, ...]:
    """The device each arm trains on, for either path.

    One definition for the placement, so the sequential and concurrent paths
    cannot disagree about it - and so every arm records a device with its
    index, rather than ``cuda`` on one path and ``cuda:0`` on the other. The
    recorded device is the published evidence that a two-device pair really
    ran on two devices.
    """
    return tuple(
        torch.device(f"cuda:{index}") if device.type == "cuda" else device
        for index in arm_device_indices(arms, device)
    )


def arm_slots(arms: int, device: torch.device) -> int:
    """How many arms this placement trains at once (1, or one per arm).

    The same placement :func:`arm_devices` decides, expressed as the
    concurrency the schedule arithmetic needs - so the plan and the execution
    cannot disagree about how many arms are in flight.
    """
    return len(set(arm_device_indices(arms, device)))


def _train_arms_in_processes(
    config: V0SessionConfig,
    root: Path,
    devices: tuple[torch.device, ...],
    deadline_budget_seconds: float | None,
    *,
    code_revision: str | None = None,
) -> tuple[V0ArmResult, ...]:
    """One subprocess per arm, joined in the declared order.

    Each child inherits stdout, so its ``[mode]`` lines land in the session log
    interleaved with the others - the honest record of two arms running at
    once. A child that fails is not a missing arm: the exit status is checked
    and the arm result must exist before the pair is assembled, because a pair
    published with a silently absent arm would corrupt the comparison the run
    exists to make.
    """
    import os
    import subprocess
    import tempfile

    from act.training.arm_worker import INTERRUPTED_EXIT_CODE, arm_job

    workdir = Path(tempfile.mkdtemp(prefix=f"act-arms-seed-{config.seed}-"))
    pending: list[tuple[str, Path, subprocess.Popen[bytes]]] = []
    environment = dict(os.environ)
    # A spawned child does not inherit this process's sys.path, and the kernel
    # runs the copied tree rather than an installed package, so whatever roots
    # the parent imported ``act`` from have to be passed on explicitly.
    roots = [entry for entry in sys.path if entry and (Path(entry) / "act").is_dir()]
    if roots:
        inherited = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            [*roots, inherited] if inherited else roots
        )
    for mode, arm_device in zip(config.arms, devices, strict=True):
        job_path = workdir / f"job-{mode}.json"
        output = workdir / f"result-{mode}.json"
        job_path.write_text(
            json.dumps(
                arm_job(
                    mode=mode,
                    config=config,
                    session_root=root,
                    device=str(arm_device),
                    deadline_seconds=deadline_budget_seconds,
                    output=output,
                    # Provenance crosses the process boundary with the job: the
                    # child writes the checkpoints, so it is the child that has
                    # to record which revision wrote them.
                    code_revision=code_revision,
                ),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        pending.append(
            (
                mode,
                output,
                subprocess.Popen(
                    [sys.executable, "-m", "act.training.arm_worker", str(job_path)],
                    env=environment,
                ),
            )
        )

    failures = []
    interrupted_steps = []
    for mode, output, process in pending:
        status = process.wait()
        if status == INTERRUPTED_EXIT_CODE:
            # A cooperative deadline stop, not a crash. Both arms are waited on
            # first (each enforces the same budget, applied from its own
            # start), and the pair is then reported at the earliest step both
            # reached - which is what makes a deadline stop still publish an
            # in-progress snapshot through the fully sequential path.
            try:
                marker = json.loads(output.read_text(encoding="utf-8"))
            except (OSError, ValueError):  # pragma: no cover - defensive
                marker = {}
            interrupted_steps.append(int(marker.get("interrupted_at_step") or 0))
        elif status != 0:
            failures.append(f"{mode} exited {status}")
    if failures:
        raise RuntimeError(
            "parallel arm training failed: " + "; ".join(failures)
        )
    if interrupted_steps:
        raise V0SessionInterrupted(
            "arms stopped at the pair deadline", min(interrupted_steps)
        )

    results = []
    for mode, output, _process in pending:
        payload = json.loads(output.read_text(encoding="utf-8"))
        payload["mutation_history"] = tuple(payload["mutation_history"])
        results.append(V0ArmResult(**payload))
    return tuple(results)


def _train_arms(
    config: V0SessionConfig,
    root: Path,
    device: torch.device,
    log: Callable[[str], None] | None,
    deadline_check: Callable[[], bool] | None,
    deadline_budget_seconds: float | None,
    *,
    code_revision: str | None = None,
) -> tuple[V0ArmResult, ...]:
    """Train the pair's arms: one per device when the host has them.

    The sequential path is the fallback for a one-device host, and it is also
    what runs when the deadline cannot be expressed as a duration for a child
    process (a caller that supplies only a bare ``deadline_check``).
    """
    devices = arm_devices(len(config.arms), device)
    parallel = len(set(devices)) > 1 and (
        deadline_check is None or deadline_budget_seconds is not None
    )
    if not parallel:
        # Named per arm even here, so a sequential pair records the same kind
        # of placement evidence a concurrent one does.
        return tuple(
            _train_arm(
                mode,
                config,
                root,
                arm_device,
                log,
                deadline_check,
                code_revision=code_revision,
            )
            for mode, arm_device in zip(config.arms, devices, strict=True)
        )
    if log is not None:
        placement = ", ".join(
            f"{mode} -> {arm_device}"
            for mode, arm_device in zip(config.arms, devices, strict=True)
        )
        log(f"arms in parallel ({placement})")
    return _train_arms_in_processes(
        config,
        root,
        devices,
        deadline_budget_seconds,
        code_revision=code_revision,
    )


def run_v0_session(
    config: V0SessionConfig = V0SessionConfig(),
    *,
    checkpoint_dir: str | Path = "artifacts/v0-sessions",
    log: Callable[[str], None] | None = None,
    deadline_check: Callable[[], bool] | None = None,
    deadline_budget_seconds: float | None = None,
    code_revision: str | None = None,
) -> V0SessionResult:
    """Run (or resume) a protocol session for one seed.

    The config defaults to the fast dev shape; pass
    ``declared_session_config()`` for the pre-registered 285M target.

    ``deadline_check`` is consulted only at checkpoint boundaries; when it
    returns True the run stops cooperatively by raising
    ``V0SessionInterrupted`` with every checkpoint up to the stop step saved.

    ``deadline_budget_seconds`` is the same deadline as a duration. It is only
    needed to run the arms concurrently, because a closure over the caller's
    clock cannot cross a process boundary - so a caller that has a clock should
    pass both, and one that passes neither gets sequential arms.

    ``code_revision`` is written into every checkpoint this session saves, so a
    published checkpoint can be attributed on its own. It is optional because a
    study's checkpoints are already attributed by that study's own summary; the
    campaign passes it (``run_pair_session`` has carried one all along for the
    manifests).
    """
    if config.declared:
        validate_declared_v0_config(config.model_config())
    device = resolve_device(config.device)
    root = Path(checkpoint_dir) / f"seed-{config.seed}"
    root.mkdir(parents=True, exist_ok=True)
    arms = _train_arms(
        config,
        root,
        device,
        log,
        deadline_check=deadline_check,
        deadline_budget_seconds=deadline_budget_seconds,
        code_revision=code_revision,
    )
    return V0SessionResult(
        config=asdict(config),
        predicted_cost_ratio=predicted_cost_ratio(config),
        arms=arms,
    )
