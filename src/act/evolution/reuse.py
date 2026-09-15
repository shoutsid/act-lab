"""Transport a discovered topology into a fresh model and retrain it.

Plan section 11 keeps topology and weights separate so that a discovered
architecture can be rebuilt with ``final topology + original initialization``,
and plan section 52 asks the question that depends on it: does the discovered
topology remain advantageous when freshly initialized and trained
independently?

This module is the minimal machinery for that question at toy fidelity. Every
Gate 3 comparison so far scores a topology *while it is trained*: an arm's
final loss contains whatever its own weights happened to learn, so no existing
result separates the structure from the run that produced it. Here a genome is
extracted from a discovery arm, rebuilt from scratch, and trained with
mutation disabled - so the genome, and not the controller, is under test.

The retraining uses ``act.training.retraining.initialize_retraining`` with the
``fresh`` policy: no weight transfer at all, which is the plan's
fresh-initialization condition rather than the rewound or teacher variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from act.evolution.experiment import (
    ExperimentConfig,
    _training_batches,
    _validation_batches,
)
from act.model import StaticMoECausalLM, model_from_genome
from act.training.retraining import InitializationReport, initialize_retraining
from act.training.runtime import resolve_device
from act.topology import ArchitectureGenome, create_genome, genome_hash

TransportMode = Literal["fresh", "rewound"]

# The expert parameters a MoE block owns, in the ``nn.Sequential`` an expert is
# built from. Used to re-point a target expert at the source expert with the
# same *id* rather than the same position.
_EXPERT_SUFFIXES = ("0.weight", "0.bias", "2.weight", "2.bias")


@dataclass(frozen=True)
class TransportResult:
    """A frozen genome's performance when rebuilt fresh and retrained."""

    final_loss: float
    parameter_count: int
    active_parameter_count: int
    training_flops: int
    architecture_hash: str
    mutation_count: int = 0
    initialization: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "final_loss": self.final_loss,
            "parameter_count": self.parameter_count,
            "active_parameter_count": self.active_parameter_count,
            "training_flops": self.training_flops,
            "architecture_hash": self.architecture_hash,
            "mutation_count": self.mutation_count,
            "initialization": dict(self.initialization),
        }


def initial_model(config: ExperimentConfig) -> StaticMoECausalLM:
    """The discovery block's initial model, rebuilt deterministically.

    Same seed and same construction call as ``run_experiment``, so this is the
    model the block starts every arm from (pinned by the transport-validity
    test: transporting an unmutated genome reproduces that arm's own run
    exactly). It is the source of *original initialization* for rewinding.
    """
    torch.manual_seed(config.seed)
    genome = create_genome(
        config.seed,
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        depth=config.depth,
        ffn="moe",
        expert_count=config.expert_count,
        active_experts=config.active_experts,
    )
    model = model_from_genome(genome, context_size=config.sequence_length)
    if not isinstance(model, StaticMoECausalLM):
        raise TypeError("the discovery block's initial model is not a static MoE")
    return model


def expert_renames(
    target: StaticMoECausalLM,
    source: StaticMoECausalLM,
) -> dict[str, str | None]:
    """Map every target expert parameter to its source by component *id*.

    A target expert whose id is absent from the source (a grown expert) maps to
    ``None`` - it has no original initialization to be rewound to, so it keeps
    the value the fresh draw gave it. Every other tensor matches by name, which
    is identity-correct here because these mutations change expert sets, not
    layer positions.
    """
    renames: dict[str, str | None] = {}
    if len(target.blocks) != len(source.blocks):
        raise ValueError("rewinding requires the same number of layers")
    for layer_index, target_block in enumerate(target.blocks):
        source_index = {
            expert_id: index
            for index, expert_id in enumerate(source.blocks[layer_index].expert_ids)
        }
        for target_expert, expert_id in enumerate(target_block.expert_ids):
            match = source_index.get(expert_id)
            for suffix in _EXPERT_SUFFIXES:
                name = f"blocks.{layer_index}.experts.{target_expert}.{suffix}"
                renames[name] = (
                    None
                    if match is None
                    else f"blocks.{layer_index}.experts.{match}.{suffix}"
                )
    return renames


def transport_genome(
    genome: ArchitectureGenome | dict[str, object],
    config: ExperimentConfig,
    *,
    initialization: TransportMode = "fresh",
    rewind_source: ExperimentConfig | None = None,
) -> TransportResult:
    """Rebuild ``genome`` and retrain it on ``config``'s seed.

    The model is built with the *same* global torch seed and the same data
    schedule the discovery block used, so transporting an unmutated genome
    reproduces that arm's own run exactly. That identity is what makes the
    measured deltas attributable to the topology rather than to the transport
    procedure (report section 5.16, secondary 1).

    ``initialization`` selects the plan section 55 treatment:

    - ``fresh``: the rebuild's own random draw, nothing transferred (plan
      section 55's model C, and section 5.16's transport).
    - ``rewound``: the same draw, then every component that existed at
      initialization is restored to its initial weights by component identity
      (plan section 55's model B, plan section 11's "final topology + original
      initialization"). Components that did not exist at initialization - grown
      experts - keep the fresh draw. On an unmutated genome, where every
      component existed at initialization, this is *identical by construction*
      to ``fresh``.

    ``rewind_source`` chooses *whose* initialization is restored, which is the
    variable that decides whether a discovered topology is paired with the
    initialization it was found with or portable to a foreign one (report
    section 5.18). It defaults to ``config`` - the block's own initial model.
    """
    if isinstance(genome, dict):
        genome = ArchitectureGenome.from_dict(genome)
    if config.steps < 1:
        raise ValueError("steps must be positive")
    if initialization not in {"fresh", "rewound"}:
        raise ValueError(f"unsupported transport initialization: {initialization}")
    if rewind_source is not None and initialization != "rewound":
        raise ValueError("rewind_source is only meaningful for rewound initialization")
    device = resolve_device(config.device)
    # Same seed and same construction call as ``run_experiment``'s initial
    # model, which is what the ``fixed`` arm trains unmutated.
    torch.manual_seed(config.seed)
    model = model_from_genome(genome, context_size=config.sequence_length)
    if not isinstance(model, StaticMoECausalLM):
        raise TypeError("transport requires a static MoE genome")
    model.to(device)
    if initialization == "rewound":
        source = initial_model(config if rewind_source is None else rewind_source)
        report: InitializationReport = initialize_retraining(
            model,
            mode="rewound",
            rewind_state=source.state_dict(),
            renames=expert_renames(model, source),
        )
    else:
        report = initialize_retraining(model, mode="fresh")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    training_flops = 0
    model.train()
    for tokens in _training_batches(config, device):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(tokens)
        loss.backward()
        optimizer.step()
        training_flops += model.parameter_count * tokens.numel() * 2
    model.eval()
    with torch.no_grad():
        validation = _validation_batches(config, device)
        final_loss = sum(float(model.loss(tokens)) for tokens in validation) / len(validation)
    return TransportResult(
        final_loss=final_loss,
        parameter_count=model.parameter_count,
        active_parameter_count=model.active_parameter_count,
        training_flops=training_flops,
        architecture_hash=genome_hash(genome),
        mutation_count=0,
        initialization={
            "mode": report.mode,
            "copied_parameters": report.copied_parameters,
            "skipped_parameters": report.skipped_parameters,
        },
    )


def transport_pair(
    genome: ArchitectureGenome | dict[str, object],
    config: ExperimentConfig,
) -> dict[str, TransportResult]:
    """Both plan-section-55 initialization treatments of one frozen genome."""
    return {
        mode: transport_genome(genome, config, initialization=mode)
        for mode in ("fresh", "rewound")
    }
