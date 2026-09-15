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

import torch

from act.evolution.experiment import (
    ExperimentConfig,
    _training_batches,
    _validation_batches,
)
from act.model import StaticMoECausalLM, model_from_genome
from act.training.retraining import InitializationReport, initialize_retraining
from act.training.runtime import resolve_device
from act.topology import ArchitectureGenome, genome_hash


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


def transport_genome(
    genome: ArchitectureGenome | dict[str, object],
    config: ExperimentConfig,
) -> TransportResult:
    """Rebuild ``genome`` with fresh weights and retrain it on ``config``'s seed.

    The model is built with the *same* global torch seed and the same data
    schedule the discovery block used, so transporting an unmutated genome
    reproduces that arm's own run exactly. That identity is what makes the
    measured deltas attributable to the topology rather than to the transport
    procedure (report section 5.16, secondary 1).
    """
    if isinstance(genome, dict):
        genome = ArchitectureGenome.from_dict(genome)
    if config.steps < 1:
        raise ValueError("steps must be positive")
    device = resolve_device(config.device)
    # Same seed and same construction call as ``run_experiment``'s initial
    # model, which is what the ``fixed`` arm trains unmutated.
    torch.manual_seed(config.seed)
    model = model_from_genome(genome, context_size=config.sequence_length)
    if not isinstance(model, StaticMoECausalLM):
        raise TypeError("transport requires a static MoE genome")
    model.to(device)
    report: InitializationReport = initialize_retraining(model, mode="fresh")
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
