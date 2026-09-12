"""Hierarchical and prior-informed topology candidate ranking."""

from dataclasses import dataclass

from act.topology import ArchitectureGenome


@dataclass(frozen=True)
class SearchPrior:
    module_scores: dict[str, float]
    depth_penalty: float = 0.0


@dataclass(frozen=True)
class PriorRankedCandidate:
    layer_id: str
    module: str
    score: float
    rationale: str


def rank_module_candidates(
    genome: ArchitectureGenome,
    prior: SearchPrior,
) -> tuple[PriorRankedCandidate, ...]:
    if prior.depth_penalty < 0:
        raise ValueError("depth_penalty must be non-negative")
    if not prior.module_scores:
        raise ValueError("module_scores must not be empty")
    candidates = [
        PriorRankedCandidate(
            layer.id,
            module,
            float(prior.module_scores[module]) - prior.depth_penalty * layer.index,
            f"prior={prior.module_scores[module]:.6g}; depth_penalty={prior.depth_penalty:.6g}",
        )
        for layer in genome.layers
        for module in sorted(prior.module_scores)
    ]
    return tuple(sorted(candidates, key=lambda item: (-item.score, item.layer_id, item.module)))
