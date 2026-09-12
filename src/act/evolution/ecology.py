"""Expert ecology summaries derived from functional sketches and routing load."""

from dataclasses import dataclass

from torch import Tensor

from act.importance import cosine_similarity


@dataclass(frozen=True)
class EcologyEdge:
    left: str
    right: str
    similarity: float


@dataclass(frozen=True)
class EcologyReport:
    edges: tuple[EcologyEdge, ...]
    node_loads: dict[str, float]


def build_ecology_report(
    sketches: dict[str, Tensor],
    loads: dict[str, float],
    *,
    similarity_threshold: float = 0.5,
) -> EcologyReport:
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between -1 and 1")
    missing = set(sketches) - set(loads)
    if missing:
        raise ValueError(f"missing routing loads for {sorted(missing)}")
    ids = sorted(sketches)
    edges: list[EcologyEdge] = []
    for index, left in enumerate(ids):
        for right in ids[index + 1:]:
            similarity = cosine_similarity(sketches[left], sketches[right])
            if similarity >= similarity_threshold:
                edges.append(EcologyEdge(left, right, similarity))
    return EcologyReport(
        edges=tuple(edges),
        node_loads={expert_id: float(loads[expert_id]) for expert_id in ids},
    )
