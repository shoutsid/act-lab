"""Small JSON-safe architecture genome used by the Python training path."""

from dataclasses import asdict, dataclass, field
import hashlib
import json


@dataclass
class ExpertGene:
    id: str
    width: int
    birth_step: int = 0


@dataclass
class LayerGene:
    id: str
    index: int
    ffn: str
    experts: list[ExpertGene] = field(default_factory=list)
    active_experts: int = 0
    enabled: bool = True
    inputs: list[str] = field(default_factory=list)


@dataclass
class ArchitectureGenome:
    version: int
    seed: int
    hidden_size: int
    vocab_size: int
    layers: list[LayerGene]
    max_params: int
    max_active_params: int

    @property
    def depth(self) -> int:
        return len(self.layers)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ArchitectureGenome":
        layers = [
            LayerGene(
                id=str(layer["id"]),
                index=int(layer["index"]),
                ffn=str(layer["ffn"]),
                experts=[
                    ExpertGene(
                        id=str(expert["id"]),
                        width=int(expert["width"]),
                        birth_step=int(expert.get("birth_step", 0)),
                    )
                    for expert in layer.get("experts", [])
                ],
                active_experts=int(layer.get("active_experts", 0)),
                enabled=bool(layer.get("enabled", True)),
                inputs=[str(source) for source in layer.get("inputs", [])],
            )
            for layer in value["layers"]
        ]
        genome = cls(
            version=int(value["version"]),
            seed=int(value["seed"]),
            hidden_size=int(value["hidden_size"]),
            vocab_size=int(value["vocab_size"]),
            layers=layers,
            max_params=int(value["max_params"]),
            max_active_params=int(value["max_active_params"]),
        )
        validation = validate_genome(genome)
        if not validation.valid:
            raise ValueError("invalid genome: " + "; ".join(validation.errors))
        return genome


def create_genome(
    seed: int,
    *,
    hidden_size: int = 64,
    vocab_size: int = 128,
    depth: int = 2,
    ffn: str = "dense",
    expert_count: int = 4,
    active_experts: int = 2,
) -> ArchitectureGenome:
    if depth < 1:
        raise ValueError("depth must be positive")
    if ffn not in {"dense", "moe"}:
        raise ValueError("ffn must be dense or moe")
    layers: list[LayerGene] = []
    for index in range(depth):
        experts = (
            [ExpertGene(id=f"L{index}E{expert}", width=hidden_size) for expert in range(expert_count)]
            if ffn == "moe"
            else []
        )
        layers.append(
            LayerGene(
                id=f"L{index}",
                index=index,
                ffn=ffn,
                experts=experts,
                active_experts=active_experts if ffn == "moe" else 0,
            )
        )
    genome = ArchitectureGenome(
        version=1,
        seed=seed,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        layers=layers,
        max_params=10**9,
        max_active_params=10**9,
    )
    validation = validate_genome(genome)
    if not validation.valid:
        raise ValueError("invalid initial genome: " + "; ".join(validation.errors))
    return genome


@dataclass(frozen=True)
class GenomeValidation:
    valid: bool
    errors: list[str]


def validate_genome(genome: ArchitectureGenome) -> GenomeValidation:
    errors: list[str] = []
    if genome.version != 1:
        errors.append(f"unsupported genome version {genome.version}")
    if genome.hidden_size <= 0 or genome.vocab_size <= 0:
        errors.append("model dimensions must be positive")
    if not genome.layers:
        errors.append("genome has no layers")
    layer_ids: set[str] = set()
    enabled = 0
    for index, layer in enumerate(genome.layers):
        if layer.index != index:
            errors.append(f"layer {layer.id} has stale index")
        if layer.id in layer_ids:
            errors.append(f"duplicate layer id {layer.id}")
        layer_ids.add(layer.id)
        enabled += int(layer.enabled)
        if layer.ffn == "dense" and layer.experts:
            errors.append(f"dense layer {layer.id} has experts")
        if layer.ffn == "moe":
            if len(layer.experts) < 2:
                errors.append(f"MoE layer {layer.id} has too few experts")
            if not 1 <= layer.active_experts <= len(layer.experts):
                errors.append(f"invalid active expert count for {layer.id}")
        elif layer.active_experts != 0:
            errors.append(f"dense layer {layer.id} has active experts")
        expert_ids: set[str] = set()
        for expert in layer.experts:
            if expert.id in expert_ids:
                errors.append(f"duplicate expert id {expert.id}")
            expert_ids.add(expert.id)
            if expert.width <= 0:
                errors.append(f"invalid expert width for {expert.id}")
    if enabled == 0:
        errors.append("genome has no enabled layers")
    layer_id_set = {layer.id for layer in genome.layers}
    for layer in genome.layers:
        if any(source not in layer_id_set for source in layer.inputs):
            errors.append(f"layer {layer.id} references an unknown input")
        if layer.id in layer.inputs:
            errors.append(f"layer {layer.id} cannot reference itself")
    if genome.max_params <= 0 or genome.max_active_params <= 0:
        errors.append("parameter ceilings must be positive")
    return GenomeValidation(valid=not errors, errors=errors)


def genome_hash(genome: ArchitectureGenome) -> str:
    return hashlib.sha256(genome.to_json().encode("utf-8")).hexdigest()[:16]