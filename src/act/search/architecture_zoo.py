"""Architecture zoo generator for static architecture search.

Produces structurally distinct candidates around a target parameter budget and
provides the structural distance metric the search uses to keep its population
spread out.

Every candidate is a first-class :class:`ArchitectureGenome`, so each one
serializes to a deterministic genome hash and can be rebuilt parameter-for-
parameter by :func:`act.model.factory.v0_moe_model_from_genome`.

Scope note: the runnable model path (``V0MoECausalLM``) implements attention
plus sparse MoE. It has no Gated DeltaNet block, so ``layer.module`` is carried
in the genome but cannot yet change what is built. The zoo therefore varies
depth, width, attention geometry, expert count, active experts and expert width;
adding GDN as a first-class gene is a separate, model-level change.
"""

import functools
from dataclasses import dataclass
from typing import Sequence

import scipy.stats

from act.model.v0_moe import V0MoEConfig, estimated_parameter_count
from act.topology.genome import (
    ArchitectureGenome,
    ExpertGene,
    LayerGene,
    validate_genome,
)


#: Default geometry of the search space. These bounds are load-bearing: every
#: search method - curated, matched or random - draws from this one grid, so a
#: method cannot gain an advantage by seeing a wider space. The curated family
#: recipes reach depth 14, 5-6 active experts and expert widths of 256-1152, all
#: of which the grid must therefore contain.
DEFAULT_DEPTHS: tuple[int, ...] = (5, 6, 8, 10, 12, 14)
DEFAULT_HIDDEN_SIZES: tuple[int, ...] = (512, 576, 640, 704, 768, 832, 896)
DEFAULT_EXPERT_COUNTS: tuple[int, ...] = (4, 6, 8, 12, 16)
DEFAULT_ACTIVE_EXPERTS: tuple[int, ...] = (2, 3, 4, 5, 6)
DEFAULT_EXPERT_WIDTHS: tuple[int, ...] = (
    128, 192, 256, 288, 320, 384, 448, 480, 512,
    576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1280, 1536,
)


@dataclass(frozen=True)
class ArchitectureCandidate:
    """A candidate architecture with its measured metadata."""

    name: str
    genome: ArchitectureGenome
    total_params: int
    active_params: int
    description: str
    family: str = "balanced"


def _heads_for(hidden_size: int) -> int:
    """Attention geometry: 64-wide heads where the width divides, else 32-wide."""
    if hidden_size % 64 == 0:
        return hidden_size // 64
    if hidden_size % 32 == 0:
        return hidden_size // 32
    return 1


def _estimate_params(
    depth: int,
    hidden_size: int,
    expert_count: int,
    active_experts: int,
    expert_width: int,
    *,
    heads: int,
    vocab_size: int = 8192,
    context_size: int = 1024,
) -> tuple[int, int]:
    """Parameter counts for a candidate without materializing the genome."""
    config = V0MoEConfig(
        vocab_size=vocab_size,
        context_size=context_size,
        hidden_size=hidden_size,
        depth=depth,
        attention_heads=heads,
        experts_per_layer=(expert_count,) * depth,
        active_experts=active_experts,
        expert_width=expert_width,
    )
    return estimated_parameter_count(config)


def make_moe_genome(
    *,
    seed: int,
    depth: int,
    hidden_size: int,
    expert_count: int,
    active_experts: int,
    expert_width: int,
    heads: int | None = None,
    vocab_size: int = 8192,
    context_size: int = 1024,
    module_pattern: Sequence[str] | None = None,
) -> ArchitectureGenome:
    """Build a uniform sparse-MoE genome.

    ``expert_width`` is independent of ``hidden_size`` so that narrow-expert and
    wide-expert designs are representable at a fixed total budget.
    ``module_pattern`` is one sequence-mixer choice per layer; omitted means
    every layer is attention.
    """
    heads = heads or _heads_for(hidden_size)
    kv_heads = heads
    if module_pattern is not None and len(module_pattern) != depth:
        raise ValueError("module_pattern must carry one entry per layer")
    layers = [
        LayerGene(
            id=f"L{index}",
            index=index,
            ffn="moe",
            experts=[ExpertGene(id=f"L{index}E{e}", width=expert_width) for e in range(expert_count)],
            active_experts=active_experts,
            ffn_width=expert_width,
            heads=heads,
            kv_heads=kv_heads,
            module=module_pattern[index] if module_pattern else "attention",
        )
        for index in range(depth)
    ]
    genome = ArchitectureGenome(
        version=1,
        seed=seed,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        layers=layers,
        max_params=10**9,
        max_active_params=10**9,
        context_size=context_size,
    )
    validation = validate_genome(genome)
    if not validation.valid:
        raise ValueError(f"invalid genome: {'; '.join(validation.errors)}")
    return genome


# Backwards-compatible alias: scripts and tests predating the rename.
_make_moe_genome = make_moe_genome


#: Family recipes spanning depth, width, attention geometry and expert structure.
#: Each entry is (family, depth, hidden_size, expert_count, active_experts, expert_width).
#: ``expert_width`` is ``None`` to mean "match hidden_size".
_FAMILY_RECIPES: tuple[tuple[str, int, int, int, int, int | None], ...] = (
    ("deep_narrow", 14, 512, 8, 2, None),
    ("deep_narrow", 12, 576, 8, 2, None),
    ("shallow_wide", 6, 896, 8, 2, None),
    ("shallow_wide", 5, 896, 8, 2, 512),
    ("balanced", 8, 768, 8, 2, None),
    ("balanced", 10, 640, 8, 2, None),
    ("many_experts", 8, 640, 16, 2, None),
    ("many_experts", 6, 704, 16, 2, 448),
    ("few_experts_broad", 10, 768, 4, 4, None),
    ("few_experts_broad", 8, 832, 4, 4, 512),
    ("narrow_experts", 8, 896, 12, 4, 256),
    ("narrow_experts", 10, 768, 12, 4, 288),
    ("wide_experts", 10, 576, 4, 2, 1024),
    ("wide_experts", 12, 512, 4, 2, 1152),
    ("high_active", 10, 640, 6, 6, None),
    ("high_active", 12, 576, 8, 5, None),
)


def generate_architecture_zoo(
    target_params: int = 100_000_000,
    tolerance: float = 0.3,
    seed: int = 42,
    *,
    context_size: int = 1024,
    vocab_size: int = 8192,
    limit: int = 14,
    min_distance: float = 0.05,
) -> list[ArchitectureCandidate]:
    """Generate a structurally diverse zoo around a target parameter budget.

    Candidates outside ``tolerance`` of ``target_params`` are dropped, then a
    greedy pass keeps the first candidate of each family and admits later ones
    only when they sit at least ``min_distance`` from everything already kept.
    The floor is what stops the zoo collapsing into 16 near-copies of one design.
    """
    candidates: list[ArchitectureCandidate] = []
    for family, depth, hidden_size, expert_count, active_experts, expert_width in _FAMILY_RECIPES:
        width = expert_width or hidden_size
        heads = _heads_for(hidden_size)
        try:
            total, active = _estimate_params(
                depth,
                hidden_size,
                expert_count,
                active_experts,
                width,
                heads=heads,
                vocab_size=vocab_size,
                context_size=context_size,
            )
        except ValueError:
            continue
        if abs(total - target_params) / target_params > tolerance:
            continue
        genome = make_moe_genome(
            seed=seed,
            depth=depth,
            hidden_size=hidden_size,
            expert_count=expert_count,
            active_experts=active_experts,
            expert_width=width,
            heads=heads,
            vocab_size=vocab_size,
            context_size=context_size,
        )
        name = f"{family}_D{depth}_H{hidden_size}_E{expert_count}_A{active_experts}_W{width}"
        candidates.append(
            ArchitectureCandidate(
                name=name,
                genome=genome,
                total_params=total,
                active_params=active,
                family=family,
                description=(
                    f"family={family} depth={depth} hidden={hidden_size} heads={heads} "
                    f"experts={expert_count} active={active_experts} expert_width={width}"
                ),
            )
        )

    # Largest designs first, so the greedy pass keeps the most distinct shapes.
    candidates.sort(key=lambda item: item.total_params, reverse=True)

    selected: list[ArchitectureCandidate] = []
    for candidate in candidates:
        if len(selected) >= limit:
            break
        if all(
            architecture_distance(candidate.genome, kept.genome) >= min_distance for kept in selected
        ):
            selected.append(candidate)
    if not selected and candidates:
        selected = [candidates[0]]

    selected.sort(key=lambda item: item.name)
    return selected


def _module_edit_distance(a: ArchitectureGenome, b: ArchitectureGenome) -> float:
    """Normalized per-layer module-type mismatch over the aligned layer prefix.

    A position present in one genome and absent from the other counts as a
    mismatch, so pure depth differences also register as module-structure
    differences.
    """
    span = max(len(a.layers), len(b.layers))
    if span == 0:
        return 0.0
    mismatches = 0
    for index in range(span):
        left = a.layers[index].module if index < len(a.layers) else None
        right = b.layers[index].module if index < len(b.layers) else None
        if left != right:
            mismatches += 1
    return mismatches / span


def architecture_distance(a: ArchitectureGenome, b: ArchitectureGenome) -> float:
    """Structural distance between two architectures, in ``[0, 1]``.

    Zero means the two genomes describe the same structure. The components are
    depth, hidden width, module-type layout, expert count, active experts,
    expert width and attention geometry, each normalized to ``[0, 1]`` before a
    weighted sum.
    """
    depth_diff = abs(a.depth - b.depth) / max(a.depth, b.depth, 1)
    hidden_diff = abs(a.hidden_size - b.hidden_size) / max(a.hidden_size, b.hidden_size, 1)

    moe_a = [layer for layer in a.layers if layer.ffn == "moe"]
    moe_b = [layer for layer in b.layers if layer.ffn == "moe"]

    def _mean(values: Sequence[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _relative(left: float, right: float) -> float:
        if left == 0.0 and right == 0.0:
            return 0.0
        return abs(left - right) / max(left, right, 1.0)

    expert_diff = _relative(
        _mean([len(layer.experts) for layer in moe_a]),
        _mean([len(layer.experts) for layer in moe_b]),
    )
    active_diff = _relative(
        _mean([layer.active_experts for layer in moe_a]),
        _mean([layer.active_experts for layer in moe_b]),
    )
    width_diff = _relative(
        _mean([layer.experts[0].width for layer in moe_a if layer.experts]),
        _mean([layer.experts[0].width for layer in moe_b if layer.experts]),
    )
    head_diff = _relative(
        _mean([layer.heads for layer in a.layers]),
        _mean([layer.heads for layer in b.layers]),
    )

    distance = (
        0.20 * depth_diff
        + 0.20 * hidden_diff
        + 0.15 * _module_edit_distance(a, b)
        + 0.15 * expert_diff
        + 0.10 * active_diff
        + 0.10 * width_diff
        + 0.10 * head_diff
    )
    return min(distance, 1.0)


#: Named sequence-mixer layouts. ``gdn_3to1`` is the Qwen3-Next-style schedule, kept
#: as one candidate among several rather than as an assumed optimum (brief
#: section 14), and ``att_3to1`` is its inverse as an explicit control.
MODULE_PATTERNS: dict[str, callable] = {
    "all_attention": lambda depth: ["attention"] * depth,
    "all_gdn": lambda depth: ["gdn"] * depth,
    "gdn_3to1": lambda depth: ["gdn" if index % 4 != 3 else "attention" for index in range(depth)],
    "att_3to1": lambda depth: ["attention" if index % 4 != 3 else "gdn" for index in range(depth)],
    "alternating": lambda depth: ["gdn" if index % 2 == 0 else "attention" for index in range(depth)],
    "gdn_att_first": lambda depth: ["attention"] * (depth // 2) + ["gdn"] * (depth - depth // 2),
    "gdn_att_last": lambda depth: ["gdn"] * (depth // 2) + ["attention"] * (depth - depth // 2),
    "all_hybrid": lambda depth: ["hybrid"] * depth,
    "hybrid_quarter": lambda depth: [
        "hybrid" if index % 4 == 0 else "gdn" for index in range(depth)
    ],
}


def generate_module_pattern_zoo(
    *,
    depth: int = 8,
    hidden_size: int = 768,
    expert_count: int = 8,
    active_experts: int = 2,
    expert_width: int | None = None,
    seed: int = 42,
    context_size: int = 1024,
    vocab_size: int = 8192,
    patterns: Sequence[str] | None = None,
) -> list[ArchitectureCandidate]:
    """Hold the block geometry fixed and vary only the sequence-mixer layout.

    Everything except ``layer.module`` is constant across the returned zoo, so a
    difference between two candidates is attributable to the mixer pattern. That
    is what makes module type testable as a gene on its own (brief section 14).
    """
    from act.model.factory import v0_moe_config_from_genome

    width = expert_width or hidden_size
    heads = _heads_for(hidden_size)
    names = list(patterns) if patterns else list(MODULE_PATTERNS)

    candidates: list[ArchitectureCandidate] = []
    for name in names:
        pattern = MODULE_PATTERNS[name](depth)
        genome = make_moe_genome(
            seed=seed,
            depth=depth,
            hidden_size=hidden_size,
            expert_count=expert_count,
            active_experts=active_experts,
            expert_width=width,
            heads=heads,
            vocab_size=vocab_size,
            context_size=context_size,
            module_pattern=pattern,
        )
        config = v0_moe_config_from_genome(genome, honor_module_types=True)
        total, active = estimated_parameter_count(config)
        candidates.append(
            ArchitectureCandidate(
                name=f"mixer_{name}_D{depth}_H{hidden_size}",
                genome=genome,
                total_params=total,
                active_params=active,
                family="module_pattern",
                description=f"pattern={''.join(p[0] for p in pattern)} depth={depth} hidden={hidden_size}",
            )
        )
    candidates.sort(key=lambda item: item.name)
    return candidates


SearchCell = tuple[int, int, int, int, int, int, int, int]
"""One point in the architecture search space.

``(total_params, active_params, depth, hidden_size, expert_count,
active_experts, expert_width, heads)``.
"""


def enumerate_search_space(
    *,
    context_size: int = 1024,
    vocab_size: int = 8192,
    depths: Sequence[int] = DEFAULT_DEPTHS,
    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
    expert_counts: Sequence[int] = DEFAULT_EXPERT_COUNTS,
    active_experts_options: Sequence[int] = DEFAULT_ACTIVE_EXPERTS,
    expert_widths: Sequence[int] = DEFAULT_EXPERT_WIDTHS,
    max_total_params: int = 10**12,
) -> list[SearchCell]:
    """Every valid cell the search may draw from.

    This one grid is the search space. Uniform random search samples exactly it,
    the curated zoo selects from it, and no method can differ from another by
    secretly seeing a wider space.

    The grid is a pure function of its arguments, so it is memoised: a 200-seed
    budget scan used to re-enumerate the same 19,320 cells for every seed (386,400
    parameter counts for 20 seeds, measured), which is what made
    ``verify_record.py``'s budget check 20 s of its 60. Callers receive a fresh
    list each time, so a caller mutating its copy cannot poison the next.
    """
    return list(
        _enumerate_cells_cached(
            context_size,
            vocab_size,
            tuple(depths),
            tuple(hidden_sizes),
            tuple(expert_counts),
            tuple(active_experts_options),
            tuple(expert_widths),
            max_total_params,
        )
    )


@functools.lru_cache(maxsize=32)
def _enumerate_cells_cached(
    context_size: int,
    vocab_size: int,
    depths: tuple[int, ...],
    hidden_sizes: tuple[int, ...],
    expert_counts: tuple[int, ...],
    active_experts_options: tuple[int, ...],
    expert_widths: tuple[int, ...],
    max_total_params: int,
) -> tuple[SearchCell, ...]:
    return tuple(
        _enumerate_cells(
            context_size=context_size,
            vocab_size=vocab_size,
            depths=depths,
            hidden_sizes=hidden_sizes,
            expert_counts=expert_counts,
            active_experts_options=active_experts_options,
            expert_widths=expert_widths,
            max_total_params=max_total_params,
        )
    )


def candidate_from_cell(
    cell: SearchCell,
    *,
    seed: int = 42,
    context_size: int = 1024,
    vocab_size: int = 8192,
    name_prefix: str = "",
    family: str = "",
    name_includes_heads: bool = False,
) -> ArchitectureCandidate:
    """Materialize a search-space cell into a genome and its measured costs.

    ``name_includes_heads`` appends the head count to the candidate name. It is
    off by default because names are ledger keys and every existing design's names
    encode only genes that vary within it; it is required when a design enumerates
    more than one head count per (depth, hidden, experts, width), where two
    candidates would otherwise share a name and silently collide in the ledger.
    """
    total, active, depth, hidden_size, expert_count, active_experts, width, heads = cell
    genome = make_moe_genome(
        seed=seed,
        depth=depth,
        hidden_size=hidden_size,
        expert_count=expert_count,
        active_experts=active_experts,
        expert_width=width,
        heads=heads,
        vocab_size=vocab_size,
        context_size=context_size,
    )
    # Heads are appended only when asked for, so a design that does not vary them
    # keeps exactly the names it had before this parameter existed.
    head_suffix = f"_Hd{heads}" if name_includes_heads else ""
    return ArchitectureCandidate(
        name=(
            f"{name_prefix}D{depth}_H{hidden_size}_E{expert_count}"
            f"_A{active_experts}_W{width}{head_suffix}"
        ),
        genome=genome,
        total_params=total,
        active_params=active,
        family=family,
        description=(
            f"depth={depth} hidden={hidden_size} heads={heads} experts={expert_count} "
            f"active={active_experts} expert_width={width}"
        ),
    )


def cell_training_flops_per_token(cell: tuple, context_size: int) -> float:
    """Training FLOPs/token for a grid cell.

    Identical to :func:`act.search.architecture_eval.estimate_flops_per_token`:
    ``3 * (2 * active + 4 * depth * seq * hidden)``. Kept here as a pure function
    of the grid tuple so the cell search does not build a config per candidate.
    """
    _, active, depth, hidden_size = cell[0], cell[1], cell[2], cell[3]
    return 3.0 * (2.0 * active + 4.0 * depth * context_size * hidden_size)


def _head_counts(hidden_size: int, head_counts: Sequence[int] | None) -> list[int]:
    """Head counts available for a hidden size.

    ``None`` reproduces the model family's own rule (``_heads_for``), so a design
    that does not ask for head variation is enumerated exactly as before. Passing
    counts enumerates those that divide the hidden size, and the point of doing so
    is that **head count is free**: attention parameters are ``4 * hidden^2``
    however that width is divided, so two head counts at the same hidden size cost
    identically in both total and active parameters. That is what makes head count
    separable from residual-stream width, with which it is otherwise the same knob.

    The option is a head *count* rather than a head width because a crossing needs
    levels that exist at every hidden size, and a width does not give that: 64-wide
    heads are 8 heads at hidden 512 but 14 at hidden 896, so a width-crossed design
    would have no shared level and no factorial. A count does transfer, provided it
    divides each hidden size in the design.
    """
    if head_counts is None:
        return [_heads_for(hidden_size)]
    counts = {count for count in head_counts if count > 0 and hidden_size % count == 0}
    return sorted(counts) or [_heads_for(hidden_size)]


def _enumerate_cells(
    *, context_size: int, vocab_size: int,
    depths: Sequence[int], hidden_sizes: Sequence[int],
    expert_counts: Sequence[int], active_experts_options: Sequence[int],
    expert_widths: Sequence[int], max_total_params: int,
    head_counts: Sequence[int] | None = None,
) -> list[SearchCell]:
    """Enumerate ``(total, active, depth, hidden, experts, active, width, heads)``.

    Expert widths are absolute rather than ratios of the hidden size, and the
    hidden sizes are always included as widths, so ``expert_width == hidden_size``
    is always reachable. ``max_total_params`` exists so a matched design cannot
    hand back a candidate the development GPU cannot hold.

    ``head_counts`` defaults to ``None``, which is the single head count the model
    family would pick anyway; passing counts enumerates the ones that fit instead,
    at identical parameter cost (see :func:`_head_counts`).
    """
    widths = sorted(set(hidden_sizes) | set(expert_widths))
    cells: list[SearchCell] = []
    for depth in depths:
        for hidden_size in hidden_sizes:
            for heads in _head_counts(hidden_size, head_counts):
                if heads < 1 or hidden_size % heads:
                    continue
                for expert_count in expert_counts:
                    for active_experts in active_experts_options:
                        if active_experts > expert_count:
                            continue
                        for width in widths:
                            if width < 16:
                                continue
                            try:
                                total, active = _estimate_params(
                                    depth, hidden_size, expert_count, active_experts, width,
                                    heads=heads, vocab_size=vocab_size, context_size=context_size,
                                )
                            except ValueError:
                                continue
                            if total > max_total_params:
                                continue
                            cells.append(
                                (total, active, depth, hidden_size, expert_count, active_experts, width, heads)
                            )
    return cells


def _stratify(ordered: Sequence[tuple], bins: int) -> list[list[tuple]]:
    """Split a sorted sequence into ``bins`` contiguous chunks of near-equal size.

    Used to guarantee a matched design covers the whole varied axis rather than
    whatever end a greedy sweep happens to reach first.

    ``bins`` is clamped to the number of cells. Without that, a band holding
    fewer cells than slots emits overlapping chunks and the same cell can be
    selected into two bins - which surfaces as a zoo containing the same
    architecture twice.
    """
    if not ordered:
        return [[]]
    bins = max(1, min(bins, len(ordered)))
    if bins == 1:
        return [list(ordered)]
    span = len(ordered)
    return [
        list(ordered[index * span // bins : (index + 1) * span // bins])
        for index in range(bins)
    ]


def _cell_matched_value(cell: tuple, match_on: str, context_size: int) -> float:
    if match_on == "total":
        return float(cell[0])
    if match_on == "active":
        return float(cell[1])
    if match_on == "flops":
        return cell_training_flops_per_token(cell, context_size)
    raise ValueError(f"unsupported match_on {match_on!r}")


def _cell_varied_value(cell: tuple, match_on: str) -> float:
    """The axis left free to spread: active parameters when total is matched,
    total parameters when the per-token budget is matched."""
    return float(cell[1] if match_on == "total" else cell[0])


def generate_matched_zoo(
    *,
    target: float = 100_000_000,
    match_on: str = "total",
    tolerance: float = 0.03,
    seed: int = 42,
    context_size: int = 1024,
    vocab_size: int = 8192,
    limit: int = 8,
    min_distance: float = 0.05,
    max_total_params: int = 150_000_000,
    depths: Sequence[int] = DEFAULT_DEPTHS,
    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
    expert_counts: Sequence[int] = DEFAULT_EXPERT_COUNTS,
    active_experts_options: Sequence[int] = DEFAULT_ACTIVE_EXPERTS,
    expert_widths: Sequence[int] = DEFAULT_EXPERT_WIDTHS,
) -> list[ArchitectureCandidate]:
    """Build a zoo matched on one cost axis and spread on the other.

    Parameter count and per-token FLOPs are different constraints, and an
    architecture can look good under one and bad under the other (brief section
    21). A zoo that varies both at once cannot tell them apart, so this holds one
    axis fixed and leaves the other deliberately wide:

    ``match_on="total"``
        total parameters matched; active parameters spread. Asks whether learning
        efficiency tracks the compute a token actually spends.
    ``match_on="active"``
        per-token active parameters matched; total parameters spread.
    ``match_on="flops"``
        training FLOPs/token matched; total parameters spread. The most faithful
        "FLOP-matched" design, because it includes the attention term that a
        depth or context change moves.

    ``tolerance`` is deliberately tight (3% by default): looser than that and the
    "matched" axis stops being matched. ``target`` is expressed in the units of
    the matched axis - parameters for ``total``/``active``, FLOPs/token for
    ``flops``.
    """
    if match_on not in {"total", "active", "flops"}:
        raise ValueError("match_on must be 'total', 'active' or 'flops'")

    cells = _enumerate_cells(
        context_size=context_size, vocab_size=vocab_size, depths=depths,
        hidden_sizes=hidden_sizes, expert_counts=expert_counts,
        active_experts_options=active_experts_options, expert_widths=expert_widths,
        max_total_params=max_total_params,
    )

    matched = [
        cell
        for cell in cells
        if abs(_cell_matched_value(cell, match_on, context_size) - target) / target <= tolerance
    ]
    if not matched:
        raise ValueError(
            f"no configuration within {tolerance:.1%} of {target:,.0f} on {match_on}"
        )
    matched.sort(key=lambda cell: _cell_varied_value(cell, match_on))

    def build(cell) -> ArchitectureCandidate:
        return candidate_from_cell(
            cell,
            seed=seed,
            context_size=context_size,
            vocab_size=vocab_size,
            name_prefix=f"m{match_on[:3]}_",
            family=f"matched_{match_on}",
        )

    # Stratify the varied axis into one bin per slot and take the most
    # structurally distinct cell from each. Filling slots by walking the band
    # from one end does not work: the diversity floor rejects most neighbours,
    # so the first `limit` survivors all come from the same end of the axis and
    # the design silently covers a fraction of the range it claims to spread.
    # At 78M +/-5% the band spans 18.0M..81.8M active parameters, and an
    # end-first sweep returned 18.0M..37.8M - half the axis, including every
    # dense configuration. Binning guarantees the extremes are represented.
    selected: list[ArchitectureCandidate] = []
    for chunk in _stratify(matched, limit):
        best: ArchitectureCandidate | None = None
        best_score = -1.0
        for cell in chunk:
            candidate = build(cell)
            score = min(
                (
                    architecture_distance(candidate.genome, kept.genome)
                    for kept in selected
                ),
                default=1.0,
            )
            if score > best_score:
                best, best_score = candidate, score
        if best is not None:
            selected.append(best)

    selected.sort(key=lambda item: _cell_varied_value(
        (item.total_params, item.active_params, item.genome.depth, item.genome.hidden_size), match_on
    ))
    return selected


#: Which element of a :data:`SearchCell` each selectable gene occupies.
#: Structural genes a crossed design can name, as indices into a search cell.
#: ``attention_heads`` is here because the model family couples it to the hidden
#: size (``heads = hidden / 64``), which makes an attribution to either one
#: unidentifiable until the two are crossed - and head count costs no parameters,
#: so the two readings imply opposite advice about where to spend a budget.
CELL_FIELDS: dict[str, int] = {
    "depth": 2,
    "hidden_size": 3,
    "expert_count": 4,
    "active_experts": 5,
    "expert_width": 6,
    "attention_heads": 7,
}


def generate_crossed_zoo(
    *,
    target_params: int = 78_000_000,
    tolerance: float = 0.05,
    factors: tuple[str, str] = ("depth", "hidden_size"),
    replicates: int = 2,
    seed: int = 42,
    context_size: int = 1024,
    vocab_size: int = 8192,
    max_total_params: int = 150_000_000,
    depths: Sequence[int] = DEFAULT_DEPTHS,
    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
    expert_counts: Sequence[int] = DEFAULT_EXPERT_COUNTS,
    active_experts_options: Sequence[int] = DEFAULT_ACTIVE_EXPERTS,
    expert_widths: Sequence[int] = DEFAULT_EXPERT_WIDTHS,
    head_counts: Sequence[int] | None = None,
) -> list[ArchitectureCandidate]:
    """Build a zoo that *crosses* two genes at a matched total-parameter budget.

    A matched design answers an attribution question, and the answer is only as
good as the design's ability to separate the genes it claims to compare.
    ``generate_matched_zoo`` spreads one axis and stratifies it, which does not
    control the *other* structural genes: at 78M the matched-total zoo's hidden
    size ends up correlated with expert width at rho = -0.74, because holding the
    budget fixed while widening the residual stream forces experts narrower.
    Those two are then one finding seen from both ends, and no amount of extra
    seeding separates them.

    Crossing fixes that where the space allows it. The two named factors are
    given ``2 levels x replicates cells`` each in a full factorial, so they are
    balanced by construction (rho = 0 exactly for two balanced levels), and the
    budget is paid for by whichever genes are left free - at a fixed depth, expert
    width and count absorb a hidden-size change completely. The genes that absorb
    are *not* independent of the crossed factors and the zoo cannot make them so;
    what it guarantees is that the crossed pair is unconfounded.

    The levels are chosen to maximize the *product* of the two spreads, subject
    to every level combination having at least ``replicates`` cells. Maximizing
    one factor alone picks whichever levels leave the other factor the most
    room: on the depth x hidden-size crossing that selects depth 5 against 6,
    the two most similar depths in the grid, because they admit the widest
    expert-width spread. Spread on both crossed genes is the entire point, so
    the objective has to contain both.
    """
    if len(factors) != 2 or len(set(factors)) != 2:
        raise ValueError("factors must be two distinct gene names")
    for factor in factors:
        if factor not in CELL_FIELDS:
            raise ValueError(
                f"unknown gene {factor!r}; choose from {sorted(CELL_FIELDS)}"
            )
    index_a, index_b = (CELL_FIELDS[factor] for factor in factors)

    cells = _enumerate_cells(
        context_size=context_size, vocab_size=vocab_size, depths=depths,
        hidden_sizes=hidden_sizes, expert_counts=expert_counts,
        active_experts_options=active_experts_options, expert_widths=expert_widths,
        max_total_params=max_total_params, head_counts=head_counts,
    )
    # A head suffix is needed only when two cells share every gene that the name
    # encodes and differ in head count - that is the collision case, and it is what
    # a head-varying design asks for. Testing whether the *grid* contains more than
    # one head count is too weak: every grid spanning two hidden sizes does, so it
    # would rename every existing design's candidates.
    name_keys: dict[tuple, set[int]] = {}
    for cell in cells:
        key = (cell[2], cell[3], cell[4], cell[5], cell[6])
        name_keys.setdefault(key, set()).add(cell[7])
    varies_heads = any(len(heads) > 1 for heads in name_keys.values())
    band = [
        cell for cell in cells
        if abs(cell[0] - target_params) / target_params <= tolerance
    ]
    if not band:
        raise ValueError(
            f"no configuration within {tolerance:.1%} of {target_params:,.0f} total params"
        )

    by_levels: dict[tuple[int, int], list[tuple]] = {}
    for cell in band:
        by_levels.setdefault((cell[index_a], cell[index_b]), []).append(cell)

    domain_a = sorted({level[0] for level in by_levels})
    domain_b = sorted({level[1] for level in by_levels})

    def full_grid(low_a: int, high_a: int, low_b: int, high_b: int) -> bool:
        return all(
            len(by_levels.get((a, b), ())) >= replicates
            for a in (low_a, high_a)
            for b in (low_b, high_b)
        )

    # Widen the second factor as far as the band and the replication floor allow.
    chosen: tuple[int, int, int, int] | None = None
    best_spread = 0.0
    for low_a in domain_a:
        for high_a in domain_a:
            if high_a == low_a:
                continue
            for low_b in domain_b:
                for high_b in domain_b:
                    if high_b == low_b:
                        continue
                    if not full_grid(low_a, high_a, low_b, high_b):
                        continue
                    spread = (high_a / low_a) * (high_b / low_b)
                    if spread > best_spread:
                        chosen, best_spread = (low_a, high_a, low_b, high_b), spread
    if chosen is None:
        raise ValueError(
            f"no full {factors[0]} x {factors[1]} crossing with {replicates} "
            f"replicates exists within {tolerance:.1%} of {target_params:,.0f}"
        )
    low_a, high_a, low_b, high_b = chosen

    names_a = {low_a: "lo", high_a: "hi"}
    names_b = {low_b: "lo", high_b: "hi"}

    def build(cell: tuple) -> ArchitectureCandidate:
        return candidate_from_cell(
            cell,
            seed=seed,
            context_size=context_size,
            vocab_size=vocab_size,
            name_prefix=f"x{names_a[cell[index_a]]}{names_b[cell[index_b]]}_",
            family=f"crossed_{factors[0]}_{factors[1]}",
            name_includes_heads=varies_heads,
        )

    selected: list[ArchitectureCandidate] = []
    for a in (low_a, high_a):
        for b in (low_b, high_b):
            available = by_levels[(a, b)]
            for _ in range(replicates):
                best: ArchitectureCandidate | None = None
                best_score = -1.0
                for cell in available:
                    candidate = build(cell)
                    if candidate.name in {kept.name for kept in selected}:
                        continue
                    # Distinctness is measured against every cell already chosen,
                    # including the other replicates of this same level
                    # combination, so a replicate cannot be a near-copy.
                    score = min(
                        (
                            architecture_distance(candidate.genome, kept.genome)
                            for kept in selected
                        ),
                        default=1.0,
                    )
                    if score > best_score:
                        best, best_score = candidate, score
                if best is not None:
                    selected.append(best)

    selected.sort(key=lambda item: (item.genome.depth, item.genome.hidden_size, item.name))
    return selected


def generate_head_paired_zoo(
    *,
    target_params: int = 78_000_000,
    tolerance: float = 0.05,
    head_counts: Sequence[int] = (8, 16),
    pairs: int = 2,
    seed: int = 42,
    context_size: int = 1024,
    vocab_size: int = 8192,
    max_total_params: int = 150_000_000,
    depths: Sequence[int] = DEFAULT_DEPTHS,
    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
    expert_counts: Sequence[int] = DEFAULT_EXPERT_COUNTS,
    active_experts_options: Sequence[int] = DEFAULT_ACTIVE_EXPERTS,
    expert_widths: Sequence[int] = DEFAULT_EXPERT_WIDTHS,
) -> list[ArchitectureCandidate]:
    """Cross head count against a *held-fixed* base configuration.

    Every other design in this module matches candidates by landing them in a
    parameter band, which leaves total parameters within about 5-10% of each other.
    That is the best a budget allows when the genes being compared both cost
    parameters - but head count does not. Attention parameters are ``4 * hidden^2``
    however that width is divided, so at a fixed ``(depth, hidden, experts, width)``
    two head counts cost **identically**, and the contrast can be exactly matched
    rather than approximately matched.

    This exists because the model family fixes ``heads = hidden_size / 64``, which
    makes an attribution to residual-stream width indistinguishable from an
    attribution to head count in every design the project had run. The two readings
    imply opposite advice - one costs budget, the other is free - so the confound is
    worth an exact design rather than a caveat.

    For each of ``pairs`` base configurations it emits one candidate per head
    count. The pairs are chosen to spread ``hidden_size`` as widely as the band
    allows while keeping every head count valid, so the resulting set gives the
    head contrast *exactly* matched within a pair and the width contrast
    band-matched across pairs.
    """
    counts = sorted({count for count in head_counts if count > 0})
    if len(counts) < 2:
        raise ValueError("a paired design needs at least two head counts")
    if pairs < 1:
        raise ValueError("pairs must be at least 1")

    cells = _enumerate_cells(
        context_size=context_size, vocab_size=vocab_size, depths=depths,
        hidden_sizes=hidden_sizes, expert_counts=expert_counts,
        active_experts_options=active_experts_options, expert_widths=expert_widths,
        max_total_params=max_total_params,
    )
    band = [
        cell for cell in cells
        if abs(cell[0] - target_params) / target_params <= tolerance
    ]
    # A base configuration is usable only if every head count divides its hidden
    # size; otherwise the pair would not be a pair.
    base = [cell for cell in band if all(cell[3] % count == 0 for count in counts)]
    if not base:
        raise ValueError(
            f"no configuration within {tolerance:.1%} of {target_params:,.0f} params "
            f"admits every head count in {counts}"
        )
    # Refuse to quietly drop a hidden size the caller asked for. Filtering is
    # otherwise silent, and a design that intended to contrast 512 against 896 but
    # returned 896 only would still produce a balanced-looking factorial - with no
    # width contrast in it at all - while reporting success.
    usable = {cell[3] for cell in base}
    unusable = sorted(
        size
        for size in {cell[3] for cell in band}
        if size not in usable
    )
    if unusable:
        raise ValueError(
            f"hidden sizes {unusable} are in the band but admit no head count in "
            f"{counts}; every requested head count must divide every hidden size "
            f"the design is meant to contrast"
        )

    # Choose base configurations greedily: maximize hidden-size spread first, since
    # that is the second contrast the set carries, then structural distinctness.
    ordered = sorted(base, key=lambda cell: (cell[3], cell[2], cell[4], cell[5], cell[6]))
    lowest, highest = ordered[0][3], ordered[-1][3]
    candidates = [cell for cell in ordered if cell[3] in (lowest, highest)] or ordered

    chosen: list[tuple] = []
    for _ in range(pairs):
        best: tuple | None = None
        best_score = -1.0
        for cell in candidates:
            if cell in chosen:
                continue
            genome = candidate_from_cell(
                cell, seed=seed, context_size=context_size, vocab_size=vocab_size
            ).genome
            score = min(
                (
                    architecture_distance(
                        genome,
                        candidate_from_cell(
                            kept, seed=seed, context_size=context_size,
                            vocab_size=vocab_size,
                        ).genome,
                    )
                    for kept in chosen
                ),
                default=1.0,
            )
            if score > best_score:
                best, best_score = cell, score
        if best is not None:
            chosen.append(best)
    if not chosen:
        raise ValueError("no base configuration could be selected")

    selected: list[ArchitectureCandidate] = []
    for pair_index, cell in enumerate(chosen, start=1):
        total, active, depth, hidden_size, expert_count, active_experts, width, _ = cell
        for count in counts:
            paired_cell = (
                total, active, depth, hidden_size, expert_count, active_experts, width, count
            )
            selected.append(
                candidate_from_cell(
                    paired_cell,
                    seed=seed,
                    context_size=context_size,
                    vocab_size=vocab_size,
                    name_prefix=f"hd{pair_index}_",
                    family="head_paired",
                    name_includes_heads=True,
                )
            )
    selected.sort(key=lambda item: (item.name,))
    return selected


def head_pair_matching(
    candidates: Sequence[ArchitectureCandidate], head_counts: Sequence[int]
) -> dict:
    """Report how exactly a paired head design matches on cost.

    The claim a paired design makes is stronger than a band's: within a pair the
    candidates should be *identical* in total and active parameters, because only
    head count differs and it is free. This measures that rather than assuming it.
    """
    counts = sorted({count for count in head_counts if count > 0})
    groups: dict[tuple, list[ArchitectureCandidate]] = {}
    for candidate in candidates:
        genome = candidate.genome
        key = (len(genome.layers), genome.hidden_size, len(genome.layers[0].experts),
               genome.layers[0].active_experts, genome.layers[0].experts[0].width)
        groups.setdefault(key, []).append(candidate)

    exact: list[dict] = []
    for key, members in sorted(groups.items()):
        present = sorted({member.genome.layers[0].heads for member in members})
        totals = {member.total_params for member in members}
        actives = {member.active_params for member in members}
        exact.append(
            {
                "base": {
                    "depth": key[0], "hidden_size": key[1], "expert_count": key[2],
                    "active_experts": key[3], "expert_width": key[4],
                },
                "head_counts": present,
                "complete": present == counts,
                "total_params": min(totals),
                "active_params": min(actives),
                "total_exactly_matched": len(totals) == 1,
                "active_exactly_matched": len(actives) == 1,
            }
        )
    totals = [candidate.total_params for candidate in candidates]
    return {
        "pairs": exact,
        "head_counts": counts,
        "all_complete": bool(exact) and all(entry["complete"] for entry in exact),
        "all_exactly_matched": bool(exact)
        and all(
            entry["total_exactly_matched"] and entry["active_exactly_matched"]
            for entry in exact
        ),
        "total_spread_ratio": (max(totals) / min(totals)) if totals else None,
    }


def gene_value(candidate: ArchitectureCandidate, gene: str) -> int:
    """Read one structural gene off a candidate's genome.

    The genes are read from the genome rather than from the name, so a design's
    balance can be checked without trusting that names encode structure.
    """
    genome = candidate.genome
    layers = genome.layers
    if gene == "depth":
        return len(layers)
    if gene == "hidden_size":
        return genome.hidden_size
    if gene == "attention_heads":
        return layers[0].heads if layers else 0
    if gene == "expert_count":
        return max((len(layer.experts) for layer in layers), default=0)
    if gene == "active_experts":
        return layers[0].active_experts if layers else 0
    if gene == "expert_width":
        widths = [expert.width for layer in layers for expert in layer.experts]
        return max(widths) if widths else 0
    raise ValueError(f"unknown gene {gene!r}")


def crossing_balance(
    candidates: Sequence[ArchitectureCandidate], factors: Sequence[str]
) -> dict:
    """How well a crossed zoo separates its two factors.

    The claim a crossed design makes is that ``factors[0]`` and ``factors[1]``
    are independent, so each one's association with loss is attributable. This
    reports the evidence for that claim instead of asserting it: the level
    values, how many candidates sit at each level, and the rank correlation
    between the two factors across the zoo. A balanced two-level crossing gives
    exactly 0.0, and anything else means the design leaked.
    """
    if len(factors) != 2:
        raise ValueError("crossing_balance needs exactly two factors")
    first = [gene_value(candidate, factors[0]) for candidate in candidates]
    second = [gene_value(candidate, factors[1]) for candidate in candidates]
    if len(candidates) < 2 or len(set(first)) < 2 or len(set(second)) < 2:
        return {"available": False, "reason": "a factor has no variance"}
    rho = scipy.stats.spearmanr(first, second).statistic
    counts: dict[str, dict[str, int]] = {}
    for factor, values in ((factors[0], first), (factors[1], second)):
        counts[factor] = {
            str(level): values.count(level) for level in sorted(set(values))
        }
    return {
        "available": True,
        "factors": list(factors),
        "levels": counts,
        "spearman": float(rho),
        "balanced": abs(float(rho)) < 1e-9,
        "candidates": len(candidates),
    }


def matching_quality(
    candidates: Sequence[ArchitectureCandidate], match_on: str, context_size: int = 1024
) -> dict:
    """How tightly the matched axis holds and how far the other axis stretches."""
    if match_on not in {"total", "active", "flops"}:
        raise ValueError("match_on must be 'total', 'active' or 'flops'")

    def cell_of(candidate: ArchitectureCandidate) -> tuple:
        return (
            candidate.total_params,
            candidate.active_params,
            candidate.genome.depth,
            candidate.genome.hidden_size,
        )

    matched_values = [
        _cell_matched_value(cell_of(c), match_on, context_size) for c in candidates
    ]
    varied_values = [_cell_varied_value(cell_of(c), match_on) for c in candidates]
    low, high = min(matched_values), max(matched_values)
    varied_low, varied_high = min(varied_values), max(varied_values)
    mean = sum(matched_values) / len(matched_values)
    return {
        "match_on": match_on,
        "matched_min": low,
        "matched_max": high,
        "matched_spread_ratio": high / low if low else float("inf"),
        "matched_spread_over_mean": (high - low) / mean if mean else float("inf"),
        "varied_min": varied_low,
        "varied_max": varied_high,
        "varied_spread_ratio": varied_high / varied_low if varied_low else float("inf"),
    }


def compute_distance_matrix(candidates: Sequence[ArchitectureCandidate]) -> list[list[float]]:
    """Pairwise structural distances between all candidates."""
    size = len(candidates)
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            distance = architecture_distance(candidates[i].genome, candidates[j].genome)
            matrix[i][j] = distance
            matrix[j][i] = distance
    return matrix


def min_pairwise_distance(candidates: Sequence[ArchitectureCandidate]) -> float:
    """Smallest structural distance between any two candidates (0 if identical)."""
    matrix = compute_distance_matrix(candidates)
    size = len(candidates)
    if size < 2:
        return 0.0
    return min(matrix[i][j] for i in range(size) for j in range(i + 1, size))


def print_zoo_summary(candidates: Sequence[ArchitectureCandidate]) -> None:
    """Print the zoo as a table."""
    print(f"\n{'=' * 108}")
    print(f"Architecture zoo: {len(candidates)} candidates")
    print(f"{'=' * 108}")
    print(
        f"{'Name':<42} {'Family':<16} {'Total':>12} {'Active':>12} "
        f"{'Depth':>6} {'Hidden':>7} {'E':>3} {'A':>3} {'W':>5}"
    )
    print("-" * 108)
    for candidate in candidates:
        genome = candidate.genome
        layer = genome.layers[0]
        experts = len(layer.experts) if layer.experts else 0
        width = layer.experts[0].width if layer.experts else 0
        print(
            f"{candidate.name:<42} {candidate.family:<16} {candidate.total_params:>12,} "
            f"{candidate.active_params:>12,} {genome.depth:>6} {genome.hidden_size:>7} "
            f"{experts:>3} {layer.active_experts:>3} {width:>5}"
        )
    print(f"{'=' * 108}\n")
