"""Random architecture search: the baseline every other search method must beat.

Samples valid genomes uniformly from the architecture search space defined by
:func:`act.search.architecture_zoo.enumerate_search_space`, and does nothing
else. There is deliberately no evolution, no mutation and no topology operator
here - the brief asks for a clean baseline before any evolutionary method, and a
baseline that quietly optimises is not a baseline.

The first scientific comparison this exists to support is:

    uniformly sampled architecture  vs  manually selected architecture

under equal compute. Two properties make that comparison honest:

* **same space**: the sampler draws from the grid the zoo generator draws from,
  so neither method sees a larger or smaller space;
* **seeded and reproducible**: a run is a pure function of its seed, so a
  baseline can be re-drawn exactly.

Uniform sampling over the grid is *not* uniform over architectures - the grid is
denser in some shapes than others. That is stated rather than corrected, because
"uniform over the declared grid" is the thing a reader can reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from act.search.architecture_zoo import (
    DEFAULT_ACTIVE_EXPERTS,
    DEFAULT_DEPTHS,
    DEFAULT_EXPERT_COUNTS,
    DEFAULT_EXPERT_WIDTHS,
    DEFAULT_HIDDEN_SIZES,
    ArchitectureCandidate,
    SearchCell,
    candidate_from_cell,
    enumerate_search_space,
)


@dataclass
class RandomArchitectureSearch:
    """Uniform sampling of valid genomes from the architecture search space.

    ``target_params`` and ``tolerance`` restrict the space to the budget under
    study; ``size`` is how many candidates one draw returns. Sampling is without
    replacement, so a single draw never returns the same cell twice.
    """

    target_params: int = 100_000_000
    tolerance: float = 0.15
    size: int = 8
    seed: int = 0
    context_size: int = 1024
    vocab_size: int = 8192
    max_total_params: int = 150_000_000
    depths: Sequence[int] = DEFAULT_DEPTHS
    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES
    expert_counts: Sequence[int] = DEFAULT_EXPERT_COUNTS
    active_experts_options: Sequence[int] = DEFAULT_ACTIVE_EXPERTS
    expert_widths: Sequence[int] = DEFAULT_EXPERT_WIDTHS
    budgeted_cells: list[SearchCell] = field(default_factory=list, init=False, repr=False)

    def space(self) -> list[SearchCell]:
        """The cells this sampler may draw from, cached after the first call."""
        if not self.budgeted_cells:
            cells = enumerate_search_space(
                context_size=self.context_size,
                vocab_size=self.vocab_size,
                depths=self.depths,
                hidden_sizes=self.hidden_sizes,
                expert_counts=self.expert_counts,
                active_experts_options=self.active_experts_options,
                expert_widths=self.expert_widths,
                max_total_params=self.max_total_params,
            )
            self.budgeted_cells = [
                cell
                for cell in cells
                if abs(cell[0] - self.target_params) / self.target_params <= self.tolerance
            ]
        return self.budgeted_cells

    def sample(self, seed: int | None = None) -> list[ArchitectureCandidate]:
        """Draw ``size`` distinct budgeted architectures, seeded and reproducible."""
        cells = self.space()
        if not cells:
            raise ValueError(
                f"search space is empty within {self.tolerance:.1%} of {self.target_params:,} params"
            )
        count = min(self.size, len(cells))
        generator = torch.Generator().manual_seed(self.seed if seed is None else seed)
        chosen = torch.randperm(len(cells), generator=generator)[:count].tolist()
        picked = [cells[index] for index in chosen]
        return [
            candidate_from_cell(
                cell,
                seed=self.seed if seed is None else seed,
                context_size=self.context_size,
                vocab_size=self.vocab_size,
                name_prefix="rand_",
                family="random",
            )
            for cell in picked
        ]

    def describe_space(self) -> dict:
        """Size and cost spread of the space a draw comes from."""
        cells = self.space()
        totals = [cell[0] for cell in cells]
        actives = [cell[1] for cell in cells]
        return {
            "cells": len(cells),
            "total_min": min(totals),
            "total_max": max(totals),
            "active_min": min(actives),
            "active_max": max(actives),
            "total_spread_ratio": max(totals) / min(totals) if totals else float("inf"),
            "active_spread_ratio": max(actives) / min(actives) if actives else float("inf"),
            "target_params": self.target_params,
            "tolerance": self.tolerance,
        }
