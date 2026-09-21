"""Section 5.62's off-policy crossing cell: the design pins, before any new seed.

Every width dose and every depth rung in this family is `prefix`-policy, so both
of the newest claims rest on one policy: the edge-dominated stage 5.60 found at
the crossing, and §10.1's ratio rule (`|adaptive - random| / (random - null)`,
open where it clears 1). 5.12's 2x2 showed the two policies agree at depths 1 and
4 at the registered width; nothing has tested them where the gate crosses.

This arm runs the crossing dose itself - depth 3, width 68 - under
`probe_candidate_policy: spread`. The question is not a new dose but the
mechanism's policy-independence, so the assertions here pin the design, the
one-gene comparison against the committed prefix cell, the fresh block, the
single-cell bar, the declared ratio floor and value band, and the branches.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import null_toy_study as study  # noqa: E402

from act.evolution.study import _gate_min_wins  # noqa: E402
from act.model import model_from_genome  # noqa: E402
from act.topology import create_genome  # noqa: E402

SUMMARIES = ROOT / "artifacts" / "study-summaries"


def _parameters(depth: int, width: int) -> int:
    genome = create_genome(
        0,
        hidden_size=width,
        vocab_size=study.REGISTERED["vocab_size"],
        depth=depth,
        ffn="moe",
        expert_count=study.REGISTERED["expert_count"],
        active_experts=study.REGISTERED["active_experts"],
    )
    return sum(
        p.numel()
        for p in model_from_genome(
            genome, context_size=study.REGISTERED["sequence_length"]
        ).parameters()
    )


def test_the_cell_changes_the_policy_and_nothing_else() -> None:
    """The comparison is one gene against the committed prefix cell."""
    assert study.PROFILES[study.DOSE3_SPREAD_PROFILE]["overrides"] == {
        "depth": 3,
        "hidden_size": 68,
        "probe_candidate_policy": "spread",
    }
    prefix = study.PROFILES[study.DOSE3_SPREAD_REFERENCE]["overrides"]
    assert prefix == {"depth": 3, "hidden_size": 68}
    # Same cell, one knob: the difference between the two override sets.
    assert set(study.PROFILES[study.DOSE3_SPREAD_PROFILE]["overrides"]) - set(prefix) == {
        "probe_candidate_policy"
    }
    assert _parameters(3, 68) == 173_672
    # The policy is *not* in REGISTERED: it is the config dataclass's default, which
    # is why every arm before this one is prefix-policy without saying so.
    from act.evolution.experiment import ExperimentConfig

    assert "probe_candidate_policy" not in study.REGISTERED
    assert ExperimentConfig().probe_candidate_policy == "prefix"


def test_the_block_is_fresh_against_every_other_claim() -> None:
    used: set[int] = set()
    own = Path(study.PROFILES[study.DOSE3_SPREAD_PROFILE]["summary"]).name
    for name, spec in study.PROFILES.items():
        if name == study.DOSE3_SPREAD_PROFILE:
            continue
        used |= set(spec["seeds"])
    for path in SUMMARIES.glob("*.json"):
        if path.name == own:
            continue
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            used |= set(payload.get("seeds") or [])
    seeds = study.PROFILES[study.DOSE3_SPREAD_PROFILE]["seeds"]
    assert seeds == tuple(range(100000, 100576))
    assert not (set(seeds) & used)


def test_the_bar_is_the_single_cell_level() -> None:
    """One primary, so alpha/2 - and the profile says so rather than the CLI."""
    spec = study.PROFILES[study.DOSE3_SPREAD_PROFILE]
    assert spec["bonferroni_divisor"] == 2.0
    assert spec["report_section"] == "5.62"
    assert _gate_min_wins(576, study.ALPHA / 2) == 313


def test_the_declared_predictions_are_the_crossing_cells_own_readings() -> None:
    """The predictions are anchored to the committed cell they compare against.

    The value band is centred on the prefix cell's reading and the ratio floor is
    the boundary §10.1's table brackets, so if either committed number moves the
    assertion fails here instead of leaving a prediction that no longer refers to
    anything.
    """
    rows = json.loads((SUMMARIES / "width-d3-dose3-576.json").read_text())["per_seed"]

    def mean(left: str, right: str) -> float:
        values = [r["losses"][left] - r["losses"][right] for r in rows]
        return sum(values) / len(values)

    edge = mean("adaptive", "random")
    pool = mean("random", "null")
    assert mean("adaptive", "null") == pytest.approx(-0.127992, abs=1e-6)
    assert edge == pytest.approx(-0.359770, abs=1e-6)
    assert pool == pytest.approx(+0.231778, abs=1e-6)
    # The prefix cell is edge-dominated, which is what the off-policy cell is
    # declared to reproduce.
    assert pool > 0 and abs(edge) / pool > study.DOSE3_SPREAD_RATIO_FLOOR
    assert study.DOSE3_SPREAD_VALUE_BAND == 0.10
    assert study.DOSE3_SPREAD_REFERENCE == study.DOSE3_BELOW


def test_the_branch_table_covers_both_declarations() -> None:
    import inspect

    source = inspect.getsource(study.dose3_spread_verdict)
    for branch in ("(True, True)", "(True, False)", "(False, True)", "(False, False)"):
        assert branch in source, branch
    assert "the crossing is policy-robust" in source
    assert "the crossing is prefix-specific" in source
    # A miss has to say which of the two declarations failed, not only report one
    # boolean, which is why the ratio is printed before the branch.
    assert "edge-dominated again" in source
    assert "within {DOSE3_SPREAD_VALUE_BAND:.2f} of the prefix" in source


def test_the_run_revision_is_declared_before_the_first_shard() -> None:
    import re

    revision = study.PROFILES[study.DOSE3_SPREAD_PROFILE]["run_revision"]
    assert revision is not None, "the off-policy cell was not frozen before the run"
    assert re.fullmatch(r"[0-9a-f]{7}", revision), revision
