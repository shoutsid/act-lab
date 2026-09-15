"""Pins for the section 5.32 statistic and its stored record.

Section 5.31 registered a paired **sign test** and failed both primaries while the
means ordered exactly as predicted, at 53-54% win rates. At that rate the sign test
needs ~1396 seeds for what a mean test resolves with ~380, so section 5.31's branch
(d) is a power failure, not evidence of absence. Section 5.32 re-registers the same
question on a ~4x block with a mean-based statistic declared in advance.

Two things are pinned, and they are pinned for different reasons:

1. **The statistic itself, before any result is read.** A permutation p-value is
   only meaningful if it is reproducible and calibrated. Reproducible is checked
   directly; calibrated is checked by measurement against symmetric heavy-tailed
   noise - the case this protocol exists because of. If the permutation test ever
   became anti-conservative, every verdict in section 5.32 would be inflated, and
   nothing else in the suite would notice.
2. **The stored record,** so a failure cannot be re-read as a success later without
   the numbers moving. Section 5.31 is the cautionary case: its registered statistic
   failed while the means pointed the other way, and the only reason that is legible
   today is that both readings were stored side by side.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import prune_target_powered_study as pow_study  # noqa: E402

from act.evolution.figures_data import summaries_dir  # noqa: E402


def test_the_permutation_p_is_reproducible() -> None:
    """Same deltas, same seed -> the same p to the digit."""
    deltas = [0.1, -0.2, 0.05, -0.3, 0.4, -0.15, 0.02, -0.08, 0.3, -0.25]
    first = pow_study._permutation_p(deltas, seed=1234)
    second = pow_study._permutation_p(deltas, seed=1234)
    assert first == second
    assert 0.0 < first <= 1.0


def test_the_permutation_p_is_calibrated_on_symmetric_heavy_tailed_noise() -> None:
    """The size of the test, measured rather than assumed.

    Under a symmetric null the sign-flip permutation p is uniform, so the
    rejection rate at 0.05 should be ~0.05. The null here is a Gaussian scale
    mixture with sd 1/5 in equal halves - symmetric and heavy-tailed, the shape
    that motivated a permutation test over a t-test.

    The band is deliberately wide (the point is a gross miscalibration, not the
    third decimal): an anti-conservative statistic, which is the failure that
    would inflate every section 5.32 verdict, would land far outside it.

    Teeth, measured rather than asserted: the observed rate under this null is 0.05,
    and the same construction with a 0.5 shift on the sample gives 0.20 and with 0.8
    gives 0.38 - so the counter is live and the 0.10 band is load-bearing rather
    than vacuously satisfied.
    """
    generator = torch.Generator().manual_seed(20260915)
    replicates = 200
    size = 40
    rejections = 0
    for index in range(replicates):
        wide = torch.randint(0, 2, (size,), generator=generator, dtype=torch.float64)
        spread = 1.0 + 4.0 * wide
        sample = torch.randn(size, generator=generator, dtype=torch.float64) * spread
        if pow_study._permutation_p([float(x) for x in sample], seed=index) <= 0.05:
            rejections += 1
    rate = rejections / replicates
    # The lower bound guards a degenerate statistic (one that never rejects); it is
    # not near the observed 0.05, because with 200 replicates the 95% interval on a
    # true 0.05 is [0.024, 0.089] and a tight lower bound would be a coin flip.
    assert 0.02 <= rate <= 0.10, f"permutation test is miscalibrated: rate {rate}"


def test_the_two_statistics_can_disagree_and_both_are_reported() -> None:
    """A structured case where the mean passes and the sign test does not.

    A majority of deltas are negatives of moderate size and the minority are small
    positives, so the mean is decisively negative while the win count sits just
    under the bar. That is exactly the section 5.31 configuration - means one way,
    win rate the other - and it is reproduced here on synthetic input so the
    reporting path of `_comparison` is pinned independently of any block.
    """
    deltas = [-0.05] * 210 + [0.01] * 174
    comparison = pow_study._comparison(deltas)
    assert comparison["n"] == 384
    assert comparison["mean_delta"] < 0
    assert comparison["passes"] is True
    assert comparison["sign_test_passes"] is False
    assert comparison["the_two_statistics_disagree"] is True
    assert comparison["min_wins_for_the_sign_test"] > comparison["wins"]


def _load(name: str) -> dict:
    path = summaries_dir() / name
    if not path.exists():
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


@pytest.mark.parametrize(
    "name,depth", [("prune-target-powered-depth2-1536.json", 2),
                   ("prune-target-powered-depth3-1536.json", 3)]
)
def test_the_committed_record_recomputes_from_its_own_per_seed_block(
    name: str, depth: int
) -> None:
    """Recompute every stored comparison from the stored per-seed rows.

    Not a re-run of the block - a re-derivation, so a summary edited after the fact
    (or a statistic changed underneath it) fails here rather than reading as the
    original result.
    """
    summary = _load(name)
    assert summary["protocol"] == (
        "random-vs-quietest-vs-busiest-prune-target-permutation-powered"
    )
    assert summary["depth"] == depth
    assert len(summary["per_seed"]) == summary["declared_n"]

    def loss(label: str) -> list[float]:
        return [row["losses"][label] for row in summary["per_seed"]]

    recomputed = {
        "fused_Q_vs_R": pow_study._comparison(
            [a - b for a, b in zip(loss("Q"), loss("R"))]
        ),
        "fused_R_vs_B": pow_study._comparison(
            [a - b for a, b in zip(loss("R"), loss("B"))]
        ),
    }
    for key, value in recomputed.items():
        stored = summary["comparisons"][key]
        assert stored["n"] == value["n"]
        assert stored["wins"] == value["wins"]
        assert stored["permutation_p"] == pytest.approx(value["permutation_p"], rel=1e-12)
        assert stored["mean_delta"] == pytest.approx(value["mean_delta"], rel=1e-12)
        assert stored["passes"] is value["passes"]
