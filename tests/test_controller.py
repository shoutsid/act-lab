import pytest

from act.evolution import AdaptiveController
from act.topology import create_genome


def test_controller_grows_overloaded_layer() -> None:
    genome = create_genome(1, hidden_size=8, vocab_size=16, depth=1, ffn="moe")
    decision = AdaptiveController(load_threshold=0.5).decide(
        genome,
        loads=((0.8, 0.1, 0.05, 0.05),),
        latest={},
    )
    assert decision.operation == "grow"
    assert decision.layer_id == "L0"


def test_controller_prunes_lowest_metric_deterministically() -> None:
    genome = create_genome(1, hidden_size=8, vocab_size=16, depth=1, ffn="moe")
    decision = AdaptiveController(metric="fisher_ema").decide(
        genome,
        loads=((0.25, 0.25, 0.25, 0.25),),
        latest={
            "L0:L0E0": {"fisher_ema": 4.0},
            "L0:L0E1": {"fisher_ema": 1.0},
            "L0:L0E2": {"fisher_ema": 2.0},
            "L0:L0E3": {"fisher_ema": 3.0},
        },
    )
    assert (decision.operation, decision.expert_id) == ("prune", "L0E1")


def test_controller_validates_policy() -> None:
    with pytest.raises(ValueError, match="metric"):
        AdaptiveController(metric="unknown")
