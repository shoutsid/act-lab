import pytest

from act.evolution import SearchPrior, rank_module_candidates
from act.topology import create_genome


def test_prior_ranking_is_deterministic_and_depth_aware() -> None:
    genome = create_genome(7, hidden_size=16, depth=2, ffn="dense")
    ranked = rank_module_candidates(
        genome,
        SearchPrior({"attention": 1.0, "gdn": 0.8, "hybrid": 0.9}, depth_penalty=0.1),
    )
    assert ranked[0].module == "attention"
    assert ranked[0].layer_id == "L0"
    assert ranked[-1].layer_id == "L1"


def test_prior_ranking_validates_inputs() -> None:
    genome = create_genome(8, hidden_size=16, depth=1, ffn="dense")
    with pytest.raises(ValueError, match="depth_penalty"):
        rank_module_candidates(genome, SearchPrior({"attention": 1.0}, depth_penalty=-1))
