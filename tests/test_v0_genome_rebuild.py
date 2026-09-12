import torch

from act.model import v0_model_from_genome
from act.topology import create_genome


def test_v0_model_reconstructs_from_genome() -> None:
    genome = create_genome(5, hidden_size=24, vocab_size=32, depth=3, ffn="dense")
    genome.layers[1].module = "gdn"
    genome.layers[2].module = "hybrid"
    genome.context_size = 12
    model = v0_model_from_genome(genome)
    tokens = torch.randint(0, 32, (2, 12))
    assert model(tokens).shape == (2, 12, 32)
