import math

import pytest

from act.model import run_v0_smoke
from act.topology import create_genome


def test_v0_smoke_reports_finite_logits_and_loss() -> None:
    result = run_v0_smoke(
        create_genome(12, hidden_size=16, vocab_size=32, depth=2, ffn="dense"),
        sequence_length=6,
    )
    assert result.parameter_count > 0
    assert result.logits_shape == (1, 5, 32)
    assert math.isfinite(result.loss)


def test_v0_smoke_rejects_invalid_bounds() -> None:
    genome = create_genome(13, hidden_size=16, vocab_size=32, depth=1, ffn="dense")
    with pytest.raises(ValueError, match="sequence"):
        run_v0_smoke(genome, sequence_length=1)
