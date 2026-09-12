import pytest
import torch

from act.evaluation import evaluate_capability_probes
from act.model.dense import DenseBaselineConfig, DenseCausalLM


def test_capability_probes_are_sorted_and_report_accuracy() -> None:
    model = DenseCausalLM(
        DenseBaselineConfig(vocab_size=16, context_size=8, hidden_size=8, num_layers=1)
    )
    probes = {
        "copy": [torch.randint(0, 16, (1, 8))],
        "addition": [torch.randint(0, 16, (1, 8))],
    }
    results = evaluate_capability_probes(model, probes)
    assert [result.name for result in results] == ["addition", "copy"]
    assert all(0.0 <= result.accuracy <= 1.0 for result in results)
    assert all(result.tokens == 7 for result in results)


def test_capability_probes_reject_invalid_inputs() -> None:
    model = DenseCausalLM(DenseBaselineConfig(hidden_size=8, num_heads=2))
    with pytest.raises(ValueError, match="at least one"):
        evaluate_capability_probes(model, {})
    with pytest.raises(ValueError, match="no batches"):
        evaluate_capability_probes(model, {"empty": []})
