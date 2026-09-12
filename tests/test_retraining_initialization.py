import pytest
import torch

from act.model.dense import DenseBaselineConfig, DenseCausalLM
from act.training import initialize_retraining


def _model(hidden_size: int = 8) -> DenseCausalLM:
    return DenseCausalLM(
        DenseBaselineConfig(vocab_size=16, context_size=8, hidden_size=hidden_size, num_layers=1)
    )


def test_retraining_modes_report_shape_safe_transfers() -> None:
    source = _model()
    target = _model()
    evolved = initialize_retraining(target, mode="evolved", source=source)
    assert evolved.copied_parameters > 0
    rewind = initialize_retraining(_model(), mode="rewound", rewind_state=source.state_dict())
    assert rewind.copied_parameters == evolved.copied_parameters
    fresh = initialize_retraining(_model(), mode="fresh")
    assert fresh.copied_parameters == 0


def test_retraining_rejects_missing_policy_inputs() -> None:
    with pytest.raises(ValueError, match="source"):
        initialize_retraining(_model(), mode="teacher")
