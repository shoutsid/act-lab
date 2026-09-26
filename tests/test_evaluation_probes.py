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


def test_capability_probe_matches_model_loss_with_one_forward_per_batch(monkeypatch) -> None:
    model = DenseCausalLM(
        DenseBaselineConfig(vocab_size=16, context_size=8, hidden_size=8, num_layers=1)
    )
    batches = [torch.randint(0, 16, (2, 6)), torch.randint(0, 16, (1, 4))]
    expected_loss = sum(
        float(model.loss(batch).detach()) * batch.shape[0] * (batch.shape[1] - 1)
        for batch in batches
    ) / sum(batch.shape[0] * (batch.shape[1] - 1) for batch in batches)
    calls = 0
    original = model.forward

    def count_forward(tokens):
        nonlocal calls
        calls += 1
        return original(tokens)

    monkeypatch.setattr(model, "forward", count_forward)
    result = evaluate_capability_probes(model, {"copy": batches})[0]

    assert result.loss == pytest.approx(expected_loss)
    assert calls == len(batches)


def test_capability_probe_rejects_logits_that_do_not_align_with_labels(monkeypatch) -> None:
    model = DenseCausalLM(DenseBaselineConfig(hidden_size=8, num_heads=2))
    monkeypatch.setattr(
        model, "forward", lambda tokens: torch.zeros(tokens.shape[0], 1, 128)
    )

    with pytest.raises(ValueError, match="do not align"):
        evaluate_capability_probes(model, {"copy": [torch.ones(1, 5, dtype=torch.long)]})


def test_capability_probes_restore_training_mode_on_success_and_error() -> None:
    model = DenseCausalLM(DenseBaselineConfig(hidden_size=8, num_heads=2))
    model.train()

    evaluate_capability_probes(model, {"copy": [torch.randint(0, 128, (1, 4))]})
    assert model.training

    with pytest.raises(ValueError, match="sequence length"):
        evaluate_capability_probes(model, {"copy": [torch.ones(1, 1, dtype=torch.long)]})
    assert model.training


def test_capability_probes_reject_nonfinite_loss_and_logits(monkeypatch) -> None:
    model = DenseCausalLM(DenseBaselineConfig(hidden_size=8, num_heads=2))
    tokens = torch.randint(0, 128, (1, 4))
    monkeypatch.setattr(
        torch.nn.functional,
        "cross_entropy",
        lambda logits, labels: torch.tensor(float("nan")),
    )
    with pytest.raises(ValueError, match="non-finite loss"):
        evaluate_capability_probes(model, {"copy": [tokens]})

    monkeypatch.undo()
    model = DenseCausalLM(DenseBaselineConfig(hidden_size=8, num_heads=2))
    monkeypatch.setattr(model, "forward", lambda batch: torch.full((*batch.shape, 128), float("inf")))
    with pytest.raises(ValueError, match="non-finite logits"):
        evaluate_capability_probes(model, {"copy": [tokens]})


def test_capability_probes_reject_invalid_inputs() -> None:
    model = DenseCausalLM(DenseBaselineConfig(hidden_size=8, num_heads=2))
    with pytest.raises(ValueError, match="at least one"):
        evaluate_capability_probes(model, {})
    with pytest.raises(ValueError, match="no batches"):
        evaluate_capability_probes(model, {"empty": []})
