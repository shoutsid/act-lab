"""Deterministic capability probes over tokenized evaluation examples."""

from dataclasses import dataclass

import torch

from ._mode import evaluation_mode
from .perplexity import LanguageModel


@dataclass(frozen=True)
class CapabilityProbeResult:
    name: str
    loss: float
    accuracy: float
    tokens: int

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "tokens": self.tokens,
        }


def evaluate_capability_probes(
    model: LanguageModel,
    probes: dict[str, tuple[torch.Tensor, ...] | list[torch.Tensor]],
) -> tuple[CapabilityProbeResult, ...]:
    if not probes:
        raise ValueError("at least one capability probe is required")
    results: list[CapabilityProbeResult] = []
    with evaluation_mode(model), torch.no_grad():
        for name in sorted(probes):
            batches = probes[name]
            if not name:
                raise ValueError("probe names must not be empty")
            if not batches:
                raise ValueError(f"probe {name} has no batches")
            total_loss = 0.0
            total_correct = 0
            total_tokens = 0
            for tokens in batches:
                if tokens.ndim != 2 or tokens.shape[1] < 2:
                    raise ValueError(f"probe {name} batches need sequence length >= 2")
                logits = model(tokens[:, :-1])
                labels = tokens[:, 1:]
                loss = model.loss(tokens)
                if not torch.isfinite(logits).all():
                    raise ValueError(f"probe {name} produced non-finite logits")
                if not torch.isfinite(loss).all():
                    raise ValueError(f"probe {name} produced non-finite loss")
                total_loss += float(loss) * labels.numel()
                total_correct += int((logits.argmax(dim=-1) == labels).sum())
                total_tokens += labels.numel()
            mean_loss = total_loss / total_tokens
            accuracy = total_correct / total_tokens
            if not torch.isfinite(torch.tensor([mean_loss, accuracy])).all():
                raise ValueError(f"probe {name} produced non-finite metrics")
            results.append(
                CapabilityProbeResult(name, mean_loss, accuracy, total_tokens)
            )
    return tuple(results)
