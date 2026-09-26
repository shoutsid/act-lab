"""Configurable V0 decoder blocks for the ACT research path."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from ._inputs import validate_token_batch


ModuleType = Literal["attention", "gdn", "hybrid"]


@dataclass(frozen=True)
class V0ModelConfig:
    vocab_size: int = 32_768
    context_size: int = 2_048
    hidden_size: int = 1_024
    depth: int = 24
    attention_heads: int = 16
    ffn_width: int = 4_096
    module_types: tuple[ModuleType, ...] = ("attention",)

    def __post_init__(self) -> None:
        if min(
            self.vocab_size,
            self.context_size,
            self.hidden_size,
            self.depth,
            self.attention_heads,
            self.ffn_width,
        ) < 1:
            raise ValueError(
                "vocab_size, context_size, hidden_size, depth, attention_heads, "
                "and ffn_width must be positive"
            )
        if self.context_size < 2:
            raise ValueError("context_size must be at least 2")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if len(self.module_types) not in {1, self.depth}:
            raise ValueError("module_types must contain one value or one per layer")
        if any(module not in {"attention", "gdn", "hybrid"} for module in self.module_types):
            raise ValueError("unsupported module type in module_types")


class _AttentionBranch(nn.Module):
    def __init__(self, config: V0ModelConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size)
        self.attention = nn.MultiheadAttention(
            config.hidden_size,
            config.attention_heads,
            dropout=0.0,
            batch_first=True,
        )

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        normalized = self.norm(hidden)
        attended, _ = self.attention(normalized, normalized, normalized, attn_mask=mask)
        return hidden + attended


class _GatedDeltaBranch(nn.Module):
    def __init__(self, config: V0ModelConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size)
        self.update = nn.Linear(config.hidden_size, config.hidden_size)
        self.gate = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden: Tensor) -> Tensor:
        normalized = self.norm(hidden)
        update = torch.tanh(self.update(normalized))
        gate = torch.sigmoid(self.gate(normalized))
        state = torch.zeros_like(update[:, 0])
        outputs = []
        for timestep in range(update.shape[1]):
            state = gate[:, timestep] * state + (1.0 - gate[:, timestep]) * update[:, timestep]
            outputs.append(state)
        return hidden + torch.stack(outputs, dim=1)


class _FeedForward(nn.Module):
    def __init__(self, config: V0ModelConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size)
        self.network = nn.Sequential(
            nn.Linear(config.hidden_size, config.ffn_width),
            nn.GELU(),
            nn.Linear(config.ffn_width, config.hidden_size),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + self.network(self.norm(hidden))


class V0Block(nn.Module):
    def __init__(self, config: V0ModelConfig, module_type: ModuleType) -> None:
        super().__init__()
        self.module_type = module_type
        self.attention = _AttentionBranch(config) if module_type in {"attention", "hybrid"} else None
        self.gdn = _GatedDeltaBranch(config) if module_type in {"gdn", "hybrid"} else None
        self.ffn = _FeedForward(config)

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        if self.attention is not None:
            hidden = self.attention(hidden, mask)
        if self.gdn is not None:
            hidden = self.gdn(hidden)
        return self.ffn(hidden)


class V0CausalLM(nn.Module):
    """Decoder-only V0 model with attention, GDN, and hybrid module choices."""

    def __init__(self, config: V0ModelConfig) -> None:
        super().__init__()
        self.config = config
        module_types = config.module_types * config.depth if len(config.module_types) == 1 else config.module_types
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.context_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            V0Block(config, module_types[index]) for index in range(config.depth)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, tokens: Tensor) -> Tensor:
        validate_token_batch(
            tokens, max_sequence_length=self.config.context_size
        )
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        mask = torch.triu(
            torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        for block in self.blocks:
            hidden = block(hidden, mask)
        hidden = self.final_norm(hidden)
        return hidden @ self.token_embedding.weight.transpose(0, 1)

    def loss(self, tokens: Tensor) -> Tensor:
        validate_token_batch(
            tokens, min_sequence_length=2, max_sequence_length=self.config.context_size + 1
        )
        logits = self(tokens[:, :-1])
        return nn.functional.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            tokens[:, 1:].reshape(-1),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def active_parameter_count(self) -> int:
        return self.parameter_count
