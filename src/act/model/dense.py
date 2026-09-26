"""Small causal language model used to validate the ACT training contract."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ._inputs import validate_token_batch


@dataclass(frozen=True)
class DenseBaselineConfig:
    vocab_size: int = 128
    context_size: int = 32
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    ffn_size: int = 256

    def __post_init__(self) -> None:
        if min(
            self.vocab_size,
            self.context_size,
            self.hidden_size,
            self.num_layers,
            self.num_heads,
            self.ffn_size,
        ) < 1:
            raise ValueError(
                "vocab_size, context_size, hidden_size, num_layers, num_heads, "
                "and ffn_size must be positive"
            )
        if self.context_size < 2:
            raise ValueError("context_size must be at least 2")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")


class DenseBlock(nn.Module):
    def __init__(self, config: DenseBaselineConfig) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(config.hidden_size)
        self.attn = nn.MultiheadAttention(
            config.hidden_size,
            config.num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.ffn_size),
            nn.GELU(),
            nn.Linear(config.ffn_size, config.hidden_size),
        )

    def forward(self, hidden: Tensor, causal_mask: Tensor) -> Tensor:
        normalized = self.norm_attn(hidden)
        attended, _ = self.attn(normalized, normalized, normalized, attn_mask=causal_mask)
        hidden = hidden + attended
        return hidden + self.ffn(self.norm_ffn(hidden))


class DenseCausalLM(nn.Module):
    """A compact decoder-only baseline with tied token embeddings."""

    def __init__(self, config: DenseBaselineConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.context_size, config.hidden_size)
        self.blocks = nn.ModuleList(DenseBlock(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.activation_checkpointing = False

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
            if self.training and self.activation_checkpointing:
                hidden = checkpoint(block, hidden, mask, use_reentrant=False)
            else:
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