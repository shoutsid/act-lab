"""Static sparse-MoE baseline sharing the dense model training contract."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class StaticMoEConfig:
    vocab_size: int = 128
    context_size: int = 32
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    ffn_size: int = 256
    num_experts: int = 4
    active_experts: int = 2
    expert_ids: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if min(
            self.vocab_size,
            self.context_size,
            self.hidden_size,
            self.num_layers,
            self.num_heads,
            self.ffn_size,
            self.num_experts,
        ) < 1:
            raise ValueError(
                "vocab_size, context_size, hidden_size, num_layers, num_heads, "
                "ffn_size, and num_experts must be positive"
            )
        if self.context_size < 2:
            raise ValueError("context_size must be at least 2")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.active_experts < 1:
            raise ValueError("active_experts must be at least 1")
        if len(self.expert_ids) not in (0, self.num_layers):
            raise ValueError("expert_ids must be empty or contain one tuple per layer")
        expert_counts = (
            tuple(len(ids) for ids in self.expert_ids)
            if self.expert_ids
            else (self.num_experts,) * self.num_layers
        )
        if any(count == 0 for count in expert_counts):
            raise ValueError("every layer needs at least one expert")
        if self.active_experts > min(expert_counts):
            raise ValueError("active_experts cannot exceed the expert count in any layer")
        if self.expert_ids and any(len(set(ids)) != len(ids) for ids in self.expert_ids):
            raise ValueError("expert ids must be unique within each layer")


class StaticMoEBlock(nn.Module):
    def __init__(self, config: StaticMoEConfig, expert_ids: tuple[str, ...]) -> None:
        super().__init__()
        if not expert_ids:
            raise ValueError("a MoE block needs at least one expert id")
        # Expert count is PER LAYER: topology evolution produces heterogeneous
        # layers, so the block derives its shape from its own ids rather than
        # the config's single ``num_experts`` field.
        num_experts = len(expert_ids)
        self.expert_ids = expert_ids
        self.num_experts = num_experts
        self.norm_attn = nn.LayerNorm(config.hidden_size)
        self.attn = nn.MultiheadAttention(
            config.hidden_size,
            config.num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm_moe = nn.LayerNorm(config.hidden_size)
        self.router = nn.Linear(config.hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(config.hidden_size, config.ffn_size),
                nn.GELU(),
                nn.Linear(config.ffn_size, config.hidden_size),
            )
            for _ in range(num_experts)
        )
        if not 1 <= config.active_experts <= num_experts:
            raise ValueError(
                f"active_experts ({config.active_experts}) must be between 1 and "
                f"the layer's expert count ({num_experts})"
            )
        self.active_experts = config.active_experts
        self.last_load = torch.zeros(num_experts)

    def forward(self, hidden: Tensor, causal_mask: Tensor) -> Tensor:
        normalized = self.norm_attn(hidden)
        attended, _ = self.attn(normalized, normalized, normalized, attn_mask=causal_mask)
        hidden = hidden + attended
        routed = self.norm_moe(hidden)
        logits = self.router(routed)
        values, indices = logits.topk(self.active_experts, dim=-1)
        weights = values.softmax(dim=-1)
        flat = routed.reshape(-1, routed.shape[-1])
        flat_indices = indices.reshape(-1, self.active_experts)
        flat_weights = weights.reshape(-1, self.active_experts)
        self.last_load = torch.bincount(
            flat_indices.reshape(-1), minlength=len(self.experts)
        ).float() / max(flat_indices.numel(), 1)
        output = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            selected = (flat_indices == expert_index).any(dim=-1)
            if not selected.any():
                continue
            positions = selected.nonzero(as_tuple=False).squeeze(-1)
            expert_input = flat[positions]
            expert_output = expert(expert_input)
            contribution = torch.zeros_like(expert_output)
            for slot in range(self.active_experts):
                slot_mask = flat_indices[positions, slot] == expert_index
                if slot_mask.any():
                    contribution[slot_mask] += expert_output[slot_mask] * flat_weights[positions[slot_mask], slot].unsqueeze(-1)
            output[positions] += contribution
        return hidden + output.reshape_as(hidden)


class StaticMoECausalLM(nn.Module):
    """A fixed sparse MoE decoder used as the non-evolving control."""

    def __init__(self, config: StaticMoEConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if config.active_experts < 1:
            raise ValueError("active_experts must be at least 1")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.context_size, config.hidden_size)
        expert_ids = config.expert_ids or tuple(
            tuple(f"L{layer}E{expert}" for expert in range(config.num_experts))
            for layer in range(config.num_layers)
        )
        if len(expert_ids) != config.num_layers:
            raise ValueError("expert_ids must match num_layers")
        self.blocks = nn.ModuleList(
            StaticMoEBlock(config, expert_ids[layer]) for layer in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.activation_checkpointing = False

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        if tokens.shape[1] > self.config.context_size:
            raise ValueError("sequence exceeds context_size")
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
        logits = self(tokens[:, :-1])
        return nn.functional.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            tokens[:, 1:].reshape(-1),
        )

    @property
    def routing_load(self) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(float(value) for value in block.last_load) for block in self.blocks)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def active_parameter_count(self) -> int:
        active = 0
        for name, parameter in self.named_parameters():
            if ".experts." not in name:
                active += parameter.numel()
        for block in self.blocks:
            for expert_index, expert in enumerate(block.experts):
                if float(block.last_load[expert_index]) > 0.0:
                    active += sum(parameter.numel() for parameter in expert.parameters())
        return active