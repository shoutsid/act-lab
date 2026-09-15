"""Declared V0-scale sparse MoE decoder for the compute-efficiency protocol.

Anatomy (pre-registered in ``RESEARCH_REPORT.md`` section 5.1 and matched to
the parameter by construction):

- RMSNorm (weight-only, no bias) before each sublayer and at the model output.
- Bias-free multi-head self-attention (fused QKV projection) with rotary
  position embeddings; no learned position table.
- Bias-free token-conditional router (``hidden_size x experts_in_layer``).
- Uniform bias-free experts: ``Linear -> SiLU -> Linear`` (``2 * h * ffn``
  parameters each), top-k routing exactly as in ``StaticMoEBlock``.
- Tied input/output embeddings; no separate LM head.
- No graph capture (no ``torch.compile`` or CUDA graphs): mutations rebuild
  the module tree at arbitrary points, so the dispatch must stay eager.
- Expert counts are **per layer** (``experts_per_layer``), because grow/prune
  mutations act on single layers and the toy path's uniform-count config can
  only represent single-layer genomes.

With the declared configuration (L=12, d=1024, heads=16, context=2048,
vocab=32768, 8 experts/layer, 2 active, expert ffn 1024) this anatomy gives

    P        = 285,336,576   (protocol total parameters)
    N_active = 134,341,632   (protocol per-token active parameters)

both reproduced to the parameter by :func:`estimated_parameter_count`.
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class V0MoEConfig:
    """Declared-scale configuration; defaults are the protocol's V0 model."""

    vocab_size: int = 32_768
    context_size: int = 2_048
    hidden_size: int = 1_024
    depth: int = 12
    attention_heads: int = 16
    experts_per_layer: tuple[int, ...] = (8,) * 12
    active_experts: int = 2
    expert_width: int = 1_024
    rotary_base: float = 10_000.0
    norm_epsilon: float = 1e-6
    expert_ids: tuple[tuple[str, ...], ...] = field(default=())

    def __post_init__(self) -> None:
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if len(self.experts_per_layer) != self.depth:
            raise ValueError("experts_per_layer must have one entry per layer")
        if any(count < 2 for count in self.experts_per_layer):
            raise ValueError("every layer needs at least 2 experts")
        if any(count > 16 for count in self.experts_per_layer):
            raise ValueError("every layer is capped at 16 experts (grow ceiling)")
        if not 2 <= self.active_experts <= min(self.experts_per_layer):
            raise ValueError("active_experts must be between 2 and the layer minimum")
        if self.depth < 1 or self.expert_width < 1:
            raise ValueError("depth and expert_width must be positive")
        if len(self.expert_ids) not in (0, self.depth):
            raise ValueError("expert_ids must be empty or one tuple per layer")
        if self.expert_ids and any(
            len(ids) != count for ids, count in zip(self.expert_ids, self.experts_per_layer)
        ):
            raise ValueError("expert_ids entries must match experts_per_layer")


DECLARED_V0_CONFIG = V0MoEConfig(
    vocab_size=32_768,
    context_size=2_048,
    hidden_size=1_024,
    depth=12,
    attention_heads=16,
    experts_per_layer=(8,) * 12,
    active_experts=2,
    expert_width=1_024,
)
"""The pre-registered section 5.1 configuration: P = 285,336,576 total and
N_active = 134,341,632 active parameters (plan section 8 V0 target)."""


DEV_V0_CONFIG = V0MoEConfig(
    vocab_size=8_192,
    context_size=512,
    hidden_size=320,
    depth=6,
    attention_heads=5,
    experts_per_layer=(8,) * 6,
    active_experts=2,
    expert_width=320,
)
"""The fast-iteration configuration: roughly 14.9M total and 7.6M active
parameters. This is the DEFAULT for development work, not a protocol config -
the pre-registered target is ``DECLARED_V0_CONFIG`` above, and it is built only
when a run at declared scale is actually being performed.

Sized by measurement on the development GPU (GTX 1070, 8 GiB) rather than by
parameter count: at these sizes the MoE block is latency-bound, not FLOP-bound,
because each block runs one kernel per expert per pass, so the SMALLEST shape is
not the fastest one. Measured, batch 8, fp16 autocast, 20-step steady state:

| config | total params | tokens/s | peak memory |
|---|---|---|---|
| h256 d6 e8 w256 v8k ctx512 | 10.0M | 3.6k | 0.76 GiB |
| h320 d6 e8 w320 v8k ctx512 (this one) | 14.9M | 9.8k | 0.90 GiB |
| h512 d6 e8 w512 v16k ctx512 | 39.9M | 6.0k | 1.77 GiB |

Cycle time is therefore governed by tokens per run, not by model size: a
400-step dev session at batch 8 x context 512 is 1.64M tokens per arm, about
2.8 minutes per arm on that GPU.

Measured end to end through ``run_v0_session`` (both arms, probes, checkpoints,
40 steps then extrapolated linearly at steady state): **0.61 s/step, 13.4k
tokens/s across the two arms, a 400-step dev cycle in ~4.1 minutes**, peak
0.90 GiB. The declared target is 285M parameters and 24.6M tokens per arm, i.e.
15x the tokens at ~1/3 the throughput - an hour plus per arm - which is why it
is no longer the default.
"""


def validate_declared_v0_config(config: V0MoEConfig) -> None:
    """Reject deviations from the pre-registered declared configuration.

    The protocol's tolerance band is derived from the exact parameter counts
    of the declared config, so a run that is not the declared model would make
    the band meaningless. Kernel-alignment constraints (hidden multiple of
    128, head_dim multiple of 64, context multiple of 128) are declared-config
    properties rather than class invariants so reduced-fidelity test configs
    stay constructible.
    """
    if config != DECLARED_V0_CONFIG:
        raise ValueError("config is not the declared V0 protocol configuration")
    if config.hidden_size % 128 or (config.hidden_size // config.attention_heads) % 64:
        raise ValueError("declared config: hidden/head geometry must stay kernel-aligned")
    if config.context_size % 128:
        raise ValueError("declared config: context_size must stay kernel-aligned")


def estimated_parameter_count(config: V0MoEConfig) -> tuple[int, int]:
    """Return ``(total, active)`` parameter counts without materializing weights.

    Active counts assume the static top-k routing contract (every token
    activates exactly ``active_experts`` experts per layer), which is the
    convention the pre-registered cost identity uses.
    """
    h = config.hidden_size
    attention = 4 * h * h  # fused QKV + output projections, bias-free
    block_norms = 2 * h  # norm_attn + norm_moe per block
    router = lambda count: h * count  # noqa: E731 - bias-free routing head
    expert = 2 * h * config.expert_width
    embeddings = config.vocab_size * h  # tied: covers input and output head
    final_norm = h
    total = embeddings + final_norm
    active = embeddings + final_norm
    for count in config.experts_per_layer:
        total += attention + block_norms + router(count) + count * expert
        active += attention + block_norms + router(count) + config.active_experts * expert
    return total, active


class _RMSNorm(nn.Module):
    def __init__(self, size: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.epsilon = epsilon

    def forward(self, hidden: Tensor) -> Tensor:
        squared = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden * torch.rsqrt(squared + self.epsilon)
        return normalized.to(hidden.dtype) * self.weight


def _apply_rope(x: Tensor, positions: Tensor, base: float) -> Tensor:
    """Rotate ``[batch, seq, heads, head_dim]`` by the rotary transform."""
    head_dim = x.shape[-1]
    half = head_dim // 2
    exponent = torch.arange(0, half, device=x.device, dtype=torch.float32)
    inv_freq = base ** (-exponent * (2.0 / head_dim))
    theta = positions.to(torch.float32).view(1, -1, 1, 1) * inv_freq.view(1, 1, 1, half)
    cos = theta.cos().to(x.dtype)
    sin = theta.sin().to(x.dtype)
    first, second = x[..., :half], x[..., half:]
    return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)


class _SelfAttention(nn.Module):
    """Bias-free multi-head self-attention with rotary positions."""

    def __init__(self, config: V0MoEConfig) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.head_dim = config.hidden_size // config.attention_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary_base = config.rotary_base

    def forward(self, hidden: Tensor, positions: Tensor) -> Tensor:
        batch, seq, _ = hidden.shape
        qkv = self.qkv(hidden)
        qkv = qkv.reshape(batch, seq, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = _apply_rope(q, positions, self.rotary_base).transpose(1, 2)
        k = _apply_rope(k, positions, self.rotary_base).transpose(1, 2)
        v = v.transpose(1, 2)
        attended = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, seq, self.heads * self.head_dim)
        return self.out(attended)


class _MoEFeedForward(nn.Module):
    """Top-k token routing over uniform bias-free experts (eager dispatch)."""

    def __init__(self, config: V0MoEConfig, expert_ids: tuple[str, ...]) -> None:
        super().__init__()
        self.expert_ids = expert_ids
        self.router = nn.Linear(config.hidden_size, len(expert_ids), bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(config.hidden_size, config.expert_width, bias=False),
                nn.SiLU(),
                nn.Linear(config.expert_width, config.hidden_size, bias=False),
            )
            for _ in range(len(expert_ids))
        )
        self.active_experts = config.active_experts
        self.last_load = torch.zeros(len(expert_ids))

    def forward(self, hidden: Tensor) -> Tensor:
        flat = hidden.reshape(-1, hidden.shape[-1])
        logits = self.router(flat)
        values, indices = logits.topk(self.active_experts, dim=-1)
        weights = values.softmax(dim=-1)
        self.last_load = torch.bincount(
            indices.reshape(-1), minlength=len(self.experts)
        ).float() / max(indices.numel(), 1)
        output = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            selected = (indices == expert_index).any(dim=-1)
            if not selected.any():
                continue
            positions = selected.nonzero(as_tuple=False).squeeze(-1)
            expert_output = expert(flat[positions])
            contribution = torch.zeros_like(expert_output)
            for slot in range(self.active_experts):
                slot_mask = indices[positions, slot] == expert_index
                if slot_mask.any():
                    contribution[slot_mask] += (
                        expert_output[slot_mask]
                        * weights[positions[slot_mask], slot].unsqueeze(-1)
                    )
            output[positions] += contribution
        return output.reshape_as(hidden)


class V0MoEBlock(nn.Module):
    def __init__(self, config: V0MoEConfig, expert_ids: tuple[str, ...]) -> None:
        super().__init__()
        self.norm_attn = _RMSNorm(config.hidden_size, config.norm_epsilon)
        self.attention = _SelfAttention(config)
        self.norm_moe = _RMSNorm(config.hidden_size, config.norm_epsilon)
        self.moe = _MoEFeedForward(config, expert_ids)

    def forward(self, hidden: Tensor, positions: Tensor) -> Tensor:
        hidden = hidden + self.attention(self.norm_attn(hidden), positions)
        return hidden + self.moe(self.norm_moe(hidden))


class V0MoECausalLM(nn.Module):
    """Declared-scale V0 sparse MoE decoder with tied embeddings."""

    def __init__(self, config: V0MoEConfig) -> None:
        super().__init__()
        self.config = config
        expert_ids = config.expert_ids or tuple(
            tuple(f"L{layer}E{expert}" for expert in range(config.experts_per_layer[layer]))
            for layer in range(config.depth)
        )
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            V0MoEBlock(config, expert_ids[layer]) for layer in range(config.depth)
        )
        self.final_norm = _RMSNorm(config.hidden_size, config.norm_epsilon)
        self.activation_checkpointing = False
        self._init_weights()

    def _init_weights(self) -> None:
        """Deterministic scaled-normal init (seeded externally by the session)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        if tokens.shape[1] > self.config.context_size:
            raise ValueError("sequence exceeds context_size")
        seq = tokens.shape[1]
        positions = torch.arange(seq, device=tokens.device)
        hidden = self.token_embedding(tokens)
        for block in self.blocks:
            if self.training and self.activation_checkpointing:
                hidden = checkpoint(block, hidden, positions, use_reentrant=False)
            else:
                hidden = block(hidden, positions)
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
        return tuple(
            tuple(float(value) for value in block.moe.last_load)
            for block in self.blocks
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def active_parameter_count(self) -> int:
        """Static top-k active count: non-expert parameters plus k experts/layer."""
        expert = 2 * self.config.hidden_size * self.config.expert_width
        inactive = sum(
            (count - self.config.active_experts) * expert
            for count in self.config.experts_per_layer
        )
        return self.parameter_count - inactive
