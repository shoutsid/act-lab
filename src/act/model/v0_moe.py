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

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


MoEModuleType = Literal["attention", "gdn", "hybrid"]
"""Per-layer sequence-mixer choice: attention, a gated delta mixer, or both."""

MODULE_TYPES: frozenset[str] = frozenset({"attention", "gdn", "hybrid"})


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
    module_types: tuple[str, ...] = ()
    """Per-layer sequence mixers; empty means every layer is ``attention``.

    Empty is the historical default and keeps every pre-existing configuration,
    model reconstruction and parameter count bit-identical. This field is only
    populated by callers that opt in (see ``honor_module_types`` on
    :func:`act.model.factory.v0_moe_config_from_genome`).
    """

    gdn_chunk_size: int = 0
    """Timesteps per checkpointed segment of the gated delta scan; 0 disables it.

    The recurrence is written as an explicit loop, which means autograd retains
    a ``(batch, heads, key_dim, value_dim)`` write *and* state tensor for every
    one of the ``context_size`` steps. At batch 2, context 1024 and hidden 768
    that is roughly 0.8 GB retained per GDN layer - measured, not estimated: 2
    GDN layers peak at 5.0 GB, 4 at 6.6 GB, and 8 exceed an 8 GB card. Segmenting
    the scan into checkpointed chunks stores only the chunk boundaries and
    recomputes within a chunk, which cuts retention by roughly an order of
    magnitude at the cost of one extra forward per segment.

    Default 0 keeps every existing configuration bit-identical; the chunked path
    is an opt-in that must produce the same numbers (see
    ``tests/test_gated_delta_mixer.py``).
    """

    def __post_init__(self) -> None:
        if min(
            self.vocab_size,
            self.context_size,
            self.hidden_size,
            self.depth,
            self.attention_heads,
            self.expert_width,
        ) < 1:
            raise ValueError(
                "vocab_size, context_size, hidden_size, depth, attention_heads, "
                "and expert_width must be positive"
            )
        if not math.isfinite(self.rotary_base) or self.rotary_base <= 0:
            raise ValueError("rotary_base must be finite and positive")
        if not math.isfinite(self.norm_epsilon) or self.norm_epsilon <= 0:
            raise ValueError("norm_epsilon must be finite and positive")
        if len(self.module_types) not in (0, self.depth):
            raise ValueError("module_types must be empty or carry one entry per layer")
        if self.gdn_chunk_size < 0:
            raise ValueError("gdn_chunk_size must not be negative")
        if any(module not in MODULE_TYPES for module in self.module_types):
            raise ValueError("unsupported module type in module_types")
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
400-step dev session at batch 8 x context 512 is 1.64M tokens per arm.

Measured end to end through ``run_v0_session`` (both arms, probes, checkpoints,
400 steps on the real TinyStories stream) by ``scripts/dev_cycle.py``: **each arm
19.6k tokens/s (0.21 s/step); a two-arm 400-step dev cycle is 2.8 minutes wall**
at 14.9M parameters, peak ~0.9 GiB - the 9.8k in the table above was that shape
before the expert block folded its per-expert launches into batched matmuls. The declared target is 285M parameters and
24.6M tokens per arm, i.e. 15x the tokens at a fraction of the throughput - an
hour plus per arm - which is why it is no longer the default.
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


def resolve_module_types(config: V0MoEConfig) -> tuple[str, ...]:
    """Per-layer mixer for every block, defaulting to all-attention."""
    if not config.module_types:
        return ("attention",) * config.depth
    if len(config.module_types) == 1:
        return config.module_types * config.depth
    return config.module_types


def mixer_parameter_count(config: V0MoEConfig, module_type: str = "attention") -> int:
    """Parameters in one block's sequence mixer.

    Attention is ``4 h^2`` (fused QKV plus output). The gated delta mixer adds
    two ``h x heads`` gate projections, keeping it within a few percent of an
    attention block at matched width.
    """
    h = config.hidden_size
    attention = 4 * h * h
    if module_type == "attention":
        return attention
    gdn = attention + 2 * h * config.attention_heads
    if module_type == "gdn":
        return gdn
    if module_type == "hybrid":
        return attention + gdn
    raise ValueError(f"unsupported module type {module_type}")


def _norm_parameter_count(module_type: str) -> int:
    """RMSNorm widths in a block: every block has ``norm_moe``, plus one norm
    per mixer present (``norm_attn`` and/or ``norm_mixer``)."""
    count = 1
    if module_type in {"attention", "hybrid"}:
        count += 1
    if module_type in {"gdn", "hybrid"}:
        count += 1
    return count


def estimated_parameter_count(config: V0MoEConfig) -> tuple[int, int]:
    """Return ``(total, active)`` parameter counts without materializing weights.

    Active counts assume the static top-k routing contract (every token
    activates exactly ``active_experts`` experts per layer), which is the
    convention the pre-registered cost identity uses.
    """
    h = config.hidden_size
    module_types = resolve_module_types(config)
    router = lambda count: h * count  # noqa: E731 - bias-free routing head
    expert = 2 * h * config.expert_width
    embeddings = config.vocab_size * h  # tied: covers input and output head
    final_norm = h
    total = embeddings + final_norm
    active = embeddings + final_norm
    for index, count in enumerate(config.experts_per_layer):
        module_type = module_types[index]
        mixer = mixer_parameter_count(config, module_type)
        norms = _norm_parameter_count(module_type) * h
        total += mixer + norms + router(count) + count * expert
        active += mixer + norms + router(count) + config.active_experts * expert
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


class _GatedDeltaMixer(nn.Module):
    """Gated delta-rule sequence mixer with a linear-time recurrent state.

    Per head, with ``S`` the ``(key_dim x value_dim)`` state and ``q_t``/``k_t``
    L2-normalized, this implements exactly::

        alpha_t = exp(-softplus(gate_t))   decay, one value per key dimension
        beta_t  = sigmoid(write_t)         write strength, one value per head
        S_t     = diag(alpha_t) (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
        o_t     = S_t^T q_t

    The erase term makes each write a *correction* of the state's current
    prediction ``S_{t-1}^T k_t`` rather than an accumulation, and the gate decays
    the retained state along the key dimension.

    This is a compact member of the Gated DeltaNet family, not Qwen3-Next's
    block: it shares one key/value dimension and omits the short convolution, so
    its cost is comparable to attention (``4 h^2`` plus the two ``h x heads``
    gates) rather than reproducing that model's geometry. The recurrence is
    deliberately written as an explicit scan and pinned by test, because a wrong
    mixer would corrupt a search for exactly the reason a wrong seed did.
    """

    def __init__(self, config: V0MoEConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.heads = config.attention_heads
        self.head_dim = hidden // self.heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, self.heads, bias=False)
        self.write_proj = nn.Linear(hidden, self.heads, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)
        self.chunk_size = config.gdn_chunk_size

    def project(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return ``(q, k, v, alpha, beta)`` with gates already activated."""
        batch, seq, _ = hidden.shape
        shape = (batch, seq, self.heads, self.head_dim)
        q = nn.functional.normalize(self.q_proj(hidden).reshape(shape), dim=-1)
        k = nn.functional.normalize(self.k_proj(hidden).reshape(shape), dim=-1)
        v = self.v_proj(hidden).reshape(shape)
        alpha = torch.exp(-nn.functional.softplus(self.gate_proj(hidden)))
        beta = torch.sigmoid(self.write_proj(hidden))
        return q, k, v, alpha, beta

    def scan_segment(
        self,
        state: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        alpha: Tensor,
        beta: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Advance the recurrence over one segment; returns ``(state, outputs)``.

        Kept parameter-free and pure so it can be wrapped in ``checkpoint``
        without the module's weights entering the segment's saved state.
        """
        seq = q.shape[1]
        outputs: list[Tensor] = []
        for step in range(seq):
            q_t, k_t, v_t = q[:, step], k[:, step], v[:, step]
            gate = alpha[:, step].unsqueeze(-1).unsqueeze(-1)
            strength = beta[:, step].unsqueeze(-1).unsqueeze(-1)
            prediction = torch.einsum("bhij,bhi->bhj", state, k_t)
            error = v_t - prediction
            write = torch.einsum("bhi,bhj->bhij", k_t, error)
            state = state * gate + strength * write
            outputs.append(torch.einsum("bhij,bhi->bhj", state, q_t))
        return state, torch.stack(outputs, dim=1)

    def mix(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        alpha: Tensor,
        beta: Tensor,
        chunk_size: int = 0,
    ) -> Tensor:
        """Run the recurrence over ``(batch, seq, heads, head_dim)`` inputs.

        ``chunk_size > 0`` splits the scan into checkpointed segments so the
        segment boundaries are the only state autograd retains. The arithmetic
        and its order are unchanged, so a chunked run is numerically identical
        to an unchunked one and the setting can never confound a comparison.
        """
        batch, seq, _, dim = q.shape
        state = q.new_zeros(batch, self.heads, dim, dim)

        if chunk_size <= 0 or chunk_size >= seq:
            state, stacked = self.scan_segment(state, q, k, v, alpha, beta)
        else:
            pieces: list[Tensor] = []
            for start in range(0, seq, chunk_size):
                stop = min(start + chunk_size, seq)
                state, piece = checkpoint(
                    self.scan_segment,
                    state,
                    q[:, start:stop],
                    k[:, start:stop],
                    v[:, start:stop],
                    alpha[:, start:stop],
                    beta[:, start:stop],
                    use_reentrant=False,
                )
                pieces.append(piece)
            stacked = torch.cat(pieces, dim=1)
        return self.out(stacked.reshape(batch, seq, self.heads * dim))

    def forward(self, hidden: Tensor) -> Tensor:
        q, k, v, alpha, beta = self.project(hidden)
        return self.mix(q, k, v, alpha, beta, self.chunk_size)


class _MoEFeedForward(nn.Module):
    """Top-k token routing over uniform bias-free experts.

    The routing contract is top-k sparse dispatch: every token activates exactly
    ``active_experts`` experts, and the FLOPs the protocol books are the
    contract's (``active_parameter_count``), not whatever the kernels happen to
    do. ``forward`` implements that contract with a *dense* expert evaluation -
    every expert applied to every token by one batched matmul, then the top-k
    rows selected and gate-weighted - because at the development model's size
    eager per-expert dispatch is launch-bound rather than FLOP-bound. Measured on
    the development GPU (GTX 1070), ``DEV_V0_CONFIG`` at batch 8 x context 512:
    51.8 ms/step eager vs 42.5 ms/step dense (79k -> 96k tokens/s, +18%). Those two
    rates are about 4x the dev shape's measured end-to-end throughput, so they are
    not what a GPU-hour budget should be sized from - see
    ``artifacts/arch-zoo/PERFORMANCE_NOTES.md`` for the measured figures and for the
    three docstring rates it records as mutually inconsistent. The
    dense form does up to ``num_experts`` times the arithmetic in three large
    kernels; ``eager_forward`` keeps the dispatch formulation and is pinned
    equivalent to it by test.
    """

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

    def _route(self, flat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Gate weights, chosen expert indices, and the recorded routing load."""
        logits = self.router(flat)
        values, indices = logits.topk(self.active_experts, dim=-1)
        weights = values.softmax(dim=-1)
        self.last_load = torch.bincount(
            indices.reshape(-1), minlength=len(self.experts)
        ).float() / max(indices.numel(), 1)
        return weights, indices, logits

    def _expert_weights(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Experts stacked as ``(in, out)`` matmul operands, transposed from ``Linear``."""
        first = torch.stack([expert[0].weight.transpose(0, 1) for expert in self.experts])
        second = torch.stack([expert[2].weight.transpose(0, 1) for expert in self.experts])
        return first.to(dtype=dtype, device=device), second.to(dtype=dtype, device=device)

    def eager_forward(self, hidden: Tensor) -> Tensor:
        """Per-expert dispatch: the reference formulation of the same contract."""
        flat = hidden.reshape(-1, hidden.shape[-1])
        weights, indices, _ = self._route(flat)
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

    def forward(self, hidden: Tensor) -> Tensor:
        flat = hidden.reshape(-1, hidden.shape[-1])
        weights, indices, _ = self._route(flat)
        first, second = self._expert_weights(flat.dtype, flat.device)
        # (tokens, hidden) @ (hidden, experts*width) is one big matmul; folding the
        # expert axis into the column dimension keeps it dense and launch-free.
        num_experts, hidden_size, width = first.shape
        wide = first.permute(1, 0, 2).reshape(hidden_size, num_experts * width)
        hidden_experts = (flat @ wide).reshape(flat.shape[0], num_experts, width)
        hidden_experts = torch.nn.functional.silu(hidden_experts)
        # Batched per-expert gemm: (experts, tokens, width) @ (experts, width, hidden).
        expert_outputs = torch.bmm(hidden_experts.transpose(0, 1), second).transpose(0, 1)
        # Select the top-k rows per token and gate-weight them.
        gathered = torch.gather(
            expert_outputs,
            1,
            indices.unsqueeze(-1).expand(-1, -1, expert_outputs.shape[-1]),
        )
        output = (gathered * weights.unsqueeze(-1)).sum(dim=1)
        return output.reshape_as(hidden)


class V0MoEBlock(nn.Module):
    """A sequence mixer followed by a sparse MoE feed-forward, both residual."""

    def __init__(
        self,
        config: V0MoEConfig,
        expert_ids: tuple[str, ...],
        module_type: str = "attention",
    ) -> None:
        super().__init__()
        if module_type not in MODULE_TYPES:
            raise ValueError(f"unsupported module type {module_type}")
        self.module_type = module_type
        self.attention = (
            _SelfAttention(config) if module_type in {"attention", "hybrid"} else None
        )
        self.mixer = (
            _GatedDeltaMixer(config) if module_type in {"gdn", "hybrid"} else None
        )
        if self.attention is not None:
            self.norm_attn = _RMSNorm(config.hidden_size, config.norm_epsilon)
        if self.mixer is not None:
            self.norm_mixer = _RMSNorm(config.hidden_size, config.norm_epsilon)
        self.norm_moe = _RMSNorm(config.hidden_size, config.norm_epsilon)
        self.moe = _MoEFeedForward(config, expert_ids)

    def forward(self, hidden: Tensor, positions: Tensor) -> Tensor:
        if self.attention is not None:
            hidden = hidden + self.attention(self.norm_attn(hidden), positions)
        if self.mixer is not None:
            hidden = hidden + self.mixer(self.norm_mixer(hidden))
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
        module_types = resolve_module_types(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            V0MoEBlock(config, expert_ids[layer], module_types[layer])
            for layer in range(config.depth)
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
