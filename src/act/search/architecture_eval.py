"""Static architecture evaluation harness for architecture search.

Trains a *fixed* architecture (no mutations) and reports loss, parameters and
cost. Everything that is not the architecture under test is pinned to the run
seed, so two candidates evaluated at the same seed see the same tokens in the
same order from the same initialization draw:

- model initialization is drawn after ``torch.manual_seed(seed)``;
- the training batch order is a seeded permutation of non-overlapping windows;
- the validation batches are a fixed prefix of the held-out stream.

Cost is reported three ways, because at this scale they disagree and the
disagreement is itself a result:

``contracted_flops_per_token``
    The declared top-k routing contract: every token activates exactly
    ``active_experts`` experts per layer. This is the quantity that makes two
    architectures comparable on paper.
``dense_runtime_flops_per_token``
    What this implementation actually asks the hardware to do. ``_MoEFeedForward``
    evaluates *every* expert on *every* token and then gathers the top-k rows
    (that is faster than eager dispatch at these sizes), so runtime arithmetic
    scales with *total* expert count, not active.
``measured_tokens_per_s`` / ``measured_step_s``
    Wall-clock. When the contract and the runtime disagree, this is the number
    that decides how many GPU-hours a candidate actually costs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from numbers import Integral
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from act.evaluation.probes import evaluate_capability_probes
from act.model.factory import v0_moe_config_from_genome
from act.model.v0_moe import V0MoECausalLM, V0MoEConfig, estimated_parameter_count
from act.training.runtime import _autocast_context
from act.topology.genome import ArchitectureGenome, genome_hash, validate_genome


@dataclass(frozen=True)
class RunConfig:
    """Every factor held fixed across a comparison, recorded for auditability."""

    steps: int = 400
    batch_size: int = 2
    context_size: int = 1024
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 40
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    val_batches: int = 16
    seed: int = 42
    train_data_path: str | None = None
    val_data_path: str | None = None
    #: Validation tokens carved off the end of the training stream when no
    #: adequately sized held-out file is supplied. The training stream is
    #: truncated by this much so train and validation never overlap.
    val_holdout_tokens: int = 2_000_000
    #: Whether ``layer.module`` selects the sequence mixer. On by default here:
    #: the search path must build the architecture its genome describes. Legacy
    #: and topology-churn paths leave it off, where ``module`` is metadata only.
    honor_module_types: bool = True
    #: ``off``, ``fp16`` or ``bf16``, matching ``TrainingConfig.mixed_precision``
    #: in :mod:`act.training.runtime`. Recorded with every result: a loss measured
    #: in one precision is not comparable to a loss measured in another, so the
    #: precision has to travel with the number.
    mixed_precision: str = "off"
    activation_checkpointing: bool = False
    #: Timesteps per checkpointed segment of the gated delta scan; 0 disables it.
    #: Only meaningful when the genome carries GDN layers. Measured at the zoo's
    #: geometry: the unchunked scan retains roughly 0.8 GB per GDN layer, so 8
    #: layers exceed an 8 GB card while the chunked path fits. Recorded with the
    #: result because segmenting changes memory, not arithmetic.
    gdn_chunk_size: int = 0

    def __post_init__(self) -> None:
        integer_fields = (
            "steps",
            "batch_size",
            "context_size",
            "warmup_steps",
            "val_batches",
            "seed",
            "val_holdout_tokens",
            "gdn_chunk_size",
        )
        if any(
            not isinstance(getattr(self, name), Integral)
            or isinstance(getattr(self, name), bool)
            for name in integer_fields
        ):
            raise ValueError("run schedule dimensions and seed must be integers")
        if min(self.steps, self.batch_size, self.context_size, self.val_batches) < 1:
            raise ValueError("steps, batch_size, context_size, and val_batches must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.val_holdout_tokens < 0:
            raise ValueError("val_holdout_tokens must be non-negative")
        if self.gdn_chunk_size < 0:
            raise ValueError("gdn_chunk_size must be non-negative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.min_lr_ratio) or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be finite and between 0 and 1")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.grad_clip) or self.grad_clip < 0:
            raise ValueError("grad_clip must be finite and non-negative")

    @property
    def train_tokens(self) -> int:
        return self.steps * self.batch_size * self.context_size

    @property
    def val_tokens(self) -> int:
        return self.val_batches * self.batch_size * self.context_size


@dataclass(frozen=True)
class EvalResult:
    """Outcome of one (architecture, seed) training run."""

    name: str
    family: str
    genome_hash: str
    seed: int
    total_params: int
    active_params: int
    contracted_flops_per_token: float
    dense_runtime_flops_per_token: float
    training_flops: float
    validation_flops: float
    final_train_loss: float
    mean_train_loss: float
    validation_loss: float
    steps: int
    train_tokens: int
    val_tokens: int
    elapsed_s: float
    measured_step_s: float
    measured_tokens_per_s: float
    peak_memory_mb: float
    device: str
    description: str
    #: Structural genes, recorded so the ledger is self-describing: an axis
    #: analysis can then ask which *structural* variable moves with loss, not
    #: only which cost variable does. Without these, a matched design can show a
    #: large spread and no cost axis to attribute it to, which is exactly what
    #: arch-zoo-100m-v2 produced. Zero means "not recorded" (pre-existing ledgers).
    depth: int = 0
    hidden_size: int = 0
    attention_heads: int = 0
    expert_count: int = 0
    active_experts: int = 0
    expert_width: int = 0
    val_source: str = ""
    run_config: dict = field(default_factory=dict)
    #: Named capability-probe results, empty unless the caller supplied probes.
    #: Brief section 19 applies probes to *finalists* only. That is also the right
    #: engineering choice: probing every candidate during search would quietly add
    #: a second objective to a search that is supposed to be steered by held-out
    #: validation loss alone, and the result would no longer be interpretable as
    #: an architecture effect.
    capability: dict = field(default_factory=dict)


def estimate_flops_per_token(
    config: V0MoEConfig,
    *,
    context_size: int | None = None,
    training: bool = True,
) -> float:
    """Analytical FLOPs per token under the declared top-k routing contract.

    Forward cost is the standard estimate

        ``2 * active_params + 4 * depth * seq * hidden_size``

    where the second term is the two attention matmuls (``QK^T`` and ``AV``),
    each ``2 * seq * hidden_size`` per token per layer. Training multiplies by
    three for forward plus backward. The dense-expert runtime this repository
    actually executes is separated out in :func:`estimate_dense_runtime_flops_per_token`.
    """
    sequence = context_size or config.context_size
    _, active = estimated_parameter_count(config)
    forward = 2.0 * active + 4.0 * config.depth * sequence * config.hidden_size
    return 3.0 * forward if training else forward


def estimate_dense_runtime_flops_per_token(
    config: V0MoEConfig,
    *,
    context_size: int | None = None,
    training: bool = True,
) -> float:
    """Arithmetic this implementation actually performs per token.

    ``_MoEFeedForward.forward`` applies all experts to all tokens before
    selecting the top-k rows, so the expert term uses *total* parameters.
    """
    sequence = context_size or config.context_size
    total, _ = estimated_parameter_count(config)
    forward = 2.0 * total + 4.0 * config.depth * sequence * config.hidden_size
    return 3.0 * forward if training else forward


def learning_rate_at(step: int, run: RunConfig) -> float:
    """Linear warmup to ``learning_rate``, then cosine decay to ``min_lr_ratio``."""
    if run.warmup_steps > 0 and step < run.warmup_steps:
        return run.learning_rate * (step + 1) / run.warmup_steps
    decay_steps = max(1, run.steps - run.warmup_steps)
    progress = min(1.0, max(0.0, (step - run.warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = run.learning_rate * run.min_lr_ratio
    return floor + (run.learning_rate - floor) * cosine


def _load_tokens(path: str | None, vocab_size: int, length: int, seed: int) -> torch.Tensor:
    """Load a token stream, falling back to a seeded synthetic stream.

    The synthetic fallback is a smoke test for the harness, never evidence.
    Uniform random tokens have no learnable structure above the entropy floor
    ``log(vocab_size)``, so a loss comparison between architectures on that
    stream ranks initialization luck. Pass a real token file for any comparison
    that is meant to support a conclusion.
    """
    if path is not None:
        values = np.load(path)
        if values.dtype.kind not in "iu":
            raise ValueError(f"{path} must contain integer token ids (got {values.dtype})")
        if values.size and (values.min() < 0 or values.max() >= vocab_size):
            raise ValueError(f"{path} contains token ids outside [0, {vocab_size})")
        tokens = torch.from_numpy(values).to(torch.long).reshape(-1)
        if tokens.numel() < length:
            raise ValueError(f"{path} holds {tokens.numel()} tokens, need {length}")
        return tokens
    print(
        "  [warn] no token file given: using uniform random tokens. "
        "Losses on this stream measure initialization, not architecture.",
        flush=True,
    )
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (length,), generator=generator)


def required_train_tokens(run: RunConfig) -> int:
    """Tokens the training stream must hold to serve ``run``'s schedule once.

    :func:`_training_batches` cuts the stream into ``context_size + 1`` spans, so
    a schedule of ``steps * batch_size`` batches consumes exactly this many
    tokens with no window repeated. That is the bar this module admits a run on.

    It used to be ``train_tokens * 2`` - twice the schedule, for no stated
    reason - which refused runs that were perfectly servable. The horizon arm is
    the case that exposed it: its 12,000-step design consumes 24.6M tokens and
    uses 24,000 of the stream's 37,115 windows (0.61 epochs, no repeat), yet a
    40.0M-token stream was rejected for holding under 51.2M, so all sixteen runs
    raised at data load and the sweep trained none of the arm it was launched
    for. A stream that meets this bar can be scheduled; one that does not cannot.
    """
    return run.steps * run.batch_size * (run.context_size + 1)


def required_holdout_tokens(run: RunConfig, *, val_tokens_available: int | None) -> int:
    """Tokens carved off the training stream so validation has its own data.

    Zero when the supplied validation file is large enough for the requested
    evaluation, because then nothing is taken from the training stream; otherwise
    the tail of the training stream is held out.
    """
    needed_val = run.val_tokens + run.context_size
    if val_tokens_available is not None and val_tokens_available >= needed_val:
        return 0
    return max(run.val_holdout_tokens, needed_val)


def required_stream_tokens(
    run: RunConfig, *, val_tokens_available: int | None = None
) -> int:
    """Tokens ``run.train_data_path`` must hold for the whole run.

    The module's own admission bar, exposed so a caller can check a plan against
    the file on disk *before* spending GPU hours on it. ``scripts/arch_zoo_screen.py``
    does exactly that at launch: under the old bar the horizon arm's plan was
    feasible on paper, infeasible in fact, and the difference only appeared as
    sixteen identical exceptions in a kernel log hours into the session.
    """
    return required_train_tokens(run) + required_holdout_tokens(
        run, val_tokens_available=val_tokens_available
    )


def _prepare_streams(
    vocab_size: int, run: RunConfig
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Return ``(train, validation, val_source)`` token streams for a run.

    A supplied validation file is used only when it is big enough for the
    requested evaluation; otherwise the tail of the training stream is held out
    and cut away from training, so the two streams never overlap. This exists
    because the repository's ``val-tokens.npy`` holds roughly 7k tokens - about
    seven windows at context 1024 - which cannot support a 16-batch evaluation.
    """
    needed_train = required_train_tokens(run)
    needed_val = run.val_tokens + run.context_size

    if run.val_data_path is not None:
        candidate = _load_tokens(run.val_data_path, vocab_size, 1, run.seed + 1)
        if candidate.numel() >= needed_val:
            train = _load_tokens(run.train_data_path, vocab_size, needed_train, run.seed)
            return train, candidate, run.val_data_path
        print(
            f"  [warn] {run.val_data_path} holds {candidate.numel()} tokens, "
            f"need {needed_val}: holding out from the training stream instead.",
            flush=True,
        )

    # Single-sourced with the admission bar so a caller checking a plan against the
    # file on disk measures the same quantity this function enforces.
    holdout = required_holdout_tokens(run, val_tokens_available=None)
    stream = _load_tokens(run.train_data_path, vocab_size, needed_train + holdout, run.seed)
    return (
        stream[: stream.numel() - holdout],
        stream[stream.numel() - holdout :],
        f"holdout:{holdout} of {run.train_data_path or 'synthetic'}",
    )


def _training_batches(data: torch.Tensor, run: RunConfig) -> list[torch.Tensor]:
    """Deterministic, seed-derived batch schedule shared by every candidate.

    The stream is cut into non-overlapping ``context_size`` windows and their
    order is a seeded permutation, so all architectures at a given seed consume
    identical tokens in identical order.
    """
    span = run.context_size + 1
    windows = data.numel() // span
    if windows < run.batch_size:
        raise ValueError("token stream is too short for a single batch")
    generator = torch.Generator().manual_seed(run.seed)
    order = torch.randperm(windows, generator=generator)
    batches: list[torch.Tensor] = []
    cursor = 0
    for _ in range(run.steps):
        if cursor + run.batch_size > windows:
            cursor = 0
        picks = order[cursor : cursor + run.batch_size]
        cursor += run.batch_size
        batches.append(torch.stack([data[p * span : p * span + span] for p in picks]))
    return batches


def _validation_batches(data: torch.Tensor, run: RunConfig) -> list[torch.Tensor]:
    """Fixed held-out batches: the same prefix for every candidate."""
    span = run.context_size + 1
    windows = data.numel() // span
    if windows < run.batch_size:
        raise ValueError("validation stream is too short for a single batch")
    return [
        torch.stack([data[i * span : i * span + span] for i in range(batch * run.batch_size, (batch + 1) * run.batch_size)])
        for batch in range(min(run.val_batches, windows // run.batch_size))
    ]


@torch.no_grad()
def evaluate_loss(model: V0MoECausalLM, batches: list[torch.Tensor]) -> float:
    """Token-weighted mean loss over held-out batches."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for tokens in batches:
        tokens = tokens.to(next(model.parameters()).device)
        loss = model.loss(tokens)
        count = tokens.shape[0] * (tokens.shape[1] - 1)
        total_loss += float(loss) * count
        total_tokens += count
    model.train()
    if total_tokens == 0:
        raise ValueError("no validation tokens were evaluated")
    return total_loss / total_tokens


def train_architecture(
    genome: ArchitectureGenome,
    *,
    name: str = "unnamed",
    family: str = "",
    description: str = "",
    run: RunConfig | None = None,
    device: str = "auto",
    probes: Mapping[str, Sequence[Any]] | None = None,
) -> EvalResult:
    """Train one static architecture and return its loss and cost.

    The genotype is fixed for the whole run: no growth, pruning, cloning,
    merging, splitting or rewiring happens here.

    ``probes`` maps a probe name to its token batches and is evaluated *after*
    training, against the trained weights. Leave it empty while screening, or the
    probe losses become a second selection criterion.
    """
    run = run or RunConfig()
    validation = validate_genome(genome)
    if not validation.valid:
        raise ValueError(f"invalid genome: {'; '.join(validation.errors)}")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_object = torch.device(device)
    if run.mixed_precision not in {"off", "fp16", "bf16"}:
        raise ValueError("mixed_precision must be off, fp16 or bf16")

    # The seed is drawn here, immediately before construction, so that the
    # recorded seed fully determines the initialization. Recording a seed
    # without consuming it was the previous harness's central defect.
    torch.manual_seed(run.seed)

    config = v0_moe_config_from_genome(
        genome, honor_module_types=run.honor_module_types
    )
    if run.gdn_chunk_size:
        config = replace(config, gdn_chunk_size=run.gdn_chunk_size)
    model = V0MoECausalLM(config).to(device)
    if hasattr(model, "activation_checkpointing"):
        model.activation_checkpointing = run.activation_checkpointing

    # fp16 needs loss scaling; bf16 has fp32's exponent range and does not. The
    # scaler is disabled rather than absent for the other modes so the step is
    # one code path, which is what the repository's own benchmark does.
    scaler = torch.amp.GradScaler(
        device_object.type, enabled=run.mixed_precision == "fp16"
    )

    total_params, active_params = estimated_parameter_count(config)
    contracted = estimate_flops_per_token(config, context_size=run.context_size)
    dense_runtime = estimate_dense_runtime_flops_per_token(config, context_size=run.context_size)

    train_data, val_data, val_source = _prepare_streams(config.vocab_size, run)
    train_batches = _training_batches(train_data, run)
    val_batches = _validation_batches(val_data, run)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=run.learning_rate, weight_decay=run.weight_decay
    )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.monotonic()

    model.train()
    losses: list[float] = []
    for step, tokens in enumerate(train_batches):
        tokens = tokens.to(device)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(step, run)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device_object, run.mixed_precision):
            loss = model.loss(tokens)
        scaler.scale(loss).backward()
        if run.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), run.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
    )

    validation_loss = evaluate_loss(model, val_batches)

    capability: dict = {}
    if probes:
        capability = {
            outcome.name: outcome.to_dict()
            for outcome in evaluate_capability_probes(model, probes)
        }

    return EvalResult(
        name=name,
        family=family,
        genome_hash=genome_hash(genome),
        seed=run.seed,
        total_params=total_params,
        active_params=active_params,
        contracted_flops_per_token=contracted,
        dense_runtime_flops_per_token=dense_runtime,
        training_flops=contracted * run.train_tokens,
        validation_flops=contracted * len(val_batches) * run.batch_size * run.context_size,
        final_train_loss=losses[-1] if losses else float("inf"),
        mean_train_loss=sum(losses) / len(losses) if losses else float("inf"),
        validation_loss=validation_loss,
        steps=run.steps,
        train_tokens=run.train_tokens,
        val_tokens=run.val_tokens,
        elapsed_s=elapsed,
        measured_step_s=elapsed / max(1, run.steps),
        measured_tokens_per_s=run.train_tokens / elapsed if elapsed > 0 else 0.0,
        peak_memory_mb=peak_memory,
        device=device,
        description=description,
        depth=config.depth,
        hidden_size=config.hidden_size,
        attention_heads=config.attention_heads,
        expert_count=max(config.experts_per_layer),
        active_experts=config.active_experts,
        expert_width=config.expert_width,
        val_source=val_source,
        run_config=asdict(run),
        capability=capability,
    )


def evaluate_batch(
    candidates: list,
    *,
    run: RunConfig | None = None,
    device: str = "auto",
    output_dir: str | None = "artifacts/arch-zoo",
    tag: str = "zoo",
) -> list[EvalResult]:
    """Evaluate candidates, appending each result to JSONL as it completes.

    Results are written incrementally: an interrupted sweep keeps every run that
    already finished. The earlier harness wrote only at the end, so the real-data
    sweep left a zoo definition and no results at all.
    """
    run = run or RunConfig()
    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
    ledger = output_path / f"{tag}_results.jsonl" if output_path else None

    results: list[EvalResult] = []
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"[{index}/{len(candidates)}] {candidate.name} seed={run.seed} "
            f"step={run.steps} ctx={run.context_size}",
            flush=True,
        )
        try:
            result = train_architecture(
                candidate.genome,
                name=candidate.name,
                family=getattr(candidate, "family", ""),
                description=getattr(candidate, "description", ""),
                run=run,
                device=device,
            )
        except Exception as error:  # keep the sweep going; OOM kills one arm, not the zoo
            print(f"  FAILED: {type(error).__name__}: {error}", flush=True)
            continue
        results.append(result)
        if ledger is not None:
            with open(ledger, "a") as handle:
                handle.write(json.dumps(asdict(result)) + "\n")
        print(
            f"  val_loss={result.validation_loss:.4f} train_loss={result.final_train_loss:.4f} "
            f"params={result.total_params:,} active={result.active_params:,} "
            f"{result.measured_tokens_per_s:,.0f} tok/s {result.elapsed_s:.0f}s "
            f"peak={result.peak_memory_mb:,.0f}MB",
            flush=True,
        )
    return results


def load_results(paths) -> list[EvalResult]:
    """Read :class:`EvalResult` records back from one or more JSONL ledgers.

    Lets a sweep be reported on at any point without re-running anything.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    results: list[EvalResult] = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line:
                results.append(EvalResult(**json.loads(line)))
    return results


def print_results_table(results: list[EvalResult]) -> None:
    """Print results ranked by held-out validation loss."""
    print(f"\n{'=' * 118}")
    print(f"Architecture zoo results ({len(results)} runs, ranked by validation loss)")
    print(f"{'=' * 118}")
    print(
        f"{'Name':<24} {'Family':<11} {'Seed':>5} {'Total':>11} {'Active':>11} "
        f"{'ValLoss':>9} {'TrainLoss':>10} {'tok/s':>8} {'s/step':>7} {'MB':>6}"
    )
    print("-" * 118)
    for result in sorted(results, key=lambda item: item.validation_loss):
        print(
            f"{result.name:<24} {result.family:<11} {result.seed:>5} "
            f"{result.total_params:>11,} {result.active_params:>11,} "
            f"{result.validation_loss:>9.4f} {result.final_train_loss:>10.4f} "
            f"{result.measured_tokens_per_s:>8,.0f} {result.measured_step_s:>7.2f} "
            f"{result.peak_memory_mb:>6,.0f}"
        )
    print(f"{'=' * 118}\n")
