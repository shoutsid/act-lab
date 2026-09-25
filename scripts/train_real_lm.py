"""Train the project's V0MoECausalLM on a real English corpus (TinyStories).

This is the "real data" leg the plan's data sections deferred: a real corpus,
a real BPE tokenizer trained on that corpus, real next-token training on the
GPU, and held-out validation loss/perplexity. Everything uses the project's
own model class so the result validates the actual training path.

Stages are idempotent and cached under artifacts/real-lm/:
  1. corpus    - download TinyStories train/valid text (network required once)
  2. tokenizer - train a BPE tokenizer (vocab 8192) on a corpus sample
  3. tokens    - encode a slice of stories into a flat token stream
  4. train     - AdamW + warmup/cosine, periodic held-out eval, checkpoints
  5. samples   - generate completions from fixed prompts (real inference)

Usage:
  PYTHONPATH=src python scripts/train_real_lm.py                 # full run
  PYTHONPATH=src python scripts/train_real_lm.py --max-steps 200 # smoke
  PYTHONPATH=src python scripts/train_real_lm.py --stage tokens  # up to stage
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn

from act.model import V0MoECausalLM, V0MoEConfig

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "real-lm"

TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
VALID_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
TRAIN_BYTES = 1_979_055_864  # approximate; used for a sanity check, not a hard gate
VOCAB_SIZE = 8_192
CONTEXT_SIZE = 512
BATCH_SIZE = 16
HIDDEN_SIZE = 384
DEPTH = 6
ATTENTION_HEADS = 6
EXPERTS_PER_LAYER = 8
ACTIVE_EXPERTS = 2
EXPERT_WIDTH = 384
LEARNING_RATE = 8e-4
WARMUP_STEPS = 200
EVAL_EVERY = 500
CHECKPOINT_EVERY = 2_000
SEED = 7

PROMPTS = (
    "Once upon a time, there was a little",
    "Tom and Lily were playing in the",
    "The dog ran to the",
    "One day, Mom said,",
)


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --- stage 1: corpus -------------------------------------------------------


def stage_corpus(force: bool = False) -> tuple[Path, Path]:
    ART.mkdir(parents=True, exist_ok=True)
    train_path = ART / "TinyStories-train.txt"
    valid_path = ART / "TinyStories-valid.txt"
    for url, path in ((TRAIN_URL, train_path), (VALID_URL, valid_path)):
        if path.is_file() and path.stat().st_size > 0 and not force:
            _log(f"corpus cached: {path.name} ({path.stat().st_size:,} bytes)")
            continue
        _log(f"downloading {url} ...")
        started = time.time()
        with urllib.request.urlopen(url, timeout=120) as response, path.open("wb") as out:
            done = 0
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if done % (200 << 20) < (1 << 20):
                    _log(f"  {path.name}: {done / 1e9:.2f} GB")
        _log(f"downloaded {path.name}: {path.stat().st_size:,} bytes in {time.time() - started:.0f}s")
    return train_path, valid_path


def read_stories(path: Path, max_bytes: int | None = None) -> list[str]:
    text = path.read_text(errors="ignore")[:max_bytes] if max_bytes else path.read_text(errors="ignore")
    stories = [story.strip() for story in text.split("<|endoftext|>") if len(story.strip()) > 100]
    return stories


# --- stage 2: tokenizer ----------------------------------------------------


def stage_tokenizer(train_path: Path, force: bool = False) -> Path:
    tokenizer_path = ART / "tokenizer.json"
    if tokenizer_path.is_file() and not force:
        _log("tokenizer cached")
        return tokenizer_path
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    _log("training BPE tokenizer on a corpus sample ...")
    stories = read_stories(train_path, max_bytes=40_000_000)[:20_000]
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<pad>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(stories, trainer)
    tokenizer.save(str(tokenizer_path))
    _log(f"tokenizer saved: vocab {tokenizer.get_vocab_size()}")
    return tokenizer_path


# --- stage 3: token stream -------------------------------------------------


def stage_tokens(train_path: Path, target_tokens: int) -> Path:
    tokens_path = ART / "tokens.npy"
    if tokens_path.is_file():
        have = int(np.load(tokens_path, mmap_mode="r").shape[0])
        if have >= target_tokens:
            _log(f"token stream cached: {have:,} tokens")
            return tokens_path
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(ART / "tokenizer.json"))
    _log(f"tokenizing stories into >= {target_tokens:,} tokens ...")
    # Token 0 is the reserved <pad> special token (BPE merges never emit it),
    # used as a story boundary so the model can learn story endings.
    boundary = 0
    ids: list[np.ndarray] = []
    total = 0
    stories = read_stories(train_path)
    batch: list[str] = []
    for story in stories:
        batch.append(story)
        if len(batch) >= 2_000:
            for e in tokenizer.encode_batch(batch):
                array = np.asarray(e.ids, dtype=np.uint16)
                if array.size >= 8:
                    ids.append(array)
                    total += array.size + 1
            batch = []
            _log(f"  {total:,} tokens so far")
            if total >= target_tokens:
                break
    if batch and total < target_tokens:
        for e in tokenizer.encode_batch(batch):
            array = np.asarray(e.ids, dtype=np.uint16)
            if array.size >= 8:
                ids.append(array)
                total += array.size + 1
    if total < target_tokens:
        raise SystemExit(f"corpus slice produced only {total:,} tokens (needed {target_tokens:,})")
    flat = np.empty(total, dtype=np.uint16)
    cursor = 0
    for array in ids:
        flat[cursor] = boundary
        flat[cursor + 1 : cursor + 1 + array.size] = array
        cursor += 1 + array.size
    np.save(tokens_path, flat)
    _log(f"token stream saved: {total:,} tokens -> {tokens_path}")
    return tokens_path


def load_val_tokens(valid_path: Path, max_stories: int = 1_500) -> np.ndarray:
    cache = ART / "val-tokens.npy"
    if cache.is_file():
        return np.load(cache)
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(ART / "tokenizer.json"))
    stories = read_stories(valid_path)[:max_stories]
    arrays = []
    for e in tokenizer.encode_batch(stories):
        array = np.asarray(e.ids, dtype=np.uint16)
        if array.size >= CONTEXT_SIZE + 1:
            arrays.append(array)
    tokens = np.concatenate(arrays)
    np.save(cache, tokens)
    return tokens


# --- stage 4: training -----------------------------------------------------


def model_config() -> V0MoEConfig:
    return V0MoEConfig(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        hidden_size=HIDDEN_SIZE,
        depth=DEPTH,
        attention_heads=ATTENTION_HEADS,
        experts_per_layer=(EXPERTS_PER_LAYER,) * DEPTH,
        active_experts=ACTIVE_EXPERTS,
        expert_width=EXPERT_WIDTH,
    )


def lr_at(step: int, max_steps: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, max_steps - WARMUP_STEPS)
    return 0.1 * LEARNING_RATE + 0.9 * LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * progress))


def draw_batch(stream: np.ndarray, generator: torch.Generator) -> torch.Tensor:
    high = stream.shape[0] - CONTEXT_SIZE - 1
    starts = torch.randint(0, high, (BATCH_SIZE,), generator=generator).numpy()
    windows = np.stack([stream[s : s + CONTEXT_SIZE + 1] for s in starts]).astype(np.int64)
    return torch.from_numpy(windows)


@torch.no_grad()
def evaluate(model: V0MoECausalLM, val: np.ndarray, device: torch.device, batches: int = 8) -> tuple[float, float]:
    model.eval()
    losses = []
    generator = torch.Generator().manual_seed(SEED)
    high = val.shape[0] - CONTEXT_SIZE - 1
    for _ in range(batches):
        starts = torch.randint(0, high, (BATCH_SIZE,), generator=generator).numpy()
        windows = np.stack([val[s : s + CONTEXT_SIZE + 1] for s in starts]).astype(np.int64)
        tokens = torch.from_numpy(windows).to(device)
        logits = model(tokens[:, :-1])
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), tokens[:, 1:].reshape(-1)
        )
        losses.append(loss.item())
    model.train()
    mean = sum(losses) / len(losses)
    return mean, math.exp(mean)


@torch.no_grad()
def baseline_losses(val: np.ndarray, vocab_size: int) -> dict:
    """Held-out next-token loss for two reference models, on the LM's own domains.

    Both references are fit on the held-out stream *itself*, which makes them a
    yardstick rather than a competitor: the LM has never seen these tokens and
    they have seen nothing else. The uniform floor is the model's own denominator
    (``ln(vocab_size)``); the unigram and bigram are smoothed with Laplace
    ``alpha=1`` so an unseen token cannot score ``-inf``.

    They are here because a perplexity with nothing beside it is not a proof of
    anything - 5.5 could be good or terrible. A reader can recompute every number
    below from the committed held-out stream, which is the whole point of
    committing it.

    One deliberate mismatch: the LM is scored on random 512-token windows of this
    stream, and these references on every adjacent pair of it. Both are next-token
    loss on the same held-out text, and the difference is in the LM's favour
    (windows give it context), so the comparison is conservative.
    """
    tokens = np.asarray(val, dtype=np.int64).tolist()
    pairs = list(zip(tokens[:-1], tokens[1:]))
    n = len(pairs)
    if n == 0:
        raise SystemExit("held-out stream is too short to score a baseline on")

    unigram = collections.Counter(target for _, target in pairs)
    bigram = collections.Counter(pairs)
    context = collections.Counter(source for source, _ in pairs)

    unigram_loss = -sum(
        math.log((unigram[target] + 1.0) / (n + vocab_size)) for _, target in pairs
    ) / n
    bigram_loss = -sum(
        math.log((bigram[pair] + 1.0) / (context[pair[0]] + vocab_size)) for pair in pairs
    ) / n

    def block(loss: float) -> dict:
        return {"loss": loss, "perplexity": math.exp(loss)}

    return {
        "held_out_tokens": len(tokens),
        "scored_pairs": n,
        "smoothing": "Laplace alpha=1",
        "fit_on": "the held-out stream itself - a reference point, not a competitor",
        "uniform": block(math.log(vocab_size)),
        "unigram": block(unigram_loss),
        "bigram": block(bigram_loss),
    }


def generate(model: V0MoECausalLM, tokenizer, prompt: str, device: torch.device, new_tokens: int = 60, temperature: float = 0.8) -> str:
    from tokenizers import Tokenizer

    assert isinstance(tokenizer, Tokenizer)
    ids = tokenizer.encode(prompt).ids
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    generator = torch.Generator(device="cpu").manual_seed(SEED + len(prompt))
    for _ in range(new_tokens):
        window = tokens[:, -CONTEXT_SIZE:]
        logits = model(window)[:, -1, :]
        if temperature <= 0:
            next_id = int(logits.argmax(dim=-1))
        else:
            probs = torch.softmax(logits[0] / temperature, dim=-1)
            next_id = int(torch.multinomial(probs.cpu(), 1, generator=generator))
        tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device)], dim=1)
    text = tokenizer.decode(tokens[0].tolist())
    return text.replace("<pad>", "")


# --- checkpointing ---------------------------------------------------------
#
# The optimizer's moments are bound to parameter *names*, not to positions.
# `Optimizer.state_dict()` is position-keyed: `load_state_dict` maps its entries
# onto `self.param_groups[i]["params"]` **by index**. `nn.Module.state_dict()`
# does the opposite and matches by name, so a model whose parameter *order*
# changed while its names and shapes stayed the same loads its weights cleanly
# and attaches its Adam moments to the wrong tensors.
#
# That is not hypothetical. The real-LM run of 2026-09-14 was resumed on
# 2026-09-25 into exactly that one-position shift, and the first step raised
# `RuntimeError: output with shape [384] doesn't match the broadcast shape
# [1152, 384]` from inside `torch._foreach_lerp_` - a message naming no
# parameter and no cause. Two things follow: the moments are now stored by name,
# and a positional payload is refused instead of guessed at.


def _named_moments(model: V0MoECausalLM, optimizer: torch.optim.Optimizer) -> dict[str, dict]:
    """The optimizer's per-parameter state, keyed by parameter name."""
    by_id = {id(param): name for name, param in model.named_parameters()}
    moments: dict[str, dict] = {}
    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state.get(param)
            if not state:
                continue
            moments[by_id[id(param)]] = {
                key: (value.detach().clone() if torch.is_tensor(value) else value)
                for key, value in state.items()
            }
    return moments


def checkpoint_payload(
    model: V0MoECausalLM, optimizer: torch.optim.Optimizer, step: int, history: list[dict]
) -> dict:
    """What a checkpoint holds: weights by name, moments by name, step, history."""
    return {
        "model": model.state_dict(),
        "optimizer": {
            "groups": [
                {key: value for key, value in group.items() if key != "params"}
                for group in optimizer.param_groups
            ],
            "moments": _named_moments(model, optimizer),
        },
        "step": step,
        "history": history,
    }


def restore_optimizer(
    model: V0MoECausalLM,
    optimizer: torch.optim.Optimizer,
    payload: dict,
    path: Path,
    *,
    reset: bool,
) -> str:
    """Attach a checkpoint's moments to the parameters they were trained against.

    A positional payload cannot be attached safely, so it is refused with the
    reason and both remedies named rather than allowed to misalign. ``reset`` is
    the named remedy for a checkpoint written before the moments were keyed by
    name: resume the weights, start Adam fresh, and say so in the log.

    Returns a short description of what was done, which the report records - a
    resumed run whose optimizer state came from somewhere other than the
    checkpoint is a different object from one trained straight through.
    """
    saved = payload.get("optimizer")
    if reset:
        _log("optimizer moments re-initialised (--reset-optimizer); weights resumed")
        return "re-initialised (--reset-optimizer)"
    if not saved:
        _log("checkpoint carries no optimizer state; moments start fresh")
        return "fresh (checkpoint carried none)"
    if "moments" not in saved:
        raise SystemExit(
            f"{path} stores its optimizer state by position, which cannot be attached "
            "to this model: the parameters are not in the order that run had them in, "
            "and a state_dict that loads by name does not imply an order that does. "
            "Re-run with --reset-optimizer to resume the weights with fresh Adam "
            "moments, or delete the checkpoint to train from scratch."
        )

    by_name = dict(model.named_parameters())
    missing: list[str] = []
    wrong_shape: list[str] = []
    for name, moment in saved["moments"].items():
        param = by_name.get(name)
        if param is None:
            missing.append(name)
            continue
        for key in ("exp_avg", "exp_avg_sq"):
            value = moment.get(key)
            if torch.is_tensor(value) and tuple(value.shape) != tuple(param.shape):
                wrong_shape.append(f"{name}.{key} {tuple(value.shape)} != {tuple(param.shape)}")
        optimizer.state[param] = dict(moment)
    if missing or wrong_shape:
        raise SystemExit(
            f"{path}'s optimizer state does not match this model - missing "
            f"{missing[:4]} of {len(missing)}, shape mismatches {wrong_shape[:4]} "
            f"of {len(wrong_shape)}. Refusing to continue rather than train with "
            "moments that belong to other parameters."
        )
    for group, metadata in zip(optimizer.param_groups, saved["groups"]):
        group.update(metadata)
    _log(f"optimizer moments restored by name for {len(saved['moments'])} parameters")
    return f"restored by name for {len(saved['moments'])} parameters"


def stage_train(
    tokens_path: Path,
    valid_path: Path,
    max_steps: int,
    log_every: int,
    *,
    reset_optimizer: bool = False,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"training on {device}")
    torch.manual_seed(SEED)
    stream = np.load(tokens_path, mmap_mode="r")
    val = load_val_tokens(valid_path)
    _log(f"stream {stream.shape[0]:,} tokens; held-out {val.shape[0]:,} tokens")

    config = model_config()
    model = V0MoECausalLM(config).to(device)
    model.activation_checkpointing = True
    model.train()
    total_params = model.parameter_count
    _log(
        f"model: {total_params:,} total / {model.active_parameter_count:,} active parameters"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    data_generator = torch.Generator().manual_seed(SEED)
    checkpoint_dir = ART / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest = checkpoint_dir / "latest.pt"
    start_step = 0
    history: list[dict] = []
    optimizer_state = "fresh (no checkpoint)"
    if latest.is_file():
        payload = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer_state = restore_optimizer(
            model, optimizer, payload, latest, reset=reset_optimizer
        )
        start_step = payload["step"]
        history = payload.get("history", [])
        _log(f"resumed from step {start_step}: optimizer state {optimizer_state}")

    tokens_per_step = BATCH_SIZE * CONTEXT_SIZE
    started = time.time()
    uniform_floor = math.log(VOCAB_SIZE)
    for step in range(start_step + 1, max_steps + 1):
        lr = lr_at(step, max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        tokens = draw_batch(stream, data_generator).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(tokens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_every == 0 or step == max_steps:
            rate = tokens_per_step * log_every / (time.time() - started)
            _log(
                f"step {step}/{max_steps} loss {loss.item():.4f} lr {lr:.2e} "
                f"{rate:,.0f} tokens/s (uniform floor {uniform_floor:.2f})"
            )
            started = time.time()
        if step % EVAL_EVERY == 0 or step == max_steps:
            val_loss, perplexity = evaluate(model, val, device)
            history.append({"step": step, "train_loss": loss.item(), "val_loss": val_loss, "perplexity": perplexity})
            _log(f"  eval: val loss {val_loss:.4f} perplexity {perplexity:.2f}")
        if step % CHECKPOINT_EVERY == 0 or step == max_steps:
            torch.save(
                checkpoint_payload(model, optimizer, step, history),
                latest.with_suffix(".tmp"),
            )
            latest.with_suffix(".tmp").replace(latest)

    final_loss, final_ppl = evaluate(model, val, device)
    samples = {}
    for prompt in PROMPTS:
        samples[prompt] = generate(model, tokenizer_from_disk(), prompt, device)
    report = {
        "corpus": "TinyStories",
        "tokenizer": "BPE vocab 8192 trained on corpus",
        "model": {
            "class": "act.model.V0MoECausalLM",
            "total_parameters": total_params,
            "active_parameters": model.active_parameter_count,
            "hidden_size": HIDDEN_SIZE,
            "depth": DEPTH,
            "attention_heads": ATTENTION_HEADS,
            "experts_per_layer": EXPERTS_PER_LAYER,
            "active_experts": ACTIVE_EXPERTS,
            "expert_width": EXPERT_WIDTH,
            "vocab_size": VOCAB_SIZE,
            "context_size": CONTEXT_SIZE,
        },
        "training": {
            "steps": max_steps,
            "batch_tokens": tokens_per_step,
            "tokens_seen": max_steps * tokens_per_step,
            "optimizer": "AdamW warmup+cosine",
            # `lr_at` is a function of the *total* budget, so a continuation whose
            # budget grew restarts the cosine decay instead of finishing the old
            # one: the 6000 -> 12000 run resumed at lr 8.0e-05 and stepped back up
            # to 4.0e-04, which is why the recorded held-out curve is not monotone.
            # That is a property of the schedule and is left in the curve rather
            # than smoothed out - a reader sees the bump, and this explains it.
            "seed": SEED,
            "device": str(device),
        },
        "validation": {
            "final_val_loss": final_loss,
            "final_perplexity": final_ppl,
            "uniform_floor_loss": uniform_floor,
            "uniform_floor_perplexity": VOCAB_SIZE,
            "history": history,
        },
        "baselines": baseline_losses(val, VOCAB_SIZE),
        "samples": samples,
    }
    report_path = ART / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _log(f"report written: {report_path}")
    return report


def tokenizer_from_disk():
    """The corpus BPE tokenizer, with the byte-level decoder attached.

    It was trained without one, so a decode returns the byte-level markers the
    BPE was built on - ``ĠOnce Ġupon Ġa Ġtime`` - and the samples this script
    writes into the report were stored in exactly that form until 2026-09-25.
    The record's qualitative half is those four generations, and an artifact whose
    proof is unreadable is not a proof; `scripts/real_lm_infer.py` had attached
    the decoder at load time all along, so the two disagreed about what a sample
    even is. Attaching it here is the same one line, in the place that writes the
    record rather than the place that reads it.
    """
    from tokenizers import Tokenizer, decoders

    tokenizer = Tokenizer.from_file(str(ART / "tokenizer.json"))
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=6_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--tokens", type=int, default=40_000_000, help="token-stream size to encode")
    parser.add_argument("--stage", choices=("all", "corpus", "tokenizer", "tokens"), default="all")
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help=(
            "resume the weights but start Adam's moments fresh; required for a "
            "checkpoint whose optimizer state was stored by position"
        ),
    )
    args = parser.parse_args()

    train_path, valid_path = stage_corpus()
    if args.stage == "corpus":
        return 0
    stage_tokenizer(train_path)
    if args.stage == "tokenizer":
        return 0
    tokens_path = stage_tokens(train_path, args.tokens)
    if args.stage == "tokens":
        return 0
    report = stage_train(
        tokens_path,
        valid_path,
        args.max_steps,
        args.log_every,
        reset_optimizer=args.reset_optimizer,
    )
    _log(
        f"final: val loss {report['validation']['final_val_loss']:.3f} "
        f"(uniform floor {report['validation']['uniform_floor_loss']:.3f}), "
        f"perplexity {report['validation']['final_perplexity']:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
