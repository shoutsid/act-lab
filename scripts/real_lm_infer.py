"""Run real inference with the trained TinyStories LM (scripts/train_real_lm.py).

Loads the checkpoint under artifacts/real-lm/checkpoints/latest.pt, rebuilds
the exact model config from artifacts/real-lm/report.json, and generates
completions from prompts. Byte-level decode is attached at load time (the
cached tokenizer was trained without a decoder, so raw output would show
byte-level markers otherwise).

Usage:
  PYTHONPATH=src python scripts/real_lm_infer.py --prompt "Once upon a time"
  PYTHONPATH=src python scripts/real_lm_infer.py --greedy --prompt "Tom and Lily"
  PYTHONPATH=src python scripts/real_lm_infer.py --score-text story.txt
  PYTHONPATH=src python scripts/real_lm_infer.py --demo
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from act.model import V0MoECausalLM

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "real-lm"

DEFAULT_PROMPTS = (
    "Once upon a time, there was a little",
    "Tom and Lily were playing in the",
    "The dog ran to the",
    "One day, Mom said,",
)


def load_model() -> tuple[V0MoECausalLM, object, torch.device, dict]:
    report = json.loads((ART / "report.json").read_text())
    spec = report["model"]
    from act.model import V0MoEConfig

    experts = spec["experts_per_layer"]
    experts_per_layer = (
        (int(experts),) * spec["depth"] if isinstance(experts, int) else tuple(experts)
    )
    config = V0MoEConfig(
        vocab_size=spec["vocab_size"],
        context_size=spec["context_size"],
        hidden_size=spec["hidden_size"],
        depth=spec["depth"],
        attention_heads=spec["attention_heads"],
        experts_per_layer=experts_per_layer,
        active_experts=spec["active_experts"],
        expert_width=spec["expert_width"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V0MoECausalLM(config).to(device)
    payload = torch.load(
        ART / "checkpoints" / "latest.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(payload["model"])
    model.eval()

    from tokenizers import Tokenizer, decoders

    tokenizer = Tokenizer.from_file(str(ART / "tokenizer.json"))
    # The cached tokenizer was trained without a decoder; attach the byte-
    # level one so generated text renders without byte-level markers.
    tokenizer.decoder = decoders.ByteLevel()
    return model, tokenizer, device, payload


@torch.no_grad()
def generate(
    model: V0MoECausalLM,
    tokenizer,
    device: torch.device,
    prompt: str,
    *,
    new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 50,
    seed: int = 1234,
) -> str:
    ids = tokenizer.encode(prompt).ids
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(new_tokens):
        logits = model(tokens[:, -model.config.context_size :])[:, -1, :]
        if temperature <= 0:
            next_id = int(logits.argmax(dim=-1))
        else:
            if top_k and top_k < logits.shape[-1]:
                cutoff = logits.topk(top_k, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probs = torch.softmax(logits[0] / temperature, dim=-1)
            next_id = int(torch.multinomial(probs.cpu(), 1, generator=generator))
        tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device)], dim=1)
    return tokenizer.decode(tokens[0].tolist())


@torch.no_grad()
def score_text(model: V0MoECausalLM, tokenizer, device: torch.device, text: str) -> dict:
    ids = tokenizer.encode(text).ids[: model.config.context_size + 1]
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(tokens[:, :-1])
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, model.config.vocab_size), tokens[:, 1:].reshape(-1)
    )
    return {"tokens": len(ids) - 1, "loss": float(loss), "perplexity": math.exp(float(loss))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--greedy", action="store_true", help="argmax decoding (temperature 0)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--demo", action="store_true", help="run the default prompt set")
    parser.add_argument("--score-text", type=Path, default=None, help="report loss/perplexity of a text file")
    args = parser.parse_args()

    model, tokenizer, device, payload = load_model()
    step = payload.get("step", "?")
    print(f"model loaded (step {step}) on {device}; vocab {model.config.vocab_size}, "
          f"context {model.config.context_size}")

    if args.score_text is not None:
        text = args.score_text.read_text(errors="ignore")
        print(json.dumps(score_text(model, tokenizer, device, text), indent=2))
        return 0

    temperature = 0.0 if args.greedy else args.temperature
    if args.demo or args.prompt is None:
        for prompt in DEFAULT_PROMPTS:
            print(f"\n--- {prompt!r} ---")
            print(generate(
                model, tokenizer, device, prompt,
                new_tokens=args.new_tokens, temperature=temperature,
                top_k=args.top_k, seed=args.seed,
            ))
        return 0
    print(generate(
        model, tokenizer, device, args.prompt,
        new_tokens=args.new_tokens, temperature=temperature,
        top_k=args.top_k, seed=args.seed,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
