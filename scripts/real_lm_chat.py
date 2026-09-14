"""Interactive chat-style inference with the trained TinyStories LM.

A REPL around ``scripts/real_lm_infer.py``'s loaded model: your lines become
part of an ongoing story, the model continues it, and tokens stream to the
terminal as they are generated. The model is a story-continuation LM, not an
instruction follower - it writes the next part of the story, which is the
honest framing for what this checkpoint can do.

Session commands (a line starting with ``/`` is never sent to the model):
  /new                 start a fresh story (clears the rolling context)
  /temp X              sampling temperature (0 = greedy/deterministic)
  /topk N              top-k sampling width (0 = disabled)
  /len N               tokens generated per turn
  /show                print the current rolling story so far
  /save FILE           write the full story to a text file
  /help                list commands
  /quit | /exit | /q   leave (Ctrl-D works too)

Usage:
  PYTHONPATH=src python scripts/real_lm_chat.py
  PYTHONPATH=src python scripts/real_lm_chat.py --temperature 0.7 --len 60
  echo "Once upon a time" | PYTHONPATH=src python scripts/real_lm_chat.py --once
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Reuse the inference module's loader so chat and one-shot inference always
# agree on config reconstruction, checkpoint path, and the byte-level decoder.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from real_lm_infer import DEFAULT_PROMPTS, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _print_header(device: torch.device, step: int, context_size: int) -> None:
    print("TinyStories LM - interactive story chat")
    print(f"checkpoint step {step} on {device}; context {context_size} tokens")
    print(
        "The model continues a story: type a line (or a few words) and it writes on.\n"
        "Type /help for commands, /quit to leave."
    )


def _print_help() -> None:
    print(
        "commands:\n"
        "  /new        fresh story (clears context)\n"
        "  /temp X     sampling temperature, 0 = greedy (now {temp})\n"
        "  /topk N     top-k width, 0 = off (now {topk})\n"
        "  /len N      tokens per turn (now {length})\n"
        "  /show       show the story so far\n"
        "  /save FILE  save the story to FILE\n"
        "  /quit       leave"
    )


def _stream_turn(
    model, tokenizer, device: torch.device, tokens: torch.Tensor, *, length: int, temperature: float, top_k: int
) -> torch.Tensor:
    """Generate ``length`` tokens, printing them as they arrive; return the new state."""
    generator = torch.Generator(device="cpu").manual_seed(int(time.time() * 1000) & 0x7FFFFFFF)
    context_size = model.config.context_size
    for _ in range(length):
        logits = model(tokens[:, -context_size:])[:, -1, :]
        if temperature <= 0:
            next_id = int(logits.argmax(dim=-1))
        else:
            if top_k and top_k < logits.shape[-1]:
                cutoff = logits.topk(top_k, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probs = torch.softmax(logits[0] / temperature, dim=-1)
            next_id = int(torch.multinomial(probs.cpu(), 1, generator=generator))
        piece = tokenizer.decode([next_id])
        print(piece, end="", flush=True)
        tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device)], dim=1)
    print(flush=True)
    return tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--len", type=int, default=60, help="tokens per turn")
    parser.add_argument(
        "--once", action="store_true",
        help="read one story-seed line from stdin, continue it, and exit (piped mode)",
    )
    args = parser.parse_args()

    model, tokenizer, device, payload = load_model()
    step = payload.get("step", "?")
    _print_header(device, step, model.config.context_size)

    # "tokens" holds a [1, T] LongTensor once a story is started, else None.
    state = {
        "tokens": None,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "length": args.len,
    }

    def start_new_story(seed_line: str | None = None) -> None:
        seed = seed_line if seed_line is not None else DEFAULT_PROMPTS[0]
        ids = tokenizer.encode(seed).ids
        state["tokens"] = torch.tensor([ids], dtype=torch.long, device=device)
        print(f"\n[story seed] {seed}", flush=True)

    if args.once:
        seed_line = sys.stdin.readline()
        if not seed_line.strip():
            seed_line = DEFAULT_PROMPTS[0]
        start_new_story(seed_line.strip())
        print("> ", end="", flush=True)
        _stream_turn(
            model, tokenizer, device, state["tokens"],
            length=state["length"], temperature=state["temperature"], top_k=state["top_k"],
        )
        return 0

    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0
        if not line:
            continue
        if line.startswith("/"):
            command, _, argument = line.partition(" ")
            command = command.lower()
            if command in {"/quit", "/exit", "/q"}:
                print("bye")
                return 0
            if command == "/new":
                if argument:
                    start_new_story(argument)
                else:
                    state["tokens"] = None
                    print("(context cleared - type your opening line)")
                continue
            if command == "/temp":
                try:
                    state["temperature"] = float(argument)
                    print(f"temperature {state['temperature']}")
                except ValueError:
                    print("usage: /temp 0.7")
                continue
            if command == "/topk":
                try:
                    state["top_k"] = int(argument)
                    print(f"top-k {state['top_k']}")
                except ValueError:
                    print("usage: /topk 50")
                continue
            if command == "/len":
                try:
                    state["length"] = max(1, int(argument))
                    print(f"tokens per turn {state['length']}")
                except ValueError:
                    print("usage: /len 60")
                continue
            if command == "/show":
                if state["tokens"] is None:
                    print("(no story yet)")
                else:
                    print(tokenizer.decode(state["tokens"][0].tolist()))
                continue
            if command == "/save":
                if state["tokens"] is None:
                    print("(no story yet)")
                    continue
                if not argument:
                    print("usage: /save story.txt")
                    continue
                path = Path(argument)
                path.write_text(tokenizer.decode(state["tokens"][0].tolist()))
                print(f"saved {len(state['tokens'][0])} tokens to {path}")
                continue
            if command == "/help":
                _print_help()
                continue
            print(f"unknown command {command!r}; /help lists commands")
            continue
        if state["tokens"] is None:
            # The user's first line seeds the story, then the model writes on.
            start_new_story(line)
        else:
            # Later lines become part of the story: append them to the rolling
            # context so the model continues from what the user actually said.
            user_ids = tokenizer.encode(line).ids
            state["tokens"] = torch.cat(
                [state["tokens"], torch.tensor([user_ids], device=device)], dim=1
            )
        print("lm> ", end="", flush=True)
        state["tokens"] = _stream_turn(
            model, tokenizer, device, state["tokens"],
            length=state["length"], temperature=state["temperature"], top_k=state["top_k"],
        )


if __name__ == "__main__":
    raise SystemExit(main())
