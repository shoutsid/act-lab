"""Score any code model on the two suites this leg reports, in comparable units.

Every model - the project's student, the 1.3B teacher it was distilled from, and
the 164M published baseline it is measured against - is scored by this one file,
on one held-out text, by one protocol. That is the point of it: numbers produced
by three different scripts are three different numbers, and a comparison nobody
can re-run is not evidence.

Two suites, chosen because between them they cover "does it model code" and "can
it write code":

``bytes``
    Bits per byte on held-out Python. Each model tokenizes the *same text* with
    its own tokenizer and is scored on the same 512-token windows, so the
    denominators differ and the numerator is renormalised per byte. This is the
    only way a 32k-vocabulary teacher, a 32k student, and a 49k-vocabulary
    baseline can sit in one table: perplexity is a function of the tokenizer and
    bits per byte is not. Lower is better; a uniform distribution over 32,256
    tokens is the trivial reference point and ``scripts/train_code_lm.py`` records
    unigram and bigram references beside it.

``humaneval``
    HumanEval pass@1, executed. 164 problems; the model completes each prompt,
    the completion is truncated at the first plausible top-level boundary, and
    ``prompt + completion + test + check(entry_point)`` is run in a subprocess
    with a wall-clock limit. A problem counts as passed only if that subprocess
    exits zero. Not a proxy metric: this is the published benchmark, and the
    baseline's published figure (7.84% pass@1) is on the same 164 problems.

Usage:
  PYTHONPATH=src python scripts/code_bench.py --model hf:bigcode/tiny_starcoder_py
  PYTHONPATH=src python scripts/code_bench.py \\
      --model ours:artifacts/code-lm/checkpoints/match-distill.pt --scale match
  PYTHONPATH=src python scripts/code_bench.py --model hf:... --suite bytes   # one suite
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "code-lm"
HUMANEVAL = ART / "humaneval" / "HumanEval.jsonl.gz"
VAL_TEXT = ART / "val-text.txt"
RESULTS = ART / "bench"

MODEL_PREFIXES = ("hf", "ours")
"""The ``--model`` kinds. One definition, because two callers build these strings.

``scripts/push_code_bench.py`` renders them into a Kaggle kernel entry from a remote
machine, so a mismatch reaches the failure several minutes and one 2.7 GB download
later: the first push of that kernel died with "--model must be hf:<repo> or
ours:<checkpoint path>" after fetching the teacher, because the driver passed a bare
repository id. The tests compare the driver's strings against this tuple rather than
against the message.
"""

WINDOW = 512
"""Tokens per scoring window, the same for every model regardless of its own limit.
Holding it fixed is what makes the teacher's and the baseline's numbers a
comparison rather than two measurements."""

DEFAULT_BYTES_LIMIT = 4_000_000
"""Held-out bytes scored when nobody says otherwise, as one definition.

Two other files pin this number: ``run_code_lm_campaign.sh`` as its ``BENCH_BYTES``
default, and ``push_code_bench.py`` as the limit it asks Kaggle for. They have to agree,
because bits per byte is only comparable at one limit and the disagreement shows up at
table time - after the local card has spent nine hours and the Kaggle run has spent GPU
quota. ``tests/test_code_lm_campaign.py`` reads all three."""

STOP_SEQUENCES = ("\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\n@", "\n\n\n")
"""Truncation points for a completion, as the published code-eval harnesses use.
Without them a model that keeps emitting plausible Python after the answer gets
charged for its verbosity, which measures generation length, not correctness."""

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95
"""The decoding protocol, and it is not a free choice: these are the values the
``bigcode-evaluation-harness`` uses - ``top_p=0.95``, ``temperature=0.2``,
``max_length_generation=512`` - and ``bigcode/tiny_starcoder_py``'s published 7.84%
HumanEval pass@1 was measured under them.

The first version of this file asked for 256 tokens and sampled from the full
multinomial at temperature 0.2 without a nucleus cut. Both differences push the number
*down* against the published one: a 164M model writing a function often needs more than
256 tokens, and sampling a 32k-token tail at temperature 0.2 without ``top_p`` spends
part of its budget on tokens the harness would never have drawn. Comparing our baseline
row against 7.84% while running a stricter protocol measures the protocol.

What is still different is the estimator, and it cannot be fixed by a flag: the
published figure comes from the harness's many-sample unbiased ``pass@1`` estimate,
while this scores one sample per problem. That is not a bias but it is noise - about
+/- 1.7 percentage points at a rate of 7.84% over 164 problems - so the published number
is a reference for the order of magnitude, not a target to land on.
"""


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def display_path(path: Path) -> str:
    """A checkpoint's path as a reader can find it: repository-relative when it is inside.

    ``--model ours:artifacts/...`` is what the campaign passes and ``--model
    ours:/abs/path`` is what a reader types, and the report needs one spelling for both.
    Calling ``relative_to`` on the argument as given raises ``ValueError`` for the first
    (a relative path against an absolute root) - after the model has loaded and the whole
    scan has run, which is how this reached a smoke test rather than a test suite.
    """
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def held_out_text(byte_limit: int = 0) -> str:
    """The held-out Python text, optionally cut to a fixed prefix of ``byte_limit`` bytes.

    A prefix, not a sample: scoring the whole 28.7 MB set costs the 1.3B teacher
    over two hours per pass, and a bits-per-byte estimate over 1.29M tokens is
    already stable well past the fourth decimal. What matters is that every model
    is cut at exactly the same byte count, so the optimisation is in the cost and
    not in the comparison. The limit is recorded in each report for that reason -
    two numbers produced at different limits are not the same measurement.
    """
    if not VAL_TEXT.is_file():
        raise SystemExit(f"{VAL_TEXT} is missing; run scripts/train_code_lm.py --stage tokens first")
    text = VAL_TEXT.read_text()
    if byte_limit and len(text.encode()) > byte_limit:
        # Decode with errors ignored so the cut cannot land inside a character.
        return text.encode()[:byte_limit].decode("utf-8", errors="ignore")
    return text


# --- suite: bits per byte ---------------------------------------------------


def bits_per_byte(nats: float, text_bytes: int) -> float:
    return nats / math.log(2) / text_bytes


def model_logits(model, tokens: torch.Tensor) -> torch.Tensor:
    """Logits from either a ``transformers`` causal LM or this project's model.

    The two disagree about the return type - ``CausalLMOutputWithPast`` against a
    plain tensor - and this file scores both kinds, so the difference is absorbed
    here rather than at each call site.
    """
    out = model(tokens)
    return out.logits if hasattr(out, "logits") else out


@torch.no_grad()
def window_nlls(model, ids: np.ndarray, device: torch.device, batch: int = 4) -> tuple[float, list[float]]:
    """Negative log-likelihood in nats, in total and per 512-token window.

    Windows do not overlap and the first token of each window has no context and is
    not scored, so the sum covers every token but one per window. Every model carries
    that same one-token-per-window omission, which is why the comparison survives it.

    The per-window list is the part that makes the headline *testable* rather than
    merely precise. A total to four decimals cannot distinguish an effect from
    numerical dust, and two students that share a tokenizer are scored on identical
    windows - so their difference can be tested as a paired one (same text, same
    window boundaries, one difference per window) instead of compared as two
    independent means. A paired test on ~2,500 windows resolves effects that a
    four-decimal aggregate hides. It is only valid between models that tokenize the
    text the same way, which is why it is recorded per model and left to the caller
    to pair correctly.
    """
    window_values: list[float] = []
    starts = range(0, len(ids) - 1, WINDOW)
    pending: list[np.ndarray] = []
    # Every window is WINDOW + 1 tokens except the last, which is whatever is left of
    # the text. A batch is a stacked array, so mixing the two lengths is a ValueError
    # rather than a measurement - which is how the Kaggle byte pass died at 4,000,000
    # bytes: 1,106,839 tokens for that tokenizer, a 407-token tail, and
    # "all input arrays must have the same shape". The tail is scored on its own.
    full = WINDOW + 1

    def flush(block: list[np.ndarray]) -> None:
        if not block:
            return
        seq = np.stack(block).astype(np.int64)
        tokens = torch.from_numpy(seq).to(device)
        logits = model_logits(model, tokens[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), tokens[:, 1:].reshape(-1), reduction="none"
        )
        per_window = loss.reshape(len(block), -1).sum(dim=1)
        window_values.extend(float(value) for value in per_window)

    for start in starts:
        window = ids[start : start + WINDOW + 1]
        if len(window) < 8:
            continue
        if len(window) != full:
            flush(pending)
            pending = []
            flush([window])
            continue
        pending.append(window)
        if len(pending) >= batch:
            flush(pending)
            pending = []
    if pending:
        flush(pending)
    if not window_values:
        raise SystemExit("held-out text is too short to score")
    return float(sum(window_values)), window_values


@torch.no_grad()
def score_hf(source: str, text: str, device: torch.device) -> dict:
    """Score a ``transformers`` causal LM from a repository id or a local directory.

    Both are accepted because the teacher's weights are fetched once into
    ``artifacts/code-lm/teacher`` and passing the repository id instead would pull
    2.7 GB a second time from the hub. The architecture is recorded beside the
    numbers: a bits-per-byte figure that does not say which implementation
    produced it is not reproducible.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    _log(f"loading {source} ...")
    config = AutoConfig.from_pretrained(source)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float16)
    model.eval().to(device)
    parameters = sum(p.numel() for p in model.parameters())
    architecture = list(getattr(config, "architectures", []) or [config.model_type])
    ids = np.asarray(tokenizer(text, add_special_tokens=False)["input_ids"], dtype=np.int32)
    _log(
        f"{source}: {architecture} {parameters/1e6:.1f}M parameters, "
        f"{len(ids):,} tokens for {len(text.encode()):,} bytes"
    )
    nats, per_window = window_nlls(model, ids, device)
    del model
    torch.cuda.empty_cache()
    return {
        "model": source,
        "kind": "hf",
        "architecture": architecture,
        # Identity of the tokenization the ids came from. Two models are only
        # pairable window by window if this agrees, and ``scripts/code_lm_table.py``
        # refuses to pair them otherwise.
        "tokenizer": getattr(tokenizer, "name_or_path", None) or source,
        "parameters": int(parameters),
        "tokens": int(len(ids)),
        "nats": nats,
        "held_out_bytes": len(text.encode()),
        "bits_per_byte": bits_per_byte(nats, len(text.encode())),
        "loss_per_token": nats / max(1, len(ids) - len(ids) // WINDOW),
        "windows": len(per_window),
        "window_nlls": per_window,
    }


@torch.no_grad()
def score_ours(checkpoint: Path, scale: str, text: str, device: torch.device) -> dict:
    """Score a student checkpoint using the *same* text protocol as the suites above.

    Deliberately re-tokenizes the held-out text here rather than reading the
    training script's ``val-tokens.npy``: that stream has a boundary token between
    files, which no other model's tokenization carries, and it would move the
    student's denominator relative to everyone else's.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_code_lm as trainer

    from act.model.v0_moe import estimated_parameter_count

    config = trainer.student_config(scale)
    model = trainer.V0MoECausalLM(config).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    total, active = estimated_parameter_count(config)
    tokenizer = trainer.shared_tokenizer()
    ids = np.asarray(tokenizer.encode(text).ids, dtype=np.int32)
    _log(
        f"{checkpoint.name}: step {payload.get('step')}, {total/1e6:.1f}M total / "
        f"{active/1e6:.1f}M active parameters, {len(ids):,} tokens"
    )
    nats, per_window = window_nlls(model, ids, device)
    del model
    torch.cuda.empty_cache()
    return {
        "model": f"ours:{scale}",
        "kind": "ours",
        "checkpoint": display_path(checkpoint),
        "step": payload.get("step"),
        "tokenizer": trainer.tokenizer_id(),
        "parameters": int(total),
        "active_parameters": int(active),
        "tokens": int(len(ids)),
        "nats": nats,
        "held_out_bytes": len(text.encode()),
        "bits_per_byte": bits_per_byte(nats, len(text.encode())),
        "loss_per_token": nats / max(1, len(ids) - len(ids) // WINDOW),
        "windows": len(per_window),
        "window_nlls": per_window,
    }


# --- suite: HumanEval, executed --------------------------------------------


def load_humaneval() -> list[dict]:
    with gzip.open(HUMANEVAL, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def truncate_at_stop(completion: str) -> str:
    cut = len(completion)
    for stop in STOP_SEQUENCES:
        found = completion.find(stop)
        if found != -1:
            cut = min(cut, found)
    return completion[:cut]


def sample_token(logits: torch.Tensor, temperature: float, top_p: float, generator) -> int:
    """One token from one row of logits, at the project's decoding protocol.

    It lives here rather than inside the student's generator so the student and the two
    ``transformers`` models are drawn from the same distribution of choices: a comparison
    where one arm samples differently from another is measuring the sampler. Greedy at
    ``temperature <= 0``; otherwise nucleus sampling, where the kept set is the smallest
    prefix of the sorted distribution whose mass reaches ``top_p`` and the rest is
    dropped and the survivor renormalised.
    """
    # Nothing here is ever differentiated, and the callers hand over whatever the model
    # returned - which for the student carries ``requires_grad``. Converting that to a
    # scalar warns once per token, i.e. 512 times per problem per model.
    logits = logits.detach()
    if temperature <= 0:
        return int(logits.argmax(dim=-1))
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        ordered, places = torch.sort(probabilities, descending=True)
        before = torch.cumsum(ordered, dim=-1) - ordered
        # Keep a token when the mass *before* it is still under the cut, which is the
        # harness's rule and keeps the first token unconditionally.
        kept = torch.where(before < top_p, ordered, torch.zeros_like(ordered))
        if float(kept.sum()) <= 0:
            kept = ordered[:1]
            places = places[:1]
        probabilities = torch.zeros_like(probabilities).scatter_(0, places, kept)
        probabilities = probabilities / probabilities.sum()
    return int(torch.multinomial(probabilities.cpu(), 1, generator=generator))


def run_program(program: str, timeout: float) -> bool:
    """Execute one assembled HumanEval program. Pass means exit code zero.

    Run in a scratch directory with the environment stripped down and a hard
    wall-clock limit. Generated code is untrusted by construction - it is whatever
    the model produced - so the limit is the safety mechanism that matters, and a
    subprocess timeout is the only reliable way to enforce one against a loop the
    model wrote.
    """
    with tempfile.TemporaryDirectory() as workdir:
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": workdir,
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        try:
            finished = subprocess.run(
                [sys.executable, "-c", program],
                cwd=workdir,
                env=environment,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False
        return finished.returncode == 0


def score_humaneval(generate, problems: list[dict], timeout: float, samples: int, seed: int) -> dict:
    """pass@1 over HumanEval, with a per-problem record of what was generated.

    Samples per problem default to one, which makes pass@1 the fraction of
    problems whose single sample passes. That is the published definition, and it
    is worth stating plainly: at one sample the estimate is high-variance, so the
    per-problem list is written to the record beside the aggregate rather than
    being summarised away.
    """
    passed = 0
    records = []
    for index, problem in enumerate(problems):
        entry = problem["entry_point"]
        completions = []
        for sample in range(samples):
            completion = generate(problem["prompt"], seed + index * 1000 + sample)
            completions.append(completion)
        oks = []
        for completion in completions:
            program = (
                problem["prompt"]
                + truncate_at_stop(completion)
                + "\n"
                + problem["test"]
                + f"\ncheck({entry})"
            )
            oks.append(run_program(program, timeout))
        if any(oks):
            passed += 1
        records.append({"task_id": problem["task_id"], "passed": bool(any(oks)), "samples": oks})
        if (index + 1) % 20 == 0 or index + 1 == len(problems):
            _log(f"  humaneval {index + 1}/{len(problems)}  pass@{samples} {passed/(index+1):.3f}")
    return {
        "problems": len(problems),
        "samples_per_problem": samples,
        "passed": passed,
        "pass_at_1": passed / max(1, len(problems)),
        "timeout_seconds": timeout,
        "records": records,
    }


def make_hf_generator(
    repo: str, device: torch.device, max_new_tokens: int, temperature: float, top_p: float = DEFAULT_TOP_P
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float16).eval().to(device)

    def generate(prompt: str, seed: int) -> str:
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True)

    return generate, tokenizer, model


def make_ours_generator(
    checkpoint: Path,
    scale: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float = DEFAULT_TOP_P,
):
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_code_lm as trainer

    config = trainer.student_config(scale)
    model = trainer.V0MoECausalLM(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    model.eval()
    tokenizer = trainer.shared_tokenizer()

    def generate(prompt: str, seed: int) -> str:
        # Decoding one prompt at a time: the student has no KV cache, so each step
        # recomputes the window, and batching would only trade that for padding.
        ids = tokenizer.encode(prompt).ids
        tokens = torch.tensor([ids[-WINDOW:]], dtype=torch.long, device=device)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        produced: list[int] = []
        # Inference, and 512 forwards deep: without this the whole window's activations
        # are retained per step for a backward pass that never comes. The ``hf:`` path
        # gets the same thing from ``model.generate``.
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = model(tokens[:, -WINDOW:])[:, -1, :]
                # The same sampler the transformers rows go through, so the protocol is
                # one protocol rather than one per model family.
                next_id = sample_token(logits[0], temperature, top_p, generator)
                produced.append(next_id)
                tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device)], dim=1)
        return tokenizer.decode(produced)

    return generate, tokenizer, model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="hf:<repo> or ours:<checkpoint path>")
    parser.add_argument("--scale", default="match", help="student scale, for --model ours:")
    parser.add_argument("--suite", default="bytes,humaneval", help="comma-separated subset of {bytes,humaneval}")
    parser.add_argument("--samples", type=int, default=1, help="HumanEval samples per problem")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help="the published harness's max_length_generation, 512")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P,
                        help="nucleus cut; 1.0 disables it and is not the published protocol")
    parser.add_argument("--timeout", type=float, default=10.0, help="seconds per HumanEval program")
    parser.add_argument("--limit", type=int, default=0, help="first N HumanEval problems (0 = all)")
    parser.add_argument(
        "--bytes-limit", type=int, default=DEFAULT_BYTES_LIMIT,
        help="held-out bytes to score (0 = all); identical across models by construction",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    requested = {name.strip() for name in args.suite.split(",") if name.strip()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kind, _, target = args.model.partition(":")
    if kind not in MODEL_PREFIXES or not target:
        raise SystemExit(f"--model must be one of {', '.join(f'{p}:<value>' for p in MODEL_PREFIXES)}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "model": args.model,
        "suite": sorted(requested),
        "device": str(device),
        # The device *name*, not only its type. ``str(torch.device("cuda"))`` is "cuda"
        # on every accelerator, so a table mixing a T4 with a GTX 1070 looks like one
        # device and the note explaining the difference never fires.
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }

    if "bytes" in requested:
        text = held_out_text(args.bytes_limit)
        report["bytes_limit"] = args.bytes_limit
        _log(f"held-out text: {len(text.encode()):,} bytes")
        report["bytes"] = score_hf(target, text, device) if kind == "hf" else score_ours(Path(target), args.scale, text, device)
        report["bytes"]["bytes_limit"] = args.bytes_limit
        _log(f"bits/byte {report['bytes']['bits_per_byte']:.4f}")

    if "humaneval" in requested:
        problems = load_humaneval()
        if args.limit:
            problems = problems[: args.limit]
        if kind == "hf":
            generate, _, model = make_hf_generator(
                target, device, args.max_new_tokens, args.temperature, args.top_p
            )
            parameters = sum(p.numel() for p in model.parameters())
        else:
            generate, _, model = make_ours_generator(
                Path(target), args.scale, device, args.max_new_tokens, args.temperature, args.top_p
            )
            parameters = None
        # Recorded at the top level, because two pass@1 numbers produced under different
        # decoding are two measurements - the same reason ``bytes_limit`` is compared.
        report["decoding"] = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "samples_per_problem": args.samples,
            "problems": len(problems),
        }
        _log(
            f"human eval: {len(problems)} problems, {args.samples} sample(s), "
            f"{args.max_new_tokens} new tokens, temperature {args.temperature}, top_p {args.top_p}"
        )
        report["humaneval"] = score_humaneval(generate, problems, args.timeout, args.samples, args.seed)
        if parameters is not None:
            report["humaneval"]["parameters"] = int(parameters)
        _log(f"pass@1 {report['humaneval']['pass_at_1']:.4f} on {len(problems)} problems")

    label = args.tag or args.model.replace("/", "_").replace(":", "-")
    out = RESULTS / f"{label}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _log(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
