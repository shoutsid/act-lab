"""Condense a large code model into the project's own sparse architecture.

The TinyStories leg (``scripts/train_real_lm.py``) proved the *path*: a real
corpus, a real tokenizer, real next-token training on the GPU, held-out loss. It
did not answer the question the project is actually about - whether a large
model's behaviour can be *condensed* into a smaller, sparse, learned-topology
network that does the heavy lifting with a fraction of the parameters.

This script is that leg. Everything here is measured against the same held-out
Python text by every model involved, so the numbers in the record are
comparable rather than merely adjacent:

  teacher   ``deepseek-ai/deepseek-coder-1.3b-base`` - 1.3B dense parameters on
            2T tokens of code, public, and native to ``transformers`` so a reader
            can load it in one call. Two rejected alternatives are worth naming.
            ``Qwen/Qwen2.5-Coder-1.5B`` is stronger and newer, but its 152k
            vocabulary would make the student's embedding matrix most of its
            parameters - this teacher's 32k leaves room for depth at the same
            total. ``bigcode/santacoder`` is the Python specialist
            ``tiny_starcoder_py`` descends from, but its repository ships
            GPT-2-with-multi-query-attention modelling code that no longer runs
            on transformers 5.x (it dies on ``transformers.onnx``, removed in
            v5), and a teacher that cannot be loaded is not a reproducible
            comparison.
  student   the project's own ``act.model.V0MoECausalLM`` - sparse, and small
            enough that the parameters it *activates* per token are a fraction
            of the teacher's.
  baseline  ``bigcode/tiny_starcoder_py`` - 164M dense, in the same size class
            as the ``match`` student, and trained on 100B tokens on 32 A100s. It
            is the number this leg is measured against, and it is also the
            honest ceiling: see the note on budget at the end of this docstring.
  metric    bits per byte on held-out Python. Deliberately not perplexity:
            perplexity is a function of the tokenizer, so a 49k-vocab teacher and
            a different student tokenizer cannot be compared with it. Bits per
            byte is tokenizer-independent and is the unit the field reports code
            models in.

Distillation is done from a *cached* teacher, not from a live one. The teacher
scores a fixed grid of non-overlapping 512-token windows once, and the student
then trains against those saved distributions. That is not only cheaper - it
removes a real confound: because both models see the identical window with
causal attention, a teacher position and the student position predicting the
same token have byte-for-byte the same context. The alternative (teacher on the
GPU beside the student) is also impossible here: this machine has an 8 GiB card
that cannot hold both.

What the cache holds, per position, is the teacher's top-32 tokens. Two details
keep that faithful:

* the stored value is ``log p_i - log p_top1`` (a non-positive number close to
  zero), not ``log p_i``. Float16 resolves the former to ~1e-3 and the latter to
  ~1e-2 near the tail, and the difference is free to avoid.
* ``log p_top1`` and the *residual* mass (``1 - sum of the top-32``) are stored
  per position too, so the KL divergence is computed over the whole vocabulary
  rather than being truncated to 32 tokens and silently renormalised.

Usage:
  PYTHONPATH=src python scripts/train_code_lm.py                      # all stages
  PYTHONPATH=src python scripts/train_code_lm.py --stage tokens        # up to tokens
  PYTHONPATH=src python scripts/train_code_lm.py --stage targets       # score teacher
  PYTHONPATH=src python scripts/train_code_lm.py --mode distill --scale toy \\
      --max-steps 200                                                  # smoke
  PYTHONPATH=src python scripts/train_code_lm.py --fetch-teacher       # 4.6 GB pull
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

from act.model.v0_moe import V0MoEConfig, V0MoECausalLM, estimated_parameter_count

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "code-lm"
CORPUS_DIR = ART / "corpus"
TEACHER_DIR = ART / "teacher"
TARGETS_DIR = ART / "targets"
CHECKPOINT_DIR = ART / "checkpoints"

CORPUS_URL = (
    "https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid"
    "/resolve/main/file-000000000054.json.gz"
)
CORPUS_FILE = "codeparrot-clean-valid.json.gz"

TEACHER_REPO = "deepseek-ai/deepseek-coder-1.3b-base"
BASELINE_REPO = "bigcode/tiny_starcoder_py"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{name}"

CONTEXT_SIZE = 512
SEED = 11
TOP_K = 32
"""Teacher tokens kept per position. The tail is not discarded - see ``residual``."""

LEARNING_RATE = 3e-4
WARMUP_STEPS = 100
EVAL_EVERY = 250
CHECKPOINT_EVERY = 500
DISTILL_WEIGHT = 0.5
"""Weight on the KL term in ``--mode distill``; the rest is ordinary cross-entropy.

Keeping cross-entropy in the objective matters. A pure-KL student may learn the
teacher's *ranking* over its top-32 candidates while putting its remaining
probability mass nowhere useful, which shows up as worse bits per byte than the
loss curve suggests.
"""

BUDGET_NOTE = """
The comparison this script supports is *matched budget*, not matched published
scores, and the difference is the whole honest ceiling of the result.
``bigcode/tiny_starcoder_py`` saw 100B tokens on 32 A100s in 18 hours. This box is
one GTX 1070 with 8 GiB, and the teacher-scoring pass alone runs at roughly a
thousand tokens a second, so the token budget here is three orders of magnitude
smaller. Nothing in this file closes that gap, and a run that lands near 0.18
HumanEval pass@1 - the published MultiPL figure for a teacher of this size -
would be evidence of a bug, not of distillation working. What the leg can show
is whether the project's sparse architecture is a *viable container* for a large
model's behaviour, measured on the same held-out Python as every model it is
compared with.
"""

SCALES: dict[str, dict] = {
    # (hidden, depth, heads, experts, active, width) at vocab 49,280, context 512.
    "toy": dict(hidden_size=384, depth=6, attention_heads=6, expert_count=8, active_experts=2, expert_width=384),
    "small": dict(hidden_size=512, depth=8, attention_heads=8, expert_count=8, active_experts=2, expert_width=512),
    "match": dict(hidden_size=768, depth=12, attention_heads=12, expert_count=8, active_experts=2, expert_width=768),
}
"""Student sizes. ``match`` is sized against ``bigcode/tiny_starcoder_py`` (164M
parameters, dense) so the comparison is between two models of the same class;
``toy`` exists so the pipeline can be smoke-tested in minutes rather than hours."""


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --- fetching ---------------------------------------------------------------
#
# A single connection to the HuggingFace CDN moves this box's ~150 KB/s, which
# would make the teacher's 4.6 GB an eight-hour download. Eight parallel range
# requests measure 5.6 MB/s - a 37x difference for twenty lines. Measured, not
# guessed: the corpus landed at 3.7 MB/s single-connection and the teacher at
# 150 KB/s, and the difference is which host the redirect lands on.


def fetch_parallel(url: str, out: Path, workers: int = 8) -> Path:
    """Fetch ``url`` to ``out`` over ``workers`` range requests, resuming."""
    out.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(request, timeout=60) as head:
        total = int(head.headers["Content-Length"])
    have = out.stat().st_size if out.is_file() else 0
    if have == total:
        _log(f"cached: {out.name} ({total / 1e6:.1f} MB)")
        return out
    if have > total:
        # A longer file than the one being fetched cannot be resumed into: the
        # spans would be empty and the truncate below would leave a corrupt mix.
        _log(f"{out.name} is {have:,} bytes but the source is {total:,}; refetching from zero")
        have = 0
    handle = out.open("w+b" if not have else "r+b")
    handle.truncate(total)
    span_size = max(1 << 22, (total - have) // (workers * 4))
    spans = [(start, min(start + span_size - 1, total - 1)) for start in range(have, total, span_size)]
    done = [have]
    started = time.time()

    def fetch(span: tuple[int, int]) -> None:
        start, end = span
        ranged = urllib.request.Request(url, headers={"User-Agent": "curl/8", "Range": f"bytes={start}-{end}"})
        last: Exception | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(ranged, timeout=120) as response:
                    payload = response.read()
                if len(payload) != end - start + 1:
                    raise IOError(f"short read {len(payload)} != {end - start + 1}")
                handle.seek(start)
                handle.write(payload)
                done[0] += len(payload)
                return
            except Exception as error:  # noqa: BLE001 - retried, then re-raised
                last = error
                time.sleep(2 + attempt * 3)
        raise SystemExit(f"range {start}-{end} failed after 4 attempts: {last}")

    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(fetch, span) for span in spans]
        for future in futures:
            future.result()
    handle.close()
    _log(
        f"fetched {out.name}: {total / 1e6:.1f} MB in {time.time() - started:.0f}s "
        f"({(total - have) / max(1e-9, time.time() - started) / 1e6:.2f} MB/s)"
    )
    return out


# --- stage 1: corpus --------------------------------------------------------


def corpus_path() -> Path:
    return CORPUS_DIR / CORPUS_FILE


def stage_corpus() -> Path:
    path = corpus_path()
    if not path.is_file():
        _log(f"downloading {CORPUS_FILE} ...")
        fetch_parallel(CORPUS_URL, path)
    return path


HOLD_EVERY = 20
"""One file in twenty is held out. Strided, not a contiguous tail, and not random.

The corpus is ordered by repository and path, so a contiguous tail would put
whole projects on one side of the split and every neighbouring file of the same
project on the other. A stride keeps the held-out files spread across the
corpus, and it is a fixed function of the input rather than of a seed.
"""


def stream_cache_usable(token_count: int, have: int, manifest: dict, current_tokenizer_id: str) -> bool:
    """Is a cached token stream still valid for a request of ``token_count``?

    There are two ways a cache goes stale and only one of them looks stale. A
    stream shorter than the request obviously needs rebuilding. A stream of the
    right *length* built by a different tokenizer does not - and it is worse,
    because the ids in it are perfectly well-formed positions in some other BPE,
    so training on them raises nothing and means nothing. This leg swapped its
    teacher (and with it the tokenizer) mid-build, which is what the check is for.

    Takes the current id as an argument rather than reading the teacher's
    tokenizer off disk, so the decision can be tested without the model present -
    the teacher weights are multi-gigabyte and deliberately not committed.
    """
    if manifest.get("tokenizer_id") != current_tokenizer_id:
        return False
    return have >= token_count or bool(manifest.get("complete"))


def iter_records(path: Path):
    """Yield each file's code from the gzipped JSON-lines corpus.

    Streamed rather than returned as a list: the corpus is 597 MB of source, and
    a 16M-token budget wants the first few thousand files of it, not all 61,373.
    """
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield json.loads(line)["content"]


def encode_flat(tokenizer, records: list[str], boundary: int) -> np.ndarray:
    """Concatenate records into one flat stream, one boundary token between."""
    pieces: list[np.ndarray] = []
    for batch_start in range(0, len(records), 2_000):
        for encoding in tokenizer.encode_batch(records[batch_start : batch_start + 2_000]):
            ids = np.asarray(encoding.ids, dtype=np.uint16)
            if ids.size >= 8:
                pieces.append(ids)
    if not pieces:
        return np.empty(0, dtype=np.uint16)
    total = sum(piece.size for piece in pieces) + len(pieces) - 1
    flat = np.empty(total, dtype=np.uint16)
    cursor = 0
    for index, ids in enumerate(pieces):
        if index:
            flat[cursor] = boundary
            cursor += 1
        flat[cursor : cursor + ids.size] = ids
        cursor += ids.size
    return flat


# --- stage 2: tokenizer -----------------------------------------------------


def teacher_tokenizer_path() -> Path:
    return TEACHER_DIR / "tokenizer.json"


def shared_tokenizer():
    """The teacher's tokenizer, used unchanged by the student as well.

    Sharing it is what makes soft-label distillation exact instead of
    approximate: the teacher's top-32 token ids are the student's token ids, so
    no probability mass has to be re-partitioned across two vocabularies. The
    baseline keeps a different tokenizer, and that is fine - the headline metric
    is bits per byte for exactly this reason.
    """
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(teacher_tokenizer_path()))


def vocab_size() -> int:
    return shared_tokenizer().get_vocab_size()


def tokenizer_id() -> str:
    """Identity of the tokenizer every derived stream was built with.

    Recorded in the manifest and checked before a cached stream is reused,
    because swapping the teacher silently invalidates every derived artifact: the
    ids in ``tokens.npy`` are positions in one specific BPE, and a stream encoded
    with another one is not merely stale, it is wrong in a way that trains. This
    guard exists because that swap happened during development.
    """
    import hashlib

    digest = hashlib.sha256(teacher_tokenizer_path().read_bytes()).hexdigest()[:16]
    return f"{TEACHER_REPO}@{digest}"


def teacher_vocab_size() -> int:
    """The teacher's output width, taken from the model's config rather than the BPE.

    These are not always the same number: a tokenizer can emit fewer ids than the
    model has output rows, because vocabularies routinely reserve placeholders
    the BPE never produces (SantaCoder's is 49,157 against 49,280 rows). Sizing
    the student to the tokenizer's number would let a teacher top-k that lands on
    a reserved row index past the student's logits; sized to the model's, every
    stored id is in range by construction.
    """
    return int(json.loads((TEACHER_DIR / "config.json").read_text())["vocab_size"])


DOCUMENT_SEPARATORS = ("<|endoftext|>", "<｜begin▁of▁sentence｜>", "<s>")
"""Names different tokenizer families give the token that starts a document.

Asked for by name, then by config id, because the two teachers considered here
name it differently and neither spelling is universal: StarCoder-family
vocabularies carry ``<|endoftext|>``, DeepSeek's carries
``<｜begin▁of▁sentence｜>``. The file separator has to be a real token of the
stream's own tokenizer, so hardcoding one spelling silently breaks the leg when
the teacher changes - which it did during development.
"""


def boundary_token_id() -> int:
    """The token separating one corpus file from the next in the flat stream."""
    raw = json.loads(teacher_tokenizer_path().read_text())
    for token in raw.get("added_tokens", []):
        if token["content"] in DOCUMENT_SEPARATORS:
            return int(token["id"])
    config = json.loads((TEACHER_DIR / "config.json").read_text())
    for key in ("bos_token_id", "eos_token_id"):
        if config.get(key) is not None:
            return int(config[key])
    raise SystemExit("no document-separator token in the teacher tokenizer or config")


# --- stage 3: token stream --------------------------------------------------


def held_out_text(records: list[str]) -> str:
    """The canonical held-out text every model is scored on.

    Written out rather than left implicit, because bits per byte divides by the
    *bytes of this text* and every model tokenizes it differently. A model's own
    token count is not a common denominator; the byte count of the text is.
    """
    return "\n".join(records)


def stage_tokens(token_count: int, force: bool = False) -> tuple[Path, Path]:
    ART.mkdir(parents=True, exist_ok=True)
    train_path = ART / "tokens.npy"
    val_path = ART / "val-tokens.npy"
    text_path = ART / "val-text.txt"
    if train_path.is_file() and val_path.is_file() and text_path.is_file() and not force:
        have = int(np.load(train_path, mmap_mode="r").shape[0])
        manifest_path = ART / "tokens-manifest.json"
        manifest = {}
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
            except (ValueError, OSError):
                manifest = {}
        if stream_cache_usable(token_count, have, manifest, tokenizer_id()):
            _log(f"token streams cached: {have:,} train tokens, {val_path.stat().st_size:,} held-out bytes")
            return train_path, val_path
        _log("cached token streams cannot be reused; re-encoding")
    tokenizer = shared_tokenizer()
    boundary = boundary_token_id()

    # Two passes over the compressed corpus. The held-out split has to be a
    # property of the corpus, not of the token budget: encoding in one pass and
    # stopping at the budget would hold out only the first slice of the file and
    # silently change the evaluation set whenever --tokens changed. Decompressing
    # twice costs about twenty seconds and removes that coupling.
    held: list[str] = []
    files_seen = 0
    for index, content in enumerate(iter_records(corpus_path())):
        if not index % HOLD_EVERY:
            held.append(content)
        files_seen += 1
    text = held_out_text(held)
    text_path.write_text(text)
    held_tokens = encode_flat(tokenizer, held, boundary)
    np.save(val_path, held_tokens)
    _log(
        f"held out {len(held):,} of {files_seen:,} files: {len(text.encode()):,} bytes -> "
        f"{held_tokens.shape[0]:,} tokens"
    )

    pieces: list[np.ndarray] = []
    total = 0
    batch: list[str] = []
    stream_done = False
    for index, content in enumerate(iter_records(corpus_path())):
        if not index % HOLD_EVERY:
            continue
        batch.append(content)
        if len(batch) >= 2_000:
            pieces.append(encode_flat(tokenizer, batch, boundary))
            total += int(pieces[-1].shape[0]) + 1
            batch = []
            _log(f"  encoded {total:,} tokens")
            if total >= token_count:
                stream_done = True
                break
    if not stream_done and batch:
        pieces.append(encode_flat(tokenizer, batch, boundary))
        total += int(pieces[-1].shape[0]) + 1
    complete = total < token_count
    if complete:
        _log(f"corpus exhausted at {total:,} tokens (asked for {token_count:,}); using all of it")
    flat = np.concatenate([piece for piece in pieces if piece.size]) if pieces else np.empty(0, dtype=np.uint16)
    np.save(train_path, flat)
    manifest = {
        "corpus": CORPUS_FILE,
        "source": CORPUS_URL,
        "files": files_seen,
        "held_out_files": len(held),
        "hold_every": HOLD_EVERY,
        "boundary_token": boundary,
        "train_tokens": int(flat.shape[0]),
        "held_out_tokens": int(held_tokens.shape[0]),
        "held_out_bytes": len(text.encode()),
        "vocab_size": vocab_size(),
        "teacher_vocab_size": teacher_vocab_size(),
        "tokenizer_id": tokenizer_id(),
        "context_size": CONTEXT_SIZE,
        "windows": window_count(int(flat.shape[0])),
        "complete": bool(complete),
    }
    (ART / "tokens-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _log(f"token stream saved: {flat.shape[0]:,} tokens -> {train_path}")
    return train_path, val_path


# --- stage 4: teacher targets ----------------------------------------------


def window_count(stream_tokens: int) -> int:
    return stream_tokens // CONTEXT_SIZE


def targets_shapes(stream_tokens: int) -> tuple[tuple[int, int, int], tuple[int, int]]:
    windows = window_count(stream_tokens)
    return (windows, CONTEXT_SIZE - 1, TOP_K), (windows, CONTEXT_SIZE - 1)


def _nearest_writable(path: Path) -> bool:
    """Is ``path`` (or its nearest existing ancestor) writable?

    Wrapped in ``try`` because ``Path.exists()`` does not answer ``False`` for a
    path under a directory the process cannot *read*: it raises
    ``PermissionError``. That is exactly this machine's shape - the ambient
    ``HF_HUB_CACHE`` lives under an unreadable mount - so an unguarded probe
    replaces one confusing error with another.
    """
    probe = path
    while True:
        try:
            if probe.exists():
                return os.access(probe, os.W_OK)
        except OSError:
            return False
        if probe.parent == probe:
            return False
        probe = probe.parent


def ensure_hf_cache() -> Path:
    """Point the HuggingFace caches at a writable directory under ``artifacts/``.

    ``trust_remote_code`` loads mean transformers needs three writable paths, not
    one: ``HF_HOME`` for downloads, ``HF_HUB_CACHE`` for the hub, and
    ``HF_MODULES_CACHE`` for the repository modelling code it compiles at load
    time. On this machine the ambient ``HF_HUB_CACHE`` points at a read-only
    mount, and loading the teacher failed with
    ``PermissionError: '/home/.../models/huggingface/modules'`` - a message that
    names neither the model nor the variable responsible. Setting the three here
    keeps the teacher path working in a fresh checkout without touching the
    user's environment, and leaves alone any value that is already writable (a
    caller who has a warm cache should keep using it).
    """
    root = ART / "hf"
    for variable, subdirectory in (("HF_HOME", ""), ("HF_HUB_CACHE", "hub"), ("HF_MODULES_CACHE", "modules")):
        target = root / subdirectory if subdirectory else root
        current = os.environ.get(variable)
        if not current or not _nearest_writable(Path(current)):
            os.environ[variable] = str(target)
    root.mkdir(parents=True, exist_ok=True)
    return root


def forward_logits(model, tokens: torch.Tensor) -> torch.Tensor:
    """Logits from either this project's model or a ``transformers`` one.

    ``V0MoECausalLM.forward`` returns a tensor; every ``transformers`` causal LM
    returns a ``CausalLMOutputWithPast``. Calling ``.float()`` on the second kind
    is an ``AttributeError``, not a wrong number, which is the good version of this
    mistake - but it is still a mistake this leg made once.
    """
    out = model(tokens)
    return out.logits if hasattr(out, "logits") else out


def load_teacher(device: torch.device):
    """Load the 1.1B teacher in half precision, frozen.

    ``trust_remote_code`` is required: SantaCoder is a GPT-2 with multi-query
    attention, and its config maps to modelling code that ships in the model
    repository rather than in ``transformers``.
    """
    ensure_hf_cache()
    from transformers import AutoModelForCausalLM

    _log(f"loading teacher {TEACHER_REPO} ...")
    model = AutoModelForCausalLM.from_pretrained(str(TEACHER_DIR), dtype=torch.float16)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def stage_targets(stream_tokens: int, batch_windows: int, force: bool = False) -> dict:
    """Score the teacher over a fixed grid of non-overlapping windows.

    Non-overlapping is not a shortcut. It is what makes the teacher's context at
    a position exactly the student's context at that position, since both see the
    same window with causal attention. A sliding window would give the teacher
    strictly more context than the student and quietly flatter the student.
    """
    TARGETS_DIR.mkdir(parents=True, exist_ok=True)
    ids_path = TARGETS_DIR / "teacher-topk-ids.npy"
    delta_path = TARGETS_DIR / "teacher-topk-delta.npy"
    top1_path = TARGETS_DIR / "teacher-top1-logp.npy"
    residual_path = TARGETS_DIR / "teacher-residual.npy"
    progress_path = TARGETS_DIR / "progress.json"
    (windows, positions, k), flat_shape = targets_shapes(stream_tokens)

    if force:
        for path in (ids_path, delta_path, top1_path, residual_path, progress_path):
            path.unlink(missing_ok=True)
    start = 0
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if int(progress.get("windows", -1)) != windows:
            # A different token budget means differently shaped arrays. Resuming
            # into them would either raise on the shape or, worse, read a window
            # of one grid as a window of another; the arrays are rebuilt instead.
            _log(
                f"target cache holds {progress.get('windows')} windows but {windows:,} are wanted; rebuilding"
            )
            for path in (ids_path, delta_path, top1_path, residual_path, progress_path):
                path.unlink(missing_ok=True)
        else:
            start = int(progress["windows_done"])
            if start >= windows:
                _log(f"teacher targets cached: {windows:,} windows")
                return {"windows": windows, "top_k": k, "resumed_from": windows}

    stream = np.load(ART / "tokens.npy", mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_teacher(device)
    ids_store = np.lib.format.open_memmap(ids_path, mode="r+" if start else "w+", dtype=np.int32, shape=(windows, positions, k))
    delta_store = np.lib.format.open_memmap(delta_path, mode="r+" if start else "w+", dtype=np.float16, shape=(windows, positions, k))
    top1_store = np.lib.format.open_memmap(top1_path, mode="r+" if start else "w+", dtype=np.float32, shape=flat_shape)
    residual_store = np.lib.format.open_memmap(residual_path, mode="r+" if start else "w+", dtype=np.float32, shape=flat_shape)

    window_tokens = windows * CONTEXT_SIZE
    _log(
        f"scoring {windows:,} windows ({window_tokens:,} tokens) from window {start:,} "
        f"on {device}; top-{k} + residual"
    )
    started = time.time()
    scored_before = start
    with torch.no_grad():
        for window_start in range(start, windows, batch_windows):
            stop = min(window_start + batch_windows, windows)
            index = np.arange(window_start, stop)
            batch = np.stack(
                [np.asarray(stream[w * CONTEXT_SIZE : (w + 1) * CONTEXT_SIZE], dtype=np.int64) for w in index]
            )
            tokens = torch.from_numpy(batch).to(device)
            logits = forward_logits(model, tokens[:, :-1]).float()
            log_probs = torch.log_softmax(logits, dim=-1)
            top_logp, top_ids = torch.topk(log_probs, k, dim=-1)
            residual = (1.0 - top_logp.exp().sum(dim=-1)).clamp_min(0.0)
            ids_store[window_start:stop] = top_ids.cpu().numpy().astype(np.int32)
            delta_store[window_start:stop] = (top_logp - top_logp[..., :1]).cpu().numpy().astype(np.float16)
            top1_store[window_start:stop] = top_logp[..., 0].cpu().numpy()
            residual_store[window_start:stop] = residual.cpu().numpy()
            if (window_start - start) // batch_windows % 20 == 0 or stop >= windows:
                seen = (stop - start) * CONTEXT_SIZE
                rate = seen / max(1e-9, time.time() - started)
                _log(f"  window {stop:,}/{windows:,}  {rate:,.0f} teacher tokens/s")
                progress_path.write_text(
                    json.dumps({"windows_done": stop, "windows": windows, "top_k": k}, indent=2) + "\n"
                )
    for store in (ids_store, delta_store, top1_store, residual_store):
        store.flush()
    progress_path.write_text(json.dumps({"windows_done": windows, "windows": windows, "top_k": k}, indent=2) + "\n")
    _log(f"teacher targets written: {windows:,} windows in {time.time() - started:.0f}s")
    return {"windows": windows, "top_k": k, "resumed_from": scored_before}


# --- checkpointing ----------------------------------------------------------
#
# Same contract as scripts/train_real_lm.py, and for the same measured reason:
# `Optimizer.state_dict()` is position-keyed while `nn.Module.state_dict()` is
# name-keyed, so a run whose parameter *order* changed between checkpoints loads
# its weights cleanly and attaches Adam's moments to the wrong tensors. That
# happened on the TinyStories leg (36 of 128 parameters, a one-position shift);
# here the moments are stored by name from the start.


def run_signature(scale: str, mode: str, windows: int, max_steps: int, batch_size: int) -> dict:
    """What makes two invocations the *same run*, for the purpose of adopting a checkpoint.

    A campaign re-launched after a crash has to resume where it stopped, and the only thing
    that makes resuming safe is knowing the checkpoint came from this run rather than from
    another one with the same shapes. The failure it prevents is silent and was found on
    disk: a 20-step smoke run leaves ``match-ce.pt`` behind, the real arm adopts it, starts
    at step 21 and reports the full budget - so the control arm began from different weights
    than the arm it is compared against and the difference between them was no longer the
    objective. Nothing raises in that case; the numbers just stop meaning what they say.
    """
    return {
        "scale": scale,
        "mode": mode,
        "seed": SEED,
        "batch_size": batch_size,
        "max_steps": max_steps,
        "windows": windows,
    }


def run_mismatches(saved: dict, wanted: dict) -> list[str]:
    """The fields on which a checkpoint's run and this invocation disagree, for a refusal."""
    return [
        f"{key} {saved.get(key)!r} -> {wanted.get(key)!r}"
        for key in sorted(set(saved) | set(wanted))
        if saved.get(key) != wanted.get(key)
    ]


def checkpoint_matches(payload: dict, run: dict, path: Path) -> None:
    """Refuse a checkpoint that is not this run's; return quietly when it is.

    Refusing costs a stage that has not started, and it is the direction that cannot be
    quietly wrong - an adopted checkpoint is a different experiment reported as this one.
    """
    saved = payload.get("run")
    if not saved:
        raise SystemExit(
            f"{path} was written before checkpoints recorded which run they belong to, so "
            "resuming from it cannot be shown to be this run. Move it aside to start from step 1."
        )
    mismatched = run_mismatches(saved, run)
    if mismatched:
        raise SystemExit(
            f"{path} belongs to a different run ({'; '.join(mismatched)}); adopting it would "
            "start this arm from another run's weights and optimiser moments. Move it aside "
            "to start this run from step 1."
        )


def tokens_trained(max_steps: int, start_step: int, batch_size: int) -> dict:
    """What an arm actually saw, against the budget it was run to.

    The two differ exactly when a checkpoint was adopted. A resumed arm reports the whole
    budget either way, and a reader who takes the budget for what the arm saw credits it
    with work that another run did - which is the same mistake in the other direction.
    """
    return {
        "tokens_budget": max_steps * batch_size * CONTEXT_SIZE,
        "tokens_seen": (max_steps - start_step) * batch_size * CONTEXT_SIZE,
        "steps_trained": max_steps - start_step,
        "resumed_from_step": start_step,
    }


def checkpoint_payload(model, optimizer, step: int, history: list[dict], run: dict) -> dict:
    by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    moments: dict[str, dict] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if state:
                moments[by_id[id(parameter)]] = {
                    key: (value.detach().clone() if torch.is_tensor(value) else value)
                    for key, value in state.items()
                }
    return {
        "model": model.state_dict(),
        "optimizer": {
            "groups": [{key: value for key, value in group.items() if key != "params"} for group in optimizer.param_groups],
            "moments": moments,
        },
        "step": step,
        "history": history,
        # Which run this belongs to, so the next launch can tell whether adopting it is
        # resuming this experiment or starting a different one from its middle.
        "run": dict(run),
    }


def restore_optimizer(model, optimizer, payload: dict, path: Path, *, reset: bool) -> str:
    saved = payload.get("optimizer")
    if reset or not saved:
        _log("optimizer moments start fresh")
        return "fresh"
    if "moments" not in saved:
        raise SystemExit(
            f"{path} stores its optimizer state by position, which cannot be attached to this "
            "model; re-run with --reset-optimizer or delete the checkpoint."
        )
    by_name = dict(model.named_parameters())
    missing = [name for name in saved["moments"] if name not in by_name]
    if missing:
        raise SystemExit(f"{path}'s optimizer state names {missing[:4]} which this model does not have")
    for name, moment in saved["moments"].items():
        optimizer.state[by_name[name]] = dict(moment)
    for group, metadata in zip(optimizer.param_groups, saved["groups"]):
        group.update(metadata)
    _log(f"optimizer moments restored by name for {len(saved['moments'])} parameters")
    return f"restored by name ({len(saved['moments'])} parameters)"


# --- stage 5: training ------------------------------------------------------


def student_config(scale: str) -> V0MoEConfig:
    if scale not in SCALES:
        raise SystemExit(f"unknown scale {scale!r}; choose from {sorted(SCALES)}")
    spec = SCALES[scale]
    depth = spec["depth"]
    return V0MoEConfig(
        vocab_size=teacher_vocab_size(),
        context_size=CONTEXT_SIZE,
        hidden_size=spec["hidden_size"],
        depth=depth,
        attention_heads=spec["attention_heads"],
        experts_per_layer=(spec["expert_count"],) * depth,
        active_experts=spec["active_experts"],
        expert_width=spec["expert_width"],
    )


def lr_at(step: int, max_steps: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, max_steps - WARMUP_STEPS)
    return 0.1 * LEARNING_RATE + 0.9 * LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * progress))


def load_targets(stream_tokens: int):
    shapes = targets_shapes(stream_tokens)
    return {
        "ids": np.load(TARGETS_DIR / "teacher-topk-ids.npy", mmap_mode="r"),
        "delta": np.load(TARGETS_DIR / "teacher-topk-delta.npy", mmap_mode="r"),
        "top1": np.load(TARGETS_DIR / "teacher-top1-logp.npy", mmap_mode="r"),
        "residual": np.load(TARGETS_DIR / "teacher-residual.npy", mmap_mode="r"),
        "shape": shapes,
    }


def distillation_terms(student_logits: torch.Tensor, targets: dict, windows: np.ndarray, device: torch.device):
    """KL(teacher || student) over the whole vocabulary, from a top-K cache.

    The teacher's distribution is ``p_i`` on the stored top-32 tokens plus one
    residual bucket holding everything else. The student's mass on that same set
    is ``q_i``, and its residual is ``1 - sum q_i`` - which is exactly the
    student's probability of *not* choosing any of the teacher's top tokens, and
    therefore has a real gradient rather than being a constant. Truncating the
    teacher and renormalising instead would make the student's tail free.
    """
    ids = torch.from_numpy(np.asarray(targets["ids"][windows]).astype(np.int64)).to(device)
    delta = torch.from_numpy(np.asarray(targets["delta"][windows]).astype(np.float32)).to(device)
    top1 = torch.from_numpy(np.asarray(targets["top1"][windows]).astype(np.float32)).to(device)
    residual = torch.from_numpy(np.asarray(targets["residual"][windows]).astype(np.float32)).to(device)

    teacher_logp = delta + top1.unsqueeze(-1)
    teacher_p = teacher_logp.exp()

    log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    student_top = log_probs.gather(-1, ids)
    student_residual = (1.0 - student_top.exp().sum(dim=-1)).clamp_min(1e-6)

    top_term = (teacher_p * (teacher_logp - student_top)).sum(dim=-1)
    rest_p = residual.clamp_min(1e-9)
    rest_term = rest_p * (rest_p.log() - student_residual.log())
    return (top_term + rest_term).mean(), teacher_p, student_top


def evaluate_bits(student_loss_nats: float, held_out_bytes: int) -> float:
    return student_loss_nats / math.log(2) / held_out_bytes


@torch.no_grad()
def evaluate(model, tokens: np.ndarray, device: torch.device, batches: int = 8) -> tuple[float, float]:
    model.eval()
    windows = tokens.shape[0] // CONTEXT_SIZE
    generator = torch.Generator().manual_seed(SEED)
    losses = []
    for _ in range(batches):
        starts = torch.randint(0, max(1, windows), (4,), generator=generator).numpy()
        batch = np.stack(
            [np.asarray(tokens[s * CONTEXT_SIZE : (s + 1) * CONTEXT_SIZE], dtype=np.int64) for s in starts]
        )
        batch_tokens = torch.from_numpy(batch).to(device)
        logits = model(batch_tokens[:, :-1])
        losses.append(
            nn.functional.cross_entropy(
                logits.reshape(-1, model.config.vocab_size), batch_tokens[:, 1:].reshape(-1)
            ).item()
        )
    model.train()
    mean = sum(losses) / len(losses)
    return mean, math.exp(mean)


def reference_bits_per_byte(val_tokens: np.ndarray, held_out_bytes: int) -> dict:
    """Uniform/unigram/bigram beselines on the held-out stream, in bits per byte.

    Same role as the TinyStories leg's references: a held-out loss with nothing
    beside it is not a result, since nobody knows in advance whether 0.9 bits per
    byte is good or terrible. Both are fit on the held-out stream itself, so they
    are a yardstick rather than a competitor.
    """
    tokens = np.asarray(val_tokens, dtype=np.int64).tolist()
    pairs = list(zip(tokens[:-1], tokens[1:]))
    n = len(pairs)
    if not n:
        raise SystemExit("held-out stream is too short to score a reference on")
    size = vocab_size()
    unigram = collections.Counter(target for _, target in pairs)
    bigram = collections.Counter(pairs)
    context = collections.Counter(source for source, _ in pairs)
    unigram_nats = -sum(math.log((unigram[t] + 1.0) / (n + size)) for _, t in pairs) / n
    bigram_nats = -sum(math.log((bigram[p] + 1.0) / (context[p[0]] + size)) for p in pairs) / n
    return {
        "held_out_bytes": held_out_bytes,
        "uniform": {"nats_per_token": math.log(size), "bits_per_byte": math.log2(size) / (held_out_bytes / n)},
        "unigram": {"nats_per_token": unigram_nats, "bits_per_byte": unigram_nats / math.log(2) / (held_out_bytes / n)},
        "bigram": {"nats_per_token": bigram_nats, "bits_per_byte": bigram_nats / math.log(2) / (held_out_bytes / n)},
        "fit_on": "the held-out stream itself - a yardstick, not a competitor",
    }


def generate(model, tokenizer, prompt: str, device: torch.device, new_tokens: int = 80) -> str:
    ids = tokenizer.encode(prompt).ids
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    generator = torch.Generator(device="cpu").manual_seed(SEED + len(prompt))
    for _ in range(new_tokens):
        logits = model(tokens[:, -CONTEXT_SIZE:])[:, -1, :]
        probabilities = torch.softmax(logits[0].float(), dim=-1)
        next_id = int(torch.multinomial(probabilities.cpu(), 1, generator=generator))
        tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device)], dim=1)
    return tokenizer.decode(tokens[0].tolist())


def stage_train(
    scale: str,
    mode: str,
    stream_tokens: int,
    max_steps: int,
    batch_size: int,
    log_every: int,
    epochs: int,
    *,
    reset_optimizer: bool = False,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    config = student_config(scale)
    total, active = estimated_parameter_count(config)
    model = V0MoECausalLM(config).to(device)
    model.activation_checkpointing = True
    model.train()
    _log(
        f"student {scale}: {total:,} total / {active:,} active parameters "
        f"(h{config.hidden_size} d{config.depth} e{config.experts_per_layer[0]} "
        f"a{config.active_experts} w{config.expert_width})"
    )
    if mode == "distill":
        _log(f"distilling from a cached {TEACHER_REPO} on top-{TOP_K} targets")

    stream = np.load(ART / "tokens.npy", mmap_mode="r")
    val_tokens = np.load(ART / "val-tokens.npy")
    held_out_bytes = len((ART / "val-text.txt").read_text().encode())
    # The training window range is pinned by ``stream_tokens`` rather than by what
    # the stream happens to hold, and deliberately *not* by whether the teacher
    # scored it. The control arm has to see the same data as the distilled arm or
    # the comparison measures two things at once: with the cap derived from the
    # targets, a cross-entropy run over the full 162M-token stream would be
    # compared against a distilled run over the scored 15M, and distillation would
    # be credited or blamed for a 10x difference in data. Pinning it here makes
    # both arms draw the same windows in the same order from the same seed, so the
    # only difference left is the objective.
    windows = min(stream.shape[0] // CONTEXT_SIZE, stream_tokens // CONTEXT_SIZE)
    _log(f"training over the first {windows:,} windows ({windows * CONTEXT_SIZE:,} tokens)")
    targets = load_targets(stream_tokens) if mode == "distill" else None
    if targets is not None:
        scored = int(targets["ids"].shape[0])
        if scored < windows:
            raise SystemExit(
                f"teacher targets cover {scored:,} windows but training would use {windows:,}; "
                "positions the teacher never scored would carry a zero KL term"
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    generator = torch.Generator().manual_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    latest = CHECKPOINT_DIR / f"{scale}-{mode}.pt"
    run = run_signature(scale, mode, windows, max_steps, batch_size)
    start_step = 0
    history: list[dict] = []
    optimizer_state = "fresh (no checkpoint)"
    if latest.is_file():
        payload = torch.load(latest, map_location=device, weights_only=False)
        checkpoint_matches(payload, run, latest)
        model.load_state_dict(payload["model"])
        optimizer_state = restore_optimizer(model, optimizer, payload, latest, reset=reset_optimizer)
        start_step = payload["step"]
        history = payload.get("history", [])
        _log(f"resumed at step {start_step} (optimizer {optimizer_state})")

    tokens_per_step = batch_size * CONTEXT_SIZE
    started = time.time()
    uniform_floor = math.log(config.vocab_size)
    for step in range(start_step + 1, max_steps + 1):
        lr = lr_at(step, max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        draws = torch.randint(0, windows, (batch_size,), generator=generator).numpy()
        batch = np.stack(
            [np.asarray(stream[d * CONTEXT_SIZE : (d + 1) * CONTEXT_SIZE], dtype=np.int64) for d in draws]
        )
        tokens = torch.from_numpy(batch).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens[:, :-1])
        cross_entropy = nn.functional.cross_entropy(
            logits.reshape(-1, config.vocab_size), tokens[:, 1:].reshape(-1)
        )
        loss = cross_entropy
        kl = None
        if targets is not None:
            kl, _, _ = distillation_terms(logits, targets, draws, device)
            loss = (1.0 - DISTILL_WEIGHT) * cross_entropy + DISTILL_WEIGHT * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_every == 0 or step == max_steps:
            rate = tokens_per_step * log_every / (time.time() - started)
            detail = f" ce {cross_entropy.item():.4f}" + (f" kl {kl.item():.4f}" if kl is not None else "")
            _log(f"step {step}/{max_steps}{detail} loss {loss.item():.4f} lr {lr:.2e} {rate:,.0f} tokens/s")
            started = time.time()
        if step % EVAL_EVERY == 0 or step == max_steps:
            value, perplexity = evaluate(model, val_tokens, device)
            entry = {"step": step, "train_loss": loss.item(), "val_loss": value, "perplexity": perplexity,
                     "bits_per_byte": evaluate_bits(value, held_out_bytes)}

            if targets is not None:
                entry["train_kl"] = float(kl.item()) if kl is not None else None
            history.append(entry)
            _log(f"  eval: val loss {value:.4f} perplexity {perplexity:.2f}")
        if step % CHECKPOINT_EVERY == 0 or step == max_steps:
            torch.save(checkpoint_payload(model, optimizer, step, history, run), latest.with_suffix(".tmp"))
            latest.with_suffix(".tmp").replace(latest)

    final_loss, final_perplexity = evaluate(model, val_tokens, device)
    tokenizer = shared_tokenizer()
    samples = {}
    for prompt in (
        "def add(a, b):\n",
        "import os\n\ndef read_config(path):\n",
        "class Stack:\n    def __init__(self):\n",
    ):
        samples[prompt] = generate(model, tokenizer, prompt, device)

    report = {
        "leg": "code-lm",
        "corpus": CORPUS_FILE,
        "corpus_source": CORPUS_URL,
        "tokenizer": f"shared with teacher {TEACHER_REPO} (vocab {config.vocab_size})",
        "teacher": {"repo": TEACHER_REPO, "parameters": "1.1B dense", "targets": "cached top-k + residual"},
        "baseline_same_class": {"repo": BASELINE_REPO, "parameters": "164M dense", "note": "100B tokens on 32 A100s"},
        "mode": mode,
        "distill_weight": DISTILL_WEIGHT if mode == "distill" else 0.0,
        "student": {
            "class": "act.model.V0MoECausalLM",
            "scale": scale,
            "total_parameters": total,
            "active_parameters": active,
            "hidden_size": config.hidden_size,
            "depth": config.depth,
            "attention_heads": config.attention_heads,
            "experts_per_layer": config.experts_per_layer[0],
            "active_experts": config.active_experts,
            "expert_width": config.expert_width,
            "vocab_size": config.vocab_size,
            "context_size": config.context_size,
        },
        "training": {
            "steps": max_steps,
            "batch_size": batch_size,
            "batch_tokens": tokens_per_step,
            "epochs_over_stream": epochs,
            # Both of these, because they are equal only when nothing was adopted.
            **tokens_trained(max_steps, start_step, batch_size),
            "run": run,
            "train_windows": windows,
            "train_tokens": windows * CONTEXT_SIZE,
            "stream_tokens": int(stream.shape[0]),
            "optimizer": "AdamW warmup+cosine",
            "optimizer_state": optimizer_state,
            "seed": SEED,
            "device": str(device),
        },
        "validation": {
            "final_val_loss": final_loss,
            "final_perplexity": final_perplexity,
            "bits_per_byte": evaluate_bits(final_loss, held_out_bytes),
            "uniform_floor_loss": uniform_floor,
            "held_out_bytes": held_out_bytes,
            "history": history,
        },
        "references": reference_bits_per_byte(val_tokens, held_out_bytes),
        "samples": samples,
    }
    report_path = ART / f"report-{scale}-{mode}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _log(f"report written: {report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("all", "corpus", "tokens", "targets", "train"), default="all")
    parser.add_argument("--scale", choices=sorted(SCALES), default="toy")
    parser.add_argument("--mode", choices=("ce", "distill"), default="distill")
    parser.add_argument("--tokens", type=int, default=160_000_000, help="token stream to encode")
    parser.add_argument(
        "--score-tokens", type=int, default=0,
        help="tokens for the teacher to score (0 = all of the encoded stream)",
    )
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=3, help="passes over the cached stream")
    parser.add_argument("--targets-batch", type=int, default=8, help="windows per teacher forward pass")
    parser.add_argument("--fetch-teacher", action="store_true", help=f"download {TEACHER_REPO} (4.6 GB)")
    parser.add_argument("--force", action="store_true", help="rebuild derived artifacts")
    parser.add_argument("--reset-optimizer", action="store_true")
    args = parser.parse_args()

    if args.fetch_teacher:
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                     "special_tokens_map.json", "generation_config.json"):
            fetch_parallel(HF_RESOLVE.format(repo=TEACHER_REPO, name=name), TEACHER_DIR / name, workers=4)
        fetch_parallel(
            HF_RESOLVE.format(repo=TEACHER_REPO, name="pytorch_model.bin"),
            TEACHER_DIR / "pytorch_model.bin",
        )
        return 0

    stage_corpus()
    if args.stage == "corpus":
        return 0
    train_path, _ = stage_tokens(args.tokens, force=args.force)
    if args.stage == "tokens":
        return 0
    score_tokens = args.score_tokens or int(np.load(train_path, mmap_mode="r").shape[0])
    if args.stage in ("all", "targets"):
        stage_targets(score_tokens, args.targets_batch, force=args.force)
    if args.stage == "targets":
        return 0
    report = stage_train(
        args.scale, args.mode, score_tokens, args.max_steps, args.batch_size, args.log_every, args.epochs,
        reset_optimizer=args.reset_optimizer,
    )
    _log(
        f"final: val loss {report['validation']['final_val_loss']:.3f} "
        f"({report['validation']['bits_per_byte']:.4f} bits/byte)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
