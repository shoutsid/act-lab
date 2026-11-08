"""Fire section 5.70: the declared-shape real-text block, twelve paired seeds on a T4.

The block is registered in ``artifacts/real-text-declared/NOTES.md``: seeds
15036-15047, the declared config (285M total / 134M active, 6000 steps,
24,576,000 tokens per arm), fp16 on one T4 x2, adaptive versus random churn on
the published TinyStories stream. This script stages three script kernels of
four seeds each (the platform's session limit, not the protocol, sets the
split), publishes the code dataset, pushes them, and fetches the per-seed rows.

    PYTHONPATH=src python scripts/push_real_text_declared.py --stage-only
    PYTHONPATH=src python scripts/push_real_text_declared.py --push
    PYTHONPATH=src python scripts/push_real_text_declared.py --status
    PYTHONPATH=src python scripts/push_real_text_declared.py --fetch

Nothing trains locally: the kernel mounts the code dataset and the 41 MB token
dataset, runs ``run_v0_session`` at the declared shape, and writes one small
JSON row per seed. No checkpoint is published and no dataset is restored, so
the container needs no Kaggle API credentials and the block cannot fail the way
the TPU null queue did.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import kaggle_queue
import kaggle_t4

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "artifacts" / "real-text-declared" / "kernel"
TOKENS_DATASET = "act-lab-tinystories-tokens"

#: The next block after the declared Gate-4 seeds (15024-15035). Twelve paired
#: seeds is the campaign's own bar (10 of 12, p = 0.019287), not a new one.
SEEDS = tuple(range(15036, 15048))
#: Four seeds per kernel: ~6.5 h at the measured 1.63 h/pair, comfortably inside
#: the 12 h GPU session limit. The split is scheduling, not protocol.
WAVES = {
    "a": SEEDS[0:4],
    "b": SEEDS[4:8],
    "c": SEEDS[8:12],
}
KERNEL_BASE = "act-realtext"
BAR_WINS = 10


def _kaggle(argv: list[str]) -> int:
    print(f"$ {' '.join(argv)}", flush=True)
    proc = subprocess.run(argv, capture_output=True, text=True)
    print(proc.stdout[-8000:], flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr, flush=True)
    return proc.returncode


ENTRY = '''"""Kaggle GPU entry for section 5.70 (generated, do not edit)."""
import glob
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

SEEDS = __SEEDS__
POLICY = __POLICY__
REVISION = __REVISION__
DTYPE = "fp16"
STEPS = 6000
BATCH = 2
CONTEXT = 2048


def find_repo():
    for marker in sorted(glob.glob("/kaggle/input/**/pyproject.toml", recursive=True)):
        root = Path(marker).parent
        if (root / "src" / "act" / "training" / "__init__.py").is_file():
            return root
    return None


def find_tokens():
    for match in sorted(glob.glob("/kaggle/input/**/tokens.npy", recursive=True)):
        path = Path(match)
        if (path.parent / "val-tokens.npy").is_file():
            return path
    return None


def main() -> int:
    root = find_repo()
    tokens = find_tokens()
    if root is None:
        print("code tree (pyproject.toml + src/act) not found under /kaggle/input", flush=True)
        return 2
    if tokens is None:
        print("token stream not found under /kaggle/input", flush=True)
        return 2
    sys.path.insert(0, str(root / "src"))
    from act.training import declared_session_config, run_v0_session

    out = Path("/kaggle/working/realtext")
    out.mkdir(parents=True, exist_ok=True)
    print(f"repo={root} tokens={tokens} seeds={list(SEEDS)} revision={REVISION}", flush=True)
    failures = []
    for seed in SEEDS:
        target = out / f"seed-{seed}.json"
        if target.exists():
            print(f"skip seed {seed}: already recorded", flush=True)
            continue
        config = declared_session_config(
            seed=seed,
            arms=("adaptive", "random"),
            data_path=str(tokens),
            val_data_path=str(tokens.parent / "val-tokens.npy"),
            dtype=DTYPE,
            probe_candidate_policy=POLICY,
            preserve_router_on_rebuild=True,
            new_expert_initialization="clone",
        )
        checkpoints = Path("/tmp/realtext-checkpoints") / str(seed)
        started = time.time()
        try:
            result = run_v0_session(
                config, checkpoint_dir=checkpoints, code_revision=REVISION
            )
        except Exception as error:  # noqa: BLE001 - the row is the diagnosis
            failures.append(seed)
            (out / f"seed-{seed}.error.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                        "code_revision": REVISION,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\\n"
            )
            print(f"FAIL seed {seed}: {type(error).__name__}: {error}", flush=True)
            continue
        row = {
            "kind": "realtext-pair",
            "report_section": "5.70",
            "seed": seed,
            "code_revision": REVISION,
            "dtype": DTYPE,
            "wall_seconds": time.time() - started,
            "config": {
                "total_steps": STEPS,
                "batch_size": BATCH,
                "context_size": CONTEXT,
                "data_path": str(tokens),
                "val_data_path": str(tokens.parent / "val-tokens.npy"),
                "probe_candidate_policy": POLICY,
                "preserve_router_on_rebuild": True,
                "new_expert_initialization": "clone",
            },
            "arms": {},
        }
        for arm in result.arms:
            row["arms"][arm.mode] = {
                "final_loss": arm.final_loss,
                "validation_loss": arm.validation_loss,
                "mutation_count": arm.mutation_count,
                "architecture_hash": arm.architecture_hash,
                "parameter_count": arm.parameter_count,
                "active_parameter_count": arm.active_parameter_count,
                "search_flops": arm.search_flops,
                "training_flops": arm.training_flops,
                "tokens_per_second": arm.tokens_per_second,
                "events": [
                    {
                        "step": event.get("step"),
                        "operation": event.get("operation"),
                        "layer_id": event.get("layer_id"),
                        "expert_id": event.get("expert_id"),
                    }
                    for event in arm.mutation_history
                ],
            }
        target.write_text(json.dumps(row, indent=2, sort_keys=True) + "\\n")
        losses = "  ".join(
            f"{mode}={arms['final_loss']:.6f}" for mode, arms in sorted(row["arms"].items())
        )
        print(f"seed {seed}: {losses}  ({row['wall_seconds']:.0f}s)", flush=True)
        shutil.rmtree(checkpoints, ignore_errors=True)

    recorded = sorted(
        int(path.stem.split("-")[1])
        for path in out.glob("seed-*.json")
        if path.name.count("-") == 1
    )
    summary = {
        "kind": "realtext-summary",
        "seeds": list(SEEDS),
        "recorded": recorded,
        "failures": failures,
        "code_revision": REVISION,
    }
    (Path("/kaggle/working") / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\\n"
    )
    print("=== REAL-TEXT SUMMARY ===")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 1 if failures or len(recorded) != len(SEEDS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def slug(wave: str) -> str:
    return f"{KERNEL_BASE}-{wave}"


def reference(owner: str, wave: str) -> str:
    return f"{owner}/{slug(wave)}"


def metadata(owner: str, wave: str) -> dict[str, object]:
    return {
        "id": reference(owner, wave),
        "title": slug(wave),
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        # The block is priced in GPU hours; this is the accelerator it is for.
        "enable_gpu": True,
        "enable_tpu": False,
        "machine_shape": "NvidiaTeslaT4",
        # Both inputs are mounted datasets; the container needs no API calls.
        "enable_internet": False,
        "dataset_sources": [
            kaggle_queue.code_dataset_id(),
            f"{owner}/{TOKENS_DATASET}",
        ],
        "competition_sources": [],
        "kernel_sources": [],
    }


def stage(owner: str, wave: str, revision: str) -> Path:
    folder = WORK / wave
    folder.mkdir(parents=True, exist_ok=True)
    entry = (
        ENTRY.replace("__SEEDS__", json.dumps(list(WAVES[wave])))
        .replace("__POLICY__", json.dumps(kaggle_t4.PRIMARY_POLICY))
        .replace("__REVISION__", json.dumps(revision))
    )
    (folder / "main.py").write_text(entry, encoding="utf-8")
    (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata(owner, wave), indent=2) + "\n", encoding="utf-8"
    )
    return folder


def push(folder: Path) -> int:
    """Push one staged kernel, reading the CLI's own success line.

    Measured in this repository: a refused push (the one-TPU-session rule) prints
    its error and exits 0, so the exit status alone cannot say the kernel exists.
    """
    argv = ["kaggle", "kernels", "push", "-p", str(folder)]
    print(f"$ {' '.join(argv)}", flush=True)
    proc = subprocess.run(argv, capture_output=True, text=True)
    print(proc.stdout[-8000:], flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr, flush=True)
        return proc.returncode
    if "successfully pushed" not in proc.stdout.lower():
        print("push was not confirmed by the CLI's own success line", file=sys.stderr, flush=True)
        return 1
    return 0


def fetch(owner: str, wave: str) -> int:
    target = WORK / wave
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"realtext-{wave}-") as temporary:
        code = _kaggle(["kaggle", "kernels", "output", reference(owner, wave), "-p", temporary])
        for path in sorted(Path(temporary).rglob("*")):
            if path.is_file() and (path.suffix == ".json" or path.suffix == ".log"):
                shutil.copy2(path, target / path.name)
                print(f"--- {path.name}\n{path.read_text(errors='replace')[:4000]}", flush=True)
    log = kaggle_queue.fetch_kernel_log(reference(owner, wave))
    if log:
        (target / f"{slug(wave)}.log").write_text(log, encoding="utf-8")
        print("--- kernel log (tail)\n" + log[-4000:], flush=True)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--wave", choices=sorted(WAVES), default=None)
    args = parser.parse_args()

    owner = kaggle_queue.code_dataset_id().split("/")[0]
    waves = [args.wave] if args.wave else sorted(WAVES)
    revision = kaggle_queue.code_revision(ROOT)

    if args.status:
        return max(
            _kaggle(["kaggle", "kernels", "status", reference(owner, wave)])
            for wave in waves
        )
    if args.fetch:
        return max(fetch(owner, wave) for wave in waves)

    folders = [stage(owner, wave, revision) for wave in waves]
    for folder in folders:
        print(f"staged {folder}", flush=True)
    if args.stage_only or not args.push:
        if not args.push:
            print("nothing pushed; pass --push to publish and run it", flush=True)
        return 0

    revision = kaggle_queue.ensure_code_dataset(ROOT)
    print(f"code dataset at {revision}", flush=True)
    for folder, wave in zip(folders, waves):
        # Re-stage with the revision the dataset actually carries, so the rows
        # name the code they ran rather than the tree at staging time.
        stage(owner, wave, revision)
        code = push(folder)
        if code != 0:
            return code
        print(f"pushed {reference(owner, wave)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
