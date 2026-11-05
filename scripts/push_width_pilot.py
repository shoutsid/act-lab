"""Push the width pilot as a CPU-only Kaggle kernel - zero GPU-hours.

The pilot is a CPU job by design (``docs/width-control-design.md`` section 10):
the dev shape at one thread per process, four processes at once, on the
published TinyStories token stream. This script is the whole local footprint
after the code is written: it stages a script kernel, pushes it, and fetches the
per-seed records. No training runs here, and the kernel requests no accelerator.

    PYTHONPATH=src python scripts/push_width_pilot.py --stage-only
    PYTHONPATH=src python scripts/push_width_pilot.py --push
    PYTHONPATH=src python scripts/push_width_pilot.py --status
    PYTHONPATH=src python scripts/push_width_pilot.py --fetch --save artifacts/width-pilot/kaggle

The kernel mounts two datasets: the code dataset (this tree, republished by
this script) and ``act-lab-tinystories-tokens`` (the 41 MB stream). It runs
every ``(seed, arm)`` pair in the plan, four at a time, and writes one JSON
record per job plus a summary the fetch keeps.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import kaggle_queue

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "artifacts" / "width-pilot" / "kernel"
ENTRY_NAME = "main.py"
KERNEL_SLUG = "act-width-pilot"
TOKENS_DATASET = "act-lab-tinystories-tokens"


def _kaggle(argv: list[str]) -> int:
    print(f"$ {' '.join(argv)}", flush=True)
    proc = subprocess.run(argv, capture_output=True, text=True)
    print(proc.stdout[-8000:], flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr, flush=True)
    return proc.returncode


ENTRY = '''"""Kaggle CPU kernel entry for the width pilot (generated, do not edit)."""
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PLAN = __PLAN__
WORKERS = __WORKERS__


def find(pattern, required=()):
    for match in sorted(glob.glob(pattern, recursive=True)):
        path = Path(match)
        if all((path.parent / name).is_file() for name in required):
            return path
    return None


def main() -> int:
    pilot = find("/kaggle/input/**/scripts/width_pilot.py")
    if pilot is None:
        print("width_pilot.py is not in the mounted code dataset", flush=True)
        return 2
    root = pilot.parents[1]
    tokens = find("/kaggle/input/**/tokens.npy", required=("val-tokens.npy",))
    if tokens is None:
        print("token stream not found under /kaggle/input", flush=True)
        return 2
    out = Path("/kaggle/working/width-pilot")
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(job["seed"], job["arm"]) for job in PLAN["jobs"]]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    def run(job):
        seed, arm = job
        target = out / f"{arm}-seed{seed}.json"
        argv = [
            sys.executable, str(pilot),
            "--seed", str(seed), "--arm", arm,
            "--steps", str(PLAN["steps"]),
            "--interval", str(PLAN["interval"]),
            "--budget", str(PLAN["budget"]),
            "--data-dir", str(tokens.parent),
            "--out", str(target),
        ]
        print(f"$ {' '.join(argv)}", flush=True)
        proc = subprocess.run(argv, env=env)
        return job, proc.returncode, target

    failures = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for (seed, arm), code, target in pool.map(run, jobs):
            ok = code == 0 and target.is_file()
            print(f"{'ok' if ok else 'FAIL'} {arm} seed {seed} exit {code}", flush=True)
            if not ok:
                failures.append(f"{arm}:{seed}")

    records = []
    for path in sorted(out.glob("*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except ValueError:
            failures.append(f"unreadable:{path.name}")
    summary = {
        "kind": "width-pilot-summary",
        "plan": PLAN,
        "jobs": len(jobs),
        "records": len(records),
        "failures": failures,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n")
    print("=== WIDTH PILOT SUMMARY ===")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _seeds(value: str) -> list[int]:
    """``71000-71011`` or ``71000,71002``, in the order written."""

    seeds: list[int] = []
    for part in value.split(","):
        piece = part.strip()
        if not piece:
            continue
        if "-" in piece:
            start, end = piece.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(piece))
    return seeds


def plan(args: argparse.Namespace) -> dict[str, object]:
    """The staged plan. ``--jobs`` scopes a resume to jobs a session has not done.

    The pilot is one session on Kaggle's CPU, and that session can be cut by the
    platform's clock before the last job. The records fetched from it are the
    completed jobs; re-running a seed would recompute them, so a resume names
    the missing ``SEED:ARM`` jobs rather than the seed block. Without ``--jobs``
    the plan is the full cross product, exactly as the block was registered.
    """

    if args.jobs:
        jobs: list[dict[str, object]] = []
        for item in args.jobs.split(","):
            piece = item.strip()
            if not piece:
                continue
            seed_text, separator, arm = piece.partition(":")
            if not separator or arm not in ("width", "width_random"):
                raise SystemExit(
                    f"bad --jobs entry {piece!r}; expected SEED:width or "
                    "SEED:width_random"
                )
            jobs.append({"seed": int(seed_text), "arm": arm})
        if not jobs:
            raise SystemExit("--jobs named no jobs")
        seeds = sorted({int(job["seed"]) for job in jobs})
        arms = sorted({str(job["arm"]) for job in jobs})
    else:
        seeds = _seeds(args.seeds)
        arms = [arm for arm in args.arms.split(",") if arm]
        jobs = [
            {"seed": seed, "arm": arm} for seed in seeds for arm in arms
        ]
    return {
        "seeds": seeds,
        "arms": arms,
        "jobs": jobs,
        "steps": args.steps,
        "interval": args.interval,
        "budget": args.budget,
    }


def metadata(owner: str) -> dict[str, object]:
    return {
        "id": f"{owner}/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": ENTRY_NAME,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        # CPU only, and explicitly not a GPU: the pilot is priced at zero
        # GPU-hours, and a kernel that quietly landed on a T4 would bill the
        # week's scarcest resource for work that does not need it.
        "enable_gpu": False,
        "enable_tpu": False,
        "machine_shape": "",
        # Both inputs are mounted datasets; nothing is downloaded at run time.
        "enable_internet": False,
        "dataset_sources": [
            kaggle_queue.code_dataset_id(),
            f"{owner}/{TOKENS_DATASET}",
        ],
        "competition_sources": [],
        "kernel_sources": [],
    }


def stage(owner: str, settings: dict[str, object], workers: int) -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    entry = ENTRY.replace("__PLAN__", json.dumps(settings, sort_keys=True)).replace(
        "__WORKERS__", str(workers)
    )
    (WORK / ENTRY_NAME).write_text(entry, encoding="utf-8")
    (WORK / "kernel-metadata.json").write_text(
        json.dumps(metadata(owner), indent=2) + "\n", encoding="utf-8"
    )
    return WORK


def fetch(reference: str, save: Path | None) -> int:
    with tempfile.TemporaryDirectory(prefix="width-pilot-out-") as temporary:
        code = _kaggle(["kaggle", "kernels", "output", reference, "-p", temporary])
        for path in sorted(Path(temporary).rglob("*")):
            if path.is_file() and path.suffix in {".json", ".log", ".txt"}:
                print(f"--- {path.name}\n{path.read_text(errors='replace')[:8000]}", flush=True)
        if save is not None:
            saved = kaggle_queue.save_fetched(Path(temporary), save)
            print(f"saved {len(saved)} file(s) under {save}", flush=True)
    log = kaggle_queue.fetch_kernel_log(reference)
    if log:
        print("--- kernel log (tail)\n" + log[-6000:], flush=True)
        if save is not None:
            save.mkdir(parents=True, exist_ok=True)
            (save / f"{KERNEL_SLUG}.log").write_text(log, encoding="utf-8")
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="71000", help="comma-separated seeds")
    parser.add_argument("--arms", default="width,width_random")
    parser.add_argument(
        "--jobs",
        default=None,
        help="comma-separated SEED:ARM jobs; overrides --seeds/--arms to resume a partial block",
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--push", action="store_true", help="publish the code dataset and push")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    owner = kaggle_queue.code_dataset_id().split("/")[0]
    reference = f"{owner}/{KERNEL_SLUG}"
    settings = plan(args)
    folder = stage(owner, settings, args.workers)
    print(f"staged {folder}", flush=True)
    print(json.dumps(metadata(owner), indent=2), flush=True)

    if args.status:
        return _kaggle(["kaggle", "kernels", "status", reference])
    if args.fetch:
        return fetch(reference, args.save)
    if args.stage_only or not args.push:
        if not args.push:
            print("nothing pushed; pass --push to publish and run it", flush=True)
        return 0
    revision = kaggle_queue.ensure_code_dataset(ROOT)
    print(f"code dataset at {revision}", flush=True)
    return _kaggle(["kaggle", "kernels", "push", "-p", str(folder)])


if __name__ == "__main__":
    raise SystemExit(main())
