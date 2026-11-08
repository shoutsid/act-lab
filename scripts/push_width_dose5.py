"""Run section 5.69's ratio-window dose as a CPU-only Kaggle kernel.

The cell is 576 seeds of the toy family's registered config at depth 3, width 61,
which is the study harness's cheapest instrument: ~1.6 s/seed, no accelerator, no
token stream (the substrate is synthetic). This script stages that work as a
script kernel, pushes it against the published code dataset, and fetches the
shard files; the verdict and the summary are pure statistics over the returned
shards and run after the fetch.

    PYTHONPATH=src python scripts/push_width_dose5.py --stage-only
    PYTHONPATH=src python scripts/push_width_dose5.py --push
    PYTHONPATH=src python scripts/push_width_dose5.py --status
    PYTHONPATH=src python scripts/push_width_dose5.py --fetch

The kernel copies the mounted code tree into its writable working directory, so
the harness's shard path exists there; nothing is trained on the local machine.
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

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "artifacts" / "width-dose5" / "kernel"
KERNEL_SLUG = "act-width-dose5"
PROFILE = "width-d3-dose5"
SHARDS = 20
WORKERS = 4
SHARD_DIR = ROOT / "artifacts" / "toy-null-study"


def _kaggle(argv: list[str]) -> int:
    print(f"$ {' '.join(argv)}", flush=True)
    proc = subprocess.run(argv, capture_output=True, text=True)
    print(proc.stdout[-8000:], flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr, flush=True)
    return proc.returncode


ENTRY = '''"""Kaggle CPU kernel entry for section 5.69 (generated, do not edit)."""
import glob
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROFILE = __PROFILE__
SHARDS = __SHARDS__
WORKERS = __WORKERS__
SEEDS = __SEEDS__


def find(pattern):
    for match in sorted(glob.glob(pattern, recursive=True)):
        return Path(match)
    return None


def main() -> int:
    script = find("/kaggle/input/**/scripts/null_toy_study.py")
    if script is None:
        print("null_toy_study.py is not in the mounted code dataset", flush=True)
        return 2
    source = script.parents[1]
    # The harness writes shards under <root>/artifacts/toy-null-study, and the
    # mounted dataset is read-only: run from a writable copy of the tree.
    root = Path("/kaggle/working/act-lab")
    if root.exists():
        shutil.rmtree(root)
    # Only the code the harness needs: the mounted tree also carries artifacts,
    # and copying those would cost minutes for files this run never reads.
    for name in ("scripts", "src"):
        shutil.copytree(source / name, root / name)
    (root / "artifacts" / "toy-null-study").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    def run(shard):
        argv = [
            sys.executable, str(root / "scripts" / "null_toy_study.py"),
            "run", "--shard", str(shard), "--shards", str(SHARDS),
            "--profile", PROFILE,
        ]
        print(f"$ {{' '.join(argv)}}", flush=True)
        proc = subprocess.run(argv, env=env)
        return shard, proc.returncode

    failures = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for shard, code in pool.map(run, range(SHARDS)):
            ok = code == 0
            print(f"{{'ok' if ok else 'FAIL'}} shard {{shard:02d}} exit {{code}}", flush=True)
            if not ok:
                failures.append(shard)

    produced = sorted(
        (root / "artifacts" / "toy-null-study" / PROFILE).glob("shard*.json")
    )
    out = Path("/kaggle/working/shards")
    out.mkdir(parents=True, exist_ok=True)
    experiments = 0
    for path in produced:
        payload = json.loads(path.read_text())
        experiments += len(payload.get("experiments") or [])
        shutil.copy2(path, out / path.name)
    summary = {
        "kind": "width-dose5-summary",
        "profile": PROFILE,
        "shards": SHARDS,
        "shard_files": len(produced),
        "seeds": experiments,
        "expected_seeds": SEEDS,
        "failures": failures,
    }
    (Path("/kaggle/working") / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\\n"
    )
    print("=== WIDTH DOSE5 SUMMARY ===")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 1 if failures or len(produced) != SHARDS else 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def metadata(owner: str) -> dict[str, object]:
    return {
        "id": f"{owner}/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        # Zero GPU-hours, stated twice: the toy family is CPU-only and a kernel
        # that landed on an accelerator would bill the scarcest resource for work
        # that does not need it.
        "enable_gpu": False,
        "enable_tpu": False,
        "machine_shape": "",
        "enable_internet": False,
        "dataset_sources": [kaggle_queue.code_dataset_id()],
        "competition_sources": [],
        "kernel_sources": [],
    }


def stage(owner: str) -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    entry = (
        ENTRY.replace("__PROFILE__", json.dumps(PROFILE))
        .replace("__SHARDS__", str(SHARDS))
        .replace("__WORKERS__", str(WORKERS))
        .replace("__SEEDS__", "576")
    )
    (WORK / "main.py").write_text(entry, encoding="utf-8")
    (WORK / "kernel-metadata.json").write_text(
        json.dumps(metadata(owner), indent=2) + "\n", encoding="utf-8"
    )
    return WORK


def fetch(reference: str) -> int:
    target = SHARD_DIR / PROFILE
    with tempfile.TemporaryDirectory(prefix="width-dose5-out-") as temporary:
        code = _kaggle(["kaggle", "kernels", "output", reference, "-p", temporary])
        temporary_path = Path(temporary)
        shards = sorted(temporary_path.rglob("shard*.json"))
        target.mkdir(parents=True, exist_ok=True)
        for path in shards:
            shutil.copy2(path, target / path.name)
        print(f"saved {len(shards)} shard file(s) under {target}", flush=True)
        for path in sorted(temporary_path.rglob("summary.json")):
            print(f"--- {path.name}\n{path.read_text(errors='replace')[:4000]}", flush=True)
    log = kaggle_queue.fetch_kernel_log(reference)
    if log:
        print("--- kernel log (tail)\n" + log[-6000:], flush=True)
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / f"{KERNEL_SLUG}.log").write_text(log, encoding="utf-8")
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--push", action="store_true", help="publish the code dataset and push")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()

    owner = kaggle_queue.code_dataset_id().split("/")[0]
    reference = f"{owner}/{KERNEL_SLUG}"
    folder = stage(owner)
    print(f"staged {folder}", flush=True)
    print(json.dumps(metadata(owner), indent=2), flush=True)

    if args.status:
        return _kaggle(["kaggle", "kernels", "status", reference])
    if args.fetch:
        return fetch(reference)
    if args.stage_only or not args.push:
        if not args.push:
            print("nothing pushed; pass --push to publish and run it", flush=True)
        return 0
    revision = kaggle_queue.ensure_code_dataset(ROOT)
    print(f"code dataset at {revision}", flush=True)
    return _kaggle(["kaggle", "kernels", "push", "-p", str(folder)])


if __name__ == "__main__":
    raise SystemExit(main())
