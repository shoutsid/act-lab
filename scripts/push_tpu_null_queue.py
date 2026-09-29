"""Push the registered null queue's TPU run as a Kaggle kernel, then collect it.

``scripts/tpu_null_queue.py`` is the runner; this stages and pushes it, the same way the probe,
the rehearsal and the price probe are pushed, and for the same two guards:

* the code dataset is republished **through the probe's own guard**
  (``push_tpu_probe.publish_code_dataset``), which refuses while a campaign session is mounted -
  a script kernel receives **only** its own ``code_file``, so the runner and the ``src/`` it
  imports both arrive in that dataset;
* ``enable_tpu`` on and ``enable_gpu`` off, because the two accelerators are mutually exclusive
  on Kaggle and a run that quietly landed on a GPU would produce twelve real-looking seeds
  answering a different question.

Unlike the three measurement kernels, this one is a **queue**: it trains the campaign's declared
block and publishes it to the null queue's own dataset
(``<owner>/act-v0-5-2-null``, via ``tpu_null_queue.DATASET_ROOT``), so what a fetch has to bring
back is a *record of a queue*, and what has to be read is whether twelve seeds landed, whether
every arm did nothing, and what the losses were. That reading is
:func:`tpu_null_queue.verdict`, printed by ``--fetch`` for the reason
``artifacts/tpu-probe/NOTES.md`` records: a result that needs a second hand-run command is a
result that can be left unread.

    PYTHONPATH=src .venv/bin/python scripts/push_tpu_null_queue.py           # stage + push
    PYTHONPATH=src .venv/bin/python scripts/push_tpu_null_queue.py --stage-only
    PYTHONPATH=src .venv/bin/python scripts/push_tpu_null_queue.py --status
    PYTHONPATH=src .venv/bin/python scripts/push_tpu_null_queue.py --fetch \\
        --save artifacts/tpu-null-queue/output
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import kaggle_queue  # noqa: E402
import push_tpu_probe  # noqa: E402
# One definition each of the record name, the dataset id and the reading of the record, so the
# kernel's output name, the fetch's lookup and the verdict cannot drift apart.
import tpu_null_queue  # noqa: E402

KERNEL_SLUG = "act-tpu-null-queue"
ENTRY_NAME = "run-tpu-null-queue.py"
#: What the run writes into the kernel's output, and so the name a fetch looks for. The entry
#: template spells the same path out as a literal (a script kernel receives only its own text, so
#: it cannot import this), and a test pins all three together.
RECORD_NAME = tpu_null_queue.RECORD_NAME
WORK = ROOT / "artifacts" / "tpu-null-queue"

#: The entry file. A script kernel gets only this text, so it must be self-sufficient: get
#: Torch-XLA onto the image (the bootstrap the other three kernels share, imported rather than
#: copied), find the runner inside the mounted repository, run it at the declared fidelity with
#: the registered dataset.
ENTRY = (
    '''"""Kaggle TPU kernel entry for the null queue (generated, do not edit)."""
import glob
import runpy
import sys


'''
    + push_tpu_probe.XLA_BOOTSTRAP
    + '''def main() -> int:
    status = ensure_xla()
    print(f"torch_xla: {status}", flush=True)
    matches = sorted(
        glob.glob("/kaggle/input/**/scripts/tpu_null_queue.py", recursive=True)
    )
    if not matches:
        print("tpu_null_queue.py is not in the mounted code dataset", flush=True)
        return 2
    runner = matches[0]
    print(f"null queue runner: {runner}", flush=True)
    sys.argv = [
        runner, "--device", "xla", "--seeds", "SEED_RANGE", "--steps", str(STEPS),
        "--workers", str(WORKERS), "--fidelity", "declared",
        "--dataset", "DATASET", "--output", "/kaggle/working/tpu-null-queue.json",
    ]
    runpy.run_path(runner, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
)


def metadata(*, owner: str) -> dict[str, object]:
    """Push metadata for the null-queue kernel: TPU on, GPU off, private, internet for the bootstrap."""
    return {
        "id": f"{owner}/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": ENTRY_NAME,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        # The runner names ``xla`` as its device; the two accelerators are exclusive on Kaggle.
        "enable_gpu": False,
        "enable_tpu": True,
        "machine_shape": "",
        # Internet is not optional here: the exit push and the startup restore are Kaggle API
        # calls, and the Torch-XLA bootstrap needs pip when the image does not carry it.
        "enable_internet": True,
        "dataset_sources": [f"{owner}/act-lab-code"],
        "competition_sources": [],
        "kernel_sources": [],
    }


def stage(
    *,
    owner: str,
    seeds: str,
    steps: int,
    workers: int,
    dataset: str,
) -> Path:
    """Write the kernel folder (entry + metadata) and return it.

    Every one of these is substituted into the entry rather than left to the runner's defaults,
    because the entry is the *only* text the kernel is staged from: a placement, a step budget or
    a dataset that is not visible here is a field a reader of the queued run cannot check.
    """
    WORK.mkdir(parents=True, exist_ok=True)
    entry = WORK / ENTRY_NAME
    entry.write_text(
        ENTRY.replace("SEED_RANGE", seeds)
        .replace("STEPS", str(steps))
        .replace("WORKERS", str(workers))
        .replace("DATASET", dataset),
        encoding="utf-8",
    )
    (WORK / "kernel-metadata.json").write_text(
        json.dumps(metadata(owner=owner), indent=2) + "\n", encoding="utf-8"
    )
    return WORK


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", default=tpu_null_queue.DEFAULT_SEEDS, help="the registered block"
    )
    parser.add_argument(
        "--steps", type=int, default=6_000, help="the declared seed's 6,000 steps"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="chips; a v5e-8 exposes 8, and 0 means every chip the host exposes",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="dataset to restore from and publish into; default is the null queue's own",
    )
    parser.add_argument("--stage-only", action="store_true", help="write the folder, do not push")
    parser.add_argument("--status", action="store_true", help="print the kernel's status")
    parser.add_argument("--fetch", action="store_true", help="download the kernel's output")
    parser.add_argument(
        "--save", type=Path, default=None,
        help="directory to keep the fetched record and log in; without it they are only "
        "printed and then deleted with the temporary directory",
    )
    args = parser.parse_args()

    owner = kaggle_queue.code_dataset_id().split("/")[0]
    dataset = args.dataset or f"{owner}/{tpu_null_queue.DATASET_ROOT}"
    folder = stage(
        owner=owner, seeds=args.seeds, steps=args.steps, workers=args.workers, dataset=dataset
    )
    print(f"staged {folder}")
    print(json.dumps(metadata(owner=owner), indent=2))

    ref = f"{owner}/{KERNEL_SLUG}"
    if args.status:
        return push_tpu_probe._kaggle(["kaggle", "kernels", "status", ref])
    if args.fetch:
        saved: list[Path] = []
        with tempfile.TemporaryDirectory(prefix="tpu-null-out-") as temporary:
            code = push_tpu_probe._kaggle(["kaggle", "kernels", "output", ref, "-p", temporary])
            for path in sorted(Path(temporary).rglob("*")):
                if path.is_file():
                    print(
                        f"--- {path.name}\n"
                        f"{path.read_text(encoding='utf-8', errors='replace')[:8000]}"
                    )
            # Inside the ``with``: this is the directory about to be deleted.
            if args.save:
                saved = kaggle_queue.save_fetched(Path(temporary), args.save)
        # The record is a *file*; why a run stopped is in the *log*, and the two come from
        # different commands (`kernels output` has never returned a log). Both are one request
        # away here, so both come back - and both are kept.
        log = kaggle_queue.fetch_kernel_log(ref)
        if args.save:
            target = args.save / f"{Path(ref).name}.log"
            target.write_text(log, encoding="utf-8")
            print(f"\nkept {len(saved) + 1} file(s) in {args.save}")
            for path in [*saved, target]:
                print(f"  {path}")
        print(f"\n--- kernel log ({len(log)} chars, last 6000)\n{log[-6000:]}")
        if args.save and (args.save / RECORD_NAME).is_file():
            record = json.loads((args.save / RECORD_NAME).read_text(encoding="utf-8"))
            print(f"\n--- null queue\n{tpu_null_queue.verdict(record)}")
        return code
    if args.stage_only:
        return 0

    # Publishing the code dataset is the one step here that can disturb a live session, so it is
    # the step that waits rather than the step that races - and it is the probe's own guard,
    # imported rather than re-implemented.
    blocked = push_tpu_probe.publish_code_dataset()
    if blocked is not None:
        return blocked
    return push_tpu_probe._kaggle(["kaggle", "kernels", "push", "-p", str(folder)])


if __name__ == "__main__":
    raise SystemExit(main())
