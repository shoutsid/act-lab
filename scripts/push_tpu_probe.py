"""Push the TPU hardware probe as a Kaggle kernel, then collect its record.

Kaggle's TPU quota is separate from the GPU quota (20 TPU-hours/week beside the 30
GPU-hours), so a TPU kernel does not queue behind the Gate-4 campaign and does not consume
the hours the campaign needs. One probe push costs a few of those TPU minutes and answers
the port question with numbers instead of a guess - which is the point, because porting the
session path to XLA is a real change (`src/act` has no XLA anywhere in it) and it should
not start until the hardware has been measured doing this workload.

The probe itself is `scripts/tpu_probe.py`; this script only stages and pushes it:

* it publishes the code dataset first (``kaggle_queue.ensure_code_dataset``), because a
  Kaggle script kernel receives **only** its own ``code_file`` - the probe arrives as part
  of the mounted repository, not beside the kernel;
* it generates the entry file, whose one job is to install Torch-XLA if the image does not
  already carry it and then run the mounted probe. On a TPU VM the right package depends on
  the image's torch build, so the entry reports which attempt (if any) worked rather than
  assuming one;
* it pushes with ``enable_tpu`` and *without* ``enable_gpu``, since the two accelerators are
  mutually exclusive in Kaggle's metadata.

    PYTHONPATH=src .venv/bin/python scripts/push_tpu_probe.py            # stage and push
    PYTHONPATH=src .venv/bin/python scripts/push_tpu_probe.py --stage-only
    PYTHONPATH=src .venv/bin/python scripts/push_tpu_probe.py --status
    PYTHONPATH=src .venv/bin/python scripts/push_tpu_probe.py --fetch
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import kaggle_queue  # noqa: E402
import kaggle_t4  # noqa: E402

KERNEL_SLUG = "act-tpu-probe"
ENTRY_NAME = "run-tpu-probe.py"
WORK = ROOT / "artifacts" / "tpu-probe"

#: The entry file. A script kernel gets only this text, so it must be self-sufficient:
#: get Torch-XLA onto the image, find the probe inside the mounted repository, run it.
ENTRY = '''"""Kaggle TPU kernel entry for the hardware probe (generated, do not edit)."""
import glob
import runpy
import subprocess
import sys


def ensure_xla() -> str:
    """Torch-XLA, installing it if the image does not already carry it."""
    try:
        import torch_xla  # noqa: F401
        return "already-present"
    except ImportError:
        pass
    for spec in ("torch_xla", "torch_xla[tpu]"):
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", spec],
            capture_output=True, text=True,
        )
        detail = (proc.stdout + proc.stderr)[-1500:]
        if proc.returncode == 0:
            try:
                import torch_xla  # noqa: F401
                return f"installed:{spec}"
            except ImportError:
                print(f"pip installed {spec} but torch_xla still will not import",
                      flush=True)
        print(f"pip {spec} failed ({proc.returncode}):\\n{detail}", flush=True)
    return "unavailable"


def main() -> int:
    status = ensure_xla()
    print(f"torch_xla: {status}", flush=True)
    matches = sorted(glob.glob("/kaggle/input/**/scripts/tpu_probe.py", recursive=True))
    if not matches:
        print("tpu_probe.py is not in the mounted code dataset", flush=True)
        return 2
    probe = matches[0]
    print(f"probe: {probe}", flush=True)
    sys.argv = [probe, "--steps", str(STEPS), "--workers", str(WORKERS),
                "--dtype", DTYPE, "--output", "/kaggle/working/tpu-probe.json"]
    runpy.run_path(probe, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def metadata(*, owner: str, steps: int, workers: int, dtype: str) -> dict[str, object]:
    """Push metadata for the probe kernel.

    ``enable_tpu`` and ``enable_gpu`` are mutually exclusive on Kaggle, and the TPU is the
    one being measured - a probe that silently landed on a GPU would answer the wrong
    question with a real-looking number.
    """
    return {
        "id": f"{owner}/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": ENTRY_NAME,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        # The GPU is off *and* the TPU is on: the probe's whole purpose is to measure the
        # accelerator the campaign is not using, so it must not fall back to the one it is.
        "enable_gpu": False,
        "enable_tpu": True,
        "machine_shape": "",
        # Internet for the Torch-XLA install and the dataset API; the code dataset is what
        # carries the probe and the repository it imports.
        "enable_internet": True,
        "dataset_sources": [f"{owner}/act-lab-code"],
        "competition_sources": [],
        "kernel_sources": [],
    }


def stage(*, owner: str, steps: int, workers: int, dtype: str) -> Path:
    """Write the kernel folder (entry + metadata) and return it."""
    WORK.mkdir(parents=True, exist_ok=True)
    entry = WORK / ENTRY_NAME
    entry.write_text(
        ENTRY.replace("STEPS", str(steps))
        .replace("WORKERS", str(workers))
        .replace("DTYPE", repr(dtype)),
        encoding="utf-8",
    )
    (WORK / "kernel-metadata.json").write_text(
        json.dumps(metadata(owner=owner, steps=steps, workers=workers, dtype=dtype), indent=2)
        + "\n",
        encoding="utf-8",
    )
    return WORK


def live_campaign_kernel() -> str | None:
    """The Gate-4 kernel training right now, if any.

    Publishing the code dataset while a session has it mounted is the other half of the
    2026-09-21 loss. A session kernel's body and its mounted ``src/`` have to be one
    revision, and the dataset is republished in place - so a publish under a live session
    can leave that session mounting a tree its own body was not staged from. The probe is
    the tool that republishes here, so the probe checks: it can wait, it is not the
    critical path, and the campaign is.
    """
    for mode in sorted(kaggle_t4.QUEUES):
        kernel = str(kaggle_t4.metadata_for(mode)["id"])
        status = kaggle_queue.active_status(kernel)
        if kaggle_queue.should_adopt(status):
            return f"{kernel} is {status}"
    return None


def publish_code_dataset() -> int | None:
    """Publish the code dataset unless a session is live. Returns an exit code to abort with.

    Split out from ``main`` so the refusal is testable without a Kaggle account: the thing
    worth pinning is not the message but that ``ensure_code_dataset`` is never reached.
    """
    live = live_campaign_kernel()
    if live is not None:
        print(
            f"refusing to publish the code dataset while {live}.\n"
            "  Republishing under a live mount can leave a session on a tree its body was "
            "not staged from, and the session is the thing that costs GPU hours. Re-run "
            "this when the campaign is between sessions.",
            file=sys.stderr,
            flush=True,
        )
        return 5
    revision = kaggle_queue.ensure_code_dataset(ROOT)
    print(f"code dataset at {revision}")
    return None


def _kaggle(argv: list[str]) -> int:
    print(f"$ {' '.join(argv)}", flush=True)
    proc = subprocess.run(argv, capture_output=True, text=True)
    print(proc.stdout[-8000:], flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr, flush=True)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12, help="steps timed per measurement")
    parser.add_argument("--workers", type=int, default=8, help="cores for the seed-parallel run")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "both"), default="both")
    parser.add_argument("--stage-only", action="store_true", help="write the folder, do not push")
    parser.add_argument("--status", action="store_true", help="print the kernel's status")
    parser.add_argument("--fetch", action="store_true", help="download the kernel's output")
    args = parser.parse_args()

    owner = kaggle_queue.code_dataset_id().split("/")[0]
    folder = stage(owner=owner, steps=args.steps, workers=args.workers, dtype=args.dtype)
    print(f"staged {folder}")
    print(json.dumps(metadata(owner=owner, steps=args.steps, workers=args.workers,
                              dtype=args.dtype), indent=2))

    ref = f"{owner}/{KERNEL_SLUG}"
    if args.status:
        return _kaggle(["kaggle", "kernels", "status", ref])
    if args.fetch:
        with tempfile.TemporaryDirectory(prefix="tpu-probe-out-") as temporary:
            _kaggle(["kaggle", "kernels", "output", ref, "-p", temporary])
            for path in sorted(Path(temporary).rglob("*")):
                if path.is_file():
                    print(f"--- {path.name}\n{path.read_text(encoding='utf-8', errors='replace')[:8000]}")
        return 0
    if args.stage_only:
        return 0

    # The code dataset has to carry the probe before the kernel that runs it is pushed,
    # and publishing it is the one step here that can disturb a live session - so it is
    # the step that waits for one to finish rather than the step that races it.
    blocked = publish_code_dataset()
    if blocked is not None:
        return blocked
    return _kaggle(["kaggle", "kernels", "push", "-p", str(folder)])


if __name__ == "__main__":
    raise SystemExit(main())
