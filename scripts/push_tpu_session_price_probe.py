"""Push the declared-shape TPU *session price* probe as a Kaggle kernel, then collect it.

``scripts/tpu_session_price_probe.py`` measures the one number section 5.14's option 2 is
priced on and that no run has measured on the path that would run it: what the registered
session path sustains at the **declared** 285M shape on a TPU. The port rule's 8.06 h comes
from ``tpu_probe.py`` v6, which timed a hand-built step loop; the rehearsal proved the session
path runs, at development fidelity, and its record says its tokens/s is not a schedule input.

This script stages and pushes the price probe the same way the rehearsal is pushed, with the
same two guards, because they are the ones the campaign has already paid for:

* the code dataset is republished **through the probe's own guard**
  (``push_tpu_probe.publish_code_dataset``), which refuses while a campaign session is mounted
  - a script kernel receives **only** its own ``code_file``, so the price probe and the
  ``src/`` it imports both arrive in that dataset;
* ``enable_tpu`` on and ``enable_gpu`` off, because the two accelerators are mutually
  exclusive on Kaggle and a price measured on a GPU would be a real-looking number answering
  a different question.

    PYTHONPATH=src .venv/bin/python scripts/push_tpu_session_price_probe.py           # stage + push
    PYTHONPATH=src .venv/bin/python scripts/push_tpu_session_price_probe.py --stage-only
    PYTHONPATH=src .venv/bin/python scripts/push_tpu_session_price_probe.py --status
    PYTHONPATH=src .venv/bin/python scripts/push_tpu_session_price_probe.py --fetch \\
        --save artifacts/tpu-session-price/output

``--fetch`` keeps what it downloads *and* prints :func:`verdict` in one line, for the reason
the probe's and the rehearsal's fetches do: a result that needs a second hand-run command is a
result that can be left unread.
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
# One definition each of the record name and the phase order, so the kernel's output name,
# the fetch's lookup and the verdict's reading of which phases ran cannot drift apart.
import tpu_session_price_probe as price_probe  # noqa: E402

KERNEL_SLUG = "act-tpu-session-price"
ENTRY_NAME = "run-tpu-session-price.py"
#: What the probe writes into the kernel's output, and so the name a fetch looks for. The
#: entry template spells the same path out as a literal (a script kernel receives only its own
#: text, so it cannot import this), and a test pins all three together.
RECORD_NAME = price_probe.RECORD_NAME
WORK = ROOT / "artifacts" / "tpu-session-price"

#: The entry file. A script kernel gets only this text, so it must be self-sufficient: get
#: Torch-XLA onto the image (the bootstrap the probe and the rehearsal share, imported rather
#: than copied), find the price probe inside the mounted repository, run it on the TPU.
ENTRY = (
    '''"""Kaggle TPU kernel entry for the session price probe (generated, do not edit)."""
import glob
import runpy
import sys


'''
    + push_tpu_probe.XLA_BOOTSTRAP
    + '''
def main() -> int:
    status = ensure_xla()
    print(f"torch_xla: {status}", flush=True)
    matches = sorted(
        glob.glob("/kaggle/input/**/scripts/tpu_session_price_probe.py", recursive=True)
    )
    if not matches:
        print("tpu_session_price_probe.py is not in the mounted code dataset", flush=True)
        return 2
    probe = matches[0]
    print(f"price probe: {probe}", flush=True)
    sys.argv = [
        probe, "--device", "xla", "--steps", str(STEPS), "--seed", str(SEED),
        "--output", "/kaggle/working/tpu-session-price-probe.json",
    ]
    runpy.run_path(probe, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
)


def metadata(*, owner: str) -> dict[str, object]:
    """Push metadata for the price probe: TPU on, GPU off, private, internet for the bootstrap."""
    return {
        "id": f"{owner}/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": ENTRY_NAME,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": True,
        "machine_shape": "",
        "enable_internet": True,
        "dataset_sources": [f"{owner}/act-lab-code"],
        "competition_sources": [],
        "kernel_sources": [],
    }


def stage(*, owner: str, steps: int, seed: int) -> Path:
    """Write the kernel folder (entry + metadata) and return it."""
    WORK.mkdir(parents=True, exist_ok=True)
    entry = WORK / ENTRY_NAME
    entry.write_text(
        ENTRY.replace("STEPS", str(steps)).replace("SEED", str(seed)), encoding="utf-8"
    )
    (WORK / "kernel-metadata.json").write_text(
        json.dumps(metadata(owner=owner), indent=2) + "\n", encoding="utf-8"
    )
    return WORK


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps", type=int, default=240, help="declared-shape training steps (one arm)"
    )
    parser.add_argument("--seed", type=int, default=price_probe.SEED)
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
    folder = stage(owner=owner, steps=args.steps, seed=args.seed)
    print(f"staged {folder}")
    print(json.dumps(metadata(owner=owner), indent=2))

    ref = f"{owner}/{KERNEL_SLUG}"
    if args.status:
        return push_tpu_probe._kaggle(["kaggle", "kernels", "status", ref])
    if args.fetch:
        saved: list[Path] = []
        with tempfile.TemporaryDirectory(prefix="tpu-price-out-") as temporary:
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
            print(f"\n--- price\n{price_probe.verdict(record)}")
        return code
    if args.stage_only:
        return 0

    # Publishing the code dataset is the one step here that can disturb a live session, so it
    # is the step that waits rather than the step that races - and it is the probe's own
    # guard, imported rather than re-implemented.
    blocked = push_tpu_probe.publish_code_dataset()
    if blocked is not None:
        return blocked
    return push_tpu_probe._kaggle(["kaggle", "kernels", "push", "-p", str(folder)])


if __name__ == "__main__":
    raise SystemExit(main())
