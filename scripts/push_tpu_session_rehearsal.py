"""Push the TPU *session* rehearsal as a Kaggle kernel, then collect its record.

Kaggle's TPU quota is **separate** from the GPU quota (20 TPU-hours/week beside the 30
GPU-hours) and was measured at **19.86 of 20 hours idle** on 2026-09-27, while the GPU week
the Gate-4 campaign is gated on stood at 1.65/30. Section 5.14's option 2 - moving the **null
queue** (twelve single-arm seeds, 16.97 h of T4 wall clock) onto that quota - is priced at
**8.06 h of TPU time at the port rule's own 2.0x bar**, and the port rule's verdict is
already in: `scripts/tpu_probe.py` v6 returned **PORT** (15,034 tokens/s, 2.96x the T4 pair).

What no probe measured is a **session**. The probe timed a step loop around a hand-built
model; the registered path adds activation checkpointing, the optimizer and scaler state, the
per-arm RNG contract, checkpoint save *and resume*, the seed handoff publish, and the two-arm
placement - and none of it has ever run on XLA. So `scripts/tpu_session_rehearsal.py` runs
that path at development fidelity on the TPU, and this script stages and pushes it:

* it publishes the code dataset first, **through the probe's own guard**
  (`push_tpu_probe.publish_code_dataset`), which refuses while a campaign session is mounted -
  a script kernel receives **only** its own ``code_file``, so the rehearsal and the ``src/``
  it imports both arrive in that dataset, and it is replaced in place;
* it generates the entry file: install Torch-XLA through the bootstrap both TPU kernels share
  (`push_tpu_probe.XLA_BOOTSTRAP`), find the mounted rehearsal, run it;
* it pushes with ``enable_tpu`` and *without* ``enable_gpu``, because the two accelerators are
  mutually exclusive on Kaggle and a rehearsal that silently landed on a GPU would answer a
  different question with a real-looking record.

    PYTHONPATH=src .venv/bin/python scripts/push_tpu_session_rehearsal.py            # stage + push
    PYTHONPATH=src .venv/bin/python scripts/push_tpu_session_rehearsal.py --stage-only
    PYTHONPATH=src .venv/bin/python scripts/push_tpu_session_rehearsal.py --status
    PYTHONPATH=src .venv/bin/python scripts/push_tpu_session_rehearsal.py --fetch \\
        --save artifacts/tpu-session-rehearsal/output

``--fetch`` keeps what it downloads *and* says what the record came to, in one line, for the
reason the probe's fetch applies its rule where the record lands: a result that needs a second
hand-run command is a result that can be left unread, and this project has paid for that
already (2026-09-25: a `COMPLETE` probe whose record was printed and deleted, three days
unread). The line it prints is :func:`verdict`, which distinguishes the three states a
rehearsal can come back in - nothing ran, some phases ran, all four ran - because a run that
died in the environment phase has said nothing at all about the session path.
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
# The rehearsal's own name for its record, and its own list of phases: one definition each,
# so the kernel's output name and the fetch's search cannot drift, and `verdict` reads the
# phases the rehearsal actually declares rather than a second copy of them.
import tpu_session_rehearsal  # noqa: E402

KERNEL_SLUG = "act-tpu-session-rehearsal"
ENTRY_NAME = "run-tpu-session-rehearsal.py"
#: What the rehearsal writes into the kernel's output, and so the name a fetch looks for. The
#: entry template below spells the same path out as a literal (a script kernel receives only
#: its own text, so it cannot import this), and a test pins all three together: a rename on one
#: side alone means a fetch that finds a log and no rehearsal.
RECORD_NAME = tpu_session_rehearsal.RECORD_NAME
WORK = ROOT / "artifacts" / "tpu-session-rehearsal"

#: The entry file. A script kernel gets only this text, so it must be self-sufficient: get
#: Torch-XLA onto the image (the bootstrap the probe uses, shared rather than copied), find
#: the rehearsal inside the mounted repository, run it on the TPU.
ENTRY = (
    '''"""Kaggle TPU kernel entry for the session rehearsal (generated, do not edit)."""
import glob
import runpy
import subprocess
import sys


'''
    + push_tpu_probe.XLA_BOOTSTRAP
    + '''

def main() -> int:
    status = ensure_xla()
    print(f"torch_xla: {status}", flush=True)
    matches = sorted(
        glob.glob("/kaggle/input/**/scripts/tpu_session_rehearsal.py", recursive=True)
    )
    if not matches:
        print("tpu_session_rehearsal.py is not in the mounted code dataset", flush=True)
        return 2
    rehearsal = matches[0]
    print(f"rehearsal: {rehearsal}", flush=True)
    sys.argv = [
        rehearsal, "--device", "xla", "--steps", str(STEPS), "--seed", str(SEED),
        "--output", "/kaggle/working/tpu-session-rehearsal.json",
    ]
    runpy.run_path(rehearsal, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
)


def metadata(*, owner: str, steps: int, seed: int) -> dict[str, object]:
    """Push metadata for the rehearsal kernel.

    ``enable_tpu`` on and ``enable_gpu`` off, for the same reason the probe does it: the
    rehearsal names ``xla`` as its device, and a kernel that fell back to a GPU would fail at
    ``resolve_device`` - or worse, answer about the wrong accelerator. The private flag and
    the internet flag match the probe's: the code dataset carries the repository, and the
    internet is what lets the entry install Torch-XLA.
    """
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
        ENTRY.replace("STEPS", str(steps)).replace("SEED", str(seed)),
        encoding="utf-8",
    )
    (WORK / "kernel-metadata.json").write_text(
        json.dumps(metadata(owner=owner, steps=steps, seed=seed), indent=2) + "\n",
        encoding="utf-8",
    )
    return WORK


def verdict(record: dict[str, object]) -> str:
    """One line for a fetched record: what ran, and what stopped the rest.

    The rehearsal has no pre-registered go/no-go to apply the way the probe's ``--evaluate``
    applies its rule - it is a reading, not a decision - so this says what the reading *is*
    and nothing more. The distinction it has to keep is between nothing having run and the
    interesting phase not having run: a record whose only phase is ``environment`` has said
    nothing about the session path, and a line that read as a result there would be the kind
    of record this project treats as worse than no record.
    """
    completed = [str(name) for name in record.get("completed") or []]
    errors = record.get("errors") or []
    detail = "; ".join(
        f"{entry.get('phase')}: {entry.get('error')}"  # type: ignore[union-attr]
        for entry in errors
    )
    missing = [name for name in tpu_session_rehearsal.PHASES if name not in completed]
    if not completed:
        return f"nothing ran ({detail or 'no phase recorded'})"
    if missing:
        return (
            f"partial rehearsal: ran {', '.join(completed)}; "
            f"{', '.join(missing)} did not run ({detail or 'no error recorded'})"
        )
    return (
        "all four phases ran: the registered session path runs on "
        f"{record.get('device_requested')} and its checkpoints are host payloads"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20, help="training steps per arm")
    parser.add_argument("--seed", type=int, default=51, help="dev-fidelity session seed")
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
    print(json.dumps(metadata(owner=owner, steps=args.steps, seed=args.seed), indent=2))

    ref = f"{owner}/{KERNEL_SLUG}"
    if args.status:
        return push_tpu_probe._kaggle(["kaggle", "kernels", "status", ref])
    if args.fetch:
        saved: list[Path] = []
        with tempfile.TemporaryDirectory(prefix="tpu-session-out-") as temporary:
            code = push_tpu_probe._kaggle(
                ["kaggle", "kernels", "output", ref, "-p", temporary]
            )
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
        # different commands (`kernels output` has never returned a log). Both are one
        # request away here, so both come back - and both are kept, which is the half the
        # probe's fetch was missing until 2026-09-25.
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
            print(f"\n--- rehearsal\n{verdict(record)}")
        return code
    if args.stage_only:
        return 0

    # Publishing the code dataset is the one step here that can disturb a live session, so it
    # is the step that waits rather than the step that races - and it is the probe's own
    # guard, imported rather than re-implemented, because two copies of that answer is how the
    # 2026-09-21 session was lost (a republish under a live mount; the session died with no
    # log and no output).
    blocked = push_tpu_probe.publish_code_dataset()
    if blocked is not None:
        return blocked
    return push_tpu_probe._kaggle(["kaggle", "kernels", "push", "-p", str(folder)])


if __name__ == "__main__":
    raise SystemExit(main())
