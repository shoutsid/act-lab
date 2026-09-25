"""Run the code-LM byte suite on a Kaggle GPU, beside the local card rather than after it.

The local GTX 1070 runs the campaign serially because one 8 GiB card cannot do two
of these at once. Scoring the 1.3B teacher's held-out bits per byte costs it about
two and a half hours on its own, and that stage sits at the *end* of the campaign -
so the comparison table waits on work that is independent of it. Kaggle's GPU quota
is separate and this job is a self-contained scoring pass, which makes it the one
piece of the pipeline worth moving.

What moves and what does not is a deliberate split, not a convenience:

* **bits per byte moves.** It is deterministic given the text and the model, and it
  is the headline comparison.
* **HumanEval stays local.** Its pass@1 comes from *sampled* decoding, and sampling
  is not bit-reproducible across devices: the same seed on a T4 and a 1070 can draw
  different tokens once a logit differs in the last place, which moves a pass rate
  by a problem or two. Scoring some models on one device and some on another would
  put that difference into the comparison. Four models on one card is worth an hour.

The kernel carries no output back beyond a few kilobytes of JSON, and it is driven
the same way as the other pushed kernels here (``--stage-only``, ``--status``,
``--fetch``). ``--fetch`` prints the kernel's log as well as its output files - the
first version printed only the files, so a run that failed arrived as a summary with
empty rows - and ``--save`` keeps those files, because ``kaggle kernels output`` writes
into a temporary directory that the fetch then deletes, which is how a run reported two
numbers and left nothing on disk to show for the GPU quotas it spent.

Three Kaggle facts this encodes, all recorded in ``docs/T4_RUNBOOK.md`` because each
already cost a run:

1. a dataset mounts **nested** (``/kaggle/input/datasets/<owner>/<slug>/`` and also
   flatter shapes), so the repository is found by searching for ``pyproject.toml``
   beside ``src/act`` rather than by assuming a depth;
2. ``machine_shape`` is the accelerator pin, not ``gpu_type`` - an unknown key is
   dropped silently and the run lands on whatever Kaggle assigns;
3. ``/kaggle/working`` is a **19.5 GiB** mount while the root overlay has 7.9 TB, and
   this job downloads 2.7 GB of teacher plus a 142 MB corpus plus its caches. All of
   it goes under ``/tmp``; only the JSON results are written to ``/kaggle/working``.

    PYTHONPATH=src .venv/bin/python scripts/push_code_bench.py --stage-only
    PYTHONPATH=src .venv/bin/python scripts/push_code_bench.py
    PYTHONPATH=src .venv/bin/python scripts/push_code_bench.py --status
    PYTHONPATH=src .venv/bin/python scripts/push_code_bench.py --fetch --save artifacts/code-lm/bench/kaggle
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import code_bench  # noqa: E402
import kaggle_queue  # noqa: E402

#: Imported rather than repeated: the limit is part of the scoring protocol, and a push
#: at a different one produces rows the comparison table has to refuse.
DEFAULT_BYTES_LIMIT = code_bench.DEFAULT_BYTES_LIMIT

KERNEL_SLUG = "act-code-bench"
ENTRY_NAME = "run-code-bench.py"
WORK = ROOT / "artifacts" / "code-bench-kernel"
MACHINE_SHAPE = "NvidiaTeslaT4"

#: The entry file. A script kernel receives only this text, so it must be
#: self-sufficient: install the one dependency the Kaggle image lacks, find the
#: repository inside the mount, copy it somewhere writable, and drive the scripts.
ENTRY = '''"""Kaggle GPU entry for the code-LM byte suite (generated, do not edit)."""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

WORK = "/tmp/act-code-bench"
MODELS = __MODELS__
BYTES_LIMIT = __BYTES_LIMIT__
SUITE = __SUITE__


def run(argv, cwd, env, label):
    print(f"\\n$ {' '.join(argv)}", flush=True)
    started = time.time()
    proc = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True)
    print(proc.stdout[-20000:], flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-8000:], file=sys.stderr, flush=True)
    print(f"[{label}] rc={proc.returncode} in {time.time() - started:.0f}s", flush=True)
    return proc.returncode


def main() -> int:
    # The caches and the 2.7 GB teacher must not land on /kaggle/working, which is a
    # 19.5 GiB mount; the root overlay has terabytes. Set before importing anything
    # that reads them.
    for variable in ("HF_HOME", "HF_HUB_CACHE", "HF_MODULES_CACHE", "TORCH_HOME"):
        os.environ[variable] = os.path.join(WORK, "hf")
    os.environ["PYTHONPATH"] = "src"
    os.makedirs(WORK, exist_ok=True)

    run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.17.0"], WORK,
        dict(os.environ), "pip")

    matches = sorted(glob.glob("/kaggle/input/**/pyproject.toml", recursive=True))
    repo = None
    for candidate in matches:
        parent = os.path.dirname(candidate)
        if os.path.isdir(os.path.join(parent, "src", "act")) and os.path.isfile(
            os.path.join(parent, "scripts", "code_bench.py")
        ):
            repo = parent
            break
    if repo is None:
        print(f"no mounted repository with scripts/code_bench.py; saw {matches}", flush=True)
        return 2
    print(f"mounted repository: {repo}", flush=True)

    # Copied out of /kaggle/input because that mount is read-only and both scripts
    # write their artifacts beside themselves.
    target = os.path.join(WORK, "repo")
    shutil.copytree(repo, target, dirs_exist_ok=True)
    print(f"working copy: {target}", flush=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(target, "src")
    failed = []
    for stage in (
        ["scripts/train_code_lm.py", "--fetch-teacher"],
        # Only the held-out split is wanted, but it is written by the token stage and
        # the stage is cheap: a 1M-token training stream takes seconds, and building it
        # from the corpus is what keeps this job reproducible from the committed code
        # rather than from a dataset someone has to remember to publish.
        ["scripts/train_code_lm.py", "--stage", "tokens", "--tokens", "1000000"],
    ):
        if run([sys.executable, *stage], target, env, stage[1]):
            failed.append(stage[1])

    for model, tag in MODELS:
        code = run(
            [sys.executable, "scripts/code_bench.py", "--model", model, "--suite", SUITE,
             "--bytes-limit", str(BYTES_LIMIT), "--tag", tag],
            target, env, tag,
        )
        if code:
            failed.append(tag)

    out = "/kaggle/working"
    bench = os.path.join(target, "artifacts", "code-lm", "bench")
    summary = {}
    for path in sorted(glob.glob(os.path.join(bench, "*.json"))):
        shutil.copy(path, out)
        try:
            report = json.load(open(path))
            if "bytes" in report:
                summary[os.path.basename(path)] = {
                    "model": report["bytes"]["model"],
                    "architecture": report["bytes"].get("architecture"),
                    "parameters": report["bytes"]["parameters"],
                    "bits_per_byte": report["bytes"]["bits_per_byte"],
                    "bytes_limit": report.get("bytes_limit"),
                }
        except Exception as error:  # noqa: BLE001 - reported, not raised
            print(f"could not summarise {path}: {error}", flush=True)
    with open(os.path.join(out, "code-bench-summary.json"), "w") as handle:
        json.dump({"summary": summary, "failed": failed, "suite": SUITE}, handle, indent=2, sort_keys=True)
    print("\\nSUMMARY " + json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"FAILED {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def entry_text(models: list[tuple[str, str]], bytes_limit: int, suite: str) -> str:
    return (
        ENTRY.replace("__MODELS__", repr(models))
        .replace("__BYTES_LIMIT__", str(bytes_limit))
        .replace("__SUITE__", repr(suite))
    )


def metadata(*, owner: str, models: list[tuple[str, str]], bytes_limit: int, suite: str) -> dict[str, object]:
    """Push metadata for the benchmark kernel.

    ``machine_shape`` rather than ``gpu_type``: the latter is not an accelerator pin
    and is dropped silently, which would leave the run on whatever Kaggle picked. A
    T4 is the one to ask for here because the teacher is evaluated in float16 and the
    P100 is the wrong card for that.
    """
    return {
        "id": f"{owner}/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": ENTRY_NAME,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "machine_shape": MACHINE_SHAPE,
        # Internet for the teacher (2.7 GB), the corpus, and the baseline tokenizer.
        "enable_internet": True,
        "dataset_sources": [f"{owner}/{kaggle_queue.CODE_DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
        # Deliberately no ``keywords``: Kaggle rejects them as tags and prints a
        # warning on every push, which trains a reader to skim push output.
    }


def stage(*, owner: str, models: list[tuple[str, str]], bytes_limit: int, suite: str) -> Path:
    """Write the kernel folder (entry + metadata) and return it."""
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / ENTRY_NAME).write_text(entry_text(models, bytes_limit, suite), encoding="utf-8")
    (WORK / "kernel-metadata.json").write_text(
        json.dumps(metadata(owner=owner, models=models, bytes_limit=bytes_limit, suite=suite), indent=2) + "\n",
        encoding="utf-8",
    )
    return WORK


def default_models() -> list[tuple[str, str]]:
    """The two models whose bytes the local card would otherwise score last.

    Local directories when the weights are already on disk, because passing the
    repository id would download the teacher's 2.7 GB a second time. Kaggle has no
    such directory, so the kernel is pushed with the hub ids - carrying the ``hf:``
    prefix, because ``code_bench.py`` selects its loader from it. The first push of
    this kernel omitted the prefix: the run downloaded the teacher, spent three
    seconds, and exited with "--model must be hf:<repo> or ours:<checkpoint path>".
    """
    return [
        ("hf:deepseek-ai/deepseek-coder-1.3b-base", "teacher"),
        ("hf:bigcode/tiny_starcoder_py", "baseline"),
    ]


#: The keep step of a fetch, shared with the TPU probe's driver because both fetch kernel
#: output now and the question it answers - *did anything survive the fetch* - is the one the
#: probe's v4 record was lost to (2026-09-25; see ``kaggle_queue.save_fetched``).
save_fetched = kaggle_queue.save_fetched


def _kaggle(argv: list[str]) -> int:
    print(f"$ {' '.join(argv)}", flush=True)
    proc = subprocess.run(argv, capture_output=True, text=True)
    print(proc.stdout[-8000:], flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr, flush=True)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bytes-limit", type=int, default=DEFAULT_BYTES_LIMIT,
                        help="held-out bytes to score; must match the local campaign's value")
    parser.add_argument("--suite", default="bytes",
                        help="benchmark suites; HumanEval is left local on purpose (see the module docstring)")
    parser.add_argument("--hours", type=float, default=1.0,
                        help="wall-clock the kernel is expected to need, in GPU-hours")
    parser.add_argument("--stage-only", action="store_true", help="write the folder, do not push")
    parser.add_argument("--status", action="store_true", help="print the kernel's status")
    parser.add_argument("--fetch", action="store_true", help="download the kernel's output")
    parser.add_argument("--save", type=Path, default=None,
                        help="directory to keep the fetched reports in; without it they are only printed")
    args = parser.parse_args()

    owner = kaggle_queue.code_dataset_id().split("/")[0]
    models = default_models()
    folder = stage(owner=owner, models=models, bytes_limit=args.bytes_limit, suite=args.suite)
    print(f"staged {folder}")
    print(json.dumps(metadata(owner=owner, models=models, bytes_limit=args.bytes_limit, suite=args.suite), indent=2))

    ref = f"{owner}/{KERNEL_SLUG}"
    if args.status:
        return _kaggle(["kaggle", "kernels", "status", ref])
    if args.fetch:
        saved: list[Path] = []
        with tempfile.TemporaryDirectory(prefix="code-bench-out-") as temporary:
            code = _kaggle(["kaggle", "kernels", "output", ref, "-p", temporary])
            for path in sorted(Path(temporary).rglob("*.json")):
                print(f"--- {path.name}\n{path.read_text(encoding='utf-8', errors='replace')[:8000]}")
            # Inside the ``with``: the temporary directory is what is about to be deleted.
            if args.save:
                saved = save_fetched(Path(temporary), args.save)
        if args.save:
            print(f"\nkept {len(saved)} fetched file(s) in {args.save}")
            for path in saved:
                print(f"  {path}")
        # The log is fetched as well as the files, and printed even though it is not a
        # result. `kaggle kernels output` returns files and never a log, so a fetch that
        # only ran that command cannot tell a run that produced nothing from one that
        # failed - and the first version of this driver reported an erroring kernel as a
        # summary with two empty rows, hiding the one line that named the cause.
        log = kaggle_queue.fetch_kernel_log(ref)
        print(f"\n--- kernel log ({len(log)} chars, last 6000)\n{log[-6000:]}")
        return code
    if args.stage_only:
        return 0

    # The code dataset has to carry the very scripts the kernel runs. Publishing it
    # while a session has it mounted can leave that session on a tree its body was not
    # staged from, so the refusal lives in kaggle_queue and is honoured here too.
    live = kaggle_queue.active_status(f"{owner}/act-t4-session")
    if kaggle_queue.should_adopt(live):
        print(
            f"refusing to publish the code dataset while {owner}/act-t4-session is {live}: "
            "re-run when the campaign is between sessions.",
            file=sys.stderr,
        )
        return 5

    # A job that runs out of weekly quota part-way still bills every hour it used and
    # produces nothing, so the gate is the whole declared window. Measured 2026-09-25:
    # 28.70 of 30 GPU-h already spent with a refresh at midnight UTC, which is the state
    # this check exists to make visible *before* a push rather than after.
    quota = kaggle_queue.active_quota()
    shortfall = kaggle_queue.quota_shortfall(quota, args.hours)
    if shortfall is not None:
        print(f"refusing to push: {shortfall}", file=sys.stderr)
        return kaggle_queue.EXIT_QUOTA_INSUFFICIENT
    if quota is not None:
        print(
            f"quota: {quota['remaining_hours']:g} GPU-h remaining of {quota.get('total_hours')} "
            f"(declaring {args.hours:g} h; refreshes {quota.get('refresh_at')})"
        )

    print(f"code dataset at {kaggle_queue.ensure_code_dataset(ROOT)}")
    return _kaggle(["kaggle", "kernels", "push", "-p", str(folder)])


if __name__ == "__main__":
    raise SystemExit(main())
