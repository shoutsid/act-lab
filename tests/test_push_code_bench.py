"""Pin the Kaggle driver's three failure modes, none of which announce themselves.

1. **The accelerator pin.** ``gpu_type`` is not a pin - an unrecognised key is dropped
   silently - so the accelerator is asked for with ``machine_shape``. A run on the
   wrong card still produces numbers, which is the dangerous part here rather than an
   error: the teacher is evaluated in float16 and a P100 is the wrong card for that.

2. **The rendered entry.** A script kernel receives only its entry text, so a
   placeholder left unreplaced ships as a ``SyntaxError`` in a run that has already
   been queued for and will still bill its minutes.

3. **Quota.** A job that runs out of the weekly allowance part-way bills every hour it
   used and produces nothing. Measured 2026-09-25: 28.70 of 30 GPU-h already spent with
   a refresh at midnight UTC - the state this guard exists to make visible before a
   push rather than after one.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import kaggle_queue  # noqa: E402
import push_code_bench  # noqa: E402

# The driver's own list, not a copy of it: a duplicate here would have hidden the missing
# ``hf:`` prefix just as well as the duplicate that was in the driver.
MODELS = push_code_bench.default_models()


def test_the_accelerator_is_pinned_not_requested():
    meta = push_code_bench.metadata(owner="someone", models=MODELS, bytes_limit=4_000_000, suite="bytes")
    assert meta["machine_shape"] == push_code_bench.MACHINE_SHAPE
    assert push_code_bench.MACHINE_SHAPE == "NvidiaTeslaT4", "P100 float16 is the wrong card for this teacher"
    assert meta["enable_gpu"] is True
    assert meta["enable_tpu"] is False, "a job that landed on a TPU would answer a question nobody asked"
    assert meta["enable_internet"] is True, "the teacher, the corpus and the baseline all come over the network"
    assert meta["dataset_sources"] == ["someone/" + kaggle_queue.CODE_DATASET_SLUG]
    assert "keywords" not in meta, "Kaggle rejects them as tags and warns on every push"


def test_the_entry_carries_the_models_and_leaves_no_placeholder():
    text = push_code_bench.entry_text(MODELS, 4_000_000, "bytes")
    assert "hf:deepseek-ai/deepseek-coder-1.3b-base" in text
    assert "hf:bigcode/tiny_starcoder_py" in text
    assert "BYTES_LIMIT = 4000000" in text
    assert "__MODELS__" not in text and "__BYTES_LIMIT__" not in text and "__SUITE__" not in text
    ast.parse(text)  # a leftover placeholder is a SyntaxError in a kernel that already billed


def test_the_fetch_prints_the_log_not_only_the_files():
    """``kaggle kernels output`` returns files and never a log.

    The first version of this fetch printed only the JSON, so an erroring kernel arrived
    as a summary with two empty rows - and the one line naming the cause (a rejected
    ``--model``) was downloaded and thrown away with the temporary directory.
    """
    assert "fetch_kernel_log" in Path(push_code_bench.__file__).read_text()


def test_the_entry_keeps_the_bulk_off_the_small_working_mount():
    """``/kaggle/working`` measured at 19.5 GiB against a 7.9 TB root overlay.

    The job downloads a 2.7 GB teacher plus the corpus and its caches. Writing those
    to the small mount is how a run dies on disk rather than on compute.
    """
    text = push_code_bench.entry_text(MODELS, 4_000_000, "bytes")
    assert 'WORK = "/tmp/act-code-bench"' in text
    assert 'out = "/kaggle/working"' in text
    for variable in ("HF_HOME", "HF_HUB_CACHE", "HF_MODULES_CACHE"):
        assert variable in text


def test_the_push_specifies_models_the_benchmark_can_actually_load():
    """Checked against the parser's own contract, because the first push passed bare ids.

    That run downloaded the teacher, then exited in three seconds: ``code_bench.py``
    selects its loader from the ``hf:``/``ours:`` prefix and the driver had left it off.
    Asserting the ids merely look like repository names is what let it through.
    """
    import code_bench

    models = push_code_bench.default_models()
    assert [tag for _, tag in models] == ["teacher", "baseline"]
    for spec, _ in models:
        kind, _, value = spec.partition(":")
        assert kind in code_bench.MODEL_PREFIXES, f"{spec!r} would be rejected by code_bench's parser"
        assert value and not value.startswith("artifacts/"), "Kaggle has no local copy to point at"


QUOTA = {"remaining_hours": 1.29, "used_hours": 28.7, "total_hours": 30.0, "refresh_at": "2026-09-26T00:00:00"}


def _prepare(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, quota, live: str = "COMPLETE", argv: list[str] | None = None):
    """Patch everything that would touch Kaggle, and record what was reached."""
    reached: dict[str, list] = {"kaggle": [], "published": []}
    monkeypatch.setattr(push_code_bench, "WORK", tmp_path)
    monkeypatch.setattr(kaggle_queue, "code_dataset_id", lambda: "someone/act-lab-code")
    monkeypatch.setattr(kaggle_queue, "active_status", lambda kernel: live)
    monkeypatch.setattr(kaggle_queue, "active_quota", lambda resource="GPU": quota)
    monkeypatch.setattr(
        kaggle_queue, "ensure_code_dataset", lambda *a, **k: reached["published"].append(a) or "abc123"
    )
    monkeypatch.setattr(push_code_bench, "_kaggle", lambda argv: reached["kaggle"].append(argv) or 0)
    monkeypatch.setattr(sys, "argv", argv or ["push_code_bench"])
    return reached


def test_a_short_week_refuses_before_publishing_or_pushing(tmp_path, monkeypatch, capsys):
    """Declaring more than remains is refused, and nothing is reached on the way."""
    reached = _prepare(monkeypatch, tmp_path, quota=QUOTA, argv=["push_code_bench", "--hours", "1.5"])
    assert push_code_bench.main() == kaggle_queue.EXIT_QUOTA_INSUFFICIENT
    err = capsys.readouterr().err
    assert "refusing to push" in err
    assert QUOTA["refresh_at"] in err
    assert reached["published"] == [], "a job that cannot finish must not republish the code dataset"
    assert reached["kaggle"] == [], "and must not reach the push"


def test_a_week_that_can_cover_the_declaration_pushes(tmp_path, monkeypatch):
    reached = _prepare(
        monkeypatch, tmp_path, quota=QUOTA, argv=["push_code_bench", "--hours", "1.0"]
    )
    assert push_code_bench.main() == 0
    assert reached["published"], "the kernel runs scripts that live in the code dataset"
    assert any(argv[:3] == ["kaggle", "kernels", "push"] for argv in reached["kaggle"])


def test_an_unreadable_quota_does_not_block(tmp_path, monkeypatch):
    """``None`` means cannot-tell, and cannot-tell is not a reason to stall."""
    reached = _prepare(monkeypatch, tmp_path, quota=None, argv=["push_code_bench", "--hours", "2.0"])
    assert push_code_bench.main() == 0
    assert any(argv[:3] == ["kaggle", "kernels", "push"] for argv in reached["kaggle"])


def test_a_live_session_stops_the_republish(tmp_path, monkeypatch, capsys):
    """A session mounting the dataset must not have it replaced underneath it."""
    reached = _prepare(monkeypatch, tmp_path, quota=QUOTA, live="RUNNING")
    assert push_code_bench.main() == 5
    assert "refusing to publish" in capsys.readouterr().err
    assert reached["published"] == []
    assert reached["kaggle"] == []


def test_staging_writes_a_folder_kaggle_accepts(tmp_path, monkeypatch):
    monkeypatch.setattr(push_code_bench, "WORK", tmp_path)
    folder = push_code_bench.stage(owner="someone", models=MODELS, bytes_limit=4_000_000, suite="bytes")
    import json

    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert meta["code_file"] == push_code_bench.ENTRY_NAME
    assert (folder / push_code_bench.ENTRY_NAME).is_file()
    assert meta["id"] == f"someone/{push_code_bench.KERNEL_SLUG}"
