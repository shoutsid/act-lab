"""Pin the TPU null queue's runner: the block it runs, the roots it writes, and its reading.

Section 5.14's option 2 is the campaign's own priced route onto the TPU week, and the script that
would take it is a *queue*, not a measurement: the registered twelve declared seeds, one per chip,
published into the null queue's own dataset. The failures worth pinning are the quiet ones - a
record whose labels read as declared-scale when the run was the development shape, a block of
checkpoints written into the kernel's 20 GB working mount (which loses the *record*, the one
artifact a fetch reads), and a verdict that cannot say whether every arm did nothing, which is the
property that makes a null seed a null seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import tpu_null_queue  # noqa: E402
import tpu_probe  # noqa: E402


def _args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "device": "xla",
        "seeds": tpu_null_queue.DEFAULT_SEEDS,
        "steps": 20,
        "workers": 1,
        "fidelity": "development",
        "dataset": None,
        "scratch": None,
        "output": Path("/tmp/tpu-null-queue.json"),
    }
    base.update(overrides)
    return argparse.Namespace(**base)


# --- the block it runs -------------------------------------------------------------------


def test_parse_seeds_takes_a_range_a_list_and_the_order_given() -> None:
    assert tpu_null_queue.parse_seeds("15024-15027") == [15024, 15025, 15026, 15027]
    assert tpu_null_queue.parse_seeds("15030, 15024") == [15030, 15024]
    assert tpu_null_queue.parse_seeds("15024-15025,15030") == [15024, 15025, 15030]
    # The registered block, spelled by the runner's own default rather than restated here.
    assert tpu_null_queue.parse_seeds(tpu_null_queue.DEFAULT_SEEDS) == list(range(15024, 15036))


@pytest.mark.parametrize("spec", ["15025-15024", "15024,15024", "  ", "15024-"])
def test_parse_seeds_refuses_a_spec_it_cannot_serve_exactly_once(spec: str) -> None:
    """A backwards range would silently train nothing; a duplicate would publish a seed twice."""
    with pytest.raises(ValueError):
        tpu_null_queue.parse_seeds(spec)


def test_the_declared_seed_is_the_registered_config_and_names_its_own_dtype() -> None:
    """fp32 is the one field this differs from the registered fp16 queue in, so it is pinned."""
    config = tpu_null_queue.seed_config("xla", seed=15024, steps=6_000, fidelity="declared")

    assert config.arms == ("null",)
    assert config.dtype == tpu_null_queue.DTYPE == "fp32"
    assert config.batch_size * config.context_size * config.total_steps == 24_576_000
    assert config.checkpoint_interval == tpu_null_queue.CHECKPOINT_INTERVAL
    # Section 5.47's treatment travels on every arm of every queue; inert for a null arm, and
    # carried anyway, because a manifest that disagrees with the registration is the defect that
    # invalidated the campaign's first block.
    assert config.preserve_router_on_rebuild is True
    assert config.new_expert_initialization == "clone"


def test_the_development_seed_is_the_dev_shape_and_not_the_declared_one() -> None:
    """A dry run that ran the declared shape would be a 43 h run, not a rehearsal."""
    dev = tpu_null_queue.seed_config("cpu", seed=15024, steps=20, fidelity="development")

    assert dev.arms == ("null",)
    assert dev.total_steps == 20
    assert dev.batch_size * dev.context_size * dev.total_steps < 24_576_000


# --- where the roots go ------------------------------------------------------------------


def test_the_local_scratch_root_is_the_output_directory_not_the_record_itself(tmp_path: Path) -> None:
    """A dry run stays inspectable beside its own record."""
    output = tmp_path / "records" / tpu_null_queue.RECORD_NAME

    scratch = tpu_null_queue.scratch_root(output)

    assert scratch == tmp_path / "records" / f"{tpu_null_queue.DATASET_ROOT}-scratch"


def test_a_kernel_scratch_root_is_the_campaign_s_own_answer(tmp_path: Path, monkeypatch) -> None:
    """On a kernel the roots go where ``workspace_root`` puts the campaign's: the overlay.

    Measured on Kaggle: ``/kaggle/working`` is a 20 GB loopback and one seed publishes ~3.2 GiB,
    so a twelve-seed block written there fills the mount - and the file that fails to be written
    is the record. The delegation is pinned rather than the path, because the path is the
    campaign's own decision and this run must not grow a second opinion about it.
    """
    monkeypatch.setattr(tpu_null_queue.kaggle_t4, "run_token", lambda: "0123456789")
    monkeypatch.setattr(
        tpu_null_queue.kaggle_t4, "workspace_root", lambda mode="": tmp_path / f"overlay-{mode}"
    )

    assert tpu_null_queue.scratch_root(tmp_path / "out.json") == tmp_path / "overlay-null-queue"


# --- what the record says ----------------------------------------------------------------


def test_the_labels_say_declared_only_when_the_run_is() -> None:
    """``declared_seed`` is not a disclaimer read backwards: it is the run's whole claim."""
    declared = tpu_null_queue.base_record(_args(fidelity="declared"), "abc123")
    dev = tpu_null_queue.base_record(_args(fidelity="development"), "abc123")

    assert declared["declared_seed"] is True and declared["declared_scale"] is True
    assert dev["declared_seed"] is False and dev["declared_scale"] is False
    for record in (declared, dev):
        assert record["dtype"] == "fp32"
        assert record["queue"] == "null"
        assert record["tpu_dataset_root_name"] == "act-v0-5-2-null"
        assert record["rebuild_treatment"] == tpu_null_queue.REBUILD_TREATMENT
        assert record["code_revision"] == "abc123"
        # The roots are described as living off the working mount, which is where they live.
        assert "overlay" in record["scratch_root"]


def test_the_record_carries_the_dataset_it_would_publish_into() -> None:
    record = tpu_null_queue.base_record(_args(dataset="someone/act-v0-5-2-null"), "abc")

    assert record["dataset"] == "someone/act-v0-5-2-null"
    assert tpu_null_queue.base_record(_args(), "abc")["dataset"] is None


def test_a_skipped_seed_is_read_back_from_its_published_summary(tmp_path: Path) -> None:
    """A block assembled over two runs still has to report every seed's losses."""
    from act.training.v0_handoff import seed_directory

    root = tmp_path / "cert"
    layout = seed_directory(root, 15024)
    (layout / "reports").mkdir(parents=True)
    (layout / "reports" / "seed-summary.json").write_text(
        json.dumps(
            {
                "seed": 15024,
                "status": "complete",
                "arms": {
                    "null": {
                        "final_loss": 10.3972,
                        "validation_loss": 10.4,
                        "search_flops": 0,
                        "mutation_count": 0,
                        "tokens_per_second": 1879.0,
                    }
                },
            }
        )
    )

    entry = tpu_null_queue.skipped_entry(root, 15024)

    assert entry["skipped"]
    assert entry["arm"]["final_loss"] == 10.3972
    assert entry["arm"]["search_flops"] == 0
    assert "summary_read" not in entry


def test_a_skipped_seed_without_a_readable_summary_still_counts_as_complete(tmp_path: Path) -> None:
    entry = tpu_null_queue.skipped_entry(tmp_path / "nothing-here", 15024)

    assert entry["published"] is True
    assert "summary_read" in entry


def test_keeping_the_summaries_copies_them_flat_under_a_seed_carrying_name(tmp_path: Path) -> None:
    """The push can fail after hours of training; these kilobyte files are the insurance."""
    from act.training.v0_handoff import seed_directory

    root = tmp_path / "cert"
    for seed in (15024, 15025):
        layout = seed_directory(root, seed)
        (layout / "reports").mkdir(parents=True)
        # The publisher writes the same payload twice - once under the seed, once under its
        # ``reports/`` - and the keeper's rule is about exactly that.
        for destination in (layout / "seed-summary.json", layout / "reports" / "seed-summary.json"):
            destination.write_text(json.dumps({"seed": seed}))
    output = tmp_path / "working"
    output.mkdir()

    kept = tpu_null_queue.keep_summaries(root, output)

    assert kept == 2
    assert json.loads((output / "seed-15024-summary.json").read_text()) == {"seed": 15024}
    assert json.loads((output / "seed-15025-summary.json").read_text()) == {"seed": 15025}


# --- the reading -------------------------------------------------------------------------


def test_the_verdict_says_how_many_landed_and_that_the_arms_do_nothing() -> None:
    record = {
        "record_type": "tpu-null-queue",
        "dtype": "fp32",
        "seeds_requested": [15024, 15025],
        "seeds_completed": 2,
        "errors": [],
        "seeds": {
            "15024": {"arm": {"final_loss": 10.30, "search_flops": 0, "mutation_count": 0,
                              "tokens_per_second": 1879.0}},
            "15025": {"arm": {"final_loss": 10.50, "search_flops": 0, "mutation_count": 0,
                              "tokens_per_second": 1881.0}},
        },
    }

    line = tpu_null_queue.verdict(record)

    assert "2 of 2 seed(s) landed in fp32 on XLA" in line
    assert "2 of 2 arm(s) do nothing (search_flops 0, mutation_count 0)" in line
    assert "final loss mean 10.4000" in line
    assert "1,880.0 tokens/s per chip" in line


def test_the_verdict_names_a_run_that_did_not_land_its_block() -> None:
    record = {
        "dtype": "fp32",
        "seeds_requested": [15024, 15025],
        "seeds_completed": 1,
        "errors": [15025],
        "seeds": {"15024": {"arm": {"final_loss": 10.3, "search_flops": 0, "mutation_count": 0}}},
    }

    line = tpu_null_queue.verdict(record)

    assert "1 of 2 seed(s) landed" in line
    assert "1 error(s): [15025]" in line


def test_the_verdict_will_not_read_a_record_with_no_seeds_in_it() -> None:
    assert "no seeds" in tpu_null_queue.verdict({"seeds": {}})


def test_an_arm_that_does_not_do_nothing_is_counted_against_the_clause() -> None:
    """The null clause's own built-in check, so a policy arm cannot be read as the null queue."""
    record = {
        "dtype": "fp32",
        "seeds_requested": [15024],
        "seeds_completed": 1,
        "errors": [],
        "seeds": {"15024": {"arm": {"final_loss": 10.3, "search_flops": 512.0,
                                    "mutation_count": 3}}},
    }

    assert "0 of 1 arm(s) do nothing" in tpu_null_queue.verdict(record)


# --- the remote half ---------------------------------------------------------------------


def test_without_a_dataset_there_is_nothing_remote_to_ask_for(tmp_path: Path) -> None:
    """A dry run must not need a Kaggle account to reach its first training step."""
    assert tpu_null_queue.restore_published(tmp_path, None) == {
        "dataset": None,
        "restored": [],
        "error": None,
    }
    assert tpu_null_queue.publish_remote(tmp_path, None, seeds=0) == {
        "dataset": None,
        "status": "not_requested",
    }


# --- the run ------------------------------------------------------------------------------


def test_main_writes_the_record_to_the_output_and_the_roots_to_the_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The defect this pins: 38 GiB of checkpoints landing in the kernel's 20 GB output mount.

    The record is the artifact a fetch reads, so it goes to ``--output``; the handoff roots go to
    ``--scratch``. A run that put them in the same directory would fill the mount and lose the
    record - which is exactly what the runner did before it was split.
    """
    output = tmp_path / "working" / tpu_null_queue.RECORD_NAME
    scratch = tmp_path / "overlay"
    seen: dict[str, Path] = {}

    def fake_run_seed(rank, seed, *, dataset_root, work, record, path, **kwargs):
        seen["dataset_root"] = dataset_root
        seen["work"] = work
        entry = {
            "rank": rank,
            "seed": seed,
            "arm": {
                "final_loss": 10.4,
                "search_flops": 0,
                "mutation_count": 0,
                "tokens_per_second": 1879.0,
            },
        }
        with tpu_null_queue._lock:
            record["seeds"][str(seed)] = entry
            record["completed"].append(seed)
            tpu_probe._save(record, path)
        return entry

    monkeypatch.setattr(tpu_null_queue, "run_seed", fake_run_seed)
    monkeypatch.setattr(tpu_null_queue, "device_count", lambda: 2)

    code = tpu_null_queue.main(
        [
            "--device", "cpu",
            "--fidelity", "development",
            "--seeds", "15024-15025",
            "--steps", "20",
            "--scratch", str(scratch),
            "--output", str(output),
        ]
    )

    assert code == 0
    record = json.loads(output.read_text())
    assert record["seeds_completed"] == 2
    assert record["scratch"] == str(scratch)
    assert record["publish"] == {"dataset": None, "status": "not_requested"}
    assert record["restore"] == {"dataset": None, "restored": [], "error": None}
    assert seen["dataset_root"] == scratch / tpu_null_queue.DATASET_ROOT
    assert seen["work"] == scratch
    assert not seen["dataset_root"].is_relative_to(output.parent)
    assert "2 of 2 seed(s) landed" in capsys.readouterr().out


def test_main_trains_only_the_seeds_the_restored_version_does_not_carry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Section 5.2 at startup: a published seed is never re-trained over."""
    trained: list[int] = []

    def fake_run_seed(rank, seed, *, record, path, **kwargs):
        trained.append(seed)
        entry = {"seed": seed, "arm": {"final_loss": 1.0, "search_flops": 0, "mutation_count": 0}}
        with tpu_null_queue._lock:
            record["seeds"][str(seed)] = entry
            record["completed"].append(seed)
            tpu_probe._save(record, path)
        return entry

    monkeypatch.setattr(tpu_null_queue, "run_seed", fake_run_seed)
    monkeypatch.setattr(tpu_null_queue, "device_count", lambda: 1)
    monkeypatch.setattr(
        tpu_null_queue,
        "restore_published",
        lambda root, dataset, *, log=None: {
            "dataset": dataset,
            "restored": [15024],
            "error": None,
        },
    )
    # Stubbed so the test never authenticates against the real API: the publish's own behaviour
    # is pinned above, and what is under test here is which seeds were handed to the pool.
    monkeypatch.setattr(
        tpu_null_queue,
        "publish_remote",
        lambda *a, **k: {"dataset": "someone/act-v0-5-2-null", "status": "not_requested"},
    )
    output = tmp_path / tpu_null_queue.RECORD_NAME

    code = tpu_null_queue.main(
        [
            "--device", "cpu",
            "--fidelity", "development",
            "--seeds", "15024-15025",
            "--steps", "20",
            "--dataset", "someone/act-v0-5-2-null",
            "--output", str(output),
        ]
    )

    assert code == 0
    assert trained == [15025]
    record = json.loads(output.read_text())
    assert record["seeds_completed"] == 2
    assert record["seeds"]["15024"]["skipped"]


def test_main_stops_before_training_when_the_restore_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that asked for remote state and cannot get it must not train over it."""
    monkeypatch.setattr(
        tpu_null_queue,
        "restore_published",
        lambda root, dataset, *, log=None: {"dataset": dataset, "restored": [], "error": "boom"},
    )
    monkeypatch.setattr(
        tpu_null_queue, "run_seed", lambda *a, **k: pytest.fail("trained after a failed restore")
    )
    output = tmp_path / tpu_null_queue.RECORD_NAME

    code = tpu_null_queue.main(
        [
            "--device", "cpu",
            "--seeds", "15024-15025",
            "--dataset", "someone/act-v0-5-2-null",
            "--output", str(output),
        ]
    )

    assert code == 2
    record = json.loads(output.read_text())
    assert record["errors"] == ["restore"]
    assert record["seeds_completed"] == 0
    assert record["restore"]["error"] == "boom"
