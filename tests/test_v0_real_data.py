"""Tests for real-data section 5.1 sessions (V0SessionConfig.data_path).

The real-data leg keeps every pre-registered property of the synthetic
sessions - batch shapes, generator discipline, checkpointed stream state,
matched budgets, FLOPs booking - and changes only where the tokens come
from: random windows into a flat uint16 corpus stream. These tests pin the
window sampler, the fail-fast vocab guard, the config validation, and the
contract that matters most: a real-data session still resumes bit-identically
across the publish -> restore -> resume round trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from act.training import (
    V0SessionConfig,
    WindowSampler,
    run_v0_session,
    sample_windows,
)
from act.training.v0_handoff import (
    load_seed_summary,
    publish_seed,
    restore_seed,
)


def _write_stream(path: Path, vocab: int = 64, length: int = 20_000, seed: int = 5) -> Path:
    generator = np.random.default_rng(seed)
    # Zipf-ish skew so some tokens dominate, like a real BPE stream; id 0 is
    # the story boundary and must be present.
    stream = generator.choice(
        np.arange(vocab, dtype=np.uint16), size=length, p=_skewed(vocab)
    )
    stream[::97] = 0  # sprinkle story boundaries
    np.save(path, stream)
    return path


def _skewed(vocab: int) -> np.ndarray:
    weights = 1.0 / np.arange(1, vocab + 1)
    return weights / weights.sum()


def _config(tmp_path: Path, stream: Path, val_stream: Path, **overrides) -> V0SessionConfig:
    values = dict(
        seed=77,
        arms=("random", "adaptive"),
        declared=False,
        device="cpu",
        batch_size=2,
        vocab_size=64,
        context_size=256,
        hidden_size=64,
        depth=2,
        attention_heads=4,
        expert_count=4,
        expert_width=32,
        total_steps=12,
        mutation_interval=6,
        mutation_budget=2,
        probe_candidate_cap=4,
        probe_batch_count=1,
        validation_batches=2,
        checkpoint_interval=4,
        max_checkpoints=2,
        data_path=str(stream),
        val_data_path=str(val_stream),
    )
    values.update(overrides)
    return V0SessionConfig(**values)


# --- window sampler --------------------------------------------------------


def test_sample_windows_shapes_and_values(tmp_path: Path) -> None:
    stream_path = _write_stream(tmp_path / "stream.npy")
    stream = np.load(stream_path)
    generator = torch.Generator().manual_seed(3)
    windows = sample_windows(stream, generator, count=8, width=64)
    assert windows.shape == (8, 64)
    assert windows.dtype == torch.int64
    for row in range(8):
        assert (windows[row] >= 0).all() and (windows[row] < 64).all()


def test_window_sampler_rejects_vocab_overflow(tmp_path: Path) -> None:
    stream_path = _write_stream(tmp_path / "stream.npy", vocab=64)
    with pytest.raises(ValueError, match="vocab_size"):
        WindowSampler(stream_path, vocab_size=32)  # stream contains ids >= 32


def test_load_stream_rejects_wrong_dtype_or_shape(tmp_path: Path) -> None:
    bad_dtype = tmp_path / "bad-dtype.npy"
    np.save(bad_dtype, np.arange(100, dtype=np.int32))
    with pytest.raises(ValueError, match="uint16"):
        WindowSampler(bad_dtype, vocab_size=64)
    bad_shape = tmp_path / "bad-shape.npy"
    np.save(bad_shape, np.zeros((10, 10), dtype=np.uint16))
    with pytest.raises(ValueError, match="flat"):
        WindowSampler(bad_shape, vocab_size=64)


# --- config validation ------------------------------------------------------


def test_config_rejects_missing_stream(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="data stream not found"):
        V0SessionConfig(declared=False, data_path=str(tmp_path / "nope.npy"))


def test_config_rejects_val_stream_without_train_stream(tmp_path: Path) -> None:
    stream = _write_stream(tmp_path / "stream.npy")
    with pytest.raises(ValueError, match="val_data_path requires data_path"):
        V0SessionConfig(declared=False, val_data_path=str(stream))


# --- real-data session behavior ---------------------------------------------


def test_real_data_session_trains_on_stream_windows(tmp_path: Path) -> None:
    stream = _write_stream(tmp_path / "stream.npy")
    val_stream = _write_stream(tmp_path / "val.npy", seed=9)
    result = run_v0_session(_config(tmp_path, stream, val_stream), checkpoint_dir=tmp_path / "s")

    assert result.arms[0].mode == "random"
    # Real text must beat the synthetic floor: a 64-vocab model trained on
    # skewed real tokens evaluates far below ln(64) ~= 4.16 on real windows.
    uniform_floor = np.log(64)
    for arm in result.arms:
        assert arm.final_loss < uniform_floor, (
            f"{arm.mode} loss {arm.final_loss:.3f} at the synthetic floor - "
            "the session trained on the stream, not on uniform tokens"
        )
    # Budgets matched, counters booked.
    assert result.arms[0].mutation_count == result.arms[1].mutation_count
    assert result.arms[1].search_flops > result.arms[0].search_flops


def test_real_data_session_resume_is_bit_identical(tmp_path: Path) -> None:
    """The real-data path keeps the section 5.2 resume contract exactly."""
    stream = _write_stream(tmp_path / "stream.npy")
    val_stream = _write_stream(tmp_path / "val.npy", seed=9)
    config = _config(tmp_path, stream, val_stream, total_steps=16, checkpoint_interval=4)

    reference = run_v0_session(config, checkpoint_dir=tmp_path / "reference")

    session_dir = tmp_path / "sessions"
    run_v0_session(
        _config(tmp_path, stream, val_stream, total_steps=8, checkpoint_interval=4),
        checkpoint_dir=session_dir,
    )
    resumed = run_v0_session(config, checkpoint_dir=session_dir)

    for before, after in zip(reference.arms, resumed.arms):
        assert after.final_loss == before.final_loss
        assert after.architecture_hash == before.architecture_hash
        assert (after.training_flops, after.search_flops) == (
            before.training_flops,
            before.search_flops,
        )
        assert json.dumps(after.mutation_history, sort_keys=True) == json.dumps(
            before.mutation_history, sort_keys=True
        )


def test_real_data_session_publishes_and_round_trips(tmp_path: Path) -> None:
    stream = _write_stream(tmp_path / "stream.npy")
    val_stream = _write_stream(tmp_path / "val.npy", seed=9)
    session_dir = tmp_path / "sessions"
    dataset_root = tmp_path / "dataset"
    result = run_v0_session(_config(tmp_path, stream, val_stream), checkpoint_dir=session_dir)
    publish_seed(
        dataset_root, result, session_dir=session_dir,
        code_revision="test-rev", clean_working_dir=True,
    )
    summary = load_seed_summary(dataset_root, 77)
    assert summary["status"] == "complete"
    # The config payload carries the stream paths for the artifact trail.
    assert summary["config"]["data_path"] == str(stream)
    assert summary["config"]["val_data_path"] == str(val_stream)

    restore_seed(dataset_root, 77, session_dir=session_dir)
    resumed = run_v0_session(_config(tmp_path, stream, val_stream), checkpoint_dir=session_dir)
    assert resumed.arms[0].final_loss == result.arms[0].final_loss


def test_synthetic_and_real_data_streams_differ(tmp_path: Path) -> None:
    """Sanity: the same seed trains on different tokens under the two paths."""
    stream = _write_stream(tmp_path / "stream.npy")
    val_stream = _write_stream(tmp_path / "val.npy", seed=9)
    synthetic = run_v0_session(
        _config(tmp_path, stream, val_stream, data_path=None, val_data_path=None),
        checkpoint_dir=tmp_path / "synthetic",
    )
    real = run_v0_session(
        _config(tmp_path, stream, val_stream),
        checkpoint_dir=tmp_path / "real",
    )
    assert synthetic.arms[0].final_loss != real.arms[0].final_loss


# --- probe data source (section 5.3 heldout diagnostic) ---------------------


def test_config_rejects_heldout_probe_source_without_val_stream() -> None:
    with pytest.raises(ValueError, match="heldout"):
        V0SessionConfig(declared=False, probe_data_source="heldout")


def test_heldout_probe_source_probes_the_val_stream(tmp_path: Path) -> None:
    """Heldout probing is a real stream switch, not a silent no-op.

    Both variants run the identical arm (same seed, same training stream,
    same argmin rule); only the probe windows' source differs. At the FIRST
    event both probe the same pre-mutation model, so if the probe windows
    really come from the held-out stream the recorded winner probe losses
    differ; a broken routing (probe windows still from the training stream)
    would reproduce the train variant's losses bit for bit.
    """
    stream = _write_stream(tmp_path / "stream.npy")
    val_stream = _write_stream(tmp_path / "val.npy", seed=9)
    common = dict(probe_landscape=True, probe_candidate_cap=4, probe_batch_count=1)
    train = run_v0_session(
        _config(tmp_path, stream, val_stream, **common),
        checkpoint_dir=tmp_path / "train-probe",
    )
    heldout = run_v0_session(
        _config(tmp_path, stream, val_stream, probe_data_source="heldout", **common),
        checkpoint_dir=tmp_path / "heldout-probe",
    )
    train_events = train.arms[1].mutation_history
    heldout_events = heldout.arms[1].mutation_history
    assert len(train_events) == len(heldout_events) == 2
    assert all(event["policy"] == "adaptive_probe_argmin" for event in heldout_events)
    assert train_events[0]["probe_candidates"] == 4
    assert heldout_events[0]["probe_candidates"] == 4
    # First event: same model, different probe windows -> different probe loss.
    assert train_events[0]["probe_loss"] != heldout_events[0]["probe_loss"]
    # The probe source is auditable in the published config payload.
    assert train.config["probe_data_source"] == "train"
    assert heldout.config["probe_data_source"] == "heldout"


def test_heldout_probe_source_books_no_extra_cost(tmp_path: Path) -> None:
    """Probe provenance changes no booked cost and no budget discipline."""
    stream = _write_stream(tmp_path / "stream.npy")
    val_stream = _write_stream(tmp_path / "val.npy", seed=9)
    train = run_v0_session(
        _config(tmp_path, stream, val_stream), checkpoint_dir=tmp_path / "train"
    )
    heldout = run_v0_session(
        _config(tmp_path, stream, val_stream, probe_data_source="heldout"),
        checkpoint_dir=tmp_path / "heldout",
    )
    for train_arm, heldout_arm in zip(train.arms, heldout.arms):
        assert heldout_arm.mutation_count == train_arm.mutation_count
        assert heldout_arm.search_flops == train_arm.search_flops
        assert heldout_arm.training_flops == train_arm.training_flops


def test_heldout_probe_resume_is_bit_identical(tmp_path: Path) -> None:
    """The heldout probe stream obeys the section 5.2 resume contract too."""
    stream = _write_stream(tmp_path / "stream.npy")
    val_stream = _write_stream(tmp_path / "val.npy", seed=9)
    config = _config(
        tmp_path,
        stream,
        val_stream,
        probe_data_source="heldout",
        total_steps=16,
        checkpoint_interval=4,
    )

    reference = run_v0_session(config, checkpoint_dir=tmp_path / "reference")

    session_dir = tmp_path / "sessions"
    run_v0_session(
        _config(
            tmp_path,
            stream,
            val_stream,
            probe_data_source="heldout",
            total_steps=8,
            checkpoint_interval=4,
        ),
        checkpoint_dir=session_dir,
    )
    resumed = run_v0_session(config, checkpoint_dir=session_dir)

    for before, after in zip(reference.arms, resumed.arms):
        assert after.final_loss == before.final_loss
        assert after.architecture_hash == before.architecture_hash
        assert (after.training_flops, after.search_flops) == (
            before.training_flops,
            before.search_flops,
        )
        assert json.dumps(after.mutation_history, sort_keys=True) == json.dumps(
            before.mutation_history, sort_keys=True
        )
