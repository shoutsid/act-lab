"""Section 5.70's registration: the declared-shape real-text block.

The block exists because the declared Gate-4 block answered its cost question
on synthetic tokens at the uniform-vocabulary floor, and the only real-text
evidence is at development fidelity. These tests pin the seeds, the venue and
resources, the stream, the bar and the caveat the record has to carry, so the
registration cannot drift between now and the twelve pairs landing.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import kaggle_t4  # noqa: E402
import push_real_text_declared as block  # noqa: E402

NOTES = ROOT / "artifacts" / "real-text-declared" / "NOTES.md"


def test_the_block_is_the_next_twelve_seeds_after_gate_4() -> None:
    assert block.SEEDS == tuple(range(15036, 15048))
    # The waves are a scheduling split, not a protocol split: together they are
    # exactly the registered block, once each, in order.
    assert [seed for wave in sorted(block.WAVES) for seed in block.WAVES[wave]] == list(
        block.SEEDS
    )
    assert sorted(block.WAVES) == ["a", "b", "c"]


def test_the_venue_is_the_declared_gpu_and_the_real_stream() -> None:
    meta = block.metadata("owner", "a")
    assert meta["enable_gpu"] is True
    assert meta["enable_tpu"] is False
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert f"owner/{block.TOKENS_DATASET}" in meta["dataset_sources"]
    # No restore, no publish: the container needs no Kaggle API credential, which
    # is the defect that left the TPU null queue at zero seeds.
    assert meta["enable_internet"] is False


def test_the_entry_runs_both_arms_on_the_declared_shape() -> None:
    folder = block.stage("owner", "a", "deadbeef")
    entry = (folder / "main.py").read_text()
    assert block.WAVES["a"] == (15036, 15037, 15038, 15039)
    assert "15036" in entry and "15039" in entry
    assert 'arms=("adaptive", "random")' in entry
    assert 'dtype=DTYPE' in entry and 'DTYPE = "fp16"' in entry
    assert "STEPS = 6000" in entry
    assert 'preserve_router_on_rebuild=True' in entry
    assert 'new_expert_initialization="clone"' in entry
    assert json.dumps(kaggle_t4.PRIMARY_POLICY) in entry


def test_the_bar_is_the_campaigns_exact_sign_test() -> None:
    p = sum(math.comb(12, wins) for wins in range(block.BAR_WINS, 13)) / 2**12
    assert p == 0.019287109375
    text = NOTES.read_text()
    assert "10 of 12" in text
    assert "0.019287" in text
    assert "5.70" in text


def test_the_registration_names_the_shared_stream_caveat() -> None:
    """The record must not let the block read as twelve independent text samples."""
    text = NOTES.read_text().lower()
    assert "40m tokens" in text
    assert "share" in text
    # The registration is wrapped for width, so the caveat is matched on
    # normalised whitespace rather than on one physical line.
    assert "not twelve independent text samples" in " ".join(text.split())


def test_the_prediction_is_declared_before_the_run() -> None:
    text = NOTES.read_text()
    assert "at most 3 of 12" in text
    # The three branches are fixed here, not chosen after the numbers land.
    for phrase in (">= 10/12", "<= 2/12", "3-9 wins"):
        assert phrase in text
