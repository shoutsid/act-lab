from pathlib import Path

import numpy as np
import pytest

from act.data import ShardManifest, build_synthetic_shards, read_tokens, write_token_shard


def test_synthetic_shards_are_reproducible(tmp_path: Path) -> None:
    first = build_synthetic_shards(tmp_path / "one", seed=11, shard_tokens=32)
    second = build_synthetic_shards(tmp_path / "two", seed=11, shard_tokens=32)
    assert [shard.checksum for shard in first.shards] == [
        shard.checksum for shard in second.shards
    ]
    assert ShardManifest.from_json(first.to_json()) == first


def test_reads_reject_truncated_or_mismatched_token_counts(tmp_path: Path) -> None:
    shard = write_token_shard(tmp_path / "tokens.bin", range(20), vocab_size=32)

    truncated = tmp_path / "truncated.bin"
    truncated.write_bytes(Path(shard.path).read_bytes()[:-2])
    with pytest.raises(ValueError, match="byte length mismatch"):
        read_tokens(
            type(shard)(str(truncated), shard.tokens, shard.checksum, shard.dtype),
            verify_checksum=False,
        )

    oversized_manifest = type(shard)(shard.path, shard.tokens + 1, shard.checksum, shard.dtype)
    with pytest.raises(ValueError, match="byte length mismatch"):
        read_tokens(oversized_manifest, verify_checksum=False)


def test_reads_reject_unsupported_token_dtype(tmp_path: Path) -> None:
    shard = write_token_shard(tmp_path / "tokens.bin", range(20), vocab_size=32)
    unsupported = type(shard)(shard.path, shard.tokens, shard.checksum, "float32")

    with pytest.raises(ValueError, match="unsupported token shard dtype"):
        read_tokens(unsupported, verify_checksum=False)


def test_bounded_reads_verify_checksum(tmp_path: Path) -> None:
    shard = write_token_shard(tmp_path / "tokens.bin", range(20), vocab_size=32)
    np.testing.assert_array_equal(read_tokens(shard, start=4, count=5), np.arange(4, 9))
    with (tmp_path / "tokens.bin").open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(ValueError, match="checksum"):
        read_tokens(shard)
