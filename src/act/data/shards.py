"""Small deterministic binary token shard format for ACT experiments."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TokenShard:
    path: str
    tokens: int
    checksum: str
    dtype: str


@dataclass(frozen=True)
class ShardManifest:
    version: int
    dataset_id: str
    tokenizer_id: str
    vocab_size: int
    seed: int
    shards: tuple[TokenShard, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> "ShardManifest":
        payload = json.loads(value)
        return cls(
            version=int(payload["version"]),
            dataset_id=str(payload["dataset_id"]),
            tokenizer_id=str(payload["tokenizer_id"]),
            vocab_size=int(payload["vocab_size"]),
            seed=int(payload["seed"]),
            shards=tuple(TokenShard(**shard) for shard in payload["shards"]),
        )


def write_token_shard(path: str | Path, tokens: Iterable[int], vocab_size: int) -> TokenShard:
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    if vocab_size > np.iinfo(np.uint32).max + 1:
        raise ValueError("vocab_size exceeds the token shard format's uint32 capacity")
    values = np.asarray(tuple(tokens), dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("token shard must contain at least one one-dimensional token")
    if values.min() < 0 or values.max() >= vocab_size:
        raise ValueError("token is outside the configured vocabulary")
    dtype = np.uint16 if vocab_size <= np.iinfo(np.uint16).max + 1 else np.uint32
    encoded = values.astype(dtype, copy=False)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(target)
    return TokenShard(
        path=str(target),
        tokens=int(encoded.size),
        checksum=_sha256(target),
        dtype=np.dtype(dtype).name,
    )


def read_tokens(
    shard: TokenShard,
    *,
    start: int = 0,
    count: int | None = None,
    verify_checksum: bool = True,
) -> np.ndarray:
    path = Path(shard.path)
    if verify_checksum and _sha256(path) != shard.checksum:
        raise ValueError(f"checksum mismatch for token shard: {path}")
    if start < 0 or start > shard.tokens:
        raise ValueError("start is outside the shard")
    available = shard.tokens - start
    length = available if count is None else count
    if length < 0 or length > available:
        raise ValueError("count is outside the shard")
    if shard.tokens < 0:
        raise ValueError("token shard length must not be negative")
    if shard.tokens == 0:
        raise ValueError("token shard must contain at least one token")
    try:
        dtype = np.dtype(shard.dtype)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported token shard dtype: {shard.dtype}") from error
    if dtype.name not in {"uint16", "uint32"}:
        raise ValueError(f"unsupported token shard dtype: {shard.dtype}")
    if shard.tokens > np.iinfo(dtype).max + 1:
        raise ValueError("token shard length exceeds dtype capacity")
    expected_bytes = shard.tokens * dtype.itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"token shard byte length mismatch for {path}: "
            f"expected {expected_bytes}, found {actual_bytes}"
        )
    return np.fromfile(path, dtype=dtype, count=length, offset=start * dtype.itemsize)


def build_synthetic_shards(
    directory: str | Path,
    *,
    dataset_id: str = "synthetic-v1",
    tokenizer_id: str = "synthetic-tokenizer-v1",
    vocab_size: int = 128,
    seed: int = 7,
    shard_tokens: int = 4096,
    shard_count: int = 2,
) -> ShardManifest:
    if shard_tokens < 1 or shard_count < 1:
        raise ValueError("shard_tokens and shard_count must be positive")
    generator = np.random.default_rng(seed)
    target = Path(directory)
    shards = []
    for index in range(shard_count):
        values = generator.integers(0, vocab_size, size=shard_tokens, dtype=np.int64)
        shards.append(write_token_shard(target / f"shard-{index:05d}.bin", values, vocab_size))
    return ShardManifest(1, dataset_id, tokenizer_id, vocab_size, seed, tuple(shards))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
