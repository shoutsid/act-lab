"""Deterministic token streams, shard manifests, and bounded readers."""

from .shards import (
    ShardManifest,
    TokenShard,
    build_synthetic_shards,
    read_tokens,
    write_token_shard,
)

__all__ = [
    "ShardManifest",
    "TokenShard",
    "build_synthetic_shards",
    "read_tokens",
    "write_token_shard",
]
