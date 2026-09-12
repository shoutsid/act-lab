"""Append-only registry for reproducible ACT experiment results."""

from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path

import torch

from .experiment import ExperimentResult


def experiment_hash(result: ExperimentResult) -> str:
    encoded = json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def append_result(path: str | Path, result: ExperimentResult) -> dict[str, object]:
    total_tokens = int(result.config["steps"]) * int(result.config["batch_size"]) * int(result.config["sequence_length"])
    total_training_flops = sum(arm.training_flops for arm in result.arms)
    total_search_flops = sum(arm.search_flops for arm in result.arms)
    record = {
        "experiment_id": f"exp_{experiment_hash(result)}",
        "experiment_hash": experiment_hash(result),
        "config_hash": _stable_hash(result.config),
        "dataset_hash": _stable_hash(
            {"id": result.config["dataset_id"], "seed": result.config["seed"]}
        ),
        "tokenizer_hash": _stable_hash(result.config["tokenizer_id"]),
        "code_revision": result.config["code_revision"],
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "tokens": total_tokens,
        "training_flops": total_training_flops,
        "search_flops": total_search_flops,
        "architecture_hashes": {
            arm.mode: arm.architecture_hash for arm in result.arms
        },
        **result.to_dict(),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record