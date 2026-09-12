"""Append-only registry for reproducible ACT experiment results."""

from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path

import torch

from .experiment import ExperimentResult
from .study import GateStudyResult


def experiment_hash(result: ExperimentResult) -> str:
    encoded = json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def study_hash(study: GateStudyResult) -> str:
    encoded = json.dumps(study.to_dict(), sort_keys=True, separators=(",", ":"))
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


def append_study(path: str | Path, study: GateStudyResult) -> dict[str, object]:
    record = {
        "record_type": "gate_study",
        "study_id": f"study_{study_hash(study)}",
        "study_hash": study_hash(study),
        "config_hash": _stable_hash(study.config),
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "seeds": list(study.seeds),
        "adaptive_wins": study.adaptive_wins,
        "win_rate": study.win_rate,
        "mean_adaptive_delta": study.mean_adaptive_delta,
        "tokens": sum(
            int(experiment.config["steps"])
            * int(experiment.config["batch_size"])
            * int(experiment.config["sequence_length"])
            for experiment in study.experiments
        ),
        "training_flops": sum(
            arm.training_flops
            for experiment in study.experiments
            for arm in experiment.arms
        ),
        "search_flops": sum(
            arm.search_flops
            for experiment in study.experiments
            for arm in experiment.arms
        ),
        "config": study.config,
        "experiments": [experiment.to_dict() for experiment in study.experiments],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record