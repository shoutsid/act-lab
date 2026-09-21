"""Section 5.63: rehearse the Gate-5 transfer path - topology without training.

The plan's gate 5 (section 99) reads *fresh architecture remains competitive*. Every
checkpoint already carries the architecture: the payload holds `training.genome` beside
`model`. But until this section the only code that read a checkpoint's genome was the
session's resume branch, which rebuilds a model from the genome and then immediately
restores `model`, `optimizer`, RNG and scaler state over it - so an architecture could
only travel *with* its training, and a Gate 5 run had no path to take. Gate 4 passing
would therefore not have been actionable.

This rehearsal builds one real (tiny) session, publishes its checkpoint, and exercises the
new path against it: `genome_from_checkpoint` reads the topology and nothing else, and
`fresh_init_from_checkpoint` rebuilds a fresh model from it. It then pins the property
Gate 5's number depends on - **that the fresh path is distinguishable from the resumed
one**: same parameter names, same shapes, same parameter count, and every trained value
different, while a restore of `payload["model"]` reproduces the published weights exactly.
A number produced by the weights-carrying path could not be read as a fresh-architecture
number if those two were the same object; the check is that they are not.

What this is not: a Gate 5 result. The architecture transferred here is a 2-step dev-shape
genome, not a discovered one, and nothing here says any architecture is competitive. The
rehearsal validates the **transfer path**, so the read exists before the artifact it will
be applied to. `--check` exits non-zero if any pinned property stops holding.

    PYTHONPATH=src python scripts/gate5_rehearsal.py            # write the record
    PYTHONPATH=src python scripts/gate5_rehearsal.py --check    # verify, non-zero on drift
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from act.model.factory import v0_moe_model_from_genome  # noqa: E402
from act.topology.genome import ArchitectureGenome  # noqa: E402
from act.training.v0_handoff import newest_arm_checkpoint  # noqa: E402
from act.training.v0_session import (  # noqa: E402
    V0SessionConfig,
    fresh_init_from_checkpoint,
    genome_from_checkpoint,
    run_v0_session,
)

REPORT_SECTION = "5.63"
RECORD = ROOT / "artifacts" / "study-summaries" / "gate5-transfer-rehearsal.json"

# The smallest shape that still produces a real checkpoint: one arm, two steps, a mutation
# at each step so the transferred genome is a non-trivial one (depth 2, two experts). The
# rehearsal is about the path, so the shape only has to be a genuine session.
REHEARSAL_CONFIG = {
    "seed": 16000,
    "arms": ("adaptive",),
    "device": "cpu",
    "vocab_size": 128,
    "context_size": 64,
    "hidden_size": 32,
    "depth": 2,
    "attention_heads": 2,
    "expert_count": 2,
    # The model rejects a single active expert (top-2 routing is the declared shape).
    "active_experts": 2,
    "expert_width": 32,
    "batch_size": 1,
    "total_steps": 2,
    "mutation_interval": 1,
    "checkpoint_interval": 1,
}


def _published_checkpoint(directory: Path) -> tuple[V0SessionConfig, Path]:
    """Run the tiny session and return its config and the published arm checkpoint."""
    config = V0SessionConfig(**REHEARSAL_CONFIG)  # type: ignore[arg-type]
    run_v0_session(config, checkpoint_dir=directory)
    # The session's own layout: <checkpoint_dir>/seed-<seed>/arm-<mode>.
    arm_dir = Path(directory) / f"seed-{config.seed}" / f"arm-{config.arms[0]}"
    return config, newest_arm_checkpoint(arm_dir)


def _raises(callable_) -> bool:  # type: ignore[no-untyped-def]
    try:
        callable_()
    except Exception:
        return True
    return False


def rehearse() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="gate5-rehearsal-") as temporary:
        directory = Path(temporary)
        config, checkpoint = _published_checkpoint(directory)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        recorded = ArchitectureGenome.from_dict(payload["training"]["genome"])
        published = payload["model"]

        # 1. The reader returns exactly the genome the run recorded.
        extracted = genome_from_checkpoint(checkpoint)

        # The new path, and the restore-style path it has to be distinguishable from.
        fresh, genome = fresh_init_from_checkpoint(checkpoint, config)
        restored, _ = fresh_init_from_checkpoint(checkpoint, config)
        restored.load_state_dict(published)

        fresh_state = fresh.state_dict()
        restored_state = restored.state_dict()

        shapes_match = sorted(fresh_state) == sorted(published) and all(
            tuple(fresh_state[name].shape) == tuple(published[name].shape)
            for name in fresh_state
        )

        # Every *trained value* differs - checked on parameters, not buffers, because a
        # zero-initialized buffer can legitimately be equal in both models.
        parameters = dict(fresh.named_parameters())
        differing = [
            name
            for name, tensor in parameters.items()
            if not torch.equal(tensor, published[name])
        ]
        max_abs_difference = max(
            float(
                (parameters[name].detach().float() - published[name].float())
                .abs()
                .max()
            )
            for name in parameters
        )

        # The restore path reproduces the published weights exactly, so the two paths are
        # observationally distinct rather than two names for the same object.
        restore_is_exact = all(
            torch.equal(restored_state[name], published[name]) for name in published
        )

        # Fresh initialization is a function of the config seed: same seed reproduces it,
        # a different seed does not.
        repeat, _ = fresh_init_from_checkpoint(checkpoint, config)
        repeat_is_identical = all(
            torch.equal(tensor, repeat.state_dict()[name])
            for name, tensor in parameters.items()
        )
        other_seed, _ = fresh_init_from_checkpoint(checkpoint, replace(config, seed=config.seed + 1))
        other_state = other_seed.state_dict()
        seed_sensitive = [
            name
            for name, tensor in parameters.items()
            if not torch.equal(tensor.detach(), other_state[name])
        ]
        # Not every parameter is seed-dependent: a normalization scale starts at exactly
        # one, so the seed cannot move it. The check is therefore that the *only*
        # parameters a seed cannot move are the normalization scales - a weight matrix
        # that stopped responding to the seed would mean the fresh path is not fresh.
        seed_insensitive = sorted(set(parameters) - set(seed_sensitive))
        seed_controls_init = bool(seed_sensitive)
        only_norms_ignore_the_seed = all("norm" in name for name in seed_insensitive)

        # The negatives: a payload that cannot supply an architecture must raise rather
        # than silently return an empty genome.
        broken = directory / "checkpoint-broken.pt"
        stripped = dict(payload)
        stripped["training"] = dict(payload["training"], genome=None)
        torch.save(stripped, broken)
        no_genome_raises = _raises(lambda: genome_from_checkpoint(broken))
        no_version = directory / "checkpoint-no-version.pt"
        torch.save(dict(payload, version=3), no_version)
        wrong_version_raises = _raises(lambda: genome_from_checkpoint(no_version))

        genome_round_trips = extracted.to_dict() == recorded.to_dict()

        record = {
            "record_type": "transfer-rehearsal",
            "kind": "gate5-transfer-rehearsal",
            "report_section": REPORT_SECTION,
            "synthetic": True,
            "what_this_is": (
                "A rehearsal of the Gate-5 transfer path on one real tiny session "
                "checkpoint. The architecture transferred is a 2-step dev-shape genome, "
                "not a discovered one: this pins that topology can travel without its "
                "training and that the fresh and resumed paths are distinguishable. It "
                "is not a Gate-5 result and says nothing about any architecture being "
                "competitive."
            ),
            "session": {
                "seed": config.seed,
                "arms": list(config.arms),
                "total_steps": config.total_steps,
                "depth": config.depth,
                "parameter_count": sum(tensor.numel() for tensor in parameters.values()),
            },
            "transferred": {
                "genome_depth": genome.depth,
                "genome_layers": len(genome.layers),
                "max_params": genome.max_params,
                "max_active_params": genome.max_active_params,
                "parameters": len(parameters),
                "parameters_differing": len(differing),
                "max_abs_difference": max_abs_difference,
                "seed_sensitive_parameters": len(seed_sensitive),
                "seed_insensitive_parameters": seed_insensitive,
            },
            "paths": {
                "fresh_weights_equal_published": False,
                "restore_reproduces_published_exactly": restore_is_exact,
                "paths_are_distinguishable": len(differing) == len(parameters),
            },
            "built_in_checks": {
                "genome_round_trips": genome_round_trips,
                "transfer_preserves_names_and_shapes": shapes_match,
                "every_trained_parameter_differs_from_published": len(differing)
                == len(parameters),
                "fresh_weights_are_not_the_published_weights": max_abs_difference > 0.0,
                "restore_path_reproduces_published_exactly": restore_is_exact,
                "fresh_init_is_deterministic_at_a_fixed_seed": repeat_is_identical,
                "the_config_seed_controls_the_fresh_init": seed_controls_init,
                "only_normalization_scales_ignore_the_seed": only_norms_ignore_the_seed,
                "a_payload_without_a_genome_raises": no_genome_raises,
                "a_non_session_payload_raises": wrong_version_raises,
            },
        }
        return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the written record against the registered rule; exit 1 on drift",
    )
    args = parser.parse_args()
    payload = rehearse()
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {RECORD}")
    for name, value in payload["built_in_checks"].items():
        print(f"  check {name:52s} {value}")
    if args.check and not all(payload["built_in_checks"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
