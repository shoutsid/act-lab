"""On-demand integration self-check for the V0 session machinery.

NOT part of the pytest suite (sessions are too slow for per-commit runs).
Run this once before launching the declared protocol, after any change to
``act.training.v0_session`` or checkpoint code:

    PYTHONPATH=src python scripts/v0_session_selfcheck.py

The declared run trains on CUDA in fp16, so run it against that pairing at
least once - the resume checks are the pre-registered determinism claim, and
they are only guaranteed for a device/dtype they have actually been run on:

    PYTHONPATH=src python scripts/v0_session_selfcheck.py --device cuda

Checks, on a tiny non-declared config:
  1. a reference session matches budgets and books search/training FLOPs;
  2. deleting the final checkpoint and resuming from the step-15 snapshot
     reproduces the reference run bit-identically (losses, FLOPs counters,
     architecture hashes, mutation histories);
  3. an fp16 session (autocast + GradScaler) also resumes bit-identically
     with its scaler state restored, and books the same FLOPs counters as
     fp32 (dtype books operations, not time);
  4. checkpoints carry the version-5 payload contract;
  5. training the pair's arms in separate processes - one per device, the way
     section 5.2 prices a pair - reproduces the sequential run's arm results
     exactly. The arms share nothing but the seed, so this must hold on any
     device count; it is checked here rather than in the suite because it
     spawns processes.
"""

import argparse
from dataclasses import asdict
import json
import sys
import tempfile
from pathlib import Path

import torch

from act.training import V0SessionConfig, run_v0_session, validate_checkpoint_payload
from act.training.runtime import resolve_device


# Set once by ``--device`` in main. A module constant rather than a parameter
# so no call site can forget it: a battery that compared a checkpoint produced
# on one device against a resume on another would still pass while testing
# nothing, and every check in here is a resume comparison.
DEVICE = "cpu"


def _config(**overrides) -> V0SessionConfig:
    values = dict(
        seed=11,
        arms=("random", "adaptive"),
        declared=False,
        device=DEVICE,
        batch_size=2,
        vocab_size=64,
        context_size=256,
        hidden_size=64,
        depth=2,
        attention_heads=4,
        expert_count=4,
        expert_width=32,
        total_steps=20,
        mutation_interval=10,
        mutation_budget=2,
        probe_candidate_cap=4,
        probe_batch_count=1,
        validation_batches=2,
        checkpoint_interval=5,
        max_checkpoints=3,
    )
    values.update(overrides)
    return V0SessionConfig(**values)


def _newest_checkpoint(arm_dir: Path) -> Path:
    return max(arm_dir.glob("checkpoint-*.pt"), key=lambda p: p.stat().st_mtime_ns)


def main(argv: list[str] | None = None) -> int:
    from act.training.v0_session import V0SessionResult

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cpu",
        help=(
            "device for the battery: auto, cpu, cuda, or an indexed cuda:N "
            "(default cpu; the declared run trains on cuda)"
        ),
    )
    args = parser.parse_args(argv)
    global DEVICE
    DEVICE = str(resolve_device(args.device))
    print(f"v0 session self-check on {DEVICE}", flush=True)

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="v0-selfcheck-") as tmp:
        root = Path(tmp)

        print("[1/5] reference session ...", flush=True)
        reference = run_v0_session(_config(), checkpoint_dir=root / "ref")
        adaptive, random_arm = reference.arms[1], reference.arms[0]
        if adaptive.mutation_count != random_arm.mutation_count:
            failures.append("mutation budgets diverged between arms")
        # Training FLOPs need not match across arms after the first mutation
        # event (architectures diverge), but the pre-event prefix is identical
        # by construction: both arms train the initial parameters for the
        # first ``mutation_interval`` steps with no event before it.
        from act.model import estimated_parameter_count

        initial_params = estimated_parameter_count(_config().model_config())[0]
        prefix_flops = (
            _config().mutation_interval
            * initial_params
            * _config().batch_size
            * _config().context_size
            * 2
        )
        for arm in reference.arms:
            if arm.training_flops < prefix_flops:
                failures.append(
                    f"arm {arm.mode} booked less training FLOPs than the "
                    "pre-mutation prefix requires"
                )
        if not (adaptive.search_flops > random_arm.search_flops > 0):
            failures.append("adaptive arm did not book more search FLOPs than random")
        for arm in reference.arms:
            if arm.parameter_count <= 0 or arm.tokens_per_second <= 0 or not arm.architecture_hash:
                failures.append(f"arm {arm.mode} recorded degenerate metrics")

        print("[2/5] resume equivalence ...", flush=True)
        run_v0_session(_config(), checkpoint_dir=root / "resume")
        for mode in ("random", "adaptive"):
            arm_dir = root / "resume" / "seed-11" / f"arm-{mode}"
            newest = _newest_checkpoint(arm_dir)
            if not newest.stem.endswith("0020"):
                failures.append(f"{mode}: newest checkpoint is {newest.name}, expected step 20")
            newest.unlink()  # force a mid-run resume from the step-15 checkpoint
        resumed = run_v0_session(_config(), checkpoint_dir=root / "resume")
        for before, after in zip(reference.arms, resumed.arms):
            if after.final_loss != before.final_loss:
                failures.append(f"{before.mode}: resumed final loss diverged")
            if after.architecture_hash != before.architecture_hash:
                failures.append(f"{before.mode}: resumed architecture diverged")
            if (after.training_flops, after.search_flops) != (
                before.training_flops,
                before.search_flops,
            ):
                failures.append(f"{before.mode}: resumed FLOPs counters diverged")
            if json.dumps(after.mutation_history, sort_keys=True) != json.dumps(
                before.mutation_history, sort_keys=True
            ):
                failures.append(f"{before.mode}: resumed mutation history diverged")

        print("[3/5] fp16 resume equivalence ...", flush=True)
        fp16_config = _config(dtype="fp16")
        fp16_reference = run_v0_session(fp16_config, checkpoint_dir=root / "fp16-ref")
        for mode in ("random", "adaptive"):
            arm_dir = root / "fp16-resume" / "seed-11" / f"arm-{mode}"
            run_v0_session(fp16_config, checkpoint_dir=root / "fp16-resume")
            newest = _newest_checkpoint(arm_dir)
            if not newest.stem.endswith("0020"):
                failures.append(f"fp16 {mode}: newest checkpoint is {newest.name}")
            newest.unlink()
        fp16_resumed = run_v0_session(fp16_config, checkpoint_dir=root / "fp16-resume")
        for before, after in zip(fp16_reference.arms, fp16_resumed.arms):
            if after.final_loss != before.final_loss:
                failures.append(f"fp16 {before.mode}: resumed final loss diverged")
            if after.architecture_hash != before.architecture_hash:
                failures.append(f"fp16 {before.mode}: resumed architecture diverged")
            if (after.training_flops, after.search_flops) != (
                before.training_flops,
                before.search_flops,
            ):
                failures.append(f"fp16 {before.mode}: resumed FLOPs counters diverged")
        for fp32_arm, fp16_arm in zip(reference.arms, fp16_reference.arms):
            if (fp16_arm.training_flops, fp16_arm.search_flops) != (
                fp32_arm.training_flops,
                fp32_arm.search_flops,
            ):
                failures.append(
                    f"{fp32_arm.mode}: fp16 changed the FLOPs counters "
                    "(dtype must book operations, not time)"
                )

        print("[4/5] checkpoint contract ...", flush=True)
        run_v0_session(
            _config(
                arms=("adaptive",),
                total_steps=10,
                checkpoint_interval=5,
                max_checkpoints=1,
            ),
            checkpoint_dir=root / "ckpt",
        )
        arm_dir = root / "ckpt" / "seed-11" / "arm-adaptive"
        retained = sorted(path.name for path in arm_dir.glob("checkpoint-*.pt"))
        if retained != ["checkpoint-000010.pt"]:
            failures.append(f"checkpoint retention produced {retained}")
        else:
            payload = torch.load(
                arm_dir / retained[-1], map_location="cpu", weights_only=False
            )
            if payload.get("version") != 5:
                failures.append("checkpoint payload is not version 5")
            validate_checkpoint_payload(payload)
            if payload["v0_session"]["mutations"] != 1:
                failures.append("checkpoint did not record the mutation count")
            if payload["training"]["step"] != 10:
                failures.append("checkpoint did not record the training step")

        print("[5/5] per-arm processes reproduce the sequential arms ...", flush=True)
        from act.training.v0_session import (
            _train_arms,
            _train_arms_in_processes,
            arm_devices,
        )

        # The arms are independent by construction - each seeds itself from the
        # session seed and writes only its own arm directory - so scheduling
        # them concurrently must not change one number. That is the property
        # section 5.2's pair time depends on, and it is checked on whatever
        # device count this host has: the placement is real on a two-GPU host
        # and the process boundary is what is under test on a one-GPU one.
        parallel_config = _config(seed=29)
        parallel_device = torch.device(DEVICE)
        devices = arm_devices(len(parallel_config.arms), parallel_device)
        if parallel_device.type == "cuda":
            # Logged next to the placement: a one-device host is a valid
            # sequential pass, and the count is how a reader tells the two
            # apart without knowing which machine ran this.
            print(f"    host devices: {torch.cuda.device_count()}", flush=True)
        sequential = _train_arms(
            parallel_config,
            root / "arms-seq",
            parallel_device,
            None,
            deadline_check=None,
            deadline_budget_seconds=None,
        )
        concurrent = _train_arms_in_processes(
            parallel_config,
            root / "arms-par",
            devices,
            None,
        )
        # Placement and timing fields, not results: the concurrent run names
        # its device per arm (cuda:0, cuda:1) where the sequential run names
        # one device for both. That difference is the parallelism working.
        volatile = {"wall_time_seconds", "tokens_per_second", "device"}
        if [arm.mode for arm in concurrent] != [arm.mode for arm in sequential]:
            failures.append("parallel arms came back in a different order")
        # The placement itself, read back from the arms: this is the only
        # durable evidence that section 5.2's "one arm per GPU" is what ran,
        # and it is the check that can only be made on the target host - two
        # arms on two devices, not two arms queued on one.
        expected_devices = [str(arm_device) for arm_device in devices]
        if [arm.device for arm in concurrent] != expected_devices:
            failures.append(
                "parallel arms recorded unexpected devices "
                f"({[arm.device for arm in concurrent]} != {expected_devices})"
            )
        elif len(set(expected_devices)) > 1:
            print(
                f"    one arm per device: {', '.join(expected_devices)}",
                flush=True,
            )

        # The same property through the session entry point, so the dispatch
        # that chooses sequential or concurrent arms is covered rather than
        # only the process runner it calls - and a two-device session is the
        # configuration section 5.2 prices.
        placed = run_v0_session(parallel_config, checkpoint_dir=root / "arms-session")
        if [arm.device for arm in placed.arms] != expected_devices:
            failures.append(
                "session-run arms recorded unexpected devices "
                f"({[arm.device for arm in placed.arms]} != {expected_devices})"
            )
        for before, after in zip(sequential, placed.arms):
            for key, value in asdict(before).items():
                if key in volatile:
                    continue
                if asdict(after)[key] != value:
                    failures.append(
                        f"{before.mode}: session-run arms changed {key} "
                        f"({value!r} -> {asdict(after)[key]!r})"
                    )
        for before, after in zip(sequential, concurrent):
            if asdict(before).keys() != asdict(after).keys():
                failures.append(f"{before.mode}: arm result shape changed")
                continue
            for key, value in asdict(before).items():
                if key in volatile:
                    continue
                if asdict(after)[key] != value:
                    failures.append(
                        f"{before.mode}: parallel arms changed {key} "
                        f"({value!r} -> {asdict(after)[key]!r})"
                    )

        assert isinstance(reference, V0SessionResult)

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("v0 session self-check: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
