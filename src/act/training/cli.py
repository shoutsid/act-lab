"""Command-line entry points for baseline and declared V0 protocol runs.

Modes: ``baseline`` (dense reference), ``v0`` (raw declared sessions), and
``pair-session`` (the section 5.2 per-seed pair loop: reads the throughput
benchmark artifact, applies the pre-committed dtype rule, and runs
``run_pair_session`` over the seed queue with the dataset handoff).
"""

import argparse
import json
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import torch

from act.model import DenseBaselineConfig, DenseCausalLM
from act.training.runtime import (
    TrainingConfig,
    TrainingState,
    make_optimizer,
    resolve_device,
    save_checkpoint,
    train_steps,
)
from act.training.session_plan import MAX_CHECKPOINTS, decide_dtype, window_left
from act.training.v0_protocol import aggregate_arm_cost_ratio, harm_guard_rejects, sign_pairs
from act.training.v0_runner import DECLARED_SEEDS, run_pair_session
from act.training.v0_session import (
    V0SessionConfig,
    declared_session_config,
    run_v0_session,
)


def _run_dense_baseline(args: argparse.Namespace) -> None:
    torch.manual_seed(19)
    model = DenseCausalLM(DenseBaselineConfig())
    config = TrainingConfig(device=args.device)
    state = TrainingState()
    optimizer = make_optimizer(model, config)
    losses = train_steps(model, optimizer, state, config, args.steps)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.checkpoint, model, optimizer, state)
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "parameters": model.parameter_count,
                "step": state.step,
                "tokens": state.tokens,
                "loss": losses[-1],
                "device": str(next(model.parameters()).device),
            },
            sort_keys=True,
        )
    )


def _run_v0(args: argparse.Namespace) -> None:
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    config = V0SessionConfig(
        arms=("random", "adaptive"),
        probe_batch_count=args.probe_batch_count,
        device=args.device,
        total_steps=args.steps,
    )
    results = []
    metrics = []
    for seed in seeds:
        seed_config = V0SessionConfig(
            arms=("random", "adaptive"),
            seed=seed,
            probe_batch_count=args.probe_batch_count,
            probe_candidate_policy=args.probe_candidate_policy,
            probe_selection=args.probe_selection,
            device=args.device,
            total_steps=args.steps,
        )
        result = run_v0_session(seed_config, checkpoint_dir=args.checkpoint_dir)
        results.append(result)
        arms = {arm.mode: arm for arm in result.arms}
        metrics.append(
            {
                "seed": seed,
                "adaptive_loss": arms["adaptive"].final_loss,
                "random_loss": arms["random"].final_loss,
                "search_flops_by_arm": {
                    arm.mode: arm.search_flops for arm in result.arms
                },
                "training_flops_by_arm": {
                    arm.mode: arm.training_flops for arm in result.arms
                },
                "wall_time_seconds": {
                    arm.mode: arm.wall_time_seconds for arm in result.arms
                },
                "tokens_per_second": {
                    arm.mode: arm.tokens_per_second for arm in result.arms
                },
            }
        )
    cost_ratio = aggregate_arm_cost_ratio(results)
    pairs = sign_pairs(results)
    summary = {
        "protocol": "v0-compute-efficiency-5.1",
        "probe_batch_count": args.probe_batch_count,
        "seeds": seeds,
        "predicted_cost_ratio": results[0].predicted_cost_ratio,
        "measured_cost_ratio": cost_ratio,
        "measured_cost_percent": 100.0 * cost_ratio,
        "adaptive_wins": sum(t <= r for t, r in pairs),
        "harm_guard_rejects": harm_guard_rejects(pairs),
        "per_seed": metrics,
    }
    output = args.output or (Path(args.checkpoint_dir) / "summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_seed"}, sort_keys=True))


#: The revision marker the published code dataset carries beside the package tree
#: (``kaggle_queue.REVISION_MARKER``, written from the same HEAD whose bytes it archived).
REVISION_MARKER = "revision.txt"


def _revision_from_marker(root: Path) -> str | None:
    """The commit recorded in ``revision.txt`` at ``root``, or ``None``.

    A Kaggle session imports the package from a *mounted dataset*, which has no ``.git`` -
    so ``git rev-parse`` answers nothing there and this marker is the only statement of
    which revision is running.
    """
    try:
        text = (Path(root) / REVISION_MARKER).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def _git_revision(root: Path | None = None) -> str:
    """The revision this code came from, for the handoff manifests.

    ``git rev-parse --short HEAD`` answers on a workstation. On Kaggle the code arrives as
    a mounted dataset with no ``.git``, so the fallback is the dataset's own
    ``revision.txt`` beside the package tree - the commit whose bytes were archived, which
    is exactly what the field means. Measured 2026-09-27: without it every published
    manifest of the whole campaign read ``code_revision: unknown``, so a reader could not
    attribute a seed to the code that trained it - and could not see that the primary
    block's first three seeds (15024-15026, retrained by the 403-era sessions that restored
    nothing and published over them) came from a different revision than its last three.

    ``root`` is for tests and for a caller that knows where its tree is; ``None`` means
    "ask git about the checkout this module is installed in, then read the marker beside
    it". Full commit id from the marker, git's short form locally: either resolves.
    """
    if root is None:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            root = Path(__file__).resolve().parents[3]
    return _revision_from_marker(root) or "unknown"


def _resolve_dtype(
    benchmark: Path | None, tokens_per_second: float | None
) -> tuple[float, str, float]:
    """Rule rate, dtype decision, and the rate that parameterizes pair time.

    The section 5.2 dtype rule is pre-committed, so the CLI re-derives the
    decision from the measured *fp32* throughput and refuses an artifact whose
    recorded ``dtype_decision`` disagrees with its own measured number (a
    stale or hand-edited artifact must not launch a session).

    ``measured_tokens_per_second`` is that fp32 number, which is the rule's
    input - but the schedule is arithmetic about the session that will run, and
    the session trains the decided dtype. The benchmark records that rate as
    ``schedule_tokens_per_second``; artifacts written before the field existed
    fall back to the rule rate, which is what they were scheduled from.
    """
    if benchmark is not None:
        if not benchmark.is_file():
            raise SystemExit(f"benchmark artifact not found: {benchmark}")
        measured = json.loads(benchmark.read_text())
        if measured.get("benchmark") != "v0-throughput-5.2":
            raise SystemExit(
                f"{benchmark} is not a v0-throughput-5.2 benchmark artifact"
            )
        tokens_per_second = float(measured["measured_tokens_per_second"])
        recorded = measured.get("dtype_decision")
        derived = decide_dtype(tokens_per_second)
        if recorded != derived:
            raise SystemExit(
                f"{benchmark}: recorded dtype_decision {recorded!r} disagrees with "
                f"the measured throughput ({derived}); artifact is stale or edited"
            )
        schedule_rate = float(measured.get("schedule_tokens_per_second", tokens_per_second))
        if schedule_rate <= 0:
            raise SystemExit(
                f"{benchmark}: schedule_tokens_per_second must be positive, "
                f"got {schedule_rate}"
            )
        return tokens_per_second, derived, schedule_rate
    if tokens_per_second is None:
        raise SystemExit(
            "provide --benchmark (the session-1 artifact) or --tokens-per-second"
        )
    if tokens_per_second <= 0:
        raise SystemExit("--tokens-per-second must be positive")
    return tokens_per_second, decide_dtype(tokens_per_second), tokens_per_second


def _run_pair_session(args: argparse.Namespace) -> None:
    rule_rate, dtype, schedule_rate = _resolve_dtype(
        args.benchmark, args.tokens_per_second
    )
    if args.dtype is not None and args.dtype != dtype:
        raise SystemExit(
            f"--dtype {args.dtype} contradicts the measured dtype decision ({dtype}); "
            "the section 5.2 dtype rule is pre-committed - rerun the benchmark or "
            "drop the flag"
        )
    if args.hours <= 0:
        raise SystemExit("--hours must be positive")
    try:
        # Fail before any training: an unusable device (``cuda:1`` on a
        # one-GPU host) should not surface from inside a training step.
        resolve_device(args.device)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(f"--device {args.device!r}: {error}") from error
    if args.seeds is None:
        seeds = DECLARED_SEEDS
    elif "-" in args.seeds and "," not in args.seeds:
        # The runbook's documented 'A-B' range form (the section 5.2 command
        # passes the declared block); also accepted by the verdict script.
        start, end = args.seeds.split("-", 1)
        if not (start.strip().isdigit() and end.strip().isdigit()):
            raise SystemExit(
                f"--seeds range must be 'A-B' with integers, got {args.seeds!r}"
            )
        seeds = tuple(range(int(start), int(end) + 1))
    else:
        try:
            seeds = tuple(
                int(value) for value in args.seeds.split(",") if value.strip()
            )
        except ValueError as error:
            raise SystemExit(f"--seeds must be a comma list or 'A-B' range: {error}")
    if not seeds:
        raise SystemExit("--seeds resolved to an empty queue")
    arms = tuple(
        arm.strip() for arm in args.session_arms.split(",") if arm.strip()
    )
    if not arms:
        raise SystemExit("--session-arms resolved to an empty tuple")
    if args.new_expert == "clone" and not args.preserve_router_on_rebuild:
        # V0SessionConfig rejects this combination too; catching it here turns an
        # operator typo into a clean one-line refusal before any training starts,
        # the way the --dtype contradiction is handled.
        raise SystemExit(
            "--new-expert clone copies rows from the surviving routers and "
            "requires --preserve-router-on-rebuild (the section 5.24 treatment); "
            "pass both or neither"
        )
    # The launch command is the DECLARED protocol run unless --profile asks for
    # the tiny smoke shape, so it must name the declared target explicitly:
    # V0SessionConfig's defaults are the fast dev shape (dev cycle time, not
    # protocol fidelity), and inheriting them here would train a 14.9M model
    # while labelling the run declared.
    config = declared_session_config(
        arms=arms,
        probe_batch_count=args.probe_batch_count,
        probe_candidate_policy=args.probe_candidate_policy,
        probe_selection=args.probe_selection,
        preserve_router_on_rebuild=args.preserve_router_on_rebuild,
        new_expert_initialization=args.new_expert,
        device=args.device,
        dtype=dtype,
        max_checkpoints=MAX_CHECKPOINTS,  # the registered section 5.2 pin (2)
    )
    if args.profile:
        # Tiny smoke config so the full launch path is checkable locally;
        # the real command runs the declared 285M config without overrides.
        config = V0SessionConfig(
            declared=False,
            arms=arms,
            device=args.device,
            dtype=dtype,
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
            probe_batch_count=args.probe_batch_count,
            probe_candidate_policy=args.probe_candidate_policy,
            probe_selection=args.probe_selection,
            preserve_router_on_rebuild=args.preserve_router_on_rebuild,
            new_expert_initialization=args.new_expert,
            validation_batches=2,
            checkpoint_interval=4,
            max_checkpoints=2,
        )
    # The clock the runner is given has to be what is *left* of the declared
    # window, and the restore below is the largest part of what has been spent:
    # it downloads the whole published dataset before a token is trained, and it
    # grows with every seed the queue publishes. Passing the declared window
    # unchanged told the runner it had the full window at entry, so a session
    # that spent 40 minutes restoring then scheduled pairs against time it did
    # not have and stopped that much late - which is exactly what
    # ``run_pair_session``'s own contract excludes ("the session clock at entry"
    # is the window minus boot and restore). The boot-and-benchmark half of that
    # subtraction is done by the launcher before it starts this process.
    entry_started = time.monotonic()
    if args.kaggle_dataset is not None:
        # Session start: restore the newest remote version before training
        # (fail-fast - a session that cannot get its requested remote state
        # must stop here, not mid-pair). Skipped seeds with a complete local
        # version stay skipped; remote seeds merge over matching locals.
        from act.ops.kaggle_remote import (
            default_client,
            restore_latest,
            shard_datasets,
        )

        # Membership is a function of the *registered* queue, not of what this
        # command asked for: a session that passes a subset must still push and
        # restore the shard the declared block puts those seeds in, or the same
        # seed would live in two datasets depending on the invocation.
        wanted_shards = [
            name
            for name, members in shard_datasets(
                args.kaggle_dataset, DECLARED_SEEDS, args.kaggle_shard_seeds
            )
            if set(members) & set(seeds)
        ]
        if not wanted_shards:
            # A development queue whose seeds are not in the declared block has
            # no shard to resolve to; it keeps the single-dataset restore rather
            # than silently restoring nothing.
            wanted_shards = [args.kaggle_dataset]
        try:
            restore_latest(
                args.dataset_root,
                datasets=wanted_shards,
                client=default_client(),
                log=print,
            )
        except Exception as error:  # noqa: BLE001 - launch-blocking, but clean
            raise SystemExit(
                f"cannot restore remote dataset {args.kaggle_dataset!r}: "
                f"{type(error).__name__}: {error}; session not started "
                "(no training ran, nothing published)"
            ) from error
    entry_minutes = (time.monotonic() - entry_started) / 60.0
    if entry_minutes / 60.0 >= args.hours:
        raise SystemExit(
            f"the declared {args.hours:g} h window was spent before training: "
            f"startup took {entry_minutes:.1f} min, so this session has nothing "
            "left to run and nothing was trained or published"
        )
    remaining_hours = window_left(args.hours, entry_minutes / 60.0)
    if args.kaggle_dataset is not None:
        # Stated rather than implied: this is the number the deadline rules act
        # on, and it is no longer the number the window was declared with.
        print(
            f"session clock: {remaining_hours:.2f} h of the declared "
            f"{args.hours:g} h remain after {entry_minutes:.1f} min of startup",
            flush=True,
        )
    # The revision is written into every seed's manifest, and it belongs in the session's
    # own account too: a kernel log that does not say which code ran cannot be attributed
    # afterwards, which is what the campaign's published manifests said (`unknown`) for
    # every seed until the marker fallback above.
    code_revision = _git_revision()
    print(f"code revision: {code_revision}", flush=True)
    # Each pair is mirrored into the versioned dataset as it finishes, so the
    # session's progress is durable before its exit publish - which is the step
    # four sessions in a row were killed inside (see ``_remote_publisher``).
    publish_each_pair = _remote_publisher(args, tokens_per_second=schedule_rate)
    summary = run_pair_session(
        seeds=seeds,
        config=config,
        session_dir=args.session_dir,
        dataset_root=args.dataset_root,
        code_revision=code_revision,
        remaining_hours=remaining_hours,
        measured_tokens_per_second=schedule_rate,
        output=args.output,
        log=print,
        remote_publish=publish_each_pair,
        # Both numbers belong on the trail: the fp32 rate is the pre-committed
        # dtype rule's input, the schedule rate is what set the pair time. They
        # differ by design whenever the session trains fp16, so a log that
        # carried only one of them could not be audited later.
        summary_extra={
            "dtype": dtype,
            "dtype_rule_tokens_per_second": rule_rate,
            "schedule_tokens_per_second": schedule_rate,
        },
    )
    # Only seeds this session trained are pushed at exit. A seed the restore found
    # complete is already in its shard's published version; pushing that shard again
    # re-uploads ~44 GiB and was the exit push that killed 2026-10-19.
    trained = tuple(
        int(entry["seed"])
        for entry in summary.get("pairs", [])
        if entry.get("action") in ("completed", "publish_failed")
    )
    _publish_remote(args, tokens_per_second=schedule_rate, seeds=trained)
    printed = {
        key: summary[key]
        for key in (
            "protocol",
            "seeds_requested",
            "declared_seed_queue",
            "remaining_hours_at_entry",
            "measured_tokens_per_second",
            "pair_time_hours",
            "pairs_completed",
            "latest_published_seed",
        )
    }
    # Stated from the CLI's own inputs rather than read back out of the runner's
    # summary: the durable session log carries them (via ``summary_extra``), but
    # what the CLI reports must not depend on that write succeeding.
    printed["dtype"] = dtype
    printed["dtype_rule_tokens_per_second"] = rule_rate
    printed["schedule_tokens_per_second"] = schedule_rate
    print(
        json.dumps(
            printed,
            sort_keys=True,
        )
    )


def _remote_publisher(
    args: argparse.Namespace, *, tokens_per_second: float
) -> Callable[[int], Mapping[str, object]] | None:
    """The per-pair half of the remote handoff (``--kaggle-dataset`` only).

    ``run_pair_session`` calls the returned hook after every pair it completes,
    and the hook pushes the whole handoff folder - so each pair is durable
    inside the session rather than only at the end of it. Measured 2026-10-05,
    that end-of-session push is where four sessions in a row were killed: every
    live log stops at ``session done`` with nothing from the publish path after
    it, and each session lost the three pairs it had just trained. Publishing
    per pair bounds the loss to the pair in flight, and what it pushes is the
    same dataset the next session restores - so the seeds it carries are
    skipped there instead of being trained a second time.

    The client is built on the first push rather than here: this runs before
    the session has trained anything, and an unavailable ``kaggle`` package
    must fail that first push loudly (inside the runner, where it is recorded)
    rather than end the session before its window is used.
    """
    if args.kaggle_dataset is None:
        return None
    from act.ops.kaggle_remote import (
        KaggleDatasetClient,
        default_client,
        publish_handoff,
        shard_for_seed,
    )

    client: KaggleDatasetClient | None = None

    def publish_after_pair(seed: int) -> Mapping[str, object]:
        nonlocal client
        if client is None:
            client = default_client()
        # One shard per push, resolved from the registered queue: a push that
        # carried every seed would carry the whole block's payload, which is
        # the size that gets the container killed (2026-10-05/06).
        shard_seeds = getattr(args, "kaggle_shard_seeds", 0)
        sharded = shard_seeds > 0 and seed in DECLARED_SEEDS
        dataset, members = (
            shard_for_seed(args.kaggle_dataset, seed, DECLARED_SEEDS, shard_seeds)
            if sharded
            else (args.kaggle_dataset, None)
        )
        result = publish_handoff(
            args.dataset_root,
            dataset=dataset,
            client=client,
            version_notes=(
                f"v0-5-1 session publish: {tokens_per_second:.0f} tokens/s, "
                f"{args.dtype or 'measured'} dtype, after seed {seed}"
            ),
            log=print,
            seeds=members,
        )
        # The same JSON line the exit publish prints, so the session's stream
        # carries one account of every push it made, not only the last.
        print(json.dumps({"remote_publish": result.to_dict()}, sort_keys=True))
        return result.to_dict()

    return publish_after_pair


def _publish_remote(
    args: argparse.Namespace, *, tokens_per_second: float, seeds: tuple[int, ...]
) -> None:
    """Optional exit-path publish to a versioned Kaggle dataset (--kaggle-dataset).

    The remote push is the last step of the launch command, after the local
    handoff has run inside ``run_pair_session``. It never raises: a failed
    remote push must not turn a completed session into a crashed one. It does
    retry before giving up (``publish_on_exit``), because in a kernel this is
    the only durable copy - the handoff roots live in the container overlay,
    so a later session cannot re-push *this* session's pairs, only start again
    from the last published version. The outcome is recorded in the publish log
    beside the dataset root, and it is the entry the driver reads.

    Since every completed pair publishes itself (``_remote_publisher``), this
    call is normally already satisfied: the handoff is unchanged since the last
    pair's push, so ``publish_handoff`` reports ``already_published`` and
    uploads nothing. That is deliberate - the exit upload is where four
    sessions in a row were killed, and a session with nothing left to upload
    has nothing left to lose there.
    """
    if args.kaggle_dataset is None:
        return
    if not seeds:
        # Nothing this session trained is unpublished; the pair-time pushes did the
        # work. Pushing with no seeds would fall through to the whole folder.
        print("remote publish skipped at exit: no seed trained this session", flush=True)
        return
    from act.ops.kaggle_remote import (
        STATUS_FAILED,
        default_client,
        publish_on_exit,
        shard_datasets,
    )

    client = default_client()
    shard_seeds = getattr(args, "kaggle_shard_seeds", 0)
    # One push per shard, normally ``already_published`` for every one of them
    # because the pair-time pushes already did the work. A failed shard stops
    # the loop so the *last* entry the driver reads is that failure - the exit
    # path is the one place a broken push must be visible rather than retried
    # into a later shard's success. A run whose seeds are not in the declared
    # block (a development queue) has no shard to resolve to, so it keeps the
    # single-version behaviour rather than publishing nothing.
    shards = [
        (name, members)
        for name, members in shard_datasets(
            args.kaggle_dataset, DECLARED_SEEDS, shard_seeds
        )
        if set(members) & set(seeds)
    ]
    if not shards:
        shards = [(args.kaggle_dataset, tuple(seeds))]
        shard_seeds = 0
    for dataset, members in shards:
        result = publish_on_exit(
            args.dataset_root,
            dataset=dataset,
            client=client,
            version_notes=(
                f"v0-5-1 session publish: {tokens_per_second:.0f} tokens/s, "
                f"{args.dtype or 'measured'} dtype"
            ),
            log=print,
            seeds=members if shard_seeds > 0 else None,
        )
        print(
            json.dumps(
                {"remote_publish": result.to_dict()},
                sort_keys=True,
            )
        )
        if result.status == STATUS_FAILED:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("baseline", "v0", "pair-session"), default="baseline"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="training horizon (baseline default 8; v0 default 6000 = declared)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="auto | cpu | cuda | cuda:N - a specific index is how the pair "
        "model addresses the second GPU (the declared run is cuda)",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/dense-baseline.pt"))
    # V0 protocol options.
    parser.add_argument(
        "--seeds", default=None,
        help="comma-separated seeds or an inclusive 'A-B' range (pair-session "
        "defaults to the declared queue, v0_runner.DECLARED_SEEDS)",
    )
    parser.add_argument("--probe-batch-count", type=int, default=1)
    # Adopted section 5.5 protocol knobs. Defaults are the registered ones, so
    # an unflagged launch reproduces the original registration; the adopted
    # primary run passes --probe-candidate-policy spread, and the pool-matched
    # control run passes --probe-selection random (same seeds, same cost).
    parser.add_argument(
        "--probe-candidate-policy", choices=("prefix", "spread"), default="prefix",
        help="how a cap-bounded probe picks candidates: 'prefix' (registered "
        "default, confined to the earliest layer(s)) or 'spread' (the adopted "
        "section 5.5 primary, a deterministic subset of the whole feasible set)",
    )
    parser.add_argument(
        "--probe-selection",
        choices=("argmin", "argmax", "random", "quietest"),
        default="argmin",
        help="decision rule over probed candidates: 'argmin' (registered "
        "default), 'argmax' (inverted control), 'random' (the adopted "
        "pool-matched loss-blind control arm) or 'quietest' (prune the "
        "least-loaded expert in the probed pool, otherwise the probe's pick)",
    )
    # The section 5.28 rebuild treatment. Section 5.16 says new work should set
    # preserve_router_on_rebuild=True and section 5.38.3 says a real-scale
    # protocol should be specified with it on; before this flag existed the
    # declared launch command had no way to express that, so it always ran the
    # historical rebuild. Defaults here are the historical behaviour, so an
    # unflagged launch is still exactly what every committed run used.
    parser.add_argument(
        "--preserve-router-on-rebuild", action="store_true", default=False,
        help="keep the router's surviving rows across a mutation rebuild "
        "instead of re-drawing them (section 5.28's treatment; default off = "
        "the historical rebuild, which is what the committed runs used)",
    )
    parser.add_argument(
        "--new-expert", choices=("random", "clone"), default="random",
        help="how a grown expert's rows are initialised: 'random' (historical "
        "default) or 'clone' (copy the nearest existing expert; requires "
        "--preserve-router-on-rebuild, the section 5.24 treatment)",
    )
    parser.add_argument("--checkpoint-dir", default="artifacts/v0-sessions")
    parser.add_argument("--output", type=Path, default=None)
    # Pair-session options (section 5.2 launch command).
    parser.add_argument(
        "--benchmark", type=Path, default=None,
        help="session-1 benchmark artifact (default artifacts/v0-benchmark.json "
        "when neither --benchmark nor --tokens-per-second is given)",
    )
    parser.add_argument(
        "--tokens-per-second", type=float, default=None,
        help="override the measured throughput (skips the benchmark artifact)",
    )
    parser.add_argument(
        "--dtype", choices=("fp32", "fp16"), default=None,
        help="must equal the measured dtype decision; pre-committed rule otherwise",
    )
    parser.add_argument(
        "--hours", type=float, default=12.0,
        help="remaining session clock at entry in hours (12-hour Kaggle limit "
        "minus boot and restore)",
    )
    parser.add_argument(
        "--session-arms", default="random,adaptive",
        help="comma-separated session arms for pair-session mode: 'random,adaptive' "
        "(the registered section 5.1 pair), 'null' (the pre-registered section "
        "5.14 do-nothing queue), 'gated' or 'heuristic' (single-arm queues)",
    )
    parser.add_argument(
        "--session-dir", default="artifacts/v0-sessions",
        help="working-directory checkpoint root",
    )
    parser.add_argument(
        "--dataset-root", default="artifacts/datasets",
        help="root of the versioned checkpoint dataset (the Kaggle dataset dir)",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="run a tiny smoke config instead of the declared 285M model",
    )
    parser.add_argument(
        "--kaggle-shard-seeds",
        type=int,
        default=0,
        help=(
            "split the handoff across dataset versions of this many declared "
            "seeds (0 = one version). Measured 2026-10-05/06: a 7-seed / "
            "44.13 GiB version lands and an 8-seed / ~50.4 GiB one does not, "
            "so the registered block needs two; shard 0 keeps the registered "
            "dataset id and later shards are '<id>-p2', '<id>-p3', ..."
        ),
    )
    parser.add_argument(
        "--kaggle-dataset", default=None,
        help="owner/slug of the versioned Kaggle checkpoint dataset: restore "
        "its latest version before the session and publish on exit "
        "(requires the kaggle package)",
    )
    args = parser.parse_args()
    if args.mode == "v0":
        if args.steps is None:
            args.steps = 6_000
        if args.seeds is None:
            args.seeds = "15000"
        _run_v0(args)
    elif args.mode == "pair-session":
        if args.output is None:
            args.output = Path(args.session_dir) / "session-log.json"
        if args.benchmark is None and args.tokens_per_second is None:
            args.benchmark = Path("artifacts/v0-benchmark.json")
        _run_pair_session(args)
    else:
        if args.steps is None:
            args.steps = 8
        _run_dense_baseline(args)


if __name__ == "__main__":
    main()
