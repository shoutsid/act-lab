"""Versioned-dataset checkpoint handoff for the section 5.2 session plan.

The Kaggle working directory does not survive a session boundary, so every
completed pair (or deadline stop) must publish its resumable state to the
versioned dataset and every new session must restore from it. This module
implements the pre-committed handoff on top of the existing manifest
contract (`act.ops.publish_artifacts` / `restore_artifacts`):

- **Layout**: `act-checkpoints/v0-5-1/seed-NNNN/` per seed, one directory per
  arm holding its latest event checkpoint, the arm metrics JSON, and the seed
  summary JSON.
- **All-or-nothing publish**: everything is staged under a temporary
  directory, the manifest is checksummed and written last, and only then is
  the seed directory swapped into place - a partial publish is never
  mistaken for a complete one (section 5.2's rule, per plan section 38's
  "a restarted session is equivalent to an uninterrupted run").
- **Restore**: checksums are verified before anything is handed back, and the
  seed directory is re-materialized into a session checkpoint root so
  `run_v0_session` resumes exactly where the publish left off.

Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

from act.ops import kaggle_layout
from act.training.v0_session import V0SessionResult

HANDOFF_DATASET = "act-checkpoints/v0-5-1"
SUMMARY_NAME = "seed-summary.json"
# Status values in the published summary. "complete" = the pair finished its
# declared horizon (resumable state AND result state are valid). "in-progress"
# = a deadline stop's safety snapshot; the next session publishes it only as
# a placeholder and deliberately restarts the pair from scratch (section 5.2:
# pairs are never resumed across sessions).
COMPLETE_STATUS = "complete"
IN_PROGRESS_STATUS = "in-progress"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_directory(dataset_root: str | Path, seed: int) -> Path:
    """Versioned-dataset location for one seed's handoff state."""
    return Path(dataset_root) / HANDOFF_DATASET / f"seed-{seed:04d}"


def session_checkpoint_root(session_dir: str | Path, seed: int) -> Path:
    """Working-directory checkpoint root `run_v0_session` resumes from."""
    return Path(session_dir) / f"seed-{seed}"


def newest_arm_checkpoint(arm_dir: str | Path) -> Path:
    """Newest checkpoint in an arm directory (step-ordered by filename)."""
    candidates: list[tuple[int, Path]] = []
    for path in Path(arm_dir).glob("checkpoint-*.pt"):
        stem = path.stem.split("-")[-1]
        if stem.isdigit():
            candidates.append((int(stem), path))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found in {arm_dir}")
    return max(candidates, key=lambda item: item[0])[1]


_arm_checkpoint = newest_arm_checkpoint  # backward-compatible alias


def publish_seed_state(
    dataset_root: str | Path,
    *,
    seed: int,
    session_dir: str | Path,
    checkpoint_paths: dict[str, str | Path],
    arm_payloads: dict[str, dict[str, object]],
    config: dict[str, object],
    predicted_cost_ratio: float,
    code_revision: str,
    status: str = COMPLETE_STATUS,
    interrupted_step: int | None = None,
    clean_working_dir: bool = False,
) -> Path:
    """Publish one seed's resumable state as an atomic dataset version.

    This is the shared staging-and-swap core: the caller supplies the newest
    per-arm checkpoint paths and metric payloads directly. ``status`` records
    whether the pair completed its horizon (``complete``) or the publish is a
    deadline-stop safety snapshot (``in-progress``). Staged under a temporary
    directory, manifest checksummed and written last, swapped into place only
    then; with ``clean_working_dir`` the working copy is removed afterwards.
    """
    if status not in {COMPLETE_STATUS, IN_PROGRESS_STATUS}:
        raise ValueError(f"unknown publish status: {status}")
    if not arm_payloads:
        raise ValueError("no arm payloads to publish")
    if set(arm_payloads) != set(checkpoint_paths):
        raise ValueError("arm payloads and checkpoint paths must cover the same arms")
    destination = seed_directory(dataset_root, seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".staging-seed-{seed:04d}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    layout = kaggle_layout(staging)
    for directory in layout.required_directories():
        directory.mkdir(parents=True, exist_ok=True)

    sources: list[tuple[Path, Path]] = []
    for mode in sorted(arm_payloads):
        checkpoint = Path(checkpoint_paths[mode])
        # Keep the original step-named filename: the session runner's resume
        # lookup globs ``checkpoint-*.pt``, so the name is part of the
        # contract that a restored seed actually resumes.
        target = layout.checkpoints / mode / checkpoint.name
        sources.append((checkpoint, target))
        arm_payloads[mode] = {
            **arm_payloads[mode],
            "checkpoint": str(target.relative_to(layout.root)),
        }

    summary = {
        "seed": seed,
        "status": status,
        "protocol": "v0-compute-efficiency-5.1",
        "interrupted_step": interrupted_step,
        "predicted_cost_ratio": predicted_cost_ratio,
        "arms": arm_payloads,
        "config": config,
    }
    summary_path = staging / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    sources.append((summary_path, layout.reports / SUMMARY_NAME))

    copied: list[str] = []
    checksums: dict[str, str] = {}
    for source, target in sources:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        relative = str(target.relative_to(layout.root))
        copied.append(relative)
        checksums[relative] = _sha256(target)
    manifest = {
        "dataset_id": HANDOFF_DATASET,
        "code_revision": code_revision,
        "seed": seed,
        "artifacts": sorted(copied),
        "checksums": checksums,
    }
    # The manifest is written last, after all payloads and checksums.
    (layout.manifests / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    # Atomic swap: replace any prior version of this seed only after the
    # complete staged copy (manifest included) exists.
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(staging, destination)

    if clean_working_dir and Path(session_dir).exists():
        shutil.rmtree(Path(session_dir) / f"seed-{seed}")
    return destination


def publish_seed(
    dataset_root: str | Path,
    result: V0SessionResult,
    *,
    session_dir: str | Path,
    code_revision: str,
    clean_working_dir: bool = False,
) -> Path:
    """Publish one completed seed's result state (status ``complete``)."""
    seed = int(result.config["seed"])
    if not result.arms:
        raise ValueError("session result has no arms to publish")
    if any(arm.mutation_count != result.arms[0].mutation_count for arm in result.arms):
        raise ValueError("refusing to publish a seed with unmatched mutation budgets")
    seed_root = Path(session_dir) / f"seed-{seed}"
    return publish_seed_state(
        dataset_root,
        seed=seed,
        session_dir=session_dir,
        checkpoint_paths={
            arm.mode: _arm_checkpoint(seed_root / f"arm-{arm.mode}")
            for arm in result.arms
        },
        arm_payloads={arm.mode: dict(asdict(arm)) for arm in result.arms},
        config=dict(result.config),
        predicted_cost_ratio=result.predicted_cost_ratio,
        code_revision=code_revision,
        status=COMPLETE_STATUS,
        clean_working_dir=clean_working_dir,
    )


def latest_published_seed(dataset_root: str | Path) -> tuple[int, Path] | None:
    """Newest complete seed version in the dataset, or None if none exist."""
    root = Path(dataset_root) / HANDOFF_DATASET
    if not root.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for path in root.glob("seed-*"):
        if not (path / "manifests" / "manifest.json").is_file():
            continue  # an incomplete publish is not a version
        stem = path.stem.split("-")[-1]
        if not stem.isdigit():
            continue
        seed = int(stem)
        if best is None or seed > best[0]:
            best = (seed, path)
    return best


def restore_seed(
    dataset_root: str | Path,
    seed: int,
    *,
    session_dir: str | Path,
) -> Path:
    """Verify a seed's dataset version and re-materialize its resumable state.

    Checksums are verified for every payload before any file is placed into
    the working session directory; on success `run_v0_session` resumes the
    seed from the restored checkpoints. Returns the seed's session directory.
    """
    source = seed_directory(dataset_root, seed)
    manifest_path = source / "manifests" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no published version for seed {seed} under {source}")
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("seed", -1)) != seed:
        raise ValueError(f"manifest seed {manifest.get('seed')} does not match {seed}")

    # Verify every checksum first; nothing moves until the version is sound.
    for relative, expected in manifest.get("checksums", {}).items():
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(f"published payload missing: {path}")
        if _sha256(path) != expected:
            raise ValueError(f"checksum mismatch for {relative}")

    target = session_checkpoint_root(session_dir, seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for relative in manifest.get("artifacts", []):
        published = source / relative
        if not relative.startswith("checkpoints/"):
            continue
        # Layout: checkpoints/<arm-mode>/<original checkpoint filename>.
        arm_mode = Path(relative).parts[1]
        arm_dir = target / f"arm-{arm_mode}"
        arm_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(published, arm_dir / Path(relative).name)
    return target


def load_seed_summary(dataset_root: str | Path, seed: int) -> dict:
    """Read (without verification) a published seed's summary for reporting."""
    path = seed_directory(dataset_root, seed) / "reports" / SUMMARY_NAME
    return json.loads(path.read_text())


def published_seeds(dataset_root: str | Path) -> list[int]:
    """Every seed version present under the dataset root that carries a summary, ascending.

    The block's membership is the set of versions *present*, not the set of seeds one
    invocation was asked for: a session launched with ``--seeds 15004-15011`` appends to
    the same block as one launched with the full declared queue, so a reader that looked
    only at its own queue would miss exactly the seeds that disagree with it.
    """
    root = Path(dataset_root) / HANDOFF_DATASET
    if not root.is_dir():
        return []
    seeds = []
    for path in root.glob("seed-*"):
        stem = path.stem.split("-")[-1]
        if stem.isdigit() and (path / "reports" / SUMMARY_NAME).is_file():
            seeds.append(int(stem))
    return sorted(seeds)


def published_seed_status(dataset_root: str | Path, seed: int) -> str | None:
    """Return a seed's publish status, or None when it has no version."""
    path = seed_directory(dataset_root, seed) / "reports" / SUMMARY_NAME
    if not path.is_file():
        return None
    summary = json.loads(path.read_text())
    status = summary.get("status")
    return status if isinstance(status, str) else None
