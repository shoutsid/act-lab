"""Re-run the published factor map on the decoupled arm-RNG path.

Every study behind the report's factor map (section 3: nine gate studies plus
the guided upper bound) ran with the arms sharing the global torch RNG, so a
``grow`` (which draws fresh expert weights) and a ``prune`` (which draws
nothing) consume different numbers of global draws. Two runs whose treatments
choose differently therefore perturb each other's *random* arm, and the
"matched random churn" baseline drifts with the treatment (report section 9.8).

This script re-runs each study with the published config, the published seed
block and **only** ``independent_arm_rng=True`` changed, and reports whether the
published verdict label survives. It is a reproduction check, not a new
hypothesis: the comparison target is the committed study summary.

Declared decision rule (fixed before the run, in RESEARCH_REPORT.md section
9.10): a study **reproduces** when its verdict label at alpha=0.05 is unchanged
(``gate_passed`` recomputed on the same n), and **flips** otherwise. A flip is a
revision to the published claim, not a new result to be preferred. Win counts
and mean deltas are reported either way, so "reproduces" cannot hide a large
effect-size move.

Reconstruction: ``gate3-legacy-signal-384`` records a config that predates the
``adaptive_policy`` field, so ``ExperimentConfig(**recorded)`` would default to
``probe_loss`` and silently reproduce a *different* experiment. The value is
recoverable rather than unknowable, so the study is re-run with it supplied
explicitly (see ``RECONSTRUCTION``), which restores the missing row to the
corrected factor map.

Device note: the published summaries record ``device: auto``. The recheck pins
``cpu`` because 16 shards contending for one GPU is far slower for this
hidden-24 model (section 9.3) and section 9.7 established that this substrate
reproduces CUDA-produced results exactly on CPU (study 3's block came back
bit-for-bit: 213/384, mean -0.025955, p 0.01814).

Usage:

    PYTHONPATH=src python scripts/factor_map_recheck.py --list
    PYTHONPATH=src python scripts/factor_map_recheck.py \
        --study gate3-probe-loss-384 --shard 0 --shards 16 --out /tmp/recheck
    PYTHONPATH=src python scripts/factor_map_recheck.py --compare '/tmp/recheck/*.json'
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from act.evolution.experiment import ExperimentConfig, run_experiment  # noqa: E402
from act.evolution.figures_data import check_summary  # noqa: E402
from act.evolution.study import (  # noqa: E402
    _binomial_survival,
    _gate_min_wins,
    _population_stddev,
)

SUMMARIES = ROOT / "artifacts" / "study-summaries"
GATE_ALPHA = 0.05

# Fields an older record predates, supplied explicitly so the run reproduces
# the *recorded* experiment rather than today's defaults.
#
# ``gate3-legacy-signal-384`` predates ``adaptive_policy`` (commit d972d4c
# replaced the cheap-signal controller with bounded probe-loss selection and
# added the field). Its value is recoverable, not guessed:
#
#   * "signal" is the *untouched* pre-transition controller - that commit's
#     diff inserts the probe branch above the legacy ``random.Random(seed)``
#     chooser, leaving the fallback path byte-identical.
#   * The raw record ``artifacts/gate3-384.jsonl`` corroborates it: its per-arm
#     ``mutation_history`` entries carry no probe fields, so the probe
#     controller never ran in it.
#   * Run coupled, it reproduces the published per-seed record to float drift
#     (max |delta| 4.8e-7), whereas ``probe_loss`` does not (|delta| ~1e-1).
#     ``tests/test_factor_map_recheck.py`` pins both directions.
RECONSTRUCTION: dict[str, dict[str, object]] = {
    "gate3-legacy-signal-384": {"adaptive_policy": "signal"},
}


def published_summaries() -> dict[str, dict]:
    """Committed study summaries that carry a config and a seed block.

    The directory also holds this script's own comparison artifact (a list of
    per-study rows, not a study) and the analysis-only probe-validity record,
    so anything that is not a single study object is skipped rather than
    mistaken for one.
    """
    summaries: dict[str, dict] = {}
    for path in sorted(SUMMARIES.glob("*.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            continue
        if data.get("config") and data.get("seeds"):
            summaries[path.stem] = data
    return summaries


def recheck_set() -> dict[str, dict]:
    """Studies this script re-runs.

    Everything with a recorded config and seed block, including the legacy
    study, whose pre-``adaptive_policy`` config is completed from
    ``RECONSTRUCTION`` rather than defaulted.
    """
    return published_summaries()


def build_config(published: dict, *, study: str | None = None) -> ExperimentConfig:
    """Reconstruct the published run's config, decoupling the arms only."""
    config = dict(published["config"])
    if study is not None:
        # A record that predates a field needs its value restored before the
        # dataclass can fill it in with today's default.
        config.update(RECONSTRUCTION.get(study, {}))
    config["independent_arm_rng"] = True
    # Device is not part of the claim (see the module docstring); CPU keeps the
    # sharded recheck fast without changing the numbers on this substrate.
    config["device"] = "cpu"
    return ExperimentConfig(**config)


def run_shard(
    study: str,
    *,
    shard: int,
    shards: int,
    independent_arm_rng: bool = True,
) -> dict:
    """Run this shard's slice of one study and return a JSON-ready record."""
    if shards < 1:
        raise SystemExit("--shards must be at least 1")
    if not 0 <= shard < shards:
        raise SystemExit("--shard must be in [0, --shards)")
    available = recheck_set()
    if study not in available:
        raise SystemExit(f"{study} is not in the recheck set")
    published = available[study]
    config = build_config(published, study=study)
    config = replace(config, independent_arm_rng=independent_arm_rng)
    treatment = published.get("treatment", "adaptive")
    seeds = tuple(published["seeds"])[shard::shards]
    experiments: list[dict] = []
    for seed in seeds:
        result = run_experiment(replace(config, seed=seed))
        arms = {arm.mode: arm for arm in result.arms}
        if treatment not in arms or "random" not in arms:
            raise SystemExit(f"{study}: seed {seed} is missing the arm pair")
        # The full per-arm payload is kept (not just final losses) so decoupled
        # study summaries can be emitted in the committed schema and drive the
        # report figures - including the probe-cost identity, which needs the
        # per-seed FLOPs counters.
        experiments.append(
            {
                "config": {"seed": seed},
                "arms": [
                    {
                        "mode": arm.mode,
                        "final_loss": arm.final_loss,
                        "validation_loss": arm.validation_loss,
                        "quality": arm.quality,
                        "mutation_count": arm.mutation_count,
                        "parameter_count": arm.parameter_count,
                        "active_parameter_count": arm.active_parameter_count,
                        "training_flops": arm.training_flops,
                        "validation_flops": arm.validation_flops,
                        "search_flops": arm.search_flops,
                        "architecture_distance": arm.architecture_distance,
                        "architecture_hash": arm.architecture_hash,
                    }
                    for arm in (arms[treatment], arms["random"])
                ],
            }
        )
    return {
        "study": study,
        "shard": shard,
        "shards": shards,
        "independent_arm_rng": independent_arm_rng,
        "treatment": treatment,
        "experiments": experiments,
    }


def aggregate(records: list[dict]) -> dict[str, list[list[object]]]:
    """Merge shard records into one (seed, treatment, random) row list per study."""
    merged: dict[str, list[list[object]]] = {}
    for record in records:
        treatment = record["treatment"]
        for experiment in record["experiments"]:
            losses = {arm["mode"]: arm["final_loss"] for arm in experiment["arms"]}
            merged.setdefault(record["study"], []).append(
                [experiment["config"]["seed"], losses[treatment], losses["random"]]
            )
    return {study: sorted(rows, key=lambda row: row[0]) for study, rows in merged.items()}


def _merge_experiments(records: list[dict]) -> dict[str, list[dict]]:
    """Merge shard records into one ordered experiment list per study."""
    merged: dict[str, list[dict]] = {}
    for record in records:
        merged.setdefault(record["study"], []).extend(record["experiments"])
    return {
        study: sorted(experiments, key=lambda item: item["config"]["seed"])
        for study, experiments in merged.items()
    }


def summaries_from_records(
    records: list[dict],
    *,
    provenance: str = "decoupled-independent-arm-rng",
) -> dict[str, dict]:
    """Emit decoupled study summaries in the committed schema.

    The figures are driven by these, so each one is put through
    ``figures_data.check_summary`` before it is returned: a summary that does
    not recompute will never reach a figure.
    """
    summaries: dict[str, dict] = {}
    for study, experiments in _merge_experiments(records).items():
        treatment = next(
            record["treatment"] for record in records if record["study"] == study
        )
        seeds = [experiment["config"]["seed"] for experiment in experiments]
        deltas = [
            {arm["mode"]: arm["final_loss"] for arm in experiment["arms"]}[treatment]
            - {arm["mode"]: arm["final_loss"] for arm in experiment["arms"]}["random"]
            for experiment in experiments
        ]
        trials = len(seeds)
        wins = sum(1 for delta in deltas if delta < 0.0)
        min_wins = _gate_min_wins(trials, GATE_ALPHA)
        search = {}
        training = {}
        for experiment in experiments:
            for arm in experiment["arms"]:
                search[arm["mode"]] = search.get(arm["mode"], 0) + int(arm["search_flops"])
                training[arm["mode"]] = training.get(arm["mode"], 0) + int(
                    arm["training_flops"]
                )
        summary = {
            "record_type": "gate_study",
            "protocol": "factor-map-recheck-9.10",
            "provenance": provenance,
            "config": {"arms": f"random,{treatment}", "independent_arm_rng": True},
            "seeds": seeds,
            "treatment": treatment,
            "experiments": experiments,
            "adaptive_wins": wins,
            "win_rate": wins / trials if trials else 0.0,
            "mean_adaptive_delta": sum(deltas) / trials if trials else 0.0,
            "delta_stddev": _population_stddev(tuple(deltas)),
            "matched_mutation_budget": True,
            "search_flops_by_arm": dict(sorted(search.items())),
            "training_flops_by_arm": dict(sorted(training.items())),
            "gate_alpha": GATE_ALPHA,
            "gate_p_value": _binomial_survival(wins, trials) if trials else 1.0,
            "gate_min_wins": min_wins,
            "gate_passed": bool(trials) and wins >= min_wins,
        }
        check_summary(summary, study)
        summaries[study] = summary
    return summaries


def compare(records: list[dict], published: dict[str, dict]) -> list[dict]:
    """Published vs recheck, per study, with the declared rule applied."""
    rows: list[dict] = []
    for study, per_seed in sorted(aggregate(records).items()):
        source = published[study]
        expected_seeds = list(source["seeds"])
        seeds = [row[0] for row in per_seed]
        complete = seeds == sorted(expected_seeds)
        trials = len(seeds)
        wins = sum(1 for _, treatment, random_loss in per_seed if treatment < random_loss)
        mean_delta = (
            sum(treatment - random_loss for _, treatment, random_loss in per_seed) / trials
            if trials
            else 0.0
        )
        # Thresholds are recomputed from n, never carried over from the summary,
        # so a truncated shard set cannot be scored against the full block's bar.
        min_wins = _gate_min_wins(trials, GATE_ALPHA) if trials else 0
        published_min = source["gate_min_wins"]
        rows.append(
            {
                "study": study,
                "trials": trials,
                "seeds_complete": complete,
                "treatment": source.get("treatment", "adaptive"),
                "published_wins": source["adaptive_wins"],
                "published_min_wins": published_min,
                "published_passed": bool(source["gate_passed"]),
                "published_p": source["gate_p_value"],
                "published_mean_delta": source["mean_adaptive_delta"],
                "recheck_wins": wins,
                "recheck_min_wins": min_wins,
                "recheck_passed": bool(trials) and wins >= min_wins,
                "recheck_p": _binomial_survival(wins, trials) if trials else 1.0,
                "recheck_mean_delta": mean_delta,
                "win_delta": wins - source["adaptive_wins"],
                "label_agreement": (
                    bool(trials)
                    and complete
                    and bool(wins >= min_wins) == bool(source["gate_passed"])
                ),
            }
        )
    return rows


def _print_list() -> None:
    available = recheck_set()
    print(f"recheck set: {len(available)} study(ies)\n")
    for name, data in sorted(available.items()):
        label = "PASS" if data["gate_passed"] else "FAIL"
        note = " (reconstructed)" if name in RECONSTRUCTION else ""
        print(
            f"  {name:28s} n={len(data['seeds']):3d} wins={data['adaptive_wins']:3d} "
            f"({label})  seeds {data['seeds'][0]}-{data['seeds'][-1]}{note}"
        )
    for name, fields in RECONSTRUCTION.items():
        supplied = ", ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))
        print(f"\nreconstructed: {name}\n  supplies {supplied} (record predates the field)")


def shard_summary(path: Path, record: dict) -> str:
    """Confirmation line for a written shard.

    Shard records key their per-seed payload as ``experiments``; ``per_seed``
    belongs to the probe-validity script's schema. Reading the wrong key here
    crashed a shard *after* it had written its file, so a completed shard
    looked like a failed one - hence this being a tested function.
    """
    return f"{path}  ({len(record['experiments'])} seed(s))"


def _print_compare(rows: list[dict]) -> None:
    print(
        f"{'study':28s} {'n':>4s} {'published':>14s} {'recheck':>14s} "
        f"{'d':>5s} {'verdict':>10s}"
    )
    for row in rows:
        pub = f"{row['published_wins']}/{row['trials']} " + (
            "PASS" if row["published_passed"] else "FAIL"
        )
        rec = f"{row['recheck_wins']}/{row['trials']} " + (
            "PASS" if row["recheck_passed"] else "FAIL"
        )
        if not row["seeds_complete"]:
            verdict = "INCOMPLETE"
        elif row["label_agreement"]:
            verdict = "reproduces"
        else:
            verdict = "FLIPS"
        print(
            f"{row['study']:28s} {row['trials']:4d} {pub:>14s} {rec:>14s} "
            f"{row['win_delta']:+5d} {verdict:>10s}"
        )
    flips = [row["study"] for row in rows if row["seeds_complete"] and not row["label_agreement"]]
    print(
        f"\n{len(rows)} study(ies) compared, {len(flips)} label flip(s)"
        + (f": {', '.join(flips)}" if flips else "")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show the recheck set")
    parser.add_argument("--study", help="study stem to re-run (see --list)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--out", type=Path, help="shard output directory")
    parser.add_argument(
        "--compare", default=None,
        help="glob of shard records; aggregates and applies the declared rule",
    )
    parser.add_argument(
        "--compare-out", type=Path, default=None,
        help="write the aggregated comparison JSON here",
    )
    parser.add_argument(
        "--summaries", type=Path, default=None,
        help="with --compare: write decoupled study summaries in the committed "
        "schema here (each validated by figures_data.check_summary) so the "
        "figures can be regenerated from the decoupled record",
    )
    parser.add_argument(
        "--coupled", action="store_true",
        help="run with the historical shared-RNG path (for a control comparison)",
    )
    args = parser.parse_args()

    if args.list:
        _print_list()
        return 0

    if args.compare:
        paths = sorted(glob.glob(args.compare))
        if not paths:
            raise SystemExit(f"no shard records match {args.compare!r}")
        records = [json.loads(Path(path).read_text()) for path in paths]
        rows = compare(records, published_summaries())
        _print_compare(rows)
        if args.compare_out is not None:
            args.compare_out.parent.mkdir(parents=True, exist_ok=True)
            args.compare_out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        if args.summaries is not None:
            summaries = summaries_from_records(records)
            args.summaries.mkdir(parents=True, exist_ok=True)
            for study, summary in summaries.items():
                target = args.summaries / f"{study}.json"
                target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            print(
                f"\nwrote {len(summaries)} decoupled summary(ies) to {args.summaries} "
                "(each re-verified by figures_data.check_summary)"
            )
        return 0

    if not args.study:
        raise SystemExit("pass --list, --study, or --compare")
    if args.out is None:
        raise SystemExit("--study requires --out (a shard output directory)")
    record = run_shard(
        args.study,
        shard=args.shard,
        shards=args.shards,
        independent_arm_rng=not args.coupled,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "coupled" if args.coupled else "decoupled"
    path = args.out / f"{args.study}.shard{args.shard:02d}.{suffix}.json"
    path.write_text(json.dumps(record) + "\n")
    print(shard_summary(path, record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
