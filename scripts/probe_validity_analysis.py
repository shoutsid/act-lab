"""Re-run the probe-validity counterfactual analysis (report section 3.2).

The analysis that backs figure 4 and section 3.2's mechanism claim was
originally produced ad hoc: nothing in the repository generated
``artifacts/study-summaries/probe-validity-analysis.json``, so its numbers were
reproducible only in principle. This script is that entry point.

Protocol (the committed ``design`` string, restated):

- Study ``gate3-probe-loss-384``: seeds 3000-3383, steps=48, cap=4 candidates,
  2 probe batches, depth 1.
- Counterfactual branches force candidate ``k`` in {1,2,3} at the **first
  mutation event only**; branch 0 is the default greedy run. Candidates are
  indexed into the probe's sorted evaluation order, so branch 0 is the
  probe-best and branch k is the (k+1)-th best.
- ``probe_losses`` is that first event's full landscape (best-first) and
  ``finals`` is each branch's realized final adaptive loss, index-aligned, so
  ``probe_losses[k]`` pairs with ``finals[k]``.
- H1: mean per-seed Spearman rho > 0, one-tailed, pre-registered.

Two re-run regimes: the historical shared-RNG path (the committed file) and the
decoupled ``independent_arm_rng=True`` path used by the section 9.11 recheck, so
figure 4 stops being the one piece of the factor map never re-run.

Definitions were recovered from the committed file and then verified against it:
every stored ``results`` field reproduces from its ``per_seed`` block (see
``--verify``), including two non-obvious ones - ``h1_one_tailed_p`` is the
**normal** one-tailed tail of the t statistic (a z-test), and
``pooled_zscored_pearson_r`` z-scores **within** each seed before pooling.

Usage:

    PYTHONPATH=src python scripts/probe_validity_analysis.py --verify
    PYTHONPATH=src python scripts/probe_validity_analysis.py \\
        --shard 0 --shards 16 --out /tmp/pv --coupled
    PYTHONPATH=src python scripts/probe_validity_analysis.py \\
        --analyze '/tmp/pv/*.json' --out artifacts/study-summaries/decoupled/probe-validity-analysis.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from act.evolution.experiment import ExperimentConfig, run_experiment  # noqa: E402
from act.evolution.figures_data import _ranks, probe_concordance  # noqa: E402

SUMMARIES = ROOT / "artifacts" / "study-summaries"
STUDY = "gate3-probe-loss-384"
BRANCHES = 4
# Branch 0 *is* the greedy run, so it must reproduce the study's adaptive arm.
# That identity is exact in principle but not portably bit-exact: the committed
# file was produced in an environment whose float arithmetic differs in the last
# digits (measured max drift 1.9e-6 on losses of order 15, 75/384 seeds exact),
# so the check is a tolerance rather than equality.
BRANCH0_TOLERANCE = 1e-5
SEED_RANGE = (3000, 3384)  # inclusive start, exclusive stop
BASE_CONFIG = {
    "steps": 48,
    "depth": 1,
    "expert_count": 4,
    "hidden_size": 24,
    "vocab_size": 32,
    "batch_size": 2,
    "sequence_length": 12,
    "validation_batches": 4,
    "mutation_interval": 12,
    "mutation_budget": 3,
    "probe_candidate_cap": 4,
    "probe_batch_count": 2,
    "probe_candidate_policy": "prefix",
    "probe_selection": "argmin",
    "probe_landscape": True,
    "arms": "random,adaptive",
    "device": "cpu",
}
ANALYSIS = (
    "probe validity: do probe-loss deltas between candidates predict final-loss "
    "deltas across seeds?"
)
DESIGN = (
    "re-run of gate3-probe-loss-384 (seeds 3000-3383, steps=48, cap=4, 2 probe "
    "batches) with probe landscape recording; counterfactual branches force "
    "candidate k in {1,2,3} at the FIRST mutation event only (branch 0 = default "
    "greedy run); per-seed Spearman between the 4 candidates' probe losses and "
    "their realized final losses; H1: mean rho > 0 (one-tailed t across seeds, "
    "alpha=0.05), pre-registered"
)
INTERPRETATION = (
    "probe loss is a statistically significant but weak predictor of realized "
    "outcome (rho ~ 0.11). Its main value is avoiding the worst candidates "
    "rather than identifying the single best one (concordance near chance; "
    "avoided excess 0.086 vs left-on-table 0.060). A weak per-event signal "
    "accumulated over only 3 mutation events is consistent with the marginal "
    "Gate 3 pass and motivates the cadence study."
)


# --- statistics (definitions pinned against the committed file) -------------


def spearman_rho(probe_losses: list[float], finals: list[float]) -> float:
    """Rank correlation with ties broken by candidate order (as the figure does)."""
    ranks_a, ranks_b = _ranks(probe_losses), _ranks(finals)
    mean_a = sum(ranks_a) / len(ranks_a)
    mean_b = sum(ranks_b) / len(ranks_b)
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(ranks_a, ranks_b))
    norm_a = math.sqrt(sum((a - mean_a) ** 2 for a in ranks_a))
    norm_b = math.sqrt(sum((b - mean_b) ** 2 for b in ranks_b))
    return numerator / (norm_a * norm_b) if norm_a and norm_b else 0.0


def normal_one_tailed_p(z: float) -> float:
    """One-tailed normal tail. The committed file stores this, not a t tail."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def pooled_zscored_pearson(per_seed: dict[str, dict]) -> float:
    """Pearson r after z-scoring *within* each seed, then pooling.

    Within-seed z-scoring removes the seed-level loss offset, so this measures
    the same within-seed association as the per-seed rho rather than the
    between-seed spread (which would swamp it).
    """

    def z(values: list[float]) -> list[float]:
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        return [(value - mean) / sd for value in values]

    probes: list[float] = []
    finals: list[float] = []
    for entry in per_seed.values():
        probes += z(entry["probe_losses"])
        finals += z(entry["finals"])
    numerator = sum(p * f for p, f in zip(probes, finals))
    norm = math.sqrt(sum(p * p for p in probes) * sum(f * f for f in finals))
    return numerator / norm if norm else 0.0


def results_block(per_seed: dict[str, dict]) -> dict:
    """Every stored ``results`` field, recomputed from ``per_seed``."""
    rhos = [spearman_rho(e["probe_losses"], e["finals"]) for e in per_seed.values()]
    mean_rho = statistics.mean(rhos)
    rho_sd = statistics.stdev(rhos)
    trials = len(rhos)
    t_stat = mean_rho / (rho_sd / math.sqrt(trials)) if rho_sd else 0.0
    concordance, total = probe_concordance(per_seed)
    k = len(next(iter(per_seed.values()))["probe_losses"])

    def chosen_index(entry: dict) -> int:
        # Ties resolve to the earliest candidate, matching the probe's own
        # (loss, order) sort and figures_data's rank convention.
        best = min(entry["probe_losses"])
        return entry["probe_losses"].index(best)

    worst_avoided = statistics.mean(
        max(e["finals"]) - e["finals"][chosen_index(e)] for e in per_seed.values()
    )
    left_on_table = statistics.mean(
        e["finals"][chosen_index(e)] - min(e["finals"]) for e in per_seed.values()
    )
    p_value = normal_one_tailed_p(t_stat)
    return {
        "n_seeds": total,
        "pooled_pairs": total * k,
        "mean_per_seed_spearman_rho": mean_rho,
        "rho_sd": rho_sd,
        "fraction_seeds_rho_positive": statistics.mean(
            1.0 if rho > 0 else 0.0 for rho in rhos
        ),
        "h1_t_statistic": t_stat,
        "h1_one_tailed_p": p_value,
        "h1_supported": bool(p_value <= 0.05 and mean_rho > 0),
        "pooled_zscored_pearson_r": pooled_zscored_pearson(per_seed),
        "argmin_concordance": concordance,
        "argmin_concordance_chance": 1.0 / k,
        "mean_worst_branch_excess_avoided": worst_avoided,
        "mean_loss_left_on_table_probe_best_vs_best_of_4": left_on_table,
    }


# --- running the counterfactual sweep ----------------------------------------


def run_seed(seed: int, *, independent_arm_rng: bool) -> dict:
    """The four counterfactual branches for one seed."""
    probe_losses: list[float] | None = None
    finals: list[float] = []
    for branch in range(BRANCHES):
        config = ExperimentConfig(
            seed=seed,
            independent_arm_rng=independent_arm_rng,
            probe_force_candidate=None if branch == 0 else branch,
            probe_force_events=None if branch == 0 else 1,
            **BASE_CONFIG,
        )
        result = run_experiment(config)
        adaptive = next(arm for arm in result.arms if arm.mode == "adaptive")
        if not adaptive.mutation_history:
            raise SystemExit(f"seed {seed}: no mutation event recorded")
        history = adaptive.mutation_history[0]
        landscape = history.get("probe_landscape")
        if not landscape:
            raise SystemExit(
                f"seed {seed}: no probe landscape recorded (probe_landscape must be on)"
            )
        if probe_losses is None:
            probe_losses = [candidate["probe_loss"] for candidate in landscape]
        finals.append(adaptive.final_loss)
    assert probe_losses is not None
    if len(probe_losses) != BRANCHES:
        raise SystemExit(
            f"seed {seed}: {len(probe_losses)} candidates evaluated, "
            f"expected {BRANCHES} (the branch design assumes a full cap)"
        )
    return {"probe_losses": probe_losses, "finals": finals}


def run_shard(shard: int, shards: int, *, independent_arm_rng: bool) -> dict:
    if shards < 1 or not 0 <= shard < shards:
        raise SystemExit("--shard must be in [0, --shards)")
    seeds = tuple(range(*SEED_RANGE))[shard::shards]
    per_seed: dict[str, dict] = {}
    for seed in seeds:
        per_seed[str(seed)] = run_seed(seed, independent_arm_rng=independent_arm_rng)
    return {
        "protocol": "probe-validity-counterfactual",
        "study": STUDY,
        "shard": shard,
        "shards": shards,
        "independent_arm_rng": independent_arm_rng,
        "seeds": list(seeds),
        "per_seed": per_seed,
    }


def merge(records: list[dict]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for record in records:
        for seed, entry in record["per_seed"].items():
            merged[seed] = entry
    return {seed: merged[seed] for seed in sorted(merged, key=int)}


def steady_state_finals(path: Path) -> dict[str, float]:
    """Per-seed adaptive final loss from a committed study summary."""
    summary = json.loads(path.read_text())
    finals: dict[str, float] = {}
    for experiment in summary["experiments"]:
        seed = str(experiment["config"]["seed"])
        arms = {arm["mode"]: arm for arm in experiment["arms"]}
        finals[seed] = float(arms["adaptive"]["final_loss"])
    return finals


def build_document(per_seed: dict[str, dict], *, independent_arm_rng: bool) -> dict:
    regime = "decoupled" if independent_arm_rng else "coupled"
    return {
        "protocol": "probe-validity-counterfactual",
        "record_type": "probe_validity_analysis",
        "rng_regime": regime,
        "independent_arm_rng": independent_arm_rng,
        "provenance": f"scripts/probe_validity_analysis.py ({regime} arm RNG)",
        "analysis": ANALYSIS,
        "design": DESIGN,
        "interpretation": INTERPRETATION,
        "per_seed": per_seed,
        "results": results_block(per_seed),
    }


def verify_against_committed() -> int:
    """Recompute the committed results block and check every stored field."""
    committed = json.loads((SUMMARIES / "probe-validity-analysis.json").read_text())
    recomputed = results_block(committed["per_seed"])
    failures = []
    for key, value in committed["results"].items():
        mine = recomputed.get(key)
        if mine is None:
            failures.append(f"  {key}: not recomputed")
        elif isinstance(value, bool):
            if bool(mine) != value:
                failures.append(f"  {key}: {mine} != {value}")
        elif isinstance(value, int):
            if mine != value:
                failures.append(f"  {key}: {mine} != {value}")
        else:
            # The committed file stores the rates rounded to 3 decimals (as
            # generate_figures documents), so agreement is judged at that
            # precision rather than to full float equality.
            tolerance = 5e-4 if abs(float(value)) <= 1.0 else 5e-5 * abs(float(value))
            if not math.isclose(
                float(mine), float(value), rel_tol=5e-5, abs_tol=tolerance
            ):
                failures.append(f"  {key}: {mine} != {value}")
    print(f"committed results block: {len(committed['results'])} field(s)")
    if failures:
        print("MISMATCH:")
        print("\n".join(failures))
        return 1
    print("every field recomputes from the committed per_seed block")
    return 0


def _display(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise (scratch output dirs)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="check the committed file")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--coupled", action="store_true", help="use the historical shared-RNG path")
    parser.add_argument(
        "--analyze", default=None,
        help="glob of shard records: merge, recompute the results block, and write the document",
    )
    parser.add_argument(
        "--cross-check", default=None,
        help="study summary whose adaptive final loss branch 0 must reproduce "
        "(default: the coupled study, or decoupled/<study> with --coupled omitted "
        "when the analyzing decoupled records)",
    )
    args = parser.parse_args()

    if args.verify:
        return verify_against_committed()

    if args.analyze:
        paths = sorted(glob.glob(args.analyze))
        if not paths:
            raise SystemExit(f"no shard records match {args.analyze!r}")
        records = [json.loads(Path(path).read_text()) for path in paths]
        regimes = {bool(record["independent_arm_rng"]) for record in records}
        if len(regimes) != 1:
            raise SystemExit("shard records mix RNG regimes; refusing to merge")
        independent = regimes.pop()
        per_seed = merge(records)
        expected = set(range(*SEED_RANGE))
        missing = sorted(expected - {int(seed) for seed in per_seed})
        if missing:
            raise SystemExit(f"incomplete sweep: {len(missing)} seed(s) missing, e.g. {missing[:5]}")
        document = build_document(per_seed, independent_arm_rng=independent)

        # Branch 0 is the greedy run, so it must reproduce the committed study's
        # adaptive arm seed for seed - the check that the counterfactual really
        # is the same experiment with one forced choice.
        cross = Path(args.cross_check) if args.cross_check else (
            SUMMARIES / (f"{STUDY}.json" if independent is False else f"decoupled/{STUDY}.json")
        )
        if cross.exists():
            steady = steady_state_finals(cross)
            mismatches = [
                seed
                for seed, entry in per_seed.items()
                if seed in steady
                and abs(entry["finals"][0] - steady[seed]) > BRANCH0_TOLERANCE
            ]
            document["branch0_reproduces_study"] = not mismatches
            document["branch0_cross_check"] = _display(cross)
            document["branch0_tolerance"] = BRANCH0_TOLERANCE
            print(
                f"branch-0 vs {cross.name}: "
                + ("reproduces" if not mismatches else f"{len(mismatches)} mismatch(es)")
            )
            if mismatches:
                raise SystemExit(
                    f"branch 0 does not reproduce the study's adaptive arm on "
                    f"{len(mismatches)} seed(s), e.g. {mismatches[:5]}"
                )
        else:
            print(f"note: no cross-check summary at {cross}")

        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
            print(f"wrote {_display(args.out)}")
        results = document["results"]
        print(
            f"rho {results['mean_per_seed_spearman_rho']:+.4f} "
            f"(p {results['h1_one_tailed_p']:.3g}), concordance "
            f"{results['argmin_concordance']:.3f} vs chance "
            f"{results['argmin_concordance_chance']:.2f}"
        )
        return 0

    if args.out is None:
        raise SystemExit("--shard requires --out (a shard output directory)")
    record = run_shard(args.shard, args.shards, independent_arm_rng=not args.coupled)
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "coupled" if args.coupled else "decoupled"
    path = args.out / f"probe-validity.shard{args.shard:02d}.{suffix}.json"
    path.write_text(json.dumps(record) + "\n")
    print(f"{path}  ({len(record['per_seed'])} seed(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
