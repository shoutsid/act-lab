"""Read section 5.70's twelve pairs and write the committed reading.

The registration (``artifacts/real-text-declared/NOTES.md``) says the reading
is written only after all twelve pairs are fetched, with the sign test and the
mean computed from the rows. This script is that instrument: it refuses to
write if any seed is missing, if the rows come from more than one code
revision, or if the registered schedule (both arms, three events at
1500/3000/4500) is not in every row.

    PYTHONPATH=src python scripts/summarize_real_text_declared.py
    PYTHONPATH=src python scripts/summarize_real_text_declared.py --preview

``--preview`` computes over whatever rows exist and prints them without writing;
it is for watching a block in flight, never for the record.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from act.evolution.study import _binomial_survival, _gate_min_wins  # noqa: E402

SEEDS = tuple(range(15036, 15048))
BAR_WINS = 10
ALPHA = 0.05
EVENT_STEPS = (1500, 3000, 4500)
#: The revision the three staged kernels mounted and their rows name.
EXPECTED_REVISION = "8fe7bbff1b3f1e10e08260e6c434e43a21085a83"
ROWS = ROOT / "artifacts" / "real-text-declared" / "kernel"
OUT = ROOT / "artifacts" / "study-summaries" / "real-text-declared-12.json"

BRANCHES = {
    "A": {
        "reading": "the edge survives at declared scale on real text",
        "consequence": (
            "Gate 4's quality premise is supported at this fidelity; the next "
            "registered step is the pool term (a declared real-text null arm), "
            "not Gate 5"
        ),
    },
    "B": {
        "reading": "the dev-fidelity negative replicates at declared scale",
        "consequence": (
            "the real-text declared arm closes; no Gate 5, no further spend on "
            "this axis"
        ),
    },
    "C": {
        "reading": "not demonstrated",
        "consequence": (
            "reported as inconclusive, same withholding as the synthetic block; "
            "no downstream authorization"
        ),
    },
}


def load_rows() -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for path in sorted(ROWS.glob("*/seed-*.json")):
        if path.name.endswith(".error.json"):
            continue
        row = json.loads(path.read_text())
        rows[int(row["seed"])] = row
    return rows


def validate(rows: dict[int, dict]) -> list[str]:
    problems: list[str] = []
    missing = [seed for seed in SEEDS if seed not in rows]
    stray = sorted(set(rows) - set(SEEDS))
    if missing:
        problems.append(f"missing seeds: {missing}")
    if stray:
        problems.append(f"seeds outside the registration: {stray}")

    revisions = {row.get("code_revision") for row in rows.values()}
    if revisions != {EXPECTED_REVISION}:
        problems.append(f"code revisions: {sorted(revisions)} != {EXPECTED_REVISION}")

    for seed in SEEDS:
        row = rows.get(seed)
        if row is None:
            continue
        arms = row.get("arms") or {}
        if set(arms) != {"adaptive", "random"}:
            problems.append(f"seed {seed}: arms {sorted(arms)}")
            continue
        for mode, arm in arms.items():
            try:
                finite = all(
                    math.isfinite(arm[key]) for key in ("final_loss", "validation_loss")
                )
            except (KeyError, TypeError):
                problems.append(f"seed {seed} {mode}: missing loss fields")
                continue
            if not finite:
                problems.append(f"seed {seed} {mode}: non-finite loss")
            if arm["final_loss"] != arm["validation_loss"]:
                problems.append(
                    f"seed {seed} {mode}: final_loss != validation_loss "
                    "(run_v0_session defines them as one held-out reading)"
                )
            if arm.get("mutation_count") != 3:
                problems.append(f"seed {seed} {mode}: mutations {arm.get('mutation_count')}")
            steps = tuple(event.get("step") for event in arm.get("events", []))
            if steps != EVENT_STEPS:
                problems.append(f"seed {seed} {mode}: event steps {steps}")
    return problems


def branch(wins: int) -> str:
    if wins >= BAR_WINS:
        return "A"
    if wins <= 2:
        return "B"
    return "C"


def build(
    rows: dict[int, dict],
    seeds: tuple[int, ...] = SEEDS,
    enforce_bar: bool = True,
) -> dict:
    per_seed = []
    deltas = []
    for seed in seeds:
        row = rows[seed]
        adaptive = row["arms"]["adaptive"]
        random = row["arms"]["random"]
        delta = adaptive["final_loss"] - random["final_loss"]
        deltas.append(delta)
        per_seed.append(
            {
                "seed": seed,
                "adaptive_loss": adaptive["final_loss"],
                "random_loss": random["final_loss"],
                "delta": delta,
                "adaptive_active_parameters": adaptive["active_parameter_count"],
                "random_active_parameters": random["active_parameter_count"],
                "adaptive_mutations": adaptive["mutation_count"],
                "random_mutations": random["mutation_count"],
                "wall_seconds": row["wall_seconds"],
            }
        )

    wins = sum(delta < 0 for delta in deltas)
    ties = sum(delta == 0 for delta in deltas)
    p = _binomial_survival(wins, len(seeds))
    bar = _gate_min_wins(len(seeds), ALPHA)
    if enforce_bar:
        if bar != BAR_WINS:
            raise SystemExit(f"declared bar {BAR_WINS} != exact binomial bar {bar}")
        if abs(p - 79 / 4096) > 1e-12 and wins == BAR_WINS:
            raise SystemExit(f"sign-test p at the bar is {p!r}, not 79/4096")

    letter = branch(wins)
    mean_delta = statistics.fmean(deltas)
    sample_sd = statistics.stdev(deltas)
    totals = [row["arms"][mode]["parameter_count"] for row in rows.values() for mode in ("adaptive", "random")]
    actives = [
        row["arms"][mode]["active_parameter_count"]
        for row in rows.values()
        for mode in ("adaptive", "random")
    ]
    walls = [row["wall_seconds"] for row in rows.values()]
    config = dict(rows[seeds[0]]["config"])

    return {
        "record_type": "study-summary",
        "kind": "real-text-declared-5.70",
        "report_section": "5.70",
        "protocol": (
            "declared shape on the real TinyStories stream, adaptive versus "
            "random churn paired within seed, three events at 1500/3000/4500"
        ),
        "code_revision": EXPECTED_REVISION,
        "seeds": list(SEEDS),
        "alpha": ALPHA,
        "declared_bar_wins": BAR_WINS,
        "declared": {
            "total_steps": 6000,
            "batch_size": 2,
            "context_size": 2048,
            "tokens_per_arm": 24_576_000,
            "vocab_size": 32768,
            "shape": "285M total / 134M active",
        },
        "config": config,
        "per_seed": per_seed,
        "primary": {
            "wins": wins,
            "ties": ties,
            "pairs": len(seeds),
            "bar": BAR_WINS,
            "binomial_p": p,
            "passes": wins >= BAR_WINS,
            "mean_delta": mean_delta,
            "sample_sd": sample_sd,
            "min_delta": min(deltas),
            "max_delta": max(deltas),
            "branch": letter,
            "reading": BRANCHES[letter]["reading"],
            "consequence": BRANCHES[letter]["consequence"],
        },
        "observed": {
            "total_parameters": [min(totals), max(totals)],
            "active_parameters": [min(actives), max(actives)],
            "wall_hours": [min(walls) / 3600, max(walls) / 3600, sum(walls) / 3600],
        },
        "built_in_checks": {
            "all_twelve_seeds_present": set(rows) == set(SEEDS),
            "one_code_revision": {row["code_revision"] for row in rows.values()} == {EXPECTED_REVISION},
            "both_arms_in_every_seed": all(
                set(row["arms"]) == {"adaptive", "random"} for row in rows.values()
            ),
            "three_events_on_the_registered_steps": all(
                tuple(event.get("step") for event in arm.get("events", [])) == EVENT_STEPS
                for row in rows.values()
                for arm in row["arms"].values()
            ),
            "final_loss_is_the_held_out_validation_loss": all(
                arm["final_loss"] == arm["validation_loss"]
                for row in rows.values()
                for arm in row["arms"].values()
            ),
            "declared_bar_matches_exact_binomial": bar == BAR_WINS,
        },
        "caveats": [
            (
                "the twelve seeds share one ~40.04M-token stream against "
                "24.576M per arm: each draws its own windows, but they are not "
                "twelve independent text samples"
            ),
            (
                "the declared vocabulary is 32768 while the stream's ids stop at "
                "8191, so 24576 embedding rows never train; identical for both "
                "arms, and absolute losses are not comparable to the real-LM "
                "model's 8192-vocab readings"
            ),
        ],
        "summary_note": (
            "Declared-shape real-text block (report section 5.70), seeds "
            "15036-15047, revision 8fe7bbf. Every quoted statistic recomputes "
            "from the embedded per-seed rows; the p-value is the exact one-tailed "
            "binomial sign test at the campaign's 10-of-12 bar."
        ),
    }


def print_table(payload: dict) -> None:
    print(f"{'seed':>6}  {'adaptive':>10}  {'random':>10}  {'delta':>10}")
    for row in payload["per_seed"]:
        print(
            f"{row['seed']:>6}  {row['adaptive_loss']:>10.6f}  "
            f"{row['random_loss']:>10.6f}  {row['delta']:>+10.6f}"
        )
    primary = payload["primary"]
    print(
        f"wins {primary['wins']}/{primary['pairs']}  ties {primary['ties']}  "
        f"mean {primary['mean_delta']:+.6f}  sd {primary['sample_sd']:.6f}  "
        f"p {primary['binomial_p']:.9f}  branch {primary['branch']}"
    )
    if primary["pairs"] == len(SEEDS):
        print(f"reading: {primary['reading']}")
    else:
        print(
            f"partial: {primary['wins']}/{primary['pairs']} so far; the "
            f"{len(SEEDS)}-seed branch is not yet decidable"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    rows = load_rows()
    problems = validate(rows)
    if args.preview:
        known = {seed: row for seed, row in rows.items() if seed in SEEDS}
        if not known:
            print("no rows yet")
            return 2
        payload = build(known, tuple(sorted(known)), enforce_bar=False)
        print_table(payload)
        if problems:
            print("NOT the record; " + "; ".join(problems))
            return 2
        return 0
    if problems:
        for problem in problems:
            print(problem)
        print("refusing to write the reading")
        return 2

    payload = build(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print_table(payload)
    print(f"summary written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
