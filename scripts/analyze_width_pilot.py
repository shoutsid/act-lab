"""Read the width pilot's records against its pre-registered rule.

The registration is ``artifacts/width-pilot/NOTES.md``: two arms, the same
three widen events at steps 100/200/300, 12 paired seeds, and a positive edge
only at 10 or more wins of 12. This script is the reader for those records.

It reports what the registration asks for and nothing else:

* the per-seed paired difference ``width - width_random`` (negative favours
  the adaptive rule), its mean and paired standard deviation;
* the exact one-sided sign test on the number of seeds the adaptive arm wins;
* the final width and the number of accepted events in each arm, so a
  difference can be read against the schedule that produced it.

A block that is not 12/12 is **not** read as a result: the fourth registered
outcome makes it undecidable, and the script exits 2. The script also checks
the schedule itself. The session machinery used to count ``mutation_budget``
against accepted mutations, not events, so a coin arm that declined could
fire a fourth event past step 300 (fixed 2026-11-05); records produced under
that scheduler do not carry the registered three events, and the script says
so (exit 3) rather than computing a reading the registration does not define.

Usage::

    PYTHONPATH=src python scripts/analyze_width_pilot.py
    PYTHONPATH=src python scripts/analyze_width_pilot.py --out artifacts/width-pilot/reading.json

Exit codes: 0 = the registered reading was produced; 1 = malformed records;
2 = the block is incomplete; 3 = the records violate the registered schedule.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

from act.evolution.study import _binomial_survival

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECORDS = ROOT / "artifacts" / "width-pilot" / "kaggle"

SEEDS = tuple(range(71000, 71012))
ARMS = ("width", "width_random")
#: The amendment's do-nothing arm. It is not part of the paired edge reading; it
#: exists so the block's decomposition gains its other term, ``random - null``.
NULL_ARM = "null"
ALLOWED_ARMS = ARMS + (NULL_ARM,)
REGISTERED_STEPS = (100, 200, 300)
BAR_WINS = 10
ACCEPTED = "widen"
DECLINED = "widen_declined"

# The record fields the registration fixes. A record from another shape (or a
# smoke run) would be pooled with these otherwise, which is the failure this
# project's "do not pool measurements that differ in configuration" rule exists
# to prevent.
REGISTERED_CONFIG = {
    "total_steps": 400,
    "mutation_interval": 100,
    "mutation_budget": 3,
    "hidden_size": 320,
    "attention_heads": 5,
    "width_heads_added": 2,
    "width_accept_margin": 0.0,
    "width_random_probability": 0.5,
}


class ReadingError(ValueError):
    """The records cannot answer the registered question as they stand."""


def load_records(directory: Path) -> dict[tuple[int, str], dict[str, object]]:
    """Every ``*-seed*.json`` record under ``directory``, keyed by (seed, arm)."""

    if not directory.is_dir():
        raise ReadingError(f"no records directory at {directory}")
    records: dict[tuple[int, str], dict[str, object]] = {}
    for path in sorted(directory.glob("*-seed*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ReadingError(f"{path.name}: unreadable record ({error})") from error
        if not isinstance(payload, dict) or payload.get("kind") != "width-pilot":
            raise ReadingError(f"{path.name}: not a width-pilot record")
        seed, arm = payload.get("seed"), payload.get("arm")
        if seed not in SEEDS or arm not in ALLOWED_ARMS:
            raise ReadingError(
                f"{path.name}: {seed}:{arm} is outside the registered block"
            )
        key = (int(seed), str(arm))
        if key in records:
            raise ReadingError(f"{path.name}: duplicate record for {seed}:{arm}")
        records[key] = payload
    if not records:
        raise ReadingError(f"no width-pilot records under {directory}")
    return records


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReadingError(f"{name} is not a number: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ReadingError(f"{name} is not finite: {value!r}")
    return number


def _validate_record(
    payload: dict[str, object], seed: int, arm: str
) -> dict[str, object]:
    """Check one record against the registration, and return what it says.

    Schedule deviations are recorded, not raised: an incomplete block is
    already undecidable, and a deviated one has to be reported rather than
    hidden behind a traceback. Structural problems *are* raised - a record
    that does not carry the fields cannot be read at all.
    """

    if payload.get("seed") != seed or payload.get("arm") != arm:
        raise ReadingError(f"record key {seed}:{arm} disagrees with its fields")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ReadingError(f"{seed}:{arm}: no config")
    if list(config.get("arms") or []) != [arm]:
        raise ReadingError(f"{seed}:{arm}: config.arms is not [{arm!r}]")
    for name, expected in REGISTERED_CONFIG.items():
        if config.get(name) != expected:
            raise ReadingError(
                f"{seed}:{arm}: config.{name} is {config.get(name)!r}, "
                f"not the registered {expected!r}"
            )

    result = payload.get("arm_result")
    if not isinstance(result, dict):
        raise ReadingError(f"{seed}:{arm}: no arm_result")
    loss = _number(result.get("validation_loss"), f"{seed}:{arm} validation_loss")
    history = result.get("mutation_history")
    if not isinstance(history, list):
        raise ReadingError(f"{seed}:{arm}: no mutation_history")

    event_steps: list[int] = []
    accepted = 0
    declined = 0
    for entry in history:
        if not isinstance(entry, dict):
            raise ReadingError(f"{seed}:{arm}: non-dict mutation_history entry")
        operation = entry.get("operation")
        step = entry.get("step")
        if not isinstance(step, int):
            raise ReadingError(f"{seed}:{arm}: event step {step!r} is not an int")
        if arm == NULL_ARM:
            # The do-nothing arm consumes each slot and records a no-op. Any
            # other operation means this record is not the arm it claims.
            if operation != NULL_ARM:
                raise ReadingError(
                    f"{seed}:{arm}: the null arm recorded a {operation!r} event"
                )
        elif operation not in {ACCEPTED, DECLINED}:
            raise ReadingError(
                f"{seed}:{arm}: unexpected mutation event {operation!r} at {step!r}"
            )
        event_steps.append(step)
        if operation == ACCEPTED:
            accepted += 1
        elif operation == DECLINED:
            declined += 1
    if event_steps != sorted(set(event_steps)):
        raise ReadingError(f"{seed}:{arm}: event steps are not strictly increasing")
    if result.get("mutation_count") != accepted:
        raise ReadingError(
            f"{seed}:{arm}: mutation_count {result.get('mutation_count')!r} "
            f"disagrees with {accepted} accepted events"
        )

    hidden = payload.get("final_hidden_size")
    heads = payload.get("final_heads")
    if not isinstance(hidden, int) or not isinstance(heads, int):
        raise ReadingError(f"{seed}:{arm}: final width is missing")
    head_dim = int(config["hidden_size"]) // int(config["attention_heads"])
    heads_added = int(config["width_heads_added"])
    expected_hidden = int(config["hidden_size"]) + accepted * heads_added * head_dim
    expected_heads = int(config["attention_heads"]) + accepted * heads_added
    if (hidden, heads) != (expected_hidden, expected_heads):
        raise ReadingError(
            f"{seed}:{arm}: final width {hidden}/{heads} does not match "
            f"{accepted} accepted events ({expected_hidden}/{expected_heads})"
        )

    return {
        "config": config,
        "loss": loss,
        "accepted": accepted,
        "declined": declined,
        "event_steps": tuple(event_steps),
        "hidden": hidden,
        "heads": heads,
    }


def _schedule_deviations(
    records: dict[tuple[int, str], dict[str, object]],
    arms: tuple[str, ...],
) -> list[dict[str, object]]:
    """``arms``' records whose events differ from the registered 100/200/300."""

    deviations: list[dict[str, object]] = []
    for (seed, arm), record in sorted(records.items()):
        if arm not in arms:
            continue
        steps = record["event_steps"]
        assert isinstance(steps, tuple)
        extra = [step for step in steps if step not in REGISTERED_STEPS]
        absent = [step for step in REGISTERED_STEPS if step not in steps]
        if extra or absent:
            deviations.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "extra_steps": extra,
                    "missing_steps": absent,
                }
            )
    return deviations


def _pool_reading(
    validated: dict[tuple[int, str], dict[str, object]],
    deviations: list[dict[str, object]],
) -> dict[str, object]:
    """The amendment's term: ``width_random - null``, negative when the graft pays.

    The three arms decompose as ``adaptive - null = (adaptive - random) +
    (random - null)``. The edge reading above measures the first bracket; this
    one measures the second, and it is the only comparison that can tell the
    registration's outcome 2 (pool positive, edge flat) from outcome 3.
    """

    has_null = any(arm == NULL_ARM for _, arm in validated)
    paired = [
        seed
        for seed in SEEDS
        if (seed, NULL_ARM) in validated and (seed, "width_random") in validated
    ]
    deltas = [
        float(validated[(seed, "width_random")]["loss"])
        - float(validated[(seed, NULL_ARM)]["loss"])
        for seed in paired
    ]
    wins = sum(delta < 0 for delta in deltas)
    losses = sum(delta > 0 for delta in deltas)

    if not has_null:
        status = "not_run"
        outcome = "NOT RUN - no null-arm records; the pool term is unmeasured"
    elif len(paired) < len(SEEDS):
        status = "incomplete"
        outcome = f"UNDECIDABLE - pool block incomplete ({len(paired)}/12 pairs)"
    elif deviations:
        status = "protocol_deviation"
        direction = "pool positive" if wins >= BAR_WINS else "pool flat or negative"
        outcome = (
            "UNDECIDABLE as registered - the null arm did not book the "
            f"registered three events; descriptive term: {direction}"
        )
    elif wins >= BAR_WINS:
        status = "complete"
        outcome = "pool positive - the graft pays and the decision rule does not"
    else:
        status = "complete"
        outcome = "pool flat or negative - widening does not pay at this dose"

    return {
        "term": "width_random - null",
        "convention": "negative = the mutating arm is better than doing nothing",
        "status": status,
        "registered_reading": status == "complete",
        "outcome": outcome,
        "protocol": {
            "registered_steps": list(REGISTERED_STEPS),
            "ok": not deviations,
            "deviations": deviations,
        },
        "paired_seeds": paired,
        "missing_jobs": [
            f"{seed}:{NULL_ARM}"
            for seed in SEEDS
            if (seed, NULL_ARM) not in validated
        ],
        "n": len(deltas),
        "wins": wins,
        "losses": losses,
        "ties": len(deltas) - wins - losses,
        "event_steps": sorted(
            {
                int(step)
                for seed in paired
                for step in validated[(seed, NULL_ARM)]["event_steps"]  # type: ignore[union-attr]
            }
        ),
        "mean_delta": statistics.fmean(deltas) if deltas else None,
        "paired_sd": statistics.stdev(deltas) if len(deltas) >= 2 else None,
        "sign_test_p": _binomial_survival(wins, len(deltas)) if deltas else None,
        "bar_wins": BAR_WINS,
    }


def summarize(
    records: dict[tuple[int, str], dict[str, object]],
) -> dict[str, object]:
    """The registered reading, or the reason it cannot be produced."""

    validated: dict[tuple[int, str], dict[str, object]] = {}
    for (seed, arm), payload in records.items():
        if seed not in SEEDS or arm not in ALLOWED_ARMS:
            raise ReadingError(f"{seed}:{arm} is outside the registered block")
        validated[(seed, arm)] = _validate_record(payload, seed, arm)

    # The pair differs only in the decision rule. A field-level check catches a
    # record that silently came from another shape before it enters a delta.
    for seed in SEEDS:
        width = validated.get((seed, "width"))
        random = validated.get((seed, "width_random"))
        if width is None or random is None:
            continue
        left, right = width["config"], random["config"]
        assert isinstance(left, dict) and isinstance(right, dict)
        differing = {
            name
            for name in set(left) | set(right)
            if name != "arms" and left.get(name) != right.get(name)
        }
        if differing:
            raise ReadingError(
                f"seed {seed}: the paired arms differ in {sorted(differing)}"
            )
        null = validated.get((seed, NULL_ARM))
        if null is not None:
            left, right = null["config"], random["config"]
            assert isinstance(left, dict) and isinstance(right, dict)
            differing = {
                name
                for name in set(left) | set(right)
                if name != "arms" and left.get(name) != right.get(name)
            }
            if differing:
                raise ReadingError(
                    f"seed {seed}: the null arm differs from its coin arm in "
                    f"{sorted(differing)}"
                )

    complete_pairs = [
        seed
        for seed in SEEDS
        if (seed, "width") in validated and (seed, "width_random") in validated
    ]
    missing_jobs = [
        f"{seed}:{arm}"
        for seed in SEEDS
        for arm in ARMS
        if (seed, arm) not in validated
    ]
    # The edge reading and the pool block are judged separately: a deviation in
    # the do-nothing arm cannot invalidate the pairs that never ran it, and the
    # pool term cannot borrow the edge's registration.
    deviations = _schedule_deviations(validated, ("width", "width_random"))
    null_deviations = _schedule_deviations(validated, (NULL_ARM,))

    per_seed: list[dict[str, object]] = []
    deltas: list[float] = []
    for seed in complete_pairs:
        width = validated[(seed, "width")]
        random = validated[(seed, "width_random")]
        delta = float(width["loss"]) - float(random["loss"])
        deltas.append(delta)
        winner = "width" if delta < 0 else "width_random" if delta > 0 else "tie"
        per_seed.append(
            {
                "seed": seed,
                "width_loss": width["loss"],
                "width_random_loss": random["loss"],
                "delta": delta,
                "winner": winner,
                "width": {
                    "hidden_size": width["hidden"],
                    "heads": width["heads"],
                    "accepted": width["accepted"],
                    "declined": width["declined"],
                    "events": len(width["event_steps"]),
                },
                "width_random": {
                    "hidden_size": random["hidden"],
                    "heads": random["heads"],
                    "accepted": random["accepted"],
                    "declined": random["declined"],
                    "events": len(random["event_steps"]),
                },
            }
        )

    wins = sum(delta < 0 for delta in deltas)
    losses = sum(delta > 0 for delta in deltas)
    ties = len(deltas) - wins - losses
    mean_delta = statistics.fmean(deltas) if deltas else None
    paired_sd = statistics.stdev(deltas) if len(deltas) >= 2 else None
    sign_test_p = _binomial_survival(wins, len(deltas)) if deltas else None

    if deviations:
        status = "protocol_deviation"
    elif missing_jobs:
        status = "incomplete"
    else:
        status = "complete"
    registered_reading = status == "complete"

    if status == "protocol_deviation":
        outcome = (
            "UNDECIDABLE - the arms did not run the registered three events; "
            "no registered reading"
        )
    elif status == "incomplete":
        outcome = (
            f"UNDECIDABLE - block incomplete ({len(complete_pairs)}/12 pairs); "
            "no registered reading"
        )
    elif wins >= BAR_WINS:
        outcome = "edge positive and clear (>= 10/12); price at declared fidelity"
    else:
        outcome = "flat or negative edge (<= 9/12); close the width arm"

    arm_schedule: dict[str, object] = {}
    for arm in ARMS:
        arm_pairs = [
            validated[(seed, arm)]
            for seed in complete_pairs
            if (seed, arm) in validated
        ]
        accepted = sum(int(record["accepted"]) for record in arm_pairs)
        events = sum(len(record["event_steps"]) for record in arm_pairs)  # type: ignore[arg-type]
        widths = Counter(int(record["hidden"]) for record in arm_pairs)
        arm_schedule[arm] = {
            "pairs": len(arm_pairs),
            "accepted": accepted,
            "events": events,
            "mean_accepted": accepted / len(arm_pairs) if arm_pairs else None,
            "final_widths": {str(width): count for width, count in sorted(widths.items())},
        }

    return {
        "kind": "width-pilot-reading",
        "registration": "artifacts/width-pilot/NOTES.md",
        "records_directory": str(DEFAULT_RECORDS),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "bar_wins": BAR_WINS,
        "sign_test_alpha": _binomial_survival(BAR_WINS, len(SEEDS)),
        "status": status,
        "registered_reading": registered_reading,
        "outcome": outcome,
        "complete_pairs": complete_pairs,
        "missing_jobs": missing_jobs,
        "protocol": {
            "registered_steps": list(REGISTERED_STEPS),
            "ok": not deviations,
            "deviations": deviations,
        },
        "per_seed": per_seed,
        "comparison": {
            "n": len(deltas),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "mean_delta": mean_delta,
            "paired_sd": paired_sd,
            "sign_test_p": sign_test_p,
            "bar_wins": BAR_WINS,
        },
        "arm_schedule": arm_schedule,
        "pool": _pool_reading(validated, null_deviations),
        "record_count": len(validated),
    }


def _deviation_groups(
    deviations: list[dict[str, object]],
) -> list[tuple[str, list[int], list[int], list[int]]]:
    """``(arm, extra, missing, seeds)`` groups for a readable report."""

    groups: dict[tuple[str, tuple[int, ...], tuple[int, ...]], list[int]] = {}
    for deviation in deviations:
        key = (
            str(deviation["arm"]),
            tuple(int(step) for step in deviation["extra_steps"]),  # type: ignore[union-attr]
            tuple(int(step) for step in deviation["missing_steps"]),  # type: ignore[union-attr]
        )
        groups.setdefault(key, []).append(int(deviation["seed"]))  # type: ignore[arg-type]
    return [
        (arm, list(extra), list(missing), seeds)
        for (arm, extra, missing), seeds in sorted(groups.items())
    ]


def render(summary: dict[str, object]) -> str:
    """The human-readable reading, in the order the registration states it."""

    comparison = summary["comparison"]
    assert isinstance(comparison, dict)
    lines = [
        "Width pilot - registered reading",
        f"registration: {summary['registration']}",
        f"records: {summary['records_directory']} "
        f"({summary['record_count']} records; "
        f"{len(summary['complete_pairs'])}/12 complete pairs)",
        f"bar: >= {summary['bar_wins']}/12 wins "
        f"(exact one-sided alpha {summary['sign_test_alpha']:.6f})",
        "",
        "seed    width_loss  random_loss       delta  winner       "
        "width h/h/a/e   random h/h/a/e",
    ]
    for row in summary["per_seed"]:
        assert isinstance(row, dict)
        width = row["width"]
        random = row["width_random"]
        assert isinstance(width, dict) and isinstance(random, dict)
        lines.append(
            f"{row['seed']:>5}  {row['width_loss']:>11.6f}  "
            f"{row['width_random_loss']:>11.6f}  {row['delta']:>+11.6f}  "
            f"{row['winner']:<10}  "
            f"{width['hidden_size']:>4}/{width['heads']:>2}/"
            f"{width['accepted']}/{width['events']}  "
            f"{random['hidden_size']:>4}/{random['heads']:>2}/"
            f"{random['accepted']}/{random['events']}"
        )

    lines.append("")
    if comparison["n"]:
        lines.append(
            f"wins: {comparison['wins']}/{comparison['n']} "
            f"(exact one-sided p = {comparison['sign_test_p']:.6f}; "
            f"losses {comparison['losses']}, ties {comparison['ties']})"
        )
        lines.append(
            f"mean delta: {comparison['mean_delta']:+.6f} nats "
            "(negative favours the adaptive rule)"
        )
        if comparison["paired_sd"] is None:
            lines.append("paired SD: n/a (fewer than two pairs)")
        else:
            lines.append(f"paired SD: {comparison['paired_sd']:.6f}")
    else:
        lines.append("no complete pairs yet")

    pool = summary.get("pool")
    if isinstance(pool, dict):
        lines.extend(["", "pool term (width_random - null; negative = the graft pays)"])
        if pool["status"] == "not_run":
            lines.append("pool: NOT RUN - no null-arm records")
        else:
            if pool["n"]:
                lines.append(
                    f"pool wins: {pool['wins']}/{pool['n']} "
                    f"(exact one-sided p = {pool['sign_test_p']:.6f}; "
                    f"losses {pool['losses']}, ties {pool['ties']})"
                )
                lines.append(f"pool mean delta: {pool['mean_delta']:+.6f} nats")
                if pool["paired_sd"] is not None:
                    lines.append(f"pool paired SD: {pool['paired_sd']:.6f}")
            lines.append(f"pool status: {pool['status']}")
            pool_protocol = pool.get("protocol")
            if isinstance(pool_protocol, dict) and not pool_protocol["ok"]:
                lines.append(
                    "pool protocol: FAILED - the null arm booked events outside "
                    "the registered schedule; no registered pool reading"
                )
                for arm, extra, missing, seeds in _deviation_groups(  # type: ignore[arg-type]
                    pool_protocol["deviations"]
                ):
                    lines.append(
                        f"  {arm}: extra {extra} missing {missing} on seeds {seeds}"
                    )
            if pool["missing_jobs"]:
                lines.append(f"pool missing: {', '.join(pool['missing_jobs'])}")

    lines.extend(["", "arm schedule (complete pairs only)", ""])
    lines.append("arm           accepted/events   mean accepted   final widths")
    schedule = summary["arm_schedule"]
    assert isinstance(schedule, dict)
    for arm in ARMS:
        entry = schedule[arm]
        assert isinstance(entry, dict)
        widths = ", ".join(
            f"{width} x{count}"
            for width, count in entry["final_widths"].items()  # type: ignore[union-attr]
        )
        mean = (
            "n/a"
            if entry["mean_accepted"] is None
            else f"{entry['mean_accepted']:.2f}"
        )
        lines.append(
            f"{arm:<13} {entry['accepted']:>8}/{entry['events']:<8} "
            f"{mean:>12}   {widths or 'n/a'}"
        )

    protocol = summary["protocol"]
    assert isinstance(protocol, dict)
    lines.append("")
    if protocol["ok"]:
        lines.append("protocol: the registered three-event schedule is present")
    else:
        lines.append("protocol: FAILED - events outside the registered schedule")
        for arm, extra, missing, seeds in _deviation_groups(protocol["deviations"]):  # type: ignore[arg-type]
            lines.append(
                f"  {arm}: extra {extra} missing {missing} on seeds {seeds}"
            )

    if summary["missing_jobs"]:
        lines.append("")
        lines.append(
            f"block: INCOMPLETE - missing {', '.join(summary['missing_jobs'])}"
        )
    lines.extend(["", f"STATUS: {summary['status']}", f"OUTCOME: {summary['outcome']}"])
    pool = summary.get("pool")
    if isinstance(pool, dict) and pool["status"] != "not_run":
        lines.append(f"POOL OUTCOME: {pool['outcome']}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=DEFAULT_RECORDS,
        help="directory of fetched width-pilot records",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the machine-readable reading here",
    )
    args = parser.parse_args()

    try:
        summary = summarize(load_records(args.records))
    except ReadingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    summary["records_directory"] = str(args.records)
    print(render(summary))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.out}")

    pool = summary.get("pool")
    if summary["status"] == "protocol_deviation" or (
        isinstance(pool, dict) and pool.get("status") == "protocol_deviation"
    ):
        return 3
    if summary["status"] == "complete":
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
