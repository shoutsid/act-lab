"""Check the code-LM leg's two trained arms against each other, from what they wrote.

The leg's claim is that **the two arms differ only in their objective**. The shell script's
flags are checked by test, but flags are not the run, and one day before the arms were
launched the run disagreed with them: a 2 GB checkpoint from a 20-step smoke run sat where
the control arm's checkpoint goes, so that arm would have started from another run's weights
at step 21 while the distilled arm started from its own initialisation at step 1 - both
reporting the same 11,250-step budget. Nothing in the numbers looks wrong when that happens.
It is the second defect of this class in the leg, after a cross-entropy arm that read the
full 162M-token stream while the distilled arm could only see the scored 15M.

So this reads the reports the arms actually produced and refuses when they disagree about
anything that is not the objective. It needs no GPU, no benchmark stage and no teacher, so it
can be run the moment the second arm finishes - hours before the table can be built.

Usage::

    PYTHONPATH=src python scripts/check_code_lm_arms.py
    PYTHONPATH=src python scripts/check_code_lm_arms.py --scale match --json arms.json

What it does not check: whether either arm is any good, or which arm won. Those are the
result, and a result is allowed to differ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "code-lm"

#: Top-level fields whose difference *is* the experiment, each with its reason. Anything
#: outside this mapping that differs between the two reports is a defect, not a result.
RESULT_FIELDS: dict[str, str] = {
    "mode": "the objective, by construction",
    "distill_weight": "0.5 in the distilled arm and 0.0 in the control",
    "samples": "three generations per arm, from models that are meant to differ",
    "validation": "the numbers each arm was run to produce",
}

#: Within ``training``, fields that are bookkeeping rather than experimental condition: they
#: differ when one arm was resumed and the other was not, which is faithful - the checkpoint
#: carries the weights, the optimiser moments and the draw sequence, so a resumed arm is the
#: same run as an uninterrupted one. They are printed rather than compared, because what the
#: comparison needs is the target (``steps``, ``tokens_budget``), which is compared.
BOOKKEEPING_FIELDS = ("optimizer_state", "resumed_from_step", "steps_trained", "tokens_seen")


def load(scale: str) -> dict[str, dict]:
    reports = {}
    for mode in ("distill", "ce"):
        path = ART / f"report-{scale}-{mode}.json"
        if not path.is_file():
            raise SystemExit(
                f"no report for the {mode} arm at {path}. The arms are trained by "
                "`bash scripts/run_code_lm_campaign.sh` (stage 2), and each writes its report "
                "when it finishes; a partial campaign reports itself here rather than as a "
                "shorter comparison."
            )
        report = json.loads(path.read_text())
        if report.get("leg") != "code-lm":
            raise SystemExit(f"{path} is not a report of this leg (leg {report.get('leg')!r})")
        reports[mode] = report
    return reports


def differences(left, right, here: str = "") -> list[str]:
    """Every leaf that differs, as ``path: left vs right``, outside the result fields."""

    if isinstance(left, dict) and isinstance(right, dict):
        found: list[str] = []
        for key in sorted(set(left) | set(right)):
            if not here and key in RESULT_FIELDS:
                continue
            # The objective is recorded in the run signature too, one level down; it is the
            # one thing that is *meant* to differ, wherever the report spells it.
            if key == "mode" and here:
                continue
            if here == "training" and key in BOOKKEEPING_FIELDS:
                continue
            found += differences(left.get(key), right.get(key), f"{here}.{key}" if here else key)
        return found
    if left != right:
        return [f"{here}: {left!r} vs {right!r}"]
    return []


def training_line(report: dict) -> str:
    training = report["training"]
    seen, budget = training.get("tokens_seen"), training.get("tokens_budget")
    if seen == budget:
        tokens = f"{seen:,} tokens"
    else:
        tokens = f"{seen:,} of a {budget:,} token budget after a resume"
    resumed = training.get("resumed_from_step", 0)
    return (
        f"{training['steps']:,} steps x {training['batch_size']} x "
        f"{training['batch_tokens'] // training['batch_size']} = {tokens}; seed "
        f"{training['seed']}; {training['optimizer']}; started at "
        f"{'step 1' if not resumed else f'step {resumed + 1}'} "
        f"({training.get('optimizer_state', 'state not recorded')})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", default="match", help="student scale the arms were trained at")
    parser.add_argument("--json", type=Path, default=None, help="write the comparison here")
    args = parser.parse_args()

    reports = load(args.scale)
    distill, control = reports["distill"], reports["ce"]
    mismatched = differences(distill, control)

    print(f"two arms at scale {args.scale}, from {ART}")
    for mode, report in reports.items():
        print(f"  {mode:<8} {training_line(report)}")
    checked = sorted(key for key in set(distill) | set(control) if key not in RESULT_FIELDS)
    print(f"  agreed on {', '.join(checked)} - everything except the objective")
    print()

    for mode, report in reports.items():
        validation = report["validation"]
        print(
            f"  {mode:<8} {validation['bits_per_byte']:.4f} bits/byte, "
            f"val loss {validation['final_val_loss']:.4f}"
        )
    print(f"  measured on {distill['validation']['bits_per_byte_measured_on']}")
    gain = control["validation"]["bits_per_byte"] - distill["validation"]["bits_per_byte"]
    print(
        f"\n  gain {gain:+.4f} bits/byte to the distilled arm (this leg's own measure of the "
        "held-out stream, not the table's)"
    )

    if mismatched:
        print("\nThese two arms were not one experiment:")
        for item in mismatched:
            print(f"  - {item}")
    else:
        print("\nThe two arms differ in their objective and nothing else.")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "scale": args.scale,
                    "differences": mismatched,
                    "agreed": checked,
                    "result_fields": RESULT_FIELDS,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"written {args.json}")
    return 1 if mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
