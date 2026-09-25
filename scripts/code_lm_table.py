"""Assemble the code-LM comparison table from the artifacts, and refuse a partial one.

Three scripts write the numbers this leg is judged on - the campaign
(``run_code_lm_campaign.sh``), the Kaggle kernel (``push_code_bench.py``), and
``code_bench.py`` underneath both - and a table assembled by hand from three places
is a table that can quietly omit a row. The rows are not interchangeable: the
comparison is only meaningful with the teacher (what the student is condensing), the
published baseline (the same size class), the distilled arm, and the cross-entropy
control it has to beat. Drop one and the table still renders, which is exactly the
failure this exists to prevent.

Two contracts are enforced rather than documented:

* **one ``bytes_limit`` across every row.** Bits per byte is only comparable on the
  same held-out text, and the limit is a compute decision that varies by machine - the
  local card scores a 4 MB prefix, and a row measured at 28.7 MB would silently be a
  different experiment.
* **HumanEval rows carry their device.** Pass@1 comes from sampled decoding, which is
  not bit-reproducible across accelerators, so a table mixing devices has a difference
  in it that no row explains. Reported, not hidden.

    PYTHONPATH=src python scripts/code_lm_table.py
    PYTHONPATH=src python scripts/code_lm_table.py --json artifacts/code-lm/bench/table.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "code-lm"
BENCH = ART / "bench"

#: The rows the comparison is made of, and what each one is for. A missing role is not
#: printed as a blank - it is an error, because every claim here is relative and the
#: relative claim is not evaluable without both sides of it.
ROLES: dict[str, str] = {
    "teacher": "the 1.3B model being condensed - the target",
    "baseline": "the published 164M model of the student's size class",
    "ours-distill": "the student distilled against the teacher's distribution",
    "ours-ce": "the student trained the same way without distillation - the control",
}


def load_reports(bench: Path) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for path in sorted(bench.glob("*.json")):
        try:
            reports[path.stem] = json.loads(path.read_text())
        except (ValueError, OSError) as error:
            raise SystemExit(f"{path} is not readable JSON: {error}")
    if not reports:
        raise SystemExit(f"no benchmark reports in {bench}; run the campaign or fetch the kernel first")
    return reports


def resolve_roles(reports: dict[str, dict], scale: str) -> dict[str, tuple[str, dict]]:
    """Map each role to its report, allowing the student tags to carry a scale prefix.

    The campaign tags its arms ``ours-<scale>-distill`` while the Kaggle kernel writes
    bare ``teacher`` / ``baseline``, so matching is on prefix for the students and exact
    for the two fixed models - and an ambiguous match is refused rather than picked.
    """
    resolved: dict[str, tuple[str, dict]] = {}
    for role in ROLES:
        # The campaign tags its arms ``ours-<scale>-distill`` while the Kaggle kernel
        # writes bare ``teacher`` / ``baseline``, so the expected tag is derived from the
        # role and the scale rather than assumed to equal the role.
        expected = role if not role.startswith("ours-") else f"ours-{scale}-{role.split('-', 1)[1]}"
        for tag in (role, expected):
            if tag in reports:
                resolved[role] = (tag, reports[tag])
                break
        else:
            raise SystemExit(
                f"no report for role {role!r} ({ROLES[role]}).\n"
                f"  Looked for a report tagged {expected!r}; have: {sorted(reports)}\n"
                f"  The comparison needs every role; a table missing one reads as a result "
                f"rather than as an unfinished run."
            )
    return resolved


def check_comparable(resolved: dict[str, tuple[str, dict]]) -> int:
    """Every row must be scored on the same held-out text."""
    limits = {}
    for role, (tag, report) in resolved.items():
        block = report.get("bytes")
        if not block:
            raise SystemExit(f"{tag} has no 'bytes' block; it was scored for a different suite")
        limits[role] = report.get("bytes_limit") or block.get("bytes_limit")
    distinct = {value for value in limits.values()}
    if len(distinct) > 1:
        detail = ", ".join(f"{role}={limits[role]}" for role in sorted(limits))
        raise SystemExit(
            f"rows were scored on different held-out text ({detail}); bits per byte is only "
            "comparable at one bytes_limit. Re-score the odd rows with matching flags."
        )
    return int(next(iter(distinct)) or 0)


def format_row(role: str, tag: str, report: dict) -> str:
    block = report["bytes"]
    total = block.get("parameters")
    active = block.get("active_parameters")
    total_text = f"{total / 1e6:.1f}M" if total else "-"
    active_text = f"{active / 1e6:.1f}M" if active else ("dense" if total else "-")
    architecture = ", ".join(block.get("architecture") or []) or block.get("kind", "-")
    # Four decimals is the precision the measurement supports; the third and fourth can
    # move with the accelerator's matmul kernels, so the comparison is read at two.
    bpb = f"{block['bits_per_byte']:.4f}"
    human = report.get("humaneval")
    if human:
        pass_at_1 = f"{human['pass_at_1']:.3f}"
    else:
        pass_at_1 = "not run"
    return f"| {role} | {architecture} | {total_text} | {active_text} | {bpb} | {pass_at_1} | {tag} |"


def build_table(resolved: dict[str, tuple[str, dict]], scale: str) -> dict:
    # Ordered by role, because ``resolved`` is keyed by role and the row carries the tag
    # separately. Keying the order by tag instead printed only the two roles whose tag
    # and name coincide, which is how a two-row table of a four-row comparison shipped
    # once: the missing rows were not blank, they were absent.
    order = [role for role in ROLES if role in resolved]
    rows = []
    for role in order:
        tag, report = resolved[role]
        block = report["bytes"]
        rows.append(
            {
                "role": role,
                "tag": tag,
                "bits_per_byte": block["bits_per_byte"],
                "parameters": block.get("parameters"),
                "active_parameters": block.get("active_parameters"),
                "architecture": block.get("architecture"),
                "pass_at_1": (report.get("humaneval") or {}).get("pass_at_1"),
                "device": report.get("device"),
            }
        )
    by_role = {row["role"]: row for row in rows}
    derived: dict[str, float | None] = {}

    def bpb(role: str) -> float | None:
        row = by_role.get(role)
        return None if row is None else float(row["bits_per_byte"])

    if bpb("ours-distill") is not None and bpb("ours-ce") is not None:
        # Positive means the distilled arm is closer to zero, i.e. distillation helped.
        derived["distillation_gain_bits_per_byte"] = bpb("ours-ce") - bpb("ours-distill")
    if bpb("teacher") is not None and bpb("ours-distill") is not None:
        derived["student_gap_to_teacher_bits_per_byte"] = bpb("ours-distill") - bpb("teacher")
    if bpb("baseline") is not None and bpb("ours-distill") is not None:
        derived["student_gap_to_baseline_bits_per_byte"] = bpb("ours-distill") - bpb("baseline")
    if bpb("ours-distill") is not None and bpb("ours-ce") is not None:
        # Ratio to the control is the one number that survives a reader disagreeing with
        # the absolute scale: it says what distillation bought at matched budget and data.
        gain = derived["distillation_gain_bits_per_byte"]
        derived["distillation_gain_percent_of_control"] = (
            100.0 * gain / bpb("ours-ce") if gain is not None and bpb("ours-ce") else None
        )
    return {"rows": rows, "derived": derived}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", type=Path, default=BENCH)
    parser.add_argument("--scale", default="match")
    parser.add_argument("--json", type=Path, default=None, help="also write the table as JSON")
    args = parser.parse_args()

    reports = load_reports(args.bench)
    resolved = resolve_roles(reports, args.scale)
    limit = check_comparable(resolved)
    table = build_table(resolved, args.scale)

    print(f"Held-out Python: first {limit:,} bytes, scored identically for every row.")
    print()
    print("| role | architecture | total params | active params | bits/byte | HumanEval pass@1 | tag |")
    print("|---|---|---|---|---|---|---|")
    for row in table["rows"]:
        print(format_row(row["role"], row["tag"], reports[row["tag"]]))
    print()
    for name, value in table["derived"].items():
        print(f"{name.replace('_', ' ')}: {value:+.4f}" if isinstance(value, float) else f"{name}: {value}")

    devices = sorted({row["device"] for row in table["rows"] if row["device"]})
    if len(devices) > 1:
        print(
            f"\nnote: rows were measured on {len(devices)} devices ({', '.join(devices)}). "
            "HumanEval pass@1 comes from sampled decoding, which is not bit-reproducible "
            "across accelerators; the bits-per-byte column is affected only in its last digits."
        )
    if any(row["pass_at_1"] is None for row in table["rows"]):
        print("\nnote: at least one row has no HumanEval number; pass@1 is not being compared.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({**table, "bytes_limit": limit}, indent=2, sort_keys=True) + "\n")
        print(f"\nwritten {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
