"""Assemble the code-LM comparison table from the artifacts, and refuse a partial one.

Three scripts write the numbers this leg is judged on - the campaign
(``run_code_lm_campaign.sh``), the Kaggle kernel (``push_code_bench.py``), and
``code_bench.py`` underneath both - and a table assembled by hand from three places
is a table that can quietly omit a row. The rows are not interchangeable: the
comparison is only meaningful with the teacher (what the student is condensing), the
published baseline (the same size class), the distilled arm, and the cross-entropy
control it has to beat. Drop one and the table still renders, which is exactly the
failure this exists to prevent.

Three contracts are enforced rather than documented:

* **one ``bytes_limit`` across every row.** Bits per byte is only comparable on the
  same held-out text, and the limit is a compute decision that varies by machine - the
  local card scores a 4 MB prefix, and a row measured at 28.7 MB would silently be a
  different experiment.
* **HumanEval rows carry their device.** Pass@1 comes from sampled decoding, which is
  not bit-reproducible across accelerators, so a table mixing devices has a difference
  in it that no row explains. Reported, not hidden.
* **the distillation claim is tested, not read off four decimals.** The headline of this
  leg is that the distilled arm beats the cross-entropy control, and the two arms are
  scored on identical windows with one tokenizer - which is a paired experiment, and a
  paired one over thousands of windows resolves effects a four-decimal aggregate cannot.
  ``paired_contrast`` runs that test and the table refuses to render without it; the
  absolute difference and its confidence interval are the result, the p-value only says
  whether it needs a caveat.

    PYTHONPATH=src python scripts/code_lm_table.py
    PYTHONPATH=src python scripts/code_lm_table.py --json artifacts/code-lm/bench/table.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.stats

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "code-lm"
BENCH = ART / "bench"

#: Below this many windows a paired test cannot separate an effect of a few
#: thousandths of a bit per byte from noise, so a short scan is refused rather than
#: dressed in a p-value that reads as proof.
MIN_PAIRED_WINDOWS = 30

#: Sign-flip settings, matching ``act.search.architecture_report``: enumerate the
#: assignments while 2**n fits, otherwise sample with a fixed seed so the printed p
#: is reproducible rather than different on every run.
PERMUTATION_CAP = 1_000_000
PERMUTATION_DRAWS = 200_000
PERMUTATION_SEED = 0

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


def window_losses(tag: str, report: dict) -> list[float]:
    """The per-window negative log-likelihoods a report carries, or a refusal.

    Three ways a report can be unusable for pairing, all of which still leave a
    plausible-looking bits-per-byte number behind, and all of which are errors rather
    than gaps: no per-window losses at all (a report written before the scan existed
    holds only a total), a window count that disagrees with the list, and a list whose
    sum is not the reported total (the two halves came from different runs).
    """
    block = report.get("bytes") or {}
    values = block.get("window_nlls")
    if not values:
        raise SystemExit(
            f"{tag} carries no per-window losses, so the two student arms cannot be paired "
            "window by window: a report written before the per-window scan is a total, and "
            "two totals are not a paired test. Re-score it with scripts/code_bench.py."
        )
    if block.get("windows") != len(values):
        raise SystemExit(
            f"{tag} advertises {block.get('windows')} windows but carries {len(values)}; "
            "the report disagrees with itself and its windows cannot be matched up."
        )
    total = float(block.get("nats", 0.0))
    if abs(sum(values) - total) > 1e-4 * max(1.0, abs(total)):
        raise SystemExit(
            f"{tag}'s per-window losses sum to {sum(values):.4f} against a reported total of "
            f"{total:.4f}; the list and the total did not come from one scan."
        )
    return [float(value) for value in values]


def report_tokenizer(tag: str, report: dict) -> str:
    """The tokenization the ids came from. Pairing two models assumes it is the same one."""
    tokenizer = (report.get("bytes") or {}).get("tokenizer")
    if not tokenizer:
        raise SystemExit(
            f"{tag} does not record which tokenizer produced its ids, so there is no way to "
            "show the two arms saw the same token boundaries. Re-score it with "
            "scripts/code_bench.py, which records the tokenizer identity."
        )
    return str(tokenizer)


def sign_flip_p(differences: list[float], draws: int = PERMUTATION_DRAWS) -> dict:
    """Two-sided sign-flip permutation p-value for a paired mean difference.

    The paired null is that each window's difference could equally well have come out
    the other way, and the permutation test says exactly that without assuming the
    differences are normal - which the paired t below does assume, and which a few
    thousand losses far in the tail would break. Exact while the ``2 ** n`` sign
    assignments fit under the cap; Monte-Carlo with a fixed seed otherwise, in batches,
    because 2,500 windows is far past enumeration and one draw at a time in Python
    would cost minutes.

    Windows cut from the same file are not independent, so this p-value is optimistic
    by an amount nothing here can measure; it is reported beside the interval rather
    than instead of it, and the printed table says so.
    """
    values = np.asarray(differences, dtype=np.float64)
    count = len(values)
    if count == 0:
        raise SystemExit("a paired test needs at least one window")
    observed = float(values.mean())
    resolutions = 2 ** count

    def at_least(flipped: np.ndarray) -> int:
        # 1e-12 absorbs the last bit of the arithmetic on either side of the boundary.
        return int(np.count_nonzero(np.abs(flipped) >= abs(observed) - 1e-12))

    if resolutions <= PERMUTATION_CAP:
        bits = (np.arange(resolutions)[:, None] >> np.arange(count)) & 1
        flipped = (2 * bits - 1) @ values / count
        return {
            "p": at_least(flipped) / resolutions,
            "method": "exact",
            "assignments": resolutions,
            "min_achievable_p": 2 / resolutions,
        }

    generator = np.random.default_rng(PERMUTATION_SEED)
    # Batched and int8: a draw is one sign per window, and 2,500 windows times 200,000
    # draws is half a billion multiplications - cheap as one matrix product, minutes as
    # a Python loop.
    batch = max(1, min(draws, 2048))
    hits = 0
    done = 0
    while done < draws:
        size = min(batch, draws - done)
        signs = (generator.integers(0, 2, size=(size, count)) * 2 - 1).astype(np.int8)
        hits += at_least(signs @ values / count)
        done += size
    return {
        "p": (hits + 1) / (draws + 1),
        "method": "monte-carlo",
        "assignments": draws,
        "min_achievable_p": 2 / (draws + 1),
        "seed": PERMUTATION_SEED,
    }


def paired_contrast(
    resolved: dict[str, tuple[str, dict]], treated: str = "ours-distill", control: str = "ours-ce"
) -> dict:
    """Test the two student arms window by window, on the windows they share.

    This is the leg's headline turned into something falsifiable. Both students share
    one tokenizer and are scored on the same held-out text with the same window
    boundaries, so window *i* of one is the same text as window *i* of the other, and
    the per-window difference removes the difficulty of the text itself - which is most
    of the variance in a loss curve. A total to four decimals cannot distinguish
    distillation from numerical dust; a mean difference with an interval and a sign
    count can.

    Direction: the gain is ``control - treated``, so positive means the treated arm
    scored lower (better), the same sign as ``distillation_gain_bits_per_byte``.
    """
    tag_treated, report_treated = resolved[treated]
    tag_control, report_control = resolved[control]

    tokenizers = {tag_treated: report_tokenizer(tag_treated, report_treated),
                  tag_control: report_tokenizer(tag_control, report_control)}
    if len(set(tokenizers.values())) > 1:
        raise SystemExit(
            "the two arms were tokenized differently "
            f"({', '.join(f'{tag}={value}' for tag, value in sorted(tokenizers.items()))}), "
            "so their windows cover different pieces of text and pairing them would "
            "compare window 7 of one against window 7 of another."
        )

    treated_values = window_losses(tag_treated, report_treated)
    control_values = window_losses(tag_control, report_control)
    if len(treated_values) != len(control_values):
        raise SystemExit(
            f"{tag_treated} was scored on {len(treated_values):,} windows and {tag_control} on "
            f"{len(control_values):,}; a paired test needs the same windows on both sides."
        )
    windows = len(treated_values)
    if windows < MIN_PAIRED_WINDOWS:
        raise SystemExit(
            f"only {windows} windows, fewer than the {MIN_PAIRED_WINDOWS} a paired test needs to "
            "say anything; score more held-out bytes before comparing the arms."
        )

    bytes_each = {
        tag: int((report.get("bytes") or {}).get("held_out_bytes") or 0)
        for tag, report in ((tag_treated, report_treated), (tag_control, report_control))
    }
    if len(set(bytes_each.values())) != 1 or not next(iter(bytes_each.values())):
        raise SystemExit(
            "the two arms report different held-out byte counts "
            f"({bytes_each}); their per-window losses are not the same measurement."
        )
    bytes_per_window = next(iter(bytes_each.values())) / windows

    gains = np.asarray(control_values) - np.asarray(treated_values)
    mean = float(gains.mean())
    deviation = float(gains.std(ddof=1))
    error = deviation / math.sqrt(windows)
    if error > 0:
        t_value = mean / error
        p_t = float(2 * scipy.stats.t.sf(abs(t_value), df=windows - 1))
        low, high = (float(bound) for bound in scipy.stats.t.interval(0.95, df=windows - 1, loc=mean, scale=error))
    else:
        # Identical losses on every window: the difference is exactly zero, and there is
        # no interval to compute rather than an interval of zero width to invent.
        t_value, p_t, low, high = 0.0, 1.0, mean, mean

    permutation = sign_flip_p(list(gains))
    per_byte = 1.0 / math.log(2) / bytes_per_window
    return {
        "treated": tag_treated,
        "control": tag_control,
        "windows": windows,
        "tokenizer": tokenizers[tag_treated],
        "bytes_per_window": bytes_per_window,
        "gain_nats_per_window": mean,
        "gain_bits_per_byte": mean * per_byte,
        "ci95_nats_per_window": [low, high],
        "ci95_bits_per_byte": [low * per_byte, high * per_byte],
        "standard_error_nats_per_window": error,
        "t": t_value,
        "p_paired_t": p_t,
        "p_sign_flip": permutation["p"],
        "sign_flip_method": permutation["method"],
        "sign_flip_assignments": permutation["assignments"],
        "min_achievable_p": permutation["min_achievable_p"],
        "windows_favouring_treated": int(np.count_nonzero(gains > 0)),
        "windows_favouring_control": int(np.count_nonzero(gains < 0)),
        "windows_tied": int(np.count_nonzero(gains == 0)),
        # The one-line verdict a reader needs, computed here so it is not retyped in prose.
        "resolved_at": 0.01 if permutation["p"] < 0.01 else None,
    }


def format_paired(paired: dict) -> str:
    """The paired result as the table prints it, numbers and direction together."""
    gain = paired["gain_bits_per_byte"]
    low, high = paired["ci95_bits_per_byte"]
    p = paired["p_sign_flip"]
    p_text = f"p<={paired['min_achievable_p']:.1e}" if p <= paired["min_achievable_p"] else f"p={p:.3g}"
    direction = "distilled arm better" if gain > 0 else "control better"
    return (
        f"distillation gain (paired, {paired['windows']:,} windows): "
        f"{gain:+.4f} bits/byte, 95% CI [{low:+.4f}, {high:+.4f}], {direction}; "
        f"{paired['windows_favouring_treated']:,}/{paired['windows']:,} windows favour the "
        f"distilled arm; sign-flip {p_text} ({paired['sign_flip_method']}, "
        f"{paired['sign_flip_assignments']:,} assignments), paired t p={paired['p_paired_t']:.3g}"
    )


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
    # Not optional and not best-effort: the whole point of scoring both arms through one
    # script on one text is that their difference can be tested, so a report that cannot
    # be paired stops the table instead of being summarised into two decimals and a shrug.
    paired = paired_contrast(resolved)
    return {"rows": rows, "derived": derived, "paired": paired}


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
    print()
    print(format_paired(table["paired"]))
    print(
        "note: the sign-flip p above treats the windows as independent, and windows cut from "
        "one file are not; read the interval as the result and the p as a lower bound on how "
        "surprising it is."
    )
    if table["paired"]["windows"] < 200:
        print(f"note: {table['paired']['windows']} windows is a short scan; the interval is wide for a reason.")

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
        args.json.write_text(
            json.dumps({**table, "bytes_limit": limit}, indent=2, sort_keys=True, default=float) + "\n"
        )
        print(f"\nwritten {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
