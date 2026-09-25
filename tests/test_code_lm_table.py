"""The comparison table refuses to be partial, and its arithmetic is pinned.

Every claim in this leg is relative - the student against the teacher it is
condensed from, against a published model of its own size class, and against a
cross-entropy control trained on the same data in the same order. A table that
renders while one of those rows is absent does not read as an unfinished run; it
reads as a result. That is the failure these tests exist for, and it is why a
missing role raises instead of printing a blank.

The other contract is the held-out text. Bits per byte is only comparable at one
``bytes_limit``, and the limit is a per-machine decision (4 MB of prefix on the
local card, because scoring the whole 28.7 MB set costs the 1.3B teacher hours).
Two rows measured at different limits are two experiments.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import code_lm_table  # noqa: E402

STUDENT_TOKENIZER = "deepseek-ai/deepseek-coder-1.3b-base@ef48ebdc8546c2d8"


def bench_report(
    bits_per_byte: float,
    *,
    parameters: int = 1_000_000,
    active: int | None = None,
    bytes_limit: int = 4_000_000,
    device: str = "cuda",
    device_name: str | None = None,
    pass_at_1: float | None = None,
    architecture: tuple[str, ...] = ("LlamaForCausalLM",),
    windows: int = 64,
    tokenizer: str | None = STUDENT_TOKENIZER,
    window_nlls: list[float] | None = None,
) -> dict:
    """A report shaped exactly like the one ``scripts/code_bench.py`` writes.

    The nats total is derived from the bits per byte rather than passed beside it, the
    per-window losses sum to that total, and the window count matches the list - so a
    test that breaks one of those has to say so explicitly instead of tripping over a
    fixture that was never internally consistent to begin with. Passing ``window_nlls``
    replaces the flat per-window losses with an explicit shape, which is how the tests
    express a noisy gain that is not significant.
    """
    if window_nlls is None:
        nats = bits_per_byte * math.log(2) * bytes_limit
        window_nlls = [nats / windows] * windows
    else:
        windows = len(window_nlls)
    nats = sum(window_nlls)
    report = {
        "model": "x",
        "suite": ["bytes"],
        "device": device,
        "bytes_limit": bytes_limit,
        "bytes": {
            "model": "x",
            "kind": "hf",
            "architecture": list(architecture),
            "parameters": parameters,
            "tokens": 10,
            "nats": nats,
            "held_out_bytes": bytes_limit,
            "bits_per_byte": bits_per_byte,
            "loss_per_token": 0.1,
            "bytes_limit": bytes_limit,
            "windows": windows,
            "window_nlls": list(window_nlls),
        },
    }
    if device_name is not None:
        report["device_name"] = device_name
    if tokenizer is not None:
        report["bytes"]["tokenizer"] = tokenizer
    if active is not None:
        report["bytes"]["active_parameters"] = active
    if pass_at_1 is not None:
        report["humaneval"] = {"problems": 10, "samples_per_problem": 1, "passed": 1, "pass_at_1": pass_at_1}
    return report


def write_bench(directory: Path, **rows: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for tag, report in rows.items():
        (directory / f"{tag}.json").write_text(json.dumps(report))
    return directory


def complete(directory: Path, *, scale: str = "match", limit: int = 4_000_000, **overrides) -> Path:
    rows = {
        "teacher": bench_report(0.5000, parameters=1_346_000_000, bytes_limit=limit),
        "baseline": bench_report(1.2000, parameters=164_100_000, pass_at_1=0.0784, bytes_limit=limit),
        f"ours-{scale}-distill": bench_report(
            0.7500, parameters=166_423_296, active=81_488_640, bytes_limit=limit
        ),
        f"ours-{scale}-ce": bench_report(0.9000, parameters=166_423_296, active=81_488_640, bytes_limit=limit),
    }
    rows.update(overrides)
    return write_bench(directory, **rows)


def test_a_complete_set_resolves_and_the_deltas_are_the_measured_differences(tmp_path: Path) -> None:
    reports = code_lm_table.load_reports(complete(tmp_path))
    resolved = code_lm_table.resolve_roles(reports, "match")
    table = code_lm_table.build_table(resolved, "match")
    derived = table["derived"]

    # Distillation "helped" means the distilled arm moved toward zero bits per byte.
    assert derived["distillation_gain_bits_per_byte"] == pytest.approx(0.9000 - 0.7500)
    assert derived["student_gap_to_teacher_bits_per_byte"] == pytest.approx(0.7500 - 0.5000)
    assert derived["student_gap_to_baseline_bits_per_byte"] == pytest.approx(0.7500 - 1.2000)
    assert derived["distillation_gain_percent_of_control"] == pytest.approx(100 * 0.15 / 0.9)


def test_a_gain_can_be_negative_and_is_reported_as_such(tmp_path: Path) -> None:
    """Distillation losing is a result, not an error, and must not be clamped."""
    reports = code_lm_table.load_reports(
        complete(tmp_path, **{"ours-match-distill": bench_report(0.9500)})
    )
    table = code_lm_table.build_table(code_lm_table.resolve_roles(reports, "match"), "match")
    assert table["derived"]["distillation_gain_bits_per_byte"] == pytest.approx(0.9 - 0.95)
    assert table["derived"]["distillation_gain_percent_of_control"] == pytest.approx(
        100 * (0.9 - 0.95) / 0.9
    )


@pytest.mark.parametrize(
    "role,missing",
    [("teacher", "teacher"), ("baseline", "baseline"),
     ("ours-distill", "ours-match-distill"), ("ours-ce", "ours-match-ce")],
)
def test_a_missing_role_is_an_error_naming_what_is_missing(tmp_path: Path, role: str, missing: str) -> None:
    """Each row is load-bearing; the message has to say which one and what it was for."""
    directory = complete(tmp_path)
    (directory / f"{missing}.json").unlink()
    reports = code_lm_table.load_reports(directory)
    with pytest.raises(SystemExit) as error:
        code_lm_table.resolve_roles(reports, "match")
    message = str(error.value)
    assert missing in message, "the operator needs the tag to produce, not the role's shorthand"
    assert "unfinished run" in message
    assert code_lm_table.ROLES[role] in message


# --- the paired test on the two student arms ---------------------------------


def test_the_paired_test_recovers_the_aggregate_difference_and_resolves_it(tmp_path: Path) -> None:
    """The same 0.15 bits/byte the totals give, now with an interval and a sign count.

    Both arms share 64 windows of identical text, so the paired gain has to come back
    as exactly the difference of the two aggregates - the value of the test is the
    interval and the p-value, not a different number.
    """
    reports = code_lm_table.load_reports(complete(tmp_path))
    table = code_lm_table.build_table(code_lm_table.resolve_roles(reports, "match"), "match")
    paired = table["paired"]

    assert paired["windows"] == 64
    assert paired["gain_bits_per_byte"] == pytest.approx(0.9000 - 0.7500)
    assert paired["gain_nats_per_window"] == pytest.approx(
        table["derived"]["distillation_gain_bits_per_byte"] * math.log(2) * 4_000_000 / 64
    )
    low, high = paired["ci95_bits_per_byte"]
    assert low <= paired["gain_bits_per_byte"] <= high
    assert paired["windows_favouring_treated"] == 64
    assert paired["windows_favouring_control"] == 0
    assert paired["p_sign_flip"] == pytest.approx(1 / 200_001)  # no sign flip got close
    assert paired["sign_flip_method"] == "monte-carlo"
    assert paired["resolved_at"] == 0.01
    assert paired["tokenizer"] == STUDENT_TOKENIZER


def test_sign_flip_is_exact_while_the_assignments_fit() -> None:
    """Few enough windows to enumerate means the p is exact, not sampled.

    Two zero differences and two positive ones, so the observed mean is 1. A sign
    vector reaches at least that only when the two positives keep their common sign -
    2 of the 4 ways to sign them - while the two zeros can be signed either way without
    moving the mean at all, which is 4 more free assignments each. 8 of 16.
    """
    result = code_lm_table.sign_flip_p([0.0, 0.0, 2.0, 2.0])
    assert result["method"] == "exact"
    assert result["assignments"] == 16
    assert result["p"] == pytest.approx(8 / 16)
    assert result["min_achievable_p"] == pytest.approx(2 / 16)


def test_a_difference_that_never_moves_off_zero_has_p_one() -> None:
    """Four ties: every sign vector reproduces the observed mean of zero."""
    result = code_lm_table.sign_flip_p([0.0, 0.0, 0.0, 0.0])
    assert result["p"] == 1.0


def test_the_sign_flip_test_needs_more_than_one_window() -> None:
    with pytest.raises(SystemExit, match="at least one"):
        code_lm_table.sign_flip_p([])


def test_arms_tokenized_differently_are_refused(tmp_path: Path) -> None:
    """Window 7 of one model must be window 7's text of the other, or the pairs are fake."""
    directory = complete(tmp_path, **{"ours-match-ce": bench_report(0.9, tokenizer="bigcode/tiny_starcoder_py")})
    resolved = code_lm_table.resolve_roles(code_lm_table.load_reports(directory), "match")
    with pytest.raises(SystemExit, match="tokenized differently"):
        code_lm_table.build_table(resolved, "match")


def test_an_unknown_tokenizer_identity_is_refused_rather_than_assumed(tmp_path: Path) -> None:
    directory = complete(tmp_path, **{"ours-match-ce": bench_report(0.9, tokenizer=None)})
    resolved = code_lm_table.resolve_roles(code_lm_table.load_reports(directory), "match")
    with pytest.raises(SystemExit, match="which tokenizer"):
        code_lm_table.build_table(resolved, "match")


def test_a_report_with_no_per_window_losses_is_refused(tmp_path: Path) -> None:
    """A pre-refactor report is a total, and two totals are not a paired test."""
    stale = bench_report(0.9000)
    stale["bytes"].pop("window_nlls")
    stale["bytes"].pop("windows")
    directory = complete(tmp_path, **{"ours-match-ce": stale})
    resolved = code_lm_table.resolve_roles(code_lm_table.load_reports(directory), "match")
    with pytest.raises(SystemExit, match="per-window losses"):
        code_lm_table.build_table(resolved, "match")


def test_a_window_list_that_disagrees_with_its_own_total_is_refused(tmp_path: Path) -> None:
    """The list and the total came from one scan or they did not; this one did not."""
    inconsistent = bench_report(0.9000)
    inconsistent["bytes"]["nats"] *= 1.5
    directory = complete(tmp_path, **{"ours-match-ce": inconsistent})
    resolved = code_lm_table.resolve_roles(code_lm_table.load_reports(directory), "match")
    with pytest.raises(SystemExit, match="did not come from one scan"):
        code_lm_table.build_table(resolved, "match")


def test_a_window_count_that_disagrees_with_the_list_is_refused(tmp_path: Path) -> None:
    inconsistent = bench_report(0.9000)
    inconsistent["bytes"]["windows"] = 7
    directory = complete(tmp_path, **{"ours-match-ce": inconsistent})
    resolved = code_lm_table.resolve_roles(code_lm_table.load_reports(directory), "match")
    with pytest.raises(SystemExit, match="disagrees with itself"):
        code_lm_table.build_table(resolved, "match")


def test_arms_scored_on_different_windows_are_refused(tmp_path: Path) -> None:
    directory = complete(tmp_path, **{"ours-match-ce": bench_report(0.9000, windows=63)})
    resolved = code_lm_table.resolve_roles(code_lm_table.load_reports(directory), "match")
    with pytest.raises(SystemExit, match="same windows on both sides"):
        code_lm_table.build_table(resolved, "match")


def test_a_scan_too_short_to_test_is_refused(tmp_path: Path) -> None:
    """Ten windows will not separate thousandths of a bit per byte, so it is not tried."""
    directory = complete(
        tmp_path,
        **{
            "ours-match-distill": bench_report(0.7500, windows=10),
            "ours-match-ce": bench_report(0.9000, windows=10),
        },
    )
    resolved = code_lm_table.resolve_roles(code_lm_table.load_reports(directory), "match")
    with pytest.raises(SystemExit, match="fewer than"):
        code_lm_table.build_table(resolved, "match")


def test_a_noisy_gain_is_not_called_significant_by_accident(tmp_path: Path) -> None:
    """A control that loses on about half the windows by a margin that varies.

    The mean gain here is a hundredth of a nat per window against a spread ten times
    that: the test has to weigh the size of the gain against the variation in it, so
    the interval must straddle zero and the p must not be small - the opposite of the
    clean 64/64 case above, on the same amount of text. A test that reported a small p
    for both would be reporting the pairing, not the effect.
    """
    distilled = [1.0] * 64
    control = [1.0 - 0.10 if index < 33 else 1.0 + 0.09 for index in range(64)]
    directory = complete(
        tmp_path,
        **{
            "ours-match-distill": bench_report(0.7500, window_nlls=distilled),
            "ours-match-ce": bench_report(0.7505, window_nlls=control),
        },
    )
    resolved = code_lm_table.resolve_roles(code_lm_table.load_reports(directory), "match")
    paired = code_lm_table.build_table(resolved, "match")["paired"]
    low, high = paired["ci95_bits_per_byte"]
    assert paired["gain_nats_per_window"] < 0  # 33 windows lost, 31 won, by less each time
    assert low < 0 < high
    assert paired["p_sign_flip"] > 0.05
    assert paired["windows_favouring_control"] == 33
    assert paired["windows_favouring_treated"] == 31
    assert paired["resolved_at"] is None


def test_the_printed_table_carries_the_paired_verdict_and_its_caveat(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_cli(monkeypatch, complete(tmp_path))
    out = capsys.readouterr().out
    assert "distillation gain (paired, 64 windows)" in out
    assert "64/64 windows favour the distilled arm" in out
    assert "bits/byte, 95% CI" in out
    # The p is a lower bound on significance and the line says as much.
    assert "read the interval as the result" in out
    assert "cut from one file are not" in out
    assert "64 windows is a short scan" in out


def test_the_json_form_carries_the_paired_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "table.json"
    run_cli(monkeypatch, complete(tmp_path / "bench"), "--json", str(out))
    paired = json.loads(out.read_text())["paired"]
    assert paired["windows"] == 64
    assert paired["gain_bits_per_byte"] == pytest.approx(0.15)
    assert len(paired["ci95_bits_per_byte"]) == 2


def test_rows_measured_at_different_limits_are_refused(tmp_path: Path) -> None:
    directory = complete(tmp_path, **{"baseline": bench_report(1.2, bytes_limit=28_707_445)})
    reports = code_lm_table.load_reports(directory)
    resolved = code_lm_table.resolve_roles(reports, "match")
    with pytest.raises(SystemExit, match="bytes_limit"):
        code_lm_table.check_comparable(resolved)


def test_the_limit_is_echoed_from_the_rows_rather_than_assumed(tmp_path: Path) -> None:
    """Any shared limit is fine - the constant is not pinned, the agreement is."""
    reports = code_lm_table.load_reports(complete(tmp_path, limit=1_000_000))
    resolved = code_lm_table.resolve_roles(reports, "match")
    assert code_lm_table.check_comparable(resolved) == 1_000_000


def test_a_row_scored_for_another_suite_is_refused(tmp_path: Path) -> None:
    directory = complete(tmp_path)
    broken = json.loads((directory / "teacher.json").read_text())
    del broken["bytes"]
    (directory / "teacher.json").write_text(json.dumps(broken))
    resolved = code_lm_table.resolve_roles(code_lm_table.load_reports(directory), "match")
    with pytest.raises(SystemExit, match="no 'bytes' block"):
        code_lm_table.check_comparable(resolved)


def test_an_empty_bench_directory_says_what_to_run(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="campaign"):
        code_lm_table.load_reports(tmp_path / "nothing")


def run_cli(monkeypatch: pytest.MonkeyPatch, bench: Path, *extra: str) -> int:
    monkeypatch.setattr(sys, "argv", ["code_lm_table", "--bench", str(bench), "--scale", "match", *extra])
    return code_lm_table.main()


def test_the_rows_print_in_comparison_order_with_roles_not_filenames(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reader meets teacher, baseline, distilled, control - in that order."""
    assert run_cli(monkeypatch, complete(tmp_path)) == 0
    out = capsys.readouterr().out
    order = [line for line in out.splitlines() if line.startswith("| ") and "---" not in line]
    assert [line.split("|")[1].strip() for line in order[1:]] == [
        "teacher",
        "baseline",
        "ours-distill",
        "ours-ce",
    ]
    # The last column names the artifact each row came from, so a reader can find it.
    assert [line.split("|")[7].strip() for line in order[1:]] == [
        "teacher",
        "baseline",
        "ours-match-distill",
        "ours-match-ce",
    ]
    assert "not run" in out  # HumanEval is absent on the student rows and is said so
    assert "4,000,000 bytes" in out
    # The decision the table exists to support is on the page, not left to the reader.
    assert "distillation gain bits per byte: +0.1500" in out


def test_mixed_devices_are_called_out(tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sampled decoding is not bit-reproducible across accelerators, so a mixed
    table has a difference in it that no row explains - and it says so."""
    bench = complete(tmp_path, **{"baseline": bench_report(1.2, device="NVIDIA Tesla T4")})
    run_cli(monkeypatch, bench)
    assert "devices" in capsys.readouterr().out


def two_machines(tmp_path: Path, **extra) -> Path:
    """The four rows as they really arrive: two scored on a T4, the students locally.

    Only the device *name* distinguishes them - ``str(torch.device("cuda"))`` is "cuda"
    on every accelerator, which is what made a Kaggle row and a local row look like one
    machine while HumanEval pass@1 quietly came from two.
    """
    t4 = {"device": "cuda", "device_name": "NVIDIA Tesla T4"}
    local = {"device": "cuda", "device_name": "NVIDIA GeForce GTX 1070"}
    rows = {
        "teacher": bench_report(0.4686, parameters=1_346_000_000, **t4),
        "baseline": bench_report(0.5372, parameters=164_100_000, pass_at_1=0.0784, **t4),
        "ours-match-distill": bench_report(0.7500, parameters=166_423_296, active=81_488_640, **local),
        "ours-match-ce": bench_report(0.9000, parameters=166_423_296, active=81_488_640, **local),
    }
    rows.update(extra)
    return write_bench(tmp_path, **rows)


def test_two_different_accelerators_are_not_one_device(tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    run_cli(monkeypatch, two_machines(tmp_path))
    out = capsys.readouterr().out
    assert "rows were measured on 2 devices (NVIDIA GeForce GTX 1070, NVIDIA Tesla T4)" in out


def test_the_same_two_machines_look_like_one_without_the_device_name(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure the device name exists to prevent, kept visible as a test.

    Strip the names and the same two machines are indistinguishable, so no note fires -
    not because the table is single-device but because the reports never said.
    """
    directory = two_machines(tmp_path)
    for path in directory.glob("*.json"):
        report = json.loads(path.read_text())
        report["device"] = "cuda"
        report.pop("device_name", None)
        path.write_text(json.dumps(report))
    run_cli(monkeypatch, directory)
    assert "devices" not in capsys.readouterr().out


def test_a_report_without_a_device_name_falls_back_to_its_type(tmp_path: Path) -> None:
    """A pre-change report is not blank in that column."""
    reports = code_lm_table.load_reports(complete(tmp_path))
    rows = code_lm_table.build_table(code_lm_table.resolve_roles(reports, "match"), "match")["rows"]
    assert {row["device"] for row in rows} == {"cuda"}


def test_single_device_tables_do_not_carry_the_note(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_cli(monkeypatch, complete(tmp_path))
    assert "devices" not in capsys.readouterr().out


def test_the_json_form_carries_the_limit_the_table_was_built_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "table.json"
    run_cli(monkeypatch, complete(tmp_path / "bench"), "--json", str(out))
    written = json.loads(out.read_text())
    assert written["bytes_limit"] == 4_000_000
    assert len(written["rows"]) == 4
