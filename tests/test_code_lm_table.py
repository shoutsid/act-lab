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
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import code_lm_table  # noqa: E402


def bench_report(
    bits_per_byte: float,
    *,
    parameters: int = 1_000_000,
    active: int | None = None,
    bytes_limit: int = 4_000_000,
    device: str = "cuda",
    pass_at_1: float | None = None,
    architecture: tuple[str, ...] = ("LlamaForCausalLM",),
) -> dict:
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
            "nats": 1.0,
            "held_out_bytes": bytes_limit,
            "bits_per_byte": bits_per_byte,
            "loss_per_token": 0.1,
            "bytes_limit": bytes_limit,
        },
    }
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
