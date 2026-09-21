"""A summary's label and level come from its own arm, and the record regenerates.

Three defects motivated these pins, all found on 2026-09-21 by regenerating the
committed summaries instead of trusting them:

1. **The label.** Sections 5.48, 5.49 and 5.52 committed summaries carried the
   5.9 vocabulary's branch string, whose two primaries are null-vs-random and
   adaptive-vs-null *of one cell* - a different pair from the ones those arms
   registered (two per-cell conjunctions, plus a rank-sum between cells in 5.50
   and 5.52). For 5.48's below-onset cell it named a *pass* where the
   registration records a failure, and for 5.52 it read "no churn harm on this
   substrate" where the recorded finding is a collapse of +0.163809. The label
   is now decided by the profile's own declaration (`per_cell_primary`) rather
   than by a hard-coded section list, so a new arm cannot silently inherit
   another family's wording.

2. **The level.** `summarize`'s CLI defaulted the Bonferroni divisor to 2.0 for
   every profile, so regenerating one of the two 5.50 cells wrote bar 313 and
   alpha 0.025 where the registration states 315 and alpha/3 - silently, and
   after the arm had already recorded that exact class of disagreement once.
   The divisor is now the profile's declaration and a mismatched override is
   refused.

3. **The record's own reproducibility.** Recomputed from the stored ``per_seed``
   triples, every committed summary's ``delta_stddev`` reproduces *bit-exactly*
   except the six ``null-control-*`` records of sections 5.9-5.15, which were
   written by a different accumulation (it matches ``statistics.pstdev``, which
   is order-sensitive). No single formula reproduces all six, and nothing in the
   suite recomputed that field before, so the drift was invisible. It is pinned
   here as enumerated and bounded rather than repaired by rewriting six
   committed records for a quantity no report section quotes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import null_toy_study as study  # noqa: E402

from act.evolution.figures_data import summaries_dir  # noqa: E402
from act.evolution.study import _gate_min_wins  # noqa: E402

# The arms that registered the 5.9 pair (a null-vs-random and an
# adaptive-vs-null primary on the same cell) and may therefore carry its
# vocabulary. Everything else in the study declares its own primaries.
LEGACY_VOCABULARY_SECTIONS = {"5.9", "5.10", "5.12", "5.15"}
# The two 5.50 cells are the family's only three-primary arm (two per-cell bars
# plus the rank-sum between cells), so they are read at alpha/3.
THREE_PRIMARY_SECTIONS = {"5.50"}
# The six summaries written by the earlier accumulation, named so that the
# enumeration is a decision rather than a discovered tolerance.
PSTDEV_VINTAGE = {
    "null-control-depth1-384.json",
    "null-control-depth1-spread-384.json",
    "null-control-depth2-384.json",
    "null-control-depth3-384.json",
    "null-control-depth4-384.json",
    "null-control-depth4-prefix-384.json",
}


def _committed(profile: str) -> dict:
    """The profile's committed summary, or a skip if it is registered but unrun.

    An arm is *registered* before it runs (that is the point of a
    pre-registration), so a profile carries ``run_revision: None`` until its
    registration commit exists. A profile that HAS a frozen revision but no
    committed summary is a failure, not a skip - which is what
    ``test_every_frozen_profile_has_a_committed_summary`` enforces.
    """
    path = summaries_dir() / Path(study.PROFILES[profile]["summary"]).name
    if not path.exists():
        if study.PROFILES[profile]["run_revision"] is None:
            pytest.skip(f"{profile} is registered but has not run yet")
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


def test_every_profile_declares_its_bonferroni_divisor() -> None:
    """The divisor is data on the profile, and it is the arm's own registration.

    A missing key is a failure rather than a fallback: the fallback is exactly
    how the 5.50 cells were nearly rewritten at bar 313.
    """
    # A tripwire, not a contract: if a new arm is added to the table and this
    # test has not been revisited, the count here is what says so.
    assert len(study.PROFILES) == 17
    for profile, spec in study.PROFILES.items():
        assert "bonferroni_divisor" in spec, f"{profile} declares no divisor"
        declared = float(spec["bonferroni_divisor"])
        expected = 3.0 if spec["report_section"] in THREE_PRIMARY_SECTIONS else 2.0
        assert declared == expected, (
            f"{profile} (section {spec['report_section']}) declares {declared:g}, "
            f"its registration implies {expected:g}"
        )
    # And the declaration is what the committed record carries: bar 315/576 for
    # the 5.50 cells, 313/576 for the two-primary cells.
    for profile, spec in study.PROFILES.items():
        divisor = float(spec["bonferroni_divisor"])
        bar = _gate_min_wins(len(spec["seeds"]), study.ALPHA / divisor)
        summary = _committed(profile)
        recorded = summary["comparisons"]["adaptive_vs_null"]["min_wins_bonferroni"]
        assert recorded == bar, f"{profile}: committed bar {recorded} != {bar}"
        assert summary["bonferroni_alpha"] == pytest.approx(study.ALPHA / divisor)


def test_a_summary_cannot_be_written_at_a_level_its_registration_does_not_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 5.50 cells refuse the sibling arms' alpha/2 rather than obey it."""
    for profile in ("width-d3-dose2", "width-d4-dose2"):
        with pytest.raises(SystemExit) as excinfo:
            study.summarize(profile, study.ALPHA, 2.0)
        assert "declares Bonferroni divisor 3" in str(excinfo.value)
    # And the refusal lands *before* the shards are read - otherwise it would
    # only hold where the run's shard files happen to be on disk.
    monkeypatch.setattr(study, "SHARD_DIR", tmp_path / "nowhere")
    with pytest.raises(SystemExit) as excinfo:
        study.summarize("width-d3-dose2", study.ALPHA, 2.0)
    assert "declares Bonferroni divisor" in str(excinfo.value)
    assert "missing shard" not in str(excinfo.value)


def test_an_unfrozen_profile_has_no_summary_and_a_missing_one_fails() -> None:
    """The registration/run boundary, in the direction that always holds.

    A profile is registered with ``run_revision: None`` before it runs, so an
    unfrozen profile cannot already have a summary. The reverse does *not*
    hold: the freeze commit sets the revision and the shards run after it, so
    frozen-without-a-summary is a legitimate intermediate state - which is why
    ``_committed`` fails loudly (rather than skipping) for a frozen profile
    whose summary is missing, catching a summary path that is simply wrong.
    """
    for profile, spec in study.PROFILES.items():
        path = summaries_dir() / Path(spec["summary"]).name
        if spec["run_revision"] is None:
            assert not path.exists(), f"{profile} is unfrozen but has a summary"
        elif path.exists():
            assert json.loads(path.read_text())["code_revision"] == spec["run_revision"]


def test_the_label_style_is_the_arms_own_declaration() -> None:
    """Per-cell labels for 5.48-5.52, the 5.9 vocabulary for 5.9-5.15 only."""
    for profile, spec in study.PROFILES.items():
        labelled = bool(spec.get("per_cell_primary"))
        legacy = spec["report_section"] in LEGACY_VOCABULARY_SECTIONS
        assert labelled is not legacy, (
            f"{profile} (section {spec['report_section']}): per_cell_primary="
            f"{labelled} is the wrong style for this arm"
        )


def test_every_committed_label_is_its_own_arms() -> None:
    """No 5.48-5.52 summary carries the 5.9 vocabulary's branch string."""
    for profile, spec in study.PROFILES.items():
        summary = _committed(profile)
        label = summary["outcome"]
        if spec.get("per_cell_primary"):
            section = spec["report_section"]
            assert label.startswith(
                f"cell creates value ({section}'s per-cell primary"
            ) or label.startswith(
                f"cell does not create value ({section}'s per-cell primary"
            ), f"{profile}: {label!r}"
            assert "branch " not in label
            creates = (
                summary["comparisons"]["adaptive_vs_null"]["wins"]
                >= summary["comparisons"]["adaptive_vs_null"]["min_wins_bonferroni"]
                and summary["comparisons"]["adaptive_vs_null"]["mean_delta"] < 0
            )
            assert label.startswith("cell creates value") is creates
        else:
            assert label.startswith("branch "), f"{profile}: {label!r}"


@pytest.mark.parametrize("profile", sorted(study.PROFILES))
def test_the_committed_summary_regenerates(profile: str) -> None:
    """The arm's own declaration reproduces the record it wrote.

    Byte-exact for every profile whose record is one vintage (the eight width
    arms and the six 5.9-5.15 records are different vintages of ``delta_stddev``
    only, which the test below bounds). The label and the level are the fields
    this refactor touches, and both now come from the profile - so a drift here
    is a wiring failure, which is exactly what the first attempt at this fix
    produced by regenerating the 5.50 cells at the CLI's default divisor.
    """
    name = Path(study.PROFILES[profile]["summary"]).name
    path = summaries_dir() / name
    if not path.exists() and study.PROFILES[profile]["run_revision"] is None:
        pytest.skip(f"{profile} is registered but has not run yet")
    fresh = study.summarize(profile, study.ALPHA)
    committed = json.loads(path.read_text())
    if name not in PSTDEV_VINTAGE:
        assert json.dumps(fresh, indent=1, sort_keys=True) + "\n" == path.read_text()
        return
    # The six earlier records: every key except delta_stddev must still match.
    moved = {
        f"comparisons.{comparison}.{field}"
        for comparison in committed["comparisons"]
        for field in committed["comparisons"][comparison]
        if committed["comparisons"][comparison][field]
        != fresh["comparisons"][comparison][field]
    }
    allowed = {
        f"comparisons.{comparison}.delta_stddev" for comparison in committed["comparisons"]
    }
    # depth1-spread agrees on one of its three (the vintage's accumulation is
    # not uniformly different), so the bound is a subset, not an equality.
    assert moved and moved <= allowed, f"{profile}: unexpected fields moved: {sorted(moved)}"
    for key in set(committed) | set(fresh):
        if key == "comparisons":
            continue
        assert committed.get(key) == fresh.get(key), f"{profile}: {key} moved"


def test_the_committed_stddevs_are_bounded_and_the_vintage_is_enumerated() -> None:
    """``delta_stddev`` recomputes exactly - except the six 5.9-5.15 records.

    Those six were written by an accumulation that matches ``statistics.pstdev``
    (order-sensitive, so not reproduced bit-exactly in the stored order either),
    and no single formula reproduces all six. The bound is the finding: ~3e-15
    relative, on a field no report section quotes and no decision reads. The
    point of the pin is that the field is recomputed at all - it was not.
    """
    pairs = {
        "adaptive_vs_null": ("adaptive", "null"),
        "adaptive_vs_random": ("adaptive", "random"),
        "null_vs_random": ("null", "random"),
    }
    inexact: set[str] = set()
    worst = 0.0
    for profile in study.PROFILES:
        name = Path(study.PROFILES[profile]["summary"]).name
        summary = _committed(profile)
        for comparison, (left, right) in pairs.items():
            recorded = summary["comparisons"][comparison]["delta_stddev"]
            deltas = [
                row["losses"][left] - row["losses"][right]
                for row in summary["per_seed"]
            ]
            n = len(deltas)
            mean = sum(deltas) / n
            recomputed = (sum((value - mean) ** 2 for value in deltas) / n) ** 0.5
            if recomputed != recorded:
                inexact.add(name)
            worst = max(worst, abs(recomputed - recorded) / abs(recorded))
    assert inexact == PSTDEV_VINTAGE, sorted(inexact ^ PSTDEV_VINTAGE)
    assert worst < 3e-15
