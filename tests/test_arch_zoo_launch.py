"""Tests for the facts a sweep states at launch, before any GPU time is spent.

Three of them, each a number the arm's interpretation depends on. A sweep that
shares the GPU is invisible from inside itself: each candidate takes proportionally
longer and the budget guard cannot tell, because it counts tokens rather than wall
clock - the head-paired arm spent three hours on its first candidate for exactly
this reason, which from its own log is indistinguishable from a hang, so the
launch warning is pinned here, including what it must *not* match (a queued launch
or an external waiter names the same script in its cmdline without being a sweep).
The other two are the parameter budget the zoo is built around and the set total it
will train, which section 11's equal-compute condition is written on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from arch_zoo_screen import (  # noqa: E402
    MATCHED_DESIGN_TARGETS,
    _is_sweep_invocation,
    competing_sweeps,
    effective_target,
    matching_report,
    parse_args,
    plan_cost,
    planned_stages,
    print_plan_cost,
    warn_if_gpu_shared,
)

DRIVER = "/home/u/code/act-lab/.venv/bin/python scripts/arch_zoo_screen.py --plan screen"


def test_a_running_driver_is_recognised():
    assert _is_sweep_invocation(DRIVER)
    assert _is_sweep_invocation("/usr/bin/python3.11 " + DRIVER.split(" ", 1)[1])


def test_the_queue_that_launched_the_sweep_is_not_a_sweep():
    """A shell wrapper or a pgrep waiter names the script without training."""
    assert not _is_sweep_invocation(
        "bash -c while pgrep -f 'arch_zoo_screen.py' >/dev/null; do sleep 30; done"
    )
    assert not _is_sweep_invocation("setsid nohup bash -c 'python scripts/arch_zoo_screen.py'")
    # A shell that only mentions the driver as an argument to something else.
    assert not _is_sweep_invocation("/usr/bin/grep -n arch_zoo_screen.py README.md")
    assert not _is_sweep_invocation("")


def test_other_sweeps_are_named_and_this_one_is_not():
    listing = [
        (101, DRIVER),
        (202, DRIVER + " --experiment-id other-arm"),
        # This process, which must never be reported as its own competition.
        (303, DRIVER),
        (404, "bash -c while pgrep -f arch_zoo_screen.py; do sleep 60; done"),
        (505, "/usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers"),
    ]
    assert competing_sweeps(listing, exclude=303) == [101, 202]
    assert competing_sweeps(listing, exclude=101) == [202, 303]


def test_nothing_running_is_reported_as_nothing():
    assert competing_sweeps([(1, "python /usr/bin/train_something_else.py")], exclude=1) == []


def test_the_warning_names_the_competition_and_the_reason(capsys):
    warned = warn_if_gpu_shared([101, 202])
    text = capsys.readouterr().out
    assert warned is True
    assert "2 other zoo sweep(s)" in text
    assert "101, 202" in text
    # The consequence, which is what makes it actionable: a longer wall clock that
    # the arm's own guard is structurally unable to notice.
    assert "share this GPU" in text
    assert "counts tokens rather than hours" in text


def test_no_warning_when_the_gpu_is_free(capsys):
    assert warn_if_gpu_shared([]) is False
    assert capsys.readouterr().out == ""


def test_the_printed_target_is_the_one_the_zoo_is_built_around():
    """`--match-target` defaults per design, and the header has to resolve it.

    It used to format the raw flag, so the documented way to take a matched
    design's default target - omitting it - crashed with a TypeError before any
    training, and an arm launched that way looked like a broken script.
    """
    from types import SimpleNamespace

    def args(design, match_target=None, target_params=100_000_000):
        return SimpleNamespace(
            design=design, match_target=match_target, target_params=target_params
        )

    assert effective_target(args("head-paired")) == 78_000_000.0
    assert effective_target(args("crossed")) == 78_000_000.0
    assert effective_target(args("matched-total")) == 100_000_000.0
    assert effective_target(args("matched-active")) == 35_000_000.0
    assert effective_target(args("matched-flops")) == 2.6e8
    # An explicit flag always wins; an unmatched design keeps --target-params.
    assert effective_target(args("crossed", match_target=60_000_000.0)) == 60_000_000.0
    assert effective_target(args("families", target_params=42_000_000)) == 42_000_000.0
    # The exact expression the header evaluates, which is where it crashed.
    assert f"{effective_target(args('head-paired')):,.0f}" == "78,000,000"


def test_only_the_designs_that_match_a_parameter_axis_have_target_defaults():
    for design in ("matched-total", "matched-active", "matched-flops", "crossed", "head-paired"):
        assert design in MATCHED_DESIGN_TARGETS
    # A family sweep matches nothing, so it must not have a matched default.
    assert "families" not in MATCHED_DESIGN_TARGETS


def _plan_args(**overrides):
    """The plan flags, as `parse_args` would hand them over."""
    from types import SimpleNamespace

    base = dict(
        plan="halving",
        screen_seeds="42,43",
        finalist_seeds="42,43,44",
        flat_steps="",
        stage1_steps=400,
        stage2_steps=800,
        stage3_steps=1200,
        keep_stage2=6,
        keep_stage3=3,
        promote_ties=False,
        batch_size=2,
        context_size=1024,
        learning_rate=3e-4,
        val_batches=16,
        data_path=None,
        val_data_path=None,
        budget_hours=20.0,
        assumed_tokens_per_s=4500.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_each_plan_is_costed_as_the_runs_it_will_actually_ask_for():
    """The estimate has to mirror the execution path, or it is a different plan.

    Cheapest to get wrong in the direction that matters: a stage that runs all
    candidates looks cheaper if it is costed at the number the *next* stage keeps,
    and the screen plan's confirmation stage is a floor rather than a count once
    ties are promoted.
    """
    halving = planned_stages(_plan_args(), zoo_size=8)
    assert [(s["label"], s["candidates"], len(s["seeds"]), s["steps"]) for s in halving] == [
        ("stage 1", 8, 2, 400),
        ("stage 2", 6, 2, 800),
        ("stage 3", 3, 3, 1200),
    ]
    screen = planned_stages(
        _plan_args(plan="screen", promote_ties=True), zoo_size=8
    )
    assert [(s["label"], s["candidates"]) for s in screen] == [
        ("screen", 8),
        ("confirmation", 3),
    ]
    assert screen[1]["may_grow"] is True
    assert planned_stages(_plan_args(plan="screen"), zoo_size=8)[1].get("may_grow") is False
    flat = planned_stages(
        _plan_args(plan="flat", flat_steps="400,1200"), zoo_size=8
    )
    assert [(s["label"], s["steps"]) for s in flat] == [("flat 400", 400), ("flat 1200", 1200)]
    # Flat costs every candidate on every length, which is why it is the expensive one.
    assert all(s["candidates"] == 8 for s in flat)


def test_the_plan_cost_counts_tokens_at_the_rate_that_trains_them():
    args = _plan_args(plan="flat", flat_steps="1200", finalist_seeds="48,49,50")
    args.budget_hours = 20.0
    cost = plan_cost(args, planned_stages(args, zoo_size=8))
    # 8 candidates x 3 seeds x 1200 steps x batch 2 x context 1024.
    assert cost["planned_runs"] == 24
    assert cost["planned_tokens"] == 24 * 1200 * 2 * 1024
    assert cost["planned_hours"] == pytest.approx(cost["planned_tokens"] / 4500 / 3600)
    assert cost["fits"] is True and cost["stops_at"] is None
    assert cost["covered_runs"] == cost["planned_runs"]


def test_a_plan_that_does_not_fit_names_where_the_guard_stops_it():
    """The guard breaks at a batch boundary after a whole stage-seed group.

    This is the prediction that replaces watching an arm stop: it must name the
    stage and seed, and its covered hours must be the batches that did fit rather
    than a fraction of a batch that cannot run.
    """
    args = _plan_args(budget_hours=1.0)
    cost = plan_cost(args, planned_stages(args, zoo_size=8))
    assert cost["fits"] is False
    # Runs are counted individually (8 candidates x 2 seeds + 6 x 2 + 3 x 3 = 37),
    # because that is the unit the phase's cost table uses.
    assert cost["planned_runs"] == 37
    # A stage-1 batch is 8 candidates x 400 steps = 6.55M tokens, 0.40 GPU-hours at
    # the assumed rate, so both screen seeds fit inside an hour and the *stage 2*
    # batch (9.83M tokens, 0.61h) does not.
    assert cost["covered_batches"] == 2
    assert cost["covered_runs"] == 16
    assert cost["stops_at"] == {"label": "stage 2", "seed": 42, "steps": 800}
    assert cost["covered_hours"] < 1.0 < cost["planned_hours"]
    assert cost["covered_hours"] == pytest.approx(
        2 * 8 * 400 * 2 * 1024 / 4500 / 3600
    )


def test_the_cost_warning_says_what_the_arm_would_report_on(capsys):
    args = _plan_args(budget_hours=1.0)
    cost = plan_cost(args, planned_stages(args, zoo_size=8))
    print_plan_cost(cost)
    text = capsys.readouterr().out
    assert "WARNING" in text
    assert "stage 2 seed 42" in text
    assert "16 of 37 runs" in text
    assert "partial design" in text
    # A plan that fits says so instead, and does not warn about a partial design.
    args = _plan_args(budget_hours=20.0)
    print_plan_cost(plan_cost(args, planned_stages(args, zoo_size=8)))
    fits = capsys.readouterr().out
    assert "within the 20.0h budget guard" in fits
    assert "WARNING" not in fits


def test_the_matching_report_states_the_budget_the_arm_will_train():
    """The arm's budget is the set *sum*, and the band is a per-candidate limit.

    Section 11's equal-compute condition is written on the sum, so it has to be
    reportable without adding up the ledger afterwards - and reported separately
    from the band, because a zoo can sit inside its band and still carry a different
    total, which is exactly the failure that section's check exists to catch.
    """
    from act.search.architecture_zoo import generate_matched_zoo

    zoo = generate_matched_zoo(
        target=78_000_000,
        match_on="total",
        tolerance=0.05,
        seed=42,
        context_size=1024,
        max_total_params=150_000_000,
        limit=4,
    )
    report = matching_report(zoo, 1024, 78_000_000, "total")
    assert report["set_total_params"] == sum(c.total_params for c in zoo)
    assert report["set_total_params"] > report["param_matched"]["spread"]["max"]
    assert set(report) >= {"set_total_params", "param_matched", "runtime_cost"}


def test_parse_args_resolves_a_default_target_end_to_end(monkeypatch):
    """The crash was in `main`, so the parsed namespace is what matters here."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "arch_zoo_screen.py",
            "--experiment-id",
            "arm",
            "--design",
            "head-paired",
            "--plan",
            "screen",
        ],
    )
    parsed = parse_args()
    assert parsed.match_target is None
    assert effective_target(parsed) == 78_000_000.0
    assert parsed.plan == "screen"
