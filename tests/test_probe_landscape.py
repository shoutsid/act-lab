import math

import pytest

from act.evolution import ExperimentConfig, run_experiment


def test_probe_landscape_records_all_candidates() -> None:
    result = run_experiment(
        ExperimentConfig(
            seed=5,
            steps=12,
            mutation_interval=3,
            mutation_budget=2,
            arms="adaptive",
            probe_candidate_cap=4,
            probe_batch_count=2,
            probe_landscape=True,
        )
    )
    adaptive = next(arm for arm in result.arms if arm.mode == "adaptive")
    for event in adaptive.mutation_history:
        landscape = event["probe_landscape"]
        assert len(landscape) == event["probe_candidates"]
        losses = [item["probe_loss"] for item in landscape]
        assert losses == sorted(losses)
        assert all(math.isfinite(loss) for loss in losses)
        selected = event["probe_selected_index"]
        assert 0 <= selected < len(landscape)
        assert event["probe_loss"] == landscape[selected]["probe_loss"]
    assert adaptive.search_flops > 0


def test_forced_candidate_overrides_selection() -> None:
    forced = run_experiment(
        ExperimentConfig(
            seed=5,
            steps=12,
            mutation_interval=3,
            mutation_budget=2,
            arms="adaptive",
            probe_candidate_cap=4,
            probe_batch_count=2,
            probe_landscape=True,
            probe_force_candidate=1,
        )
    )
    adaptive = next(arm for arm in forced.arms if arm.mode == "adaptive")
    for event in adaptive.mutation_history:
        assert event["forced_candidate"] == 1
        assert event["probe_selected_index"] == 1
        assert event["probe_loss"] == event["probe_landscape"][1]["probe_loss"]


def test_forced_candidate_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError):
        run_experiment(
            ExperimentConfig(
                seed=5,
                steps=3,
                mutation_interval=2,
                mutation_budget=1,
                arms="adaptive",
                probe_candidate_cap=2,
                probe_force_candidate=2,
            )
        )


def test_forced_zero_matches_default_selection() -> None:
    default = run_experiment(
        ExperimentConfig(
            seed=9,
            steps=6,
            mutation_interval=3,
            mutation_budget=1,
            arms="random,adaptive",
        )
    )
    forced = run_experiment(
        ExperimentConfig(
            seed=9,
            steps=6,
            mutation_interval=3,
            mutation_budget=1,
            arms="random,adaptive",
            probe_force_candidate=0,
        )
    )
    default_arm = next(arm for arm in default.arms if arm.mode == "adaptive")
    forced_arm = next(arm for arm in forced.arms if arm.mode == "adaptive")
    assert default_arm.final_loss == forced_arm.final_loss
    assert default_arm.search_flops == forced_arm.search_flops
    instrumentation_keys = {"forced_candidate", "probe_selected_index", "probe_landscape"}
    strip = lambda events: [
        {key: value for key, value in event.items() if key not in instrumentation_keys}
        for event in events
    ]
    assert strip(default_arm.mutation_history) == strip(forced_arm.mutation_history)


def test_forced_events_limit_applies_to_first_events_only() -> None:
    forced = run_experiment(
        ExperimentConfig(
            seed=5,
            steps=12,
            mutation_interval=3,
            mutation_budget=3,
            arms="adaptive",
            probe_candidate_cap=4,
            probe_batch_count=2,
            probe_landscape=True,
            probe_force_candidate=1,
            probe_force_events=1,
        )
    )
    adaptive = next(arm for arm in forced.arms if arm.mode == "adaptive")
    assert adaptive.mutation_history[0]["forced_candidate"] == 1
    assert adaptive.mutation_history[0]["probe_selected_index"] == 1
    assert all("forced_candidate" not in event for event in adaptive.mutation_history[1:])
    assert all(event["probe_selected_index"] == 0 for event in adaptive.mutation_history[1:])


def test_uninstrumented_runs_omit_landscape_fields() -> None:
    result = run_experiment(
        ExperimentConfig(
            seed=5,
            steps=6,
            mutation_interval=3,
            mutation_budget=1,
            arms="adaptive",
        )
    )
    adaptive = next(arm for arm in result.arms if arm.mode == "adaptive")
    assert all("probe_landscape" not in event for event in adaptive.mutation_history)
    assert all("forced_candidate" not in event for event in adaptive.mutation_history)
