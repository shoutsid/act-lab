"""Section 5.34: the probe's scoring rebuild (``probe_scoring``).

Four claims pinned here, all teeth-checkable:

1. **The default path is untouched.** ``probe_scoring`` defaults to ``"default"``, and a
   committed section 5.33 block re-runs bit-identically under it - so every number in
   sections 5.16-5.33 keeps its meaning.
2. **The knob reaches the scored model, and reaches it with the right settings.** Under
   ``"applied"`` the probe's trial rebuilds are called with the arm's own
   ``preserve_router``/``new_expert_initialization``; under ``"default"`` they are called
   with the historical defaults while the applied rebuild still uses the arm's settings.
3. **The applied rebuild is RNG-independent at these settings**, which is what makes the
   contrast readable as "the ranking changed": the rebuilt model holds the same values
   whatever the global RNG state, so a different score cannot leak into the applied
   weights through the initialisation draw.
4. **Pool membership does not depend on the scoring rule.** With
   ``probe_candidate_cap=1`` the pool holds one candidate, and then the applied run is
   bit-identical under both rules - the study's registered integrity check, at the unit
   level.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from act.evolution import experiment as experiment_module
from act.evolution.experiment import ExperimentConfig, run_experiment
from act.model import model_from_genome
from act.model.factory import rebuild_model_from_genome
from act.topology import ArchitectureGenome, grow_expert

SUMMARY = Path("artifacts/study-summaries/probe-selection-depth2-1536.json")

# The section 5.30/5.33 registered configuration with the section 5.28 delivery settings.
REGISTERED = dict(
    hidden_size=24,
    vocab_size=32,
    depth=2,
    expert_count=4,
    active_experts=2,
    batch_size=2,
    sequence_length=12,
    steps=48,
    mutation_interval=12,
    mutation_budget=3,
    learning_rate=0.0003,
    validation_batches=4,
    adaptive_policy="probe_loss",
    adaptive_expert_metric="random",
    probe_candidate_cap=4,
    probe_batch_count=2,
    probe_candidate_policy="spread",
    independent_arm_rng=True,
    device="cpu",
    forced_operation=None,
    preserve_router_on_rebuild=True,
    new_expert_initialization="clone",
)


@pytest.fixture(autouse=True)
def _fixed_thread_count():
    """Pin torch's thread count for every test in this file.

    Report section 6: bit-exact reproduction is only defined at a fixed thread count
    (reductions are thread-count sensitive), and the committed numbers this file
    compares against were produced with one thread. Without this the file passes or
    fails depending on the shell's ``OMP_NUM_THREADS``, which is exactly the kind of
    silent substitution the pin exists to prevent.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _adaptive(seed: int, **overrides):
    config = ExperimentConfig(
        **{**REGISTERED, **overrides, "seed": seed, "arms": "fixed,adaptive"}
    )
    return {arm.mode: arm for arm in run_experiment(config).arms}["adaptive"]


def test_the_default_scoring_field_is_default() -> None:
    assert ExperimentConfig().probe_scoring == "default"
    assert ExperimentConfig(probe_scoring="applied").probe_scoring == "applied"


def test_an_unsupported_scoring_rule_is_rejected() -> None:
    with pytest.raises(ValueError):
        run_experiment(ExperimentConfig(probe_scoring="whatever"))


def test_the_default_scoring_reproduces_a_committed_section_5_33_block() -> None:
    """Pinned as literals from the committed record: the historical path must not move.

    Two selections and their expected probe losses are literals read from the
    summary's own per-seed rows, so this fails if either the code path or the record
    moves.
    """
    summary = json.loads(SUMMARY.read_text())
    record = next(exp for exp in summary["per_seed"] if exp["seed"] == 119000)
    config = {
        key: value
        for key, value in summary["config"].items()
        if key in ExperimentConfig.__dataclass_fields__ and key not in {"seed", "arms"}
    }
    for label, selection, expected in (
        ("A", "argmin", [14.171257495880127, 13.766163349151611, 13.312538623809814]),
        ("Q", "quietest", [14.305128574371338, 13.775033473968506, 13.308974266052246]),
    ):
        arm = _adaptive(119000, **{**config, "probe_selection": selection})
        assert arm.final_loss == record["losses"][label], label
        assert [
            float(event["probe_loss"]) for event in arm.mutation_history
        ] == expected, label
        assert all(
            event["probe_scoring"] == "default" for event in arm.mutation_history
        )


def test_the_scored_rebuild_uses_the_settings_each_rule_names(monkeypatch) -> None:
    """The code path itself: what the probe builds when it scores a candidate.

    ``"applied"`` must score with the arm's settings, ``"default"`` with the
    historical defaults, while the applied rebuild uses the arm's settings in both
    cases - which is the mismatch section 5.34 measures.
    """
    calls: list[tuple[bool, str]] = []
    original = experiment_module.rebuild_model_from_genome

    def spy(model, genome, **kwargs):
        calls.append(
            (
                bool(kwargs.get("preserve_router", False)),
                str(kwargs.get("new_expert_initialization", "random")),
            )
        )
        return original(model, genome, **kwargs)

    monkeypatch.setattr(experiment_module, "rebuild_model_from_genome", spy)
    historical, arm_settings = (False, "random"), (True, "clone")
    for scoring, probe_agrees_with_the_application in (
        ("default", False),
        ("applied", True),
    ):
        calls.clear()
        _adaptive(123000, probe_scoring=scoring)
        assert calls, f"no rebuilds were made under {scoring}"
        assert arm_settings in calls
        if probe_agrees_with_the_application:
            assert set(calls) == {arm_settings}, scoring
        else:
            # Under the historical rule the probe scores with defaults while the
            # winner is applied with the arm's settings: the mismatch itself.
            assert historical in calls, scoring
            assert calls.count(historical) >= 3, scoring


def test_the_applied_rebuild_does_not_depend_on_the_rng_state() -> None:
    """Why the contrast is readable: at these settings the rebuild is deterministic.

    With the router preserved and a cloned new expert, every mutated-layer parameter
    is transferred by id, so the rebuilt model cannot carry a different draw - which is
    what stops a changed score from leaking into the applied weights. The historical
    defaults are asserted to *depend* on the draw in the same setup, so this test fails
    if it is ever satisfied vacuously.
    """
    genome = ArchitectureGenome.from_dict(_adaptive(123000).genome)
    model = model_from_genome(genome, context_size=12)
    layer_id = genome.layers[0].id
    grown = deepcopy(genome)
    assert grow_expert(grown, layer_id, step=12) is not None

    def rebuild(seed: int, **settings):
        torch.manual_seed(seed)
        return rebuild_model_from_genome(model, grown, context_size=12, **settings)

    applied = [
        rebuild(seed, preserve_router=True, new_expert_initialization="clone")
        for seed in (1, 999)
    ]
    for (name, first), (_, second) in zip(
        applied[0].state_dict().items(), applied[1].state_dict().items()
    ):
        assert torch.equal(first, second), name

    historical = [rebuild(seed) for seed in (1, 999)]
    assert any(
        not torch.equal(first, second)
        for first, second in zip(
            historical[0].state_dict().values(), historical[1].state_dict().values()
        )
    )


SUMMARIES = {
    2: Path("artifacts/study-summaries/probe-scoring-depth2-384.json"),
    3: Path("artifacts/study-summaries/probe-scoring-depth3-384.json"),
}
IDENTITY = {
    2: Path("artifacts/study-summaries/probe-scoring-identity-depth2-64.json"),
    3: Path("artifacts/study-summaries/probe-scoring-identity-depth3-64.json"),
}


def _reload(path: Path) -> dict:
    return json.loads(path.read_text())


def test_the_committed_verdict_is_branch_b_at_both_depths() -> None:
    """The registered verdicts, re-derived from the stored per-seed rows.

    The means and win counts quoted in section 5.34 are recomputed here from the
    per-seed losses, so a tampered summary cannot keep its quoted numbers.
    """
    expected_mean = {2: -0.04043, 3: -0.03478}
    for depth, path in SUMMARIES.items():
        summary = _reload(path)
        deltas = [
            exp["losses"]["S"] - exp["losses"]["D"]
            for exp in summary["per_seed"]
        ]
        assert len(deltas) == 384
        assert sum(deltas) / len(deltas) == pytest.approx(expected_mean[depth], abs=5e-05)
        fused = summary["comparisons"]["fused_S_vs_D"]
        assert sum(d < 0 for d in deltas) == fused["wins"]
        assert fused["passes"]  # P1: the fused run improves
        assert not summary["comparisons"]["own_S_vs_D"]["passes"]  # P2 fails
        assert summary["outcome"].startswith("branch (b)")
        # The penalty secondary fails *because* the fused loss improved.
        assert summary["comparisons"]["penalty_S_less_than_D"]["mean_delta"] > 0


def test_the_undisagreed_seeds_are_bit_identical() -> None:
    """A design check the block carries for free: same selection, same run.

    On every seed where the two rules chose the same candidate sequence the fused
    losses must be exactly equal. This is the same claim the cap-1 identity blocks
    test, but at the registered cap and without being arranged.
    """
    for path in SUMMARIES.values():
        summary = _reload(path)
        undisagreed = [
            exp for exp in summary["per_seed"] if exp["selection_changes"] == 0
        ]
        assert undisagreed, "no seeds agreed; the check would be vacuous"
        for exp in undisagreed:
            assert exp["losses"]["D"] == exp["losses"]["S"]
            assert exp["losses"]["D_own"] == exp["losses"]["S_own"]


def test_the_identity_blocks_are_all_bit_identical() -> None:
    for path in IDENTITY.values():
        summary = _reload(path)
        assert summary["all_identical"] is True
        assert len(summary["per_seed"]) == 64
        assert all(exp["identical_fused"] for exp in summary["per_seed"])


def test_the_search_cost_difference_is_booked_not_hidden() -> None:
    """The scoring rule books the scored model's active parameters, so cost differs.

    Sections 5.32/5.33's "the cost identity holds exactly" is a selection-rule
    claim; this pins that the scoring-rule variant is not held to it, and that the
    measured difference is recorded rather than asserted away.
    """
    for path in SUMMARIES.values():
        summary = _reload(path)
        assert 0.0 < summary["search_flops_relative_delta_mean"] < 0.10
        assert summary["search_flops_relative_delta_max"] >= (
            summary["search_flops_relative_delta_mean"]
        )
        zeroed = [
            exp for exp in summary["per_seed"] if exp["search_flops_relative_delta"] == 0
        ]
        assert zeroed, "no seed booked identical cost; the mechanism claim is untested"


def test_forced_operation_is_inert_on_the_probe_path() -> None:
    """The structural finding: this knob cannot force a probe-path operation.

    ``forced_operation`` is consulted only in ``_mutate_arm``'s non-probe branch, so
    on the probe path both values produce the identical run - which is why section
    5.34's forced blocks were deleted and the mechanism answered by partition.
    """
    runs = {
        operation: _adaptive(123000, forced_operation=operation)
        for operation in (None, "grow", "prune")
    }
    reference = runs[None]
    for operation, arm in runs.items():
        assert arm.final_loss == reference.final_loss, operation
        assert [
            (event["operation"], event["layer_id"], event["expert_id"])
            for event in arm.mutation_history
        ] == [
            (event["operation"], event["layer_id"], event["expert_id"])
            for event in reference.mutation_history
        ], operation


def test_cap_one_makes_the_scoring_rule_a_no_op() -> None:
    """The registered integrity check: with one candidate the score decides nothing."""
    default = _adaptive(128000, probe_candidate_cap=1)
    applied = _adaptive(128000, probe_candidate_cap=1, probe_scoring="applied")
    assert default.final_loss == applied.final_loss
    assert [
        (event["operation"], event["layer_id"], event["expert_id"])
        for event in default.mutation_history
    ] == [
        (event["operation"], event["layer_id"], event["expert_id"])
        for event in applied.mutation_history
    ]
    # ...while the scores themselves do differ, which is what the contrast measures.
    assert [event["probe_loss"] for event in default.mutation_history] != [
        event["probe_loss"] for event in applied.mutation_history
    ]
