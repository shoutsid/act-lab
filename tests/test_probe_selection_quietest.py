"""The ``quietest`` probe-selection rule, and the contract it has to keep.

Section 5.32 established that on the non-probe paths, removing the least-loaded
expert is worth 27.6-41.6% of the historical prune penalty. It also turned up a
structural fact: ``prune_expert_policy`` is **inert** on the probe-guided path,
because ``_mutate_by_probe`` enumerates every expert as a candidate
(``feasible_expert_mutations``) and returns before the policy branch is reached.

So the discovery controller already owns the prune-target lever and exercises it by
measured probe loss. The open question is whether that selection is *better than the
cheap load rule* - and answering it needs the load rule to exist on the probe path
as a selection mode, which is what ``probe_selection="quietest"`` adds.

Pinned here:

1. **The helper, on its own.** ``quietest_prune_index`` indexes the decided pool,
   ignores non-prune candidates, skips experts whose load was never recorded rather
   than guessing, returns the fallback when the pool holds no scoreable prune, and
   breaks ties to the lowest index.
2. **The isolation the comparison depends on.** The rule overrides the prune target
   *only*: the applied candidate must always come from the probed pool, and the
   selected index must point at a prune candidate whenever the pool holds one. A
   rule that quietly changed which event was picked, or grew instead, would not
   isolate the target.
3. **The mode is accepted on both probe paths** (toy evolution and the V0 session
   config), which is the mirror contract the two already share.
"""

from __future__ import annotations

import pytest

from act.evolution import ExperimentConfig, run_experiment
from act.topology import quietest_prune_index

BASE = dict(
    hidden_size=24,
    vocab_size=32,
    depth=2,
    expert_count=4,
    active_experts=2,
    batch_size=2,
    sequence_length=12,
    steps=36,
    mutation_interval=12,
    mutation_budget=2,
    validation_batches=2,
    adaptive_policy="probe_loss",
    probe_candidate_cap=4,
    probe_batch_count=2,
    probe_candidate_policy="spread",
    independent_arm_rng=True,
    device="cpu",
    preserve_router_on_rebuild=True,
    new_expert_initialization="clone",
    arms="adaptive",
)


def test_the_helper_indexes_the_decided_pool_by_least_load() -> None:
    candidates = [
        ("grow", "L0", None),
        ("prune", "L0", "L0E2"),
        ("prune", "L0", "L0E0"),
        ("prune", "L1", "L1E1"),
    ]
    experts = {"L0": ["L0E0", "L0E1", "L0E2"], "L1": ["L1E0", "L1E1"]}
    # Expert loads, positional over each layer's expert list:
    #   L0 -> L0E0 0.5, L0E1 0.1, L0E2 0.4     L1 -> L1E0 0.2, L1E1 0.05
    loads = {"L0": (0.5, 0.1, 0.4), "L1": (0.2, 0.05)}
    # L1E1 is the least-loaded expert anywhere in the pool, so the last entry wins.
    assert quietest_prune_index(candidates, experts, loads) == 3
    # Drop L1E1 and the least-loaded remaining prune is L0E2 (0.4) at index 1. The
    # grow at index 0 is never eligible however quiet it might look.
    assert quietest_prune_index(candidates[:3], experts, loads) == 1
    assert quietest_prune_index(candidates[:1], experts, loads, fallback=5) == 5


def test_the_helper_falls_back_rather_than_guessing() -> None:
    experts = {"L0": ["L0E0", "L0E1"]}
    # No prune in the pool: the caller's own rule decides.
    assert quietest_prune_index(
        [("grow", "L0", None)], experts, {"L0": (0.5, 0.5)}, fallback=7
    ) == 7
    # A prune whose layer has no recorded load at all is skipped, not treated as
    # zero - an unrecorded load is missing data, not evidence of quiet.
    assert quietest_prune_index(
        [("prune", "L0", "L0E0")], experts, {}, fallback=7
    ) == 7
    # An expert no longer present in the layer is skipped the same way.
    assert quietest_prune_index(
        [("prune", "L0", "L0E9")], experts, {"L0": (0.5, 0.5)}, fallback=7
    ) == 7


def test_the_helper_breaks_ties_to_the_lowest_index() -> None:
    candidates = [("prune", "L0", "L0E1"), ("prune", "L0", "L0E0")]
    experts = {"L0": ["L0E0", "L0E1"]}
    assert quietest_prune_index(candidates, experts, {"L0": (0.3, 0.3)}) == 0


def test_the_config_accepts_the_mode_and_rejects_others() -> None:
    for bad in ("Quietest", "loudest", ""):
        with pytest.raises(ValueError, match="probe_selection"):
            run_experiment(
                ExperimentConfig(**{**BASE, "probe_selection": bad, "steps": 12})
            )
    run_experiment(
        ExperimentConfig(**{**BASE, "probe_selection": "quietest", "steps": 12})
    )


def test_v0_session_config_accepts_the_mode() -> None:
    """The mirror contract: both probe paths take the same selection values."""
    from act.training.v0_session import V0SessionConfig

    config = V0SessionConfig(probe_selection="quietest")
    assert config.probe_selection == "quietest"
    with pytest.raises(ValueError, match="probe_selection"):
        V0SessionConfig(probe_selection="loudest")


def test_the_rule_is_confined_to_prune_targets_inside_the_probed_pool() -> None:
    """The isolation the section 5.33 comparison rests on.

    The rule has to differ from ``argmin`` **only** in the prune target. So, for
    every event: the applied candidate is one the probe actually scored; if the
    probe's own pick was not a prune, the selection is that pick unchanged; and if
    it was a prune, the selection is still a prune (possibly a different one).
    Anything else would mean ``quietest`` was also changing which operation happens,
    and the comparison would no longer isolate the target.
    """
    seen_override = False
    for seed in (1, 2, 3, 4):
        arm = run_experiment(
            ExperimentConfig(**BASE, seed=seed, probe_selection="quietest", probe_landscape=True)
        ).arms[0]
        for event in arm.mutation_history:
            landscape = event["probe_landscape"]
            selected = event["probe_selected_index"]
            chosen = landscape[selected]
            assert chosen["operation"] == event["operation"]
            assert chosen["expert_id"] == event["expert_id"]
            probe_pick = landscape[0]
            if probe_pick["operation"] != "prune":
                assert selected == 0, (
                    "the probe would have grown or pruned nothing, so its pick must "
                    "stand; the rule is a prune-target rule, not an operation rule"
                )
                continue
            assert chosen["operation"] == "prune"
            if selected != 0:
                seen_override = True
    assert seen_override, (
        "the rule never overrode the probe's own pick: either the quietest prune in "
        "the pool is always the probe argmin on these seeds, or the mode is not live"
    )


def test_the_mode_changes_the_run() -> None:
    """A mode that produced identical runs would make section 5.33 vacuous."""
    for seed in (1, 2, 3):
        outcomes = {}
        for selection in ("argmin", "quietest", "random"):
            arm = run_experiment(
                ExperimentConfig(**BASE, seed=seed, probe_selection=selection)
            ).arms[0]
            outcomes[selection] = (
                arm.final_loss,
                tuple(event["expert_id"] for event in arm.mutation_history),
            )
        assert len(set(outcomes.values())) == 3, (
            f"selection modes coincide on seed {seed}: {outcomes}"
        )
