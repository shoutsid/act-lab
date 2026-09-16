"""Configuration liveness contract for both training paths (report section 5.37).

**Why this file exists.** Section 5.33 found ``prune_expert_policy`` *inert* on the
probe-guided path, and section 5.34 found ``forced_operation`` inert on it too. Both
are declared config fields that no code path reads under the policy a block was
running, so registered compute was spent on a mechanism that was never applied. That
is a cheap-to-prevent, recurring failure: the config surface is a promise and nothing
checked it.

This file checks it, by **measuring** rather than by reading code. Every field of both
config dataclasses is classified, and each classification is asserted. The two known
no-ops are pinned as *scoped* facts - inert on the probe-guided arm, live on the
random arm - so that wiring either into the probe path fails here and forces sections
5.33/5.34 to be updated instead of being silently contradicted.

**A liveness flip is only as good as its value.** ``adaptive_load_threshold`` moves
nothing at ``0.5 -> 0.9`` and everything at ``0.05 -> 0.95``: a mid-range flip does not
cross the controller's decision boundary and would have been misread as a dead knob.
Thresholds are therefore flipped to their extremes, indices to a value that is *not*
the argmin default, and dependent knobs are flipped **with** the field they depend on.

**Dependencies are part of the contract, and each was measured:**
- ``probe_scoring`` does nothing when the applied rebuild already *is* the historical
  default: the arm must carry a treatment for the scoring mismatch to exist at all.
  This is exactly why section 5.34's block needed ``preserve_router_on_rebuild`` /
  ``clone`` arms.
- ``optimizer_reset_steps`` does nothing when a reset step coincides with a mutation
  step on a **mutating** arm - the rebuild already discards Adam, so the reset is
  masked. Section 5.23's own control is unaffected (its reset twin is the ``fixed``
  arm, which never mutates), but that design re-used on a mutating arm measures zero.
- ``probe_force_events`` is a modifier: without ``probe_force_candidate`` it is inert by
  construction, so it is exercised with its parent set.
- Under the default shared arm RNG (``independent_arm_rng=False``) a change to the
  random arm perturbs the adaptive arm's *numbers* without changing its *decisions*
  (section 9.8's coupling). Arm-scoped claims are therefore measured with the arms'
  RNG decoupled, or the drift would be misattributed to the knob.

**Cost.** A tiny shape, one seed unless a field needs more, with the base run shared
across entries: ~35 measured fields per path in about a minute. The shapes are
deliberately small; they are chosen to make each knob's effect visible, not to be
representative of anything.
"""

from __future__ import annotations

import dataclasses
import tempfile

import pytest
import torch

from act.evolution.experiment import ExperimentConfig, run_experiment
from act.training import V0SessionConfig, run_v0_session

CHANGE = "change"
SAME = "same"
EITHER = "either"
CHANNELS = ("decisions", "loss", "validation", "booking", "probe", "record")

_DECIDES = {"decisions": CHANGE}
_SCORES = {"probe": CHANGE}
_LOSSES = {"loss": CHANGE}
_RECORDS = {
    "record": CHANGE,
    "decisions": SAME,
    "loss": SAME,
    "probe": SAME,
    "booking": SAME,
}
_INERT = dict.fromkeys(CHANNELS, SAME)

# --- the two measured shapes -------------------------------------------------

BASE_V0 = dict(
    device="cpu",
    vocab_size=256,
    context_size=64,
    hidden_size=64,
    attention_heads=4,
    depth=2,
    expert_count=4,
    active_experts=2,
    expert_width=64,
    batch_size=2,
    total_steps=8,
    mutation_interval=4,
    mutation_budget=2,
    probe_candidate_cap=2,
    probe_batch_count=1,
    validation_batches=2,
    activation_checkpointing=False,
    checkpoint_interval=8,
    max_checkpoints=1,
    seed=7,
)

BASE_TOY = dict(
    device="cpu",
    steps=8,
    mutation_interval=2,
    mutation_budget=2,
    arms="random,adaptive",
    seed=7,
)

# --- observation -------------------------------------------------------------


@pytest.fixture(autouse=True)
def one_thread():
    """Report section 6: reproduction is only defined at a fixed thread count.

    Without this the contract would pass or fail depending on the shell's
    ``OMP_NUM_THREADS``, which is exactly the silent substitution the pin prevents.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _channels(arm) -> dict[str, object]:
    return {
        "decisions": (
            arm.architecture_hash,
            tuple(
                (e.get("operation"), e.get("layer_id"), e.get("expert_id"))
                for e in arm.mutation_history
            ),
        ),
        "loss": repr(arm.final_loss),
        "validation": repr(arm.validation_loss),
        "booking": (arm.training_flops, arm.search_flops),
        "probe": tuple(repr(e.get("probe_loss")) for e in arm.mutation_history),
        "record": tuple(sorted(map(str, e.keys())) for e in arm.mutation_history),
    }


def run_v0(overrides: dict) -> dict[str, dict[str, object]]:
    config = V0SessionConfig(**{**BASE_V0, **overrides})
    with tempfile.TemporaryDirectory() as tmp:
        result = run_v0_session(config, checkpoint_dir=tmp)
    return {arm.mode: _channels(arm) for arm in result.arms}


def run_toy(overrides: dict) -> dict[str, dict[str, object]]:
    result = run_experiment(ExperimentConfig(**{**BASE_TOY, **overrides}))
    return {arm.mode: _channels(arm) for arm in result.arms}


def _changed(base: dict, flipped: dict, mode: str, channel: str) -> bool:
    """Whether ``channel`` moved on ``mode``. A differing arm set counts as a change."""
    if mode not in base or mode not in flipped:
        return True
    return base[mode][channel] != flipped[mode][channel]


def _assert_channel(mode: str | None, channel: str, want: str, observed: bool) -> None:
    where = f" on arm {mode!r}" if mode else ""
    if want == CHANGE:
        assert observed, (
            f"{channel}{where} did not move when the field was flipped, but the "
            f"contract says it must - a declared knob that reaches no decision"
        )
    elif want == SAME:
        assert not observed, (
            f"{channel}{where} moved when the field was flipped, but the contract "
            f"says it must not - update report section 5.37 and this table"
        )


# --- the contract -----------------------------------------------------------
#  field -> {base?, flip, expect? / per_arm?, seeds?}

V0_TABLE = {
    "seed": {"flip": {"seed": 8}, "expect": _DECIDES},
    "arms": {"flip": {"arms": ("random",)}, "expect": _DECIDES},
    "vocab_size": {"flip": {"vocab_size": 512}, "expect": _DECIDES},
    "context_size": {"flip": {"context_size": 32}, "expect": _DECIDES},
    "hidden_size": {"flip": {"hidden_size": 32}, "expect": _DECIDES},
    "depth": {"flip": {"depth": 3}, "expect": _DECIDES},
    "attention_heads": {"flip": {"attention_heads": 2}, "expect": _DECIDES},
    "expert_count": {"flip": {"expert_count": 5}, "expect": _DECIDES},
    # Widened so a third active expert leaves room for the events to keep every
    # layer's active count uniform; the active parameter count moves
    # (83008 -> 99392) without changing which mutations are selected.
    "active_experts": {
        "base": {"expert_count": 6},
        "flip": {"expert_count": 6, "active_experts": 3},
        "expect": _LOSSES,
    },
    "expert_width": {"flip": {"expert_width": 32}, "expect": _DECIDES},
    "batch_size": {"flip": {"batch_size": 3}, "expect": _DECIDES, "seeds": (7, 8)},
    "total_steps": {"flip": {"total_steps": 12}, "expect": _LOSSES},
    "learning_rate": {"flip": {"learning_rate": 6e-4}, "expect": _SCORES},
    "mutation_interval": {"flip": {"mutation_interval": 6}, "expect": _DECIDES},
    # The horizon has to be long enough for the third budget slot to be reached, or
    # the flip only adds a permission nobody exercises (measured: it did not bind).
    "mutation_budget": {
        "base": {"total_steps": 12},
        "flip": {"total_steps": 12, "mutation_budget": 3},
        "expect": _DECIDES,
    },
    "probe_candidate_cap": {"flip": {"probe_candidate_cap": 4}, "expect": _DECIDES},
    "probe_batch_count": {
        "flip": {"probe_batch_count": 2},
        "expect": _DECIDES,
        "seeds": (7, 8),
    },
    "probe_candidate_policy": {
        "flip": {"probe_candidate_policy": "spread"},
        "expect": _DECIDES,
    },
    "probe_selection": {"flip": {"probe_selection": "argmax"}, "expect": _DECIDES},
    "preserve_router_on_rebuild": {
        "flip": {"preserve_router_on_rebuild": True},
        "expect": {"decisions": CHANGE, "probe": CHANGE},
        "seeds": (7, 8),
    },
    "new_expert_initialization": {
        "base": {"preserve_router_on_rebuild": True},
        "flip": {"preserve_router_on_rebuild": True, "new_expert_initialization": "clone"},
        "expect": _DECIDES,
    },
    "probe_landscape": {"flip": {"probe_landscape": True}, "expect": _RECORDS},
    "validation_batches": {
        "flip": {"validation_batches": 1},
        "expect": {"decisions": SAME, "loss": CHANGE},
    },
    "activation_checkpointing": {
        "flip": {"activation_checkpointing": True},
        "expect": _INERT,
    },
    "dtype": {"flip": {"dtype": "fp16"}, "expect": _SCORES},
    "checkpoint_interval": {"flip": {"checkpoint_interval": 4}, "expect": _INERT},
    "max_checkpoints": {"flip": {"max_checkpoints": 2}, "expect": _INERT},
    "checkpoint_max_bytes": {
        "flip": {"checkpoint_max_bytes": 5_000_000},
        "expect": _INERT,
    },
    "probe_finetune_steps": {"flip": {"probe_finetune_steps": 1}, "expect": _SCORES},
    "gate_margin": {
        "base": {"arms": ("gated",)},
        "flip": {"arms": ("gated",), "gate_margin": 1e9},
        "expect": _DECIDES,
    },
}

V0_UNMEASURED = {
    "data_path": "needs a real token stream; pinned by tests/test_v0_real_data.py",
    "val_data_path": "needs a real token stream; pinned by tests/test_v0_real_data.py",
    "probe_data_source": "needs a real token stream; pinned by tests/test_v0_real_data.py",
    "declared": "a validation gate rather than a trajectory knob (guard below)",
    "device": "placement only (resolve_device), so a same-device flip is degenerate",
}

V0_GUARDS = {
    "declared": ({"declared": True}, "not the declared V0 protocol configuration"),
    "probe_selection": ({"probe_selection": "bogus"}, "unsupported probe_selection"),
    "probe_data_source": (
        {"probe_data_source": "heldout"},
        "requires val_data_path",
    ),
    "new_expert_initialization": (
        {"new_expert_initialization": "clone"},
        "requires preserve_router_on_rebuild",
    ),
    "active_experts": (
        {"active_experts": 1},
        "active_experts must be between 2",
    ),
    "checkpoint_max_bytes": (
        {"checkpoint_max_bytes": 1},
        "exceeds max_bytes",
    ),
}

TOY_TABLE = {
    "seed": {"flip": {"seed": 8}, "expect": _DECIDES},
    "steps": {"flip": {"steps": 20}, "expect": _LOSSES},
    "batch_size": {"flip": {"batch_size": 3}, "expect": _DECIDES, "seeds": (7, 8)},
    "sequence_length": {"flip": {"sequence_length": 6}, "expect": _DECIDES},
    "validation_batches": {
        "flip": {"validation_batches": 2},
        "expect": {"decisions": SAME, "loss": CHANGE},
    },
    "mutation_interval": {"flip": {"mutation_interval": 4}, "expect": _DECIDES},
    "mutation_budget": {"flip": {"mutation_budget": 3}, "expect": _DECIDES},
    "hidden_size": {"flip": {"hidden_size": 32}, "expect": _DECIDES},
    "vocab_size": {"flip": {"vocab_size": 64}, "expect": _DECIDES},
    "depth": {"flip": {"depth": 2}, "expect": _DECIDES},
    "expert_count": {"flip": {"expert_count": 5}, "expect": _DECIDES},
    "active_experts": {
        "flip": {"active_experts": 3},
        "expect": _DECIDES,
        "seeds": (7, 8),
    },
    "learning_rate": {"flip": {"learning_rate": 6e-4}, "expect": _SCORES, "seeds": (7, 8)},
    "dataset_id": {"flip": {"dataset_id": "synthetic-v2"}, "expect": _INERT},
    "tokenizer_id": {
        "flip": {"tokenizer_id": "synthetic-tokenizer-v2"},
        "expect": _INERT,
    },
    "code_revision": {"flip": {"code_revision": "abc123"}, "expect": _INERT},
    # Flipped to the extremes: 0.5 -> 0.9 does not cross the decision boundary and
    # would have been misread as a dead knob.
    "adaptive_load_threshold": {
        "base": {"adaptive_policy": "signal", "adaptive_load_threshold": 0.05},
        "flip": {"adaptive_policy": "signal", "adaptive_load_threshold": 0.95},
        "expect": _DECIDES,
        "seeds": (7, 8),
    },
    "adaptive_expert_metric": {
        "base": {"adaptive_policy": "signal"},
        "flip": {"adaptive_policy": "signal", "adaptive_expert_metric": "routing_load"},
        "expect": _LOSSES,
        "seeds": (7, 8),
    },
    "adaptive_policy": {"flip": {"adaptive_policy": "signal"}, "expect": _DECIDES},
    "probe_candidate_cap": {"flip": {"probe_candidate_cap": 2}, "expect": _DECIDES},
    "probe_batch_count": {"flip": {"probe_batch_count": 1}, "expect": _DECIDES, "seeds": (7, 8)},
    "probe_candidate_policy": {
        "flip": {"probe_candidate_policy": "spread"},
        "expect": _DECIDES,
    },
    "probe_selection": {"flip": {"probe_selection": "argmax"}, "expect": _DECIDES},
    "independent_arm_rng": {
        "flip": {"independent_arm_rng": True},
        "expect": _DECIDES,
        "seeds": (7, 8),
    },
    "probe_landscape": {"flip": {"probe_landscape": True}, "expect": _RECORDS},
    # Index 1, not 0: the argmin default is index 0, so forcing 0 is degenerate.
    "probe_force_candidate": {"flip": {"probe_force_candidate": 1}, "expect": _DECIDES},
    # A modifier: inert alone (tested below), live with its parent set.
    "probe_force_events": {
        "base": {"probe_force_candidate": 1},
        "flip": {"probe_force_candidate": 1, "probe_force_events": 1},
        "expect": _DECIDES,
    },
    # Step 3, which is not a mutation step under this shape - see the masking test.
    "optimizer_reset_steps": {
        "flip": {"optimizer_reset_steps": (3,)},
        "expect": _LOSSES,
    },
    "preserve_router_on_rebuild": {
        "flip": {"preserve_router_on_rebuild": True},
        "expect": _SCORES,
        "seeds": (7, 8),
    },
    # The probe-guided arm running alone: bit-identical, which is sections 5.33/5.34's
    # finding in its cleanest form. The random arm's half of the fact is the
    # companion test below.
    "forced_operation": {
        "base": {"arms": "adaptive"},
        "flip": {"arms": "adaptive", "forced_operation": "prune"},
        "expect": _INERT,
    },
    "new_expert_initialization": {
        "base": {"preserve_router_on_rebuild": True},
        "flip": {"preserve_router_on_rebuild": True, "new_expert_initialization": "clone"},
        "expect": _SCORES,
        "seeds": (7, 8),
    },
    "prune_expert_policy": {
        "base": {"arms": "adaptive"},
        "flip": {"arms": "adaptive", "prune_expert_policy": "quietest"},
        "expect": _INERT,
    },
    "probe_scoring": {
        "base": {"preserve_router_on_rebuild": True, "new_expert_initialization": "clone"},
        "flip": {
            "preserve_router_on_rebuild": True,
            "new_expert_initialization": "clone",
            "probe_scoring": "applied",
        },
        "per_arm": {"adaptive": {"decisions": CHANGE, "probe": CHANGE}},
    },
    "arms": {"flip": {"arms": "adaptive"}, "expect": _DECIDES},
}

TOY_UNMEASURED = {
    "device": "placement only (resolve_device), so a same-device flip is degenerate",
}

TOY_GUARDS = {
    "adaptive_policy": ({"adaptive_policy": "load"}, "unsupported adaptive_policy"),
    "adaptive_expert_metric": (
        {"adaptive_expert_metric": "importance"},
        "unsupported adaptive_expert_metric",
    ),
    "probe_force_candidate": (
        {"probe_force_candidate": -1},
        "must be non-negative",
    ),
    "probe_force_events": ({"probe_force_events": 0}, "must be at least 1"),
    "optimizer_reset_steps": (
        {"optimizer_reset_steps": (99,)},
        "must be strictly increasing positive steps within the run",
    ),
}


def _check(spec: dict, run) -> None:
    seeds = spec.get("seeds", (7,))
    touches_seed = "seed" in spec.get("flip", {})
    expect = spec.get("expect", {})
    per_arm = spec.get("per_arm", {})
    seen = {channel: [] for channel in CHANNELS}
    scope_seen: dict[tuple[str, str], list[bool]] = {}

    for seed in seeds:
        base_extra = {**spec.get("base", {})}
        flip_extra = {**base_extra, **spec["flip"]}
        if not touches_seed:
            base_extra["seed"] = seed
            flip_extra["seed"] = seed
        base_obs, flip_obs = run(base_extra), run(flip_extra)
        modes = sorted(set(base_obs) | set(flip_obs))
        for channel in CHANNELS:
            seen[channel].append(
                any(_changed(base_obs, flip_obs, mode, channel) for mode in modes)
            )
        for mode, wanted in per_arm.items():
            for channel, want in wanted.items():
                scope_seen.setdefault((mode, channel), []).append(
                    _changed(base_obs, flip_obs, mode, channel)
                )
        if touches_seed:
            break

    for channel, want in expect.items():
        _assert_channel(None, channel, want, any(seen[channel]))
    for (mode, channel), observations in scope_seen.items():
        want = per_arm[mode][channel]
        if want == CHANGE:
            _assert_channel(mode, channel, want, any(observations))
        else:
            _assert_channel(mode, channel, want, not any(observations))


def _memoize(run):
    cache: dict[tuple, dict] = {}

    def wrapped(overrides: dict):
        key = tuple(sorted(overrides.items()))
        if key not in cache:
            cache[key] = run(overrides)
        return cache[key]

    return wrapped


def _run_whole_table(table: dict, run, label: str) -> None:
    """Every field in one test, so the shared base run is measured once."""
    memo = _memoize(run)
    failures = []
    for field, spec in sorted(table.items()):
        try:
            _check(spec, memo)
        except AssertionError as exc:
            failures.append(f"{label}.{field}: {exc}")
    assert failures == [], failures


# --- every declared field is classified --------------------------------------


@pytest.mark.parametrize(
    "config,table,unmeasured,label",
    [
        (V0SessionConfig, V0_TABLE, V0_UNMEASURED, "v0"),
        (ExperimentConfig, TOY_TABLE, TOY_UNMEASURED, "toy"),
    ],
)
def test_every_config_field_is_classified(config, table, unmeasured, label):
    """The drift guard: a new field fails here until it is classified.

    This is the check whose absence let section 5.34's ``forced_operation`` block be
    declared, registered and run as a duplicate of the forced-grow block.
    """
    declared = {field.name for field in dataclasses.fields(config)}
    classified = set(table) | set(unmeasured)
    assert classified == declared, {
        "unclassified": sorted(declared - classified),
        "stale": sorted(classified - declared),
    }


# --- the measured contract, one test per path --------------------------------


def test_v0_config_liveness_contract():
    """Each declared knob of the declared path either moves the trajectory or must not."""
    _run_whole_table(V0_TABLE, run_v0, "v0")


def test_toy_config_liveness_contract():
    """Each declared knob of the toy path either moves the trajectory or must not."""
    _run_whole_table(TOY_TABLE, run_toy, "toy")


# --- guards: an invalid value is rejected, not silently ignored ----------------


@pytest.mark.parametrize("field", sorted(V0_GUARDS))
def test_v0_invalid_values_are_rejected(field):
    overrides, fragment = V0_GUARDS[field]
    with pytest.raises(ValueError, match=fragment):
        run_v0(overrides)


@pytest.mark.parametrize("field", sorted(TOY_GUARDS))
def test_toy_invalid_values_are_rejected(field):
    overrides, fragment = TOY_GUARDS[field]
    with pytest.raises(ValueError, match=fragment):
        run_toy(overrides)


# --- the dependency facts the contract rests on ------------------------------


def _identical(a: dict, b: dict) -> bool:
    return a == b


def test_probe_scoring_is_inert_without_a_treatment():
    """``probe_scoring`` needs an arm whose applied rebuild differs from the default.

    With ``preserve_router_on_rebuild`` off and ``new_expert_initialization=random``
    the applied rebuild already *is* the historical one, so scoring under it changes
    nothing. This is why section 5.34's block had to run treated arms.
    """
    assert _identical(run_toy({}), run_toy({"probe_scoring": "applied"}))


def test_optimizer_reset_is_masked_by_a_colliding_mutation():
    """A reset on a mutating arm is a no-op when it lands on a mutation step.

    The rebuild already re-creates AdamW, so the reset has nothing left to discard.
    The off-collision step moves the loss, so the knob itself is applied: this is a
    masking hazard, not a dead field. Section 5.23's control is unaffected because its
    reset twin is the ``fixed`` arm, which never mutates.
    """
    colliding = {"steps": 8, "mutation_interval": 4, "mutation_budget": 2}
    assert _identical(
        run_toy(colliding), run_toy({**colliding, "optimizer_reset_steps": (4,)})
    )
    assert not _identical(
        run_toy(colliding), run_toy({**colliding, "optimizer_reset_steps": (3,)})
    )


def test_probe_force_events_alone_is_inert():
    """``probe_force_events`` is a modifier, so alone it must change nothing."""
    assert _identical(run_toy({}), run_toy({"probe_force_events": 1}))


@pytest.mark.parametrize(
    "field,value",
    [("forced_operation", "prune"), ("prune_expert_policy", "quietest")],
)
def test_the_arm_scoped_no_ops_are_live_on_the_random_arm(field, value):
    """The other half of sections 5.33/5.34: both knobs *are* read on the random arm.

    The table above pins bit-identical behaviour on the probe-guided arm. This pins
    that the fields are not dead everywhere - they reach the random arm's trajectory -
    which is what makes the scoping a finding about the probe path rather than a
    complaint about two unused fields.
    """
    assert not _identical(
        run_toy({"arms": "random"}),
        run_toy({"arms": "random", field: value}),
    )
