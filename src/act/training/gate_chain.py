"""The gate chain of the plan's section 99, written down as declarations.

The plan states its gates as one line each - "Gate 5: Fresh architecture remains
competitive", "Gate 6: Architecture transfers across seeds", "Gate 7: Architecture transfers
across scale" - and leaves the operational content (which artifact, which comparison, what
bar, what has to be true first) to whoever runs them. Gate 4 is the exception: it has a
pre-registered verdict, a rehearsal of that verdict and a lock on its knobs, because it is
the one that was reached. The rest were prose.

The reason to write them down *now*, before the artifacts they read exist, is that the
alternative is defining them afterwards, against the numbers that came back. That is the
same failure the study family spends pre-registration avoiding, one level up: a gate whose
measurement is chosen after the measurement is not a gate. So each gate below declares what
it reads, the comparison it makes, the bar it applies, the gate it requires, and - the part
that matters most for honesty - **what is still an open question about it**. The open
questions are declared rather than hidden because several are genuine judgement calls that
belong to whoever is running the gate, and a declaration that pretends otherwise would be
the thing to distrust later.

``plan_text`` is verbatim from the plan and pinned against the document by
``tests/test_gate_chain.py``, so a change to the plan fails the suite rather than quietly
leaving this module describing something else.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateRead:
    """One gate's read: what it consumes, what it compares, what it requires."""

    gate: int
    plan_text: str
    artifact: str
    comparison: str
    bar: str
    requires: int | None
    implemented: bool
    open_questions: tuple[str, ...] = ()


READS: dict[int, GateRead] = {
    4: GateRead(
        gate=4,
        plan_text="Adaptive > static MoE.",
        artifact=(
            "the declared v0-5-1 block: all 12 seeds of the primary queue and the "
            "pool-matched control and null queues, read by scripts/v0_verdict.py"
        ),
        comparison=(
            "aggregate per-arm search FLOPs over training FLOPs against the pre-registered "
            "prediction, plus the paired adaptive-vs-random harm guard and the do-nothing "
            "contrast"
        ),
        bar=(
            "cost <= 5%, measured C inside the [0.5, 2]x band, harm guard clear (failures "
            "under 10 of 12), the section 5.44 pool-matched control clause, and the section "
            "5.14 do-nothing clause"
        ),
        requires=None,
        implemented=True,
    ),
    5: GateRead(
        gate=5,
        plan_text="Fresh architecture remains competitive.",
        artifact=(
            "the winning arm's genome, transferred with "
            "v0_session.fresh_init_from_checkpoint (topology only, weights discarded), then "
            "trained at the declared protocol"
        ),
        comparison=(
            "the transferred, freshly initialized architecture against the same static "
            "baselines Gate 4 scores, on the same protocol and token budget"
        ),
        bar=(
            "the paired comparison clears the same win threshold Gate 4 applies to its harm "
            "guard, and the do-nothing contrast still holds"
        ),
        requires=4,
        implemented=False,
        open_questions=(
            "whether 'competitive' means clearing the same win threshold as Gate 4 or a "
            "non-inferiority margin against the *searched* arm - the two are different "
            "claims and the plan's line does not choose",
            "whether the transferred arm is compared at the declared token budget (the "
            "search's own training is not part of what is transferred, so its FLOP cost is "
            "near zero and the Gate 4 cost criterion is trivially satisfied here)",
        ),
    ),
    6: GateRead(
        gate=6,
        plan_text="Architecture transfers across seeds.",
        artifact=(
            "the same genome, identified by v0_topology.genome_hash equality with the gate-5 "
            "genome, freshly initialized at **held-out seeds** - seeds that took part in "
            "neither the search that discovered it nor gate 5, on the architecture track's "
            "own fresh-seed convention"
        ),
        comparison=(
            "the gate-5 comparison, repeated at the held-out seeds: a transfer that only "
            "works at the seed that found it is a seed effect, not an architecture"
        ),
        bar="the same bar as gate 5, holding across the held-out seeds",
        requires=5,
        implemented=False,
        open_questions=(
            "how many held-out seeds, and whether they should be the architecture track's "
            "fresh-seed convention (the finalist re-ranking uses 45-47) or a new block",
            "whether every held-out seed must clear the bar, or the block's aggregate - the "
            "'every' reading is stricter and would fail on a single unlucky seed",
        ),
    ),
    7: GateRead(
        gate=7,
        plan_text="Architecture transfers across scale.",
        artifact=(
            "a **declared scale map** applied to the gate-5 genome - which fields move "
            "(hidden size, expert width, depth, attention heads) and which are held - plus "
            "the scaled genome's hash, and a static baseline matched at the new scale"
        ),
        comparison=(
            "the gate-5 comparison at the new scale, with the baseline parameter-matched "
            "there rather than carried over"
        ),
        bar="the same bar as gate 5, at the new scale",
        requires=6,
        implemented=False,
        open_questions=(
            "what 'scale' means here: changing hidden size or expert width scales a fixed "
            "topology, while changing depth changes the *topology* - so a scale map that "
            "moves depth is testing a different claim, and the map has to say which it is",
            "which scale to test, and whether one target is enough for a gate that will "
            "authorize serious compute",
            "the plan says only after this gate should serious compute be invested, so this "
            "is the read with the most riding on it and the least written about it",
        ),
    ),
}


def chain() -> tuple[int, ...]:
    """The declared gates, in order."""
    return tuple(sorted(READS))


def declared_but_unimplemented() -> tuple[int, ...]:
    """The gates this module declares a read for that nothing runs yet.

    Reported rather than inferred from prose: 'the chain is declared' and 'the chain runs'
    are different statements, and the difference is exactly what a reader of the plan would
    otherwise have to discover for themselves.
    """
    return tuple(gate for gate, read in sorted(READS.items()) if not read.implemented)


def next_gate() -> int:
    """The lowest-numbered gate that is not implemented."""
    unimplemented = declared_but_unimplemented()
    return unimplemented[0] if unimplemented else chain()[-1] + 1
