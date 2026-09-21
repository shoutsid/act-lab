"""Pin the gate chain: the plan's own words, and what each gate requires.

The declaration is only worth having if it cannot drift from the plan it claims to read, so
the first test here reads section 99 of the plan document and requires each gate's
``plan_text`` to appear in it verbatim. The rest pin the structure that makes the chain a
chain - each gate requiring its predecessor, exactly one gate implemented - and that the
declaration says out loud which gates nothing runs yet, because "declared" and "runnable"
are different statements and the gap between them is where a reader would otherwise assume
more progress than there is.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from act.training import gate_chain  # noqa: E402

PLAN = ROOT / "docs" / "complete_implementation_plan.md"


def _plan_gate_text(gate: int) -> str:
    """The plan's own sentence for one gate, from its ``## Gate N`` section."""
    text = PLAN.read_text(encoding="utf-8")
    section = re.search(
        rf"^## Gate {gate}\s*\n+(.+?)\s*$", text, re.MULTILINE
    )
    assert section is not None, f"the plan has no '## Gate {gate}' section"
    return section.group(1).strip()


@pytest.mark.parametrize("gate", sorted(gate_chain.READS))
def test_the_declaration_quotes_the_plan_verbatim(gate: int) -> None:
    """A gate whose text drifts from the plan is a gate describing something else."""
    assert gate_chain.READS[gate].plan_text == _plan_gate_text(gate)


def test_the_chain_is_gapless_and_each_gate_requires_its_predecessor() -> None:
    """Gate 7 authorized by a failed gate 6 would not be the plan's chain."""
    gates = gate_chain.chain()
    assert gates == (4, 5, 6, 7)
    for gate in gates:
        read = gate_chain.READS[gate]
        if gate == gates[0]:
            assert read.requires is None
        else:
            assert read.requires == gate - 1


def test_exactly_one_gate_is_implemented_and_it_is_gate_four() -> None:
    implemented = [gate for gate, read in gate_chain.READS.items() if read.implemented]
    assert implemented == [4]
    assert gate_chain.declared_but_unimplemented() == (5, 6, 7)


def test_the_next_gate_is_the_lowest_unimplemented_one() -> None:
    assert gate_chain.next_gate() == 5


@pytest.mark.parametrize("gate", [5, 6, 7])
def test_an_unimplemented_gate_declares_what_is_still_open(gate: int) -> None:
    """The open questions are the point: they are what a later reader must not invent.

    A declaration that silently resolved them would be the thing to distrust when the
    numbers arrive, because the resolution would have been chosen with the numbers in view.
    """
    read = gate_chain.READS[gate]
    assert read.open_questions, f"gate {gate} declares no open questions"
    assert read.artifact and read.comparison and read.bar


def test_every_gate_names_a_comparison_and_a_bar() -> None:
    for gate, read in gate_chain.READS.items():
        assert read.artifact, gate
        assert read.comparison, gate
        assert read.bar, gate


def test_gate_six_requires_identity_and_held_out_seeds() -> None:
    """The two things that make it a *transfer* claim rather than a repeat of gate 5."""
    read = gate_chain.READS[6]
    assert "genome_hash" in read.artifact
    assert "held-out" in read.artifact or "held out" in read.artifact


def test_gate_seven_requires_a_declared_scale_map() -> None:
    """Without one, 'across scale' is whatever the runner decides it means afterwards."""
    read = gate_chain.READS[7]
    assert "scale map" in read.artifact
    # And the ambiguity that matters: scaling width is not the same claim as scaling depth.
    assert any("depth" in question for question in read.open_questions)
