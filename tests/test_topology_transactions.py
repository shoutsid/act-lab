from act.topology import (
    apply_transaction,
    create_genome,
    feasible_expert_mutations,
    genome_hash,
    grow_expert,
    prune_expert,
)


def test_failed_transaction_restores_every_genome_field() -> None:
    genome = create_genome(3, ffn="moe")
    before = genome_hash(genome)

    result = apply_transaction(
        genome,
        "invalid",
        lambda value: (value.layers[0].experts.clear() or True),
    )

    assert not result.applied
    assert result.before_hash == result.after_hash == before
    assert genome_hash(genome) == before
    assert len(genome.layers[0].experts) == 4


def test_callback_exception_restores_every_genome_field() -> None:
    genome = create_genome(7, ffn="moe")
    before = genome_hash(genome)

    def change_then_fail(value):
        value.seed = 99
        value.max_params = 1
        value.layers[0].experts.clear()
        raise RuntimeError("injected callback failure")

    result = apply_transaction(genome, "raises", change_then_fail)

    assert not result.applied
    assert result.errors == ("injected callback failure",)
    assert result.before_hash == result.after_hash == before
    assert genome_hash(genome) == before
    assert genome.seed == 7
    assert genome.max_params == 10**9
    assert len(genome.layers[0].experts) == 4


def test_successful_transaction_records_hash_transition() -> None:
    genome = create_genome(4, ffn="moe")
    result = apply_transaction(
        genome,
        "disable-layer",
        lambda value: setattr(value.layers[-1], "enabled", False) or True,
    )
    assert result.applied
    assert result.before_hash != result.after_hash


def test_no_reachable_operation_can_move_the_gene_the_search_phase_measured() -> None:
    """The whole reachable operation pool is expert grow/prune (report section 10.4).

    `feasible_expert_mutations` is the pool every V0 selection policy draws its
    candidates from, and every entry it yields is an expert grow or prune. The
    architecture phase measured the loss lever on real text as residual-stream width
    (-0.1562 nats at the factorial's own floor) and found active parameters
    uncorrelated with loss (rho = +0.214, p = 0.62), so this search spends its
    operations on the one variable that arm could not attribute an effect to. Nor can
    any of them reach the measured lever: no operation sets `hidden_size`, head count
    or depth, and the only assignment to `hidden_size` in the mutation module is the
    restore on a rejected change. If that ever changes, section 10.4 must be updated
    with the new reachable set rather than left describing a search space that no
    longer exists.
    """
    genome = create_genome(3, ffn="moe")
    candidates = feasible_expert_mutations(genome)
    assert candidates
    assert {operation for operation, _, _ in candidates} <= {"grow", "prune"}
    assert all(expert_id is None for operation, _, expert_id in candidates if operation == "grow")

    def shape() -> tuple:
        return (
            genome.hidden_size,
            genome.depth,
            tuple((layer.heads, layer.kv_heads, layer.ffn_width) for layer in genome.layers),
        )

    before = shape()
    for operation, layer_id, expert_id in candidates:
        if operation == "grow":
            grow_expert(genome, layer_id, 5)
        else:
            assert expert_id is not None
            prune_expert(genome, layer_id, expert_id)
        assert shape() == before
