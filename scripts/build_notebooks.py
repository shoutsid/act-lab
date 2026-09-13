"""Build the first reproducible Kaggle notebooks (plan deliverable 2).

The plan's deliverable list (RESEARCH_REPORT.md section 7, item 2) defers the
reproducible Kaggle notebooks "until a gate passes"; Gate 3 has now passed
three times, and plan sections 39 and 104 name the first notebooks. This
builder emits them deterministically - every re-run over the same source
produces byte-identical ``.ipynb`` files (no execution counts, no outputs,
fixed key order), so the notebooks are versionable and reviewable like code.

Usage (from the repository root):

    PYTHONPATH=src python scripts/build_notebooks.py            # build all
    PYTHONPATH=src python scripts/build_notebooks.py --check    # verify bytes

The notebooks are written to run inside a Kaggle session after the
repository checkout step, i.e. with the repository root as the working
directory and ``PYTHONPATH=src`` (or an editable install). Every cell wraps
a real code-path contract - CLI modes, session-plan arithmetic, checkpoint
round trips - and each notebook ends by recording a summary JSON under
``artifacts/notebook-records/`` so a session's evidence is appendable to the
experiment registry trail. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = ROOT / "notebooks"
RECORD_DIR = "artifacts/notebook-records"

NBFORMAT = 4
NBFORMAT_MINOR = 5


def _code(source: str) -> dict:
    """A code cell with the source split into ``nbformat`` line chunks."""
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.splitlines(keepends=True)}


def _markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def _notebook(title: str, purpose: str, cells: list[dict]) -> dict:
    """Assemble a notebook with the fixed header every session begins by."""
    header = [
        f"# {title}\n",
        "\n",
        f"{purpose}\n",
        "\n",
        "Reproducibility contract (plan section 40): every session begins by\n",
        "determining its configuration, locating and validating prior state,\n",
        "and continues from there; it ends by saving metrics and a summary JSON.\n",
        "Run inside a Kaggle session (or locally) after checking out this\n",
        "repository, with the repository root as the working directory.\n",
    ]
    bootstrap = [
        "import sys\n",
        "from pathlib import Path\n",
        "\n",
        "REPO = Path.cwd()\n",
        "assert (REPO / 'src' / 'act').is_dir(), (\n",
        "    f'run this notebook with the repository root as the working directory; '\n",
        "    f'cwd was {REPO!r}'\n",
        ")\n",
        "sys.path.insert(0, str(REPO / 'src'))\n",
        "print('python', sys.version.split()[0], '| repo', REPO)\n",
    ]
    return {
        "cells": [_markdown("".join(header)), _code("".join(bootstrap)), *cells],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "act": {"protocol": "act-kaggle-notebooks-v1"},
        },
        "nbformat": NBFORMAT,
        "nbformat_minor": NBFORMAT_MINOR,
    }


def _record_cell(filename: str, payload_expr: str) -> dict:
    """A closing cell that appends the notebook's summary record.

    ``payload_expr`` must be a Python expression evaluating to a JSON-safe
    dict; the record is written with sorted keys so re-running a session on
    the same state produces byte-identical records.
    """
    return _code(
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        f"record_dir = Path('{RECORD_DIR}')\n"
        f"record_dir.mkdir(parents=True, exist_ok=True)\n"
        f"payload = {payload_expr}\n"
        f"record = json.dumps(payload, sort_keys=True, indent=2) + '\\n'\n"
        f"target = record_dir / '{filename}'\n"
        f"existing = target.read_text() if target.exists() else None\n"
        f"if existing != record:\n"
        f"    target.write_text(record)\n"
        f"print('record:', target)"
    )


def notebook_00_environment() -> dict:
    """Plan section 39 `00_environment`: verify the compute contract.

    The declared V0 model must reproduce the pre-registered parameter counts
    (285,336,576 total / 134,341,632 active) before any session trusts the
    machine it is running on; the tolerance band in section 5.1 is derived
    from exactly those counts.
    """
    cells = [
        _markdown(
            "# Environment check (plan section 39)\n"
            "\n"
            "Verifies CUDA visibility, GPU count, host RAM, disk, PyTorch, and -\n"
            "the part the protocol actually depends on - that the declared V0\n"
            "configuration materializes the pre-registered parameter counts\n"
            "(total 285,336,576, active 134,341,632). The section 5.1 tolerance\n"
            "band is derived from these counts, so a machine that cannot\n"
            "reproduce them must not launch a session.\n"
        ),
        _code(
            "import json, shutil, torch\n"
            "\n"
            "env = {\n"
            "    'torch': torch.__version__,\n"
            "    'cuda_available': torch.cuda.is_available(),\n"
            "    'cuda_version': torch.version.cuda,\n"
            "    'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,\n"
            "    'gpus': [\n"
            "        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())\n"
            "    ],\n"
            "    'host_ram_gb': round(shutil.disk_usage('/').total / 2**30, 1),\n"
            "    'free_disk_gb': round(shutil.disk_usage(os.getcwd()).free / 2**30, 1),\n"
            "}\n"
            "print(json.dumps(env, indent=2))"
        ),
        _code(
            "from act.model import DECLARED_V0_CONFIG, estimated_parameter_count\n"
            "\n"
            "total, active = estimated_parameter_count(DECLARED_V0_CONFIG)\n"
            "assert total == 285_336_576, f'declared total parameters drifted: {total}'\n"
            "assert active == 134_341_632, f'declared active parameters drifted: {active}'\n"
            "print(f'declared V0 parameters: total={total:,} active={active:,}')"
        ),
        _code(
            "from act.training.session_plan import (\n"
            "    CHECKPOINT_PAYLOAD_GB,\n"
            "    MAX_CHECKPOINTS,\n"
            "    PairSchedule,\n"
            "    decide_dtype,\n"
            "    working_dir_headroom_gb,\n"
            ")\n"
            "\n"
            "# The declared schedule at the fp32 floor (section 5.2's registered row).\n"
            "floor = PairSchedule(tokens_per_second=2_500.0)\n"
            "print('fp32 floor pair time (h):', round(floor.pair_time_hours, 2))\n"
            "print('full pairs per session:', floor.full_pairs_per_session)\n"
            "print('primary sessions:', floor.sessions_for_primary,\n"
            "      '| total incl. secondary:', floor.sessions_total)\n"
            "print('checkpoint payload GB:', CHECKPOINT_PAYLOAD_GB,\n"
            "      '| max_checkpoints pin:', MAX_CHECKPOINTS)\n"
            "print('working-dir headroom GB:', round(working_dir_headroom_gb(), 2))\n"
            "# The dtype rule is pre-committed; the session runner re-derives it\n"
            "# from the benchmark artifact and refuses a stale one.\n"
            "print('dtype at 2,500 tokens/s:', decide_dtype(2_500.0))\n"
            "print('dtype below floor:', decide_dtype(2_499.0))"
        ),
        _record_cell(
            '00-environment.json',
            "{'notebook': '00_environment', 'torch': env['torch'],\n"
            " 'gpu_count': env['gpu_count'],\n"
            " 'declared_total_parameters': total,\n"
            " 'declared_active_parameters': active,\n"
            " 'dtype_rule_floor_tokens_per_second': 2500.0}",
        ),
    ]
    return _notebook(
        "00 - Environment",
        "Plan section 39: verify CUDA, GPUs, RAM, disk, PyTorch, and the declared "
        "V0 parameter contract before any session starts.",
        cells,
    )


def notebook_01_dense_baseline() -> dict:
    """Plan section 104 `01_dense_baseline`: train, resume, and verify.

    Proves the three claims the plan asks of the first notebook: the model
    trains, a checkpoint resumes parameter-for-parameter, and metrics are
    reproducible (a second training run from the same seed is identical).
    """
    cells = [
        _markdown(
            "# Dense baseline (plan section 104)\n"
            "\n"
            "Proves the first notebook's three claims: the model trains,\n"
            "checkpoint resume is parameter-for-parameter, and metrics are\n"
            "reproducible from a fixed seed. Uses the tiny deterministic gate\n"
            "model and the shared runtime - not the declared 285M config.\n"
        ),
        _code(
            "import torch\n"
            "from act.model import DenseBaselineConfig, DenseCausalLM\n"
            "from act.training.runtime import (\n"
            "    TrainingConfig,\n"
            "    TrainingState,\n"
            "    load_checkpoint,\n"
            "    make_optimizer,\n"
            "    save_checkpoint,\n"
            "    train_steps,\n"
            ")\n"
            "\n"
            "CHECKPOINT = 'artifacts/dense-baseline.pt'\n"
            "STEPS = 8\n"
            "\n"
            "torch.manual_seed(19)\n"
            "model = DenseCausalLM(DenseBaselineConfig())\n"
            "optimizer = make_optimizer(model, TrainingConfig())\n"
            "state = TrainingState()\n"
            "losses = train_steps(model, optimizer, state, TrainingConfig(), STEPS)\n"
            "print('steps', state.step, '| tokens', state.tokens,\n"
            "      '| final loss', round(losses[-1], 6))"
        ),
        _code(
            "# Claim 2: a checkpoint resumes parameter-for-parameter.\n"
            "save_checkpoint(CHECKPOINT, model, optimizer, state)\n"
            "resumed_model = DenseCausalLM(DenseBaselineConfig())\n"
            "resumed_optimizer = make_optimizer(resumed_model, TrainingConfig())\n"
            "resumed_state = load_checkpoint(CHECKPOINT, resumed_model, resumed_optimizer)\n"
            "same = all(\n"
            "    torch.equal(a.cpu(), b.cpu())\n"
            "    for a, b in zip(model.state_dict().values(), resumed_model.state_dict().values())\n"
            ")\n"
            "assert same, 'resume did not reproduce parameters exactly'\n"
            "assert resumed_state.step == state.step\n"
            "print('resume reproduced', len(resumed_model.state_dict()),\n"
            "      'tensors at step', resumed_state.step)"
        ),
        _code(
            "# Claim 3: a fresh run from the same seed is metric-identical.\n"
            "torch.manual_seed(19)\n"
            "fresh_model = DenseCausalLM(DenseBaselineConfig())\n"
            "fresh_optimizer = make_optimizer(fresh_model, TrainingConfig())\n"
            "fresh_losses = train_steps(\n"
            "    fresh_model, fresh_optimizer, TrainingState(), TrainingConfig(), STEPS\n"
            ")\n"
            "assert fresh_losses == losses, 'training is not seed-reproducible'\n"
            "print('fresh run reproduced every loss exactly')"
        ),
        _record_cell(
            '01-dense-baseline.json',
            "{'notebook': '01_dense_baseline', 'steps': STEPS,\n"
            " 'final_loss': losses[-1], 'tokens': state.tokens,\n"
            " 'checkpoint': CHECKPOINT,\n"
            " 'resume_reproduced': same, 'seed_reproducible': True}",
        ),
    ]
    return _notebook(
        "01 - Dense Baseline",
        "Plan section 104: the dense baseline trains, resumes, and reproduces "
        "its metrics from seed.",
        cells,
    )


def notebook_02_random_topology_churn() -> dict:
    """Plan section 104 `02_random_topology_churn`: the mutation engine works.

    Applies grow/prune mutations directly through the topology operators,
    verifies expert-ID weight preservation, hashes, and the registry round
    trip - the claims the plan asks of the second notebook.
    """
    cells = [
        _markdown(
            "# Random topology churn (plan section 104)\n"
            "\n"
            "Proves the second notebook's claims: the topology mutation engine\n"
            "works, mutation cost is measurable, and the random control is\n"
            "reproducible. Mutations are applied directly through the\n"
            "prune/grow operators on a real genome, then rebuilt\n"
            "weight-preserving into a live model.\n"
        ),
        _code(
            "from act.model import model_from_genome\n"
            "from act.topology import (\n"
            "    create_genome,\n"
            "    feasible_expert_mutations,\n"
            "    genome_hash,\n"
            "    grow_expert,\n"
            "    prune_expert,\n"
            ")\n"
            "\n"
            "genome = create_genome(42, hidden_size=24, vocab_size=32, depth=1,\n"
            "                       ffn='moe', expert_count=4, active_experts=2)\n"
            "model = model_from_genome(genome, context_size=12)\n"
            "hash_before = genome_hash(genome)\n"
            "print('feasible mutations:', feasible_expert_mutations(genome)[:4], '...')"
        ),
        _code(
            "# Grow one expert: the operator returns the new expert's ID, the\n"
            "# genome hash changes, and the layer gains exactly one expert.\n"
            "layer = genome.layers[0]\n"
            "count_before = len(layer.experts)\n"
            "added_id = grow_expert(genome, layer.id, step=12)\n"
            "assert added_id is not None, 'grow was not feasible on a fresh genome'\n"
            "assert added_id == f\"{layer.id}E{count_before}\"\n"
            "assert genome_hash(genome) != hash_before\n"
            "assert len(genome.layers[0].experts) == count_before + 1\n"
            "print('grew', added_id, '->', len(genome.layers[0].experts), 'experts')"
        ),
        _code(
            "# Rebuild the mutated model: by-expert-ID weight transfer must retain\n"
            "# every existing expert's weights and report more parameters.\n"
            "import torch\n"
            "from act.model import rebuild_model_from_genome\n"
            "\n"
            "params_before = model.parameter_count\n"
            "grown_model = rebuild_model_from_genome(model, genome, context_size=12)\n"
            "assert grown_model.parameter_count > params_before\n"
            "retained = grown_model.state_dict()[\n"
            "    'blocks.0.experts.0.0.weight']\n"
            "source = model.state_dict()['blocks.0.experts.0.0.weight']\n"
            "assert torch.equal(retained, source), 'expert weights did not survive'\n"
            "model = grown_model\n"
            "print('rebuilt:', model.parameter_count, 'parameters (was', params_before, ')')"
        ),
        _code(
            "# Prune the added expert back and verify the hash round trip.\n"
            "assert prune_expert(genome, layer.id, added_id)\n"
            "assert genome_hash(genome) == hash_before, 'prune did not restore the genome'\n"
            "print('prune restored hash', hash_before)"
        ),
        _code(
            "# Mutation cost is measurable: every mutation rebuilds the model and\n"
            "# books parameter_count x 4 rebuild FLOPs into search, while the fixed\n"
            "# arm books none. The registry round trip: one experiment, one record.\n"
            "from act.evolution.experiment import ExperimentConfig, run_experiment\n"
            "\n"
            "result = run_experiment(ExperimentConfig(\n"
            "    seed=42, steps=48, arms='fixed,random', device='cpu',\n"
            "))\n"
            "random_arm = next(arm for arm in result.arms if arm.mode == 'random')\n"
            "fixed_arm = next(arm for arm in result.arms if arm.mode == 'fixed')\n"
            "assert random_arm.mutation_count == 3\n"
            "assert fixed_arm.mutation_count == 0 and fixed_arm.search_flops == 0\n"
            "assert random_arm.search_flops > 0  # 3 rebuilds booked\n"
            "assert len(random_arm.mutation_history) == 3\n"
            "print('random arm:', random_arm.mutation_count, 'mutations,',\n"
            "      random_arm.training_flops, 'training FLOPs,',\n"
            "      random_arm.search_flops, 'search FLOPs')"
        ),
        _code(
            "# Registry round trip: one experiment appends exactly one JSONL record.\n"
            "import json\n"
            "from act.evolution.registry import append_result\n"
            "\n"
            "registry = 'artifacts/notebook-records/experiments.jsonl'\n"
            "record = append_result(registry, result)\n"
            "lines = Path(registry).read_text().strip().splitlines()\n"
            "assert lines[-1] == json.dumps(record, sort_keys=True)\n"
            "print('appended', record['experiment_id'], 'to', registry)"
        ),
        _record_cell(
            '02-random-churn.json',
            "{'notebook': '02_random_topology_churn',\n"
            " 'mutations': random_arm.mutation_count,\n"
            " 'training_flops': random_arm.training_flops,\n"
            " 'architecture_hash': random_arm.architecture_hash,\n"
            " 'registry_round_trip': True}",
        ),
    ]
    return _notebook(
        "02 - Random Topology Churn",
        "Plan section 104: the mutation engine works, cost is measurable, and "
        "the random control is reproducible.",
        cells,
    )


def notebook_03_adaptive_topology_v0() -> dict:
    """Plan section 104 `03_adaptive_topology_v0`: the sole Gate 3 question.

    Compares adaptive probe-loss selection against exactly matched random
    churn on real seeds through the gate-study machinery, then reports the
    objective verdict (exact binomial sign test) the same way every study in
    the factor map was decided.
    """
    cells = [
        _markdown(
            "# Adaptive topology vs matched random churn (plan section 104)\n"
            "\n"
            "The sole question of the third notebook: does information-guided\n"
            "topology evolution beat exactly the same amount of random topology\n"
            "evolution? The gate-study machinery pairs seeds, enforces matched\n"
            "mutation budgets (it aborts otherwise), and emits the objective\n"
            "verdict - the exact one-tailed binomial sign test used by every\n"
            "factor-map study.\n"
            "\n"
            "This notebook runs 4 seeds so it finishes in minutes - far below the\n"
            "384-seed powered studies in the factor map, so its verdict is a\n"
            "demo of the machinery, not a Gate 3 claim. The versioned studies\n"
            "live in `artifacts/study-summaries/`; the same CLI reproduces them\n"
            "(`PYTHONPATH=src python -m act.evolution.cli --seeds ...`).\n"
        ),
        _code(
            "from act.evolution.experiment import ExperimentConfig\n"
            "from act.evolution.study import run_gate_study\n"
            "\n"
            "config = ExperimentConfig(\n"
            "    steps=48,\n"
            "    mutation_interval=12,\n"
            "    mutation_budget=3,\n"
            "    adaptive_policy='probe_loss',\n"
            "    probe_candidate_cap=4,\n"
            "    probe_batch_count=2,\n"
            "    arms='random,adaptive',\n"
            "    device='cpu',\n"
            ")\n"
            "seeds = (1, 2, 3, 42)\n"
            "study = run_gate_study(config, seeds=seeds, treatment='adaptive')"
        ),
        _code(
            "print('wins:', study.adaptive_wins, '/', len(study.seeds))\n"
            "print('mean delta:', round(study.mean_adaptive_delta, 6))\n"
            "print('matched budgets:', study.matched_mutation_budget)\n"
            "print('min wins to pass at alpha=0.05:', study.gate_min_wins)\n"
            "print('binomial p:', study.gate_p_value)\n"
            "print('gate passed:', study.gate_passed,\n"
            "      '(4 seeds - a demo verdict, not the powered 384-seed study)')"
        ),
        _code(
            "# Persist the verdict exactly as every gate study records it.\n"
            "from act.evolution.registry import append_study\n"
            "\n"
            "study_record = append_study(\n"
            "    'artifacts/notebook-records/notebook-gate-study.jsonl', study\n"
            ")\n"
            "print('study', study_record['study_id'], '-> gate_passed:',\n"
            "      study_record['gate_passed'])"
        ),
        _record_cell(
            '03-adaptive-v0.json',
            "{'notebook': '03_adaptive_topology_v0', 'seeds': list(seeds),\n"
            " 'wins': study.adaptive_wins,\n"
            " 'mean_delta': study.mean_adaptive_delta,\n"
            " 'gate_min_wins': study.gate_min_wins,\n"
            " 'gate_p_value': study.gate_p_value,\n"
            " 'gate_passed': study.gate_passed,\n"
            " 'study_id': study_record['study_id']}",
        ),
    ]
    return _notebook(
        "03 - Adaptive Topology V0",
        "Plan section 104: information-guided topology evolution versus exactly "
        "matched random churn - the Gate 3 question at notebook fidelity.",
        cells,
    )


def notebook_04_importance() -> dict:
    """Plan section 39 `04_measurement`: validate the importance system.

    Extracts the real per-expert signals after a backward pass (the same
    records the experiment runner persists), measures a temporary ablation
    delta, and writes the append-only lifecycle fossil trail.
    """
    cells = [
        _markdown(
            "# Importance-system measurement (plan section 39)\n"
            "\n"
            "Validates that the measurable signals exist and behave: per-expert\n"
            "routing load, gradient norm, parameter norm, and temporal stability\n"
            "extracted after a real backward pass, a temporary-ablation loss\n"
            "delta, and the append-only lifecycle fossil ledger the studies\n"
            "serialize.\n"
        ),
        _code(
            "import torch\n"
            "from act.model import StaticMoEConfig, StaticMoECausalLM\n"
            "from act.importance import (\n"
            "    LifecycleLedger,\n"
            "    expert_ablation_delta,\n"
            "    measure_expert_signals,\n"
            ")\n"
            "\n"
            "model = StaticMoECausalLM(StaticMoEConfig(\n"
            "    vocab_size=32, context_size=12, hidden_size=24, num_layers=2,\n"
            "    num_heads=4, ffn_size=24, num_experts=4, active_experts=2,\n"
            "))\n"
            "tokens = torch.randint(0, 32, (2, 12))\n"
            "loss = model.loss(tokens)\n"
            "loss.backward()  # gradient signals exist only after a real step\n"
            "print('training loss:', round(float(loss.detach()), 4))\n"
            "signals = measure_expert_signals(model)\n"
            "assert len(signals) == 8  # 2 layers x 4 experts\n"
            "first = signals[0]\n"
            "print(f\"{first.layer_id}/{first.expert_id}: load={first.routing_load:.3f} \"\n"
            "      f\"grad={first.gradient_norm:.4f} param={first.parameter_norm:.3f}\")"
        ),
        _code(
            "# Temporary ablation: zeroing one expert moves the loss by a measured\n"
            "# delta (positive or negative at toy scale - an unused expert can be\n"
            "# harmless) and the parameters must be restored afterwards.\n"
            "baseline = float(model.loss(tokens).detach())\n"
            "delta = expert_ablation_delta(model, tokens, 'L0E0')\n"
            "restored = float(model.loss(tokens).detach())\n"
            "assert restored == baseline, 'ablation did not restore parameters'\n"
            "print('ablation delta for L0E0:', round(delta, 6),\n"
            "      '| baseline restored:', restored == baseline)"
        ),
        _code(
            "# Lifecycle fossils: the append-only record of every accepted mutation.\n"
            "from act.topology import create_genome, genome_hash, grow_expert\n"
            "\n"
            "genome = create_genome(42, hidden_size=24, vocab_size=32, depth=1,\n"
            "                       ffn='moe', expert_count=4, active_experts=2)\n"
            "hash_before = genome_hash(genome)\n"
            "added = grow_expert(genome, genome.layers[0].id, step=12)\n"
            "assert added is not None\n"
            "ledger = LifecycleLedger()\n"
            "fossil = ledger.record(\n"
            "    step=12, operation='grow', expert_id=added, layer_id='L0',\n"
            "    architecture_hash_before=hash_before,\n"
            "    architecture_hash_after=genome_hash(genome),\n"
            "    signal=first,\n"
            ")\n"
            "assert fossil.signal['expert_id'] == first.expert_id\n"
            "print('fossil:', fossil.operation, fossil.expert_id,\n"
            "      '| ledger lines:', len(ledger.to_jsonl().splitlines()))"
        ),
        _record_cell(
            '04-importance.json',
            "{'notebook': '04_importance',\n"
            " 'signals': len(signals),\n"
            " 'ablation_delta': delta,\n"
            " 'ablation_restored': restored == baseline,\n"
            " 'ablation_finite': delta == delta and abs(delta) != float('inf'),\n"
            " 'fossils': len(ledger.records)}",
        ),
    ]
    return _notebook(
        "04 - Importance Measurement",
        "Plan section 39: the importance system produces real, device-safe, "
        "restoring signals and fossils.",
        cells,
    )


def notebook_05_adaptive() -> dict:
    """Plan section 39 `05_adaptive`: the signal-informed controller.

    Runs the experiment with all three arms (fixed control, random churn,
    signal-informed adaptive) so the controller's decisions are visible in
    the mutation history, then reports the comparable per-arm ablations.
    """
    cells = [
        _markdown(
            "# Adaptive topology evolution (plan section 39)\n"
            "\n"
            "The actual topology-evolving model: three arms in one matched-\n"
            "budget experiment - the fixed control, matched random churn, and\n"
            "the adaptive controller deciding mutations from measurable\n"
            "signals. The comparable per-arm ablation report is the notebook's\n"
            "scientific output.\n"
        ),
        _code(
            "from act.evolution.experiment import ExperimentConfig, run_experiment\n"
            "\n"
            "result = run_experiment(ExperimentConfig(\n"
            "    seed=42, steps=48, arms='fixed,random,adaptive', device='cpu',\n"
            "))\n"
            "arms = {arm.mode: arm for arm in result.arms}\n"
            "assert arms['fixed'].mutation_count == 0\n"
            "assert arms['random'].mutation_count == arms['adaptive'].mutation_count\n"
            "for arm in result.arms:\n"
            "    print(f\"{arm.mode:>8}: loss={arm.validation_loss:.4f} \"\n"
            "          f\"mutations={arm.mutation_count} params={arm.parameter_count}\")"
        ),
        _code(
            "# The adaptive arm's mutation history names the policy behind each\n"
            "# structural event, so every decision is inspectable afterwards.\n"
            "adaptive = arms['adaptive']\n"
            "assert all(event['policy'] for event in adaptive.mutation_history)\n"
            "for event in adaptive.mutation_history:\n"
            "    print(event['step'], event['policy'], event['operation'])"
        ),
        _code(
            "# Comparable reports: efficiency against total FLOPs and ablation\n"
            "# deltas against the fixed control.\n"
            "from act.evaluation.reports import ablation_report, efficiency_report\n"
            "\n"
            "efficiency = efficiency_report(result)\n"
            "ablation = ablation_report(result, reference_mode='fixed')\n"
            "assert {row['mode'] for row in efficiency['rows']} == {'fixed', 'random', 'adaptive'}\n"
            "assert {row['mode'] for row in ablation['comparisons']} == {'random', 'adaptive'}\n"
            "print('adaptive - random:', round(efficiency['adaptive_minus_random'], 6))\n"
            "for row in ablation['comparisons']:\n"
            "    print(row['mode'], 'delta:', round(row['validation_loss_delta'], 6))"
        ),
        _record_cell(
            '05-adaptive.json',
            "{'notebook': '05_adaptive',\n"
            " 'adaptive_minus_random': efficiency['adaptive_minus_random'],\n"
            " 'rows': efficiency['rows'],\n"
            " 'comparisons': ablation['comparisons']}",
        ),
    ]
    return _notebook(
        "05 - Adaptive",
        "Plan section 39: the signal-informed controller mutates a live model "
        "under matched budgets, with comparable per-arm reports.",
        cells,
    )


def notebook_06_search() -> dict:
    """Plan section 39 `06_search`: candidate generation and selection.

    Exercises the full search stack at toy fidelity: module/depth variant
    generation, prior-informed candidate ranking, the matched-budget
    frontier run, and deterministic Pareto population selection over the\n    measured quality/parameter/FLOPs trade-offs.
    """
    cells = [
        _markdown(
            "# Population search (plan section 39)\n"
            "\n"
            "Generate and evaluate candidate genomes: one-step module/depth\n"
            "variants in stable order, prior-informed candidate ranking with\n"
            "explicit depth penalties, the matched-budget static-vs-adaptive\n"
            "frontier run, and deterministic Pareto selection over measured\n"
            "quality, active parameters, and search FLOPs.\n"
        ),
        _code(
            "from act.evolution.module_depth import enumerate_module_depth_variants\n"
            "from act.evolution.hierarchical import SearchPrior, rank_module_candidates\n"
            "from act.topology import create_genome\n"
            "\n"
            "genome = create_genome(42, hidden_size=24, vocab_size=32, depth=2,\n"
            "                       ffn='moe', expert_count=4, active_experts=2)\n"
            "variants = enumerate_module_depth_variants(genome)\n"
            "assert len(variants) == 6  # 2 layers x (2 module switches + 1 depth toggle)\n"
            "assert len({v.architecture_hash for _, v in variants}) == len(variants)\n"
            "# every variant is a distinct architecture\n"
            "prior = SearchPrior(module_scores={'attention': 0.9, 'gdn': 0.4},\n"
            "                    depth_penalty=0.1)\n"
            "ranked = rank_module_candidates(genome, prior)\n"
            "assert ranked[0].module == 'attention'  # highest prior wins\n"
            "print('variants:', [(v.operation, v.target, v.module) for _, v in variants[:3]],\n"
            "      '...')\n"
            "print('top-ranked:', ranked[0].layer_id, ranked[0].module,\n"
            "      round(ranked[0].score, 4))"
        ),
        _code(
            "# Matched-budget frontier: static MoE control versus adaptive MoE.\n"
            "from act.evolution.experiment import ExperimentConfig\n"
            "from act.evolution.frontier import run_frontier\n"
            "\n"
            "points = run_frontier(\n"
            "    ExperimentConfig(seed=42, steps=48, arms='fixed,adaptive', device='cpu'),\n"
            "    seeds=(1,),\n"
            ")\n"
            "assert len(points) == 2\n"
            "for point in points:\n"
            "    print(f\"{point.family}: loss={point.validation_loss:.4f} \"\n"
            "          f\"active={point.active_parameter_count} vflops={point.validation_flops}\")"
        ),
        _code(
            "# Deterministic Pareto selection over measured trade-offs: each\n"
            "# frontier point becomes a candidate over (quality, active params,\n"
            "# search FLOPs) with its own genome.\n"
            "from act.evolution.pareto import (\n"
            "    PopulationCandidate,\n"
            "    pareto_frontier,\n"
            "    select_population,\n"
            ")\n"
            "\n"
            "point_genomes = (\n"
            "    genome,\n"
            "    create_genome(7, hidden_size=24, vocab_size=32, depth=2,\n"
            "                  ffn='moe', expert_count=4, active_experts=2),\n"
            ")\n"
            "candidates = tuple(\n"
            "    PopulationCandidate(\n"
            "        genome=point_genome,\n"
            "        quality=-point.validation_loss,\n"
            "        active_params=point.active_parameter_count,\n"
            "        search_flops=point.validation_flops,\n"
            "    )\n"
            "    for point, point_genome in zip(points, point_genomes)\n"
            ")\n"
            "frontier = pareto_frontier(candidates)\n"
            "assert frontier, 'the frontier must not be empty'\n"
            "population = select_population(candidates, size=2)\n"
            "assert len(population) == 2\n"
            "print('frontier size:', len(frontier),\n"
            "      '| selected:', [c.architecture_hash[:8] for c in population])"
        ),
        _record_cell(
            '06-search.json',
            "{'notebook': '06_search',\n"
            " 'variant_count': len(variants),\n"
            " 'top_ranked': {'layer': ranked[0].layer_id, 'module': ranked[0].module},\n"
            " 'frontier_size': len(frontier),\n"
            " 'selected_hashes': [c.architecture_hash for c in population]}",
        ),
    ]
    return _notebook(
        "06 - Population Search",
        "Plan section 39: variants, prior-ranked candidates, the matched-budget "
        "frontier, and Pareto population selection.",
        cells,
    )


def notebook_07_extract() -> dict:
    """Plan section 39 `07_extract`: freeze, extract, publish.

    The evolved architecture is frozen: the genome serializes to and from a\n    stable hash, the checkpoint packages model plus genome plus mutation\n    history in the versioned format, and the artifact set is published\n    through the checksummed Kaggle manifest contract (plan section 40's\n    end-of-session publish).\n    """
    cells = [
        _markdown(
            "# Freeze and extract (plan section 39)\n"
            "\n"
            "The evolved architecture becomes a portable artifact: the genome\n"
            "round-trips its stable hash through serialization, the versioned\n"
            "checkpoint packages model + optimizer + genome + mutation history,\n"
            "and the session publishes through the checksummed manifest\n"
            "contract (plan section 40: save, then publish output).\n"
        ),
        _code(
            "import torch\n"
            "from act.topology import (\n"
            "    ArchitectureGenome,\n"
            "    create_genome,\n"
            "    genome_hash,\n"
            "    grow_expert,\n"
            ")\n"
            "from act.model import model_from_genome, rebuild_model_from_genome\n"
            "\n"
            "genome = create_genome(42, hidden_size=24, vocab_size=32, depth=1,\n"
            "                       ffn='moe', expert_count=4, active_experts=2)\n"
            "model = model_from_genome(genome, context_size=12)\n"
            "added = grow_expert(genome, genome.layers[0].id, step=12)\n"
            "model = rebuild_model_from_genome(model, genome, context_size=12)\n"
            "assert added is not None\n"
            "hash_before = genome_hash(genome)\n"
            "restored = ArchitectureGenome.from_dict(genome.to_dict())\n"
            "assert genome_hash(restored) == hash_before, 'genome hash did not round trip'\n"
            "print('evolved genome', hash_before, '| experts:',\n"
            "      len(genome.layers[0].experts))"
        ),
        _code(
            "# Freeze the full resumable state in the versioned checkpoint format.\n"
            "from act.training.runtime import (\n"
            "    TrainingConfig,\n"
            "    TrainingState,\n"
            "    load_checkpoint,\n"
            "    make_optimizer,\n"
            "    save_checkpoint,\n"
            ")\n"
            "\n"
            "CHECKPOINT = 'artifacts/notebook-records/07-extracted.pt'\n"
            "optimizer = make_optimizer(model, TrainingConfig())\n"
            "state = TrainingState(\n"
            "    step=12,\n"
            "    tokens=288,\n"
            "    genome=genome.to_dict(),\n"
            "    mutation_history=[\n"
            "        {'step': 12, 'operation': 'grow', 'expert_id': added},\n"
            "    ],\n"
            ")\n"
            "save_checkpoint(CHECKPOINT, model, optimizer, state)\n"
            "payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)\n"
            "assert payload['version'] == 3\n"
            "assert payload['training']['genome'] == genome.to_dict()\n"
            "print('frozen', CHECKPOINT, '| genome + mutation history persisted')"
        ),
        _code(
            "# Publish the session's artifacts behind the checksummed manifest.\n"
            "import json\n"
            "from pathlib import Path\n"
            "from act.ops import kaggle_layout, publish_artifacts, restore_artifacts\n"
            "\n"
            "report = Path('artifacts/notebook-records/07-extract-summary.json')\n"
            "report.write_text(json.dumps({\n"
            "    'architecture_hash': genome_hash(genome),\n"
            "    'mutation_history': state.mutation_history,\n"
            "}, sort_keys=True) + '\\n')\n"
            "layout = kaggle_layout('artifacts/notebook-records/07-kaggle')\n"
            "manifest = publish_artifacts(\n"
            "    layout,\n"
            "    (CHECKPOINT, report),\n"
            "    dataset_id='notebook-07-extract',\n"
            "    code_revision='notebook',\n"
            "    config_hash=genome_hash(genome),\n"
            ")\n"
            "assert Path(layout.manifests / 'manifest.json').is_file()\n"
            "restored_files = restore_artifacts(layout)\n"
            "assert len(restored_files) == 2\n"
            "print('published:', manifest.artifacts)"
        ),
        _record_cell(
            '07-extract.json',
            "{'notebook': '07_extract',\n"
            " 'architecture_hash': genome_hash(genome),\n"
            " 'checkpoint_version': payload['version'],\n"
            " 'published': list(manifest.artifacts)}",
        ),
    ]
    return _notebook(
        "07 - Extract",
        "Plan section 39: freeze the evolved architecture and publish it "
        "through the checksummed manifest contract.",
        cells,
    )


def notebook_08_retrain() -> dict:
    """Plan section 39 `08_retrain`: fresh-initialization training.

    Compares the explicit initialization policies - fresh, rewound, and\n    teacher - on the same evolved genome, reporting the exact copied/skipped\n    parameter counts the retraining comparisons are built on.
    """
    cells = [
        _markdown(
            "# Fresh-initialization retraining (plan section 39)\n"
            "\n"
            "The same evolved genome retrained from different initializations:\n"
            "``fresh`` copies nothing, ``rewound`` restores a pre-mutation\n"
            "checkpoint, and ``teacher`` transfers an existing model's\n"
            "compatible tensors. Each policy reports its exact copied/skipped\n"
            "parameter counts - the numbers the retraining comparisons record.\n"
        ),
        _code(
            "from act.model import model_from_genome, rebuild_model_from_genome\n"
            "from act.topology import create_genome, genome_hash, grow_expert\n"
            "from act.training.retraining import initialize_retraining\n"
            "\n"
            "genome = create_genome(42, hidden_size=24, vocab_size=32, depth=1,\n"
            "                       ffn='moe', expert_count=4, active_experts=2)\n"
            "base = model_from_genome(genome, context_size=12)\n"
            "rewound_state = {k: v.clone() for k, v in base.state_dict().items()}\n"
            "added = grow_expert(genome, genome.layers[0].id, step=12)\n"
            "assert added is not None\n"
            "evolved = rebuild_model_from_genome(base, genome, context_size=12)\n"
            "print('evolved genome:', genome_hash(genome), '| added expert:', added)"
        ),
        _code(
            "# Apply every policy to the same evolved model.\n"
            "reports = {}\n"
            "reports['fresh'] = initialize_retraining(evolved, mode='fresh')\n"
            "reports['rewound'] = initialize_retraining(\n"
            "    evolved, mode='rewound', rewind_state=rewound_state\n"
            ")\n"
            "reports['teacher'] = initialize_retraining(\n"
            "    evolved, mode='teacher', source=base\n"
            ")\n"
            "assert reports['fresh'].copied_parameters == 0\n"
            "# The grown expert has no pre-mutation source: rewound/teacher must\n"
            "# skip exactly its tensors and copy the rest.\n"
            "assert (\n"
            "    reports['rewound'].copied_parameters\n"
            "    == reports['teacher'].copied_parameters\n"
            "    > 0\n"
            ")\n"
            "assert reports['rewound'].skipped_parameters > 0\n"
            "for mode, report in reports.items():\n"
            "    print(f\"{mode:>8}: copied={report.copied_parameters} \"\n"
            "          f\"skipped={report.skipped_parameters}\")"
        ),
        _code(
            "# Each initialized model trains under the shared runtime.\n"
            "from act.training.runtime import (\n"
            "    TrainingConfig,\n"
            "    TrainingState,\n"
            "    make_optimizer,\n"
            "    train_steps,\n"
            ")\n"
            "\n"
            "train_config = TrainingConfig(sequence_length=12)\n"
            "losses_by_mode = {}\n"
            "for mode in ('fresh', 'rewound', 'teacher'):\n"
            "    train_model = rebuild_model_from_genome(base, genome, context_size=12)\n"
            "    if mode == 'rewound':\n"
            "        initialize_retraining(\n"
            "            train_model, mode='rewound', rewind_state=rewound_state\n"
            "        )\n"
            "    elif mode == 'teacher':\n"
            "        initialize_retraining(train_model, mode='teacher', source=base)\n"
            "    optimizer = make_optimizer(train_model, train_config)\n"
            "    losses = train_steps(\n"
            "        train_model, optimizer, TrainingState(), train_config, 8\n"
            "    )\n"
            "    losses_by_mode[mode] = losses[-1]\n"
            "assert set(losses_by_mode) == {'fresh', 'rewound', 'teacher'}\n"
            "for mode, final in losses_by_mode.items():\n"
            "    print(f\"{mode:>8}: final loss {final:.4f}\")"
        ),
        _record_cell(
            '08-retrain.json',
            "{'notebook': '08_retrain',\n"
            " 'evolved_hash': genome_hash(genome),\n"
            " 'reports': {mode: {\n"
            "     'copied': report.copied_parameters,\n"
            "     'skipped': report.skipped_parameters,\n"
            " } for mode, report in reports.items()},\n"
            " 'final_losses': losses_by_mode}",
        ),
    ]
    return _notebook(
        "08 - Retrain",
        "Plan section 39: fresh, rewound, and teacher initializations compared "
        "on the same evolved genome.",
        cells,
    )


def notebook_09_compare() -> dict:
    """Plan section 39 `09_compare`: final scientific analysis.

    Cross-seed stability over the gate study's paired deltas (mean,\n    standard deviation, range), the strict architecture-equivalence check,\n    and a compact markdown factor summary - the shape of the paper's final\n    comparison table.
    """
    cells = [
        _markdown(
            "# Final comparison (plan section 39)\n"
            "\n"
            "The scientific analysis notebook: a multi-seed gate study reduced\n"
            "to its cross-seed stability summary (mean, standard deviation,\n"
            "range of the paired deltas) and the strict architecture-hash\n"
            "equivalence check, rendered as the compact markdown table shape\n"
            "the final report uses.\n"
        ),
        _code(
            "from act.evolution.experiment import ExperimentConfig\n"
            "from act.evolution.study import run_gate_study\n"
            "\n"
            "study = run_gate_study(\n"
            "    ExperimentConfig(\n"
            "        steps=48, arms='random,adaptive', device='cpu',\n"
            "    ),\n"
            "    seeds=(1, 2, 3, 42),\n"
            "    treatment='adaptive',\n"
            ")\n"
            "assert study.matched_mutation_budget\n"
            "print('wins:', study.adaptive_wins, '/', len(study.seeds),\n"
            "      '| mean delta:', round(study.mean_adaptive_delta, 6))"
        ),
        _code(
            "# Cross-seed stability of the paired deltas.\n"
            "from act.evaluation.stability import (\n"
            "    architecture_equivalence,\n"
            "    summarize_stability,\n"
            ")\n"
            "\n"
            "deltas = tuple(\n"
            "    next(arm for arm in experiment.arms if arm.mode == 'adaptive').validation_loss\n"
            "    - next(arm for arm in experiment.arms if arm.mode == 'random').validation_loss\n"
            "    for experiment in study.experiments\n"
            ")\n"
            "summary = summarize_stability(deltas)\n"
            "assert summary.minimum <= summary.mean <= summary.maximum\n"
            "print('delta mean/stddev/min/max:',\n"
            "      [round(value, 6) for value in (\n"
            "          summary.mean, summary.stddev, summary.minimum, summary.maximum\n"
            "      )])"
        ),
        _code(
            "# Architecture equivalence: identical hashes when (and only when) the\n"
            "# architectures coincide - whichever way this seed's mutations land.\n"
            "hashes = tuple(arm.architecture_hash for arm in study.experiments[0].arms)\n"
            "assert len(hashes) == 2\n"
            "same = architecture_equivalence(hashes)\n"
            "assert same == (hashes[0] == hashes[1])\n"
            "assert not architecture_equivalence((hashes[0], '0' * 16))\n"
            "print('arm hashes:', [value[:8] for value in hashes],\n"
            "      '| equivalent:', same)"
        ),
        _code(
            "# The final table shape: one row per arm, objective statistics only.\n"
            "rows = [\n"
            "    '| metric | value |',\n"
            "    '|---|---|',\n"
            "    f\"| wins | {study.adaptive_wins}/{len(study.seeds)} |\",\n"
            "    f\"| mean delta | {study.mean_adaptive_delta:.6f} |\",\n"
            "    f\"| delta stddev | {study.delta_stddev:.6f} |\",\n"
            "    f\"| binomial p | {study.gate_p_value:.6g} |\",\n"
            "    f\"| gate passed | {study.gate_passed} |\",\n"
            "]\n"
            "print('\\n'.join(rows))"
        ),
        _record_cell(
            '09-compare.json',
            "{'notebook': '09_compare',\n"
            " 'wins': study.adaptive_wins,\n"
            " 'seeds': list(study.seeds),\n"
            " 'mean_delta': study.mean_adaptive_delta,\n"
            " 'delta_stddev': study.delta_stddev,\n"
            " 'gate_p_value': study.gate_p_value,\n"
            " 'gate_passed': study.gate_passed,\n"
            " 'delta_summary': {\n"
            "     'mean': summary.mean,\n"
            "     'stddev': summary.stddev,\n"
            "     'minimum': summary.minimum,\n"
            "     'maximum': summary.maximum,\n"
            " },\n"
            " 'arms_architecture_equivalent': same}",
        ),
    ]
    return _notebook(
        "09 - Compare",
        "Plan section 39: cross-seed stability, architecture equivalence, and "
        "the final comparison table.",
        cells,
    )


NOTEBOOKS = {
    "00_environment.ipynb": notebook_00_environment,
    "01_dense_baseline.ipynb": notebook_01_dense_baseline,
    "02_random_topology_churn.ipynb": notebook_02_random_topology_churn,
    "03_adaptive_topology_v0.ipynb": notebook_03_adaptive_topology_v0,
    "04_importance.ipynb": notebook_04_importance,
    "05_adaptive.ipynb": notebook_05_adaptive,
    "06_search.ipynb": notebook_06_search,
    "07_extract.ipynb": notebook_07_extract,
    "08_retrain.ipynb": notebook_08_retrain,
    "09_compare.ipynb": notebook_09_compare,
}


def build_notebooks(directory: Path = NOTEBOOK_DIR) -> dict[str, bytes]:
    """Render every notebook to canonical bytes (sorted keys, one trailing newline)."""
    rendered: dict[str, bytes] = {}
    for name, builder in NOTEBOOKS.items():
        notebook = builder()
        text = json.dumps(notebook, indent=1, sort_keys=True) + "\n"
        rendered[name] = text.encode("utf-8")
    return rendered


def write_notebooks(directory: Path = NOTEBOOK_DIR) -> list[Path]:
    """Write the rendered notebooks; returns the paths written."""
    directory.mkdir(parents=True, exist_ok=True)
    rendered = build_notebooks(directory)
    paths = []
    for name, payload in rendered.items():
        path = directory / name
        path.write_bytes(payload)
        paths.append(path)
    return paths


def notebooks_match(directory: Path = NOTEBOOK_DIR) -> bool:
    """True when the on-disk notebooks are byte-identical to a fresh build."""
    rendered = build_notebooks(directory)
    for name, payload in rendered.items():
        path = directory / name
        if not path.is_file() or path.read_bytes() != payload:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="verify the built notebooks are byte-identical instead of writing",
    )
    args = parser.parse_args()
    if args.check:
        if not notebooks_match():
            sys.exit(
                "notebooks drifted from the builder output; "
                "re-run scripts/build_notebooks.py"
            )
        print(f"checked {len(NOTEBOOKS)} notebooks: byte-identical")
        return
    for path in write_notebooks():
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
