from act.evaluation import ablation_report, efficiency_report
from act.evolution import ExperimentConfig, run_experiment


def test_efficiency_report_contains_all_experiment_arms() -> None:
    result = run_experiment(ExperimentConfig(steps=2, validation_batches=1))
    report = efficiency_report(result)
    assert [row["mode"] for row in report["rows"]] == ["adaptive", "fixed", "random"]
    assert all(row["total_flops"] > 0 for row in report["rows"])


def test_ablation_report_compares_against_selected_reference() -> None:
    result = run_experiment(ExperimentConfig(steps=2, validation_batches=1))
    report = ablation_report(result, reference_mode="fixed")
    assert report["reference_mode"] == "fixed"
    assert len(report["comparisons"]) == 2
