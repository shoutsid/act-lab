import pytest

from act.evaluation import architecture_equivalence, summarize_stability


def test_stability_summary_reports_cross_seed_statistics() -> None:
    summary = summarize_stability([1.0, 2.0, 3.0])
    assert summary.mean == 2.0
    assert summary.minimum == 1.0
    assert summary.maximum == 3.0
    assert summary.stddev > 0.0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_stability_rejects_nonfinite_samples(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        summarize_stability([1.0, invalid])


def test_architecture_equivalence_is_strict() -> None:
    assert architecture_equivalence(["abc", "abc"])
    assert not architecture_equivalence(["abc", "def"])
    with pytest.raises(ValueError, match="must not be empty"):
        architecture_equivalence([])
    with pytest.raises(ValueError, match="empty values"):
        architecture_equivalence(["", ""])
