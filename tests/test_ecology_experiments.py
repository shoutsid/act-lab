import pytest
import torch

from act.evolution import build_ecology_report


def test_ecology_report_builds_bounded_similarity_edges() -> None:
    report = build_ecology_report(
        {
            "L0E0": torch.tensor([1.0, 0.0]),
            "L0E1": torch.tensor([0.9, 0.1]),
            "L0E2": torch.tensor([0.0, 1.0]),
        },
        {"L0E0": 0.4, "L0E1": 0.3, "L0E2": 0.3},
        similarity_threshold=0.9,
    )
    assert [(edge.left, edge.right) for edge in report.edges] == [("L0E0", "L0E1")]
    assert report.node_loads["L0E2"] == 0.3


def test_ecology_report_rejects_missing_loads() -> None:
    with pytest.raises(ValueError, match="missing"):
        build_ecology_report({"L0E0": torch.ones(2)}, {})
