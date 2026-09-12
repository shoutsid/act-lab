"""Low-fidelity topology evolution experiments."""

from .experiment import ExperimentConfig, ExperimentResult, run_experiment
from .controller import AdaptiveController, ControllerDecision
from .pareto import PopulationCandidate, dominates, pareto_frontier, select_population
from .frontier import FrontierPoint, run_frontier
from .ecology import EcologyEdge, EcologyReport, build_ecology_report
from .registry import append_result, append_study, experiment_hash, study_hash
from .study import GateStudyResult, run_gate_study

__all__ = [
	"ExperimentConfig",
	"ExperimentResult",
	"AdaptiveController",
	"ControllerDecision",
	"PopulationCandidate",
	"dominates",
	"pareto_frontier",
	"select_population",
	"FrontierPoint",
	"run_frontier",
	"EcologyEdge",
	"EcologyReport",
	"build_ecology_report",
	"GateStudyResult",
	"append_result",
	"append_study",
	"experiment_hash",
	"run_gate_study",
	"run_experiment",
	"study_hash",
]