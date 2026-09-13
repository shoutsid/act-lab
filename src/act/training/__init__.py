"""Deterministic training and checkpoint utilities for ACT baselines."""

from .runtime import (
    TrainingConfig,
    TrainingState,
    load_checkpoint,
    make_optimizer,
    make_scheduler,
    move_optimizer_state,
    resolve_device,
    save_checkpoint,
    train_steps,
    validate_checkpoint_payload,
)
from .distributed import (
    DistributedConfig,
    distributed_config,
    initialize_distributed,
    wrap_distributed,
)
from .session import SessionCheckpointPolicy, retain_checkpoints
from .session_plan import (
	PairSchedule,
	SessionDecision,
	deadline_decision,
	decide_dtype,
	expected_probe_cost_percent,
	tolerance_band,
	working_dir_headroom_gb,
)
from .retraining import InitializationReport, initialize_retraining
from .v0_session import (
    V0ArmResult,
    V0SessionConfig,
    V0SessionResult,
    predicted_cost_ratio,
    run_v0_session,
)
from .v0_protocol import aggregate_arm_cost_ratio, harm_guard_rejects, sign_pairs

__all__ = [
	"TrainingConfig",
	"TrainingState",
	"load_checkpoint",
	"make_optimizer",
	"make_scheduler",
	"move_optimizer_state",
	"resolve_device",
	"save_checkpoint",
	"train_steps",
	"validate_checkpoint_payload",
	"DistributedConfig",
	"distributed_config",
	"initialize_distributed",
	"wrap_distributed",
	"SessionCheckpointPolicy",
	"retain_checkpoints",
	"PairSchedule",
	"SessionDecision",
	"deadline_decision",
	"decide_dtype",
	"expected_probe_cost_percent",
	"tolerance_band",
	"working_dir_headroom_gb",
	"InitializationReport",
	"initialize_retraining",
	"V0ArmResult",
	"V0SessionConfig",
	"V0SessionResult",
	"predicted_cost_ratio",
	"run_v0_session",
	"aggregate_arm_cost_ratio",
	"harm_guard_rejects",
	"sign_pairs",
]