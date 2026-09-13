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