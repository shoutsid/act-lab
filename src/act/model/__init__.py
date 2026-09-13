"""Minimal model implementations used by the ACT research gates."""

from .dense import DenseBaselineConfig, DenseCausalLM
from .factory import (
    model_from_genome,
    rebuild_model_from_genome,
    rebuild_v0_moe_from_genome,
    v0_moe_config_from_genome,
    v0_moe_model_from_genome,
    v0_model_from_genome,
)
from .moe import StaticMoECausalLM, StaticMoEConfig
from .v0 import V0CausalLM, V0ModelConfig
from .v0_moe import (
    DECLARED_V0_CONFIG,
    V0MoEBlock,
    V0MoECausalLM,
    V0MoEConfig,
    estimated_parameter_count,
    validate_declared_v0_config,
)
from .smoke import V0SmokeResult, run_v0_smoke

__all__ = [
	"DenseBaselineConfig",
	"DenseCausalLM",
	"model_from_genome",
	"rebuild_model_from_genome",
	"rebuild_v0_moe_from_genome",
	"v0_moe_config_from_genome",
	"v0_moe_model_from_genome",
	"v0_model_from_genome",
	"StaticMoECausalLM",
	"StaticMoEConfig",
	"V0CausalLM",
	"V0ModelConfig",
	"DECLARED_V0_CONFIG",
	"V0MoEBlock",
	"V0MoECausalLM",
	"V0MoEConfig",
	"estimated_parameter_count",
	"validate_declared_v0_config",
	"V0SmokeResult",
	"run_v0_smoke",
]