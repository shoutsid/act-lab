"""Typed, versioned configuration contracts for ACT research runs."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Config):
    dataset_id: str = "synthetic-v1"
    tokenizer_id: str = "synthetic-tokenizer-v1"
    train_shards: tuple[str, ...] = ()
    validation_shards: tuple[str, ...] = ()
    sequence_length: int = Field(default=2048, ge=2)
    seed: int = 7

    @model_validator(mode="after")
    def validate_shards(self) -> "DataConfig":
        if bool(self.train_shards) != bool(self.validation_shards):
            raise ValueError("train_shards and validation_shards must be provided together")
        return self


class ModelConfig(_Config):
    name: str = "act-v0"
    hidden_size: int = Field(default=1024, gt=0)
    depth: int = Field(default=24, gt=0)
    vocab_size: int = Field(default=32768, gt=0)
    attention_heads: int = Field(default=16, gt=0)
    kv_heads: int = Field(default=4, gt=0)
    ffn_width: int = Field(default=4096, gt=0)
    module_types: tuple[Literal["attention", "gdn", "hybrid"], ...] = ("attention",)
    moe_experts: int = Field(default=0, ge=0)
    active_experts: int = Field(default=0, ge=0)
    shared_experts: int = Field(default=0, ge=0)
    max_params: int = Field(default=350_000_000, gt=0)
    max_active_params: int = Field(default=350_000_000, gt=0)

    @model_validator(mode="after")
    def validate_dimensions(self) -> "ModelConfig":
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if self.attention_heads % self.kv_heads:
            raise ValueError("attention_heads must be divisible by kv_heads")
        if self.moe_experts == 0 and self.active_experts:
            raise ValueError("active_experts requires moe_experts")
        if self.moe_experts and not 1 <= self.active_experts <= self.moe_experts:
            raise ValueError("active_experts must be between 1 and moe_experts")
        if self.max_active_params > self.max_params:
            raise ValueError("max_active_params cannot exceed max_params")
        if len(self.module_types) not in {1, self.depth}:
            raise ValueError("module_types must contain one value or one value per layer")
        return self


class TrainingRunConfig(_Config):
    seed: int = 7
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = Field(default=2, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0)
    max_steps: int = Field(default=1000, gt=0)
    mixed_precision: Literal["off", "fp16", "bf16"] = "off"
    activation_checkpointing: bool = False
    checkpoint_interval: int = Field(default=100, gt=0)


class SearchConfig(_Config):
    controller: Literal["random", "routing", "gradient", "fisher", "pareto"] = "random"
    mutation_interval: int = Field(default=100, gt=0)
    mutation_budget: int = Field(default=10, ge=0)
    max_search_flops: int = Field(default=10**12, gt=0)
    population_size: int = Field(default=1, gt=0)


class EvaluationConfig(_Config):
    validation_batches: int = Field(default=16, gt=0)
    capability_probes: tuple[str, ...] = ()
    report_dir: str = "artifacts/evaluations"


class ACTConfig(_Config):
    version: Literal[1] = 1
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingRunConfig = TrainingRunConfig()
    search: SearchConfig = SearchConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    @model_validator(mode="after")
    def validate_contract(self) -> "ACTConfig":
        if self.training.batch_size * self.data.sequence_length > 2**22:
            raise ValueError("batch_size * sequence_length exceeds the safety budget")
        if self.search.population_size > 1 and self.search.controller != "pareto":
            raise ValueError("population search requires the pareto controller")
        return self


def load_config(path: str | Path) -> ACTConfig:
    """Load and validate a YAML configuration from a repository or artifact path."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload: Any = yaml.safe_load(handle)
    if payload is None:
        raise ValueError(f"configuration file is empty: {source}")
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    return ACTConfig.model_validate(payload)
