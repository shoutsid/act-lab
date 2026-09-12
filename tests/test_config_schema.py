from pathlib import Path

import pytest

from act.config import ACTConfig, ModelConfig, load_config


def test_default_config_is_reconstructible() -> None:
    config = ACTConfig()
    assert config.version == 1
    assert config.model.hidden_size % config.model.attention_heads == 0
    assert config.model.module_types == ("attention",)


def test_config_rejects_invalid_moe_dimensions() -> None:
    with pytest.raises(ValueError, match="active_experts"):
        ModelConfig(moe_experts=4, active_experts=5)


def test_yaml_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(
        """
version: 1
model:
  hidden_size: 128
  attention_heads: 8
  kv_heads: 2
  depth: 4
  module_types: [attention, gdn, attention, hybrid]
training:
  device: cpu
  max_steps: 12
search:
  controller: pareto
  population_size: 4
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.model.module_types[-1] == "hybrid"
    assert config.training.max_steps == 12
    assert config.search.population_size == 4


def test_population_requires_pareto_controller() -> None:
    with pytest.raises(ValueError, match="population search"):
        ACTConfig(search={"population_size": 2})
