from __future__ import annotations

import ast
from pathlib import Path

import pytest

from slm_lab.backends.mlx.cache import stored_cache_shape
from slm_lab.backends.mlx.config import (
    CacheLayout,
    MlxRuntimeConfigurationError,
    Qwen3MlxConfig,
)
from slm_lab.contracts.static_cache import CONTEXT_VARIANTS


def tiny_config() -> Qwen3MlxConfig:
    return Qwen3MlxConfig(
        model_type="qwen3",
        hidden_size=16,
        num_hidden_layers=2,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        rope_theta=10_000,
        tie_word_embeddings=True,
    )


def test_config_freezes_grouped_query_ratio() -> None:
    config = tiny_config()
    assert config.query_heads_per_kv_head == 2
    assert config.num_key_value_heads < config.num_attention_heads


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        (CacheLayout.HEAD_MAJOR, (1, 8, 160, 128)),
        (CacheLayout.SEQUENCE_MAJOR, (1, 160, 8, 128)),
    ],
)
def test_cache_layouts_keep_eight_physical_heads(
    layout: CacheLayout,
    expected: tuple[int, ...],
) -> None:
    assert (
        stored_cache_shape(
            layout=layout,
            batch_size=1,
            num_key_value_heads=8,
            capacity=CONTEXT_VARIANTS[128],
            head_dim=128,
        )
        == expected
    )


def test_config_rejects_non_integral_gqa_group() -> None:
    with pytest.raises(MlxRuntimeConfigurationError, match="divisible"):
        Qwen3MlxConfig(
            model_type="qwen3",
            hidden_size=12,
            num_hidden_layers=1,
            intermediate_size=16,
            num_attention_heads=3,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=16,
            rms_norm_eps=1e-6,
            max_position_embeddings=32,
            rope_theta=10_000,
            tie_word_embeddings=True,
        )


def test_attention_implementation_has_no_repeat_or_tile_materialization() -> None:
    source_path = (
        Path(__file__).resolve().parents[2] / "src/slm_lab/backends/mlx/model.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "repeat" not in calls
    assert "tile" not in calls
