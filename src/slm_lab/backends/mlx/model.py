"""Custom Qwen3 MLX modules with native grouped-query attention."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from slm_lab.backends.mlx.cache import LayerKVCache
from slm_lab.backends.mlx.config import (
    MlxRuntimeConfigurationError,
    Qwen3MlxConfig,
    load_pinned_config,
)


@dataclass(frozen=True)
class GQAAttentionContract:
    """Concrete head counts presented to MLX's native GQA operator."""

    batch_size: int
    query_heads: int
    key_value_heads: int
    query_length: int
    key_value_length: int
    head_dim: int

    @property
    def query_heads_per_kv_head(self) -> int:
        return self.query_heads // self.key_value_heads


def validate_gqa_attention_shapes(
    queries: Any,
    keys: Any,
    values: Any,
) -> GQAAttentionContract:
    """Reject any attention boundary that pre-materializes repeated K/V."""

    q_shape = tuple(int(value) for value in queries.shape)
    k_shape = tuple(int(value) for value in keys.shape)
    v_shape = tuple(int(value) for value in values.shape)
    if len(q_shape) != 4 or len(k_shape) != 4 or k_shape != v_shape:
        raise MlxRuntimeConfigurationError(
            "attention requires Q/K/V rank-four tensors and matching K/V shapes"
        )
    if q_shape[0] != k_shape[0] or q_shape[3] != k_shape[3]:
        raise MlxRuntimeConfigurationError(
            "Q/K/V batch and head dimensions must match"
        )
    if q_shape[1] % k_shape[1]:
        raise MlxRuntimeConfigurationError(
            "query heads must be divisible by physical K/V heads"
        )
    if k_shape[1] >= q_shape[1]:
        raise MlxRuntimeConfigurationError(
            "Qwen3 GQA requires fewer physical K/V heads than query heads"
        )
    return GQAAttentionContract(
        batch_size=q_shape[0],
        query_heads=q_shape[1],
        key_value_heads=k_shape[1],
        query_length=q_shape[2],
        key_value_length=k_shape[2],
        head_dim=q_shape[3],
    )


def grouped_query_attention(
    queries: Any,
    keys: Any,
    values: Any,
    *,
    scale: float,
    mask: str | Any | None,
) -> Any:
    """Call MLX SDPA with physical K/V heads; MLX groups queries internally."""

    validate_gqa_attention_shapes(queries, keys, values)
    # MLX requires a common attention dtype. The T12 cache boundary is FP16,
    # so queries cross the same boundary without expanding K/V heads.
    queries = queries.astype(keys.dtype)
    return mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=scale,
        mask=mask,
    )


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3MlxConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(
            config.head_dim,
            traditional=False,
            base=config.rope_theta,
        )

    def __call__(
        self,
        hidden_states: Any,
        cache: LayerKVCache,
        *,
        offset: int,
        mask: str | Any | None,
    ) -> tuple[Any, LayerKVCache]:
        batch_size, sequence_length, _ = hidden_states.shape
        queries = self.q_proj(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.head_dim,
        )
        keys = self.k_proj(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        values = self.v_proj(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        next_cache = cache.update(keys, values, offset=offset)
        active_keys, active_values = next_cache.active(offset + sequence_length)
        attention = grouped_query_attention(
            queries,
            active_keys,
            active_values,
            scale=self.scale,
            mask=mask,
        )
        attention = attention.transpose(0, 2, 1, 3).reshape(
            batch_size,
            sequence_length,
            -1,
        )
        return self.o_proj(attention), next_cache


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3MlxConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, hidden_states: Any) -> Any:
        return self.down_proj(
            nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MlxConfig):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: Any,
        cache: LayerKVCache,
        *,
        offset: int,
        mask: str | Any | None,
    ) -> tuple[Any, LayerKVCache]:
        attention, next_cache = self.self_attn(
            self.input_layernorm(hidden_states),
            cache,
            offset=offset,
            mask=mask,
        )
        hidden_states = hidden_states + attention
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, next_cache


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3MlxConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: Any,
        cache_layers: tuple[LayerKVCache, ...],
        *,
        offset: int,
        mask: str | Any | None,
    ) -> tuple[Any, tuple[LayerKVCache, ...]]:
        if len(cache_layers) != len(self.layers):
            raise MlxRuntimeConfigurationError(
                "cache layer count differs from model layer count"
            )
        hidden_states = self.embed_tokens(input_ids)
        next_layers = []
        for layer, cache in zip(self.layers, cache_layers, strict=True):
            hidden_states, next_cache = layer(
                hidden_states,
                cache,
                offset=offset,
                mask=mask,
            )
            next_layers.append(next_cache)
        return self.norm(hidden_states), tuple(next_layers)


class CustomQwen3ForCausalLM(nn.Module):
    """Custom implementation; parameter names match upstream Safetensors."""

    def __init__(self, config: Qwen3MlxConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            )

    def __call__(
        self,
        input_ids: Any,
        cache_layers: tuple[LayerKVCache, ...],
        *,
        offset: int,
        mask: str | Any | None,
    ) -> tuple[Any, tuple[LayerKVCache, ...]]:
        hidden_states, next_layers = self.model(
            input_ids,
            cache_layers,
            offset=offset,
            mask=mask,
        )
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        return logits, next_layers


def load_custom_qwen3(
    model_path: Path | str,
    *,
    verify_weights: bool = True,
) -> CustomQwen3ForCausalLM:
    """Load pinned local weights without constructing an MLX-LM model."""

    config, weight_paths = load_pinned_config(
        model_path,
        verify_weights=verify_weights,
    )
    weights: dict[str, Any] = {}
    for path in weight_paths:
        weights.update(mx.load(str(path)))
    if config.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    model = CustomQwen3ForCausalLM(config)
    model.eval()
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model
