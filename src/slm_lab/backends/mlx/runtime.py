"""Explicit custom-MLX prompt prefill and one-token decode entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from slm_lab.backends.mlx.cache import MLXKVCacheState, allocate_cache
from slm_lab.backends.mlx.config import (
    CacheLayout,
    MlxRuntimeConfigurationError,
)
from slm_lab.backends.mlx.model import CustomQwen3ForCausalLM
from slm_lab.contracts.static_cache import CONTEXT_VARIANTS


@dataclass(frozen=True)
class PrefillOutput:
    """Next-token logits and full fixed-capacity cache after prompt execution."""

    last_logits: Any
    cache: MLXKVCacheState


@dataclass(frozen=True)
class DecodeOutput:
    """Next-token logits and full fixed-capacity cache after one decode token."""

    next_logits: Any
    cache: MLXKVCacheState


def _validate_input_ids(input_ids: Any, *, expected_length: int | None = None) -> int:
    shape = tuple(int(value) for value in input_ids.shape)
    if len(shape) != 2 or shape[0] != 1 or shape[1] < 1:
        raise MlxRuntimeConfigurationError(
            "input_ids must have shape [1, non_empty_sequence]"
        )
    if expected_length is not None and shape[1] != expected_length:
        raise MlxRuntimeConfigurationError(
            f"input_ids must contain exactly {expected_length} token"
        )
    if "int" not in str(input_ids.dtype):
        raise MlxRuntimeConfigurationError("input_ids must use an integer dtype")
    return shape[1]


def prefill(
    model: CustomQwen3ForCausalLM,
    input_ids: Any,
    *,
    capacity: int,
    layout: CacheLayout = CacheLayout.HEAD_MAJOR,
) -> PrefillOutput:
    """Execute a non-empty prompt and materialize its fixed-capacity KV state."""

    prompt_length = _validate_input_ids(input_ids)
    if capacity < prompt_length:
        raise MlxRuntimeConfigurationError(
            f"prompt length {prompt_length} exceeds cache capacity {capacity}"
        )
    state = allocate_cache(model.config, capacity=capacity, layout=layout)
    logits, layers = model(
        input_ids,
        state.layers,
        offset=0,
        mask="causal" if prompt_length > 1 else None,
    )
    state = state.advanced(layers, token_count=prompt_length)
    last_logits = logits[:, -1, :].astype(mx.float32)
    return PrefillOutput(last_logits=last_logits, cache=state)


def prefill_t12_variant(
    model: CustomQwen3ForCausalLM,
    input_ids: Any,
    *,
    layout: CacheLayout = CacheLayout.HEAD_MAJOR,
) -> PrefillOutput:
    """Prefill one exact T12 prompt length with its frozen reserve capacity."""

    prompt_length = _validate_input_ids(input_ids)
    try:
        capacity = CONTEXT_VARIANTS[prompt_length]
    except KeyError as exc:
        raise MlxRuntimeConfigurationError(
            f"T12 prompt length must be one of {tuple(CONTEXT_VARIANTS)}"
        ) from exc
    return prefill(model, input_ids, capacity=capacity, layout=layout)


def decode(
    model: CustomQwen3ForCausalLM,
    input_ids: Any,
    cache: MLXKVCacheState,
) -> DecodeOutput:
    """Execute exactly one token and write it at cache.valid_length."""

    _validate_input_ids(input_ids, expected_length=1)
    if len(cache.layers) != model.config.num_hidden_layers:
        raise MlxRuntimeConfigurationError(
            "cache layer count differs from the model configuration"
        )
    if cache.valid_length >= cache.capacity:
        raise MlxRuntimeConfigurationError(
            f"decode write at {cache.valid_length} exceeds capacity {cache.capacity}"
        )
    logits, layers = model(
        input_ids,
        cache.layers,
        offset=cache.valid_length,
        mask=None,
    )
    next_cache = cache.advanced(layers, token_count=1)
    next_logits = logits[:, -1, :].astype(mx.float32)
    return DecodeOutput(next_logits=next_logits, cache=next_cache)
