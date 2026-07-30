"""Deterministic greedy generation over the custom MLX runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from slm_lab.backends.mlx.cache import MLXKVCacheState
from slm_lab.backends.mlx.config import CacheLayout, MlxRuntimeConfigurationError
from slm_lab.backends.mlx.model import CustomQwen3ForCausalLM
from slm_lab.backends.mlx.runtime import decode, prefill


@dataclass(frozen=True)
class MLXGenerationResult:
    """Generated IDs and resumable state.

    The final selected token is intentionally pending: it is consumed by
    ``decode`` only when another token is requested, avoiding an unreturned
    look-ahead computation.
    """

    generated_token_ids: tuple[int, ...]
    stopped_on_eos: bool
    cache: MLXKVCacheState
    pending_token_id: int | None


def _greedy_token(logits: Any) -> int:
    import mlx.core as mx

    selected = mx.argmax(logits, axis=-1)
    mx.eval(selected)
    return int(selected.item())


def greedy_generate(
    model: CustomQwen3ForCausalLM,
    prompt_token_ids: Any,
    *,
    max_new_tokens: int,
    capacity: int,
    eos_token_id: int | None = None,
    layout: CacheLayout = CacheLayout.HEAD_MAJOR,
) -> MLXGenerationResult:
    """Generate deterministically with explicit prefill/decode transitions."""

    import mlx.core as mx

    if max_new_tokens < 0:
        raise MlxRuntimeConfigurationError("max_new_tokens must be non-negative")
    if max_new_tokens == 0:
        raise MlxRuntimeConfigurationError(
            "max_new_tokens=0 has no generated state; call prefill directly"
        )
    input_ids = mx.array(prompt_token_ids, dtype=mx.int64)
    if len(input_ids.shape) == 1:
        input_ids = input_ids[None, :]
    first = prefill(model, input_ids, capacity=capacity, layout=layout)
    logits = first.last_logits
    cache = first.cache
    generated: list[int] = []
    stopped = False

    for step in range(max_new_tokens):
        token_id = _greedy_token(logits)
        generated.append(token_id)
        if eos_token_id is not None and token_id == eos_token_id:
            stopped = True
            break
        if step + 1 < max_new_tokens:
            token = mx.array([[token_id]], dtype=mx.int64)
            output = decode(model, token, cache)
            logits = output.next_logits
            cache = output.cache

    return MLXGenerationResult(
        generated_token_ids=tuple(generated),
        stopped_on_eos=stopped,
        cache=cache,
        pending_token_id=generated[-1] if generated and not stopped else None,
    )
