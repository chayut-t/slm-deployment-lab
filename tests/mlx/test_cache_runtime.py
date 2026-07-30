from __future__ import annotations

# ruff: noqa: E402

import pytest

mx = pytest.importorskip("mlx.core")

from slm_lab.backends.mlx.cache import allocate_cache
from slm_lab.backends.mlx.config import CacheLayout
from slm_lab.backends.mlx.model import (
    CustomQwen3ForCausalLM,
    validate_gqa_attention_shapes,
)
from slm_lab.backends.mlx.runtime import decode, prefill
from slm_lab.generation.mlx import greedy_generate
from tests.mlx.test_config import tiny_config


@pytest.mark.parametrize("layout", list(CacheLayout))
def test_cache_update_is_fixed_capacity_and_preserves_gqa_heads(
    layout: CacheLayout,
) -> None:
    config = tiny_config()
    state = allocate_cache(config, capacity=6, layout=layout)
    update = mx.arange(1 * 2 * 2 * 4).reshape(1, 2, 2, 4).astype(mx.float16)
    next_layer = state.layers[0].update(update, update + 1, offset=0)
    keys, values = next_layer.active(2)
    mx.eval(keys, values)

    assert tuple(keys.shape) == (1, 2, 2, 4)
    assert tuple(values.shape) == (1, 2, 2, 4)
    assert keys.tolist() == update.tolist()
    assert values.tolist() == (update + 1).tolist()
    assert state.valid_length == 0
    assert next_layer.capacity == 6
    assert state.nbytes == 2 * 2 * 1 * 2 * 6 * 4 * 2


def test_native_gqa_boundary_keeps_queries_grouped_over_physical_kv() -> None:
    queries = mx.zeros((1, 4, 3, 4), dtype=mx.float16)
    keys = mx.zeros((1, 2, 3, 4), dtype=mx.float16)
    contract = validate_gqa_attention_shapes(queries, keys, keys)
    assert contract.query_heads == 4
    assert contract.key_value_heads == 2
    assert contract.query_heads_per_kv_head == 2


@pytest.mark.parametrize("layout", list(CacheLayout))
def test_decode_matches_full_forward_and_advances_one_position(
    layout: CacheLayout,
) -> None:
    mx.random.seed(7)
    model = CustomQwen3ForCausalLM(tiny_config())
    prompt = mx.array([[1, 2, 3]], dtype=mx.int64)
    token = mx.array([[4]], dtype=mx.int64)

    prompt_output = prefill(model, prompt, capacity=8, layout=layout)
    decode_output = decode(model, token, prompt_output.cache)
    full_output = prefill(
        model,
        mx.concatenate([prompt, token], axis=1),
        capacity=8,
        layout=layout,
    )
    mx.eval(decode_output.next_logits, full_output.last_logits)

    assert decode_output.cache.valid_length == 4
    assert mx.allclose(
        decode_output.next_logits,
        full_output.last_logits,
        atol=2e-3,
        rtol=2e-3,
    ).item()


def test_head_and_sequence_major_layouts_are_numerically_equivalent() -> None:
    mx.random.seed(11)
    model = CustomQwen3ForCausalLM(tiny_config())
    prompt = mx.array([[2, 5, 7]], dtype=mx.int64)
    head = prefill(model, prompt, capacity=7, layout=CacheLayout.HEAD_MAJOR)
    sequence = prefill(
        model,
        prompt,
        capacity=7,
        layout=CacheLayout.SEQUENCE_MAJOR,
    )
    mx.eval(head.last_logits, sequence.last_logits)
    assert mx.array_equal(head.last_logits, sequence.last_logits).item()


def test_multistep_generation_is_deterministic_and_avoids_lookahead() -> None:
    mx.random.seed(13)
    model = CustomQwen3ForCausalLM(tiny_config())
    first = greedy_generate(
        model,
        [1, 2, 3],
        max_new_tokens=4,
        capacity=8,
    )
    second = greedy_generate(
        model,
        [1, 2, 3],
        max_new_tokens=4,
        capacity=8,
    )

    assert first.generated_token_ids == second.generated_token_ids
    assert len(first.generated_token_ids) == 4
    assert first.cache.valid_length == 6
    assert first.pending_token_id == first.generated_token_ids[-1]
