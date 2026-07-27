from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from slm_lab.contracts import (
    CACHE_DTYPE,
    CONTEXT_VARIANTS,
    CacheContractError,
    apply_decode_updates,
    build_contract_family,
    build_decode_contract,
    build_prefill_contract,
    cache_bytes,
    cache_pairs_from_reference,
    materialize_reference_cache,
    validate_tensor_mapping,
)
from slm_lab.contracts.static_cache import (
    HEAD_DIM,
    NUM_KEY_VALUE_HEADS,
    NUM_LAYERS,
    VOCAB_SIZE,
)
from slm_lab.models import load_reference_model


ROOT = Path(__file__).resolve().parents[2]
TOKEN_FIXTURES = ROOT / "tests/fixtures/t10/token-fixtures-v1.json"


def test_contract_family_covers_all_contexts_without_dynamic_dimensions() -> None:
    family = build_contract_family()

    assert tuple(family) == (128, 512, 1024, 4096)
    assert {length: pair["prefill"].cache_capacity for length, pair in family.items()} == (
        CONTEXT_VARIANTS
    )
    for prompt_length, pair in family.items():
        assert set(pair) == {"prefill", "decode"}
        for graph_kind, contract in pair.items():
            payload = contract.as_dict()
            assert payload["variant_id"] == f"S{prompt_length}"
            assert payload["graph_kind"] == graph_kind
            assert all(
                isinstance(dimension, int) and dimension > 0
                for tensor in (*contract.inputs, *contract.outputs)
                for dimension in tensor.shape
            )
            assert all(
                len(tensor.shape) == len(tensor.layout)
                for tensor in (*contract.inputs, *contract.outputs)
            )


def test_prefill_names_dtypes_shapes_and_gqa_layout_are_explicit() -> None:
    contract = build_prefill_contract(1024)
    payload = contract.as_dict()

    assert [tensor.name for tensor in contract.inputs] == [
        "input_ids",
        "attention_mask",
        "position_ids",
    ]
    assert contract.tensor("input_ids").dtype == "int64"
    assert contract.tensor("input_ids").shape == (1, 1024)
    assert contract.tensor("last_logits").dtype == "float32"
    assert contract.tensor("last_logits").shape == (1, VOCAB_SIZE)
    assert contract.tensor("key_cache.0").shape == (
        1,
        NUM_KEY_VALUE_HEADS,
        1152,
        HEAD_DIM,
    )
    assert contract.tensor("key_cache.0").layout == (
        "batch",
        "kv_head",
        "cache_position",
        "head_dim",
    )
    assert contract.tensor("key_cache.27").dtype == CACHE_DTYPE
    assert contract.tensor("value_cache.27").dtype == CACHE_DTYPE
    assert contract.tensor("valid_length").shape == (1,)
    assert payload["cache_update"] == {
        "strategy": "prefill_prefix_materialization",
        "written_range": "[0, prompt_length)",
        "zero_filled_range": "[prompt_length, cache_capacity)",
        "output_valid_length": "prompt_length",
    }
    assert "write_index" not in payload["cache_update"]
    assert "input_valid_range" not in payload["cache_update"]
    assert len(contract.outputs) == 1 + 2 * NUM_LAYERS + 1


def test_decode_names_shapes_and_update_semantics_are_explicit() -> None:
    contract = build_decode_contract(4096)
    payload = contract.as_dict()

    assert contract.tensor("input_ids").shape == (1, 1)
    assert contract.tensor("attention_mask").shape == (1, 4224)
    assert contract.tensor("position_ids").shape == (1, 1)
    assert contract.tensor("key_cache.0").shape == (1, 8, 4224, 128)
    assert contract.tensor("present_key.27").shape == (1, 8, 4224, 128)
    assert contract.tensor("updated_valid_length").dtype == "int64"
    assert payload["cache_update"] == {
        "strategy": "fixed_capacity_indexed_copy",
        "input_valid_range": "[0, valid_length)",
        "write_index": "valid_length",
        "output_valid_length": "valid_length + 1",
    }
    assert "written_range" not in payload["cache_update"]
    assert "zero_filled_range" not in payload["cache_update"]
    assert len(contract.inputs) == 3 + 2 * NUM_LAYERS + 1
    assert len(contract.outputs) == 1 + 2 * NUM_LAYERS + 1


def test_cache_byte_accounting_matches_qwen_gqa_dimensions() -> None:
    mib = 1024 * 1024

    assert cache_bytes(128) == 14 * mib
    assert cache_bytes(512) == 56 * mib
    assert cache_bytes(1024) == 112 * mib
    assert cache_bytes(4096) == 448 * mib
    assert cache_bytes(CONTEXT_VARIANTS[1024]) == 126 * mib
    assert cache_bytes(CONTEXT_VARIANTS[4096]) == 462 * mib

    with pytest.raises(CacheContractError, match="non-negative"):
        cache_bytes(-1)


class _TensorStub:
    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        self.shape = shape
        self.dtype = dtype


def test_tensor_mapping_rejects_name_shape_and_dtype_drift() -> None:
    specs = build_prefill_contract(128).inputs
    tensors = {
        spec.name: _TensorStub(spec.shape, spec.dtype)
        for spec in specs
    }
    validate_tensor_mapping(tensors, specs)

    with pytest.raises(CacheContractError, match="missing=.*position_ids"):
        validate_tensor_mapping(
            {name: tensor for name, tensor in tensors.items() if name != "position_ids"},
            specs,
        )
    tensors["input_ids"] = _TensorStub((1, 127), "int64")
    with pytest.raises(CacheContractError, match="input_ids: expected shape"):
        validate_tensor_mapping(tensors, specs)
    tensors["input_ids"] = _TensorStub((1, 128), "int32")
    with pytest.raises(CacheContractError, match="expected dtype int64"):
        validate_tensor_mapping(tensors, specs)


def _torch_or_skip():
    return pytest.importorskip("torch")


class _TinyGQAReference:
    """Weightless PyTorch model with the T11 growing-cache protocol."""

    def __init__(self, torch) -> None:
        self.torch = torch

    def __call__(
        self,
        *,
        input_ids,
        past_key_values=None,
        use_cache: bool,
        return_dict: bool,
    ):
        assert use_cache is True
        assert return_dict is True
        batch, sequence = input_ids.shape
        assert batch == 1
        prior = (
            0
            if past_key_values is None
            else past_key_values[0][0].shape[2]
        )
        positions = self.torch.arange(
            prior,
            prior + sequence,
            dtype=self.torch.float16,
        ).reshape(1, 1, sequence, 1)
        heads = self.torch.arange(
            NUM_KEY_VALUE_HEADS,
            dtype=self.torch.float16,
        ).reshape(1, NUM_KEY_VALUE_HEADS, 1, 1)
        dimensions = self.torch.arange(
            HEAD_DIM,
            dtype=self.torch.float16,
        ).reshape(1, 1, 1, HEAD_DIM)
        token_values = input_ids.to(self.torch.float16).reshape(1, 1, sequence, 1)
        pairs = []
        for layer in range(NUM_LAYERS):
            key_delta = token_values + positions + heads + dimensions + layer
            value_delta = token_values - positions + heads - dimensions - layer
            if past_key_values is None:
                key = key_delta
                value = value_delta
            else:
                key = self.torch.cat((past_key_values[layer][0], key_delta), dim=2)
                value = self.torch.cat(
                    (past_key_values[layer][1], value_delta),
                    dim=2,
                )
            pairs.append((key, value))
        logits = self.torch.zeros((1, sequence, VOCAB_SIZE), dtype=self.torch.float32)
        return SimpleNamespace(logits=logits, past_key_values=tuple(pairs))


def test_multistep_static_updates_reproduce_pytorch_reference_cache() -> None:
    torch = _torch_or_skip()
    model = _TinyGQAReference(torch)
    prompt = torch.arange(128, dtype=torch.int64).reshape(1, 128)
    prefill = model(
        input_ids=prompt,
        use_cache=True,
        return_dict=True,
    )
    state = materialize_reference_cache(
        prefill.past_key_values,
        prompt_length=128,
    )

    reference_cache = prefill.past_key_values
    for token_id in (17, 23):
        decode = model(
            input_ids=torch.tensor([[token_id]], dtype=torch.int64),
            past_key_values=reference_cache,
            use_cache=True,
            return_dict=True,
        )
        reference_pairs = cache_pairs_from_reference(decode.past_key_values)
        key_updates = tuple(key[:, :, -1:, :] for key, _ in reference_pairs)
        value_updates = tuple(value[:, :, -1:, :] for _, value in reference_pairs)
        prior_keys = tuple(key.clone() for key in state.keys)
        prior_values = tuple(value.clone() for value in state.values)
        state = apply_decode_updates(state, key_updates, value_updates)

        assert state.valid_length == reference_pairs[0][0].shape[2]
        for layer, (reference_key, reference_value) in enumerate(reference_pairs):
            assert torch.equal(
                state.keys[layer][:, :, : state.valid_length, :],
                reference_key,
            )
            assert torch.equal(
                state.values[layer][:, :, : state.valid_length, :],
                reference_value,
            )
            assert torch.equal(
                state.keys[layer][:, :, : state.valid_length - 1, :],
                prior_keys[layer][:, :, : state.valid_length - 1, :],
            )
            assert torch.equal(
                state.values[layer][:, :, : state.valid_length - 1, :],
                prior_values[layer][:, :, : state.valid_length - 1, :],
            )
        reference_cache = decode.past_key_values


def test_decode_update_rejects_overflow() -> None:
    torch = _torch_or_skip()
    prompt_pairs = tuple(
        (torch.zeros((1, 8, 128, 128), dtype=torch.float16),) * 2
        for _ in range(NUM_LAYERS)
    )
    state = materialize_reference_cache(prompt_pairs, prompt_length=128)
    updates = tuple(
        torch.zeros((1, 8, 1, 128), dtype=torch.float16)
        for _ in range(NUM_LAYERS)
    )

    for _ in range(32):
        state = apply_decode_updates(state, updates, updates)
    assert state.valid_length == state.capacity == 160
    with pytest.raises(CacheContractError, match="exceeds capacity"):
        apply_decode_updates(state, updates, updates)


@pytest.mark.skipif(
    os.environ.get("SLM_LAB_RUN_QWEN_CACHE_CONTRACT") != "1",
    reason="set SLM_LAB_RUN_QWEN_CACHE_CONTRACT=1 with pinned weights available",
)
def test_real_qwen_static_updates_reproduce_t11_reference() -> None:
    torch = _torch_or_skip()
    bundle = json.loads(TOKEN_FIXTURES.read_text(encoding="utf-8"))
    fixture = next(
        item for item in bundle["context_workloads"] if item["id"] == "S128"
    )
    input_ids = torch.tensor([fixture["token_ids"]], dtype=torch.int64)
    loaded = load_reference_model(
        device="cpu",
        dtype="bfloat16",
        seed=0,
        local_files_only=True,
        attn_implementation="eager",
    )
    with torch.inference_mode():
        prefill = loaded.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=True,
            return_dict=True,
        )
    state = materialize_reference_cache(
        prefill.past_key_values,
        prompt_length=128,
    )
    reference_cache = prefill.past_key_values
    logits = prefill.logits[:, -1, :]

    for _ in range(2):
        token = logits.argmax(dim=-1, keepdim=True)
        attention_mask = torch.ones(
            (1, state.valid_length + 1),
            dtype=torch.int64,
        )
        with torch.inference_mode():
            decode = loaded.model(
                input_ids=token,
                attention_mask=attention_mask,
                past_key_values=reference_cache,
                use_cache=True,
                return_dict=True,
            )
        pairs = cache_pairs_from_reference(decode.past_key_values)
        key_updates = tuple(
            key[:, :, -1:, :].to(dtype=torch.float16) for key, _ in pairs
        )
        value_updates = tuple(
            value[:, :, -1:, :].to(dtype=torch.float16) for _, value in pairs
        )
        state = apply_decode_updates(state, key_updates, value_updates)
        for layer, (reference_key, reference_value) in enumerate(pairs):
            assert torch.equal(
                state.keys[layer][:, :, : state.valid_length, :],
                reference_key[:, :, : state.valid_length, :].to(torch.float16),
            )
            assert torch.equal(
                state.values[layer][:, :, : state.valid_length, :],
                reference_value[:, :, : state.valid_length, :].to(torch.float16),
            )
        reference_cache = decode.past_key_values
        logits = decode.logits[:, -1, :]
