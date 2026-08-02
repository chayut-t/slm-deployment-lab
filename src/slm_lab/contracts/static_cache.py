"""Static Qwen3-0.6B prefill, decode, and GQA KV-cache contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
NUM_LAYERS = 28
NUM_ATTENTION_HEADS = 16
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
VOCAB_SIZE = 151_936
BATCH_SIZE = 1

TOKEN_DTYPE = "int64"
MASK_DTYPE = "int64"
POSITION_DTYPE = "int64"
LENGTH_DTYPE = "int64"
CACHE_DTYPE = "float16"
LOGITS_DTYPE = "float32"

# Each capacity is the exact T10 prompt length plus its frozen maximum output
# token count. A capacity equal to the prompt length would make the first
# one-token decode an overflow.
CONTEXT_VARIANTS: dict[int, int] = {
    128: 160,
    512: 576,
    1024: 1152,
    4096: 4224,
}

DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "int32": 4,
    "int64": 8,
}


class CacheContractError(ValueError):
    """A tensor or cache state violates the frozen T12 contract."""


@dataclass(frozen=True)
class TensorSpec:
    """One explicit graph tensor boundary."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    layout: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        if not self.name:
            raise CacheContractError("tensor name must be non-empty")
        if self.dtype not in DTYPE_BYTES:
            raise CacheContractError(f"{self.name}: unsupported dtype {self.dtype!r}")
        if not self.shape or any(
            not isinstance(dimension, int) or dimension < 1 for dimension in self.shape
        ):
            raise CacheContractError(
                f"{self.name}: shape must contain positive static dimensions"
            )
        if len(self.layout) != len(self.shape) or any(not axis for axis in self.layout):
            raise CacheContractError(
                f"{self.name}: layout must name every tensor dimension"
            )

    @property
    def nbytes(self) -> int:
        elements = 1
        for dimension in self.shape:
            elements *= dimension
        return elements * DTYPE_BYTES[self.dtype]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "layout": list(self.layout),
            "description": self.description,
        }


@dataclass(frozen=True)
class GraphContract:
    """Frozen input/output boundary for one static graph variant."""

    graph_kind: str
    prompt_length: int
    cache_capacity: int
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]

    def __post_init__(self) -> None:
        if self.graph_kind not in {"prefill", "decode"}:
            raise CacheContractError(
                f"graph_kind must be 'prefill' or 'decode', found {self.graph_kind!r}"
            )
        expected_capacity = CONTEXT_VARIANTS.get(self.prompt_length)
        if expected_capacity != self.cache_capacity:
            raise CacheContractError(
                f"S{self.prompt_length}: expected cache capacity "
                f"{expected_capacity}, found {self.cache_capacity}"
            )
        for boundary, tensors in (("input", self.inputs), ("output", self.outputs)):
            names = [tensor.name for tensor in tensors]
            if len(names) != len(set(names)):
                raise CacheContractError(
                    f"{self.graph_kind} {boundary} tensor names must be unique"
                )

    @property
    def variant_id(self) -> str:
        return f"S{self.prompt_length}"

    def tensor(self, name: str) -> TensorSpec:
        match = next(
            (tensor for tensor in (*self.inputs, *self.outputs) if tensor.name == name),
            None,
        )
        if match is None:
            raise CacheContractError(
                f"{self.graph_kind} {self.variant_id}: unknown tensor {name!r}"
            )
        return match

    def as_dict(self) -> dict[str, Any]:
        if self.graph_kind == "prefill":
            cache_update = {
                "strategy": "prefill_prefix_materialization",
                "written_range": "[0, prompt_length)",
                "zero_filled_range": "[prompt_length, cache_capacity)",
                "output_valid_length": "prompt_length",
            }
        else:
            cache_update = {
                "strategy": "fixed_capacity_indexed_copy",
                "input_valid_range": "[0, valid_length)",
                "write_index": "valid_length",
                "output_valid_length": "valid_length + 1",
            }
        return {
            "schema_version": 1,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "graph_kind": self.graph_kind,
            "variant_id": self.variant_id,
            "prompt_length": self.prompt_length,
            "cache_capacity": self.cache_capacity,
            "cache_update": cache_update,
            "inputs": [tensor.as_dict() for tensor in self.inputs],
            "outputs": [tensor.as_dict() for tensor in self.outputs],
        }


@dataclass(frozen=True)
class StaticCacheState:
    """Full fixed-capacity K/V buffers and their valid prefix length."""

    keys: tuple[Any, ...]
    values: tuple[Any, ...]
    valid_length: int
    capacity: int

    def __post_init__(self) -> None:
        if len(self.keys) != NUM_LAYERS or len(self.values) != NUM_LAYERS:
            raise CacheContractError(
                f"static cache must contain {NUM_LAYERS} key/value layers"
            )
        if self.capacity not in CONTEXT_VARIANTS.values():
            raise CacheContractError(
                f"capacity must be one of {tuple(CONTEXT_VARIANTS.values())}, "
                f"found {self.capacity}"
            )
        if not 0 <= self.valid_length <= self.capacity:
            raise CacheContractError(
                f"valid_length {self.valid_length} exceeds capacity {self.capacity}"
            )


def _variant_capacity(prompt_length: int) -> int:
    try:
        return CONTEXT_VARIANTS[prompt_length]
    except KeyError as exc:
        raise CacheContractError(
            f"prompt length must be one of {tuple(CONTEXT_VARIANTS)}, "
            f"found {prompt_length}"
        ) from exc


def _cache_spec(prefix: str, layer: int, capacity: int, description: str) -> TensorSpec:
    return TensorSpec(
        name=f"{prefix}.{layer}",
        dtype=CACHE_DTYPE,
        shape=(BATCH_SIZE, NUM_KEY_VALUE_HEADS, capacity, HEAD_DIM),
        layout=("batch", "kv_head", "cache_position", "head_dim"),
        description=description,
    )


def build_prefill_contract(prompt_length: int) -> GraphContract:
    """Build the static prompt graph for one frozen T10 workload."""

    capacity = _variant_capacity(prompt_length)
    inputs = (
        TensorSpec(
            "input_ids",
            TOKEN_DTYPE,
            (BATCH_SIZE, prompt_length),
            ("batch", "sequence"),
            "Exact T10 prompt token IDs.",
        ),
        TensorSpec(
            "attention_mask",
            MASK_DTYPE,
            (BATCH_SIZE, prompt_length),
            ("batch", "sequence"),
            "One for every real prompt token; padding is not permitted.",
        ),
        TensorSpec(
            "position_ids",
            POSITION_DTYPE,
            (BATCH_SIZE, prompt_length),
            ("batch", "sequence"),
            "Zero-based positions [0, prompt_length).",
        ),
    )
    outputs: list[TensorSpec] = [
        TensorSpec(
            "last_logits",
            LOGITS_DTYPE,
            (BATCH_SIZE, VOCAB_SIZE),
            ("batch", "vocabulary"),
            "Float32 next-token logits for the last prompt position.",
        )
    ]
    for layer in range(NUM_LAYERS):
        outputs.extend(
            (
                _cache_spec(
                    "key_cache",
                    layer,
                    capacity,
                    "GQA key cache; prompt prefix is valid and reserve is zero-filled.",
                ),
                _cache_spec(
                    "value_cache",
                    layer,
                    capacity,
                    "GQA value cache; prompt prefix is valid and reserve is zero-filled.",
                ),
            )
        )
    outputs.append(
        TensorSpec(
            "valid_length",
            LENGTH_DTYPE,
            (BATCH_SIZE,),
            ("batch",),
            "Equals prompt_length after prefill.",
        )
    )
    return GraphContract(
        graph_kind="prefill",
        prompt_length=prompt_length,
        cache_capacity=capacity,
        inputs=inputs,
        outputs=tuple(outputs),
    )


def build_decode_contract(prompt_length: int) -> GraphContract:
    """Build a fixed-capacity one-token decode graph."""

    capacity = _variant_capacity(prompt_length)
    inputs: list[TensorSpec] = [
        TensorSpec(
            "input_ids",
            TOKEN_DTYPE,
            (BATCH_SIZE, 1),
            ("batch", "sequence"),
            "Exactly one teacher-forced or selected decode token.",
        ),
        TensorSpec(
            "attention_mask",
            MASK_DTYPE,
            (BATCH_SIZE, capacity),
            ("batch", "cache_position"),
            "One through the current token at valid_length; zero after it.",
        ),
        TensorSpec(
            "position_ids",
            POSITION_DTYPE,
            (BATCH_SIZE, 1),
            ("batch", "sequence"),
            "Equals valid_length for the current token.",
        ),
    ]
    for layer in range(NUM_LAYERS):
        inputs.extend(
            (
                _cache_spec(
                    "key_cache",
                    layer,
                    capacity,
                    "Incoming GQA key cache; only [0, valid_length) is valid.",
                ),
                _cache_spec(
                    "value_cache",
                    layer,
                    capacity,
                    "Incoming GQA value cache; only [0, valid_length) is valid.",
                ),
            )
        )
    inputs.append(
        TensorSpec(
            "valid_length",
            LENGTH_DTYPE,
            (BATCH_SIZE,),
            ("batch",),
            "Number of valid incoming cache positions and current write index.",
        )
    )

    outputs: list[TensorSpec] = [
        TensorSpec(
            "next_logits",
            LOGITS_DTYPE,
            (BATCH_SIZE, VOCAB_SIZE),
            ("batch", "vocabulary"),
            "Float32 logits after incorporating the current decode token.",
        )
    ]
    for layer in range(NUM_LAYERS):
        outputs.extend(
            (
                _cache_spec(
                    "present_key",
                    layer,
                    capacity,
                    "Key cache with the current token written at valid_length.",
                ),
                _cache_spec(
                    "present_value",
                    layer,
                    capacity,
                    "Value cache with the current token written at valid_length.",
                ),
            )
        )
    outputs.append(
        TensorSpec(
            "updated_valid_length",
            LENGTH_DTYPE,
            (BATCH_SIZE,),
            ("batch",),
            "Equals incoming valid_length + 1.",
        )
    )
    return GraphContract(
        graph_kind="decode",
        prompt_length=prompt_length,
        cache_capacity=capacity,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
    )


def build_contract_family() -> dict[int, dict[str, GraphContract]]:
    """Return all four prefill/decode pairs without hand-written variants."""

    return {
        prompt_length: {
            "prefill": build_prefill_contract(prompt_length),
            "decode": build_decode_contract(prompt_length),
        }
        for prompt_length in CONTEXT_VARIANTS
    }


def cache_bytes(
    sequence_length: int,
    *,
    dtype: str = CACHE_DTYPE,
    num_layers: int = NUM_LAYERS,
    num_key_value_heads: int = NUM_KEY_VALUE_HEADS,
    head_dim: int = HEAD_DIM,
) -> int:
    """Return logical bytes for both K and V across all layers."""

    if sequence_length < 0:
        raise CacheContractError("sequence_length must be non-negative")
    try:
        element_bytes = DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise CacheContractError(f"unsupported cache dtype {dtype!r}") from exc
    dimensions = (num_layers, num_key_value_heads, head_dim)
    if any(dimension < 1 for dimension in dimensions):
        raise CacheContractError("cache dimensions must be positive")
    return (
        BATCH_SIZE
        * 2
        * num_layers
        * num_key_value_heads
        * sequence_length
        * head_dim
        * element_bytes
    )


def _shape(tensor: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dimension) for dimension in tensor.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise CacheContractError("cache value has no concrete tensor shape") from exc


def _dtype_name(tensor: Any) -> str:
    name = str(getattr(tensor, "dtype", ""))
    return name.rsplit(".", maxsplit=1)[-1]


def validate_tensor_mapping(
    tensors: Mapping[str, Any],
    specs: Sequence[TensorSpec],
    *,
    exact_names: bool = True,
) -> None:
    """Validate concrete tensors against explicit names, dtypes, and shapes."""

    expected = {spec.name: spec for spec in specs}
    actual_names = set(tensors)
    missing = set(expected) - actual_names
    extra = actual_names - set(expected)
    if missing or (exact_names and extra):
        raise CacheContractError(
            f"tensor names mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for name, spec in expected.items():
        tensor = tensors[name]
        actual_shape = _shape(tensor)
        if actual_shape != spec.shape:
            raise CacheContractError(
                f"{name}: expected shape {spec.shape}, found {actual_shape}"
            )
        actual_dtype = _dtype_name(tensor)
        if actual_dtype != spec.dtype:
            raise CacheContractError(
                f"{name}: expected dtype {spec.dtype}, found {actual_dtype!r}"
            )


def cache_pairs_from_reference(past_key_values: Any) -> tuple[tuple[Any, Any], ...]:
    """Normalize Transformers DynamicCache or legacy K/V tuples."""

    if past_key_values is None:
        raise CacheContractError("reference returned no past_key_values")
    legacy = past_key_values
    converter = getattr(past_key_values, "to_legacy_cache", None)
    if callable(converter):
        legacy = converter()
    try:
        pairs = tuple((layer[0], layer[1]) for layer in legacy)
    except (TypeError, IndexError) as exc:
        raise CacheContractError(
            "reference cache must expose per-layer (key, value) tensors"
        ) from exc
    if len(pairs) != NUM_LAYERS:
        raise CacheContractError(
            f"reference cache must contain {NUM_LAYERS} layers, found {len(pairs)}"
        )
    for layer, (key, value) in enumerate(pairs):
        key_shape = _shape(key)
        value_shape = _shape(value)
        if key_shape != value_shape:
            raise CacheContractError(
                f"layer {layer}: key/value shapes differ: {key_shape} != {value_shape}"
            )
        if (
            len(key_shape) != 4
            or key_shape[0] != BATCH_SIZE
            or key_shape[1] != NUM_KEY_VALUE_HEADS
            or key_shape[3] != HEAD_DIM
        ):
            raise CacheContractError(
                f"layer {layer}: expected reference layout "
                f"[{BATCH_SIZE}, {NUM_KEY_VALUE_HEADS}, sequence, {HEAD_DIM}], "
                f"found {key_shape}"
            )
    return pairs


def materialize_reference_cache(
    past_key_values: Any,
    *,
    prompt_length: int,
) -> StaticCacheState:
    """Copy a growing PyTorch reference cache into fixed FP16 buffers."""

    capacity = _variant_capacity(prompt_length)
    pairs = cache_pairs_from_reference(past_key_values)
    sequence_lengths = {_shape(key)[2] for key, _ in pairs}
    if len(sequence_lengths) != 1:
        raise CacheContractError("reference layers have inconsistent sequence lengths")
    valid_length = sequence_lengths.pop()
    if valid_length != prompt_length:
        raise CacheContractError(
            f"S{prompt_length} prefill requires reference length {prompt_length}, "
            f"found {valid_length}"
        )

    keys: list[Any] = []
    values: list[Any] = []
    target_shape = (BATCH_SIZE, NUM_KEY_VALUE_HEADS, capacity, HEAD_DIM)
    for key, value in pairs:
        try:
            static_key = key.new_zeros(target_shape, dtype=_torch_float16(key))
            static_value = value.new_zeros(target_shape, dtype=_torch_float16(value))
            static_key[:, :, :valid_length, :] = key.to(dtype=static_key.dtype)
            static_value[:, :, :valid_length, :] = value.to(dtype=static_value.dtype)
        except (AttributeError, TypeError, RuntimeError) as exc:
            raise CacheContractError(
                "reference cache tensors must support PyTorch-style allocation "
                "and indexed assignment"
            ) from exc
        keys.append(static_key)
        values.append(static_value)
    return StaticCacheState(
        keys=tuple(keys),
        values=tuple(values),
        valid_length=valid_length,
        capacity=capacity,
    )


def _torch_float16(tensor: Any) -> Any:
    module = type(tensor).__module__.split(".", maxsplit=1)[0]
    if module != "torch":
        raise CacheContractError(
            "reference materialization currently requires PyTorch tensors"
        )
    try:
        import torch
    except ImportError as exc:
        raise CacheContractError(
            "PyTorch is required to materialize the T11 reference cache"
        ) from exc
    return torch.float16


def apply_decode_updates(
    state: StaticCacheState,
    key_updates: Sequence[Any],
    value_updates: Sequence[Any],
) -> StaticCacheState:
    """Copy one GQA K/V position per layer into the fixed cache."""

    if state.valid_length >= state.capacity:
        raise CacheContractError(
            f"decode write at {state.valid_length} exceeds capacity {state.capacity}"
        )
    if len(key_updates) != NUM_LAYERS or len(value_updates) != NUM_LAYERS:
        raise CacheContractError(
            f"decode update must contain {NUM_LAYERS} key/value layers"
        )
    expected_update_shape = (BATCH_SIZE, NUM_KEY_VALUE_HEADS, 1, HEAD_DIM)
    updated_keys: list[Any] = []
    updated_values: list[Any] = []
    for layer, (key, value, key_update, value_update) in enumerate(
        zip(
            state.keys,
            state.values,
            key_updates,
            value_updates,
            strict=True,
        )
    ):
        expected_cache_shape = (
            BATCH_SIZE,
            NUM_KEY_VALUE_HEADS,
            state.capacity,
            HEAD_DIM,
        )
        for label, tensor, expected_shape in (
            ("key cache", key, expected_cache_shape),
            ("value cache", value, expected_cache_shape),
            ("key update", key_update, expected_update_shape),
            ("value update", value_update, expected_update_shape),
        ):
            if _shape(tensor) != expected_shape:
                raise CacheContractError(
                    f"layer {layer} {label}: expected {expected_shape}, "
                    f"found {_shape(tensor)}"
                )
            if _dtype_name(tensor) != CACHE_DTYPE:
                raise CacheContractError(
                    f"layer {layer} {label}: expected {CACHE_DTYPE}, "
                    f"found {_dtype_name(tensor)!r}"
                )
        next_key = key.clone()
        next_value = value.clone()
        write = slice(state.valid_length, state.valid_length + 1)
        next_key[:, :, write, :] = key_update
        next_value[:, :, write, :] = value_update
        updated_keys.append(next_key)
        updated_values.append(next_value)
    return StaticCacheState(
        keys=tuple(updated_keys),
        values=tuple(updated_values),
        valid_length=state.valid_length + 1,
        capacity=state.capacity,
    )
