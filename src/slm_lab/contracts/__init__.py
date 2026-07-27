"""Runtime-neutral deployment graph and explicit KV-cache contracts."""

from slm_lab.contracts.static_cache import (
    CACHE_DTYPE,
    CONTEXT_VARIANTS,
    CacheContractError,
    GraphContract,
    StaticCacheState,
    TensorSpec,
    apply_decode_updates,
    build_contract_family,
    build_decode_contract,
    build_prefill_contract,
    cache_bytes,
    cache_pairs_from_reference,
    materialize_reference_cache,
    validate_tensor_mapping,
)

__all__ = [
    "CACHE_DTYPE",
    "CONTEXT_VARIANTS",
    "CacheContractError",
    "GraphContract",
    "StaticCacheState",
    "TensorSpec",
    "apply_decode_updates",
    "build_contract_family",
    "build_decode_contract",
    "build_prefill_contract",
    "cache_bytes",
    "cache_pairs_from_reference",
    "materialize_reference_cache",
    "validate_tensor_mapping",
]
