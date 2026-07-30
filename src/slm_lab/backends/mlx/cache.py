"""Fixed-capacity, GQA-aware MLX KV-cache state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from slm_lab.backends.mlx.config import (
    CacheLayout,
    MlxRuntimeConfigurationError,
    Qwen3MlxConfig,
)


CACHE_DTYPE_NAME = "float16"


def stored_cache_shape(
    *,
    layout: CacheLayout,
    batch_size: int,
    num_key_value_heads: int,
    capacity: int,
    head_dim: int,
) -> tuple[int, int, int, int]:
    """Return a physical cache shape without importing MLX."""

    if min(batch_size, num_key_value_heads, capacity, head_dim) < 1:
        raise MlxRuntimeConfigurationError(
            "cache batch, K/V heads, capacity, and head dimension must be positive"
        )
    if layout is CacheLayout.HEAD_MAJOR:
        return (batch_size, num_key_value_heads, capacity, head_dim)
    if layout is CacheLayout.SEQUENCE_MAJOR:
        return (batch_size, capacity, num_key_value_heads, head_dim)
    raise MlxRuntimeConfigurationError(f"unsupported cache layout {layout!r}")


@dataclass(frozen=True)
class LayerKVCache:
    """One layer's full FP16 K/V allocation."""

    keys: Any
    values: Any
    layout: CacheLayout

    @property
    def capacity(self) -> int:
        return int(self.keys.shape[2 if self.layout is CacheLayout.HEAD_MAJOR else 1])

    @property
    def num_key_value_heads(self) -> int:
        return int(self.keys.shape[1 if self.layout is CacheLayout.HEAD_MAJOR else 2])

    @property
    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.values.nbytes)

    def _stored(self, update: Any) -> Any:
        if self.layout is CacheLayout.HEAD_MAJOR:
            return update
        return update.transpose(0, 2, 1, 3)

    def _head_major_prefix(self, tensor: Any, valid_length: int) -> Any:
        if self.layout is CacheLayout.HEAD_MAJOR:
            return tensor[:, :, :valid_length, :]
        return tensor[:, :valid_length, :, :].transpose(0, 2, 1, 3)

    def active(self, valid_length: int) -> tuple[Any, Any]:
        """Expose an active head-major view required by MLX GQA SDPA."""

        if not 0 <= valid_length <= self.capacity:
            raise MlxRuntimeConfigurationError(
                f"valid_length {valid_length} exceeds capacity {self.capacity}"
            )
        return (
            self._head_major_prefix(self.keys, valid_length),
            self._head_major_prefix(self.values, valid_length),
        )

    def update(self, keys: Any, values: Any, *, offset: int) -> LayerKVCache:
        """Functionally write a contiguous head-major K/V update."""

        import mlx.core as mx

        if tuple(keys.shape) != tuple(values.shape) or len(keys.shape) != 4:
            raise MlxRuntimeConfigurationError(
                "K/V updates must share [batch, kv_head, sequence, head_dim]"
            )
        update_length = int(keys.shape[2])
        if int(keys.shape[1]) != self.num_key_value_heads:
            raise MlxRuntimeConfigurationError(
                "K/V update head count differs from physical cache head count"
            )
        if offset < 0 or offset + update_length > self.capacity:
            raise MlxRuntimeConfigurationError(
                f"cache write [{offset}, {offset + update_length}) exceeds "
                f"capacity {self.capacity}"
            )
        axis = 2 if self.layout is CacheLayout.HEAD_MAJOR else 1
        stored_keys = self._stored(keys).astype(mx.float16)
        stored_values = self._stored(values).astype(mx.float16)
        start = mx.array([offset], dtype=mx.int32)
        return LayerKVCache(
            keys=mx.slice_update(self.keys, stored_keys, start, axes=(axis,)),
            values=mx.slice_update(self.values, stored_values, start, axes=(axis,)),
            layout=self.layout,
        )


@dataclass(frozen=True)
class MLXKVCacheState:
    """All layer buffers plus their shared valid prefix length."""

    layers: tuple[LayerKVCache, ...]
    valid_length: int
    capacity: int
    layout: CacheLayout

    def __post_init__(self) -> None:
        if not self.layers:
            raise MlxRuntimeConfigurationError("cache state needs at least one layer")
        if not 0 <= self.valid_length <= self.capacity:
            raise MlxRuntimeConfigurationError(
                f"valid_length {self.valid_length} exceeds capacity {self.capacity}"
            )
        for index, layer in enumerate(self.layers):
            if layer.layout is not self.layout:
                raise MlxRuntimeConfigurationError(
                    f"layer {index} layout differs from cache state"
                )
            if layer.capacity != self.capacity:
                raise MlxRuntimeConfigurationError(
                    f"layer {index} capacity differs from cache state"
                )

    @property
    def nbytes(self) -> int:
        return sum(layer.nbytes for layer in self.layers)

    def advanced(
        self,
        layers: Sequence[LayerKVCache],
        *,
        token_count: int,
    ) -> MLXKVCacheState:
        if len(layers) != len(self.layers):
            raise MlxRuntimeConfigurationError(
                "updated cache must preserve the layer count"
            )
        if token_count < 1:
            raise MlxRuntimeConfigurationError("token_count must be positive")
        return MLXKVCacheState(
            layers=tuple(layers),
            valid_length=self.valid_length + token_count,
            capacity=self.capacity,
            layout=self.layout,
        )


def allocate_cache(
    config: Qwen3MlxConfig,
    *,
    capacity: int,
    layout: CacheLayout = CacheLayout.HEAD_MAJOR,
    batch_size: int = 1,
) -> MLXKVCacheState:
    """Allocate exactly eight physical K/V heads for pinned Qwen3."""

    import mlx.core as mx

    shape = stored_cache_shape(
        layout=layout,
        batch_size=batch_size,
        num_key_value_heads=config.num_key_value_heads,
        capacity=capacity,
        head_dim=config.head_dim,
    )
    layers = tuple(
        LayerKVCache(
            keys=mx.zeros(shape, dtype=mx.float16),
            values=mx.zeros(shape, dtype=mx.float16),
            layout=layout,
        )
        for _ in range(config.num_hidden_layers)
    )
    return MLXKVCacheState(
        layers=layers,
        valid_length=0,
        capacity=capacity,
        layout=layout,
    )
