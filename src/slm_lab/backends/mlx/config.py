"""Configuration and immutable-source validation for the custom MLX runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


EXPECTED_CONFIG_SHA256 = (
    "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
)
EXPECTED_WEIGHTS_SHA256 = (
    "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
)
EXPECTED_WEIGHTS_SIZE = 1_503_300_328


class MlxRuntimeConfigurationError(ValueError):
    """The model or runtime request violates the custom MLX contract."""


class CacheLayout(str, Enum):
    """Physical fixed-capacity cache layouts supported by T51."""

    HEAD_MAJOR = "head_major"
    SEQUENCE_MAJOR = "sequence_major"


@dataclass(frozen=True)
class Qwen3MlxConfig:
    """Qwen3 dimensions needed by the custom implementation."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    max_position_embeddings: int
    rope_theta: float
    tie_word_embeddings: bool

    def __post_init__(self) -> None:
        integer_fields = (
            self.hidden_size,
            self.num_hidden_layers,
            self.intermediate_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.vocab_size,
            self.max_position_embeddings,
        )
        if self.model_type != "qwen3":
            raise MlxRuntimeConfigurationError(
                f"custom runtime requires model_type='qwen3', found {self.model_type!r}"
            )
        if any(value < 1 for value in integer_fields):
            raise MlxRuntimeConfigurationError(
                "all Qwen3 architecture dimensions must be positive"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise MlxRuntimeConfigurationError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise MlxRuntimeConfigurationError(
                "rms_norm_eps and rope_theta must be positive"
            )

    @property
    def query_heads_per_kv_head(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> Qwen3MlxConfig:
        """Read only the explicit, supported Qwen3 architecture fields."""

        names = (
            "model_type",
            "hidden_size",
            "num_hidden_layers",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "rms_norm_eps",
            "max_position_embeddings",
            "rope_theta",
            "tie_word_embeddings",
        )
        try:
            values = {name: document[name] for name in names}
        except KeyError as exc:
            raise MlxRuntimeConfigurationError(
                f"model config is missing {exc.args[0]!r}"
            ) from exc
        try:
            return cls(**values)
        except TypeError as exc:
            raise MlxRuntimeConfigurationError(
                f"model config contains invalid field types: {exc}"
            ) from exc


def file_sha256(path: Path) -> str:
    """Hash a local source artifact without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_pinned_config(
    model_path: Path | str,
    *,
    verify_weights: bool = True,
) -> tuple[Qwen3MlxConfig, tuple[Path, ...]]:
    """Validate and load the immutable Qwen3 source accepted by T50."""

    root = Path(model_path).resolve()
    config_path = root / "config.json"
    weight_paths = tuple(sorted(root.glob("model*.safetensors")))
    if not config_path.is_file():
        raise MlxRuntimeConfigurationError(f"missing model config: {config_path}")
    if not weight_paths:
        raise MlxRuntimeConfigurationError(
            f"no model*.safetensors files found under {root}"
        )
    if file_sha256(config_path) != EXPECTED_CONFIG_SHA256:
        raise MlxRuntimeConfigurationError(
            "config.json does not match the immutable Qwen3-0.6B revision"
        )
    if len(weight_paths) != 1:
        raise MlxRuntimeConfigurationError(
            "the pinned Qwen3-0.6B source must contain one model.safetensors file"
        )
    weight_path = weight_paths[0]
    if weight_path.name != "model.safetensors":
        raise MlxRuntimeConfigurationError(
            f"unexpected weight filename {weight_path.name!r}"
        )
    if weight_path.stat().st_size != EXPECTED_WEIGHTS_SIZE:
        raise MlxRuntimeConfigurationError(
            "model.safetensors size does not match the immutable revision"
        )
    if verify_weights and file_sha256(weight_path) != EXPECTED_WEIGHTS_SHA256:
        raise MlxRuntimeConfigurationError(
            "model.safetensors digest does not match the immutable revision"
        )
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MlxRuntimeConfigurationError(
            f"cannot parse model config {config_path}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise MlxRuntimeConfigurationError("config.json must contain one object")
    config = Qwen3MlxConfig.from_mapping(document)
    expected_dimensions = {
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "intermediate_size": 3072,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151_936,
        "max_position_embeddings": 40_960,
        "rope_theta": 1_000_000,
        "tie_word_embeddings": True,
    }
    actual_dimensions = {
        name: getattr(config, name) for name in expected_dimensions
    }
    if actual_dimensions != expected_dimensions:
        raise MlxRuntimeConfigurationError(
            "model dimensions differ from the frozen Qwen3-0.6B contract"
        )
    return config, weight_paths
