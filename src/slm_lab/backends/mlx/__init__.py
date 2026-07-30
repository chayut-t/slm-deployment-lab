"""Custom MLX runtime for explicit Qwen3 prefill and one-token decode.

MLX is imported only by the implementation modules. Importing this package's
metadata remains safe in the repository's platform-neutral environment.
"""

from .config import CacheLayout, Qwen3MlxConfig

__all__ = ["CacheLayout", "Qwen3MlxConfig"]
