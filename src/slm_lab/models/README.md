# Model integration

`qwen3_reference.py` validates the immutable T00 Qwen3-0.6B contract and loads
that exact revision for deterministic PyTorch inference. Network access is
opt-in; model weights remain external artifacts and are never committed.

The loader freezes eval mode, disables gradients, selects an explicit dtype,
requires eager attention, and enables deterministic algorithms. Evidence
records both the requested and model-reported attention implementation along
with the exact Python, PyTorch, Transformers, and Safetensors versions; a
fallback to another attention implementation is rejected.
