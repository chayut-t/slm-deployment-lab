# Model integration

`qwen3_reference.py` validates the immutable T00 Qwen3-0.6B contract and loads
that exact revision for deterministic PyTorch inference. Network access is
opt-in; model weights remain external artifacts and are never committed.

The loader freezes eval mode, disables gradients, selects an explicit dtype
and attention implementation, enables deterministic algorithms, and records
the exact Python, PyTorch, Transformers, and Safetensors versions beside
reference evidence.
