# Model contracts

`slm_lab.contracts` is the runtime-neutral, machine-checkable boundary between
the pinned Qwen3-0.6B reference and later ONNX, QNN, MLX, and CUDA work.

The T12 contract freezes:

- four static prompt/capacity variants;
- every prefill and one-token-decode tensor name, dtype, shape, and layout;
- per-layer, GQA-aware K/V buffers; and
- fixed-capacity indexed cache updates.

See `docs/architecture/static-cache-contract.md` for the tensor diagrams,
mask/update semantics, byte accounting, and downstream conformance rules.
