# macOS M4 environment

The common repository environment is pinned by the root `.python-version`,
`pyproject.toml`, and `uv.lock`. On the primary Mac:

```bash
uv sync --extra dev --locked
uv run slm-lab-validate-manifest host results/hosts/apple-m4-primary.json
uv run slm-lab-storage-preflight
```

T50 owns the MLX-LM baseline and must pin and smoke-test exact MLX, MLX-LM,
PyTorch, and Transformers versions before recording them here. T51 owns the
custom MLX runtime. T52 owns Instruments and MLX profiling evidence. Until
those tasks run, null tool versions in the host manifest mean “checked and
deferred,” not compatibility.

The storage preflight writes and removes only a tiny temporary probe. It does
not download weights or perform a heavy storage benchmark.
