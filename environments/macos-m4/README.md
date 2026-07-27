# macOS M4 environment

The common repository environment is pinned by the root `.python-version`,
`pyproject.toml`, and `uv.lock`. On the primary Mac:

```bash
uv sync --extra dev --locked
uv run slm-lab-validate-manifest host results/hosts/apple-m4-primary.json
uv run slm-lab-storage-preflight
```

T50's MLX-LM baseline uses the separately pinned
`mlx-baseline-requirements.txt`. Keeping MLX out of the cross-platform root
environment allows ordinary repository checks to run on non-macOS hosts:

```bash
UV_CACHE_DIR=/tmp/slm-lab-mlx-cache \
  uv venv --python 3.11.13 .ai-local/envs/t50-mlx
UV_CACHE_DIR=/tmp/slm-lab-mlx-cache \
  uv pip install --python .ai-local/envs/t50-mlx/bin/python \
  -r environments/macos-m4/mlx-baseline-requirements.txt
```

Prepare a local directory containing the pinned revision's `config.json`,
`generation_config.json`, `model.safetensors`, `tokenizer.json`, and
`tokenizer_config.json`. Model weights remain outside Git. Run the real M4
baseline with:

```bash
PYTHONPATH=src .ai-local/envs/t50-mlx/bin/python \
  -m slm_lab.backends.mlx_baseline \
  --model-path "${SLM_LAB_MLX_MODEL_DIR}" \
  --output-dir results/raw/apple/baseline
```

The runner verifies the source checksums, all five T10 tokenizer canaries, and
the T11 three-token generation oracle before writing
`results/raw/apple/baseline/mlx-lm-baseline-run-v2.json`. The single run bundle
is validated against `mlx-baseline-run-v2.schema.json` and links its timestamp,
run ID, source commit, runner, benchmark protocol, fixtures, host/runtime,
workload, raw samples, and recomputed summaries. The adjacent
`.json.sha256` file anchors the complete result digest independently of the
document's self-digest.

Timed regions fence `mlx_lm.generate.generation_stream` directly. The TTFT
probe uses `generate_step(max_tokens=0)` to materialize the first token without
scheduling a later decode. The three-token library loop includes MLX-LM's one
unreturned look-ahead token and labels its throughput accordingly. The result
is an MLX Metal GPU baseline and makes no Apple Neural Engine (ANE) execution
claim.

T51 owns the custom MLX runtime. T52 owns the four-context sweep, Instruments,
MLX profiling, power, and thermal evidence. Null Xcode, Metal compiler, or
Instruments versions in T50 evidence mean the command was checked and the
tool was unavailable, not compatibility.

The storage preflight writes and removes only a tiny temporary probe. It does
not download weights or perform a heavy storage benchmark.
