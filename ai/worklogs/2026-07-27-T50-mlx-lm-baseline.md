# T50: MLX-LM Baseline

Date: 2026-07-27
Task: `T50`
Visibility: `public`
Status: draft

## Outcome

The T50 implementation checkpoint establishes a real, versioned MLX-LM
baseline for the immutable Qwen3-0.6B revision on the repository's exact Apple
M4 Mac mini. The runner
validated all five T10 tokenizer canaries and reproduced the T11 PyTorch/BF16
greedy generation oracle exactly: token IDs `576, 8356, 3950`.

The synchronized warm baseline used two warm-ups and ten retained
measurements. For the 18-token prompt and three-token output, median
time-to-first-token was 55.979 ms, median complete generation-loop latency was
87.620 ms, and median throughput including prefill was 34.243 output
tokens/second. MLX reported 1,255,817,508 bytes peak memory for every measured
run. This small canary is a correctness/performance baseline, not the T52
four-context sweep.

## Changes

- Added a reusable MLX-LM runner that verifies the immutable model and
  tokenizer checksums, loads the local checkpoint without remote code,
  validates all T10 tokenization canaries, and rejects T11 generation drift.
- Added synchronized generation-loop measurement with explicit model-load,
  lazy-evaluation, warm-up, repetition, sample-retention, and memory
  boundaries.
- Added self-digested structured parity, host/runtime, and performance
  evidence under `results/raw/apple/baseline/`.
- Added an exact task-local MLX environment pin and macOS reproduction guide
  without changing the shared cross-platform dependency lock.
- Sanitized host evidence to retain the Mac model, model number, chip, memory,
  processor topology, OS, and MLX Metal device while excluding serial,
  platform UUID, and provisioning identifiers.

## Verification

- `PYTHONPATH=src .ai-local/envs/t50-mlx/bin/python -m
  slm_lab.backends.mlx_baseline --model-path
  .ai-local/models/qwen3-0.6b --output-dir
  results/raw/apple/baseline --warmup-repetitions 2
  --measured-repetitions 10`
  - Passed all tokenizer and generation canaries and wrote three evidence
    documents from real MLX Metal execution on the Apple M4.
- `PYTHONPATH=src .venv/bin/python -m slm_lab.backends.mlx_baseline
  --validate results/raw/apple/baseline/host-runtime-v1.json
  results/raw/apple/baseline/mlx-lm-parity-v1.json
  results/raw/apple/baseline/mlx-lm-performance-v1.json`
  - All evidence self-digests and explicit no-ANE claim boundaries passed.
- `ruff check src/slm_lab/backends/mlx_baseline.py`
  - Passed.
- `PYTHONPATH=src .venv/bin/python -m pytest -q
  tests/repo/test_t10_fixtures.py tests/repo/test_model_contract.py
  tests/reference/test_model_contract.py
  tests/reference/test_pytorch_reference.py`
  - 34 passed, 3 intentional upstream/real-weight-gated skips.
- `python3 scripts/ai/render_task_status.py --check`
  - Passed with T50 retained in progress pending independent review.
- `python3 scripts/repo/check_hygiene.py --all`
  - Passed with the final public file set.

## Decisions and evidence

- Exact host: Mac mini, `Mac16,10`, model `MU9D3VC/A`, Apple M4,
  16 GiB unified memory, arm64, macOS 15.7.7 build 24G720.
- Exact runtime: Python 3.11.13, MLX 0.32.0, MLX-LM 0.31.3,
  MLX-Metal 0.32.0, Transformers 5.14.1, Tokenizers 0.22.2,
  Safetensors 0.8.0, and NumPy 2.4.6.
- MLX reported `Device(gpu, 0)`, device name `Apple M4`, and architecture
  `applegpu_g16g`. This is MLX Metal GPU evidence and does not establish Apple
  Neural Engine (ANE) execution.
- Model loading took 0.703 seconds in the measurement process and was outside
  steady-state timing. File-cache state was uncontrolled, so the observation
  is not labeled cold start.
- The full 1,503,300,328-byte BF16 Safetensors file remained external. Its
  committed evidence checksum is
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.

## Risks and limitations

- The baseline covers one short frozen correctness canary. T52 owns the
  128/512/1,024/4,096-token sweep, sustained runs, Instruments traces, power,
  thermal, swap, and detailed profiling.
- MLX-LM pipelines generation and may enqueue a later decode before the first
  token is consumed. The recorded TTFT is the library's first-yield boundary,
  not an isolated prefill measurement.
- Full Xcode, the standalone Metal compiler, and Instruments were not
  installed. The exact failed check commands are retained with null values;
  this does not block the T50 MLX runtime baseline.
- The project-root dependency lock intentionally does not install MLX. The
  exact macOS-only requirements file and ignored task-local environment keep
  the ordinary repository environment portable.

## Follow-up

- Review state: implementation and evidence are ready for a fresh independent
  agent; T50 remains in progress until that review passes.
- After review: address any findings, set this worklog to completed, move the
  active plan to completed, mark T50 completed in the task graph, and
  regenerate task status. T51 will then need only its T12 dependency.

## Learner debrief checklist

- [ ] Compare the exact-token parity in
  `results/raw/apple/baseline/mlx-lm-parity-v1.json` with the T11 PyTorch
  reference.
- [ ] Explain why the timing boundary in
  `results/raw/apple/baseline/mlx-lm-performance-v1.json` is a generation-loop
  baseline rather than isolated prefill/decode evidence.
- [ ] Verify the exact host/runtime identity in
  `results/raw/apple/baseline/host-runtime-v1.json` and explain why an MLX
  Metal result is not an ANE claim.
