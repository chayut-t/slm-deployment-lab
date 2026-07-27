# T50: MLX-LM Baseline

Date: 2026-07-27
Task: `T50`
Visibility: `public`
Status: completed

## Outcome

The T50 implementation checkpoint establishes a real, versioned MLX-LM
baseline for the immutable Qwen3-0.6B revision on the repository's exact Apple
M4 Mac mini. The runner
validated all five T10 tokenizer canaries and reproduced the T11 PyTorch/BF16
greedy generation oracle exactly: token IDs `576, 8356, 3950`.

The synchronized warm baseline used two warm-ups and ten retained
measurements. For the 18-token prompt and three-token output, median
time-to-first-token was 39.216 ms. The pinned MLX-LM loop returned three
tokens in a median 78.332 ms while also computing one unreturned look-ahead
token, for 38.299 returned output tokens/second including prefill and that
look-ahead. MLX reported 1,255,817,508 bytes peak memory for every measured
region. This small canary is a correctness/performance baseline, not the T52
four-context sweep.

## Changes

- Added a reusable MLX-LM runner that verifies the immutable model and
  tokenizer checksums, loads the local checkpoint without remote code,
  validates all T10 tokenization canaries, and rejects T11 generation drift.
- Added explicit pre/post timer fences on
  `mlx_lm.generate.generation_stream`. TTFT uses
  `generate_step(max_tokens=0)` to avoid scheduling a later decode, while the
  three-token library-loop metric accounts for its one unreturned look-ahead.
- Added one schema-validated v2 run bundle linking run ID/time, clean source
  commit, runner, schema, benchmark protocol, fixtures, model, canaries,
  host/runtime, workload, raw samples, and recomputed summaries. An external
  digest anchor rejects a document whose self-digest alone was recomputed.
- Bound the generation canary's prompt-token digest to the canonical JSON
  digest of the exact T11 prompt token IDs in both schema and semantic
  validation.
- Froze the canonical TTFT, generation-loop, look-ahead, and model-load
  boundary semantics in the schema and validator so contradictory prose
  cannot accompany otherwise valid measurements.
- Added dedicated repetition, synchronization, schema, provenance,
  cross-field, and adversarial validator tests.
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
  - Passed all tokenizer and generation canaries and wrote run
    `t50-mlx-lm-20260727T155820Z-e8c7e2dd33fa` from clean source commit
    `e8c7e2dd33fa29f85d05004e16d521dad4ca99e0`.
- `PYTHONPATH=src uv run --extra dev --locked python -m
  slm_lab.backends.mlx_baseline
  --validate
  results/raw/apple/baseline/mlx-lm-baseline-run-v2.json`
  - Schema, external digest anchor, Git blob provenance, immutable contracts,
    exact environment, canaries, repetitions, raw samples, throughput, and
    summaries passed.
- `uv run --extra dev --locked ruff check
  src/slm_lab/backends/mlx_baseline.py
  tests/backends/test_mlx_baseline.py`
  - Passed.
- `PYTHONPATH=src uv run --extra dev --locked python -m pytest -q
  tests/backends/test_mlx_baseline.py`
  - 16 passed.
- `PYTHONPATH=src uv run --extra dev --locked python -m pytest -q
  tests/backends/test_mlx_baseline.py
  tests/repo/test_t10_fixtures.py tests/repo/test_model_contract.py
  tests/reference/test_model_contract.py
  tests/reference/test_pytorch_reference.py`
  - 42 passed, 3 intentional upstream/real-weight-gated skips.
- `PYTHONPATH=src uv run --extra dev --locked python -m pytest -q`
  - 134 passed, 3 intentional external/upstream-gated skips.
- `uv run --extra dev --locked ruff check src tests`
  - Passed.
- `python3 scripts/ai/render_task_status.py --check`
  - Passed after T50 was marked completed and generated status was refreshed.
- `python3 scripts/repo/check_hygiene.py --all`
  - Passed with the final completed lifecycle file set.
- Fresh independent review of checkpoint
  `31f07dcfa3d40ee01fd5f3707c414778c8dcc3ce`
  - Approved with no findings.
  - Reproduced the evidence validator, Ruff, dedicated `16 passed`, focused
    `42 passed, 3 skipped`, full `134 passed, 3 skipped`, task-status, hygiene,
    and clean-status checks.
  - Confirmed the exact T11 prompt digest binding, canonical timing semantics,
    source Git blobs, external digest anchor, stream fences, no-look-ahead
    TTFT, and one unreturned generation-loop look-ahead.

## Decisions and evidence

- Exact host: Mac mini, `Mac16,10`, model `MU9D3VC/A`, Apple M4,
  16 GiB unified memory, arm64, macOS 15.7.7 build 24G720.
- Exact runtime: Python 3.11.13, MLX 0.32.0, MLX-LM 0.31.3,
  MLX-Metal 0.32.0, Transformers 5.14.1, Tokenizers 0.22.2,
  Safetensors 0.8.0, and NumPy 2.4.6.
- MLX reported `Device(gpu, 0)`, device name `Apple M4`, and architecture
  `applegpu_g16g`. This is MLX Metal GPU evidence and does not establish Apple
  Neural Engine (ANE) execution.
- Provenance source commit:
  `e8c7e2dd33fa29f85d05004e16d521dad4ca99e0`. Evidence digest:
  `25af82bcc8372d0817341b48af47f2c19877e59f4fc8fd75458e61b8e59477de`.
- Model loading took 0.469 seconds in the measurement process and was outside
  steady-state timing. File-cache state was uncontrolled, so the observation
  is not labeled cold start.
- The full 1,503,300,328-byte BF16 Safetensors file remained external. Its
  committed evidence checksum is
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.

## Risks and limitations

- The baseline covers one short frozen correctness canary. T52 owns the
  128/512/1,024/4,096-token sweep, sustained runs, Instruments traces, power,
  thermal, swap, and detailed profiling.
- TTFT is a separate first-token-materialization probe. The generation-loop
  metric is not directly comparable to a runtime that does not compute an
  extra look-ahead token; the metric name and workload record preserve that
  boundary.
- Full Xcode, the standalone Metal compiler, and Instruments were not
  installed. The exact failed check commands are retained with null values;
  this does not block the T50 MLX runtime baseline.
- The project-root dependency lock intentionally does not install MLX. The
  exact macOS-only requirements file and ignored task-local environment keep
  the ordinary repository environment portable.

## Follow-up

- T50 engineering acceptance and independent review are complete. T51 now
  needs only its T12 dependency.
- The learner debrief below remains intentionally unchecked and user-owned. It
  is a study follow-up, not a blocker for the completed engineering task.

## Learner debrief checklist

These study items require explicit learner confirmation and therefore remain
unchecked:

- [ ] Compare the exact-token parity in
  `results/raw/apple/baseline/mlx-lm-baseline-run-v2.json` with the T11
  PyTorch reference.
- [ ] Explain why the timing boundary in
  `results/raw/apple/baseline/mlx-lm-baseline-run-v2.json` separates
  no-look-ahead TTFT from a library loop containing one look-ahead token.
- [ ] Verify the exact host/runtime identity in
  `results/raw/apple/baseline/mlx-lm-baseline-run-v2.json` and explain why an
  MLX Metal result is not an ANE claim.
