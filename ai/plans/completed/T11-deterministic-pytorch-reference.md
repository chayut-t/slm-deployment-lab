# T11: Deterministic PyTorch reference

Status: completed
Owner: Codex T11 agent
Updated: 2026-07-25

## Objective

Provide the correctness-first PyTorch oracle for Qwen3-0.6B: pinned model
loading, full-forward execution, cached one-token decoding, deterministic
greedy generation, and reproducible numerical evidence that later graph and
runtime tasks can consume.

## Scope

### In scope

- Load only the immutable Qwen3-0.6B revision from the T00 model contract.
- Make deterministic execution settings, reference dtype, device, and package
  versions explicit.
- Compare full-prefix and cached next-token logits at every decode step.
- Implement greedy generation with the T10 tie-break, EOS, and output policies.
- Define and validate a small, committed numerical fixture derived from an
  authored T10 canary.
- Test the reusable algorithms with a deterministic PyTorch causal-model
  fixture and optionally reproduce the golden Qwen evidence when its public
  weights are locally available.

### Out of scope

- Static KV-cache tensor contracts (T12), ONNX export (T20), benchmarking
  methodology (T13), quantization, or hardware profiling.
- Committing model weights, complete vocabulary-sized logit arrays, cache
  tensors, local cache paths, credentials, or private prompts.
- Claiming FP32, ONNX, MLX, QNN, or CUDA parity.

## Dependencies and resources

- Required task dependencies: T10, completed and integrated.
- Resource locks: none.
- External access: public pinned Qwen weights only for optional golden-evidence
  regeneration.
- Cost boundary: no paid services or cloud jobs.

## Important paths

- Inputs: `configs/models/qwen3-0.6b.yaml`,
  `configs/workloads/t10-token-fixtures.json`,
  `tests/fixtures/t10/token-fixtures-v1.json`.
- Outputs: `src/slm_lab/models/`, `src/slm_lab/generation/`,
  `tests/reference/`.
- Shared contracts: T10 greedy generation policy and immutable model revision.

## Milestones

- [x] Pinned model loader and deterministic execution metadata are implemented.
- [x] Full-forward, cache-prefill, one-token decode, and generation APIs are
  reusable and reject unsupported or ambiguous model outputs.
- [x] Stepwise parity metrics and frozen tolerances detect numerical or token
  drift.
- [x] Deterministic tests and available golden-Qwen verification pass.
- [x] Public worklog, completed plan, and task graph record T11 completion.

## Verification and acceptance

- Commands:
  - `pytest -q tests/reference`
  - `pytest -q`
  - `ruff check src tests`
  - `python3 scripts/ai/render_task_status.py --check`
  - `python3 scripts/repo/check_hygiene.py --all`
- Numerical or behavioral criteria:
  - Full-prefix and cached next-token logits pass the committed absolute,
    relative, cosine, top-k, and token-agreement thresholds at every step.
  - Repeated greedy runs return exactly the same token IDs and evidence digest.
  - Loader metadata matches the frozen Qwen model and tokenizer revisions.
- Hardware/profile evidence: not applicable; CPU/MPS are reference execution
  locations, not benchmark targets in this task.

## Artifact and privacy handling

- Committed evidence: authored fixture input/output IDs, compact logit
  fingerprints and metrics, exact revisions/versions, tolerances, and commands.
- External artifacts: Qwen safetensors remain under the configured artifact or
  Hugging Face cache and are never committed.
- Private/local material: cache paths and environment-specific diagnostics
  remain ignored and are not printed into public evidence.

## Decisions and discoveries

- 2026-07-25: Use a model-protocol boundary rather than coupling generation
  logic to a concrete Transformers class, so tests can exercise cache behavior
  deterministically without distributing Qwen weights.
- 2026-07-25: Store compact numerical fingerprints and comparison metrics
  instead of complete vocabulary logits; exact regeneration remains the
  authority.

## Progress and restart instructions

T11 is complete. The implementation, deterministic PyTorch fixture, pinned
Qwen fixture, tests, public worklog, and task metadata are ready for an
independent review. T12 and T50 may consume the committed reference after this
branch is integrated.
