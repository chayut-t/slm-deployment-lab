# T51: Custom MLX runtime

Status: completed
Owner: Codex T51 agent
Updated: 2026-07-30

## Objective

Implement a reusable Qwen3-0.6B MLX runtime with explicit prompt prefill,
one-token decode, and fixed-capacity GQA KV state that conforms to T12.

## Scope

### In scope

- Custom MLX Qwen3 modules and pinned local-weight loading.
- Explicit prefill and decode entry points.
- Head-major and sequence-major fixed-capacity cache variants.
- Grouped-query attention without materializing repeated K/V heads.
- Greedy multi-step generation and numerical/canary tests.
- Real local Apple M4 correctness evidence when the pinned runtime and weights
  are available.

### Out of scope

- The T52 four-context performance sweep, `mx.compile`, Instruments, power,
  thermal, and sustained profiling.
- MLX-LM as the runtime implementation.
- Apple Neural Engine claims.

## Dependencies and resources

- Required task dependencies: completed T12 and T50.
- Resource locks: `apple_m4_heavy`.
- External access: none required when the pinned model is available locally.
- Cost boundary: no paid services or jobs.

## Important paths

- Inputs: `src/slm_lab/contracts/static_cache.py`, the pinned model contract,
  T10/T11 fixtures, and T50 local model format.
- Outputs: `src/slm_lab/backends/mlx/`, `src/slm_lab/generation/mlx.py`,
  `tests/mlx/`.
- Shared contracts: immutable T12 shapes and cache update semantics.

## Milestones

- [x] Freeze custom runtime configuration and cache APIs against T12.
- [x] Implement grouped-query Qwen3 attention, model loading, prefill, decode,
  and generation.
- [x] Pass lightweight structural/state tests and real MLX numerical/canary
  tests where local runtime assets permit.
- [x] Complete repository lifecycle checks and publish the task worklog.

## Verification and acceptance

- Commands: focused `tests/mlx`, Ruff, full pytest, task-status check, and
  repository hygiene.
- Numerical or behavioral criteria: fixed-capacity state transitions,
  prefill/decode agreement with full-forward execution, deterministic
  multi-step generation, exact T11 canary where real weights are available.
- Hardware/profile evidence: record exact MLX/Metal/M4 identity for executed
  correctness evidence; performance profiling remains T52.

## Artifact and privacy handling

- Committed evidence: implementation, tests, and sanitized worklog.
- External artifacts: full model weights remain outside Git.
- Private/local material: task-local environments and temporary evidence stay
  under ignored `.ai-local/`.

## Decisions and discoveries

- 2026-07-30: Preserve eight physical K/V heads in both cache layouts and
  express GQA as a grouped query-head axis, never as repeated K/V storage.

## Progress and restart instructions

Implementation and task-local verification are complete. Reviewers should
start with `tests/mlx/test_real_qwen.py`, then inspect the GQA boundary in
`src/slm_lab/backends/mlx/model.py` and the functional cache updates in
`src/slm_lab/backends/mlx/cache.py`.
