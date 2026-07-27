# T12: Static cache and tensor contract

Status: active
Owner: Codex T12 agent
Updated: 2026-07-27

## Objective

Freeze explicit, machine-checkable prefill and one-token decode tensor
contracts for Qwen3-0.6B at the 128, 512, 1,024, and 4,096-token context
variants. The contract must define GQA-aware cache shapes and updates that
reproduce the deterministic T11 reference.

## Scope

### In scope

- Prefill and decode input/output tensor names, shapes, dtypes, and layouts.
- GQA-aware per-layer key/value cache dimensions and byte accounting.
- Contract serialization or reusable Python interfaces under
  `src/slm_lab/contracts/`.
- Conformance and reference-update tests under `tests/contracts/`.
- Architecture documentation that later ONNX and MLX tasks can consume.

### Out of scope

- ONNX export or compiler-specific graph rewrites.
- Runtime-specific performance measurements.
- Changing the frozen model, tokenizer, or benchmark contracts.

## Dependencies and resources

- Required task dependencies: T11, completed.
- Resource locks: none.
- External access: none.
- Cost boundary: no paid resources.

## Important paths

- Inputs: `src/slm_lab/models/`, T11 reference tests and fixtures.
- Outputs: `src/slm_lab/contracts/`, `docs/architecture/`,
  `tests/contracts/`.
- Shared contracts: T10 token fixtures and the T11 deterministic reference.

## Milestones

- [ ] Inventory Qwen3-0.6B dimensions and T11 cache behavior.
- [ ] Implement explicit prefill/decode contracts for all four contexts.
- [ ] Verify multi-step cache updates against the T11 reference.
- [ ] Document tensor diagrams, layouts, and cache-byte calculations.
- [ ] Pass independent review and address all findings.

## Verification and acceptance

- Commands: focused contract tests, task-status check, repository hygiene.
- Numerical or behavioral criteria: cache updates reproduce the T11 reference;
  every tensor name, dtype, layout, and static dimension is explicit.
- Hardware/profile evidence: not applicable.

## Artifact and privacy handling

- Committed evidence: source, tests, architecture guide, completed plan, and
  sanitized worklog.
- External artifacts: none.
- Private/local material: raw agent sessions only.

## Decisions and discoveries

- 2026-07-27: Work begins from the committed dependency-complete `main`
  checkpoint and is isolated on `task/T12-static-cache-contract`.

## Progress and restart instructions

Inspect the T11 reference cache APIs first, then implement the smallest
runtime-neutral contract surface that later ONNX and MLX tasks can import.
