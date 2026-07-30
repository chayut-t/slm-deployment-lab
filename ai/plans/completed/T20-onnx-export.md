# T20: Four-context ONNX export matrix

Status: completed
Owner: Codex T20 agent
Updated: 2026-07-30

## Objective

Export reproducible, fixed-shape Qwen3-0.6B reference ONNX prefill and
one-token decode graphs for the T12 S128, S512, S1024, and S4096 contracts.
Keep graph payloads and weights on external storage while committing exact
commands, contracts, versions, and content hashes.

## Scope

### In scope

- A deterministic PyTorch-to-ONNX wrapper for static prefill and decode.
- Exact ONNX input/output conformance checks against the T12 contracts.
- External-data packaging and checksums for every graph and data shard.
- Four context manifests containing both prefill and decode provenance.
- Focused exporter, manifest, shape, and failure-path tests.

### Out of scope

- ONNX Runtime numerical parity, which belongs to T21.
- QNN-specific graph transformations, which belong to T22.
- Compiler acceptance, accelerator placement, latency, or hardware claims.
- Committing model weights or ONNX graph payloads.

## Dependencies and resources

- Required task dependencies: T12, completed.
- Resource locks: `t9_heavy_io`, held by the T20 task claim.
- External access: none for execution; the pinned model is locally cached.
- Cost boundary: no paid resources and no external jobs.

## Important paths

- Inputs: `src/slm_lab/contracts/static_cache.py`,
  `configs/models/qwen3-0.6b.yaml`, pinned local Hugging Face snapshot.
- Outputs: `src/slm_lab/export/`, `configs/models/`,
  `results/manifests/onnx/`, `tests/export/`.
- Shared contracts: T12 tensor/cache contract and T00 model identity.

## Milestones

- [x] Implement static prefill/decode wrappers and deterministic export CLI.
- [x] Enforce external data and validate ONNX I/O against T12.
- [x] Export and hash all eight graphs on external storage.
- [x] Commit four context manifests with exact source/toolchain provenance.
- [x] Pass focused tests, full repository tests, task-status, and hygiene.
- [x] Prepare the implementation and evidence for independent review.

## Verification and acceptance

- Commands: focused exporter tests, real export matrix validation, full pytest,
  Ruff, task-status check, and repository hygiene.
- Numerical or behavioral criteria: all public ONNX I/O names, dtypes, and
  dimensions exactly match T12; all graph/data hashes match external files.
- Hardware/profile evidence: not applicable; no compiler or device claim.

## Artifact and privacy handling

- Committed evidence: implementation, tests, configuration, manifests,
  completed plan, and sanitized worklog.
- External artifacts: ONNX protobufs and data shards under
  `SLM_LAB_ARTIFACT_ROOT/onnx/reference/T20/`.
- Private/local material: raw command logs and agent session identifiers only.

## Decisions and discoveries

- 2026-07-30: Use the pinned T11 eager-attention model loader and T12-generated
  contracts; do not duplicate context-specific interfaces.
- 2026-07-30: Treat export conformance as graph evidence only. T21 remains
  responsible for runtime numerical parity and graph-risk inspection.
- 2026-07-30: Transformers 4.51.3 Qwen3 requires a `Cache` implementation
  rather than legacy K/V tuples. Decode constructs `DynamicCache` from the
  explicit T12 inputs while retaining tensor-only public ONNX boundaries.
- 2026-07-30: All eight real graphs passed ONNX checker, exact static-I/O, and
  external-data validation. Four manifests tie them to exporter source commit
  `631fd70bcff9b73b81c08a2a2e0127cad07f09ca`.

## Progress and restart instructions

T20 implementation and evidence are complete and ready for independent review.
T21 can start from the completed branch after the review findings, if any, are
addressed and integrated.
