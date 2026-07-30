# T20: Four-context ONNX export matrix

Status: active
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

- [ ] Implement static prefill/decode wrappers and deterministic export CLI.
- [ ] Enforce external data and validate ONNX I/O against T12.
- [ ] Export and hash all eight graphs on external storage.
- [ ] Commit four context manifests with exact source/toolchain provenance.
- [ ] Pass focused tests, full repository tests, task-status, and hygiene.
- [ ] Address independent review findings.

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

## Progress and restart instructions

Implement the exporter and focused tests first. Make an implementation commit,
then run the real export matrix from that commit so each manifest can identify
the exact exporting source revision without circular provenance.
