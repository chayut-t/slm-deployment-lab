# T20: Four-context ONNX export matrix

Status: active — review fixes ready for rereview
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
- [x] Address P1 deterministic-manifest and exporter-provenance findings.
- [x] Address P2 configured T10 fixture and workload-provenance findings.
- [x] Anchor exporter revision and runtime Python outside the manifests.
- [x] Bind configuration to the fixed code-pinned and HEAD-committed blob.
- [x] Enforce actual Python against the recorded export runtime.
- [ ] Pass fresh independent rereview.

## Verification and acceptance

- Commands: focused exporter tests, real export matrix validation, full pytest,
  Ruff, task-status check, and repository hygiene.
- Numerical or behavioral criteria: all public ONNX I/O names, dtypes, and
  dimensions exactly match T12; all graph/data hashes match external files.
- Hardware/profile evidence: not applicable; no compiler or device claim.

## Artifact and privacy handling

- Committed evidence: implementation, tests, configuration/attestation,
  manifests, active plan, and sanitized in-progress worklog.
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
- 2026-07-30: Independent review found that validation covered artifacts and
  selected identity fields but not every deterministic manifest claim.
  Validation now reconstructs the full manifest and binds the historical
  exporter commit to exact Git blobs for source, configuration, model, and T10
  inputs.
- 2026-07-30: Independent review found that export configuration did not
  control the tracing fixture and same-length token drift could pass.
  Configuration now resolves the fixture actually consumed by export, runs
  the frozen T10 validators and canonical digest check, and records the exact
  workload prompt/token hashes in each manifest.
- 2026-07-30: Second rereview found that a coherently changed manifest could
  select another valid ancestor or invent a paired Python version. A committed
  run attestation in the export configuration now independently pins the
  DynamicCache-capable exporter commit, runtime Python, source weights, eight
  graph hashes, and external-data hash.
- 2026-07-30: Third rereview found that callers could coherently substitute
  the parsed configuration itself. Loading is now restricted to the fixed
  tracked path whose bytes must match both a code-pinned digest and its exact
  `HEAD` Git blob; every in-memory field is compared with a fresh trusted
  parse before evidence validation.
- 2026-07-30: Third rereview also found that Python was recorded but not
  compared with the executing interpreter. Runtime validation now requires
  actual Python `3.11.15`, with a direct mismatch regression.

## Progress and restart instructions

Both third-rereview findings are implemented with coherent-tamper regressions
for every pinned field. Focused validation is green; the only repository-wide
failure remains the known concurrent-claim fixture baseline recorded in the
worklog. Commit this trust-root fix and request another fresh independent
rereview. Keep T20 `in_progress` with a null graph worklog until rereview
passes.
