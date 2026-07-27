# T50: MLX-LM baseline

Status: active
Owner: Codex T50 agent
Updated: 2026-07-27

## Objective

Establish a correct, versioned MLX-LM generation and performance baseline for
Qwen3-0.6B on the repository's exact Apple M4 Mac mini, compared with the T11
reference and reported without any unsupported ANE claim.

## Scope

### In scope

- Reproducible macOS/MLX-LM environment information.
- A reusable MLX baseline runner and structured result records.
- Output/canary comparison with the deterministic reference.
- Baseline latency and memory measurements with warm-up and synchronization
  boundaries made explicit.
- Exact host and runtime provenance.

### Out of scope

- The custom MLX cache runtime owned by T51.
- Apple Neural Engine execution claims.
- Unversioned or fabricated benchmark data.

## Dependencies and resources

- Required task dependencies: T11, completed.
- Resource locks: `apple_m4_heavy`.
- External access: model/runtime artifacts already available locally where
  possible; downloads require normal environment constraints.
- Cost boundary: no paid resources.

## Important paths

- Inputs: T10 fixtures and T11 deterministic reference.
- Outputs: `environments/macos-m4/`,
  `src/slm_lab/backends/mlx_baseline.py`,
  `results/raw/apple/baseline/`.
- Shared contracts: frozen canaries and benchmark timing terminology.

## Milestones

- [x] Audit the current M4 host, Python, MLX, and model artifact state.
- [x] Implement a versioned MLX-LM baseline runner and result schema.
- [x] Produce real local correctness, latency, and memory evidence.
- [x] Verify canaries and provenance, including the explicit no-ANE boundary.
- [ ] Pass independent fresh-agent review and address all findings.

## Verification and acceptance

- Commands: focused MLX baseline tests, task-status check, repository hygiene.
- Numerical or behavioral criteria: output passes frozen canary checks and
  timing records are reproducible and honestly labeled.
- Hardware/profile evidence: exact Mac model, OS, MLX/MLX-LM revisions,
  latency, and memory.

## Artifact and privacy handling

- Committed evidence: small structured results, source, environment files,
  active plan, and draft sanitized worklog pending final review.
- External artifacts: model weights remain under the artifact root.
- Private/local material: large/raw traces and machine-private details.

## Decisions and discoveries

- 2026-07-27: This task owns only the MLX-LM baseline; T51 will own the custom
  runtime and explicit cache implementation.
- 2026-07-27: The existing project environments did not contain MLX. The
  measured task-local environment pins MLX 0.32.0, MLX-LM 0.31.3, and their
  exact runtime dependencies without changing the shared lockfile.
- 2026-07-27: All five T10 token sequences matched exactly and MLX-LM
  reproduced the T11 tokens `576, 8356, 3950`.
- 2026-07-27: Review fixes fence MLX-LM's actual generation stream. Median
  no-look-ahead TTFT was 39.216 ms. The three-returned-token loop, including
  one unreturned look-ahead, had median latency 78.332 ms and 38.299 returned
  output tokens/second including prefill and look-ahead. MLX peak memory was
  1,255,817,508 bytes.
- 2026-07-27: The v2 run bundle records source commit
  `e8c7e2dd33fa29f85d05004e16d521dad4ca99e0`, validates against an exact JSON
  Schema, cross-checks Git blobs and raw-sample summaries, and uses an external
  digest anchor in addition to its self-digest.
- 2026-07-27: A second review fix binds the generation canary to the exact T11
  prompt-token digest and freezes the canonical TTFT, generation-loop,
  look-ahead, and model-load boundary semantics in both schema and validator.

## Progress and restart instructions

Implementation, exact environment pins, real M4 evidence, and a draft public
worklog are ready for independent review. A fresh agent should review the
committed state, rerun the evidence validator and focused checks, address any
findings, then finalize the worklog, move this plan to completed, and mark T50
completed in the task graph.
