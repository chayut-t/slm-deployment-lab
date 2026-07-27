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

- [ ] Audit the current M4 host, Python, MLX, and model artifact state.
- [ ] Implement a versioned MLX-LM baseline runner and result schema.
- [ ] Produce real local correctness, latency, and memory evidence.
- [ ] Verify canaries and provenance, including the explicit no-ANE boundary.
- [ ] Pass independent review and address all findings.

## Verification and acceptance

- Commands: focused MLX baseline tests, task-status check, repository hygiene.
- Numerical or behavioral criteria: output passes frozen canary checks and
  timing records are reproducible and honestly labeled.
- Hardware/profile evidence: exact Mac model, OS, MLX/MLX-LM revisions,
  latency, and memory.

## Artifact and privacy handling

- Committed evidence: small structured results, source, environment files,
  completed plan, and sanitized worklog.
- External artifacts: model weights remain under the artifact root.
- Private/local material: large/raw traces and machine-private details.

## Decisions and discoveries

- 2026-07-27: This task owns only the MLX-LM baseline; T51 will own the custom
  runtime and explicit cache implementation.

## Progress and restart instructions

Inspect the frozen T10/T11 fixtures and the currently installed MLX stack,
then build the baseline runner around real local measurements and commit only
small, reproducible evidence.
