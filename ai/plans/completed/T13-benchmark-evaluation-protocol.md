# T13: Benchmark and evaluation protocol

Status: completed
Owner: Codex T13 agent
Updated: 2026-07-25

## Objective

Freeze a machine-checkable protocol for numerical, quality, latency, memory,
power, and reporting evidence so downstream Apple, NVIDIA, and Qualcomm tasks
measure the same workloads without erasing system differences.

## Scope

### In scope

- Freeze benchmark timing classes, synchronization boundaries, warm-up,
  repetition, invalid-run, and statistics policies.
- Define a strict result schema with provenance, timing scope, samples,
  computed summaries, evidence class, and comparison limitations.
- Pin the T10 workload matrix into the benchmark contract.
- Select and pin a small academic regression subset without committing dataset
  rows or reporting fabricated scores.
- Provide reusable offline validation and summary logic.
- Write the deployment-benchmarking learning guide and study checkpoint.

### Out of scope

- Running model, hardware, cloud, academic-quality, memory, power, or thermal
  measurements.
- Adding benchmark dependencies or downloading evaluation datasets.
- Producing cross-platform conclusions, which belongs to T81.
- Creating the T80 notebook.

## Dependencies and resources

- Required task dependencies: T10, completed and integrated.
- Resource locks: none.
- External access: read-only inspection of public dataset cards and the
  lm-evaluation-harness release/task definitions.
- Cost boundary: no paid resources and no external jobs.

## Important paths

- Inputs: `configs/workloads/t10-token-fixtures.json`,
  `tests/fixtures/t10/token-fixtures-v1.json`, `docs/project/plan.md`.
- Outputs: `configs/workloads/benchmark-protocol-v1.json`,
  `configs/workloads/benchmark-result-v1.schema.json`,
  `configs/workloads/academic-evaluation-v1.json`,
  `src/slm_lab/benchmark/`, `docs/learning/deployment_benchmarking.md`.
- Shared contracts: `ai/tasks/task_graph.yaml`,
  `ai/tasks/status.generated.md`.

## Milestones

- [x] Protocol fixes every timing class, workload, synchronization rule, and
  statistics definition.
- [x] Result-schema validation rejects missing provenance, mixed timing
  scopes, system-erasing comparisons, and inconsistent summaries.
- [x] Academic subset pins harness/task/dataset revisions, split, limit,
  scoring, prompt interface, and license handling.
- [x] Learning guide makes the measurement boundaries and review checkpoint
  concrete.
- [x] Required checks, worklog, completed task metadata, and local commit are
  complete.

## Verification and acceptance

- Commands:
  - `uv run python -m slm_lab.benchmark.protocol check --root .`
  - focused Python checks for statistics and invalid-result rejection
  - `uv run --extra dev ruff check src/slm_lab/benchmark`
  - `python3 scripts/ai/render_task_status.py --check`
  - `python3 scripts/repo/check_hygiene.py --all`
- Numerical or behavioral criteria:
  - Type-7 median/p90/p95, mean, sample standard deviation, MAD, and IQR are
    deterministically recomputed from retained raw samples.
  - Warm-up and measured repetition counts are fixed per timing class.
  - Graph, runtime-stage, generation-loop, end-to-end, and cold-start scopes
    cannot be silently combined.
  - Every comparison retains complete system identity and explicit
    non-comparable dimensions.
- Hardware/profile evidence: not applicable; this task creates methodology,
  not measurements.

## Artifact and privacy handling

- Committed evidence: protocol, schema, dataset metadata/revisions, validation
  logic, guide, plan, and public worklog.
- External artifacts: future dataset caches and raw profiles remain outside
  Git.
- Private/local material: session identity stays only in the ignored shared
  registry.

## Decisions and discoveries

- 2026-07-25: Treat each metric record as one timing scope; composite reports
  link records rather than averaging graph and end-to-end samples together.
- 2026-07-25: Keep all statistically valid samples. Exclusion is limited to
  predeclared integrity/environment failures, never post-hoc outlier removal.
- 2026-07-25: Use WikiText-2 raw, HellaSwag, and ARC Easy as regression
  sentinels; omit PIQA because its currently rendered public license metadata
  remains inconsistent, and commit no external rows.
- 2026-07-25: Resolve lm-evaluation-harness `v0.4.12` to full commit
  `6d642546f4688648fced259eb3302efd36ece5af`; future results also archive the
  resolved task-configuration hash.
- 2026-07-25: Keep non-timing quality, numerical, and power/thermal evidence
  explicit in the same result contract without forcing it into a graph or
  request timing class.
- 2026-07-25: Independent review showed that a strict outer schema was not
  enough: semantic validation must bind metrics to timing classes, sync
  methods to backends/platforms, throughput to actual token counts, cold
  evidence to fresh process identities, and quality records to the exact
  academic task definition.
- 2026-07-25: Idle-baseline subtraction is evidence, not a boolean
  presentation option; subtraction now requires a finite, non-negative
  measured watt value.

## Progress and restart instructions

T13 is complete. Downstream platform tasks should load the frozen protocol and
validate one representative result record before running a full sweep. T81
owns actual academic evaluation and cross-platform reporting; T80 owns the
related notebook and must import the reusable statistics implementation.
