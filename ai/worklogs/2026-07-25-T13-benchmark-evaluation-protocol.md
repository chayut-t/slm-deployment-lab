# T13: Benchmark Evaluation Protocol

Date: 2026-07-25
Task: `T13`
Visibility: `public`
Status: completed

## Outcome

Froze a machine-checkable benchmark and evaluation contract for the exact T10
workloads. The contract distinguishes graph, runtime-stage, generation-loop,
end-to-end request, and cold-start evidence; fixes warm-up/repetition and
synchronization rules; defines raw-sample-backed statistics; and retains full
system identity for cross-platform claims. No performance, quality, power, or
hardware measurements were created.

## Changes

- Added `benchmark-protocol-v1.json` with the four workload contexts, decode
  probes, five timing classes, non-timing quality/numerical/power evidence,
  TTFT/throughput definitions, synchronization, invalid-series handling,
  memory/power methods, numerical metrics, and reporting boundaries.
- Added a strict JSON Schema for individual metric records. The semantic
  validator checks the protocol digest, T10 workload linkage, timing scope,
  base units, evidence method, raw sample count, valid/invalid accounting,
  recomputed summaries, and headline bootstrap interval.
- Implemented Hyndman–Fan type-7 quantiles, sample standard deviation, MAD,
  IQR, and a seeded 10,000-resample percentile-bootstrap median interval.
- Pinned lm-evaluation-harness `v0.4.12` at full commit
  `6d642546f4688648fced259eb3302efd36ece5af` and selected WikiText-2 raw,
  HellaSwag 1,000, and ARC Easy as regression sentinels. Dataset revisions,
  splits, selection, scoring, license handling, and no-row-commit policy are
  explicit.
- Added the deployment-benchmarking guide and its learner debrief checklist.
- Independent review hardening added the missing cold artifact/model load
  metric definitions, timing-class/metric allowlists, structured
  backend-specific synchronization, fresh-process identity evidence,
  mandatory token denominators, and exact academic-task cross-checks.
- Added adversarial regression tests that attempt graph/request/cold
  mislabeling, false synchronization, warm-as-cold evidence, missing or
  incorrect token denominators, and quality dataset/revision/split/selection
  drift.

## Verification

- `uv run python -m slm_lab.benchmark.protocol check --root .`
  - Passed protocol digest/semantics, T10 linkage, JSON Schema, academic
    subset, statistics, valid-result, and inconsistent-summary rejection.
- `uv run --extra dev pytest -q`
  - `88 passed, 2 skipped` after review corrections.
- `uv run --extra dev ruff check src tests`
  - Passed.
- `python3 scripts/ai/render_task_status.py --check`
  - Passed after completion metadata regeneration.
- `python3 scripts/repo/check_hygiene.py --all`
  - Passed.
- `git diff --check`
  - Passed.

## Decisions and evidence

- One result record has exactly one evidence scope. Composite reports link
  graph and end-to-end records rather than pooling their samples.
- Valid statistical outliers remain. Only predeclared integrity,
  environment, device/provider, runtime, timer, or external-interruption
  failures can invalidate a sample; an incomplete series is diagnostic-only
  and must be rerun in full.
- Warm performance series use 5/30 graph and runtime-stage repetitions or
  2/10 loop/request repetitions. Cold-start uses 0/5 fresh processes.
- Headline medians require a deterministic 95% bootstrap interval.
- Public dataset cards and the official harness release were inspected on
  2026-07-25. Read-only Git queries confirmed the full pinned HEAD revisions
  for WikiText, HellaSwag, and AI2 ARC. PIQA was excluded because its rendered
  license metadata and licensing discussion were inconsistent.
- A metric record must be a member of its timing class's frozen metric set;
  a valid metric name alone is insufficient. Cold component records also
  declare artifact/model-load inclusion separately.

## Risks and limitations

- Academic task rows and sample logs remain external/ignored; the repository
  stores only metadata, hashes, aggregates, and permitted excerpts.
- The academic subset is frozen methodology, not executed quality evidence.
  T81 must label the HellaSwag 1,000-row score as limited, not canonical.
- Power/thermal evidence remains unavailable until platform tasks run the
  specified instruments and ten-minute steady-state intervals.
- The T80 notebook does not yet exist; it must import reusable protocol logic
  rather than redefine statistics.

## Follow-up

- Newly unblocked tasks: T52 and T60 satisfy their T13 dependency once this
  commit is integrated; both retain other incomplete dependencies.
- Recommended next action: platform adapter authors should construct one
  schema-valid metric record before expensive sweeps, then T81 should link
  records without erasing system differences.
