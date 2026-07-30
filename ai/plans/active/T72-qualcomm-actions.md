# T72: Manual GitHub Actions AI Hub workflow

Status: active
Owner: Codex T72 agent
Updated: 2026-07-30

## Objective

Add a default-branch-only, manually dispatched GitHub Actions workflow that
selects one Qualcomm AI Hub target/context/precision/stage tuple, retrieves a
trusted private request bundle, and delegates job execution to the existing
T30 local stage scripts. Teach the learner how the security, artifact, and
hardware boundaries work without creating secrets or running external jobs.

## Scope

### In scope

- Add `.github/workflows/qualcomm-benchmark.yml` with typed target, context,
  precision, stage, and trusted request-bundle inputs.
- Add `.github/workflows/qualcomm-request-bundle.yml` as the fixed reviewed
  no-Qualcomm-secret producer. Its trust terminus is a separately pre-staged
  same-repository release asset bound by tag, asset name, and reviewed SHA-256.
- Gate secret-bearing work to the upstream default branch and a protected
  GitHub environment.
- Pin the Python and official GitHub actions used by the workflow.
- Keep `scripts/qualcomm/{compile,inference,profile}.py` as the only job
  implementation entry points.
- Add a detailed learning and operations guide.
- Add offline workflow structure, security, and delegation regression tests.

### Out of scope

- Creating or changing GitHub secrets or environment protection rules.
- Staging, creating, or publishing a GitHub release or source asset.
- Uploading a real request bundle, dispatching the workflow, or submitting an
  AI Hub job.
- Changing the T30 adapters or their request contract.
- Publishing remote GitHub state, large model artifacts, or private service
  output.

## Dependencies and resources

- Required task dependencies: T03 and T30, both completed in
  `ai/tasks/task_graph.yaml`.
- Resource locks: none for offline implementation; a real learner-approved
  dispatch would consume Qualcomm service capacity.
- External access: GitHub Actions and Qualcomm AI Hub are learner-controlled
  and intentionally not used during implementation.
- Cost boundary: no service job is submitted; the guide requires checking
  quota/cost immediately before dispatch.

## Important paths

- Inputs: `ai/tasks/definitions/T72.yaml`, `scripts/qualcomm/README.md`,
  `src/slm_lab/deployment/qualcomm/ai_hub.py`, T03/T30 worklogs.
- Outputs: `.github/workflows/qualcomm-benchmark.yml`,
  `.github/workflows/qualcomm-request-bundle.yml`,
  `docs/learning/github_actions_for_ai_hub.md`, `tests/workflows/`.
- Shared contracts: T30 schema-v2 request files and sanitized manifest
  boundary; T03 generated task status, privacy, and completion rules.

## Milestones

- [x] Workflow dispatch inputs and fork/default-branch/environment gates are
  mechanically validated.
- [x] Workflow downloads only a fixed reviewed producer workflow/revision
  request bundle, validates its manifest, request semantics, paths, and
  digests before secret configuration, configures the pinned client without
  printing the token, and calls exactly one local T30 stage script.
- [x] Guide explains learner setup, request-bundle lineage, security model,
  dispatch, evidence, failure handling, and the unverified external boundary.
- [ ] Focused and repository-wide required checks pass, with a public worklog and T72
  completion metadata.

## Verification and acceptance

- Commands:
  - `PYTHONPATH=src python3 -m unittest discover -s tests/workflows -p 'test_*.py' -v`
  - `PYTHONPATH=src <project-python> -m pytest -q`
  - `<project-python> -m ruff check tests/workflows`
  - `python3 scripts/ai/render_task_status.py --check`
  - `python3 scripts/repo/check_hygiene.py --all`
  - `git diff --check`
- Numerical or behavioral criteria: all enumerated target/context/precision/
  stage choices parse; secret use is confined to the protected job; scripts
  are selected from a fixed allowlist rather than interpolated command text.
- Hardware/profile evidence: none claimed; a real dispatch and Qualcomm
  result remain learner-controlled and unverified.

## Artifact and privacy handling

- Committed evidence: workflow, guide, offline tests, plan, worklog, and
  generated task status.
- External artifacts: private request bundle, model/dataset/compiled binaries,
  raw profile, and uploaded workflow result artifact.
- Private/local material: API token, client config, request paths, raw SDK
  output, service job identity/URLs, accounts, and quota details.

## Decisions and discoveries

- 2026-07-30: Preserve T30's independent-stage restart boundary: one dispatch
  runs one of compile, inference, or profile from a content-addressed request
  bundle rather than chaining service job IDs.
- 2026-07-30: A secret-bearing run must use the upstream default branch and a
  protected `qualcomm-ai-hub` environment; forks and alternate refs skip it.
- 2026-07-30: The request bundle must come from a successful same-repository
  default-branch Actions run, and environment approval remains the last human
  check before external submission.
- 2026-07-30: Offline implementation and acceptance checks passed. Real secret
  setup, environment approval, bundle production, workflow dispatch, and
  Qualcomm execution remain explicitly learner-controlled and unverified.
- 2026-07-30: Independent review found that same-repository/default-branch
  provenance alone did not bind the bundle to a reviewed producer workflow or
  prove tuple/path/digest semantics. T72 is reopened until those P1 findings
  are fixed and freshly reviewed.
- 2026-07-30: The coordinator authorized the narrow producer-workflow scope
  expansion. The producer uses a content-addressed same-repository release
  asset as the non-dangling source mechanism; staging that asset remains an
  explicit learner-controlled prerequisite and is never performed by T72.

## Progress and restart instructions

Review fixes are in progress. Validate the producer, immutable bundle manifest,
selected request semantics, source/archive safety, and runner-private paths;
add adversarial regressions, then rerun focused/full checks. Keep T72 in
progress with no graph worklog until fresh review accepts it.
