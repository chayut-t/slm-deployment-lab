# T30: Workbench compile, inference, and profile adapters

Status: completed
Owner: Codex T30 agent
Updated: 2026-07-25

## Objective

Provide a safe, testable adapter layer and three independently runnable
commands for Qualcomm AI Hub Workbench compile, inference, and profile jobs.
Every stage must preserve source-to-result traceability while keeping
credentials, private job identifiers, job URLs, and raw service responses out
of committed or console-visible output.

## Scope

### In scope

- A dependency-injected Python adapter for compile, inference, and profile
  submission, waiting, artifact retrieval, and normalization.
- Versioned, sanitized JSON stage manifests with content hashes and public
  lineage references.
- Independent command-line entry points under `scripts/qualcomm/`.
- Mocked tests covering successful lifecycles, stage independence, lineage,
  failure behavior, and privacy/logging safeguards.
- Offline examples and documentation sufficient for T31 and T72 to invoke the
  adapter without submitting a job during T30.

### Out of scope

- Any real Workbench submission, Qwen compilation, paid job, device
  reservation, or credential configuration.
- Owning QNN candidate generation, target benchmark results, quantization, or
  Device Cloud execution.
- Committing downloaded binaries, datasets, raw profile output, job IDs,
  private URLs, account identifiers, credentials, or raw service responses.

## Dependencies and resources

- Required task dependencies: T01 and T02 are completed.
- Resource locks: `qai_hub_submission`; no external submission will be made.
- External access: none for implementation or verification.
- Cost boundary: zero; external and paid actions are prohibited for this task.

## Important paths

- Inputs: T01 artifact/storage contracts, T02 sanitized Workbench lifecycle,
  `ai/tasks/definitions/T30.yaml`, and the AI Hub client protocol represented
  through dependency injection.
- Outputs: `src/slm_lab/deployment/qualcomm/ai_hub.py`,
  `scripts/qualcomm/`, and `tests/deployment/qualcomm/`.
- Shared contracts: artifact SHA-256 provenance, exact tool/runtime/device
  revisions, sanitized public evidence, and ignored external artifacts.

## Milestones

- [x] Define validated stage requests, manifests, lineage, and redaction rules.
- [x] Implement compile, inference, and profile adapters plus independent CLIs.
- [x] Add mocked success, independence, traceability, and privacy tests.
- [x] Run focused and full repository checks and resolve all findings.
- [x] Publish the worklog, complete task status, and create one local commit.

## Verification and acceptance

- Commands:
  - `PYTHONPATH=src python3 -m unittest tests.deployment.qualcomm.test_ai_hub -v`
  - `PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'`
  - `ruff check src tests`
  - `python3 scripts/ai/render_task_status.py --check`
  - `python3 scripts/repo/check_hygiene.py --all`
  - `git diff --check`
- Behavioral criteria:
  - No console, exception, or public manifest exposes credentials, job IDs,
    job URLs, account identifiers, or raw service responses.
  - Compile, inference, and profile can be invoked independently from an
    explicit request and predecessor manifest.
  - Raw artifact/profile hashes and normalized results share stable public
    lineage without needing private identifiers.
  - Mocked failure and malformed-result paths fail closed.
- Hardware/profile evidence: T30 produces no hardware measurement; normalized
  profile shapes must preserve units, device/runtime identity, placement, and
  raw private artifact hashes for downstream real runs.

## Artifact and privacy handling

- Committed evidence: adapter code, scripts, mocked fixtures/tests, completed
  plan, and sanitized worklog.
- External artifacts: models, compiled binaries, datasets, inference outputs,
  and raw profiles live below `SLM_LAB_ARTIFACT_ROOT` and are represented only
  by SHA-256 and safe metadata.
- Private/local material: credentials, job IDs/URLs, account data, and raw
  service payloads remain unlogged and uncommitted under approved ignored
  storage.

## Decisions and discoveries

- 2026-07-25: Keep the Qualcomm SDK behind a small protocol so tests and stage
  orchestration are deterministic and do not import or authenticate to the
  optional external client.
- 2026-07-25: Public lineage uses request IDs derived from canonical,
  sanitized request content plus artifact hashes; service job identity is
  deliberately not part of the public contract.
- 2026-07-25: Qualcomm's documented APIs accept local model and HDF5 paths for
  independently submitted compile/inference/profile jobs. The client is
  explicitly set non-verbose and every SDK call is additionally captured so
  its automatic job URL output cannot reach task logs.

## Progress and restart instructions

The adapter, three scripts, mocked lifecycle/privacy tests, public worklog, and
downstream handoff are complete. No authenticated client method or external
job was invoked. T31 should use the handoff checklist for the first bounded
production-backend smoke after T22 completes; T72 can reuse the independent
commands in a manual workflow.
