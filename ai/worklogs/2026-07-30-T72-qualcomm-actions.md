# T72: Qualcomm Actions

Date: 2026-07-30
Task: `T72`
Visibility: `public`
Status: completed

## Outcome

Implemented a manually dispatched, fork-safe Qualcomm AI Hub workflow that
selects one target/context/precision/stage tuple and delegates remote execution
to the existing T30 local adapter for that stage. The workflow is structurally
validated offline and teaches the full learner-controlled secret, environment,
request-bundle, dispatch, and evidence lifecycle. No secret was created or
inspected and no GitHub Actions or Qualcomm job was run.

## Changes

- Added typed choices for Snapdragon X Elite, Dragonwing IQ-9075, and
  Snapdragon 8 Elite; 128/512/1,024/4,096 contexts; FP16/W8A16/W8A8/W4A8
  precision labels; and independently restartable compile/inference/profile
  stages.
- Added a no-AI-Hub-secret authorization job. It permits only manual upstream
  default-branch dispatches and validates fixed choices plus the successful
  same-repository/default-branch provenance of the private request-bundle run.
- Added a protected `qualcomm-ai-hub` environment boundary, minimal read-only
  repository/Actions permissions, bounded timeouts, non-cancelling concurrency,
  immutable official-action pins, Python 3.11.13, and exact
  `qai-hub==0.53.0` verification.
- Configured the AI Hub client through an ephemeral mode-`600` file without
  calling the credential-printing client configuration command. Secret scope
  is limited to that step, and an always-run cleanup removes the file.
- Kept T30's three scripts as the sole stage implementations. A fixed shell
  allowlist maps the stage to a script; no YAML submission, polling,
  normalization, or redaction logic duplicates the adapter.
- Uploaded only the sanitized stage manifest with bounded retention. Requests,
  models, datasets, compiled artifacts, result tensors, raw profiles, SDK
  output, credentials, and service identity remain private.
- Added a learning guide covering Actions concepts, the independent-stage
  contract, input meaning, immutable request-bundle layout/provenance,
  protected-environment setup, secret hygiene, approval, dispatch, evidence,
  incident handling, and the unverified external boundary.
- Added offline workflow tests for parsing, triggers, choices, fork/default-ref
  gating, provenance, permissions, secret placement/cleanup, pinned
  dependencies, local-script delegation, private-artifact exclusion, bounds,
  embedded Python syntax, and guide coverage.

## Verification

- Command:
  `PYTHONPATH=src python3 -m unittest discover -s tests/workflows -p 'test_*.py' -v`
- Result: 10 focused workflow tests passed.
- Command:
  `PYTHONPATH=src <project-python> -m unittest tests.deployment.qualcomm.test_ai_hub -v`
- Result: all 21 T30 adapter contract tests passed.
- Command: `<project-python> -m ruff check tests/workflows`
- Result: passed.
- Command: `PYTHONPATH=src <project-python> -m pytest -q`
- Result: with an isolated writable uv cache, 170 tests passed and six
  intentional opt-in tests skipped. One T03 repository-automation test failed:
  `test_staged_graph_requires_matching_staged_status` assumes at least one
  dependency-ready task remains `planned`, while the public parallel-work
  claim has every currently ready task marked `in_progress` or `completed`.
  T72 code was not in the traceback.
- Command:
  `UV_CACHE_DIR=<writable-temp> PYTHONPATH=src <project-python> -m pytest -q -k 'not test_staged_graph_requires_matching_staged_status'`
- Result: 170 tests passed, six intentional opt-in tests skipped, and the one
  coordination-sensitive T03 test was explicitly deselected.
- Commands: `python3 scripts/ai/render_task_status.py --check`,
  `python3 scripts/repo/check_hygiene.py --all`, and `git diff --check`.
- Result: all passed on the final completion snapshot.

## Decisions and evidence

- One workflow dispatch runs one stage. Inference/profile reuse a downloaded
  compiled artifact and sanitized compile manifest by hash, never an in-memory
  SDK job or public service ID.
- Workload choices select a private request artifact; they do not prove the
  observed device, compiler support, achieved precision, placement, numerical
  quality, or latency. Those claims require the sanitized stage result.
- The request bundle is accepted only from a successful same-repository
  default-branch push/manual run. Protected-environment review remains the
  final human authorization before any quota-consuming submission.
- GitHub-hosted runner time and service turnaround are orchestration evidence,
  not Qualcomm device latency.

## Risks and limitations

- GitHub does not provide local execution evidence for its environment
  protections. The learner must create and review the protected environment
  and secret before the first dispatch.
- A reviewed default-branch producer workflow must create the private
  `qualcomm-request-bundle`; that producer depends on the export/artifact lane
  and is outside T72.
- The pinned client and action commits are immutable, but a future deliberate
  upgrade must revalidate authentication configuration, request contracts,
  and official-action behavior.
- No GitHub environment, secret, request bundle, workflow run, Qualcomm job,
  quota/cost observation, or hardware/profile result was created or verified.

## Follow-up

- Newly unblocked tasks: T72 satisfies its dependency contribution to T80 and
  T82; their other dependencies remain authoritative.
- Recommended next action: the learner should complete the guide's security
  checklist, create the protected environment and secret, inspect a trusted
  default-branch request-bundle producer run, then approve one bounded free
  compile dispatch before inference or profile.
- Learner debrief:
  - [ ] Trace a dispatch through authorization, environment approval, client
    setup, the local stage script, credential cleanup, and manifest upload in
    [`qualcomm-benchmark.yml`](../../.github/workflows/qualcomm-benchmark.yml).
  - [ ] Explain why a successful workflow run is not a device-latency result.
  - [ ] Review every item under “Learner setup: stop before adding the secret”
    in the [Actions guide](../../docs/learning/github_actions_for_ai_hub.md).
  - [ ] Inspect the selected request and predecessor hashes before approving
    the first real run.
  - [ ] Do not mark the T72 learning checkpoint complete until the learner has
    personally reviewed the security model.
