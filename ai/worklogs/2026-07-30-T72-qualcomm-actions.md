# T72: Qualcomm Actions

Date: 2026-07-30
Task: `T72`
Visibility: `public`
Status: in_progress

## Outcome

Initial implementation produced a manually dispatched Qualcomm AI Hub workflow
and offline validation. Independent review then identified incomplete producer
identity and bundle semantic/path/digest validation, so T72 is reopened while
those findings are addressed and freshly reviewed. No secret was created or
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
- The benchmark result step uploads only the sanitized stage manifest with
  bounded retention. Its request bundle is a separate two-day producer
  artifact; result tensors, raw profiles, SDK output, credentials, and service
  identity are never included in the result upload.
- Added a learning guide covering Actions concepts, the independent-stage
  contract, input meaning, immutable request-bundle layout/provenance,
  protected-environment setup, secret hygiene, approval, dispatch, evidence,
  incident handling, and the unverified external boundary.
- Added offline workflow tests for parsing, triggers, choices, fork/default-ref
  gating, provenance, permissions, secret placement/cleanup, pinned
  dependencies, local-script delegation, private-artifact exclusion, bounds,
  embedded Python syntax, and guide coverage.
- Independent-review fixes add the fixed
  `.github/workflows/qualcomm-request-bundle.yml` producer. It has no Qualcomm
  secret, reads one same-repository pre-staged release ZIP by conservative
  tag/name, verifies its learner-reviewed SHA-256, safely extracts bounded ZIP
  contents without `extractall`, checks the tuple, builds a complete digest
  inventory, and uploads a two-day immutable request bundle.
- The benchmark now binds the producer workflow path, exact current commit,
  run ID, event/repository/default branch, and independently reviewed
  `bundle-manifest.json` SHA-256. Before secret configuration it rejects
  missing/extra/modified/symlinked files, tuple mismatches, wrong exact device
  selectors, context mismatches, predecessor/compiled-artifact drift, and
  input/output paths outside distinct runner-private roots.
- Added executable adversarial regressions for successful wrong-producer runs,
  wrong producer revisions, wrong source/manifest digests, mislabeled request
  tuples, source ZIP path escapes, and request path escapes.

## Verification

- Command:
  `PYTHONPATH=src python3 -m unittest discover -s tests/workflows -p 'test_*.py' -v`
- Result: 22 focused producer/benchmark workflow tests passed.
- Command:
  `PYTHONPATH=src <project-python> -m unittest tests.deployment.qualcomm.test_ai_hub -v`
- Result: all 21 T30 adapter contract tests passed.
- Command: `<project-python> -m ruff check tests/workflows`
- Result: passed.
- Command: `PYTHONPATH=src <project-python> -m pytest -q`
- Result: with an isolated writable uv cache, 182 tests passed and six
  intentional opt-in tests skipped. One T03 repository-automation test failed:
  `test_staged_graph_requires_matching_staged_status` assumes at least one
  dependency-ready task remains `planned`, while the public parallel-work
  claim has every currently ready task marked `in_progress` or `completed`.
  T72 code was not in the traceback.
- Command:
  `UV_CACHE_DIR=<writable-temp> PYTHONPATH=src <project-python> -m pytest -q -k 'not test_staged_graph_requires_matching_staged_status'`
- Result: 182 tests passed, six intentional opt-in tests skipped, and the one
  coordination-sensitive T03 test was explicitly deselected.
- Commands: `python3 scripts/ai/render_task_status.py --check`,
  `python3 scripts/repo/check_hygiene.py --all`, and `git diff --check`.
- Result: all passed on the current rereview snapshot.

## Decisions and evidence

- One workflow dispatch runs one stage. Inference/profile reuse a downloaded
  compiled artifact and sanitized compile manifest by hash, never an in-memory
  SDK job or public service ID.
- Workload choices select a private request artifact; they do not prove the
  observed device, compiler support, achieved precision, placement, numerical
  quality, or latency. Those claims require the sanitized stage result.
- The request bundle is accepted only from the fixed reviewed producer
  workflow at the benchmark dispatch's exact commit. The producer terminates
  trust at a same-repository release asset plus a separately reviewed source
  digest; it does not create a release or upload from a learner's machine.
  Protected-environment review remains the final human authorization before
  any quota-consuming submission.
- GitHub-hosted runner time and service turnaround are orchestration evidence,
  not Qualcomm device latency.

## Risks and limitations

- GitHub does not provide local execution evidence for its environment
  protections. The learner must create and review the protected environment
  and secret before the first dispatch.
- A learner must separately stage the source ZIP as a same-repository release
  asset. T72 does not create that external state, and repository visibility
  and artifact licensing must be reviewed before staging.
- The pinned client and action commits are immutable, but a future deliberate
  upgrade must revalidate authentication configuration, request contracts,
  and official-action behavior.
- No GitHub environment, secret, request bundle, workflow run, Qualcomm job,
  quota/cost observation, or hardware/profile result was created or verified.

## Follow-up

- Independent review status: P1 producer identity/revision and bundle
  validation findings have implementation and adversarial regression fixes.
  The coordinator authorized and the T72 definition records the narrow
  producer-workflow scope expansion. This worklog remains unreferenced by the
  task graph until a fresh reviewer accepts the fixes.
- Newly unblocked tasks: none while T72 remains in progress.
- Recommended next action: perform a fresh independent rereview of the
  producer trust terminus, benchmark pre-secret validator, adversarial tests,
  and lifecycle metadata. Only after acceptance should T72 be completed and
  the learner proceed to external setup.
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
