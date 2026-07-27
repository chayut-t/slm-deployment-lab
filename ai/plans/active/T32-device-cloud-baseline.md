# T32: Device Cloud Qwen GenieX baseline and generation loop

Status: active
Owner: Codex T32 agent
Updated: 2026-07-27

## Objective

Produce a reproducible Qwen3-0.6B GenieX/`llama.cpp` baseline and persistent
device-side generation-loop record for a Device Cloud Snapdragon X Elite,
with loading, tokenization, prefill, decode, and total generation timing kept
distinct.

## Scope

### In scope

- A sanitized Device Cloud environment manifest and reproducible commands.
- Ready-made Qwen baseline output with valid multi-token generation.
- Explicit end-to-end timing boundaries and provenance labels.
- Reusable local tooling and documentation for recording a live session.
- A bounded blocker record if authentication or device scheduling prevents
  real-device completion.

### Out of scope

- Claiming the GenieX route proves the custom static QNN path.
- Mislabeling hosted single-graph timing as persistent end-to-end generation.
- Spending money or activating paid services.
- Publishing private job, account, device-session, or access identifiers.

## Dependencies and resources

- Required task dependencies: T02, completed.
- Resource locks: `device_cloud_x_elite`.
- External access: Qualcomm Device Cloud; learner interaction may be needed
  for authentication, free-minute activation, or scheduling.
- Cost boundary: free resources only; paid work requires separate approval.

## Important paths

- Inputs: T02 access evidence and Qualcomm environment helpers.
- Outputs: `src/slm_lab/deployment/qualcomm/device_cloud.py`,
  `scripts/qualcomm/device_cloud/`,
  `docs/results/qualcomm/device-cloud.md`.
- Shared contracts: privacy/sanitization policy and benchmark timing terms.

## Milestones

- [x] Audit T02 access evidence and current Device Cloud prerequisites.
- [x] Implement sanitized manifest/timing capture and reproducible commands.
- [x] Prepare the live Snapdragon X Elite GenieX generation session.
- [ ] Run the live session after learner login and free-minute confirmation.
- [ ] Validate multi-token output and timing-boundary labels.
- [ ] Pass independent review and address all findings.

## Verification and acceptance

- Commands: focused Qualcomm tests, task-status check, repository hygiene.
- Numerical or behavioral criteria: valid multi-token output; load,
  tokenization, prefill, decode, and total timing are distinguished.
- Hardware/profile evidence: exact Snapdragon X Elite and runtime identity,
  or an explicit non-completion blocker if live access requires the learner.

## Artifact and privacy handling

- Current committed evidence: capture tooling, commands, bounded result page,
  active plan, and sanitized handoff.
- Completion evidence: sanitized environment/timing manifest, completed plan,
  and public worklog, only after the live acceptance criteria pass.
- External artifacts: ready-made model/runtime assets remain external.
- Private/local material: account/session identifiers and raw service output
  under `.ai-local/`.

## Decisions and discoveries

- 2026-07-27: The task may pause only at the documented learner-authentication
  or live-device boundary; no paid resource is authorized.
- 2026-07-27: Public catalog discovery confirmed a Snapdragon X Elite Compute
  Reference Design (`CRD8380X`, Windows) and an `Unlock Free Minutes` label,
  but the browser session is logged out. This does not establish account
  minutes, allocated hardware, or availability.
- 2026-07-27: The pinned v0.58.0 Qwen3-0.6B model card names the ready asset
  `geniex_llamacpp` at `Q4_0`; this baseline is intentionally distinct from a
  custom `qairt`/QNN bundle.
- 2026-07-27: The coordinating agent approved one narrow ownership expansion
  for `tests/deployment/qualcomm/test_device_cloud.py`; no existing T30 test
  is modified.
- 2026-07-27: Independent review required structured evidence enums with
  digests/private references, affirmative NPU/HTP placement, strict RFC3339
  time, boolean-safe numeric validation, a normalized prompt digest, native
  failure checks, and unique no-clobber transcripts.
- 2026-07-27: Raw and sanitized session results remain under
  `.ai-local/profiles/T32/`; T31 owns `results/processed/qualcomm/`. T32
  publishes the reviewed normalized record or digest/link only in its owned
  result page.

## Progress and restart instructions

The offline sanitizer, PowerShell workflow, fixed prompt, private template,
tests, and bounded result page are ready. Resume by having the learner sign in
to the Device Cloud tab, confirm free minutes, and start the X Elite CRD
session. Follow `scripts/qualcomm/device_cloud/README.md`; keep raw logs under
`.ai-local/profiles/T32/`. Do not complete the task unless the allocated-device
identity, exact GenieX/runtime route, NPU/HTP evidence, valid multi-token
output, and every timing boundary pass the sanitizer and learner review.
