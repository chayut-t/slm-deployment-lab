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

- [ ] Audit T02 access evidence and current Device Cloud prerequisites.
- [ ] Implement sanitized manifest/timing capture and reproducible commands.
- [ ] Run or prepare the live Snapdragon X Elite GenieX generation session.
- [ ] Validate multi-token output and timing-boundary labels.
- [ ] Pass independent review and address all findings.

## Verification and acceptance

- Commands: focused Qualcomm tests, task-status check, repository hygiene.
- Numerical or behavioral criteria: valid multi-token output; load,
  tokenization, prefill, decode, and total timing are distinguished.
- Hardware/profile evidence: exact Snapdragon X Elite and runtime identity,
  or an explicit non-completion blocker if live access requires the learner.

## Artifact and privacy handling

- Committed evidence: sanitized environment/timing manifest, commands,
  documentation, completed plan, and worklog.
- External artifacts: ready-made model/runtime assets remain external.
- Private/local material: account/session identifiers and raw service output
  under `.ai-local/`.

## Decisions and discoveries

- 2026-07-27: The task may pause only at the documented learner-authentication
  or live-device boundary; no paid resource is authorized.

## Progress and restart instructions

Start from T02 evidence, make all capture tooling testable without live access,
then attempt the live free Device Cloud session. Record any access boundary
precisely without fabricating hardware results.
