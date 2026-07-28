# T32: Device Cloud Qwen GenieX baseline and generation loop

Status: completed
Owner: Codex T32 agent
Updated: 2026-07-28

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
- [x] Run the live session after learner login and free-minute confirmation.
- [x] Validate multi-token output and timing-boundary labels.
- [x] Pass independent review and address all findings.
- [x] Research and record the learner-selected publication boundary.
- [x] Complete the learner timing and claim-boundary debrief.

## Verification and acceptance

- Commands: focused Qualcomm tests, task-status check, repository hygiene.
- Numerical or behavioral criteria: valid multi-token output; load,
  tokenization, prefill, decode, and total timing are distinguished.
- Hardware/profile evidence: exact Snapdragon X Elite and runtime identity,
  or an explicit non-completion blocker if live access requires the learner.

## Artifact and privacy handling

- Tracked completion evidence: capture tooling, commands, bounded result page,
  completed plan, sanitized handoff, and public worklog.
- Private completion evidence: the sanitized environment/timing manifest under
  ignored `.ai-local/` storage.
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
  publishes only the learner-approved generic setup and aggregate latency in
  its owned result page; the normalized record and evidence digests remain
  private.
- 2026-07-28: The learner authenticated and created a free Snapdragon X Elite
  interactive session. A native probe recorded the model, prompt,
  multi-token output, device/runtime evidence, placement, and all eight timing
  boundaries in ignored private storage.
- 2026-07-28: Complete request uses the observed host wall from artifact
  open/map through generation completion, including tokenizer subprocess and
  probe overhead. It is not Device Cloud allocation time, SSH transport, or a
  Workbench graph measurement.
- 2026-07-28: The session UI labels allocated-device information confidential.
  Public Qualcomm terms and Device Cloud documentation provide no express
  permission to publish the complete live-device record.
- 2026-07-28: After reviewing that research, the learner directed publication
  of the generic reproducibility setup and aggregate latency measurements
  while withholding allocated-device evidence, observed placement proof,
  exact installed versions, session/account identifiers, logs, manifests, and
  evidence digests. The result page records that this is the learner's
  publication decision rather than a legal determination or a general
  publication license.
- 2026-07-28: Three successive fresh independent reviews found and then
  verified fixes for request-wall semantics, same-run placement evidence,
  evidence freezing, runtime-version evidence, immutable model provenance,
  output validation, timing-source constraints, and the public confidentiality
  hold. The final review approved with no findings.
- 2026-07-28: A fresh post-publication review verified every published setup
  field and latency value against the frozen evidence. It found two stale
  instructions to publish a normalized manifest or digest; both were corrected
  to keep manifests and evidence digests private, and re-review approved with
  no remaining findings.

## Progress and restart instructions

T32 is complete. The live capture, sanitizer acceptance pass, publication
decision, generic-setup/aggregate-latency publication, independent review, and
learner debrief all passed. Raw transcripts, hardware identity, and the
private/sanitized manifests remain under
`.ai-local/profiles/T32/qdc-2026-07-28/`. T33 may consume this result after its
other dependency, T31, completes.
