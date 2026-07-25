# T02: AI Hub, Device Cloud, and GPU access report with toy jobs

Status: active — implementation complete, final review pending
Owner: Codex T02 agent
Updated: 2026-07-25

## Objective

Prove which public Qualcomm and free NVIDIA resources are usable now, capture
sanitized and reproducible evidence for current devices, quotas, and versions,
and exercise a bounded toy AI Hub Workbench compile, inference, and profile
lifecycle when authentication permits it.

## Scope

### In scope

- Inspect local and browser-authenticated access to Qualcomm AI Hub Workbench
  and Qualcomm Device Cloud without exposing credentials or account details.
- Record public service/package versions, target availability, quota or free
  allowance evidence, and access blockers with observation timestamps.
- Submit one minimal, free toy Workbench compile, inference, and profile
  lifecycle if the account and service permit it.
- Record current Colab and Kaggle free-GPU availability and document a
  paid-rental fallback without launching it.
- Commit only sanitized reports and small evidence under T02-owned paths.

### Out of scope

- Qwen export or compilation, which belongs to later graph and Qualcomm tasks.
- Running the Device Cloud Qwen/GenieX baseline, which belongs to T32.
- Paid GPU or cloud jobs, purchases, reservations, or external publication.
- Committing job URLs, account identifiers, credentials, raw service
  responses, or private quota/account pages.

## Dependencies and resources

- Required task dependencies: T00 is completed.
- Resource locks: `qai_hub_submission`, `device_cloud_x_elite`.
- External access: AI Hub Workbench, Qualcomm Device Cloud, Colab, and Kaggle.
- Cost boundary: free/public actions only; do not launch a paid fallback.

## Important paths

- Inputs: `docs/project/plan.md`, `ai/tasks/definitions/T02.yaml`,
  `ai/tasks/task_graph.yaml`.
- Outputs: `docs/results/access/`, `docs/failures/access/`, `results/hosts/`.
- Shared contracts: T00 version-capture policy in
  `docs/decisions/0001-model-and-version-pins.md`.

## Milestones

- [x] Record sanitized Workbench authentication, device, quota, and version
  evidence, preserving numeric quota as unknown because the client does not
  expose it.
- [x] Exercise and record one toy Workbench compile, inference, and profile
  lifecycle.
- [x] Confirm Device Cloud access or document the precise pending-access
  blocker and public target availability.
- [x] Record free NVIDIA options and a non-executed paid fallback.
- [ ] Run repository gates and fresh independent review, then complete the task
  graph entry and public worklog if every acceptance criterion passes.

## Verification and acceptance

- Commands:
  - `python3 -m unittest discover -v`
  - `python3 scripts/ai/render_task_status.py --check`
  - `python3 scripts/repo/check_hygiene.py --all`
  - `git status --short --ignored`
- Behavioral criteria:
  - Every access claim names its evidence and observation time.
  - The toy Workbench record separates service turnaround from device
    inference/profile latency.
  - Unavailable access is reported as a bounded blocker, never inferred.
  - The paid-GPU fallback is documentation only.
- Hardware/profile evidence: device name/type, public service or client
  version where exposed, compile/runtime options, job states, profile latency
  where available, and sanitized warnings.

## Artifact and privacy handling

- Committed evidence: sanitized Markdown/JSON records and small deterministic
  fixtures under T02-owned paths.
- External artifacts: any service-produced model/profile files remain ignored
  and are referenced only by checksums if downloaded.
- Private/local material: raw account pages, job URLs and IDs, tokens, quota
  identifiers, and unsanitized responses stay under `.ai-local/`.

## Decisions and discoveries

- 2026-07-24: The base environment has no importable `qai_hub` package and no
  Qualcomm credential variable exposed to the shell. Browser-authenticated
  access and safe local installation remain to be tested.
- 2026-07-24: The only available browser showed unauthenticated Workbench and
  Device Cloud states. Public target/version/free-resource evidence is
  committed without treating it as account or hardware proof.
- 2026-07-24: Colab account UI access is available, but no GPU runtime was
  allocated. Kaggle was not authenticated. The Runpod fallback remains an
  unexecuted, approval-gated command template.
- 2026-07-25: Exact `qai-hub==0.53.0` authenticated with a rotated local
  credential and exposed Snapdragon X Elite CRD plus QAIRT 2.45 (`default`),
  2.47, and 2.48 (`latest`).
- 2026-07-25: One ONNX Add graph compiled to a QAIRT 2.45 QNN context binary,
  ran numerically correct inference, and profiled successfully on Snapdragon X
  Elite CRD. The profile reported 127 microseconds estimated inference time,
  14,450,688 bytes estimated inference peak memory, and NPU placement.
- 2026-07-25: `qai-hub configure` can print its generated configuration,
  including the credential, and created `client.ini` with permissions broader
  than desired. All future credential configuration must capture and discard
  stdout/stderr and force both secret files to mode `600`.

## Progress and restart instructions

The single free Workbench lifecycle is complete and its raw state remains under
`.ai-local/profiles/T02/`. Public evidence contains no job IDs, URLs, account
identifiers, credential material, or raw service responses. The next
coordinator must run all focused/full tests, task-status and hygiene checks,
then obtain a fresh independent review. If it passes, set the worklog and task
to `completed`, archive this plan, and regenerate task status. T30 and T32
become ready when T02 completes; T60 still requires T13 and T20.
