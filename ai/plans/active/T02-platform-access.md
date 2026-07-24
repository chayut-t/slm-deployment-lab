# T02: AI Hub, Device Cloud, and GPU access report with toy jobs

Status: active — blocked on Qualcomm authentication
Owner: Codex T02 agent
Updated: 2026-07-24

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

- [ ] Record sanitized Workbench authentication, device, quota, and version
  evidence. Public versions and authentication boundary are recorded; account
  device/quota evidence remains blocked.
- [ ] Exercise and record one toy Workbench compile, inference, and profile
  lifecycle.
- [x] Confirm Device Cloud access or document the precise pending-access
  blocker and public target availability.
- [x] Record free NVIDIA options and a non-executed paid fallback.
- [ ] Run repository gates and, only if every T02 acceptance criterion passes,
  complete the task graph entry and public worklog.

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

## Progress and restart instructions

Read this plan and the public access/failure reports. Have the user sign in to
Qualcomm in the retained Workbench and Device Cloud Chrome tabs, then configure
an API token locally without placing it in chat or the repository. Verify
`qai-hub list-devices` and `qai-hub list-frameworks`, keeping raw output under
`.ai-local/`. Identify a minimal supported toy model and run exactly one free
compile, inference, and profile lifecycle. Sanitize all durable evidence before
placing it under the owned public paths, rerun every repository gate, then move
this plan to `ai/plans/completed/` only if all T02 acceptance criteria pass.
