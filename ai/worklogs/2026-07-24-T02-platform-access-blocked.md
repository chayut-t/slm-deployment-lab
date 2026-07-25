# T02: Platform Access Blocked

Date: 2026-07-24
Task: `T02`
Visibility: `public`
Status: blocked

## Outcome

Recorded a sanitized, machine-readable snapshot of the public Workbench,
Device Cloud, Colab, Kaggle, and paid-fallback boundaries. Workbench and Device
Cloud were reachable but unauthenticated, so no account-visible quota or toy
compile/inference/profile job could be proven. T02 remains incomplete and is
marked blocked rather than presenting public service documentation as account
or hardware access.

## Changes

- Added a reader-facing public-access report separating service, account, and
  hardware evidence.
- Added a reusable Qualcomm authentication failure record with a privacy-safe
  reproducer and exact unblock sequence.
- Added JSON evidence for public versions, target catalogs, account-state
  boundaries, free NVIDIA options, and the unexecuted paid fallback.
- Added regression tests that prevent an unauthenticated state from looking
  like a completed Workbench lifecycle, preserve unknown values as null, scan
  the evidence for common private markers, and require the paid command to
  remain unexecuted.
- Corrected the reviewed restart path to create an ignored Python 3.11
  environment, install exact `qai-hub==0.53.0`, and verify the client version
  before any local token configuration.
- Kept the execution plan active with restart instructions and marked T02
  blocked with no completion worklog reference.

## Verification

- Command: `python3 -m unittest tests.repo.test_access_evidence -v`
- Result: 6 tests passed.
- Command: `python3 scripts/ai/render_task_status.py --check`
- Result: passed after regenerating task status.
- Command: `python3 scripts/repo/check_hygiene.py --all`
- Result: passed for tracked and untracked public files.
- Command: `git diff --check`
- Result: passed.
- Command: `python3 -m unittest discover -s tests -p 'test_*.py' -v`
- Result: 22 passed, 1 skipped, 1 failed. The failure is the pre-existing
  T03-owned
  `test_staged_graph_requires_matching_staged_status`: its temporary T00
  mutation invalidates the coordinator's `in_progress` T01 and T03 states
  before reaching the assertion under test. T02 is `blocked` and did not
  change that shared automation test.

## Decisions and evidence

- Public `qai-hub` `0.53.0`, hosted QAIRT `2.45.0`/`2.46.0`/`2.47.0`,
  ONNX Runtime `1.26.0`, QNN EP `2.2.0`, and Quantize Job AIMET `2.34` were
  recorded as public discovery, not tested account/job versions.
- The public Qwen catalog lists IQ-9075 EVK, Snapdragon 8 Elite QRD, and
  Snapdragon X Elite CRD. The Device Cloud partial public catalog showed X
  Elite CRD8380X and 8 Elite QRD8750 with free-minute labels.
- Colab account UI access was observed with identity omitted, but no GPU was
  allocated. Kaggle was not authenticated; its published P100/weekly-quota
  policy is not treated as account quota.
- Runpod was selected only as a documented paid fallback. Its creation command
  was not run and remains subject to price capture and explicit approval.

## Risks and limitations

- The required toy Workbench compile → inference → profile lifecycle was not
  exercised. There are no job IDs, artifacts, latency, memory, or NPU-placement
  results.
- Workbench account access, quota, device scheduling, default QAIRT tag, and
  Device Cloud minute/session access remain unproven.
- A free NVIDIA GPU allocation and CUDA provider placement remain unproven.
- T30, T32, and T60 must remain blocked on T02.

## Follow-up

- Newly unblocked tasks: none.
- Recommended next action: sign in to Qualcomm in Chrome, configure the
  Workbench API token locally, run the exact toy lifecycle, capture account
  quota/device/framework evidence privately, publish only a sanitized summary,
  and rerun all gates before completing T02.
