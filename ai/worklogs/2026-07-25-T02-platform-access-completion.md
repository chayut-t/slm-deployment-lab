# T02: Platform Access Completion

Date: 2026-07-25
Task: `T02`
Visibility: `public`
Status: in_progress

## Outcome

Authenticated to Qualcomm AI Hub Workbench with exact `qai-hub==0.53.0` and
completed one bounded, free ONNX compile → inference → profile lifecycle on a
physical Snapdragon X Elite CRD. The toy compiled to a QNN context binary,
produced numerically correct output, and returned NPU profile evidence. Raw job
and account data remain local; the public record contains no job IDs, URLs,
credentials, or account identifiers.

## Changes

- Added a sanitized machine-readable lifecycle record with exact client,
  framework, device, graph, compile, numerical, profile, turnaround, cost, and
  privacy boundaries.
- Added a reader-facing result report separating service turnaround from
  measured graph latency and explicitly limiting the toy's benchmark meaning.
- Preserved the 2026-07-24 blocked evidence as historical truth and marked its
  Workbench authentication blocker resolved without claiming Device Cloud
  session access.
- Replaced the invalid `qai-hub --version` restart check with package metadata
  and documented that `qai-hub configure` output must always be captured and
  discarded because version 0.53.0 can print the credential.
- Added regression coverage for lifecycle completion, exact versions, numeric
  validation, profile units, NPU placement, unknown quota, remaining access
  boundaries, zero paid-resource use, credential safeguards, and privacy.

## Verification

- Pending the implementation verification and fresh independent review.

## Decisions and evidence

- Source model: ONNX opset 13, one Add node, float32 input `[1, 4]`, SHA-256
  `04992ab2a0ef479902430d6eb466927001c43d706e84aaf922aea9976559fca7`.
- Compile: Snapdragon X Elite CRD (Windows 11), QNN context binary, QAIRT
  `2.45.0.260326154327` default variant, HTP backend, Hexagon v73, SoC model
  60, optimization level 3.
- Inference: maximum absolute error `4.76837158203125e-7`; output passed
  `rtol=1e-5`, `atol=1e-6`.
- Profile: 127 microseconds estimated inference time across 100 samples,
  14,450,688 bytes estimated inference peak memory, and NPU placement for all
  reported entries.
- Observed submit-to-terminal turnaround was 96 seconds for compile, 298
  seconds for inference, and 364 seconds for profile. These include service
  queueing and provisioning and are not hardware latency.
- The authenticated framework query exposed QAIRT 2.45 as `default`, 2.47, and
  2.48 as `latest`. The client exposed no numeric quota; successful jobs prove
  submission access but not an unlimited allowance.
- Workbench cost was zero and no paid fallback was launched.

## Risks and limitations

- The graph is deliberately trivial access evidence and says nothing about
  Qwen3 compiler support, transformer throughput, or application latency.
- Device Cloud account minutes, live X Elite availability, and a session were
  not reverified. T32 owns the real Qwen/GenieX device path.
- No free NVIDIA runtime was allocated; T60 must perform its own provider and
  hardware preflight after T13 and T20.
- The raw private profile format is not a stable API contract; the public
  record includes its SHA-256 and only documented summary fields.

## Follow-up

- Newly unblocked tasks after final review and graph completion: T30 and T32.
- Recommended next action: run all gates, obtain fresh independent review,
  then archive the plan and complete T02.
