# Workbench toy lifecycle — 2026-07-25

Task: `T02`
Observed: 2026-07-25 12:10–12:24 UTC
Status: completed

## Outcome

An authenticated Qualcomm AI Hub Workbench account completed one minimal
compile → inference → profile lifecycle on a physical Snapdragon X Elite CRD.
This proves that the account can query the hosted catalog, submit jobs, obtain
a QNN context binary, run inference, and retrieve a hardware profile. It does
not prove Device Cloud session access, numeric account quota, or Qwen3 graph
support.

No paid resource was launched. Workbench reported no charge for this lifecycle.
Job identifiers, URLs, account details, the API token, and raw service
responses remain under `.ai-local/`.

## Reproducible toy contract

| Field | Value |
|---|---|
| Source format | ONNX, opset 13 |
| Graph | `y = x + constant_bias` |
| Input | `x`, float32, shape `[1, 4]` |
| Source SHA-256 | `04992ab2a0ef479902430d6eb466927001c43d706e84aaf922aea9976559fca7` |
| Client | `qai-hub==0.53.0`, Python `3.11.13` |
| Local ONNX package | `1.13.1`, selected by the `qai-hub[onnx]` extra |
| Device | Snapdragon X Elite CRD, Windows 11 |
| Compile option | `--target_runtime qnn_context_binary` |
| Resolved target | QNN context binary, HTP backend, Hexagon v73, SoC model 60 |
| Resolved QAIRT | `2.45.0.260326154327`, `default` variant |

The authenticated framework query also returned QAIRT
`2.47.0.260601114230` and `2.48.0.260626120635`; `2.48` carried the `latest`
tag at observation time. The client did not expose a numeric quota, so the
report preserves it as unknown. Successful jobs prove submission access, not
an unlimited allowance.

## Lifecycle result

| Stage | State | Observed service turnaround |
|---|---|---:|
| Compile | success | 96 s |
| Inference | success | 298 s |
| Profile | success | 364 s |

These turnaround values run from local submission to the first observed
terminal status and include service queueing and physical-device provisioning.
They are not model latency.

The inference input was `[1, 2, 3, 4]`; the expected output was
`[1.5, 1, 5, 3.75]`. The device output matched with maximum absolute error
`4.76837158203125e-7` using `rtol=1e-5` and `atol=1e-6`.

## Physical-device profile

| Metric | Value |
|---|---:|
| Estimated inference time | 127 µs (0.127 ms) |
| Profile iterations | 100 |
| Observed iteration range | 127–1,087 µs |
| Estimated inference peak memory | 14,450,688 bytes |
| First load | 311,671 µs; 14,786,560 bytes peak |
| Warm load | 198,597 µs; 14,573,568 bytes peak |
| Reported compute units | NPU for input, graph node, and output |

Workbench reports time in microseconds and memory in bytes. The estimated
inference time is its minimum observed microbenchmark time; it is distinct
from end-to-end application latency and job turnaround.

The graph is intentionally tiny and is access evidence, not a meaningful
performance benchmark. These numbers must not be compared with Qwen3 or used
to infer transformer throughput.

## Remaining bounded access boundaries

- Device Cloud account minutes, live X Elite availability, and session launch
  were not reverified. T32 owns the actual GenieX/device-side execution path.
- No Colab or Kaggle GPU runtime was allocated. Their public free-tier
  availability remains recorded without an account-quota claim.
- The Runpod paid fallback remains documentation only and was not launched.

## Evidence

- Machine-readable sanitized record:
  [`results/hosts/workbench-toy-lifecycle-2026-07-25.json`](../../../results/hosts/workbench-toy-lifecycle-2026-07-25.json)
- Historical pre-authentication observation:
  [2026-07-24 public platform access](2026-07-24-public-platform-access.md)
- Qualcomm documentation for profile units and semantics:
  [Working with Jobs](https://workbench.aihub.qualcomm.com/docs/hub/jobs.html)
  and
  [How it works](https://workbench.aihub.qualcomm.com/docs/hub/howitworks.html)
