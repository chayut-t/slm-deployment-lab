# T31 handoff: how to run the first submission

Date: 2026-08-03
From: the T22/T41 integration at `f856386`
To: T31

Three specifics for how to run T31. They come out of integrating
`task/T22-qnn-candidates` and `task/T41-w8-quantization-evidence` into `main`,
and each one is anchored to something in the committed evidence rather than to
a preference. `ai/handoffs/T22-qnn-candidates.md` remains the source for what
T31 consumes; this file only covers how to sequence the first attempt.

## 1. Fix the `claim_boundary` defect before building any new candidate

`src/slm_lab/graph/qnn/build.py:122` puts
`onnx_checker_accepted_the_candidate_graph` in the `establishes` tuple
unconditionally. `check_candidate` at `src/slm_lab/graph/qnn/build.py:384`
does not raise when the checker rejects a graph — it returns
`{"status": "failed", ...}` and lets the caller continue.

All eight manifests committed under `results/manifests/qnn/` read `passed`, so
no committed claim is false today. The exposure is prospective: T31 is the
first task that builds candidates the checker has not already accepted, and a
failing check would still stamp a manifest asserting the checker accepted it.

Fix it first. It is small, and T31 is exactly the task that would trip it.
This is the same defect class the T22 review caught in round 1 — a stamped
claim that the surrounding measurement does not support.

## 2. Smoke-test with the S128 prefill candidate

Use `results/manifests/qnn/S128.json`, prefill, as the first thing submitted.

From `results/manifests/qnn/inspection/S128.json`, it is the only graph in the
matrix that comes out of the rewrite clean:

| Graph | `R-DATA-DEPENDENT-SHAPE-INPUT` | `R-INTERNAL-DYNAMIC-SHAPE` |
|---|---|---|
| S128 prefill | 804 → 0 | none |
| S4096 prefill | 804 → 6 | 0 → 9 |
| decode (all variants) | 1,231 → 423 | 0 → 1,069 |

Note the decode row. The shape fold converts one high-severity risk class into
another: decode's rank-1 finding falls, but 1,069 interior tensors whose shape
ONNX shape inference cannot resolve appear in its place, so by raw
high-severity count decode comes out worse than the reference. That is
disclosed in section 6.1 of `docs/results/qualcomm/qnn-candidates.md` and in
the T22 handoff, and it is the reason not to lead with decode.

S128 prefill is also the smallest artifact in the matrix (949,559 bytes of
protobuf). If it fails, the failure is about the pipeline, not about the graph.

## 3. Expect external-data packaging to break first

The compile request names only the `.onnx` file, because the committed T30
adapter requires `source_artifact.path` to be a single existing file. Whether
the service reads the `.onnx.data` sidecar from the same directory, or wants a
directory or an archive, is untested against AI Hub.

Every candidate in the matrix carries a sidecar of roughly 1.19 GB, so this is
not an edge case — it is on the path of the first submission. Treat a failure
here as a packaging result, not a graph result, and record it as such.

## Why T31 before T41

T41 merged `blocked`, and the T22 merge did not unblock it.
`src/slm_lab/deployment/qualcomm/ai_hub.py` exposes `submit_compile_job`,
`submit_inference_job` and `submit_profile_job` and nothing else; there is no
quantize stage, and `/Volumes/T9/slm-deployment-lab/onnx/quantized/` is empty.

T41's own capability record
(`results/quantization/t41-ai-hub-capability-2026-08-03.json`) establishes that
the service accepts INT4/INT8/INT16, so unblocking T41 is an adapter extension
rather than a hardware purchase. The reason to still do T31 first is
attribution: quantizing before anything has compiled means a failure cannot be
separated into "W8 did this" and "the graph did this."

T22's parity measurement is what makes that separation possible. The candidates
are bit-identical to the reference on the ONNX Runtime CPU provider, so any
numerical divergence T31 observes on a device belongs to the compiler, the
runtime or the hardware. Nothing in the Qualcomm lane has yet established
`compiler_acceptance` — every manifest's `claim_boundary` lists it under
`does_not_establish` — and T33 and all profiling work sit behind it.

## Budget state at handoff

The user authorized AI Hub compile/profile jobs and up to 120 minutes of Device
Cloud, free capacity only, with instructions to stop and report rather than
incur a charge. T41 spent none of it: it ran one read-only capability query
(`qai-hub==0.53.0`, 2026-08-03) recording 0 jobs, 0 device minutes, US$0.00.
The full 120 minutes is intact. Compile is the free tier; the metered pool is
what on-device profiling draws from.
