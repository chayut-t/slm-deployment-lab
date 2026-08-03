# T41 handoff: the W8 submission boundary

Date: 2026-08-03
From: T41
To: T42, T43, and whichever session first receives submission authorization

T41 built everything up to, and stopped exactly at, the Qualcomm AI Hub
submission boundary for the two W8 candidates. It quantized no weight, produced
no measurement, submitted no job, and wrote no request file. This handoff says
what exists, what to run in what order, what the standing authorization does
and does not cover, and what would invalidate the frozen candidates.

Read `docs/results/quantization/w8.md` first if you want the argument. This
file is the operational half.

**Read "The standing authorization" and "What the service exposes" below before
anything else.** Submission authorization already exists — you do not need to
ask again for what it covers — and the reason you still cannot satisfy the
acceptance criterion is not permission. It is that no W8 artifact exists, and
the one repository component that could request a hosted one belongs to another
task.

## What exists at this commit

- `configs/quantization/w8/w8a16.yaml` and `w8a8.yaml` — two frozen candidate
  specifications, byte-identical fixed points of their generator, bound by hash
  to the T40 calibration revision, the four committed float16 baseline
  manifests, the frozen T13 protocol digest, and the T21 graph inventory.
- `src/slm_lab/quantization/w8.py` — generation, validation, the analytic
  weight-storage projection, the precision-state function, the quality-delta
  comparison, and the stage-request emitter.
- `results/quantization/t41-w8-readiness-2026-08-03.json` — the readiness
  record. Both candidates read `precision_state: specified`; the plan 7.3
  ledger reads 1 `satisfied`, 3 `not_run`, 6 `blocked`.
- `tests/quantization/test_w8_candidates.py` and `test_w8_evidence.py` — 132
  offline tests, including mutation tests that prove the refusals rather than
  asserting them.

- `results/quantization/t41-ai-hub-capability-2026-08-03.json` — what the live
  service exposed on 2026-08-03, from one read-only query. No job, no device
  minutes, no cost.

There is no quantized artifact, no encodings file, no simulation record, no
stage manifest, and no quality result anywhere in this repository.

## The standing authorization

Granted 2026-08-03. Record it in any external-job entry you write, and check
your job against it before you submit.

| Term | Value |
|---|---|
| Scope | Hosted AI Hub compile, profile, and inference jobs for T41 |
| Device Cloud | Interactive minutes permitted, capped at **120** |
| Capacity | **Free capacity only** |
| Spend | **None.** Any cost needs a fresh decision from the user |
| Still withheld | Pushing, and any public GitHub state |
| Consumed by T41 | 0 jobs, 0 of 120 minutes, US$0.00 |

The 120-minute ceiling is **fully unspent** and you inherit all of it. What
needs a fresh decision, not this grant: a paid job, a device outside free
capacity, the 121st Device Cloud minute, a push, or anything public on GitHub.

What T41 did with the authorization: exactly one read-only capability query,
below. Nothing else.

## What the service exposes

Observed on 2026-08-03 with client `qai-hub==0.53.0`, by a read-only query that
called exactly one service function, `get_devices`, and submitted no job,
uploaded no model, and leased no device. The entry-point names came from
`dir()` and the signature from `inspect.getattr_static`, so no `submit_*`
function was called or even fetched.

The live query is `uv run python -m slm_lab.quantization.w8 capabilities`. It
needs an authenticated `qai-hub` client, which this repository does not pin, so
it does not run on the primary host — it ran elsewhere and the sanitized result
was carried in. Replay the committed record offline with:

```bash
uv run python -m slm_lab.quantization.w8 capabilities \
  --offline-input results/quantization/t41-ai-hub-capability-2026-08-03.json
```

**There is a quantize entry point, and both candidates fit it.** The client
exposes:

```text
submit_quantize_job(model, calibration_data, weights_dtype, activations_dtype,
                    name, options, project)
```

and `QuantizeDtype` carries `INT4`, `INT8`, and `INT16`. So:

| Candidate | `weights_dtype` | `activations_dtype` |
|---|---|---|
| `w8a16` | `QuantizeDtype.INT8` | `QuantizeDtype.INT16` |
| `w8a8` | `QuantizeDtype.INT8` | `QuantizeDtype.INT8` |

`INT4` exists as well, which is T42's business, not T41's.

**This makes the Lane A gap an adapter gap, not a platform limitation.** The
service takes the request; this repository has nothing that sends it.
`slm_lab.deployment.qualcomm.ai_hub` declares
`STAGES = {"compile", "inference", "profile"}`, and the module lives under
`src/slm_lab/deployment/qualcomm/`, an **owned path of T22**
(`ai/tasks/definitions/T22.yaml`), which is being worked concurrently on
`task/T22-qnn-candidates`. Do not add a quantize stage to it from a T41
session. Raise it with T22, or take it as a scoped handover, and record the
decision. This is the single most useful thing to fix if you want Lane A.

**The three plan section 3.2 targets are live.** 79 devices were enumerated;
the candidates' existing selectors all resolve:

| Candidate selector | os | Hexagon | Advertised |
|---|---|---|---|
| `Snapdragon X Elite CRD` | 11 | v73 | `framework:qnn`, `htp-supports-fp16:true` |
| `Snapdragon 8 Elite QRD` | 15 | v79 | `framework:qnn`, `htp-supports-fp16:true` |
| `Dragonwing IQ-9075 EVK` | 1.7 | v73 | `framework:qnn`, `htp-supports-fp16:true` |

The candidate files carry these names together with the full attribute
vocabulary the query observed for each device, and two mechanisms in
`src/slm_lab/quantization/w8.py` keep that honest.
`assert_selectors_match_observation` compares the module constants
`PRIMARY_DEVICE` and `COMPARISON_DEVICES` against
`results/quantization/t41-ai-hub-capability-2026-08-03.json` on every
generation, so drift in what the service says fails rather than silently
re-anchoring; `validate_repository` re-renders both documents from those same
constants and compares them to the files on disk byte for byte, which is what
catches an edited YAML. Do not strip those lists to make a selector resolve:
`check` and
`test_target_device_selectors_carry_the_observed_vocabulary` will both fail,
and the drift you would have hidden is the thing worth seeing. The lists are
deliberately over-constrained, so if a selector stops resolving, trim it to
the device name in the request you are about to submit before concluding the
device is gone. And an advertised attribute is still not a compile result.

**No separate KV-cache dtype knob exists.** The quantize parameter list above
is the whole surface — weights and activations, nothing addressing the cache.
This is consistent with the frozen `CACHE_DTYPE = "float16"` finding and is not
an escape from it. `w8a8` remains Q2-with-a-float16-cache on the hosted lane
exactly as on the local one, and must be reported that way.

**What the query does not establish.** Nothing about this model. Not compiler
acceptance, not operator support, not NPU placement, not 16-bit activation
datapath support, not latency, not memory, not accuracy, and not that a
quantize job for either candidate would succeed. **No job was submitted and no
W8 candidate traversed the public pipeline.** A capability query is not a
pipeline traversal; do not cite it as one.

## Stable interface

Run everything from the repository root; `--repo-root` defaults to the current
directory.

```bash
uv run python -m slm_lab.quantization.w8 check    # offline validation gate
uv run python -m slm_lab.quantization.w8 status   # state, projection, ledger
uv run python -m slm_lab.quantization.w8 record   # write the readiness record
uv run python -m slm_lab.quantization.w8 generate # rewrite both candidates
```

The request emitter composes one schema-v2 AI Hub stage request from the
committed candidate specification plus caller-supplied private paths, writes
it, and stops:

```bash
uv run python -m slm_lab.quantization.w8 request \
  --candidate <w8a16|w8a8> --stage <compile|inference|profile> \
  --context <128|512|1024|4096> --graph <prefill|decode> \
  --quantized-artifact <private-path>.onnx \
  --output-artifact <private-path>.serialized.bin \
  --request-out .ai-local/profiles/T41/compile-request.json
```

Stage-specific arguments: `compile` takes `--quantized-artifact`;
`inference` and `profile` take `--predecessor-manifest` (the sanitized compile
manifest) and `--compiled-artifact`; `inference` additionally takes
`--input-dataset`. `--output-artifact` and `--request-out` are always required.
`--timeout-seconds` defaults to 3,600 and `--stage-manifest` controls only the
public manifest path printed in the follow-up command.

What the emitter derives from committed material rather than from the caller:
`input_specs` from the frozen T12 tensor contract for the requested variant and
graph kind; the pinned `qai-hub` client version `0.53.0` and QAIRT version
`2.45.0.260326154327`; the stage option strings; the target-device selector
from the plan's section 3.2 policy; `retry: false`; and path-free logical names
of the form `<candidate>-S<context>-<graph>-<role>`.

What it guarantees:

- It never imports `qai_hub`, never constructs a backend, and never calls a
  submission path. A test asserts the module is not imported.
- It fails closed. Every input artifact must exist and is hashed; the quantized
  artifact does not exist at this commit, so `--stage compile` exits 1, writes
  nothing, and explains that no W8 artifact exists.
- It refuses a committable output location. Request files, output artifacts,
  and raw profiles must be external to the repository or under `.ai-local/` or
  `artifacts/`.
- Its output is accepted by `ai_hub.load_request` unchanged, for all three
  stages.

After writing a request it prints the exact `scripts/qualcomm/<stage>.py`
command to run next, states that this session submitted nothing and that what
is missing is a W8 artifact rather than permission, and repeats that the
request must not be committed.

## Private artifact layout

Same convention as T30. Requests, raw profiles, and unsanitized service
material go under `.ai-local/profiles/T41/`. Quantized graphs, external data,
encodings, compiled models, and inference datasets go under
`${SLM_LAB_ARTIFACT_ROOT}` or ignored `artifacts/`. Sanitized stage manifests
belong under `results/processed/qualcomm/` and may be committed after evidence
review.

Request files are never committed: they carry machine-local paths
(`scripts/qualcomm/README.md`). The emitter enforces this rather than trusting
it.

Precision evidence for a candidate goes to
`results/quantization/t41-<candidate_id>-precision-evidence.json`. That file is
committed — it holds digests and versions, not artifacts — and it is the only
input that can move a candidate off `specified`.

## Order of operations

The order matters, and the first step is no longer the authorization. That is
granted. The first step is obtaining a W8 artifact, because every step after it
consumes one.

0. **Pick a route to a W8 artifact.** There are two, and only two. *Route A,
   hosted quantize*: the service accepts it (see "What the service exposes"),
   this repository cannot request it, and the module that would is T22's. Take
   this route only as a cross-task decision with T22. *Route B, local AIMET*:
   step 1 below, and the one a T41 session can start on its own. Nothing in
   steps 2-7 is reachable until one of them delivers an artifact.
1. **B1, the host.** Provision Linux x86-64 + CUDA and build the pinned
   environment from `environments/linux-aimet/`. Record which AIMET
   distribution was installed: the PyPI and GitHub `+cu126` distributions share
   the release number `2.36.0` and declare different `torch` majors.
2. **Close T40's numerical parity half first.** It is recorded `not_run` in
   `results/quantization/t40-baseline-parity-2026-08-02.json` and owned by T21.
   A quality delta whose floating side was never verified against the PyTorch
   reference is not a quality delta.
3. **Simulate `w8a16` before `w8a8`.** The conservative candidate isolates
   weight error; a W8A8 result is only interpretable next to it. The Lane B
   runner does not exist at this commit and has to be written.
4. **Write the simulation record**, then confirm with
   `uv run python -m slm_lab.quantization.w8 status` that the state moved to
   `simulated`. Do not assert the state anywhere; it is computed.
5. **B3, the upstream floating path.** T31 and T33 must show that a floating
   Qwen graph compiles, infers, and profiles on a public target. If Qwen is
   blocked there, plan section 3.4 applies and the resulting W8 evidence is
   labelled fallback evidence, never Qwen evidence.
6. **Submission, under the standing authorization.** Check the job against the
   grant's terms — free capacity, zero spend, at most the unspent 120 Device
   Cloud minutes — and go back to the user for anything outside them. Then
   compile, then inference, then profile, as three separate processes through
   the T30 adapters.
7. **Promotion to `deployed` is a data change with a verified chain**, not a
   relabelling: add the three sanitized manifests to the candidate's
   precision-evidence record and let `assess_precision_state` verify them.

## First real-run checklist

Everything on the T30 checklist in `ai/handoffs/T30-ai-hub-adapters.md` still
applies. These are the additions T41 introduces.

- Hold the `qai_hub_submission` resource lock. T41 holds it while open and has
  never exercised it.
- Regenerate every request from the committed candidate with `w8 request`. Do
  not hand-write one and do not reuse a stale request: schema-v1 requests and
  predecessor manifests are intentionally incompatible.
- Verify the quantized artifact's sha256 *before* composing the compile
  request, and record the same digest in the candidate's precision-evidence
  `simulation.quantized_artifact_sha256`. The chain check compares the compile
  manifest's source-artifact digest against that value; if they differ, the
  compiled artifact is not this candidate's and the state stays `simulated`.
- Submit compile, inference, and profile in that order and keep the sanitized
  compile manifest: both downstream stages cite its recomputed digest as
  `predecessor_manifest_sha256`, and both must consume a `compiled_model`
  source artifact matching the compile manifest's target digest.
- Run one candidate through one context and one graph kind at a time. The job
  name encodes all four (`slm-lab-t41-<candidate>-<stage>-S<context>-<graph>`),
  so a mixed set is easy to create and hard to unpick later.
- A W8 quality result must set `source.precision` to
  `<candidate_id>+<simulated|deployed>` and `system.evidence_level` to
  `simulated`, `observed_real_device`, or `observed_hosted_device` to match.
  `compare_quality` derives the comparison scope from that declaration and has
  no caller argument that could override it.
- Attach the inherited T40 calibration bias to every reported delta. The
  comparison does this automatically; a human summary of the delta must not
  drop it.
- Any future W8 artifact manifest must carry `precision`, `quantization`,
  `calibration_dataset_revision`, and `source_artifact_sha256` (plan section
  17.4). A manifest whose calibration revision differs from the frozen one was
  calibrated on a different corpus and its delta is not comparable.
- Sanitize everything through the T30 adapter before committing. No token,
  account identifier, job identifier, job URL, raw service response, or local
  path in any committed file.

## What the standing authorization covers, and what still needs a decision

Covered by the 2026-08-03 grant, for T41, without asking again:

- Submitting AI Hub compile, profile, and inference jobs on **free capacity**.
- Device Cloud interactive minutes, up to the **120-minute** cap, of which
  **0 have been used**.

Still needs a fresh decision from the user, every time:

- Any job that costs money, or any capacity outside the free tier. The grant is
  explicitly zero-spend.
- The 121st Device Cloud minute, or any use once the ceiling is reached.
- Pushing, publishing a release, or creating any public GitHub state.
- Any submission for a task other than T41. This grant is T41-scoped.

When you do ask, name the target device or devices, the exact stages, the
expected cost, and which candidate and context variant. Record the answer as an
external-job event with target, runtime and compiler versions, configuration,
cost, and artifact hashes, per `AGENTS.md`.

## What would invalidate the frozen candidates

`uv run python -m slm_lab.quantization.w8 check` fails, rather than
re-anchoring silently, if any of these drifts. If one does, a delta measured
against the old specification is not comparable with one measured against the
new, and the honest response is a new candidate id, not an edited one.

- **The calibration revision.** Any change to
  `configs/quantization/calibration.yaml` that moves
  `calibration_dataset_revision` off
  `t40-qwen3-0.6b-t10-derived-v1+sha256.d2b749e15dd5d987`.
- **A baseline manifest digest.** Any edit to
  `results/manifests/onnx/S{128,512,1024,4096}.json`, including a re-export
  that changes a graph digest.
- **The T13 protocol digest.** Any change that moves the frozen protocol off
  `2541fa76fb088de3ebb559aeb300aed5cd62e215994b8db0faa2fbc6273f947e`.
- **The cache dtype.** Any change to
  `slm_lab.contracts.static_cache.CACHE_DTYPE`, which would also re-open the
  T20/T23 export boundary.
- **The model contract.** Any change to the architecture fields in
  `configs/models/qwen3-0.6b.yaml`; untying the embeddings makes the projection
  refuse outright rather than miscount.
- **The T21 graph inventory.** A change in initializer counts or the largest
  inline initializer size breaks the projection's cross-check against the
  committed float16 export.

## Known boundary

T41 produced no quantized weight, no encodings, no simulation, no job, and no
measurement, and none of its outputs may be quoted as one. The one live-service
action it took was a read-only capability query, which established an API
surface and a device list on 2026-08-03 and nothing about this model. **No W8
candidate traversed the public pipeline**, and the acceptance criterion that
asks for one is not satisfied. The analytic weight-storage projection is
arithmetic over committed inputs, labelled `analytic_projection` on every row;
it is not an artifact size, an encoding overhead, a peak memory, or a latency.

Two capability gaps are not defects and are not T41's to close.
`slm_lab.deployment.qualcomm.ai_hub` implements compile, inference, and profile
and has **no quantize stage**, so Lane A cannot itself produce the quantized
artifact — the compile stage consumes one. The service does offer the stage, so
this is a missing adapter in a **T22-owned** module, not a platform limit. And
committed T02 access evidence records the Workbench Quantize Job at AIMET
`2.34` while `environments/linux-aimet/` pins `2.36.0`, so a Lane A artifact
and a Lane B artifact are different artifacts; their encodings must not be
compared as though a difference between them were a property of the model.

T42 owns W4A8, LPBQ, LiteMP, mixed precision, and sensitivity analysis, and
should extend the candidate model rather than fork it. T43 owns quantized
compile, inference, and profile evidence and the quality-latency-memory
comparison built on it. Neither may widen the T40 calibration contract or the
T13 evaluation contract to make a result look better.
