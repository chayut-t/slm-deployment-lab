# Qwen3-0.6B on three Qualcomm Workbench targets

Task: `T31`
Date: 2026-08-04
Measured: nothing on any device
Status: **complete run plan; zero device evidence on all three targets**

> **No Qualcomm AI Hub job was submitted, no service call was made, and no
> device produced any evidence.** There is no latency, no memory figure, no
> accelerator placement, no operator-support verdict, and no numerical
> comparison in this report, for any of the three targets. Every number below
> is one of three things: a digest or byte count read off a committed file, a
> finding count read off a committed T22 inspection manifest, or a count of
> plan entries. The machine-readable record is
> `results/raw/qualcomm/workbench/t31-workbench-run-plan-2026-08-04.json`.

T31 was scoped to compile, run inference on, and profile the T22 Qwen
candidate graphs on Snapdragon X Elite CRD, Dragonwing IQ-9075 EVK, and
Snapdragon 8 Elite QRD. It produced everything up to the submission boundary
and stopped there, because two things a submission requires are absent and
neither can be manufactured inside this task.

## The two blockers, exactly

**B1 — the client is not installed.** The optional `qai-hub` client does not
exist in any environment in this repository. `pyproject.toml` pins no
Qualcomm client. Both the project virtual environment and the system
interpreter fail on `import qai_hub`. A client configuration file exists on
this host, but the library that would read it does not. This is not a
permission problem and not a service problem; it is an absent dependency.

**B2 — no submission permission is held by this session.** `AGENTS.md`
requires explicit user permission before any external job. The "Budget state
at handoff" paragraph in `ai/handoffs/T31-first-submission.md` records that an
earlier session was granted hosted compile/profile/inference plus up to 120
Device Cloud minutes on free capacity. That paragraph is a note written into a
file by an earlier agent. It is useful context for a future session and it is
not consent held by this one, so it was treated as context and nothing was
submitted.

Both are recorded in the run plan under
`submission_boundary.required_before_any_submission`, with an offline recheck
for the first and an explicit `not_machine_checkable` for the second.

Neither blocker is a defect in the pipeline. What T31 could build without
them, it built.

## Per-target status

The acceptance criterion is "unavailable targets are reported without proxy
claims". All three are unavailable, for the same reason, and none stands in
for another.

| Target | Device evidence | Compile | Inference | Profile | Status |
|---|---|---|---|---|---|
| Snapdragon X Elite CRD | authenticated T02 device query (toy model, not Qwen) | 8 requests `ready` | 8 stages `pending_predecessor` | 8 stages `pending_predecessor` | **No Qwen evidence.** Blocked on B1 and B2 |
| Dragonwing IQ-9075 EVK | unauthenticated public catalog listing only | 8 requests `ready` | 8 stages `pending_predecessor` | 8 stages `pending_predecessor` | **No Qwen evidence.** Blocked on B1 and B2, and the selector itself is unconfirmed |
| Snapdragon 8 Elite QRD | unauthenticated public catalog listing only | 8 requests `ready` | 8 stages `pending_predecessor` | 8 stages `pending_predecessor` | **No Qwen evidence.** Blocked on B1 and B2, and the selector itself is unconfirmed |

Three things this table must not be read as saying.

The X Elite row is *not* stronger because of T02. The 2026-07-25 authenticated
lifecycle compiled a toy model on that device. It establishes the device
identity and the QAIRT default that the selector copies, and nothing about
Qwen3-0.6B. Presenting it as a Qwen result would be exactly the substitution
the acceptance criterion forbids.

The two catalog rows are weaker in a second, independent way. Their device
names were read off a public Qwen3-0.6B model page without authentication. No
authenticated device query has ever confirmed that either name resolves, that
this account can schedule either device, or that the pinned QAIRT version is
supported on either. Each config says so in its own `claim_boundary`, and the
planner reads the strength off that boundary rather than assigning one — a
target config that declared neither marker would be refused rather than
guessed at.

And no row is a proxy for another. Nothing measured on X Elite would be
reported as an IQ-9075 or 8 Elite result, and the plan carries that rule as a
field (`no_proxy_rule`) on every target rather than as prose here.

## What was built

`src/slm_lab/deployment/qualcomm/workbench.py` derives the whole T31 matrix
from committed inputs: three target selectors, four T22 package records, four
T22 candidate manifests bound through them by digest, four T22 structural
inspections, and four T21/T22 ONNX Runtime CPU parity records. Every input is
bound by repository-relative path and SHA-256.

The result is 3 targets x 4 context variants x 2 graph kinds = 24 plan
entries, each with three stages: **24 stages `ready`, 48 stages
`pending_predecessor`**.

### `ready` means one specific thing

A compile stage reads `ready` when the request satisfies the committed T30
compile validation chain — schema version, stage, exact field set, client
version, device selector, runtime identity, option allowlist, job name,
timeout, `retry: false`, input specs, path-free logical names, and the
public-safety projection. That chain is not reimplemented here; the planner
calls the same `ai_hub` functions the stage runner calls.

Two checks in that chain cannot run without artifact bytes: the source
artifact's existence and rehash, and the preparation of the output artifact's
private parent. Both are recorded on the plan under
`deferred_to_submission_time` rather than quietly skipped, and the optional
`--preflight` mode runs the real `ai_hub.preflight_compile_request` over all
24 requests against the assembled packages on the external artifact root.

The check that makes the offline derivation trustworthy is an equality. The
planner derives each compile request's deterministic `request_id` without
reading a single artifact byte, and for the eight X Elite entries that id is
**identical to the id already committed in
`results/manifests/qnn/packages/S*.json`**, which T22 obtained by running the
real preflight. The first one is `t30-compile-83b8813c19a37ac036ad`. A test
asserts all eight, and `--preflight` asserts all 24 against the real function
and refuses any mismatch.

### `pending_predecessor` means the opposite

An inference or profile request cannot be materialized now, and the plan says
so rather than filling the gap. `ai_hub._load_predecessor` requires a
sanitized schema-v2 compile manifest with `status: success`, and
`ai_hub._compiled_artifact` requires the compiled artifact's logical name and
digest to equal that manifest's. Neither exists before a real compile job
runs, so those fields are `null` and listed under `unresolved_input_ids`:

| Stage | Unresolved | Why |
|---|---|---|
| inference | `predecessor_manifest` | No successful compile manifest exists for any target |
| inference | `compiled_artifact_sha256` | Read off the bytes the service returns |
| inference | `input_dataset` | This repository contains no AI Hub-compatible HDF5 dataset at this commit |
| inference | `output_path` | A private machine-local path chosen at submission time |
| profile | `predecessor_manifest`, `compiled_artifact_sha256`, `output_path` | Same, minus the dataset |

Emitting a synthetic predecessor so the request would validate here was
deliberately not done. It would produce a plan that passes locally and fails
at the service, which is the failure mode this module exists to prevent. A
test asserts that no stage carries a fabricated predecessor and that
`readiness` cannot be promoted without `--check` catching it.

## The submission order, and why it is that order

The matrix is emitted in one deterministic order. Targets sort by device
evidence strength then `config_id`; graphs sort by the residual population of
high-severity shape findings on the *candidate*, then by candidate protobuf
bytes; the matrix sorts by target then graph, so one target is exhausted
before the next begins.

Nothing in that ordering is asserted here. The shape populations are read off
`results/manifests/qnn/inspection/S*.json`:

| Order | Variant | Graph | `R-DATA-DEPENDENT-SHAPE-INPUT` | `R-INTERNAL-DYNAMIC-SHAPE` | Candidate total | Reference total | Nodes | Protobuf bytes |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | S128 | prefill | 0 | 0 | **0** | 804 | 2,785 | 949,559 |
| 2 | S512 | prefill | 5 | 9 | 14 | 804 | 2,817 | 956,509 |
| 3 | S1024 | prefill | 6 | 9 | 15 | 804 | 2,824 | 958,649 |
| 4 | S4096 | prefill | 6 | 9 | 15 | 804 | 2,824 | 958,654 |
| 5 | S128 | decode | 423 | 1,069 | **1,492** | 1,231 | 5,421 | 1,752,536 |
| 6 | S512 | decode | 423 | 1,069 | 1,492 | 1,231 | 5,421 | 1,752,536 |
| 7 | S1024 | decode | 423 | 1,069 | 1,492 | 1,231 | 5,421 | 1,752,538 |
| 8 | S4096 | decode | 423 | 1,069 | 1,492 | 1,231 | 5,421 | 1,752,538 |

S128 prefill leads because it is the only graph in the matrix that comes out
of the T22 shape rewrite clean — 804 residual findings on the reference, zero
on the candidate — and because it is the smallest candidate protobuf. If it
fails, the failure is about the pipeline, not about the graph.

Decode trails, and the reason is in the table rather than in a preference.
The fold takes decode's rank-1 finding from 1,231 to 423 and produces 1,069
interior tensors whose shape ONNX shape inference cannot resolve, so by raw
high-severity count the decode candidate scores **worse** than its reference.
A static-shape ahead-of-time compiler is precisely the consumer that will care.
Leading with decode would mean a first failure that could not be separated
from that population.

The first submission is therefore Snapdragon X Elite CRD / S128 / prefill /
compile, request `t30-compile-83b8813c19a37ac036ad`.

## What is expected to break first, and how to attribute it

The plan carries this as a field, `first_failure_hypothesis`, not as advice.

The compile request names only the `.onnx` file, because the committed T30
adapter requires `source_artifact.path` to be one existing file. Every
candidate in the matrix carries an external-data sidecar beside it — between
1,192,085,504 and 1,240,451,072 bytes, depending on variant and graph kind.
Whether the service reads that sidecar from the same directory, or wants a
directory or an archive instead, has never been tested against AI Hub. It is
on the path of the very first submission.

**The attribution rule.** A failure whose diagnostic names the model upload, a
missing external data file, or the source artifact is a *packaging* result. It
must not be recorded as a graph result, a compiler result, or an
operator-support result, and it says nothing about the target it happened on.

The rule has a cheap test attached. A packaging failure is
target-independent, so it reproduces identically on every target and every
variant. A compiler or operator failure varies with the graph. The submission
order makes that discrimination available immediately: positions 1 and 2 are
the same graph kind at different residual shape populations.

## The numerical comparison that was not made

The acceptance criterion asks that numerical output be compared with a
reference. No inference stage ran, so no comparison exists.

What the plan does carry, per variant, is the reference such a comparison
would use: `results/manifests/qnn/parity/S*-ort-cpu.json`, evidence tier
`real_onnxruntime_cpu`, bound by digest, with `status: not_compared` and the
two things blocking it — no inference stage has run on any target, and this
repository contains no AI Hub-compatible input dataset.

The reason that reference is the right one is a T22 measurement rather than a
convenience: the candidates are bit-identical to the reference graphs on the
ONNX Runtime CPU provider across all 20 recorded steps. So any divergence a
device eventually shows belongs to the compiler, the runtime, or the
hardware — not to the T22 rewrites. That property is what makes the eventual
comparison interpretable, and it is also the reason nothing may be inferred
from it today.

## Device and toolchain versions

The criterion asks that device and toolchain versions be captured. What is
captured is the *requested* identity of each, from committed evidence. No
observed version exists, because nothing ran.

| Field | Value | Where it comes from |
|---|---|---|
| Client | `qai-hub` 0.53.0 | The 2026-07-25 authenticated T02 lifecycle; not installed here |
| Runtime | QAIRT 2.45.0.260326154327 | The resolved default from the same authenticated framework query |
| Compile options | `--target_runtime qnn_context_binary --qairt_version 2.45.0.260326154327` | Each target config, validated by the T30 compile allowlist |
| Inference/profile options | `--qairt_framework 2.45.0.260326154327` | Composed by the planner, validated by the T30 allowlist for each stage |
| Devices | `Snapdragon X Elite CRD` (os `Windows 11`), `Dragonwing IQ-9075 EVK`, `Snapdragon 8 Elite QRD` | Target configs; only the first carries an OS, and only because an authenticated query returned it |
| Timeout / retry | 3600 s, `retry: false` | Target configs |

The QAIRT version is device-independent evidence: it came from a service-wide
`get_frameworks` query, so reusing it on the two catalog-only targets imports
no device claim. Whether it is supported on either is unverified, and each
config's boundary says so.

No `--compute_unit` flag is set on any stage. The T30 allowlist accepts one,
but pinning it would decide the placement question the profile stage exists to
observe; the normalized profile reports the compute units it actually saw, and
a plan that pinned them would return its own input.

## Cost

| Term | Value |
|---|---|
| Jobs submitted | 0 |
| Service calls made | 0 |
| Device minutes consumed | 0 |
| Cost incurred | US$0.00 |

Nothing was submitted, leased, or spent. The 120-minute Device Cloud ceiling
recorded in `ai/handoffs/T31-first-submission.md` is untouched and remains
whole for a future session that holds permission.

## What would run the moment the client and permission exist

Every command below exists at this commit. Placeholders in angle brackets are
private, machine-local paths.

1. **Re-check the plan.** Offline, no artifact root, no client:

   ```bash
   PYTHONPATH=src python3 scripts/qualcomm/plan_workbench_run.py --check
   ```

   It re-derives the plan from committed inputs and fails on the first
   differing key path. Run it first: if a target config or a T22 record moved,
   the plan below is stale.

2. **Install the client at the pinned version, in an isolated environment.**
   The common environment deliberately does not carry it. The exact version
   every target config names is `0.53.0`; the stage adapter refuses to run if
   the installed version differs from the request.

3. **Confirm the permission covers what is about to happen.** Hosted compile
   is the free tier; on-device profiling draws on the metered pool. Hold the
   `qai_hub_submission` resource lock for the duration.

4. **Assemble the packages and preflight the whole plan for real.** This
   writes request files into `.ai-local/profiles/T31/` and submits nothing:

   ```bash
   SLM_LAB_ARTIFACT_ROOT=<external-root> PYTHONPATH=src python3 \
     scripts/qualcomm/package_qnn_candidate.py \
     --manifest results/manifests/qnn/S128.json --check

   SLM_LAB_ARTIFACT_ROOT=<external-root> PYTHONPATH=src python3 \
     scripts/qualcomm/plan_workbench_run.py --preflight
   ```

5. **Submit position 1 and nothing else.** Snapdragon X Elite CRD, S128,
   prefill, compile:

   ```bash
   PYTHONPATH=src python3 scripts/qualcomm/compile.py \
     --request .ai-local/profiles/T31/qualcomm-snapdragon-x-elite-crd/S128/prefill-compile-request.json \
     --manifest results/processed/qualcomm/t31-x-elite-S128-prefill-compile.json
   ```

   Stop and read the result. If it fails, apply the attribution rule above
   before touching anything else: re-run the same graph with a directory or
   archive source, and record a packaging failure as a packaging failure.

6. **Only then chain inference and profile.** Both requests need
   `--predecessor-manifest` pointing at the sanitized compile manifest from
   step 5 and `--compiled-artifact` pointing at the downloaded compiled model;
   inference additionally needs an AI Hub-compatible input dataset, which this
   repository does not contain and which must be produced from the T22
   workload before the numerical comparison is possible.

7. **Then widen.** Positions 2 to 4 are the remaining prefill graphs on the
   same target; position 5 is the first decode graph; positions 9 and 17 move
   to the two catalog-only targets. The first request against either of those
   is also the first test that its selector resolves at all, and a resolution
   failure there is a *selector* result, not a graph result.

## Reproducing this report

```bash
PYTHONPATH=src python3 scripts/qualcomm/plan_workbench_run.py          # writes the record
PYTHONPATH=src python3 scripts/qualcomm/plan_workbench_run.py --check  # re-derives it
PYTHONPATH=src python3 -m pytest tests/deployment/qualcomm -q
```

All three run offline on the primary macOS host with no Qualcomm client
installed. The planner imports no Qualcomm client on any code path, and a test
asserts `qai_hub` never enters `sys.modules`.

The optional full check needs the assembled T22 packages on the external
artifact root and still submits nothing:

```bash
SLM_LAB_ARTIFACT_ROOT=<external-root> PYTHONPATH=src python3 \
  scripts/qualcomm/plan_workbench_run.py --preflight
```

It ran on this host at this commit and validated all 24 compile requests
through `ai_hub.preflight_compile_request`, each returning the request id the
planner had already derived offline. The committed record carries that as a
dated observation under `run_observation.preflight`, alongside
`run_observation.client_probe`, which records `importable: false` for
`qai_hub` — read by `importlib.util.find_spec`, which locates a module without
executing it. Both live outside the block `--check` compares, because they
describe one machine on one day rather than the plan's contract.

## Evidence boundaries

From the record's own `claim_boundary`, nothing here establishes:

- that Qualcomm AI Hub accepted, or would accept, any request in this plan;
- that AI Hub accepts the external-data package layout the plan names;
- compiler acceptance or operator support for any Qwen3 graph;
- that any of the three devices resolves, is schedulable, or is reachable by
  this account;
- accelerator placement or fallback behaviour;
- latency, throughput, peak memory, or energy on any target;
- device numerical parity, or any comparison against the T22 reference logits;
- that a result on one target transfers to another.

What it does establish, and only this:

- a deterministic three-target run plan derived only from committed inputs;
- that every compile request in it satisfies the committed T30 compile
  validation chain offline;
- that each compile request id equals the value the T30 preflight records for
  the same request, verified against eight committed T22 ids and re-verified
  for all 24 through the real preflight;
- inference and profile specifications that are explicitly blocked on a real
  compile output rather than completed with a placeholder;
- a deterministic submission order whose reason is derived from the committed
  T22 inspection manifests;
- each target's device-evidence strength, read from that target's own
  committed claim boundary;
- a dated record that no job was submitted, no service was contacted, and
  nothing was spent.

## See also

- `results/raw/qualcomm/workbench/README.md` — how to read the record.
- `docs/results/qualcomm/qnn-candidates.md` — the T22 candidates this plan
  consumes, including section 6.1 on the decode shape population.
- `ai/handoffs/T22-qnn-candidates.md` — what T22 established and what it
  deliberately did not.
- `ai/handoffs/T31-first-submission.md` — the sequencing this plan encodes.
- `scripts/qualcomm/README.md` — the stage adapters and the request contract.
- `configs/targets/README.md` — the three selectors and their evidence.
- `ai/plans/active/T31-qwen-workbench.md` — the execution plan, the milestone
  state, and the restart sequence.
- `ai/worklogs/2026-08-04-T31-workbench-run-plan-submission-boundary.md` — the
  worklog for this stop at the submission boundary. The task graph cannot
  reference it by field while T31 is not `completed`, so it is named here.
