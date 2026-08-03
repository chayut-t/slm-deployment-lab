# T22: QNN candidates and packaging

Status: active — all five milestones met on `task/T22-qnn-candidates`; the
plan stays here because the branch is unmerged and the task graph reads
`in_progress`
Owner: Claude t22-main agent
Updated: 2026-08-03

## Objective

Turn the T20/T23 reference ONNX exports into a distinct, compiler-oriented
`qnn_candidate` artifact stage, record every transformation that produced it,
and leave a package plus a validated Workbench compile request that T31 can
submit without re-deriving anything. The reference artifacts must remain
byte-identical and independently identifiable throughout.

## Scope

### In scope

- A declarative, ordered transformation catalogue and a target-neutral
  transform engine under `src/slm_lab/graph/qnn/`.
- Candidate graphs for all four context variants and both graph kinds, written
  to `${SLM_LAB_ARTIFACT_ROOT}/onnx/qnn-candidate/T22/`.
- A committed transformation manifest per variant under
  `results/manifests/qnn/`, carrying source digests, candidate digests, the
  ordered transformation records, and the structural before/after.
- Structural re-inspection of every candidate through the committed T21 rule
  engine, and ONNX Runtime CPU parity of the candidate measured with the T21
  protocol.
- A package builder and an AI Hub compile-request generator under
  `src/slm_lab/deployment/qualcomm/`, plus a preflight that validates the
  request through the committed T30 adapter without contacting the service.

### Out of scope

- Submitting any AI Hub job. T22 holds `t9_heavy_io` only; `qai_hub_submission`
  belongs to T31.
- Quantization. The candidate stays float16 weights / float32 logits.
- Changing the T12 static-cache contract. The `fixed_capacity_indexed_copy`
  decode write and the `prefill_prefix_materialization` prefill write are
  preserved exactly; replacing them is a contract decision, not a T22 rewrite.
- Re-exporting from PyTorch. T22 rewrites graphs, it does not produce them.

## Dependencies and resources

- Required task dependencies: T21 (completed), T23 (completed).
- Resource locks: `t9_heavy_io`.
- External access: none. No network, no service, no spending.
- Cost boundary: zero. Local CPU and local storage only.

## Important paths

- Inputs: `results/manifests/onnx/S*.json`, `results/graph/S*.json`,
  `docs/results/onnx/graph-inspection.md`,
  `configs/graph/onnx-risk-rules-v1.json`,
  `${SLM_LAB_ARTIFACT_ROOT}/onnx/reference/T20/`.
- Outputs: `src/slm_lab/graph/qnn/`,
  `src/slm_lab/deployment/qualcomm/packaging.py`,
  `configs/graph/qnn-transforms-v1.json`, `results/manifests/qnn/`,
  `${SLM_LAB_ARTIFACT_ROOT}/onnx/qnn-candidate/T22/`,
  `docs/results/qualcomm/qnn-candidates.md`.
- Shared contracts: `slm_lab.graph.onnx_reader` and `slm_lab.graph.inspection`
  are consumed as a library and not modified. `slm_lab.backends.onnx_cpu`
  receives one generalization (below) and no behaviour change for T20
  manifests.

## The transformation profile

Catalogue id `qnn-candidate-v1`, in `configs/graph/qnn-transforms-v1.json`.
Passes are ordered and every one records observed issue, exact action,
parameters, and measured effect per graph.

1. `X-CONSTANT-TO-INITIALIZER` — rewrite every `Constant` node as a graph
   initializer. Addresses `R-LARGE-INLINE-CONSTANT`: the O(S^2) causal mask and
   the 56 zero cache reserves are `Constant` attributes today, so they bypass
   T20's 1,024-byte external-data threshold and sit inline in the protobuf.
2. `X-STATIC-SHAPE-FOLD` — evaluate and replace nodes whose every input is a
   known constant, restricted to an allowlisted operator set and a byte budget.
   What keeps a weight tensor out of a fold is neither of those on its own: an
   initializer that lives in external data is not in the constant pool at all,
   so a float16 weight cannot be a fold input under *any* budget. The budget
   bounds what a fold may materialize, and the allowlist keeps `MatMul`,
   `Softmax` and normalization out; a sub-1 MiB inline float tensor would be
   foldable under the budget alone. Addresses
   `R-DATA-DEPENDENT-SHAPE-INPUT` (804 prefill / 1,231 decode) and
   `R-SHAPE-COMPUTATION-CHAIN` (127 / 556), the T21 rank-1 and rank-7 findings.
3. `X-DEAD-NODE-ELIMINATION` — drop nodes and initializers no graph output
   depends on, which is what makes pass 2's node reduction real.
4. `X-EXTERNALIZE-LARGE-TENSORS` — write initializers at or above the
   threshold into the candidate's own external-data sidecar, so the candidate
   protobuf stops carrying the mask and the reserves.
5. `X-INFER-VALUE-INFO` — run ONNX shape inference and keep the inferred
   `value_info`. This closes the T21 evidence boundary in
   `docs/results/onnx/graph-inspection.md` section 7: `R-INTERNAL-DYNAMIC-SHAPE`
   inspected 0 of 0 entries because the exports declare no interior shapes.
6. `X-STAMP-CANDIDATE-PROVENANCE` — set the producer and `metadata_props` to
   record source graph digest, catalogue id and digest, and the applied pass
   list, so a candidate is self-identifying in-band and can never be mistaken
   for the reference.

### One rejected transformation, recorded with its evidence

`X-ORT-CPU-OFFLINE-OPTIMIZATION` (ONNX Runtime `optimized_model_filepath` at
`ORT_ENABLE_BASIC`) is the obvious way to fold these graphs and it must be
rejected on measured grounds, not asserted away. A planning probe on
`S128/prefill.onnx` with onnxruntime 1.28.0 on the CPU execution provider
observed that it:

- decomposes all 28 `Softmax` nodes into `Sub`/`Exp`/`ReduceMax`/`ReduceSum`/
  `Div`, replacing a single operator an NPU implements natively with five, on
  exactly the operator T21 section 5.6 flags as precision-sensitive;
- raises `Cast` from 348 to 830, which is the CPU execution provider's float16
  fallback inserting casts around kernels it lacks — a host-specific artifact
  with no business in a candidate aimed at a different target;
- stamps eight opset imports no node uses, including `com.microsoft`,
  `com.microsoft.nchwc` and `org.pytorch.aten`, weakening the single-domain
  portability T21 section 6 records as one of the three clean blocking results.

The build tool must reproduce these numbers itself and write them into the
manifest as a rejected transformation. A probe measurement is not evidence
until the committed tooling reproduces it.

## Committed manifest schema

`results/manifests/qnn/S<context>.json`, `schema_version: 1`, `task_id: T22`,
`stage: "qnn_candidate"`:

- `variant_id`, `context_length`, `cache_capacity`, `opset`, `precision`.
- `source`: the T20 manifest path and digest, and per graph kind the reference
  `relative_path`, `sha256`, `size_bytes`, and external-data records.
- `transform_catalogue`: path, `catalogue_id`, `sha256`.
- `toolchain`: exact `python`, `onnx`, `onnxruntime`, `numpy` versions read
  from the running interpreter, never copied from prose.
- `artifacts`: `root` as `${SLM_LAB_ARTIFACT_ROOT}/onnx/qnn-candidate/T22`,
  plus a `prefill` and a `decode` record with `relative_path`, `sha256`,
  `size_bytes`, `external_data`, `input_tensors`, `output_tensors`. This block
  is deliberately shaped like the T20 manifest's so
  `slm_lab.graph.inspection` and `slm_lab.backends.onnx_cpu` can both read it.
- `transformations`: the ordered records, each with `id`, `applied`,
  `observed_issue`, `rule_ids`, `transformation`, `parameters`, and `effect`
  per graph kind. The rejected pass appears here with `applied: false` and its
  measured `rejection_evidence`.
- `structural_delta`: node count, operator histogram, boundary counts,
  initializer counts, inline bytes, and per-rule finding counts before and
  after, produced by the T21 rule engine used as a library.
- `verification`: `onnx_checker`, `graph_inspection`, `ort_cpu_parity`, each
  either a real result or an explicit `not_measured` with a reason.
- `claim_boundary`: `establishes` / `does_not_establish`, in the same shape T20
  and T21 use.

## Milestones

- [x] Transform engine and catalogue exist with unit tests that run in the
      locked root environment, skipping the `onnx`-dependent cases.
- [x] Candidate graphs built for all four variants and both kinds, with
      manifests committed and `--check` reproducing them. `--check` exited 0
      for all four variants on 2026-08-03.
- [x] Every candidate passes `onnx.checker` and is re-inspected by the T21 rule
      engine, with the rank-1 and rank-4 findings measurably reduced. Rank 1:
      prefill 804 → 0/5/6/6, decode 1,231 → 423. Rank 4
      (`R-LARGE-INLINE-CONSTANT`) cleared at every variant where it fired.
- [x] ONNX Runtime CPU parity measured on the candidates with the T21 protocol
      and the T23 tolerance, recorded as a real pass or a real failure. All
      four passed, and the candidate logits are bit-identical to the committed
      reference records.
- [x] A package and a validated compile request exist for the primary variant,
      with the request proven to satisfy the committed T30 adapter contract
      without any service call. Delivered for all four variants and both graph
      kinds, not only the primary variant.

## Verification and acceptance

- Commands: the repository test suite; `python -m slm_lab.graph.qnn.build
  --all-manifests --check`; `python3 scripts/ai/render_task_status.py --check`;
  `python3 scripts/repo/check_hygiene.py --all`.
- Numerical criteria: candidate ORT CPU parity is measured under the same
  tolerance the reference records in `results/graph/parity/`. A parity failure
  is recorded as a failure; no threshold is widened to absorb one.
- Hardware/profile evidence: none. T22 produces no device evidence and must not
  claim any.

## Artifact and privacy handling

- Committed evidence: `results/manifests/qnn/`, the transform catalogue, the
  reader-facing report, and the source code.
- External artifacts: the candidate graphs and their external-data sidecars,
  under `${SLM_LAB_ARTIFACT_ROOT}/onnx/qnn-candidate/T22/`, never committed.
- Private/local material: generated AI Hub compile requests contain
  machine-local paths and are written under `.ai-local/` only, per
  `scripts/qualcomm/README.md`.

## Decisions and discoveries

- 2026-08-03: Rejected ONNX Runtime CPU offline optimization as the folding
  mechanism on measured grounds; see the rejected-transformation section.
- 2026-08-03: Chose a target-neutral, allowlisted, byte-budgeted folder so no
  float weight is ever evaluated and no host-specific kernel choice enters the
  candidate. A planning probe folded 4,850 of 7,634 S128 prefill nodes in a
  single topological pass without touching `Softmax`.
- 2026-08-03: Candidate parity reuses the T21 runner rather than a second
  implementation, which requires the runner to resolve its artifact directory
  from the manifest instead of the hard-coded `onnx/reference/T20`. For every
  committed T20 manifest that expression already resolves to the same path, so
  no committed T21 or T23 evidence changes.
- 2026-08-03: The rejected pass's grounds grew from three to **four** when the
  build tool measured what the plan had asserted. The fourth is the one that
  turns a preference into a constraint: the optimizer inlines the weights, so
  the S4096 prefill output is a single 1,811,439,962-byte protobuf — 1.69 GiB,
  84% of protobuf's 2 GiB serialization ceiling — with no sidecar written. One
  probe figure did not reproduce (S128 prefill inline bytes) and the cause is
  recorded as unknown rather than guessed.
- 2026-08-03: `X-INFER-VALUE-INFO` closed the `R-INTERNAL-DYNAMIC-SHAPE`
  evidence boundary and closed it **unfavourably**. 9 non-static interior
  tensors in prefill at S512 and above, and 1,069 in decode of 5,363 annotated.
  The clean answer the static public boundary made "likely" is not the answer.
- 2026-08-03: `X-DEAD-NODE-ELIMINATION` removed **zero nodes** on all eight
  graphs. Its initializer arm removed 3,335–3,399 per prefill graph and 1,629
  per decode graph. Both arms are recorded separately so the zero is legible as
  a measurement rather than as a defect.
- 2026-08-03: Decode's rank-1 count falls only to 423 because it builds its
  attention mask at run time from `Range`/`Greater` over activations, so 429 of
  its 459 `Shape` nodes read non-constant tensors whose *shapes* are static. A
  shape-aware second fold running after `X-INFER-VALUE-INFO` is the obvious
  follow-up and is out of scope here.

## Progress and restart instructions

**Current state, 2026-08-03: all five milestones are met and none of the work
is merged.** The task graph reads `in_progress` with owner
`Claude t22-main agent` and branch `task/T22-qnn-candidates`. `T31` is blocked
on the integration, not on any missing T22 work. This plan stays in
`ai/plans/active/` because `ai/plans/completed/README.md` admits a plan only
"after its task is integrated and the task graph marks it completed with a
matching public worklog", and neither is true yet.

The worklog is
`ai/worklogs/2026-08-03-T22-qnn-candidates-and-packaging.md`. It is committed
but **not referenced from the task graph**, because
`scripts/ai/render_task_status.py` refuses a `worklog` field on any status
other than `completed`. No learning checkpoint was added for the same class of
reason: `configs/learning/checkpoints.yaml` covers completed tasks only, and
`scripts/learning/build_learning_sheet.py` enforces it. The ready-to-paste
`LEARN-12` entry and the four-step close-out sequence are in
`ai/handoffs/T22-qnn-candidates.md`.

**The exact next action is to merge `task/T22-qnn-candidates` into the
integration branch, then run the close-out sequence in that handoff.** Nothing
else in this plan is outstanding.

If you are picking this up cold instead, read
`docs/results/qualcomm/qnn-candidates.md` first — it is the reader-facing
account of everything below — then `docs/results/onnx/graph-inspection.md`
sections 5 and 7, which is where every transformation in the profile comes
from. The build tool is `python -m slm_lab.graph.qnn.build`; run it with
`--check` first, in an environment with `onnx`, `onnxruntime` and `numpy` and
with `SLM_LAB_ARTIFACT_ROOT` set, to see whether the committed manifests still
reproduce from the artifacts on disk.
