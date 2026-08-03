# T22: QNN Candidates And Packaging

Date: 2026-08-03
Task: `T22`
Visibility: `public`
Status: implementation complete on `task/T22-qnn-candidates`; not merged, so
the task graph reads `in_progress`

## Outcome

The `qnn_candidate` artifact stage exists. Six declarative graph rewrites turn
the eight hash-verified T20/T23 reference exports into eight candidate graphs
that pass `onnx.checker`, preserve the frozen T12 boundary name-for-name, keep
all 56 cache writes on the contract's operator in both graph kinds, and are
**numerically indistinguishable from the reference** on the ONNX Runtime CPU
execution provider.

The structural result is large. Prefill falls from 7,634 nodes to 2,785–2,824,
its protobuf stops scaling with `S` entirely (a 9.70x spread across the matrix
collapses to 1.01x), and the T21 rank-1 finding falls from 804 to at most 6.
Decode falls from 10,191 nodes to 5,421 and its rank-1 finding from 1,231 to
423.

The numerical result is stronger than "within tolerance". Every one of the 20
recorded parity steps produced a `candidate_logits_sha256` **equal to the
reference record's for the same step**, and every step's cache report is equal
too, while the records' `graph_digests` prove different bytes were loaded. That
is the predicted outcome of a semantics-preserving catalogue — none of the six
passes touches float arithmetic or its order — and the prediction was made
before the measurement.

A seventh transformation is recorded as **rejected**, with evidence the build
tool re-measures on every run rather than citing a planning probe.

Everything is evidence about *graphs*. **No compiler, no converter, no QAIRT
tool, no device, and no Qualcomm AI Hub job was involved at any point.**

**This work is not merged.** Merging was not authorized and was not done, so
`T22` is `in_progress`. See "Task status" below.

## Changes

On `task/T22-qnn-candidates`, in this working tree:

- `configs/graph/qnn-transforms-v1.json` — catalogue `qnn-candidate-v1`, six
  applied passes and one recorded rejection. Every pass names the T21 rule ids
  it addresses, the numbered `graph-inspection.md` finding that motivated it,
  the exact rewrite, and its declarative parameters.
- `src/slm_lab/graph/qnn/` — the transform engine and the `build` CLI. Reads a
  committed T20 manifest, re-hashes both the protobuf and its sidecar before
  parsing anything, applies the catalogue in order, writes the candidate and
  its own sidecar, runs `onnx.checker`, asserts two fatal post-conditions on
  the bytes it wrote, re-scores both sides with the committed T21 rule engine,
  and measures the rejected pass into a scratch directory it then deletes.
- `results/manifests/qnn/` — four candidate manifests, four full before/after
  inspection reports, four candidate parity records, four package records, and
  three READMEs. No timestamp in the manifests or inspection reports, so
  `--check` is a genuine drift check.
- `src/slm_lab/deployment/qualcomm/packaging.py` and
  `scripts/qualcomm/package_qnn_candidate.py` — the package builder and its
  `--check` re-verifier.
- `src/slm_lab/deployment/qualcomm/ai_hub.py` — `preflight_compile_request`,
  which runs the committed T30 compile validation chain against a generated
  request and returns the same `request_id` `run_compile` would record. It
  builds no backend, imports no client, and contacts no service.
- `configs/targets/qualcomm-snapdragon-x-elite-crd.json` — the compile target
  selector, every field copied from committed T02 evidence.
- `src/slm_lab/backends/onnx_cpu.py` — one generalization: the runner resolves
  its graph directory from the manifest's `artifacts.root` instead of a
  hard-coded `onnx/reference/T20`. For every committed T20 manifest that
  expression already resolves to the same path, so no committed T21 or T23
  evidence moved.
- `tests/qnn/`, `tests/deployment/qualcomm/test_packaging.py`, and additions to
  `tests/deployment/qualcomm/test_ai_hub.py` and
  `tests/onnx/test_onnx_cpu_parity.py`.
- `docs/results/qualcomm/qnn-candidates.md` — the reader-facing report.
- `configs/graph/README.md`, `scripts/qualcomm/README.md`,
  `src/slm_lab/deployment/README.md` — documentation for the three new
  surfaces.
- `results/graph/README.md` — one accuracy fix. It said `record_kind` "only
  reads `t21_ort_cpu_parity` for the four files above"; four T22 candidate
  records now carry the same kind, so the sentence named `graph_digests` as
  what actually separates them.
- `ai/tasks/task_graph.yaml`, `ai/tasks/learning_lane.yaml`,
  `ai/tasks/status.generated.md`, `docs/dashboard/index.html` — the public T22
  claim, a learning-lane rebuild for the two READMEs this task edited, and the
  dashboard. The dashboard needed a prose edit as well as a regeneration: its
  builder refuses a task badge inside `<section id="next">` whose effective
  status is not `ready`, so the T22 card was rewritten to say the task is in
  flight and that the next action is a merge.
- `ai/plans/active/T22-qnn-candidates.md` — milestones ticked, three
  discoveries added, and the header and restart instructions rewritten to
  describe the unmerged state and the exact next action. The plan stays active;
  see "Task status".
- `ai/handoffs/T22-qnn-candidates.md` — what T31 consumes, what T22 did not
  establish, the four-step close-out sequence, and the ready-to-paste
  `LEARN-12` entry.

## Verification

Run in this worktree with the locked root interpreter unless stated otherwise.

- Command: `PYTHONPATH=src .venv/bin/python -m pytest tests -q`
  Result: **773 passed, 31 skipped, 0 failed** in 50 s, after the review
  corrections below added two cases (the `onnx`-dependent one skips here and
  passes in the parity environment, where `pytest tests/qnn` is 69 passed).
  Two rounds of failures
  cleared along the way, and both were generated files rather than code. Three
  `tests/repo/test_task_automation.py` cases assert that a staged
  `ai/tasks/status.generated.md` matches the staged task graph; they cleared
  when the generated status was rebuilt. Three `tests/repo/test_dashboard.py`
  cases then fired for the same reason on `docs/dashboard/index.html`, and one
  of them was a genuine prose-drift error rather than a stale render — the
  dashboard listed T22 as ready.
- Command: `python3 scripts/ai/render_task_status.py --check`
  Result: pass — task graph valid; 31 tasks; 12 learning checkpoints;
  generated status current.
- Command: `python3 scripts/repo/check_hygiene.py --all`
  Result: pass — 357 tracked and untracked public files.
- Command: `python3 scripts/dashboard/build_dashboard.py --check`
  Result: pass — generated regions current and prose matches the graph.
- Command: `python3 scripts/audit/audit_reference_graph_claims.py citations`
  Result: **0 disagreements**, 1,020 measured facts, 6 documents bound. Note
  what that does and does not cover, below.
- Command: `python3 scripts/learning/build_learning_sheet.py --all --record`
  Result: 12 sheets rebuilt and recorded. This cleared the stale `LEARN-06` and
  `LEARN-10` sheets, whose pinned digests of `scripts/qualcomm/README.md` and
  `configs/graph/README.md` moved when this task documented the two new
  surfaces.
- Command: `python -m slm_lab.graph.qnn.build --all-manifests --check` in the
  parity environment, with `SLM_LAB_ARTIFACT_ROOT` set
  Result: exit 0 for all four variants. Every committed manifest and inspection
  report re-derives from reference bytes whose SHA-256 matches the T20
  manifest, down to the candidate digests.
- Command: `python -m slm_lab.backends.onnx_cpu --manifest
  results/manifests/qnn/S<N>.json --steps 4 --reference torch` in the parity
  environment, once per variant
  Result: all four `passed: true` with `failures: []` and `failure_kinds: []`,
  at `evidence_tier="real_onnxruntime_cpu"`.

Not run, and named rather than implied:

- No vendor toolchain, converter, compiler, or device. None is installed.
- No AI Hub call of any kind. T22 holds `t9_heavy_io` only; `qai_hub_submission`
  belongs to T31.
- No `ORT_ENABLE_ALL` run on either the reference or the candidate, so the
  fusion delta remains unmeasured on both sides.
- The new report is **not** covered by
  `scripts/audit/audit_reference_graph_claims.py`. Its `CLAIM_DOCUMENTS` tuple
  is a fixed list and `docs/results/qualcomm/qnn-candidates.md` is not in it, so
  the audit neither binds nor enumerates a single number in that document.
  Adding it is a source change this task did not take; see "Risks".

## Decisions and evidence

**ONNX Runtime offline optimization was rejected on four measured grounds, not
three asserted ones.** Writing a session's `optimized_model_filepath` at
`ORT_ENABLE_BASIC` is the obvious way to fold these graphs, and it works: node
count falls to 3,040 in prefill and 4,245 in decode, the latter beating this
catalogue's 5,421 by 22% of the decode graph. Rejecting it costs something
real. The build tool therefore re-measures it on every graph it builds rather
than citing the planning probe, and records the result in each manifest's
`rejection_evidence`:

1. All 28 `Softmax` nodes decompose into `Sub`/`Exp`/`ReduceMax`/`ReduceSum`/
   `Div` — one operator an NPU implements natively replaced by five, on exactly
   the operator `graph-inspection.md` 5.6 ranks as precision-sensitive.
2. `Cast` rises 348 → 830 in prefill and 464 → 860 in decode, which is the CPU
   execution provider's float16 fallback inserting casts around kernels it
   lacks. A host-specific artifact in a candidate aimed at a different target.
3. Eight opset domains are added — including `com.microsoft`,
   `com.microsoft.nchwc` and `org.pytorch.aten` — and the tool checks each
   against the node list: **all eight are used by no node.** That is a pure
   regression in the single-standard-domain portability `graph-inspection.md`
   section 6 records as one of three clean blocking results.
4. The external-data layout is destroyed. External initializers fall 254 → 58,
   no sidecar is written, and inline initializer bytes rise from 14,336 to
   between 1,191,990,214 and 1,810,643,510. **The S4096 prefill output is a
   single 1,811,439,962-byte protobuf — 1.69 GiB, 84% of protobuf's 2 GiB
   serialization ceiling.**

Only the first three were in the plan. The fourth appeared when the tooling
measured what the plan had asserted, and it is the one that turns a design
preference into an engineering constraint.

**The catalogue's own rejection rationale still carried the falsified
three-ground claim, and correcting it forced a full rebuild.** Review found
that while the manifests, the report and this log all record four measured
grounds, `configs/graph/qnn-transforms-v1.json` still opened with "Three
measured effects" and closed with "The protobuf shrinks and the sidecar grows"
— the opposite of what every `rejection_evidence` block measures. The evidence
records the protobuf *growing* on all eight graphs (S128 prefill 5,131,850 →
1,197,039,898 bytes, S4096 prefill 49,790,614 → 1,811,439,962) with
`external_data_files: []` and `external_data_bytes.after: 0`, so no sidecar is
written beside the optimized model at all. That planning-probe sentence was the
last place in the repository still carrying it. The rationale now names four
grounds and defers every byte count to `rejection_evidence` instead of
restating one, which is what stops the same class of drift recurring.

**The correction was not free, and its cost is the interesting part.**
`X-STAMP-CANDIDATE-PROVENANCE` writes `slm_lab.transform_catalogue_sha256` into
every candidate, so editing one string in a committed config changed all eight
candidate `.onnx` digests. The catalogue SHA-256 moved from
`64b266971b5e…` to `0f6ffb3a3647…`, and the whole cascade had to be carried
through: `build --all-manifests`, a fresh T21 parity run on all four variants,
a second `build --all-manifests` so each manifest re-derived
`verification.ort_cpu_parity` from records measuring the new bytes, then
`package_qnn_candidate.py` for all four. Left half-applied, the parity records
would have gone `stale_record` and `claim_boundary` would have silently stopped
claiming parity. Both `--check` modes exit 0 afterwards.

**The re-measured parity is a genuine re-measurement, and it came back
identical.** All four records are `passed: true` with `failures: []` and
`failure_kinds: []` at `evidence_tier="real_onnxruntime_cpu"`; all 20 steps are
still bit-identical to the committed T20 reference records, 1,120 cache tensor
checks still show zero violations, and every metric in the report's section-8
table is unchanged to the digit. That is the predicted result — the catalogue
digest is metadata, not arithmetic — but it was measured rather than assumed.
The candidate sidecar digests and all eight protobuf byte sizes did not move
either, because the stamped digest is a fixed-width `metadata_props` value
inside the protobuf and never reaches the sidecar. What did move: eight
candidate `.onnx` digests, four manifest digests, and therefore the eight
deterministic compile-request ids in the package records.

**Four smaller accuracy fixes rode along in the same rebuild.**

- `dead_node_elimination` now calls `assert_topological_order`, which
  `static_shape_fold` already did. Its backward liveness sweep depends on the
  same ordering, and as a public helper it would otherwise silently drop a live
  producer that appeared after its consumer. Covered by a new case in
  `tests/qnn/test_qnn_transforms.py`.
- `tests/qnn/test_qnn_build.py` now asserts from the committed manifests alone
  that every candidate `.onnx` digest differs from the T20 reference digest it
  was derived from, and that each candidate cites the reference digest it read.
  That is T22's first acceptance criterion and it had no committed-file
  assertion. Sidecars are deliberately excluded: the four decode sidecars are
  byte-identical to the reference sidecars by measurement, not by collision.
- The execution plan's transformation profile attributed weight safety to the
  allowlist and the byte budget. The load-bearing guard is neither on its own:
  an initializer in external data is not in the constant pool at all, and a
  sub-1 MiB inline float tensor would be foldable under the budget alone. The
  catalogue and report section 4.2 already stated it correctly; the plan now
  matches them.
- `results/manifests/qnn/README.md` said "Both files are produced by
  `slm_lab.graph.qnn.build`" directly after listing three file kinds, one of
  which the T21 runner produces — as the same README says two paragraphs later.

**One number did not reproduce the planning probe, and the cause is recorded as
unknown.** The probe recorded the S128 prefill optimized output as an
828,930-byte protobuf plus a 1,507,512,320-byte sidecar. This tooling measures
1,197,039,898 bytes inline and no sidecar for the same graph on onnxruntime
1.28.0. The two runs must have differed in session configuration — this one
sets only `graph_optimization_level` and `optimized_model_filepath` — but the
repository holds no record of the probe's, so the cause is **not stated as
known**. Everything else the probe reported reproduced exactly. The committed
numbers are the tool's.

**Candidate parity reuses the T21 runner rather than a second implementation.**
That required generalizing `slm_lab.backends.onnx_cpu` to resolve its artifact
directory from the manifest. The alternative — a second parity implementation
for the candidate stage — would have produced numbers with no defensible
relationship to the reference numbers, and the bit-identity result would have
been unavailable: it is only meaningful because both sides ran the same
protocol, the same frozen T10 workload, the same bfloat16 reference at the same
revision, the same four decode steps, and the same `DEFAULT_ORT_CPU_TOLERANCE`.
Verified that no committed reference evidence moved.

**Bit-identical logits were predicted, and the prediction could have failed.**
None of the six applied passes touches float arithmetic or its order, and at
`ORT_DISABLE_ALL` the runtime adds no fusion of its own on either side, so both
runs execute the same float operations in the same order over the same weights.
Had any metric moved, the honest reading would have been that one of the six
passes is not semantics-preserving. That is what the measurement was for. It is
also the sharpest statement of why the rejected pass was never a candidate: a
pass that decomposes 28 `Softmax` nodes and inserts 482 casts cannot produce
bit-identical logits, so comparison alone could no longer separate a legitimate
rewrite from a defective one.

**The folder is allowlisted and byte-budgeted, and both guards were load-
bearing.** The allowlist admits 33 shape, index, and small elementwise
operators — no `MatMul`, no `Softmax`, no normalization — so no attention
arithmetic is ever evaluated at build time. The 1 MiB budget caps both consumed
and produced tensors. What keeps the *weights* out is neither of those: an
initializer that lives in external data is not in the constant pool at all. Their measured effect is visible in
the residue: S128 prefill reaches 0 residual shape-defining inputs while S512,
S1024 and S4096 stop at 5, 6 and 6, because the causal mask is 32,768 bytes at
S128 but 524,288 to 33,554,432 above it, and a handful of mask-consuming nodes
are refused rather than evaluated. The alternative is a folder that
materializes a 33 MB tensor at build time and writes it into the protobuf,
which is the thing pass 4 exists to prevent.

**Three findings surfaced that were not the expected answer**, and all three
are in the report rather than smoothed over:

- `R-INTERNAL-DYNAMIC-SHAPE` went from silent to loud. `graph-inspection.md`
  section 7 named shape inference as the follow-up that would turn its 0-of-0
  silence into a real answer. It did, and the answer is **not** the clean one
  the static public boundary made "likely": 9 non-static interior tensors in
  prefill at S512 and above, and **1,069** in decode, out of 5,363 annotated.
  Those are not new defects; they are the first measurement of something the
  reference graphs never declared. They are also a statement about what
  opset-18 inference could resolve, not a proof that any dimension varies.
- Decode's rank-1 count falls only to 423 because it builds its attention mask
  at run time from `Range`/`Greater` arithmetic over activations, so 429 of its
  459 `Shape` nodes read tensors that are not constants even though their
  *shapes* are static. A shape-aware second fold using the inferred `value_info`
  is the obvious follow-up; it is not in this catalogue and no number assumes
  it. Note that its likely yield is bounded by the finding above.
- `X-DEAD-NODE-ELIMINATION` removed **zero nodes** on all eight graphs, because
  the folder already deletes what it replaces and these exports contain no
  unreachable computation. Neither fact was known before the pass ran. Its
  initializer arm did the work — 3,335 to 3,399 dead initializers per prefill
  graph, 1,629 per decode graph — and the pass records the two arms separately
  so the distinction is legible in the manifest rather than reconstructible.

**"Ready for Workbench submission" was given an exact, narrow meaning.** The
package record carries it in-band, in `submission_status`: the candidate and
sidecar digests were re-verified against the committed T22 manifest, a compile
request was generated into private storage, and that request was accepted by
the committed T30 adapter's own validation. `job_submitted`,
`service_contacted` and `package_layout_verified_against_service` are all
`false`. The package layout for an external-data ONNX model is unverified
against the service, and the compile request names only the `.onnx` file
because the T30 adapter requires `source_artifact.path` to be one existing
file — whether the service reads the sidecar from the same directory is
recorded as unverified on every graph record.

## Risks and limitations

- **1,069 unresolved interior shapes in decode.** The T21 evidence boundary is
  closed and it closed unfavourably. Whether those dimensions are provably
  constant under a stronger analysis is unknown, and a static-shape
  ahead-of-time compiler is exactly the consumer that will care.
- **The parity result is narrow.** One frozen workload, four variants, 20
  steps, one host, the CPU execution provider, onnxruntime 1.28.0,
  `ORT_DISABLE_ALL`. Bit-identity says the graph still computes the same
  function on that provider; it says nothing about a converter, and the T21
  protocol's own boundary carries over — the newly written cache slot is
  checked for having been written and being finite, not for holding the right
  values.
- **The fusion delta remains unmeasured on both stages.** Inherited from T23
  and not resolved here.
- **Nothing in this task was validated against a QNN toolchain.** The stage
  name `qnn_candidate` is an intent, not a verdict. Every severity in the risk
  catalogue is still a reviewed structural judgement.
- **The new report is outside the citation audit.**
  `scripts/audit/audit_reference_graph_claims.py` binds a fixed
  `CLAIM_DOCUMENTS` tuple, and `docs/results/qualcomm/qnn-candidates.md` is not
  in it. Every number in that report was read from the committed manifests by
  hand and cross-checked against `results/graph/parity/` for the parity
  columns, but it has no automated guard against drift. Adding it to the tuple
  is the obvious follow-up and was deliberately not taken here, because it is a
  source change to a tool this task does not own and would need its own
  regression coverage.
- **`X-STATIC-SHAPE-FOLD` folds by evaluating with
  `onnx.reference.ReferenceEvaluator`.** The folded values are therefore
  produced by the ONNX reference implementation of each operator, on this host,
  at `onnx` 1.18.0. The allowlist is integer and small-elementwise only and the
  parity result is bit-identical, which is strong evidence that no value moved,
  but the mechanism is an evaluation and not a proof.
- **The 1 MiB fold budget is a chosen number.** It is justified by what it
  refuses at S1024 and S4096, not derived from any target's constraint.
- **The candidate tree costs about 9.0 GB** on top of the reference tree's 9.0
  GB, on the external artifact root.

## Task status

`T22` is **`in_progress`**, not `completed`, and that is the honest call rather
than a formality.

AGENTS.md defines `completed` as including "changes are integrated into the
branch downstream tasks will use". This work lives on the unmerged task branch
`task/T22-qnn-candidates`. Merging was not authorized and was not performed.
The task graph's `allowed_statuses` are `planned`, `in_progress`, `blocked` and
`completed`, with no state meaning "finished on its branch, awaiting merge" —
the same situation `T23` documented for itself before its merge on 2026-08-03.

**`T31` is blocked on that integration, not on any missing T22 work.** T31
consumes the candidate manifests, the package records, and the target selector,
and it must start from a commit that contains them.

Two repository invariants prevent the close-out from being completed while the
status is `in_progress`, and both are deliberate:

- **The `worklog` field is `null`.** `scripts/ai/render_task_status.py` raises
  `only completed tasks may set the worklog field`. This log therefore exists
  and is committed, but the graph does not reference it yet.
- **No learning checkpoint was added.** `configs/learning/checkpoints.yaml`
  states the rule in its own header — "cover only tasks whose task-graph status
  is `completed`" — and `scripts/learning/build_learning_sheet.py` enforces it,
  failing with `LEARN-12 cites T22, whose status is 'in_progress'; checkpoints
  cover completed work only`. That was verified by adding the entry and running
  the build, not inferred from the code.

The complete, ready-to-paste `LEARN-12` entry is in
`ai/handoffs/T22-qnn-candidates.md`, together with the exact sequence the
integrator must run. `docs/project/learning-checkpoints.md` marks T22 as a
deep-study checkpoint, and section 12 of the report is its self-check.

The execution plan stays in `ai/plans/active/`, per
`ai/plans/completed/README.md`: "Move an execution plan here only after its task
is integrated and the task graph marks it completed with a matching public
worklog." Its header and its "Progress and restart instructions" section were
updated to describe the real state and the exact next action.

## Follow-up

- Newly unblocked tasks: **none yet**. `T31` unblocks when
  `task/T22-qnn-candidates` is merged and `T22` flips to `completed`.
- Recommended next action: integrate the branch, then run the four-step
  close-out in `ai/handoffs/T22-qnn-candidates.md` — flip the status, set the
  `worklog` field and this log's `Status:` line in the same edit, paste the
  `LEARN-12` entry, and rebuild the lane and the generated status.
- Deferred, for whoever next owns the transform catalogue: a **shape-aware
  second fold** that runs after `X-INFER-VALUE-INFO` and folds `Shape` nodes
  whose operand shape is declared even though the operand itself is not
  constant. That is the pass that would move decode's 423.
- Deferred, for whoever next owns the audit tool: add
  `docs/results/qualcomm/qnn-candidates.md` to `CLAIM_DOCUMENTS` with a
  regression, so its numbers are bound the way the T21 reports' are.
- Deferred, still: the paired `ORT_ENABLE_ALL` run on both stages, and a
  `total-inline-bytes` companion to `R-LARGE-INLINE-CONSTANT` — both carried
  forward unchanged from T23.
