# T41: W8 Candidate Evidence Boundary

Date: 2026-08-03
Task: `T41`
Visibility: `public`
Status: blocked

## Outcome

T41 froze two regenerable W8 candidate specifications — W8A16 (plan matrix row
Q1) and W8A8 (row Q2) — and the fail-closed evidence framework that decides
what will and will not count as a W8 result. It then stopped at the submission
boundary. **No weight was quantized, no simulation was run, no job was
submitted, and no measurement exists.** Every number in this task's outputs is
arithmetic over committed repository inputs, a hash read off a committed file,
or a count of files, tests, or graph nodes.

Three external blockers stand between this commit and a real W8 result. AIMET is
Linux + CUDA only and remains specified rather than installed. Submission
authorization was granted during this task and did not make the criterion
reachable, because there is still no W8 artifact to submit and the only
repository component that could request a hosted one belongs to T22. And no
Qwen graph of any precision has yet traversed the public Workbench pipeline,
because T31 and T33 are still `planned`. The task-definition acceptance
criterion "at least one W8 candidate traverses the public pipeline" is
therefore not satisfied, and T41 ends at stored status `blocked` rather than
`completed`.

What is delivered is the durable half: a frozen, hash-checked candidate
specification that regenerates byte-identically offline, a precision-state
function that composes evidence into a state rather than accepting a claimed
label, a quality-delta comparison bound to the frozen T13 protocol, and a
stage-request emitter that stops exactly one step short of submission.

## External-service authorization and use

Recorded as an event, per the `AGENTS.md` external-services rules.

**Authorization.** On 2026-08-03 the user authorized Qualcomm AI Hub submission
for T41: hosted compile, profile, and inference jobs, plus Device Cloud
interactive minutes capped at **120**, on **free capacity only**, with **no
spend** permitted without a fresh decision, and with pushing and public GitHub
state still withheld.

**What was done with it.** One read-only capability query against the live
service, with client `qai-hub==0.53.0`, on 2026-08-03. It called exactly one
service function, `get_devices`, which returns public device metadata; the
`submit_*` entry-point names came from `dir()` and the quantize signature from
`inspect.getattr_static`, so no submission function was called or even fetched
through the descriptor protocol. It submitted no job, uploaded no model, leased
no device, ran no inference, and downloaded no artifact. This repository pins
no Qualcomm client, so the query ran in an environment that carried one and the
sanitized result was carried in.

**What it consumed.** 0 jobs. 0 of the 120 Device Cloud minutes. US$0.00. The
ceiling is **fully unspent** and passes to whoever works T41 next.

**What it established.** Three things, all about the service on that date:

- The client exposes `submit_quantize_job(model, calibration_data,
  weights_dtype, activations_dtype, name, options, project)`, and
  `QuantizeDtype` carries `INT4`, `INT8`, and `INT16`. Both T41 candidates are
  therefore expressible as public quantize requests — `w8a16` as INT8 weights
  with INT16 activations, `w8a8` as INT8 with INT8 — and `INT4` exists for T42.
- 79 devices are live, including all three plan section 3.2 targets:
  `Snapdragon X Elite CRD` (os 11, hexagon v73), `Snapdragon 8 Elite QRD`
  (os 15, hexagon v79), and `Dragonwing IQ-9075 EVK` (os 1.7, hexagon v73),
  each advertising `framework:qnn` and `htp-supports-fp16:true`.
- The quantize API exposes no separate KV-cache dtype knob, which is consistent
  with the frozen `CACHE_DTYPE = "float16"` finding below and is not an escape
  from it.

**What it did not establish, and why no job followed.** It says nothing about
this model: not compiler acceptance, not operator support, not NPU placement,
not 16-bit activation datapath support, not latency, not memory, not accuracy.
**No job was submitted and no W8 candidate traversed the public pipeline.** A
capability query is not a pipeline traversal.

No job followed for two independent reasons, neither of them permission. First,
there is nothing to submit: no W8 artifact exists, Lane B cannot produce one
without the AIMET host, and Lane A would need a hosted quantize job that this
repository cannot request — `slm_lab.deployment.qualcomm.ai_hub` declares
`STAGES = {"compile", "inference", "profile"}` and lives under
`src/slm_lab/deployment/qualcomm/`, an owned path of T22
(`ai/tasks/definitions/T22.yaml`) worked concurrently on
`task/T22-qnn-candidates`, so T41 must not add a stage to it. Second, even with
an artifact, a submission now would not establish what the acceptance criterion
wants: no floating Qwen graph has ever been accepted by the Workbench compiler,
so a quantize or compile failure could not be attributed to the W8 policy
rather than to the graph.

The record is `results/quantization/t41-ai-hub-capability-2026-08-03.json`. The
live query is `uv run python -m slm_lab.quantization.w8 capabilities`; the
committed record replays offline with the same subcommand and
`--offline-input`, and `check` recomputes it on every run and fails if the two
disagree.

## Changes

- `src/slm_lab/quantization/w8.py` — the whole T41 implementation:
  specification generation and validation, the analytic weight-storage
  projection and its cross-check, the precision-state composition, the
  frozen-protocol quality comparison, and the AI Hub stage-request emitter.
  Dependency-free beyond the standard library and PyYAML, so it runs on a host
  with no quantization stack.
- `configs/quantization/w8/w8a16.yaml` and `configs/quantization/w8/w8a8.yaml`
  — the two generated candidate specifications, each a byte-identical fixed
  point of its generator and bound by hash to the T40 calibration revision, the
  committed float16 baseline manifests, the frozen T13 protocol digest, and the
  T21 graph inventory.
- `configs/quantization/w8/README.md` — how the directory is regenerated, what
  each candidate claims, and what it does not.
- `tests/quantization/test_w8_candidates.py` and
  `tests/quantization/test_w8_evidence.py` — 132 offline tests, including
  mutation tests that prove the refusals rather than asserting them.
- `results/quantization/t41-w8-readiness-2026-08-03.json` — the committed
  readiness record. Both candidates read `precision_state: specified`; the plan
  7.3 ledger reads 1 `satisfied`, 3 `not_run`, 6 `blocked`.
- `docs/results/quantization/w8.md` — the reader-facing report, which opens by
  stating that it contains no W8 measurement.
- `results/quantization/t41-ai-hub-capability-2026-08-03.json` — what the
  read-only capability query observed: the quantize API surface, the dtype
  members, and the live device fleet. No job, no minutes, no cost.
- `ai/handoffs/T41-w8-submission-boundary.md` — the operational handoff: what
  exists, what to run in what order, what the standing authorization covers and
  what still needs a decision, and what would invalidate the frozen candidates.
- New sections in `configs/quantization/README.md`,
  `src/slm_lab/quantization/README.md`, and `results/quantization/README.md`.
- `w8` entries in `src/slm_lab/quantization/__init__.py`. Four `w8` names
  (`build_document`, `generate_repository`, `load_inputs`,
  `validate_repository`) collide with `calibration`'s and are deliberately not
  exported, because the export map is flat and one name may only mean one
  module.
- `ai/plans/active/T41-w8-quantization-evidence.md` — corrections recorded under
  "Decisions and evidence" below, plus the revised blocker state.
- `ai/tasks/task_graph.yaml` — the T41 entry moved to `status: blocked` with
  this task's owner and branch, and `worklog: null`. See "Risks and
  limitations" for why the field cannot carry the worklog path yet.

## Verification

All commands were run from the worktree root on the primary macOS host.

- Command: `uv run --extra dev python -m pytest tests/quantization -q`
  Result: 225 passed, 1 skipped in 5.47 s. The skip is T40's opt-in 9.6 GB
  artifact re-hash.
- Command: `uv run --extra dev python -m pytest tests -q`
  Result: 787 passed, 15 skipped in 69.19 s.
- Command: `uv run --extra dev python -m pytest
  tests/quantization/test_w8_candidates.py tests/quantization/test_w8_evidence.py -q`
  Result: 132 passed in 4.09 s — this task's tests in isolation.
- Command: `uv run python -m slm_lab.quantization.w8 check`
  Result: exit 0, "T41 W8 candidate check passed: <checkout path>", where the
  trailing field is this machine's absolute checkout path and is not reproduced
  here. The two committed specifications are byte-identical fixed points of
  their generator.
- Command: `uv run python -m slm_lab.quantization.w8 status`
  Result: both candidates `precision_state=specified` with scope
  `candidate_specification_only_no_weight_was_quantized`, 10 unsatisfied
  evidence checks each, evidence `absent_at_this_commit`; ledger 1 satisfied,
  3 not_run, 6 blocked; lane A and lane B both unavailable.
- Command: `uv run python scripts/ai/render_task_status.py --check`
  Result: "task graph valid; 31 tasks; 12 learning checkpoints; generated
  status is current".
- Command: `uv run python scripts/repo/check_hygiene.py --all`
  Result: "repository hygiene passed for 335 tracked and untracked public
  files".
- Command: `uv run --extra dev ruff format --check` and `ruff check` on
  `src/slm_lab/quantization/w8.py`,
  `src/slm_lab/quantization/__init__.py`, and the two new test modules
  Result: "4 files already formatted"; "All checks passed!".

Not run, and not partially run:

- AIMET. It was never installed and never executed, on this host or any other.
- Any fake-quant simulation, calibration pass, or encoding computation.
- Any Qualcomm AI Hub compile, inference, profile, or quantize submission. No
  request file was written to a committable path, and a test pins that the
  stage-request emitter leaves `sys.modules` free of `qai_hub`. The one
  live-service action taken in this task was the read-only capability query
  described above, which submits nothing.
- Any device measurement, hosted or real. Zero Device Cloud minutes were
  consumed against the authorized 120-minute ceiling.
- Anything requiring `torch`, `onnx`, `onnxruntime`, `numpy`, or
  `transformers`. Confirmed absent from the environment at the end of the task
  and deliberately not installed, exactly as at the end of T40.

## Decisions and evidence

- **Precision state is a three-value ladder computed from a digest chain, not
  from a claimed label.** `assess_precision_state` reads the evidence
  positively: it looks for a simulation record and three chained stage
  manifests, and never consults a `state`, `precision_state`, `verdict`, or
  `deployed` key planted in its input, so a record cannot assert its own
  conclusion. `deployed` requires compile, inference, and profile manifests
  that are schema-v2, declare the stage they are filed under, report success,
  and chain: the compile manifest's source digest equal to the simulation's
  quantized-artifact digest, the inference and profile `compiled_model` digests
  equal to the compile manifest's target-artifact digest, and the
  predecessor-manifest digest recomputed from the manifest actually supplied
  rather than trusted as declared.
- **Unlike T40's parity verdict, `deployed` is reachable from data.**
  `parity.overall_verdict` has no branch that returns `verified`, so promoting
  it needs a code change. `w8.assess_precision_state` deliberately does not
  copy that property: `test_a_complete_and_chained_evidence_set_reads_deployed`
  proves it says yes to a complete, correctly chained set, so that the refusal
  tests around it are refusals rather than a function that cannot say yes at
  all. `deployed` is unreachable today because no such manifest set
  exists anywhere in this repository, not because the function declines to emit
  it. The guarantee is the chain, not an unreachable branch. The execution
  plan's step 7 asserted the opposite — that promotion "is a code change to the
  verdict function ... not a data change" — which imported T40's property into
  a task that does not have it and contradicted both the shipped code and its
  tests. Step 7 was corrected to describe the digest chain and to say that
  promotion is a data change backed by auditable manifests. Three further
  sentences carried the same imported claim and were corrected in the same
  direction: the scope bullet that said "no branch may assign `deployed` ...
  mirroring the precedent T40 set", the acceptance criterion that said "the
  verdict function has no input that yields `deployed`", and the decision entry
  that attributed the unreachability to "T40's parity precedent". `deployed` is
  unreachable because this repository holds no chained manifest set, not
  because the function declines to emit it, and the difference from T40 is
  deliberate.
- **W8A8 cannot satisfy plan matrix row Q2 without a contract change.** Q2 asks
  for an INT8 cache. `slm_lab.contracts.static_cache.CACHE_DTYPE` is frozen at
  `float16`, and that dtype is the declared dtype of 56 cache inputs and 56
  cache outputs on the exported graph boundaries, asserted by the committed
  manifests under `results/manifests/onnx/`. Lowering it would change a public
  T12 contract that T41 does not own, and would invalidate the T20 exports and
  the T23 promoted prefill export. The candidate therefore specifies float16,
  records `satisfied_without_contract_change: false`, and files the INT8-cache
  variant as an out-of-scope change request against T12, T20, and T23 rather
  than quietly specifying a cache precision the exported graphs do not accept.
- **The weight-storage projection is arithmetic and is labelled
  `analytic_projection` in its own field names.** 596,049,920 parameters are
  derived from `configs/models/qwen3-0.6b.yaml` — layer count, both feature
  widths, both head counts, head dimension, vocabulary, and the tied-embedding
  flag — with nothing hardcoded from the published model size. It is
  independently cross-checked against the committed artifact: the T20 external
  data is 1,192,085,504 bytes, which is 596,042,752 float16 elements, and the
  56 initializers the exporter leaves inline (310 total minus 254 external,
  exactly `2 x num_hidden_layers`) carry 128 elements each for 7,168 more,
  reproducing 596,049,920 exactly. Two independent committed sources have to
  agree and the projection fails closed if they stop agreeing. This cross-check
  corrected an initial figure of 596,107,264 that the task brief carried; that
  wrong number appears nowhere in the repository.
- **Tied embeddings mean one weight set, not two.**
  `tie_word_embeddings: true` in the model contract makes the embedding table
  and the output logits projection the same parameters, so they cannot carry
  two different precisions without untying the model. The projection counts the
  vocabulary table once and refuses to run against an untied contract until a
  second weight class is added.
- **The precision declaration rides on `source.precision`.** The frozen T13
  result schema is `additionalProperties: false` at the record root and inside
  `source`, so a W8 result has nowhere to add a bespoke `precision_state`
  field. The state is therefore carried as
  `"<candidate_id>+<simulated|deployed>"` on `source.precision`, which the
  schema already requires, and is cross-checked against
  `system.evidence_level`, which the schema already constrains. Consequently
  `compare_quality` derives `comparison_scope` from the candidate record itself
  and has no caller argument that could label a simulated comparison as
  deployed; a test pins the function's parameter set to exactly
  `{baseline_record, candidate_record, root}`.
- **The calibration corpus stays frozen at the T40 revision for both
  candidates.** The T40 worklog explicitly left widening the corpus with its
  declared tier-2 material to T41 as a recommended follow-up. T41 declines it
  on purpose: changing the corpus and introducing the first quantization policy
  in the same task would leave any future W8-versus-float delta unattributable
  to either change. The corpus is widened later, as its own change, against a
  W8 delta that already exists.

## Risks and limitations

- **Every future W8 quality delta inherits T40's calibration bias.** The corpus
  is 6,912 tokens, 83% of the token budget comes from one repeated seed, and
  four tier-1 samples are both calibrated on and evaluated on. The bias is
  optimistic: the range observers are fitted on prompts the delta is measured
  on, so any measured degradation is a lower bound on what an unseen prompt
  would show. `compare_quality` attaches this statement, read from the T40
  contract rather than restated, to every comparison it emits, so a consumer
  that reads only the numbers still receives the reason they are optimistic.
- **Lane A and Lane B would not produce the same artifact.**
  `docs/results/access/2026-07-24-public-platform-access.md` records the
  Workbench Quantize Job at AIMET `2.34`, while `environments/linux-aimet`
  pins `aimet-onnx==2.36.0` and `aimet-torch==2.36.0`. A hosted quantization
  and a local one are therefore two different quantizers. Encoding differences
  between the two lanes must be read as differences between tool versions, not
  as properties of the model.
- **Lane A cannot itself produce the quantized artifact at this commit, and
  this is now the binding blocker rather than authorization.**
  `slm_lab.deployment.qualcomm.ai_hub` declares `STAGES = {"compile",
  "inference", "profile"}` and has no quantize stage, while the capability
  query showed the service does accept quantize jobs for both candidates' dtype
  pairs. So the gap is a missing adapter, recorded as
  `capability:no_quantize_stage_adapter_in_this_repository`. It is not T41's to
  close: the module sits under `src/slm_lab/deployment/qualcomm/`, an owned
  path of T22 (`ai/tasks/definitions/T22.yaml`), worked concurrently on
  `task/T22-qnn-candidates`. Closing it is a cross-task decision with T22.
- **The task graph cannot reference this worklog while T41 is not completed.**
  `scripts/ai/render_task_status.py` rejects a `worklog` field on any task that
  is not `completed`, so the T41 entry carries `worklog: null` and the link to
  this file is by filename from the execution plan and the report until T41
  completes. The validator is a shared repository invariant and was not
  weakened, and T41's status was not falsified to satisfy it.
- **The readiness record was generated on a dirty working tree.** It records
  `git_commit: 0085a2c5ee01ef9fa1360ac130c88f574a68c3aa` and
  `git_tree_clean: false`, and states both rather than hiding them, exactly as
  T40's parity record did. Its inputs are committed files and committed
  manifests, so the record stands; a post-commit re-run would flip the flag.
- **Nothing in this repository has yet traversed the public Workbench
  pipeline, floating or quantized.** Every statement about compiler acceptance,
  operator support, NPU placement, 16-bit activation datapath support, latency,
  or memory for either candidate is unverified. The capability query does not
  change this: it read an API surface and a device list, and
  `htp-supports-fp16:true` is a device advertisement, not a compile result. The
  candidates say so in their own `boundary` fields rather than leaving a reader
  to infer it.

## Follow-up

Three blockers remain, they clear independently, and they clear in a specific
order. Naming them separately paid off the same day this task ran: the
authorization blocker cleared and the acceptance criterion did not move,
which would have been invisible had it been merged with the capability gap
underneath it.

- `hardware:linux_cuda_aimet_host` — cleared by provisioning a Linux x86-64
  CUDA host from `environments/linux-aimet/` using its lock files and filling
  `aimet-host.template.json` into a real host manifest. The manifest must record
  which AIMET *distribution* was installed, not only the release number: the
  PyPI and GitHub `+cu126` builds share the release string 2.36.0 but declare
  different `torch` majors. This unblocks the simulated W8A16 and W8A8 quality
  deltas.
- `user_authorization:qai_hub_submission` — **cleared 2026-08-03**, on the
  terms recorded above, with the 120-minute Device Cloud ceiling fully unspent.
  What replaced it is
  `capability:no_quantize_stage_adapter_in_this_repository`: there is no W8
  artifact to submit, and producing one needs either the AIMET host above or a
  hosted quantize job this repository cannot request from a T22-owned module.
  Do not re-ask the user for authorization; ask T22 for the adapter, or provide
  the artifact from Lane B. T41 continues to hold the `qai_hub_submission`
  resource lock and has still submitted nothing.
- `upstream_task:T31` and T33 — cleared when a floating Qwen graph has been
  shown to traverse compile, inference, and profile on a public target. Both
  are `planned`. Clear this before submitting anything quantized: a quantized
  job submitted before the floating one has ever succeeded cannot distinguish a
  quantization failure from a pipeline failure, which is why a submission today
  would not establish what the acceptance criterion asks for even with an
  artifact and authorization in hand. If Qwen is blocked there, plan section
  3.4 applies — isolate the boundary, record the reproduction, and complete the
  pipeline with the smallest verified fallback, labelling the result as
  fallback evidence and never as Qwen evidence.

Start from `ai/handoffs/T41-w8-submission-boundary.md` for the operational
sequence and the list of changes that would invalidate the frozen candidates,
and from `docs/results/quantization/w8.md` for the argument behind the two
candidates.

Newly unblocked tasks: none. T42 depends on T41 and T43 depends on T33 and
T42, so both stay `planned` until a W8 candidate actually traverses the public
pipeline.

The `configs/learning/checkpoints.yaml` entry and the corresponding `LEARN-NN`
sheet are deliberately deferred. The `AGENTS.md` protocol adds a task to the
learning lane on completion, and T41 is not completed; the sheet is written
when the evidence half lands.
