# T41: W8 quantization evidence

Status: active
Owner: Claude T41 agent
Updated: 2026-08-03

## Objective

Specify, freeze, and gate the two deployable eight-bit-weight candidates that
the plan's experiment matrix (section 7.2) calls Q1, conservative PTQ with INT8
weights and INT16 activations, and Q2, aggressive PTQ with INT8 weights and
INT8 activations. The outcome is that the first agent with a Linux + CUDA host
and submission authorization can produce quality and hardware evidence for both
candidates without re-deciding anything this task already decided.

No weight is quantized here, and no number in this task's outputs is a
measurement of a quantized model. Three external blockers stand between T41 and
a real W8 result, and this plan names them separately because they clear
separately:

- **B1 — no Linux + CUDA host.** AIMET is pinned in `environments/linux-aimet/`
  but has never been installed or executed; T40 established that it publishes
  no macOS or Apple-silicon build. The primary host also carries no `torch`,
  `onnx`, `onnxruntime`, `numpy`, or `transformers`, so nothing can be
  fake-quantized or numerically evaluated locally.
- **B2 — no W8 artifact to submit. Authorization CLEARED 2026-08-03.** Every
  Qualcomm AI Hub compile, inference, or profile job requires explicit user
  authorization under `AGENTS.md`, and on 2026-08-03 the user gave it for T41:
  hosted compile, profile, and inference jobs plus Device Cloud interactive
  minutes capped at 120, free capacity only, no spend without a fresh decision.
  Clearing it did not make the criterion reachable. A submission needs an
  input, and no W8 artifact exists. Producing one takes Lane B, blocked by B1,
  or a hosted quantize job, which this repository cannot request:
  `slm_lab.deployment.qualcomm.ai_hub` implements compile, inference, and
  profile only, and it sits under `src/slm_lab/deployment/qualcomm/`, an owned
  path of **T22** (`ai/tasks/definitions/T22.yaml`) that is being worked
  concurrently. T41 does not add a stage to another task's module. The
  surviving blocker is therefore a capability and ownership gap with T22 named,
  not an authorization problem.
- **B3 — the upstream floating path is unproven.** T31, "Qwen Workbench results
  on three Qualcomm targets", and T33, the integrated floating milestone, are
  both `planned` in `ai/tasks/task_graph.yaml`. No Qwen graph of any precision
  has traversed the public Workbench pipeline yet, so a quantize or compile
  failure could not be attributed to the W8 policy rather than to the graph,
  and a success would isolate neither. That is the evidence-interpretability
  reason a submission today would not establish what the acceptance criterion
  wants. The criterion therefore also waits on work T41 does not own.

Under the 2026-08-03 authorization one read-only capability query was run
against the live service with client `qai-hub==0.53.0`. It submitted no job,
consumed no Device Cloud minutes, and cost nothing; the 120-minute ceiling is
fully unspent. It established that `submit_quantize_job(model,
calibration_data, weights_dtype, activations_dtype, name, options, project)`
exists and that `QuantizeDtype` carries INT4, INT8, and INT16, so both
candidates are expressible as public quantize requests; that all three plan
section 3.2 target devices are live among 79 enumerated devices; and that the
quantize API exposes no separate KV-cache dtype knob, consistent with the
frozen `CACHE_DTYPE` finding. It established nothing about compiler acceptance,
NPU placement, latency, or accuracy, and **no W8 candidate traversed the public
pipeline**.

What T41 can deliver against those blockers is the contract and its
enforcement: a frozen, hash-checked candidate specification, a module that
regenerates and validates it, a fail-closed precision-state machine that cannot
be talked into calling a simulation a deployment, and a stage-request path that
stops exactly at the submission boundary. That is the durable half of the task,
and it is worth doing before the hardware exists precisely because it decides
in advance what will and will not count as evidence.

## Scope

### In scope

- `configs/quantization/w8/` — the frozen Q1 (W8A16) and Q2 (W8A8) candidate
  specifications: weight and activation precision, granularity, per-op
  inclusion and exclusion policy, cache precision, the target-runtime and
  target-device intent for each, and the `calibration_dataset_revision` binding
  that ties every candidate to T40's corpus.
- `src/slm_lab/quantization/w8.py` — generation and validation of those
  specifications, dependency-free beyond the standard library and PyYAML so it
  runs on any host; the analytic weight-storage projection; the precision-state
  and verdict function; the frozen-evaluation quality-delta comparison; and the
  emission of Qualcomm AI Hub stage requests up to, never across, the
  submission boundary.
- An explicitly labelled analytic weight-storage projection derived from the
  committed architecture fields in `configs/models/qwen3-0.6b.yaml` and the
  committed external-data sizes in `results/manifests/onnx/S*.json`. It is a
  projection, carries that label in its own field name, and is never presented
  as an artifact size, a memory measurement, or a compression result.
- A fail-closed precision-state and verdict function. A candidate is
  `specified` until a simulation exists, `simulated` until compile, inference,
  and profile evidence all exist, and only then `deployed`. `deployed` is
  reachable from data — a complete, correctly digest-chained evidence set
  returns it — and is unreachable at this commit only because no such manifest
  set exists anywhere in this repository. That is a deliberate difference from
  T40's `slm_lab.quantization.parity`, whose terminal verdict has no branch at
  all: here the guarantee is the digest chain, so promotion stays an auditable
  data change rather than a code edit.
- The frozen-evaluation quality-delta comparison, bound to the T13 contracts in
  `configs/workloads/academic-evaluation-v1.json` and
  `configs/workloads/benchmark-protocol-v1.json` and gated through
  `slm_lab.benchmark.protocol`. It is implemented and enforced as a protocol
  here; it computes nothing until a quantized model exists.
- A committed readiness record under `results/quantization/`, following the
  `t40-baseline-parity-<date>.json` precedent, recording which gates pass,
  which are `not_run`, and which blocker owns each one.
- `docs/results/quantization/w8.md` — the reader-facing report, which must open
  by stating that it contains no W8 measurement.
- Tests under `tests/quantization/`, a public worklog, and the task-graph
  status update.

### Out of scope

- Quantizing any weight, running AIMET, or producing any fake-quant simulation
  number. That is B1.
- Submitting any AI Hub compile, inference, or profile job, reserving any
  device, or making any device measurement. Authorization for these exists as
  of 2026-08-03 and they still cannot be done, because no W8 artifact exists to
  submit. That is B2.
- Adding a quantize stage to `slm_lab.deployment.qualcomm.ai_hub`. The service
  supports quantize jobs, but the module is T22's owned path and T22 is open.
  The route is named here so a future session pursues it as a cross-task
  decision rather than an edit.
- W4A8, LPBQ, LiteMP, mixed precision, and layer or block sensitivity analysis.
  That is T42.
- Quantized compile, inference, and profile evidence and the quality-latency-
  memory comparison built on it. That is T43.
- Editing T40's calibration contract or T13's evaluation contract. T41 consumes
  both and may not widen either to make a result look better.
- Installing any heavy dependency into the repository environment.

## Dependencies and resources

- Required task dependencies: T40, completed
  (`ai/worklogs/2026-08-02-T40-aimet-calibration-environment.md`). The evidence
  half of this task additionally depends on T31 and T33, which are `planned`
  and which T41 does not own; that is B3. It also depends on a W8 artifact
  existing, which needs either B1 cleared or a quantize-stage adapter in
  `src/slm_lab/deployment/qualcomm/`, an owned path of T22.
- Resource locks: `qai_hub_submission`. The lock is held so that no other
  writer submits a Workbench job under a competing precision policy while this
  task is open. It is not exercised: no job was submitted at any stage.
- External access: one read-only capability query against Qualcomm AI Hub on
  2026-08-03 with client `qai-hub==0.53.0`, run under the authorization granted
  that day. It enumerated the API surface and the device fleet, submitted no
  job, reserved no device, and consumed no Device Cloud minutes. No committed
  check depends on the network; every gate in this task runs offline.
- Cost boundary: zero, and zero was spent. The authorization permits free
  capacity only and caps Device Cloud interactive use at 120 minutes; 0 of
  those 120 minutes were used and the ceiling is fully unspent. Any spend, or
  the 121st minute, needs a fresh decision.

## Important paths

- Inputs: `configs/quantization/calibration.yaml` (specifically
  `calibration_dataset_revision:
  t40-qwen3-0.6b-t10-derived-v1+sha256.d2b749e15dd5d987`),
  `src/slm_lab/quantization/parity.py`,
  `results/quantization/t40-baseline-parity-2026-08-02.json`,
  `configs/models/qwen3-0.6b.yaml`, `results/manifests/onnx/S*.json`,
  `src/slm_lab/manifests/schemas/artifact-v1.schema.json`,
  `configs/workloads/academic-evaluation-v1.json`,
  `configs/workloads/benchmark-protocol-v1.json`,
  `src/slm_lab/benchmark/protocol.py`,
  `src/slm_lab/deployment/qualcomm/ai_hub.py`, `scripts/qualcomm/`,
  `environments/linux-aimet/`.
- Outputs: as listed under "In scope".
- Shared contracts: the T12 static prefill and decode tensor contract; the
  section 17.4 artifact manifest envelope, whose `precision`, `quantization`,
  and `calibration_dataset_revision` fields every W8 manifest must fill; the
  T13 frozen benchmark and evaluation protocol; the T30 sanitized stage-request
  and stage-manifest schema.

## Milestones

- [ ] `configs/quantization/w8/` freezes exactly two candidates, Q1 and Q2,
      each carrying the T40 `calibration_dataset_revision` verbatim, and is a
      byte-identical fixed point of its generator.
- [ ] `slm_lab.quantization.w8` regenerates and validates those specifications
      offline, on a host with no quantization stack installed, and fails closed
      on drift in any pinned input, hash, or revision string.
- [ ] The analytic weight-storage projection is emitted with a name and a
      `claim_boundary` that make it unusable as a measurement, and its inputs
      are traceable to committed files.
- [ ] The precision-state and verdict function refuses to report `deployed`
      without compile, inference, and profile evidence, and a mutation test
      proves the refusal rather than asserting it.
- [ ] The frozen-evaluation comparison rejects any quality delta whose floating
      and quantized sides do not cite the same T13 task set, harness release,
      and dataset revisions, and attaches the inherited calibration-bias
      statement to every delta it would emit.
- [ ] Qualcomm AI Hub stage requests for both candidates are generated and
      validated locally, and the run stops before submission with an explicit
      non-submission outcome rather than an error.
- [ ] `results/quantization/` carries a committed readiness record naming every
      `not_run` gate and its owning blocker.
- [ ] `docs/results/quantization/w8.md` lets a reader answer what W8A16 and
      W8A8 would cost and what is still unknown, without implying a
      measurement exists.
- [ ] `tests/quantization/` proves the above, offline and deterministically.
- [x] Submission authorization obtained, and recorded with its terms and its
      unspent 120-minute Device Cloud ceiling. Exercised only for one read-only
      capability query: no job submitted, no minutes consumed, no cost.
- [ ] **Blocked (B1)** — a simulated W8A16 and W8A8 quality delta against the
      frozen evaluation. Needs a Linux + CUDA host with AIMET installed from
      `environments/linux-aimet/`.
- [ ] **Blocked (B2)** — compile, inference, and profile manifests for at least
      one W8 candidate. Authorization is no longer what is missing; a W8
      artifact is. Needs B1 cleared, or a quantize-stage adapter in
      `src/slm_lab/deployment/qualcomm/`, which is T22's owned path and T22's
      decision.
- [ ] **Blocked (B3)** — at least one W8 candidate traversing the public
      pipeline end to end. Needs T31 and T33 to establish that the floating
      Qwen graph traverses it first, so that a W8 outcome is attributable to
      the quantization policy rather than to the graph.

## Verification and acceptance

- Commands:
  - `uv run pytest tests/quantization`
  - `uv run python -m slm_lab.quantization.w8 check` (introduced by this task)
  - `uv run python -m slm_lab.quantization.parity verify`
  - `uv run pytest tests`
  - `uv run ruff format --check` and `uv run ruff check` on the changed paths
  - `uv run python scripts/ai/render_task_status.py --check`
  - `uv run python scripts/repo/check_hygiene.py --all`
- Numerical or behavioral criteria: regenerating `configs/quantization/w8/`
  from committed inputs reproduces the committed hashes exactly, and mutating
  any policy knob changes them; a candidate whose
  `calibration_dataset_revision` does not equal the value
  `slm_lab.quantization.calibration` regenerates is rejected rather than
  reported as incomparable; the precision-state function yields `deployed` only
  for a complete, correctly digest-chained evidence set, and no input this
  repository contains is one; a quality delta with mismatched
  frozen-evaluation identifiers on its two sides is rejected; a stage request
  whose input artifact does not exist stops with an explicit non-submission
  outcome rather than an error.
- Hardware/profile evidence: none is claimed and none exists. Of the three
  acceptance criteria in `ai/tasks/definitions/T41.yaml`, "simulated and
  deployed precision are distinguished" and "quality deltas use frozen
  evaluation" are satisfiable here as enforced protocol, with no measurement
  behind either. "At least one W8 candidate traverses the public pipeline" is
  **not satisfied**: no candidate was submitted to any stage, and the read-only
  capability query of 2026-08-03 is not a traversal. It remains blocked by B1,
  B2, and B3 together, with B2 now resting on the missing artifact and the
  T22-owned adapter rather than on authorization. T41 therefore ends at stored
  status `blocked`, which section 10.1 defines as the state "stored explicitly
  when an external blocker remains after dependencies are complete". It is not
  `completed`.

## Artifact and privacy handling

- Committed evidence: the W8 candidate specifications and their hashes, the
  analytic projection, the readiness record, and the reader-facing report. No
  weights, no encodings, no quantized graph.
- External artifacts: the T20 floating ONNX graphs and their external data stay
  on `${SLM_LAB_ARTIFACT_ROOT}`; only digests are cited. Any future quantized
  export belongs there too, with only its manifest and digests committed.
- Private/local material: generated AI Hub stage requests are treated as
  private until sanitized through the T30 adapter's redaction rules. No token,
  account identifier, job identifier, job URL, raw service response, or session
  identifier appears in any file this task commits.

## Decisions and discoveries

- 2026-08-03: the three blockers are recorded separately rather than as one
  "no hardware" note because they clear independently and in a specific order.
  B1 needs a machine, B3 needs two upstream tasks, and B2 needed a person. A
  single merged blocker would hide that a future agent can clear B1 and B3
  without ever asking for authorization, and would invite the mistake of
  submitting a quantized job before the floating one has ever succeeded. The
  separation paid off the same day: B2's authorization cleared and the
  acceptance criterion did not move an inch, which is only visible because the
  authorization was never the same record as the capability gap beneath it.
- 2026-08-03: the user authorized AI Hub submission for T41 — hosted compile,
  profile, and inference jobs, plus Device Cloud interactive minutes capped at
  120, free capacity only, no spend without a fresh decision, and still no
  pushing and no public GitHub state. Under it, one read-only capability query
  was run with client `qai-hub==0.53.0`: no job, no device reservation, zero of
  the 120 minutes, US$0.00. The ceiling is fully unspent. The query established
  that the client's `submit_quantize_job` covers both candidates' dtype pairs
  through `QuantizeDtype.INT8` and `INT16` (with `INT4` present for T42), that
  all three plan section 3.2 devices are live among 79 enumerated, and that the
  quantize API has no separate KV-cache dtype knob. It established nothing
  about this model, and **no W8 candidate traversed the public pipeline**.
- 2026-08-03: the precision-state function is deliberately built so that
  `deployed` is unreachable from present inputs, because the digest chain it
  requires has no witness in this repository — not because the function
  declines to emit the state. The alternative, a state that becomes reachable
  when a configuration flag is set, is the exact failure the plan's rule "do
  not report a simulated precision as deployed without artifact and compiler
  evidence" (section 7.2) exists to prevent.
- 2026-08-03: the calibration bias T40 measured is inherited and must travel
  with every future W8 quality delta. The corpus is 6,912 tokens, 83% of the
  token budget comes from one repeated seed, and four tier-1 samples are both
  calibrated on and evaluated on. Any W8 delta computed against that corpus is
  optimistically biased, and this task attaches that statement to the
  comparison itself so a later reader cannot receive the number without it.
- 2026-08-03: T41 keeps only Q1 and Q2. Widening the calibration corpus with
  T40's declared tier-2 candidates was left open for T41 by the T40 worklog,
  but changing the corpus and introducing the first quantization policy in one
  task would leave any future delta unattributable to either change. The corpus
  stays frozen at the T40 revision for both W8 candidates.
- 2026-08-03: the W8 manifests reuse the existing
  `src/slm_lab/manifests/schemas/artifact-v1.schema.json` envelope rather than
  a new schema. It already requires `precision`, `quantization`, and
  `calibration_dataset_revision`, which are the three fields that make a
  quantized artifact comparable to the floating baseline.

## Progress and restart instructions

Work is executed on the `task/T41-w8-quantization-evidence` branch in its own
worktree; the checkout location is a local detail and is deliberately not
recorded here. T41 owns `configs/quantization/w8/`,
`src/slm_lab/quantization/w8.py`, and `docs/results/quantization/w8.md`. It
does not own `configs/quantization/calibration.yaml`,
`src/slm_lab/quantization/calibration.py`, or
`src/slm_lab/quantization/parity.py`, all of which are T40's frozen output and
must be read, not edited.

The task ends at stored status `blocked`. Everything that does not require
hardware, a W8 artifact, or an upstream task is done and verified. There is no
W8 measurement in this repository, and no file this task commits should be read
as containing one. **No job was submitted and no W8 candidate traversed the
public pipeline**; the one live-service action taken was a read-only capability
query, which is not a traversal.

Submission authorization is already in hand and does not need to be requested
again for what it covers: hosted compile, profile, and inference for T41, plus
Device Cloud interactive minutes capped at 120, free capacity only, zero spend.
Zero minutes were used, so the whole ceiling is inherited. A paid job, capacity
outside the free tier, or the 121st minute needs a fresh decision. The
`worklog` field of the T41 entry in `ai/tasks/task_graph.yaml` is `null`,
because the validator permits it only on a completed task; the worklog exists
at `ai/worklogs/2026-08-03-T41-w8-candidate-evidence-boundary.md` and is linked
by filename from this plan and from the report until T41 completes.

Next actions, in order. The first is no longer authorization — it is getting
something to submit:

1. Re-establish the offline state before changing anything. Run
   `uv run pytest tests/quantization`,
   `uv run python -m slm_lab.quantization.w8 check`, and
   `uv run python -m slm_lab.quantization.parity verify`, then read
   `configs/quantization/w8/` top to bottom. Every claim there is either
   regenerated by the module or labelled with the command that would verify it.
2. Choose a route to a W8 artifact, because nothing after this can start
   without one. Route A is a hosted quantize job: the service accepts it —
   `submit_quantize_job` with `QuantizeDtype.INT8` and `INT16` covers both
   candidates — but `slm_lab.deployment.qualcomm.ai_hub` cannot request one,
   and that module is T22's owned path, so Route A is a cross-task decision
   with T22 and not an edit T41 may make. Route B is local AIMET, which is
   step 3 and which a T41 session can start unilaterally.
3. Clear B1. Provision a Linux x86-64 CUDA host from
   `environments/linux-aimet/` using its lock files, and fill
   `aimet-host.template.json` into a real host manifest that records which
   AIMET distribution was installed — the PyPI distribution and the GitHub
   `+cu126` distribution share the release number 2.36.0 but declare different
   `torch` majors, so the release number alone does not identify the build.
4. On that host, close the numerical half of T40's parity gate before
   publishing any delta. It needs `torch` and `onnxruntime`, is recorded as
   `not_run` in `results/quantization/t40-baseline-parity-2026-08-02.json`, and
   is owned by T21. A quality delta whose floating side has never been
   numerically verified against the PyTorch reference is not a quality delta.
5. Run W8A16 first, then W8A8, against the frozen specifications and the frozen
   T40 corpus. Record the simulated deltas through the T13 gate, keep every
   candidate at precision state `simulated`, and carry the calibration-bias
   statement into every reported number. Do not renumber, rename, or relax a
   frozen candidate to make a result look better; add a new candidate id
   instead and say why.
6. Clear B3 before submitting. T31 and T33 must show that a floating Qwen graph
   traverses compile, inference, and profile on a public target. If Qwen is
   blocked there, plan section 3.4 applies: isolate the boundary, record the
   exact reproduction, and complete the pipeline with the smallest verified
   fallback model, labelling the resulting W8 evidence as fallback evidence and
   never as Qwen evidence.
7. Submit last, under the standing authorization rather than a new one. Confirm
   the job stays inside its terms — free capacity, no spend, at most the
   unspent 120 Device Cloud minutes — then submit through
   `scripts/qualcomm/compile.py`, `inference.py`, and `profile.py` while
   holding the `qai_hub_submission` lock. Anything outside those terms needs a
   fresh decision from the user. Sanitize every result through the T30 adapter
   before anything is committed.
8. Only once compile, inference, and profile manifests exist for a candidate
   may it move from `simulated` to `deployed`, and that transition is a data
   change backed by auditable manifests, not a code change.
   `assess_precision_state` in `src/slm_lab/quantization/w8.py` already returns
   `deployed` for a complete evidence set: a simulation record plus compile,
   inference, and profile stage manifests that are schema-v2, declare the stage
   they are filed under, report success, and chain — the compile manifest's
   source digest equal to the simulation's quantized-artifact digest, the
   inference and profile `compiled_model` digests equal to the compile
   manifest's target-artifact digest, and the predecessor-manifest digest
   recomputed from the manifest supplied rather than trusted as declared.
   `deployed` is unreachable today because no such manifest set exists anywhere
   in this repository, not because the function refuses to emit it. Then update
   `ai/tasks/task_graph.yaml` from `blocked` through
   `in_progress` to `completed`, regenerate with
   `uv run python scripts/ai/render_task_status.py`, publish the worklog, and
   move this plan to `ai/plans/completed/`.
