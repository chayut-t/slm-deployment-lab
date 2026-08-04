# T31: Qwen Workbench results on three Qualcomm targets

Status: active
Owner: Claude T31 agent
Updated: 2026-08-04

## Objective

Compile, run inference on, and profile the committed T22 Qwen3-0.6B candidate
graphs on the three public Qualcomm AI Hub Workbench targets — Snapdragon X
Elite CRD, Dragonwing IQ-9075 EVK, and Snapdragon 8 Elite QRD — and report
either a comparison or the exact blocker for each, with no target standing in
for another.

**No device evidence exists at this commit.** No job was submitted, no service
was contacted, and nothing was measured on any device. What this task has
delivered so far is everything up to the submission boundary: the three-target
run plan, the two missing target selectors, the offline validation of all 24
compile requests through the committed T30 validators, and the recorded
boundary. Two blockers stand between this commit and a real result, and they
clear independently:

- **B1 — the `qai-hub` client is not installed.** `pyproject.toml` pins no
  Qualcomm client, and both the project virtual environment and the system
  interpreter fail on `import qai_hub`. A client configuration file exists on
  this host; the library that would read it does not. This is an absent
  dependency, not a permission or service problem, and it is machine-checkable:
  the run plan records it through `importlib.util.find_spec`, which locates a
  module without executing it.
- **B2 — this session holds no submission authorization.** `AGENTS.md`
  requires explicit user permission before any external job. The "Budget state
  at handoff" paragraph in `ai/handoffs/T31-first-submission.md` records that an
  *earlier* session was granted hosted compile/profile/inference plus up to 120
  Device Cloud minutes on free capacity. That paragraph is a note an earlier
  agent wrote into a file. It is context for a resuming session and it is not
  consent held by this one, so it was treated as context and nothing was
  submitted. This blocker is recorded as `not_machine_checkable` on purpose.

Both are recorded in the run plan under
`submission_boundary.required_before_any_submission`.

## Scope

### In scope

- `configs/targets/qualcomm-dragonwing-iq-9075-evk.json` and
  `configs/targets/qualcomm-snapdragon-8-elite-qrd.json` — the two target
  selectors T31 needs and T22 did not build. Their device evidence is an
  unauthenticated public catalog listing and is deliberately weaker than the X
  Elite selector's authenticated claim; each says so in its own
  `claim_boundary`.
- `src/slm_lab/deployment/qualcomm/workbench.py` — the offline three-target run
  plan builder. It reuses the committed T30 validators rather than
  reimplementing them, so a request it accepts is a request the compile stage
  would accept, and it never imports `qai_hub` under any code path.
- `scripts/qualcomm/plan_workbench_run.py` — the `--record`, `--check`, and
  `--preflight` command surface.
- `results/raw/qualcomm/workbench/t31-workbench-run-plan-2026-08-04.json` — the
  committed run plan: 3 targets x 4 variants x 2 graph kinds = 24 entries, 24
  compile stages `ready`, 48 inference and profile stages
  `pending_predecessor`, with the boundary, the first-failure hypothesis, and a
  zero cost record.
- `docs/results/qualcomm/workbench.md` — the reader-facing report, which opens
  by stating that it contains no device measurement.
- `src/slm_lab/graph/qnn/build.py` — the conditional
  `onnx_checker_accepted_the_candidate_graph` claim, fixed before building any
  new candidate as `ai/handoffs/T31-first-submission.md` section 1 requires.
- Tests under `tests/deployment/qualcomm/` and `tests/qnn/`, a public worklog,
  and the task-graph owner/branch claim.

### Out of scope

- Submitting any compile, inference, or profile job, reserving any device, or
  taking any measurement. That is B1 and B2, and no part of it may be
  simulated, estimated, or filled with a placeholder.
- Installing `qai-hub` or pinning it in `pyproject.toml`. The client is
  optional by design; adding a dependency is a separate decision.
- Quantized graphs of any precision. That is T41, T42, and T43, and it waits on
  this task establishing that a floating graph traverses the pipeline at all.
- The Device Cloud generation loop. That is T32, completed, and its results are
  not Workbench results.
- Substituting a result measured on one target for another. The acceptance
  criterion forbids it and the plan carries the rule as a `no_proxy_rule` field
  on every target.

## Dependencies and resources

- Required task dependencies: T22, completed
  (`ai/worklogs/2026-08-03-T22-qnn-candidates-and-packaging.md`), and T30,
  completed (`ai/worklogs/2026-07-25-T30-ai-hub-adapters.md`).
- Resource locks: `qai_hub_submission`. Held so no other writer submits a
  Workbench job while this task is open. It has not been exercised: nothing was
  submitted at any stage.
- External access: none. No network call of any kind was made in this task, and
  every gate runs offline.
- Cost boundary: zero, and zero was spent — 0 jobs, 0 Device Cloud minutes,
  US$0.00. The 120-minute ceiling described in the handoff belongs to an
  earlier session's authorization and was not drawn on here.

## Important paths

- Inputs: `configs/targets/qualcomm-snapdragon-x-elite-crd.json`,
  `results/manifests/qnn/packages/S*.json`, `results/manifests/qnn/S*.json`,
  `results/manifests/qnn/inspection/S*.json`,
  `results/manifests/qnn/parity/S*-ort-cpu.json`,
  `results/hosts/public-platform-access-2026-07-24.json`,
  `results/hosts/workbench-toy-lifecycle-2026-07-25.json`.
- Outputs: `results/raw/qualcomm/workbench/`, `docs/results/qualcomm/`,
  `configs/targets/`, `src/slm_lab/deployment/qualcomm/workbench.py`,
  `scripts/qualcomm/plan_workbench_run.py`.
- Shared contracts: `src/slm_lab/deployment/qualcomm/ai_hub.py` (T30) and
  `src/slm_lab/deployment/qualcomm/packaging.py` (T22) are consumed, not
  edited. `src/slm_lab/graph/qnn/build.py` is T22's module and was changed only
  for the claim-boundary defect the T31 handoff names.
- Private material: compile requests written by `--preflight` go to
  `.ai-local/profiles/T31/`, never into the repository.

## Milestones

- [x] The two missing target selectors exist, are accepted by the committed T30
      validators, and state device evidence weaker than X Elite's.
- [x] The `onnx_checker_accepted_the_candidate_graph` claim is conditional on
      the checker verdict, and all eight committed `results/manifests/qnn/`
      reports re-derive their exact committed `claim_boundary`.
- [x] The 24-entry run plan re-derives byte for byte from committed inputs, and
      the X Elite S128 prefill compile request id equals the id already
      committed in `results/manifests/qnn/packages/S128.json`.
- [x] The submission boundary and both blockers are recorded machine-readably
      and in prose.
- [ ] **Blocked on B1 and B2.** S128 prefill compile submitted on Snapdragon X
      Elite CRD, with a real compile manifest.
- [ ] Inference and profile stages run against that compile manifest, with the
      numerical output compared against
      `results/manifests/qnn/parity/S128-ort-cpu.json`.
- [ ] IQ-9075 EVK and 8 Elite QRD each produce a comparison or an exact
      blocker, with no proxy claim.

## Verification and acceptance

- Commands:
  - `PYTHONPATH=src python3 -m pytest tests -q`
  - `PYTHONPATH=src python3 scripts/qualcomm/plan_workbench_run.py --check`
  - `python3 scripts/ai/render_task_status.py --check`
  - `python3 scripts/repo/check_hygiene.py --all`
- Numerical or behavioral criteria: the run plan is a deterministic function of
  committed inputs, every input is bound by repository-relative path and
  SHA-256, and `--check` fails on the first differing key path. `--check` also
  refuses any record claiming a submitted job or a contacted service. The
  device-evidence strength of each target is read off that target's own
  committed `claim_boundary`; a config declaring neither marker is refused
  rather than guessed at.
- Hardware/profile evidence: **none exists.** The task-definition acceptance
  criteria "device and toolchain versions are captured" and "unavailable
  targets are reported without proxy claims" are satisfied for the plan;
  "numerical output is compared with reference" is **not satisfied** and cannot
  be until B1 and B2 clear. T31 therefore stays `status: planned`.

## Artifact and privacy handling

- Committed evidence: the run plan record, the two target selectors, and the
  report. All are small, all are offline-derivable, and none carries a job id,
  a job URL, an account identifier, or a raw service response.
- External artifacts: the assembled T22 packages live under
  `${SLM_LAB_ARTIFACT_ROOT}`, are never committed, and are only read by the
  optional `--preflight` mode.
- Private/local material: `.ai-local/profiles/T31/` holds the preflight compile
  requests. `_private_output_path` refuses to write one inside the repository.

## Decisions and discoveries

- 2026-08-04: The two new selectors declare no `os` and no `attributes`. The
  committed T30 `_device` validator treats both as optional, and no committed
  evidence records an operating system, chipset, Hexagon version, or SoC model
  for either device. The Device Cloud catalog's `Android` entry for a
  `Snapdragon 8 Elite` / `QRD8750` was deliberately not copied: Device Cloud is
  a different service with its own device namespace, and that listing is also
  unauthenticated and flagged partial. Inventing any of these would be
  fabrication.
- 2026-08-04: The client version and QAIRT version are the only fields carried
  over from the authenticated 2026-07-25 lifecycle. They are device-independent
  — the QAIRT default came from a service-wide `get_frameworks()` query — so
  reusing them imports no device claim. Whether that QAIRT version is supported
  on either new device is unverified.
- 2026-08-04: Inference and profile stages are `pending_predecessor` rather
  than materialized. Both need a compile manifest and the digest of the
  artifact it produced, and neither exists before a real compile job runs.
  Emitting a placeholder so the request validates would produce a plan that
  passes here and fails at the service.
- 2026-08-04: The handoff's "Budget state at handoff" paragraph was read as
  context, not as consent. See B2.
- 2026-08-04: The `run_observation.preflight` block derives
  `service_contacted` and `job_submitted` from the per-request preflight
  observations instead of stamping literals, so the published claim is backed
  by the measurement and an unreported flag is refused rather than read as
  `False`.

## Progress and restart instructions

Everything up to the submission boundary is built, committed, and checkable
offline. Nothing beyond it has been attempted.

A resuming session should, in this order:

1. Obtain a `qai-hub` client in the environment it will submit from (B1) and
   obtain explicit submission authorization from the user in its own session
   (B2). Do not read the handoff's budget paragraph as consent.
2. Re-run `PYTHONPATH=src python3 scripts/qualcomm/plan_workbench_run.py
   --check` to confirm the plan still matches its committed inputs, then run
   the optional `--preflight` mode with `SLM_LAB_ARTIFACT_ROOT` set to validate
   all 24 compile requests against the assembled T22 packages.
3. Submit **only** the first plan entry: Snapdragon X Elite CRD, S128, prefill,
   request id `t30-compile-83b8813c19a37ac036ad`. It is the only graph in the
   matrix with zero residual high-severity shape findings and the smallest
   candidate protobuf, on the only target with an authenticated device
   identity, so a failure there is attributable to the pipeline rather than to
   the graph.
4. Expect external-data packaging to fail first. Every candidate carries a
   roughly 1.19 GB `.onnx.data` sidecar that the compile request does not name,
   because the T30 adapter requires `source_artifact.path` to be a single
   existing file. **Record such a failure as a packaging result, not a graph
   result**, and file it under
   `first_failure_hypothesis` rather than as a compiler verdict about Qwen.
5. Only after a real compile manifest exists, run inference and profile against
   it and compare the numerical output with
   `results/manifests/qnn/parity/S128-ort-cpu.json`.

Read `ai/worklogs/2026-08-04-T31-workbench-run-plan-submission-boundary.md` for
what this session did and why it stopped — the task graph cannot reference it
by field while T31 is not `completed`, so it is named here —
`ai/handoffs/T31-first-submission.md` for the sequencing argument, and
`docs/results/qualcomm/workbench.md` for what the plan does and does not claim.
