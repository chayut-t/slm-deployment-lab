# T31: Workbench Run Plan Submission Boundary

Date: 2026-08-04
Task: `T31`
Visibility: `public`
Status: planned

## Outcome

T31 built the complete three-target Qualcomm AI Hub Workbench run plan for the
committed T22 Qwen3-0.6B candidate graphs and stopped at the submission
boundary. **No job was submitted, no service was contacted, no device was
reached, and nothing was measured on any target.** There is no latency figure,
no memory figure, no accelerator placement, no operator-support verdict, and no
numerical comparison in any of this task's outputs. Every number in them is a
digest or byte count read off a committed file, a finding count read off a
committed T22 inspection manifest, or a count of plan entries.

Two blockers stand between this commit and a real result, and neither can be
manufactured inside the task:

- **B1 — the `qai-hub` client is not installed.** `pyproject.toml` pins no
  Qualcomm client, and both the project virtual environment and the system
  interpreter fail on `import qai_hub`. A client configuration file exists on
  this host; the library that would read it does not. This is an absent
  dependency, not a permission or service problem. The run plan records it
  through `importlib.util.find_spec`, which locates a module without executing
  it, so the planner never imports the client under any code path.
- **B2 — this session holds no submission authorization.** `AGENTS.md` requires
  explicit user permission before any external job. The "Budget state at
  handoff" paragraph in `ai/handoffs/T31-first-submission.md` records that an
  *earlier* session was granted hosted compile/profile/inference plus up to 120
  Device Cloud minutes on free capacity. That paragraph is a note an earlier
  agent wrote into a file: it is useful context for a resuming session and it
  is not consent held by this one. It was treated as context, and nothing was
  submitted.

Both are recorded machine-readably in the run plan under
`submission_boundary.required_before_any_submission`, with an offline recheck
for B1 and an explicit `not_machine_checkable` for B2.

The task-definition acceptance criterion "numerical output is compared with
reference" is therefore **not satisfied**, and T31 stays at stored status
`planned` with `worklog: null` rather than being promoted. What is delivered is
the durable half: two target selectors that state weaker evidence than they
might have, a run plan that re-derives byte for byte from committed inputs, a
recorded first-failure hypothesis with an attribution rule, and a fixed
claim-boundary defect that would otherwise have stamped a false claim on the
first candidate this task built.

## External-service authorization and use

Recorded as an event, per the `AGENTS.md` external-services rules.

**Authorization.** None was requested and none was held by this session.

**What was done with it.** Nothing. No network call of any kind was made. No
job was submitted, no model was uploaded, no device was leased, no inference
was run, and no artifact was downloaded. The Qualcomm client was never
imported, and a test pins that building the record leaves `sys.modules` free of
`qai_hub`.

**What it consumed.** 0 jobs. 0 Device Cloud minutes. US$0.00. The 120-minute
ceiling described in the handoff belongs to an earlier session's authorization
and was not drawn on here.

## Changes

- `configs/targets/qualcomm-dragonwing-iq-9075-evk.json` and
  `configs/targets/qualcomm-snapdragon-8-elite-qrd.json` — the two selectors
  T31 needs and T22 did not build. The device names are the exact strings under
  `workbench.public_qwen_targets` in
  `results/hosts/public-platform-access-2026-07-24.json`, read from a public
  model page without authentication. Neither declares an `os` or any
  `attributes`, and neither claims authenticated device confirmation.
- `src/slm_lab/deployment/qualcomm/workbench.py` — the offline run plan
  builder. It reuses the committed T30 validators rather than reimplementing
  them, derives the deterministic compile `request_id` the T30 preflight would
  record, reads each target's device-evidence strength off that target's own
  committed `claim_boundary`, and orders the whole matrix from committed
  measurements.
- `scripts/qualcomm/plan_workbench_run.py` and a new section in
  `scripts/qualcomm/README.md` — the `--record`, `--check`, and `--preflight`
  command surface.
- `results/raw/qualcomm/workbench/t31-workbench-run-plan-2026-08-04.json` and
  its `README.md` — the committed run plan: 3 targets x 4 variants x 2 graph
  kinds = 24 entries, 24 compile stages `ready`, 48 inference and profile
  stages `pending_predecessor`, plus the boundary, the first-failure
  hypothesis, and a zero cost record.
- `docs/results/qualcomm/workbench.md` — the reader-facing report, which opens
  by stating that it contains no device measurement.
- `src/slm_lab/graph/qnn/build.py` and `tests/qnn/test_qnn_build.py` — the
  `onnx_checker_accepted_the_candidate_graph` claim is now conditional on the
  recorded checker verdict. See "Decisions and evidence".
- `configs/targets/README.md` — the evidence asymmetry between the three
  selectors, and the naming rule that keeps the Qualcomm-only validators off a
  future Apple or NVIDIA target file.
- `tests/deployment/qualcomm/test_target_configs.py` and
  `tests/deployment/qualcomm/test_workbench.py` — the offline contract tests
  for the selectors and the planner.
- `ai/plans/active/T31-qwen-workbench.md` — the execution plan, with both
  blockers, the milestone state, and the restart sequence.
- `ai/tasks/task_graph.yaml` — the T31 entry now carries this task's owner and
  branch. `status` stays `planned` and `worklog` stays `null`; see "Risks and
  limitations" for why the field cannot carry this file's path.

## Verification

All commands were run from the `task/T31-qwen-workbench` worktree on the
primary macOS host. Nothing here contacts a service.

- Command: `PYTHONPATH=src .venv/bin/python -m pytest tests -q`
  Result: 993 passed, 31 skipped in 56.02 s.
- Command: `PYTHONPATH=src .venv/bin/python -m pytest
  tests/deployment/qualcomm tests/qnn -q`
  Result: 229 passed, 16 skipped in 0.93 s — this task's tests in isolation.
  The skips are the opt-in checks that need the external artifact root.
- Command: `PYTHONPATH=src python3 scripts/qualcomm/plan_workbench_run.py
  --check`
  Result: exit 0, `status: ok`, 3 targets, 24 plan entries, 24 stages ready, 48
  pending predecessor, `jobs_submitted: 0`, `service_contacted: false`. The
  committed record still re-derives from its committed inputs.
- Command: `python3 scripts/ai/render_task_status.py --check`
  Result: "task graph valid; 32 tasks; 13 learning checkpoints; generated
  status is current".
- Command: `python3 scripts/repo/check_hygiene.py --all`
  Result: repository hygiene passed.
- Command: `ruff format --check` and `ruff check` over the changed modules and
  test files.
  Result: formatted; "All checks passed!".

Not run, and not partially run:

- Any Qualcomm AI Hub compile, inference, profile, or quantize submission.
- Any device measurement, hosted or real. Zero Device Cloud minutes.
- Any network call at all, including a read-only capability query. The `qai-hub`
  client does not exist in this environment and was deliberately not installed.
- The `--preflight` mode was not re-run in this session. It needs the assembled
  T22 packages on the external artifact root. The committed record was written
  by that mode in the session that produced it and reads `mode: ran`,
  `requests_validated: 24`, `all_request_ids_matched_the_plan: true`.

## Decisions and evidence

- **The handoff's budget paragraph is context, not consent.** It records what
  an earlier session was authorized to do. Authorization is held by a session
  and granted by the user; a sentence in a committed file cannot transfer it.
  Treating it as consent would have made the repository able to authorize its
  own spending, which is precisely the failure mode the `AGENTS.md`
  external-services rule exists to prevent.
- **The `onnx.checker` claim is now conditional on the verdict.**
  `check_candidate` does not raise when the checker rejects a graph — it
  returns `{"status": "failed", ...}` and lets the build continue — while the
  claim `onnx_checker_accepted_the_candidate_graph` sat unconditionally in
  `CLAIM_BOUNDARY["establishes"]`. `checker_claim_boundary` now withdraws that
  claim unless every graph kind returned `passed`, states
  `onnx_checker_did_not_accept_at_least_one_candidate_graph` in its place so
  the failure is recorded rather than silently omitted, and adds
  `onnx_checker_acceptance_of_every_candidate_graph` to `does_not_establish`.
  It is fail-closed: a graph kind with no recorded verdict counts as a
  rejection, because a missing verdict is not a return of `passed`. All eight
  committed reports under `results/manifests/qnn/` read `passed`, so no
  committed claim was false and all eight re-derive their exact committed
  `claim_boundary`. The exposure was prospective, and T31 is the first task
  that builds candidates the checker has not already accepted — which is why
  `ai/handoffs/T31-first-submission.md` asked for it first.
- **The two new selectors state weaker evidence on purpose.** The planner reads
  each target's strength off the config's own `claim_boundary` markers rather
  than assigning one, and refuses a config that declares neither marker or
  both. So a selector cannot be promoted to `authenticated_device_query`
  without editing the selector itself, in public, where a reviewer sees it.
- **No device attribute was invented for either new selector.** No `os`, no
  chipset, no Hexagon version, no SoC model. The T30 `_device` validator treats
  all of these as optional, and no committed evidence records any of them for
  either device. The Device Cloud catalog's `Android` entry for a `Snapdragon 8
  Elite` / `QRD8750` was deliberately not copied: Device Cloud is a different
  service with its own device namespace, that listing is also unauthenticated,
  and the record flags it partial.
- **Inference and profile stages are `pending_predecessor`, not materialized.**
  Both need a compile manifest and the digest of the artifact it produced, and
  `ai_hub._load_predecessor` and `ai_hub._compiled_artifact` enforce that.
  Neither exists before a real compile job runs, so those fields are `null`
  rather than filled with a placeholder that would validate here and fail at
  the service.
- **The stamped preflight claim is derived, not asserted.**
  `build_record` previously wrote `service_contacted: false` and
  `job_submitted: false` into `run_observation.preflight` as literals while
  discarding the per-request values the preflight had actually collected. Both
  are now folded from those observations, and a preflight entry that does not
  report a flag is refused rather than read as `False` — an unmeasured field
  may not be published as a negative claim. The committed record is unchanged,
  because every observation in it reported `False`.
- **Test discovery over `configs/targets/` is scoped to `qualcomm-*.json`.**
  The directory is documented as also holding Apple host profiles and NVIDIA
  runtime targets, while `load_target_config` requires `client.name ==
  "qai-hub"`. A directory-wide glob would have failed the first non-Qualcomm
  target committed there. The README was corrected in the same direction rather
  than narrowed to Qualcomm, because the multi-platform intent is the older
  statement and matches `configs/README.md`.

## Risks and limitations

- **Nothing in this repository has yet traversed the public Workbench
  pipeline.** Every statement about compiler acceptance, operator support, NPU
  placement, latency, or memory for Qwen3-0.6B on any of the three targets is
  unverified. The plan's `claim_boundary` says so in its own fields rather than
  leaving a reader to infer it.
- **Two of the three selectors have never been confirmed to resolve.** Their
  device names came from an unauthenticated catalog page. Catalog support does
  not prove that the service resolves the selector for this account, that the
  device is schedulable, or that the pinned QAIRT version is supported on it.
- **External-data packaging is expected to fail first.** The compile request
  names only the `.onnx` file, because the T30 adapter requires
  `source_artifact.path` to be a single existing file, and every candidate
  carries a roughly 1.19 GB `.onnx.data` sidecar the request does not name.
  Whether the service reads the sidecar from the same directory, or wants a
  directory or an archive, is untested. **A failure there is a packaging
  result, not a graph result, and must be attributed as one.**
- **The X Elite row is not stronger for Qwen because of T02.** The 2026-07-25
  authenticated lifecycle compiled a *toy* model on that device. It establishes
  the device identity and the QAIRT default the selector copies, and nothing
  about Qwen3-0.6B. Presenting it as a Qwen result would be exactly the
  substitution the acceptance criterion forbids.
- **The task graph cannot reference this worklog while T31 is not completed.**
  `scripts/ai/render_task_status.py` rejects a `worklog` field on any task that
  is not `completed`, so the T31 entry carries `worklog: null` and the link to
  this file is by filename from the execution plan and the report. The
  validator is a shared repository invariant and was not weakened, and T31's
  status was not falsified to satisfy it. This is the same shape as T41.
- **`LEARN-06` is now marked stale** in `ai/tasks/status.generated.md` because
  `scripts/qualcomm/README.md` changed. Rebuilding and republishing the sheet
  belongs with the evidence half of this task, not with a plan that measured
  nothing.

## Follow-up

Both blockers are external to the code and clear independently.

- `capability:no_qai_hub_client_installed` — cleared by obtaining a `qai-hub`
  client in the environment the submission will run from. Whether that means
  pinning it in `pyproject.toml` or carrying it in a separate environment is an
  open decision this task did not take.
- `user_authorization:qai_hub_submission` — cleared only by the user, to the
  session that will submit. Do not read
  `ai/handoffs/T31-first-submission.md` as consent.

When both clear, submit exactly one thing first: Snapdragon X Elite CRD, S128,
prefill, request id `t30-compile-83b8813c19a37ac036ad`. It is the only graph in
the matrix with zero residual high-severity shape findings and the smallest
candidate protobuf, on the only target with an authenticated device identity,
so a failure there is attributable to the pipeline rather than to the graph.
Expect the external-data packaging boundary to break first and record it as a
packaging result. Only after a real compile manifest exists may inference and
profile run, and the numerical comparison is against
`results/manifests/qnn/parity/S128-ort-cpu.json`.

Newly unblocked tasks: none. T33 depends on T31 and stays `planned` until a
Qwen graph has actually traversed compile, inference, and profile on a public
target.

The `configs/learning/checkpoints.yaml` entry and the corresponding `LEARN-NN`
sheet are deliberately deferred. The `AGENTS.md` protocol adds a task to the
learning lane on completion, and T31 is not completed.

Start from `ai/plans/active/T31-qwen-workbench.md` for the restart sequence,
`ai/handoffs/T31-first-submission.md` for the sequencing argument, and
`docs/results/qualcomm/workbench.md` for what the plan does and does not claim.
