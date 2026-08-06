# T31: X Elite Selector Reconciliation

Date: 2026-08-06
Task: `T31`
Visibility: `public`
Status: final

## Outcome

The Snapdragon X Elite CRD device selector correction (os `Windows 11` →
`11`, discovered by the authenticated 2026-08-04 device query) is now fully
reconciled across the repository: the run plan record re-derives with the
corrected selector, every test that pinned the superseded value or a
superseded request id states the corrected contract, and the public docs
explain the divergence from the historical T22 package records instead of
claiming an equality that no longer holds. The repository is back to a state
where `plan_workbench_run.py --check`, the full test suite, the task-status
check, and the hygiene check all pass.

## Changes

- `results/raw/qualcomm/workbench/t31-workbench-run-plan-2026-08-04.json` —
  regenerated with `--preflight` over all 24 compile requests against the
  assembled T22 packages on the artifact root. No service contact, no job
  submitted. The eight X Elite `request_id` values changed because the device
  selector enters the hashed public request; the first submission is now
  `t30-compile-4002ded9a30ed87a692c` (was `t30-compile-83b8813c19a37ac036ad`).
- `tests/deployment/qualcomm/test_workbench.py` — split the single committed-id
  constant into the planned id (corrected selector) and the historical T22 id.
  The all-eight equality test became a divergence proof: for every X Elite
  entry, rebuilding the request with the superseded os value (read from the
  device-query evidence, not hardcoded) reproduces the committed T22 id, so
  the id change is exactly the os correction and nothing else.
- `tests/deployment/qualcomm/test_packaging.py`,
  `tests/deployment/qualcomm/test_target_configs.py` — expectations updated to
  the corrected selector; a new test ties the target config's
  `device_os_correction` evidence pointer to the device-query record and
  asserts the two agree.
- `docs/results/qualcomm/workbench.md`,
  `results/raw/qualcomm/workbench/README.md`,
  `ai/plans/active/T31-qwen-workbench.md` — first-submission id updated and
  the equality claim rewritten as the explained divergence.
- `docs/results/qualcomm/qnn-candidates.md` — dated correction note appended:
  the T22 records keep the superseded selector as history; no submission
  should reuse it.
- `ai/tasks/learning_lane.yaml`, `ai/tasks/status.generated.md`,
  `build/learning/*.html` — mechanical rebuild after the doc edits changed
  learning-sheet source hashes.

This log also covers the 2026-08-04 working-tree changes it reconciles: the
target-config os correction with its evidence notes, the
`_observed_preflight_count` derivation in `workbench.py` (top-level
`jobs_submitted`/`service_contacted` are now folded from per-request preflight
observations rather than stamped as literals, with tests), the `qualcomm`
extra pinning `qai-hub==0.53.0`, and the committed device-query record
`t31-device-query-2026-08-04.json`.

## Verification

- Command: `uv run python -m pytest tests/ -q`
- Result: 1001 passed, 31 skipped, 0 failed.
- Command: `PYTHONPATH=src uv run python scripts/qualcomm/plan_workbench_run.py --check`
- Result: ok; 24 plan entries, jobs_submitted 0, service_contacted false.
- Command: `python3 scripts/ai/render_task_status.py --check`
- Result: task graph valid; generated status current.
- Command: `python3 scripts/repo/check_hygiene.py --all`
- Result: passed for 383 tracked and untracked public files.

## Decisions and evidence

- The T22 package records under `results/manifests/qnn/packages/` keep the
  superseded `Windows 11` selector as the historical record of what T22
  requested. Rewriting them would fabricate history; the divergence proof in
  the tests documents the relationship instead.
- The run plan record keeps its `t31-workbench-run-plan-2026-08-04.json`
  filename (it is the module-level `RECORD_NAME` referenced across the docs);
  `run_observation.created_at_utc` truthfully dates the 2026-08-06 rebuild,
  and `--check` excludes the observation block by design.
- The superseded os value in the tests is read from the device-query evidence
  record rather than hardcoded, so the evidence file and the tests cannot
  silently disagree.

## Risks and limitations

- All request ids remain evidence about this repository's own contract only;
  Qualcomm AI Hub has never seen any of these requests. The corrected selector
  resolves in an authenticated device query, but no compile job has been
  submitted with it.
- The single-file `source_artifact.path` caveat from the T22 records is
  unchanged and remains the first-failure hypothesis.

## Follow-up

- Newly unblocked tasks: none; T31 remains the active front.
- Recommended next action: submit the first plan entry (X Elite / S128 /
  prefill / compile, `t30-compile-4002ded9a30ed87a692c`) once submission
  authorization is given in the submitting session.
