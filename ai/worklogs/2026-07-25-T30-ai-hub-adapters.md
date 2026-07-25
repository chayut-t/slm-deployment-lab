# T30: AI Hub compile, inference, and profile adapters

Date: 2026-07-25
Task: `T30`
Visibility: `public`
Status: completed

## Outcome

Implemented safe Qualcomm AI Hub Workbench adapters for compile, inference, and
profile as three independently runnable processes. Each stage begins from
content-addressed local artifacts, emits a sanitized public manifest, and
keeps SDK output, service identity, filesystem paths, raw responses, and
credentials private. Inference and profile restart from a compile manifest and
downloaded target model without depending on an in-memory job or committed job
ID.

## Changes

- Added a dependency-injected adapter module with an optional production
  `qai-hub` backend, exact client/runtime version checks, bounded waits,
  single-attempt submission, input/lineage validation, and fail-closed errors.
- Added capture around client initialization and every SDK operation. Any SDK
  stdout/stderr is discarded, and external exception messages are replaced
  with stable sanitized errors so job URLs, IDs, tokens, accounts, or raw
  service details cannot reach console output.
- Added versioned stage manifests containing public request IDs, exact target
  and runtime identity, options, service-turnaround boundaries, predecessor
  and artifact SHA-256 lineage, byte sizes, and explicit privacy assertions.
  Local filesystem paths and service job identities are excluded.
- Added compile normalization for source/input contracts and target tensor
  specs; inference normalization for output-artifact traceability; and profile
  normalization for latency, memory, load, sample ranges, compute-unit
  placement, warning counts, and documented units.
- Added private-output enforcement: compiled targets, inference outputs, and
  raw profiles must be external or under ignored `.ai-local/`/`artifacts`
  storage. Raw profiles are linked to normalized manifests by SHA-256.
- Added separate compile, inference, and profile scripts plus a request/lifecycle
  guide for T31 and T72.
- Added mocked tests covering stage independence, disk-based restart,
  traceability, exact versions, target tensor specs, flat and nested profiles,
  private raw/public normalized evidence, path policy, malformed results,
  digest mismatches, SDK output capture, service-error suppression, and
  malicious request text.

## Verification

- Command:
  `PYTHONPATH=src python3 -m unittest tests.deployment.qualcomm.test_ai_hub -v`
- Result: 13 focused mocked adapter tests passed.
- Command:
  `PYTHONPATH=src <primary-checkout>/.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v`
- Result: 96 tests passed with two intentional opt-in upstream/tokenizer
  checks skipped.
- Command:
  `<primary-checkout>/.venv/bin/ruff check src tests scripts/qualcomm`
- Result: passed.
- Commands: direct `--help` invocation for all three scripts and
  `PYTHONPYCACHEPREFIX=<writable-temp>/slm-lab-T30-pycache PYTHONPATH=src python3 -m compileall -q src/slm_lab/deployment/qualcomm scripts/qualcomm tests/deployment/qualcomm`
- Result: all entry points parsed independently and all new Python compiled.
- Commands: `python3 scripts/ai/render_task_status.py --check`,
  `python3 scripts/repo/check_hygiene.py --all`, and `git diff --check`.
- Result: all passed after task completion metadata was generated.

## Decisions and evidence

- Qualcomm's documented API accepts local model paths for all three jobs and
  local HDF5 paths for inference, so a downloaded compiled artifact is a
  sufficient cross-process boundary. Public service IDs are unnecessary for
  normal operation and intentionally excluded from the contract.
- The public request ID is derived from canonical sanitized request content;
  predecessor manifests and every source/result artifact retain their full
  SHA-256. This preserves raw-to-normalized traceability without service
  identity.
- Both current documented profile layouts are accepted: flat keys and keys
  under `execution_summary`. Missing latency or peak-memory evidence and
  malformed samples fail closed. Times are normalized as microseconds and
  memory as bytes.
- Warning text is not a stable or privacy-safe public field. The normalized
  record preserves its count while the raw private profile retains the
  original text and is content-addressed.
- SDK retry is required to be false and each wait has a positive timeout. This
  gives every stage one bounded submission attempt rather than an indefinite
  retry loop.
- No Workbench or paid job was submitted. T30 validates the adapter contract
  and mocked behavior; it does not publish hardware measurements.

## Risks and limitations

- The optional `qai-hub==0.53.0` environment was not imported or authenticated
  in this worktree because T30 was explicitly offline. T31 must run a bounded
  free smoke lifecycle before relying on the production backend and record any
  client-format drift.
- AI Hub job artifacts and raw profile structures are not stable APIs. The
  adapter uses only documented summary fields and rejects missing essentials;
  T31 may need a task-scoped normalizer extension when it captures sanitized
  evidence from the exact pinned client.
- Public manifests prove adapter behavior and content lineage, not Qwen
  compiler support, numerical parity, NPU placement, or device performance.

## Follow-up

- Newly unblocked tasks: T72. T31 still requires T22 in addition to T30.
- Recommended next action: after T22 completes, prepare private per-stage
  request files under `.ai-local/profiles/T31/`, run one bounded free compile
  smoke, and inspect the sanitized manifest before inference/profile.
- Learner debrief:
  - [ ] Trace a source SHA through compile, inference, and profile in
    [`ai_hub.py`](../../src/slm_lab/deployment/qualcomm/ai_hub.py) and the
    [stage guide](../../scripts/qualcomm/README.md).
  - [ ] Explain why the compile-manifest SHA replaces a public service job ID
    in the independently runnable inference/profile stages.
  - [ ] Compare the private raw-profile boundary with the allowlisted
    normalized profile and identify every unit.
  - [ ] Reproduce one mocked lifecycle from
    [`test_ai_hub.py`](../../tests/deployment/qualcomm/test_ai_hub.py).
  - [ ] Review related notebooks `04_ai_hub_pipeline.ipynb` and
    `05_qnn_profile_analysis.ipynb` when T80 creates them; they do not yet
    exist and this study checkpoint remains learner-owned.
