# T32: Device Cloud Baseline

Date: 2026-07-28
Task: `T32`
Visibility: `public`
Status: completed

## Outcome

Completed the engineering and narrow publication portions of the T32
Device Cloud baseline. A free interactive session produced valid multi-token
Qwen3-0.6B output through the ready-made GenieX/`llama.cpp` route, and the
validated record separates artifact access, model loading, tokenization,
prefill, first decode, remaining decode, generation total, and complete
request wall time.

The learner explicitly confirmed the timing, GenieX-versus-custom-QNN, and
private/public evidence-split debrief on 2026-07-28, completing the final T32
acceptance gate.

## Changes

- Added a Windows-native boundary probe, strict private-capture normalizer,
  immutable model provenance checks, semantic timing-source validation,
  evidence digests, and regression coverage.
- Preserved the frozen raw and sanitized evidence under ignored
  `.ai-local/profiles/T32/qdc-2026-07-28/`.
- Published the learner-approved generic reproducibility setup and aggregate
  latency measurements in `docs/results/qualcomm/device-cloud.md`.
- Withheld allocated-device evidence, observed placement proof, exact
  installed software versions, session/account identifiers, logs, manifests,
  and evidence digests.

## Verification

- `.venv/bin/python -m pytest -q
  tests/deployment/qualcomm/test_device_cloud.py`: 21 passed.
- `.venv/bin/python -m pytest -q`: 160 passed, 6 skipped.
- Focused Ruff lint and format checks: passed.
- `python3 scripts/ai/render_task_status.py --check`: task graph valid and
  generated status current.
- `python3 scripts/repo/check_hygiene.py --all`: passed for 226 tracked and
  untracked public files.
- Frozen private checksum ledger: all 16 entries verified.
- Regenerating the sanitized manifest from the frozen capture produced a
  byte-identical file.
- `git diff --check`: passed.
- PowerShell was not available on the macOS verification host, so the
  already-captured Windows probe was not rerun.
- Fresh independent post-publication review found two stale instructions to
  publish a normalized manifest or digest. Both were corrected; re-review
  approved with no remaining findings.

## Decisions and evidence

- The complete-request metric is the observed host wall from artifact open/map
  through synchronous generation completion. It includes instrumentation and
  orchestration overhead but excludes Device Cloud allocation, downloads, SSH
  transport, and interactive turnaround.
- The accepted capture is one run, not a distribution benchmark.
- Public Qualcomm terms and Device Cloud documentation did not provide express
  permission to publish the complete live-device record. After reviewing that
  research, the learner directed publication of generic setup information and
  aggregate latency values. This is recorded as the learner's decision, not
  as a legal determination or general publication license.
- The ready-made GenieX/`llama.cpp` GGUF route does not prove the custom
  QNN/QAIRT static-graph path.

## Risks and limitations

- The published latency values have no repeat count, variance, percentile, or
  repeatability claim.
- All detailed live-device evidence remains private and local.
- The accepted live result remains one capture rather than a repeated
  distribution benchmark.

## Follow-up

- Newly unblocked tasks: T33 has satisfied its T32 dependency but still waits
  for T31.
- Recommended next action: complete T31, then integrate its floating QNN/QAIRT
  artifacts with this Device Cloud generation evidence in T33.
