# T30 handoff: Qualcomm AI Hub stage adapters

Date: 2026-07-25
From: T30
To: T31 and T72

## Stable interface

Run `scripts/qualcomm/compile.py`, `inference.py`, and `profile.py` as separate
processes with `PYTHONPATH=src`. Each takes `--request` and `--manifest`.
Request files are private because they contain machine-local paths. The
manifest is sanitized and may be committed after task-specific evidence
review.

Inference and profile require:

1. The sanitized successful compile-manifest path.
2. The downloaded compiled model path.
3. A logical name and SHA-256 matching `result.target_artifact` in that
   compile manifest.

They never require an AI Hub job ID, URL, or live compile-job object. The
request contract and commands are documented in
`scripts/qualcomm/README.md`.

The current contract is schema v2. It is intentionally incompatible with
schema-v1 request and manifest files, so downstream work must regenerate all
three requests and rerun compile before inference or profile. Runtime evidence
now separates requested identity, submitted identity/SDK option, artifact
metadata, and unobserved execution identity.

## Private artifact layout

Keep requests, raw profiles, and unsanitized service material under
`.ai-local/profiles/TNN/`. Keep large/downloaded model and inference artifacts
under `SLM_LAB_ARTIFACT_ROOT` or ignored `artifacts/`. The adapter rejects raw
or binary outputs elsewhere inside the repository.

Public manifests retain source, predecessor-manifest, output, and raw-profile
SHA-256 values. They never retain local paths, credentials, accounts, service
job identity, job URLs, or raw warning text.

## First real-run checklist

- Use an exact installed `qai-hub` version and exact QAIRT version in the
  request. T30 mocked `qai-hub==0.53.0` and QAIRT
  `2.45.0.260326154327`, matching T02 evidence.
- Use `runtime.name: "QAIRT"` and put the exact QAIRT version in `options`
  once. With pinned `qai-hub==0.53.0`, compile uses `--qairt_version`;
  inference and profile use `--qairt_framework`.
- Use only the documented stage allowlists in the request guide. Credential,
  account, identity, model/path, unknown, and misspelled flags fail closed,
  including prefixed and underscore variants. Add a reviewed regression before
  extending an allowlist.
- Coordinate the `qai_hub_submission` resource lock and use a free bounded
  submission only.
- Set `retry` to false and a positive `timeout_seconds`.
- Verify every input SHA-256 before running the command.
- Capture request and raw paths privately. Do not capture client output through
  a separate wrapper.
- Inspect the public manifest's requested device selector separately from the
  service-reported job device. Do not require equality when a compatible
  family or successor device is intentional.
- Inspect requested runtime options, artifact runtime metadata, and observed
  execution runtime as separate evidence. A null observed execution runtime
  must not be promoted to the requested or artifact version.
- Inspect service-turnaround boundaries, hashes, units, placement, and privacy
  flags.
- Treat service turnaround as queue/provisioning time, never graph latency.
- If the exact client returns a different documented profile shape, preserve
  the private raw hash and add a focused mocked normalizer regression before
  changing public evidence.

## Known boundary

T30 submitted no external job and reports no hardware result. T31 owns the
first bounded production-backend smoke, real target evidence, numerical
validation, and any compatible adapter adjustment. T72 may reuse the same
scripts but must provide secrets through the approved CI secret store and
publish only the sanitized manifest.
