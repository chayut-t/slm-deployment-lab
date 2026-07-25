# Qualcomm AI Hub Workbench stage adapters

These three commands deliberately run compile, inference, and profile as
separate processes:

```bash
PYTHONPATH=src python3 scripts/qualcomm/compile.py \
  --request .ai-local/profiles/T30/compile-request.json \
  --manifest results/processed/qualcomm/compile-manifest.json

PYTHONPATH=src python3 scripts/qualcomm/inference.py \
  --request .ai-local/profiles/T30/inference-request.json \
  --manifest results/processed/qualcomm/inference-manifest.json

PYTHONPATH=src python3 scripts/qualcomm/profile.py \
  --request .ai-local/profiles/T30/profile-request.json \
  --manifest results/processed/qualcomm/profile-manifest.json
```

The common environment does not install the optional `qai-hub` client. Use an
isolated environment with the exact `client_version` named in the request.
The authenticated T02 environment used `qai-hub==0.53.0`; later tasks must
record and deliberately update the request if they validate another version.

## Request contract

All requests use `schema_version: 1`, name the exact client/runtime/device, set
`retry: false`, and provide a bounded `timeout_seconds`. Every input artifact
has a private `path`, a path-free `logical_name`, and its lowercase SHA-256.
Output artifacts and raw profiles must be external or below ignored
`.ai-local/` or `artifacts/` storage.

Compile additionally provides:

- `stage: "compile"`
- `source_artifact`
- `input_specs` mapping tensor names to positive `shape` dimensions and exact
  `dtype`
- `output_artifact` and `output_logical_name`

Inference additionally provides:

- `stage: "inference"`
- `predecessor_manifest` pointing to the sanitized compile manifest
- `compiled_artifact`, whose logical name and digest must match that manifest
- `input_dataset` for an AI Hub-compatible HDF5 dataset
- `output_artifact` and `output_logical_name`

Profile additionally provides:

- `stage: "profile"`
- the same compile predecessor and compiled-artifact checks
- `raw_profile_output` and `raw_profile_logical_name`

The SDK is always non-verbose and all SDK stdout/stderr is captured and
discarded. Printed summaries and public manifests never contain filesystem
paths, service job IDs/URLs, tokens, accounts, raw responses, or warning text.
Stage lineage uses request, predecessor-manifest, and artifact SHA-256 values.
Service turnaround is explicitly not device latency. Profile time is
normalized to microseconds and memory to bytes per the AI Hub profile
contract.

Do not commit request files: they contain machine-local paths. Do not redirect
raw SDK output into public files, run `qai-hub configure` in task logs, or use
job URLs/IDs as cross-stage state.
