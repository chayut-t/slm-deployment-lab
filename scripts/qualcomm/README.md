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

All requests use `schema_version: 2`, name the exact client, set
`runtime.name` to `QAIRT`, provide an exact runtime version, set `retry: false`,
and provide a bounded `timeout_seconds`. Schema-v1 requests and predecessor
manifests are intentionally incompatible and must be regenerated. `device` is
an SDK selector with `name` and optional `os`/`attributes`; it is not published
as an observed physical-device identity. Every input artifact has a private
`path`, a path-free `logical_name`, and its lowercase SHA-256. Output artifacts
and raw profiles must be external or below ignored `.ai-local/` or
`artifacts/` storage.

Options are parsed before backend initialization using small, stage-specific
allowlists. Unknown, positional, short-form, misspelled, valueless, URL-like,
email-like, path-like, or private-looking options fail closed. This includes
credential/account/identity prefixes such as `--access-token`,
`--access_token`, `--billing-account`, and path-bearing `--model`. Values are
never copied to an error. Filesystem paths belong only in the private,
purpose-specific request fields.

For the pinned `qai-hub==0.53.0` interface, compile must bind the requested
runtime version exactly once with `--qairt_version`. Inference and profile must
bind it exactly once with `--qairt_framework`. The manifest records the
submitted framework name, version, and option separately from target-model
runtime metadata and leaves execution runtime unobserved. Supported option
flags are deliberately narrow:

- compile: `--target_runtime`, `--qairt_version`, and `--qnn_options`
- inference/profile: `--qairt_framework`, `--compute_unit`, and
  `--qnn_options`

At schema v2, `--qnn_options` accepts only the reviewed
`context_enable_graphs=<logical-name>` sub-option. Extend an allowlist only
with a documented use case and an adversarial regression.

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
contract. Each manifest separates the requested device selector from the
service job's reported device, and it does not require them to match. This
preserves legitimate compatible/successor-device reuse. Runtime fields
separate the requested identity, submitted name/version/SDK option,
target-model metadata when exposed, and the unobserved execution-runtime
boundary; an exact execution runtime is never inferred from request text
alone.

Do not commit request files: they contain machine-local paths. Do not redirect
raw SDK output into public files, run `qai-hub configure` in task logs, or use
job URLs/IDs as cross-stage state.

## Generating a compile request from a T22 QNN candidate

`package_qnn_candidate.py` builds the package a compile job would consume and
generates the compile request for it. It never contacts the service, never
imports `qai-hub`, and never submits a job, so it runs in the common
environment:

```bash
SLM_LAB_ARTIFACT_ROOT=<external-root> PYTHONPATH=src python3 \
  scripts/qualcomm/package_qnn_candidate.py \
  --manifest results/manifests/qnn/S128.json

SLM_LAB_ARTIFACT_ROOT=<external-root> PYTHONPATH=src python3 \
  scripts/qualcomm/package_qnn_candidate.py \
  --manifest results/manifests/qnn/S128.json --check
```

Defaults: target selector
`configs/targets/qualcomm-snapdragon-x-elite-crd.json`, package root
`$SLM_LAB_ARTIFACT_ROOT/onnx/qnn-package/T22`, committed record
`results/manifests/qnn/packages/S<context>.json`, and requests under
`.ai-local/profiles/T22/`. `--graph prefill|decode|all`, `--artifact-root`,
`--package-root`, `--record`, and `--request-dir` override them. A request
directory inside the public repository tree is refused by the same rule that
governs stage output artifacts.

The committed record carries logical names, digests, byte sizes, input specs,
the target selector, the runtime identity, the option string, and the request
id. It never carries a filesystem path. The generated request does carry
machine-local paths and therefore stays private, exactly as above.

`--check` re-verifies an assembled package against its committed record. It
re-reads every package member, re-derives the record from the candidate
manifest and the target selector, regenerates the request into private
storage, and re-runs the preflight. It never relinks, copies, or rebuilds
anything.

### The honest boundary

The package layout for an external-data ONNX model has not been verified
against the Qualcomm AI Hub service. No compile job was submitted and no
service call was made. T31 owns the first real submission. Ready for
submission means exactly three things: the candidate and sidecar digests were
re-verified against the committed T22 manifest, a compile request was
generated, and that request was accepted by the committed T30 adapter's own
validation. It does not mean AI Hub accepted it.

The compile request names only the `.onnx` file, because `source_artifact.path`
must be one existing file. Whether the service reads the `.onnx.data` sidecar
from the same directory, or requires a directory or an archive instead, is
unverified. Nothing here establishes a compile result, a device result, or a
latency number.

The offline validation itself is `ai_hub.preflight_compile_request`, which
runs the full committed compile validation chain — schema version, stage,
public-safety projection, field set, client version, device selector, runtime,
option allowlist, timeout, retry, source-artifact existence and digest, input
specs, and output-path policy — and returns the same `request_id` the compile
stage would record. It builds no backend and touches no network.
