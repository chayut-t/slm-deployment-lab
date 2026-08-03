# Deployment adapters

Compiler, hosted-job, Device Cloud, packaging, and artifact-retrieval adapters
belong here, separated by platform.

## `qualcomm/packaging.py` — QNN candidate packaging (T22)

`slm_lab.deployment.qualcomm.packaging` turns one committed T22 candidate
manifest (`results/manifests/qnn/S<context>.json`) into a Workbench-ready
package and a validated compile request. It submits nothing, imports nothing
from `qai_hub`, and makes no network call.

What it does, in order:

1. Reads the candidate manifest and the committed target selector, and
   validates the selector through the T30 request validators so a bad device,
   runtime, or option string fails before any large file is read.
2. Resolves the candidate graph and its external-data sidecar under
   `$SLM_LAB_ARTIFACT_ROOT` and re-verifies every SHA-256 and byte size
   against the manifest.
3. Assembles `<artifact-root>/onnx/qnn-package/T22/<variant>/<graph-kind>/`
   containing the `.onnx`, its `.onnx.data` sidecar, and a `SHA256SUMS` file
   in `sha256sum` format. Hardlinking is preferred over copying; the record
   states which was used per file. A hardlinked member is proven identical by
   its inode, a copied member is re-hashed after placement.
4. Derives the compile `input_specs` from the manifest's recorded input
   tensors. Prefill contributes three `int64` boundary tensors; decode
   contributes 60, of which 56 are the `[1, 8, C, 128]` float16 cache tensors
   named `key_cache.<layer>` and `value_cache.<layer>`.
5. Writes a path-free, committable record to
   `results/manifests/qnn/packages/S<context>.json`.
6. Generates the AI Hub compile request into private storage only
   (`.ai-local/profiles/T22/` by default) and validates it with
   `ai_hub.preflight_compile_request`.

One package directory per graph kind is deliberate. An ONNX external-data
`location` is resolved relative to the directory holding the `.onnx`, so the
sidecar must keep the exact name the graph references and must sit beside it.
A nested or absolute `location` is rejected rather than flattened, because
flattening would silently break the reference the graph carries.

### What "ready for submission" means here

The package layout for an external-data ONNX model has not been verified
against the Qualcomm AI Hub service. No compile job was submitted and no
service call was made. T31 owns the first real submission. Ready for
submission means exactly three things: the candidate and sidecar digests were
re-verified against the committed T22 manifest, a compile request was
generated, and that request was accepted by the committed T30 adapter's own
validation. It does not mean AI Hub accepted it.

Two consequences follow, and both are recorded in the package record rather
than assumed away:

- The compile request names only the `.onnx` file, because the committed T30
  adapter requires `source_artifact.path` to be one existing file. Whether the
  service reads the sidecar from the same directory, or requires a directory
  or an archive instead, is unverified.
- No compile result, device result, placement, or latency number is produced
  or implied by anything in this module.

## `qualcomm/ai_hub.py` — offline compile preflight

`ai_hub.preflight_compile_request(path)` runs the committed compile
validation chain against a request file and returns the sanitized public
projection plus the deterministic `request_id` that `run_compile` would record
in its manifest. It reuses the same private validators the stage runner uses,
in the same order, so a request it accepts is a request `run_compile` accepts.
It constructs no backend, imports no client, and contacts no service.
