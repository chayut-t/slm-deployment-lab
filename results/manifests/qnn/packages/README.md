# QNN candidate package records

One `S<context>.json` per packaged variant, written by
`scripts/qualcomm/package_qnn_candidate.py`. Each record is deliberately
path-free: logical names, SHA-256 digests, byte sizes, compile input specs,
the target selector, the runtime identity, the option string, and the
deterministic request id. The packages themselves live under
`$SLM_LAB_ARTIFACT_ROOT/onnx/qnn-package/T22/` and are never committed. The
generated compile requests carry machine-local paths and stay under
`.ai-local/`.

Re-verify a record against an assembled package with `--check`. It re-reads
every package member, re-derives the record from the candidate manifest and
the target selector, and re-runs the offline request preflight.

## What a record does and does not claim

The package layout for an external-data ONNX model has not been verified
against the Qualcomm AI Hub service. No compile job was submitted and no
service call was made. T31 owns the first real submission. Ready for
submission means exactly three things: the candidate and sidecar digests were
re-verified against the committed T22 manifest, a compile request was
generated, and that request was accepted by the committed T30 adapter's own
validation. It does not mean AI Hub accepted it.

No record here contains a compile result, a device result, an accelerator
placement, or a latency number.
