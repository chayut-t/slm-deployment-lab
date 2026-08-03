# Graph inspection reports

One JSON report per T20 context variant: `S128.json`, `S512.json`,
`S1024.json`, `S4096.json`. Each holds the prefill and decode structural
summary — node count, operator histogram, boundary tensor counts, non-static
dimensions, initializer metadata — plus the ranked deployment-risk findings for
that variant, together with the source graph SHA-256, the T20 manifest digest,
and the risk-catalogue digest the report was produced against.

Produced by `slm_lab.graph.inspection` (task T21) from the committed manifests
under `results/manifests/onnx/`, scored against the declarative catalogue at
`configs/graph/onnx-risk-rules-v1.json`. The reader-facing interpretation is
`docs/results/onnx/graph-inspection.md`.

These are structural reads of the ONNX protobuf. No compiler, runtime, or
device is involved, and no numerical claim is made.

Regenerate and verify:

```bash
SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab PYTHONPATH=src \
  python -m slm_lab.graph.inspection --all-manifests

SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab PYTHONPATH=src \
  python -m slm_lab.graph.inspection --all-manifests --check
```

`--check` re-hashes every graph against its manifest and exits non-zero if a
committed report would change.

Each report opens with a `claim_boundary` block listing what it does and does
not establish, mirroring the block T20 writes into
`results/manifests/onnx/S*.json`. It is in-band on purpose: a reader who only
ever sees this JSON — and therefore only ever sees `"severity": "blocking"` next
to a rule id — needs to know from the file itself that severities are review
judgements and that no compiler was run.

The eight `.onnx` graphs and their eight external-data sidecars stay under the
external artifact root and are never committed. There is one
1,192,085,504-byte `.onnx.data` file *per graph*, not one shared file; the eight
are byte-identical and therefore share a single SHA-256, but the set still needs
about 8.9 GB of storage (8 × 1,192,085,504 = 9,536,684,032 bytes). Only these
compact summaries are committed.

Neither figure moved with the `T23` re-export: the sidecars are byte-identical
before and after, which is the whole reason the export attestation can keep
recording one shared `external_data_sha256`. What did move is the total of all
sixteen files, which `results/quantization/t40-baseline-parity-2026-08-02.json`
records as `recorded_total_bytes`: 9,586,211,364 → 9,626,186,972, all of it in
the four prefill protobufs.

## `parity/` — ORT CPU multi-step parity evidence

`results/graph/parity/` is the committed home of the `ParityEvidence` JSON that
`slm_lab.backends.onnx_cpu` writes. It holds one record per context variant —
`S{128,512,1024,4096}-ort-cpu.json` — each measured on 2026-08-02 against the
committed reference graphs with `onnxruntime` 1.28.0 on the CPU execution
provider, and each carrying `evidence_tier="real_onnxruntime_cpu"`.

All four are taken at the runner's `ORT_DISABLE_ALL` default, on both the
prefill and the decode session, so they measure the exported graph rather than
ONNX Runtime's fusion choices. That was not possible before `T23`: the float16
prefill graphs could not be loaded at that level until their cache write was
re-exported from a `Pad` to a `Concat`. Every record ends `passed: true` with
an empty `failures[]`, against a `tolerance.status` of `derived_and_measured…`
— thresholds derived from reference and candidate dtype ULP, logit scale and
layer depth, with no observed candidate error used to set any of them. The
cache invariants pass on all 20 steps. See
`docs/results/onnx/ort-cpu-parity.md` for the numbers, the derivation, and what
they license, and
`docs/failures/runtime/2026-08-02-t20-fp16-prefill-pad-unloadable.md` for the
load failure that made the re-export necessary.

`parity/diagnostics/` holds the records that justify the tolerance rather than
apply it: a reference-against-itself run at float32, bfloat16 and float16 with
no ONNX Runtime session anywhere, and a float16-reference parity probe. They
are **not** T21 parity records and must never be read as one; each says so in
its own `record_kind`, which only reads `t21_ort_cpu_parity` for the four files
above. See the README in that directory.

Unlike the inspection reports above, a parity record *is* a numerical claim,
made by a named runtime on a named host, and it carries its own
`evidence_sha256` over a canonical payload that contains no absolute host path.
