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
about 8.9 GB of storage. Only these compact summaries are committed.

## `parity/` — ORT CPU multi-step parity evidence

`results/graph/parity/` is the committed home of the `ParityEvidence` JSON that
`slm_lab.backends.onnx_cpu` writes. It holds one record per context variant —
`S{128,512,1024,4096}-ort-cpu.json` — each measured on 2026-08-02 against the
committed reference graphs with `onnxruntime` 1.28.0 on the CPU execution
provider, and each carrying `evidence_tier="real_onnxruntime_cpu"`.

Two caveats travel with these records. They were taken at `ORT_ENABLE_BASIC`
rather than the runner's `ORT_DISABLE_ALL` default, because the float16
reference prefill graphs cannot be loaded at that level; and every record ends
`passed: false` with `failure_kinds: ["numerical_tolerance"]`, because the
proposed tolerances remain `proposed_unvalidated` and were not widened to fit
the measurement. The cache invariants pass on all 20 steps. See
`docs/results/onnx/ort-cpu-parity.md` for the numbers and what they license,
and `docs/failures/runtime/2026-08-02-t20-fp16-prefill-pad-unloadable.md` for
the load failure. `T23` re-measures at `ORT_DISABLE_ALL` after re-export.

Unlike the inspection reports above, a parity record *is* a numerical claim,
made by a named runtime on a named host, and it carries its own
`evidence_sha256` over a canonical payload that contains no absolute host path.
