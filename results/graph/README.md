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
`slm_lab.backends.onnx_cpu` writes, starting with
`results/graph/parity/S128-ort-cpu.json`. **No such record exists yet, and the
directory has not been created:** no ONNX Runtime has ever been run against
these graphs. Producing the first one — which is also what confirms or replaces
the proposed tolerances — is described in `docs/results/onnx/ort-cpu-parity.md`.

Unlike the inspection reports above, a parity record *is* a numerical claim,
made by a named runtime on a named host, and it carries its own
`evidence_sha256` over a canonical payload that contains no absolute host path.
