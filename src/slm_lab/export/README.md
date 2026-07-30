# Model export

Static ONNX export, external-data handling, graph variants, and artifact
manifest generation belongs here.

`onnx_matrix.py` exports the pinned Qwen3-0.6B model as four fixed-shape
prefill/decode pairs generated directly from the T12 contract. The exporter:

- uses the exact T10 token workload as each prefill tracing input;
- keeps 8-head FP16 GQA caches explicit at every graph boundary;
- performs one dynamic-index decode cache update without changing the fixed
  public cache shapes;
- forces model initializers into ONNX external data;
- runs the ONNX checker and exact name/dtype/shape conformance checks; and
- validates the configured T10 bundle against its frozen canonical digest and
  every context workload's prompt/token hashes; and
- records graph, data-file, source-weight, toolchain, host, contract, export
  source/config, and T10 input hashes in four commit-safe manifests.

The graph protobufs and data shards are deliberately excluded from Git. Run
one context with the pinned isolated environment:

```bash
HF_HOME=/Volumes/T9/slm-deployment-lab/hf-cache \
TRANSFORMERS_OFFLINE=1 \
SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab \
PYTHONPATH=src \
/path/to/python -m slm_lab.export.onnx_matrix export --context 128
```

After exporting both graphs, validate and write the context manifest:

```bash
HF_HOME=/Volumes/T9/slm-deployment-lab/hf-cache \
SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab \
PYTHONPATH=src \
/path/to/python -m slm_lab.export.onnx_matrix validate \
  --context 128 --write-manifests
```

Plain `validate` is intentionally stronger than schema validation. It
re-hashes the external graph/data files, proves the historical exporter commit
exists in the current branch ancestry, checks the exporter/config/model/T10
Git blobs from that commit, reconstructs every deterministic manifest field,
and rejects any claim or evidence drift. `created_at` is the only
run-generated manifest field reused during reconstruction.

Export evidence proves only that the pinned host toolchain produced
ONNX-checker-valid graphs whose public I/O matches T12 and whose external
files match the recorded hashes. T21 owns runtime numerical parity and graph
inspection; T22 owns compiler-oriented transformations. No manifest here is
evidence of compiler acceptance, accelerator placement, or performance.
