# T20: ONNX Export Matrix

Date: 2026-07-30
Task: `T20`
Visibility: `public`
Status: completed

## Outcome

T20 exported the pinned Qwen3-0.6B model into eight reference ONNX graphs:
static prefill and one-token decode for S128, S512, S1024, and S4096. Every
graph passed ONNX checker validation and exact public name, dtype, and shape
conformance against the frozen T12 contract.

The large protobuf tensor payloads remain under
`SLM_LAB_ARTIFACT_ROOT/onnx/reference/T20/`. Four committed context manifests
record exact commands, source/model/toolchain revisions, serialized T12
contracts, graph hashes, external-data hashes, and explicit evidence
boundaries. No compiler, runtime-parity, accelerator-placement, or performance
claim is made.

## Changes

- Added a deterministic `torch.onnx.export` workflow using the exact T10
  prefill token IDs and T12-generated graph contracts.
- Added prefill wrapping that returns last-token FP32 logits, 28 pairs of
  fixed-capacity FP16 GQA cache tensors, and an explicit valid length.
- Added decode wrapping that converts explicit cache tensor inputs into the
  Transformers 4.51.3 `DynamicCache` protocol, slices the valid prefix,
  performs one model step, and scatters only the new slice back into each
  fixed-capacity output cache.
- Forced initializers larger than 1,024 bytes into ONNX external data and
  rejected missing, unsafe, non-external, or oversized-inline payloads.
- Added resumable export behavior: an existing graph is accepted only after
  checker, contract, and external-data validation and is never overwritten.
- Added focused configuration, fixture, cache-transition, real ONNX export,
  overwrite-protection, committed-manifest, and shape regression tests.
- Added four manifests under `results/manifests/onnx/`; each identifies
  exporting source commit `631fd70bcff9b73b81c08a2a2e0127cad07f09ca`.

## Verification

- `uv pip install --python /private/tmp/slm-t12-venv/bin/python
  onnx==1.18.0 jsonschema==4.23.0 pyyaml==6.0.2`
  - Prepared an isolated exact-version T20 verification runtime alongside the
    pinned T11/T12 Torch 2.7.1, Transformers 4.51.3, and Safetensors 0.8.0
    packages.
- For each context in `128 512 1024 4096`:

  ```bash
  HF_HOME=/Volumes/T9/slm-deployment-lab/hf-cache \
  TRANSFORMERS_OFFLINE=1 \
  SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab \
  PYTHONPATH=src \
  /private/tmp/slm-t12-venv/bin/python \
    -m slm_lab.export.onnx_matrix export --context CONTEXT
  ```

  - Produced and validated all eight real pinned-Qwen graph pairs.
  - S4096 eager CPU prefill graph capture took approximately eleven minutes;
    this observation is not a benchmark result.
- For each context, ran the matching `validate --context CONTEXT
  --write-manifests` command.
  - Re-ran ONNX checker and static I/O checks, hashed every graph/data file,
    validated the artifact schema, and wrote the four public manifests.
- `PYTHONPATH=src /private/tmp/slm-t12-venv/bin/python -m pytest -q
  tests/export tests/contracts`
  - Passed: `15 passed, 1 skipped`; the skip is the explicit real-Qwen T12
    numerical gate, which T12 already completed.
- `PATH=/Users/chayut/projects/slm-deployment-lab/.venv/bin:$PATH
  PYTHONPATH=src /Users/chayut/projects/slm-deployment-lab/.venv/bin/python
  -m pytest -q`
  - Passed after T20 completion made T21 ready in the generated graph.
- `/Users/chayut/projects/slm-deployment-lab/.venv/bin/ruff check src tests`
  - Passed.
- `python3 scripts/ai/render_task_status.py --check`
  - Passed after completion metadata regeneration.
- `python3 scripts/repo/check_hygiene.py --all`
  - Passed.

## Decisions and evidence

- The pinned source `model.safetensors` hash is
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- Every external ONNX data file is 1,192,085,504 bytes with SHA-256
  `e9d4b051fa86283dc96a29ceb4eb99107dbe8aff1036e54628e8725e3dac5cde`.
  The repeated digest is expected because the eight reference graphs use the
  same pinned FP16 model initializers; graph structure and static constants
  remain in separately hashed protobufs.
- Prefill protobuf size grows from 1,560,358 bytes at S128 to 35,209,213 bytes
  at S4096 because the legacy static export materializes more context-shaped
  graph constants. Decode protobufs are 1,759,947 bytes each but have distinct
  hashes because their fixed cache capacities differ.
- The first real decode attempt exposed that Qwen3 in Transformers 4.51.3
  rejects legacy cache tuples. Constructing `DynamicCache` inside the wrapper
  preserves the model's required Python protocol without changing the
  tensor-only public ONNX contract.
- Public graph validation requires every T12 dimension to be a positive
  `dim_value`; symbolic public shapes are rejected even when tracing could
  infer them at runtime.

## Risks and limitations

- T20 validates export structure and content identity, not numerical runtime
  behavior. T21 must compare multi-step ONNX Runtime outputs against T11 and
  inspect dynamic internal slices and other deployment risks.
- Transformers and Torch emit trace warnings for tensor-dependent Python
  conditions inside `DynamicCache` and Qwen causal-mask code. The public
  boundary is static and checked, but T21 must verify that internal
  valid-length behavior remains correct across multiple decode positions.
- External data files are intentionally not committed and must be copied or
  regenerated from the exact commands before use on another machine.
- Eight graph packages currently store identical external weight payloads
  separately. Content hashes make safe storage-level deduplication possible,
  but no portability-sensitive symlink or hard-link scheme is imposed.
- ONNX Runtime execution, Qualcomm compiler acceptance, QNN transformation,
  hardware placement, latency, and memory remain unverified by design.

## Follow-up

- Newly unblocked tasks: T21 and T40.
- Recommended next action: T21 should consume the manifests, verify every
  external hash before loading, run multi-position decode parity in ONNX
  Runtime CPU, and inspect whether `valid_length` remains a live internal
  slice/scatter dependency rather than a traced constant.
