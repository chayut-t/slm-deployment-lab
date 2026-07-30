# T20: ONNX Export Matrix

Date: 2026-07-30
Task: `T20`
Visibility: `public`
Status: in progress — independent review fixes ready for rereview

## Outcome

T20 implementation exported the pinned Qwen3-0.6B model into eight reference
ONNX graphs: static prefill and one-token decode for S128, S512, S1024, and
S4096. Every graph passed ONNX checker validation and exact public name, dtype,
and shape conformance against the frozen T12 contract.

The large protobuf tensor payloads remain under
`SLM_LAB_ARTIFACT_ROOT/onnx/reference/T20/`. Four committed context manifests
record exact commands, source/model/toolchain revisions, serialized T12
contracts, graph hashes, external-data hashes, and explicit evidence
boundaries. No compiler, runtime-parity, accelerator-placement, or performance
claim is made.

The task remains `in_progress` and its task-graph worklog remains null while
the fifth independent-review remediation awaits fresh rereview.

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
- Independent review remediation now reconstructs every deterministic
  manifest field, validates the historical exporter commit and branch
  ancestry, and binds it to content hashes for exporter source, export
  configuration, model contract, T10 contract/source/bundle, and the exact
  context workload.
- Export configuration now resolves and supplies the tracing token fixture.
  The fixture must pass the complete offline T10 validation, the frozen
  canonical bundle digest, and explicit per-workload prompt/token hashes.
- Added adversarial tests for altered commits, chat-template hash, all
  recorded toolchain versions/settings, status, commands, input/cache
  summaries, claim boundaries, source provenance, artifact hashes, configured
  fixture path, stale workload hashes, and coherent token-content tampering.
- Second-rereview remediation adds an independent committed run attestation to
  the export configuration. It pins exporter commit `631fd70...` (the
  DynamicCache fix), actual Python `3.11.15`, source-weight hash, every graph
  hash, and the shared external-data hash rather than trusting values selected
  by each manifest.
- Added coherent-tamper regressions that change the manifest to the valid but
  pre-DynamicCache `14518d7...` ancestor with its matching source hash, and
  that change both Python fields to `9.9.9`. Both are rejected against the
  independent attestation.
- Third-rereview remediation makes the fixed tracked export configuration an
  immutable trust root: its exact bytes must match both a code-pinned SHA-256
  and the `HEAD` Git blob at
  `configs/models/qwen3-0.6b-onnx-export.json`.
- Alternate config paths and modified in-memory `ExportConfig` instances are
  rejected. Coherent config/manifest substitution regressions now cover every
  attested field: run ID, exporter commit, runtime Python, source weights,
  external data, and all graph hashes.
- Runtime verification now compares the actual interpreter version with the
  attested Python `3.11.15`, before package-version checks.
- Fourth-rereview remediation removes dataclass equality from the trust
  decision. It requires exact types for `ExportConfig`, every nested
  dataclass, tuple, path, and primitive, then compares an explicit canonical
  primitive serialization with a freshly parsed trusted configuration.
- The fixed config boundary now rejects symlink aliases as well as any
  lexically different path. A delegated equality-adapter regression confirms
  that nested objects cannot substitute attested values.
- Fifth-rereview remediation applies the same primitive-first boundary to
  manifests. Every key and value must use exact builtin JSON types before
  schema validation; actual and reconstructed manifests are then compared as
  canonical serialized bytes without caller-defined equality.
- Config loading now accepts only an exact builtin string containing the
  fixed absolute spelling before any `Path` construction. `Path` objects,
  dot segments, repeated separators, copied paths, and symlink aliases are
  rejected explicitly.

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
  - Passed: `16 passed, 1 skipped`; the skip is the explicit real-Qwen T12
    numerical gate, which T12 already completed.
- `HF_HOME=/Volumes/T9/slm-deployment-lab/hf-cache
  SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab PYTHONPATH=src
  /private/tmp/slm-t12-venv/bin/python -m
  slm_lab.export.onnx_matrix validate`
  - Passed: all four committed manifests matched freshly checked graph,
    external-data, source-weight, host, toolchain, and contract identities.
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

### Independent-review remediation verification

- `HF_HOME=/Volumes/T9/slm-deployment-lab/hf-cache
  SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab PYTHONPATH=src
  /private/tmp/slm-t12-venv/bin/python -m
  slm_lab.export.onnx_matrix validate`
  - Passed after reconstructing all four manifests and matching their
    historical Git blobs, T10 inputs, contracts, graph protobufs, and external
    data.
- `PYTHONPATH=src /private/tmp/slm-t12-venv/bin/python -m pytest -q
  tests/export tests/contracts`
  - `40 passed, 1 skipped`; the skip remains the explicit real-Qwen T12
    numerical gate already completed by T12.
- `PATH=/Users/chayut/projects/slm-deployment-lab/.venv/bin:$PATH
  PYTHONPATH=src /Users/chayut/projects/slm-deployment-lab/.venv/bin/python
  -m pytest -q`
  - `189 passed, 10 skipped, 1 failed`.
  - The sole failure is the known claim-baseline
    `GitSnapshotTests.test_staged_graph_requires_matching_staged_status`
    `StopIteration`: while T20, T51, and T72 are concurrently `in_progress`,
    the test fixture cannot find a planned task whose dependencies are all
    completed. The unrelated automation test was not changed; it passed on
    the earlier completion baseline and will become applicable again when a
    claimed ready task completes.
- `/Users/chayut/projects/slm-deployment-lab/.venv/bin/ruff check src tests`
  - Passed after review remediation.

### Third-rereview remediation verification

- `HF_HOME=/Volumes/T9/slm-deployment-lab/hf-cache
  SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab PYTHONPATH=src
  /private/tmp/slm-t12-venv/bin/python -m
  slm_lab.export.onnx_matrix validate`
  - Passed under the exact attested Python 3.11.15 runtime after enforcing the
    immutable config trust root.
- `PYTHONPATH=src /private/tmp/slm-t12-venv/bin/python -m pytest -q
  tests/export tests/contracts`
  - `48 passed, 1 skipped`; the skip remains the explicit real-Qwen T12
    numerical gate already completed by T12.
- `PATH=/Users/chayut/projects/slm-deployment-lab/.venv/bin:$PATH
  PYTHONPATH=src /Users/chayut/projects/slm-deployment-lab/.venv/bin/python
  -m pytest -q`
  - `197 passed, 10 skipped, 1 failed`.
  - The sole failure remains the concurrent-claim baseline
    `GitSnapshotTests.test_staged_graph_requires_matching_staged_status`
    `StopIteration`; no planned task has all dependencies completed while
    T20, T51, and T72 remain concurrently `in_progress`.
- `/Users/chayut/projects/slm-deployment-lab/.venv/bin/ruff check src tests`
  - Passed.

### Fourth-rereview remediation verification

- `HF_HOME=/Volumes/T9/slm-deployment-lab/hf-cache
  SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab PYTHONPATH=src
  /private/tmp/slm-t12-venv/bin/python -m
  slm_lab.export.onnx_matrix validate`
  - Passed after replacing equality-based trust comparison with strict type
    validation and canonical primitive serialization.
- `PYTHONPATH=src /private/tmp/slm-t12-venv/bin/python -m pytest -q
  tests/export tests/contracts`
  - `51 passed, 1 skipped`; the skip remains the explicit real-Qwen T12
    numerical gate already completed by T12.
- `PATH=/Users/chayut/projects/slm-deployment-lab/.venv/bin:$PATH
  PYTHONPATH=src /Users/chayut/projects/slm-deployment-lab/.venv/bin/python
  -m pytest -q`
  - `200 passed, 10 skipped, 1 failed`.
  - The sole failure remains the concurrent-claim baseline
    `GitSnapshotTests.test_staged_graph_requires_matching_staged_status`
    `StopIteration`; no planned task has all dependencies completed while
    T20, T51, and T72 remain concurrently `in_progress`.

### Fifth-rereview remediation verification

- `HF_HOME=/Volumes/T9/slm-deployment-lab/hf-cache
  SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab PYTHONPATH=src
  /private/tmp/slm-t12-venv/bin/python -m
  slm_lab.export.onnx_matrix validate`
  - Passed after enforcing canonical primitive manifest comparison and the
    raw fixed config spelling.
- `PYTHONPATH=src /private/tmp/slm-t12-venv/bin/python -m pytest -q
  tests/export tests/contracts`
  - `55 passed, 1 skipped`; the skip remains the explicit real-Qwen T12
    numerical gate already completed by T12.
- `PATH=/Users/chayut/projects/slm-deployment-lab/.venv/bin:$PATH
  PYTHONPATH=src /Users/chayut/projects/slm-deployment-lab/.venv/bin/python
  -m pytest -q`
  - `204 passed, 10 skipped, 1 failed`.
  - The sole failure remains the concurrent-claim baseline
    `GitSnapshotTests.test_staged_graph_requires_matching_staged_status`
    `StopIteration`; no planned task has all dependencies completed while
    T20, T51, and T72 remain concurrently `in_progress`.

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

- Newly unblocked tasks: none while independent rereview is pending.
- Recommended next action: fresh independent rereview should rerun the
  adversarial manifest/fixture tests and external evidence validation. After
  approval, T21 should consume the manifests, verify every
  external hash before loading, run multi-position decode parity in ONNX
  Runtime CPU, and inspect whether `valid_length` remains a live internal
  slice/scatter dependency rather than a traced constant.
