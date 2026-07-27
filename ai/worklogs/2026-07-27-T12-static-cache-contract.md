# T12: Static Cache and Tensor Contract

Date: 2026-07-27
Task: `T12`
Visibility: `public`
Status: completed

## Outcome

T12 freezes one runtime-neutral, machine-checkable Qwen3-0.6B static graph
family for the exact 128, 512, 1,024, and 4,096-token T10 prompts. Every
prefill and one-token decode tensor has an explicit name, dtype, static shape,
layout, and state-transition meaning.

The contract preserves Qwen GQA as 8 physical K/V heads across all 28 layers.
Fixed cache capacities of 160, 576, 1,152, and 4,224 reserve the corresponding
T10 generation budgets, so a full prompt prefill still has legal decode
positions. Multi-step tests reproduce both a weightless PyTorch growing-cache
reference and the real pinned T11 Qwen reference after the explicit
BF16-to-FP16 deployment-boundary cast.

## Changes

- Added immutable tensor and graph specifications with serialization,
  lookup, validation, and generation of all four prefill/decode pairs from one
  definition.
- Added explicit per-layer cache names `key_cache.L`, `value_cache.L`,
  `present_key.L`, and `present_value.L` for `L=0..27`, using layout
  `[batch, kv_head, cache_position, head_dim]`.
- Added exact indexed-update semantics: `[0, valid_length)` remains unchanged,
  the new slice is written at `valid_length`, overflow is rejected, and the
  output length advances by one.
- Added PyTorch/Transformers cache normalization, fixed FP16 cache
  materialization, concrete tensor-map conformance checks, and GQA byte
  accounting.
- Independent review found that the first serializer attached decode-only
  `valid_length` write metadata to prefill. Prefill now serializes prefix
  materialization and zero-fill ranges, while decode alone serializes the
  indexed write transition; regressions freeze both complete mappings.
- Added deterministic contract, shape/dtype/name drift, memory, multi-step
  reference-equivalence, and overflow tests.
- Added an architecture guide with tensor diagrams, mask/position rules,
  prompt-versus-capacity memory tables, and the T12 learner debrief.

## Verification

- `UV_CACHE_DIR=/private/tmp/slm-t12-uv-cache uv pip install --python
  /private/tmp/slm-t12-venv/bin/python torch==2.7.1 pytest==8.3.5`
- `UV_CACHE_DIR=/private/tmp/slm-t12-uv-cache uv pip install --python
  /private/tmp/slm-t12-venv/bin/python transformers==4.51.3
  safetensors==0.8.0`
  - Created an isolated ignored verification runtime using the same pinned
    Torch/Transformers/Safetensors versions as T11.
- `PYTHONPATH=src /Users/chayut/projects/slm-deployment-lab/.venv/bin/python
  -m pytest -q tests/contracts`
  - `5 passed, 3 skipped`; the lightweight environment intentionally lacks
    optional PyTorch and real-weight execution.
- `PYTHONPATH=src /private/tmp/slm-t12-venv/bin/python -m pytest -q
  tests/contracts`
  - `7 passed, 1 skipped`; all weightless PyTorch cache-transition tests
    passed and only the explicit real-Qwen gate skipped.
- `HF_HOME=/Volumes/T9/slm-deployment-lab/hf-cache
  SLM_LAB_RUN_QWEN_CACHE_CONTRACT=1 PYTHONPATH=src
  /private/tmp/slm-t12-venv/bin/python -m pytest -q
  tests/contracts/test_static_cache_contract.py::test_real_qwen_static_updates_reproduce_t11_reference`
  - `1 passed`; pinned Qwen3-0.6B BF16 prefill plus two decode steps matched
    the fixed FP16 cache after explicit casting at every valid position and
    layer.
- `PYTHONPATH=src /Users/chayut/projects/slm-deployment-lab/.venv/bin/python
  -m pytest -q`
  - `123 passed, 6 skipped`; skips are intentional optional/external gates.
- `/Users/chayut/projects/slm-deployment-lab/.venv/bin/ruff check src tests`
  - Passed.
- `python3 scripts/ai/render_task_status.py --check`
  - Passed after completion metadata regeneration.
- `python3 scripts/repo/check_hygiene.py --all`
  - Passed.

## Decisions and evidence

- A cache capacity equal to a fully occupied prompt length cannot accept even
  the first decode token. The frozen capacities therefore add the already
  specified T10 output budgets without changing the named prompt workloads.
- The logical FP16 K+V cache costs 112 KiB per position. Prompt-resident
  S1024 and S4096 state is 112 MiB and 448 MiB; reserved allocation is 126 MiB
  and 462 MiB respectively.
- The cache has 8 physical K/V heads, not 16 repeated query heads. Preserving
  GQA prevents a 2x memory and traffic error.
- Per-layer tensors were selected over one stacked tensor because they match
  the T11/Transformers cache boundary and keep later ONNX/QNN conformance
  failures attributable to a specific layer.
- Cache tensors cross the deployment boundary as FP16 while T11 remains BF16.
  The real-Qwen verification compared the explicitly cast reference and did
  not claim BF16/FP16 byte identity.

## Risks and limitations

- T12 proves graph-boundary and cache state-transition equivalence; it does not
  establish ONNX exportability, compiler acceptance, accelerator placement, or
  performance.
- The explicit out-of-place decode interface can expose two full cache
  allocations. Later runtimes may safely alias or recycle buffers only if
  their own API and numerical tests prove that behavior.
- `int64` is the reference graph boundary. A target-specific rewrite to
  `int32` must be separately recorded and validated rather than silently
  changing this contract.
- Independent review identified one P1 machine-contract inconsistency:
  prefill inherited decode-only update metadata despite lacking a
  `valid_length` input. The graph-kind-specific serializer and regression
  assertions resolve that finding without changing tensor or cache behavior.

## Follow-up

- Newly unblocked task: T20, the four-context ONNX export matrix.
- T51 may consume the cache contract after its other dependency, T50,
  completes.
- Recommended next action: T20 should serialize every generated contract into
  its artifact manifest, validate concrete ONNX I/O mappings, and compare
  logits and multi-step cache updates with T11.
