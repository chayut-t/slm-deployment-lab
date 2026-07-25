# T11: Deterministic PyTorch Reference

Date: 2026-07-25
Task: `T11`
Visibility: `public`
Status: completed

## Outcome

T11 provides the deterministic PyTorch numerical oracle for the pinned
Qwen3-0.6B revision. The reusable implementation supports complete-prefix
forward execution, cache prefill, one-token cached decode, greedy generation,
and lockstep comparison at every decode step.

The real Qwen3-0.6B CPU/BF16 canary generated token IDs `576, 8356, 3950`.
Full-forward and cached logits were byte-identical at all three steps, and an
immediate reproduction returned the same compact evidence digest
`f5a37682c216120f0c10748a98a7fb44885081e699ea077a46d2caedb39a840b`.

## Changes

- Added strict loading of the immutable T00 Qwen model contract, including
  revision, reference dtype, remote-code policy, architecture, and special
  token validation.
- Added an opt-in pinned Transformers loader that freezes eval mode, gradients,
  dtype, eager attention, seed, and deterministic PyTorch algorithms while
  recording exact runtime versions.
- Added reusable full-forward and cached greedy loops that preserve T10's
  no-sampling, lowest-token-ID tie break, EOS inclusion, and output limit.
- Added stepwise numerical metrics: maximum/mean absolute error, protected
  relative error, cosine similarity, top-1 agreement, top-5 overlap, and the
  reference top-1/top-2 margin.
- Added compact SHA-256 logit fingerprints and a canonical evidence digest
  rather than committing vocabulary-sized logits or cache tensors.
- Added a CC0 deterministic PyTorch causal-model fixture that exercises
  prefill, cached decode, EOS, token selection, tolerance failure, and repeated
  evidence reproduction without depending on distributed model weights.
- Added a real pinned-Qwen fixture for the authored T10 `raw_ascii` canary and
  a gated test that reproduces its tokens, metrics, runtime, and digest from
  external weights.

## Verification

- `UV_CACHE_DIR=/private/tmp/slm-t11-uv-cache uv pip install --python
  .venv/bin/python torch==2.7.1 transformers==4.51.3 pytest==8.3.5
  ruff==0.11.0 pyyaml==6.0.2 jsonschema==4.23.0`
  - Installed an ignored, isolated verification environment. The resolved
    reference runtime also recorded `safetensors==0.8.0`.
- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/reference`
  - 13 passed and the explicit real-Qwen weight-gated test skipped.
- `HF_HOME=/Volumes/T9/slm-deployment-lab/hf-cache
  SLM_LAB_RUN_QWEN_REFERENCE=1 PYTHONPATH=src .venv/bin/python -m pytest -q
  tests/reference`
  - 14 passed, including the real Qwen golden reproduction.
- `PYTHONPATH=src .venv/bin/python -m pytest -q`
  - 94 passed, 3 intentional external/upstream-gated skips.
- `.venv/bin/ruff check src tests`
  - Passed.
- `python3 scripts/ai/render_task_status.py --check`
  - Task graph valid and generated status current.
- `python3 scripts/repo/check_hygiene.py --all`
  - Passed for 180 tracked and untracked public files including completion
    metadata.

## Decisions and evidence

- The numerical oracle uses the checkpoint/reference BF16 dtype and eager
  attention on CPU. The same-model tolerance allows bounded BF16
  accumulation-order noise but requires allclose, protected-relative, cosine,
  top-5, and exact top-1 criteria simultaneously.
- The observed Qwen run was stronger than the tolerance: all three full/cache
  logit pairs had zero absolute error and identical float32 fingerprints.
- The real reference command downloads only the immutable public revision when
  `--allow-download` is explicit. Normal invocation is local-files-only.
- Complete logits and all 28 layers of cache state remain external. The
  committed fixture contains generated IDs, metrics, fingerprints, exact
  runtime identity, and a reproducible command.

## Risks and limitations

- PyTorch is intentionally an optional runtime import because T11 does not own
  the shared dependency manifest. The repository's standard dev environment
  therefore skips the real-Qwen integration unless an exact Torch runtime and
  external weights are supplied; T11 separately exercised that path in the
  isolated recorded environment.
- The golden host evidence is CPU/BF16 on the observed Apple M4 Mac mini and is
  a correctness canary, not latency, throughput, accelerator, or
  cross-platform evidence.
- Fingerprints may legitimately change under a different exact PyTorch,
  Safetensors, device, dtype, or attention implementation. Such a run must be
  recorded as a new reference surface, not silently update this fixture.
- T12 still owns fixed-capacity cache tensor names, layouts, update mechanics,
  and per-layer cache-error thresholds.

## Follow-up

- Newly unblocked tasks: T12 and T50.
- Recommended next action: T12 should trace the full/cache reference through
  explicit static prefill/decode tensor contracts, and the learner should
  reproduce the T11 canary before accepting those downstream cache updates.
