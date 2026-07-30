# T51: Custom MLX Runtime

Date: 2026-07-30
Task: `T51`
Visibility: `public`
Status: completed

## Outcome

T51 implements a reusable custom Qwen3 MLX runtime with explicit prompt
prefill and one-token decode boundaries. It uses full fixed-capacity FP16
caches with either head-major `[batch, kv_head, cache_position, head_dim]` or
sequence-major `[batch, cache_position, kv_head, head_dim]` storage.

The attention boundary passes 16 query heads and eight physical K/V heads
directly to MLX native grouped-query SDPA. It does not repeat or tile K/V
heads. Real execution on the repository's Apple M4 target loaded the immutable
Qwen3-0.6B Safetensors without constructing an MLX-LM model and reproduced the
T11 three-token oracle exactly as `576, 8356, 3950` through both cache
layouts.

## Changes

- Added immutable-source configuration validation for the exact Qwen3-0.6B
  config and weights used by T50, while retaining small configurations for
  deterministic unit tests.
- Added functional fixed-capacity cache allocation and indexed updates for
  head-major and sequence-major layouts. Both store only physical K/V heads
  and expose active head-major views to attention.
- Added custom Qwen3 embedding, RMSNorm, RoPE, grouped-query attention, MLP,
  decoder-layer, tied-output, and strict Safetensors-loading modules.
- Added explicit `prefill`, T12-variant prefill, and exactly-one-token `decode`
  APIs. Logits cross the runtime boundary as FP32 and cache tensors as FP16.
- Added deterministic greedy generation that does not compute an unreturned
  look-ahead token; the final selected token remains explicit pending state.
- Added pure configuration/shape tests, tiny-model state/numerical tests,
  multi-step generation tests, structural no-repeat/no-tile regression, and a
  gated real-weight Apple M4 canary.

## Verification

- `SLM_LAB_RUN_MLX_QWEN=1 SLM_LAB_MLX_MODEL_DIR=<pinned-snapshot>
  PYTHONPATH=src .ai-local/envs/t51-mlx/bin/python -m pytest -q tests/mlx`
  - `13 passed`; includes both real Qwen cache layouts on the Apple M4.
- `PYTHONPATH=src uv run --extra dev --locked python -m pytest -q tests/mlx`
  - `5 passed, 2 skipped`; real MLX modules are intentionally skipped in the
    platform-neutral environment.
- `PYTHONPATH=src uv run --extra dev --locked python -m pytest -q`
  - `166 passed, 8 skipped`; skips are intentional optional/external/runtime
    gates.
- `uv run --extra dev --locked ruff check src tests`
  - Passed.
- `python3 scripts/ai/render_task_status.py --check`
  - Passed after T51 completion metadata regeneration.
- `python3 scripts/repo/check_hygiene.py --all`
  - Passed.

## Decisions and evidence

- Exact real runtime: Python 3.11.13, MLX 0.32.0, MLX-Metal 0.32.0 on
  `Device(gpu, 0)`, device name `Apple M4`, architecture `applegpu_g16g`.
  This establishes MLX Metal GPU correctness and makes no Apple Neural Engine
  claim.
- The source loader validates the frozen config digest
  `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`
  and 1,503,300,328-byte weight digest
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- MLX native SDPA explicitly accepts GQA inputs without K/V pre-tiling. T51
  validates the concrete boundary before every call and regression-tests that
  the implementation contains no repeat/tile operation.
- Decode versus full-forward numerical checks pass through both physical
  layouts after the explicit FP16 cache boundary. The real T11 generation
  canary passes both layouts exactly.
- Sequence-major cache access uses a transposed active view. Whether it
  improves memory behavior on the M4 is deliberately left to T52 profiling.

## Risks and limitations

- T51 provides correctness evidence, not performance evidence. T52 owns the
  128/512/1,024/4,096 sweep, lazy-evaluation fences, `mx.compile`, Instruments,
  memory pressure, power, and thermal measurements.
- The project-wide dependency lock stays platform-neutral. The real gate uses
  T50's separately pinned macOS MLX requirements in an ignored task-local
  environment.
- The real canary validates exact selected tokens and layout parity. It does
  not claim byte identity with the BF16 PyTorch oracle because cache state
  intentionally crosses T12's FP16 deployment boundary.

## Follow-up

- Newly unblocked task: T52, Apple profiling and context sweep.
- T70 still depends on T33 and T60 in addition to T51.
- Recommended next action: T52 should measure both layouts under identical
  synchronization and allocation boundaries before selecting a default.

## Learner debrief checklist

- [ ] Walk through
  `src/slm_lab/backends/mlx/model.py::Qwen3Attention.__call__` and explain why
  16 query heads can consume eight physical K/V heads without repeated K/V
  storage.
- [ ] Compare the head-major and sequence-major update/view logic in
  `src/slm_lab/backends/mlx/cache.py` and calculate cache traffic for one
  decode position.
- [ ] Reproduce the focused tests in `tests/mlx/` and inspect the future
  `08_mlx_gqa_kv_layout.ipynb` alongside T52 evidence.
