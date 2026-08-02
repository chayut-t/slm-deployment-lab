# T20: Prefill cache write without float16 `Pad`

Status: active — exporter fix committed; promotion into the reference
artifacts is not authorized and not done
Owner: Claude prefill-rewrite agent (branch `task/prefill-scatter-cache-write`)
Updated: 2026-08-02

## Objective

Make the four float16 T20 prefill graphs loadable by ONNX Runtime's CPU
execution provider at `ORT_DISABLE_ALL`, without changing the frozen T12 graph
contract and without changing what the graphs compute.

They were not loadable. ORT's CPU provider has no float16 `Pad` kernel, and the
prefill cache write used `torch.nn.functional.pad`, so `CastFloat16Transformer`
rejected all four at session creation. Decode, same precision and same export
run, loaded fine because it writes its cache with a scatter. This blocks every
downstream consumer that needs an unoptimized graph — which is the level the
T21 parity runner asks for, deliberately, so that measured numbers belong to
the graph and not to the optimizer.

## Scope

### In scope

- Re-express the prefill zero-extension in `PrefillWrapper`.
- Export candidate graphs for all four contexts and measure: load matrix,
  operator census, numerical equivalence against the old graphs, and multi-step
  parity.
- A regression test that pins the lowering.
- A failure analysis recording the defect, the fix and the promotion path.

### Out of scope

- Changing the T12 contract. If the contract had to move, stop and report.
- Confirming or replacing `DEFAULT_ORT_CPU_TOLERANCE`. That is T21's, and it is
  a separate blocker to a green tree.
- Replacing the attested reference artifacts, re-forging the T20 attestation,
  or regenerating any committed evidence record. All commit-gated.
- Any task-status change in `ai/tasks/task_graph.yaml`.

## Dependencies and resources

- Required task dependencies: `T12` (frozen contract), `T20` (the export
  matrix being modified), `T21` (found the defect; owns the parity runner).
- Resource locks: `t9_heavy_io` — candidate exports write ~4.8 GB of external
  data under `SLM_LAB_ARTIFACT_ROOT`.
- External access: none. No network, no cloud, no spend.
- Cost boundary: zero. Local CPU only.

## Important paths

- Inputs: `configs/models/qwen3-0.6b-onnx-export.json`,
  `tests/fixtures/t10/token-fixtures-v1.json`, the pinned Qwen3-0.6B snapshot
  under `HF_HOME`, `results/manifests/onnx/S*.json`.
- Outputs: `src/slm_lab/export/onnx_matrix.py`,
  `tests/export/test_onnx_matrix.py`,
  `docs/failures/runtime/2026-08-02-t20-fp16-prefill-pad-unloadable.md`.
- Shared contracts: `src/slm_lab/contracts/static_cache.py` — read only, not
  modified. `src/slm_lab/backends/onnx_cpu.py` (T21-owned) — read only, run
  unmodified.
- External, never committed:
  `${SLM_LAB_ARTIFACT_ROOT}/onnx/candidate/concat-reserve/`,
  `${SLM_LAB_ARTIFACT_ROOT}/staging/T20-concat-reserve/`.
- Private: `.ai-local/scratch/` — export/census/load/compare scripts,
  provisional manifests, parity evidence.

## Milestones

- [x] Reproduce the defect and confirm the operator gap is the cause.
- [x] Re-express the cache write; evaluate `Concat`, scatter, `index_copy`,
      `slice_scatter` and a broadcast reserve on their merits.
- [x] Export all four prefill graphs from the fixed exporter.
- [x] All four load at `ORT_DISABLE_ALL`; all four unfixed still fail.
- [x] Operator census before and after, all four contexts.
- [x] Numerical equivalence: bitwise-identical outputs versus the old graphs.
- [x] T12 boundary verified identical against the committed manifests.
- [x] Real multi-step parity at `ORT_DISABLE_ALL`, all four contexts.
- [x] Regression test pinning the lowering and the reserve operand.
- [x] Failure analysis with reproduction commands.
- [ ] **Blocked on commit authorization:** promote into the reference
      artifacts and re-forge the attestation.
- [ ] **Blocked on T21:** resolve `DEFAULT_ORT_CPU_TOLERANCE` so the tree can
      go green.

## Verification and acceptance

- Commands: full `pytest` in the locked root environment; `pytest tests/export
  tests/onnx` in the parity environment, both with and without
  `SLM_LAB_ARTIFACT_ROOT`; `ruff format --check` and `ruff check` on touched
  paths; `scripts/ai/render_task_status.py --check`;
  `scripts/repo/check_hygiene.py --all`; `git status --short --ignored`.
- Numerical criteria: every prefill output bitwise identical to the superseded
  graph at a level where both load; `prefill_reserve_zero` clean on all 56
  cache tensors at all four contexts; top-1 agreement on all 20 parity steps;
  zero non-finite logits; `cache_report.passed` true.
- Contract criterion: exported `input_tensors` and `output_tensors` equal to
  the committed manifests exactly, 3 in and 58 out, same names, dtypes, shapes
  and order.
- Hardware/profile evidence: none. No compiler, no accelerator, no device, no
  latency or memory claim.

## Artifact and privacy handling

- Committed evidence: the source change, the regression test, and the failure
  analysis. No manifest, inspection report or parity record was regenerated,
  because the reference artifacts were not replaced.
- External artifacts: candidate and staging graph trees under
  `SLM_LAB_ARTIFACT_ROOT`; never committed, identified in the analysis by
  SHA-256 and byte size.
- Private/local material: `.ai-local/scratch/`. Contains no credentials, no
  service identifiers and no host-specific data beyond mount paths, which the
  parity evidence format already excludes from its digest.

## Decisions and discoveries

- 2026-08-02: `Concat` chosen over mirroring decode's scatter. A scatter loads
  equally well but adds 56 runtime-indexed writes to the one graph that has
  none, extending the §5.2 rank-2 Qualcomm risk, and needs a capacity-sized
  rather than reserve-sized zero buffer.
- 2026-08-02: the missing kernel confirmed positively, not inferred from the
  error string. Dumping `optimized_model_filepath` at `ORT_ENABLE_BASIC` shows
  ORT executing `Add(f32) -> Pad(f32) -> Cast(to=f16) -> Reshape` for the old
  graph: the cast the exporter emitted *before* the `Pad` is still there, and
  ORT hoisted the `Pad` above it to keep the pad in float32. The earlier
  account — that higher levels folded the cast away — was wrong and has been
  corrected in the source, the test and the failure analysis.
- 2026-08-02: broadcasting the reserve from a scalar zero to collapse 56
  duplicate constants produces a byte-identical file. The eliminating pass is
  TorchScript's `_jit_pass_constant_propagation`, run unconditionally by
  `torch.onnx.utils._optimize_graph`, not the `do_constant_folding=False` this
  module passes — exporting with that flag either way gives the identical
  result. Abandoned.
- 2026-08-02: the duplicated reserve constants stay out of external data
  because torch emits them as node attributes and the save uses
  `convert_attribute=False`, not because of the 1024-byte size threshold, which
  they exceed by 64x to 256x. This is what preserves the single shared
  `external_data_sha256` the attestation records.
- 2026-08-02: the fix is numerically inert — bitwise-identical outputs, not
  merely within float16 rounding.
- 2026-08-02: `test_real_onnxruntime_cpu_parity_when_available` is red on any
  host with the artifacts mounted, pre-existing. Measured that promotion alone
  does **not** turn it green: `assert evidence.passed` still fails on
  `numerical_tolerance`.
- 2026-08-02: promotion requires an exporter commit whose config carries no
  `evidence_attestation` block, because `_export_provenance` compares the
  attested commit's config against the current one with that block removed.
- 2026-08-02: the prefill `R-DATA-DEPENDENT-SHAPE-INPUT` count is **unchanged**
  at 804 after the fix — only its denominator moves, 1,257/1,258 to 922. Found
  by regenerating the T21 inspection reports against the staged graphs. Reading
  the change off the census alone would have wrongly marked
  `graph-inspection.md:193` as needing an edit.
- 2026-08-02: enumerating the promotion blast radius by reasoning failed four
  times in review. Replaced with `.ai-local/scratch/promotion_audit.py claims`,
  which classifies every numeric token in the affected documents and prints an
  explicit `UNCLASSIFIED` queue, so completeness does not depend on anyone
  remembering a value. It is a cross-check, not an oracle: `MOVES` and
  `AMBIGUOUS` still need human rulings.

## Progress and restart instructions

The exporter fix, its regression test and the failure analysis are complete,
verified, and committed on branch `task/prefill-scatter-cache-write` after a
five-round review loop. Nothing is pushed or merged, and no task status was
changed. This plan stays in `ai/plans/active/` because two of its milestones
are open: the promotion into the reference artifacts, and the tolerance work
that a green tree also needs.

The next action depends on a decision that is the user's, not an agent's:

1. Resolve the coordination gap recorded in
   `ai/worklogs/2026-08-02-T20-prefill-concat-cache-write.md` — `T20` and `T21`
   are `completed` in the task graph with nothing recording that T20's attested
   outputs are known-defective, while `T22` depends on `T21` and would consume
   them.
2. With commit authorization, run the five-step promotion sequence in the
   failure analysis. Do not hand-edit any recorded digest; regenerate each
   record with the tool that produces it.
3. Independently, T21 must settle `DEFAULT_ORT_CPU_TOLERANCE` against a real
   measurement. Until then the tree cannot be green on a host equipped to take
   one, whatever happens to the export.
