# T20: Prefill Concat Cache Write

Date: 2026-08-02
Task: `T20`
Visibility: `public`
Status: implementation complete; promotion into the reference artifacts is
commit-gated and not done

## Outcome

`PrefillWrapper` no longer expresses the prefill cache zero-extension with
`torch.nn.functional.pad`. It uses `torch.cat` against a zero reserve, which
lowers to ONNX `Concat` instead of ONNX `Pad`.

ONNX Runtime's CPU execution provider has no float16 `Pad` kernel. All four
float16 prefill graphs were therefore unloadable at `ORT_DISABLE_ALL` — not
slow, not inaccurate, but rejected at session creation by
`CastFloat16Transformer`. All four decode graphs, same precision and same
export run, loaded fine, because decode writes its cache with a scatter. One
operator choice was the whole difference.

With the fix, all four prefill graphs load at all four optimization levels and
produce a real multi-step parity measurement at `ORT_DISABLE_ALL` for the first
time. The full analysis, including reproduction commands, is
`docs/failures/runtime/2026-08-02-t20-fp16-prefill-pad-unloadable.md`.

**This did not replace the reference artifacts.** The graphs measured here were
exported into `${SLM_LAB_ARTIFACT_ROOT}/onnx/candidate/concat-reserve/`. The
attested `onnx/reference/T20` tree is untouched, so no committed record was
regenerated and none is stale.

## Changes

- `src/slm_lab/export/onnx_matrix.py` — `PrefillWrapper` builds a
  `[1, 8, capacity - prompt_length, 128]` float16 zero reserve once and
  `torch.cat`s it onto each layer's key and value along the `cache_position`
  axis. Docstring and comment record why `Concat` was chosen over a scatter,
  and why the reserve stays inline in the protobuf.
- `tests/export/test_onnx_matrix.py` —
  `test_prefill_cache_write_lowers_to_concat_and_never_pad` asserts zero `Pad`
  nodes, that all 56 cache outputs are `Reshape` over a 2-input `Concat` at
  `axis=2`, and that the second `Concat` operand is a `Constant` node holding
  an all-zero float16 tensor of exactly the reserve shape.
- `docs/failures/runtime/2026-08-02-t20-fp16-prefill-pad-unloadable.md` — new
  failure analysis.

## Verification

- Command: `PYTHONPATH=<worktree>/src .venv/bin/python -m pytest -q`
  Result: **574 passed, 15 skipped** (branch baseline 574/14; the extra skip is
  the new torch-gated test, which the locked root environment cannot run).
- Command: `PYTHONPATH=<worktree>/src <parity-venv>/bin/python -m pytest
  tests/export tests/onnx -q`
  Result: **214 passed, 2 skipped**.
- Command: the same, with `SLM_LAB_ARTIFACT_ROOT` pointing at the real T20
  graphs.
  Result: **1 failed, 214 passed, 1 skipped** — see "Risks and limitations".
- Command: `ruff format --check` and `ruff check` on both touched files.
  Result: clean.
- Command: `scripts/ai/render_task_status.py --check`.
  Result: `task graph valid; 30 tasks; 12 learning checkpoints; generated
  status is current`.
- Command: `scripts/repo/check_hygiene.py --all`.
  Result: passed.
- Command: `git status --short --ignored`.
  Result: modified/untracked public paths only, no `.onnx` or `.onnx.data`
  anywhere near the index.

Measurements (all `evidence_tier="real_onnxruntime_cpu"`, onnxruntime 1.28.0,
CPU EP, `ORT_DISABLE_ALL`, single-threaded):

- All four fixed prefill graphs create a session at all four optimization
  levels. All four unfixed ones still fail at `ORT_DISABLE_ALL`.
- 20 parity steps across four contexts: top-1 agreement everywhere, zero
  non-finite logits, `cache_report.passed` true everywhere,
  `prefill_reserve_zero` clean on all 56 tensors per context, teacher-forced
  sequences reproduced exactly.
- Prefill node count 8,753/8,755 → 7,634; `Pad` 56 → 0.

## Decisions and evidence

- **`Concat`, not a scatter.** Mirroring `DecodeWrapper` would also load, but
  it would add 56 indexed scatters to the one graph that has none, and
  `docs/results/onnx/graph-inspection.md` §5.2 already ranks the indexed cache
  write as the second-highest deployment risk for the Qualcomm lane precisely
  because a runtime write address defeats compile-time DMA descriptor
  generation. The reserve is a compile-time constant, so `Concat` keeps the
  write fully static.
- **The missing kernel is confirmed positively, not inferred from the error
  string.** Dumping `optimized_model_filepath` at `ORT_ENABLE_BASIC` shows ORT
  executing `Add(float32) -> Pad(float32) -> Cast(to=float16) -> Reshape` for
  the old graph and
  `Add(float32) -> Cast(to=float16) -> Concat(float16) -> Reshape` for the new
  one. The cast the exporter emitted before the `Pad` is still present; ORT
  hoisted the `Pad` above it so the pad runs in float32 rather than run `Pad`
  in float16. An earlier draft of this work claimed higher levels folded the
  cast away — that was wrong, and is corrected in the source comment, the test
  docstring and the failure analysis.
- **The change is numerically inert, and that is measured, not assumed.** Old
  and new S128 prefill produce all 58 outputs bitwise identical at
  `ORT_ENABLE_BASIC` and `ORT_ENABLE_ALL`, and identical
  `candidate_logits_sha256` at all five steps through the parity runner. The
  T12 boundary is byte-identical to the committed manifests for all four
  contexts.
- **Blast radius: recommend replacing the reference export, option (a).** The
  `Pad` is a defect in the reference artifact; `artifacts/onnx/qnn-candidate/`
  is reserved for compiler/quantized candidates, not for "the reference but
  loadable". Keeping a parallel variant would leave the reference broken and
  oblige every consumer to know to avoid it.
- **Promotion is commit-gated, which is why it is not done here.** The T20
  attestation is anchored to Git on purpose: the config pins eight graph
  digests, the exporter commit and the runtime Python version;
  `FROZEN_EXPORT_CONFIG_SHA256` pins the config bytes;
  `_trusted_export_config_bytes` requires the on-disk config to equal `HEAD`'s;
  and `_export_provenance` requires the attested commit's config to equal the
  current one with the attestation block removed — which is why the attested
  commit `631fd70` carries a config with no attestation at all. Re-forging that
  chain needs at least two chained commits, and this task was instructed not to
  commit. The five-step promotion order is in the failure analysis.

## Risks and limitations

- **A committed test is red on any host that has the artifacts.**
  `tests/onnx/test_onnx_cpu_parity.py::test_real_onnxruntime_cpu_parity_when_available`
  fails whenever `SLM_LAB_ARTIFACT_ROOT` points at real T20 graphs, because it
  uses the committed manifest and `onnxruntime_cpu_session_factory()` defaults
  to `ORT_DISABLE_ALL`. Pre-existing, not caused by this change. Without the
  artifact root it skips, which is how it stayed invisible.
- **Promotion alone does not turn that test green.** Measured against the fixed
  staged graphs: the tier and `cache_report.passed` assertions pass, and
  `assert evidence.passed` still fails on `numerical_tolerance`. A green tree
  needs the `DEFAULT_ORT_CPU_TOLERANCE` work too — the thresholds still carry
  `status: proposed_unvalidated`. That is T21 work, and it is a hard blocker
  this change neither performs nor makes easier.
- **The shared-`external_data_sha256` invariant is load-bearing, and the new
  test is the only thing holding it.** The 56 duplicated reserve constants stay
  out of external data only because torch emits them as node attributes and
  `export_onnx_graph` saves with `convert_attribute=False`. Measured by
  breaking each half through the production export path: with
  `convert_attribute=True` the test fails with `ValidationError`, and with a
  registered-buffer reserve it fails with `KeyError`. So the guard is real —
  but it is one test, and weakening its assertions silently removes the only
  protection.
- **Only one onnxruntime version is evidenced.** 1.28.0. Reports of 1.20.1 and
  1.22.0 reproducing the defect are unrecorded anywhere in the repository and
  are not treated as evidence.
- **Environment mutation.** `uv pip install jsonschema pytest` into the shared
  `.ai-local/envs/t21-ort-cpu` added nine packages
  (`jsonschema`, `jsonschema-specifications`, `referencing`, `rpds-py`,
  `attrs`, `pytest`, `pluggy`, `iniconfig`, `pygments`). Nothing was upgraded;
  `torch`, `transformers`, `onnx`, `onnxruntime` and `numpy` are untouched. Any
  future pin taken from a `pip freeze` of that environment will capture all
  nine.
- **Provisional inputs.** The parity runs used uncommitted manifests under
  `.ai-local/scratch/manifests/` against a hard-linked staging tree, so the
  parity CLI ran unmodified. Their `export_provenance` still names the
  superseded export run.

## Coordination gap for the user to decide

**No task status was changed, as instructed.** But the graph now says something
that is not true, and a fresh session reads the graph, not `docs/failures/`:

- `T20` is `completed`, and its acceptance criteria ("Artifacts conform to
  frozen contracts") are literally satisfied — that check is a shape and dtype
  check and it passes. Nothing in the node records that T20's attested outputs
  cannot be loaded by the runtime T21 exists to run.
- `T21` is `completed` while the one test that can produce a real ONNX Runtime
  measurement is red on an equipped host, and while its own tolerances remain
  `proposed_unvalidated`.
- `T22` (`planned`, "QNN candidates and packaging") depends on `T21` and will
  consume the reference exports. `T40` and `T60` also depend on `T20`. A T22
  agent starting from the committed graph today would build QNN candidates from
  prefill graphs carrying a defect that is already understood and already fixed
  in the working tree.

Options, for the user rather than for me: reopen T20; add a defect or
`known_issues` field to the T20 node pointing at the failure analysis; or hold
T22 until promotion lands. A failure-analysis document is not the surface that
prevents the wrong start.

## Follow-up

- Newly unblocked tasks: none. Nothing is unblocked until the reference export
  is replaced.
- Recommended next action: decide the coordination gap above, then run the
  five-step promotion sequence in the failure analysis, which requires
  authorization to commit. The tolerance work is independent and can start now.
