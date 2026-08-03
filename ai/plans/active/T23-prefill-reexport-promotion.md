# T23: Prefill re-export promotion and evidence refresh

Status: draft
Owner: unassigned
Updated: 2026-08-02

## Objective

Promote the `Concat` prefill cache write into the attested reference export, and
bring every committed record that describes those graphs back into agreement
with the bytes on disk.

The four reference prefill graphs zero-extend each layer's KV cache with a
float16 `Pad`. ONNX Runtime's CPU provider registers no float16 `Pad` kernel, so
its `CastFloat16Transformer` rewrites the graph to run `Pad` in float32 and then
fails type inference doing it. The graphs cannot be loaded at `ORT_DISABLE_ALL`
on onnxruntime 1.20.1, 1.22.0 or 1.28.0. Decode, at the same precision, uses
`ScatterElements` and loads everywhere. The fix — writing the reserve with
`Concat` — is measured and committed at `e9edc8a`; only the promotion is left.

Full analysis:
[`docs/failures/runtime/2026-08-02-t20-fp16-prefill-pad-unloadable.md`](../../../docs/failures/runtime/2026-08-02-t20-fp16-prefill-pad-unloadable.md).

## Scope

### In scope

- Re-export the four contexts from the committed `Concat` exporter, and
  regenerate the export attestation with its own tooling.
- Regenerate `results/manifests/onnx/S*.json`, the T21 inspection reports, and
  the ORT CPU parity evidence — the last of these at `ORT_DISABLE_ALL`.
- Confirm or replace `DEFAULT_ORT_CPU_TOLERANCE`, recording the derivation.
- Reconcile every committed number describing the reference graphs: digests,
  sizes, node counts, operator counts, and every ratio derived from them.
- Promote the numeric-claim audit from a scratch script into `scripts/`.
- Refresh `LEARN-10` and republish it.

### Out of scope

- Any change to the T12 graph contract. The boundary is frozen and the fix was
  built to leave it byte-identical; if promotion appears to require a contract
  change, stop and escalate.
- The decode graphs. They are unaffected and their digests must not move.
- Any QNN, Device Cloud, or quantization work. `T22` is blocked on this task and
  starts after it.
- Re-litigating `T20`. Its acceptance criteria were about export correctness and
  it met them; nothing required the graph to load in a runtime.

## Dependencies and resources

- Required task dependencies: `T20` (completed), `T21` (completed).
- Resource locks: `t9_heavy_io`.
- External access: none. No paid job, no cloud submission, no network beyond a
  local Hugging Face cache.
- Cost boundary: zero spend. Local CPU only.

## Important paths

- Inputs: `src/slm_lab/export/onnx_matrix.py`,
  `configs/models/qwen3-0.6b-onnx-export.json`,
  `${SLM_LAB_ARTIFACT_ROOT}/onnx/reference/T20/`, the failure analysis above.
- Outputs: `results/manifests/onnx/S*.json`, `results/graph/S*.json`,
  `results/graph/parity/S*-ort-cpu.json`, `docs/results/onnx/*.md`,
  `scripts/` audit tool.
- Shared contracts: `src/slm_lab/contracts/static_cache.py` (read-only here),
  `results/quantization/t40-baseline-parity-2026-08-02.json` (T40's record,
  which re-hashed all 16 files and must be regenerated, not hand-edited).

## Milestones

- [ ] Re-export succeeds and the export attestation verifies against its own
      tooling, with no digest hand-edited.
- [ ] All four prefill graphs create a CPU-EP session at `ORT_DISABLE_ALL`.
- [ ] All four decode digests and the shared `external_data_sha256` are
      unchanged from the pre-promotion artifacts.
- [ ] `results/graph/S*.json` regenerated; no new risk finding appears that was
      not predicted, and any that does is explained rather than accepted.
- [ ] Parity re-measured at `ORT_DISABLE_ALL` for all four contexts, at
      `evidence_tier="real_onnxruntime_cpu"`.
- [ ] `DEFAULT_ORT_CPU_TOLERANCE` no longer reads `proposed_unvalidated`, with
      its derivation recorded.
- [ ] The audit tool reports zero unreconciled numeric claims, and lives in
      `scripts/` rather than scratch.
- [ ] `LEARN-10` rebuilt and republished.

## Verification and acceptance

- Commands:
  - `PYTHONPATH=src <parity-env-python> -m slm_lab.export.onnx_matrix …`
  - `PYTHONPATH=src <parity-env-python> -m slm_lab.graph.inspection --all-manifests`
  - `PYTHONPATH=src <parity-env-python> -m slm_lab.backends.onnx_cpu --manifest results/manifests/onnx/S<ctx>.json --steps 4 --reference torch --output results/graph/parity/S<ctx>-ort-cpu.json`
    (no `--graph-optimization-level`: the default `ORT_DISABLE_ALL` is the point)
  - `python3 scripts/ai/render_task_status.py --check`
  - `python3 scripts/repo/check_hygiene.py --all`
  - `python3 scripts/dashboard/build_dashboard.py --check`
  - the promoted audit tool, in both its citation and claim modes
- Numerical or behavioral criteria: cache invariants hold on every step of every
  context; decode digests unchanged; the tolerance decision is a derivation, not
  a fit. **A parity failure is recorded as a failure.** The acceptance criteria
  deliberately do not require parity to pass, so that widening a threshold until
  the measurement agrees with it cannot close this task.
- Hardware/profile evidence: none. This task makes no performance claim; the
  single-threaded, deterministic session configuration would be a poor one.

## Artifact and privacy handling

- Committed evidence: manifests, inspection reports, parity records, checksums.
- External artifacts: the eight `.onnx` and `.onnx.data` files, ~8.9 GB, under
  `SLM_LAB_ARTIFACT_ROOT`. Never committed. Keep the superseded set until the
  new one verifies.
- Private/local material: the parity environment at `.ai-local/envs/t21-ort-cpu`
  and any scratch probes stay ignored. Nine packages were added to that
  environment beyond its documented pins (`jsonschema`, `jsonschema-specifications`,
  `referencing`, `rpds-py`, `attrs`, `pytest`, `pluggy`, `iniconfig`,
  `pygments`); a `pip freeze`-based pin would capture them.

## Decisions and discoveries

- 2026-08-02: `Concat` chosen over mirroring decode's `ScatterElements`. Both
  load, but a scatter reintroduces a runtime index for a write whose address is
  known at export time — already a ranked deployment risk for the Qualcomm lane.
- 2026-08-02: promotion is structurally commit-gated.
  `_trusted_export_config_bytes` requires the on-disk config to equal `HEAD`'s
  and hash to `FROZEN_EXPORT_CONFIG_SHA256`; `_export_provenance` requires the
  attested commit's config to equal the current one with the attestation block
  removed. Re-attesting therefore needs at least two chained commits.
- 2026-08-02: the reserve constant is re-materialised at all 56 sites and
  escapes external-data conversion because `save_model(convert_attribute=False)`
  externalises only initializers. At S1024/S4096 it is exactly 262,144 bytes,
  which equals `max_bytes` in `configs/graph/onnx-risk-rules-v1.json`, so the
  inline-constant risk rule does not fire — by one byte, on a `<=` comparison.
- 2026-08-02: four numeric-claim reconciliations attempted by reasoning were
  each incomplete. Work the list from tool output, never from memory.

## Progress and restart instructions

The fix and its analysis are merged; the reference artifacts still carry the
defect. Nothing about the promotion has been started.

Next action, in order:

1. Resolve the interpreter question before exporting anything. `_verify_runtime`
   pins `runtime_python_version: 3.11.15`; the parity host has 3.11.13. Either
   provision 3.11.15, or change the pin as a recorded decision — it is part of
   what the attestation asserts, so it is a decision, not a workaround.
2. Re-export and re-attest through the two-commit sequence in §"Promotion" of
   the failure analysis.
3. Regenerate manifests, then inspection reports, then parity — in that order;
   each reads the previous one's output.
4. Take the tolerance decision on the `ORT_DISABLE_ALL` numbers, not on the
   superseded `ORT_ENABLE_BASIC` ones in `results/graph/parity/` today.
5. Work the audit tool's `MOVES`, `AMBIGUOUS` and `UNCLASSIFIED` queues to zero.

Claim the task in `ai/tasks/task_graph.yaml` before starting, and expect `T22`
to unblock automatically when this completes.
