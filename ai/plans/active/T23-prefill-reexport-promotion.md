# T23: Prefill re-export promotion and evidence refresh

Status: complete on `task/T23-prefill-reexport-promotion`, awaiting merge
Owner: Claude t23-main agent
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

- [x] Re-export succeeds and the export attestation verifies against its own
      tooling, with no digest hand-edited. Re-attested by
      `scripts/export/write_export_attestation.py` across `321b11b` and
      `d3494fd`.
- [x] All four prefill graphs create a CPU-EP session at `ORT_DISABLE_ALL`.
- [x] All four decode digests and the shared `external_data_sha256` are
      unchanged from the pre-promotion artifacts.
- [x] `results/graph/S*.json` regenerated; no new risk finding appears that was
      not predicted, and any that does is explained rather than accepted.
      Finding totals are 14/15/15/15; S128 is 14 because
      `R-LARGE-INLINE-CONSTANT` does not fire there.
- [x] Parity re-measured at `ORT_DISABLE_ALL` for all four contexts, at
      `evidence_tier="real_onnxruntime_cpu"`, on onnxruntime 1.28.0.
- [x] `DEFAULT_ORT_CPU_TOLERANCE` no longer reads `proposed_unvalidated`, with
      its derivation recorded. `TOLERANCE_STATUS` is now `derived_and_measured`.
- [x] The audit tool reports zero unreconciled numeric claims, and lives in
      `scripts/audit/audit_reference_graph_claims.py` rather than scratch.
- [x] `LEARN-10` rebuilt and republished.

One in-scope item was deliberately **not** taken, and is recorded as a
follow-up rather than done: the risk catalogue still has no total-inline-bytes
rule. See the last entry under "Decisions and discoveries".

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

### Discovered while executing the plan

- 2026-08-02: **the prescribed promotion order was not executable.** Step 1 of
  "Progress and restart instructions" says to re-export before re-attesting, but
  `load_export_config` refuses a config carrying no attestation block, so the
  export CLI cannot run at the very commit the plan said to export from. The
  commit-gating described above is real, but it is tighter than the plan
  realised: the config must first be *un*attested as its own commit (`321b11b`),
  which is why the sequence ran to three commits rather than two. A plan that
  prescribes an order should be executed once against the tooling before it is
  trusted.
- 2026-08-02: **the interpreter pin moved to the measured value.**
  `_verify_runtime` pinned `runtime_python_version: 3.11.15` while the parity
  host runs 3.11.13. The decision was to re-attest on 3.11.13 rather than
  provision 3.11.15, because the attestation exists to record the interpreter
  that actually ran. Making the record match the machine is the point; making
  the machine match the record would have preserved a number at the cost of the
  thing it asserts.
- 2026-08-02: **the blast-radius enumeration missed
  `configs/quantization/calibration.yaml`**, which pins the
  `canonical_json_sha256` of the export *config* — not of any graph. No search
  for a graph digest, size, or node count could have found it, because it cites
  none of them. The test suite found it, via 22 failures. Enumerating a blast
  radius by searching for the values that moved cannot find a record that pins a
  digest of the thing that moved.
- 2026-08-02: **the old ORT CPU tolerance rejected float32 itself.** The
  load-bearing control was a float32-vs-bfloat16 self-comparison with no ONNX,
  no runtime and no graph involved: it missed the old `atol` of 0.25 by 0.609,
  where the actual graph missed by 0.578. The instrument was broken, not the
  graph. A tolerance that the reference dtype cannot pass is not a threshold the
  candidate failed. Any tolerance change must lead with a control like this,
  because otherwise widening a threshold until the measurement agrees with it is
  indistinguishable from validating it.
- 2026-08-02: **the re-export inverted which inline-constant family dominates
  the protobuf.** The `Concat` reserves are 71.5% / 78.9% / 80.5% / 29.5% of the
  four prefill files against the causal mask's 0.6% / 5.6% / 11.5% / 67.4%, so
  the mask is the largest inline family only at S4096. A fix aimed at one
  deployment risk created a larger instance of the same risk, and neither the
  risk catalogue nor the reports said so until they were re-read against bytes.
- 2026-08-02: **the risk catalogue cannot see the new family.**
  `R-LARGE-INLINE-CONSTANT` is per-tensor and fires strictly above 262,144 bytes;
  every reserve is at most exactly 262,144, so 71.5% of the S128 prefill protobuf
  is invisible to the catalogue. Adding a total-inline-bytes rule would change
  findings and is therefore **out of scope for T23** — recorded here as a
  follow-up for the task that next owns the catalogue. Note also that the clean
  `R-LARGE-INLINE-CONSTANT` report at S1024 and S4096 rests on a one-byte
  boundary.
- 2026-08-02: **reconciliation-by-reasoning failed five times**, not the four
  recorded above — the fifth surfaced after that entry was written. Two of the
  five were in briefs written from this plan's own analysis, so a careful
  document derived from correct analysis is not a substitute for re-running the
  producer. The audit tool caught every one. The rule this ticket ends with:
  regenerate from the producer, then let the tool enumerate; never reconcile
  from a document, however good.

## Progress and restart instructions

All eight milestones are met on `task/T23-prefill-reexport-promotion`. The
promotion is done, the evidence is regenerated, and the audit tool reports zero
citation disagreements. `CLAIM_DOCUMENTS` holds eight in-scope documents, of
which `citations` binds the six with `role="reconcile"`; the other two are
`role="historical"` and are enumerated by `claims` rather than bound.

**The work is not merged, and merging was not authorized.** `T23` therefore
stays `in_progress` in `ai/tasks/task_graph.yaml`: the schema allows only
`planned`, `in_progress`, `blocked` and `completed`, and AGENTS.md defines
`completed` as requiring the changes to be integrated into the branch downstream
tasks will use. There is no state for "finished on its branch", so the truthful
choice is `in_progress` with the branch and worklog recorded.

The worklog is `ai/worklogs/2026-08-02-T23-prefill-reexport-promotion.md`. It is
**not** referenced from the task graph and cannot be: the validator enforces
`only completed tasks may set the worklog field`, so `T23`'s `worklog` stays
`null` until it is `completed`. Set both in the same edit at merge time.

Next action, for whoever picks this up:

1. Review and merge `task/T23-prefill-reexport-promotion` into `main`.
2. Promote `T23` to `completed` **and** set its `worklog` to the path above in
   the same edit, then re-run `python3 scripts/ai/render_task_status.py`. `T22`
   unblocks at that point, not before.
3. Move this plan to `ai/plans/completed/`.

Carried forward as unresolved, and listed in the worklog:

- The **fusion delta is unmeasured**: every committed record is
  `ORT_DISABLE_ALL`; the paired `ORT_ENABLE_ALL` run has never been taken.
- The parity evidence is **one build, one EP, one host**.
- The committed parity records use a **bfloat16** reference. The
  float16-reference probe, 6.9x-9.8x tighter on S128, is a diagnostic only;
  switching the reference dtype is a contract decision nobody has taken.
- The claim that the prefill `Pad` defect reproduces on onnxruntime 1.20.1 and
  1.22.0 is **hearsay** — no recorded run exists for either.
- The risk catalogue has no total-inline-bytes rule.
- The clean `R-LARGE-INLINE-CONSTANT` report at S1024/S4096 sits on a one-byte
  boundary.
