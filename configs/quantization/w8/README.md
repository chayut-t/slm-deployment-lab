# W8 candidate specifications (T41)

Two generated files, one per row of the plan's experiment matrix
(`docs/project/plan.md` section 7.2):

| File | Plan row | Weights | Activations | KV cache |
|---|---|---|---|---|
| `w8a16.yaml` | Q1, conservative PTQ | int8 | int16 | `float16` |
| `w8a8.yaml` | Q2, aggressive PTQ | int8 | int8 | `float16` |

**Neither file is a result.** No weight was quantized to produce them and
nothing in them was measured on any device. They are the frozen *inputs* to a
W8 experiment that has not been run: the precision policies, the inclusion and
exclusion policy, the bindings that make a future result comparable, and an
analytic projection that is labelled as one on every row. The reader-facing
account is `docs/results/quantization/w8.md`.

## Generated files. Do not hand-edit

```bash
uv run python -m slm_lab.quantization.w8 generate  # rewrite both files
uv run python -m slm_lab.quantization.w8 check     # offline validation
```

`generate` is a fixed point: at this commit it rewrites both files
byte-identically, so `generate` followed by `check` still passes and, once this
directory is tracked, `git diff -- configs/quantization/w8` stays empty across
it. `check` is the CI and test gate. It reaches no network and needs no
quantization stack — the module depends on nothing beyond the standard library
and PyYAML, so both files regenerate on any host.

Hand-editing is not merely discouraged; it fails. `check` rebuilds each
document from committed inputs and compares the rendered text byte for byte, so
even a prose edit inside a comment or a `note` field is rejected with the
regenerate command. There is no field a human is meant to type here. To change
a policy, change `CANDIDATE_DEFINITIONS`, `EXCLUSION_ENTRIES`, or
`build_weight_policy` in `src/slm_lab/quantization/w8.py` and regenerate.

`check` additionally verifies six things that hand-editing cannot fake:

1. The cache dtype the candidate claims equals
   `slm_lab.contracts.static_cache.CACHE_DTYPE`.
2. `calibration_dataset_revision` still equals the value
   `configs/quantization/calibration.yaml` carries.
3. Every float16 baseline manifest under `results/manifests/onnx/` still hashes
   to the value recorded in the candidate, and still declares the same
   precision.
4. The frozen T13 protocol digest has not moved.
5. The weight-storage projection is recomputed from the model contract and the
   candidate's own exclusion policy, and must match what is recorded.
6. `candidate_canonical_json_sha256` covers the `candidate` block that is
   actually present.

A committed candidate may only ever read `precision_state: specified`.
Validation rejects any other value: `simulated` and `deployed` are evidence,
and evidence does not live in a configuration file.

## Reading a candidate top to bottom

The generated YAML preserves insertion order, so it reads as an argument rather
than as an alphabetized dump. The header comment block states the six things a
reader should take away before any field appears.

- `schema_version`, `task_id`, `candidate_id`, `plan_matrix_row`, `title` —
  identity, and which plan row this candidate answers.
- `summary`, `intent` — what the candidate is and what question it exists to
  answer. `w8a16` isolates weight error; `w8a8` asks whether the activation
  path survives 8 bits. Neither is interpretable alone.
- `model` — the T00 model contract identity, including
  `tie_word_embeddings`, which is why the vocabulary table is excluded.
- `candidate` — everything that determines what would actually be quantized,
  and nothing else. This is the block the canonical hash covers:
  - `precision_state`, `precision_state_note`, `precision_state_scope` — the
    evidence state, always `specified` here, and the scope string that must
    travel with it.
  - `weights` — the shared int8 policy, its granularity argument, the caveat
    about which ONNX axis a per-output-channel scale indexes, and a per-class
    table with logical shapes and output-channel counts.
  - `activations` — the only block that differs between the two candidates:
    dtype, granularity, the chosen range estimator *with its reason*, the
    estimators explicitly rejected and why, the tensors expected to break
    first, and a `boundary` field stating that none of those ranges has been
    measured anywhere.
  - `kv_cache` — where the candidate meets a contract it does not own. Read
    `satisfied_without_contract_change` and `change_control` before proposing
    to lower the cache dtype.
  - `graph_inventory` — the committed T21 structural evidence the exclusion
    arguments point at, so a reader can check that the operators being excluded
    are operators this graph contains, and in what number.
  - `excluded_from_quantization` — the entries, split into `policy_choice_ids`
    and `frozen_graph_constraint_ids`. Read `kind_note`.
  - `calibration` — the T40 binding, the prefill variants the observers would
    run on, the open decode-side observer question T40 left, and the inherited
    bias that must travel with any future quality delta.
  - `baseline` — the four committed float16 manifests, by canonical digest.
  - `evaluation` — the frozen T13 protocol and academic contract, the metric
    set, and the `<candidate_id>+<state>` precision-label convention a W8
    result record must use.
  - `weight_storage_projection` — arithmetic, labelled `analytic_projection` on
    every row, with its own `does_not_establish` list and a byte-level
    cross-check against the committed float16 export.
- `candidate_canonical_json_sha256` — the candidate revision.
- `deployment_routes` — Lane A and Lane B, each with what this repository
  provides, what is missing, and what it is blocked on. Both read
  `available: false`, for different reasons.
- `evidence_requirements` — the plan 7.3 measurement list as a status ledger.
  One row is `satisfied`, and it is a property of committed inputs rather than
  a measurement.
- `artifact_manifest_contract` — the four fields (plan section 17.4) that any
  future W8 artifact manifest must carry before it may name this candidate.
- `commands` — every command that regenerates, validates, or consumes this
  file.

## Policy choices versus frozen constraints

The distinction runs through the whole file and is the first thing to check
before proposing a change.

**T41 policy choices** — revisable by T41 on evidence, and the reason the
candidate hash moves when they change:

- `weights.granularity` (`per_output_channel`), `weights.scale_dtype`
  (`float32`), and the symmetric zero-zero-point encoding.
- `activations.range_estimator` for each candidate (`min_max` at int16,
  `mse` at int8) and the per-tensor activation granularity.
- The five exclusion entries tagged `policy_choice`: `rmsnorm_scales`,
  `qwen3_per_head_qk_norm`, `softmax`, `rope_sin_cos`, `residual_adds`.
- `calibration.observe_ranges_on` — all four exported prefill contexts, because
  ranges fitted at 128 tokens do not bound those at 4,096.
- `calibration.decode_side_observer_pass.t41_position` —
  `required_pending_measurement`, an argument with no measurement behind it.
- The bounded submission parameters: `retry: false`, a 3,600-second timeout,
  and the target-device selectors drawn from the plan's section 3.2 policy.
  These are T41 policy as well, but they sit under `deployment_routes`, outside
  the `candidate` block that `candidate_canonical_json_sha256` covers, so
  changing one does not move the candidate revision — it does fail the
  byte-identity check. Each selector carries the full attribute vocabulary the
  capability query observed, which identifies the device exactly and therefore
  over-constrains it: if a selector stops resolving, trim it to the device name
  before concluding the device is gone.

**Hard constraints of the frozen graph or of another task's contract** —
reversing one of these is a change request against its owner, not an edit here:

- `kv_cache.applied_dtype` is `float16` because
  `slm_lab.contracts.static_cache.CACHE_DTYPE` is, and that dtype is declared
  on 56 cache outputs of every prefill graph and on 56 inputs plus 56 outputs
  of every decode graph. Owners: **T12** (graph contract), **T20** (export
  boundary), **T23** (promoted prefill export).
- The three exclusion entries tagged `frozen_graph_constraint`:
  `tied_embedding_table` and `final_logits_projection` (the model contract sets
  `tie_word_embeddings: true`, so the embedding table and the output projection
  are one stored tensor and cannot carry two precisions) and
  `kv_cache_read_write`.
- Every shape and instance count under `weights.per_class`, derived from
  `configs/models/qwen3-0.6b.yaml` (**T00**).
- `calibration.calibration_dataset_revision` (**T40**), read at generation time
  and never transcribed.
- `baseline.manifests` digests (**T20**/**T23**).
- `evaluation.benchmark_protocol.contract_sha256` and the academic contract
  digest (**T13**).
- `graph_inventory` (**T21**).
- The pinned `qai-hub` client and QAIRT versions under
  `deployment_routes.lane_a_ai_hub_workbench.submission_parameters`, which come
  from committed T02 access evidence and the T30 handoff.

`w8a8` is the case where the two collide: plan row Q2 asks for an INT8 cache,
the frozen contract carries `float16`, and the candidate records
`satisfied_without_contract_change: false` with
`request_status: out_of_scope_change_request_for_T12_T20_T23` rather than
writing a cache precision the exported graphs do not carry.

## What invalidates these files

Any drift in the calibration revision, in a baseline manifest digest, in the
T13 protocol digest, or in the frozen cache dtype makes `check` fail rather
than silently re-anchor the comparison. That is the intended behaviour: a
candidate that quietly re-anchored would produce a quality delta that looks
comparable and is not. Regenerate, read the diff, and decide whether the frozen
candidate is still the candidate you meant.
