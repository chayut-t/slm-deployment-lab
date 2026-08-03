# T20 defect: float16 prefill graphs cannot be loaded by the ONNX Runtime CPU provider

Date: 2026-08-02
Task: `T20` (defect), found by `T21`
Status: **closed.** Fixed in the exporter, and `T23` promoted the fix into the
attested reference artifacts and regenerated every committed record against
them. See "Downstream implications", which records what the promotion did and
where three predictions in this analysis turned out to be wrong.

> **Reading note.** Everything from here to "Downstream implications" describes
> the investigation as it stood before promotion, and is left intact: the root
> cause, the ruled-out hypotheses, the operator census and the numerical
> equivalence evidence are the durable content and none of it was invalidated.
> Where the analysis made a *prediction* about what promotion would require or
> affect, and the prediction was wrong, the correction is recorded in
> "Downstream implications" rather than by editing the original claim.

## Intended outcome

Run the four T20 float16 reference graph pairs through ONNX Runtime's CPU
execution provider at `ORT_DISABLE_ALL`. That level is the one the T21 parity
runner asks for, deliberately: with every graph transformation switched off,
the numbers that come back are the graph's, not the optimizer's.

## Environment

- Repository branch: `task/prefill-scatter-cache-write`, based on `main` @ `11a9c57`
- Host: Darwin `24.6.0`, `arm64`, macOS 15.7.7 (reported by the runner as
  `macOS-15.7.7-arm64-arm-64bit`)
- CPython `3.11.13` (the T20 attestation pinned `3.11.15` at the time; see
  "What is not proven". The promotion moved that pin to `3.11.13`, the
  interpreter that actually runs the exporter.)
- `torch` 2.7.1, `transformers` 4.51.3, `onnx` 1.18.0, `onnxruntime` 1.28.0,
  `numpy` 2.4.6. `environments/onnx-cpu/README.md` requires the `onnxruntime`
  and `numpy` versions to be chosen and recorded together; every number below
  reached the comparators through `numpy_tensor_factory`.
- Environment: `.ai-local/envs/t21-ort-cpu`, the separate parity environment —
  ONNX Runtime is deliberately absent from the locked root environment
- Artifacts: `${SLM_LAB_ARTIFACT_ROOT}/onnx/reference/T20`, opset 18, float16

## Observed symptom

All four prefill graphs fail identically at session creation. All four decode
graphs load. The graphs are the same precision, from the same export run.

```
onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL :
Type Error: Type (tensor(float)) of output arg
(InsertedPrecisionFreeCast_/Cast_163_output_0) of node (/Cast_163)
does not match expected type (tensor(float16)).
```

The failure is at *load*, not at inference: no kernel ever runs, so nothing
downstream of T20 that needs the strict level can proceed at all.

| Graph | `ORT_DISABLE_ALL` | `ORT_ENABLE_BASIC` | `ORT_ENABLE_EXTENDED` | `ORT_ENABLE_ALL` |
|---|---|---|---|---|
| prefill S128/S512/S1024/S4096 | load fails | loads | loads | loads |
| decode S128/S512/S1024/S4096 | loads | loads | loads | loads |

Measured here on onnxruntime **1.28.0**, the only version this document can
support. The session that found the defect reported reproducing it on 1.20.1
and 1.22.0 as well, but neither run is recorded anywhere in the repository —
not in a manifest, not in the T21 worklog — so those two are hearsay for the
purposes of this analysis and are not evidence that the behaviour is
version-independent. Establishing that would need a recorded run per version.

## Root cause

ONNX Runtime's CPU execution provider registers no float16 kernel for `Pad`.
`Reshape`, `Cast`, `Concat` and the scatter operators all have one.

The exporter's prefill cache write was
`torch.nn.functional.pad(key.to(torch.float16), (0, 0, 0, capacity - prompt_length))`.
The `.to(torch.float16)` becomes a `Cast`, and its consumer is a `Pad` with no
float16 kernel. ORT's `CastFloat16Transformer` walks the graph looking for
float16 nodes it must demote, inserts a "precision-free" cast around the
unsupported node, and then fails type inference on the result. The error names
the inserted cast, not the `Pad`, which is why the message points away from the
cause.

`CastFloat16Transformer` and `InsertCastTransformer` are one transformer under
two names — the first is the registered name of the second's instance, and both
strings appear in the ORT binary. This document uses whichever name the
surrounding evidence used, and they are interchangeable throughout.

The operator census is the whole difference. At S128:

| | prefill | decode |
|---|---|---|
| `Pad` | 56 | 0 |
| `ScatterElements` | 0 | 56 |

56 is 28 layers x 2 tensors. Decode writes its cache with a scatter, loads
fine, and runs fine at the same precision. One operator choice separated a
usable graph from an unloadable one.

### The missing kernel, read off ORT's own optimized graph

Setting `SessionOptions.optimized_model_filepath` and running ONNX shape
inference over the result shows what ORT actually intends to execute. At
`ORT_ENABLE_BASIC`, where the unfixed graph does load, the chain producing
`key_cache.0` is:

```
old:  Add(float32) -> Pad(float32) -> Cast(to=float16) -> Reshape(float16)
new:  Add(float32) -> Cast(to=float16) -> Concat(float16) -> Reshape(float16)
```

The exporter emitted `Cast(->float16)` *before* the `Pad` in both cases. In the
old graph ORT moved the `Pad` back above that cast so it runs in float32, then
cast to float16 afterwards. It rewrote the graph rather than run `Pad` in
float16, which is the missing kernel stated as a positive observation rather
than inferred from an error message. In the new graph the cast stays exactly
where the exporter put it and `Concat` runs natively in float16, with no
inserted cast at all.

At `ORT_DISABLE_ALL` that rewrite is unavailable, and the type mismatch it
would otherwise have resolved surfaces as the load failure above.

`ORT_ENABLE_BASIC` and above load because the optimizer clears the way for the
rewrite shown above before the float16 transformer's check can fail. That is
what made this look like an optimization-level quirk rather than a missing
kernel, and it is
why the defect survived export-time validation: `onnx.checker` accepts the
graph, and the T20 manifest's `claim_boundary` correctly says it does not
establish `onnxruntime_numerical_parity`.

### What was ruled out

- Not a precision problem in general: decode is float16 too and loads.
- Not an artefact of one graph or one context: all four prefill graphs fail
  identically and all four decode graphs load, on one measured onnxruntime
  version. Whether it is version-independent is **not** established here; see
  the version note above.
- Not fixable by disabling the transformer: disabling `InsertCastTransformer`
  does not help.
- No minimal reproducer was found. A standalone
  `Add -> Cast(fp16) -> Pad(dynamic pads) -> Reshape -> fp16 output` graph
  loads at all four levels, so the trigger needs more surrounding structure
  than that. This analysis therefore rests on the full graphs, not a toy.

## Fix

`PrefillWrapper` in `src/slm_lab/export/onnx_matrix.py` now expresses the
zero-extension as a concatenation with a zero reserve:

```python
reserve_length = capacity - prompt_length
...
reserve = torch.zeros(
    BATCH_SIZE, NUM_KEY_VALUE_HEADS, reserve_length, HEAD_DIM,
    dtype=torch.float16, device=input_ids.device,
)
...
torch.cat((key.to(torch.float16), reserve), dim=2).reshape(...)
```

Both forms compute the same tensor: the prompt prefix in
`[0, prompt_length)` and zeros in `[prompt_length, capacity)`. Only the
lowering differs. `cat` becomes ONNX `Concat`, which has a CPU float16 kernel.

`tests/export/test_onnx_matrix.py::test_prefill_cache_write_lowers_to_concat_and_never_pad`
pins the lowering: it exports a tiny prefill graph and asserts zero `Pad` nodes
and that every one of the 56 cache outputs is produced by `Reshape` over
`Concat`. Run against the old lowering, that test sees 56 `Pad` and `Reshape`
over `Pad`, and fails — so it is a guard, not a tautology.

### Why `Concat` and not a scatter

The obvious alternative is to mirror `DecodeWrapper` exactly: preallocate a
capacity-sized buffer and scatter the prompt prefix into it. It would load —
decode proves `ScatterElements` works in float16 — but it is the worse graph
for the platform this repository targets first:

- The reserve is a compile-time constant, so `Concat` keeps the write fully
  static. A scatter reintroduces an index operand for a write whose address is
  known at export time.
- `docs/results/onnx/graph-inspection.md` section 5.2 ranks the indexed cache
  write as the second-highest deployment risk precisely because a runtime write
  address defeats compile-time DMA descriptor generation on an accelerator.
  Adding 56 more indexed scatters to prefill would extend that risk to the
  graph that currently does not have it.
- A scatter needs a capacity-sized zero destination — `[1, 8, 4224, 128]`
  float16 at S4096, in the `[1, 8, C, 128]` layout the contract uses
  everywhere. `Concat` needs only a reserve-sized one, `[1, 8, C - S, 128]`;
  the reserve is 32, 64, 128 and 128 positions for S128, S512, S1024 and
  S4096.

`index_copy` and `slice_scatter` were considered and not pursued: they are the
same shape of graph as the scatter with less certain support in the legacy
`torch.onnx` exporter at opset 18.

Expressing the reserve as a broadcast from a scalar zero
(`torch.zeros(1, 1, 1, 1).expand(...)`) was tried in the hope of emitting one
node instead of 56 constants. It produced a **byte-identical** graph.

The pass responsible is not the one the export call disables. Probing the
stages separately: `torch.jit.trace` records `aten::zeros` and `aten::expand`
as distinct nodes, and both are gone by the time ONNX is emitted. The
eliminating pass is TorchScript's `_jit_pass_constant_propagation`, which
`torch.onnx.utils._optimize_graph` runs unconditionally — its
`_disable_torch_constant_prop` switch is private and `torch.onnx.export` does
not expose it. It is **not** `do_constant_folding`, which this module sets to
`False` and which governs the later ONNX-level fold: exporting the same module
with that flag both `True` and `False` yields the identical single
65,610-byte `Constant` and no `Expand` in either case.

The 56 duplicated reserve constants are therefore accepted. They stay inline in
the protobuf rather than moving to external data, which is what keeps every
graph's `.onnx.data` file byte-identical and lets the T20 attestation keep
recording one shared `external_data_sha256`.

The mechanism that keeps them inline is worth stating precisely, because it is
not the size threshold. Each copy is 65,536 bytes at S128 and 262,144 at
S1024/S4096 — 64x to 256x the configured
`external_data_threshold_bytes = 1024`. They are not externalized because torch
emits the reserve as an `onnx::Constant` node **attribute**, and
`export_onnx_graph` calls `onnx.save_model(..., convert_attribute=False)`, so
only entries in `graph.initializer` are considered for externalization.
`inspect_onnx_artifact`'s inline-initializer guard walks `graph.initializer`
too, so it does not see them either.

The shared-`external_data_sha256` invariant therefore rests on two things:
`convert_attribute=False` at the save call, and torch choosing an attribute
rather than an initializer for a traced constant. Hoisting the reserve to a
registered buffer would make it an initializer, at which point the 64-256 KB
tensor would be externalized and the invariant would break. That is why it was
not hoisted.

**Both halves are guarded**, which was verified by breaking each one through
the production export path and running the committed test's own assertion body
against the result:

| Export | `test_prefill_cache_write_lowers_to_concat_and_never_pad` |
|---|---|
| unmodified | passes |
| `convert_attribute=True` | fails — `ValidationError: Data of TensorProto ... should be stored in prefill.onnx.data` |
| reserve as a registered buffer | fails — `KeyError: 'reserve'` |

The first breaks because the reserve moves into external data and
`numpy_helper.to_array` can no longer read it; the second because the reserve
becomes an initializer rather than a node output, so the producer lookup misses.
Neither assertion should be weakened without replacing the guarantee.

## Evidence

All measurements below are real, taken on the host described above, against
graphs exported from the fixed exporter into
`${SLM_LAB_ARTIFACT_ROOT}/onnx/candidate/concat-reserve/`. The attested
reference artifacts were not modified.

### Loading

All four fixed prefill graphs create a CPU-EP session at all four optimization
levels, including `ORT_DISABLE_ALL`, with the contracted 3 inputs and 58
outputs. All four unfixed prefill graphs still fail at `ORT_DISABLE_ALL` with
the message quoted above. The S128 pair was additionally driven through the
unmodified T21 parity CLI at `ORT_DISABLE_ALL` in both states: the fixed graph
ran to completion, the unfixed one raised the same error out of
`onnxruntime_cpu_session_factory` before any reference was built.

### Operator census, prefill, before and after

Only operators whose count changed are listed. Counts are identical across all
four contexts after the change.

| Operator | S128 before | S512/S1024/S4096 before | after (all) |
|---|---|---|---|
| `Pad` | 56 | 56 | 0 |
| `Sub` | 56 | 56 | 0 |
| `Gather` | 58 | 58 | 2 |
| `Shape` | 121 | 121 | 65 |
| `Slice` | 200 | 200 | 144 |
| `Cast` | 460 | 460 | 348 |
| `Reshape` | 395 | 396 | 284 |
| `Transpose` | 394 | 394 | 338 |
| `ConstantOfShape` | 118 | 118 | 62 |
| `Mul` | 542 | 542 | 486 |
| `Constant` | 3177 | 3178 | 2729 |
| **node count** | **8753** | **8755** | **7634** |

`Concat` stays at 341. The 56 `Concat` nodes that used to assemble each `Pad`'s
padding vector are replaced one-for-one by the 56 `Concat` nodes that now
perform the cache write.

The other ~1,120 removed nodes are the rest of that padding-vector plumbing:
`torch.onnx`'s `constant_pad_nd` lowering rebuilds the ONNX `pads` operand from
the input rank on every call, which cost about 20 nodes per cache output. The
new form asks for none of it. The two-node S128-vs-others anomaly recorded in
`docs/results/onnx/graph-inspection.md` section 4 disappears with it: all four
prefill graphs are now 7,634 nodes.

Graph protobufs grow because the reserve constant is duplicated 56 times. Net
measured growth, which is smaller than the raw 56-copy payload because the
removed padding-vector plumbing also carried constants:

| Variant | reserve positions | one copy | 56 copies | `.onnx` before | `.onnx` after | net growth |
|---|---|---|---|---|---|---|
| S128 | 32 | 65,536 | 3,670,016 | 1,560,358 | 5,131,850 | 3,571,492 (3.57 MB) |
| S512 | 64 | 131,072 | 7,340,032 | 2,064,360 | 9,305,674 | 7,241,314 (7.24 MB) |
| S1024 | 128 | 262,144 | 14,680,064 | 3,653,613 | 18,235,014 | 14,581,401 (14.58 MB) |
| S4096 | 128 | 262,144 | 14,680,064 | 35,209,213 | 49,790,614 | 14,581,401 (14.58 MB) |

The 1,192,085,504-byte `.onnx.data` file is unchanged in every case and still
hashes to
`e9d4b051fa86283dc96a29ceb4eb99107dbe8aff1036e54628e8725e3dac5cde`.

### Boundary and identity

The T12 boundary is unchanged. `validate_onnx_contract` passed inside
`export_onnx_graph` for all four graphs, and comparing the `input_tensors` and
`output_tensors` that `inspect_onnx_artifact` records against the committed
`results/manifests/onnx/S*.json` gives an exact match for every context: 3
inputs and 58 outputs, same names, dtypes, shapes and ordering. The `contract`
block, including `prefill_sha256`, is byte-identical.

| Variant | fixed prefill `.onnx` sha256 | bytes |
|---|---|---|
| S128 | `464892a720e208a62932a6189e200ecc7433e2f629cbb6ee29775679ddf4efc3` | 5,131,850 |
| S512 | `6fafbe126f4758b6590e697c70b7bb83a5bca58181b193bf2bebe9bb1383670f` | 9,305,674 |
| S1024 | `61d1b8b8b56f97dc44c93c27b02cafece5a2691ac33e51105c91444122521940` | 18,235,014 |
| S4096 | `cbed215ca4cda9e5ac6fe1d8545795bd853cab321ca14e9dcadd167d720490f0` | 49,790,614 |

All eight `.onnx.data` files in the staged tree — four fixed prefill, four
untouched decode — still carry the single attested
`e9d4b051fa86283dc96a29ceb4eb99107dbe8aff1036e54628e8725e3dac5cde`.

### Numerical equivalence

The change is numerically inert, which is stronger than "within float16
rounding":

- Run directly, at both `ORT_ENABLE_BASIC` and `ORT_ENABLE_ALL`, the old and
  new S128 prefill graphs produce **all 58 outputs bitwise identical** on the
  frozen T10 workload.
- Through the full T21 parity runner at `ORT_ENABLE_BASIC`, the old and new
  S128 graphs produce **identical `candidate_logits_sha256` at all five
  steps**.
- The reserve `[prompt_length, capacity)` is exactly zero in every cache output
  of every fixed graph.

### Parity, `evidence_tier = real_onnxruntime_cpu`

Four teacher-forced decode steps after prefill, PyTorch reference, ONNX Runtime
1.28.0 CPU EP, `ORT_DISABLE_ALL`, `intra_op_num_threads = 1`,
`ORT_SEQUENTIAL` — a level at which these graphs could not previously be
loaded at all.

| Variant | step | graph | cosine | max abs err | mean abs err | top-1 | top-5 |
|---|---|---|---|---|---|---|---|
| S128 | 0 | prefill | 0.99990096 | 0.34375 | 0.063132 | agree | 1.0 |
| S128 | 1 | decode | 0.99982154 | 0.296875 | 0.051860 | agree | 1.0 |
| S128 | 2 | decode | 0.99975692 | 0.4609375 | 0.066077 | agree | 1.0 |
| S128 | 3 | decode | 0.99987728 | 0.431640625 | 0.062377 | agree | 1.0 |
| S128 | 4 | decode | 0.99989854 | 0.3671875 | 0.056241 | agree | 1.0 |
| S512 | 0 | prefill | 0.99994149 | 0.3125 | 0.059332 | agree | 1.0 |
| S512 | 1 | decode | 0.99978278 | 0.578125 | 0.128828 | agree | 1.0 |
| S512 | 2 | decode | 0.99982865 | 0.189453125 | 0.032551 | agree | 1.0 |
| S512 | 3 | decode | 0.99982764 | 0.22265625 | 0.035690 | agree | 1.0 |
| S512 | 4 | decode | 0.99989362 | 0.25390625 | 0.044923 | agree | 1.0 |
| S1024 | 0 | prefill | 0.99996710 | 0.21875 | 0.037873 | agree | 1.0 |
| S1024 | 1 | decode | 0.99992436 | 0.3515625 | 0.075896 | agree | 1.0 |
| S1024 | 2 | decode | 0.99995413 | 0.34375 | 0.056867 | agree | 1.0 |
| S1024 | 3 | decode | 0.99991771 | 0.3515625 | 0.066467 | agree | 1.0 |
| S1024 | 4 | decode | 0.99984423 | 0.546875 | 0.085654 | agree | 1.0 |
| S4096 | 0 | prefill | 0.99995351 | 0.265625 | 0.054479 | agree | 1.0 |
| S4096 | 1 | decode | 0.99995675 | 0.390625 | 0.050768 | agree | 1.0 |
| S4096 | 2 | decode | 0.99996573 | 0.28125 | 0.037631 | agree | 1.0 |
| S4096 | 3 | decode | 0.99994314 | 0.28125 | 0.039618 | agree | 1.0 |
| S4096 | 4 | decode | 0.99994254 | 0.2265625 | 0.040918 | agree | 0.8 |

Every step across all four contexts: top-1 agreement, zero non-finite logits,
and `cache_report.passed` true with no slot-immutability violation. The
teacher-forced token sequence was reproduced exactly in every case.

The cache invariants are not the same set on every step, and the distinction
matters here. Step 0 is prefill, and `check_prefill_cache_tensor` checks
exactly one invariant per cache tensor — **`prefill_reserve_zero`**, that
`[prompt_length, capacity)` is zero — plus a single `prefill_valid_length`
check that the graph reported `valid_length == prompt_length`.
`prefill_reserve_zero` is the invariant this change is most directly
responsible for, since the reserve is precisely what the `Concat` now
contributes, and it passed on all 56 tensors at all four contexts. Steps 1-4
are decode, and `compare_cache_tensor` checks the five-invariant set
`prefix_preserved`, `slot_written`, `slot_finite`, `tail_untouched` and
`valid_length_increment`. Those five are decode's, not prefill's; the decode
graph is unmodified here and they are reported as controls.

Top-5 overlap is 1.0 on 19 of the 20 steps. The exception is S4096 step 4, at
0.8: one of the five highest-scoring tokens differs between the runtime and the
reference while the top-1 choice still agrees. That still clears
`top5_overlap_min = 0.8`, and it is a reordering deep in a float16-vs-bfloat16
logit tail rather than a state fault — the cache invariants for that step pass
on all 56 tensors. It is recorded here because it is the only step that is not
a clean 1.0, not because anything about it is diagnosed.

`passed` is nevertheless `false` on every run, for one reason and it is not
this change: the logits fall outside `DEFAULT_ORT_CPU_TOLERANCE`, whose own
`status` field says `proposed_unvalidated: no ONNX Runtime run has confirmed
these thresholds`. Across all 20 steps the observed `max_absolute_error` is
0.1895-0.5781 against `atol = 0.25`, and `max_protected_relative_error` is
0.1738-0.4880 against `protected_relative_max = 0.10`. `cosine_similarity`
(0.999757-0.999967) clears `cosine_min = 0.999` everywhere, and `top5_overlap`
clears `top5_overlap_min = 0.8` everywhere.
The reference runs in bfloat16 while the graph is float16, so a gap
of this size is expected rather than surprising. The old graphs produce the same
logits bit-for-bit at `ORT_ENABLE_BASIC`, which is the direct evidence that
retolerancing — not this fix — is the open question. Confirming or replacing
those thresholds remains T21's, not this change's, work.

## Reproduction

Every command below ran on the host in the Environment block. `$P` is the
parity interpreter and `$ROOT` the external artifact storage; both are host
paths and neither is repository state.

```bash
P=.ai-local/envs/t21-ort-cpu/bin/python          # in the primary checkout
ROOT=/Volumes/T9/slm-deployment-lab
export SLM_LAB_ARTIFACT_ROOT=$ROOT
export HF_HOME=$ROOT/hf-cache
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PWD/src                       # must be THIS worktree
$P -c 'import slm_lab; print(slm_lab.__file__)'  # confirm the source tree
```

`PYTHONPATH` is load-bearing: the parity environment carries an editable
install pointing at the primary checkout, so without it every command silently
exercises the wrong source tree.

**1. Observe the defect, and that decode does not share it.** For each
`S128 S512 S1024 S4096`, create a CPU-EP session at each of `ORT_DISABLE_ALL`,
`ORT_ENABLE_BASIC`, `ORT_ENABLE_EXTENDED`, `ORT_ENABLE_ALL`:

```python
import onnxruntime
options = onnxruntime.SessionOptions()
options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
onnxruntime.InferenceSession(path, sess_options=options,
                             providers=["CPUExecutionProvider"])
```

Prefill raises `Fail: ... InsertedPrecisionFreeCast_/Cast_163_output_0 ...` at
`ORT_DISABLE_ALL` only; decode and the three higher levels succeed.

**2. Operator census.** Load with `onnx.load_model(path,
load_external_data=False)` and count `collections.Counter(n.op_type for n in
model.graph.node)`. The "before" column of the census table is also readable
without any artifact, straight from the committed
`results/graph/S{128,512,1024,4096}.json` under `graphs.prefill.op_histogram`.

**2b. What ORT intends to execute.** Set
`SessionOptions.optimized_model_filepath` at `ORT_ENABLE_BASIC`, load the dump
with `onnx.shape_inference.infer_shapes(..., strict_mode=False)`, and walk back
from `key_cache.0` through `{output: node}`, reading each `Cast`'s `to`
attribute and each edge's `value_info` element type. That produces the two
chains in "The missing kernel, read off ORT's own optimized graph".

**3. Export the fixed graphs.** The T20 CLI refuses on this host because
`_verify_runtime` pins CPython 3.11.15, so the graphs were produced by calling
the production `export_onnx_graph` directly with the production contract,
config and example inputs — see "What is not proven". Once the attestation is
re-forged the supported command is:

```bash
$P -m slm_lab.export.onnx_matrix export --context 128
$P -m slm_lab.export.onnx_matrix validate --context 128 --write-manifests
```

**4. Parity.** The graphs were hard-linked into a staging tree laid out exactly
like the reference one (`<staging>/onnx/reference/T20/S{N}/`) so that the T21
CLI resolves them with nothing but `--artifact-root`, and driven with a
provisional manifest:

```bash
$P -m slm_lab.backends.onnx_cpu \
  --manifest <provisional-manifest>.json \
  --artifact-root $ROOT/staging/T20-concat-reserve \
  --steps 4 --reference torch \
  --graph-optimization-level ORT_DISABLE_ALL \
  --output <evidence>.json
```

Exit status is 1 on every context, from `evidence.passed` being false for the
tolerance reason above, not from any error. The control that shows the defect
through the same CLI is the identical command with the committed manifest and
`--artifact-root $ROOT`; it raises out of `onnxruntime_cpu_session_factory`
before a reference is ever built.

After promotion the same measurement is the documented one, with no
`--artifact-root` and the committed manifest:

```bash
$P -m slm_lab.backends.onnx_cpu \
  --manifest results/manifests/onnx/S128.json \
  --steps 4 --reference torch \
  --graph-optimization-level ORT_DISABLE_ALL \
  --output results/graph/parity/S128-ort-cpu.json
```

**5. The red committed test.**

```bash
SLM_LAB_ARTIFACT_ROOT=$ROOT HF_HOME=$ROOT/hf-cache TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=$PWD/src $P -m pytest tests/export tests/onnx -q
# 1 failed, 214 passed, 1 skipped
```

Dropping `SLM_LAB_ARTIFACT_ROOT` turns the failure into a skip and reports
`214 passed, 2 skipped`. That is the difference between a host that can run the
measurement and one that cannot, and it is why the failure is easy to miss.

**6. The regression guard, and that it guards both halves of the
external-data invariant.**

```bash
PYTHONPATH=$PWD/src $P -m pytest \
  tests/export/test_onnx_matrix.py::test_prefill_cache_write_lowers_to_concat_and_never_pad -q
```

To show it is not a tautology, re-run that test's assertion body against
exports that break each half — `onnx.save_model` patched to
`convert_attribute=True`, and a `PrefillWrapper` variant holding the reserve in
a registered buffer. Both fail, with `ValidationError` and `KeyError`
respectively.

**7. The blast radius.**

```bash
python .ai-local/scratch/promotion_audit.py claims
```

This extracts every numeric token from the four hand-written documents that
describe these graphs and classifies each against measured post-promotion
values: `MOVES` (54), `AMBIGUOUS` (49), `STATIC` (69, suppressed), and
`UNCLASSIFIED` (969 occurrences, 179 distinct, printed in full).

**What it is and is not.** It is a cross-check that no number in those
documents goes unexamined, because every token lands in exactly one bucket. It
is *not* an oracle for step 5. `MOVES` and `AMBIGUOUS` are candidate lists
whose classification a human still has to confirm — the first version of this
tool inherited the hand-curated needle list it was meant to replace and missed
eight occurrences, and even now `118` means both "prefill `ConstantOfShape`
count" and "decode boundary tensors" depending on the line. `UNCLASSIFIED` is a
review queue, not a clean bill of health.

An earlier note here claimed a `git grep` sweep was "the authority for
promotion step 5". It was not: absence of a hit proves only that the spelling
searched for was absent. The `` `Pad=56` `` search reported "no committed
occurrence" while `results/graph/S*.json` contained `"Pad": 56` — true that no
hand edit was needed there, since step 4 regenerates those files, but the
output read as proof of absence and was not.

## What is not proven

- **These graphs are not T20 evidence.** They were exported by calling
  `export_onnx_graph` directly, which skips `_verify_runtime`, because this host
  runs CPython 3.11.13 and the attestation pins 3.11.15. The T12 contract check
  inside `export_onnx_graph` did run and did pass for all four; the attestation
  chain did not. Their digests are recorded here as measurements, not as
  attested artifact identities. *(Since resolved: the promotion re-ran the
  export through the attested chain on 3.11.13, and `validate
  --write-manifests` confirmed these same four digests. They are T20 evidence
  now.)*
- **The manifests used to drive the parity runner are provisional.** They are
  the committed T20 manifests with the prefill artifact record replaced by a
  freshly computed `inspect_onnx_artifact` record — the same producer T20 uses —
  so the runner could hash-verify the graphs it was about to execute. Their
  `export_provenance` blocks still describe the superseded export run. They live
  under `.ai-local/scratch/` and are not committed. *(Since resolved: the
  committed `results/manifests/onnx/S*.json` were regenerated and describe the
  promoted export.)*
- **No compiler and no device were involved.** The argument that `Concat` is
  friendlier than a scatter for the Qualcomm lane follows from the ranked risks
  in `docs/results/onnx/graph-inspection.md`; it is reasoning about the graph,
  not a QNN conversion result. Nothing here establishes compiler acceptance,
  accelerator placement, or any latency or memory claim.
- ~~**The tolerance question is untouched.** Whether
  `DEFAULT_ORT_CPU_TOLERANCE` should be widened, kept, or replaced by a
  different comparison against a float16 rather than bfloat16 reference remains
  open, and belongs to T21. It is also a blocker to a green tree; see
  "Downstream implications".~~ **Resolved by `T23` — correction 3 below.** The
  framing was also wrong about the shape of the answer: the choice was not
  "widen, keep, or change the reference dtype", because the superseded `atol`
  rejected float32 and so was not a threshold any implementation could meet.
- **The parity environment was mutated to take these measurements.** `uv pip
  install jsonschema pytest` into `.ai-local/envs/t21-ort-cpu` added
  `jsonschema`, `jsonschema-specifications`, `referencing`, `rpds-py`, `attrs`,
  `pytest`, `pluggy`, `iniconfig` and `pygments`. Nothing was upgraded or
  removed, and `torch`, `transformers`, `onnx`, `onnxruntime` and `numpy` are
  untouched, so the measurements stand. But `environments/onnx-cpu/README.md`
  asks for the environment's versions to be recorded, and any pin taken from a
  `pip freeze` of this environment from now on will silently capture all nine
  packages, none of which the parity runner needs.

## Downstream implications

### The promotion happened

`T23` promoted the fix. The four `Concat` prefill graphs replaced the defective
ones under `${SLM_LAB_ARTIFACT_ROOT}/onnx/reference/T20`, the export was
re-attested on the interpreter that actually performed it, and every
machine-generated record was regenerated against the promoted artifacts. What
that settled:

- **All four prefill graphs now create a CPU-EP session at `ORT_DISABLE_ALL`.**
  The defect this document describes is not reachable from the committed
  manifests any more.
- **The decode graphs and the external-data sidecar are byte-identical after
  re-export.** All four decode digests and the single
  `external_data_sha256` `e9d4b051fa86…` are unchanged, which is what lets one
  attestation keep covering the whole set. The prefill digests all moved.
- **The attestation now records CPython 3.11.13**, moving from 3.11.15 — an
  interpreter that had never run the exporter, and whose pin is what forced the
  investigation above to bypass the CLI and call `export_onnx_graph` directly.
- **`results/graph/parity/S*-ort-cpu.json` were re-measured** against the
  promoted graphs at `ORT_DISABLE_ALL`, replacing records taken at
  `ORT_ENABLE_BASIC` against the defective ones.

The paragraph this section used to open with — that the reference artifacts
still contain the defect and that every committed record remains accurate for
the artifact it describes — was true when written and is now history. It is
worth keeping the distinction it drew, because it is the one that made the
promotion safe: a record describing a superseded artifact is not *wrong*, it is
*stale*, and the two are repaired differently. A wrong record is corrected in
place; a stale one is regenerated by its producer and its prose reconciled
afterwards.

### Three predictions in this analysis were falsified

Recorded as corrections rather than silently removed, because the way each one
failed is more useful than the claim itself.

**Correction 1 — "`LEARN-11` is not affected" was wrong, and no search for a
graph number could have found it.**

The blast-radius list below states: *"`LEARN-11` is **not** affected — AIMET and
calibration material for T40, citing no `Pad`, node count or T20 digest."* That
enumeration is correct about what `LEARN-11` cites and still reaches the wrong
conclusion. `configs/quantization/calibration.yaml` lists
`configs/models/qwen3-0.6b-onnx-export.json` under `inputs[]` and pins its
`canonical_json_sha256`. The re-export rewrote that config's
`evidence_attestation` block, so the pin went stale and **22 tests in
`tests/quantization/test_calibration.py` failed** against the promoted export
(repaired in commit `45d10f9`, which moved the pin from `e9b47945…` to
`2d38d1ad…` using the contract's own `calibration generate` producer rather
than by hand). `ai/tasks/learning_lane.yaml` in turn pins a digest of
`calibration.yaml`, so `LEARN-11` went stale with it.

The lesson is about how blast-radius enumeration fails. Every search this
analysis proposed was for a *graph* fact — a digest, a node count, an operator
name, a byte size. The dependency that broke was a pin on the digest of a
**config**, one link further out: nothing in `calibration.yaml` mentions `Pad`,
7,634, or any graph SHA-256, and it would not have matched any of those
searches at any level of diligence. Enumerating a blast radius by searching for
the values that changed finds only first-order readers. Second-order readers
pin the *identity of a file* that contains those values, and they are found by
walking the pin graph — `inputs[]` blocks, `canonical_json_sha256` fields,
`learning_lane.yaml` sources — not by grepping for numbers.

**Correction 2 — the promotion order prescribed below is not executable.**

Step 1 says to commit the fixed exporter "together with a config that has no
`evidence_attestation` block", and step 2 says to re-export from that commit.
That cannot be done: `load_export_config` calls `_load_export_attestation`,
which raises `ExportConfigurationError("evidence_attestation must be a
schema-version 1 mapping")` when the key is absent, and every `onnx_matrix`
entry point begins with `load_export_config`. The CLI cannot run at that commit
at all. The order is not merely awkward, it is a fixed point that does not
exist: the commit the export must be attested *to* is a commit at which the
exporter refuses to run.

The chain that actually worked breaks it into two commits and moves
re-attestation out of the CLI entirely:

1. **Commit A** (`321b11b`) — the attested commit. The `evidence_attestation`
   block is removed from the config and `FROZEN_EXPORT_CONFIG_SHA256` is
   repinned to the block-less bytes (`be885020…` → `02a2fe3c…`). The
   `onnx_matrix` CLI cannot run at this commit; the state is inherent and
   transient. This commit also adds
   `scripts/export/write_export_attestation.py`, a separate script rather than
   an `onnx_matrix attest` subcommand precisely because the subcommand would
   have to bypass its own trust root before it could restore it.
2. **Commit B** (`d3494fd`) — the attestation is regenerated from artifacts on
   disk by that script, which hashes every file it records so no digest is ever
   hand-edited. It names `321b11ba` as `exporter_commit`, moves
   `runtime_python_version` to 3.11.13, and `FROZEN_EXPORT_CONFIG_SHA256` is
   repinned again (`02a2fe3c…` → `ede6cfc0…`).
3. **Export** from commit B, which now loads.
4. **`validate --write-manifests`** — this is the step that *confirms* the
   digests. The attestation written in commit B asserts what the bytes should
   be; `validate` re-checks all of them against the bytes the re-export
   actually wrote, and writes the manifests only if they agree. Attestation and
   verification are therefore separate acts by separate producers, which is the
   property the one-commit order silently gave up.

**Correction 3 — "the tolerance question is untouched" is resolved, and the
green-tree framing was wrong.**

Both halves of the claim under "A committed test is red on any host that has
the artifacts" — that a green tree needs two independent pieces of work, and
that the tolerance one is a hard blocker — were right about the *count* and
wrong about the *nature* of the second piece. The choice offered was "confirm,
widen, or replace `DEFAULT_ORT_CPU_TOLERANCE` against a real measurement", with
widening understood as the suspect option. What the measurement showed is that
`atol = 0.25` **rejects float32**: running the pinned reference at float32
against itself at bfloat16, with no ONNX anywhere, misses that threshold too,
and at S512 step 1 the exact answer misses by 0.609 against the graph's 0.578.

A threshold no implementation can meet is not a tolerance that was too tight;
it is a broken instrument, and the action it calls for is neither "confirm" nor
"widen" but "re-derive". `DEFAULT_ORT_CPU_TOLERANCE` was replaced by a budget
derived from the two dtypes' unit roundoff, the measured logit scale and the
28-layer residual depth — `atol` 0.25 → 1.15, `protected_relative_max` 0.10 →
1.05, `cosine_min` 0.999 → **0.9993, tightened** — with `rtol`,
`top5_overlap_min`, `require_top1`, `relative_floor` and the exact cache rules
confirmed unchanged. `TOLERANCE_STATUS` reads `derived_and_measured…`. All four
contexts pass at `ORT_DISABLE_ALL`. The derivation, the float32 control, and
why this is a repair rather than an accommodation are in
`docs/results/onnx/ort-cpu-parity.md`; the analysis above was right that this
work did not belong to the exporter change.

### One prediction held: the clean report sits on a one-byte boundary

"A risk finding that does not fire, by one byte", below, predicted that the
reserve constants would not trigger `R-LARGE-INLINE-CONSTANT` because the S1024
and S4096 reserve is *exactly* 262,144 bytes against
`inspection.py`'s `inline_bytes <= max_bytes` skip and a rule `max_bytes` of
262,144. **Confirmed against the regenerated reports.** `results/graph/S*.json`
gained no findings: `R-LARGE-INLINE-CONSTANT` still fires once per prefill
graph at S512, S1024 and S4096 and not at all at S128, and in each case the
detail's total equals its largest — 524,288, 2,097,152 and 33,554,432 bytes
respectively, the causal mask alone. Not one of the 56 reserves is counted at
any context. The clean report is therefore sitting on the boundary exactly as
predicted, and the warning stands: one more cache position, or a rule tightened
by a single byte, and all 56 become findings at two contexts.

### A committed test is red on any host that has the artifacts

> **Resolved by `T23`.** Both pieces of work below were done: the export was
> promoted, and `DEFAULT_ORT_CPU_TOLERANCE` was re-derived (correction 3). The
> section is kept because the *diagnosis* — a red test that hides as a skip on
> any host without the artifact root — is a durable hazard, and because the
> two-item list below is exactly where the framing went wrong.

`tests/onnx/test_onnx_cpu_parity.py::test_real_onnxruntime_cpu_parity_when_available`
**fails** in the parity environment whenever `SLM_LAB_ARTIFACT_ROOT` points at
real T20 graphs. It loads the committed S128 manifest and calls
`onnxruntime_cpu_session_factory()`, which defaults to `ORT_DISABLE_ALL`, so it
dies on exactly the defect this document describes:

```
1 failed, 214 passed, 1 skipped
```

The failure is pre-existing and is not caused by the exporter change — the test
reads the committed manifest and therefore the unfixed reference graphs. It is
recorded here because the repository is otherwise green and nothing else says
so: without the artifact root set the test skips, which is how it stays
invisible on a host that has no external storage mounted.

**Replacing the reference export does not turn it green.** Measured by
replicating the test's own code path against the fixed staged graphs:

```
sessions created at ORT_DISABLE_ALL: yes
assert evidence.evidence_tier == REAL_ONNXRUNTIME_CPU -> True
assert evidence.cache_report.passed                   -> True
assert evidence.passed                                -> False
failure kinds: ('numerical_tolerance',)
```

The two assertions the fix satisfies are at lines 1734 and 1735. Line 1739 is
`assert evidence.passed`, and `evidence.passed` is false on every context for
the `DEFAULT_ORT_CPU_TOLERANCE` reason documented above. So a green tree needs
**two** independent pieces of work, and the tolerance one is a hard blocker
that this change neither performs nor makes easier:

1. Promote the exporter fix into the reference artifacts (steps 1-5 below).
2. Confirm, widen, or replace `DEFAULT_ORT_CPU_TOLERANCE` against a real
   measurement — the work its own `status` field
   (`proposed_unvalidated`) has been waiting for. Until that lands, the one
   test in the repository that can produce a real ONNX Runtime measurement is
   red on any host equipped to run it.

Promoting the fix means replacing the reference export, and that is a
commit-gated sequence because the T20 attestation is deliberately anchored to
Git. `configs/models/qwen3-0.6b-onnx-export.json` pins the eight graph digests,
the exporter commit, and the runtime Python version; `FROZEN_EXPORT_CONFIG_SHA256`
in the exporter pins that config's bytes; `_trusted_export_config_bytes`
requires the on-disk config to equal `HEAD`'s copy; and `_export_provenance`
requires the attested commit's copy of the config to equal the current one with
the `evidence_attestation` block removed — which is why the currently attested
commit `631fd70` carries a config with no attestation block at all.

The promotion order that satisfies those checks — **not executable as written;
see correction 2 above for the chain that was actually used**:

1. Commit the fixed exporter together with a config that has no
   `evidence_attestation` block. This is the commit the new export is attested
   to. *(This is the step that cannot work: `load_export_config` refuses a
   config with no attestation block, so nothing in the `onnx_matrix` CLI runs
   at this commit.)*
2. Re-export all eight graphs into `onnx/reference/T20` from that commit, on an
   interpreter whose version will be recorded truthfully.
3. Add the attestation block back with the new run id, the commit from step 1,
   that interpreter version, and the eight measured digests; update
   `FROZEN_EXPORT_CONFIG_SHA256` to the new config bytes.
4. Regenerate, in order: `results/manifests/onnx/S*.json`
   (`onnx_matrix validate --write-manifests`), `results/graph/S*.json`
   (`slm_lab.graph.inspection --all-manifests`), and the T40 baseline parity
   record, which re-hashed all 16 files. `results/graph/parity/` now holds four
   records measured against the pre-promotion graphs at `ORT_ENABLE_BASIC`;
   every one must be re-measured against the promoted graphs, and at
   `ORT_DISABLE_ALL`, which promotion is what makes possible.
5. Update the hand-written records. **Do not work this list from memory, and
   do not trust prose — including this prose.** Four successive attempts to
   enumerate it by reasoning were each incomplete. Generate it instead:

   ```bash
   python .ai-local/scratch/promotion_audit.py claims
   ```

   *(That private prototype is superseded. Its committed successor is
   `scripts/audit/audit_reference_graph_claims.py`, which adds a strict
   `citations` mode that exits non-zero on any bound claim disagreeing with the
   evidence, and takes `--baseline-ref <pre-promotion commit>` to populate
   `MOVES`. Two further "enumerate it by reasoning" attempts were found
   incomplete after this was written, bringing the count to six; the rule below
   stands unchanged and stood up.)*

   That extracts *every* numeric token from `graph-inspection.md`,
   `ort-cpu-parity.md`, `results/graph/README.md` and the T20 export worklog
   and classifies each against measured post-promotion values. It currently
   reports **54 `MOVES` occurrences** (edit these), **49 `AMBIGUOUS`**
   (literals with two meanings in these documents — rule on each), and **969
   `UNCLASSIFIED`** across 179 distinct values, printed in full. Its
   completeness claim is only that every number is *accounted for*: `MOVES` and
   `AMBIGUOUS` are candidate lists that still need a human, and
   `UNCLASSIFIED` is a review queue, not a clean bill of health. The summary
   below is a reading of that output, not a substitute for it.

   **Digests — the most damaging class, because a stale one names a graph that
   no longer exists.**
   - `graph-inspection.md` §2, lines 62-71: four manifest SHA-256s and eight
     graph SHA-256s. The four prefill rows go stale, and so do all four
     manifest rows, since regenerating the manifests changes their own digests.
     Lines 57-60 assert these are *simultaneously* the T20 manifest values and
     the digests of the bytes actually inspected, so a partial edit makes the
     document contradict itself. The four decode digests must **not** move.
   - `ort-cpu-parity.md:725` quotes the S128 prefill digest `a61ed2ef…`; the
     decode digest beside it stays. This file is a `LEARN-10` reading.
   - Regenerated by step 4, listed so the audit reconciles:
     `results/manifests/onnx/S*.json` — each manifest embeds the attestation
     for **all four** contexts (`S128.json:4902/4906/4910/4914`), so all four
     files change when any one prefill digest does — `results/graph/S*.json`,
     and `t40-baseline-parity-2026-08-02.json` (16 dual-digest entries).

   **Counts and sizes in `graph-inspection.md`.**
   - §4 table lines 136-142: prefill node counts, `.onnx` sizes, and the
     **"Op types" column, 28 → 26** — `Pad` and `Sub` are the only operators
     that disappear. Line 147's `28` is `NUM_LAYERS` and stays; the audit
     flags both spellings as `AMBIGUOUS` for exactly this reason.
   - Line 163 and the §9 learner question at line 712: "grows 22.6x" → 9.70x.
   - Line 167 explains the two-node S128-vs-S512 anomaly, which disappears —
     all four prefill graphs become 7,634 — and lines 168-169 carry the
     `1,257`/`1,258` shape-defining-input populations that become 922.
   - Line 197 rank table `8,753-8,755`; §5.5 line 414 repeats it *and* says
     "across **28 operator types**" → 26.
   - Line 200 rank-7 population `239` → 127, matching §5.1 line 218's
     "`Shape=121`, `ConstantOfShape=118` … (239 total)" → 65, 62, 127.
   - Line 206 "804 of 1,257": **804 does not move** — measured unchanged in the
     regenerated report — only the denominator does, to 922. Line 193's
     rank-table `804` therefore needs **no** edit. That is a measurement, not
     an inference; without running the inspection it would look like it moved.
   - Line 403 "459 `Shape` nodes against prefill's 121" → 65.
   - Line 421 top-five operators: `Constant=3178`, `Mul=542`, `Cast=460`,
     `Reshape=396` → 2729, 486, 348, 284. `Unsqueeze=1568` stays.
   - Lines 424-426: "254 `MatMul` … under 3% of the node list (2.9% in
     prefill)" → 254/7,634 = **3.33%**, so both the figure and the "under 3%"
     framing become false; and "**31 distinct operator types** across all eight
     graphs" → 29, since `Pad` and `Sub` occurred only in prefill.
   - §5.6 line 443 "285 of 460 prefill `Cast` nodes" and the line 449 table's
     total-casts column `460` → 348. The `142`/`143` split and the `285`
     crossing count must be **recomputed** by re-running §5.6's own snippet;
     do not scale them. The decode row is unchanged. Lines 198, 465 and 482
     repeat `285` — check which graph each refers to.
   - §6 line 503, `R-CONTROL-FLOW-OP`: "no `If`, `Loop`, or `Scan` in any of
     the **31 operator types**" → 29. The conclusion holds; the population it
     is argued from does not. §6 is otherwise untouched by this change.
   - §5.4 lines 353-358 "Share of `.onnx` file": `2.1 / 25.4 / 57.4 / 95.3%` →
     `0.64 / 5.63 / 11.50 / 67.39%`. That materially weakens the section's
     "localizes essentially all of that growth to one node" framing — the mask
     stops dominating at S128 through S1024.
   - §5.4 line 367 and the §8 claims table line 607: the S4096 residual
     `157,928` → 14,833,032, and "four 32,768-byte int64 vectors" becomes four
     262,144-byte reserves. Line 607 is a different row from line 611.
   - §8 reproduction output, lines 659-661. **The S128 largest inline
     attribute changes identity**: the 65,536-byte reserve displaces the
     32,768-byte mask, so `largest=` becomes
     `(65536, '/Constant_99', (1, 8, 32, 128))`, `next4` becomes four more
     65,536-byte reserves, `inline_attr_total` 63,704 → 3,728,776 and `rest`
     30,936 → 3,663,240.

   **`ai/worklogs/2026-07-30-T20-onnx-export-matrix.md` line 258** carries the
   same `1,560,358` / `35,209,213` figures. Easy to miss because it is a
   worklog, not a results document, and it is what makes `LEARN-05` a
   downstream reader.

   **`results/graph/README.md` needs no size edit.** Its figures are the
   1,192,085,504-byte sidecar, which does not change, and the ~8.9 GB storage
   total, which does not move materially against ~40 MB of added protobuf.
   *(Held, and now checked rather than reasoned: the sidecar is byte-identical
   before and after, and the ~8.9 GB figure is the eight sidecars alone,
   8 × 1,192,085,504 = 9,536,684,032 bytes. What moved is the T40 record's
   `recorded_total_bytes` over all sixteen files, 9,586,211,364 →
   9,626,186,972 — the +39,975,608 of prefill protobuf this note anticipated.)*

   **`results/graph/S*.json` gains no findings.** `R-LARGE-INLINE-CONSTANT`
   still does not fire — see the boundary note below — and
   `R-DATA-DEPENDENT-SHAPE-INPUT` keeps its count of 804. The counts that do
   move in the regenerated reports are `R-GRAPH-NODE-COUNT` (8,753/8,755 →
   7,634) and `R-SHAPE-COMPUTATION-CHAIN` (239 → 127).

   Then rebuild the sheets with
   `scripts/learning/build_learning_sheet.py --all --record` and re-run
   `scripts/ai/render_task_status.py`, because `ai/tasks/learning_lane.yaml`
   records a digest per cited document. Per `configs/learning/checkpoints.yaml`:
   - **`LEARN-10`** reads `graph-inspection.md` and `ort-cpu-parity.md`
     directly. This is the one that matters.
   - **`LEARN-05`** is affected indirectly, through the T20 worklog above.
   - ~~`LEARN-11` is **not** affected — AIMET and calibration material for T40,
     citing no `Pad`, node count or T20 digest.~~ **False; see correction 1.**
     `LEARN-11` is affected through `configs/quantization/calibration.yaml`,
     which pins the export config's `canonical_json_sha256`. The enumeration of
     what it cites is accurate and the conclusion drawn from it is wrong,
     which is the point.

### A risk finding that does not fire, by one byte

The reserve does not trigger a new `large_inline_constant` finding, so
`results/graph/S*.json` gains no findings from this change. That is a
coincidence, not a margin: `inspection.py:773` skips a tensor when
`inline_bytes <= max_bytes`, `configs/graph/onnx-risk-rules-v1.json:233`
sets `max_bytes` to 262,144, and the S1024 and S4096 reserve is **exactly**
262,144 bytes. One more cache position, or a rule tightened by a single
byte, and all 56 reserves become findings at those two contexts. Anyone
re-tuning that rule or changing `CONTEXT_VARIANTS` should know the current
clean report is sitting on the boundary.

**Confirmed after promotion.** The regenerated `results/graph/S*.json` gained no
findings, and `R-LARGE-INLINE-CONSTANT`'s detail totals equal its largest at
every context where it fires — the mask alone, with none of the 56 reserves
counted. The warning is therefore live, not hypothetical.

~~Until then, anything needing a loadable float16 prefill graph at
`ORT_DISABLE_ALL` must use the candidate export, not the reference.~~ The
promotion made the reference export the loadable one; there is no longer a
separate candidate to reach for.
