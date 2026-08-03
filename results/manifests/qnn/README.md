# T22 QNN candidate manifests

`S<context>.json` records how one T20/T23 reference variant was turned into the
`qnn_candidate` stage: which reference bytes went in, which transformation
catalogue was applied, what each pass measurably did, what the candidate bytes
are, and what was and was not verified. `inspection/S<context>.json` carries the
full before/after findings from the committed T21 rule engine, including the
sampled locations the manifest's compact delta leaves out.
`parity/S<context>-ort-cpu.json` is the candidate's ONNX Runtime CPU parity
record, measured by the T21 runner and only *read* by the build tool.

The first two are produced by `slm_lab.graph.qnn.build`. Neither contains a
timestamp, so `--check` is a genuine drift check. The third is produced by the
T21 runner, `slm_lab.backends.onnx_cpu`, and is only read here.

```bash
SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab PYTHONPATH=src \
  python -m slm_lab.graph.qnn.build --all-manifests

SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab PYTHONPATH=src \
  python -m slm_lab.graph.qnn.build --all-manifests --check
```

Both commands need `onnx`, `onnxruntime`, and `numpy`. The locked root
environment has none of them; that is deliberate for the T21 inspection path and
is why the `onnx`-dependent cases in `tests/qnn/` skip there.

The candidate graphs themselves are **not** committed. They live at
`${SLM_LAB_ARTIFACT_ROOT}/onnx/qnn-candidate/T22/S<context>/{prefill,decode}.onnx`
with a per-candidate `<kind>.onnx.data` sidecar; `du -sh` on that tree reports
**9.0 GB**, the same order as the reference tree it derives from. The
reference artifacts under `onnx/reference/T20/` are read and re-hashed, never
written.

## What the build tool does, in order

1. Reads the committed T20 manifest and re-computes the SHA-256 of both the
   `.onnx` protobuf and its external-data sidecar, failing if either differs
   from the manifest. Nothing is parsed before that check passes.
2. Applies the six applied passes of `configs/graph/qnn-transforms-v1.json`
   (catalogue id `qnn-candidate-v1`) in declared order, recording a structured
   effect per pass per graph kind.
3. Writes the candidate and its own sidecar, runs `onnx.checker.check_model` on
   it, and reads it back with `slm_lab.graph.onnx_reader`.
4. Asserts two post-conditions on the bytes it just wrote and fails loudly on
   either: the public boundary must be name-for-name, dtype-for-dtype and
   shape-for-shape identical to the reference (prefill 3 in / 58 out, decode
   60 in / 58 out), and all 56 T12 cache outputs must still be written by the
   contract's operator — `Concat` in prefill, `ScatterElements` in decode.
5. Scores the reference and the candidate with `slm_lab.graph.inspection` used
   as a library, against `configs/graph/onnx-risk-rules-v1.json`.
6. Measures the rejected ONNX Runtime pass into a scratch directory under the
   artifact root, records the result, and deletes the scratch output.

7. Reads `parity/S<context>-ort-cpu.json` if it exists and derives
   `verification.ort_cpu_parity` from it. The build tool never executes a
   graph, so it adopts that record's verdict only when the record's
   `graph_digests` are the candidate digests this manifest just wrote.

Nothing is written to `results/graph/`, which T21 owns. The `packages/`
subdirectory here is written by the T22 package builder and is documented with
that tool; `parity/` is written by the T21 runner, never by the build tool.

## Manifest schema, version 1

| Field | Contents |
|---|---|
| `variant_id`, `context_length`, `cache_capacity`, `opset`, `precision` | Copied from the source T20 manifest |
| `cache_contract` | The frozen T12 strategy for both graph kinds, copied forward so a reader of this file alone can see what had to survive |
| `source` | The T20 manifest path and digest, plus per graph kind the reference `relative_path`, `sha256`, `size_bytes` and external-data records. Every digest was recomputed from disk during this build |
| `transform_catalogue`, `risk_catalogue` | Path, id, and SHA-256 of both committed catalogues |
| `toolchain` | `python`, `onnx`, `onnxruntime`, `numpy`, read from the running interpreter |
| `artifacts` | `root` as `${SLM_LAB_ARTIFACT_ROOT}/onnx/qnn-candidate/T22`, plus a `prefill` and a `decode` record with `relative_path`, `sha256`, `size_bytes`, `external_data`, `input_tensors`, `output_tensors`. Deliberately the same shape as the T20 manifest's, so the same readers work on both |
| `transformations` | The ordered pass records: `id`, `order`, `applied`, `observed_issue`, `rule_ids`, `transformation`, `parameters`, and `effect` per graph kind. The rejected pass appears with `applied: false` and a measured `rejection_evidence` |
| `structural_delta` | Node counts, operator-histogram delta, boundary counts, initializer counts, inline bytes, `value_info` counts, protobuf and sidecar byte sizes, and per-rule finding counts, before and after |
| `contract_preservation` | The two post-conditions above, as recorded results |
| `verification` | `onnx_checker`, `graph_inspection`, and `ort_cpu_parity`, each a real result or an explicit `not_measured` with a reason. `ort_cpu_parity` is derived from `parity/S<context>-ort-cpu.json`; see *ONNX Runtime CPU parity of the candidate, measured* below for what it carries and for the `stale_record` state |
| `claim_boundary` | `establishes` / `does_not_establish`, in the shape T20 and T21 use |

## What the committed numbers say

Measured on 2026-08-03 with python 3.11.13, onnx 1.18.0, onnxruntime 1.28.0,
numpy 2.4.6. `onnx.checker.check_model` passed on all eight candidates and all
eight preserved 56 of 56 T12 cache writes.

| Variant | Kind | Nodes | `R-DATA-DEPENDENT-SHAPE-INPUT` | `R-SHAPE-COMPUTATION-CHAIN` | `R-LARGE-INLINE-CONSTANT` | `R-INTERNAL-DYNAMIC-SHAPE` | `.onnx` bytes |
|---|---|---|---|---|---|---|---|
| S128 | prefill | 7,634 -> 2,785 | 804 -> 0 | 127 -> 0 | 0 -> 0 | 0 -> 0 | 5,131,850 -> 949,559 |
| S512 | prefill | 7,634 -> 2,817 | 804 -> 5 | 127 -> 9 | 1 -> 0 | 0 -> 9 | 9,305,674 -> 956,509 |
| S1024 | prefill | 7,634 -> 2,824 | 804 -> 6 | 127 -> 11 | 1 -> 0 | 0 -> 9 | 18,235,014 -> 958,649 |
| S4096 | prefill | 7,634 -> 2,824 | 804 -> 6 | 127 -> 11 | 1 -> 0 | 0 -> 9 | 49,790,614 -> 958,654 |
| all four | decode | 10,191 -> 5,421 | 1,231 -> 423 | 556 -> 496 | 0 -> 0 | 0 -> 1,069 | 1,759,947 -> 1,752,536 (1,752,538 at S1024/S4096) |

Six things in that table are worth reading rather than skimming.

**The prefill protobuf stops scaling with `S`.** It was 5.1 MB to 49.8 MB across
the matrix and is now 949,559 to 958,654 bytes — a 9.70x spread collapsed to
1.01x. The reason is not compression: `X-CONSTANT-TO-INITIALIZER` turned the
O(S^2) causal mask and the 56 zero cache reserves from `Constant` attributes
into initializers, and `X-EXTERNALIZE-LARGE-TENSORS` then moved everything at or
above 1,024 bytes into the candidate's own sidecar. Inline bytes fell from
48,401,800 to 31,368 at S4096. The bytes did not vanish; they moved, and the
prefill sidecar grew correspondingly, from 1,192,085,504 to between
1,196,378,112 and 1,240,451,072 bytes.

**`R-LARGE-INLINE-CONSTANT` is cleared at every variant where it fired.** It
fires once per prefill graph at S512 and above and is 0 in all four candidates.

**The rank-1 finding is essentially gone in prefill and only cut by two-thirds
in decode.** Prefill falls from 804 to 0, 5, 6, 6; decode falls from 1,231 to
423 at every capacity, and 429 of decode's 459 `Shape` nodes survive. The reason
is structural and worth stating rather than hiding: `X-STATIC-SHAPE-FOLD` folds
only nodes whose every input is a known constant, and decode computes its
attention mask at run time from `Range`/`Greater` position arithmetic over
activation tensors (graph-inspection.md 5.4), so its `Shape` nodes read tensors
that are not constants even though their *shapes* are static. Folding those
would need the inferred `value_info`, which `X-INFER-VALUE-INFO` produces only
afterwards. A shape-aware second fold is the obvious follow-up; it is not in
this catalogue and no number here assumes it.

**The prefill residue is a byte-budget effect, not a capacity effect.** S128
reaches 0 and S512, S1024 and S4096 stop at 5, 6 and 6. The folder budgets both
the tensors it consumes and the tensors it produces at 1 MiB, and the causal
mask is 32,768 bytes at S128 but 524,288, 2,097,152 and 33,554,432 bytes above
it, so at the larger contexts a handful of mask-consuming nodes are refused
rather than evaluated. That is the budget doing exactly its job: the alternative
is a folder that will happily materialize a 33 MB tensor at build time.

**`X-DEAD-NODE-ELIMINATION` removed zero nodes, and that is a result rather than
a bug.** `X-STATIC-SHAPE-FOLD` already deletes the nodes it replaces, and these
exports contain no unreachable computation, so the node arm of pass 3 finds
nothing to do on all eight graphs. Its initializer arm does the work: 3,335 to
3,399 dead initializers removed per prefill graph and 1,629 per decode graph,
which is what stops the operands of the folded chains from staying in the
protobuf. The pass records the two arms separately for exactly this reason.

**`R-INTERNAL-DYNAMIC-SHAPE` went from silent to loud, and that is the point.**
Section 6 of `docs/results/onnx/graph-inspection.md` flags this rule as the one
silent rule that is *not* a clean result: `value_info` was empty in all eight
reference graphs, so the detector inspected 0 of 0 entries. Running shape
inference annotated every intermediate tensor in every candidate — 2,727 to
2,766 in prefill, 5,363 in decode — and the rule now has something real to
score. It reports 9 non-static interior tensors in prefill at S512 and above and
1,069 in decode. Those are not new defects introduced by T22; they are the first
measurement of something the reference graphs never declared. The evidence
boundary section 7 named is closed, and the answer is not the clean one the
static public boundary made "likely".

## The rejected transformation, measured

`X-ORT-CPU-OFFLINE-OPTIMIZATION` is recorded in the catalogue with
`applied: false`, and the build tool re-measures it on every reference graph it
reads rather than citing a planning probe. Per graph kind, in
`transformations[].rejection_evidence`:

- Node count falls hard: prefill 7,634 -> 3,040 and decode 10,191 -> 4,245,
  identically at all four capacities.
- All 28 `Softmax` nodes are decomposed into `Sub`/`Exp`/`ReduceMax`/
  `ReduceSum`/`Div`: `Sub`, `Exp`, `ReduceMax` and `ReduceSum` each go 0 -> 28
  in both graph kinds, and `Div` goes 113 -> 141 in prefill. One operator an NPU
  implements natively is replaced by five, on exactly the operator
  graph-inspection.md 5.6 ranks as precision-sensitive. Decode's `Div` count
  *falls*, 167 -> 141, so there the decomposition's `Div` arm is netted out by
  other rewrites in the same run; the other four arms are unambiguous.
- `Cast` rises from 348 to 830 in prefill and 464 to 860 in decode, which is the
  CPU execution provider's float16 fallback inserting casts around kernels it
  lacks. That is a host-specific artifact and it moves the 285 real float16 /
  float32 crossings section 5.6 counts.
- Eight opset domains are added — `ai.onnx.ml`, `ai.onnx.preview`,
  `ai.onnx.preview.training`, `ai.onnx.training`, `com.microsoft`,
  `com.microsoft.experimental`, `com.microsoft.nchwc`, `org.pytorch.aten` — and
  the tool checks each against the node list: **all eight are used by no node.**
  That weakens the single-standard-domain portability section 6 records as one
  of the three clean blocking results.
- The external-data layout is destroyed. External initializers fall from 254 to
  58, inline initializer bytes rise from 14,336 to between 1,191,990,214 and
  1,810,643,510, and no sidecar is written next to the optimized model. The
  S4096 prefill output is a single **1,811,439,962-byte** protobuf, 1.69 GiB,
  against protobuf's 2 GiB serialization ceiling.

That last point disagrees with the T22 planning probe, which recorded the S128
prefill output as an 828,930-byte protobuf plus a 1,507,512,320-byte sidecar.
This tooling measures 1,197,039,898 bytes inline and no sidecar for the same
graph, with onnxruntime 1.28.0 on the CPU execution provider. **The committed
numbers are the tool's, not the probe's.** The two runs must have differed in
their session configuration — this one sets only `graph_optimization_level` and
`optimized_model_filepath`, and no external-data session option — but this
repository has no record of the probe's, so the cause is not stated as known.
Everything else the probe reported is reproduced here exactly: the node counts,
the `Softmax` decomposition, `Cast` 348 -> 830, and all eight added domains.

## ONNX Runtime CPU parity of the candidate, measured

`parity/S<context>-ort-cpu.json` holds one `ParityEvidence` record per variant,
written by the **T21 runner**, `slm_lab.backends.onnx_cpu`, pointed at this
manifest instead of the T20 one. That reuse is the point: same protocol, same
frozen T10 workload, same bfloat16 PyTorch reference at revision
`c1899de289a04d12100db370d81485cdf75e47ca`, same four decode steps, same
`ORT_DISABLE_ALL` sessions on the CPU execution provider, and the same
`DEFAULT_ORT_CPU_TOLERANCE` the reference records in `results/graph/parity/`
were measured against. A second parity implementation would not have produced
comparable evidence. The runner resolves the graph directory from the
manifest's `artifacts.root`, so it read the `qnn_candidate` bytes and verified
their SHA-256 against this manifest before creating a session.

```bash
SLM_LAB_ARTIFACT_ROOT=<artifact-root> HF_HOME=<local-hf-cache> \
TRANSFORMERS_OFFLINE=1 PYTHONPATH=src <parity-env-python> \
  -m slm_lab.backends.onnx_cpu \
  --manifest results/manifests/qnn/S128.json --steps 4 --reference torch \
  --output results/manifests/qnn/parity/S128-ort-cpu.json
```

Measured on 2026-08-03 with onnxruntime 1.28.0, CPython 3.11.13,
`macOS-15.7.7-arm64`, `CPUExecutionProvider` alone, `ORT_SEQUENTIAL`, both
thread counts at 1. Every one of the eight candidate graphs **loaded** on the
CPU provider, every record carries `evidence_tier="real_onnxruntime_cpu"`, and
all four end `passed: true` with `failures: []` and `failure_kinds: []`.

| Variant | Kind | `cosine_similarity` | `max_absolute_error` | worst protected rel. | `mean_absolute_error` | top-1 | worst top-5 | cache |
|---|---|---|---|---|---|---|---|---|
| S128 | prefill | 0.999901 | 0.343750 | 0.261475 | 0.063132 | 1/1 | 1.00 | pass |
| S128 | decode | 0.999757 – 0.999899 | 0.296875 – 0.460938 | 0.363953 | 0.051860 – 0.066077 | 4/4 | 1.00 | pass |
| S512 | prefill | 0.999941 | 0.312500 | 0.299316 | 0.059332 | 1/1 | 1.00 | pass |
| S512 | decode | 0.999783 – 0.999894 | 0.189453 – 0.578125 | 0.488037 | 0.032551 – 0.128828 | 4/4 | 1.00 | pass |
| S1024 | prefill | 0.999967 | 0.218750 | 0.188477 | 0.037873 | 1/1 | 1.00 | pass |
| S1024 | decode | 0.999844 – 0.999954 | 0.343750 – 0.546875 | 0.449951 | 0.056867 – 0.085654 | 4/4 | 1.00 | pass |
| S4096 | prefill | 0.999954 | 0.265625 | 0.238281 | 0.054479 | 1/1 | 1.00 | pass |
| S4096 | decode | 0.999943 – 0.999966 | 0.226562 – 0.390625 | 0.308655 | 0.037631 – 0.050768 | 4/4 | 0.80 | pass |

Thresholds, for reading that table: `cosine_min` 0.9993, `atol` 1.15,
`protected_relative_max` 1.05, `top5_overlap_min` 0.8, `require_top1` true, and
**exact equality** on every cache region the T12 contract calls untouched. The
cache invariants were checked on 56 tensors per decode step, 280 tensor checks
per variant, 1,120 across the matrix, with **zero** violations and zero
`written_slot_immutable` violations.

**The candidate is bit-identical to the reference.** This is the result worth
reading twice, and it is stronger than "within tolerance". Every one of the 20
recorded steps produced a `candidate_logits_sha256` **equal to the reference
record's for the same step**, and every step's `cache_report` entry is equal
too. The metric columns above are therefore not merely close to the ones in
`docs/results/onnx/ort-cpu-parity.md` — they are the same floats, including
S4096 step 4's `top5_overlap` of exactly 0.80, the single step in the reference
set whose top-5 was not 1.00. The two runs differ only in which bytes ONNX
Runtime loaded, which the records' `graph_digests` prove: the candidate digests
are this manifest's, not the T20 manifest's.

That is the expected outcome if the catalogue is semantics-preserving, and it
is worth naming why rather than treating it as luck. None of the six applied
passes touches float arithmetic or its order: `X-CONSTANT-TO-INITIALIZER` moves
a tensor between two encodings of the same values, `X-STATIC-SHAPE-FOLD`
evaluates integer shape arithmetic whose inputs are already constants,
`X-DEAD-NODE-ELIMINATION` removes computation no output reads,
`X-EXTERNALIZE-LARGE-TENSORS` relocates bytes, `X-INFER-VALUE-INFO` adds
annotations, and `X-STAMP-CANDIDATE-PROVENANCE` writes metadata. At
`ORT_DISABLE_ALL` the runtime adds no fusion of its own on either side. Had any
number moved, the honest reading would have been that one of those six passes
is not semantics-preserving — which is exactly what this measurement was for,
and it is the reason the rejected `X-ORT-CPU-OFFLINE-OPTIMIZATION` pass, which
decomposes all 28 `Softmax` nodes, was never a candidate for this catalogue.

`verification.ort_cpu_parity` in each manifest carries the record's path and
SHA-256, its `evidence_sha256`, `evidence_tier`, `record_kind`, `task_id`,
step counts, `passed` verdict, `failure_kinds`, the tolerance, the reference
provenance, the runtime block, and the worst step per graph kind. Two things
about it are deliberate:

- The block is **derived, never asserted**. The build tool reads the record and
  adopts its verdict only when the record's `graph_digests` are the candidate
  digests the same build just wrote. A record that measured other bytes is
  reported as `status: "stale_record"` with both digest sets and no verdict,
  and a missing record stays an explicit `not_measured` naming the command that
  would produce one. `--check` re-derives all of this, so a rebuilt candidate
  whose digest moved turns the block stale on stderr instead of silently
  carrying a verdict about bytes that no longer exist.
- `record_task_id` reads `T21`, because the runner stamps its own task id into
  a fixed schema field. It is not evidence that T21 produced the file; the
  `graph_digests` are what identify what was measured.

### What the parity measurement does not license

It is 20 steps of one frozen workload per variant, on one host, on the CPU
execution provider, at one ONNX Runtime version, at one optimization level. It
says nothing about whether the candidate compiles, whether any operator is
supported by a vendor toolchain, how it behaves on an accelerator, or what it
costs. It also inherits every boundary of the T21 protocol, including the one
that matters most here: the newly written cache slot is checked for having been
written and being finite, **not** for holding the right values.
`claim_boundary` in each manifest is adjusted to match — a passing,
on-these-bytes measurement removes `onnxruntime_numerical_parity_of_the_candidate`
from `does_not_establish` and replaces it with a narrower entry naming the
recorded steps, the one frozen workload, and the CPU execution provider as the
limits of what was shown. A failing measurement would have left the original
entry in place, and a `stale_record` or a missing record leaves it too.

## What `--check` proves, and what it does not

`--check` re-derives everything: it re-hashes the reference graphs, rebuilds
both candidates into the artifact root, re-runs `onnx.checker`, re-inspects both
graphs, re-runs the ONNX Runtime probe, and compares the rendered JSON against
the committed files. It exits non-zero with `stale report:` or `missing report:`
on stderr if anything would change. On 2026-08-03 it exited 0 for all four
variants.

So `--check` proves that the committed manifests and inspection reports are
exactly what this toolchain reproduces from these reference bytes, and that the
build is deterministic down to the candidate SHA-256.

It does **not** prove that any candidate compiles, that any operator is
supported by any vendor toolchain, or that any accelerator will place it. Nor
does it, by itself, prove the candidate runs: this tool rewrites and measures
graphs, it never executes one. What proves that is the separate parity record
above, which `--check` re-reads and re-derives the manifest block from — it
does not re-measure it. Regenerating that evidence means re-running the T21
runner in the parity environment. A reduced finding count is a smaller
population of a structural pattern, not evidence of convertibility, and
`claim_boundary.does_not_establish` says so in every file.
