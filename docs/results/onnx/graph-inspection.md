# ONNX graph inspection: the T20 static reference graphs

Task: `T21`
Date: 2026-08-02
Status: **structural measurement; no compiler, runtime, or device**

That line describes the state of *this report's evidence*, not the state of the
task. The task's status lives in `ai/tasks/task_graph.yaml`, and nothing here
should be read as a completion claim for it. The sibling report
(`docs/results/onnx/ort-cpu-parity.md`) opens with `Status: **no measurement**`
for the same reason: the two deliverables sit at different evidence tiers and
each says which one it is in its own header.

## 1. What this report is and is not

This report describes what the eight T20 reference ONNX graphs for Qwen3-0.6B
*contain*, and ranks the structural patterns in them by how much trouble each
is likely to cause when T22 turns these graphs into a Qualcomm QNN candidate.

It **is** a byte-level structural read of eight `.onnx` protobufs, hash-bound
to the committed T20 manifests, scored against a committed, declarative risk
catalogue.

It is **not** a compile result, a runtime result, a placement result, or a
numerical result. Nothing in this document has been through a converter, an
execution provider, or a device. Where a risk is described in terms of
converter rejection, host fallback, or partitioning, that is a *mechanism* by
which the observed structure could cause a problem, not a prediction that it
will. ONNX Runtime CPU parity is a separate deliverable
(`docs/results/onnx/ort-cpu-parity.md`); no numerical claim is made here.

The headline is good news, and it should be read before anything else in
section 5: **the T12 static contract holds end to end.** Across all eight
graphs there are zero symbolic and zero unset dimensions, zero control-flow
operators, zero nested subgraphs, and exactly one standard opset import. The
problems this report identifies are all problems *inside* an otherwise
completely static, standard, single-domain graph.

## 2. Provenance

| Field | Value |
|---|---|
| Model | `Qwen/Qwen3-0.6B` |
| Model revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Export task | T20, exporter commit `631fd70bcff9b73b81c08a2a2e0127cad07f09ca` |
| Exporter | `torch.onnx.export`, torch 2.7.1, transformers 4.51.3, onnx 1.18.0, python 3.11.15 |
| Attention implementation | `eager`; constant folding disabled at export |
| Producer string in every graph | `pytorch 2.7.1` |
| IR version in every graph | 8 |
| Opset import in every graph | default domain (`""`), version 18 |
| Precision | float16 weights and cache, float32 logits |
| Risk catalogue | `configs/graph/onnx-risk-rules-v1.json`, id `onnx-deployment-risk-v1`, 15 rules |
| Catalogue SHA-256 | `21f0cf537c8aef98691cdbd26c11f1f21c530d9e97cbf2e54d0c83d40cb412dc` |
| Inspection module | `slm_lab.graph.inspection` (schema version 1), reader `slm_lab.graph.onnx_reader` |
| Inspection output | `results/graph/S128.json`, `S512.json`, `S1024.json`, `S4096.json` |

Every graph is identified by the SHA-256 the inspection recomputed from the
file on disk and matched against the T20 manifest before parsing it. A
mismatch is a hard error, so the digests below are simultaneously the T20
manifest values and the values of the bytes actually inspected.

| Variant | Manifest | Manifest SHA-256 | Graph | Graph SHA-256 |
|---|---|---|---|---|
| S128 | `results/manifests/onnx/S128.json` | `2d0212d7deeba5d11d6a3e4ea065cc39cbe8b3b92b37474eb58d94355f98adb7` | `S128/prefill.onnx` | `a61ed2ef1e3f1ef9313f33ee13ab5af5dc79029291afbd328cb0aaaea470dfd1` |
| S128 | | | `S128/decode.onnx` | `e200ecd27e1ab83d2bea17de030c0a0c8a0eea08c6f182eed41c04a457c421d2` |
| S512 | `results/manifests/onnx/S512.json` | `bf06fd5293d3aeab87ac4ea35be661b74d810b57f993d872db2fa4acb8362cd4` | `S512/prefill.onnx` | `05ea2106fdd4dcd37fab537be08dd9c03a0ecfe974c40d876796236c7a48280e` |
| S512 | | | `S512/decode.onnx` | `ed2c8b52bd284685a6c549b7ebadd4db257c93e17ec7cae6b09c5b7561e36c8f` |
| S1024 | `results/manifests/onnx/S1024.json` | `9280bf9d8845bd1a57db75e441d781cd39971888ef9910d689f53c23de6cac67` | `S1024/prefill.onnx` | `b19aec5f5b6f9a71462b3777d5de0154b5eac6242e0baaa68ab4fe7992bbb5a6` |
| S1024 | | | `S1024/decode.onnx` | `4d25bbd1894213d3539827ddd6bf10ea07bdfa653db5421de40a6fdb726f8759` |
| S4096 | `results/manifests/onnx/S4096.json` | `8c3a426c7a5bb685a3c2c2315c8adb3ed94941033303f096c1dca8dba3d0817a` | `S4096/prefill.onnx` | `c70a7facdb4428c6dc612aa1ed9e4e9bd43e3e1155b097eb4d9d241691a9e988` |
| S4096 | | | `S4096/decode.onnx` | `ace3468aee92e93a2db33f54f6dbd07e4af2163d683ef5fd066e62b60fbf94cd` |

Each of the eight graphs has its **own** external-data sidecar next to it —
`S128/prefill.onnx.data`, `S128/decode.onnx.data`, `S512/prefill.onnx.data`, and
so on. The eight files are byte-identical: each is 1,192,085,504 bytes and each
has SHA-256 `e9d4b051fa86283dc96a29ceb4eb99107dbe8aff1036e54628e8725e3dac5cde`,
which is the single digest each T20 manifest records under
`artifacts.<kind>.external_data`. Identical content is not one shared file:
storing the set costs **8 x 1.19 GB**, and `du -sh` on
`${SLM_LAB_ARTIFACT_ROOT}/onnx/reference/T20` reports **8.9 GB**. Size storage
from that number, not from 1.19 GB. None of these files is ever opened by this
tooling (section 3).

## 3. What was measured, and how

No `onnx`, `onnxruntime`, `torch`, `numpy`, or vendor compiler was installed or
used. `slm_lab.graph.onnx_reader` decodes the protobuf wire format directly
with the standard library and recovers the model's *structure*: opset imports,
IR version, producer, graph inputs/outputs/`value_info`, initializer metadata,
and the full node list with attributes. It retains only `len(raw_data)` for
tensor payloads, so a 33 MB graph costs bounded memory. It never opens,
follows, or reads an external-data sidecar, so every statement below about
weights is a statement about a declaration, not about weight bytes.

`slm_lab.graph.inspection` then applies the 15-rule catalogue. Each rule names
a detector, a severity, a rationale, and a mitigation, and each finding carries
the concrete numbers the detector observed plus up to eight sample locations.
Findings are ranked in the JSON by severity, then by descending count.

Three consequences of this method must be carried into every conclusion:

1. **Structure is not behaviour.** The reader is explicitly not a validator: it
   does not check operator schemas, type consistency, topological order, or
   opset compatibility. It reports what the graph contains.
2. **The catalogue's severities are review judgements**, bound to the stated
   target context (QNN / Hexagon HTP ahead-of-time compilation, secondarily
   ONNX Runtime CPU and CUDA). The catalogue itself says no severity in it is
   derived from an executed compile job. T22 is the task that replaces these
   judgements with measured compiler evidence.
3. **`value_info` is empty in all eight graphs.** The exports carry no
   intermediate shape or type declarations whatsoever. This has two direct
   effects. First, the internal-dynamic-shape rule cannot fire on these graphs
   at all: it is silent because there is nothing to inspect, not because the
   interior was found to be static. Second, dtypes for intermediate tensors had
   to be reconstructed by `resolve_element_types`, a forward pass that seeds
   declared boundary, initializer, `Constant`, and `Cast` types and then uses
   two tables of opset-18 type constraints: `TYPE_PRESERVING_OPS`, whose
   entries tie the output dtype to a named input, and `FIXED_OUTPUT_TYPE_OPS`,
   whose entries fix the output to one concrete type whatever flows in (the
   comparison and logical operators produce `bool`; `Shape` produces `int64`).
   Anything outside both tables is left unresolved rather than guessed, and a
   `Cast` whose source stays unresolved is reported as `unknown->` instead of
   being assigned a direction. The not-guessed rule applies *per output* too:
   for the two listed operators whose secondary outputs are bound to a
   different type variable — `Dropout`'s bool `mask` and
   `LayerNormalization`'s `Mean`/`InvStdDev` — only the constrained output is
   resolved. Neither operator occurs in these eight graphs, so no number in this
   report depends on that; it matters for the `value_info` follow-up in
   section 7. With both tables in place, no `unknown->` cast remains in any of
   the eight graphs — see section 5.6 for what that changed.

## 4. The eight graphs

| Variant | Capacity `C` | Graph | Nodes | Op types | Inputs | Outputs | Initializers (external) | Non-static dims | `value_info` | `.onnx` bytes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S128 | 160 | prefill | 8,753 | 28 | 3 | 58 | 310 (254) | 0 | 0 | 1,560,358 |
| S128 | 160 | decode | 10,191 | 29 | 60 | 58 | 310 (254) | 0 | 0 | 1,759,947 |
| S512 | 576 | prefill | 8,755 | 28 | 3 | 58 | 310 (254) | 0 | 0 | 2,064,360 |
| S512 | 576 | decode | 10,191 | 29 | 60 | 58 | 310 (254) | 0 | 0 | 1,759,947 |
| S1024 | 1152 | prefill | 8,755 | 28 | 3 | 58 | 310 (254) | 0 | 0 | 3,653,613 |
| S1024 | 1152 | decode | 10,191 | 29 | 60 | 58 | 310 (254) | 0 | 0 | 1,759,947 |
| S4096 | 4224 | prefill | 8,755 | 28 | 3 | 58 | 310 (254) | 0 | 0 | 35,209,213 |
| S4096 | 4224 | decode | 10,191 | 29 | 60 | 58 | 310 (254) | 0 | 0 | 1,759,947 |

The boundary matches the frozen T12 contract in `src/slm_lab/contracts/static_cache.py`:
prefill takes `input_ids`, `attention_mask`, `position_ids` and returns
`last_logits` plus 28 `key_cache.L` / `value_cache.L` pairs plus
`valid_length`; decode adds the 56 incoming cache tensors and `valid_length` to
its inputs (3 + 56 + 1 = 60) and returns `next_logits`, 56 `present_*.L`
tensors, and `updated_valid_length` (58).

Two contrasts in that table are worth reading carefully, because they teach
more than the totals do.

**Prefill scales with `S`; decode does not.** Every decode protobuf is
1,759,947 bytes with an identical node count of 10,191 and an identical
26,706 bytes of inline node-attribute tensors, differing only in its hash.
The decode graph's capacity-dependent tensors are all boundary declarations,
and the four capacities (160, 576, 1152, 4224) all encode as two-byte protobuf
varints, so the serialized size does not move. The deeper reason is that decode
stores no mask at all and builds one at run time; section 5.4 gives that
evidence in full, because it is the contrast that explains prefill's growth.
The prefill protobuf grows 22.6x from S128 to S4096, and section 5.4 localizes
essentially all of that growth to one node.

**S128 prefill has two fewer nodes than the other three prefill graphs**
(8,753 vs 8,755: one fewer `Constant`, one fewer `Reshape`). The extra
`Reshape` at S >= 512 raises the shape-defining-input population from 1,257 to
1,258 while the flagged count stays at 804, so its shape input is a constant.
This inspection observes the difference and does not explain it; anyone
rewriting the export should find out why the traced graph is not identical
across variants before assuming the four prefill graphs are interchangeable.

## 5. Findings ranked by deployment impact

Eight of the 15 catalogue rules fired. The catalogue ranks by severity then
count; the ranking below is different, and deliberately so. It ranks by
**consequence for T22**, using three questions in order:

1. Can this stop a QNN conversion outright, or force the pipeline back to a
   contract-level redesign?
2. Is it upstream of the other findings, so that fixing it invalidates their
   measurements and they must be re-taken?
3. Is the cost certain (arithmetic) or possible (depends on a toolchain)?

By those criteria the traced shape residue comes first even though it shares a
`high` severity with two other rules, and the precision findings come last even
though they are numerically the most interesting, because they do not affect
whether T22 produces a package at all.

| Rank | Rule | Severity | Prefill count | Decode count | Why it is ranked here |
|---:|---|---|---:|---:|---|
| 1 | `R-DATA-DEPENDENT-SHAPE-INPUT` | high | 804 | 1,231 | Largest population, most common cause of a static-shape conversion failure, and upstream of every other count |
| 2 | `R-SCATTER-GATHER-INDEXING` | high | 1 | 57 | Load-bearing: if the indexed cache write cannot be lowered, the T12 decode contract itself has to change |
| 3 | `R-WIDE-IO-BOUNDARY` | high | 58 | 118 | The one certain cost; it is arithmetic, not a toolchain question, and it grows linearly with capacity |
| 4 | `R-LARGE-INLINE-CONSTANT` | medium | 1 (S >= 512) | not present | Single localized node, but it is 95% of the S4096 prefill protobuf and it is O(S^2) |
| 5 | `R-GRAPH-NODE-COUNT` | medium | 8,753-8,755 | 10,191 | Drives converter time and partition search; a budget to track, not a defect |
| 6 | `R-FLOAT-PRECISION-CAST` | medium | 285 | 285 | Real precision boundaries, but they bind T21 numerics and quantization, not T22 acceptance |
| 6 | `R-FLOAT-SENSITIVE-ELEMENTWISE` | medium | 367 | 367 | Same reasoning; these are where a quantized graph will diverge first |
| 7 | `R-SHAPE-COMPUTATION-CHAIN` | low | 239 | 556 | Population context for rank 1 and the natural before/after metric for a folding fix — covered inside section 5.1 rather than in its own subsection |

### 5.1 Rank 1 — Shape-defining inputs are computed, not constant

**Observed.** In the decode graph, 1,231 of 1,326 shape-defining operator
inputs are produced by another node rather than by an initializer, a graph
input, a `Constant`, or a `ConstantOfShape`. In prefill it is 804 of 1,257
(S128) or 1,258 (S512, S1024, S4096). The counts are identical across all four
context lengths for each graph kind. The watched positions are `Reshape[1]`,
`Slice[1,2,3,4]`, `Expand[1]`, and `Tile[1]` — the target-shape and bound
inputs. The sampled locations show the classic tracing residue directly, for
example `node[11] ai.onnx.Slice /Slice input[1]=/Unsqueeze_1_output_0`.
(Sampled locations spell the default domain `ai.onnx` for readability. On the
wire it is the empty string, which is what section 6's `R-NON-DEFAULT-DOMAIN`
verdict is about — no node in these graphs declares a domain at all.)

The raw material for those chains is counted separately by rank 7:
`Shape=459`, `ConstantOfShape=91`, `Range=6` in decode (556 total) and
`Shape=121`, `ConstantOfShape=118` in prefill (239 total). Note the asymmetry:
decode has nearly four times as many `Shape` nodes as prefill, and is the only
graph kind containing `Range`.

**Why it matters on a static-shape NPU.** An ahead-of-time compiler for a
fixed-function accelerator emits one command stream. It plans allocations,
tiling, and DMA descriptors from concrete dimensions. When the target shape of
a `Reshape` arrives as a tensor computed by a `Shape` -> `Gather` -> arithmetic
chain, the compiler cannot in general prove the result at compile time even
though, here, the boundary is fully static and the value provably never varies.

**Symptom to expect.** Converter rejection of the region, or the region being
partitioned onto the host, or (on ONNX Runtime) execution succeeding but with
fusion blocked and a copy at every partition edge.

**Important caveat.** This detector reports *where the pattern occurs*. It does
not prove any value actually varies. A computed shape input can be entirely
constant in practice and removable by constant folding — and given that T20
exported with `do_constant_folding=False` and every boundary dimension is
static, that is the likely situation for most of these 1,231. The count is a
measure of unfolded residue, not of genuine dynamism.

**Mitigation direction.** Run constant folding and re-inspect, or rewrite the
export wrapper so shapes come from Python integers rather than tensor
arithmetic on `Shape` outputs. Then re-run this inspection and compare: the
drop in this count, and in the rank-7 count, is the evidence that the fix
worked. Do this *first*, because every other number in this report moves when
it lands.

### 5.2 Rank 2 — The KV-cache write is an indexed scatter at a runtime index

**Observed.** The decode graph contains 56 `ScatterElements` nodes, which is
exactly 28 layers x 2 tensors, plus one `ScatterND`. The prefill graph contains
no `ScatterElements` at all and one `ScatterND`, so the single `ScatterND` —
`/model/model/ScatterND`, whose inputs come from the `Expand`/`Concat`/`Reshape`
mask prologue — is *not* a cache write; it belongs to the attention-mask
construction and is present in both graph kinds. The 56 `ScatterElements` are
the cache writes.

Reading one of them directly confirms the structure: `/ScatterElements` takes
`key_cache.0` (a graph input) as its data operand, has `axis=2` (the
`cache_position` axis of the `[1, 8, C, 128]` layout), and produces
`present_key.0` (a graph output). The write is therefore a direct
graph-input-to-graph-output edge, at an index derived from `valid_length`.

Prefill instead materializes its cache with 56 `Pad` nodes (`/Pad` through
`/Pad_55`), matching the contract's `prefill_prefix_materialization` strategy:
write `[0, prompt_length)` and zero-fill the reserve.

**Why it matters.** This is the single most load-bearing structure in the
decode graph. It implements `fixed_capacity_indexed_copy` from the T12 contract,
and if a backend cannot lower it the decode graph loses its reason to exist.
It is awkward for accelerators for a specific reason: the write address is a
runtime value, which defeats compile-time DMA descriptor generation. In a later
quantized pipeline it adds a second constraint — the scattered update and the
destination buffer must share one quantization encoding, or the cache
accumulates re-quantization error across decode steps.

The input-to-output edge form matters too: because the whole cache is declared
both as an input and as an output, an implementation that wanted to update in
place has to be granted that by the runtime; the graph itself asks for a copy.

**Symptom to expect.** Converter rejection of `ScatterElements`, or acceptance
followed by placement of the write (and everything data-dependent on it) on the
host, which would be the worst case: the accelerator would compute attention
and then hand the cache back to the CPU 56 times per token.

**Mitigation direction.** Plan section 6.5 already names the alternatives to
evaluate: *"fixed-capacity indexed update, growing/bucketed cache, shift
buffer, and runtime-managed state where public APIs expose it."* One more that
the plan does not list, and that this report suggests: a mask-and-blend write,
using `Where` against a position-comparison mask instead of a scatter at a
runtime index. Any replacement must be validated numerically
against the reference before adoption, and the replacement changes the T12
contract, which is why this is ranked above the boundary cost.

### 5.3 Rank 3 — The whole fixed-capacity cache crosses the boundary every token

**Observed.** The decode boundary carries 118 tensors (60 in, 58 out) against
the catalogue's review limit of 16 each. Every one of them has a static shape,
so the detector could size all 118 exactly.

| Variant | Capacity `C` | Cache per copy | Decode in | Decode out | Decode total per token | Cache share | Prefill in | Prefill out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S128 | 160 | 18,350,080 B (17.50 MiB) | 18,351,384 B | 18,957,832 B | 37,309,216 B (35.58 MiB) | 98.37% | 3,072 B | 18,957,832 B (18.08 MiB) |
| S512 | 576 | 66,060,288 B (63.00 MiB) | 66,064,920 B | 66,668,040 B | 132,732,960 B (126.58 MiB) | 99.54% | 12,288 B | 66,668,040 B (63.58 MiB) |
| S1024 | 1152 | 132,120,576 B (126.00 MiB) | 132,129,816 B | 132,728,328 B | 264,858,144 B (252.59 MiB) | 99.77% | 24,576 B | 132,728,328 B (126.58 MiB) |
| S4096 | 4224 | 484,442,112 B (462.00 MiB) | 484,475,928 B | 485,049,864 B | 969,525,792 B (924.61 MiB) | 99.93% | 98,304 B | 485,049,864 B (462.58 MiB) |

Note that the prefill **output** side is byte-for-byte the decode output side at
every capacity: both emit the 56 cache tensors, a `[1, 151936]` float32 logit
vector, and one `int64`. Only the input sides differ, and prefill's is trivial —
three `[1, S]` `int64` tensors, so `3 x S x 8` bytes exactly (3,072 / 12,288 /
24,576 / 98,304). Prefill's total boundary is therefore `prefill in + prefill
out`: 18,960,904 B at S128 up to 485,148,168 B at S4096.

The in/out totals are the values the detector reported; the cache column is
`cache_bytes(C)` from the T12 contract
(`1 * 2 * 28 layers * 8 KV heads * C * 128 * 2 bytes`), and it accounts for the
totals exactly. At S128 the entire non-cache part of the decode input is
1,304 bytes: `input_ids` (8) + `attention_mask` (160 x 8 = 1,280) +
`position_ids` (8) + `valid_length` (8). The output side adds 607,744 bytes of
float32 logits and 8 bytes of `updated_valid_length`. Everything else is cache.

**The arithmetic that matters.** One decode step produces one token and writes
exactly one cache position per layer — 28 x 2 x 8 x 128 x 2 = 114,688 bytes of
genuinely new data, independent of `C`. At S4096 the graph boundary declares
969,525,792 bytes for that step. The useful fraction is 0.012%. Generating 100
tokens at S4096 declares 90.3 GiB across the boundary. The ratio between the
S4096 and S128 per-token boundary is 26.0x, and it is linear in `C`, so it
keeps going: this is the term that decides whether long-context decode is
viable at all, long before any kernel efficiency question arises.

**Why it matters beyond bandwidth.** Separate input and output cache buffers
roughly double live memory: at S4096 the two sides are 462 MiB each. Descriptor
count matters too — 118 boundary tensors is 118 allocations, 118 bindings, and
118 opportunities for a layout or alignment mismatch at the accelerator edge.

**Honest limit on this number.** These are the *declared* boundary sizes. They
say what the graph asks the runtime to hand it, not what any runtime physically
copies. A runtime with I/O binding, or one that can alias an input buffer to
the matching output, may move far less. This inspection cannot tell you which;
measuring it is a runtime task.

**Mitigation direction.** Runtime-managed or in-place cache state if the target
exposes it; or stack the 56 per-layer tensors into fewer larger buffers to cut
descriptor count. Plan section 6.6 is explicit on the general point —
*"Node-count reduction is not evidence of an optimization"* — and the same
caution applies to tensor count: measure before and after.

### 5.4 Rank 4 — An O(S^2) causal mask materialized as one inline constant

**Observed.** The rule fires once, on prefill only, for S512 and above, and the
inspection localizes it to a single node in every case:

| Variant | Node | Attribute tensor | Inline bytes | Share of `.onnx` file |
|---|---|---|---:|---:|
| S128 | `node[6] /model/model/Constant_4` | float16 `[1, 1, 128, 128]` | 32,768 | 2.1% |
| S512 | `node[6] /model/model/Constant_4` | float16 `[1, 1, 512, 512]` | 524,288 | 25.4% |
| S1024 | `node[6] /model/model/Constant_4` | float16 `[1, 1, 1024, 1024]` | 2,097,152 | 57.4% |
| S4096 | `node[6] /model/model/Constant_4` | float16 `[1, 1, 4096, 4096]` | 33,554,432 | 95.3% |

The S128 row is included for honesty: the rule does **not** fire at S128,
because 32,768 bytes is below the catalogue's 262,144-byte threshold. The node
is there; it is simply small enough not to be flagged. That row was read
directly from `S128/prefill.onnx` with the same reader, not inferred.

This is exactly the growth T20's worklog attributed to "context-shaped graph
constants": at S4096 the mask alone is 33,554,432 of the file's 35,209,213
bytes. The remaining inline node-attribute tensors at S4096 total only 157,928
bytes, of which the largest four are 32,768-byte int64 index vectors.

**Why an O(S^2) materialized mask is a distinct problem, not merely a big one.**
Three reasons, in increasing order of nastiness:

1. It is stored **inline in the graph protobuf**, not in external data. T20's
   exporter forces initializers over 1,024 bytes into external data, but this
   is a `Constant` node's `value` attribute, not an initializer, so it bypasses
   that rule. Every tool that parses the graph — checker, converter, quantizer,
   partitioner — pays to load and copy 32 MiB of mask before it does anything
   useful, and some of them will do so more than once.
2. It grows quadratically while the thing it represents is a boolean triangle
   with an O(1) description. At S4096 the mask alone is 21.5x the size of the
   entire S128 prefill graph, and the same tensor at a hypothetical S8192
   variant would be 134,217,728 bytes (128 MiB). The scaling is structural,
   not incidental.
3. It becomes a quantization liability. A materialized additive mask in float16
   contains large-magnitude negative fill values alongside zeros. Any quantizer
   that treats it as an ordinary constant tensor has to choose an encoding
   spanning that range, and the sensible answer — recognizing it as a mask and
   keeping it symbolic — requires the converter to pattern-match it back out of
   the constant it was baked into.

**Mitigation direction.** The mask is the ideal candidate for reconstruction
rather than materialization: because each variant is a fixed static shape, the
mask is fully determined by `S` and `C`, and the QNN candidate can build or
supply it rather than carry it. **The decode graph already does exactly that**,
and that is the contrast that proves this is a prefill problem only. Decode
carries 26,706 bytes of inline attribute tensors at *every* capacity including
S4096 — not because it stores a smaller `[1, 1, 1, C]` mask row, but because it
stores no mask at all. Not one attribute tensor in any of the eight graphs
contains the capacity dimension; decode's largest inline attribute is a
256-byte float32 `[1, 64, 1]` on `Constant_4107`, identical at all four
capacities. Instead decode computes its mask at run time from the position
arithmetic: it is the only graph kind containing `Range` (6 nodes) and `Greater`
(1 node), and it carries 459 `Shape` nodes against prefill's 121. The chain is
visible by name — `/model/model/Range_1` produces the cache-position ramp,
`/model/model/Greater` compares it against the reshaped current position, and
`Cast_3947` widens that `bool` into the arithmetic dtype. A materialized
constant would have to scale with `C` (320 bytes at `C=160`, 8,448 bytes at
`C=4224` in float16); the observed constancy is itself the evidence that no
such constant exists. That is the same trade this section asks prefill to make:
spend a handful of shape nodes to avoid an O(S^2) inline constant.

### 5.5 Rank 5 — Graph scale

**Observed.** 8,753-8,755 nodes across 28 operator types in prefill;
10,191 nodes across 29 types in decode; identical across context lengths (bar
the S128 prefill anomaly in section 4). The review threshold is 2,000. The
busiest operators are overwhelmingly bookkeeping:

| Graph | Top five operators |
|---|---|
| prefill (S512/S1024/S4096) | `Constant=3178`, `Unsqueeze=1568`, `Mul=542`, `Cast=460`, `Reshape=396` |
| decode (all) | `Constant=3399`, `Unsqueeze=2083`, `Mul=571`, `Cast=464`, `Shape=459` |

There are only 254 `MatMul` nodes in either graph — the actual arithmetic is
under 3% of the node list (2.9% in prefill, 2.5% in decode). 31 distinct
operator types appear across all eight graphs; `Pad` and `Sub` occur only in
prefill, and `Greater`, `Range`, and `ScatterElements` only in decode.

**Why it matters.** Node count drives converter and compiler wall-clock time,
the size of the partitioning search space, and how many small operators can end
up stranded on the host between accelerator partitions. The 2,000 threshold is
a review convention chosen so the four variants are compared against a fixed
line — it is not a published limit of any toolchain, and this report makes no
claim that 10,191 nodes exceeds anyone's supported maximum.

**Mitigation direction.** Treat it as a budget tracked across the
`reference_onnx`, `qnn_candidate`, and `quantized_candidate` stages of plan
section 6.6, and record it at each stage so a change in compile behaviour can
be attributed. Most of the reduction should fall out of the rank-1 fix.

### 5.6 Rank 6 — Precision boundaries

**Observed, casts.** 285 of 460 prefill `Cast` nodes and 285 of 464 decode
`Cast` nodes cross the float16/float32 boundary. Every direction is resolved;
the breakdown is identical in all eight graphs:

| Graph | `float16->float32` | `float32->float16` | `unknown->` | Total casts |
|---|---:|---:|---:|---:|
| prefill (all four) | 142 | 143 | 0 | 460 |
| decode (all four) | 142 | 143 | 0 | 464 |

An earlier revision of this report showed 2 prefill and 3 decode casts as
`unknown->float16` and treated their sources as unresolvable. They were not:
they were operators missing from `resolve_element_types`' opset-18 tables.
`/model/model/rotary_emb/Cast_4` and `Cast_5` are fed by
`rotary_emb/Mul_1`/`Mul_2`, whose chain runs back through `Cos`/`Sin` — both
`T -> T`, both simply absent from `TYPE_PRESERVING_OPS` — to two `Cast` nodes
that declare float32. With `Cos` and `Sin` listed, that chain resolves and both
casts are observed `float32->float16` in prefill and in decode, which is where
the extra two `float32->float16` in each row come from (141 previously,
143 now).
Decode's remaining unknown, `Cast_3947`, is fed by `/model/model/Greater`,
whose opset-18 output is unconditionally `tensor(bool)`; it is a bool widening,
not a float crossing, so it drops out of the finding entirely and the decode
count falls from 286 to 285. Nothing here is a guess: `Cos`/`Sin` are `T -> T`
and the comparison operators produce `bool` by specification, and any operator
whose opset-18 constraint is not that definite is still left unresolved.

**Observed, sensitive operators.** 367 nodes per graph, identical in every one
of the eight: `Pow=113`, `ReduceMean=113`, `Sqrt=113`, `Softmax=28`. The 113s
are the RMSNorm decomposition, and reading the node names confirms the
arithmetic exactly: 28 x `input_layernorm`, 28 x `post_attention_layernorm`,
28 x `self_attn/q_norm`, 28 x `self_attn/k_norm`, plus the single final
`/model/model/norm` — 28 x 4 + 1 = 113. The 28 `Softmax` nodes are one per
layer. No `Erf`, `Exp`, `Log`, or `Reciprocal` appears anywhere.

**Why it matters.** The float16/float32 crossings are *expected by design* —
the T12 contract stores the cache in float16 and returns logits in float32 —
so they are not a defect. They matter because each one is a real precision
boundary: a float32-to-float16 cast is where overflow and underflow enter, and
a float16-to-float32 cast is where a later quantizer must decide which side it
owns. 285 of them is the number of boundaries the T21 numerical comparison and
any future quantization has to account for.

The 367 sensitive nodes are where a quantized graph will diverge from float
first: `Softmax` is transcendental and is usually implemented on fixed-point
accelerators by piecewise approximation or lookup, and the RMSNorm reduction
plus reciprocal square root has a large dynamic range.

**Mitigation direction.** Keep normalization and softmax in higher precision if
the backend supports a mixed-precision partition, and make these the first
candidates for per-tensor error inspection rather than only checking the graph
output. Plan section 6.7 lists the metrics that comparison must record.

## 6. What is not a risk here, with evidence

Seven of the 15 rules did not fire. Six of those are genuine clean results.
One is not, and is called out explicitly.

| Rule | Severity | Silent because |
|---|---|---|
| `R-BOUNDARY-DYNAMIC-SHAPE` | blocking | **Genuinely clean.** `dynamic_dimensions` is empty in all eight reports; 0 of 118 decode and 0 of 61 prefill boundary tensors has a symbolic or unset dimension. Every dimension is a concrete `dim_value`. |
| `R-CONTROL-FLOW-OP` | blocking | **Genuinely clean.** No `If`, `Loop`, or `Scan` in any of the 31 operator types across all eight op histograms. The prefill/decode split already hoisted the generation loop into host code, which is exactly the shape plan section 6.5 asks for. |
| `R-NON-DEFAULT-DOMAIN` | blocking | **Genuinely clean.** Every node is in the default domain. Each graph declares exactly one opset import, `("", 18)`. No custom operators, so the artifact is portable across the three target platforms rather than tied to one runtime's extension library. |
| `R-SUBGRAPH-ATTRIBUTE` | high | **Genuinely clean.** No node carries a `GRAPH`-typed attribute. The reader recurses into subgraphs when they exist and flattens them into the node list with a scope path; every node in all eight graphs has an empty scope. |
| `R-DATA-DEPENDENT-OUTPUT-SHAPE` | high | **Genuinely clean.** No `NonZero`, `Compress`, or `Unique`. Nothing in these graphs produces an output whose element count is a function of data values, so a static allocator has a bound for every tensor. |
| `R-STRUCTURED-TENSOR-OP` | medium | **Genuinely clean.** No `Trilu`, `CumSum`, `Einsum`, or `TopK`. Sampling stayed in host code, contractions are plain `MatMul`, and the causal mask is a constant rather than a `Trilu` — which is precisely why it shows up as the rank-4 finding instead. |
| `R-INTERNAL-DYNAMIC-SHAPE` | high | **Not a clean result.** `value_info` is empty in all eight graphs, so the detector inspected 0 of 0 entries. This is absence of information, not evidence of a static interior. See section 7. |

The blocking-severity trio being silent is the strongest single statement this
inspection can make: whatever else is wrong with these graphs, they are static,
standard, and branch-free at the boundary and in the operator set.

## 7. Evidence boundaries

This report does not establish, and must not be cited as establishing:

- **That any graph compiles.** No converter, compiler, or quantizer was run.
- **That any operator is or is not supported by any QNN converter version.**
  No vendor toolchain was consulted, installed, or queried. Every severity in
  the catalogue is a reviewed structural judgement about the stated target
  context, and the catalogue says so itself.
- **That any graph runs, or runs correctly.** No execution provider was
  invoked. Numerical claims belong to `docs/results/onnx/ort-cpu-parity.md`.
- **Anything about latency, throughput, memory residency, or placement.** The
  byte totals in section 5.3 are declared boundary sizes computed from static
  shapes, not measured transfers.
- **That the interior of the graphs is statically shaped.** `value_info` is
  empty, so `R-INTERNAL-DYNAMIC-SHAPE` had nothing to examine. The static
  public boundary makes a static interior likely, but likely is not measured.
  The follow-up that closes this is running ONNX shape inference on a machine
  with `onnx` installed, writing the inferred `value_info` back, and re-running
  this inspection; the same rule would then produce a real answer.
- **That every dtype in the graphs was declared.** No intermediate dtype in
  these exports is declared at all; each one in section 5.6 was derived by
  `resolve_element_types` from the boundary declarations plus opset-18 type
  constraints. That derivation is a specification reading, not a measurement,
  and it is only as good as its two tables — an earlier revision of this report
  escalated five casts to an evidence boundary that turned out to be nothing
  more than `Cos`, `Sin`, and `Greater` missing from those tables. An operator
  whose constraint is not definite is still left unresolved, and any
  `unknown->` direction that appears in a future report means exactly that.
- **That the reported shape-defining inputs are genuinely dynamic.** They are
  computed rather than declared constant; given a fully static boundary and
  `do_constant_folding=False`, most are probably foldable. The count measures
  unfolded residue.
- **Anything about weight values.** The external-data sidecar is never opened.
  Initializer statements are statements about declarations.

The four JSON reports carry that boundary in-band, as a `claim_boundary` object
with `establishes` and `does_not_establish` lists, in the same shape T20 uses in
`results/manifests/onnx/S*.json`. A reader who sees only the JSON — and so sees
only `"severity": "blocking"` next to a rule id — learns from the file itself
that severities are review judgements and that no compiler was run.

The reader is also a structural decoder, not a validator: it does not check
operator schemas, type consistency, topological order, or opset compatibility,
so a graph that this tooling parses cleanly can still be an invalid ONNX model.
T20's `onnx.checker` acceptance, recorded in the manifests, is the evidence for
validity; this report is not.

## 8. Reproduction

Requires the T20 graphs under the external artifact root
(`/Volumes/T9/slm-deployment-lab` on the primary machine). The eight graphs and
their eight 1.19 GB external-data sidecars — about 8.9 GB in total — are
deliberately not committed; regenerate them with the `export` commands recorded
in `results/manifests/onnx/*.json` if the artifact root is empty. Only the
compact JSON reports under `results/graph/` are committed.

Regenerate the four inspection reports:

```bash
cd /path/to/slm-deployment-lab
SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab \
PYTHONPATH=src \
python -m slm_lab.graph.inspection --all-manifests
```

Verify that the committed reports are exactly what the tooling reproduces:

```bash
SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab \
PYTHONPATH=src \
python -m slm_lab.graph.inspection --all-manifests --check
```

`--check` re-runs the full inspection, including re-hashing every graph against
its manifest, and exits non-zero with `stale report:` or `missing report:` on
stderr if any committed file would change. On 2026-08-02 it exited 0 with no
output for all four variants.

**What `--check` does and does not prove.** It proves exactly the contents of
`results/graph/*.json`: every node count, operator histogram, boundary count,
`dynamic_dimensions` entry, initializer statistic, and every finding's count,
detail, and sampled locations. That covers most of this document — every count
and byte total in sections 4, 5.1, 5.2, 5.3, 5.5, and 5.6, and the whole of
section 6, which is an argument about which rules stayed silent.

It does **not** cover the following, because they are not fields of the JSON.
Each was read directly from the named graphs with `slm_lab.graph.onnx_reader`:

| Number | Where | Why `--check` misses it |
|---|---|---|
| The `.onnx bytes` column | §4 | File size on disk, not a report field |
| Decode inline node-attribute total `26,706` | §4, §5.4 | Summed over all attribute tensors; the JSON keeps only initializer sizes |
| S4096 prefill residual `157,928` and its four 32,768-byte int64 vectors | §5.4 | Same: below the rule's 262,144-byte threshold, so never a finding |
| The S128 `32,768` mask row | §5.4 | The rule does not fire at S128, so the node is in no finding |
| "No attribute tensor in any graph carries the capacity dimension" | §4, §5.4 | Attribute tensor shapes are not report fields at all |
| The RMSNorm `28 x 4 + 1 = 113` name decomposition | §5.6 | `locations` is capped at 8 samples per finding |
| The per-node reads: `/ScatterElements`'s `axis=2` and its `key_cache.0` → `present_key.0` operands, the `/Pad` through `/Pad_55` names, `/model/model/Constant_4`'s node index and attribute shape | §5.2, §5.4 | Findings record node type, index, and name — not attributes or operand names |

The snippet below reproduces the first six rows; only the last row, the
per-node reads, is left out. Those come from walking
`read_onnx_model(path).nodes` and inspecting `node.inputs`, `node.outputs`, and
`node.attributes` for the named node.

```bash
SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab PYTHONPATH=src python - <<'PY'
import os
from collections import Counter
from pathlib import Path
from slm_lab.graph.onnx_reader import read_onnx_model

root = Path(os.environ["SLM_LAB_ARTIFACT_ROOT"]) / "onnx/reference/T20"
for variant in ("S128", "S512", "S1024", "S4096"):
    for kind in ("prefill", "decode"):
        path = root / variant / f"{kind}.onnx"
        summary = read_onnx_model(path)
        inline = [
            (attribute.tensor.inline_bytes, node.name, attribute.tensor.dims)
            for node in summary.nodes
            for attribute in node.attributes
            if attribute.tensor is not None and not attribute.tensor.external
        ]
        inline.sort(reverse=True)
        total = sum(size for size, _, _ in inline)
        capacity = {"S128": 160, "S512": 576, "S1024": 1152, "S4096": 4224}[variant]
        with_capacity = [name for _, name, dims in inline if capacity in dims]
        print(
            f"{variant:6} {kind:8} file={path.stat().st_size:>10,} "
            f"inline_attr_total={total:>10,} largest={inline[0]} "
            f"rest={total - inline[0][0]:>9,} next4={[s for s, _, _ in inline[1:5]]} "
            f"attrs_with_capacity_dim={len(with_capacity)}"
        )

pows = [n.name for n in read_onnx_model(root / "S128/prefill.onnx").nodes
        if n.op_type == "Pow"]
print(len(pows), Counter(
    next((k for k in ("input_layernorm", "post_attention_layernorm",
                      "q_norm", "k_norm") if k in name), name)
    for name in pows))
PY
```

Run on 2026-08-02 it printed, among the rest:

```
S128   prefill  file= 1,560,358 inline_attr_total=    63,704 largest=(32768, '/model/model/Constant_4', (1, 1, 128, 128)) rest=   30,936 next4=[1024, 1024, 1024, 1024] attrs_with_capacity_dim=0
S128   decode   file= 1,759,947 inline_attr_total=    26,706 largest=(256, 'Constant_4107', (1, 64, 1)) rest=   26,450 next4=[32, 32, 24, 16] attrs_with_capacity_dim=0
S4096  prefill  file=35,209,213 inline_attr_total=33,712,360 largest=(33554432, '/model/model/Constant_4', (1, 1, 4096, 4096)) rest=  157,928 next4=[32768, 32768, 32768, 32768] attrs_with_capacity_dim=0
S4096  decode   file= 1,759,947 inline_attr_total=    26,706 largest=(256, 'Constant_4107', (1, 64, 1)) rest=   26,450 next4=[32, 32, 24, 16] attrs_with_capacity_dim=0
113 Counter({'input_layernorm': 28, 'q_norm': 28, 'k_norm': 28, 'post_attention_layernorm': 28, '/model/model/norm/Pow': 1})
```

The decode line is identical at all four capacities, its largest inline
attribute is a 256-byte `[1, 64, 1]` tensor, and `attrs_with_capacity_dim=0`
holds for every one of the eight graphs — the §5.4 claim that decode carries no
materialized mask and builds one at run time instead. The four 32,768-byte
entries after the S4096 mask are the int64 index vectors named in §5.4. The
snippet prints sizes only; their `int64` element type comes from reading
`attribute.tensor.dtype` on those four nodes (`Constant_3250`,
`/model/model/Constant_34`, `/model/model/Constant_33`,
`/model/model/Constant_30`).

The §5.3 byte columns are `--check`-covered in a slightly indirect way: the
per-variant totals appear inside the `R-WIDE-IO-BOUNDARY` finding's `detail`
string, so the JSON does pin them.

Useful variations: `--manifest results/manifests/onnx/S4096.json` for one
variant, `--graph-kind decode` to restrict the graph kind,
`--rules <path>` to score against a different catalogue, and
`--location-sample-limit N` to widen the sampled locations recorded per finding
(default 8). `--artifact-root` overrides `SLM_LAB_ARTIFACT_ROOT`; with neither
set, the tool falls back to `./artifacts` and fails loudly if that is missing.

The inspection itself needs no third-party packages. It was run here with the
repository virtual environment, which has no `onnx`, `onnxruntime`, `torch`, or
`numpy` installed — that absence is a feature of this deliverable, not a
limitation of the host.

## 9. Learner checkpoint

`docs/project/learning-checkpoints.md` marks T21 as a deep-study checkpoint
whose task is to review this report and explain the most important deployment
risks. The questions below are answerable from this document plus the four JSON
reports under `results/graph/`; none of them can be answered by restating a
severity.

- [ ] Every dimension in all eight graphs is static and there is no control
  flow, yet the top-ranked finding is in the `dynamic_shape` category. Explain
  how a graph can be fully static at the boundary and still contain 1,231
  data-dependent shape inputs, and why a static-shape compiler cares.
- [ ] `R-INTERNAL-DYNAMIC-SHAPE` did not fire. Explain why that is not evidence
  that the graph interiors are statically shaped, and name the specific
  follow-up that would turn this into a real answer.
- [ ] Compute the useful fraction of the S4096 decode boundary traffic — new
  cache bytes written per token divided by total declared boundary bytes — and
  explain why that ratio, not the absolute megabytes, is the number that
  decides whether long-context decode is viable.
- [ ] The decode `.onnx` protobuf is the same 1,759,947 bytes at every capacity
  while the prefill protobuf grows 22.6x from S128 to S4096. Explain both facts
  from the graph structure — one graph materializes its mask as an inline
  constant and the other builds one at run time; name the operators that show
  which is which — and say what would happen to each at a hypothetical S8192
  variant.
- [ ] The prefill graph has one `ScatterND` and no `ScatterElements`; the decode
  graph has one `ScatterND` and 56 `ScatterElements`. Explain which of these is
  the KV-cache write, how you can tell from the committed JSON alone, and what
  the other one is doing.
- [ ] Rank 2 is ranked above rank 3 even though rank 3 is a certain cost and
  rank 2 is a possible one. Justify that ordering for T22, then argue the
  opposite case.
- [ ] `resolve_element_types` reports a `Cast` as `unknown->` when it cannot
  resolve the source dtype, and section 5.6 now has none. Explain what the two
  opset-18 tables can and cannot infer, why `Greater`'s output is resolvable
  without knowing its inputs while `Loop`'s is not, and why an `unknown->`
  count is a statement about the tooling rather than about the graph.
- [ ] Name one thing in this report that would change if T22 ran constant
  folding before conversion, and one thing that would not. Then say which
  findings' counts would have to be re-measured afterwards.
