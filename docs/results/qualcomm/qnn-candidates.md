# The T22 QNN candidate graphs: what six rewrites did, and what they proved

Task: `T22`
Date: 2026-08-03
Status: **graph rewrite plus an ONNX Runtime CPU numerical result; no compiler,
converter, or device**

That line describes the state of *this report's evidence*, not the state of the
task. The task's status lives in `ai/tasks/task_graph.yaml`, and nothing here
should be read as a completion claim for it. The two sibling reports —
`docs/results/onnx/graph-inspection.md` and
`docs/results/onnx/ort-cpu-parity.md` — open the same way, because the three
deliverables sit at different evidence tiers and each says which one it is in
its own header.

## 1. What this report is and is not

This report describes what happened when the eight T20/T23 reference ONNX
graphs for Qwen3-0.6B were put through a declarative, ordered catalogue of
graph rewrites to produce the `qnn_candidate` artifact stage, and what
measurements were taken on the result.

It **is** two things:

1. A **structural** result. Eight candidate graphs were built from
   hash-verified reference bytes, accepted by `onnx.checker`, and re-scored by
   the same committed T21 rule engine that scored the reference. Every count in
   sections 4 through 7 is a before/after pair from that engine or from the
   transform engine's own per-pass records.
2. A **numerical** result. All eight candidates were executed on the ONNX
   Runtime **CPU execution provider** at `ORT_DISABLE_ALL` using the T21
   runner, protocol, workload, reference, and tolerance without modification,
   and every recorded step passed.

It is **not** a compile result, a converter result, a placement result, or a
performance result. **No QNN converter, no QAIRT tool, no Hexagon HTP, no
device, and no Qualcomm AI Hub job was involved at any point.** No vendor
toolchain was installed, consulted, or queried. Nothing in this document
licenses a claim about QNN acceptance of any operator, about HTP placement or
partitioning, about latency, about throughput, or about memory residency.

The name `qnn_candidate` is a **stage name and an intent**, not a verdict. It
means "the artifact that will be offered to a QNN toolchain", not "the artifact
a QNN toolchain accepted". Section 9 says exactly what "ready for Workbench
submission" was allowed to mean here, and section 10 lists what none of it
establishes.

Two headlines, and the second is the more interesting one.

**The rewrites are structurally large.** Prefill falls from 7,634 nodes to
between 2,785 and 2,824, the prefill protobuf stops scaling with `S` entirely
(a 9.70x spread across the matrix collapses to 1.01x), and the rank-1 T21
finding falls from 804 to at most 6 in prefill.

**The rewrites are numerically invisible.** Every one of the 20 recorded parity
steps produced a `candidate_logits_sha256` **equal to the reference record's
for the same step**, and every step's cache report is equal too. That is the
expected outcome of a semantics-preserving catalogue, and section 8 explains
both why it was expected and what it does not prove.

## 2. Provenance

| Field | Value |
|---|---|
| Model | `Qwen/Qwen3-0.6B` |
| Model revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Source stage | `reference_onnx`, task T20; the four prefill graphs re-exported under T23 with a `Concat` cache write |
| Source producer string | `pytorch 2.7.1` in all eight reference graphs |
| Candidate producer string | `slm_lab.graph.qnn qnn-candidate-v1` in all eight candidates |
| IR version | 8, before and after, in all eight |
| Opset import | default domain (`""`), version 18, before and after, in all eight |
| Precision | float16 weights and cache, float32 logits — unchanged |
| Transform catalogue | `configs/graph/qnn-transforms-v1.json`, id `qnn-candidate-v1`, 6 applied passes and 1 recorded rejection |
| Transform catalogue SHA-256 | `0f6ffb3a3647fe5d04e0623ec766452712da3673de174beed2ed7977e8eabc28` |
| Risk catalogue | `configs/graph/onnx-risk-rules-v1.json`, id `onnx-deployment-risk-v1`, 15 rules |
| Risk catalogue SHA-256 | `f769acd86e83bc7163c4211672180b392b4134bcea6e74da66960b42c65e6c1d` |
| Transform engine / build tool | `slm_lab.graph.qnn.transforms` / `slm_lab.graph.qnn.build`, schema version 1 |
| Committed manifests | `results/manifests/qnn/S{128,512,1024,4096}.json` |
| Full before/after findings | `results/manifests/qnn/inspection/S*.json` |
| Candidate parity records | `results/manifests/qnn/parity/S*-ort-cpu.json` |
| Package records | `results/manifests/qnn/packages/S*.json` |

The toolchain block in every manifest is read from the running interpreter, not
copied from prose:

| Component | Version |
|---|---|
| CPython | 3.11.13 |
| `onnx` | 1.18.0 |
| `onnxruntime` | 1.28.0 |
| `numpy` | 2.4.6 |

Every reference graph and every reference sidecar was re-hashed from disk
before it was parsed, and the build fails if either digest differs from the
committed T20 manifest. The digests below are therefore simultaneously the T20
manifest values and the values of the bytes actually read.

| Variant | Source manifest | Manifest SHA-256 |
|---|---|---|
| S128 | `results/manifests/onnx/S128.json` | `f66d628fb59d069aa095cde1e258d29cee417eed941702e472acdd2c1db353dd` |
| S512 | `results/manifests/onnx/S512.json` | `92d03e3fa513da43b696b1352d6768448cf2ae5ca009d4723da8c2765545df56` |
| S1024 | `results/manifests/onnx/S1024.json` | `adcf51a3bc665254626c0b332f1a65a93966a4fb50520516f08c0785401cfac6` |
| S4096 | `results/manifests/onnx/S4096.json` | `7ea2db7643c8ca6b4b44c1869204e1ddd5c1f3e67258b0ddb95c1eaaf9ee8788` |

| Variant | Kind | Reference `.onnx` SHA-256 | Candidate `.onnx` SHA-256 |
|---|---|---|---|
| S128 | prefill | `464892a720e208a62932a6189e200ecc7433e2f629cbb6ee29775679ddf4efc3` | `463313d4ba05bc0f769b53efa91496cfb9b59be818966e0a496ba4da9782ee3f` |
| S128 | decode | `e200ecd27e1ab83d2bea17de030c0a0c8a0eea08c6f182eed41c04a457c421d2` | `fc220a31ddf3dc3b460d8e0a843d38376c89ec49be9dae1a7e9b66c0cd16111a` |
| S512 | prefill | `6fafbe126f4758b6590e697c70b7bb83a5bca58181b193bf2bebe9bb1383670f` | `9097e1873f23765c696f17bb7c3125c5aa06f5aeccd71e304a69d92945f406c6` |
| S512 | decode | `ed2c8b52bd284685a6c549b7ebadd4db257c93e17ec7cae6b09c5b7561e36c8f` | `47dbad3a2f4fe1356d4083506e7cff2af303e5545d6af5dacba29c8beeb31871` |
| S1024 | prefill | `61d1b8b8b56f97dc44c93c27b02cafece5a2691ac33e51105c91444122521940` | `da0ae3dcd8f27464fed7b6db92e2f52d33087e50f78e71ee0cf2b4c7dd79613d` |
| S1024 | decode | `4d25bbd1894213d3539827ddd6bf10ea07bdfa653db5421de40a6fdb726f8759` | `7d7e816d110df7d48b584f7ef37b17f844920a52f5172694e64e9a5f4dcdc1b6` |
| S4096 | prefill | `cbed215ca4cda9e5ac6fe1d8545795bd853cab321ca14e9dcadd167d720490f0` | `b2534ad68999f7df60c2be2118a44f473db66b075e4cdfd40f343b150e3613b7` |
| S4096 | decode | `ace3468aee92e93a2db33f54f6dbd07e4af2163d683ef5fd066e62b60fbf94cd` | `81394648522eaf1cbe1365915b6a6019022c19d9270390b469c07bd29312c79d` |

Each candidate carries **its own** external-data sidecar, written next to it and
never shared with the reference. That separation is a hard requirement, not
tidiness: the reference artifacts must stay byte-identical and independently
identifiable, and a shared sidecar would let one stage's rewrite invalidate the
other stage's committed digest.

| Variant | Kind | Candidate sidecar SHA-256 | Sidecar bytes |
|---|---|---|---:|
| S128 | prefill | `adbca551d245106eefb3faf0a6fe850ea1f859ffff9a28b5d68ce2d9a3fe2760` | 1,196,378,112 |
| S512 | prefill | `1a73b5dc83be5288c5247709ffd49cd1101d6cd8a70f8ad71552601a8a991b0a` | 1,201,014,784 |
| S1024 | prefill | `272d846256cc22a384a4a6f7a33779339d5153d128381b3efb0eb32dad9ba531` | 1,208,895,488 |
| S4096 | prefill | `f476f51267e0c7ee09e9e97166afa9900158b22adcf5acec895b94d96e3d0dd3` | 1,240,451,072 |
| all four | decode | `e9d4b051fa86283dc96a29ceb4eb99107dbe8aff1036e54628e8725e3dac5cde` | 1,192,085,504 |

Read the last row carefully. **The four decode sidecars are byte-identical to
the reference sidecars** — that digest is the single value all eight T20
manifests record. Pass 4 externalized nothing new in decode (section 4.4), so
the sidecar came out unchanged, which is a measurement rather than a shortcut:
the file was written by the candidate build and then hashed.

The candidate graphs themselves are **not** committed. They live under
`${SLM_LAB_ARTIFACT_ROOT}/onnx/qnn-candidate/T22/S<context>/`, and
`results/manifests/qnn/README.md` records `du -sh` on that tree as **9.0 GB**,
the same order as the 9.0 GB reference tree it derives from.

## 3. What was measured, and how

The build tool does six things per variant, in order, and every one of them is
a check rather than an assertion:

1. **Verify.** Read the committed T20 manifest, re-compute the SHA-256 of both
   the `.onnx` protobuf and its sidecar, and fail if either differs. Nothing is
   parsed before that passes.
2. **Rewrite.** Apply the six applied passes of the catalogue in declared
   order, recording a structured `effect` per pass per graph kind.
3. **Validate.** Write the candidate and its own sidecar, run
   `onnx.checker.check_model` on it, and read it back with
   `slm_lab.graph.onnx_reader`.
4. **Assert the contract.** Two post-conditions on the bytes just written, both
   fatal: the public boundary must be name-for-name, dtype-for-dtype and
   shape-for-shape identical to the reference (prefill 3 in / 58 out, decode
   60 in / 58 out), and all 56 T12 cache outputs must still be written by the
   contract's operator — `Concat` in prefill, `ScatterElements` in decode.
5. **Re-score.** Run `slm_lab.graph.inspection` as a library over both the
   reference and the candidate, against the same committed risk catalogue.
6. **Measure the rejection.** Run the rejected ONNX Runtime pass into a scratch
   directory under the artifact root, record what it did, and delete the
   output.

A seventh step reads the candidate parity record if one exists and derives
`verification.ort_cpu_parity` from it. The build tool never executes a graph,
so it adopts a record's verdict **only** when that record's `graph_digests` are
the candidate digests the same build just wrote; otherwise it reports
`status: "stale_record"` with both digest sets and no verdict.

Three consequences of this method must be carried into every conclusion.

1. **A finding count is a population, not a verdict.** The T21 rule engine
   counts occurrences of a structural pattern. A count that falls from 804 to 0
   means the pattern is gone, not that a compiler will accept the graph.
   `claim_boundary.does_not_establish` in every manifest names this directly:
   `that_a_reduced_finding_count_makes_the_graph_convertible`.
2. **The transform engine is target-neutral by construction.** No pass encodes
   a host kernel choice, an execution provider, or a vendor operator set. This
   is what makes the rejection in section 7 a design decision rather than a
   preference.
3. **`onnx.checker` acceptance is the validity evidence, and it is narrow.**
   `full_check` is `false` on all eight, so the candidate was checked against
   the ONNX operator schemas and not through a full shape-inference validation
   pass. The separate `X-INFER-VALUE-INFO` result in section 4.5 is what
   reports on shapes.

## 4. The six passes

Each subsection names the T21 finding the pass addresses, the exact rewrite,
and its measured effect. Every number is from `transformations[].effect` in
`results/manifests/qnn/S*.json`.

### 4.1 `X-CONSTANT-TO-INITIALIZER` — a canonicalization that unlocks the rest

**The T21 finding.** Section 5.4 of `graph-inspection.md`, rank 4. The two
largest inline tensor families in every prefill protobuf are `Constant` node
`value` attributes rather than initializers: the O(S^2) float16 causal mask on
`/model/model/Constant_4` (32,768 bytes at S128 rising to 33,554,432 at S4096)
and the 56 float16 `[1, 8, C - S, 128]` zero cache reserves on `/Constant_1`
through `/Constant_111`. Section 5.4 point 1 gives the mechanism exactly: T20
forces initializers over 1,024 bytes into external data, but a `Constant`
attribute is not an initializer, so it bypasses that threshold and stays inside
the protobuf. `Constant` is also the largest operator population in both graph
kinds — 2,729 in prefill and 3,399 in decode (section 5.5).

**The transformation.** For every `Constant` node carrying a `value` tensor,
delete the node and append that tensor to `graph.initializer` under the name
the node produced. No consumer edge is rewritten, because the initializer takes
over the produced name. This changes *where* a tensor is stored, never its
dtype, shape, or bytes.

**Measured.**

| Kind | Variant | `Constant` before | Converted | Skipped | Nodes | Initializers | Bytes moved | Largest tensor |
|---|---|---:|---:|---|---:|---:|---:|---:|
| prefill | S128 | 2,729 | 2,728 | 1 (`output_is_graph_output`) | 7,634 → 4,906 | 310 → 3,038 | 3,728,272 | 65,536 |
| prefill | S512 | 2,729 | 2,728 | 1 | 7,634 → 4,906 | 310 → 3,038 | 7,902,096 | 524,288 |
| prefill | S1024 | 2,729 | 2,728 | 1 | 7,634 → 4,906 | 310 → 3,038 | 16,831,376 | 2,097,152 |
| prefill | S4096 | 2,729 | 2,728 | 1 | 7,634 → 4,906 | 310 → 3,038 | 48,386,960 | 33,554,432 |
| decode | all four | 3,399 | 3,399 | 0 | 10,191 → 6,792 | 310 → 3,709 | 25,984 | 256 |

Two things are worth reading in that table. The single skipped prefill
`Constant` is skipped because its output *is* a graph output, and promoting it
would have changed the public boundary — the guard exists so a
canonicalization can never quietly alter the T12 contract. And the largest
converted tensor tracks the causal mask exactly at S512 and above, which is the
whole point: pass 4 cannot externalize a tensor that is not an initializer, so
this pass is what makes the mask reachable at all.

### 4.2 `X-STATIC-SHAPE-FOLD` — evaluate what is already constant

**The T21 finding.** Section 5.1, rank 1. 804 of 922 prefill and 1,231 of 1,326
decode shape-defining operator inputs — the watched positions `Reshape[1]`,
`Slice[1,2,3,4]`, `Expand[1]`, `Tile[1]` — are produced by another node rather
than by an initializer, a graph input, or a `Constant`. Section 5.1's own
caveat is why this pass exists: the detector counts *unfolded residue*, not
proven dynamism, and T20 exported with `do_constant_folding=False` while every
boundary dimension is static, so most of those values provably never vary. The
rank-7 companion counts the raw material: `Shape=65`, `ConstantOfShape=62` in
prefill; `Shape=459`, `ConstantOfShape=91`, `Range=6` in decode.

**The transformation.** Walk the node list once in declaration order — which
the ONNX specification requires to be topological, and which the engine
*asserts* rather than assumes (`topological_order_verified: true` in every
record). Fold a node when its operator is allowlisted, it carries no subgraph
attribute, none of its outputs is a graph output, every input is either omitted
or an inline initializer of known value, and both the input and output declared
byte totals are at most 1,048,576. Evaluate with
`onnx.reference.ReferenceEvaluator` on a single-node graph; each output becomes
an initializer under the produced name.

Two guards keep this target-neutral and away from the weights. The allowlist
admits only shape, index, and small elementwise arithmetic — 33 operators, no
`MatMul`, no `Softmax`, no normalization — so no float weight and no attention
arithmetic is ever evaluated at build time. And an initializer that lives in
external data is not in the constant pool at all
(`external_initializers_excluded_from_pool: 254` in every record), so a float16
weight cannot be a fold input under *any* budget.

**Measured, prefill.**

| Variant | Folded | Nodes | Initializers created | Bytes created |
|---|---:|---:|---:|---:|
| S128 | 2,121 | 4,906 → 2,785 | 2,121 | 1,866,775 |
| S512 | 2,089 | 4,906 → 2,817 | 2,089 | 2,131,495 |
| S1024 | 2,082 | 4,906 → 2,824 | 2,082 | 38,375 |
| S4096 | 2,082 | 4,906 → 2,824 | 2,082 | 62,951 |

Decode is identical at all four capacities: 1,371 folded, 6,792 → 5,421 nodes,
1,371 initializers created, 15,239 bytes created.

**Measured, what was refused and why.** Every unfolded node is counted by
reason, which is how the residue in section 5 becomes explicable rather than
mysterious.

| Reason | S128 pf | S512 pf | S1024 pf | S4096 pf | decode (all) |
|---|---:|---:|---:|---:|---:|
| `input_not_constant` | 1,820 | 1,851 | 1,857 | 1,857 | 4,456 |
| `operator_not_allowlisted` | 653 | 653 | 653 | 653 | 708 |
| `input_in_external_data` | 255 | 255 | 255 | 255 | 255 |
| `output_is_graph_output` | 57 | 57 | 57 | 57 | 2 |
| `output_bytes_over_budget` | — | 1 | 1 | 1 | — |
| `input_bytes_over_budget` | — | — | 1 | 1 | — |

**The prefill residue is a byte-budget effect, not a capacity effect.** S128
reaches zero; S512, S1024 and S4096 stop at 5, 6 and 6. The budget caps both
consumed and produced tensors at 1 MiB, and the causal mask is 32,768 bytes at
S128 but 524,288, 2,097,152 and 33,554,432 above it — so at the larger contexts
a handful of mask-consuming nodes are refused rather than evaluated. That is
the budget doing its job: the alternative is a folder that will cheerfully
materialize a 33 MB tensor at build time and write it into the protobuf.

**One number in the execution plan does not reproduce.** The plan's
"Decisions and discoveries" records a planning probe that "folded 4,850 of
7,634 S128 prefill nodes in a single topological pass". The committed tooling
reduces S128 prefill from 7,634 to 2,785 nodes across passes 1 to 3, a net
removal of **4,849**, of which pass 1 accounts for 2,728 and pass 2 for 2,121.
The one-node difference is the `Constant` whose output is a graph output and
which this engine deliberately refuses to promote (section 4.1). The committed
number is the tool's.

### 4.3 `X-DEAD-NODE-ELIMINATION` — and the zero that is a result

**The T21 finding.** Section 5.5, rank 5: 7,634 prefill and 10,191 decode nodes
against a 2,000-node review convention, of which only 254 are `MatMul`. Pass 2
deletes the nodes it folds, but their operands stay behind as initializers that
nothing reads any more — above all the 2,728 prefill and 3,399 decode tensors
pass 1 promoted out of `Constant` nodes. Those are protobuf bytes.

**The transformation.** Compute the tensor names transitively required by the
graph outputs; keep every node producing at least one required name; drop every
initializer and `value_info` entry whose name is not required. Graph inputs are
**never** removed, whether or not any surviving node consumes them, because the
public boundary is the frozen T12 contract and its 60 decode / 3 prefill inputs
are part of it (`graph_inputs_preserved` records 60 and 3).

**Measured.**

| Kind | Variant | Nodes removed | Initializers removed | Initializer count | Bytes removed |
|---|---|---:|---:|---:|---:|
| prefill | S128 | **0** | 3,399 | 5,159 → 1,760 | 1,285,615 |
| prefill | S512 | **0** | 3,354 | 5,127 → 1,773 | 1,087,439 |
| prefill | S1024 | **0** | 3,335 | 5,120 → 1,785 | 42,775 |
| prefill | S4096 | **0** | 3,335 | 5,120 → 1,785 | 67,351 |
| decode | all four | **0** | 1,629 | 5,080 → 3,451 | 16,519 |

**The node arm removed zero nodes on all eight graphs, and that is a
measurement rather than a bug.** Two facts produce it together: pass 2 already
deletes every node it replaces, and these exports contain no unreachable
computation at all. Neither fact was known before the pass ran; the zero is the
evidence for the second one. The initializer arm is where the pass earns its
place — 3,335 to 3,399 dead initializers per prefill graph and 1,629 per decode
graph — and the two arms are recorded separately for exactly this reason. A
pass that reported only a single "removed" figure would have hidden the more
interesting half.

The bytes-removed column is worth a second look, because it is not monotone in
`S`: 1,285,615 at S128 and 42,775 at S1024. That tracks pass 2's
`bytes_created`, not the graph size — at the smaller contexts the folder
successfully evaluated the mask-consuming chains and produced large
intermediate tensors that then became dead; at S1024 and above the budget
refused those same chains, so there was less to reclaim.

### 4.4 `X-EXTERNALIZE-LARGE-TENSORS` — move the bytes out of the protobuf

**The T21 finding.** Section 5.4 again. `R-LARGE-INLINE-CONSTANT` fires once
per prefill graph at S512 and above, on the mask, whose share of the protobuf
is 5.6%, 11.5% and 67.4% at S512, S1024 and S4096. Section 5.4 also records
that the per-tensor rule is *blind* to the 56 reserves, which are 71.5%, 78.9%,
80.5% and 29.5% of the four prefill files and are the largest inline family at
three of the four variants. Both families are inline only because they were
`Constant` attributes; once pass 1 has made them initializers, the same
1,024-byte threshold T20 already applies to weights applies to them too.

**The transformation.** Serialize the candidate with ONNX external-data
conversion at a 1,024-byte threshold into one sidecar per candidate graph,
named `<graph_kind>.onnx.data` beside the candidate protobuf. The candidate
never shares, references, or modifies the reference graph's sidecar.

**Measured.**

| Kind | Variant | Newly externalized | Bytes externalized | Largest | Kept inline | Bytes kept inline |
|---|---|---:|---:|---:|---:|---:|
| prefill | S128 | 60 | 4,292,608 | 524,288 | 1,446 | 31,160 |
| prefill | S512 | 63 | 8,929,280 | 524,288 | 1,456 | 31,208 |
| prefill | S1024 | 61 | 16,809,984 | 2,097,152 | 1,470 | 31,328 |
| prefill | S4096 | 61 | 48,365,568 | 33,554,432 | 1,470 | 31,328 |
| decode | all four | **0** | **0** | 0 | 3,197 | 39,040 |

`already_external` is 254 in every record — the weight tensors T20 had already
externalized, carried through untouched.

**Decode externalized nothing, and that is why its sidecar digest did not
move.** Decode's entire inline attribute population was 26,706 bytes and its
largest single attribute is a 256-byte `[1, 64, 1]` float32 tensor
(`graph-inspection.md` 5.4). Nothing pass 1 promoted reaches 1,024 bytes, so
the threshold has nothing to catch, and the candidate sidecar came out
byte-identical to the reference's. The prefill sidecars each grew by exactly
what this pass moved into them.

### 4.5 `X-INFER-VALUE-INFO` — close a T21 evidence boundary

**The T21 finding.** Sections 6 and 7. `R-INTERNAL-DYNAMIC-SHAPE` is the one
silent rule that graph-inspection explicitly refuses to call a clean result:
`value_info` is empty in all eight reference graphs, so the detector inspected
**0 of 0** entries. Section 7 names this pass as the follow-up, in these words:
run ONNX shape inference on a machine with `onnx` installed, write the inferred
`value_info` back, and re-run the inspection, and the same rule then produces a
real answer.

**The transformation.** Run `onnx.shape_inference` with `data_prop=true`,
`strict_mode=false`, `check_type=false`, and keep the entries it produced. The
engine then counts how many intermediate tensors carry an entry and how many of
those entries are fully static, and records both. Partial coverage is reported
as partial with the exact fraction; the pass never claims coverage it did not
achieve.

**Measured.**

| Kind | Variant | `value_info` | Intermediate tensors | Annotated | Fully static | **Not fully static** |
|---|---|---:|---:|---:|---:|---:|
| prefill | S128 | 0 → 2,727 | 2,727 | 2,727 | 2,727 | **0** |
| prefill | S512 | 0 → 2,759 | 2,759 | 2,759 | 2,750 | **9** |
| prefill | S1024 | 0 → 2,766 | 2,766 | 2,766 | 2,757 | **9** |
| prefill | S4096 | 0 → 2,766 | 2,766 | 2,766 | 2,757 | **9** |
| decode | all four | 0 → 5,363 | 5,363 | 5,363 | 4,294 | **1,069** |

`coverage` reads `complete` on all eight and `intermediate_tensors_unannotated`
is 0 everywhere, so the not-fully-static counts are a real answer rather than
an artefact of partial inference. Section 6.1 is what those numbers mean.

### 4.6 `X-STAMP-CANDIDATE-PROVENANCE` — make the candidate self-identifying

**The T21 finding.** None. This pass addresses no risk rule, and its
`addresses` list is empty for that reason rather than by omission. The
motivation is section 2 of graph-inspection: the producer string in all eight
reference graphs is `pytorch 2.7.1`, so a rewritten candidate that kept it
would be indistinguishable in-band from the export it came from, and the only
thing separating the two stages would be the directory it happened to land in.

**The transformation.** Set `producer_name`/`producer_version` to this engine
and its catalogue id, and set `metadata_props` to nine keys: task id, stage,
variant id, graph kind, source relative path, source SHA-256, transform
catalogue id, transform catalogue SHA-256, and the ordered applied-pass list.
The graph body, the IR version, and the opset imports are left exactly as the
preceding passes produced them.

**Measured.** All eight: producer `pytorch 2.7.1` → `slm_lab.graph.qnn
qnn-candidate-v1`; `ir_version` 8 preserved; `opset_imports` `[["", 18]]`
preserved. `slm_lab.applied_passes` reads, in order,
`X-CONSTANT-TO-INITIALIZER,X-STATIC-SHAPE-FOLD,X-DEAD-NODE-ELIMINATION,X-EXTERNALIZE-LARGE-TENSORS,X-INFER-VALUE-INFO,X-STAMP-CANDIDATE-PROVENANCE`,
and `slm_lab.source_sha256` reads the reference digest in section 2's table. A
reader holding only the `.onnx` file can recover which reference bytes it came
from and which catalogue produced it, without the manifest.

### 4.7 One graph, end to end: S128 prefill

This subsection serves the study checkpoint in
`docs/project/learning-checkpoints.md` — *"Compare one reference ONNX graph
with its QNN candidate and explain every transformation."* Everything below is
one variant, one graph kind, read from `results/manifests/qnn/S128.json` and
`results/manifests/qnn/inspection/S128.json`.

**Before.** `S128/prefill.onnx`, SHA-256 `464892a7…`, 5,131,850 bytes on disk
plus a 1,192,085,504-byte sidecar. 7,634 nodes across 26 operator types. 3
inputs, 58 outputs. 310 initializers, 254 of them external. `value_info` empty.
3,743,112 bytes inline, of which 3,728,776 are node-attribute tensors. Findings:
`R-DATA-DEPENDENT-SHAPE-INPUT` 804, `R-WIDE-IO-BOUNDARY` 58,
`R-SCATTER-GATHER-INDEXING` 1, `R-GRAPH-NODE-COUNT` 7,634,
`R-FLOAT-SENSITIVE-ELEMENTWISE` 367, `R-FLOAT-PRECISION-CAST` 285,
`R-SHAPE-COMPUTATION-CHAIN` 127.

**Pass 1.** 2,728 of 2,729 `Constant` nodes become initializers, moving
3,728,272 bytes out of node attributes and into the initializer pool. Node
count 7,634 → 4,906; initializers 310 → 3,038. One `Constant` is refused
because its output is a graph output. Nothing else changes: no edge is
rewritten, because each new initializer takes the name its node produced.

**Pass 2.** With 2,728 more values visible as constants, the folder evaluates
2,121 nodes in a single topological sweep. By operator: `Unsqueeze` 1,507,
`Concat` 228, `Shape` 65, `ConstantOfShape` 62, `Equal` 62, `Mul` 62, `Where`
62, `Reshape` 58, `Expand` 6, `Slice` 4, `Add` 3, `Cast` 2. Node count
4,906 → 2,785; 2,121 new initializers holding 1,866,775 bytes. 1,820 nodes were
refused because an input was not constant, 653 because the operator is not
allowlisted, 255 because an input lives in external data, 57 because an output
is a graph output. The `Shape` population goes 65 → 0 and `ConstantOfShape`
62 → 0, which is why `R-SHAPE-COMPUTATION-CHAIN` reaches 0.

**Pass 3.** The node arm finds nothing — 0 removed, because pass 2 already
deleted what it replaced and the graph has no unreachable computation. The
initializer arm removes 3,399 dead initializers holding 1,285,615 bytes,
bringing the pool 5,159 → 1,760. All 3 graph inputs are preserved.

**Pass 4.** 60 initializers at or above 1,024 bytes are written into
`prefill.onnx.data`, moving 4,292,608 bytes; the largest is 524,288 bytes.
1,446 initializers stay inline at 31,160 bytes. The 254 already-external weight
tensors are carried through. Sidecar 1,192,085,504 → 1,196,378,112 bytes.

**Pass 5.** Shape inference annotates all 2,727 intermediate tensors, and every
one of them is fully static. `R-INTERNAL-DYNAMIC-SHAPE` still does not fire on
this graph — but for the first time that is because the interior was inspected
and found static, not because there was nothing to inspect.

**Pass 6.** Producer becomes `slm_lab.graph.qnn qnn-candidate-v1`;
`metadata_props` records `slm_lab.source_sha256 = 464892a7…` and the six-pass
list. IR version 8 and opset `("", 18)` are untouched.

**After.** `S128/prefill.onnx`, SHA-256 `463313d4…`, 949,559 bytes plus a
1,196,378,112-byte sidecar. 2,785 nodes across 24 operator types. 3 inputs, 58
outputs — identical names, dtypes and shapes. 1,760 initializers, 314 external.
2,727 `value_info` entries. 31,168 bytes inline. `onnx.checker` passed. All 56
cache writes still `Concat`. Findings: `R-WIDE-IO-BOUNDARY` 58,
`R-SCATTER-GATHER-INDEXING` 1, `R-GRAPH-NODE-COUNT` 2,785,
`R-FLOAT-SENSITIVE-ELEMENTWISE` 367, `R-FLOAT-PRECISION-CAST` 285. The two
shape findings are gone entirely.

**What did not move, and why that matters more than what did.**
`R-WIDE-IO-BOUNDARY` is still 58 and `R-SCATTER-GATHER-INDEXING` still 1,
because both are properties of the T12 contract, and no pass here is allowed to
touch it. `R-FLOAT-PRECISION-CAST` is still 285 and
`R-FLOAT-SENSITIVE-ELEMENTWISE` still 367, because no pass touches float
arithmetic — the 28 `Softmax` nodes, the 113 `Pow`/`ReduceMean`/`Sqrt` RMSNorm
triples, and all 285 float16/float32 crossings are exactly where the export put
them. That invariance is not a limitation of the catalogue. It is the reason
section 8's parity result came out the way it did, and it is what separates
this catalogue from the rejected pass in section 7, which moves all four of
those numbers.

## 5. Before and after, across the matrix

Prefill, per variant. A dash means the rule did not fire on that side.

| Variant | Nodes | Op types | `R-DATA-DEPENDENT-SHAPE-INPUT` | `R-SHAPE-COMPUTATION-CHAIN` | `R-LARGE-INLINE-CONSTANT` | `R-INTERNAL-DYNAMIC-SHAPE` | `value_info` | `.onnx` bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S128 | 7,634 → 2,785 | 26 → 24 | 804 → — | 127 → — | — → — | — → — | 0 → 2,727 | 5,131,850 → 949,559 |
| S512 | 7,634 → 2,817 | 26 → 26 | 804 → 5 | 127 → 9 | 1 → — | — → 9 | 0 → 2,759 | 9,305,674 → 956,509 |
| S1024 | 7,634 → 2,824 | 26 → 26 | 804 → 6 | 127 → 11 | 1 → — | — → 9 | 0 → 2,766 | 18,235,014 → 958,649 |
| S4096 | 7,634 → 2,824 | 26 → 26 | 804 → 6 | 127 → 11 | 1 → — | — → 9 | 0 → 2,766 | 49,790,614 → 958,654 |

Decode, identical at all four capacities except for two protobuf bytes:

| Variant | Nodes | Op types | `R-DATA-DEPENDENT-SHAPE-INPUT` | `R-SHAPE-COMPUTATION-CHAIN` | `R-INTERNAL-DYNAMIC-SHAPE` | `value_info` | `.onnx` bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| S128, S512 | 10,191 → 5,421 | 29 → 28 | 1,231 → 423 | 556 → 496 | — → 1,069 | 0 → 5,363 | 1,759,947 → 1,752,536 |
| S1024, S4096 | 10,191 → 5,421 | 29 → 28 | 1,231 → 423 | 556 → 496 | — → 1,069 | 0 → 5,363 | 1,759,947 → 1,752,538 |

Storage and boundary, both kinds:

| Kind | Variant | Initializers (external) | Inline bytes | Sidecar bytes | Inputs / outputs |
|---|---|---|---:|---:|---|
| prefill | S128 | 310 (254) → 1,760 (314) | 3,743,112 → 31,168 | 1,192,085,504 → 1,196,378,112 | 3 / 58 unchanged |
| prefill | S512 | 310 (254) → 1,773 (317) | 7,916,936 → 31,248 | 1,192,085,504 → 1,201,014,784 | 3 / 58 unchanged |
| prefill | S1024 | 310 (254) → 1,785 (315) | 16,846,216 → 31,368 | 1,192,085,504 → 1,208,895,488 | 3 / 58 unchanged |
| prefill | S4096 | 310 (254) → 1,785 (315) | 48,401,800 → 31,368 | 1,192,085,504 → 1,240,451,072 | 3 / 58 unchanged |
| decode | all four | 310 (254) → 3,451 (254) | 41,042 → 39,522 | 1,192,085,504 unchanged | 60 / 58 unchanged |

Four readings of those tables.

**The prefill protobuf stops scaling with `S`.** It ran 5,131,850 to 49,790,614
bytes across the matrix and now runs 949,559 to 958,654 — a **9.70x spread
collapsed to 1.01x**. This is not compression. Pass 1 turned the mask and the
56 reserves from `Constant` attributes into initializers and pass 4 moved
everything at or above 1,024 bytes into the candidate's own sidecar. Inline
bytes fell from 48,401,800 to 31,368 at S4096. The bytes did not vanish; they
moved, and the prefill sidecar grew correspondingly.

**`R-LARGE-INLINE-CONSTANT` is cleared at every variant where it fired.** It
fires once per prefill graph at S512 and above and does not fire on any of the
four candidates.

**Contract preservation is measured, not assumed.** `contract_preservation` in
all four manifests records `boundary_identical_to_reference: true`,
`cache_writes_preserved: 56`, and `cache_write_operator` `Concat` for prefill
and `ScatterElements` for decode. `onnx.checker.check_model` passed on all
eight. Both are hard post-conditions inside the build, so a candidate that
broke either would not have been written at all.

**Every unchanged count is unchanged in the same place.** `R-WIDE-IO-BOUNDARY`
58 / 118, `R-SCATTER-GATHER-INDEXING` 1 / 57, `R-FLOAT-PRECISION-CAST` 285, and
`R-FLOAT-SENSITIVE-ELEMENTWISE` 367 hold on both sides of all eight graphs, and
`highest_severity` stays `high` throughout. The rewrites removed bookkeeping;
they did not remove any of the four findings that describe the contract or the
arithmetic.

## 6. Three things the build surfaced that were not the expected answer

### 6.1 `R-INTERNAL-DYNAMIC-SHAPE` went from silent to loud

`graph-inspection.md` section 6 is careful to say that this rule's silence is
**not** a clean result: `value_info` was empty in all eight reference graphs,
so the detector inspected 0 of 0 entries, and section 7 records that "the
static public boundary makes a static interior likely, but likely is not
measured."

Running shape inference annotated every intermediate tensor in every candidate
— 2,727 to 2,766 in prefill, 5,363 in decode — and the rule now has real input.
The answer is **not** the clean one that "likely" pointed at.

| Kind | Variant | Non-static interior tensors | Of |
|---|---|---:|---:|
| prefill | S128 | 0 | 2,727 |
| prefill | S512, S1024, S4096 | 9 | 2,759 / 2,766 / 2,766 |
| decode | all four | **1,069** | 5,363 |

The sampled locations name the shapes directly. In S512 prefill the nine are
paired `Expand`/`Unsqueeze` outputs whose dimensions inference could not
resolve, for example `value_info: /model/model/Expand_2_output_0 int64
[unk__0, unk__1, unk__2, unk__3]` and
`value_info: /model/model/Unsqueeze_18_output_0 int64 [unk__0, unk__1, unk__2,
unk__3, 1]`. In decode the population is dominated by the attention prologue's
`Slice` outputs — `value_info: /Slice_output_0 float16 [unk__0, unk__1,
unk__2, unk__3]` and its 55 siblings begin the list.

Three things must be said about those 1,069 together, and the order matters.

1. **They are not new defects.** No pass introduced a dynamic shape. They are
   the *first measurement* of something the reference graphs never declared.
   The reference count of 0 was never a zero; it was an absence.
2. **They are a statement about shape inference, not about the graph.**
   `unk__0` means opset-18 shape inference could not prove a dimension from the
   declarations available to it, at `strict_mode=false` with `data_prop=true`.
   A dimension that inference cannot resolve may still be provably constant by
   a stronger analysis. The count measures what this inference pass could
   establish.
3. **The evidence boundary named in section 7 of graph-inspection is closed,
   and it closed unfavourably.** That is worth stating plainly rather than
   filing under progress. A follow-up that was expected to confirm a static
   interior instead found 1,069 interior tensors it could not confirm.

### 6.2 Decode's residue is a run-time mask, not a folding failure

Prefill's rank-1 finding falls from 804 to 0, 5, 6, 6. Decode's falls from
1,231 to **423** at every capacity, and 429 of its 459 `Shape` nodes survive.
That two-thirds reduction is real, and the remaining third has a structural
explanation that must not be hidden behind the headline.

`X-STATIC-SHAPE-FOLD` folds only nodes whose every input is a *known constant*.
Decode computes its attention mask at run time from `Range`/`Greater` position
arithmetic over activation tensors — `graph-inspection.md` 5.4 traces the chain
by name, `/model/model/Range_1` producing the cache-position ramp and
`/model/model/Greater` comparing it against the reshaped current position — so
decode's `Shape` nodes read tensors that are **not** constants even though
their *shapes* are static. The candidate's own residue confirms it: the
surviving 496 `R-SHAPE-COMPUTATION-CHAIN` nodes are `Shape=429`,
`ConstantOfShape=61`, `Range=6`, and the sampled locations begin
`node[174] ai.onnx.Shape /model/model/Shape`,
`node[179] ai.onnx.Range /model/model/Range`.

Folding those would need the inferred `value_info` — which
`X-INFER-VALUE-INFO`, pass 5, produces only *afterwards*. **A shape-aware
second fold is the obvious follow-up.** It is not in this catalogue, it has not
been implemented, and no number in this report assumes it. Its likely effect is
also bounded by section 6.1: the 1,069 tensors inference could not resolve are
exactly the ones such a fold would have least to say about.

Note the trade this exposes. Decode carries no materialized mask at any
capacity — which is why its protobuf is the same size at S128 and S4096, and
why it needed no externalization at all — and it pays for that in shape
arithmetic that survives folding. Prefill made the opposite choice and pays in
inline bytes. The catalogue fixed prefill's cost almost completely and decode's
only partially, because the two costs are not the same kind of thing.

### 6.3 Dead-node elimination removed zero nodes

Covered in full at section 4.3, and repeated here because it is the finding
most likely to be misread as a bug. On all eight graphs, the node arm of pass 3
removed nothing. The cause is that pass 2 already deletes every node it
replaces and these exports contain no unreachable computation. The initializer
arm did the work — 3,335 to 3,399 dead initializers per prefill graph, 1,629
per decode graph — which is what stops the operands of the folded chains from
staying in the protobuf. The pass reports its two arms separately so that this
distinction is legible in the manifest rather than reconstructible from it.

## 7. The rejected transformation, measured

`X-ORT-CPU-OFFLINE-OPTIMIZATION` is in the catalogue with `applied: false`.
Writing an ONNX Runtime session's `optimized_model_filepath` at
`ORT_ENABLE_BASIC` is the obvious, one-line way to fold these graphs, and it
does fold them hard. It is rejected on **measured** grounds, and the build tool
re-measures it on every reference graph it reads rather than citing a planning
probe. A probe measurement is not evidence until the committed tooling
reproduces it.

The measurement runs on the CPU execution provider at `ORT_ENABLE_BASIC` with
onnxruntime 1.28.0, writes into a scratch directory under the artifact root,
and deletes the output (`scratch_output_deleted: true`). Wall-clock duration is
deliberately not recorded, so `--check` stays a genuine drift check; the tool
prints it on stderr instead.

**It works, on the metric it is chosen for.** Node count falls hard: prefill
7,634 → **3,040** and decode 10,191 → **4,245**, identically at all four
capacities. Set against this catalogue's 2,785–2,824 and 5,421, the committed
passes win on prefill by roughly 220 nodes and **lose on decode by 1,176** —
about 22% of the decode graph. Rejecting this pass therefore costs something
real, which is why the grounds have to be measured rather than asserted.

Four grounds, in increasing order of severity.

### 7.1 It decomposes all 28 `Softmax` nodes

`Softmax` goes **28 → 0** in both graph kinds, and `Sub`, `Exp`, `ReduceMax`
and `ReduceSum` each go **0 → 28**. One operator that an NPU implements
natively is replaced by five — on exactly the operator `graph-inspection.md`
5.6 ranks as precision-sensitive, transcendental, and usually implemented on
fixed-point accelerators by piecewise approximation or lookup.

`Div` is the fifth arm and it does not read cleanly, so it is reported as
observed rather than tidied: prefill `Div` goes 113 → **141**, exactly the +28
the decomposition predicts, while decode `Div` *falls*, 167 → **141**. In
decode the decomposition's `Div` arm is netted out by other rewrites in the
same run. The other four arms are unambiguous in both kinds; this one is not,
and both graph kinds land on the same 141.

This is the ground that matters most for a quantized pipeline. A candidate
whose `Softmax` has been pre-decomposed into a five-node reduction denies the
converter the chance to pattern-match a `Softmax` and hand it to a native
implementation — and it hands a fixed-point quantizer five separate tensors to
choose encodings for where there was one operator.

### 7.2 It inserts host-specific float16 casts

`Cast` rises **348 → 830** in prefill and **464 → 860** in decode. That is the
CPU execution provider's float16 fallback inserting casts around kernels it
lacks. It is a **host-specific artifact of the machine that ran the
optimizer**, with no business in a candidate aimed at a different target, and
it moves the 285 real float16/float32 crossings that section 5.6 counts and
that T21's numerical work is calibrated against.

This is the ground that most clearly separates an *optimizer* from a
*transformation catalogue*. ONNX Runtime's CPU optimizer is doing its job
correctly: it is producing a graph that runs well on this host's CPU. That is
precisely the wrong objective. The committed catalogue's six passes are
target-neutral by construction and produce no host-specific operator choice at
all — which is exactly why they leave `Cast` at 346 and 462, two below the
reference in each kind, rather than more than doubling it.

### 7.3 It stamps eight opset imports that no node uses

The optimized graphs declare nine opset imports where the reference declares
one. The eight added are `ai.onnx.ml`, `ai.onnx.preview`,
`ai.onnx.preview.training`, `ai.onnx.training`, `com.microsoft`,
`com.microsoft.experimental`, `com.microsoft.nchwc`, and `org.pytorch.aten`.

The build tool checks each one against the node list, and records the result in
its own field: `added_opset_domains_used_by_no_node` contains **all eight**.
Not one node in either optimized graph declares a domain outside `""`.

That is a pure regression in portability for no functional gain.
`graph-inspection.md` section 6 records `R-NON-DEFAULT-DOMAIN` staying silent
as one of the three clean **blocking**-severity results — the evidence that the
artifact is portable across the three target platforms rather than tied to one
runtime's extension library. A converter or a validator that reads the opset
imports and refuses `com.microsoft` would reject this graph over domains no
node uses.

### 7.4 It destroys the external-data layout, and at S4096 that is nearly fatal

This is the ground that turns a design preference into an engineering
constraint.

| Measure | Reference | After the pass |
|---|---:|---:|
| External initializers, prefill and decode | 254 | 58 |
| External data files written | 1 sidecar | **none** |
| External data bytes | 1,192,085,504 | **0** |
| Inline initializer bytes | 14,336 | 1,191,990,214 – 1,810,643,510 |

The optimizer inlines the weights. `external_data_files` is empty in all eight
records: no sidecar is written next to the optimized model, and 1.19 GB of
float16 weights that lived in a companion file are serialized back into the
protobuf itself.

| Variant | Kind | Reference `.onnx` bytes | Optimized `.onnx` bytes |
|---|---|---:|---:|
| S128 | prefill | 5,131,850 | 1,197,039,898 |
| S512 | prefill | 9,305,674 | 1,209,557,276 |
| S1024 | prefill | 18,235,014 | 1,245,208,920 |
| **S4096** | **prefill** | **49,790,614** | **1,811,439,962** |
| all four | decode | 1,759,947 | 1,193,053,329 |

**The S4096 prefill output is a single 1,811,439,962-byte protobuf — 1.69 GiB,
against protobuf's 2 GiB serialization ceiling, at 84% of it.** A candidate
built this way is one context variant away from being unserializable, and the
margin is consumed by a mask that this catalogue's passes move *out* of the
protobuf entirely.

Compare the same column for the committed catalogue: 949,559 to 958,654 bytes.
The two approaches move the same tensors in opposite directions.

### 7.5 What disagreed with the planning probe

One number does not reproduce, and it is recorded rather than smoothed over.
The T22 planning probe recorded the S128 prefill optimized output as an
828,930-byte protobuf plus a 1,507,512,320-byte sidecar. This tooling measures
**1,197,039,898 bytes inline and no sidecar** for the same graph, with
onnxruntime 1.28.0 on the CPU execution provider.

**The committed numbers are the tool's, not the probe's.** The two runs must
have differed in their session configuration — this one sets only
`graph_optimization_level` and `optimized_model_filepath`, and no external-data
session option — but this repository holds no record of the probe's
configuration, so the cause is **not stated as known**. Everything else the
probe reported is reproduced here exactly: the node counts, the `Softmax`
decomposition, `Cast` 348 → 830, and all eight added domains.

The lesson is the one the catalogue's own comment makes: a rejection recorded
as prose is an assertion, and a rejection re-measured by the committed tooling
on every build is evidence. Three of the four grounds above were in the plan;
the fourth — the protobuf ceiling — only appeared when the tool measured what
the plan had asserted.

## 8. ONNX Runtime CPU parity of the candidate, measured

`results/manifests/qnn/parity/S<context>-ort-cpu.json` holds one
`ParityEvidence` record per variant, written by the **T21 runner**,
`slm_lab.backends.onnx_cpu`, pointed at the T22 manifest instead of the T20
one.

That reuse is the point. Same protocol, same frozen T10 workload, same
bfloat16 PyTorch reference at revision `c1899de289a04d12100db370d81485cdf75e47ca`,
same teacher-forced token IDs, same four decode steps, same `ORT_DISABLE_ALL`
sessions on the CPU execution provider with both thread counts at 1 and
`ORT_SEQUENTIAL`, and the same `DEFAULT_ORT_CPU_TOLERANCE` the reference
records in `results/graph/parity/` were measured against. A second parity
implementation would not have produced comparable evidence; it would have
produced a second set of numbers with no defensible relationship to the first.

The one change the reuse required is that the runner resolves its graph
directory from the manifest's `artifacts.root` rather than a hard-coded
`onnx/reference/T20`. For every committed T20 manifest that expression already
resolves to the same path, so no committed T21 or T23 evidence moved.

Measured on 2026-08-03 with onnxruntime 1.28.0, CPython 3.11.13,
`macOS-15.7.7-arm64`, `CPUExecutionProvider` alone. **Every one of the eight
candidate graphs loaded**, every record carries
`evidence_tier="real_onnxruntime_cpu"` — a tier derived from the session
objects, which a caller cannot assert — and all four end `passed: true` with
`failures: []` and `failure_kinds: []`.

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
`protected_relative_max` 1.05, `rtol` 0.02, `top5_overlap_min` 0.8,
`require_top1` true, and **exact equality** on every cache region the T12
contract calls untouched. The static-cache invariants were checked on 56
tensors per step, 280 tensor checks per variant, **1,120 across the matrix**,
with zero violations and zero `written_slot_immutable` violations.

### 8.1 The candidate is bit-identical to the reference

This is the result worth reading twice, and it is stronger than "within
tolerance".

Every one of the 20 recorded steps produced a `candidate_logits_sha256`
**equal to the reference record's for the same step**, and every step's
`cache_report` entry is equal too. The metric columns above are therefore not
merely close to the ones in `docs/results/onnx/ort-cpu-parity.md` — they are
the same floats, including S4096 step 4's `top5_overlap` of exactly 0.80, the
single step in the whole reference set whose top-5 was not 1.00.

The two runs differ only in which bytes ONNX Runtime loaded, and the records
prove it: `graph_digests` in each candidate record are the candidate digests
from section 2, not the T20 digests, and each record's `evidence_sha256`
differs from its reference counterpart accordingly.

**Why that is the expected outcome, and why saying so is not a way of
discounting it.** None of the six applied passes touches float arithmetic or
its order:

- `X-CONSTANT-TO-INITIALIZER` moves a tensor between two encodings of the same
  values;
- `X-STATIC-SHAPE-FOLD` evaluates integer shape arithmetic whose inputs are
  already constants, and its allowlist excludes every float-reduction operator;
- `X-DEAD-NODE-ELIMINATION` removes computation no output reads;
- `X-EXTERNALIZE-LARGE-TENSORS` relocates bytes between a protobuf and a
  sidecar;
- `X-INFER-VALUE-INFO` adds annotations;
- `X-STAMP-CANDIDATE-PROVENANCE` writes metadata.

At `ORT_DISABLE_ALL` the runtime adds no fusion of its own on either side, so
both runs execute the same float operations in the same order over the same
weights. Bit-identity is what a semantics-preserving catalogue *should*
produce. Had any number moved, the honest reading would have been that one of
those six passes is not semantics-preserving — which is exactly what this
measurement was for. The prediction was made before the measurement and the
measurement could have falsified it; that it did not is the result.

It is also the sharpest possible statement of why section 7's pass was never a
candidate for this catalogue. A pass that decomposes all 28 `Softmax` nodes and
inserts 482 casts cannot produce bit-identical logits, and there would then be
no way to tell a legitimate rewrite from a defective one by comparison alone.

### 8.2 What the parity measurement does not license

It is 20 steps of one frozen workload per variant, on **one host**, on the
**CPU execution provider**, at **one ONNX Runtime version** (1.28.0), at **one
optimization level** (`ORT_DISABLE_ALL`). It says nothing about whether the
candidate compiles, whether any operator is supported by a vendor toolchain,
how it behaves on an accelerator, or what it costs.

Three specific limits deserve naming.

- **It says nothing about a converter.** Bit-identity on the ONNX Runtime CPU
  provider is evidence that the *graph* still computes the same function. It is
  not evidence about how a QNN converter will lower it, whether the folded
  initializers will be quantized differently from the `Constant` attributes
  they replaced, or whether 1,069 unresolved interior shapes (section 6.1) will
  survive a compiler's own shape analysis.
- **It inherits every boundary of the T21 protocol**, including the one that
  matters most here: the newly written cache slot is checked for having been
  written and for being finite, **not** for holding the right values. Verifying
  its contents needs a reference cache comparison, which is not implemented.
- **The fusion delta is still unmeasured.** No `ORT_ENABLE_ALL` run exists on
  either the reference or the candidate, so nothing here says what graph
  optimization would do to these numbers — including whether it reintroduces
  the class of problem the T23 failure analysis documented.

`claim_boundary` in each manifest is adjusted to match, and it is derived
rather than asserted: a passing, on-these-bytes measurement replaces
`onnxruntime_numerical_parity_of_the_candidate` in `does_not_establish` with
the narrower
`candidate_parity_beyond_the_recorded_steps_of_one_frozen_workload_on_the_cpu_execution_provider`,
and adds
`candidate_logit_and_static_cache_parity_held_on_every_recorded_step_under_the_T21_protocol_and_tolerance`
to `establishes`. A failing measurement, a `stale_record`, or a missing record
would each have left the original, broader entry in place.

One field is deliberately not what it looks like. `record_task_id` reads `T21`
in every candidate record, because the runner stamps its own task id into a
fixed schema field. It is **not** evidence that T21 produced the file; the
`graph_digests` are what identify what was measured.

## 9. Packaging: what "ready for Workbench submission" means

The T22 task definition's third acceptance criterion is "Packages are ready for
Workbench submission". That phrase can be read two ways and only one of them is
supported, so the package record says which in-band, in its
`submission_status.caveat` field:

> The package layout for an external-data ONNX model has not been verified
> against the Qualcomm AI Hub service. No compile job was submitted and no
> service call was made. T31 owns the first real submission. Ready for
> submission means exactly three things: the candidate and sidecar digests were
> re-verified against the committed T22 manifest, a compile request was
> generated, and that request was accepted by the committed T30 adapter's own
> validation. It does not mean AI Hub accepted it.

`job_submitted`, `service_contacted`, and
`package_layout_verified_against_service` are all `false` in all four records,
and `first_submission_owner` reads `T31`.

**What exists.** For each of the four variants and both graph kinds, a package
directory under `${SLM_LAB_ARTIFACT_ROOT}/onnx/qnn-package/T22/` containing the
`.onnx`, its `.onnx.data` sidecar, and a `SHA256SUMS` file in `sha256sum`
format, plus a path-free committed record at
`results/manifests/qnn/packages/S<context>.json`. Members are hardlinked where
possible; the record states the link mode and the digest evidence per file
(`same_inode_as_verified_source` for a hardlink, a post-placement re-hash for a
copy).

One package directory per graph kind is deliberate: an ONNX external-data
`location` resolves relative to the directory holding the `.onnx`, so the
sidecar must keep the exact name the graph references and must sit beside it.

**The compile request.** Derived from the target selector
`configs/targets/qualcomm-snapdragon-x-elite-crd.json` (SHA-256
`20b284060beb6de64bac3cf903858f87a6e5683e0669af539d177d4c92327846`), whose
device and runtime identity are copied from the committed T02 lifecycle
evidence rather than from vendor documentation:

| Field | Value |
|---|---|
| Client | `qai-hub` 0.53.0 |
| Device | `Snapdragon X Elite CRD`, Windows 11, no selector attributes |
| Runtime | QAIRT 2.45.0.260326154327 |
| Compile options | `--target_runtime qnn_context_binary --qairt_version 2.45.0.260326154327` |
| Timeout / retry | 3600 s / no retry |

Input specs are derived from the manifest's recorded boundary tensors: 3 for
prefill, 60 for decode, of which 56 are the `[1, 8, C, 128]` float16 cache
tensors. The request id is deterministic — `t30-compile-83b8813c19a37ac036ad`
for S128 prefill, and a distinct id per graph — and each request was validated
by `ai_hub.preflight_compile_request`, which runs the committed T30 compile
validation chain (schema version, stage, public-safety projection, field set,
client version, device selector, runtime, option allowlist, timeout, retry,
source-artifact existence and digest, input specs, output-path policy) and
returns the same `request_id` `run_compile` would record. It constructs no
backend, imports no client, and contacts no service.

**One known unverified assumption, recorded rather than assumed away.** The
compile request names only the `.onnx` file, because the committed T30 adapter
requires `source_artifact.path` to be a single existing file. Whether the
service reads the `.onnx.data` sidecar from the same directory, or requires a
directory or an archive instead, is **unverified**. The record carries that
sentence in `compile_request.single_file_source_caveat` on every graph.

The generated request files carry machine-local paths and stay under
`.ai-local/`; the committed record is deliberately path-free.

## 10. Evidence boundaries

This report does not establish, and must not be cited as establishing:

- **That any candidate compiles.** No QNN converter, no QAIRT tool, no
  quantizer, and no vendor toolchain of any kind was run, installed, or
  queried.
- **That any operator is or is not supported by any QNN converter version.**
  Every severity in the risk catalogue remains a reviewed structural judgement
  about a stated target context, and the catalogue says so itself.
- **That a reduced finding count makes a graph convertible.** A smaller
  population of a structural pattern is a smaller population of a structural
  pattern. Every manifest's `claim_boundary.does_not_establish` names this
  explicitly.
- **Anything about accelerator placement or partitioning.** Nothing here says
  which operators would run on a Hexagon HTP, which would fall back to a host,
  or where a partition boundary would land.
- **Anything about latency, throughput, memory residency, or power.** No
  performance number of any kind was measured, and the single-threaded,
  optimization-disabled parity configuration would be a poor one for measuring
  any.
- **That Qualcomm AI Hub accepted, or would accept, any request or package.**
  No job was submitted and no service call was made. The package layout for an
  external-data ONNX model is unverified against the service.
- **That the candidate's numerical behaviour generalizes.** The parity result
  is one frozen workload, four variants, 20 steps, one host, one provider, one
  runtime build, one optimization level — see 8.2.
- **That the candidate's interior is statically shaped.** Section 6.1 measures
  the opposite for decode, and the 1,069 figure is a statement about what
  opset-18 shape inference could resolve, not a proof that those dimensions
  vary.
- **That `onnx.checker` acceptance implies validity under a stricter check.**
  `full_check` is `false` on all eight.
- **Anything about ONNX Runtime versions other than 1.28.0**, or about the
  rejected pass's behaviour under any other version or execution provider. The
  numbers in section 7 are what onnxruntime 1.28.0's CPU optimizer did on this
  host.

Each of the four manifests carries this boundary in-band as a `claim_boundary`
object with `establishes` and `does_not_establish` lists, in the same shape T20
and T21 use. A reader who sees only the JSON — and so sees only a set of
falling finding counts — learns from the file itself that no compiler was run.

## 11. Reproduction

Requires the T20/T23 reference graphs under the external artifact root
(`/Volumes/T9/slm-deployment-lab` on the primary machine), and about 9.0 GB of
free space for the candidates. Neither tree is committed; regenerate the
reference tree with the `export` commands recorded in
`results/manifests/onnx/*.json` if the artifact root is empty.

Build the candidates and their manifests, then verify that the committed files
are exactly what the tooling reproduces:

```bash
SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab PYTHONPATH=src \
  <onnx-env-python> -m slm_lab.graph.qnn.build --all-manifests

SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab PYTHONPATH=src \
  <onnx-env-python> -m slm_lab.graph.qnn.build --all-manifests --check
```

Both need `onnx`, `onnxruntime`, and `numpy`. The locked root environment has
none of them — deliberately, for the T21 inspection path — which is why the
`onnx`-dependent cases in `tests/qnn/` skip there. Use the parity environment
described in `environments/onnx-cpu/README.md`.

Measure candidate parity with the T21 runner, one invocation per variant:

```bash
SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab HF_HOME=<local-hf-cache> \
TRANSFORMERS_OFFLINE=1 PYTHONPATH=src <parity-env-python> \
  -m slm_lab.backends.onnx_cpu \
  --manifest results/manifests/qnn/S128.json --steps 4 --reference torch \
  --output results/manifests/qnn/parity/S128-ort-cpu.json
```

Build and re-verify a package and its compile request, with no service call:

```bash
SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab PYTHONPATH=src python3 \
  scripts/qualcomm/package_qnn_candidate.py \
  --manifest results/manifests/qnn/S128.json

SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab PYTHONPATH=src python3 \
  scripts/qualcomm/package_qnn_candidate.py \
  --manifest results/manifests/qnn/S128.json --check
```

Run the unit tests, which need no artifact root:

```bash
PYTHONPATH=src python -m pytest tests/qnn tests/deployment/qualcomm -q
```

Useful variations on the build tool: `--manifest results/manifests/onnx/S4096.json`
for one variant, `--catalogue <path>` and `--rules <path>` to score against
different catalogues, `--location-sample-limit N` to widen the sampled
locations per finding (default 8), `--artifact-root` to override
`SLM_LAB_ARTIFACT_ROOT`, and `--output-directory` / `--inspection-directory` /
`--parity-directory` to redirect the three committed outputs.

### What `--check` proves, and what it does not

`--check` re-derives everything the build derives: it re-hashes the reference
graphs against the T20 manifests, rebuilds both candidates into the artifact
root, re-runs `onnx.checker`, re-inspects both graphs with the T21 rule engine,
re-runs the ONNX Runtime probe of the rejected pass, re-reads the parity record
and re-derives the `verification.ort_cpu_parity` block from it, and compares
the rendered JSON against the committed files. It exits non-zero with
`stale report:` or `missing report:` on stderr if anything would change. On
2026-08-03 it exited 0 for all four variants.

Neither the manifests nor the inspection reports contain a timestamp, so this
is a genuine drift check rather than a re-render.

**What that covers.** Every number in sections 4, 5, 6 and 7 of this report,
and every digest in section 2. Those are all fields of
`results/manifests/qnn/S*.json` or `results/manifests/qnn/inspection/S*.json`:
the per-pass `effect` records, the `structural_delta` counts and byte totals,
the per-rule finding counts and their sampled locations, the
`contract_preservation` post-conditions, the `onnx_checker` verdicts, and the
whole of `rejection_evidence`. It also covers the candidate SHA-256 values,
which is the strongest single statement here: the build is deterministic down
to the candidate bytes.

**What it does not cover.** Four things, each named rather than implied.

| Number or claim | Where | Why `--check` misses it |
|---|---|---|
| The parity metrics and the bit-identity result | §8 | `--check` *reads* `parity/S*-ort-cpu.json` and re-derives the manifest block from it; it never re-measures. Regenerating that evidence means re-running the T21 runner in the parity environment. |
| The `du -sh` figure of 9.0 GB for the candidate tree | §2 | Filesystem state, not a report field. Recorded in `results/manifests/qnn/README.md`. |
| The package contents, digests, and request validation | §9 | Written and re-verified by `package_qnn_candidate.py --check`, a separate tool with its own record. The build tool neither writes nor reads `packages/`. |
| The reference-side T21 numbers cited for context (804 of 922, `Shape=459`, the 262,144-byte threshold reasoning) | §4, §6 | Fields of `results/graph/S*.json`, covered by `python -m slm_lab.graph.inspection --all-manifests --check`. The `before` side of every table here is re-derived independently by this tool and agrees with them. |

And what it proves about the world, as opposed to about the files: nothing. It
does **not** prove that any candidate compiles, that any operator is supported
by any vendor toolchain, or that any accelerator will place it. Nor does it, by
itself, prove the candidate runs — this tool rewrites and measures graphs, it
never executes one. What proves that is the separate parity record in section
8.

## 12. Learner checkpoint

`docs/project/learning-checkpoints.md` marks T22 as a deep-study checkpoint
whose task is to *"compare one reference ONNX graph with its QNN candidate and
explain every transformation"*. Section 4.7 walks one graph end to end for
that. The questions below are answerable from this document plus the committed
files under `results/manifests/qnn/`; none can be answered by restating a pass
title.

- [ ] Pass 1 moves tensors between two encodings and changes no value. Explain
  why it must nevertheless run **first**, and what would have happened to the
  33,554,432-byte S4096 causal mask if passes 2 and 4 had run without it.
- [ ] `X-STATIC-SHAPE-FOLD` reaches 0 residual shape-defining inputs at S128
  and stops at 5, 6 and 6 above it. Explain that pattern from the byte budget
  and the mask size, and say why the correct response is *not* to raise
  `max_input_bytes` to 64 MiB.
- [ ] `X-DEAD-NODE-ELIMINATION` removed **zero nodes** on all eight graphs.
  Argue first that this proves the pass is useless, then refute yourself from
  the committed `effect` record, and say what the zero is actually evidence
  about.
- [ ] `R-INTERNAL-DYNAMIC-SHAPE` reported nothing on the reference and 1,069 on
  the decode candidate. Explain why the first number was never a zero, why the
  second is a statement about shape inference rather than about the graph, and
  what a compiler would do with either.
- [ ] Prefill's rank-1 finding falls to at most 6 and decode's only to 423.
  Explain the structural reason using the operators decode uses to build its
  mask at run time, and describe the follow-up pass that would close the gap —
  including why it cannot be pass 2.
- [ ] The four decode candidate sidecars are byte-identical to the reference
  sidecars, and the four prefill sidecars are not. Derive both facts from the
  measured `X-EXTERNALIZE-LARGE-TENSORS` record without looking at the graphs.
- [ ] The rejected ORT pass reduces decode from 10,191 nodes to 4,245, better
  than this catalogue's 5,421. Make the strongest case for adopting it anyway.
  Then say which single measured number defeats that case most decisively, and
  why the other three grounds are not merely aesthetic.
- [ ] Every candidate produced logits **bit-identical** to the reference.
  Explain why that was the predicted outcome from the six passes alone, what a
  1-ULP difference would have implied, and why the same measurement on the
  rejected pass could not have been read the same way.
- [ ] The parity records carry `record_task_id: "T21"` and
  `evidence_tier: "real_onnxruntime_cpu"`. Explain what each field does and
  does not identify, and name the field that actually proves which bytes were
  executed.
- [ ] A colleague writes "the T22 candidates are QNN-ready, with a 63% node
  reduction and verified numerical parity". Rewrite that sentence so every
  clause is supported by committed evidence, and list what you had to remove.
