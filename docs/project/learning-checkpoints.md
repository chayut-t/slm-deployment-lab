# Learner attention and study checkpoints

Updated: 2026-07-25

## Purpose

This project is both an implementation and a deployment-engineering
curriculum. Agents may implement tasks autonomously, but the learner should not
miss the design decisions, hardware evidence, experiments, and failure
analysis that build the intended skillset.

For every task listed below, the coordinating agent should:

1. At task start, explain the skill being developed and point to relevant
   prerequisite reading or notebooks.
2. Before a hands-on checkpoint, tell the learner exactly what account,
   hardware interaction, approval, or judgment is needed.
3. During long tasks, call out an important result or decision when it becomes
   concrete rather than waiting until final completion.
4. At task completion, provide links to the implementation, worklog, results,
   guide, and notebook and give a short study checklist.
5. Never mark the learner's study checkbox complete without explicit
   confirmation from the learner.

These reminders should not block safe implementation unless the task requires
the learner's authentication, hardware interaction, spending approval, or
subjective/public decision.

## Attention levels

- **Deep study:** Understand the implementation and be able to explain the
  engineering trade-offs.
- **Hands-on:** Participate directly in an account, hardware, profiling, or
  approval step.
- **Review:** Inspect the result and its evidence; detailed implementation can
  mostly be delegated.

## Immediate ready tasks

### T10 — Token, prompt, and evaluation fixtures

Attention: Deep study

Learn:

- Why immutable tokenizer revisions and token IDs matter.
- How prompt fixtures control numerical and benchmark comparability.
- How the 128, 512, 1,024, and 4,096-token workloads are constructed.
- Why canaries and licensed/private evaluation data require different
  handling.

Study checkpoint:

- [ ] Review the frozen prompts, token sequences, hashes, and four context
  variants before T10 is accepted.

Related notebook: `00_model_shape_memory_budget.ipynb`

### T30 — Workbench compile, inference, and profile adapters

Attention: Deep study

Learn:

- How source artifacts, compilation jobs, target models, inference datasets,
  and profile jobs relate.
- How raw service results become sanitized, traceable manifests.
- How to keep tokens, job URLs, and account identifiers out of logs.
- Why compile, inference, and profile must remain independently runnable.

Study checkpoint:

- [ ] Walk through one adapter lifecycle from source manifest to normalized
  profile without using private job identifiers.

Related notebooks: `04_ai_hub_pipeline.ipynb`,
`05_qnn_profile_analysis.ipynb`

### T32 — Device Cloud Qwen GenieX baseline and generation loop

Attention: Hands-on and deep study

Learner action may be required for Device Cloud authentication, free-minute
activation, session scheduling, or interactive access.

Learn:

- The difference between a Workbench single-graph measurement and a persistent
  device-side generation loop.
- How loading, tokenization, prefill, decode, and total generation time are
  separated.
- What the ready-made GenieX/`llama.cpp` path proves and what it does not prove
  about the custom QNN path.

Study checkpoint:

- [x] Participate in or review the Device Cloud session and explain every
  timing boundary before T32 is accepted.

Related notebook: `04_ai_hub_pipeline.ipynb`

## Core model, graph, and benchmark skills

### T11 — Deterministic PyTorch reference

Attention: Deep study

- Full-forward versus cached autoregressive execution.
- Deterministic multi-token generation and tolerance design.
- Reference logits as the numerical oracle for later backends.

Study checkpoint:

- [ ] Reproduce and explain full-forward/cached parity and one deterministic
  generation fixture.

Related notebook: `01_prefill_decode_cache_contracts.ipynb`

### T12 — Static cache and tensor contract

Attention: Deep study

- Static prefill and one-token decode graph interfaces.
- Explicit tensor names, layouts, dtypes, and shapes.
- GQA-aware K/V cache dimensions and cache-update cost.

Study checkpoint:

- [x] Draw the prefill/decode tensor contract and calculate KV-cache bytes for
  at least two context sizes.

Related notebooks: `00_model_shape_memory_budget.ipynb`,
`01_prefill_decode_cache_contracts.ipynb`

### T13 — Benchmark and evaluation protocol

Attention: Deep study

- Warm-up, synchronization, repetition, percentiles, and dispersion.
- TTFT, prefill throughput, decode latency, and sustained generation.
- The distinction between graph, runtime, and end-to-end measurements.
- Why cross-platform numbers are system comparisons, not isolated software
  comparisons.

Study checkpoint:

- [ ] Review and approve the frozen measurement definitions before final
  benchmark runs.

Related notebook: `11_cross_platform_benchmark.ipynb`

### T20 — Four-context ONNX export matrix

Attention: Deep study

- Static ONNX export for prefill/decode variants.
- External data, artifact hashes, shape conformance, and provenance.
- Why export success does not establish compiler or hardware success.

Study checkpoint:

- [ ] Inspect at least one prefill and decode export and trace their shapes to
  the T12 contract.

Related notebook: `02_onnx_export_and_shapes.ipynb`

### T21 — ONNX Runtime CPU parity and graph inspection

Attention: Deep study

- Multi-step cache validation in ONNX Runtime.
- Finding dynamic shapes, unsupported patterns, and compiler risks.
- Distinguishing numerical errors from export or state-update errors.

Study checkpoint:

- [ ] Review the graph inspection report and explain the most important
  deployment risks.

Related notebook: `03_graph_inspection.ipynb`

### T22 — QNN candidates and packaging

Attention: Deep study

- Compiler-oriented graph transformations.
- Keeping the reference artifact distinct from the QNN candidate.
- Recording every transformation, package boundary, and checksum.

Study checkpoint:

- [ ] Compare one reference ONNX graph with its QNN candidate and explain every
  transformation.

Related notebooks: `02_onnx_export_and_shapes.ipynb`,
`03_graph_inspection.ipynb`

## Qualcomm deployment skills

### T31 — Qwen Workbench results on three Qualcomm targets

Attention: Deep study and review

- X Elite compile, numerical inference, and physical-device profile evidence.
- IQ-9075 and Snapdragon 8 Elite compatibility or bounded blockers.
- Device/toolchain identity, NPU placement, memory, and latency.
- Why proxy-device claims are unacceptable.

Study checkpoint:

- [ ] Review all three target records and explain which comparisons are valid.

Related notebooks: `04_ai_hub_pipeline.ipynb`,
`05_qnn_profile_analysis.ipynb`

### T33 — Integrated floating-point Qualcomm milestone or fallback

Attention: Deep study and learner decision

- Integrating export, compile, inference, profiling, and generation into one
  traceable path.
- Isolating a Qwen compiler/runtime boundary.
- Choosing the smallest verified fallback without mislabeling its evidence as
  Qwen evidence.

Study checkpoint:

- [ ] Personally review and approve any fallback decision and the final
  floating-path claim.

Related notebooks: `04_ai_hub_pipeline.ipynb`,
`05_qnn_profile_analysis.ipynb`

## Quantization skills

### T40 — AIMET and calibration environment

Attention: Deep study

- Representative calibration inputs and preprocessing.
- Calibration data revision, licensing, and reproducibility.
- Baseline parity before quantization.

Study checkpoint:

- [ ] Review the calibration corpus manifest and explain why it represents the
  target workloads.

Related notebook: `06_w8_w4_calibration.ipynb`

### T41 — W8 quantization evidence

Attention: Deep study and review

- W8A16 versus W8A8.
- Simulated precision versus a compiled/deployed artifact.
- Quality deltas using the frozen evaluation protocol.

Study checkpoint:

- [ ] Compare floating, W8A16, and W8A8 quality and hardware evidence.

Related notebooks: `06_w8_w4_calibration.ipynb`,
`12_quantization_quality.ipynb`

### T42 — W4A8, LPBQ, LiteMP, and sensitivity evidence

Attention: Deep study

- Layer/block sensitivity analysis.
- LPBQ and mixed-precision candidate selection.
- Why low-bit support must be measured rather than assumed.
- Quality and memory effects of precision allocation.

Study checkpoint:

- [ ] Explain the sensitivity map and justify the selected mixed-precision
  candidate.

Related notebooks: `07_litemp_lpbq_sensitivity.ipynb`,
`12_quantization_quality.ipynb`

### T43 — Quantized compile, inference, and profile

Attention: Deep study and review

- Connecting quantization simulation to deployable hardware artifacts.
- Comparing quality, latency, memory, and compute placement.
- Reporting unsupported low-bit paths without overstating deployment.

Study checkpoint:

- [ ] Review the complete W8 hardware path and any low-bit fallback before
  accepting the public conclusions.

Related notebooks: `05_qnn_profile_analysis.ipynb`,
`12_quantization_quality.ipynb`

## Apple M4 and MLX skills

### T50 — MLX-LM baseline

Attention: Review

- Establishing a versioned correctness/performance baseline.
- Exact host/runtime identity and avoiding unsupported ANE claims.

Study checkpoint:

- [x] Compare MLX-LM output and timings with the PyTorch reference.

### T51 — Custom MLX runtime

Attention: Deep study

- Explicit MLX prefill/decode and cache state.
- GQA-aware layouts without unnecessary K/V materialization.
- Lazy evaluation and reusable runtime design outside notebooks.

Study checkpoint:

- [ ] Walk through one layer's GQA/cache implementation and explain its memory
  behavior.

Related notebook: `08_mlx_gqa_kv_layout.ipynb`

### T52 — Apple profiling and context sweep

Attention: Hands-on and deep study

Learner action may be useful for Instruments captures, thermal stabilization,
and interpreting power or memory-pressure observations on the M4.

Learn:

- TTFT, prefill, decode, dispatch, and synchronization.
- Unified-memory traffic and GQA layout effects.
- `mx.compile`, thermal drift, peak memory, and power behavior.

Study checkpoint:

- [ ] Review an Instruments/MLX trace and explain the dominant bottleneck with
  evidence.

Related notebook: `09_mlx_compile_and_profile.ipynb`

## NVIDIA CUDA skills

### T60 — ONNX Runtime CUDA context sweep

Attention: Hands-on and deep study

Learner action may be required to allocate a Colab/Kaggle GPU or approve a paid
fallback.

Learn:

- CUDA Execution Provider placement and explicit failure on CPU fallback.
- GPU synchronization and host/device transfer accounting.
- Ordinary feeds versus I/O binding.

Study checkpoint:

- [ ] Confirm the actual GPU/runtime manifest and explain the I/O-binding
  result.

Related notebook: `10_ort_cuda_iobinding.ipynb`

## Integration, automation, and publication skills

### T70 — FastAPI backend contract

Attention: Review

- Backend-neutral generation and capability contracts.
- Why remote hosted jobs are not local token backends.

Study checkpoint:

- [ ] Review the backend interface and capability schema.

### T71 — React UI and local demo

Attention: Review

- Presenting backend capabilities and evidence honestly.
- Separating live backends from replay/mock results.

Study checkpoint:

- [ ] Review the recorded walkthrough and verify every displayed claim.

### T72 — Manual GitHub Actions AI Hub workflow

Attention: Hands-on and deep study

Learner action is required to manage the GitHub secret and approve workflow
execution.

Learn:

- `workflow_dispatch`, runners, jobs, steps, inputs, secrets, and artifacts.
- Fork-safe secret handling and why local scripts remain the implementation
  source.

Study checkpoint:

- [ ] Review the workflow security model before adding the secret or running
  the workflow.

### T80 — Integrated guides, notebooks, and MkDocs

Attention: Deep study

T80 owns the final notebook and guide integration, but study each notebook when
its underlying task finishes rather than waiting until T80.

Study checkpoint:

- [ ] Complete or review every notebook in the notebook map below.
- [ ] Confirm notebooks call reusable package logic instead of hiding the only
  implementation.

### T81 — Final evaluation and report

Attention: Deep study and learner decision

- Regenerating results from committed data and manifests.
- Cross-platform interpretation without erasing hardware/runtime differences.
- Tracing every public claim to numerical or hardware evidence.

Study checkpoint:

- [ ] Personally review the final plots, limitations, and conclusions.

Related notebooks: `11_cross_platform_benchmark.ipynb`,
`12_quantization_quality.ipynb`

### T82 — Reproduction and release audit

Attention: Hands-on final approval

- Clean-checkout reproduction, secrets, privacy, licenses, and artifact policy.
- Ensuring README, demo, portfolio, and resume claims do not exceed evidence.

Study checkpoint:

- [ ] Personally approve the final README, limitations, demo, and resume
  bullets.

## Planned notebook map

The notebooks are created and integrated by T80. Agents should remind the
learner to study them alongside their underlying implementation tasks.

| Notebook | Related tasks | Main skill |
|---|---|---|
| `00_model_shape_memory_budget.ipynb` | T10–T12 | Tensor, KV-cache, bandwidth, and artifact budgets |
| `01_prefill_decode_cache_contracts.ipynb` | T11–T12 | Static prefill/decode and cache updates |
| `02_onnx_export_and_shapes.ipynb` | T20–T22 | ONNX export and compiler shapes |
| `03_graph_inspection.ipynb` | T21–T22 | Graph risks and rewrites |
| `04_ai_hub_pipeline.ipynb` | T30–T33 | Qualcomm Workbench and Device Cloud pipeline |
| `05_qnn_profile_analysis.ipynb` | T31, T43 | NPU placement and profile interpretation |
| `06_w8_w4_calibration.ipynb` | T40–T41 | Calibration and W8/W4 behavior |
| `07_litemp_lpbq_sensitivity.ipynb` | T42–T43 | Sensitivity and mixed precision |
| `08_mlx_gqa_kv_layout.ipynb` | T51 | GQA-efficient MLX cache layouts |
| `09_mlx_compile_and_profile.ipynb` | T52 | MLX compilation and M4 profiling |
| `10_ort_cuda_iobinding.ipynb` | T60 | CUDA provider placement and I/O binding |
| `11_cross_platform_benchmark.ipynb` | T13, T81 | Fair benchmarking and statistics |
| `12_quantization_quality.ipynb` | T41–T43, T81 | Quantization quality/performance trade-offs |

## Completed-task review

The following completed foundations should be reviewed once, with T02 receiving
the most technical attention:

- [ ] **T00:** Review the model/version ADR and understand why immutable
  revisions and artifact fields are required.
- [ ] **T01:** Review the pinned environment, artifact schemas, host manifests,
  and storage preflight.
- [ ] **T02:** Review the successful Workbench compile → inference → profile
  report, numerical output, NPU placement, latency/memory units, privacy
  boundary, and the distinction between service turnaround and graph latency.
- [ ] **T03:** Review the task graph, dependency gates, worktree rules, privacy
  boundary, and completion/worklog contract.

## Reminder template for agents

At task start:

> Learning checkpoint for TNN: this task develops [skill]. Before completion,
> study [guide/notebook/output]. Your hands-on action is [action or "none"].

During the task:

> TNN learning checkpoint: [specific result or decision] is now concrete.
> Review [artifact] and focus on [trade-off].

At task completion:

> TNN is technically complete. Before considering your study complete, review
> [links], reproduce [small check], and be able to explain [three concepts].
