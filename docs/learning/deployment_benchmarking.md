# Deployment benchmarking without false equivalence

Task: `T13`

## What this protocol protects

A benchmark is a claim about a defined system boundary, not merely a timer.
The same Qwen weights can produce several legitimate but incomparable
latencies:

```text
hosted graph invocation
  != synchronized local prefill/decode stage
  != persistent generation loop
  != API request visible to a user
  != cold process plus model load
```

T13 freezes those scopes before hardware runs. The machine-readable authority
is
[`benchmark-protocol-v1.json`](../../configs/workloads/benchmark-protocol-v1.json),
and every future metric record must pass
[`benchmark-result-v1.schema.json`](../../configs/workloads/benchmark-result-v1.schema.json)
plus semantic validation in
[`protocol.py`](../../src/slm_lab/benchmark/protocol.py).

This task contains no model or hardware measurements. It defines how later
tasks may collect and describe them.

## Frozen workload matrix

T13 inherits exact token IDs and generation behavior from the
[T10 fixture contract](../../configs/workloads/t10-token-fixtures.json).
Backends may not substitute a more convenient prompt.

| Workload | Prompt tokens | Output limit | Main use |
|---|---:|---:|---|
| S128 | 128 | 32 | Short feasibility |
| S512 | 512 | 64 | Interactive/cache growth |
| S1024 | 1,024 | 128 | Main system comparison |
| S4096 | 4,096 | 128 | Long-context pressure |

Decode probes use preloaded cache lengths 128, 512, 1,024, and 4,096. Every
measured token begins from equivalent cache state. Timing consecutive decode
steps while the cache grows would answer a different question.

## Timing classes and repetition policy

| Class | Scope | Warm-up | Measured | Load included? |
|---|---|---:|---:|---|
| `single_graph` | One hosted/deployed graph | 5 | 30 | No |
| `runtime_stage` | Prefill or one-token decode | 5 | 30 | No |
| `generation_loop` | Warm persistent token loop | 2 | 10 | No |
| `end_to_end_request` | Measured API request | 2 | 10 | No |
| `cold_start` | Fresh process through first token | 0 | 5 | Yes |

Compilation is excluded from all five latency series and reported separately.
Cold-start repetitions require a fresh process. Model/artifact load is excluded
from warm scopes and must appear explicitly in their `excludes` list.

Warm-up is not discarded evidence of a desired speedup. It removes lazy
initialization, cache population, graph specialization, and allocator setup
from a deliberately steady-state question. Cold-start is a separate measured
question so those costs remain visible.

## Synchronization is part of the timer

CPU calls normally return after work completes. GPU/NPU APIs may return after
enqueue. A host timer around only enqueue measures scheduling overhead.

The protocol therefore requires:

- ONNX Runtime CUDA: synchronize the active CUDA stream/device before timer
  start and after requested outputs materialize.
- MLX: materialize setup state before timing and timed outputs with `mx.eval`
  before timer stop.
- Qualcomm Workbench: use service-reported device graph/profile time for graph
  scope; queue and download wall time are service metrics.
- Device Cloud: record the blocking runtime API/fence and separately expose
  loading, tokenization, prefill, decode, and detokenization.

The result stores the exact synchronization method, not merely `synchronized:
true`.

## Metric boundaries

### TTFT

TTFT needs a start boundary. This protocol distinguishes:

- `ttft_warm`: token IDs and loaded model at generation-loop entry through the
  first output token.
- `request_ttft`: complete API request receipt through the first token at that
  API boundary.
- `cold_ttft`: fresh process before artifact/model load through first token.

Queue/dispatch, artifact load, model load, tokenization, prefill, first decode,
and detokenization/transfer are separately named components. If a platform
cannot expose one component, record it as unavailable rather than silently
assigning it to another.

### Prefill and decode

Prefill throughput is exact prompt tokens divided by synchronized prefill
seconds. Decode throughput is actual generated tokens divided by summed
synchronized decode seconds. Also report:

- Decode time per output token.
- Generation throughput including prefill.
- Generation throughput excluding prefill.
- Actual generated count when EOS stops early.

AI Hub graph latency is useful compiler/device evidence, but it provides none
of these persistent-loop rates by itself.

### Memory

“Peak memory” is incomplete without a domain and mechanism. Process RSS,
system memory, Apple unified memory, accelerator allocation, runtime-reported
memory, and a hosted-service estimate remain separate. Each result records
method, sampling interval, measurement interval, and baseline policy.

### Power and thermal behavior

Steady-state power/thermal claims require three repetitions of at least ten
minutes, an instrument and sample rate, the measurement domain, idle-baseline
policy, start/end temperature, thermal state, and ambient notes. An estimate is
allowed only with `evidence_level=estimated`; it cannot be styled as an
observation.

## Statistics policy

Store unrounded base-unit samples. For valid samples report:

- Minimum, maximum, and arithmetic mean.
- Sample standard deviation (`n - 1` denominator).
- Median, p90, and p95 using Hyndman–Fan type 7 linear quantiles.
- Median absolute deviation and interquartile range.
- Sample counts, including invalid samples.

Headline comparisons also require a seeded 10,000-resample percentile
bootstrap 95% interval for the median.

The outlier rule is intentionally strict: keep every valid sample. A value is
not invalid merely because it is slow or distant. Only predeclared timer,
runtime, integrity, device/provider, environmental-threshold, or external
interruption failures may be excluded. Preserve such a sample and its reason,
mark the series incomplete, and rerun the whole series before headline use.

The validator recomputes summaries from raw valid samples, which prevents a
presentation table from drifting away from its evidence.

## Numerical and quality evaluation

The numerical oracle remains a named reference level:

1. Golden PyTorch reference.
2. Deployment floating baseline.
3. Quantized candidate.

Logit/cache comparisons record maximum and mean absolute error, protected
relative error with denominator `max(abs(reference), 1e-6)`, cosine
similarity, KL divergence or a documented top-k approximation, top-1
agreement, top-5 overlap, reference margin, per-layer cache error, cached
versus full-forward error by decode step, and deterministic token canaries.
Tolerance policies differ for dtype conversion, export, backend, and
quantization. Token equality alone is insufficient.

The pinned
[academic regression subset](../../configs/workloads/academic-evaluation-v1.json)
uses lm-evaluation-harness `v0.4.12` with raw completion, no chat template, and
zero-shot scoring:

| Sentinel | Pinned selection | Role |
|---|---|---|
| WikiText-2 raw | Full test split | Perplexity/NLL regression |
| HellaSwag | First 1,000 validation rows in pinned order | Commonsense completion regression |
| ARC Easy | Full validation split | Small science multiple-choice regression |

The dataset and harness revisions, split, selection, scoring, and window
metadata are part of every quality result. Third-party rows and sample logs are
not committed. These sentinels detect deployment regressions; their limited
scores are not broad capability claims. PIQA is excluded from v1 because its
currently rendered dataset-card license metadata and merged licensing
discussion are inconsistent.

Primary references:

- [lm-evaluation-harness releases](https://github.com/EleutherAI/lm-evaluation-harness/releases)
- [lm-evaluation-harness task interface](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)
- [WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext)
- [HellaSwag dataset card](https://huggingface.co/datasets/Rowan/hellaswag)
- [ARC dataset card](https://huggingface.co/datasets/allenai/ai2_arc)

## Cross-platform interpretation

Every result is a system result. Keep device, OS, runtime, compiler, provider,
placement evidence, graph/cache contract, artifact hash, precision, and
measurement method attached.

A fair report can hold workload and metric definition constant while still
listing non-comparable dimensions:

```text
same token IDs + same output policy + same timing scope + same statistics
    while retaining
hardware + OS + compiler + runtime + precision + placement + cache strategy
```

This supports a valid system comparison. It does not isolate “runtime
software” as the cause of the difference. Context scaling and uncertainty are
more informative than one unqualified token/s ranking.

## Offline hands-on check

Validate the contracts and statistics implementation:

```bash
uv run python -m slm_lab.benchmark.protocol check --root .
```

Inspect the timing classes:

```bash
jq '.timing_classes |
  to_entries[] |
  {class: .key, scope: .value.scope,
   warmup: .value.warmup_repetitions,
   measured: .value.measured_repetitions}' \
  configs/workloads/benchmark-protocol-v1.json
```

The related `11_cross_platform_benchmark.ipynb` is intentionally created later
by T80. It must call this reusable implementation rather than redefine the
statistics in notebook cells.

## Study/debrief checklist

Do not mark these complete without personally reviewing the frozen contract:

- [ ] Explain why five graph warm-ups and thirty samples answer a different
  question from five fresh-process cold starts.
- [ ] Draw the start/stop boundary for warm TTFT, request TTFT, and cold TTFT.
- [ ] Explain why CUDA/MLX synchronization changes the number being measured.
- [ ] Recompute median, p95, MAD, and IQR from one sample series.
- [ ] Explain why a slow but valid sample stays in the result.
- [ ] Identify at least three system differences that a Qualcomm/Apple/NVIDIA
  comparison must retain.
- [ ] Review and approve the frozen measurement definitions before final
  benchmark runs.
