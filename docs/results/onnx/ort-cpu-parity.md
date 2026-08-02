# ONNX Runtime CPU parity for the T20 Qwen3-0.6B reference graphs

Task: `T21`
Date: 2026-08-02
Status: **no measurement**

> **No real ONNX Runtime measurement has been taken.** There is no parity
> number in this repository, and none is claimed here. `onnxruntime`, `torch`,
> `numpy`, `onnx`, and `transformers` are all absent from the environment in
> which this work was done, so the runner has never executed a real graph. What
> follows documents validated comparison machinery, a set of *proposed and
> unvalidated* tolerances, and the exact command a host with the runtime must
> run to produce the real result. Every number the machinery can currently
> produce comes from injected fake sessions and is labelled
> `evidence_tier="fake_session_self_test"` in its own evidence record.

The T21 acceptance criteria "ORT outputs satisfy numerical tolerances" and
"Multiple decode steps update cache correctly" are therefore **not yet
satisfied by measurement**. They are satisfied by construction: the criteria
are encoded as executable, fault-injected checks, and the run that evaluates
them is a single recorded command away.

## What is being compared

| Side | Definition |
|---|---|
| Reference | Qwen3-0.6B, `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca`, loaded in **bfloat16** (`configs/models/qwen3-0.6b.yaml`, `reference_dtype`), `attn_implementation="eager"`, device `cpu`, seed 0, driven by the T11 deterministic loop through `slm_lab.backends.onnx_cpu.TorchReferenceSource` |
| Candidate | The T20 static ONNX graphs `S128/prefill.onnx` and `S128/decode.onnx`, exported at **float16** precision, opset 18, by `torch.onnx.export` 2.7.1 with constant folding disabled and no dynamic axes (`configs/models/qwen3-0.6b-onnx-export.json`), executed on the ONNX Runtime **CPU execution provider** |
| Comparison surface | Next-token logits (FP32 on both sides) after prefill and after each decode step, plus the full fixed-capacity KV cache tensors between steps |

Both sides use the same model revision, the same frozen T10 prompt token IDs
for S128, eager attention, CPU, and seed 0. Four things are therefore held
constant and two are deliberately varied:

**Held constant:** model weights and revision, prompt tokens, attention
implementation, device class.

**Varied — the two error sources this comparison admits:**

1. **A dtype conversion.** The reference computes in BF16; the graph stores
   weights and the entire KV cache in FP16 (`CACHE_DTYPE = "float16"` in
   `src/slm_lab/contracts/static_cache.py`). BF16 and FP16 have the same width
   but different exponent/mantissa splits, so this is not a widening — values
   are re-rounded on a different grid at every cache and weight boundary.
2. **A backend and accumulation-order change.** ONNX Runtime selects its own
   kernels and reduction orders. Even at identical dtype this moves the last
   bits of every reduction, and 28 layers of residual accumulation amplify it.

Note that the *compared* tensors are FP32 on both sides: the T12 contract fixes
`last_logits` and `next_logits` at `float32` (`LOGITS_DTYPE`), and
`TorchReferenceSource._materialize` casts the reference logits with
`.to(torch.float32)`. The FP16 lives in the weights, the internal compute, and
the cache — not in the comparison boundary. The comparison is therefore
measuring accumulated internal precision loss, not a truncation applied at the
final step.

Per `docs/project/plan.md` section 6.7, distinct tolerances are required for
**dtype conversion**, **ONNX export**, **backend parity**, and **quantized
quality**. This comparison stacks the first three and is reported as a
**backend-parity** tolerance: it is the first check in the chain that actually
executes the exported graph on a foreign runtime. It is not a quantization
tolerance — nothing here is quantized; `quantization` is `null` in
`results/manifests/onnx/S128.json`.

### Why this is a strictly harder comparison than T11

`src/slm_lab/generation/reference.py` `DEFAULT_TOLERANCE` governs T11, whose
comment is explicit: *"Same pinned model, dtype, device, and eager attention
implementation. These thresholds admit BF16 accumulation-order noise, not
backend or dtype changes."* T11 compares one model against **itself** — a full
forward pass against a cached decode loop — in the same BF16, on the same
backend. The only error source is reduction order inside one library.

T21 keeps every T11 error source and adds two independent ones on top: the
BF16→FP16 grid change and a different runtime's kernels. A tolerance derived
for T11 is therefore a *lower bound* on what T21 needs, never an upper bound.

## Tolerances

The thresholds live in `DEFAULT_ORT_CPU_TOLERANCE` in
`src/slm_lab/backends/onnx_cpu.py`. `ParityTolerance` mirrors T11's
`NumericalTolerance` field for field so a T21 number and a T11 number are
directly comparable.

| Threshold | T21 proposed | T11 in use, confirmed by a real run | Reason for the T21 value |
|---|---:|---:|---|
| `atol` | 0.25 | 0.25 | Qwen3-0.6B next-token logits span roughly ±20; FP16 spacing near 20 is about 0.016, and 28 layers accumulate rounding through the residual stream. 0.25 is ~1% of the logit range — above dtype noise, far below the whole-unit shifts a mis-wired graph produces. |
| `rtol` | 0.02 | 0.02 | Proportional room where FP16 spacing is widest, without granting that room to the near-zero bulk of the 151,936-entry vocabulary. |
| `protected_relative_max` | 0.10 | 0.10 | Raw relative error is meaningless where the reference logit is ~0. Above the floor, 10% is generous for dtype noise and still far below what a wrong cache read produces. |
| `relative_floor` | 1.0 | 1.0 | Denominator floor for the protected relative error, identical to T11. |
| `cosine_min` | 0.999 | 0.999 | Whole-vocabulary direction check that no per-element bound provides. Dtype noise barely rotates a 151,936-dimensional vector; reading the wrong cache slot rotates it a lot. |
| `top5_overlap_min` | 0.8 | 0.8 | At most one of the reference's top five may leave the candidate's top five. Reordering within the top five is plausible FP16 behaviour at small margins; losing two is not. |
| `require_top1` | `True` | `True` | Greedy decoding is only reproducible if argmax agrees. This is the decision-level criterion a deployment actually depends on. |
| `cache_state` | `EXACT_CACHE_STATE_TOLERANCE` | *(no equivalent)* | Cache regions the contract calls untouched are compared for **exact equality**, never closeness. See the next section. |

What the middle column means, exactly: T11's `DEFAULT_TOLERANCE` values were
*authored*, not derived from a measurement — its own comment at
`src/slm_lab/generation/reference.py:50-51` claims only that they "admit BF16
accumulation-order noise, not backend or dtype changes". What makes them more
than a proposal is that a real Qwen3-0.6B run passed them with room to spare:
`ai/worklogs/2026-07-25-T11-deterministic-pytorch-reference.md:87` records
"zero absolute error and identical float32 fingerprints" on all three
full-versus-cache logit pairs. So they are in use and unfalsified, not fitted to
data — which is a strictly stronger position than T21's column, where no run of
any kind has happened.

The logit thresholds are deliberately **no tighter than T11's**, even though
T21 admits strictly more error. Setting them tighter would guarantee a failure
that says nothing; setting them looser without evidence would hide real
defects. Equal-to-T11 is the honest starting hypothesis: it asserts "the extra
dtype and backend error should not visibly exceed the same-model noise", and it
is a falsifiable claim that the first real run will settle.

### These tolerances are proposed and unvalidated

No ONNX Runtime execution has ever checked them. The code says so in three
places, and the evidence record says so in its own serialized payload:

- The block comment above `DEFAULT_ORT_CPU_TOLERANCE` opens with
  `# PROPOSED, NOT MEASURED.`
- `TOLERANCE_STATUS` is the string
  `"proposed_unvalidated: no ONNX Runtime run has confirmed these thresholds"`.
- `ParityTolerance.as_dict()` emits that string under the key `status`, so
  every evidence JSON — real or fake — carries
  `tolerance.status: proposed_unvalidated…`. This is asserted by
  `test_evidence_json_is_deterministic_and_digest_is_sensitive`.

**What the first real run must do about them.** Confirm or replace. There is no
third option, and "loosen until green" is not "confirm":

1. If the run passes, record the observed `max_absolute_error`,
   `max_protected_relative_error`, `cosine_similarity`, and `top5_overlap` from
   the evidence, and tighten each threshold to a documented margin above the
   observed value. A threshold ten times larger than the measured error is not
   a validated threshold — it is an unfalsified one.
2. If the run fails, do **not** widen the threshold before reading
   `failures[].kind` — which is ordered so the most fundamental class is
   `failures[0]`. A `cache_state_update` failure is not a tolerance problem and
   must never be answered by retolerancing. Neither is a `non_finite_logits`
   failure: no threshold admits a NaN, and the fix is on the export's precision
   side. Only a `numerical_tolerance`
   failure with a clean cache report is even a candidate for retolerancing, and
   then only with the measured error and a written justification for the new
   value.
3. Either way, replace `TOLERANCE_STATUS` and record the evidence digest that
   justified the change.

### The `allclose` convention

This is easy to get wrong when reproducing, so it is stated exactly.
`compare_logits` implements torch's convention:

```
|reference - candidate| <= atol + rtol * |candidate|
```

`rtol` scales the **candidate** — the ONNX Runtime output, the `other` operand
of `torch.allclose` — not the reference. The comparison is therefore *not*
symmetric: swapping the two arguments can change the verdict. Two tests pin
this down: `test_allclose_boundary_matches_the_torch_convention` (a difference
of exactly `atol` passes; `atol + 1e-7` fails) and
`test_rtol_scales_the_candidate_operand` (`|8-4| = 4 > 0.5·|4|` fails, while
`|4-8| = 4 <= 0.5·|8|` passes with the operands swapped).

Two further conventions matter for reproduction:

- **Cosine denominator floor.** The denominator is the product of the two norms
  floored at `COSINE_DENOMINATOR_FLOOR = 1e-8`. This is a floor, not a special
  case: a degenerate pair yields `dot / 1e-8`, not a defined-away `0.0`; it is
  only `0.0` when the dot product itself is zero, which is the all-zero-vector
  case. Pinned by `test_cosine_denominator_is_floored_and_does_not_zero_out`,
  and, on the input class that distinguishes a product floor from a per-norm
  floor, by `test_the_cosine_floor_applies_to_the_norm_product_not_each_norm`
  (norms `1e-12` and `1e+6`: the product `1e-6` is above the floor, so nothing
  is clamped and the result is `1.0`; flooring each norm separately would
  report `1e-4`).

  **Whether this differs from torch is unverified.**
  `torch.nn.functional.cosine_similarity` also floors its denominator at a
  default `eps` of `1e-8`. Its **published formula** writes that floor per
  norm — `max(‖x1‖, eps) · max(‖x2‖, eps)` — which would differ from this
  module on exactly the degenerate input above. Its **implementation** is
  reported to clamp the product of the two squared norms before taking the
  square root, which would agree with this module everywhere. No host in this
  task has torch installed, so neither reading has been checked against a real
  `F.cosine_similarity` call, and an earlier revision of this document was
  wrong to call the difference a deliberate deviation. Like the ORT tolerances
  above, this stays labelled unverified until a host with torch settles it. It
  affects no number published here: the two conventions agree for every norm at
  or above the floor, which is every non-degenerate logit vector.

  The result is **not** clamped to `[-1, 1]`, because torch does not clamp its
  output either — `F.cosine_similarity` can return a few ULPs above `1.0` for
  identical vectors, and clamping here would diverge from
  `slm_lab.generation.reference.compare_logits`, which calls the real torch
  function. `compare_logits([1.0, 5.0], [1.0, 5.0])` returns
  `1.0000000000000002`, and `test_cosine_similarity_is_not_clamped_to_one`
  fails if a clamp is reintroduced. Since `cosine_min` is a lower bound, an
  above-one value cannot mask a failure.
- **Tie-breaking.** `top_indices` breaks ties to the *lowest* index, matching
  `torch.argmax`'s first-maximum rule and the T10 fixture convention. Pinned by
  `test_top1_ties_break_to_the_lowest_index`.

`compare_logits` refuses non-finite input on either side: it raises
`ParityInputError("logits contain NaN or infinite values")` rather than
producing a metric, so a NaN can never be averaged into a passing score. That
is the pure-function contract, and it matches T11. The *runner* does not let
that exception decide the run's fate, because the two sides are not
symmetrical:

- **Candidate** logits that are NaN or Inf are a graph behaviour — the most
  likely one being an FP16 export that overflows — so `OrtCpuParityRunner`
  screens for them before calling `compare_logits`, records the step with
  `metrics: null` and a `non_finite_candidate_logits` count, classifies it as
  `non_finite_logits`, and carries on with the remaining steps. The run still
  produces evidence, the cache invariants are still checked on every step, and
  the CLI exits `1`.
- **Reference** logits that are NaN or Inf are a broken input: the golden
  fixture or the reference run itself is unusable and there is no graph
  behaviour to classify. That stays `ParityInputError`, and the CLI exits `2`.

Pinned by `test_non_finite_candidate_logits_are_classified_not_fatal`,
`test_a_single_non_finite_candidate_logit_is_counted_exactly`,
`test_cli_exits_one_on_non_finite_logits_not_two`, and
`test_non_finite_reference_logits_stay_a_configuration_error`.

## Multi-step cache validation

This is the substance of the task. "The cache updated correctly" is not a
feeling about a diff; it is a named set of invariants derived from the frozen
T12 contract in `src/slm_lab/contracts/static_cache.py`.

### The contract, in prose

**Prefill** (`prefill_prefix_materialization`). Given a prompt of length `S`
and a fixed cache capacity `C`, the prefill graph emits, for each of 28 layers,
a `key_cache.L` and a `value_cache.L` of shape `[1, 8, C, 128]` in FP16 with
layout `(batch, kv_head, cache_position, head_dim)`. Cache positions `[0, S)`
hold the prompt's keys and values. Positions `[S, C)` are **exactly zero** —
not "small", not "unspecified": zero. The graph also emits `valid_length = S`.
For S128, `S = 128` and `C = 160` (`CONTEXT_VARIANTS[128] = 160`); the capacity
exceeds the prompt because a capacity equal to the prompt length would make the
very first decode an overflow.

**Decode** (`fixed_capacity_indexed_copy`). The decode graph consumes the full
fixed-capacity cache plus a scalar `valid_length`, and treats only `[0,
valid_length)` as valid. It writes exactly one position — index `valid_length`
— in every `present_key.L` / `present_value.L`. Everything before that index
must come back byte-identical to what went in. Everything after it must come
back byte-identical too. The graph emits `updated_valid_length =
valid_length + 1`. Nothing grows: the tensor is the same shape on the way out
as on the way in, which is the entire point of a static-shape deployment graph.

### Why one decode step proves almost nothing

A decode graph that writes the *wrong* slot still returns plausible logits at
step one, because at step one there is only one previously-written region and
the corruption has not yet been read back. The defect surfaces only when a
later step attends over the position that the earlier step damaged. That is why
`OrtCpuParityRunner` threads state rather than replaying it: each step's output
cache tensors are fed straight back in as the next step's input cache tensors
(`cache = {source: outputs[target] for source, target, _ in self._cache_pairs}`
in `run()`), and `test_decode_feeds_thread_the_cache_and_build_masks` asserts
that step two's `key_cache.0` differs from step one's — i.e. that the runner is
genuinely chaining, not re-feeding the prefill cache.

The point is made concrete by
`test_prefix_corruption_at_step_three_is_caught_after_two_clean_steps`: with a
single element of `present_key.0` at cache position 5 perturbed on step 3 only,
steps 1 and 2 report `passed is True` and step 3 reports `passed is False`. A
single-step parity check would have reported success.

### Why the cache checks are exact, not tolerant

`CacheStateTolerance`'s docstring states the reasoning directly: *a KV cache
slot is copied, not recomputed*. The prefix and tail regions of a decode output
are the result of a data movement, not an arithmetic reduction, so there is no
legitimate source of a one-ulp difference in them. The comparison in
`compare_cache_tensor` is Python `!=` on FP16-derived float values —
bit-identical or a violation.

This is the design decision that makes T21 mean something. A tolerant cache
check with, say, `atol=0.25` would silently accept a graph that reads the wrong
cache row, because a wrong-but-nearby row differs by less than FP16 noise in
many elements. The state bug would then reappear downstream as an unexplained
"numerical" failure — exactly the confusion the task exists to prevent. Exact
comparison on the regions the contract says are untouched is what keeps the two
failure classes separable.

Note the one region that is *not* checked exactly, and cannot be: the newly
written slot itself. Its correct value is the result of real attention
arithmetic and would need its own reference cache to compare against. T21
checks only that it was written and is finite. This is a deliberate,
acknowledged boundary — see "Evidence boundaries".

### The invariants, by name

| Invariant | Where | What it asserts |
|---|---|---|
| `prefill_valid_length` | prefill | The emitted `valid_length` equals the contract's `prompt_length`. |
| `prefill_reserve_zero` | prefill | Every element of `[prompt_length, capacity)` is exactly `0.0`. |
| `write_index_within_capacity` | decode | `0 <= valid_length < capacity`; checked before any indexing *within* the step, and short-circuits the rest of that tensor's checks. Reachable in a real run because `run()` threads the graph's own `updated_valid_length` into the next step (see below), so a graph that reports a length outside its fixed capacity is caught on that next step — after that length has been used to build one feed, which is the ordering caveat noted below the table. |
| `prefix_preserved` | decode | `[0, valid_length)` is bit-identical before and after, per `kv_head` block. |
| `slot_written` | decode | The slot at `valid_length` **changed**; if it is identical to the incoming value, the step wrote nothing there. |
| `slot_finite` | decode | Every element of the written slot is finite (no NaN, no Inf). |
| `tail_untouched` | decode | `(valid_length, capacity)` is bit-identical before and after. |
| `valid_length_increment` | decode | `updated_valid_length == valid_length + 1`. |
| `written_slot_immutable` | whole run | A slot written at step *n* still holds those exact values in the final cache at the end of the run. |

Each violation is reported as a `CacheInvariantViolation` carrying `invariant`,
`tensor`, `layer`, `position`, `element`, and a human-readable `detail` — so a
failure names the layer and the cache position, not just "the cache is wrong".
`_layer_of` recovers the layer index from the tensor name suffix
(`present_key.0` → layer 0).

`written_slot_immutable` is the only whole-run invariant. It is computed in
`_check_slot_immutability` against the *final* cache state, so it detects
corruption that survives to the end of the run; corruption that appears and is
overwritten mid-run is caught instead by the per-step `prefix_preserved` check.
The two together cover the run.
`test_a_slot_that_changes_later_breaks_written_slot_immutability` pins it with a
graph that writes the previous step's slot: over three steps and four cache
tensors, eight recorded slots have moved by the end of the run, and each
violation is reported with no `step` because it is a property of the run rather
than of a step.

**Where the next step's `valid_length` comes from.** `run()` reads it out of the
decode graph's own `updated_valid_length` output rather than incrementing an
internal counter. That is deliberate for a validation tool: substituting the
value the runner expected would make it impossible to observe the graph
disagreeing. The disagreement is already recorded as `valid_length_increment`,
and carrying the reported value forward is what makes
`write_index_within_capacity` reachable — asserted by
`test_a_reported_valid_length_outside_capacity_is_caught_on_the_next_step`,
where a graph reporting `updated_valid_length = 160` at capacity 160 is caught
on the following step.

**Ordering caveat on that check.** The invariant table says
`write_index_within_capacity` is checked before any indexing, and inside
`compare_cache_tensor` and `_slot_values` that is exactly true — both guard
`0 <= valid_length < capacity` before any slice arithmetic, so an out-of-range,
negative, or skipped index is classified rather than raised. It is *not* true
across step boundaries. A bad `updated_valid_length` reported at the end of step
*N* is first used to build the **feed** for step *N+1* — `position_ids`,
`valid_length`, and the truncated `attention_mask` — and only then does step
*N+1*'s output get range-checked. A real session therefore sees one garbage
feed before the violation is recorded. That is benign here by construction:
the mask is capped at capacity and a negative length is tolerated, and if the
session raises on the bad feed the step is classified as `runtime_error`
instead. Either way the run produces evidence and the defect is named; the
difference is only which of the two classes appears.

### Four decode steps at S128, as the contract requires

This is the contract's prescription, not a measurement. Nothing below has been
observed on ONNX Runtime.

Capacity `C = 160`, prompt `S = 128`, 28 layers × 2 tensors = **56** cache
tensors checked per decode step (asserted at
`test_real_s128_contract_runs_end_to_end_with_fakes`:
`evidence.cache_report.steps[1].tensors_checked == 56`).

| Step | Graph | `valid_length` in | Write index | Prefix that must be preserved | Tail that must be untouched | `valid_length` out |
|---:|---|---:|---:|---|---|---:|
| 0 | prefill | — | — | — | `[128, 160)` must be exactly zero | 128 |
| 1 | decode | 128 | 128 | `[0, 128)` | `[129, 160)` | 129 |
| 2 | decode | 129 | 129 | `[0, 129)` | `[130, 160)` | 130 |
| 3 | decode | 130 | 130 | `[0, 130)` | `[131, 160)` | 131 |
| 4 | decode | 131 | 131 | `[0, 131)` | `[132, 160)` | 132 |

Each step's protected prefix grows by exactly the slot the previous step wrote,
which is what turns `prefix_preserved` into a cumulative check: by step 4 it
re-validates all three previously written slots as well as the whole prompt.
The reserved tail shrinks by one position per step and, because prefill
zero-filled it, remains zero throughout — `tail_untouched` compares
before-against-after, so it detects any write into the reserve regardless of
value.

The `attention_mask` fed at each decode step is built to match: `[1] *
min(valid_length + 1, capacity)` followed by zeros to capacity, so step 1 sends
129 ones and 31 zeros over the 160-wide mask.
`test_decode_feeds_thread_the_cache_and_build_masks` asserts exactly this
shape, along with `position_ids == (valid_length,)`.

Tokens are **teacher-forced from the reference**: `run()` feeds
`self._reference.expected_token_id(step - 1)`, the reference's own greedy
choice, rather than the graph's argmax. A single disagreeing step therefore
diagnoses that step instead of compounding into every later one — the same
convention `TorchReferenceSource` uses internally, and the same one T11 uses.

The runner also refuses configurations the contract cannot honour: `steps < 1`
and `prompt_length + steps > capacity` both raise `OnnxCpuError` before
anything executes (`test_runner_rejects_impossible_configurations`).

## Distinguishing a numerical failure from a state-update failure

This is the intellectual core of T21. The two failures look nearly identical in
a log — "the logits are wrong at step 3" — and have completely different fixes:

- **`numerical_tolerance`**: the graph is wired correctly, but FP16 storage and
  a different backend's accumulation order moved the logits outside the written
  tolerance. The fix is precision work or retolerancing.
- **`cache_state_update`**: the decode graph wrote the wrong slot, lost the
  valid prefix, disturbed the reserved tail, or reported the wrong
  `valid_length`. The fix is a corrected export. **Retolerancing this is
  malpractice** — it converts a correctness defect into a documented allowance.
- **`non_finite_logits`**: the graph emitted NaN or Inf logits. This is a third
  diagnosis, not a wider tolerance: nothing is "outside" anything, an FP16
  export overflowed or divided by zero, and no metric computed from those
  values would mean anything. It is the logit-side counterpart of the
  cache-side `slot_finite` invariant, and it exists so that the most likely
  real ONNX Runtime numerical failure is classified rather than fatal.

`FailureKind` names five classes:

| Kind | Raised when | Fix domain |
|---|---|---|
| `numerical_tolerance` | any step's `LogitParityMetrics.passed` is false | precision / tolerance |
| `non_finite_logits` | any step's **candidate** logits contain NaN or Inf; no metric is computed for that step and `steps[].metrics` is `null` | precision / export overflow |
| `cache_state_update` | any `CacheInvariantViolation`, per-step or whole-run | export correctness |
| `contract_violation` | declared output names, count, dtype, or shape disagree with the T12 contract | export / contract |
| `runtime_error` | `session.run` or `session.get_outputs` raises | environment / runtime |

A non-finite **reference** logit is not in this table: it is a configuration
error, not a graph behaviour, and it exits `2` (see "The `allclose`
convention").

### `failures[]` is ordered, and the order is the diagnosis

`failures[]` lists, in this order: any `contract_violation` or `runtime_error`
that aborted the loop, then every `cache_state_update`, then every
`non_finite_logits`, then every `numerical_tolerance`. The realistic combined
case is a wrong cache read that also moves the logits; listing the tolerance
failure first would put the wrong class under the operator's eye and invite the
retolerancing the previous paragraph calls malpractice. `failures[0]` is
therefore always the most fundamental class that fired. The same order is used
by the `_parity_failure_message` diagnostic. The whole order is pinned by
`test_all_three_fault_classes_are_reported_in_the_documented_order`, which
drives one run into all three classes — a cache fault on every step, non-finite
logits at step 1, out-of-tolerance logits at step 2 — and asserts each adjacent
pair. Two narrower tests pin one pair each:
`test_simultaneous_faults_report_both_classes` (state before tolerance) and
`test_a_state_fault_is_reported_before_a_non_finite_logit_fault` (state before
non-finite). Neither of those two produces a `non_finite_logits` *and* a
`numerical_tolerance` failure, so before the three-class test that pair was
unpinned and a refactor that swapped the two loops would have shipped green.

### The structural guarantee that neither class masks the other

Three properties, all traceable to `OrtCpuParityRunner.run()`:

1. **Independent code paths.** Logit metrics come from `compare_logits`; cache
   invariants come from `compare_cache_tensor`, `check_prefill_cache_tensor`,
   and `_check_slot_immutability`. Neither reads the other's result.
2. **Both always run.** The per-step loop appends a `ParityStepRecord` *and* a
   `CacheStepReport` for every step. Failure classification happens afterwards,
   in four separate passes over `cache_steps`, `slot_violations`, and
   `records` (twice: non-finite, then out-of-tolerance). There is no early
   return that skips one because the other failed — in particular, NaN or Inf
   candidate logits are recorded as a `non_finite_logits` failure and the loop
   continues, so a graph that overflows its logits at step 1 still has its
   cache invariants checked at every later step.
3. **Every class that fired is reported.** `ParityEvidence.failures` is a
   tuple, not a single value, and `failure_kinds` is the sorted set of distinct
   kinds. `passed` is `not self.failures` — one failure of any class fails the
   run, but the record still names all of them.

The one exception is deliberate and visible: a `_ClassifiedFault`
(`contract_violation` or `runtime_error`) aborts the step loop, because a
session that returns the wrong tensors cannot be meaningfully compared. The
steps completed before the abort keep their records and cache reports, and
`_check_slot_immutability` still runs on whatever final cache exists.

### The fault-injection evidence

These are the tests that make the distinction real rather than aspirational.
All use `FakeDecodeSession`, which is faithful by default — every fault mode is
an explicit injection through a named keyword argument.

| Test | Injected fault | Asserted outcome |
|---|---|---|
| `test_writing_the_previous_slot_is_a_state_fault_not_a_tolerance_fault` | `write_offset=-1` — writes slot `valid_length - 1` instead of `valid_length`; logits untouched | `failure_kinds == ("cache_state_update",)`; **every** step's logit metrics still pass; invariants `prefix_preserved` **and** `slot_written` both fire; the prefix violation is located at layer 0, position 127 |
| `test_prefix_corruption_at_step_three_is_caught_after_two_clean_steps` | one element of `present_key.0` at cache position 5 perturbed, on step 3 only | steps 1 and 2 `passed is True`, step 3 `passed is False`; `failure_kinds == ("cache_state_update",)`; all logit metrics pass; violation located at `present_key.0`, layer 0, position 5 |
| `test_missing_valid_length_increment_is_a_state_fault` | `valid_length_increment=0` | `failure_kinds == ("cache_state_update",)`; all logit metrics pass; `valid_length_increment` violation on `updated_valid_length` at position 128 |
| `test_writing_past_the_current_slot_is_a_tail_fault` | `tail_write_offset=2` — an extra write into the reserved tail | `failure_kinds == ("cache_state_update",)`; all logit metrics pass; `tail_untouched` violation at layer 0, position 130 |
| `test_logit_fault_with_a_perfect_cache_is_a_tolerance_fault_only` | `logit_bias=5.0` on step 2 only; cache handling perfect | `failure_kinds == ("numerical_tolerance",)`; `cache_report.passed is True`; the only failing step is step 2 |
| `test_simultaneous_faults_report_both_classes` | `logit_bias=5.0` on steps 1–3 **and** `write_offset=-1` | `failure_kinds == {"numerical_tolerance", "cache_state_update"}`; `cache_report.passed is False`; no decode step's logit metrics pass; all four cache reports (prefill + 3 decode) present, so neither computation was skipped |
| `test_a_slot_that_changes_later_breaks_written_slot_immutability` | `write_offset=-1` over three steps | exactly 8 `written_slot_immutable` violations, at positions 128 and 129, across all four cache tensors of both layers; each reported with `step is None` |
| `test_a_non_finite_value_in_the_new_slot_is_a_state_fault` | the newly written slot filled with `inf`, `-inf`, or `nan` | `failure_kinds == ("cache_state_update",)`; all logit metrics pass; `slot_finite` violation located at layer 0, position 128 |
| `test_a_dirty_prefill_reserve_is_a_state_fault` | **prefill** emits a `key_cache.0` whose reserve `[128, 160)` is not exactly zero | `failure_kinds == ("cache_state_update",)`; the violation is on the step-0 prefill report as `prefill_reserve_zero`, at layer 0, position 128 |
| `test_a_wrong_prefill_valid_length_is_a_state_fault` | **prefill** reports `valid_length = 127` for a 128-token prompt | `failure_kinds == ("cache_state_update",)`; `prefill_valid_length` violation on `valid_length` |
| `test_a_reported_valid_length_outside_capacity_is_caught_on_the_next_step` | decode reports `updated_valid_length = 160` at capacity 160 | `failure_kinds == ("cache_state_update",)`; all logit metrics pass; step 2 reports `write_index_within_capacity` at position 160 alongside `valid_length_increment`, and nothing else |
| `test_non_finite_candidate_logits_are_classified_not_fatal` | every decode step returns all-`nan`, all-`inf`, or all-`-inf` `next_logits`; cache handling perfect | `failure_kinds == ("non_finite_logits",)`; the run completes; `cache_report.passed is True` with all four reports present; each decode step has `metrics is None`, a full non-finite count, and a recorded candidate digest |
| `test_a_single_non_finite_candidate_logit_is_counted_exactly` | one `nan` in an otherwise clean logit vector | `failure_kinds == ("non_finite_logits",)`; `non_finite_candidate_logits == 1` |
| `test_a_state_fault_is_reported_before_a_non_finite_logit_fault` | `write_offset=-1` **and** all-`inf` logits | both classes present; `failures[0]` is `cache_state_update` and the last failure is `non_finite_logits` |
| `test_cli_exits_one_on_non_finite_logits_not_two` | all-`nan` `next_logits` through `main()` | exit code `1`, not `2`; the written evidence carries `failure_kinds == ["non_finite_logits"]`, a passing cache report, and `steps[1].metrics is null` |
| `test_non_finite_reference_logits_stay_a_configuration_error` | a `nan` in the **reference** logits | `ParityInputError` naming the step; the golden side is an input, so this is a configuration error and not a classified graph failure |
| `test_all_three_fault_classes_are_reported_in_the_documented_order` | `write_offset=-1` on every step, all-`nan` logits on step 1, `logit_bias=5.0` on step 2 — the only test that drives one run into all three classes | all three kinds present; `failures[0]` is `cache_state_update`; the last state failure precedes the first non-finite one, which precedes the first tolerance one; the logit-side sequence is exactly `[("non_finite_logits", 1), ("numerical_tolerance", 2)]` |

The first four prove a state fault is *never* misfiled as a tolerance fault:
the logits are pristine in every one of them, so a logit-only check would have
reported success. The fifth proves the converse. The sixth proves neither
masks the other when both are present — the decisive case, because a naive
implementation that short-circuits on the first failure would report only one.
The next five cover the remaining named invariants, including the two that live
on the **prefill** side of the T12 contract and are injected through a faulty
prefill session rather than a faulty decode session. The final five cover the
`non_finite_logits` class: that it is classified rather than fatal, that it is
counted exactly, that it is ordered after the state faults, that it exits `1`
through the CLI, and that the reference side is treated as a configuration
error instead. The seventeenth is the one that ties the taxonomy together: it
drives a single run into all three classes at once and pins the full documented
order of `failures[]`, which no other test does.

The four `contract_violation` tests
(`test_wrong_output_shape_is_a_contract_violation`,
`test_wrong_output_dtype_is_a_contract_violation`,
`test_missing_output_name_is_a_contract_violation`,
`test_wrong_output_count_is_a_contract_violation`) and
`test_session_run_failure_is_a_runtime_error` complete the taxonomy: a graph
that does not match the T12 contract is diagnosed as a contract problem, never
as bad numbers.

## Evidence tiers, and why a fake run cannot lie

`EvidenceTier` has exactly two members: `real_onnxruntime_cpu` and
`fake_session_self_test`. The tier is **derived, never declared**.
`detect_evidence_tier(sessions)` returns `fake_session_self_test` unless
`onnxruntime` imports, exposes an `InferenceSession` that is a real `type`, and
**every** session satisfies `issubclass(type(session), real)`. It uses
`issubclass(type(x), …)` rather than `isinstance(x, …)` precisely because
`isinstance` consults `x.__class__`, which a Python object can lie about with a
`__class__` property.

`test_evidence_tier_cannot_be_forged` demonstrates this with a `Liar` class
whose `__class__` property returns a fake `InferenceSession`: `isinstance(liar,
InferenceSession)` is `True`, and `detect_evidence_tier([liar])` is still
`FAKE_SESSION_SELF_TEST`. It also checks that a mixed list of one genuine
session and one liar degrades to the fake tier — the tier is the minimum across
all sessions, not the maximum.

The same test then closes the caller-side routes by introspection:
`evidence_tier` appears in neither `OrtCpuParityRunner.__init__`'s signature,
nor `OrtCpuParityRunner.run`'s, nor the `dest` set of any argument in
`build_parser()`. There is no flag, no constructor argument, and no environment
variable that can promote a run to the real tier. The only way to obtain
`evidence_tier="real_onnxruntime_cpu"` is to have really run ONNX Runtime.

`runtime_record` completes the picture by recording what actually loaded rather
than what was configured: `platform.python_version()`,
`platform.platform()`, the live `onnxruntime.__version__` (or `null` when the
import fails), each session's `get_providers()` result, and — under
`runtime.session_settings` — the graph optimization level, both thread counts,
and the execution mode as read back from each session's own
`get_session_options()`. That read-back is the point: the factory's arguments
are a request, and ONNX Runtime is free to normalize or override them, so the
evidence records the configuration that executed. A session exposing no options
(every fake) records `null` rather than a guess. Because the runtime block is
inside `digest_payload()`, two runs of the same graphs that differ only in
optimization level produce different `evidence_sha256` values, which is what
makes the `ORT_DISABLE_ALL` / `ORT_ENABLE_ALL` fusion-delta experiment below a
comparison of two distinguishable records rather than two indistinguishable
ones. Asserted by `test_runtime_record_carries_the_applied_session_settings` and
`test_the_optimization_level_changes_the_evidence_digest`.

The tier itself is validated as well as derived: `ParityEvidence.__post_init__`
rejects any `evidence_tier` that is not a member of `EvidenceTier`, so a
hand-constructed record cannot carry an invented tier such as
`"real_onnxruntime_cpu_honest"` (`test_an_unknown_evidence_tier_is_rejected_at_construction`).
That is a guard against a typo or an invented string, not against a code owner
writing a *valid* tier onto a record that was never measured; only
`OrtCpuParityRunner.run()` derives the tier from live sessions.

## What was verified in this environment

Commands run on 2026-08-02 in the `task/T21-ort-cpu-parity` worktree, using the
locked root interpreter:

```bash
PYTHONPATH=src python -m pytest tests/onnx/test_onnx_cpu_parity.py -v
```

Result: **61 passed, 1 skipped** (62 collected), 1.51 s, Python 3.11.13,
pytest 8.3.5.

The single skip is `test_real_onnxruntime_cpu_parity_when_available`, which
begins with `pytest.importorskip("onnxruntime")` and
`pytest.importorskip("torch")`. It skips here because neither is installed.
That test is the guarded real-runtime path: on a host that has the runtime, the
manifest, and `SLM_LAB_ARTIFACT_ROOT`, it verifies the graph digests, builds
real sessions, runs a two-step parity check, and asserts
`evidence_tier == "real_onnxruntime_cpu"` alongside `cache_report.passed` and
`evidence.passed`. That last assertion is the acceptance criterion itself —
without it the one path that can produce a measurement would go green with
logits arbitrarily far outside the tolerance. Both assertions report through
`_parity_failure_message`, which names every failure class that fired and prints
each failing step's worst metric next to the threshold it missed, so a red run
on a parity host says immediately whether it is a tolerance problem or a state
problem. That helper is itself exercised here, on a fake run with both faults
injected (`test_the_parity_failure_message_names_both_classes`), because it
would otherwise only ever execute on a host this one is not. The guarded test is
the only test in the file that can produce a measurement, and it did not run.

Verified here:

- Dependency absence, checked directly with `importlib.util.find_spec`:
  `onnxruntime`, `torch`, `numpy`, `onnx`, and `transformers` are **all
  absent**.
- `python -m slm_lab.backends.onnx_cpu --help` runs and exits `0` with none of
  those installed. The module imports nothing heavy at import time; ONNX
  Runtime, numpy, and torch are imported lazily inside
  `_require_onnxruntime`, `numpy_tensor_factory`, and
  `TorchReferenceSource._materialize` respectively.
- The pure-Python logit metrics against hand-computed values, including the
  protected-relative floor, tie-breaking, top-5 overlap, top-1/top-2 margin,
  the `allclose` boundary, the `rtol`-scales-the-candidate convention, the
  cosine denominator floor on a degenerate pair, the absence of an output clamp
  on the cosine, and the norm-magnitude-asymmetry case that separates a
  product floor from a per-norm floor.
- The array adapter's rejection of ragged nesting, non-numeric leaves, scalars,
  and objects without `.tolist()`.
- A clean four-step run over the reduced-shape contracts, and a one-step run
  over the **real** S128 T12 contracts (28 layers, 8 KV heads, head_dim 128,
  vocab 151,936, capacity 160) with fake sessions: `variant_id == "S128"`,
  `cache_capacity == 160`, 56 cache tensors checked per decode step.
- All seventeen fault-injection scenarios tabulated above, which between them
  exercise every one of the nine named invariants — including the two prefill
  invariants, injected through a faulty **prefill** session — and the whole
  `non_finite_logits` class, including its CLI exit code and its position in
  `failures[]`.
- All four contract-violation classes and the runtime-error class.
- Evidence determinism: two independent runs produce byte-identical JSON and
  identical `evidence_sha256`, and a 0.01 logit perturbation changes the
  digest.
- Evidence-tier forgery resistance, including the `__class__`-lying case, and
  rejection of an evidence tier that is not an `EvidenceTier` member.
- That `runtime.session_settings` carries the level, thread counts, and
  execution mode each session reports, and that changing only the optimization
  level changes `evidence_sha256`.
- That a T20 manifest cannot point the runner outside the artifact root: an
  absolute or `..`-bearing `relative_path` is rejected before any file is
  opened, matching the guard on the inspection side.
- CLI behaviour: default `--steps 4`, default `--reference torch`, default
  `--graph-optimization-level ORT_DISABLE_ALL`; digest mismatch exits `2`
  **without constructing a session**; a missing graph file exits `2` with no
  traceback; a missing `onnxruntime` exits `2` naming
  `environments/onnx-cpu/README.md`; and an end-to-end CLI run with injected
  fakes writes an evidence file whose `evidence_tier` is
  `fake_session_self_test`.

## What could not be verified in this environment

- Any parity number whatsoever: no `max_absolute_error`, no
  `cosine_similarity`, no `top1_agreement`, no `top5_overlap`.
- Whether the proposed tolerances are correct, too tight, or too loose.
- Whether the real T20 decode graph actually honours the T12 cache contract at
  runtime. T20's own worklog flags this precisely: it verified export structure
  and content identity, *not* numerical runtime behaviour, and asked T21 to
  check whether `valid_length` remains a live internal slice/scatter dependency
  rather than a traced constant. **That question is still open.** The machinery
  to answer it exists and is tested; it has not been pointed at the real graph.
- Whether ONNX Runtime loads an opset-18 model with a 1.19 GB external-data
  file on the CPU execution provider at all.
- Whether the reference model loads, and whether its BF16 logits match T11's
  recorded values on the parity host.
- Any behaviour of `TorchReferenceSource._materialize`, `numpy_tensor_factory`,
  or `onnxruntime_cpu_session_factory`'s inner `factory` — all three are marked
  `# pragma: no cover` because they cannot execute here.

## Reproducing the real measurement

Build the separate parity environment first. ONNX Runtime is deliberately
**not** in the locked root environment: `pyproject.toml` and `uv.lock` pin a
cross-platform environment and every documented setup command uses `--locked`,
so adding an ONNX Runtime extra would invalidate that lock for every host,
including hosts that will never run a parity job. T50 keeps MLX out of the root
environment for the same reason. There is **no `uv sync --extra onnx-cpu`** —
that extra does not exist.

Per `environments/onnx-cpu/README.md`:

```bash
UV_CACHE_DIR=/tmp/slm-lab-ort-cache \
  uv venv --python 3.11.13 .ai-local/envs/t21-ort-cpu
UV_CACHE_DIR=/tmp/slm-lab-ort-cache \
  uv pip install --python .ai-local/envs/t21-ort-cpu/bin/python \
  torch==2.7.1 transformers==4.51.3 onnx==1.18.0 \
  "onnxruntime==<smoke-tested-version>" "numpy==<smoke-tested-version>"
```

`torch` 2.7.1, `transformers` 4.51.3, and `onnx` 1.18.0 are already pinned by
T20 in `configs/models/qwen3-0.6b-onnx-export.json`. The `onnxruntime` and
`numpy` versions are **not yet chosen**: `environments/README.md` requires a
platform task to pin exact versions *after* a compatibility smoke test, no
ONNX Runtime build has been smoke-tested in this repository, and the guide
correctly declines to invent one. The task that first executes a real parity
run owns those pins. `latest` is not an acceptable version.

Then run the measurement:

```bash
SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab \
HF_HOME=<local-hf-cache> TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  .ai-local/envs/t21-ort-cpu/bin/python -m slm_lab.backends.onnx_cpu \
  --manifest results/manifests/onnx/S128.json \
  --steps 4 \
  --reference torch \
  --output results/graph/parity/S128-ort-cpu.json
```

Environment variables, in order: `SLM_LAB_ARTIFACT_ROOT` locates
`onnx/reference/T20/` (the graphs and their external-data sidecars are not in
Git — one 1,192,085,504-byte `.onnx.data` file per graph, eight in all, so the
root needs about 8.9 GB for the set even though all eight are byte-identical and
share the single SHA-256 recorded in the manifests); `HF_HOME` points at a local
Hugging Face cache;
`TRANSFORMERS_OFFLINE=1` forbids any network fetch of the reference model —
the runner defaults to `local_files_only=True` and only `--allow-download`
relaxes it; `PYTHONPATH=src` makes `slm_lab` importable without installation.

Before any session is constructed, `verified_graph_paths` resolves
`S128/prefill.onnx` and `S128/decode.onnx` under
`<artifact-root>/onnx/reference/T20/` and compares each file's SHA-256 against
the digest committed in `results/manifests/onnx/S128.json`
(`a61ed2ef1e3f1ef9313f33ee13ab5af5dc79029291afbd328cb0aaaea470dfd1` for
prefill, `e200ecd27e1ab83d2bea17de030c0a0c8a0eea08c6f182eed41c04a457c421d2` for
decode). A mismatch or a missing file aborts with exit `2` and no session is
created — asserted by `test_cli_digest_mismatch_exits_without_creating_a_session`.

Sessions are pinned for determinism, not speed:
`intra_op_num_threads=1`, `inter_op_num_threads=1`,
`execution_mode=ORT_SEQUENTIAL`, `providers=["CPUExecutionProvider"]`, and
`graph_optimization_level="ORT_DISABLE_ALL"` by default. Disabling
optimizations first means the initial number measures *the exported graph*
rather than ONNX Runtime's fusion choices; a second run with
`--graph-optimization-level ORT_ENABLE_ALL` then isolates the fusion delta as a
separate, named experiment. The level, both thread counts, and the execution
mode are applied to the `SessionOptions` and then read back off each constructed
session into `runtime.session_settings` in the evidence, so the two runs are
distinguishable by their recorded configuration and by their `evidence_sha256`.

### Exit codes

| Code | Meaning |
|---:|---|
| `0` | parity passed — `ParityEvidence.failures` is empty |
| `1` | parity failed — read `failures[]` in the evidence and classify before acting |
| `2` | configuration or dependency error — bad manifest, missing/mismatched graph, missing artifact root, absent `onnxruntime`, non-finite **reference** logits, or a contract error. Printed as `error: …` on stderr with no traceback. |

Anything the graph itself did wrong — including NaN or Inf logits — is exit `1`
with a classified `failures[]`, never exit `2`.

### The evidence record the run produces

Written to the `--output` path (stdout if omitted), as sorted-key JSON with
`allow_nan=False`. Fields, from `ParityEvidence.digest_payload()`:

| Field | Content |
|---|---|
| `schema_version`, `task_id` | `1`, `"T21"` |
| `evidence_tier` | derived from the session objects; `real_onnxruntime_cpu` only for genuine sessions |
| `variant_id`, `prompt_length`, `cache_capacity`, `steps_requested` | `"S128"`, `128`, `160`, and the `--steps` value |
| `graph_digests` | per graph kind: the verified `sha256` and the manifest `relative_path` (e.g. `S128/prefill.onnx`). Built by `graph_digests_payload`, which the CLI and the guarded real-runtime test both call, so the two produce comparable records. No absolute host path appears in the evidence: it is committed under `results/graph/parity/` and covered by `evidence_sha256`, so it must not depend on where the artifact root is mounted. |
| `runtime` | `python_version`, `platform`, live `onnxruntime_version`, per-session `providers`, and per-session `session_settings` (graph optimization level, both thread counts, execution mode) read back off the session |
| `tolerance` | every threshold plus the nested `cache_state` rules and the `status` string |
| `reference_provenance` | reference source, `model_id`, `model_revision`, runtime record, `teacher_forced`, prompt token count, prompt-token-IDs digest, per-step reference logit digests, and the expected token IDs |
| `steps[]` | per step: `step`, `graph_kind`, `input_token_id`, `input_valid_length`, `output_valid_length`, `reference_logits_sha256`, `candidate_logits_sha256`, `non_finite_candidate_logits`, and the full `metrics` block — `metrics` is `null` exactly when `non_finite_candidate_logits` is non-zero, because no honest metric exists for a NaN vector |
| `cache_report` | `passed`, per-step reports (with `write_index`, `tensors_checked`, and located `violations`), and `slot_immutability_violations` |
| `failures[]`, `failure_kinds`, `passed` | every classified failure, the distinct kinds, and the overall verdict |
| `evidence_sha256` | SHA-256 over canonical JSON of everything above |

Full logit vectors are **not** committed — only their SHA-256 digests, computed
over little-endian float64 by `values_sha256`. The digest payload contains no
timestamps and no host-dependent paths at all, so two runs of the same graphs
produce the same digest even from different artifact roots; a 0.01 change in one
logit changes it.

## Evidence boundaries

None of the work described here establishes:

- **Any parity result.** No logit error, no cosine similarity, no top-1
  agreement, no top-5 overlap has been measured against ONNX Runtime.
- **That the proposed tolerances are correct.** They are a hypothesis recorded
  in code and serialized as `proposed_unvalidated`.
- **That the real T20 decode graph updates the cache correctly.** The
  invariants are defined and their detection is tested against injected faults;
  the real graph has not been checked against them.
- **That the newly written cache slot holds the right values.** T21 checks only
  that the slot was written and is finite. Verifying its contents needs a
  reference cache comparison, which is not implemented. Relatedly,
  `slot_written` is a change detector: a graph writing exactly the bytes
  already present would be indistinguishable from one writing nothing.
- **Any performance claim.** No latency, throughput, memory footprint, or
  memory-bandwidth number. The single-threaded, optimization-disabled session
  configuration is chosen for determinism and would be a poor performance
  configuration.
- **Anything about specific ONNX Runtime versions.** No `onnxruntime` build has
  been installed, run, or smoke-tested in this repository. No claim is made
  about which versions load an opset-18 model with external data, or about any
  execution provider's behaviour.
- **Anything about accelerators.** Nothing about CUDA, QNN/QAIRT, Hexagon NPU,
  Adreno GPU, MLX, or Apple Neural Engine. The CPU execution provider is the
  only target contemplated, and it has not been exercised.
- **Compiler acceptance or hardware placement.** `results/manifests/onnx/S128.json`
  already records these under `claim_boundary.does_not_establish`, alongside
  `onnxruntime_numerical_parity` — which this report does not move.
- **Anything about S512, S1024, or S4096.** The recorded command targets S128
  only. The other three variants have committed manifests but no parity route
  has been exercised for them.

## Learner checkpoint

- [ ] Explain why the T21 tolerance is a *backend-parity* tolerance under
  `docs/project/plan.md` §6.7, and name the two error sources it admits that
  T11's tolerance does not. Then explain why T21's logit thresholds are
  nonetheless numerically identical to T11's, and why that is a hypothesis
  rather than a conclusion.
- [ ] `rtol` scales the candidate, not the reference. Construct a pair of logit
  vectors for which `compare_logits(a, b)` passes `allclose` and
  `compare_logits(b, a)` fails, and say which of the two is the ONNX Runtime
  output.
- [ ] A decode graph reads cache row `valid_length - 1` instead of
  `valid_length` but writes the correct slot. Which invariants fire, at which
  step, and would a step-1-only parity check have caught it? Now answer the
  same question for a graph that *writes* the wrong slot.
- [ ] Argue for and then against making `prefix_preserved` a tolerant check
  with `atol=1e-3`. What class of defect would become invisible, and what would
  it be misdiagnosed as?
- [ ] `detect_evidence_tier` uses `issubclass(type(session), real)` rather than
  `isinstance(session, real)`. Explain the attack the second form permits, and
  why the tier is the minimum over all sessions rather than the maximum.
- [ ] The first real run passes with `max_absolute_error = 0.03`. State what
  must change in `DEFAULT_ORT_CPU_TOLERANCE` and `TOLERANCE_STATUS`, and
  explain why leaving `atol=0.25` in place would leave the criterion
  unfalsified.
- [ ] The first real run fails at step 3 with both `numerical_tolerance` and
  `cache_state_update` in `failure_kinds`. Which do you investigate first, and
  why is retolerancing not an available response? Then say how `failures[]`
  answers the "which first" question for you before you have read a single
  metric.
- [ ] The graph returns `inf` logits at step 1 while its cache stays perfect.
  Explain why that is `non_finite_logits` rather than `numerical_tolerance`,
  why the run continues to step 4 instead of aborting, and why the
  corresponding cache-side invariant (`slot_finite`) has existed since the
  first draft while the logit-side class did not.
