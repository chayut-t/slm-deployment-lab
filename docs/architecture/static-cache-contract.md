# Qwen3-0.6B static prefill, decode, and KV-cache contract

Status: frozen by T12

Model: `Qwen/Qwen3-0.6B`

Revision: `c1899de289a04d12100db370d81485cdf75e47ca`

## Contract boundary

The machine-readable authority is
`src/slm_lab/contracts/static_cache.py`. It generates every variant from one
definition. Later exporters and runtimes must consume that definition or prove
that their concrete tensors conform to it; they must not duplicate four
hand-written interfaces.

The contract is a correctness-first deployment boundary:

- batch size is fixed at one;
- prompt lengths are `128`, `512`, `1024`, and `4096`;
- cache capacity reserves each T10 workload's complete generation budget;
- cache state is explicit, per layer, and head-major;
- prefill and decode graph inputs and outputs have no dynamic dimensions; and
- one decode call performs one indexed cache update.

## Why capacity is larger than prompt length

The project plan permits `C = S` or a documented larger fixed capacity. A
prefill that writes exactly `S` positions into a capacity `C = S` leaves no
legal position for the first decode token. T12 therefore freezes:

| Variant | Prompt `S` | T10 output budget | Capacity `C` |
|---|---:|---:|---:|
| S128 | 128 | 32 | 160 |
| S512 | 512 | 64 | 576 |
| S1024 | 1,024 | 128 | 1,152 |
| S4096 | 4,096 | 128 | 4,224 |

The named context remains the exact prompt workload. Capacity is an allocation
property and must be reported separately in manifests and measurements.

## Prefill graph

```text
input_ids       int64    [1, S]  (batch, sequence)
attention_mask  int64    [1, S]  (batch, sequence)
position_ids    int64    [1, S]  (batch, sequence)
        │
        ▼
static prefill for S, with fixed cache capacity C
        │
        ├─ last_logits     float32  [1, 151936]
        ├─ key_cache.L     float16  [1, 8, C, 128]  L=0..27
        ├─ value_cache.L   float16  [1, 8, C, 128]  L=0..27
        └─ valid_length    int64    [1] = S
```

The prompt has no padding. `attention_mask` is all ones and `position_ids` is
`[0, 1, ..., S-1]`. Every cache tensor contains valid values in `[0, S)` and
zeros in `[S, C)`.

Serialized prefill metadata names this transition
`prefill_prefix_materialization`: `written_range` is
`[0, prompt_length)`, `zero_filled_range` is
`[prompt_length, cache_capacity)`, and `output_valid_length` is
`prompt_length`. It deliberately has no decode-only `write_index` or
`input_valid_range`.

`float16` is the frozen deployment cache boundary. T11 remains the
checkpoint-dtype BF16 numerical oracle. Crossing from T11 to this contract
therefore performs an explicit BF16-to-FP16 cache cast, and downstream parity
must retain T11's numerical comparison rather than imply byte identity.
Logits cross the graph boundary as `float32`.

## One-token decode graph

```text
input_ids       int64    [1, 1]
attention_mask  int64    [1, C]
position_ids    int64    [1, 1]
key_cache.L     float16  [1, 8, C, 128]  L=0..27
value_cache.L   float16  [1, 8, C, 128]  L=0..27
valid_length    int64    [1]
        │
        ▼
fixed-capacity indexed decode
        │
        ├─ next_logits          float32  [1, 151936]
        ├─ present_key.L        float16  [1, 8, C, 128]  L=0..27
        ├─ present_value.L      float16  [1, 8, C, 128]  L=0..27
        └─ updated_valid_length int64    [1] = valid_length + 1
```

For an incoming `valid_length = P`:

1. cache positions `[0, P)` contain prior K/V state;
2. `position_ids[0, 0] = P`;
3. `attention_mask[0, 0:P+1] = 1` and the remaining entries are zero;
4. each layer writes the new K/V slice at cache position `P`; and
5. the output valid length is `P + 1`.

Calling decode with `P >= C` is an error. No shifting, wraparound, eviction, or
silent truncation is allowed.

Serialized decode metadata names this transition
`fixed_capacity_indexed_copy`: the input valid range is
`[0, valid_length)`, the write index is `valid_length`, and the output valid
length is `valid_length + 1`.

## GQA layout and byte accounting

Qwen3-0.6B has 16 query heads but only 8 K/V heads. Cache state must preserve
those 8 physical K/V heads; materializing 16 repeated K/V heads would double
cache memory and traffic without adding information.

The per-layer layout is:

```text
[batch, kv_head, cache_position, head_dim]
[   1,       8,              C,      128]
```

For FP16, logical K+V bytes are:

```text
2 (K,V) × 28 layers × 8 KV heads × positions × 128 head dim × 2 bytes
= 114,688 bytes per valid or allocated position
= 112 KiB per position
```

| Prompt | Prompt-resident K+V | Capacity | Allocated K+V |
|---:|---:|---:|---:|
| 128 | 14 MiB | 160 | 17.5 MiB |
| 512 | 56 MiB | 576 | 63 MiB |
| 1,024 | 112 MiB | 1,152 | 126 MiB |
| 4,096 | 448 MiB | 4,224 | 462 MiB |

An out-of-place decode interface exposes both input and output buffers, so a
runtime that cannot alias or recycle them can temporarily require twice the
allocated cache bytes. That live-memory effect must be measured, not inferred
away from the logical table.

## Conformance and numerical validation

`validate_tensor_mapping` checks concrete name, dtype, and shape mappings.
`materialize_reference_cache` copies the growing T11/Transformers cache into
fixed FP16 buffers. `apply_decode_updates` clones those buffers, writes exactly
one `[1, 8, 1, 128]` slice per layer, rejects overflow, and advances the valid
length.

Contract tests assert the complete graph-kind-specific serialized transition
for both prefill and decode so a manifest cannot attach decode-only state
fields to prefill again.

The weightless PyTorch conformance test uses the same growing
`past_key_values` protocol as T11. Across multiple decode steps it verifies:

- all 28 layers use the GQA dimensions;
- the fixed valid prefix equals the PyTorch growing cache;
- earlier positions remain unchanged after each indexed update; and
- only the newly appended reference slice is written.

This proves cache state-transition equivalence. T20 and later runtime tasks
must additionally compare graph logits with the T11 oracle using the frozen
T11 tolerances.

## Learner debrief

- [ ] Draw both graph boundaries without consulting the diagrams.
- [ ] Explain why `C = S` cannot support a decode after a full-length prefill.
- [ ] Calculate the 112 MiB and 448 MiB prompt-resident figures from model
  dimensions.
- [ ] Calculate the S1024 and S4096 allocation figures including generation
  reserve.
- [ ] Explain why 8 K/V heads, rather than 16 query heads, determine cache
  bytes.
- [ ] Trace one update from `valid_length` to `updated_valid_length` and name
  the exact cache slice that changes.

Do not mark these study items complete without the learner's explicit
confirmation.
