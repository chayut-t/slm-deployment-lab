# Token and prompt fixtures as a deployment contract

Task: `T10`

## Why this exists

Deployment systems do not receive words; they receive integer token IDs. Two
runs that display the same prompt can still differ if they use another
tokenizer revision, chat template, thinking-mode setting, or implicit special
token. That difference can change logits, generated text, graph shapes,
prefill latency, cache occupancy, and every comparison made afterward.

T10 therefore treats tokenization like an application binary interface:

```text
authored prompt
  → immutable tokenizer revision
  → exact token IDs
  → static graph/runtime input
  → numerical and performance evidence
```

The committed entrypoint is
[`configs/workloads/t10-token-fixtures.json`](../../configs/workloads/t10-token-fixtures.json).
It links the T00 model contract to a hashed bundle of prompts and token IDs.

## The three fixture layers

### 1. Authored source

[`tests/fixtures/t10/source-prompts-v1.json`](../../tests/fixtures/t10/source-prompts-v1.json)
contains repository-authored CC0 prompts. It explicitly states that it contains
neither private material nor third-party dataset rows.

This source separates:

- Raw-completion canaries for punctuation, whitespace, Unicode, and structured
  text.
- One chat-shaped canary that calls Qwen's pinned template with
  `enable_thinking=false`.
- A deployment-focused seed used to construct context workloads.
- Small CC0 quality checks with reference answers.

### 2. Generated token bundle

[`tests/fixtures/t10/token-fixtures-v1.json`](../../tests/fixtures/t10/token-fixtures-v1.json)
stores the exact Qwen token IDs, prompt hashes, token-array hashes, package
versions, and tokenizer identity.

The 128, 512, 1,024, and 4,096-token prompts are not estimated by word count.
The generator repeats authored text, tokenizes it, truncates the integer
sequence, decodes that sequence, and re-encodes the resulting text. Generation
fails unless the round trip reproduces every ID exactly.

### 3. Workload manifest

The workload manifest maps each exact prompt to its future generation contract:

| Workload | Prompt tokens | Generated tokens | Role |
|---|---:|---:|---|
| S128 | 128 | 32 | Earliest compiler and latency feasibility |
| S512 | 512 | 64 | Interactive/cache-growth workload |
| S1024 | 1,024 | 128 | Main cross-platform comparison |
| S4096 | 4,096 | 128 | Long-context memory and compiler pressure |

Later backends must consume these IDs rather than inventing platform-specific
prompts.

## Why both text and IDs are committed

Text makes a fixture understandable; IDs make it executable and exact.
Committing both lets validation answer two different questions:

1. Did a person edit the prompt or token array? Canonical hashes detect drift.
2. Does the immutable upstream tokenizer still reproduce the bundle? The
   optional upstream verification re-encodes every record.

Raw-completion encoding uses `add_special_tokens=false`. Qwen's tokenizer does
not add a BOS token, while the model configuration still assigns token
`151643` a BOS meaning and the tokenizer uses the same ID for padding. The
fixture preserves that distinction rather than inferring behavior from the
model config.

Qwen's pinned template still emits an empty `<think>…</think>` control block
when thinking is disabled. The fixture records those tokens. Non-thinking mode
means the block contains no reasoning; it does not mean the template markers
disappear.

## Privacy and licensing boundary

The committed quality cases are small and authored for this repository under
CC0. External academic candidates—WikiText-2, HellaSwag, ARC Easy, and
PIQA—are named but their rows are not copied here. T13 must pin exact dataset
revisions, document licenses, and materialize permitted data outside Git.

This is why a hash alone is insufficient: reproducibility does not grant a
right to redistribute data.

## Hands-on study

Run the offline structural check:

```bash
uv run slm-lab-fixtures check
```

Then install the locked tokenizer-only environment and reproduce every token:

```bash
uv sync --extra tokenizer --locked
uv run --extra tokenizer slm-lab-fixtures verify
```

Inspect one short canary:

```bash
jq '.raw_canaries[0] |
  {id, prompt, token_count, token_ids, token_ids_sha256}' \
  tests/fixtures/t10/token-fixtures-v1.json
```

Compare context sizes:

```bash
jq '.context_workloads[] |
  {id, context_length, generated_tokens, token_ids_sha256}' \
  tests/fixtures/t10/token-fixtures-v1.json
```

## Questions to answer before marking your study complete

1. Why can changing only the tokenizer invalidate numerical and latency
   comparisons?
2. Why does T10 use raw completion for canonical validation but retain one chat
   canary?
3. How does the generator prove a prompt contains exactly 4,096 tokens?
4. Why are external benchmark names committed while their dataset rows are
   not?
5. Which later tasks depend on these fixtures, and what kind of drift would
   each detect?
