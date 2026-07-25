# Workload configurations

Store frozen prompts, token counts, context variants, generation settings,
warm-up, repetitions, and evaluation subsets here.

## T10 fixture contract

`t10-token-fixtures.json` is the small entrypoint for the frozen Qwen3-0.6B
inputs. It records:

- The immutable tokenizer revision and chat-template hash inherited from T00.
- Exact static contexts `128`, `512`, `1024`, and `4096`.
- Generated-token counts `32`, `64`, `128`, and `128`.
- Greedy/no-sampling generation, tie-breaking, seed, EOS/PAD, and output-limit
  semantics shared by later backends.
- Canonical SHA-256 values for authored source prompts and generated token
  fixtures.
- IDs for raw-completion, non-thinking chat, and CC0 quality canaries.

The full token arrays live in
`../../tests/fixtures/t10/token-fixtures-v1.json`. They are committed because
they are small, deterministic test vectors rather than model artifacts.

Validate hashes and structure without downloading anything:

```bash
uv run slm-lab-fixtures check
```

Re-encode every prompt with the pinned public tokenizer:

```bash
uv sync --extra tokenizer --locked
uv run --extra tokenizer slm-lab-fixtures verify
```

Regeneration uses the same exact dependency and tokenizer revisions:

```bash
uv run --extra tokenizer slm-lab-fixtures generate
git diff --exit-code -- configs/workloads/ tests/fixtures/t10/
```

No model weights are needed for these commands.

## T13 benchmark and evaluation contract

`benchmark-protocol-v1.json` freezes the shared context matrix, decode probes,
timing classes, synchronization rules, statistics, memory/power methods,
numerical metrics, and claim boundaries. `benchmark-result-v1.schema.json`
defines one raw-sample-backed metric record. A graph-latency record and an
end-to-end generation record therefore cannot be silently combined.

`academic-evaluation-v1.json` pins the small WikiText-2 raw, HellaSwag, and ARC
Easy regression subset. It commits metadata, not third-party rows or scores.
Dataset caches and sample logs remain external or ignored.

Validate all three contracts offline:

```bash
uv run python -m slm_lab.benchmark.protocol check --root .
```
