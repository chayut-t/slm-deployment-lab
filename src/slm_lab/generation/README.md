# Generation

`reference.py` provides three correctness surfaces:

- full-forward greedy generation, which recomputes the complete prefix;
- cached generation, which prefills once and decodes one token at a time; and
- lockstep parity evidence with absolute/relative error, cosine similarity,
  top-k agreement, margins, compact logit fingerprints, and exact token IDs.

Lockstep comparison teacher-forces the full-forward token into cached decode
after recording any token mismatch. Later steps therefore remain on the same
prefix instead of conflating cache error with autoregressive divergence.
Every cached prefill and decode call must return cache state or execution fails
with the exact missing-cache boundary.

The T10 policy is preserved: no sampling, lowest-token-ID argmax tie breaking,
EOS included in output, and a fixed maximum-new-token limit. A pinned Qwen run
is available without introducing a console-script dependency:

```bash
python -m slm_lab.generation.reference \
  --fixture raw_ascii --max-new-tokens 8 --device cpu --dtype bfloat16
```

Add `--allow-download` only when the immutable public weights are not already
in the external model cache.
