# Evaluation

Numerical canaries, perplexity, small academic benchmark subsets, and
precision-aware comparison logic belongs here.

`fixtures.py` implements the T10 reproducibility boundary:

- Canonical JSON and token-ID hashing.
- Exact-length prompt construction from authored CC0 material.
- Raw-completion and explicitly non-thinking chat canaries.
- Structural checks that do not need network access.
- Optional exact regeneration with the immutable Qwen tokenizer.

T10 intentionally does not commit WikiText, HellaSwag, ARC, or PIQA rows.
Those external candidates are named for T13, which owns revision pinning and
the final academic subset. This module only commits repository-authored CC0
quality cases and their token IDs.
