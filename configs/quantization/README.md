# Quantization configurations

Store calibration, W8, W4, LPBQ, LiteMP, and mixed-precision policies with
explicit simulated-versus-deployed semantics.

## `calibration.yaml` (T40)

Generated file. Do not hand-edit; regenerate with

```bash
uv run python -m slm_lab.quantization.calibration generate
```

and verify with

```bash
uv run python -m slm_lab.quantization.calibration check
```

It is the frozen contract for the *inputs* to quantization. It records the
model and tokenizer revisions, the canonical hashes of every committed input
it derives from, the deterministic preprocessing contract, the tier-1 sample
table with a per-sample selection rationale, the corpus token budget, the
declared-but-never-committed tier-2 candidates with their real Hugging Face
revisions and card licences, and the licensing boundary that keeps this
repository Apache-2.0 clean.

Read it top to bottom to answer the T40 study checkpoint: *why does this
corpus represent the target workloads?* Every selection argues for itself in
its `rationale` field — but read `calibration_corpus.coverage` first. That
block is computed from the emitted token IDs at generation time and bounds what
the rationales are allowed to mean: how many distinct token IDs the corpus
touches, what fraction of the vocabulary that is, how the token budget splits
across source groups, and the fact that the four context workloads are nested
token-ID prefixes of one repeated T10 seed rather than four independent bodies
of text. A `rationale` is a qualitative argument about token classes and
prefill shapes; it is never a coverage measurement.

`calibration_dataset_revision` is the value every T41+ artifact manifest must
carry. A quantized result whose manifest does not match the value this file
regenerates was calibrated on a different corpus and is not comparable.
