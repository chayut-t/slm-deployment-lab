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

## `w8/` (T41)

Two generated files, `w8a16.yaml` and `w8a8.yaml`, freezing the plan's Q1 and
Q2 eight-bit-weight candidates. Do not hand-edit; regenerate with

```bash
uv run python -m slm_lab.quantization.w8 generate
```

and verify with

```bash
uv run python -m slm_lab.quantization.w8 check
```

They are specifications, not results. No weight was quantized to produce them
and nothing in them was measured: every number is arithmetic over committed
inputs (labelled `analytic_projection` where it is a projection) or a hash read
off a committed file. `candidate.precision_state` may only ever read
`specified` in a committed file, because `simulated` and `deployed` are
evidence states and evidence does not live in configuration.

Each candidate binds this directory's `calibration.yaml` revision, the four
committed float16 manifests under `results/manifests/onnx/`, the frozen T13
protocol digest, and the T21 graph inventory, and `check` fails rather than
re-anchoring if any of them moves. `w8/README.md` explains how to read a
candidate top to bottom and which fields are T41 policy choices versus hard
constraints of the frozen graph — including the one place they collide, where
plan row Q2 asks for an INT8 KV cache that the frozen T12 contract does not
carry. The reader-facing report is `docs/results/quantization/w8.md`.
