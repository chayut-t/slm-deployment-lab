# ORT CPU parity diagnostics — not T21 parity records

Everything in this directory exists to justify or check
`DEFAULT_ORT_CPU_TOLERANCE` in `src/slm_lab/backends/onnx_cpu.py`. **None of it
is a T21 parity measurement.** The T21 records are the four files one level up:

```
results/graph/parity/S{128,512,1024,4096}-ort-cpu.json
```

Those compare the float16 ONNX graph against the pinned PyTorch reference at
its contract dtype, **bfloat16**, on the ONNX Runtime CPU provider at
`ORT_DISABLE_ALL`. That pairing is what `DEFAULT_ORT_CPU_TOLERANCE` is derived
for, and nothing here may be copied over them.

**Read `record_kind`, at the top level of every file here and of every record
one level up.** A T21 parity record carries `t21_ort_cpu_parity`; anything else
is a diagnostic whose `passed` is not a T21 verdict. The field is derived from
the reference's own recorded dtype, is covered by `evidence_sha256`, and the CLI
refuses to write a non-contract-dtype run to an `S<N>-ort-cpu.json` name at all.

## `S<N>-reference-dtype-self-error.json`

`record_kind: diagnostic_reference_dtype_self_error`. No ONNX Runtime session
is created and no graph is executed — this runs the **PyTorch reference against
itself** at float32, bfloat16 and float16.

It supplies the three inputs the tolerance derivation needs and cannot get from
a parity record:

- `lambda_max_abs_logit` — the logit scale an absolute tolerance binds at.
  Measured 19.25 … 30.89 over 20 steps, so every logit sits in the binade
  [16, 32) where bfloat16's ULP is 0.125 and float16's is 0.015625.
- `pairwise.float32_vs_*` — the irreducible error floor of the comparison,
  0.189 … 0.609 against the graph's own 0.189 … 0.578. The headline: at S512
  step 1, **float32 differs from the bfloat16 reference by 0.609**, more than
  the ONNX graph's 0.578. The superseded `atol=0.25` therefore rejected the
  exact answer.
- `consecutive_step_distance` — a mis-wiring reference scale. One decode step
  of extra context moves the logits by 13.29 … 30.44, drops cosine to
  0.034 … 0.951, drops top-5 overlap to 0.0 … 0.6 and flips top-1 on all 16
  pairs. A cache read landing one slot off is at least that visible.

`tolerance_verdict` rolls up the two-sided property a tolerance must have: every
reference-dtype pair passes (it accepts the exact answer) and every consecutive
step pair fails (it still rejects a cache offset).

Reproduce:

```
python -m slm_lab.backends.onnx_cpu --reference-self-error \
  --manifest results/manifests/onnx/S<N>.json --steps 4 \
  --output results/graph/parity/diagnostics/S<N>-reference-dtype-self-error.json
```

## `S128-ort-cpu-float16-reference-probe.json`

The same runner and schema as a T21 record, but with the reference model in
**float16** instead of bfloat16, so both sides of the comparison carry the same
11 significand bits. It says so structurally:
`record_kind: diagnostic_off_contract_reference_dtype`, derived from
`reference_provenance.runtime.dtype` and covered by `evidence_sha256`. Note that
`task_id` still reads `T21` and `evidence_tier` still reads
`real_onnxruntime_cpu` — both are true and neither distinguishes anything, so
**identify a record by `record_kind`**, not by those two.

Result: max absolute error 0.031 … 0.066 against 0.297 … 0.461 for the same
steps with a bfloat16 reference — 6.9x to 9.8x tighter. The derivation predicts
5.7x from the ULP ratio alone, and the excess is the two float16 pipelines'
errors being partly correlated. It clears the float16-appropriate bound of
0.201 with 3x to spare.

That is the evidence that the graph is faithful and the whole gap was the
reference dtype. Whether the T21 comparison should move to a float16 reference
is a contract decision that T23 did not take.

Reproduce:

```
python -m slm_lab.backends.onnx_cpu --reference-dtype float16 \
  --manifest results/manifests/onnx/S128.json --steps 4 --reference torch \
  --output results/graph/parity/diagnostics/S128-ort-cpu-float16-reference-probe.json
```

Both commands need the parity environment in `environments/onnx-cpu/README.md`,
`SLM_LAB_ARTIFACT_ROOT` pointing at the T20 graphs, and a local Hugging Face
cache.
