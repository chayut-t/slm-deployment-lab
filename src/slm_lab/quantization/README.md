# Quantization

Calibration, W8/W4 simulation, LPBQ, sensitivity analysis, mixed precision,
export, and deployed-artifact validation belongs here.

## `calibration` (T40)

`slm_lab.quantization.calibration` builds and verifies the frozen calibration
corpus that every W8/W4/LPBQ/LiteMP experiment must cite. It quantizes
nothing: AIMET is Linux + CUDA only and is pinned in
`environments/linux-aimet/`, not executed on the development host.

The module is deliberately dependency-free beyond the standard library and
PyYAML, so the corpus can be regenerated on any machine:

```bash
uv run python -m slm_lab.quantization.calibration generate  # rewrite the YAML
uv run python -m slm_lab.quantization.calibration check     # offline validation
uv run python -m slm_lab.quantization.calibration verify    # the same, exactly
uv run python -m slm_lab.quantization.calibration verify --online  # + re-fetch
```

`check` and `verify` are the same offline validation: both call
`validate_repository`, and both emit and contract-check every sample's prefill
tensors. Only `--online` differs, adding the tier-2 revision and licence
re-fetch. `check` is the name to use in CI because it cannot reach the network.

What it freezes:

- **Tier 1 (`t10_derived`)** — 13 samples derived deterministically from the
  CC0-1.0 T10 token fixtures, covering all four exported prefill shapes, the
  four token-class canaries, the chat-template canary, and the CC0 quality
  subset. This is the only tier that is committed and used.
- **Tier 2 (`external_diversity`)** — revision-pinned public corpora recorded
  as candidates for T41. No dataset row is committed.

Two properties are load-bearing and enforced by `tests/quantization`:

1. `configs/quantization/calibration.yaml` is a **fixed point** of
   `generate`. Any drift in the T10 bundle, the preprocessing contract, the
   sample set, the ordering, or the token budget changes the corpus hash and
   fails `check`.
2. Every emitted sample satisfies the frozen T12 prefill contract in
   `slm_lab.contracts.static_cache`. That contract forbids padding, so short
   fixtures are tiled to an exact frozen prompt length rather than padded, and
   `attention_mask` is all ones for every sample — by construction, not by a
   check; see the `validate_prefill_tensors` docstring.

`calibration_corpus.coverage` in the generated YAML is measured from the
emitted token IDs and bounds what the per-sample rationales may be read to
mean. `calibration_dataset_revision` is the value T41+ must copy into every
artifact manifest (`docs/project/plan.md` section 17.4).

## `parity` (T40)

`slm_lab.quantization.parity` is the fail-closed preflight that runs before any
weight is quantized. It has two halves and never lets a caller conflate them:

```bash
SLM_LAB_ARTIFACT_ROOT=<external-root> PYTHONPATH=src \
  python -m slm_lab.quantization.parity verify   # exits 0 only on full identity
SLM_LAB_ARTIFACT_ROOT=<external-root> PYTHONPATH=src \
  python -m slm_lab.quantization.parity record   # + write committed evidence
```

Half one, artifact identity, runs anywhere `hashlib` runs: the T20 attestation,
the committed manifests, the frozen T12 tensor contracts, and the bytes on the
external artifact root must still describe one floating baseline. Half two,
PyTorch-versus-ONNX logit parity, needs `torch` and `onnxruntime` and is
recorded as a declared `not_run` requirement owned by T21.

Because half two is a declaration rather than a measurement, the overall
`verdict` at this commit can only be `failed`, `unavailable`, or `partial`: no
branch of `parity.overall_verdict` assigns `verified` to the overall verdict,
even when the identity half is itself `verified`. The release field is named
`released_for_calibration_on_artifact_identity` so that its scope travels with
its value. See `results/quantization/README.md` for how to read a record.

## `w8` (T41)

`slm_lab.quantization.w8` freezes the two eight-bit-weight candidates the plan
calls Q1 (`w8a16`) and Q2 (`w8a8`), and gates the evidence that would promote
them. Like `calibration`, it quantizes nothing and depends on nothing beyond
the standard library and PyYAML, so it runs on any host:

```bash
uv run python -m slm_lab.quantization.w8 generate  # rewrite the candidates
uv run python -m slm_lab.quantization.w8 check     # offline validation gate
uv run python -m slm_lab.quantization.w8 status    # state, projection, ledger
uv run python -m slm_lab.quantization.w8 record    # write the readiness record
uv run python -m slm_lab.quantization.w8 compare \
  --baseline <float16-result.json> --candidate <w8-result.json>
uv run python -m slm_lab.quantization.w8 request --candidate <w8a16|w8a8> \
  --stage <compile|inference|profile> ...             # composes, never submits
```

Four properties are load-bearing and pinned by `tests/quantization`:

1. **Both candidate files are byte-identical fixed points of `generate`.** Any
   drift in the T40 calibration revision, a committed float16 baseline manifest
   digest, the frozen T13 protocol digest, the T21 graph inventory, or the
   frozen cache dtype fails `check` rather than silently re-anchoring the
   comparison.
2. **`assess_precision_state` computes `specified` / `simulated` / `deployed`
   from evidence and never from a claim.** It reads the input positively — a
   `state`, `precision_state`, `verdict`, or `deployed` key planted anywhere in
   the record is not consulted. `simulated` needs a record naming the tool, its
   exact version, the host, and the quantized-artifact digest. `deployed`
   additionally needs all three schema-v2 AI Hub stage manifests with a
   verified digest chain: compile's source must be the simulated artifact, and
   inference and profile must each cite the *recomputed* compile-manifest
   digest and consume the compile target artifact. Unlike `parity`'s terminal
   verdict, `deployed` is reachable from data — deliberately, so promotion is
   an auditable manifest set rather than a code edit — but no input available
   at this commit reaches it, and mutation tests prove each refusal.
3. **`weight_storage_projection` is arithmetic and says so.** Every row carries
   `"measurement": "analytic_projection"`, the mapping carries its own
   `does_not_establish` list, and the derived parameter total is reconciled
   byte for byte against the committed float16 export before any ratio is
   trusted.
4. **`compare_quality` refuses more than it computes.** Both records must
   validate against the frozen T13 result schema, cite this repository's
   protocol digest, and measure the same workload; the baseline must be
   floating; and `comparison_scope` is derived from the candidate record's own
   declared precision, with no caller argument that could relabel a simulated
   comparison as deployed. It has nothing to compare at this commit, and
   `validate_repository` fails if a W8 quality record appears.

The request emitter composes a real schema-v2 stage request from the committed
candidate plus caller-supplied private paths, derives `input_specs` from the
frozen T12 contract, refuses any committable output location, never imports
`qai_hub`, and fails closed because the quantized artifact does not exist. See
`ai/handoffs/T41-w8-submission-boundary.md` for the operational path and
`docs/results/quantization/w8.md` for the report.
