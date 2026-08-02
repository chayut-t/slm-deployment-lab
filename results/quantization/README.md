# Quantization evidence

Compact, committed evidence for the W8/W4 lane. Large artifacts stay under the
external artifact root; only digests, manifests, and small records belong here.

## Baseline parity records

`t40-baseline-parity-<date>.json` is one run of the T40 pre-quantization
baseline parity gate:

```bash
SLM_LAB_ARTIFACT_ROOT=<external-root> PYTHONPATH=src \
  python -m slm_lab.quantization.parity record
```

A record is a snapshot of one host at one commit. Re-running the gate is
cheaper than trusting an old record, so treat these as evidence of what was
checked and when, not as a standing guarantee.

`repository.git_commit` identifies the tree and `repository.git_tree_clean`
says whether the run happened at that commit exactly or on top of uncommitted
work. The `repository` block carries no checkout path: an absolute checkout
path is meaningless on the machine that later reads the record. The record is
not path-free overall — `artifact_root.requested`, `artifact_root.resolved`,
and `artifact_root.artifact_directory` are absolute, because the external
artifact root is a published location
(`configs/storage/external-ssd.example.yaml`) and which root was measured is
part of the claim.

The record is not inert. `tests/quantization/test_baseline_parity.py` asserts
that every `recorded_sha256` and `recorded_size_bytes` in the committed record
still equals what `results/manifests/onnx/S*.json` declares, so the record
cannot rot silently against the manifests it was measured from.

### What a passing record licenses

Only `artifact_identity.verdict == "verified"`, and only these claims, which
the record spells out under `claim_boundary.establishes`:

- the T20 evidence attestation and the four committed
  `results/manifests/onnx/S*.json` manifests still agree on every graph and
  external-data digest, with no missing or orphan context;
- the model and tokenizer revision still matches `configs/models/qwen3-0.6b.yaml`
  and `slm_lab.contracts.static_cache.MODEL_REVISION`;
- the graph tensor boundaries recorded by T20 still satisfy the frozen T12
  prefill and decode contracts by name, order, dtype, and static shape;
- when the artifact root was mounted, every recorded `.onnx` and `.onnx.data`
  file re-hashed to its committed sha256 at its committed `size_bytes`. Those
  entries are labelled `"measurement": "recomputed_sha256"`; anything else is
  labelled `"not_measured"`.

### What it does not license

`claim_boundary.does_not_establish` is authoritative. Most importantly, a
record **never** establishes numerical parity between the T11 PyTorch reference
and the T20 ONNX export. That half needs `torch` and `onnxruntime`, which the
primary macOS host does not carry, so it is always written as
`numerical_parity.status == "not_run"` together with the command that would
close it and the owning task (T21).

Be exact about what that implies for `verdict`. At this commit the gate can
emit only `failed`, `unavailable`, or `partial`. `parity.overall_verdict` is
the pure function that composes the two halves, and no branch in it assigns
`verified` to the overall verdict — not even when the identity half is itself
`verified` — because half two is a declared requirement rather than a
measurement and no input can turn it into one. (The identity half does have a
`verified` state of its own; read `artifact_identity.verdict` for that, never
`verdict`.) The overall `verified` becomes reachable only when T21 supplies a
real numerical result and this module is changed to consume it — a code change,
not a data change. Two tests pin it: a unit test over every identity verdict,
and an end-to-end run whose identity half is fully `verified` against a stub
artifact root.

The machine-readable release field is named for its scope:
`released_for_calibration_on_artifact_identity`. It is `true` only when all
four identity checks pass, and its name says the one thing it is about, so it
cannot be read as "parity is done". Read it together with `verdict` and
`verdict_scope`, and never quote a record as "parity checked" without saying
which half.

### Verification

- `python -m slm_lab.quantization.parity verify` exits non-zero unless artifact
  identity fully verifies, including the on-disk bytes. A missing artifact root
  produces an explicit `unavailable` outcome, never a pass.
- `pytest tests/quantization` is offline and does not read the ~9 GB of
  external artifacts. It includes a cheap existence and `st_size` probe that
  runs whenever the volume happens to be mounted.
- The full re-hash is opt-in:

  ```bash
  SLM_LAB_T40_VERIFY_ARTIFACT_BYTES=1 PYTHONPATH=src \
    python -m pytest tests/quantization -q
  ```
