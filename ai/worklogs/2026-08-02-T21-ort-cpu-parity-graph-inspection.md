# T21: ONNX Runtime CPU parity and graph inspection

Date: 2026-08-02
Task: `T21`
Visibility: `public`
Status: completed

## Outcome

T21 delivers two things at very different evidence tiers, and keeping that
boundary unambiguous was as much of the work as the code.

**The graph inspection is a real measurement.** All eight T20 reference graphs
were parsed byte by byte and scored against a committed risk catalogue. The
headline result is that the T12 static contract holds end to end: across the
eight graphs there are zero symbolic and zero unset dimensions, zero
control-flow operators, zero nested subgraphs, and exactly one standard opset
import. Every risk found is a risk *inside* an otherwise static, standard,
single-domain graph. The four committed reports under `results/graph/` are
reproducible from the hash-verified artifacts with one command.

**The ORT CPU parity is not a measurement, and says so.** There is no
`onnxruntime`, `torch`, `numpy`, or `onnx` in this environment, so the parity
runner has never executed a real graph. `docs/results/onnx/ort-cpu-parity.md`
opens with `Status: no measurement`, states that acceptance criteria (a) and
(b) are satisfied by construction rather than by measurement, marks the
tolerances `proposed_unvalidated`, and records the exact command a host with
the runtime must run. No parity number exists anywhere in the repository.

The distinction the task exists to draw — a numerical-tolerance failure versus
a cache state-update failure — is implemented as two independent code paths
that always both run, and is pinned by seventeen fault-injection scenarios.

## Changes

- Added `src/slm_lab/graph/onnx_reader.py`: a read-only ONNX protobuf
  structural decoder built on the standard library alone. No `onnx`, no
  `protobuf`. It recovers opset imports, IR version, producer, boundary and
  `value_info` tensors, initializer metadata, and the full node list with
  attributes, recursing into subgraphs with a scope path. It retains only
  `len(raw_data)` for tensor payloads and never opens an external-data sidecar,
  so a 35 MB graph costs bounded memory.
- Added `src/slm_lab/graph/inspection.py` and
  `configs/graph/onnx-risk-rules-v1.json`: a 15-rule declarative catalogue and
  a detector engine that ranks findings by deployment impact, plus a CLI that
  verifies each graph's SHA-256 against the committed T20 manifest before
  parsing it. An inspection is always bound to a named, hash-verified graph.
- Added `src/slm_lab/backends/onnx_cpu.py`: the ORT CPU parity runner.
  Dependency-injected sessions and reference source, pure-Python metrics
  matching the T11 definitions, exact (bit-identical) static-cache invariants,
  a five-class failure taxonomy, and a forgery-resistant evidence tier derived
  from the session objects rather than from a caller's claim.
- Added `tests/onnx/` — 167 tests, of which 166 pass here and one skips because
  it needs a real runtime. Nothing heavy is required to run the rest.
- Added `results/graph/S{128,512,1024,4096}.json`: compact inspection reports
  with an in-band `claim_boundary`, regenerable and `--check`-verifiable.
- Added `docs/results/onnx/graph-inspection.md` and
  `docs/results/onnx/ort-cpu-parity.md`, plus `results/graph/README.md`,
  `configs/graph/README.md`, and `environments/onnx-cpu/README.md`.

## Verification

- Command: `PYTHONPATH=src python -m pytest tests -q`
- Result: **461 passed, 13 skipped** (pre-T21 baseline 295 passed, 12 skipped;
  +166 passing, +1 skipping, 167 added in total). The single new skip is
  `test_real_onnxruntime_cpu_parity_when_available`, guarded by
  `pytest.importorskip("onnxruntime")`.

- Command: `PYTHONPATH=src python -m ruff format --check` and `ruff check` on
  `src/slm_lab/graph/`, `src/slm_lab/backends/onnx_cpu.py`, `tests/onnx/`
- Result: 8 files already formatted; all checks passed (ruff 0.11.0).

- Command: `SLM_LAB_ARTIFACT_ROOT=<external-root> PYTHONPATH=src python -m
  slm_lab.graph.inspection --all-manifests --check`
- Result: exit 0 — the four committed reports are exactly what the tooling
  reproduces from the real graphs.

- Command: `python scripts/ai/render_task_status.py --check`
- Result: task graph valid; generated status current.

- Command: `python scripts/repo/check_hygiene.py --all`
- Result: passed.

Not run, and why: no ONNX Runtime parity measurement, no compile job, no
device job. None is possible in this environment and none is claimed.

## Decisions and evidence

- **No `onnx-cpu` extra was added to `pyproject.toml`, deliberately. Do not
  "fix" this back.** `uv.lock` records `provides-extras = ["dev",
  "tokenizer"]`, and every documented setup command in `DEVELOPMENT.md`,
  `environments/README.md`, and `environments/macos-m4/README.md` uses
  `uv sync … --locked`. Adding an extra without regenerating the lock — which
  cannot be done offline — would make `--locked` fail on every host, including
  hosts that will never run a parity job. Separately,
  `environments/README.md` forbids pinning versions that have not passed a
  compatibility smoke test, and no ONNX Runtime build has been smoke-tested
  here. `environments/onnx-cpu/README.md` therefore describes a separate
  virtual environment, following the precedent T50 set by keeping MLX out of
  the root lock. The task that first runs a real parity job owns the
  `onnxruntime`/`numpy` pins and must record them in a host manifest.
- The T20 `.onnx` files are small (1.5–35 MB) because every weight is external.
  That is what made a pure-Python protobuf walk viable and turned the graph
  inspection report from a prediction into a measurement.
- Cache-state checks are exact, not tolerant. A tolerant prefix check would let
  a state bug hide inside FP16 noise, which is precisely the confusion this
  task exists to prevent.
- The runner threads the decode graph's own `updated_valid_length` rather than
  an internal counter. A validation tool that substitutes the value it expected
  cannot observe the graph disagreeing with it.
- Non-finite candidate logits are a distinct `non_finite_logits` class, not a
  tolerance failure and not a configuration error: nothing is "outside" a
  threshold, and the fix domain is export precision. The run continues so the
  cache evidence for later steps is not lost. Non-finite *reference* logits
  remain a configuration error.
- `failures[]` is ordered state → non-finite → tolerance, so the most
  fundamental diagnosis is read first. Retolerancing a state fault is the
  failure mode this ordering is designed to prevent.
- Severities in the risk catalogue are reviewed structural judgements bound to
  a stated target context, not compiler results. The catalogue and every
  emitted report say so in band.

## Risks and limitations

- **The parity tolerances are unvalidated.** They are argued from the two error
  sources the comparison admits (a BF16→FP16 conversion and a backend change)
  and are deliberately looser than T11's same-model thresholds, but no run has
  confirmed them. The first real run must confirm or replace them.
- **`value_info` is empty in all eight graphs**, so the internal-dynamic-shape
  rule inspected nothing. Its silence is absence of information, not evidence
  of a static interior. Running ONNX shape inference on a host with `onnx`,
  writing the inferred `value_info` back, and re-running the inspection is the
  follow-up that closes it.
- The reader is a structural decoder, not a validator. It does not check
  operator schemas, type consistency, topological order, or opset
  compatibility. T20's `onnx.checker` acceptance remains the evidence for
  validity.
- The 1,231 flagged shape-defining inputs in the decode graph measure unfolded
  tracing residue, not proven dynamism. T20 exported with
  `do_constant_folding=False` and every boundary dimension is static, so most
  are probably foldable. The count is the natural before/after metric for that
  fix, not a defect count.
- Boundary byte totals are *declared* sizes computed from static shapes. They
  say what the graph asks the runtime for, not what any runtime copies.
- Whether this module's cosine denominator floor matches torch's exactly is
  unverified — the published formula floors each norm, the implementation is
  reported to floor the product. No published number depends on it.

## Follow-up

- Newly unblocked tasks: **T22** (QNN candidates and packaging). T21's ranked
  findings are its input: the shape-residue population, the 56
  `ScatterElements` cache writes, the 118-tensor decode boundary, and the
  O(S^2) inline mask are the four things a QNN candidate has to answer for.
- Recommended next action: run constant folding and re-inspect before anything
  else. It is upstream of every other count in the report, and every other
  number moves when it lands.
- Also outstanding: execute the recorded parity command on a host with the
  runtime, commit the evidence under `results/graph/parity/`, and update
  `docs/results/onnx/ort-cpu-parity.md` from "no measurement" to a result.
