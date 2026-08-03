# ONNX Runtime CPU parity environment

This environment exists to run one job: the T21 ONNX Runtime CPU parity and
multi-step static-cache validation of the T20 reference graphs, driven by
`slm_lab.backends.onnx_cpu`.

## Why it is separate from the root lock

`pyproject.toml` and `uv.lock` pin the cross-platform repository environment,
and every documented setup command uses `--locked` (see `DEVELOPMENT.md` and
`environments/README.md`). Adding an ONNX Runtime extra to the root project
would invalidate that lock for every host, including hosts that will never run
a parity job. T50 solved the same problem for MLX by keeping the runtime out of
the root environment; this follows that precedent.

Everything in `src/slm_lab/graph/` and the unit tests in `tests/onnx/` are
deliberately dependency-free and run in the locked root environment. Only the
real parity measurement needs this second environment.

## What has to be pinned, and by whom

`environments/README.md` requires that a platform task pin exact versions
**after a compatibility smoke test** and record them in a host manifest. T21 ran
the first real parity measurement and T23 re-ran it against the promoted
`Concat` prefill graphs, so the `onnxruntime` and `numpy` pins below are now
smoke-tested rather than proposed.

Pinned by T20 for the reference side:

| Package | Version | Source |
|---|---|---|
| `torch` | 2.7.1 | `configs/models/qwen3-0.6b-onnx-export.json` |
| `transformers` | 4.51.3 | `configs/models/qwen3-0.6b-onnx-export.json` |
| `onnx` | 1.18.0 | `configs/models/qwen3-0.6b-onnx-export.json` |

Pinned by the parity runs, read from the built environment with
`importlib.metadata.version` rather than copied from prose:

| Package | Version | Smoke test that justifies it |
|---|---|---|
| `onnxruntime` | 1.28.0 | loads all eight opset 18 graphs with external data on the CPU EP at `ORT_DISABLE_ALL`, and produces the committed `results/graph/parity/S*-ort-cpu.json` |
| `numpy` | 2.4.6 | the version this `onnxruntime` and `torch` 2.7.1 build resolve together in the environment that produced that evidence |

These two must be chosen and recorded together: `onnxruntime` constrains the
`numpy` ABI, so a pin of one without the other does not reproduce anything.

Do not record `latest` as a version.

The export attestation in `configs/models/qwen3-0.6b-onnx-export.json` records
`runtime_python_version: 3.11.13`. T20's original export ran on CPython 3.11.15;
T23 re-exported and re-attested the prefill graphs on 3.11.13, the repository's
pinned interpreter and the one this environment is built on, so the attested
interpreter and the parity interpreter now agree. Parity would not have required
that — the graphs are hash-verified and the reference logits are recomputed from
the pinned weights at run time — but recording the interpreter that actually ran
is the point of the attestation, so the pin was moved to the real value rather
than the environment being bent to the pin. The runner still captures
`platform.python_version()` itself, so the evidence never depends on this file.

### Known contamination of the built environment

`.ai-local/envs/t21-ort-cpu` has **nine packages beyond the pins above**:
`jsonschema`, `jsonschema-specifications`, `referencing`, `rpds-py`, `attrs`,
`pytest`, `pluggy`, `iniconfig`, and `pygments`. They were added to run the test
suite and the manifest validation in the same interpreter as the measurement.
Nothing was upgraded or removed, so the versions above are the versions that
produced the evidence.

The consequence is procedural: **do not derive a pin set from `pip freeze` in
this environment.** It would silently capture all nine as though they were
parity requirements. Reproduce from the explicit install command below instead.

## Building the environment

Use a separate virtual environment outside the locked root one, exactly as the
M4 MLX baseline does:

```bash
UV_CACHE_DIR=/tmp/slm-lab-ort-cache \
  uv venv --python 3.11.13 .ai-local/envs/t21-ort-cpu
UV_CACHE_DIR=/tmp/slm-lab-ort-cache \
  uv pip install --python .ai-local/envs/t21-ort-cpu/bin/python \
  torch==2.7.1 transformers==4.51.3 onnx==1.18.0 \
  onnxruntime==1.28.0 numpy==2.4.6
```

Confirm what was actually installed, rather than trusting this file:

```bash
.ai-local/envs/t21-ort-cpu/bin/python -c \
  "import importlib.metadata as m; print(m.version('onnxruntime'), m.version('numpy'))"
```

The parity runner reads the live `onnxruntime.__version__`, the active execution
providers, and the Python version at run time and writes them into its evidence
record, so the evidence never depends on this file being correct.

## Running the parity measurement

The graphs and their external-data sidecars live under the artifact root, not in
Git. Each graph has its **own** 1,192,085,504-byte `.onnx.data` file — eight in
all, byte-identical and therefore sharing one SHA-256, but costing about 9.0 GB
of storage for the set (`du -sh` on the T20 reference directory; it read 8.9 GB
before the T23 re-export grew the four prefill protobufs). Size the artifact
root from that number, not from 1.19 GB. The
runner verifies each graph's SHA-256 against the digest committed in
`results/manifests/onnx/S<context>.json` before it constructs a session.

```bash
SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab \
HF_HOME=<local-hf-cache> TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  .ai-local/envs/t21-ort-cpu/bin/python -m slm_lab.backends.onnx_cpu \
  --manifest results/manifests/onnx/S128.json \
  --steps 4 \
  --reference torch \
  --output results/graph/parity/S128-ort-cpu.json
```

Exit codes: `0` parity passed, `1` parity failed (read `failures` in the
evidence), `2` configuration or dependency error.

Only a run whose sessions are genuine `onnxruntime.InferenceSession` objects is
recorded as `evidence_tier="real_onnxruntime_cpu"`. The tier is derived from
the session objects and cannot be set by a caller or a command-line flag, so a
fake-session self-test can never be mistaken for a measurement.

## State after the T21 and T23 runs

All three follow-ups this section used to list are done:

1. The evidence JSONs are committed under `results/graph/parity/`.
2. `DEFAULT_ORT_CPU_TOLERANCE` no longer reads `proposed_unvalidated`. T23
   derived it from dtype and depth and measured it; `TOLERANCE_STATUS` is now
   `derived_and_measured`.
3. `docs/results/onnx/ort-cpu-parity.md` carries the measurement.

What this environment has **not** produced, and what a later run still owes:

- Every committed parity record is `ORT_DISABLE_ALL`. The paired
  `ORT_ENABLE_ALL` run has never been taken, so the fusion delta is unmeasured.
- The evidence is one build, one execution provider, one host. Nothing here
  supports a claim about another `onnxruntime` version, another EP, or another
  machine.
- The committed records compare against a **bfloat16** reference. A
  float16-reference probe was run as a diagnostic only; changing the reference
  dtype is a contract decision that has not been taken.
