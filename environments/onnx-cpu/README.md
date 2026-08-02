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
**after a compatibility smoke test** and record them in a host manifest.
No ONNX Runtime build has been smoke-tested in this repository yet, so this
file does not invent one. The task that first executes a real parity run owns
those pins.

Known-good and already pinned by T20 for the reference side:

| Package | Version | Source |
|---|---|---|
| `torch` | 2.7.1 | `configs/models/qwen3-0.6b-onnx-export.json` |
| `transformers` | 4.51.3 | `configs/models/qwen3-0.6b-onnx-export.json` |
| `onnx` | 1.18.0 | `configs/models/qwen3-0.6b-onnx-export.json` |

T20's export itself ran on CPython 3.11.15 (`runtime_python_version` in the
run attestation). Parity does not require the same patch release: the graphs
are already exported and hash-verified, and the reference logits are
recomputed from the pinned weights at run time. Build this environment on the
repository's pinned CPython 3.11.13 unless a real run shows a reason not to,
and record whatever interpreter actually ran in the evidence — the runner
captures `platform.python_version()` itself.

Still to be chosen and smoke-tested by the first real parity run:

| Package | Constraint |
|---|---|
| `onnxruntime` | must load an opset 18 model with external data on the CPU execution provider |
| `numpy` | whatever the selected `onnxruntime` and `torch` builds agree on |

Do not record `latest` as a version.

## Building the environment

Use a separate virtual environment outside the locked root one, exactly as the
M4 MLX baseline does:

```bash
UV_CACHE_DIR=/tmp/slm-lab-ort-cache \
  uv venv --python 3.11.13 .ai-local/envs/t21-ort-cpu
UV_CACHE_DIR=/tmp/slm-lab-ort-cache \
  uv pip install --python .ai-local/envs/t21-ort-cpu/bin/python \
  torch==2.7.1 transformers==4.51.3 onnx==1.18.0 \
  "onnxruntime==<smoke-tested-version>" "numpy==<smoke-tested-version>"
```

Then record the observed versions in a host manifest under `results/hosts/`
and add the chosen `onnxruntime`/`numpy` pins to the table above in the same
change. The parity runner reads the live `onnxruntime.__version__`, the active
execution providers, and the Python version at run time and writes them into
its evidence record, so the evidence never depends on this file being correct.

## Running the parity measurement

The graphs and their external-data sidecars live under the artifact root, not in
Git. Each graph has its **own** 1,192,085,504-byte `.onnx.data` file — eight in
all, byte-identical and therefore sharing one SHA-256, but costing about 8.9 GB
of storage for the set. Size the artifact root from that, not from 1.19 GB. The
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

## After the first real run

1. Commit the evidence JSON under `results/graph/parity/`.
2. Confirm or replace the proposed tolerances in
   `DEFAULT_ORT_CPU_TOLERANCE`; they are currently a documented hypothesis and
   are serialized with `status: proposed_unvalidated`.
3. Update `docs/results/onnx/ort-cpu-parity.md`, which today states plainly
   that no measurement exists.
