# Linux AIMET environment

T40 freezes this environment. It is the Lane B host from `docs/project/plan.md`
section 7: local AIMET quantization simulation, calibration, LPBQ, and layer
sensitivity analysis. T41 (W8 evidence), T42 (W4A8/LPBQ/LiteMP/sensitivity),
and T43 (quantized compile, inference, profile) consume it. Lane A (AI Hub
Workbench, T30-T33) does not use this environment; it records hosted service
versions separately.

This directory specifies the environment. It does not contain a measurement.
No AIMET code has been executed for this repository yet.

## This environment cannot run on the primary macOS host

The primary machine is a Mac mini (2024), Apple M4, `arm64` Darwin. AIMET's
compiled component — the `aimet-onnx` extension — is published for Linux x86-64
and for Windows on both x86-64 and ARM64. It has no macOS wheel and therefore
no Apple-silicon wheel:

- `aimet-onnx` 2.36.0 ships exactly one file on PyPI,
  `aimet_onnx-2.36.0-cp310-abi3-manylinux_2_34_x86_64.whl`. The
  `manylinux_2_34_x86_64` platform tag excludes both macOS and `arm64`, and
  requires glibc 2.34 or newer. The same is true of every release from 2.32.1
  through 2.36.0.
- The GitHub release for 2.36.0 additionally publishes two Windows wheels that
  PyPI does not carry: `aimet_onnx-2.36.0+cpu-cp310-abi3-win_amd64.whl` and
  `…-win_arm64.whl`. Both are CPU-only. That release publishes two `+cu126`
  assets, and `aimet-cuda-wheels.lock` pins both, but only
  `aimet_onnx-2.36.0+cu126-cp310-abi3-manylinux_2_34_x86_64.whl` carries CUDA
  kernels; `aimet_torch-2.36.0+cu126-py310-none-any.whl` is 792612 bytes of
  pure Python and differs from its `+cpu` twin only in its version metadata and
  the `torch` build string in `aimet_torch/common/_version.py` (see
  "CUDA-variant wheels" below). Neither wheel contains a compiled extension.
  So Windows is a supported *platform* but not a supported *CUDA* platform,
  and neither Windows wheel helps an `arm64` Mac: `win_arm64` is
  Windows-on-ARM, not Darwin.
- `aimet-torch` 2.36.0 ships `aimet_torch-2.36.0-py310-none-any.whl`, which is
  pure Python and therefore installable anywhere. Installing it on macOS does
  not produce a working AIMET: the wheel contains no compiled extension at all,
  and its only `aimet_common` content is a deprecation shim that re-exports
  `aimet_onnx.common` or `aimet_torch.common`. The quantization kernels live in
  `aimet-onnx`.
- The AIMET installation page carries a "Supported platforms" table whose three
  rows are "Linux (x86-64, Ubuntu 22.04 and above)", "Windows", and "macOS
  (Apple Silicon)". Its `aimet-onnx` column reads "Prebuilt wheel", "Prebuilt
  wheel (x86-64, ARM64)", and "Build from source" respectively, so macOS is the
  one platform with no prebuilt `aimet-onnx`. The page states Python support
  once, for both distributions: "Both packages support Python 3.10+ (tested
  through 3.13)." It names no CPU vendor and no Windows release. On GPUs it
  says "CUDA 12.x is validated for aimet-torch and aimet-onnx on Linux
  (x86-64), using the `+cu126` wheels. Windows and macOS builds are CPU-only."

So the CUDA ONNX-graph quantization path this lab needs is Linux x86-64. Every
pin below is a specification to be executed on such a host, not something
observed here.

## Verified in this repository

Everything in this section was confirmed against a live source on 2026-08-02,
and the source is named. Nothing here required an AIMET installation.

### Package versions and digests

| Package | Version | Source of truth |
|---|---|---|
| `aimet-torch` | 2.36.0 | PyPI JSON API, uploaded 2026-07-29 |
| `aimet-onnx` | 2.36.0 | PyPI JSON API, uploaded 2026-07-29 |
| `torch` | 2.7.1 | held from T20 `qwen3-0.6b-onnx-export.json` |
| `onnx` | 1.18.0 | held from T20 `qwen3-0.6b-onnx-export.json` |
| `transformers` | 4.51.3 | held from T20, also the repo `tokenizer` extra |

Both AIMET distributions are BSD-3-Clause and declare `requires_python >=3.10`.
Re-check with:

```bash
curl -sS https://pypi.org/pypi/aimet-onnx/json | python3 -m json.tool | less
curl -sS https://pypi.org/pypi/aimet-torch/json | python3 -m json.tool | less
```

The `aimet-torch` wheel was downloaded to a scratch directory, hashed, and its
contents listed. The computed sha256 matched the PyPI digest
`f4d7d49f603febacf660144049081e18988b72683b114eb2afb4315aee5463a7` at 792550
bytes. It was not installed and is not committed.

### Resolved dependency lock

`aimet-requirements.lock` is a **complete, real transitive resolution** with
hashes, produced by `uv` 0.11.32 (the version pinned in
`common-toolchain.json`) on 2026-08-02 with an explicit target platform:

```bash
uv pip compile environments/linux-aimet/aimet-requirements.in \
  --python-platform x86_64-manylinux_2_34 \
  --python-version 3.11 \
  --generate-hashes \
  --no-build \
  --custom-compile-command "uv pip compile environments/linux-aimet/aimet-requirements.in --python-platform x86_64-manylinux_2_34 --python-version 3.11 --generate-hashes --no-build --output-file environments/linux-aimet/aimet-requirements.lock" \
  --output-file environments/linux-aimet/aimet-requirements.lock
```

`--no-build` means the whole graph resolved from published wheel metadata with
no source build, so the resolution reproduces from any host, including the
`arm64` macOS primary. The command is recorded in the lock's own header.

What the resolution establishes, and only this:

- AIMET 2.36.0 and T20's `torch==2.7.1`, `onnx==1.18.0`, and
  `transformers==4.51.3` pins **co-resolve** on Linux x86-64 / CPython 3.11,
  with no backtracking conflict and no version relaxation.
- The resolved CUDA stack is `nvidia-cuda-runtime-cu12==12.6.77` and
  `nvidia-cudnn-cu12==9.5.1.17`, pulled in by the `torch` 2.7.1 manylinux
  wheel. That is CUDA 12.6, the same minor AIMET's own validated `+cu126`
  wheels target. A strong compatibility signal, but still metadata.
- Notable resolved transitive pins: `numpy==2.4.6`, `onnxruntime==1.28.0`,
  `onnxscript==0.7.1`, `onnx-ir==0.2.1`, `onnx2torch==1.5.15`.

What the resolution does **not** establish: that the packages import, that
AIMET's compiled kernels load, that CUDA initializes, or that any Qwen3-0.6B
graph quantizes. Those are runtime facts, listed as unverified below.

### CUDA-variant wheels

`aimet-cuda-wheels.lock` hash-pins the `+cu126` and `+cpu` AIMET wheels by
direct URL. These local-version wheels are published only as GitHub release
assets, never to PyPI, so `uv pip compile` cannot discover them and they are
necessarily a separate overlay file. Digests came from the GitHub releases API:

```bash
curl -sS https://api.github.com/repos/qualcomm/aimet/releases/tags/2.36.0 \
  | python3 -c 'import json,sys; [print(a["name"], a["size"], a["digest"]) for a in json.load(sys.stdin)["assets"]]'
```

The size split is informative: `aimet_onnx-2.36.0+cu126-...` is 68864490 bytes
against 778078 bytes for `aimet_onnx-2.36.0+cpu-...` of the same release, so
the CUDA kernels ship inside the aimet-onnx wheel rather than in a separate
package. The plain PyPI `aimet-onnx` wheel is 68858273 bytes, i.e. essentially
the CUDA build.

The two `aimet-torch` variants were downloaded on 2026-08-02, hashed against
the digests above (`+cpu` 792588 bytes,
sha256 `9d0f9c2945e58f3931cb184b208ac7be35a25fbd990b987f7612ff264cf50e7c`;
`+cu126` 792612 bytes,
sha256 `c659e35192561e6cdd7a41fa84395f1b5f91155e08537faf2f9db4b31c46a18d`),
and diffed entry by entry. They were not installed and are not committed. Both
carry 380 entries and **zero** compiled objects — no `.so`, `.pyd`, `.dll` or
`.dylib` in either — which is the load-bearing fact: the `+cu126` `aimet-torch`
wheel contains no CUDA code. They differ in exactly two places:

- the four `dist-info/*` files — `METADATA`, `RECORD`, `WHEEL`, and
  `licenses/LICENSE` — which sit under a directory whose name carries the local
  version, so all four differ by path. By *content*, `WHEEL` and `LICENSE` are
  byte-identical (identical sha256 in both `RECORD`s), the `METADATA` diff is a
  single line — `Version: 2.36.0+cpu` against `Version: 2.36.0+cu126` — and
  `RECORD` differs in five lines: the four renamed `dist-info` paths and the
  `aimet_torch/common/_version.py` digest;
- one shipped **source** file, `aimet_torch/common/_version.py`, which is
  package content rather than wheel metadata:

  ```python
  # +cpu                              # +cu126
  __version__ = '2.36.0+cpu'          __version__ = '2.36.0+cu126'
  torch = '2.13.0+cpu'                torch = '2.13.0+cu126'
  ```

So "differs only in metadata" would be too strong: the variants also record a
different pinned `torch` build. Only `__version__` is read at runtime, by
`aimet_torch/common/utils.py::_get_version_string`; nothing in either wheel
reads the `torch` field, so it is a provenance record rather than a constraint.
It is still evidence of what Qualcomm built against — see "Relationship to the
T20 export pins" below.

That release also carries `aimet_onnx-2.36.0+cpu-cp310-abi3-win_amd64.whl`
(677407 bytes) and `…-win_arm64.whl` (647209 bytes). Neither is pinned here:
this lab targets Linux, and neither is a CUDA build. They are named because
"AIMET is Linux-only" would be wrong. The accurate statement is narrower and
is about the compiled distribution: `aimet-onnx` has no macOS wheel at all —
the installation page's macOS (Apple Silicon) row reads "Build from source" —
and its only CUDA build is Linux x86-64. `aimet-torch` is pure Python and the
same page lists a prebuilt wheel for macOS, which is true and useless here,
because the quantization kernels are not in it.

### Documented host requirements

From the AIMET 2.36.0 installation page, refetched on 2026-08-02. These are
documented requirements, not observations of any machine this project owns.
Each bullet quotes the page and then says what it means for this lab:

- Platform: "Linux (x86-64, Ubuntu 22.04 and above)", the row of the supported
  platforms table with a prebuilt `aimet-onnx` wheel and CUDA. Consistent with
  the `manylinux_2_34_x86_64` tag. The page names no CPU vendor, so `x86-64`
  here covers AMD as well as Intel.
- Python: "Both packages support Python 3.10+ (tested through 3.13)." This lab
  uses the repo-pinned 3.11.13, inside that range.
- GPU: "Nvidia GPU card (Compute capability 5.2 or later)".
- Driver: "Nvidia driver version 455 or later (using the latest driver is
  recommended; both CUDA and cuDNN are supported)".
- CUDA: "CUDA 12.x is validated for aimet-torch and aimet-onnx on Linux
  (x86-64), using the `+cu126` wheels. Windows and macOS builds are CPU-only."

The page is served from the `releases/latest/` path. The versioned
`releases/2.36.0/overview/install/index.html` was fetched on the same day and
is byte-identical, so the quotations above are pinned to 2.36.0 and not only to
whatever "latest" resolves to later.

## Unverified from this host

None of the following has been observed. Each row gives the exact command that
would produce the value on a real Linux CUDA host. Do not copy any of these
into a manifest or a result document until the command has actually run there.

| Fact | Status | Command that would verify it |
|---|---|---|
| Host OS name and version | unverified | `. /etc/os-release && echo "$NAME $VERSION_ID"` |
| Kernel | unverified | `uname -r` |
| glibc >= 2.34 | unverified | `ldd --version \| head -1` |
| GPU model | unverified | `nvidia-smi --query-gpu=name --format=csv,noheader` |
| Compute capability >= 5.2 | unverified | `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` |
| Driver version >= 455 | unverified | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` |
| CUDA toolkit version | unverified | `nvcc --version` |
| cuDNN version | unverified | `python -c 'import torch; print(torch.backends.cudnn.version())'` |
| Torch sees the GPU | unverified | `python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)'` |
| `aimet_onnx` imports | unverified | `python -c 'import aimet_onnx; print(aimet_onnx.__version__)'` |
| `aimet_torch` imports | unverified | `python -c 'import aimet_torch; print(aimet_torch.__version__)'` |
| ORT CUDA provider available | unverified | `python -c 'import onnxruntime as o; print(o.get_available_providers())'` |
| AIMET quantizes the T20 graph | unverified | the smoke test below |
| Calibration or quantization time | unverified | not yet designed; T41 owns it |
| Any perplexity or accuracy delta | unverified | T41/T42 own it |

The lock's resolved versions are verified **as a resolution**. They become
verified **as an installation** only after `uv pip install --require-hashes`
succeeds on the real host and the import commands above run there.

## Build the environment on the Linux CUDA host

Start from a checkout of this repository at the commit that owns the lock.

```bash
uv python install 3.11.13
uv venv --python 3.11.13 .ai-local/envs/t40-aimet
export AIMET_VENV="$PWD/.ai-local/envs/t40-aimet"

# 1. The verified, fully hash-pinned base environment.
uv pip install --python "${AIMET_VENV}/bin/python" \
  --require-hashes -r environments/linux-aimet/aimet-requirements.lock

# 2. Optional: swap to the AIMET CUDA build validated by Qualcomm.
uv pip install --python "${AIMET_VENV}/bin/python" --no-deps \
  --require-hashes -r environments/linux-aimet/aimet-cuda-wheels.lock
uv pip install --python "${AIMET_VENV}/bin/python" onnxruntime-gpu
```

Step 2 is a deliberate, recorded divergence from the lock: it replaces the two
AIMET distributions with their `+cu126` local versions and adds a package that
is not hash-pinned. If step 2 is applied, say so in the run's artifact
manifest, record the observed `onnxruntime-gpu` version, and then pin it in
`aimet-cuda-wheels.lock`.

Keep the venv under `.ai-local/`, which is never committed. Point `HF_HOME` and
the calibration corpus beneath `SLM_LAB_ARTIFACT_ROOT`; weights and calibration
data do not belong in the environment or in Git.

## Verify the environment on the Linux CUDA host

```bash
"${AIMET_VENV}/bin/python" - <<'PY'
import sys
import onnx
import torch
import transformers
import aimet_onnx
import aimet_torch

print("python     ", sys.version)
print("torch      ", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available())
print("onnx       ", onnx.__version__)
print("transformers", transformers.__version__)
print("aimet_onnx ", aimet_onnx.__version__)
print("aimet_torch", aimet_torch.__version__)
PY
```

Then exercise the compiled quantization kernel, not only the Python import
surface. Two caveats shape the command below, and both were found by reading
the wheel rather than by running it.

First, `aimet_common/__init__.py` is deprecated since AIMET 2.20 and **raises
`ImportError` when both `aimet_onnx` and `aimet_torch` are installed**, which
is exactly this environment. The AIMET docs write the smoke test as `from
aimet_common import libpymo`, so the documented form fails here. Use the
explicit package the shim itself recommends.

Second, the AIMET docs' `libpymo.EncodingAnalyzerForPython(...)` form does not
exist in this wheel. `libpymo.py` is `from ._libpymo import *` with a
`py_libpymo` fallback, and the compiled Cython module
`aimet_onnx/common/_libpymo` defines exactly three public Cython objects:
`TfEncoding`, `BlockTensorQuantizer`, and `PtrToInt64`. (It also carries the
`QuantizationMode`, `RoundingMode`, and `TensorQuantizerOpMode` `IntEnum`s,
which live in the plain-Python `_quant_enums.py` next to it.)
`TfEncodingAnalyzer`, `MinMaxEncodingAnalyzer`, `PercentileEncodingAnalyzer`
and friends exist only as C++ symbols inside the shared object; nothing binds
them into Python. `BlockTensorQuantizer` is what the shipped
`aimet_onnx/qc_quantize_op.py` actually drives.

```bash
"${AIMET_VENV}/bin/python" - <<'PY'
import numpy as np
from aimet_onnx.common import libpymo

x = np.random.randn(4096).astype(np.float32)

# ([] = per-tensor, bitwidth, quant scheme) - the same three arguments
# aimet_onnx/qc_quantize_op.py passes when it builds a quantizer.
quantizer = libpymo.BlockTensorQuantizer(
    [], 8, libpymo.QuantizationMode.QUANTIZATION_TF
)
quantizer.updateStats(x)
for encoding in quantizer.computeEncodings(False):  # False = asymmetric
    print(encoding.min, encoding.max, encoding.delta, encoding.offset,
          encoding.bw)
PY
```

**Derived from the shipped wheel, not executed.** `aimet-onnx` cannot be
installed on this host, so every name above comes from reading
`aimet_onnx-2.36.0+cpu-cp310-abi3-manylinux_2_34_x86_64.whl` (778078 bytes,
sha256 `6d91b581da9e22f332c78d7e127cfb4ac4928bf950cc74409bb97e4d28c6dc70`,
downloaded from the 2.36.0 GitHub release on 2026-08-02, hashed, never
installed and never committed):

- the exported Python names come from the Cython qualified-name strings in
  `aimet_onnx/common/_libpymo.abi3.so`. Reproduce with
  `strings -a aimet_onnx/common/_libpymo.abi3.so | grep -oE '_libpymo\.[A-Za-z_][A-Za-z0-9_.]*' | sort -u`,
  which lists `TfEncoding`, `BlockTensorQuantizer` and their methods
  (`updateStats`, `computeEncodings`, `getEncodings`, `setQuantScheme`,
  `quantizeDequantize`, …) and no analyzer class;
- the three-argument constructor matches both `qc_quantize_op.py:77` and the
  exported C++ symbol
  `DlQuantization::BlockTensorQuantizer::BlockTensorQuantizer(std::vector<long>, int, QuantizationMode)`;
- `updateStats(tensor)` and `computeEncodings(use_symmetric)` are the call
  shapes used at `qc_quantize_op.py:665` and `:582`;
- `libpymo.QuantizationMode.QUANTIZATION_TF` is used by the shipped
  `aimet_onnx/common/defs.py` (`MAP_QUANT_SCHEME_TO_PYMO`) on the ordinary
  compiled-import path — it would `AttributeError` on every AIMET install if
  the compiled module did not carry it — and
  `aimet_onnx/common/_quant_enums.py` documents those `IntEnum`s as "shared
  between _libpymo and libquant_info Cython modules". It is therefore not
  merely a `py_libpymo` fallback name. This one is an inference from the
  shipped call sites rather than a directly observed export, because the enum
  is a Python object rather than a Cython type and leaves no qualified-name
  string in the shared object.

Note that the shared object is built with `CYTHON_COMPRESS_STRINGS`, so most of
its Python-level string constants are compressed: grepping it for a name and
finding nothing does **not** prove the name is absent. The qualified-name
strings above are uncompressed and are the reliable signal.

If any of this is wrong on the real host, that is a finding: record the actual
`AttributeError` or `TypeError`, the real signature from
`help(libpymo.BlockTensorQuantizer)`, and correct this file. Record the actual
numeric output. Do not copy sample values from the AIMET docs into any
repository evidence.

Finally, before quantizing anything, check baseline parity against the T20
reference export, which is the T40 acceptance criterion "baseline model parity
is checked before quantization". Load the T20 float16 ONNX graph, confirm its
`source_artifact_sha256`, `external_data_sha256`, and per-context
`graph_sha256` values from `configs/models/qwen3-0.6b-onnx-export.json`, then
run the T10 token fixtures and the T11 three-token generation oracle on this
host before any `QuantizationSimModel` is constructed. A parity failure here is
an environment bug, not a quantization result.

## Relationship to the T20 export pins

T20 exported the reference ONNX graph with `torch 2.7.1`, `transformers
4.51.3`, `onnx 1.18.0`, opset 18, float16, on CPython 3.11.15. T23 re-exported
the four prefill graphs and re-attested on 3.11.13, so
`configs/models/qwen3-0.6b-onnx-export.json` now records
`runtime_python_version: 3.11.13`; the three library pins are unchanged by that
re-export. This environment holds those three pins.

Holding them is currently free, which is a fact to state precisely rather than
a coincidence to rely on:

- The **PyPI** `aimet-torch` 2.36.0 distribution declares `torch` and
  `torchvision` with **no version bound** in its base requirements. A hard
  `torch==2.12.*` appears only under the optional `v1-deps` extra, which this
  environment does not install. Re-read from the PyPI JSON API on 2026-08-02:
  that holds for every release from 2.32.1 through 2.36.0, and the extra has
  been `torch==2.12.*` since 2.31.0 (2.27.0-2.30.0 pinned `torch==2.11.*`).
- `aimet-onnx` 2.36.0 likewise declares `torch` and `onnx` unbounded.
  `onnxruntime>=1.19`, `onnxscript>=0.4.0`, and `onnx_ir>=0.1.16` are the only
  floors that constrain this stack; the remaining floors it declares
  (`bokeh>=3.3.0`, `hvplot>=0.10.0`) bound plotting packages only.
- The resolver therefore accepted `torch==2.7.1` and `onnx==1.18.0`, which is
  what `aimet-requirements.lock` records.

**The GitHub `+cpu` and `+cu126` assets are not the PyPI distribution, and their
torch numbers are one minor ahead.** Both were read on 2026-08-02 from the
downloaded wheels named in `aimet-cuda-wheels.lock`, not from PyPI metadata:

| Distribution | `v1-deps` extra | `_version.py` `torch` |
|---|---|---|
| `aimet_torch-2.36.0` (PyPI) | `torch==2.12.*` | `2.12.1+cu126` |
| `aimet_torch-2.36.0+cpu` (GitHub) | `torch==2.13.*` | `2.13.0+cpu` |
| `aimet_torch-2.36.0+cu126` (GitHub) | `torch==2.13.*` | `2.13.0+cu126` |

Same release number, different build. This has no effect on the install,
because step 2 applies the overlay with `--no-deps` and `v1-deps` is never
requested — but it does mean the CUDA wheel this environment installs was built
against `torch 2.13.0`, not the `2.12.*` the PyPI metadata advertises. Read the
`_version.py` `torch` string as provenance, not as a constraint: nothing in
either wheel reads it.

So the signal of what Qualcomm actually tests against is `torch` 2.12-2.13
depending on which artifact you read, and this environment's `torch==2.7.1`
sits well behind either. Treat "resolves" and "is tested" as different claims,
and when recording the AIMET version in a run manifest, record *which*
distribution was installed, not only `2.36.0`.

**Divergence rule.** The export host and the calibration/quantization host are
separate environments and are allowed to differ. They are *not* allowed to
differ silently.

1. Prefer holding T20's pins. A quantization result is comparable to the T20
   float16 baseline only if the graph it consumed is the T20 graph.
2. If a future AIMET release forces a different `torch`, `onnx`, or
   `transformers`, do not quietly relax `aimet-requirements.in`. Record the
   forced version, the AIMET release that forced it, and the resolver error,
   then re-run the T20 baseline parity check above on the new stack.
3. The CPython interpreter is covered by this rule too, and a patch-level
   difference between the interpreter that produced the graph and the one that
   will consume it counts. Any run on a stack whose interpreter differs from
   the attested one must carry both interpreter versions in its artifact
   manifest, exactly as it carries both library version sets. There is no such
   difference right now: T20 originally exported on CPython 3.11.15, but T23
   re-exported and moved the pin, so `runtime_python_version` in
   `configs/models/qwen3-0.6b-onnx-export.json` is **3.11.13** — the same
   repository-pinned interpreter this environment builds on
   (`environments/common-toolchain.json`). The rule stands; it currently has no
   live instance.
4. Any run produced on a diverged stack must record both version sets in its
   artifact manifest and be labelled as such wherever its numbers appear. Do
   not compare a diverged-stack quantization number to a T20-stack baseline
   without saying so.
5. Never re-export the reference graph from this environment to make the pins
   agree. T20 owns the export, and a re-export changes the `graph_sha256`
   values that downstream comparisons are keyed on.

## Host manifest

`aimet-host.template.json` is a **template, not evidence**. It deliberately
fails validation while its `<REPLACE: ...>` placeholders remain, because
`host-v1.schema.json` requires an exact version for the four version-typed
`platform_details.nvidia` fields — `compute_capability`, `driver_version`,
`cuda_version`, and `cudnn_version` — and there is no truthful "not yet
provisioned" value for a GPU that does not exist yet. The fifth field in that
block, `gpu_name`, is only `{"type": "string", "minLength": 1}`, so a
placeholder passes it; that is the first of the four gaps listed below.

**The validator is not a completeness gate.** Run it, but do not treat a clean
run as "everything is filled in":

```bash
PYTHONPATH=src python -m slm_lab.manifests.cli host \
  environments/linux-aimet/aimet-host.template.json
```

`host-v1.schema.json` types several fields loosely, so a `<REPLACE: ...>`
string is a perfectly valid value for them. Filling in only the fields the
validator names produces `host manifest valid` while four placeholders are
still in the document. Verified on 2026-08-02 by filling exactly the reported
fields and re-running: the validator exited 0 with four `<REPLACE:` strings
remaining. Those four are:

- `hardware.product`
- `hardware.cpu.cores`
- `hardware.gpu.count`
- `platform_details.nvidia.gpu_name`

So the fill-in procedure is two checks, not one, and the second is mandatory:

```bash
# 1. Schema validity.
PYTHONPATH=src python -m slm_lab.manifests.cli host results/hosts/<host-id>.json

# 2. Placeholder scan. Must print nothing; any hit is an unfilled field the
#    schema is too loose to catch.
grep -n '<REPLACE:' results/hosts/<host-id>.json && echo 'UNFILLED PLACEHOLDERS'
```

The `tools` block is already honest and complete: every entry carries
`version: null`, `status: "deferred"`, a real `capture_command`, and a
non-empty reason, which is what the schema requires for a checked-but-absent
tool. Fill the placeholders on the real host, delete the `_template` key,
run both checks above until the first passes and the second prints nothing,
and move the result to `results/hosts/<host-id>.json`. That directory belongs
to the task that captures the evidence, not to this environment directory.

## References

- [AIMET](https://github.com/quic/aimet)
- [AIMET documentation](https://quic.github.io/aimet-pages/)
- [AIMET LPBQ](https://quic.github.io/aimet-pages/releases/latest/techniques/lpbq.html)
- [AIMET LLM quantization recipes](https://quic.github.io/aimet-pages/releases/latest/tutorials/quantization_recipe.html)
- [AIMET installation](https://qualcomm.github.io/aimet-pages/releases/latest/overview/install/index.html)
- [AIMET quick start](https://qualcomm.github.io/aimet-pages/releases/latest/overview/install/quick-start.html)
- [AIMET GitHub releases](https://github.com/qualcomm/aimet/releases)

The plan's four AIMET links use the `quic.github.io` host. Both
`quic.github.io` and `qualcomm.github.io` serve `aimet-pages` and both returned
HTTP 200 on 2026-08-02. The PyPI metadata and the project README both point at
`qualcomm.github.io`, so treat that as primary and `quic.github.io` as the
alias the plan already records. The installation pages exist only under
`overview/install/`; there is no top-level `install/` path.

## Scope boundaries

T30 records public AI Hub client and hosted compiler/runtime versions; do not
duplicate them here. T60 owns the general Linux CUDA extension in
`environments/linux-cuda/` for ONNX Runtime GPU work; that is a different
environment with a different purpose, and neither should be assumed to satisfy
the other. A locally resolvable wheel establishes nothing about QAIRT, target
device, or hosted-runtime compatibility.
