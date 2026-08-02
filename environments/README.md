# Reproducible environments

The repository foundation uses CPython 3.11.13 and uv 0.11.32, recorded in
`common-toolchain.json`. Exact runtime and development dependency pins live in
`pyproject.toml`; `uv.lock` freezes that full transitive solution. The PEP 517
build backend is independently hash-locked by
`build-requirements.lock` because build-system requirements are not members of
the project lock. A clean setup is:

```bash
uv python install 3.11.13
uv sync --extra dev --locked
uv run python -m unittest discover -s tests -p 'test_*.py'
```

Build the package with the committed build constraint and mandatory hash
verification:

```bash
uv build \
  --python 3.11.13 \
  --build-constraints environments/build-requirements.lock \
  --require-hashes \
  --out-dir /tmp/slm-lab-dist
```

Regenerate the build lock only after intentionally changing the matching
`build-system.requires` pin:

```bash
uv pip compile environments/build-requirements.in \
  --no-deps \
  --generate-hashes \
  --custom-compile-command "uv pip compile environments/build-requirements.in --no-deps --generate-hashes --output-file environments/build-requirements.lock" \
  --output-file environments/build-requirements.lock
```

`uv` keeps the environment in the ignored `.venv/` directory. Do not install
model weights or caches there. Set `HF_HOME` beneath
`SLM_LAB_ARTIFACT_ROOT` before later model tasks.

Environment-specific setup and compatibility evidence belongs in:

- `macos-m4/`
- `linux-cuda/`
- `linux-aimet/`
- `onnx-cpu/`

T01 deliberately does not choose untested versions for MLX, CUDA, AIMET,
QAIRT, ONNX Runtime, or hosted runtimes. Their owning platform tasks must pin
exact package,
SDK, driver, and operating-system versions after a compatibility smoke test,
then record them in a host manifest. Floating labels such as `latest` are not
reproducible evidence.

`linux-aimet/` is a partial exception whose limits are stated in its own
README. AIMET's compiled wheel is published for Linux x86-64 and Windows, with
no macOS or Apple-silicon build and a CUDA build only for Linux x86-64, so its
package pins and hash-locked resolution were produced and verified from package
metadata on the macOS primary, while every host fact (OS, driver, CUDA, GPU)
and every runtime smoke-test result remains explicitly unverified until a real
Linux CUDA host runs the commands that file records.
