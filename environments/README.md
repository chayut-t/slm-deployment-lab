# Reproducible environments

The repository foundation uses CPython 3.11.13 and uv 0.11.32, recorded in
`common-toolchain.json`. Exact direct dependency pins live in
`pyproject.toml`; `uv.lock` freezes the full transitive solution. A clean setup
is:

```bash
uv python install 3.11.13
uv sync --extra dev --locked
uv run python -m unittest discover -s tests -p 'test_*.py'
```

`uv` keeps the environment in the ignored `.venv/` directory. Do not install
model weights or caches there. Set `HF_HOME` beneath
`SLM_LAB_ARTIFACT_ROOT` before later model tasks.

Environment-specific setup and compatibility evidence belongs in:

- `macos-m4/`
- `linux-cuda/`
- `linux-aimet/`

T01 deliberately does not choose untested versions for MLX, CUDA, AIMET,
QAIRT, or hosted runtimes. Their owning platform tasks must pin exact package,
SDK, driver, and operating-system versions after a compatibility smoke test,
then record them in a host manifest. Floating labels such as `latest` are not
reproducible evidence.
