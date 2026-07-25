# T01: Environments, manifests, and storage preflight

Status: completed
Owner: Codex T01 agent
Updated: 2026-07-25

## Objective

Provide a small reproducible Python foundation, machine-checkable artifact and
host evidence contracts, and a safe storage preflight so later export,
deployment, and profiling tasks fail early when their environment or artifact
root is unsuitable.

## Scope

### In scope

- Pin the repository's Python and development dependencies in `uv.lock`.
- Document platform-environment boundaries without claiming untested SDK or
  hardware compatibility.
- Add versioned JSON Schemas and reusable validation for artifact and host
  manifests.
- Add representative valid and invalid fixtures and offline tests.
- Add a safe T9 preflight that verifies mount and resolved-directory containment,
  writability with a temporary probe, expected layout, and free-space reserve.
- Capture a sanitized primary-machine host manifest.

### Out of scope

- Downloading model weights, SDKs, or large artifacts.
- Proving MLX, CUDA, AIMET, QAIRT, or hosted-service compatibility; those
  belong to their platform tasks.
- Running model inference, benchmarks, or paid cloud jobs.
- Publishing mutable free-space figures as benchmark results.

## Dependencies and resources

- Required task dependencies: T00 (completed).
- Resource locks: `t9_heavy_io` (held for the lightweight preflight only).
- External access: Python package index metadata to resolve `uv.lock`.
- Cost boundary: no paid services; zero expected cost.

## Important paths

- Inputs:
  - `docs/project/plan.md`
  - `configs/models/qwen3-0.6b.yaml`
  - `ai/tasks/definitions/T01.yaml`
- Outputs:
  - `pyproject.toml`
  - `uv.lock`
  - `environments/`
  - `configs/storage/external-ssd.example.yaml`
  - `src/slm_lab/manifests/`
  - `results/hosts/apple-m4-primary.json`
- Shared contracts:
  - `ai/tasks/task_graph.yaml`
  - `ai/tasks/status.generated.md`

## Milestones

- [x] Lock a clean, installable repository environment and document platform
      extension ownership.
- [x] Implement artifact and host schemas with reusable validation.
- [x] Validate representative fixtures, including negative cases.
- [x] Pass the T9 preflight on the primary machine without material artifact
      I/O.
- [x] Complete task records and all repository gates.

## Verification and acceptance

- Commands:
  - `uv lock --check`
  - `uv sync --extra dev --locked`
  - `uv build --python 3.11.13 --build-constraints environments/build-requirements.lock --require-hashes --out-dir /tmp/slm-lab-dist`
  - `uv run --extra dev python -m unittest discover -s tests -p 'test_*.py'`
  - `uv run slm-lab-validate-manifest artifact tests/fixtures/manifests/artifact.valid.json`
  - `uv run slm-lab-validate-manifest host results/hosts/apple-m4-primary.json`
  - `uv run slm-lab-validate-manifest host tests/fixtures/manifests/host.linux-nvidia.valid.json`
  - `uv run slm-lab-validate-manifest host tests/fixtures/manifests/host.qualcomm-hosted.valid.json`
  - `uv run slm-lab-storage-preflight --json`
  - `python3 scripts/ai/render_task_status.py --check`
  - `python3 scripts/repo/check_hygiene.py --all`
- Behavioral criteria:
  - A clean locked sync succeeds on Python 3.11.
  - Schemas accept representative Apple, NVIDIA/Linux, and Qualcomm-hosted
    fixtures and reject missing, floating, inconsistent, or malformed
    provenance.
  - The primary storage preflight proves the configured root is on T9,
    writable, has the expected layout, and exceeds the configured reserve.
- Hardware/profile evidence: host facts and current storage capacity only; no
  model or performance claims.

## Artifact and privacy handling

- Committed evidence: schemas, synthetic fixtures, exact package lock,
  sanitized host facts, commands, and pass/fail verification.
- External artifacts: none created; the preflight writes and removes one tiny
  temporary probe.
- Private/local material: no usernames, serial numbers, volume UUIDs, tokens,
  or raw system profiles are committed.

## Decisions and discoveries

- 2026-07-24: The primary T9 root is mounted on local journaled HFS+ and has
  approximately 1.5 TiB free; capacity is mutable and will not be frozen as a
  benchmark.
- 2026-07-24: The invoking shell exposes Apple Python 3.9.6, below the project
  contract. `uv` will provision the pinned Python 3.11 toolchain for clean
  setup and tests.
- 2026-07-24: Heavy platform stacks remain separately owned. T01 pins the
  common repository tooling and records explicit handoff boundaries instead
  of inventing compatibility claims.
- 2026-07-25: Fresh review found that the first host schema encoded Apple and
  T9 facts as universal. The corrected schema has a portable base, a required
  Apple/NVIDIA/Qualcomm discriminator, mutually exclusive platform details,
  and separate local-external versus hosted-service storage shapes.
- 2026-07-25: `uv.lock` covers project runtime and development dependencies,
  not PEP 517 build requirements. Setuptools is now separately constrained by
  an exact, hash-bearing build lock consumed by the documented `uv build`
  command.
- 2026-07-25: Every exact-version field rejects comparison, compatible-release,
  comma-range, caret, and wildcard syntax. Storage validation resolves every
  expected directory and rejects symlink targets outside the artifact root.
- 2026-07-25: Fresh independent review reproduced all acceptance checks and
  returned a final pass with no remaining findings.

## Progress and restart instructions

T01 is complete. The reviewed implementation, completed worklog, archived
plan, task graph, and generated status are integrated. T30 remains gated by
blocked T02; T10 is the next ready model-path task.
