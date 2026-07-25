# T01: Environments, manifests, and storage

Date: 2026-07-24
Task: `T01`
Visibility: `public`
Status: in_progress

## Outcome

Established the repository's reproducible Python foundation, versioned
artifact and portable host evidence schemas, a sanitized Apple M4 host
manifest, and a safe external-storage preflight. Later export and deployment
tasks can now validate provenance and refuse heavy I/O when T9 is unmounted,
unwritable, incomplete, below its reserve, or contains a directory symlink
that escapes the approved root.

## Changes

- Pinned CPython 3.11.13, all direct common/development dependencies, and their
  complete hash-verified transitive solution in `uv.lock`.
- Added a separate exact, distribution-hashed setuptools build constraint
  because PEP 517 build requirements are outside the project lock. The
  documented build command consumes it with `--require-hashes`.
- Documented clean `uv` setup and explicit ownership boundaries for MLX,
  CUDA, AIMET, QAIRT, hosted runtimes, and profiling tools. Those versions are
  deferred until their owning tasks can smoke-test compatibility.
- Added packaged JSON Schemas for the stable T00 artifact envelope and
  portable sanitized host evidence, plus JSON/YAML loading, aggregated
  validation, and a command-line validator.
- Added valid Apple, NVIDIA/Linux, and hosted-Qualcomm host fixtures without
  cross-platform hardware or storage fabrication. Added invalid fixtures for
  ranges/wildcards, inconsistent tool status, malformed privacy, and
  platform-shape mismatches.
- Added valid and invalid representative artifact fixtures and regression
  tests that keep the artifact schema aligned with the T00 contract and reject
  non-exact exporter/runtime/QAIRT versions.
- Captured the project-plan Apple M4/macOS facts, exact common environment,
  verified local tools, and explicit null/deferred platform tools without
  serial numbers, accounts, UUIDs, credentials, or private paths.
- Added a storage preflight that resolves the root and every expected
  directory, rejects lexical and symlink escapes, validates mount containment,
  enforces a 100 GiB free-space reserve, and proves writability using one tiny
  temporary file that is fsynced and removed automatically.

## Verification

- Command: `uv lock --check`
- Result: passed; 28 packages resolve under `>=3.11.13,<3.12`.
- Command: `uv sync --extra dev --locked`
- Result: passed from a clean `.venv` using CPython 3.11.13; 27 packages
  installed from the locked solution.
- Command:
  `uv run --extra dev ruff check src/slm_lab/manifests tests/repo/test_manifests_and_storage.py`
- Result: passed.
- Command:
  `uv run --extra dev python -m unittest tests.repo.test_manifests_and_storage tests.repo.test_model_contract`
- Result: 23 focused tests passed with one intentional opt-in network test
  skipped.
- Commands:
  `uv run slm-lab-validate-manifest artifact tests/fixtures/manifests/artifact.valid.json`
  plus host validation for `results/hosts/apple-m4-primary.json`,
  `host.linux-nvidia.valid.json`, and `host.qualcomm-hosted.valid.json`.
- Result: all four representative manifests passed. The artifact-invalid,
  host-contract-invalid, and host-platform-shape-invalid fixtures all exited
  nonzero with the intended field-specific errors.
- Command: `uv run slm-lab-storage-preflight --json`
- Result: passed on `/Volumes/T9`; all 10 directories were present, the
  write/fsync/delete probe passed, and 1,677,834,244,096 free bytes exceeded
  the 107,374,182,400-byte reserve. Free space is mutable and is not benchmark
  evidence.
- Command:
  `.venv/bin/python -m unittest discover -s tests -p 'test_*.py'`
- Result after integrating current `main`: 42 tests passed with one
  intentional opt-in upstream-network test skipped.
- Command:
  `uv build --python 3.11.13 --build-constraints environments/build-requirements.lock --require-hashes --out-dir /tmp/slm-lab-dist`
- Result: the exact hash-enforced command built the source distribution and
  wheel successfully; both versioned JSON Schemas and command entry points are
  present in the wheel.
- Commands:
  `ruff check src tests`,
  `python3 scripts/ai/render_task_status.py --check`,
  `python3 scripts/repo/check_hygiene.py --all`, and `git diff --check`
- Result: all passed after current-main integration.

## Decisions and evidence

- The common lock is intentionally small; installing model or platform stacks
  in every contributor environment would be slow and would conflate metadata
  resolution with tested hardware compatibility.
- `uv.lock` is described only as the project runtime/development lock.
  `environments/build-requirements.lock` independently constrains setuptools
  80.9.0 with its published distribution hashes.
- Artifact manifests require every stable field from the project plan and T00.
  Non-applicable tool fields remain present as JSON null rather than being
  omitted or guessed.
- Host tool records pair null versions with a capture command and a
  `deferred` or `not_installed` reason. Verified entries require an exact
  version and no reason.
- The portable host schema requires exactly one platform extension. Apple-only
  neural-engine and unified-memory facts never appear in NVIDIA or Qualcomm
  shapes; hosted storage explicitly records the absence of an exposed
  filesystem.
- The storage command rejects the filesystem root, requires the artifact root
  and each resolved expected directory to remain beneath the configured mount,
  and never creates layout directories implicitly.

## Risks and limitations

- MLX, CUDA, AIMET, QAIRT, ONNX, PyTorch, Transformers, hosted runtime, and
  profiling versions are not yet compatibility-tested; their owning tasks
  must extend the host/environment evidence.
- The primary host facts originate from the approved project-plan machine
  inventory plus direct version/mount commands. No hardware benchmark was
  performed.
- The synthetic artifact fixture proves the schema, not the existence of its
  dummy artifact digest or a successful model export.
- Independent final review remains pending; task status intentionally remains
  `in_progress` until the coordinator completes that review.

## Follow-up

- Newly unblocked tasks: none until T01 is finalized; T30 also requires T02.
- Recommended next action: run the fresh independent T01 review, then archive
  the plan and mark the task/worklog completed if the reviewer passes it.
