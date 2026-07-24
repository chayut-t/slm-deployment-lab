# T01: Environments, manifests, and storage

Date: 2026-07-24
Task: `T01`
Visibility: `public`
Status: in_progress

## Outcome

Established the repository's reproducible Python foundation, versioned
artifact and host evidence schemas, a sanitized Apple M4 host manifest, and a
safe external-storage preflight. Later export and deployment tasks can now
validate provenance and refuse heavy I/O when T9 is unmounted, unwritable,
incomplete, or below its reserve.

## Changes

- Pinned CPython 3.11.13, all direct common/development dependencies, and the
  complete hash-verified transitive solution in `uv.lock`.
- Documented clean `uv` setup and explicit ownership boundaries for MLX,
  CUDA, AIMET, QAIRT, hosted runtimes, and profiling tools. Those versions are
  deferred until their owning tasks can smoke-test compatibility.
- Added packaged JSON Schemas for the stable T00 artifact envelope and
  sanitized host evidence, plus JSON/YAML loading, aggregated validation, and
  a command-line validator.
- Added valid and invalid representative artifact fixtures and regression
  tests that keep the artifact schema aligned with the T00 contract.
- Captured the project-plan Apple M4/macOS facts, exact common environment,
  verified local tools, and explicit null/deferred platform tools without
  serial numbers, accounts, UUIDs, credentials, or private paths.
- Added a storage preflight that validates mount containment, the expected
  directory layout, a 100 GiB free-space reserve, and writability using one
  tiny temporary file that is fsynced and removed automatically.

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
- Result: 18 tests passed with one intentional opt-in network test skipped.
- Commands:
  `uv run slm-lab-validate-manifest artifact tests/fixtures/manifests/artifact.valid.json`
  and
  `uv run slm-lab-validate-manifest host results/hosts/apple-m4-primary.json`
- Result: both representative manifests passed.
- Command: `uv run slm-lab-storage-preflight --json`
- Result: passed on `/Volumes/T9`; all 10 directories were present, the
  write/fsync/delete probe passed, and 1,677,834,244,096 free bytes exceeded
  the 107,374,182,400-byte reserve. Free space is mutable and is not benchmark
  evidence.
- Command:
  `uv run --extra dev python -m unittest discover -s tests -p 'test_*.py'`
- Result: 27 tests ran; all T01 and T00 tests passed, one opt-in test skipped,
  and one inherited T03 automation assertion
  failed because the shared start commit marks downstream tasks in progress,
  so its staged-snapshot mutation now encounters an incomplete-dependency
  validation first. T01 did not edit T03-owned tests; the coordinator is
  integrating the T03 fix before independent T01 review.
- Command: `uv build --out-dir /private/tmp/slm-lab-T01-dist-final`
- Result: source distribution and wheel built successfully with both versioned
  JSON Schemas included in the wheel.

## Decisions and evidence

- The common lock is intentionally small; installing model or platform stacks
  in every contributor environment would be slow and would conflate metadata
  resolution with tested hardware compatibility.
- Artifact manifests require every stable field from the project plan and T00.
  Non-applicable tool fields remain present as JSON null rather than being
  omitted or guessed.
- Host tool records pair null versions with a capture command and a
  `deferred` or `not_installed` reason. Verified entries require an exact
  version and no reason.
- The storage command rejects the filesystem root, requires the artifact root
  to resolve beneath the configured mounted volume, and never creates layout
  directories implicitly.

## Risks and limitations

- MLX, CUDA, AIMET, QAIRT, ONNX, PyTorch, Transformers, hosted runtime, and
  profiling versions are not yet compatibility-tested; their owning tasks
  must extend the host/environment evidence.
- The primary host facts originate from the approved project-plan machine
  inventory plus direct version/mount commands. No hardware benchmark was
  performed.
- The synthetic artifact fixture proves the schema, not the existence of its
  dummy artifact digest or a successful model export.
- The full shared suite retains the inherited T03 test failure described
  above until the coordinator integrates T03's correction.

## Follow-up

- Newly unblocked tasks: none until T01 is finalized; T30 also requires T02.
- Recommended next action: integrate T03's test correction, run a fresh
  independent T01 review and the full shared suite, then archive the plan and
  mark the task/worklog completed.
