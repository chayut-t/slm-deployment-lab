# T40: AIMET and calibration environment

Date: 2026-08-02
Task: `T40`
Visibility: `public`
Status: completed

## Outcome

The quantization lane now has frozen inputs. A Linux AIMET environment is
pinned to an exact, hash-locked, reproducible resolution; the calibration
corpus is a hash-frozen fixed point regenerated from committed CC0 material;
and the floating baseline that every future quality delta will be compared
against is re-verified before any weight is quantized.

Nothing was quantized. AIMET is Linux + CUDA only and cannot run on the
primary macOS host, so this task specifies and verifies the environment
rather than executing it. That boundary is stated in every artifact.

## Changes

- `environments/linux-aimet/` — replaces the stub. `aimet-requirements.in`
  holds the direct pins; `aimet-requirements.lock` is a complete 104-package
  hash-pinned transitive resolution produced by `uv pip compile
  --python-platform x86_64-manylinux_2_34 --python-version 3.11
  --generate-hashes`, reproduced byte-identically on three separate runs.
  `aimet-cuda-wheels.lock` overlays the GitHub-release `+cu126`/`+cpu` wheels,
  which are never published to PyPI and so cannot appear in a resolver lock.
  `aimet-host.template.json` is a template, explicitly not evidence.
- `configs/quantization/calibration.yaml` — the generated, hash-frozen corpus
  contract: 13 tier-1 samples, 6,912 tokens, per-sample rationale, a computed
  coverage block, licensing, and the `calibration_dataset_revision` that T41+
  must copy into every artifact manifest.
- `src/slm_lab/quantization/calibration.py` — deterministic corpus
  construction, prefill-tensor emission against the T12 contract, and a
  validator that rejects drift in any pinned input, hash, count, or licence.
- `src/slm_lab/quantization/parity.py` — the pre-quantization baseline parity
  preflight and its evidence recorder.
- `tests/quantization/` — 94 tests.
- `results/quantization/` — the committed parity record and its README.
- `docs/learning/calibration_and_aimet.md` — the study surface for the T40
  learning checkpoint.

## Verification

- Command: `uv run pytest tests`
  Result: 388 passed, 13 skipped. Pre-T40 baseline was 295 passed, 12 skipped;
  the delta is exactly this task's tests.
- Command: `uv run pytest tests/quantization`
  Result: 93 passed, 1 skipped (the skip is the opt-in 9.6 GB re-hash). With
  `SLM_LAB_T40_VERIFY_ARTIFACT_BYTES=1`, 94 passed in 12.8 s.
- Command: `uv run python -m slm_lab.quantization.calibration check`
  Result: passed; the committed contract is a byte-identical fixed point of
  its generator.
- Command: `uv run python -m slm_lab.quantization.calibration verify --online`
  Result: passed; all three tier-2 dataset revisions and card licences still
  match upstream.
- Command: `uv run python -m slm_lab.quantization.parity verify`
  Result: exit 0, `verdict: partial (scope: artifact_identity_only)`. All 16
  T20 ONNX files, 9,586,211,364 bytes, re-hashed from the external volume and
  matching the committed digests exactly.
- Command: `uv run ruff format --check` / `ruff check` on the changed paths
  Result: clean.
- Command: `uv run python scripts/ai/render_task_status.py --check`
  Result: task graph valid, generated status current.
- Command: `uv run python scripts/repo/check_hygiene.py --all`
  Result: passed.

Not run: anything requiring `torch`, `onnx`, `onnxruntime`, `numpy`, or
`transformers` — none are installed and none were installed. AIMET itself was
never executed.

## Decisions and evidence

- **AIMET is not a `pyproject.toml` extra.** `aimet-onnx` publishes a single
  `manylinux_2_34_x86_64` wheel, so an extra would break `uv sync` on the
  primary macOS host. The stack is pinned under `environments/`, where
  platform extensions already live.
- **The corpus is two-tier.** Tier 1 is derived from the committed CC0-1.0 T10
  fixtures and is the only tier committed or required. Tier 2 records three
  public corpora by real repository revision and real card licence, with
  `data_committed: false`, mirroring T10's `external_quality_candidates`
  pattern. Recording identifiers rather than rows is what keeps the repository
  Apache-2.0 clean under CC-BY-SA-3.0, GFDL, and ODC-BY obligations.
- **Short fixtures are tiled, never padded.** The T12 prefill contract states
  that `attention_mask` is "One for every real prompt token; padding is not
  permitted." Padding would emit a tensor the exported graph does not accept
  and would teach the observers pad-embedding statistics that never occur at
  inference. Each sample is therefore emitted at exactly one frozen prompt
  length.
- **Coverage is measured, not asserted.** The contract carries a computed
  `coverage` block: the corpus touches 175 of 151,936 embedding rows (0.115%),
  the four context workloads are strict token-ID prefixes of one repeated T10
  seed, and they carry 83% of the token budget. Review found the first draft's
  rationales implied vocabulary breadth the corpus does not have; they were
  rewritten to claim only what they establish.
- **Baseline parity is split and scoped.** The artifact-identity half ran here
  and verified. The numerical logit half needs `torch` and `onnxruntime`, is
  recorded as `not_run` with its owner (T21) and its exact command, and
  `overall_verdict` has no branch that can emit `verified` — a property pinned
  by tests that fail under mutation.
- **"AIMET 2.36.0" is ambiguous.** The PyPI distribution declares
  `torch==2.12.*` under `v1-deps` and records `2.12.1+cu126`; the GitHub
  `+cpu`/`+cu126` wheels declare `torch==2.13.*`. No functional impact here
  because neither is installed with its dependencies, but a run manifest must
  record which distribution was installed, not only the release number.

## Risks and limitations

- 6,912 calibration tokens is small for post-training quantization, and 83% of
  the budget comes from one repeated seed. Widening coverage is tier 2's
  purpose and T41's decision.
- Four tier-1 samples are both calibrated on and evaluated on. This biases any
  quality delta optimistically and T41-T43 must report it rather than present
  the delta as generalization evidence.
- Tiled samples are valid token sequences but not valid tokenizations of any
  string; they must never be reused as quality or latency workloads.
- The AIMET environment is verified as a resolution, not as an installation.
  No wheel was installed and the smoke test was derived from the shipped
  wheel's symbols rather than executed. The README separates verified from
  unverified facts and gives the command for each unverified one.
- The committed parity record was generated on a dirty working tree
  (`git_tree_clean: false`), which it states. Its inputs are committed files
  and pinned external artifacts, so the measurement stands; a post-commit
  re-run would flip the flag.
- `onnxruntime-gpu` is deliberately not hash-pinned because the correct build
  depends on host CUDA and cuDNN minors that cannot be determined from here.

## Follow-up

- Newly unblocked tasks: T41 (W8 quantization evidence), and the calibration
  inputs T42/T43 inherit through it.
- Recommended next action: T41 provisions the Linux AIMET host from
  `environments/linux-aimet/`, fills the host manifest template, and closes
  the numerical parity half that T21 owns before publishing any W8 quality
  delta.
