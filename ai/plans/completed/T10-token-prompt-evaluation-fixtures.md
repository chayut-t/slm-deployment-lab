# T10: Token, prompt, and evaluation fixtures

Status: completed
Owner: Codex T10 agent
Updated: 2026-07-25

## Objective

Freeze deterministic, inspectable inputs for Qwen3-0.6B so numerical,
benchmark, export, quantization, and hardware tasks consume identical token
sequences. The result must make tokenizer drift obvious, cover all four static
contexts, and keep private or redistribution-restricted evaluation data out of
Git.

## Scope

### In scope

- Add an exact optional tokenizer dependency and lock it for reproducible
  fixture generation and verification.
- Define self-authored raw-completion canaries and one explicitly non-thinking
  chat-template canary.
- Generate exact 128, 512, 1,024, and 4,096-token workloads from the pinned
  Qwen tokenizer revision.
- Record prompt text, token IDs, generation settings, source revisions, tool
  versions, and canonical content hashes.
- Provide reusable loading, validation, hashing, generation, and upstream
  verification logic under `src/slm_lab/evaluation/`.
- Record quality-evaluation dataset references and selection policies without
  committing restricted dataset rows.
- Add tests for deterministic regeneration, hash drift, context coverage,
  special-token behavior, and license/privacy boundaries.

### Out of scope

- Model-weight download or inference.
- Golden logits and deterministic model outputs, which belong to T11.
- Final benchmark statistics and academic task selection, which belong to T13.
- Calibration corpus materialization, which belongs to T40.
- Any private prompt, licensed dataset row, or Hugging Face credential.

## Dependencies and resources

- Required task dependencies: T00, completed on the branch base.
- Resource locks: none.
- External access: public pinned Qwen tokenizer files from Hugging Face.
- Cost boundary: no paid resources and no cloud jobs.

## Important paths

- Inputs: `configs/models/qwen3-0.6b.yaml`,
  `docs/decisions/0001-model-and-version-pins.md`,
  `docs/project/plan.md`.
- Outputs: `configs/workloads/`, `tests/fixtures/t10/`,
  `src/slm_lab/evaluation/`, `docs/learning/token_prompt_fixtures.md`.
- Shared contracts: `pyproject.toml`, `uv.lock`,
  `ai/tasks/task_graph.yaml`, `ai/tasks/status.generated.md`.

## Milestones

- [x] Exact tokenizer environment is locked and can load the immutable revision.
- [x] Canary prompts and all four exact-length workload fixtures regenerate.
- [x] Fixture and dataset-reference manifests validate and detect drift.
- [x] Focused and full repository verification pass.
- [x] Public worklog, completed plan, and task graph record T10 completion.
- [x] Fresh independent reviewer reports no unresolved findings.

## Verification and acceptance

- Commands:
  - `uv sync --extra dev --extra tokenizer --locked`
  - `uv run --extra tokenizer slm-lab-fixtures verify`
  - `uv run --extra dev --extra tokenizer pytest -q`
  - `uv run --extra dev ruff check src tests`
  - `python3 scripts/ai/render_task_status.py --check`
  - `python3 scripts/repo/check_hygiene.py --all`
- Numerical or behavioral criteria:
  - Every stored token sequence exactly matches a fresh encoding from the
    pinned tokenizer and immutable revision.
  - Workload token counts are exactly 128, 512, 1,024, and 4,096.
  - Canonical SHA-256 values fail validation after any covered-content change.
  - Raw completion adds no implicit BOS token; the chat canary explicitly
    disables thinking.
- Hardware/profile evidence: not applicable.

## Artifact and privacy handling

- Committed evidence: authored prompt text, token IDs, manifests, hashes,
  dataset identities/revisions, and test fixtures.
- External artifacts: public tokenizer cache only; no model weights.
- Private/local material: cache paths and any local Hugging Face state remain
  ignored and are never printed or committed.

## Decisions and discoveries

- 2026-07-25: Start from published `origin/main` in an isolated worktree
  because unrelated T04 coordination commits are present on local `main`.
- 2026-07-25: Use raw completion as the canonical workload interface and keep
  chat formatting limited to a dedicated non-thinking canary, matching ADR
  0001.
- 2026-07-25: Add a task-local learning guide because tokenization is a deep
  study checkpoint and the related notebook is intentionally deferred to T80.
- 2026-07-25: Qwen's pinned template renders an empty `<think>…</think>` block
  even with thinking disabled. Preserve those control tokens instead of
  treating non-thinking mode as template-marker removal.
- 2026-07-25: Independent review required whole-config provenance checks and
  an explicit deterministic generation contract. Offline validation now
  derives and compares the full manifest, while upstream verification remains
  the authority for token-array reproduction.

## Progress and restart instructions

T10 is complete on `codex/T10-token-fixtures`. The implementation, independent
review corrections, public worklog, completed plan, and task metadata are
ready for integration. T11 and T13 should start from the integrated T10
commit and consume the committed fixture IDs and generation policy unchanged.
