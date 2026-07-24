# T00: Model and version contract

Status: completed
Owner: Codex
Updated: 2026-07-24

## Objective

Freeze the Qwen3-0.6B source model and tokenizer to an immutable upstream
revision, define the model/runtime assumptions shared by downstream tasks, and
record a toolchain-version policy that makes later artifacts and measurements
reproducible.

## Scope

### In scope

- Pin the primary Hugging Face model and tokenizer to a full commit SHA.
- Record model architecture, tokenizer, chat-template, special-token, dtype,
  thinking-mode, and loading decisions required by the project plan.
- Define mandatory version fields and capture rules for common, Qualcomm,
  Apple, and NVIDIA workflows.
- Add the formal architecture decision record and declarative model config.
- Complete the T00 public worklog and task-graph state.

### Out of scope

- Downloading model weights or large tokenizer artifacts.
- Selecting and locking installable Python/toolchain package versions; T01
  owns environment definitions and `uv.lock`.
- Proving public cloud/device access; T02 owns access discovery.
- Defining prompt fixtures, graph tensor contracts, or benchmark tolerances.

## Dependencies and resources

- Required task dependencies: none.
- Resource locks: none.
- External access: read-only public Hugging Face repository metadata.
- Cost boundary: no paid services or jobs; zero expected cost.

## Important paths

- Inputs:
  - `AGENTS.md`
  - `docs/project/plan.md`
  - `ai/tasks/definitions/T00.yaml`
- Outputs:
  - `docs/decisions/0001-model-and-version-pins.md`
  - `configs/models/qwen3-0.6b.yaml`
- Shared contracts:
  - `ai/tasks/task_graph.yaml`
  - `ai/tasks/status.generated.md`
  - `ai/worklogs/`

## Milestones

- [x] Resolve and independently verify the immutable upstream revision.
- [x] Hash the pinned model config, tokenizer config, and extracted chat
  template without downloading weights.
- [x] Write the ADR and declarative model/version contract.
- [x] Add narrow contract validation and complete the public task record.
- [x] Pass required repository tests and hygiene checks.
- [x] Complete independent subagent review and resolve actionable findings.

## Verification and acceptance

- Commands:
  - `git ls-remote https://huggingface.co/Qwen/Qwen3-0.6B refs/heads/main`
  - `python3 -m unittest discover -s tests -p 'test_*.py'`
  - `python3 scripts/ai/render_task_status.py --check`
  - `python3 scripts/repo/check_hygiene.py --all`
- Behavioral criteria:
  - Model and tokenizer use the same full 40-character immutable revision.
  - Recorded SHA-256 values reproduce from files at that revision.
  - The config contains every version field required by the ADR.
  - Scope, platform priority, context sizes, and fallback policy agree with
    `docs/project/plan.md`.
- Hardware/profile evidence: not applicable; T00 performs no model execution.

## Artifact and privacy handling

- Committed evidence: immutable revision, small metadata hashes, ADR, model
  config, execution plan, generated task status, and sanitized worklog.
- External artifacts: none.
- Private/local material: real Codex task ownership remains only in
  `.ai-local/tasks/thread-registry.yaml`.

## Decisions and discoveries

- 2026-07-24: Hugging Face `refs/heads/main` resolved to
  `c1899de289a04d12100db370d81485cdf75e47ca`; the full SHA is pinned instead of
  the floating branch name.
- 2026-07-24: The pinned repository uses one revision for model and tokenizer.
- 2026-07-24: Exact installable toolchain versions remain a T01 responsibility.
  T00 defines the mandatory capture schema and forbids floating versions in
  reproducible evidence.
- 2026-07-24: Independent review found that the first draft's nested evidence
  paths conflicted with the plan's flat artifact schema. The config and ADR now
  preserve the plan fields exactly and use flat names for platform extensions.
- 2026-07-24: Added an opt-in live provenance test after review identified that
  the original offline tests checked internal consistency but not upstream
  provenance.

## Progress and restart instructions

T00 is complete. The ADR, model contract, offline and opt-in provenance tests,
public worklog, and task status are finalized. T01, T02, T03, and T10 may start
from the commit containing this plan.
