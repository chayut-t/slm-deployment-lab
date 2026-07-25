# Agent operating rules

These instructions apply to the entire repository. A more specific
`AGENTS.md` in a subtree may add or refine rules for that subtree.

## Mission and priorities

Build a reproducible, educational SLM deployment lab that demonstrates the
complete engineering path from source model to graph contract, compiler,
runtime, numerical validation, hardware profile, and explanation.

The platform priority is:

1. Qualcomm public AI Hub and Device Cloud workflows.
2. Apple M4 and MLX on the current Mac mini.
3. NVIDIA CUDA through ONNX Runtime.

Use Qwen3-0.6B first. Read `docs/project/plan.md` before changing scope,
platform priorities, model choice, benchmark contracts, or acceptance criteria.

## Start-of-task protocol

Before substantial work:

1. Read this file, `docs/project/plan.md`, and the relevant task definition.
2. Read `docs/project/learning-checkpoints.md`. If the assigned task appears
   there, give the user a concise learning reminder at task start, call out
   concrete study or hands-on checkpoints during the work, and provide a
   linked study/debrief checklist at completion. Do not block implementation
   unless the checkpoint requires user authentication, hardware interaction,
   spending approval, or a subjective/public decision.
3. Inspect `git status` and preserve unrelated user or agent changes.
4. Check dependencies in `ai/tasks/task_graph.yaml`. Do not start a blocked
   task unless the work is explicitly a dependency-breaking investigation.
5. For a substantial or multi-session task, create an execution plan from
   `ai/plans/templates/execution-plan.md`.
6. When parallel work is useful, use one task-scoped branch and worktree per
   active writer. Do not rely on another worktree's uncommitted files.
7. Record private task/session ownership through
   `scripts/ai/session_registry.py`, which resolves the primary checkout's
   `.ai-local/tasks/thread-registry.yaml`. Never put real session IDs in public
   files.

The coordinating agent session is replaceable. Repository state must be
sufficient for a new Codex or Claude Code session to resume without private
conversation history.

Task definitions live at `ai/tasks/definitions/TNN.yaml` and specify objective,
owned paths, outputs, and acceptance criteria. The task graph records status,
dependencies, resource locks, public owner/branch coordination, and optional
GitHub issue linkage. Real agent session IDs remain local.

`AGENTS.md` is the canonical cross-tool policy. Every tracked `AGENTS.md`,
including a subtree-specific file, must have a same-directory `CLAUDE.md` that
imports it with `@AGENTS.md`. Claude-specific adapters must not duplicate
repository policy.

## Where files belong

| Material | Location | Git policy |
|---|---|---|
| Stable project scope and roadmap | `docs/project/` | Commit |
| Architecture and interfaces | `docs/architecture/` | Commit |
| Architecture decision records | `docs/decisions/` | Commit |
| Hardware-focused learning guides | `docs/learning/` | Commit |
| Reproducible failure analyses | `docs/failures/` | Commit after sanitizing |
| Polished agentic case studies | `docs/agentic/case-studies/` | Commit |
| Approved execution plans | `ai/plans/active/` or `ai/plans/completed/` | Commit |
| Task graph and public task definitions | `ai/tasks/` | Commit |
| Reusable prompts | `ai/prompts/` | Commit |
| Sanitized cross-task handoffs | `ai/handoffs/` | Commit |
| Curated engineering worklogs | `ai/worklogs/` | Commit |
| Draft plans and brainstorming | `.ai-local/plans/` | Never commit |
| Raw transcripts and worklogs | `.ai-local/worklogs/` | Never commit |
| Real agent task/session identifiers | `.ai-local/tasks/` | Never commit |
| Private inputs and feedback | `.ai-local/inputs/` | Never commit |
| Unsanitized cloud/profile output | `.ai-local/profiles/` | Never commit |
| Temporary experiments | `.ai-local/scratch/` | Never commit |
| Small reproducible metrics and plots | `results/` | Commit |
| Weights, ONNX files, binaries, and full traces | `artifacts/` | Never commit |

Do not create alternative directories such as `notes/`, `tmp/`, `agent/`, or
root-level plan files when an assigned location already exists.

## Source, documentation, and notebook rules

- Python packages live under `src/slm_lab/`; applications live under `apps/`.
- Reusable commands belong in `scripts/`; tests belong in `tests/`.
- Configuration is declarative and belongs in `configs/`. Environment setup
  belongs in `environments/`.
- Notebooks are learning and investigation surfaces, not the only
  implementation. Move reusable logic into `src/slm_lab/`.
- Learning material assumes transformer fundamentals. Focus on deployment
  engineering: graph lowering, cache layouts, GQA efficiency, quantization,
  compiler constraints, runtime scheduling, memory traffic, profiling, and
  numerical validation.
- Never fabricate benchmark data. Clearly label simulated, estimated, hosted,
  and real-device measurements.
- Record exact model/runtime/device revisions and commands needed to reproduce
  every published result.

## Task completion and worklogs

A task is substantial if it completes a task-graph node, changes runtime or
deployment behavior, changes a public contract, produces benchmark/profile
evidence, or takes enough investigation that another agent would benefit from
the reasoning.

After every substantial task:

1. Run the narrowest relevant tests and repository hygiene checks.
2. Create a worklog with:

   ```bash
   python3 scripts/ai/new_worklog.py --task TNN --slug short-description
   ```

3. Write the outcome, important changes, verification, decisions, unresolved
   risks, and next dependencies. Summarize reasoning; do not paste raw agent
   transcripts.
4. If the log contains private data, use `--visibility local` and publish a
   sanitized `ai/worklogs/` version before marking a public task complete.
5. Update the task's status and `worklog` field in
   `ai/tasks/task_graph.yaml`, then regenerate:

   ```bash
   python3 scripts/ai/render_task_status.py
   ```

6. Move a completed execution plan from `ai/plans/active/` to
   `ai/plans/completed/`.
7. Leave a sanitized handoff in `ai/handoffs/` when downstream work needs more
   than the task outputs and worklog provide.

Small typo fixes and purely mechanical formatting do not require a worklog.

`completed` means that all dependencies are completed, task-definition outputs
and acceptance criteria are satisfied, changes are integrated into the branch
downstream tasks will use, and the graph references a matching public worklog
under `ai/worklogs/`. Draft or merely local work is not completed.

## Git and multi-agent rules

- Use tool-neutral task branches such as `task/T31-qwen-workbench`. Historical
  `codex/TNN-*` branches remain valid and must not be renamed.
- Use one active writer per file set. Coordinate ownership before editing
  shared contracts, dependency manifests, or release documents.
- Create project task worktrees explicitly with Git from the committed public
  claim; do not rely on tool-specific implicit branch or worktree naming.
- Start downstream work from a commit containing its completed dependencies.
- Keep commits task-scoped and do not mix unrelated cleanup.
- Never stage, commit, discard, or rewrite another agent's changes.
- Do not push, publish a release, create public GitHub state, spend money, or
  submit paid cloud jobs without explicit authorization.
- Local commits are allowed only when the current task authorizes them. An
  instruction such as "do not commit" overrides the default workflow.

## Artifacts, secrets, and external services

- Use `SLM_LAB_ARTIFACT_ROOT` for large artifacts. On the primary machine it
  resolves to `/Volumes/T9/slm-deployment-lab`.
- Commit checksums, manifests, commands, and small evidence instead of large
  binaries.
- Keep tokens in ignored environment files or approved secret stores. Never
  print, log, notebook-save, or commit credentials.
- Sanitize Qualcomm job URLs, account details, private identifiers, and raw
  service responses before publication.
- External jobs must record target, runtime/compiler versions, configuration,
  estimated or actual cost, and artifact hashes.

## Required verification

Before presenting work as complete:

1. Run relevant unit/integration tests.
2. Run `python3 scripts/ai/render_task_status.py --check`.
3. Run `python3 scripts/repo/check_hygiene.py --all`.
4. Inspect `git status --short --ignored`.
5. Report what was verified, what was not run, and any environmental blocker.

The versioned pre-commit hook repeats task-status, privacy, secret, and
large-file checks for staged changes. Install it with
`scripts/setup/install_git_hooks.sh`.
