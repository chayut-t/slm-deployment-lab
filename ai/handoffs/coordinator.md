# Coordinator handoff

Updated: 2026-07-24

## Current state

- `T00` is planned: scope and priorities are drafted, but immutable version
  pins and the formal ADR are not complete.
- `T03` is planned and dependency-blocked by `T00`: agent rules, task graph,
  worklogs, local/private separation, and repository automation are drafted
  for review.
- The repository is intentionally uncommitted pending user review.

## Ready work

- `T00`: immutable version pins and the scope/version ADR.

After T00 is integrated, T01, T02, T03, and T10 become logically ready and can
proceed in separate Codex tasks and worktrees.

## Resume

1. Read `AGENTS.md` and `docs/project/plan.md`.
2. Validate `ai/tasks/task_graph.yaml` with
   `python3 scripts/ai/render_task_status.py --check`.
3. Inspect `git status --short --ignored`.
4. Assign one ready task per branch/worktree and record real task/thread
   identity only in `.ai-local/tasks/thread-registry.yaml`.
