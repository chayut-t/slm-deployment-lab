# Public AI workspace

The `ai/` tree contains curated artifacts that make agent-assisted development
auditable and reproducible without publishing private conversations.

## Lifecycle

1. Select a ready task from `ai/tasks/task_graph.yaml`.
2. Create a task-scoped execution plan in `ai/plans/active/` when required.
3. Use a reusable prompt from `ai/prompts/` or record a new broadly useful one.
4. Implement and verify in an isolated branch/worktree when parallel work is
   active.
5. Record a sanitized worklog in `ai/worklogs/`.
6. Update the task graph and regenerate `ai/tasks/status.generated.md`.
7. Leave a handoff in `ai/handoffs/` when downstream agents need extra context.
8. Move the execution plan to `ai/plans/completed/`.

## Public versus private

Approved plans, reusable prompts, sanitized handoffs, task dependencies, and
curated engineering logs belong here. Draft thinking, raw transcripts, real
Codex task IDs, and unsanitized outputs belong under `.ai-local/`, which is
ignored by Git.
