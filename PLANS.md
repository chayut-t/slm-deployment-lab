# Execution plans

Execution plans make substantial work resumable across agent sessions,
worktrees, tools, and context boundaries.

## When a plan is required

Create a plan for work that:

- spans multiple implementation or verification steps;
- is expected to continue across sessions;
- changes a shared contract or architecture;
- coordinates multiple agents or worktrees;
- involves external compilation, benchmarking, or profiling; or
- carries meaningful cost, compatibility, or data-loss risk.

A small, self-contained edit does not need its own execution plan.

## Locations and lifecycle

- Template: `ai/plans/templates/execution-plan.md`
- Active, approved plans: `ai/plans/active/`
- Completed plans: `ai/plans/completed/`
- Private drafts: `.ai-local/plans/`

Name committed plans `TNN-short-slug.md`. Keep plans self-contained and update
them as facts change. Do not use a plan as a raw diary; durable outcomes belong
in code, documentation, task manifests, decisions, and worklogs.

## Required plan content

Every committed execution plan includes:

- task ID and objective;
- in-scope and out-of-scope work;
- dependencies and required resource locks;
- important repository paths;
- implementation milestones;
- verification and acceptance criteria;
- artifact and privacy handling;
- decisions and discoveries;
- current progress and restart instructions.

## Source of truth

`docs/project/plan.md` controls stable project scope and priority.
`ai/tasks/task_graph.yaml` controls task dependencies and current task status.
Execution plans describe how one task or coordinated group will be completed.
GitHub Issues or Projects may mirror the task graph, but must not silently
override repository state.

## Active-plan index

- `T20`: `ai/plans/active/T20-prefill-concat-cache-write.md` — exporter fix
  committed; promotion into the reference artifacts is not authorized and not
  done
- `T31`: `ai/plans/active/T31-qwen-workbench.md` — three-target Workbench run
  plan built; submission evidence outstanding
- `T41`: `ai/plans/active/T41-w8-quantization-evidence.md` — candidate
  specifications frozen; waiting on `T34`

A plan leaves this index when it moves to `ai/plans/completed/`. `T02` was
listed here after its plan had already been completed and moved; the index is
now derived from the contents of `ai/plans/active/`.
