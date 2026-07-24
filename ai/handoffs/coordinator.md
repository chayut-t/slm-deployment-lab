# Coordinator handoff

Updated: 2026-07-24

## Current state

- `T00` is completed on `codex/T00-version-adr`: the immutable Qwen3-0.6B
  model/tokenizer revision, metadata hashes, model contract, version-evidence
  policy, ADR, tests, worklog, and completed execution plan are integrated.
- An independent review reproduced the public metadata hashes, identified an
  artifact-field mismatch, and passed the corrected contract and provenance
  tests on follow-up.
- `T03` remains planned, but its dependency is now satisfied. Agent rules, task
  graph, worklogs, local/private separation, and repository automation are
  already drafted and should be validated as part of T03.

## Ready work

- `T01`: repository environments, artifact schemas, and T9 preflight.
- `T02`: AI Hub, Device Cloud, and GPU access report with bounded toy jobs.
- `T03`: task manifest, generated DAG, worktree, and GitHub conventions.
- `T10`: token, prompt, and evaluation fixtures from the pinned tokenizer.

These tasks may proceed in separate Codex tasks and worktrees. Respect the
`t9_heavy_io`, `qai_hub_submission`, and `device_cloud_x_elite` resource locks.

## Resume

1. Read `AGENTS.md` and `docs/project/plan.md`.
2. Validate `ai/tasks/task_graph.yaml` with
   `python3 scripts/ai/render_task_status.py --check`.
3. Inspect `git status --short --ignored`.
4. Start from the commit completing T00.
5. Assign one ready task per branch/worktree and record real task/thread
   identity only in `.ai-local/tasks/thread-registry.yaml`.
