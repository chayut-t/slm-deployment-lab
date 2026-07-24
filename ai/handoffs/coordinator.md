# Coordinator handoff

Updated: 2026-07-24

## Current state

- `T00` is completed on `codex/T00-version-adr`: the immutable Qwen3-0.6B
  model/tokenizer revision, metadata hashes, model contract, version-evidence
  policy, ADR, tests, worklog, and completed execution plan are integrated.
- An independent review reproduced the public metadata hashes, identified an
  artifact-field mismatch, and passed the corrected contract and provenance
  tests on follow-up.
- `T01`, `T02`, and `T03` are in progress on isolated task branches. T03's
  agent rules, task graph, worklogs, local/private separation, and repository
  automation were drafted earlier and are now being validated.

## Active work

- `T01` on `codex/T01-environments-manifests`.
- `T02` on `codex/T02-platform-access`.
- `T03` on `codex/T03-agent-workflow`.

T10 remains ready but unassigned. Respect the `t9_heavy_io`,
`qai_hub_submission`, and `device_cloud_x_elite` resource locks.

## Resume

1. Read `AGENTS.md` and `docs/project/plan.md`.
2. Validate `ai/tasks/task_graph.yaml` with
   `python3 scripts/ai/render_task_status.py --check`.
3. Inspect `git status --short --ignored`.
4. Start from the commit completing T00.
5. Assign one ready task per branch/worktree and record real task/thread
   identity only in `.ai-local/tasks/thread-registry.yaml`.
