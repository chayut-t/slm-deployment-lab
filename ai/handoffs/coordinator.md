# Coordinator handoff

Updated: 2026-07-25

## Current state

- `T00` is completed on `codex/T00-version-adr`: the immutable Qwen3-0.6B
  model/tokenizer revision, metadata hashes, model contract, version-evidence
  policy, ADR, tests, worklog, and completed execution plan are integrated.
- An independent review reproduced the public metadata hashes, identified an
  artifact-field mismatch, and passed the corrected contract and provenance
  tests on follow-up.
- `T01` is completed on `codex/T01-environments-manifests`: the pinned common
  environment, portable artifact/host schemas, exact build lock, sanitized
  Apple host evidence, and hardened T9 preflight passed independent review.
- `T02` is blocked on Qualcomm Workbench and Device Cloud authentication. Its
  sanitized public-access evidence and executable restart procedure are
  committed on `codex/T02-platform-access`; no compile, inference, or profile
  job is claimed.
- `T03` is now completed on `codex/T03-agent-workflow`: regression tests
  demonstrate staged graph/status drift rejection, dependency and public
  worklog completion gates, and clean-clone reconstruction of ignored local
  coordination state. Independent review passed the deletion-snapshot fix.

## Active work

- `T02` is restartable from `codex/T02-platform-access` after the user signs
  into Qualcomm Workbench and Device Cloud.

T10 remains ready but unassigned. Respect the `t9_heavy_io`,
`qai_hub_submission`, and `device_cloud_x_elite` resource locks.

## Resume

1. Read `AGENTS.md` and `docs/project/plan.md`.
2. Validate `ai/tasks/task_graph.yaml` with
   `python3 scripts/ai/render_task_status.py --check`.
3. Inspect `git status --short --ignored`.
4. Start from the commit completing T00.
5. Integrate reviewed task branches in topological order, then assign one ready
   task per branch/worktree and record real task/thread
   identity only in `.ai-local/tasks/thread-registry.yaml`.
