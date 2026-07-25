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
- `T02` is completed after fresh independent review. One authenticated, free
  Workbench compile, inference, and profile lifecycle on Snapdragon X Elite
  CRD produced a QAIRT 2.45 QNN context binary, numerically correct output,
  127-microsecond toy latency, NPU placement, and zero paid-resource use.
- `T03` is now completed on `codex/T03-agent-workflow`: regression tests
  demonstrate staged graph/status drift rejection, dependency and public
  worklog completion gates, and clean-clone reconstruction of ignored local
  coordination state. Independent review passed the deletion-snapshot fix.
- `T10` is completed on `codex/T10-token-fixtures`: the exact Qwen tokenizer
  reproduces committed raw/chat canaries and 128/512/1,024/4,096-token
  workloads; greedy generation and stopping semantics, CC0 quality cases,
  privacy/licensing boundaries, hashes, CLI validation, tests, and a learning
  guide passed independent review and re-review.
- `T04` is completed on `task/T04-dual-agent-compatibility`: Codex and Claude
  Code now share canonical `AGENTS.md` policy through a thin `CLAUDE.md`
  adapter, private state is protected, and linked worktrees coordinate through
  a locked registry with committed-graph and exact-checkpoint validation.
  Independent review and re-review passed the final unsafe-claim and
  path-changing-transfer regressions.

## Active work

T11, T13, T30, and T32 are ready but unassigned. T11 and T13 must consume T10's
fixture IDs and generation policy rather than create new prompts. Device Cloud
account minutes/session access remain a bounded T32-owned boundary. Respect
the `t9_heavy_io`, `qai_hub_submission`, and `device_cloud_x_elite` resource
locks.

## Resume

1. Read `AGENTS.md`, `docs/project/plan.md`, and
   `docs/agentic/dual-agent-setup.md`.
2. Validate `ai/tasks/task_graph.yaml` with
   `python3 scripts/ai/render_task_status.py --check`.
3. Inspect `git status --short --ignored`.
4. Start T11 and T13 from the commit completing T10; other tasks start from
   commits containing all of their completed dependencies.
5. Integrate reviewed task branches in topological order, then assign one ready
   task per branch/worktree and record real session identity through
   `scripts/ai/session_registry.py`; the resolved registry remains private.
