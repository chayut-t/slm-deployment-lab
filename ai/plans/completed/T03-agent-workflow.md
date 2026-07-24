# T03: Task manifest, generated DAG, worktree, and GitHub conventions

Status: completed
Owner: Codex T03 agent
Updated: 2026-07-24

## Objective

Finish and verify the repository workflow that makes task delivery
dependency-aware, resumable across fresh clones, safe for private coordination
data, and mechanically consistent at commit time.

## Scope

### In scope

- Validate the existing task graph, definitions, generated status, agent rules,
  templates, hooks, hygiene checks, and local-state bootstrap.
- Add narrow regression tests that directly demonstrate all T03 acceptance
  criteria.
- Clarify clone/bootstrap and staged-snapshot behavior where documentation is
  ambiguous.
- Publish a completed T03 worklog, update the graph/status, and archive this
  plan.

### Out of scope

- Publishing or synchronizing GitHub Issues or Projects.
- Creating GitHub Actions workflows assigned to later tasks.
- Changing model, platform, benchmark, environment, or artifact contracts.
- Pushing, opening a pull request, or changing any external service.

## Dependencies and resources

- Required task dependencies: T00, completed at the branch base.
- Resource locks: none.
- External access: none.
- Cost boundary: local standard-library and Git checks only; no paid service.

## Important paths

- Inputs: `AGENTS.md`, `PLANS.md`, `docs/project/plan.md`,
  `ai/tasks/definitions/T03.yaml`, `ai/tasks/task_graph.yaml`.
- Outputs: `ai/tasks/status.generated.md`, workflow scripts and tests,
  `ai/worklogs/`, and this plan under `ai/plans/completed/`.
- Shared contracts: public/private coordination boundaries and task lifecycle
  validation.

## Milestones

- [x] Audit the existing scaffold and reproduce its focused tests.
- [x] Repair the drift regression so it tests a valid staged graph transition.
- [x] Demonstrate fresh-clone local-state reconstruction, including installed
  hooks and ignored private files.
- [x] Complete T03 metadata and public evidence.
- [x] Run focused tests, full tests, status validation, hygiene, and staged
  pre-commit validation.

## Verification and acceptance

- Commands:
  - `python3 -m unittest tests.repo.test_task_automation -v`
  - `python3 -m unittest discover -s tests -p 'test_*.py' -v`
  - `python3 scripts/ai/render_task_status.py --check`
  - `python3 scripts/repo/check_hygiene.py --all`
  - `python3 scripts/repo/check_hygiene.py --staged`
- Behavioral criteria:
  - A valid staged graph change with stale generated status is rejected.
  - Completion validation rejects unfinished dependencies and missing,
    private, or mismatched public worklogs.
  - A clean clone can install hooks and reconstruct ignored local
    coordination state without external storage.
- Hardware/profile evidence: not applicable.

## Artifact and privacy handling

- Committed evidence: regression tests, documentation, generated status,
  completed plan, and sanitized public worklog.
- External artifacts: none.
- Private/local material: `.ai-local/` remains ignored; no real task/thread IDs
  or unsanitized state are committed.

## Decisions and discoveries

- 2026-07-24: The scaffold already implements all three mechanisms, but the
  staged status-drift regression currently fails before exercising drift
  because it invalidates dependencies. Use a valid state transition instead.
- 2026-07-24: Validate reconstruction through the documented hook installer in
  a real local clone, not only by invoking the bootstrap helper in a copied
  directory.
- 2026-07-24: Independent review found that deletion-only staged snapshots were
  excluded by the `ACMR` diff filter and skipped whole-index validation. Include
  deletions, always validate required files and task state under `--staged`,
  and cover generated-status and referenced-worklog deletion explicitly.

## Progress and restart instructions

Implementation and lifecycle updates are complete. The focused and full test
suites pass, and the graph/status and public-worktree hygiene gates pass. If
review finds an issue, restart from the task-scoped commit, reproduce it with a
narrow regression, update this completed plan and the public worklog if the
resolution changes durable behavior, then rerun all completion gates.
