# T04: Dual Agent Compatibility

Date: 2026-07-25
Task: `T04`
Visibility: `public`
Status: completed

## Outcome

Implemented a tool-neutral repository workflow for Codex and Claude Code.
`AGENTS.md` remains the canonical policy, a minimal `CLAUDE.md` imports it,
private state stays ignored, and task sessions coordinate through one
primary-checkout registry with locked atomic mutations and Git-backed
compare-and-swap checks.

## Changes

- Added the Claude Code instruction adapter, privacy exclusions, and mechanical
  root/nested adapter checks.
- Made forward-looking coordination language agent-neutral while preserving
  historical Codex provenance.
- Added the dual-agent setup, manual worktree, parallel-writing, handoff,
  recovery, and registry documentation.
- Added schema-v1/v2 registry reads, conservative and idempotent migration,
  primary-checkout resolution, atomic initialization, writer lifecycle,
  reviewer lifecycle, and exact branch/worktree/checkpoint validation.
- Added regressions for private force-adds, adapter drift, registry invariants,
  migration refusal, concurrent initialization, linked-worktree bootstrap, and
  a real temporary Git claim/checkpoint/transfer/release lifecycle.

## Verification

- `python3 -m pytest tests/repo/test_session_registry.py
  tests/repo/test_task_automation.py -q`
  - 36 passed.
- Project Python 3.11: `python -m pytest -q`
  - Final merged branch: 81 passed, 2 hardware-dependent tests skipped.
- `python scripts/ai/render_task_status.py --check`
  - Task graph valid; 30 tasks; generated status current.
- `python scripts/repo/check_hygiene.py --all`
  - Final merged branch: passed for 169 tracked and untracked public files.
- `ruff check --no-cache .` and targeted `ruff format --check`
  - Passed for all code; all T04 Python files formatted.
- Two isolated temporary clones at
  `2d96afcc1b7465bf713c37685b99b96497f60257`
  - Both reconstructed ignored schema-v2 state, installed hooks, passed task
    status and hygiene, and showed only `.ai-local/` as ignored local state.
- Claude Code `2.1.220`, non-interactive `/context`
  - Listed both project memory files: `CLAUDE.md` and imported `AGENTS.md`.
- Codex CLI `0.146.0-alpha.3.1`; Codex desktop/subagent review
  - The active task and fresh independent reviewer both loaded `AGENTS.md` and
    used the project plan, task graph, and required completion checks.

## Decisions and evidence

- The committed graph remains authoritative; the private registry supplements
  it and cannot claim tasks that are not publicly `in_progress` with owner and
  branch.
- Active registry mutations require a clean registered worktree on the public
  branch at an exact 40-character commit SHA.
- The previous inline-empty schema-v1 form remains readable.
- Migration changes only explicitly completed v1 entries. Missing, unknown,
  active, or otherwise ambiguous states are refused without mutation.
- No registry migration was performed because live tasks remained present.
- A fresh independent implementation review identified nine safety and
  evidence gaps. The implementation added the missing invariants, Git checks,
  atomic initialization, reviewer release, regressions, and corrected
  instruction-inspection guidance before final review.
- Follow-up review found and reproduced two remaining issues: uncommitted graph
  or unrelated-clone claims, and an impossible different-path transfer. Claims
  now read clean committed graph state, verify the claim inside the checkpoint,
  and require a registered linked worktree. Transfer now validates a clean
  detached outgoing worktree before attaching the branch at the same commit in
  the incoming worktree.
- The final independent review reported no actionable findings. Its focused
  registry suite, merged full suite, task-status, hygiene, Ruff, formatting,
  diff, and clean-status checks all passed.

## Risks and limitations

- The authenticated Claude semantic orientation prompt could not run because
  the installed CLI reported that it was not logged in. The local `/context`
  command still verified the complete memory-file import chain.
- A separate Codex CLI semantic prompt was not authorized because it would
  transmit repository coordination content to a remotely authenticated
  service. Equivalent Codex orientation was demonstrated by this desktop task
  and its fresh read-only review subagent.
- Authenticated semantic prompts remain optional evidence; the repository
  contract and local instruction-loading evidence do not depend on tool login.

## Follow-up

- Newly unblocked tasks: none; T04 is intentionally not a dependency of
  existing SLM tasks.
- Recommended next action: use
  `docs/agentic/dual-agent-setup.md` when assigning the next ready task, and
  record its private writer through the shared registry helper.
