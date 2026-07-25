# Codex and Claude Code setup

This repository supports Codex and Claude Code through one versioned workflow.
Neither tool's private conversation history is a project dependency.

## Shared sources of truth

- `AGENTS.md` is the canonical operating policy.
- `CLAUDE.md` is a thin Claude Code adapter that imports `AGENTS.md`.
- `docs/project/plan.md` controls stable project scope and priorities.
- `ai/tasks/task_graph.yaml` controls public dependencies, status, ownership,
  branches, resource locks, and worklogs.
- The primary checkout's ignored
  `.ai-local/tasks/thread-registry.yaml` records private session details.

If a subtree adds an `AGENTS.md`, add a same-directory `CLAUDE.md` containing
`@AGENTS.md`. Repository hygiene checks enforce this mapping.

## Fresh-clone setup

Run:

```bash
scripts/setup/install_git_hooks.sh
python3 scripts/ai/render_task_status.py --check
python3 scripts/repo/check_hygiene.py --all
```

The installer reconstructs ignored local state. New registries use schema v2.
An existing schema-v1 registry remains readable and is never migrated
automatically.

For Claude Code, use interactive `/memory` to inspect loaded project memory.
For a non-interactive smoke check, `/context` also lists the Memory Files;
confirm that `CLAUDE.md` and imported `AGENTS.md` both appear. For Codex, ask a
read-only orientation question and confirm it identifies `AGENTS.md`, the
project plan, the task graph, dependency checks, and completion gates.

Keep these local:

- `CLAUDE.local.md`
- `.claude/settings.local.json`
- `.claude/worktrees/`
- credentials, OAuth/browser state, raw transcripts, and auto-memory
- real Codex or Claude Code session identifiers

Do not commit shared MCP servers or broad permission allowlists until a real
task requires them and their security boundary has been reviewed.

## Starting a task

1. Run the non-disruption preflight from the task execution plan.
2. Confirm dependencies are completed and resource locks are available.
3. Register the public task claim on the coordination/integration checkout:
   set `in_progress`, owner, and intended branch together, regenerate status,
   validate, and commit.
4. Create the worktree from that exact claim commit:

   ```bash
   git worktree add /explicit/task/path \
     -b task/TNN-short-slug CLAIM_SHA
   ```

5. Launch either tool inside the new worktree.
6. Record the private writer session in the shared registry before editing.

Use manual Git worktree creation for project tasks. Claude Code's implicit
worktree naming and base selection are not this repository's task contract.
Historical `codex/TNN-*` branches remain valid and must not be renamed.

## Shared registry

From any linked worktree:

```bash
python3 scripts/ai/session_registry.py path
python3 scripts/ai/session_registry.py validate
python3 scripts/ai/session_registry.py show
```

The helper resolves the primary checkout from `git worktree list --porcelain`.
Set `SLM_LAB_COORDINATION_ROOT` only when an explicit primary checkout is
needed; it must resolve to that repository's primary checkout exactly.

Schema-v2 writer changes are locked, atomic, and compare-and-swap guarded.
Examples:

```bash
python3 scripts/ai/session_registry.py claim TNN \
  --session PRIVATE_SESSION \
  --tool codex \
  --worktree /explicit/task/path \
  --checkpoint CLAIM_SHA

python3 scripts/ai/session_registry.py checkpoint TNN \
  --writer PRIVATE_SESSION \
  --expected-checkpoint OLD_SHA \
  --new-checkpoint NEW_SHA

python3 scripts/ai/session_registry.py release TNN \
  --expected-writer PRIVATE_SESSION \
  --expected-checkpoint FINAL_SHA

python3 scripts/ai/session_registry.py release-reviewer TNN \
  --reviewer PRIVATE_REVIEW_SESSION \
  --expected-state active
```

Never place real session values in commands copied to public logs or worklogs.

Schema-v1 migration is explicit:

```bash
python3 scripts/ai/session_registry.py migrate
```

Migration refuses active or ambiguous sessions, creates a private backup, and
does not modify malformed or unknown schemas.

## Parallel work

- One active writer owns a task and file set.
- Each writer uses a dedicated worktree.
- Read-only reviewers may coexist but cannot become writers implicitly.
- Existing SLM tasks do not depend on the compatibility task.
- Non-overlapping tasks may continue while compatibility work is isolated.
- Shared coordination paths require a short exclusive edit window.

Treat a registry/graph disagreement as occupied state. A clean worktree, idle
UI, or missing API response is not permission to mark another session stale.
Only its owner or the user may reconcile it.

## Cross-tool handoff

Prefer stopping the outgoing tool and launching the incoming tool in the same
task worktree. If the path must change, verify a clean checkpoint, detach the
old worktree at that exact commit, attach the branch to the new registered
worktree, perform the registry transfer, and only then remove the detached old
worktree. The transfer validates both paths and never requires one branch to be
checked out twice.

A handoff records:

- checkpoint commit and clean Git status;
- branch and worktree;
- artifact manifest paths and hashes;
- environment, model, compiler, and runtime revisions;
- pending external jobs and sanitized state;
- resource-lock release or transfer;
- verification already run and the exact restart command.

Uncommitted source files and private conversation context are not dependencies.
External artifacts are usable only through versioned manifests and hashes.

## Recovery and safety

Before registration, task start, migration, and integration:

- inspect every worktree and active task;
- compare dirty paths with task ownership;
- validate graph/status and repository hygiene;
- preserve hashes of pre-existing dirty files; and
- stop on ambiguous ownership or overlap.

Do not resolve a collision with stash, reset, forced staging, branch deletion,
worktree cleanup, task interruption, or unilateral stale-state edits.

If a public task claim was committed but worktree setup fails, leave a
truthful follow-up metadata commit or ask the user for direction. Never erase
the claim with destructive history rewriting.
