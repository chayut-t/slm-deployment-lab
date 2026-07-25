# T04: Codex and Claude Code Repository Compatibility

Status: in_progress
Owner: Codex T04 agent
Updated: 2026-07-25
Implementation readiness: active after public claim

## Objective

Prepare the repository so Codex and Claude Code can work on it concurrently or
hand tasks between one another without duplicating policy, losing task state,
colliding in a checkout, exposing private session data, or depending on either
tool's conversation history.

The repository, not either agent product, remains the source of truth. A fresh
agent session must be able to determine scope, dependencies, ownership,
resource locks, verification requirements, and restart instructions from
versioned files plus the ignored local registry.

This plan deliberately does not assign an owner, start a task, change a task
status, or reserve a resource.

## Scope

### In scope

- Establish `AGENTS.md` as the canonical shared operating policy.
- Add a minimal root `CLAUDE.md` that imports `AGENTS.md` and contains only
  Claude Code-specific compatibility notes.
- Make normative, forward-looking coordination terminology agent-neutral in a
  narrowly inventoried set of policy files.
- Preserve historical Codex branch names, owners, plans, worklogs, handoffs,
  and failure records as historical evidence.
- Define a task-centered branch and worktree convention for new work.
- Extend the ignored local task registry to distinguish Codex and Claude Code
  sessions without publishing real session identifiers.
- Keep tool-specific permissions, credentials, browser sessions, MCP
  authentication, and auto-memory local.
- Document setup, handoff, parallel-work, and recovery procedures for both
  tools.
- Add mechanical checks that prevent shared instructions and generated task
  state from drifting.
- Make nested instruction files portable by pairing every nested `AGENTS.md`
  with a thin `CLAUDE.md` import adapter.
- Verify the workflow from a fresh-clone-equivalent state with both tools.

### Out of scope

- Assigning Codex or Claude Code during this planning turn or during
  registration; implementation assignment requires separate authorization.
- Reassigning any existing task owner.
- Starting an SLM implementation, benchmarking, profiling, or cloud job.
- Interrupting, archiving, reassigning, stopping, or cleaning up another
  Codex or Claude Code session, branch, worktree, registry entry, or resource
  lock.
- Rewriting historical records to replace the word `Codex`.
- Committing credentials, API tokens, OAuth state, browser profiles, private
  session identifiers, raw transcripts, or agent auto-memory.
- Requiring the same plugins, connectors, or MCP servers in both tools.
- Making tool-specific hooks the only enforcement of repository policy.
- Comparing model quality, token cost, or speed between Codex and Claude.

## Dependencies and resources

- Required task dependencies:
  - `T03` must remain completed because this work refines its agent workflow.
  - Before implementation, register a dedicated task using an approved unused
    task ID. Add its definition, `docs/project/plan.md` DAG node and task-table
    row, `task_graph.yaml` node, active plan, generated status, and any required
    documentation index updates as one coherent parity-preserving change.
  - Register it with `depends_on: ["T03"]`, `status: "planned"`,
    `owner: null`, and `branch: null`.
  - Registration does not authorize implementation. After separate
    implementation authorization, set `status: "in_progress"`, a non-null
    `owner`, and a non-null task branch together before editing implementation
    paths, because the graph validator requires those fields for active work.
  - Do not reopen or rewrite `T03`; preserve its completed evidence.
- Resource locks: none.
  - The absence of a hardware lock does not authorize concurrent edits to
    shared coordination files. Treat those paths as an exclusive coordination
    surface during registration and lifecycle commits.
- External access:
  - Local Codex access for an instruction-loading smoke test.
  - Local Claude Code access for an instruction-loading and worktree smoke
    test.
  - No external service login is required for repository preparation.
- Cost boundary:
  - No paid cloud, hosted inference, Qualcomm, or GPU jobs.
  - Normal local agent subscription or API usage only.

## Non-disruption gate

Run this gate before any tracked edit, task registration, owner assignment,
branch creation, registry migration, or implementation step. A request to
"implement the plan" authorizes work only after this gate passes; it does not
authorize overriding a failed gate.

### Read-only inventory

Capture a local, timestamped preflight record under `.ai-local/` containing:

- current branch, `HEAD`, staged/unstaged/untracked/ignored status, and explicit
  hashes for every pre-existing dirty file;
- `git worktree list --porcelain` plus tracked status for every worktree;
- task-graph statuses, owners, branches, dependencies, and resource locks;
- all local-registry sessions and states, regardless of whether they agree
  with the public graph;
- active execution plans and coordinator handoff state;
- exact plan-owned paths and their intersection with dirty files, active task
  definitions, and other worktrees; and
- the ready-task set produced by the validated task graph.

Use tool/session APIs for a second read-only liveness check when available, but
do not infer that a session is inactive merely because an API is unavailable,
a worktree is clean, or graph and registry state disagree.

When the Codex app task API is available, list tasks and read the recent status
of every active task whose `cwd` is this repository or one of its worktrees.
Treat any active peer task as live until it reaches a terminal state or the
user explicitly releases its paths. Never send it a message, interrupt it,
archive it, hand it off, or change its Git state as part of this preflight.

### Hard stop conditions

Stop without making tracked changes if any of these is true:

- a pre-existing dirty file overlaps a proposed compatibility-task output;
- an active, blocked-but-resumable, or ambiguously stale session may own an
  overlapping path or shared coordination file;
- another writer is editing `AGENTS.md`, `CLAUDE.md`, `PLANS.md`,
  `docs/project/plan.md`, `ai/tasks/`, `ai/plans/`, `ai/handoffs/`,
  `scripts/ai/`, `scripts/setup/`, `scripts/repo/`, `.githooks/`, or the
  relevant repository tests;
- a registry entry says `active` or `in_progress` and its owner has not
  explicitly released or reconciled it;
- a resource lock, branch, worktree, or task claim would be changed for an
  existing task;
- graph/status validation fails or the registration change would alter an
  existing task's ready/blocked result;
- the exact dependency/claim commit is unavailable; or
- the implementation cannot be isolated from every existing worktree and
  uncommitted change.

Do not resolve a stop condition by stashing, resetting, force-adding,
rewriting, migrating, deleting, detaching, interrupting, or marking another
task stale. Only the owning session or user may reconcile ambiguous state.

### Isolation and preservation rules

- Create a new compatibility-task worktree at a new explicit path; never reuse,
  detach, remove, rename, or clean an existing task worktree.
- Preserve all current branches and historical `codex/TNN-*` names.
- Stage only an explicit allowlist of compatibility-task paths and inspect
  `git diff --cached --name-only` before each commit.
- Recheck every pre-existing dirty file's status and hash before and after each
  coordination commit, before integration, and at final handoff.
- If a claim/worktree setup fails, create a truthful follow-up metadata commit
  or stop for user direction; never use destructive reset to erase the claim.
- Do not merge, push, delete worktrees, or alter external state without the
  authorization required by repository policy.
- The compatibility node must not become a dependency of existing SLM tasks.
  Its registration must leave the pre-existing ready-task set, dependency
  edges, owners, branches, statuses, and resource locks unchanged.
- Future tasks may proceed while compatibility work is planned or isolated,
  provided their owned paths and shared coordination windows do not overlap.

### Current audit snapshot

As observed on 2026-07-25:

- the main checkout is on `main` at
  `6cdfe450ee7e7df697de2a5a379deef2fbeda656`;
- `AGENTS.md` has a pre-existing modification and
  `docs/project/learning-checkpoints.md` is pre-existing and untracked;
- a separate Codex task is actively using the main checkout and owns those two
  changes; its current work is not part of this compatibility plan;
- T01, T02, and T03 worktrees exist and have no tracked changes;
- the local schema-v1 registry marks T02 and T03 `in_progress`;
- the public graph marks T02 `blocked` and T03 `completed`; and
- task-graph/status validation passes.

Treat the registry/graph mismatch as ambiguous live state, not permission to
clean or migrate it. Because this plan proposes editing `AGENTS.md` and shared
coordination files, immediate implementation is on hold until the active peer
task finishes or explicitly releases its paths and the user or owning sessions
reconcile the T02/T03 session claims. Registration and implementation must
rerun the gate; this snapshot is evidence, not a substitute for a fresh check.

## Important paths

- Inputs:
  - `AGENTS.md`
  - `PLANS.md`
  - `docs/project/plan.md`
  - `ai/tasks/task_graph.yaml`
  - `ai/tasks/definitions/T03.yaml`
  - `ai/tasks/thread_registry.example.yaml`
  - `ai/handoffs/coordinator.md`
  - `ai/prompts/`
  - `scripts/setup/bootstrap_local_state.py`
  - `scripts/repo/check_hygiene.py`
  - `tests/repo/test_task_automation.py`
  - `.gitignore`
- Proposed committed outputs:
  - `CLAUDE.md`
  - Updated agent-neutral operating text only in the normative coordination
    portions of `AGENTS.md`, `PLANS.md`, `docs/project/plan.md` section 11,
    `ai/README.md`, `ai/tasks/README.md`, and generic reusable prompts.
  - `docs/agentic/dual-agent-setup.md`
  - Updated `ai/tasks/thread_registry.example.yaml`
  - A local-registry helper plus updated bootstrap, hygiene checks, and
    repository tests.
  - A new task definition, active execution plan, task-graph entry, generated
    status, completion worklog, and any necessary coordinator handoff for the
    eventual implementation task.
- Proposed local-only outputs:
  - `CLAUDE.local.md`, only for personal project instructions.
  - `.claude/settings.local.json`, only if the user wants personal Claude Code
    permissions or environment settings.
  - Real Codex and Claude Code session identifiers in the designated
    coordination checkout's `.ai-local/tasks/thread-registry.yaml`.
  - Tool auto-memory and authentication state in each tool's normal local
    storage.
- Shared contracts:
  - `AGENTS.md` is the only canonical repository policy.
  - `docs/project/plan.md` controls stable project scope and agentic-delivery
    claims.
  - `ai/tasks/task_graph.yaml` controls dependencies, public ownership,
    branches, resource locks, and status.
  - The designated coordination checkout's
    `.ai-local/tasks/thread-registry.yaml` records private live-session state;
    the committed task graph remains authoritative for public ownership,
    branch, dependency, status, and resource-lock claims.

## Target operating model

### Shared policy and tool adapters

Use a hub-and-adapter structure:

```text
                    AGENTS.md
                 canonical policy
                   /          \
          Codex native       CLAUDE.md
                              @AGENTS.md
                         Claude-only notes
```

`CLAUDE.md` must import `AGENTS.md` instead of copying its content. It may
explain Claude Code terminology or local tool behavior, but it must not restate
branch, privacy, verification, task-lifecycle, artifact, or cost rules.

The same rule applies below the repository root. If a subtree adds an
`AGENTS.md`, it must also add a same-directory `CLAUDE.md` that imports that
file. A repository test must enumerate tracked `AGENTS.md` files and reject a
missing or incorrect paired adapter. This preserves the existing hierarchical
instruction model for both tools.

Do not add a committed Claude permission allowlist initially. Repository Git
hooks and explicit verification commands should remain the common enforcement
layer. Add shared `.mcp.json` or tool-specific project settings later only when
a real project task requires them and their security boundary is reviewed.

### Branch and worktree convention

Use task-centered branches for new work:

```text
task/T10-evaluation-fixtures
task/T31-qwen-workbench
task/T51-mlx-runtime
```

The branch name should survive a handoff between Codex and Claude Code. Record
the active tool and private session ID in the ignored local registry rather
than encoding tool identity into the new branch name.

Existing `codex/TNN-*` branches remain valid historical branches and must not
be renamed. A task already in progress on a historical branch may finish on
that branch.

Create task worktrees with Git, not either tool's implicit worktree feature, so
the exact dependency commit, directory, and branch are deterministic:

```bash
git worktree add /explicit/task/worktree -b task/TNN-short-slug CLAIM_SHA
```

`CLAIM_SHA` is the committed public ownership/branch claim made on top of the
completed dependency commit. Launch Codex or Claude Code from that worktree.
Do not use Claude Code's
default `--worktree` branch creation for project tasks because its branch and
base-selection behavior do not implement this repository contract. A future
custom worktree hook is out of scope until separately reviewed.

Each active writer gets a dedicated worktree. Never run Codex and Claude Code
as simultaneous writers in the same checkout, on the same task branch, or
against overlapping owned paths. Read-only reviewers must use separate
sessions and must not mutate files or Git state.

For a tool handoff, prefer reusing the existing task worktree after the
outgoing tool has stopped. If the incoming tool must use a different path,
first verify a clean checkpoint, detach or remove the old worktree without
discarding changes, and only then attach the existing task branch to the new
worktree. Never attempt to check out one branch in two worktrees.

### Ownership and handoff

This draft remains unassigned. The lifecycle is:

1. Register the compatibility task as `planned`, `owner: null`, and
   `branch: null`.
2. Stop unless the user separately authorizes implementation.
3. In the designated coordination/integration checkout, atomically set the
   task to `in_progress` with a non-null owner and intended task branch,
   regenerate status, and commit that public claim.
4. Create the task branch/worktree from the exact claim commit, then record the
   private session in the shared local registry before implementation edits.
5. Mark the task `completed` only after its dependencies, outputs, acceptance
   criteria, integration, public worklog, and verification all satisfy the
   repository completion contract.

When ordinary project work resumes:

- Public `owner` identifies the currently responsible agent or coordinator.
- The shared local registry identifies the active writer and all current or
  historical private sessions, including their tool, role, state, worktree,
  and timestamps. It does not override the committed graph.
- A tool handoff records the checkpoint SHA, clean Git status, artifact
  manifest paths and hashes, environment and tool revisions, pending external
  job state, resource-lock transfer or release, updated execution plan, and a
  sanitized handoff when necessary.
- The outgoing writer stops before the incoming writer begins editing.
- The incoming writer starts from the exact checkpoint commit and reruns the
  narrow verification relevant to the task.
- Uncommitted source files and private conversation context are never
  dependencies. Large external artifacts are valid only when versioned
  manifests and hashes make them reproducible.

### Proposed local registry shape

Keep real values only in the designated coordination checkout's
`.ai-local/tasks/thread-registry.yaml`. Every worktree must access it through a
versioned helper rather than reading or writing a per-worktree copy. The helper
resolves an explicit `SLM_LAB_COORDINATION_ROOT` when set; otherwise it resolves
the primary checkout from `git worktree list --porcelain`. It must use atomic
writes and an inter-process file lock.

The committed task graph is the authoritative coordination claim. The local
registry adds private session detail and must never be treated as an
independent lock. The committed example may use obvious placeholders:

```yaml
schema_version: 2
tasks:
  TNN:
    branch: task/TNN-short-slug
    checkpoint_sha: full-git-sha
    active_writer: session-placeholder
    sessions:
      session-placeholder:
        tool: codex-or-claude-code
        role: writer-or-reviewer
        state: active
        worktree: /private/local/path
        started_at: YYYY-MM-DDTHH:MM:SSZ
        updated_at: YYYY-MM-DDTHH:MM:SSZ
    updated_at: YYYY-MM-DDTHH:MM:SSZ
```

The helper must read and validate both schema v1 and schema v2. Bootstrap must
create schema v2 for a new registry but must never auto-migrate or rewrite an
existing registry. Migration is a separate explicit operation that:

- refuses to run while any registry session is active, `in_progress`, or
  ambiguously stale;
- requires a fresh non-disruption preflight and explicit user approval;
- performs an atomic, backup-backed v1-to-v2 conversion preserving every
  existing entry;
- is idempotent;
- refuses malformed or unknown schema versions without changing the source;
  and
- retains a recoverable backup until successful validation.

Compatibility implementation may add the dual-schema reader before migration,
but it must not migrate the currently observed live/ambiguous registry.

Writer operations must be compare-and-swap safe. The helper must reject:

- a second active-writer claim for the same task;
- a registry branch that disagrees with the task graph;
- a checkpoint update based on a stale checkpoint;
- a writer transfer or release that does not name the current writer and
  checkpoint state; and
- promotion of a reviewer session to writer without an explicit validated
  transfer.

Multiple read-only reviewers may coexist. Every rejection must leave the
registry unchanged and return a clear recovery instruction.

## Milestones

### 1. Register the compatibility task without assigning it

- [ ] Run the non-disruption gate and stop if any hard condition remains.
- [ ] Obtain owner/user reconciliation for every ambiguous active session and
  overlapping dirty path; do not perform that reconciliation within this task.
- [ ] Select an approved unused task ID.
- [ ] In one coherent change, add the task definition, project-plan DAG node and
  edge, project-plan task-table row, graph node, active execution plan, and
  generated status.
- [ ] Set the graph node to `depends_on: ["T03"]`, `status: planned`,
  `owner: null`, `branch: null`, `github_issue: null`, and no resource locks.
- [ ] Promote this draft into `ai/plans/active/TNN-dual-agent-compatibility.md`
  after replacing the placeholder ID.
- [ ] Regenerate and validate `ai/tasks/status.generated.md`.
- [ ] Prove that every pre-existing task retains the same dependency edges,
  status, owner, branch, resource locks, and ready/blocked result.

Observable result: the work is dependency-aware and restartable, but no agent
has been assigned and no implementation has started.

### 2. Start implementation only after separate authorization

- [ ] Confirm the user has authorized implementation, not only registration.
- [ ] Rerun the non-disruption gate immediately before the public claim.
- [ ] In the designated coordination/integration checkout, set
  `status: in_progress`, the assigned `owner`, and intended task branch;
  regenerate status and commit the public claim.
- [ ] Create the task-centered branch and manual Git worktree from that exact
  claim commit.
- [ ] Register the private writer session through the shared local-registry
  helper.
- [ ] Stop if authorization or any required lifecycle field is missing.
- [ ] If worktree or registry setup fails, do not implement; repair the setup
  or restore a truthful planned/unassigned graph state in the coordination
  checkout.

Observable result: implementation begins only from a valid, assigned,
dependency-complete graph state. This draft does not perform this milestone.

### 3. Establish one shared instruction source

- [ ] Add root `CLAUDE.md` containing `@AGENTS.md`.
- [ ] Keep Claude-only notes short and free of duplicated policy.
- [ ] Require every tracked nested `AGENTS.md` to have a same-directory
  `CLAUDE.md` import adapter.
- [ ] Update hygiene requirements so a dual-agent repository cannot silently
  lose `CLAUDE.md`.
- [ ] Add regression tests that verify root and nested import mappings.
- [ ] Document how to inspect loaded instructions in each tool.

Observable result: both tools receive the same repository rules, and drift is
mechanically detectable.

### 4. Neutralize only normative workflow language

- [ ] Inventory exact Codex-specific occurrences and classify each as
  normative policy, current operational state, historical provenance, or
  product-specific documentation.
- [ ] Change `Codex task/thread` to `agent task/session` only in normative
  future-facing policy, coordination READMEs, and generic prompts.
- [ ] Update the multi-agent operating-model section of
  `docs/project/plan.md`.
- [ ] Update the branch examples and explain the historical `codex/` prefix.
- [ ] Update public/private-data rules to cover both tools' sessions,
  transcripts, auto-memory, and identifiers.
- [ ] Leave historical worklogs, completed plans, branch references, and
  failure evidence unchanged.

Observable result: future policy applies equally to Codex and Claude Code
without falsifying project history.

### 5. Harden shared local coordination and private configuration

- [ ] Ignore `CLAUDE.local.md`, `.claude/worktrees/`, and
  `.claude/settings.local.json`.
- [ ] Add staged-index rejection for those paths so `git add -f` cannot bypass
  privacy policy.
- [ ] Inventory actual repository-local Codex private paths; enumerate and
  protect only paths that exist or are explicitly introduced.
- [ ] Add force-add privacy regressions.
- [ ] Add a shared local-registry helper that resolves the designated
  coordination checkout and uses locking plus atomic writes.
- [ ] Update the local registry example to schema version 2.
- [ ] Implement and test v1/v2 reading without mutation.
- [ ] Implement explicit backup-backed v1-to-v2 migration with active-session
  refusal, idempotence, entry preservation, malformed-input failure, and
  unknown-version failure.
- [ ] Test rejection of duplicate writers, graph/registry branch mismatch,
  stale checkpoints, invalid transfer/release, and implicit reviewer promotion.
- [ ] Expand privacy/hygiene tests for both tools' private session material and
  verify two worktrees resolve the same registry.
- [ ] Do not ignore all of `.claude/`, because future shared agents, skills, or
  project settings may legitimately be versioned.

Observable result: shareable compatibility files can be committed while
machine-local state remains private and live coordination is shared across
worktrees.

### 6. Document parallel work and cross-tool handoffs

- [ ] Add `docs/agentic/dual-agent-setup.md`.
- [ ] Document fresh-clone setup for Codex and Claude Code.
- [ ] Document task selection, dependency checking, owner assignment, branch
  creation with manual `git worktree add`, and shared local registry updates.
- [ ] Document both handoff paths: reuse the existing worktree, or cleanly
  detach/remove it before attaching the branch to a different worktree.
- [ ] Require checkpoint SHA, clean status, manifests and hashes, environment
  revisions, pending external jobs, and resource-lock state in handoffs.
- [ ] Document resource-lock coordination and the prohibition on overlapping
  writers.
- [ ] Document recovery when a session disappears, a worktree is stale, or a
  handoff checkpoint fails verification.
- [ ] Explain that browser connectors, MCP servers, plugins, and authenticated
  sessions are optional tool adapters rather than repository dependencies.

Observable result: a user can start either tool or transfer a task without
needing private conversation history.

### 7. Verify both tools reproducibly from clean state

- [ ] Run the repository unit tests and hygiene checks.
- [ ] Capture the exact Codex and Claude Code versions, product surfaces
  (desktop, CLI, or extension), launch modes, OS, and common base commit.
- [ ] Create two isolated temporary clones from the same local commit.
- [ ] Run the versioned setup installer and verify local state reconstruction.
- [ ] In Codex, use a read-only orientation prompt and record a sanitized
  checklist showing it identified the canonical instructions, project plan,
  task graph, ready tasks, and required completion checks.
- [ ] In Claude Code, use `/context` and its memory-files view to confirm
  `CLAUDE.md` and imported
  `AGENTS.md`, then run the same read-only orientation prompt and checklist.
- [ ] Verify both tools identify the same repository source of truth and do
  not assign an owner or begin a blocked task.
- [ ] Exercise the documented manual Git worktree creation, tool launch, stop,
  and cleanup path for each tool without modifying project files.
- [ ] Confirm `git status --short --ignored` contains no unexpected tracked or
  unignored private tool state.

Observable result: both tools can orient, isolate work, and recover from
the same versioned state while producing equivalent coordination decisions.

### 8. Complete the repository task

- [ ] Rerun the non-disruption inventory and compare it with the preflight
  record before integration.
- [ ] Record exact commands and results in a sanitized worklog.
- [ ] Mark the task `completed` only after every acceptance criterion passes,
  all outputs are integrated, and the matching public worklog exists.
- [ ] Regenerate task status.
- [ ] Move the approved execution plan to `ai/plans/completed/`.
- [ ] Update the coordinator handoff if future tasks need new startup
  instructions.
- [ ] Preserve `owner: null` for unrelated tasks.
- [ ] Confirm every pre-existing worktree, dirty-file hash, branch, task claim,
  ready/blocked result, and resource lock is unchanged except for changes
  separately made by its owner during the task.

Observable result: dual-tool support is versioned, tested, documented, and
resumable without changing unrelated task ownership.

## Verification and acceptance

### Commands

Run at minimum:

```bash
python3 -m unittest tests.repo.test_task_automation
python3 scripts/ai/render_task_status.py --check
python3 scripts/repo/check_hygiene.py --all
git status --short --ignored
```

Also run any narrower tests added for instruction imports, ignored local state,
shared-registry resolution and migration, manual-worktree behavior, force-added
private paths, and staged-snapshot validation.

### Behavioral criteria

- The non-disruption gate passes before registration, task start, registry
  migration, and integration.
- Any ambiguous active session or overlapping dirty path causes a safe stop
  without tracked changes.
- Codex loads and follows `AGENTS.md`.
- Claude Code loads `CLAUDE.md`, which imports `AGENTS.md`.
- Every tracked nested `AGENTS.md` has a paired thin `CLAUDE.md` adapter.
- Shared policy exists in one canonical file and is not copied into
  tool-specific adapters.
- Both tools identify the same project plan, task graph, ready tasks,
  dependencies, resource locks, privacy rules, and verification commands.
- A task can move between tools using repository state and a checkpoint commit.
- Two active writers cannot legitimately claim the same task or overlapping
  owned paths under the documented procedure.
- All worktrees resolve one locked, atomically updated local registry in the
  designated coordination checkout; the committed graph remains authoritative.
- Registry compare-and-swap checks reject duplicate writers, branch mismatch,
  stale checkpoints, invalid release/transfer, and implicit reviewer promotion
  without changing registry state.
- A handoff records source checkpoint and non-Git dependencies sufficiently to
  reproduce the incoming session.
- New task branches are tool-neutral; historical branches remain intact.
- No existing task is assigned or reassigned by this preparation work.
- No existing task branch, worktree, owner, status, dependency, resource lock,
  ready/blocked result, uncommitted file, or private session is modified.
- The compatibility task is not a dependency of existing SLM tasks and does not
  prevent future non-overlapping tasks from starting.
- Planned, in-progress, and completed lifecycle transitions satisfy graph
  validation, including owner and branch requirements for `in_progress`.
- The new task definition, project-plan DAG/table, graph, active/completed plan,
  worklog, and generated status remain in parity at their lifecycle stages.

### Privacy and hygiene criteria

- Real Codex and Claude Code session identifiers remain ignored.
- Raw transcripts, auto-memory, browser state, OAuth state, credentials, and
  local permission settings remain ignored or outside the repository.
- `CLAUDE.md` is present and imports `AGENTS.md`.
- `CLAUDE.local.md`, `.claude/worktrees/`, and
  `.claude/settings.local.json` are ignored and rejected if force-added.
- Fresh clones receive registry schema v2.
- Existing schema-v1 registries remain readable and unchanged until an explicit
  approved migration is safe.
- Explicit migration refuses active or ambiguous sessions, uses an atomic
  recoverable backup, preserves all entries, and leaves malformed or unknown
  versions unchanged.
- Staged and working-tree hygiene checks pass.

### Hardware/profile evidence

None. This is a repository workflow task and must not run hardware benchmarks,
external compiler jobs, or hosted profiles.

## Artifact and privacy handling

- Committed evidence:
  - Shared policy adapter, agent-neutral documentation, sanitized registry
    example, tests, worklog, task metadata, and generated status.
- External artifacts:
  - None.
- Private/local material:
  - Real session IDs, local worktree paths, tool settings, permissions,
    authentication, transcripts, smoke-test conversation details, and
    non-disruption preflight/comparison records.
- Sanitization:
  - Smoke-test results may state whether a tool loaded instructions and reached
    the expected decision, but must not include raw conversations or private
    identifiers.

## Risks and mitigations

- Policy duplication:
  - Mitigation: import `AGENTS.md`; prohibit copied policy in `CLAUDE.md`; test
    for the import.
- Simultaneous edits:
  - Mitigation: one active writer per task and owned path; dedicated
    worktrees; explicit registry state.
- Disruption of current or future tasks:
  - Mitigation: mandatory read-only preflight, occupied-until-released
    semantics, explicit-path staging, isolated new worktree, ready-set parity,
    and before/after preservation checks.
- Tool-specific feature dependency:
  - Mitigation: keep core workflow in Git, scripts, and standard shell
    commands; treat connectors and MCP as optional.
- Historical record churn:
  - Mitigation: update only forward-looking policy; preserve completed
    evidence and branch names.
- Local registry data loss:
  - Mitigation: dual-schema reads; no automatic migration; explicit
    active-session refusal; atomic backup-backed migration; locked writes; test
    entry preservation, malformed input, unknown versions, and idempotence.
- Per-worktree registry divergence:
  - Mitigation: resolve one designated coordination checkout through a
    versioned helper and verify resolution from multiple worktrees.
- Conflicting or stale writer claims:
  - Mitigation: compare-and-swap writer operations validated against the graph,
    current writer, branch, and checkpoint; rejection leaves state unchanged.
- Branch ambiguity during handoff:
  - Mitigation: use manual Git worktree creation for `task/TNN-*`; reuse the
    worktree or detach it cleanly before reattaching the branch elsewhere.
- Nested instruction drift:
  - Mitigation: require and test one thin `CLAUDE.md` adapter beside every
    tracked `AGENTS.md`.
- Excessive instruction context:
  - Mitigation: keep `CLAUDE.md` minimal and move explanations into
    `docs/agentic/dual-agent-setup.md`.
- False parity:
  - Mitigation: require equivalent coordination outcomes, not identical tools,
    permissions, plugins, or user interfaces.

## Decisions and discoveries

- 2026-07-25: Keep this plan local and unassigned until the user approves a
  task-graph ID and implementation start.
- 2026-07-25: Use `AGENTS.md` as the canonical policy and `CLAUDE.md` as a thin
  import adapter.
- 2026-07-25: Recommend task-centered `task/TNN-*` branches for new work so a
  branch can survive a tool handoff.
- 2026-07-25: Create project task worktrees manually from an explicit
  dependency commit instead of relying on either tool's implicit worktree
  naming or base-selection behavior.
- 2026-07-25: Keep one locked local registry in a designated coordination
  checkout and access it through a versioned helper from every worktree.
- 2026-07-25: Pair every nested `AGENTS.md` with a thin same-directory
  `CLAUDE.md` import adapter.
- 2026-07-25: Read both registry schemas and never auto-migrate; use an explicit
  atomic, backup-backed v1-to-v2 migration only when no live or ambiguous
  sessions remain.
- 2026-07-25: Preserve historical Codex-specific records rather than rewriting
  provenance.
- 2026-07-25: Prefer repository Git hooks and verification scripts over
  mandatory tool-specific hooks.
- 2026-07-25: Do not commit shared MCP or broad permission configuration
  without a concrete project requirement.
- 2026-07-25: An independent subagent review identified lifecycle/parity,
  shared-registry, worktree, privacy, nested-instruction, handoff, migration,
  reproducibility, and scope issues; all actionable findings were incorporated
  into this revision.

## Progress and restart instructions

Current state:

- This draft plan exists only under `.ai-local/plans/`.
- No task ID, owner, branch, worktree, resource lock, or task status has been
  assigned.
- No public repository file has been changed by this planning work.
- Pre-existing changes are present in `AGENTS.md` and
  `docs/project/learning-checkpoints.md`.
- A separate active Codex task owns those changes in the main checkout.
- T02 and T03 have ambiguous active local-registry claims that disagree with
  public graph status.
- Implementation readiness is `hold`; do not register or implement until a
  fresh non-disruption gate passes.

To restart:

1. Read `AGENTS.md`, `docs/project/plan.md`, this draft, and the current
   `ai/tasks/task_graph.yaml`.
2. Inspect `git status --short --ignored` and preserve unrelated changes.
3. Confirm `T03` is still completed.
4. Run the non-disruption gate. Stop if dirty-path ownership, session liveness,
   registry/graph disagreement, or shared-path availability is unresolved.
5. Ask the user or owning sessions to reconcile blockers; do not clean or
   override their state.
6. Ask the user to approve implementation and an unused task ID if approval
   has not already been given.
7. As one coherent change, add the definition, project-plan DAG/table entries,
   planned graph node with `owner: null` and `branch: null`, promoted active
   plan, and generated status.
8. Validate parity, then stop before assigning or starting the task unless the
   user separately authorizes implementation.
9. After implementation authorization, rerun the gate; commit the
   `in_progress`, owner, and
   intended-branch claim on the coordination/integration checkout; create the
   manual task worktree from that claim commit; register the private session;
   and only then edit implementation paths.
