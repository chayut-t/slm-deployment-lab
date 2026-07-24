# T03: Agent workflow scaffold

Date: 2026-07-24
Task: `T03`
Visibility: `public`
Status: draft

## Outcome

The repository has a drafted public AI workspace, ignored local workspace,
dependency-aware task graph, durable agent rules, reusable templates, and
versioned workflow automation. T03 remains planned until T00 is complete and
this scaffold is reviewed, integrated, and accepted.

## Changes

- Added `AGENTS.md`, `PLANS.md`, and the `ai/` public coordination tree.
- Added `.ai-local/` for feedback, raw logs, draft plans, private registries,
  unsanitized profiles, and scratch work.
- Added a JSON-compatible YAML task graph and generated Markdown renderer.
- Added worklog scaffolding, repository hygiene checks, and versioned Git
  hooks.
- Moved the public plan to `docs/project/plan.md` and private planning feedback
  under the ignored `.ai-local/inputs/` tree.

## Verification

- Task graph validation and generated status check:
  `python3 scripts/ai/render_task_status.py --check`
- Repository privacy and organization check:
  `python3 scripts/repo/check_hygiene.py --all`
- Git ignore and hook configuration are verified separately before handoff.

## Decisions and evidence

- Public worklogs contain summarized engineering evidence, not raw transcripts.
- A completed task must name an existing worklog in the task graph.
- Local task/thread IDs never enter committed task artifacts.
- Git hooks mechanically enforce privacy, size, secret, and generated-status
  policy; `AGENTS.md` governs judgment-dependent lifecycle behavior.

## Risks and limitations

- GitHub Issue/Project synchronization remains a later implementation step.
- Hook checks reduce accidental secret exposure but do not replace dedicated
  secret scanning in CI.

## Follow-up

- Newly unblocked tasks: none while T00 remains planned.
- Recommended next action: finish and integrate T00, then validate and
  integrate this scaffold before starting T01, T02, and T10.
