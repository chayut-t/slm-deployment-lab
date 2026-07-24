# T03: Independent review fixes

Date: 2026-07-24
Task: `T03`
Visibility: `public`
Status: draft

## Outcome

Resolved every finding from the independent scaffold review while keeping T03
honestly planned until T00 is integrated and the scaffold itself is committed.

## Changes

- Reworked task validation so the pre-commit check reads graph, definitions,
  worklogs, plan, and generated status from the staged Git index.
- Enforced completed dependencies, safe public worklog paths, matching worklog
  metadata, task definitions, ownership fields, and plan/DAG/resource parity.
- Reconciled missing plan edges, status semantics, and resource names.
- Returned T00 and T03 to planned state because version pins, the ADR, and
  integration are not complete.
- Added all 29 task definitions and preserved every intended public directory
  with meaningful content.
- Added minimal `pyproject.toml` and `mkdocs.yml`.
- Hardened the artifact-symlink rule, local bootstrap, and hook line endings.

## Verification

- Command: `python3 -m unittest discover -s tests -p 'test_*.py' -v`
- Result: ten task-automation tests, including adversarial staged-index cases,
  pass.
- Command: `python3 scripts/ai/render_task_status.py --check`
- Result: graph, definitions, plan edges/resources, and generated status agree.
- Command: `python3 scripts/repo/check_hygiene.py --all`
- Result: public working-tree privacy and size checks pass.

## Decisions and evidence

- A completed task is an integrated state, not a local drafting state.
- Public task definitions own objective, paths, outputs, and acceptance; the
  graph owns dynamic coordination and dependencies.
- The NVIDIA resource lock represents either free capacity or a rental, with
  approval required only for the paid fallback.

## Risks and limitations

- The first commit remains intentionally pending user approval.
- T00 still requires immutable revisions and a formal ADR.
- Dedicated dependency/environment locking remains T01 work.

## Follow-up

- Newly unblocked tasks: none until T00 is integrated.
- Recommended next action: complete T00 version pins and ADR, then integrate
  and complete T03 before launching downstream work.
