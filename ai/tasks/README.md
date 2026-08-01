# Task coordination

`task_graph.yaml` is the repository source of truth for task dependencies,
status, resource locks, and worklog linkage. It is written in JSON-compatible
YAML so the standard-library status renderer can validate it without requiring
a bootstrapped Python environment.

Every graph node references `definitions/TNN.yaml`, which owns the task's
objective, file boundaries, outputs, and acceptance criteria. Public
owner/branch/issue fields support coordination; real agent session identifiers
remain in the ignored local registry.

After editing the graph, run:

```bash
python3 scripts/ai/render_task_status.py
```

Never edit `status.generated.md` by hand. Access real task/session identifiers
through `scripts/ai/session_registry.py`; the resolved registry remains ignored.

Validation rejects plan/DAG drift, dependency cycles, unknown resources,
in-progress tasks with unfinished dependencies, and completed tasks without a
matching public worklog and completed dependencies. The pre-commit hook applies
the same checks to the staged Git snapshot.

## Learning lane

`learning_lane.yaml` holds the `LEARN-NN` study checkpoints and the completed
tasks each one covers. Checkpoints are **terminal**: they depend on tasks, and
no task may depend on them, so a missing or stale sheet never blocks
implementation work — consistent with `docs/project/plan.md` section 8.1.

The lane is generated, not authored. Its content comes from
`configs/learning/checkpoints.yaml`, which also holds the reading lists and
questions that do not belong in a dependency graph:

```bash
python3 scripts/learning/build_learning_sheet.py --all --record
python3 scripts/ai/render_task_status.py
```

The renderer validates that every covered task exists and is completed, that
no task is covered twice, and that no completed task is left uncovered. It
also compares each recorded source digest against the current file, so
`status.generated.md` names the sheets that need rebuilding and republishing.
That is a report, never a gate.
