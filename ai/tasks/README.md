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
