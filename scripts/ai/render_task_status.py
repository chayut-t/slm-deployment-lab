#!/usr/bin/env python3
"""Validate the task DAG and render its generated Markdown status page."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = REPO_ROOT / "ai" / "tasks" / "task_graph.yaml"
OUTPUT_PATH = REPO_ROOT / "ai" / "tasks" / "status.generated.md"
LANE_PATH = "ai/tasks/learning_lane.yaml"
ReadText = Callable[[str], str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when the generated file is missing or stale",
    )
    return parser.parse_args()


def normalize_public_path(
    value: object,
    *,
    task_id: str,
    field: str,
    prefix: tuple[str, ...],
    suffix: str,
) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{task_id}: {field} must be a repository-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{task_id}: unsafe {field} path: {value}")
    if path.parts[: len(prefix)] != prefix or path.suffix != suffix:
        expected = "/".join(prefix) + f"/*{suffix}"
        raise ValueError(f"{task_id}: {field} must be under {expected}: {value}")
    return path.as_posix()


def working_text(path: str) -> str:
    filesystem_path = REPO_ROOT / path
    if not filesystem_path.is_file():
        raise FileNotFoundError(path)
    return filesystem_path.read_text(encoding="utf-8")


def validate_graph_text(
    graph_text: str,
    read_text: ReadText,
) -> tuple[dict, list[dict]]:
    try:
        graph = json.loads(graph_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"task_graph.yaml is not valid JSON-compatible YAML: {exc}"
        ) from exc

    tasks = graph.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("task_graph.yaml must contain a tasks list")

    ids = [task.get("id") for task in tasks]
    duplicates = sorted(task_id for task_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate task IDs: {', '.join(duplicates)}")

    task_map = {task["id"]: task for task in tasks}
    allowed_statuses = set(graph.get("allowed_statuses", []))
    known_locks = set(graph.get("resource_locks", {}))

    for task in tasks:
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id.startswith("T"):
            raise ValueError(f"invalid task ID: {task_id!r}")
        if task.get("status") not in allowed_statuses:
            raise ValueError(f"{task_id}: invalid status {task.get('status')!r}")

        dependencies = task.get("depends_on")
        if not isinstance(dependencies, list):
            raise ValueError(f"{task_id}: depends_on must be a list")
        missing = [dep for dep in dependencies if dep not in task_map]
        if missing:
            raise ValueError(f"{task_id}: unknown dependencies: {', '.join(missing)}")

        resource_locks = task.get("resource_locks")
        if not isinstance(resource_locks, list):
            raise ValueError(f"{task_id}: resource_locks must be a list")
        invalid_locks = [lock for lock in resource_locks if lock not in known_locks]
        if invalid_locks:
            raise ValueError(
                f"{task_id}: unknown resource locks: {', '.join(invalid_locks)}"
            )

        for coordination_field in ("owner", "branch", "github_issue"):
            if coordination_field not in task:
                raise ValueError(f"{task_id}: missing {coordination_field} field")

        definition = normalize_public_path(
            task.get("definition"),
            task_id=task_id,
            field="definition",
            prefix=("ai", "tasks", "definitions"),
            suffix=".yaml",
        )
        try:
            definition_data = json.loads(read_text(definition))
        except FileNotFoundError as exc:
            raise ValueError(
                f"{task_id}: definition does not exist: {definition}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{task_id}: invalid definition {definition}: {exc}"
            ) from exc

        matching_fields = {
            "id": task_id,
            "title": task.get("title"),
            "depends_on": dependencies,
            "resource_locks": resource_locks,
        }
        for field, expected in matching_fields.items():
            if definition_data.get(field) != expected:
                raise ValueError(
                    f"{task_id}: definition {field} does not match task graph"
                )
        for field in ("outputs", "acceptance", "owned_paths"):
            value = definition_data.get(field)
            if not isinstance(value, list) or not value:
                raise ValueError(
                    f"{task_id}: definition must include non-empty {field}"
                )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ValueError(f"task dependency cycle includes {task_id}")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in task_map[task_id].get("depends_on", []):
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in task_map:
        visit(task_id)

    for task in tasks:
        task_id = task["id"]
        status = task["status"]
        incomplete_dependencies = [
            dependency
            for dependency in task["depends_on"]
            if task_map[dependency]["status"] != "completed"
        ]
        if status in {"in_progress", "completed"} and incomplete_dependencies:
            raise ValueError(
                f"{task_id}: {status} task has incomplete dependencies: "
                f"{', '.join(incomplete_dependencies)}"
            )
        if status == "blocked" and incomplete_dependencies:
            raise ValueError(
                f"{task_id}: keep status planned while dependencies are incomplete; "
                "blocked is reserved for an external blocker after dependency completion"
            )
        if status == "in_progress" and (
            not task.get("owner") or not task.get("branch")
        ):
            raise ValueError(
                f"{task_id}: in_progress task must name an owner and branch"
            )

        if status == "completed":
            worklog = normalize_public_path(
                task.get("worklog"),
                task_id=task_id,
                field="worklog",
                prefix=("ai", "worklogs"),
                suffix=".md",
            )
            try:
                worklog_text = read_text(worklog)
            except FileNotFoundError as exc:
                raise ValueError(
                    f"{task_id}: worklog does not exist: {worklog}"
                ) from exc

            required_metadata = {
                "Task": rf"`{re.escape(task_id)}`",
                "Visibility": "`public`",
                "Status": "completed",
            }
            for label, expected in required_metadata.items():
                if not re.search(
                    rf"^{label}:\s*{expected}\s*$",
                    worklog_text,
                    flags=re.MULTILINE,
                ):
                    raise ValueError(
                        f"{task_id}: worklog must contain {label}: {expected}"
                    )
        elif task.get("worklog") is not None:
            raise ValueError(
                f"{task_id}: only completed tasks may set the worklog field"
            )

    return graph, tasks


def validate_plan_parity(graph: dict, tasks: list[dict], plan_text: str) -> None:
    dag_match = re.search(
        r"### 10\.2 Core DAG.*?```mermaid\n(?P<dag>.*?)```",
        plan_text,
        flags=re.DOTALL,
    )
    if not dag_match:
        raise ValueError("project plan is missing the section 10.2 Mermaid DAG")
    dag = dag_match.group("dag")
    plan_ids = set(re.findall(r"^\s*(T\d+)\[", dag, flags=re.MULTILINE))
    plan_edges = set(
        re.findall(
            r"^\s*(T\d+)\s*-->\s*(T\d+)\s*$",
            dag,
            flags=re.MULTILINE,
        )
    )
    graph_ids = {task["id"] for task in tasks}
    graph_edges = {
        (dependency, task["id"])
        for task in tasks
        for dependency in task.get("depends_on", [])
    }
    if plan_ids != graph_ids:
        missing = sorted(graph_ids - plan_ids)
        extra = sorted(plan_ids - graph_ids)
        raise ValueError(
            "project-plan task IDs differ from task graph; "
            f"missing={missing or 'none'}, extra={extra or 'none'}"
        )
    if plan_edges != graph_edges:
        missing = sorted(graph_edges - plan_edges)
        extra = sorted(plan_edges - graph_edges)
        raise ValueError(
            "project-plan DAG edges differ from task graph; "
            f"missing={missing or 'none'}, extra={extra or 'none'}"
        )

    resource_match = re.search(
        r"### 10\.4 Resource locks.*?```yaml\n(?P<resources>.*?)```",
        plan_text,
        flags=re.DOTALL,
    )
    if not resource_match:
        raise ValueError("project plan is missing the section 10.4 resource block")
    plan_resources = set(
        re.findall(
            r"^  ([a-z][a-z0-9_]+):\s*$",
            resource_match.group("resources"),
            flags=re.MULTILINE,
        )
    )
    graph_resources = set(graph.get("resource_locks", {}))
    if plan_resources != graph_resources:
        raise ValueError(
            "project-plan resources differ from task graph; "
            f"plan={sorted(plan_resources)}, graph={sorted(graph_resources)}"
        )


def load_learning(read_text: ReadText, tasks: list[dict]) -> list[dict]:
    """Load the learning lane: checkpoints that depend on completed tasks.

    The lane is generated from `configs/learning/checkpoints.yaml` by
    `scripts/learning/build_learning_sheet.py --record`. It is projected into a
    JSON-compatible file so this script stays dependency-free and reads the
    same bytes from a staged snapshot as from the working tree.

    Checkpoints are terminal: tasks never depend on them, so a missing or
    stale checkpoint never blocks implementation work.
    """

    try:
        lane_text = read_text(LANE_PATH)
    except FileNotFoundError:
        return []
    try:
        lane = json.loads(lane_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{LANE_PATH} is not valid JSON-compatible YAML: {exc}"
        ) from exc

    entries = lane.get("checkpoints")
    if not isinstance(entries, list):
        raise ValueError(f"{LANE_PATH} must contain a checkpoints list")

    task_map = {task["id"]: task for task in tasks}
    completed = {task["id"] for task in tasks if task["status"] == "completed"}
    claimed: dict[str, str] = {}
    seen: set[str] = set()
    loaded: list[dict] = []

    for entry in entries:
        checkpoint_id = entry.get("id")
        if not isinstance(checkpoint_id, str) or not re.fullmatch(
            r"LEARN-\d{2}", checkpoint_id
        ):
            raise ValueError(f"invalid checkpoint ID: {checkpoint_id!r}")
        if checkpoint_id in seen:
            raise ValueError(f"duplicate checkpoint ID: {checkpoint_id}")
        seen.add(checkpoint_id)

        covers = entry.get("covers")
        if not isinstance(covers, list) or not covers:
            raise ValueError(f"{checkpoint_id}: covers must be a non-empty list")
        for task_id in covers:
            if task_id not in task_map:
                raise ValueError(f"{checkpoint_id}: unknown task {task_id}")
            if task_map[task_id]["status"] != "completed":
                raise ValueError(
                    f"{checkpoint_id}: covers {task_id}, which is "
                    f"{task_map[task_id]['status']!r}; checkpoints cover "
                    "completed work only"
                )
            if task_id in claimed:
                raise ValueError(
                    f"{task_id} is covered by both {claimed[task_id]} and "
                    f"{checkpoint_id}"
                )
            claimed[task_id] = checkpoint_id

        sources = entry.get("sources")
        if not isinstance(sources, dict) or not sources:
            raise ValueError(f"{checkpoint_id}: sources must be a non-empty mapping")
        changed = []
        for path, stamp in sorted(sources.items()):
            try:
                current = hashlib.sha256(read_text(path).encode("utf-8")).hexdigest()[
                    :12
                ]
            except FileNotFoundError:
                changed.append(path)
                continue
            if current != stamp:
                changed.append(path)

        loaded.append({**entry, "changed": changed})

    uncovered = sorted(completed - set(claimed))
    if uncovered:
        raise ValueError(
            "completed tasks without a learning checkpoint: "
            f"{', '.join(uncovered)}; add them to configs/learning/checkpoints.yaml"
        )
    return loaded


def load_and_validate() -> tuple[dict, list[dict], list[dict]]:
    try:
        graph_text = GRAPH_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"cannot read {GRAPH_PATH.relative_to(REPO_ROOT)}: {exc}"
        ) from exc
    graph, tasks = validate_graph_text(graph_text, working_text)
    plan_path = graph.get("project_plan")
    if not isinstance(plan_path, str):
        raise ValueError("task graph must name a project_plan path")
    try:
        plan_text = working_text(plan_path)
    except FileNotFoundError as exc:
        raise ValueError(f"project plan does not exist: {plan_path}") from exc
    validate_plan_parity(graph, tasks, plan_text)
    learning = load_learning(working_text, tasks)
    return graph, tasks, learning


def effective_status(task: dict, task_map: dict[str, dict]) -> str:
    status = task["status"]
    if status in {"completed", "in_progress", "blocked"}:
        return status
    dependencies = task.get("depends_on", [])
    if all(task_map[dep]["status"] == "completed" for dep in dependencies):
        return "ready"
    return "blocked"


def render_learning(learning: list[dict]) -> list[str]:
    """Render the learning lane: terminal study nodes over completed tasks."""

    if not learning:
        return []

    lines = [
        "",
        "## Learning checkpoints",
        "",
        "Study units built from completed tasks. They depend on tasks; no task "
        "depends on them, so they never gate implementation work. Source: "
        f"`{LANE_PATH}`.",
        "",
        "```mermaid",
        "graph LR",
    ]
    for entry in learning:
        subject = entry["subject"].replace('"', "'")
        node = entry["id"].replace("-", "_")
        lines.append(f'    {node}{{{{"{entry["id"]}: {subject}"}}}}')
    for entry in learning:
        node = entry["id"].replace("-", "_")
        for task_id in entry["covers"]:
            lines.append(f"    {task_id} --> {node}")
    lines.extend(["```", ""])

    lines.append("| Checkpoint | Subject | Covers | Built | Sheet |")
    lines.append("|---|---|---|---|---|")
    for entry in learning:
        state = entry["built"]
        if entry["changed"]:
            state = f"{entry['built']} (stale: {len(entry['changed'])})"
        lines.append(
            f"| {entry['id']} — {entry['title']} | {entry['subject']} | "
            f"{', '.join(entry['covers'])} | {state} | `{entry['sheet']}` |"
        )

    stale = [entry for entry in learning if entry["changed"]]
    if stale:
        lines.extend(["", "Rebuild and republish these sheets:", ""])
        for entry in stale:
            changed = ", ".join(f"`{path}`" for path in entry["changed"])
            lines.append(
                f"- {entry['id']}: {changed} changed since {entry['built']}. "
                f"Run `scripts/learning/build_learning_sheet.py {entry['id']} "
                "--record` after republishing."
            )
    return lines


def render(graph: dict, tasks: list[dict], learning: list[dict] | None = None) -> str:
    task_map = {task["id"]: task for task in tasks}
    lines = [
        "<!-- Generated by scripts/ai/render_task_status.py; do not edit. -->",
        "",
        "# Task status",
        "",
        f"Source: `{GRAPH_PATH.relative_to(REPO_ROOT)}`",
        "",
        "## Dependency graph",
        "",
        "```mermaid",
        "graph TD",
    ]

    for task in tasks:
        title = task["title"].replace('"', "'")
        lines.append(f'    {task["id"]}["{task["id"]}: {title}"]')
    for task in tasks:
        for dependency in task.get("depends_on", []):
            lines.append(f"    {dependency} --> {task['id']}")
    lines.extend(["```", "", "## Status", ""])
    lines.append("| Task | Status | Dependencies | Resource locks | Worklog |")
    lines.append("|---|---|---|---|---|")

    for task in tasks:
        deps = ", ".join(task.get("depends_on", [])) or "—"
        locks = ", ".join(task.get("resource_locks", [])) or "—"
        worklog = task.get("worklog") or "—"
        status = effective_status(task, task_map)
        lines.append(
            f"| {task['id']} — {task['title']} | {status} | "
            f"{deps} | {locks} | {worklog} |"
        )

    lines.extend(["", "## Summary", ""])
    counts = Counter(effective_status(task, task_map) for task in tasks)
    for status in ("ready", "in_progress", "blocked", "completed"):
        lines.append(f"- {status}: {counts.get(status, 0)}")

    lines.extend(["", "## Resource capacities", ""])
    lines.append("| Resource | Capacity | Spending approval |")
    lines.append("|---|---:|---|")
    for name, resource in graph.get("resource_locks", {}).items():
        if resource.get("requires_spending_approval"):
            approval = "required"
        elif resource.get("paid_fallback_requires_approval"):
            approval = "required only for paid fallback"
        else:
            approval = "no"
        lines.append(f"| {name} | {resource.get('capacity', 1)} | {approval} |")

    lines.extend(render_learning(learning or []))

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    try:
        graph, tasks, learning = load_and_validate()
        rendered = render(graph, tasks, learning)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.check:
        if not OUTPUT_PATH.is_file():
            print(
                f"error: missing {OUTPUT_PATH.relative_to(REPO_ROOT)}", file=sys.stderr
            )
            return 1
        if OUTPUT_PATH.read_text(encoding="utf-8") != rendered:
            print(
                f"error: stale {OUTPUT_PATH.relative_to(REPO_ROOT)}; "
                "run scripts/ai/render_task_status.py",
                file=sys.stderr,
            )
            return 1
        print(
            f"task graph valid; {len(tasks)} tasks; "
            f"{len(learning)} learning checkpoints; generated status is current"
        )
        return 0

    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
