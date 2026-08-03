#!/usr/bin/env python3
"""Regenerate the project dashboard's generated regions and cross-check its prose.

The dashboard at ``docs/dashboard/index.html`` is an authored HTML page with
machine-generated regions delimited by::

    <!-- BEGIN GENERATED: <name> -->
    ...
    <!-- END GENERATED: <name> -->

This script rewrites those regions from ``ai/tasks/task_graph.yaml`` and the
learning lane, and cross-checks the authored prose against the graph:

* every task badge inside ``<section id="done">`` names a completed task;
* every task badge inside ``<section id="next">`` names a ready task;
* every checkpoint in ``ai/tasks/learning_lane.yaml`` has a digest card and a
  table-of-contents row in ``<section id="learning">``, and vice versa.

Prose drift is reported, never rewritten: the numbers are generated, the words
stay authored. Graph *validation* is owned by ``scripts/ai/render_task_status.py``;
this script only reads the same JSON-compatible bytes.

The page body is an artifact-ready fragment (no ``<html>``/``<head>``/``<body>``
shell); browsers and the Claude Code Artifact tool both supply the shell.

    python3 scripts/dashboard/build_dashboard.py          # rewrite regions
    python3 scripts/dashboard/build_dashboard.py --check  # fail if stale or drifted
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = REPO_ROOT / "ai" / "tasks" / "task_graph.yaml"
LANE_PATH = REPO_ROOT / "ai" / "tasks" / "learning_lane.yaml"
DASHBOARD_PATH = REPO_ROOT / "docs" / "dashboard" / "index.html"

# The Qualcomm critical path shown on the "road to the NPU" rail, in order.
# Titles and sublines are display copy owned by this script; statuses come
# from the task graph, and the "you are here" marker lands on the first
# node that is not yet completed.
CRITICAL_PATH = [
    ("T20", "Export 8 ONNX graphs", "Prefill + decode at 4 context sizes"),
    (
        "T21",
        "Inspect graphs + parity machinery",
        "Graph inspection + ORT parity runner",
    ),
    (
        "T23",
        "Fix prefill export + measure parity",
        "Concat cache write · ORT CPU 1.28.0",
    ),
    ("T22", "Package QNN candidates", "Compiler-friendly rewrites"),
    ("T31", "Compile &amp; profile on 3 Qualcomm targets", "X Elite, IQ-9075, 8 Elite"),
    ("T33", "Qualcomm floating-point milestone", "End-to-end NPU generation"),
    ("T43", "Quantized NPU deployment", "W8 / W4 on device"),
]

CHIP = {
    "completed": ("done", "Done"),
    "ready": ("ready", "Ready"),
    "in_progress": ("ready", "In progress"),
    "blocked": ("blocked", "Blocked"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when the dashboard is missing, stale, or drifted from the graph",
    )
    return parser.parse_args()


def load_tasks(graph_text: str) -> dict[str, dict]:
    graph = json.loads(graph_text)
    tasks = graph.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("task_graph.yaml must contain a tasks list")
    return {task["id"]: task for task in tasks}


def effective_status(task: dict, task_map: dict[str, dict]) -> str:
    # Mirrors scripts/ai/render_task_status.py::effective_status.
    status = task["status"]
    if status in {"completed", "in_progress", "blocked"}:
        return status
    if all(
        task_map[dep]["status"] == "completed" for dep in task.get("depends_on", [])
    ):
        return "ready"
    return "blocked"


def render_status_region(task_map: dict[str, dict]) -> str:
    counts = Counter(effective_status(task, task_map) for task in task_map.values())
    total = len(task_map)
    completed = counts.get("completed", 0)
    ready = counts.get("ready", 0)
    in_progress = counts.get("in_progress", 0)
    blocked = counts.get("blocked", 0)

    lines = [
        '    <div class="tiles">',
        '      <div class="stat"><div class="num">'
        f'{completed}<span style="color:var(--ink-3);font-size:1.1rem;">/{total}</span></div>'
        '<div class="lbl"><span class="dot" style="background:var(--bar-done)"></span>Completed</div></div>',
        f'      <div class="stat"><div class="num">{ready}</div>'
        '<div class="lbl"><span class="dot" style="background:var(--bar-ready)"></span>Ready to start</div></div>',
        f'      <div class="stat"><div class="num">{blocked}</div>'
        '<div class="lbl"><span class="dot" style="background:var(--bar-blocked)"></span>Blocked (waiting on others)</div></div>',
        f'      <div class="stat"><div class="num">{in_progress}</div><div class="lbl">In progress</div></div>',
        "    </div>",
        "",
    ]

    label = f"Task progress: {completed} completed, "
    if in_progress:
        label += f"{in_progress} in progress, "
    label += f"{ready} ready, {blocked} blocked, of {total} tasks"
    lines.append(f'    <div class="bar" role="img" aria-label="{label}">')
    lines.append(
        f'      <span style="flex:{completed};background:var(--bar-done)"></span>'
    )
    if in_progress:
        lines.append(
            f'      <span style="flex:{in_progress};background:var(--copper)"></span>'
        )
    lines.append(
        f'      <span style="flex:{ready};background:var(--bar-ready)"></span>'
    )
    lines.append(
        f'      <span style="flex:{blocked};background:var(--bar-blocked)"></span>'
    )
    lines.append("    </div>")
    return "\n".join(lines)


def render_rail_region(task_map: dict[str, dict]) -> str:
    unknown = [task_id for task_id, _, _ in CRITICAL_PATH if task_id not in task_map]
    if unknown:
        raise ValueError(f"critical path names unknown tasks: {', '.join(unknown)}")

    here = next(
        (
            task_id
            for task_id, _, _ in CRITICAL_PATH
            if effective_status(task_map[task_id], task_map) != "completed"
        ),
        None,
    )

    lines = ['      <div class="rail">']
    for index, (task_id, title, subline) in enumerate(CRITICAL_PATH):
        if index:
            lines.append('        <div class="arrow">→</div>')
        chip_class, chip_label = CHIP[effective_status(task_map[task_id], task_map)]
        node_class = "node here" if task_id == here else "node"
        chip = f'<span class="chip {chip_class}">{chip_label}</span>'
        if task_id == here:
            chip += ' <span class="here-tag">← you are here</span>'
        lines.extend(
            [
                f'        <div class="{node_class}">',
                f"          {chip}",
                f'          <div class="t">{task_id} · {title}</div>',
                '          <div style="font-size:0.8rem;color:var(--ink-2);">'
                f"{subline}</div>",
                "        </div>",
            ]
        )
    lines.append("      </div>")
    return "\n".join(lines)


def replace_region(text: str, name: str, content: str) -> str:
    pattern = re.compile(
        rf"(<!-- BEGIN GENERATED: {re.escape(name)} -->\n).*?"
        rf"(^\s*<!-- END GENERATED: {re.escape(name)} -->)",
        flags=re.DOTALL | re.MULTILINE,
    )
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise ValueError(
            f"dashboard must contain exactly one generated region named {name!r}"
        )
    return pattern.sub(lambda m: f"{m.group(1)}{content}\n{m.group(2)}", text, count=1)


def section(text: str, section_id: str) -> str:
    match = re.search(
        rf'<section id="{re.escape(section_id)}".*?</section>', text, flags=re.DOTALL
    )
    if not match:
        raise ValueError(f'dashboard is missing <section id="{section_id}">')
    return match.group(0)


def check_prose(text: str, task_map: dict[str, dict], lane_text: str) -> list[str]:
    errors: list[str] = []

    def badge_ids(section_id: str) -> list[str]:
        return re.findall(r'<span class="tid">(T\d+)</span>', section(text, section_id))

    for task_id in badge_ids("done"):
        if task_id not in task_map:
            errors.append(f"done section names unknown task {task_id}")
        elif effective_status(task_map[task_id], task_map) != "completed":
            errors.append(
                f"done section lists {task_id}, which is "
                f"{effective_status(task_map[task_id], task_map)!r}"
            )

    for task_id in badge_ids("next"):
        if task_id not in task_map:
            errors.append(f"next section names unknown task {task_id}")
        elif effective_status(task_map[task_id], task_map) != "ready":
            errors.append(
                f"next section lists {task_id}, which is "
                f"{effective_status(task_map[task_id], task_map)!r}; update the prose"
            )

    ready = sorted(
        task_id
        for task_id, task in task_map.items()
        if effective_status(task, task_map) == "ready"
    )
    missing_ready = sorted(set(ready) - set(badge_ids("next")))
    if missing_ready:
        errors.append(
            f"ready tasks missing from the next section: {', '.join(missing_ready)}"
        )

    lane = json.loads(lane_text)
    lane_numbers = set()
    for entry in lane.get("checkpoints", []):
        match = re.fullmatch(r"LEARN-(\d{2})", entry.get("id", ""))
        if not match:
            errors.append(
                f"learning lane has invalid checkpoint ID {entry.get('id')!r}"
            )
            continue
        lane_numbers.add(match.group(1))
    learning = section(text, "learning")
    cards = set(re.findall(r'<article class="learn" id="l(\d{2})">', learning))
    toc = set(re.findall(r'href="#l(\d{2})"', learning))
    for label, present in (("digest card", cards), ("table-of-contents row", toc)):
        missing = sorted(lane_numbers - present)
        if missing:
            errors.append(
                f"checkpoints missing a {label}: "
                + ", ".join(f"LEARN-{number}" for number in missing)
            )
        extra = sorted(present - lane_numbers)
        if extra:
            errors.append(
                f"{label} for unknown checkpoints: "
                + ", ".join(f"LEARN-{number}" for number in extra)
            )
    return errors


def build(
    dashboard_text: str, graph_text: str, lane_text: str
) -> tuple[str, list[str]]:
    """Return the regenerated dashboard text and any prose-drift errors."""
    task_map = load_tasks(graph_text)
    text = replace_region(dashboard_text, "status", render_status_region(task_map))
    text = replace_region(text, "critical-path", render_rail_region(task_map))
    return text, check_prose(text, task_map, lane_text)


def main() -> int:
    args = parse_args()
    try:
        dashboard_text = DASHBOARD_PATH.read_text(encoding="utf-8")
        graph_text = GRAPH_PATH.read_text(encoding="utf-8")
        lane_text = LANE_PATH.read_text(encoding="utf-8")
        rendered, errors = build(dashboard_text, graph_text, lane_text)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for message in errors:
        print(f"error: {message}", file=sys.stderr)

    if args.check:
        if rendered != dashboard_text:
            print(
                f"error: stale {DASHBOARD_PATH.relative_to(REPO_ROOT)}; "
                "run scripts/dashboard/build_dashboard.py",
                file=sys.stderr,
            )
            return 1
        if errors:
            return 1
        print("dashboard generated regions are current and prose matches the graph")
        return 0

    if rendered != dashboard_text:
        DASHBOARD_PATH.write_text(rendered, encoding="utf-8")
        print(f"wrote {DASHBOARD_PATH.relative_to(REPO_ROOT)}")
    else:
        print(f"{DASHBOARD_PATH.relative_to(REPO_ROOT)} already current")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
