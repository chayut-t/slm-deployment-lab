#!/usr/bin/env python3
"""Render a learning-checkpoint study sheet from the repository's own documents.

A checkpoint groups completed task-graph work into one subject so the learner
studies the subject instead of reverse-engineering several task definitions.
The checkpoints are declared in ``configs/learning/checkpoints.yaml``; the
Markdown files they cite are the source of truth.

This script mirrors those documents into a self-contained HTML page
**verbatim**: it extracts, converts, and styles, but never rewrites prose.
``tests/learning/test_learning_sheet.py`` enforces that by tokenizing every
source document and asserting each token survives into the rendered page.

    uv run python scripts/learning/build_learning_sheet.py --all
    uv run python scripts/learning/build_learning_sheet.py LEARN-03
    uv run python scripts/learning/build_learning_sheet.py --check

Output lives under ``build/learning/`` and is generated, not authored: it is
not committed. Re-run after editing any cited document, then republish the
sheet as an artifact and record it:

    uv run python scripts/learning/build_learning_sheet.py LEARN-03 --record
    uv run python scripts/ai/render_task_status.py

``--record`` refreshes ``ai/tasks/learning_lane.yaml``, which places each
checkpoint in the dependency graph over the completed tasks it covers and lets
the task status report which sheets have gone stale.

Requires ``markdown``, which arrives with the ``dev`` extra (through mkdocs):

    uv sync --extra dev
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

try:
    import markdown
except ModuleNotFoundError:  # pragma: no cover - exercised by the error path only
    sys.exit(
        "markdown is not installed. It arrives with the 'dev' extra:\n"
        "    uv sync --extra dev"
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent / "learning_sheet"
CONFIG = REPO_ROOT / "configs" / "learning" / "checkpoints.yaml"
TASK_GRAPH = REPO_ROOT / "ai" / "tasks" / "task_graph.yaml"
LANE_PATH = REPO_ROOT / "ai" / "tasks" / "learning_lane.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "build" / "learning"


class ConfigError(RuntimeError):
    """The checkpoint configuration disagrees with the repository."""


@dataclass(frozen=True)
class Reading:
    """One mirrored document, or a named slice of one."""

    key: str
    label: str
    title: str
    source: str
    required: bool
    why: str
    text: str
    stamp: str

    @property
    def lines(self) -> int:
        return len(self.text.splitlines())


# -- source extraction ----------------------------------------------------


def digest(path: Path) -> str:
    """Digest the decoded text, so a staged and a working read agree."""

    return text_digest(path.read_text(encoding="utf-8"))


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def slice_between(text: str, start: str, stop: str, source: str) -> str:
    """Take the lines from the line starting with ``start`` up to ``stop``."""

    lines = text.splitlines()
    try:
        first = next(i for i, line in enumerate(lines) if line.startswith(start))
        last = next(
            i for i, line in enumerate(lines) if i > first and line.startswith(stop)
        )
    except StopIteration:  # a heading was renamed upstream of this config
        raise ConfigError(
            f"{source}: cannot locate the range {start!r} .. {stop!r}"
        ) from None
    return "\n".join(lines[first:last]).strip() + "\n"


def drop_title(text: str) -> str:
    """Remove a document's own H1: the reading card already names it."""

    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).lstrip("\n")


def load_reading(spec: dict) -> Reading:
    path = REPO_ROOT / spec["path"]
    if not path.is_file():
        raise ConfigError(f"reading {spec['key']!r} points at a missing file: {path}")

    text = path.read_text(encoding="utf-8")
    source = spec["path"]
    if "slice" in spec:
        bounds = spec["slice"]
        text = slice_between(text, bounds["from"], bounds["to"], source)
        source = f"{source} ({bounds['from'].lstrip('# ')} …)"
    else:
        text = drop_title(text)

    return Reading(
        key=spec["key"],
        label=spec["label"],
        title=spec["title"],
        source=source,
        required=bool(spec.get("required", False)),
        why=spec["why"],
        text=text,
        stamp=digest(path),
    )


# -- markdown to html -----------------------------------------------------

MERMAID = re.compile(
    r'<pre><code class="language-mermaid">(.*?)</code></pre>', re.DOTALL
)
# Repository-relative links cannot resolve inside a published page. Render them
# as inert path references rather than as links that lead nowhere.
LOCAL_LINK = re.compile(r'<a href="(?!https?:)[^"]*">(.*?)</a>', re.DOTALL)


def to_html(text: str) -> str:
    rendered = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    rendered = MERMAID.sub(
        lambda match: f'<pre class="mermaid">{html.unescape(match.group(1)).strip()}</pre>',
        rendered,
    )
    rendered = LOCAL_LINK.sub(r'<span class="reflink">\1</span>', rendered)
    # Wide tables scroll inside their own container so the page body never does.
    rendered = rendered.replace("<table>", '<div class="scroller"><table>')
    return rendered.replace("</table>", "</table></div>")


def inline(text: str) -> str:
    """Render one short field: escaped, with `code` spans and em dashes kept."""

    escaped = html.escape(text.strip())
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)


# -- repository state -----------------------------------------------------


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def load_tasks() -> dict[str, dict]:
    graph = yaml.safe_load(TASK_GRAPH.read_text(encoding="utf-8"))
    return {task["id"]: task for task in graph["tasks"]}


def slug_for(checkpoint_id: str) -> str:
    return checkpoint_id.lower().replace("_", "-")


def display_path(path: Path) -> str:
    """Repository-relative when it can be, absolute when it cannot."""

    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


# -- page assembly --------------------------------------------------------


def render_parts(checkpoint: dict, tasks: dict[str, dict]) -> str:
    cells = []
    for task_id in checkpoint["tasks"]:
        task = tasks.get(task_id)
        if task is None:
            raise ConfigError(f"{checkpoint['id']} cites unknown task {task_id}")
        if task["status"] != "completed":
            raise ConfigError(
                f"{checkpoint['id']} cites {task_id}, whose status is "
                f"{task['status']!r}; checkpoints cover completed work only"
            )
        worklog = task.get("worklog") or "no public worklog"
        cells.append(
            '<div class="part">'
            f'<div class="no">{html.escape(task_id)}</div>'
            f'<div class="what">{inline(task.get("title", ""))}</div>'
            f'<div class="ev">{html.escape(Path(str(worklog)).name)}</div>'
            "</div>"
        )
    return "\n      ".join(cells)


def render_list(items: list[str], extra_class: str = "") -> str:
    classes = f' class="{extra_class}"' if extra_class else ""
    return "\n        ".join(f"<li{classes}>{inline(item)}</li>" for item in items)


def render_readings(readings: list[Reading]) -> str:
    blocks = []
    for index, item in enumerate(readings):
        badge = (
            '<span class="chip is-accent">required</span>'
            if item.required
            else '<span class="chip is-quiet">supplementary</span>'
        )
        blocks.append(
            f'<details class="reading" data-key="{item.key}" '
            f'data-required="{1 if item.required else 0}"'
            f"{' open' if index == 0 else ''}>"
            "<summary>"
            f'<span class="rlabel">{html.escape(item.label)}</span>'
            f'<span class="rtitle">{inline(item.title)}</span>'
            f'<span class="rmeta">{item.lines} lines · <span class="state"></span></span>'
            "</summary>"
            '<div class="rbody">'
            f'<div class="chips">{badge}'
            f'<span class="chip is-quiet">{html.escape(item.source)}</span>'
            f'<span class="chip is-quiet">sha {item.stamp}</span></div>'
            f'<p class="rwhy">{inline(item.why)}</p>'
            f'<article class="doc">{to_html(item.text)}</article>'
            '<div class="rfoot">'
            f'<label class="readmark"><input type="checkbox" data-read="{item.key}" />'
            "<span>Read in full</span></label>"
            "</div>"
            "</div>"
            "</details>"
        )
    return "\n      ".join(blocks)


def render_labs(labs: list[dict]) -> str:
    blocks = []
    for lab in labs:
        command = html.escape(str(lab["run"]).strip())
        blocks.append(
            '<div class="lab">'
            f'<div class="what">{inline(lab["what"])}</div>'
            f"<pre><code>{command}</code></pre>"
            f'<div class="proves">{inline(lab["proves"])}</div>'
            "</div>"
        )
    return "\n      ".join(blocks)


def render_notebooks(notebooks: list[dict]) -> str:
    if not notebooks:
        return ""
    cells = []
    for notebook in notebooks:
        cells.append(
            '<div class="nbi">'
            f'<div class="name">{html.escape(notebook["name"])}</div>'
            f'<div class="chips"><span class="chip is-bound">'
            f"{html.escape(notebook['status'])}</span>"
            f'<span class="chip is-quiet">owner {html.escape(notebook["owner"])}</span></div>'
            f'<div class="focus">{inline(notebook["focus"])}</div>'
            "</div>"
        )
    return (
        '<div class="head" style="border-top:0;padding-top:0.5rem">'
        "<h3>Jupyter labs for this subject</h3>"
        "<p>The notebooks are the experiment surface: runnable baseline, editable "
        "parameters, assertions, plots, interpretation. They are created and "
        "integrated by their owning task, so a planned notebook is a promise, not "
        "a file you can open yet.</p></div>"
        '<div class="nb">' + "\n      ".join(cells) + "</div>"
    )


def render_questions(questions: list[str]) -> str:
    return "\n      ".join(
        '<div class="ask">'
        f'<span class="n">{index:02d}</span>'
        f'<span class="q">{inline(question)}</span>'
        "</div>"
        for index, question in enumerate(questions, start=1)
    )


def render_provenance(readings: list[Reading]) -> str:
    return "\n            ".join(
        f'<tr><td class="path">{html.escape(item.source)}</td>'
        f'<td class="num">{item.lines}</td>'
        f"<td>{item.stamp}</td></tr>"
        for item in readings
    )


def render_series(config: dict, current: str) -> str:
    parts = []
    for checkpoint in config["checkpoints"]:
        css = ' class="self"' if checkpoint["id"] == current else ""
        parts.append(
            f"<span{css}>{html.escape(checkpoint['id'])} "
            f"{html.escape(checkpoint['title'])}</span>"
        )
    return "\n      ".join(parts)


def render_chips(
    checkpoint: dict, tasks: dict[str, dict], readings: list[Reading]
) -> str:
    required = sum(1 for item in readings if item.required)
    chips = [
        f'<span class="chip is-accent">{len(checkpoint["tasks"])} completed '
        f"{'task' if len(checkpoint['tasks']) == 1 else 'tasks'}</span>",
        f'<span class="chip">{required} required '
        f"{'reading' if required == 1 else 'readings'}</span>",
        f'<span class="chip">{len(readings) - required} supplementary</span>',
        '<span class="chip is-proven">study surface, not a gate</span>',
    ]
    if checkpoint.get("hands_on"):
        chips.append('<span class="chip is-bound">hands-on step</span>')
    return "\n      ".join(chips)


def build(checkpoint: dict, config: dict, tasks: dict[str, dict]) -> str:
    readings = [load_reading(spec) for spec in checkpoint["readings"]]
    required = sum(1 for item in readings if item.required)
    style = (ASSETS / "sheet.css").read_text(encoding="utf-8")
    page = (ASSETS / "template.html").read_text(encoding="utf-8")

    hands_on = ""
    if checkpoint.get("hands_on"):
        hands_on = f'<div class="handson">{inline(checkpoint["hands_on"])}</div>'

    built = (
        f"Generated {dt.date.today().isoformat()} by "
        "scripts/learning/build_learning_sheet.py from "
        "configs/learning/checkpoints.yaml. Read marks live in this browser only; "
        "nothing on this page writes to the repository."
    )

    replacements = {
        "<!--TITLE-->": html.escape(f"{checkpoint['id']} — {checkpoint['title']}"),
        "<!--ID-->": html.escape(checkpoint["id"]),
        "<!--SLUG-->": slug_for(checkpoint["id"]),
        "<!--ATTENTION-->": html.escape(checkpoint["attention"]),
        "<!--HEADLINE-->": inline(checkpoint["title"]),
        "<!--SUBJECT-->": inline(checkpoint["subject"]),
        "<!--LEDE-->": inline(checkpoint["lede"]),
        "<!--CHIPS-->": render_chips(checkpoint, tasks, readings),
        "<!--PARTS-->": render_parts(checkpoint, tasks),
        "<!--OUTCOMES-->": render_list(checkpoint["outcomes"]),
        "<!--HANDSON-->": hands_on,
        "<!--READINGS-->": render_readings(readings),
        "<!--REQUIRED-->": str(required),
        "<!--LABS-->": render_labs(checkpoint["labs"]),
        "<!--NOTEBOOKS-->": render_notebooks(checkpoint.get("notebooks", [])),
        "<!--QUESTIONS-->": render_questions(checkpoint["questions"]),
        "<!--BOUNDARIES-->": render_list(checkpoint["boundaries"]),
        "<!--PROVENANCE-->": render_provenance(readings),
        "<!--SERIES-->": render_series(config, checkpoint["id"]),
        "<!--BUILT-->": built,
        "<!--STYLE-->": f"<style>\n{style}\n</style>",
    }
    for placeholder, value in replacements.items():
        if placeholder not in page:
            raise ConfigError(f"template is missing the {placeholder} placeholder")
        page = page.replace(placeholder, value)
    return page


# -- the learning lane ----------------------------------------------------
#
# The task-status renderer must be able to place checkpoints in the dependency
# graph without importing YAML or Markdown, and must read identical bytes from
# a staged snapshot. So the human-facing configuration here is projected into
# one small JSON-compatible file that the renderer consumes, exactly like
# ai/tasks/task_graph.yaml.


def sheet_path(checkpoint_id: str) -> str:
    relative = DEFAULT_OUTPUT.relative_to(REPO_ROOT)
    return f"{relative.as_posix()}/{slug_for(checkpoint_id)}.html"


def lane_entry(checkpoint: dict, built: str) -> dict:
    sources = {}
    for spec in checkpoint["readings"]:
        sources[spec["path"]] = digest(REPO_ROOT / spec["path"])
    return {
        "id": checkpoint["id"],
        "title": checkpoint["title"],
        "subject": checkpoint["subject"],
        "covers": list(checkpoint["tasks"]),
        "sheet": sheet_path(checkpoint["id"]),
        "built": built,
        "sources": dict(sorted(sources.items())),
    }


def load_lane() -> dict[str, dict]:
    if not LANE_PATH.is_file():
        return {}
    lane = json.loads(LANE_PATH.read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in lane.get("checkpoints", [])}


def write_lane(entries: dict[str, dict]) -> None:
    document = {
        "schema_version": 1,
        "generated_by": "scripts/learning/build_learning_sheet.py --record",
        "config": str(CONFIG.relative_to(REPO_ROOT)),
        "checkpoints": [entries[key] for key in sorted(entries)],
    }
    LANE_PATH.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def structure(entry: dict) -> dict:
    return {key: entry[key] for key in ("id", "title", "subject", "covers", "sheet")}


def lane_report(config: dict, recorded: dict[str, dict]) -> tuple[list[str], list[str]]:
    """Return (structural drift, freshness notes) against the recorded lane.

    The split is deliberate. Drift is everything decided by the configuration
    alone — which checkpoints exist, what they cover, and which documents they
    mirror. It is independent of document *content*, so it can be asserted in a
    test without failing every time a cited document is edited. Freshness is
    the content half: a digest that no longer matches the file it was taken
    from, which means the published sheet is out of date.
    """

    drift: list[str] = []
    freshness: list[str] = []
    expected = {
        checkpoint["id"]: lane_entry(checkpoint, "")
        for checkpoint in config["checkpoints"]
    }

    for checkpoint_id in sorted(set(expected) - set(recorded)):
        drift.append(f"{checkpoint_id}: missing from {LANE_PATH.name}")
    for checkpoint_id in sorted(set(recorded) - set(expected)):
        drift.append(f"{checkpoint_id}: recorded but no longer configured")

    for checkpoint_id in sorted(set(expected) & set(recorded)):
        if structure(expected[checkpoint_id]) != structure(recorded[checkpoint_id]):
            drift.append(f"{checkpoint_id}: recorded entry does not match the config")
            continue

        expected_sources = expected[checkpoint_id]["sources"]
        recorded_sources = recorded[checkpoint_id].get("sources", {})
        added = sorted(set(expected_sources) - set(recorded_sources))
        removed = sorted(set(recorded_sources) - set(expected_sources))
        if added or removed:
            detail = ", ".join(
                [f"+{path}" for path in added] + [f"-{path}" for path in removed]
            )
            drift.append(
                f"{checkpoint_id}: recorded sources do not match the config: {detail}"
            )
            continue

        changed = [
            path
            for path, stamp in expected_sources.items()
            if recorded_sources[path] != stamp
        ]
        if changed:
            freshness.append(
                f"{checkpoint_id}: {len(changed)} source(s) changed since "
                f"{recorded[checkpoint_id]['built']} — {', '.join(changed)}"
            )
    return drift, freshness


# -- entry point ----------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "checkpoints",
        nargs="*",
        metavar="LEARN-NN",
        help="checkpoint IDs to render (default: all)",
    )
    parser.add_argument("--all", action="store_true", help="render every checkpoint")
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "validate the configuration and the recorded learning lane, render "
            "in memory, and write nothing"
        ),
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help=(
            "update ai/tasks/learning_lane.yaml for the rendered checkpoints. "
            "Run it when you republish a sheet, so the task status can show "
            "which sheets have gone stale."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"where to write sheets (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})",
    )
    return parser.parse_args(argv)


def select(config: dict, wanted: list[str]) -> list[dict]:
    if not wanted:
        return list(config["checkpoints"])
    known = {checkpoint["id"]: checkpoint for checkpoint in config["checkpoints"]}
    chosen = []
    for name in wanted:
        key = name.upper()
        if key not in known:
            raise ConfigError(f"unknown checkpoint {name!r}; known: {', '.join(known)}")
        chosen.append(known[key])
    return chosen


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config()
    tasks = load_tasks()

    try:
        chosen = select(config, [] if args.all else args.checkpoints)
        pages = {
            checkpoint["id"]: build(checkpoint, config, tasks) for checkpoint in chosen
        }
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    recorded = load_lane()

    if args.check:
        drift, freshness = lane_report(config, recorded)
        for note in freshness:
            print(f"stale: {note}")
        if drift:
            for note in drift:
                print(f"error: {note}", file=sys.stderr)
            print(
                "run: scripts/learning/build_learning_sheet.py --all --record",
                file=sys.stderr,
            )
            return 1
        print(
            f"checked {len(pages)} checkpoint(s); learning lane matches the "
            f"config; {len(freshness)} sheet(s) need rebuilding"
        )
        return 0

    output_dir = args.output_dir.resolve()
    if args.record and output_dir != DEFAULT_OUTPUT:
        print(
            "error: --record writes the committed learning lane, which names the "
            f"canonical sheet location {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}. "
            "Drop --output-dir, or record a separate build.",
            file=sys.stderr,
        )
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint_id, page in pages.items():
        target = output_dir / f"{slug_for(checkpoint_id)}.html"
        target.write_text(page, encoding="utf-8")
        print(f"wrote {display_path(target)} ({len(page):,} bytes)")

    if args.record:
        today = dt.date.today().isoformat()
        for checkpoint in chosen:
            recorded[checkpoint["id"]] = lane_entry(checkpoint, today)
        for stale_id in set(recorded) - {c["id"] for c in config["checkpoints"]}:
            del recorded[stale_id]
        write_lane(recorded)
        print(
            f"recorded {len(chosen)} checkpoint(s) in "
            f"{LANE_PATH.relative_to(REPO_ROOT)}; "
            "re-run scripts/ai/render_task_status.py"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
