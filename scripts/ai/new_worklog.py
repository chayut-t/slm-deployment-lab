#!/usr/bin/env python3
"""Create a public or local worklog from the repository convention."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Task ID such as T31")
    parser.add_argument(
        "--slug",
        required=True,
        help="Short lowercase description used in the filename",
    )
    parser.add_argument(
        "--visibility",
        choices=("public", "local"),
        default="public",
        help="Public logs are committed; local logs remain under .ai-local",
    )
    parser.add_argument(
        "--date",
        default=dt.date.today().isoformat(),
        help="ISO date for the log (default: today)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task = args.task.upper()
    if not re.fullmatch(r"T\d+", task):
        print("error: --task must look like T31", file=sys.stderr)
        return 2

    slug = re.sub(r"[^a-z0-9]+", "-", args.slug.lower()).strip("-")
    if not slug:
        print("error: --slug must contain a letter or number", file=sys.stderr)
        return 2

    try:
        dt.date.fromisoformat(args.date)
    except ValueError:
        print("error: --date must use YYYY-MM-DD", file=sys.stderr)
        return 2

    base = (
        REPO_ROOT / "ai" / "worklogs"
        if args.visibility == "public"
        else REPO_ROOT / ".ai-local" / "worklogs"
    )
    base.mkdir(parents=True, exist_ok=True)
    destination = base / f"{args.date}-{task}-{slug}.md"
    if destination.exists():
        print(f"error: worklog already exists: {destination}", file=sys.stderr)
        return 1

    destination.write_text(
        f"""# {task}: {args.slug.replace("-", " ").title()}

Date: {args.date}
Task: `{task}`
Visibility: `{args.visibility}`
Status: draft

## Outcome

Describe the completed engineering outcome.

## Changes

- TODO

## Verification

- Command:
- Result:

## Decisions and evidence

- TODO

## Risks and limitations

- TODO

## Follow-up

- Newly unblocked tasks:
- Recommended next action:
""",
        encoding="utf-8",
    )
    print(destination.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
