#!/usr/bin/env python3
"""Create ignored local AI directories and an optional artifact-root symlink."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIRS = (
    ".ai-local/inputs",
    ".ai-local/plans",
    ".ai-local/tasks",
    ".ai-local/handoffs",
    ".ai-local/worklogs",
    ".ai-local/profiles",
    ".ai-local/scratch",
)
LOCAL_README = """# Local AI workspace

This directory is intentionally ignored by Git. It holds private inputs, draft
plans, raw worklogs, real Codex task/thread identifiers, unsanitized profiles,
and scratch experiments.

Move only sanitized, durable material into the corresponding public `ai/` or
`docs/` directory.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default=os.environ.get(
            "SLM_LAB_ARTIFACT_ROOT",
            "/Volumes/T9/slm-deployment-lab",
        ),
        help="Existing external artifact directory",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for relative in LOCAL_DIRS:
        (REPO_ROOT / relative).mkdir(parents=True, exist_ok=True)

    local_readme = REPO_ROOT / ".ai-local" / "README.md"
    if not local_readme.exists():
        local_readme.write_text(LOCAL_README, encoding="utf-8")

    registry = REPO_ROOT / ".ai-local" / "tasks" / "thread-registry.yaml"
    registry_example = REPO_ROOT / "ai" / "tasks" / "thread_registry.example.yaml"
    if not registry.exists():
        if registry_example.is_file():
            registry.write_text(
                registry_example.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        else:
            registry.write_text(
                "schema_version: 1\ntasks: {}\n",
                encoding="utf-8",
            )

    artifact_root = Path(args.artifact_root).expanduser()
    link = REPO_ROOT / "artifacts"
    if link.is_symlink():
        if link.resolve() != artifact_root.resolve():
            print(
                f"error: artifacts symlink points to {link.resolve()}, "
                f"not {artifact_root}",
                file=sys.stderr,
            )
            return 1
    elif link.exists():
        print("error: artifacts exists but is not a symlink", file=sys.stderr)
        return 1
    elif artifact_root.is_dir():
        link.symlink_to(artifact_root, target_is_directory=True)
    elif not args.quiet:
        print(
            f"warning: artifact root is unavailable; no symlink created: "
            f"{artifact_root}",
            file=sys.stderr,
        )

    if not args.quiet:
        print("local AI workspace is ready")
        if link.is_symlink():
            print(f"artifacts -> {link.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
