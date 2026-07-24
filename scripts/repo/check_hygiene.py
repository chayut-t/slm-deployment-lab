#!/usr/bin/env python3
"""Check repository privacy, generated state, secrets, and file-size policy."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_COMMITTED_BYTES = 10 * 1024 * 1024
FORBIDDEN_ROOTS = {".ai-local", "artifacts"}
FORBIDDEN_PREFIXES = (".ai-local/", "artifacts/")
FORBIDDEN_NAMES = {
    ".env",
    "slm_deployment_lab_project_plan_feedback.md",
    "thread_registry.local.yaml",
}
REQUIRED_FILES = {
    ".gitattributes",
    ".gitignore",
    ".githooks/pre-commit",
    "README.md",
    "AGENTS.md",
    "PLANS.md",
    "mkdocs.yml",
    "pyproject.toml",
    "docs/project/plan.md",
    "ai/tasks/task_graph.yaml",
    "ai/tasks/status.generated.md",
    "scripts/ai/render_task_status.py",
    "scripts/repo/check_hygiene.py",
}
SECRET_PATTERNS = (
    (re.compile(rb"hf_[A-Za-z0-9]{24,}"), "Hugging Face token"),
    (re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"), "GitHub token"),
    (re.compile(rb"sk-[A-Za-z0-9_-]{32,}"), "API key"),
    (re.compile(rb"AKIA[0-9A-Z]{16}"), "AWS access key"),
    (
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "private key",
    ),
)


def load_task_renderer():
    renderer_path = REPO_ROOT / "scripts" / "ai" / "render_task_status.py"
    spec = importlib.util.spec_from_file_location("task_status_renderer", renderer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load task renderer: {renderer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", help="Check tracked and untracked files")
    mode.add_argument("--staged", action="store_true", help="Check the staged snapshot")
    return parser.parse_args()


def git_output(*args: str) -> bytes:
    return subprocess.check_output(("git", *args), cwd=REPO_ROOT)


def candidate_paths(staged: bool) -> list[str]:
    if staged:
        output = git_output(
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACMRD",
            "-z",
        )
    else:
        output = git_output(
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        )
    return sorted(path for path in output.decode().split("\0") if path)


def staged_content(path: str) -> bytes:
    return git_output("show", f":{path}")


def index_has(path: str) -> bool:
    check = subprocess.run(
        ("git", "cat-file", "-e", f":{path}"),
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return check.returncode == 0


def index_text(path: str) -> str:
    if not index_has(path):
        raise FileNotFoundError(path)
    return staged_content(path).decode("utf-8")


def working_content(path: str) -> bytes:
    return (REPO_ROOT / path).read_bytes()


def main() -> int:
    args = parse_args()
    staged = args.staged
    errors: list[str] = []

    try:
        paths = candidate_paths(staged)
    except subprocess.CalledProcessError as exc:
        print(f"error: cannot enumerate Git files: {exc}", file=sys.stderr)
        return 1

    if staged:
        for required in sorted(REQUIRED_FILES):
            if not index_has(required):
                errors.append(f"required file is missing from staged snapshot: {required}")
    elif not staged:
        for required in sorted(REQUIRED_FILES):
            if not (REPO_ROOT / required).is_file():
                errors.append(f"required repository file is missing: {required}")

    for path in paths:
        if staged and not index_has(path):
            # Deletions must trigger whole-index validation, but there is no
            # staged blob to scan for size or secret patterns.
            continue

        normalized = path.replace(os.sep, "/")
        name = Path(normalized).name
        if (
            normalized in FORBIDDEN_ROOTS
            or normalized.startswith(FORBIDDEN_PREFIXES)
        ):
            errors.append(f"private or external path cannot be committed: {normalized}")
            continue
        if name in FORBIDDEN_NAMES or (
            name.startswith(".env.") and name != ".env.example"
        ):
            errors.append(f"private filename cannot be committed: {normalized}")
            continue
        if "thread_registry.local." in name:
            errors.append(f"local thread registry cannot be committed: {normalized}")
            continue

        filesystem_path = REPO_ROOT / path
        if not staged and (filesystem_path.is_symlink() or not filesystem_path.is_file()):
            continue

        try:
            content = staged_content(path) if staged else working_content(path)
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"cannot inspect {normalized}: {exc}")
            continue

        if len(content) > MAX_COMMITTED_BYTES:
            errors.append(
                f"file exceeds 10 MiB repository limit: {normalized} "
                f"({len(content)} bytes)"
            )
            continue

        for pattern, label in SECRET_PATTERNS:
            if pattern.search(content):
                errors.append(f"possible {label} detected in {normalized}")

    renderer_path = REPO_ROOT / "scripts" / "ai" / "render_task_status.py"
    if renderer_path.is_file():
        if staged:
            graph_path = "ai/tasks/task_graph.yaml"
            status_path = "ai/tasks/status.generated.md"
            if index_has(graph_path) and index_has(status_path):
                try:
                    renderer = load_task_renderer()
                    graph, tasks = renderer.validate_graph_text(
                        index_text(graph_path),
                        index_text,
                    )
                    plan_path = graph.get("project_plan")
                    if not isinstance(plan_path, str):
                        raise ValueError("task graph must name a project_plan path")
                    renderer.validate_plan_parity(
                        graph,
                        tasks,
                        index_text(plan_path),
                    )
                    expected_status = renderer.render(graph, tasks)
                    if index_text(status_path) != expected_status:
                        errors.append(
                            "staged ai/tasks/status.generated.md does not match "
                            "the staged task graph"
                        )
                except (
                    FileNotFoundError,
                    RuntimeError,
                    UnicodeDecodeError,
                    ValueError,
                ) as exc:
                    errors.append(f"staged task snapshot validation failed: {exc}")
        elif not staged:
            check = subprocess.run(
                (sys.executable, str(renderer_path), "--check"),
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            if check.returncode:
                detail = (check.stderr or check.stdout).strip()
                errors.append(f"task status validation failed: {detail}")

    if errors:
        print("repository hygiene check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    scope = "staged files" if staged else "tracked and untracked public files"
    print(f"repository hygiene passed for {len(paths)} {scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
