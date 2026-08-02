"""Regression tests for repository task and privacy automation."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = REPO_ROOT / "ai" / "tasks" / "task_graph.yaml"
PLAN_PATH = REPO_ROOT / "docs" / "project" / "plan.md"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RENDERER = load_module(
    "render_task_status_under_test",
    REPO_ROOT / "scripts" / "ai" / "render_task_status.py",
)


class TaskGraphValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))

    def working_text(self, path: str) -> str:
        return (REPO_ROOT / path).read_text(encoding="utf-8")

    def validate(self, graph: dict, read_text=None):
        return RENDERER.validate_graph_text(
            json.dumps(graph),
            read_text or self.working_text,
        )

    def test_current_graph_and_plan_are_valid(self) -> None:
        graph, tasks = self.validate(self.graph)
        RENDERER.validate_plan_parity(
            graph,
            tasks,
            PLAN_PATH.read_text(encoding="utf-8"),
        )

    def test_completed_task_requires_completed_dependencies(self) -> None:
        graph = copy.deepcopy(self.graph)
        tasks_by_id = {item["id"]: item for item in graph["tasks"]}
        task = next(
            item
            for item in graph["tasks"]
            if any(
                tasks_by_id[dependency]["status"] != "completed"
                for dependency in item["depends_on"]
            )
        )
        task["status"] = "completed"
        task["worklog"] = "ai/worklogs/example.md"
        with self.assertRaisesRegex(ValueError, "incomplete dependencies"):
            self.validate(graph)

    def test_completed_task_rejects_private_worklog(self) -> None:
        graph = copy.deepcopy(self.graph)
        task = next(item for item in graph["tasks"] if item["id"] == "T00")
        task["status"] = "completed"
        task["worklog"] = ".ai-local/worklogs/private.md"
        with self.assertRaisesRegex(ValueError, "worklog must be under"):
            self.validate(graph)

    def test_completed_task_rejects_mismatched_worklog_metadata(self) -> None:
        graph = copy.deepcopy(self.graph)
        task = next(item for item in graph["tasks"] if item["id"] == "T00")
        task["status"] = "completed"
        task["worklog"] = "ai/worklogs/fake.md"

        def read_text(path: str) -> str:
            if path == "ai/worklogs/fake.md":
                return (
                    "# Wrong task\n\n"
                    "Task: `T99`\n"
                    "Visibility: `public`\n"
                    "Status: completed\n"
                )
            return self.working_text(path)

        with self.assertRaisesRegex(ValueError, "Task:"):
            self.validate(graph, read_text)

    def test_completed_task_requires_existing_public_worklog(self) -> None:
        graph = copy.deepcopy(self.graph)
        task = next(item for item in graph["tasks"] if item["id"] == "T10")
        task["status"] = "completed"
        task["worklog"] = "ai/worklogs/missing.md"

        with self.assertRaisesRegex(ValueError, "worklog does not exist"):
            self.validate(graph)

    def test_plan_parity_rejects_missing_edge(self) -> None:
        graph, tasks = self.validate(self.graph)
        plan = PLAN_PATH.read_text(encoding="utf-8").replace(
            "    T13 --> T60\n",
            "",
            1,
        )
        with self.assertRaisesRegex(ValueError, "DAG edges differ"):
            RENDERER.validate_plan_parity(graph, tasks, plan)


class GitSnapshotTests(unittest.TestCase):
    def copy_repository(self) -> Path:
        temporary_root = Path(self.addCleanupDirectory())
        clone = temporary_root / "repo"
        shutil.copytree(
            REPO_ROOT,
            clone,
            ignore=shutil.ignore_patterns(
                ".git",
                ".ai-local",
                "artifacts",
                "__pycache__",
            ),
        )
        return clone

    def make_repository(self) -> Path:
        clone = self.copy_repository()
        self.run_git(clone, "init", "-q")
        self.run_git(clone, "config", "user.email", "test@example.invalid")
        self.run_git(clone, "config", "user.name", "Repository Test")
        self.run_git(clone, "add", ".")
        self.run_git(clone, "commit", "-qm", "baseline")
        return clone

    def addCleanupDirectory(self) -> str:
        directory = tempfile.mkdtemp(prefix="slm-lab-test-")
        self.addCleanup(shutil.rmtree, directory, True)
        return directory

    @staticmethod
    def run_git(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("git", *args),
            cwd=repository,
            text=True,
            capture_output=True,
            check=True,
        )

    @staticmethod
    def run_hygiene(repository: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            (
                sys.executable,
                "scripts/repo/check_hygiene.py",
                "--staged",
            ),
            cwd=repository,
            text=True,
            capture_output=True,
            check=False,
        )

    @staticmethod
    def run_pre_commit(repository: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            (str(repository / ".githooks" / "pre-commit"),),
            cwd=repository,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_complete_initial_staged_snapshot_passes(self) -> None:
        repository = self.copy_repository()
        self.run_git(repository, "init", "-q")
        self.run_git(repository, "add", ".")

        result = self.run_hygiene(repository)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pre_commit_uses_project_venv_without_system_python(self) -> None:
        repository = self.make_repository()
        venv_bin = repository / ".venv" / "bin"
        venv_bin.mkdir(parents=True, exist_ok=True)
        venv_python = venv_bin / "python"
        venv_python.unlink(missing_ok=True)
        venv_python.write_text(
            f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n',
            encoding="utf-8",
        )
        venv_python.chmod(0o755)

        git_executable = shutil.which("git")
        self.assertIsNotNone(git_executable)
        isolated_bin = Path(self.addCleanupDirectory()) / "bin"
        isolated_bin.mkdir()
        (isolated_bin / "git").symlink_to(git_executable)

        result = subprocess.run(
            (str(repository / ".githooks" / "pre-commit"),),
            cwd=repository,
            env={**os.environ, "PATH": str(isolated_bin)},
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_staged_invalid_graph_cannot_hide_behind_valid_worktree(self) -> None:
        repository = self.make_repository()
        graph_path = repository / "ai" / "tasks" / "task_graph.yaml"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        graph["tasks"][1]["depends_on"] = ["T999"]
        graph_path.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
        self.run_git(repository, "add", "ai/tasks/task_graph.yaml")

        baseline = self.run_git(
            repository,
            "show",
            "HEAD:ai/tasks/task_graph.yaml",
        ).stdout
        graph_path.write_text(baseline, encoding="utf-8")

        result = self.run_hygiene(repository)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown dependencies", result.stderr)

    def test_staged_graph_requires_matching_staged_status(self) -> None:
        repository = self.make_repository()
        graph_path = repository / "ai" / "tasks" / "task_graph.yaml"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        completed = {
            item["id"] for item in graph["tasks"] if item["status"] == "completed"
        }
        task = next(
            item
            for item in graph["tasks"]
            if item["status"] == "planned"
            and set(item["depends_on"]).issubset(completed)
        )
        task["status"] = "in_progress"
        task["owner"] = "Repository Test"
        task["branch"] = f"codex/{task['id']}-test"
        graph_path.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
        self.run_git(repository, "add", "ai/tasks/task_graph.yaml")

        baseline = self.run_git(
            repository,
            "show",
            "HEAD:ai/tasks/task_graph.yaml",
        ).stdout
        graph_path.write_text(baseline, encoding="utf-8")

        result = self.run_pre_commit(repository)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "staged ai/tasks/status.generated.md does not match the staged task graph",
            result.stderr,
        )

    def test_staged_generated_status_deletion_is_rejected(self) -> None:
        repository = self.make_repository()
        self.run_git(repository, "rm", "ai/tasks/status.generated.md")

        result = self.run_pre_commit(repository)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "required file is missing from staged snapshot: "
            "ai/tasks/status.generated.md",
            result.stderr,
        )

    def test_staged_referenced_worklog_deletion_is_rejected(self) -> None:
        repository = self.make_repository()
        graph = json.loads(
            (repository / "ai" / "tasks" / "task_graph.yaml").read_text(
                encoding="utf-8"
            )
        )
        task = next(item for item in graph["tasks"] if item["id"] == "T03")
        worklog = task["worklog"]
        self.assertIsInstance(worklog, str)
        self.run_git(repository, "rm", worklog)

        result = self.run_pre_commit(repository)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            f"T03: worklog does not exist: {worklog}",
            result.stderr,
        )

    def test_force_added_artifact_symlink_is_rejected(self) -> None:
        repository = self.make_repository()
        external = Path(self.addCleanupDirectory()) / "external-artifacts"
        external.mkdir()
        (repository / "artifacts").symlink_to(external, target_is_directory=True)
        self.run_git(repository, "add", "-f", "artifacts")

        result = self.run_hygiene(repository)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "private or external path cannot be committed: artifacts",
            result.stderr,
        )

    def test_force_added_claude_private_state_is_rejected(self) -> None:
        for relative in (
            "CLAUDE.local.md",
            ".claude/settings.local.json",
            ".claude/worktrees/private/file.txt",
        ):
            with self.subTest(relative=relative):
                repository = self.make_repository()
                path = repository / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("private\n", encoding="utf-8")
                self.run_git(repository, "add", "-f", relative)

                result = self.run_hygiene(repository)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    f"private or external path cannot be committed: {relative}"
                    if relative != "CLAUDE.local.md"
                    else "private filename cannot be committed: CLAUDE.local.md",
                    result.stderr,
                )

    def test_nested_agents_requires_claude_adapter(self) -> None:
        repository = self.make_repository()
        agents = repository / "src" / "example" / "AGENTS.md"
        agents.parent.mkdir(parents=True)
        agents.write_text("# Nested policy\n", encoding="utf-8")
        self.run_git(repository, "add", str(agents.relative_to(repository)))

        result = self.run_hygiene(repository)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing same-directory Claude adapter", result.stderr)

    def test_nested_agents_accepts_thin_claude_adapter(self) -> None:
        repository = self.make_repository()
        nested = repository / "src" / "example"
        nested.mkdir(parents=True)
        (nested / "AGENTS.md").write_text("# Nested policy\n", encoding="utf-8")
        (nested / "CLAUDE.md").write_text("@AGENTS.md\n", encoding="utf-8")
        self.run_git(repository, "add", "src/example")

        result = self.run_hygiene(repository)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_nested_agents_rejects_duplicated_claude_policy(self) -> None:
        repository = self.make_repository()
        nested = repository / "src" / "example"
        nested.mkdir(parents=True)
        (nested / "AGENTS.md").write_text("# Nested policy\n", encoding="utf-8")
        (nested / "CLAUDE.md").write_text(
            "@AGENTS.md\n\nDo all of the policy again.\n",
            encoding="utf-8",
        )
        self.run_git(repository, "add", "src/example")

        result = self.run_hygiene(repository)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be a thin adapter", result.stderr)

    def test_fresh_clone_installer_reconstructs_local_coordination_state(
        self,
    ) -> None:
        repository = self.make_repository()
        clone = Path(self.addCleanupDirectory()) / "fresh-clone"
        self.run_git(
            repository.parent,
            "clone",
            "-q",
            str(repository),
            str(clone),
        )
        missing_artifact_root = Path(self.addCleanupDirectory()) / "unmounted"
        result = subprocess.run(
            ("./scripts/setup/install_git_hooks.sh",),
            cwd=clone,
            env={
                **os.environ,
                "SLM_LAB_ARTIFACT_ROOT": str(missing_artifact_root),
            },
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        registry = clone / ".ai-local" / "tasks" / "thread-registry.yaml"
        self.assertTrue((clone / ".ai-local" / "README.md").is_file())
        self.assertEqual(
            json.loads(registry.read_text(encoding="utf-8")),
            json.loads(
                (clone / "ai" / "tasks" / "thread_registry.example.yaml").read_text(
                    encoding="utf-8"
                )
            ),
        )
        self.assertEqual(
            self.run_git(clone, "config", "--get", "core.hooksPath").stdout.strip(),
            ".githooks",
        )
        ignored = self.run_git(
            clone,
            "check-ignore",
            ".ai-local/tasks/thread-registry.yaml",
        )
        self.assertEqual(
            ignored.stdout.strip(),
            ".ai-local/tasks/thread-registry.yaml",
        )
        self.assertEqual(self.run_git(clone, "status", "--porcelain").stdout, "")

    def test_linked_worktree_bootstrap_uses_primary_registry(self) -> None:
        repository = self.make_repository()
        linked = Path(self.addCleanupDirectory()) / "linked"
        self.run_git(
            repository,
            "worktree",
            "add",
            "-q",
            "-b",
            "task/bootstrap-test",
            str(linked),
        )
        missing_artifact_root = Path(self.addCleanupDirectory()) / "unmounted"
        result = subprocess.run(
            (sys.executable, "scripts/setup/bootstrap_local_state.py", "--quiet"),
            cwd=linked,
            env={
                **os.environ,
                "SLM_LAB_ARTIFACT_ROOT": str(missing_artifact_root),
            },
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        shared = repository / ".ai-local" / "tasks" / "thread-registry.yaml"
        duplicate = linked / ".ai-local" / "tasks" / "thread-registry.yaml"
        self.assertTrue(shared.is_file())
        self.assertFalse(duplicate.exists())


if __name__ == "__main__":
    unittest.main()
