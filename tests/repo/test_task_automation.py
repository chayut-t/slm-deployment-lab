"""Regression tests for repository task and privacy automation."""

from __future__ import annotations

import copy
import importlib.util
import json
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
        task = next(item for item in graph["tasks"] if item["id"] == "T03")
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

    def test_complete_initial_staged_snapshot_passes(self) -> None:
        repository = self.copy_repository()
        self.run_git(repository, "init", "-q")
        self.run_git(repository, "add", ".")

        result = self.run_hygiene(repository)
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
        graph["tasks"][0]["status"] = "blocked"
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
        self.assertIn("staged task graph", result.stderr)

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

    def test_bootstrap_recreates_local_coordination_files(self) -> None:
        repository = self.make_repository()
        missing_artifact_root = Path(self.addCleanupDirectory()) / "unmounted"
        result = subprocess.run(
            (
                sys.executable,
                "scripts/setup/bootstrap_local_state.py",
                "--artifact-root",
                str(missing_artifact_root),
                "--quiet",
            ),
            cwd=repository,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((repository / ".ai-local" / "README.md").is_file())
        self.assertTrue(
            (
                repository
                / ".ai-local"
                / "tasks"
                / "thread-registry.yaml"
            ).is_file()
        )


if __name__ == "__main__":
    unittest.main()
