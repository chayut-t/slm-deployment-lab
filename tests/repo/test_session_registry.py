"""Tests for shared, private agent-session coordination."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SHA_A = "a" * 40
SHA_B = "b" * 40
SHA_C = "c" * 40


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REGISTRY = load_module(
    "session_registry_under_test",
    REPO_ROOT / "scripts" / "ai" / "session_registry.py",
)


class SessionRegistryTests(unittest.TestCase):
    def make_repository(self) -> tuple[Path, Path]:
        temporary = tempfile.TemporaryDirectory(prefix="slm-registry-test-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / "repo"
        linked = Path(temporary.name) / "linked"
        root.mkdir()
        self.run_git(root, "init", "-q")
        self.run_git(root, "config", "user.email", "test@example.invalid")
        self.run_git(root, "config", "user.name", "Registry Test")
        (root / "seed").write_text("seed\n", encoding="utf-8")
        self.run_git(root, "add", "seed")
        self.run_git(root, "commit", "-qm", "seed")
        self.run_git(root, "worktree", "add", "-q", "-b", "task/test", str(linked))
        return root, linked

    @staticmethod
    def run_git(repository: Path, *args: str) -> str:
        return subprocess.check_output(
            ("git", *args),
            cwd=repository,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

    @staticmethod
    def v1_text(*, state: str = "completed") -> str:
        completed = "    completed_at: 2026-07-25\n" if state == "completed" else ""
        return (
            "schema_version: 1\n"
            "tasks:\n"
            "  T03:\n"
            "    branch: codex/T03-agent-workflow\n"
            "    worktree: /private/tmp/slm-lab-T03\n"
            "    agent: /root/t03_writer\n"
            f"    state: {state}\n"
            "    started_at: 2026-07-24\n"
            f"{completed}"
        )

    @staticmethod
    def graph(
        branch: str = "task/T04-dual-agent-compatibility",
        *,
        status: str = "in_progress",
    ) -> dict:
        return {
            "tasks": [
                {
                    "id": "T04",
                    "status": status,
                    "owner": "Registry Test",
                    "branch": branch,
                }
            ]
        }

    @staticmethod
    def empty_v2() -> dict:
        return {"schema_version": 2, "tasks": {}}

    def test_worktrees_resolve_one_coordination_root(self) -> None:
        root, linked = self.make_repository()
        self.assertEqual(REGISTRY.resolve_coordination_root(root), root.resolve())
        self.assertEqual(REGISTRY.resolve_coordination_root(linked), root.resolve())

    def test_explicit_root_must_share_git_common_directory(self) -> None:
        root, linked = self.make_repository()
        other, _ = self.make_repository()
        with self.assertRaisesRegex(REGISTRY.RegistryError, "different Git"):
            REGISTRY.resolve_coordination_root(linked, explicit_root=other)
        self.assertEqual(
            REGISTRY.resolve_coordination_root(linked, explicit_root=root),
            root.resolve(),
        )
        with self.assertRaisesRegex(REGISTRY.RegistryError, "primary checkout"):
            REGISTRY.resolve_coordination_root(linked, explicit_root=linked)

    def test_schema_v1_is_read_without_mutation(self) -> None:
        text = self.v1_text()
        parsed = REGISTRY.load_registry_text(text)
        self.assertEqual(parsed["schema_version"], 1)
        self.assertEqual(parsed["tasks"]["T03"]["state"], "completed")
        self.assertEqual(text, self.v1_text())

    def test_schema_v1_inline_empty_tasks_is_read(self) -> None:
        text = "schema_version: 1\ntasks: {}\n"
        self.assertEqual(
            REGISTRY.load_registry_text(text),
            {"schema_version": 1, "tasks": {}},
        )

    def test_migration_refuses_active_registry_without_mutation(self) -> None:
        root, _ = self.make_repository()
        path = root / ".ai-local" / "tasks" / "thread-registry.yaml"
        path.parent.mkdir(parents=True)
        original = self.v1_text(state="in_progress")
        path.write_text(original, encoding="utf-8")

        with self.assertRaisesRegex(REGISTRY.RegistryError, "migration refused"):
            REGISTRY.migrate_registry(path)

        self.assertEqual(path.read_text(encoding="utf-8"), original)
        self.assertEqual(list(path.parent.glob("*.bak")), [])

    def test_completed_v1_migrates_with_backup_and_preserves_fields(self) -> None:
        root, _ = self.make_repository()
        path = root / ".ai-local" / "tasks" / "thread-registry.yaml"
        path.parent.mkdir(parents=True)
        original = self.v1_text()
        path.write_text(original, encoding="utf-8")

        backup = REGISTRY.migrate_registry(path)
        migrated = REGISTRY.load_registry(path)
        session = migrated["tasks"]["T03"]["sessions"]["/root/t03_writer"]

        self.assertEqual(backup.read_text(encoding="utf-8"), original)
        self.assertEqual(migrated["schema_version"], 2)
        self.assertIsNone(migrated["tasks"]["T03"]["active_writer"])
        self.assertEqual(session["legacy_fields"]["state"], "completed")
        self.assertEqual(
            session["legacy_fields"]["worktree"],
            "/private/tmp/slm-lab-T03",
        )

    def test_migration_refuses_missing_or_unknown_state_without_mutation(self) -> None:
        for state_line in ("", "    state: stale\n"):
            with self.subTest(state_line=state_line):
                root, _ = self.make_repository()
                path = root / ".ai-local" / "tasks" / "thread-registry.yaml"
                path.parent.mkdir(parents=True)
                original = (
                    "schema_version: 1\n"
                    "tasks:\n"
                    "  T03:\n"
                    "    branch: codex/T03-agent-workflow\n"
                    f"{state_line}"
                )
                path.write_text(original, encoding="utf-8")

                with self.assertRaisesRegex(REGISTRY.RegistryError, "ambiguous"):
                    REGISTRY.migrate_registry(path)

                self.assertEqual(path.read_text(encoding="utf-8"), original)
                self.assertEqual(list(path.parent.glob("*.bak")), [])

    def test_schema_v2_migration_is_idempotent(self) -> None:
        root, _ = self.make_repository()
        path = root / ".ai-local" / "tasks" / "thread-registry.yaml"
        path.parent.mkdir(parents=True)
        original = json.dumps(self.empty_v2(), indent=2) + "\n"
        path.write_text(original, encoding="utf-8")

        self.assertIsNone(REGISTRY.migrate_registry(path))

        self.assertEqual(path.read_text(encoding="utf-8"), original)
        self.assertEqual(list(path.parent.glob("*.bak")), [])

    def test_unknown_schema_is_rejected_without_mutation(self) -> None:
        root, _ = self.make_repository()
        path = root / ".ai-local" / "tasks" / "thread-registry.yaml"
        path.parent.mkdir(parents=True)
        original = '{"schema_version": 99, "tasks": {}}\n'
        path.write_text(original, encoding="utf-8")

        with self.assertRaisesRegex(REGISTRY.RegistryError, "unsupported"):
            REGISTRY.migrate_registry(path)

        self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_duplicate_writer_claim_is_rejected_without_mutation(self) -> None:
        original = self.empty_v2()
        claimed = REGISTRY.claim_writer(
            original,
            self.graph(),
            task_id="T04",
            session_id="writer-a",
            tool="codex",
            worktree="/tmp/a",
            checkpoint_sha=SHA_A,
            now="now",
        )
        snapshot = copy.deepcopy(claimed)

        with self.assertRaisesRegex(REGISTRY.RegistryError, "already exists"):
            REGISTRY.claim_writer(
                claimed,
                self.graph(),
                task_id="T04",
                session_id="writer-b",
                tool="claude-code",
                worktree="/tmp/b",
                checkpoint_sha=SHA_A,
                now="later",
            )

        self.assertEqual(claimed, snapshot)
        self.assertEqual(original, self.empty_v2())

    def test_branch_mismatch_is_rejected(self) -> None:
        data = self.empty_v2()
        data["tasks"]["T04"] = {
            "branch": "task/wrong",
            "checkpoint_sha": None,
            "active_writer": None,
            "sessions": {},
            "updated_at": None,
        }
        with self.assertRaisesRegex(REGISTRY.RegistryError, "disagrees"):
            REGISTRY.claim_writer(
                data,
                self.graph(),
                task_id="T04",
                session_id="writer-a",
                tool="codex",
                worktree="/tmp/a",
                checkpoint_sha=SHA_A,
                now="now",
            )

    def test_claim_requires_public_in_progress_task_and_exact_values(self) -> None:
        for graph, worktree, checkpoint, message in (
            (self.graph(status="planned"), "/tmp/a", SHA_A, "in_progress"),
            (self.graph(), "relative", SHA_A, "absolute"),
            (self.graph(), "/tmp/a", "not-a-sha", "40-character"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(REGISTRY.RegistryError, message):
                    REGISTRY.claim_writer(
                        self.empty_v2(),
                        graph,
                        task_id="T04",
                        session_id="writer-a",
                        tool="codex",
                        worktree=worktree,
                        checkpoint_sha=checkpoint,
                        now="now",
                    )

    def test_orphan_active_writer_session_is_invalid(self) -> None:
        data = self.empty_v2()
        data["tasks"]["T04"] = {
            "branch": "task/T04-dual-agent-compatibility",
            "checkpoint_sha": SHA_A,
            "active_writer": None,
            "sessions": {
                "writer-a": {
                    "tool": "codex",
                    "role": "writer",
                    "state": "active",
                    "worktree": "/tmp/a",
                }
            },
        }
        with self.assertRaisesRegex(REGISTRY.RegistryError, "active writer sessions"):
            REGISTRY.validate_registry(data)

    def test_stale_checkpoint_operations_are_rejected(self) -> None:
        claimed = REGISTRY.claim_writer(
            self.empty_v2(),
            self.graph(),
            task_id="T04",
            session_id="writer-a",
            tool="codex",
            worktree="/tmp/a",
            checkpoint_sha=SHA_A,
            now="now",
        )
        snapshot = copy.deepcopy(claimed)

        with self.assertRaisesRegex(REGISTRY.RegistryError, "stale checkpoint"):
            REGISTRY.update_checkpoint(
                claimed,
                self.graph(),
                task_id="T04",
                writer="writer-a",
                expected_checkpoint=SHA_B,
                new_checkpoint=SHA_C,
                now="later",
            )
        with self.assertRaisesRegex(REGISTRY.RegistryError, "stale checkpoint"):
            REGISTRY.release_writer(
                claimed,
                self.graph(),
                task_id="T04",
                expected_writer="writer-a",
                expected_checkpoint=SHA_B,
                now="later",
            )
        self.assertEqual(claimed, snapshot)

    def test_reviewer_cannot_be_promoted_implicitly(self) -> None:
        claimed = REGISTRY.claim_writer(
            self.empty_v2(),
            self.graph(),
            task_id="T04",
            session_id="writer-a",
            tool="codex",
            worktree="/tmp/a",
            checkpoint_sha=SHA_A,
            now="now",
        )
        reviewed = REGISTRY.add_reviewer(
            claimed,
            self.graph(),
            task_id="T04",
            session_id="reviewer-a",
            tool="claude-code",
            worktree="/tmp/review",
            now="now",
        )
        snapshot = copy.deepcopy(reviewed)

        with self.assertRaisesRegex(REGISTRY.RegistryError, "cannot become writer"):
            REGISTRY.transfer_writer(
                reviewed,
                self.graph(),
                task_id="T04",
                expected_writer="writer-a",
                expected_checkpoint=SHA_A,
                new_session_id="reviewer-a",
                new_tool="claude-code",
                new_worktree="/tmp/review",
                now="later",
            )

        self.assertEqual(reviewed, snapshot)

    def test_reviewer_release_is_cas_safe_and_preserves_writer(self) -> None:
        claimed = REGISTRY.claim_writer(
            self.empty_v2(),
            self.graph(),
            task_id="T04",
            session_id="writer-a",
            tool="codex",
            worktree="/tmp/a",
            checkpoint_sha=SHA_A,
            now="now",
        )
        reviewed = REGISTRY.add_reviewer(
            claimed,
            self.graph(),
            task_id="T04",
            session_id="reviewer-a",
            tool="claude-code",
            worktree="/tmp/a",
            now="now",
        )
        released = REGISTRY.release_reviewer(
            reviewed,
            self.graph(),
            task_id="T04",
            reviewer="reviewer-a",
            expected_state="active",
            now="later",
        )
        task = released["tasks"]["T04"]
        self.assertEqual(task["active_writer"], "writer-a")
        self.assertEqual(task["sessions"]["writer-a"]["state"], "active")
        self.assertEqual(task["sessions"]["reviewer-a"]["state"], "completed")
        with self.assertRaisesRegex(REGISTRY.RegistryError, "stale state"):
            REGISTRY.release_reviewer(
                released,
                self.graph(),
                task_id="T04",
                reviewer="reviewer-a",
                expected_state="active",
                now="latest",
            )

    def test_initialize_is_parallel_and_preserves_existing_registry(self) -> None:
        root, _ = self.make_repository()
        (root / "ai" / "tasks").mkdir(parents=True)
        template = root / "ai" / "tasks" / "registry.json"
        template.write_text(
            json.dumps(self.empty_v2(), indent=2) + "\n",
            encoding="utf-8",
        )
        helper = REPO_ROOT / "scripts" / "ai" / "session_registry.py"
        command = (
            sys.executable,
            str(helper),
            "--start",
            str(root),
            "initialize",
            "--template",
            str(template),
        )
        processes = [
            subprocess.Popen(
                command,
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for _ in range(4)
        ]
        results = [process.communicate(timeout=10) for process in processes]
        self.assertTrue(all(process.returncode == 0 for process in processes), results)
        path = root / ".ai-local" / "tasks" / "thread-registry.yaml"
        self.assertEqual(REGISTRY.load_registry(path), self.empty_v2())
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

        populated = {
            "schema_version": 2,
            "tasks": {
                "T04": {
                    "branch": "task/T04-dual-agent-compatibility",
                    "checkpoint_sha": None,
                    "active_writer": None,
                    "sessions": {},
                }
            },
        }
        path.write_text(json.dumps(populated), encoding="utf-8")
        self.assertFalse(
            REGISTRY.initialize_registry(path, template.read_text(encoding="utf-8"))
        )
        self.assertEqual(REGISTRY.load_registry(path), populated)

    def test_cli_claim_checkpoint_transfer_and_release_verify_git_state(
        self,
    ) -> None:
        root, linked = self.make_repository()
        graph_path = root / "ai" / "tasks" / "task_graph.yaml"
        graph_path.parent.mkdir(parents=True)
        graph_path.write_text(
            json.dumps(self.graph(branch="task/test")),
            encoding="utf-8",
        )
        self.run_git(root, "add", "ai/tasks/task_graph.yaml")
        self.run_git(root, "commit", "-qm", "claim task")
        primary_branch = self.run_git(root, "branch", "--show-current")
        self.run_git(linked, "merge", "-q", primary_branch)
        registry = root / ".ai-local" / "tasks" / "thread-registry.yaml"
        registry.parent.mkdir(parents=True)
        registry.write_text(json.dumps(self.empty_v2()), encoding="utf-8")
        helper = REPO_ROOT / "scripts" / "ai" / "session_registry.py"
        first_checkpoint = self.run_git(linked, "rev-parse", "HEAD")

        def run_helper(*arguments: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                (
                    sys.executable,
                    str(helper),
                    "--start",
                    str(linked),
                    *arguments,
                ),
                cwd=linked,
                text=True,
                capture_output=True,
                check=False,
            )

        committed_graph = graph_path.read_text(encoding="utf-8")
        graph_path.write_text(
            json.dumps(self.graph(branch="task/test", status="planned")),
            encoding="utf-8",
        )
        uncommitted_graph = run_helper(
            "claim",
            "T04",
            "--session",
            "writer-a",
            "--tool",
            "codex",
            "--worktree",
            str(linked),
            "--checkpoint",
            first_checkpoint,
        )
        self.assertEqual(uncommitted_graph.returncode, 2)
        self.assertIn("uncommitted changes", uncommitted_graph.stderr)
        self.assertEqual(REGISTRY.load_registry(registry), self.empty_v2())
        graph_path.write_text(committed_graph, encoding="utf-8")

        invalid = run_helper(
            "claim",
            "T04",
            "--session",
            "writer-a",
            "--tool",
            "codex",
            "--worktree",
            str(linked),
            "--checkpoint",
            SHA_A,
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("is at", invalid.stderr)
        self.assertEqual(REGISTRY.load_registry(registry), self.empty_v2())

        dirty_path = linked / "dirty"
        dirty_path.write_text("dirty\n", encoding="utf-8")
        dirty = run_helper(
            "claim",
            "T04",
            "--session",
            "writer-a",
            "--tool",
            "codex",
            "--worktree",
            str(linked),
            "--checkpoint",
            first_checkpoint,
        )
        self.assertEqual(dirty.returncode, 2)
        self.assertIn("uncommitted changes", dirty.stderr)
        self.assertEqual(REGISTRY.load_registry(registry), self.empty_v2())
        dirty_path.unlink()

        independent = root.parent / "independent"
        subprocess.check_call(
            (
                "git",
                "clone",
                "-q",
                "--branch",
                "task/test",
                str(root),
                str(independent),
            )
        )
        unrelated = run_helper(
            "claim",
            "T04",
            "--session",
            "writer-a",
            "--tool",
            "codex",
            "--worktree",
            str(independent),
            "--checkpoint",
            first_checkpoint,
        )
        self.assertEqual(unrelated.returncode, 2)
        self.assertIn("not linked", unrelated.stderr)
        self.assertEqual(REGISTRY.load_registry(registry), self.empty_v2())

        claimed = run_helper(
            "claim",
            "T04",
            "--session",
            "writer-a",
            "--tool",
            "codex",
            "--worktree",
            str(linked),
            "--checkpoint",
            first_checkpoint,
        )
        self.assertEqual(claimed.returncode, 0, claimed.stderr)

        (linked / "next").write_text("next\n", encoding="utf-8")
        self.run_git(linked, "add", "next")
        self.run_git(linked, "commit", "-qm", "next")
        second_checkpoint = self.run_git(linked, "rev-parse", "HEAD")
        checkpointed = run_helper(
            "checkpoint",
            "T04",
            "--writer",
            "writer-a",
            "--expected-checkpoint",
            first_checkpoint,
            "--new-checkpoint",
            second_checkpoint,
        )
        self.assertEqual(checkpointed.returncode, 0, checkpointed.stderr)

        self.run_git(linked, "switch", "--detach")
        incoming = root.parent / "incoming"
        self.run_git(root, "worktree", "add", "-q", str(incoming), "task/test")
        transferred = run_helper(
            "transfer",
            "T04",
            "--expected-writer",
            "writer-a",
            "--expected-checkpoint",
            second_checkpoint,
            "--new-session",
            "writer-b",
            "--new-tool",
            "claude-code",
            "--new-worktree",
            str(incoming),
        )
        self.assertEqual(transferred.returncode, 0, transferred.stderr)
        released = subprocess.run(
            (
                sys.executable,
                str(helper),
                "--start",
                str(incoming),
                "release",
                "T04",
                "--expected-writer",
                "writer-b",
                "--expected-checkpoint",
                second_checkpoint,
            ),
            cwd=incoming,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(released.returncode, 0, released.stderr)
        task = REGISTRY.load_registry(registry)["tasks"]["T04"]
        self.assertIsNone(task["active_writer"])
        self.assertEqual(task["sessions"]["writer-a"]["state"], "transferred")
        self.assertEqual(task["sessions"]["writer-b"]["state"], "completed")


if __name__ == "__main__":
    unittest.main()
