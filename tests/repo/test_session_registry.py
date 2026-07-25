"""Tests for shared, private agent-session coordination."""

from __future__ import annotations

import copy
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


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
    def graph(branch: str = "task/T04-dual-agent-compatibility") -> dict:
        return {"tasks": [{"id": "T04", "branch": branch}]}

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

    def test_schema_v1_is_read_without_mutation(self) -> None:
        text = self.v1_text()
        parsed = REGISTRY.load_registry_text(text)
        self.assertEqual(parsed["schema_version"], 1)
        self.assertEqual(parsed["tasks"]["T03"]["state"], "completed")
        self.assertEqual(text, self.v1_text())

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
            checkpoint_sha="claim",
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
                checkpoint_sha="claim",
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
                checkpoint_sha="claim",
                now="now",
            )

    def test_stale_checkpoint_operations_are_rejected(self) -> None:
        claimed = REGISTRY.claim_writer(
            self.empty_v2(),
            self.graph(),
            task_id="T04",
            session_id="writer-a",
            tool="codex",
            worktree="/tmp/a",
            checkpoint_sha="claim",
            now="now",
        )
        snapshot = copy.deepcopy(claimed)

        with self.assertRaisesRegex(REGISTRY.RegistryError, "stale checkpoint"):
            REGISTRY.update_checkpoint(
                claimed,
                self.graph(),
                task_id="T04",
                writer="writer-a",
                expected_checkpoint="old",
                new_checkpoint="new",
                now="later",
            )
        with self.assertRaisesRegex(REGISTRY.RegistryError, "stale checkpoint"):
            REGISTRY.release_writer(
                claimed,
                self.graph(),
                task_id="T04",
                expected_writer="writer-a",
                expected_checkpoint="old",
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
            checkpoint_sha="claim",
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
                expected_checkpoint="claim",
                new_session_id="reviewer-a",
                new_tool="claude-code",
                new_worktree="/tmp/review",
                now="later",
            )

        self.assertEqual(reviewed, snapshot)


if __name__ == "__main__":
    unittest.main()
