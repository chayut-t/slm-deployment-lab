#!/usr/bin/env python3
"""Coordinate private agent sessions safely across Git worktrees."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator


REGISTRY_RELATIVE_PATH = Path(".ai-local/tasks/thread-registry.yaml")
GRAPH_RELATIVE_PATH = Path("ai/tasks/task_graph.yaml")
ACTIVE_STATES = {"active", "in_progress"}
V2_SESSION_STATES = {"active", "completed", "transferred"}
FULL_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class RegistryError(ValueError):
    """A safe registry operation could not be completed."""


def _git_output(cwd: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ("git", *args),
            cwd=cwd,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RegistryError(
            f"git {' '.join(args)} failed in {cwd}: {exc.output.strip()}"
        ) from exc


def _git_common_dir(cwd: Path) -> Path:
    raw = Path(_git_output(cwd, "rev-parse", "--git-common-dir"))
    return (raw if raw.is_absolute() else cwd / raw).resolve()


def resolve_coordination_root(
    start: Path | str,
    *,
    explicit_root: Path | str | None = None,
) -> Path:
    """Resolve one primary checkout shared by every linked worktree."""

    start_path = Path(start).resolve()
    listing = _git_output(start_path, "worktree", "list", "--porcelain")
    first = next(
        (
            line.removeprefix("worktree ")
            for line in listing.splitlines()
            if line.startswith("worktree ")
        ),
        None,
    )
    if not first:
        raise RegistryError("git worktree list returned no primary checkout")
    primary = Path(first).resolve()

    override = explicit_root or os.environ.get("SLM_LAB_COORDINATION_ROOT")
    root = Path(override).expanduser().resolve() if override else primary

    if not root.is_dir():
        raise RegistryError(f"coordination root does not exist: {root}")
    if _git_common_dir(root) != _git_common_dir(start_path):
        raise RegistryError(
            "coordination root belongs to a different Git repository; "
            "set SLM_LAB_COORDINATION_ROOT to this repository's primary checkout"
        )
    if root != primary:
        raise RegistryError(
            f"coordination root must be the primary checkout {primary}, not {root}"
        )
    return root


def registry_path(coordination_root: Path | str) -> Path:
    return Path(coordination_root) / REGISTRY_RELATIVE_PATH


def _parse_scalar(value: str) -> object:
    value = value.strip()
    if value in {"null", "~"}:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if value.startswith(('"', "'")):
        try:
            return json.loads(value) if value.startswith('"') else value[1:-1]
        except (json.JSONDecodeError, IndexError) as exc:
            raise RegistryError(f"invalid quoted registry value: {value}") from exc
    try:
        return int(value)
    except ValueError:
        return value


def parse_v1_registry(text: str) -> dict:
    """Parse the deliberately small schema-v1 YAML subset without PyYAML."""

    data: dict[str, object] = {}
    tasks: dict[str, dict[str, object]] = {}
    current_task: str | None = None
    saw_tasks = False

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if indent == 0:
            current_task = None
            if stripped == "tasks:":
                data["tasks"] = tasks
                saw_tasks = True
                continue
            if stripped == "tasks: {}":
                data["tasks"] = tasks
                saw_tasks = True
                continue
            if ":" not in stripped:
                raise RegistryError(f"invalid schema-v1 line {line_number}: {raw_line}")
            key, value = stripped.split(":", 1)
            data[key] = _parse_scalar(value)
        elif indent == 2 and stripped.endswith(":") and saw_tasks:
            current_task = stripped[:-1]
            if not current_task or current_task in tasks:
                raise RegistryError(
                    f"invalid or duplicate task at line {line_number}: {raw_line}"
                )
            tasks[current_task] = {}
        elif indent == 4 and current_task and ":" in stripped:
            key, value = stripped.split(":", 1)
            tasks[current_task][key] = _parse_scalar(value)
        else:
            raise RegistryError(
                f"unsupported schema-v1 structure at line {line_number}: {raw_line}"
            )

    validate_registry(data)
    return data


def load_registry_text(text: str) -> dict:
    stripped = text.lstrip()
    if not stripped:
        raise RegistryError("registry is empty")
    if stripped.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RegistryError(f"registry JSON is invalid: {exc}") from exc
    else:
        data = parse_v1_registry(text)
    validate_registry(data)
    return data


def load_registry(path: Path | str) -> dict:
    registry = Path(path)
    try:
        return load_registry_text(registry.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RegistryError(f"cannot read registry {registry}: {exc}") from exc


def validate_registry(data: object) -> None:
    if not isinstance(data, dict):
        raise RegistryError("registry must be a mapping")
    schema = data.get("schema_version")
    if schema not in {1, 2}:
        raise RegistryError(
            f"unsupported registry schema_version {schema!r}; expected 1 or 2"
        )
    tasks = data.get("tasks")
    if not isinstance(tasks, dict):
        raise RegistryError("registry tasks must be a mapping")
    for task_id, task in tasks.items():
        if not isinstance(task_id, str) or not task_id.startswith("T"):
            raise RegistryError(f"invalid registry task ID: {task_id!r}")
        if not isinstance(task, dict):
            raise RegistryError(f"{task_id}: registry entry must be a mapping")
        if schema == 2:
            branch = task.get("branch")
            if not isinstance(branch, str) or not branch:
                raise RegistryError(f"{task_id}: branch must be a nonempty string")
            checkpoint = task.get("checkpoint_sha")
            if checkpoint is not None and (
                not isinstance(checkpoint, str)
                or not FULL_SHA_PATTERN.fullmatch(checkpoint)
            ):
                raise RegistryError(
                    f"{task_id}: checkpoint_sha must be null or a full Git SHA"
                )
            sessions = task.get("sessions", {})
            if not isinstance(sessions, dict):
                raise RegistryError(f"{task_id}: sessions must be a mapping")
            active_writer = task.get("active_writer")
            active_writer_sessions: list[str] = []
            for session_id, session in sessions.items():
                if not isinstance(session_id, str) or not isinstance(session, dict):
                    raise RegistryError(f"{task_id}: invalid session entry")
                if session.get("role") not in {"writer", "reviewer"}:
                    raise RegistryError(
                        f"{task_id}/{session_id}: role must be writer or reviewer"
                    )
                state = session.get("state")
                if state not in V2_SESSION_STATES:
                    raise RegistryError(
                        f"{task_id}/{session_id}: invalid session state {state!r}"
                    )
                if session.get("role") == "reviewer" and state == "transferred":
                    raise RegistryError(
                        f"{task_id}/{session_id}: reviewer cannot be transferred"
                    )
                worktree = session.get("worktree")
                if state == "active" and (
                    not isinstance(worktree, str) or not Path(worktree).is_absolute()
                ):
                    raise RegistryError(
                        f"{task_id}/{session_id}: active worktree must be "
                        "an absolute path"
                    )
                if worktree is not None and (
                    not isinstance(worktree, str) or not Path(worktree).is_absolute()
                ):
                    raise RegistryError(
                        f"{task_id}/{session_id}: worktree must be null or "
                        "an absolute path"
                    )
                if session.get("role") == "writer" and state == "active":
                    active_writer_sessions.append(session_id)
            if active_writer is None:
                if active_writer_sessions:
                    raise RegistryError(
                        f"{task_id}: active writer sessions exist while "
                        "active_writer is null"
                    )
            elif (
                active_writer not in sessions
                or sessions[active_writer].get("role") != "writer"
                or sessions[active_writer].get("state") != "active"
            ):
                raise RegistryError(
                    f"{task_id}: active_writer {active_writer!r} must identify "
                    "an active writer session"
                )
            if active_writer_sessions != ([active_writer] if active_writer else []):
                raise RegistryError(
                    f"{task_id}: active_writer must be the only active writer session"
                )


def has_active_or_ambiguous_sessions(data: dict) -> list[str]:
    active: list[str] = []
    schema = data["schema_version"]
    for task_id, task in data["tasks"].items():
        if schema == 1:
            if task.get("state") != "completed":
                active.append(task_id)
        else:
            if task.get("active_writer"):
                active.append(task_id)
            for session_id, session in task.get("sessions", {}).items():
                if session.get("state") in ACTIVE_STATES:
                    active.append(f"{task_id}/{session_id}")
    return sorted(set(active))


def migrate_v1_data(data: dict) -> dict:
    validate_registry(data)
    if data["schema_version"] != 1:
        raise RegistryError("migration requires a schema-v1 registry")
    active = has_active_or_ambiguous_sessions(data)
    if active:
        raise RegistryError(
            "migration refused while sessions are active or ambiguous: "
            + ", ".join(active)
            + "; reconcile them with their owners before retrying"
        )

    migrated: dict[str, object] = {"schema_version": 2, "tasks": {}}
    migrated_tasks: dict[str, dict] = migrated["tasks"]  # type: ignore[assignment]
    for task_id, legacy in data["tasks"].items():
        session_id = str(
            legacy.get("thread_id")
            or legacy.get("agent")
            or f"legacy-{task_id.lower()}"
        )
        completed_at = legacy.get("completed_at")
        started_at = legacy.get("started_at")
        session = {
            "tool": "codex"
            if legacy.get("thread_id") or legacy.get("agent")
            else "unknown",
            "role": "writer",
            "state": legacy["state"],
            "worktree": legacy.get("worktree"),
            "started_at": started_at,
            "updated_at": completed_at or started_at,
            "legacy_fields": copy.deepcopy(legacy),
        }
        migrated_tasks[task_id] = {
            "branch": legacy.get("branch"),
            "checkpoint_sha": legacy.get("checkpoint_sha"),
            "active_writer": None,
            "sessions": {session_id: session},
            "updated_at": completed_at or started_at,
        }
    validate_registry(migrated)
    return migrated


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def locked_registry(path: Path | str) -> Iterator[Path]:
    registry = Path(path)
    registry.parent.mkdir(parents=True, exist_ok=True)
    lock_path = registry.with_name(f"{registry.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield registry
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def migrate_registry(path: Path | str) -> Path | None:
    registry = Path(path)
    with locked_registry(registry):
        source = load_registry(registry)
        if source["schema_version"] == 2:
            return None
        migrated = migrate_v1_data(source)
        backup = registry.with_name(f"{registry.name}.v1.{_timestamp()}.bak")
        try:
            shutil.copy2(registry, backup)
            os.chmod(backup, 0o600)
            _write_atomic(registry, migrated)
        except OSError as exc:
            raise RegistryError(f"registry migration failed: {exc}") from exc
        return backup


def initialize_registry(path: Path | str, template_text: str) -> bool:
    """Create a new registry atomically without replacing an existing one."""

    registry = Path(path)
    template = load_registry_text(template_text)
    if template["schema_version"] != 2:
        raise RegistryError("new registry template must use schema v2")
    with locked_registry(registry):
        if registry.exists():
            load_registry(registry)
            return False
        _write_atomic(registry, template)
        return True


def load_task_graph(coordination_root: Path | str) -> dict:
    path = Path(coordination_root) / GRAPH_RELATIVE_PATH
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"cannot read task graph {path}: {exc}") from exc
    if not isinstance(graph.get("tasks"), list):
        raise RegistryError("task graph has no tasks list")
    return graph


def _graph_task(graph: dict, task_id: str) -> dict:
    task = next((item for item in graph["tasks"] if item.get("id") == task_id), None)
    if task is None:
        raise RegistryError(f"{task_id}: task is absent from the public graph")
    if task.get("status") != "in_progress":
        raise RegistryError(
            f"{task_id}: public graph status must be in_progress, "
            f"not {task.get('status')!r}"
        )
    owner = task.get("owner")
    if not isinstance(owner, str) or not owner:
        raise RegistryError(f"{task_id}: public graph has no active owner")
    branch = task.get("branch")
    if not isinstance(branch, str) or not branch:
        raise RegistryError(f"{task_id}: public graph has no claimed branch")
    return task


def _require_full_sha(value: str, field: str) -> None:
    if not FULL_SHA_PATTERN.fullmatch(value):
        raise RegistryError(f"{field} must be a full 40-character Git SHA")


def _require_absolute_worktree(value: str, field: str = "worktree") -> None:
    if not Path(value).is_absolute():
        raise RegistryError(f"{field} must be an absolute path")


def _v2_copy(data: dict) -> dict:
    validate_registry(data)
    if data["schema_version"] != 2:
        raise RegistryError(
            "writer operations require schema v2; validate schema v1 read-only "
            "or migrate it explicitly after every active session is reconciled"
        )
    return copy.deepcopy(data)


def _task_for_branch(data: dict, graph: dict, task_id: str) -> dict:
    branch = _graph_task(graph, task_id)["branch"]
    task = data["tasks"].setdefault(
        task_id,
        {
            "branch": branch,
            "checkpoint_sha": None,
            "active_writer": None,
            "sessions": {},
            "updated_at": None,
        },
    )
    if task.get("branch") != branch:
        raise RegistryError(
            f"{task_id}: registry branch {task.get('branch')!r} disagrees with "
            f"public graph branch {branch!r}; reconcile the graph before retrying"
        )
    return task


def claim_writer(
    data: dict,
    graph: dict,
    *,
    task_id: str,
    session_id: str,
    tool: str,
    worktree: str,
    checkpoint_sha: str,
    now: str,
) -> dict:
    _require_absolute_worktree(worktree)
    _require_full_sha(checkpoint_sha, "checkpoint")
    updated = _v2_copy(data)
    task = _task_for_branch(updated, graph, task_id)
    current = task.get("active_writer")
    if current:
        raise RegistryError(
            f"{task_id}: active writer {current!r} already exists; "
            "release or transfer it explicitly"
        )
    active_writer_sessions = [
        identifier
        for identifier, session in task["sessions"].items()
        if session.get("role") == "writer" and session.get("state") == "active"
    ]
    if active_writer_sessions:
        raise RegistryError(
            f"{task_id}: active writer sessions already exist: "
            + ", ".join(active_writer_sessions)
        )
    if session_id in task["sessions"]:
        raise RegistryError(f"{task_id}: session {session_id!r} already exists")
    task["active_writer"] = session_id
    task["checkpoint_sha"] = checkpoint_sha
    task["sessions"][session_id] = {
        "tool": tool,
        "role": "writer",
        "state": "active",
        "worktree": worktree,
        "started_at": now,
        "updated_at": now,
    }
    task["updated_at"] = now
    validate_registry(updated)
    return updated


def add_reviewer(
    data: dict,
    graph: dict,
    *,
    task_id: str,
    session_id: str,
    tool: str,
    worktree: str,
    now: str,
) -> dict:
    _require_absolute_worktree(worktree)
    updated = _v2_copy(data)
    task = _task_for_branch(updated, graph, task_id)
    if session_id in task["sessions"]:
        raise RegistryError(f"{task_id}: session {session_id!r} already exists")
    task["sessions"][session_id] = {
        "tool": tool,
        "role": "reviewer",
        "state": "active",
        "worktree": worktree,
        "started_at": now,
        "updated_at": now,
    }
    task["updated_at"] = now
    validate_registry(updated)
    return updated


def update_checkpoint(
    data: dict,
    graph: dict,
    *,
    task_id: str,
    writer: str,
    expected_checkpoint: str,
    new_checkpoint: str,
    now: str,
) -> dict:
    _require_full_sha(expected_checkpoint, "expected checkpoint")
    _require_full_sha(new_checkpoint, "new checkpoint")
    updated = _v2_copy(data)
    task = _task_for_branch(updated, graph, task_id)
    if task.get("active_writer") != writer:
        raise RegistryError(
            f"{task_id}: writer compare-and-swap failed; current writer is "
            f"{task.get('active_writer')!r}"
        )
    if task.get("checkpoint_sha") != expected_checkpoint:
        raise RegistryError(
            f"{task_id}: stale checkpoint {expected_checkpoint!r}; current "
            f"checkpoint is {task.get('checkpoint_sha')!r}"
        )
    task["checkpoint_sha"] = new_checkpoint
    task["sessions"][writer]["updated_at"] = now
    task["updated_at"] = now
    validate_registry(updated)
    return updated


def release_writer(
    data: dict,
    graph: dict,
    *,
    task_id: str,
    expected_writer: str,
    expected_checkpoint: str,
    now: str,
) -> dict:
    _require_full_sha(expected_checkpoint, "expected checkpoint")
    updated = _v2_copy(data)
    task = _task_for_branch(updated, graph, task_id)
    if task.get("active_writer") != expected_writer:
        raise RegistryError(
            f"{task_id}: release refused; current writer is "
            f"{task.get('active_writer')!r}, not {expected_writer!r}"
        )
    if task.get("checkpoint_sha") != expected_checkpoint:
        raise RegistryError(
            f"{task_id}: release refused for stale checkpoint "
            f"{expected_checkpoint!r}; current checkpoint is "
            f"{task.get('checkpoint_sha')!r}"
        )
    task["sessions"][expected_writer]["state"] = "completed"
    task["sessions"][expected_writer]["updated_at"] = now
    task["active_writer"] = None
    task["updated_at"] = now
    validate_registry(updated)
    return updated


def release_reviewer(
    data: dict,
    graph: dict,
    *,
    task_id: str,
    reviewer: str,
    expected_state: str,
    now: str,
) -> dict:
    updated = _v2_copy(data)
    task = _task_for_branch(updated, graph, task_id)
    session = task["sessions"].get(reviewer)
    if not session or session.get("role") != "reviewer":
        raise RegistryError(
            f"{task_id}: reviewer release refused; {reviewer!r} is not a reviewer"
        )
    if session.get("state") != expected_state:
        raise RegistryError(
            f"{task_id}: reviewer release refused for stale state "
            f"{expected_state!r}; current state is {session.get('state')!r}"
        )
    if expected_state != "active":
        raise RegistryError("reviewer release requires expected state active")
    session["state"] = "completed"
    session["updated_at"] = now
    task["updated_at"] = now
    validate_registry(updated)
    return updated


def transfer_writer(
    data: dict,
    graph: dict,
    *,
    task_id: str,
    expected_writer: str,
    expected_checkpoint: str,
    new_session_id: str,
    new_tool: str,
    new_worktree: str,
    now: str,
) -> dict:
    _require_full_sha(expected_checkpoint, "expected checkpoint")
    _require_absolute_worktree(new_worktree, "new worktree")
    updated = _v2_copy(data)
    task = _task_for_branch(updated, graph, task_id)
    if task.get("active_writer") != expected_writer:
        raise RegistryError(
            f"{task_id}: transfer refused; current writer is "
            f"{task.get('active_writer')!r}"
        )
    if task.get("checkpoint_sha") != expected_checkpoint:
        raise RegistryError(
            f"{task_id}: transfer refused for stale checkpoint; current "
            f"checkpoint is {task.get('checkpoint_sha')!r}"
        )
    if new_session_id in task["sessions"]:
        session = task["sessions"][new_session_id]
        if session.get("role") == "reviewer":
            raise RegistryError(
                f"{task_id}: reviewer {new_session_id!r} cannot become writer "
                "implicitly; use a new writer session ID"
            )
        raise RegistryError(f"{task_id}: session {new_session_id!r} already exists")
    task["sessions"][expected_writer]["state"] = "transferred"
    task["sessions"][expected_writer]["updated_at"] = now
    task["sessions"][new_session_id] = {
        "tool": new_tool,
        "role": "writer",
        "state": "active",
        "worktree": new_worktree,
        "started_at": now,
        "updated_at": now,
    }
    task["active_writer"] = new_session_id
    task["updated_at"] = now
    validate_registry(updated)
    return updated


def validate_task_worktree(
    worktree: str,
    *,
    branch: str,
    checkpoint_sha: str,
) -> None:
    """Verify that an exact task checkpoint is checked out on its public branch."""

    _require_absolute_worktree(worktree)
    _require_full_sha(checkpoint_sha, "checkpoint")
    path = Path(worktree).resolve()
    if not path.is_dir():
        raise RegistryError(f"worktree does not exist: {path}")
    top_level = Path(_git_output(path, "rev-parse", "--show-toplevel")).resolve()
    if top_level != path:
        raise RegistryError(f"worktree path must be its Git root: {path}")
    actual_branch = _git_output(path, "branch", "--show-current")
    if actual_branch != branch:
        raise RegistryError(
            f"worktree {path} is on branch {actual_branch!r}, not {branch!r}"
        )
    actual_checkpoint = _git_output(path, "rev-parse", "HEAD")
    if actual_checkpoint != checkpoint_sha:
        raise RegistryError(
            f"worktree {path} is at {actual_checkpoint}, not {checkpoint_sha}"
        )
    if _git_output(path, "status", "--porcelain"):
        raise RegistryError(
            f"worktree {path} has uncommitted changes; commit or reconcile them "
            "before changing registry state"
        )


def validate_writer_worktree(
    data: dict,
    graph: dict,
    *,
    task_id: str,
    writer: str,
    checkpoint_sha: str,
) -> None:
    task = data["tasks"].get(task_id)
    if not isinstance(task, dict) or task.get("active_writer") != writer:
        raise RegistryError(f"{task_id}: {writer!r} is not the active writer")
    session = task.get("sessions", {}).get(writer)
    if not isinstance(session, dict):
        raise RegistryError(f"{task_id}: active writer session is missing")
    graph_task = _graph_task(graph, task_id)
    validate_task_worktree(
        str(session.get("worktree")),
        branch=graph_task["branch"],
        checkpoint_sha=checkpoint_sha,
    )


def mutate_registry(
    path: Path | str,
    operation: Callable[[dict], dict],
) -> None:
    registry = Path(path)
    with locked_registry(registry):
        original = load_registry(registry)
        updated = operation(original)
        _write_atomic(registry, updated)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=".", help="Path inside the Git repository")
    parser.add_argument("--coordination-root", help="Explicit primary checkout")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("path", help="Print the shared registry path")
    subparsers.add_parser("show", help="Print the validated registry as JSON")
    subparsers.add_parser("validate", help="Validate the registry")
    subparsers.add_parser("migrate", help="Explicitly migrate schema v1 to v2")
    initialize = subparsers.add_parser(
        "initialize",
        help="Atomically create a missing schema-v2 registry",
    )
    initialize.add_argument("--template", required=True)

    claim = subparsers.add_parser("claim", help="Claim one active writer")
    claim.add_argument("task_id")
    claim.add_argument("--session", required=True)
    claim.add_argument("--tool", required=True)
    claim.add_argument("--worktree", required=True)
    claim.add_argument("--checkpoint", required=True)

    reviewer = subparsers.add_parser("add-reviewer", help="Add a read-only reviewer")
    reviewer.add_argument("task_id")
    reviewer.add_argument("--session", required=True)
    reviewer.add_argument("--tool", required=True)
    reviewer.add_argument("--worktree", required=True)

    checkpoint = subparsers.add_parser("checkpoint", help="Advance writer checkpoint")
    checkpoint.add_argument("task_id")
    checkpoint.add_argument("--writer", required=True)
    checkpoint.add_argument("--expected-checkpoint", required=True)
    checkpoint.add_argument("--new-checkpoint", required=True)

    release = subparsers.add_parser("release", help="Release the active writer")
    release.add_argument("task_id")
    release.add_argument("--expected-writer", required=True)
    release.add_argument("--expected-checkpoint", required=True)

    release_reviewer_parser = subparsers.add_parser(
        "release-reviewer",
        help="Complete an active reviewer session",
    )
    release_reviewer_parser.add_argument("task_id")
    release_reviewer_parser.add_argument("--reviewer", required=True)
    release_reviewer_parser.add_argument("--expected-state", default="active")

    transfer = subparsers.add_parser("transfer", help="Transfer writer ownership")
    transfer.add_argument("task_id")
    transfer.add_argument("--expected-writer", required=True)
    transfer.add_argument("--expected-checkpoint", required=True)
    transfer.add_argument("--new-session", required=True)
    transfer.add_argument("--new-tool", required=True)
    transfer.add_argument("--new-worktree", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        root = resolve_coordination_root(
            args.start,
            explicit_root=args.coordination_root,
        )
        path = registry_path(root)
        if args.command == "path":
            print(path)
            return 0
        if args.command == "show":
            print(json.dumps(load_registry(path), indent=2, sort_keys=True))
            return 0
        if args.command == "validate":
            data = load_registry(path)
            print(f"registry schema v{data['schema_version']} is valid: {path}")
            return 0
        if args.command == "migrate":
            backup = migrate_registry(path)
            if backup is None:
                print(f"registry already uses schema v2; no change: {path}")
            else:
                print(f"registry migrated to schema v2; backup: {backup}")
            return 0
        if args.command == "initialize":
            try:
                template = Path(args.template).read_text(encoding="utf-8")
            except OSError as exc:
                raise RegistryError(
                    f"cannot read registry template {args.template}: {exc}"
                ) from exc
            created = initialize_registry(path, template)
            action = "created" if created else "preserved existing"
            print(f"{action} registry: {path}")
            return 0

        graph = load_task_graph(root)
        now = _utc_now()
        if args.command == "claim":
            graph_task = _graph_task(graph, args.task_id)

            def claim_operation(data: dict) -> dict:
                validate_task_worktree(
                    args.worktree,
                    branch=graph_task["branch"],
                    checkpoint_sha=args.checkpoint,
                )
                return claim_writer(
                    data,
                    graph,
                    task_id=args.task_id,
                    session_id=args.session,
                    tool=args.tool,
                    worktree=args.worktree,
                    checkpoint_sha=args.checkpoint,
                    now=now,
                )

            mutate_registry(
                path,
                claim_operation,
            )
        elif args.command == "add-reviewer":
            graph_task = _graph_task(graph, args.task_id)

            def reviewer_operation(data: dict) -> dict:
                task = data.get("tasks", {}).get(args.task_id, {})
                checkpoint = task.get("checkpoint_sha")
                if not isinstance(checkpoint, str):
                    raise RegistryError(
                        f"{args.task_id}: no writer checkpoint exists for review"
                    )
                validate_task_worktree(
                    args.worktree,
                    branch=graph_task["branch"],
                    checkpoint_sha=checkpoint,
                )
                return add_reviewer(
                    data,
                    graph,
                    task_id=args.task_id,
                    session_id=args.session,
                    tool=args.tool,
                    worktree=args.worktree,
                    now=now,
                )

            mutate_registry(
                path,
                reviewer_operation,
            )
        elif args.command == "checkpoint":

            def checkpoint_operation(data: dict) -> dict:
                validate_writer_worktree(
                    data,
                    graph,
                    task_id=args.task_id,
                    writer=args.writer,
                    checkpoint_sha=args.new_checkpoint,
                )
                return update_checkpoint(
                    data,
                    graph,
                    task_id=args.task_id,
                    writer=args.writer,
                    expected_checkpoint=args.expected_checkpoint,
                    new_checkpoint=args.new_checkpoint,
                    now=now,
                )

            mutate_registry(
                path,
                checkpoint_operation,
            )
        elif args.command == "release":

            def release_operation(data: dict) -> dict:
                validate_writer_worktree(
                    data,
                    graph,
                    task_id=args.task_id,
                    writer=args.expected_writer,
                    checkpoint_sha=args.expected_checkpoint,
                )
                return release_writer(
                    data,
                    graph,
                    task_id=args.task_id,
                    expected_writer=args.expected_writer,
                    expected_checkpoint=args.expected_checkpoint,
                    now=now,
                )

            mutate_registry(
                path,
                release_operation,
            )
        elif args.command == "release-reviewer":
            mutate_registry(
                path,
                lambda data: release_reviewer(
                    data,
                    graph,
                    task_id=args.task_id,
                    reviewer=args.reviewer,
                    expected_state=args.expected_state,
                    now=now,
                ),
            )
        elif args.command == "transfer":
            graph_task = _graph_task(graph, args.task_id)

            def transfer_operation(data: dict) -> dict:
                validate_writer_worktree(
                    data,
                    graph,
                    task_id=args.task_id,
                    writer=args.expected_writer,
                    checkpoint_sha=args.expected_checkpoint,
                )
                validate_task_worktree(
                    args.new_worktree,
                    branch=graph_task["branch"],
                    checkpoint_sha=args.expected_checkpoint,
                )
                return transfer_writer(
                    data,
                    graph,
                    task_id=args.task_id,
                    expected_writer=args.expected_writer,
                    expected_checkpoint=args.expected_checkpoint,
                    new_session_id=args.new_session,
                    new_tool=args.new_tool,
                    new_worktree=args.new_worktree,
                    now=now,
                )

            mutate_registry(
                path,
                transfer_operation,
            )
        else:
            raise RegistryError(f"unknown command: {args.command}")
        print(f"{args.command} completed for {args.task_id}")
        return 0
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
