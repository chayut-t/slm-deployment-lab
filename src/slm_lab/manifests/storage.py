"""Verify the external artifact root before storage-heavy work."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG = Path("configs/storage/external-ssd.example.yaml")


class StoragePreflightError(RuntimeError):
    """Raised when artifact storage is unsafe or unavailable."""


def load_config(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise StoragePreflightError("storage config schema_version must be 1")
    return config


def artifact_root(config: Mapping[str, Any], override: Path | None) -> Path:
    configured_env = config["artifact_root_env"]
    raw = (
        str(override)
        if override is not None
        else os.environ.get(configured_env, config["primary_machine_default"])
    )
    if not raw.strip():
        raise StoragePreflightError(
            f"{configured_env} is empty; provide a concrete artifact root"
        )
    requested = Path(raw).expanduser()
    try:
        return requested.resolve(strict=True)
    except OSError as exc:
        raise StoragePreflightError(
            f"artifact root cannot be resolved: {requested}: {exc}"
        ) from exc


def find_mount(path: Path) -> Path:
    candidate = path
    while not candidate.is_mount():
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return candidate


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def run_preflight(
    config: Mapping[str, Any],
    *,
    root_override: Path | None = None,
    write_probe: bool = True,
) -> dict[str, Any]:
    root = artifact_root(config, root_override)
    requested_mount = Path(config["required_mount"]).expanduser()
    try:
        required_mount = requested_mount.resolve(strict=True)
    except OSError as exc:
        raise StoragePreflightError(
            f"required external mount cannot be resolved: {requested_mount}: {exc}"
        ) from exc

    if root == Path("/"):
        raise StoragePreflightError("artifact root may not be the filesystem root")
    if not root.is_dir():
        raise StoragePreflightError(f"artifact root is not a directory: {root}")
    if not required_mount.is_mount():
        raise StoragePreflightError(
            f"required external mount is not mounted: {required_mount}"
        )
    if not _is_within(root, required_mount):
        raise StoragePreflightError(
            f"artifact root {root} is outside required mount {required_mount}"
        )

    actual_mount = find_mount(root)
    if actual_mount != required_mount:
        raise StoragePreflightError(
            f"artifact root resolves to mount {actual_mount}, "
            f"expected {required_mount}"
        )

    expected_directories = config["expected_directories"]
    unsafe = [
        relative
        for relative in expected_directories
        if Path(relative).is_absolute() or ".." in Path(relative).parts
    ]
    if unsafe:
        raise StoragePreflightError(
            "artifact layout contains unsafe paths: " + ", ".join(unsafe)
        )
    missing: list[str] = []
    escaped: list[str] = []
    for relative in expected_directories:
        candidate = root / relative
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            missing.append(relative)
            continue
        if not _is_within(resolved, root) or not _is_within(
            resolved,
            required_mount,
        ):
            escaped.append(f"{relative} -> {resolved}")
            continue
        if not resolved.is_dir():
            missing.append(relative)
    if escaped:
        raise StoragePreflightError(
            "artifact layout resolves outside the artifact root or mount: "
            + ", ".join(escaped)
        )
    if missing:
        raise StoragePreflightError(
            "artifact layout is incomplete; missing: " + ", ".join(missing)
        )

    if write_probe:
        prefix = config["write_probe_prefix"]
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=prefix,
                dir=root,
                delete=True,
            ) as probe:
                probe.write(b"slm-lab-storage-preflight\n")
                probe.flush()
                os.fsync(probe.fileno())
        except OSError as exc:
            raise StoragePreflightError(
                f"artifact root write probe failed: {root}: {exc}"
            ) from exc

    usage = shutil.disk_usage(root)
    minimum_free = int(config["minimum_free_bytes"])
    if usage.free < minimum_free:
        raise StoragePreflightError(
            f"artifact root has {usage.free} free bytes; "
            f"{minimum_free} required"
        )

    return {
        "schema_version": 1,
        "status": "pass",
        "artifact_root": str(root),
        "mount_point": str(actual_mount),
        "write_probe": "passed" if write_probe else "skipped",
        "expected_directory_count": len(expected_directories),
        "free_bytes": usage.free,
        "minimum_free_bytes": minimum_free,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--skip-write-probe", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        result = run_preflight(
            config,
            root_override=args.artifact_root,
            write_probe=not args.skip_write_probe,
        )
    except (OSError, ValueError, StoragePreflightError) as exc:
        print(f"storage preflight failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        gib = result["free_bytes"] / (1024**3)
        reserve_gib = result["minimum_free_bytes"] / (1024**3)
        print(
            f"storage preflight passed: {result['artifact_root']} "
            f"on {result['mount_point']}; {gib:.1f} GiB free "
            f"({reserve_gib:.1f} GiB required)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
