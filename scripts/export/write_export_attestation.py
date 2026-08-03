#!/usr/bin/env python3
"""Regenerate the T20 export attestation from the bytes on disk.

Every digest the T20 export config asserts -- the eight graph SHA-256s, the one
shared external-data SHA-256, the source-weight SHA-256, and the
``FROZEN_EXPORT_CONFIG_SHA256`` pin in the exporter itself -- is produced here
by hashing a file. Nothing in the attestation is meant to be typed by hand, and
a digest that was typed by hand is indistinguishable from a digest that was
wrong.

Why this is a separate script rather than an ``attest`` subcommand on
``slm_lab.export.onnx_matrix``: every entry point of that CLI begins with
``load_export_config``, which refuses to return until the on-disk config
carries an ``evidence_attestation`` block, hashes to the code-pinned
``FROZEN_EXPORT_CONFIG_SHA256``, and is byte-identical to ``HEAD``'s copy.
Those are exactly the conditions re-attestation exists to restore, so an
``attest`` subcommand would have to bypass its own module's trust root before
it could do anything -- weakening the invariant that every ``onnx_matrix``
command loads the trusted config first. Keeping the re-attestation outside that
CLI leaves the invariant unconditional.

Promotion is commit-gated and needs two chained commits, because
``_export_provenance`` requires the config at the attested commit to equal the
current config with the ``evidence_attestation`` key removed:

  1. ``strip``  -- commit the exporter plus an unattested config. This is the
     commit the next export is attested *to*. The CLI cannot load a config in
     this state; that is inherent and transient.
  2. ``write``  -- re-add the attestation naming commit 1, then export, then
     ``validate --write-manifests``, whose ``_validate_attested_artifacts``
     confirms the attested digests against the freshly written bytes. Re-run
     ``write`` and commit the correction if any digest disagrees.

``runtime_python_version`` is always taken from the interpreter running this
script, never from an argument. The attestation must describe the interpreter
that actually performs the export, so run this with that interpreter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# Deliberately the exporter's own producers: a digest recorded here and a digest
# checked by ``_validate_attested_artifacts`` must come from the same code.
from slm_lab.contracts import CONTEXT_VARIANTS  # noqa: E402
from slm_lab.export.onnx_matrix import (  # noqa: E402
    ARTIFACT_SUBDIRECTORY,
    DEFAULT_CONFIG_PATH,
    EXPORTER_SOURCE_PATH,
    _sha256,
    _source_weights_path,
)


GRAPH_KINDS = ("prefill", "decode")
ATTESTATION_KEY = "evidence_attestation"
ATTESTATION_AFTER_KEY = "export"
RUN_ID_PATTERN = re.compile(r"T20-[A-Za-z0-9._-]+")
FROZEN_PIN_PATTERN = re.compile(
    r"(FROZEN_EXPORT_CONFIG_SHA256 = \(\n    \")[0-9a-f]{64}(\"\n\))"
)


class AttestationError(RuntimeError):
    """The attestation cannot be regenerated from the bytes on disk."""


def _read_config(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AttestationError(f"invalid export config {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise AttestationError(f"export config {path} is not a JSON object")
    return document


def _config_bytes(document: dict[str, Any]) -> bytes:
    """Serialize exactly as the tracked config is spelled, so bytes round-trip."""

    return (json.dumps(document, indent=2) + "\n").encode("utf-8")


def _place_attestation(
    document: dict[str, Any],
    attestation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the document with the attestation removed or put back in place."""

    rebuilt: dict[str, Any] = {}
    for key, value in document.items():
        if key == ATTESTATION_KEY:
            continue
        rebuilt[key] = value
        if key == ATTESTATION_AFTER_KEY and attestation is not None:
            rebuilt[ATTESTATION_KEY] = attestation
    if attestation is not None and ATTESTATION_KEY not in rebuilt:
        raise AttestationError(
            f"export config has no {ATTESTATION_AFTER_KEY!r} key to anchor the "
            "attestation position"
        )
    return rebuilt


def _repin_exporter(exporter_source: Path, config_sha256: str) -> bool:
    """Rewrite ``FROZEN_EXPORT_CONFIG_SHA256``; report whether it moved."""

    try:
        text = exporter_source.read_text(encoding="utf-8")
    except OSError as exc:
        raise AttestationError(
            f"cannot read exporter {exporter_source}: {exc}"
        ) from exc
    replacement, count = FROZEN_PIN_PATTERN.subn(
        lambda match: f"{match.group(1)}{config_sha256}{match.group(2)}",
        text,
    )
    if count != 1:
        raise AttestationError(
            f"expected exactly one FROZEN_EXPORT_CONFIG_SHA256 literal in "
            f"{exporter_source}, found {count}"
        )
    if replacement == text:
        return False
    exporter_source.write_text(replacement, encoding="utf-8")
    return True


def _resolve_commit(revision: str) -> str:
    try:
        resolved = subprocess.check_output(
            ("git", "rev-parse", f"{revision}^{{commit}}"),
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AttestationError(f"cannot resolve exporter commit {revision}") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise AttestationError(f"exporter commit resolved oddly: {resolved!r}")
    ancestry = subprocess.run(
        ("git", "merge-base", "--is-ancestor", resolved, "HEAD"),
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if ancestry.returncode != 0:
        raise AttestationError(
            f"exporter commit {resolved} is not an ancestor of HEAD; the "
            "exporter refuses such an attestation"
        )
    return resolved


def _artifact_directory(explicit: str | None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    root = os.environ.get("SLM_LAB_ARTIFACT_ROOT")
    if not root:
        raise AttestationError(
            "SLM_LAB_ARTIFACT_ROOT must identify external artifact storage, or "
            "pass --artifact-directory"
        )
    return (Path(root).expanduser() / ARTIFACT_SUBDIRECTORY).resolve()


def _graph_path(root: Path, context: int, graph_kind: str) -> Path:
    path = root / f"S{context}" / f"{graph_kind}.onnx"
    if not path.is_file():
        raise AttestationError(f"missing S{context} {graph_kind} graph: {path}")
    return path


def measure_graphs(
    *,
    prefill_root: Path,
    decode_root: Path,
    contexts: Iterable[int],
) -> tuple[dict[str, dict[str, str]], str, list[dict[str, Any]]]:
    """Hash every graph and its sidecar; require one shared external digest."""

    graph_sha256: dict[str, dict[str, str]] = {}
    external_digests: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for context in contexts:
        entry: dict[str, str] = {}
        for graph_kind in GRAPH_KINDS:
            root = prefill_root if graph_kind == "prefill" else decode_root
            path = _graph_path(root, context, graph_kind)
            sidecar = path.with_name(f"{path.name}.data")
            if not sidecar.is_file():
                raise AttestationError(f"missing external data sidecar: {sidecar}")
            digest = _sha256(path)
            entry[graph_kind] = digest
            external_digests[str(sidecar)] = _sha256(sidecar)
            rows.append(
                {
                    "context": context,
                    "graph_kind": graph_kind,
                    "path": str(path),
                    "sha256": digest,
                    "size_bytes": path.stat().st_size,
                    "external_data_sha256": external_digests[str(sidecar)],
                    "external_data_size_bytes": sidecar.stat().st_size,
                }
            )
        graph_sha256[f"S{context}"] = entry
    distinct = sorted(set(external_digests.values()))
    if len(distinct) != 1:
        detail = "\n".join(
            f"  {digest}  {path}" for path, digest in sorted(external_digests.items())
        )
        raise AttestationError(
            "the export attestation records one shared external_data_sha256, but "
            f"{len(distinct)} distinct sidecar digests were measured:\n{detail}"
        )
    return graph_sha256, distinct[0], rows


def build_attestation(
    *,
    run_id: str,
    exporter_commit: str,
    source_weights: Path,
    prefill_root: Path,
    decode_root: Path,
    contexts: Iterable[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the attestation block entirely from measured bytes."""

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise AttestationError(
            f"run_id must match {RUN_ID_PATTERN.pattern}, found {run_id!r}"
        )
    if not source_weights.is_file():
        raise AttestationError(f"missing pinned source weights: {source_weights}")
    graph_sha256, external_data_sha256, rows = measure_graphs(
        prefill_root=prefill_root,
        decode_root=decode_root,
        contexts=contexts,
    )
    attestation = {
        "schema_version": 1,
        "run_id": run_id,
        "exporter_commit": exporter_commit,
        # Never an argument: the attestation must name the interpreter that
        # actually runs the export, and this is that interpreter.
        "runtime_python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "source_artifact_sha256": _sha256(source_weights),
        "external_data_sha256": external_data_sha256,
        "graph_sha256": graph_sha256,
    }
    return attestation, rows


def _apply(
    *,
    config_path: Path,
    exporter_source: Path,
    attestation: dict[str, Any] | None,
) -> tuple[str, bool, bool]:
    document = _place_attestation(_read_config(config_path), attestation)
    raw = _config_bytes(document)
    digest = hashlib.sha256(raw).hexdigest()
    config_changed = raw != config_path.read_bytes()
    if config_changed:
        config_path.write_text(raw.decode("utf-8"), encoding="utf-8")
    exporter_changed = _repin_exporter(exporter_source, digest)
    return digest, config_changed, exporter_changed


def _report(
    *,
    action: str,
    config_path: Path,
    exporter_source: Path,
    digest: str,
    config_changed: bool,
    exporter_changed: bool,
    rows: Sequence[dict[str, Any]] = (),
) -> None:
    for row in rows:
        print(
            f"S{row['context']:<5} {row['graph_kind']:<8} "
            f"{row['size_bytes']:>12,} bytes  {row['sha256']}"
        )
    if rows:
        shared = rows[0]["external_data_sha256"]
        print(f"shared external_data_sha256 ({len(rows)} sidecars): {shared}")
    print(f"{action}: {config_path.relative_to(REPO_ROOT)}")
    print(
        f"  config sha256 {digest} ({'rewritten' if config_changed else 'unchanged'})"
    )
    print(
        f"  FROZEN_EXPORT_CONFIG_SHA256 in "
        f"{exporter_source.relative_to(REPO_ROOT)} "
        f"({'repinned' if exporter_changed else 'unchanged'})"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/export/write_export_attestation.py",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--exporter-source", type=Path, default=EXPORTER_SOURCE_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "strip",
        help="remove the attestation block and repin the config digest",
    )
    write = subparsers.add_parser(
        "write",
        help="measure the exported artifacts and write the attestation block",
    )
    write.add_argument("--run-id", required=True, help="T20-<slug> evidence run id")
    write.add_argument(
        "--exporter-commit",
        required=True,
        help="revision of the unattested commit the export is attested to",
    )
    write.add_argument(
        "--artifact-directory",
        default=None,
        help=(
            "directory holding S<context>/{prefill,decode}.onnx "
            "(default $SLM_LAB_ARTIFACT_ROOT/onnx/reference/T20)"
        ),
    )
    write.add_argument(
        "--prefill-root",
        default=None,
        help="override the artifact directory for prefill graphs only",
    )
    write.add_argument(
        "--decode-root",
        default=None,
        help="override the artifact directory for decode graphs only",
    )
    write.add_argument(
        "--source-weights",
        default=None,
        help="pinned safetensors weights (default resolved from HF_HOME)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).resolve()
    exporter_source = Path(args.exporter_source).resolve()
    try:
        if args.command == "strip":
            digest, config_changed, exporter_changed = _apply(
                config_path=config_path,
                exporter_source=exporter_source,
                attestation=None,
            )
            _report(
                action="stripped attestation from",
                config_path=config_path,
                exporter_source=exporter_source,
                digest=digest,
                config_changed=config_changed,
                exporter_changed=exporter_changed,
            )
            return 0

        artifact_directory = _artifact_directory(args.artifact_directory)
        prefill_root = (
            Path(args.prefill_root).expanduser().resolve()
            if args.prefill_root
            else artifact_directory
        )
        decode_root = (
            Path(args.decode_root).expanduser().resolve()
            if args.decode_root
            else artifact_directory
        )
        source_weights = (
            Path(args.source_weights).expanduser().resolve()
            if args.source_weights
            else _source_weights_path()
        )
        attestation, rows = build_attestation(
            run_id=args.run_id,
            exporter_commit=_resolve_commit(args.exporter_commit),
            source_weights=source_weights,
            prefill_root=prefill_root,
            decode_root=decode_root,
            contexts=CONTEXT_VARIANTS,
        )
        digest, config_changed, exporter_changed = _apply(
            config_path=config_path,
            exporter_source=exporter_source,
            attestation=attestation,
        )
        _report(
            action="wrote attestation into",
            config_path=config_path,
            exporter_source=exporter_source,
            digest=digest,
            config_changed=config_changed,
            exporter_changed=exporter_changed,
            rows=rows,
        )
    except AttestationError as exc:
        print(f"attestation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
