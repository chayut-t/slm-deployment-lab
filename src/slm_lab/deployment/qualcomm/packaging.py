"""Assemble a Workbench-ready package from a committed T22 QNN candidate.

The builder never contacts a service, never imports ``qai_hub``, and never
submits a job. It re-verifies the candidate graph and its external-data
sidecar against the committed manifest, assembles them into a package under
the external artifact root, writes a path-free record under ``results/``, and
generates the AI Hub compile request into private storage only.

The generated request is then validated with
:func:`slm_lab.deployment.qualcomm.ai_hub.preflight_compile_request`. That is
the entire meaning of "ready for submission" here: digests verified, request
generated, request accepted by the committed T30 adapter's own validation. The
package layout for an external-data ONNX model has not been verified against
the Qualcomm AI Hub service.

Errors use the sanitized register of ``ai_hub``: a message names the logical
object that failed, never a filesystem path, a service response, or a
credential.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The private helpers below are deliberate reuse, not reimplementation: a
# target config or a request path this module accepts must be one the
# committed T30 compile stage would accept.
from .ai_hub import (
    ALLOWED_DTYPES,
    SAFE_LOGICAL_NAME_PATTERN,
    SHA256_PATTERN,
    AiHubAdapterError,
    _assert_public_safe,
    _device,
    _private_output_path,
    _runtime,
    _safe_exact_version,
    _SafeArgumentParser,
    _validate_options,
    preflight_compile_request,
    sha256_file,
)


SCHEMA_VERSION = 1
RECORD_TYPE = "slm_lab.qualcomm.qnn_package"
TASK_ID = "T22"
CANDIDATE_STAGE = "qnn_candidate"
PACKAGE_STAGE = "qnn_package"
GRAPH_KINDS = ("prefill", "decode")
ARTIFACT_ROOT_TOKEN = "${SLM_LAB_ARTIFACT_ROOT}"
PACKAGE_ROOT_TEMPLATE = f"{ARTIFACT_ROOT_TOKEN}/onnx/qnn-package/{TASK_ID}"
COMPILED_ROOT_TEMPLATE = f"{ARTIFACT_ROOT_TOKEN}/qualcomm/compiled/{TASK_ID}"
CHECKSUM_FILE = "SHA256SUMS"
RECORD_DIRECTORY = "results/manifests/qnn/packages"
REQUEST_DIRECTORY = ".ai-local/profiles/T22"
DEFAULT_TARGET_CONFIG = "configs/targets/qualcomm-snapdragon-x-elite-crd.json"

# Written verbatim into every record so a reader who only ever sees the JSON
# cannot mistake an offline validation for a service result.
SERVICE_CAVEAT = (
    "The package layout for an external-data ONNX model has not been verified "
    "against the Qualcomm AI Hub service. No compile job was submitted and no "
    "service call was made. T31 owns the first real submission. Ready for "
    "submission means exactly three things: the candidate and sidecar digests "
    "were re-verified against the committed T22 manifest, a compile request "
    "was generated, and that request was accepted by the committed T30 "
    "adapter's own validation. It does not mean AI Hub accepted it."
)
SINGLE_FILE_CAVEAT = (
    "The compile request names only the .onnx file because the committed T30 "
    "adapter requires source_artifact.path to be one existing file. Whether "
    "the service reads the sidecar from the same directory, or requires a "
    "directory or an archive instead, is unverified."
)


class QnnPackagingError(AiHubAdapterError):
    """A sanitized packaging error safe to print in public task logs."""


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _repository_label(path: Path) -> str:
    """Return a committable label: repository-relative inside, else basename."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(_repository_root()).as_posix()
    except ValueError:
        return resolved.name


def _observed_at() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise QnnPackagingError(f"{field} is not readable valid JSON") from None
    if not isinstance(value, Mapping):
        raise QnnPackagingError(f"{field} must be a JSON object")
    return value


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QnnPackagingError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise QnnPackagingError(f"{field} must be a list")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise QnnPackagingError(f"{field} must be a nonempty string")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise QnnPackagingError(f"{field} must be a positive integer")
    return value


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise QnnPackagingError(f"{field} must be a lowercase SHA-256")
    return value


def _logical_name(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SAFE_LOGICAL_NAME_PATTERN.fullmatch(value):
        raise QnnPackagingError(f"{field} must be a path-free logical name")
    return value


def resolve_artifact_root(explicit: str | Path | None = None) -> Path:
    """``--artifact-root``, then ``SLM_LAB_ARTIFACT_ROOT``, then ``artifacts/``."""

    if explicit:
        root = Path(explicit).expanduser()
    else:
        configured = os.environ.get("SLM_LAB_ARTIFACT_ROOT", "").strip()
        root = (
            Path(configured).expanduser()
            if configured
            else _repository_root() / "artifacts"
        )
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    if not root.is_dir():
        raise QnnPackagingError(
            "artifact root does not exist; set SLM_LAB_ARTIFACT_ROOT or pass "
            "--artifact-root"
        )
    return root


def _expand_root(template: str, artifact_root: Path, field: str) -> Path:
    expanded = _text(template, field).replace(
        ARTIFACT_ROOT_TOKEN, artifact_root.as_posix()
    )
    directory = Path(expanded)
    if not directory.is_absolute():
        raise QnnPackagingError(f"{field} did not resolve to an absolute path")
    return directory


def load_candidate_manifest(path: Path) -> Mapping[str, Any]:
    """Read the T22 candidate manifest and check only the fields consumed here.

    Unknown top-level blocks are ignored on purpose: the transform engine owns
    the manifest schema and may add evidence blocks at any time. Every field
    this module reads is validated strictly.
    """

    manifest = _load_json(path, "candidate manifest")
    if manifest.get("schema_version") != 1:
        raise QnnPackagingError("candidate manifest schema_version must be 1")
    if manifest.get("stage") != CANDIDATE_STAGE:
        raise QnnPackagingError(f"candidate manifest stage must be {CANDIDATE_STAGE}")
    if manifest.get("task_id") != TASK_ID:
        raise QnnPackagingError(f"candidate manifest task_id must be {TASK_ID}")
    _logical_name(manifest.get("variant_id"), "candidate manifest variant_id")
    _positive_int(manifest.get("context_length"), "candidate manifest context_length")
    _positive_int(manifest.get("cache_capacity"), "candidate manifest cache_capacity")
    _positive_int(manifest.get("opset"), "candidate manifest opset")
    _text(manifest.get("precision"), "candidate manifest precision")
    _mapping(manifest.get("artifacts"), "candidate manifest artifacts")
    return manifest


def load_target_config(path: Path) -> Mapping[str, Any]:
    """Read the committed target selector and check the fields consumed here."""

    config = _load_json(path, "target config")
    if config.get("schema_version") != 1:
        raise QnnPackagingError("target config schema_version must be 1")
    _text(config.get("config_id"), "target config config_id")
    client = _mapping(config.get("client"), "target config client")
    if client.get("name") != "qai-hub":
        raise QnnPackagingError("target config client.name must be qai-hub")
    compile_block = _mapping(config.get("compile"), "target config compile")
    _text(compile_block.get("job_name_prefix"), "target config compile.job_name_prefix")
    _positive_int(
        compile_block.get("timeout_seconds"), "target config compile.timeout_seconds"
    )
    if compile_block.get("retry") is not False:
        raise QnnPackagingError("target config compile.retry must be false")
    normalize_target(config)
    return config


def normalize_target(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the selector through the committed T30 request validators.

    Reusing ``ai_hub`` here means a target config that this module accepts is
    a target config the compile stage would accept, and a bad option string
    fails before any large artifact is read or linked.
    """

    runtime = _runtime(config.get("runtime"))
    return {
        "client_version": _safe_exact_version(
            _mapping(config.get("client"), "target config client").get("version"),
            "target config client.version",
        ),
        "device": _device(config.get("device")),
        "runtime": runtime,
        "options": _validate_options(
            _mapping(config.get("compile"), "target config compile").get("options"),
            runtime,
            "compile",
        ),
    }


def _graph_record(manifest: Mapping[str, Any], graph_kind: str) -> Mapping[str, Any]:
    artifacts = _mapping(manifest["artifacts"], "candidate manifest artifacts")
    if graph_kind not in artifacts:
        raise QnnPackagingError(
            f"candidate manifest has no {graph_kind} graph to package"
        )
    return _mapping(artifacts[graph_kind], f"candidate manifest artifacts.{graph_kind}")


def _input_specs(record: Mapping[str, Any], graph_kind: str) -> dict[str, Any]:
    """Derive AI Hub compile input specs from the manifest's input tensors."""

    tensors = _sequence(
        record.get("input_tensors"), f"artifacts.{graph_kind}.input_tensors"
    )
    if not tensors:
        raise QnnPackagingError(f"{graph_kind} graph declares no input tensors")
    specs: dict[str, Any] = {}
    for index, raw in enumerate(tensors):
        field = f"artifacts.{graph_kind}.input_tensors[{index}]"
        tensor = _mapping(raw, field)
        name = _text(tensor.get("name"), f"{field}.name")
        if name in specs:
            raise QnnPackagingError(
                f"{graph_kind} graph declares a duplicate input tensor name"
            )
        dtype = _text(tensor.get("dtype"), f"{field}.dtype")
        if dtype not in ALLOWED_DTYPES:
            raise QnnPackagingError(
                f"{graph_kind} input tensor dtype {dtype} is unsupported by the "
                "compile adapter"
            )
        shape = _sequence(tensor.get("shape"), f"{field}.shape")
        if not shape:
            raise QnnPackagingError(f"{graph_kind} input tensor has an empty shape")
        specs[name] = {
            "shape": [
                _positive_int(dimension, f"{field}.shape") for dimension in shape
            ],
            "dtype": dtype,
        }
    return specs


def _sidecar_names(record: Mapping[str, Any], graph_kind: str) -> list[dict[str, Any]]:
    entries = _sequence(
        record.get("external_data", []), f"artifacts.{graph_kind}.external_data"
    )
    sidecars: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(entries):
        field = f"artifacts.{graph_kind}.external_data[{index}]"
        entry = _mapping(raw, field)
        location = _text(entry.get("location"), f"{field}.location")
        if location != Path(location).name or location in {".", ".."}:
            raise QnnPackagingError(
                f"{graph_kind} external data location must be a plain file name "
                "beside the graph; a nested location cannot be packaged flat"
            )
        _logical_name(location, f"{field}.location")
        if location in seen:
            raise QnnPackagingError(
                f"{graph_kind} graph declares a duplicate external data location"
            )
        seen.add(location)
        sidecars.append(
            {
                "role": "external_data",
                "logical_name": location,
                "sha256": _digest(entry.get("sha256"), f"{field}.sha256"),
                "size_bytes": _positive_int(
                    entry.get("size_bytes"), f"{field}.size_bytes"
                ),
            }
        )
    return sidecars


def _graph_members(record: Mapping[str, Any], graph_kind: str) -> list[dict[str, Any]]:
    """Return the manifest-declared package members, graph first."""

    relative = _text(
        record.get("relative_path"), f"artifacts.{graph_kind}.relative_path"
    )
    graph_name = Path(relative).name
    _logical_name(graph_name, f"artifacts.{graph_kind}.relative_path")
    members = [
        {
            "role": "candidate_graph",
            "logical_name": graph_name,
            "sha256": _digest(record.get("sha256"), f"artifacts.{graph_kind}.sha256"),
            "size_bytes": _positive_int(
                record.get("size_bytes"), f"artifacts.{graph_kind}.size_bytes"
            ),
        }
    ]
    for sidecar in _sidecar_names(record, graph_kind):
        if sidecar["logical_name"] == graph_name:
            raise QnnPackagingError(
                f"{graph_kind} external data collides with the graph file name"
            )
        members.append(sidecar)
    if len({member["logical_name"] for member in members}) != len(members):
        raise QnnPackagingError(f"{graph_kind} package members are not unique")
    return members


def _checksums_text(members: Sequence[Mapping[str, Any]]) -> str:
    rows = sorted(
        (member["logical_name"], member["sha256"])
        for member in members
        if member["role"] != "checksums"
    )
    return "".join(f"{sha}  {name}\n" for name, sha in rows)


def _checksums_member(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payload = _checksums_text(members).encode("utf-8")
    return {
        "role": "checksums",
        "logical_name": CHECKSUM_FILE,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _job_name(config: Mapping[str, Any], variant_id: str, graph_kind: str) -> str:
    prefix = config["compile"]["job_name_prefix"]
    return f"{prefix}-{variant_id}-{graph_kind}"


def _deterministic_record(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    config: Mapping[str, Any],
    config_path: Path,
    graph_kinds: Sequence[str],
) -> dict[str, Any]:
    """Build the committable record from the manifest and the target config.

    This is a pure function of its inputs and the layout policy in this
    module. Nothing observed from the local filesystem enters it, so ``build``
    and ``--check`` can compare their results directly.
    """

    variant_id = manifest["variant_id"]
    compile_block = config["compile"]
    target = normalize_target(config)
    graphs: list[dict[str, Any]] = []
    for graph_kind in graph_kinds:
        record = _graph_record(manifest, graph_kind)
        members = _graph_members(record, graph_kind)
        members.append(_checksums_member(members))
        graphs.append(
            {
                "graph_kind": graph_kind,
                "package_relative_path": f"{variant_id}/{graph_kind}",
                "files": members,
                "compile_request": {
                    "request_id": None,
                    "source_logical_name": members[0]["logical_name"],
                    "output_logical_name": f"{variant_id}-{graph_kind}.serialized.bin",
                    "job_name": _job_name(config, variant_id, graph_kind),
                    "input_specs": _input_specs(record, graph_kind),
                    "external_data_members": [
                        member["logical_name"]
                        for member in members
                        if member["role"] == "external_data"
                    ],
                    "single_file_source_caveat": SINGLE_FILE_CAVEAT,
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": RECORD_TYPE,
        "task_id": TASK_ID,
        "stage": PACKAGE_STAGE,
        "variant_id": variant_id,
        "context_length": manifest["context_length"],
        "cache_capacity": manifest["cache_capacity"],
        "opset": manifest["opset"],
        "precision": manifest["precision"],
        "source_manifest": {
            "path": _repository_label(manifest_path),
            "sha256": sha256_file(manifest_path),
            "stage": CANDIDATE_STAGE,
            "task_id": TASK_ID,
        },
        "target_config": {
            "path": _repository_label(config_path),
            "sha256": sha256_file(config_path),
            "config_id": config["config_id"],
        },
        "client": {"name": "qai-hub", "version": target["client_version"]},
        "device": target["device"],
        "runtime": target["runtime"],
        "compile": {
            "options": target["options"],
            "timeout_seconds": compile_block["timeout_seconds"],
            "retry": False,
        },
        "package": {
            "root_token": PACKAGE_ROOT_TEMPLATE,
            "checksum_file": CHECKSUM_FILE,
            "checksum_format": "sha256sum",
            "graphs": graphs,
        },
        "submission_status": {
            "job_submitted": False,
            "service_contacted": False,
            "package_layout_verified_against_service": False,
            "first_submission_owner": "T31",
            "ready_for_submission_means": [
                "candidate_and_sidecar_digests_reverified_against_the_manifest",
                "compile_request_generated_into_private_storage",
                "request_accepted_by_the_committed_T30_adapter_validation",
            ],
            "caveat": SERVICE_CAVEAT,
        },
        "claim_boundary": {
            "establishes": [
                "package_contents_match_the_committed_candidate_manifest_digests",
                "package_is_self_verifying_through_a_sha256sum_checksum_file",
                "compile_request_satisfies_the_committed_T30_adapter_contract",
            ],
            "does_not_establish": [
                "qualcomm_ai_hub_accepted_or_would_accept_this_request",
                "qualcomm_ai_hub_accepts_this_external_data_package_layout",
                "compiler_acceptance_of_the_candidate_graph",
                "accelerator_placement_or_device_numerical_parity",
                "latency_or_memory_performance",
            ],
        },
    }


def _package_directory(package_root: Path, graph: Mapping[str, Any]) -> Path:
    return package_root / graph["package_relative_path"]


def _source_paths(
    manifest: Mapping[str, Any],
    artifact_root: Path,
    graph_kind: str,
) -> tuple[Path, Path]:
    artifacts = _mapping(manifest["artifacts"], "candidate manifest artifacts")
    root = _expand_root(
        artifacts.get("root", ""), artifact_root, "candidate manifest artifacts.root"
    )
    record = _graph_record(manifest, graph_kind)
    relative = _text(
        record.get("relative_path"), f"artifacts.{graph_kind}.relative_path"
    )
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise QnnPackagingError(
            f"artifacts.{graph_kind}.relative_path must stay under the candidate root"
        )
    graph_path = root / relative
    return graph_path, graph_path.parent


def _verify_source(path: Path, member: Mapping[str, Any], graph_kind: str) -> None:
    name = member["logical_name"]
    if not path.is_file():
        raise QnnPackagingError(f"{graph_kind} package member {name} is missing")
    try:
        size = path.stat().st_size
    except OSError:
        raise QnnPackagingError(
            f"{graph_kind} package member {name} metadata is unavailable"
        ) from None
    if size != member["size_bytes"]:
        raise QnnPackagingError(f"{graph_kind} package member {name} has wrong size")
    if sha256_file(path) != member["sha256"]:
        raise QnnPackagingError(f"{graph_kind} package member {name} digest mismatch")


def _place(source: Path, destination: Path) -> str:
    """Hardlink when the filesystem allows it, otherwise copy."""

    try:
        if destination.exists() or destination.is_symlink():
            destination.unlink()
    except OSError:
        raise QnnPackagingError("stale package member could not be replaced") from None
    try:
        os.link(source, destination)
        return "hardlink"
    except (OSError, NotImplementedError, AttributeError):
        pass
    try:
        shutil.copy2(source, destination)
    except OSError:
        raise QnnPackagingError("package member could not be copied") from None
    return "copy"


def _same_inode(left: Path, right: Path) -> bool:
    try:
        first = left.stat()
        second = right.stat()
    except OSError:
        return False
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _write_bytes(path: Path, payload: bytes, field: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    except OSError:
        raise QnnPackagingError(f"{field} could not be written") from None


def _compile_request(
    record: Mapping[str, Any],
    graph: Mapping[str, Any],
    *,
    package_directory: Path,
    compiled_root: Path,
) -> dict[str, Any]:
    request_block = graph["compile_request"]
    source_name = request_block["source_logical_name"]
    source = next(
        member for member in graph["files"] if member["logical_name"] == source_name
    )
    output = compiled_root / record["variant_id"] / request_block["output_logical_name"]
    return {
        "schema_version": 2,
        "stage": "compile",
        "client_version": record["client"]["version"],
        "device": dict(record["device"]),
        "runtime": dict(record["runtime"]),
        "source_artifact": {
            "path": str(package_directory / source_name),
            "logical_name": source_name,
            "sha256": source["sha256"],
        },
        "output_artifact": str(output),
        "output_logical_name": request_block["output_logical_name"],
        "input_specs": json.loads(json.dumps(request_block["input_specs"])),
        "options": record["compile"]["options"],
        "job_name": request_block["job_name"],
        "timeout_seconds": record["compile"]["timeout_seconds"],
        "retry": False,
    }


def _request_path(
    request_directory: Path, record: Mapping[str, Any], kind: str
) -> Path:
    return request_directory / record["variant_id"] / f"{kind}-compile-request.json"


def _generate_and_preflight(
    record: Mapping[str, Any],
    graph: Mapping[str, Any],
    *,
    package_directory: Path,
    compiled_root: Path,
    request_directory: Path,
) -> dict[str, Any]:
    """Write the request to private storage and validate it through T30."""

    request = _compile_request(
        record,
        graph,
        package_directory=package_directory,
        compiled_root=compiled_root,
    )
    path = _private_output_path(
        str(_request_path(request_directory, record, graph["graph_kind"])),
        "compile request",
    )
    _write_bytes(
        path,
        (json.dumps(request, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        "compile request",
    )
    return preflight_compile_request(path)


def _first_difference(expected: Any, actual: Any, prefix: str = "") -> str | None:
    """Return the first differing key path, never a value."""

    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                return f"{prefix}.{key}" if prefix else str(key)
            found = _first_difference(
                expected[key], actual[key], f"{prefix}.{key}" if prefix else str(key)
            )
            if found is not None:
                return found
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return prefix or "record"
        for index, (left, right) in enumerate(zip(expected, actual)):
            found = _first_difference(left, right, f"{prefix}[{index}]")
            if found is not None:
                return found
        return None
    if expected != actual:
        return prefix or "record"
    return None


def _resolve_graph_kinds(
    manifest: Mapping[str, Any], requested: Sequence[str]
) -> list[str]:
    artifacts = _mapping(manifest["artifacts"], "candidate manifest artifacts")
    kinds = [kind for kind in requested if kind in GRAPH_KINDS]
    if not kinds:
        raise QnnPackagingError("no supported graph kind was requested")
    missing = [kind for kind in kinds if kind not in artifacts]
    if missing:
        raise QnnPackagingError(
            f"candidate manifest has no {missing[0]} graph to package"
        )
    return kinds


def _record_path(explicit: Path | None, variant_id: str) -> Path:
    if explicit is not None:
        return explicit
    return _repository_root() / RECORD_DIRECTORY / f"{variant_id}.json"


def build_package(
    manifest_path: Path,
    *,
    target_path: Path,
    artifact_root: Path,
    graph_kinds: Sequence[str] = GRAPH_KINDS,
    record_path: Path | None = None,
    request_directory: Path | None = None,
    package_root: Path | None = None,
) -> dict[str, Any]:
    """Verify, assemble, record, and preflight one candidate variant."""

    manifest = load_candidate_manifest(manifest_path)
    config = load_target_config(target_path)
    kinds = _resolve_graph_kinds(manifest, graph_kinds)
    record = _deterministic_record(manifest, manifest_path, config, target_path, kinds)

    resolved_package_root = (
        package_root
        if package_root is not None
        else _expand_root(PACKAGE_ROOT_TEMPLATE, artifact_root, "package root")
    )
    compiled_root = _expand_root(
        COMPILED_ROOT_TEMPLATE, artifact_root, "compiled output root"
    )
    requests_at = (
        request_directory
        if request_directory is not None
        else _repository_root() / REQUEST_DIRECTORY
    )

    observations: list[dict[str, Any]] = []
    for graph in record["package"]["graphs"]:
        graph_kind = graph["graph_kind"]
        graph_path, source_directory = _source_paths(
            manifest, artifact_root, graph_kind
        )
        destination = _package_directory(resolved_package_root, graph)
        try:
            destination.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise QnnPackagingError(
                f"{graph_kind} package directory could not be prepared"
            ) from None

        placements: list[dict[str, str]] = []
        for member in graph["files"]:
            if member["role"] == "checksums":
                continue
            name = member["logical_name"]
            source = (
                graph_path
                if member["role"] == "candidate_graph"
                else source_directory / name
            )
            _verify_source(source, member, graph_kind)
            target = destination / name
            link_mode = _place(source, target)
            if link_mode == "hardlink" and _same_inode(source, target):
                evidence = "same_inode_as_verified_source"
            else:
                _verify_source(target, member, graph_kind)
                evidence = "rehashed_after_placement"
            placements.append(
                {
                    "logical_name": name,
                    "link_mode": link_mode,
                    "digest_evidence": evidence,
                }
            )

        checksums = next(
            member for member in graph["files"] if member["role"] == "checksums"
        )
        _write_bytes(
            destination / CHECKSUM_FILE,
            _checksums_text(graph["files"]).encode("utf-8"),
            "checksum file",
        )
        _verify_source(destination / CHECKSUM_FILE, checksums, graph_kind)

        preflight = _generate_and_preflight(
            record,
            graph,
            package_directory=destination,
            compiled_root=compiled_root,
            request_directory=requests_at,
        )
        graph["compile_request"]["request_id"] = preflight["request_id"]
        observations.append(
            {
                "graph_kind": graph_kind,
                "request_id": preflight["request_id"],
                "placements": placements,
            }
        )

    record["build_observation"] = {
        "created_at_utc": _observed_at(),
        "artifact_root_committed": False,
        "request_written_to_repository": False,
        "graphs": observations,
    }
    _assert_public_safe(record, "package record")

    destination_record = _record_path(record_path, record["variant_id"])
    _write_bytes(
        destination_record,
        (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        "package record",
    )
    return {
        "mode": "build",
        "status": "ok",
        "variant_id": record["variant_id"],
        "record": _repository_label(destination_record),
        "graphs": [
            {
                "graph_kind": item["graph_kind"],
                "request_id": item["request_id"],
                "link_modes": sorted(
                    {placement["link_mode"] for placement in item["placements"]}
                ),
            }
            for item in observations
        ],
        "job_submitted": False,
        "service_contacted": False,
        "package_layout_verified_against_service": False,
    }


def check_package(
    manifest_path: Path,
    *,
    target_path: Path,
    artifact_root: Path,
    graph_kinds: Sequence[str] = GRAPH_KINDS,
    record_path: Path | None = None,
    request_directory: Path | None = None,
    package_root: Path | None = None,
) -> dict[str, Any]:
    """Re-verify an assembled package against its committed record.

    Nothing is relinked, copied, or rebuilt. The request is regenerated into
    private storage because its acceptance by the committed T30 adapter is
    part of what the record claims.
    """

    manifest = load_candidate_manifest(manifest_path)
    config = load_target_config(target_path)
    kinds = _resolve_graph_kinds(manifest, graph_kinds)
    expected = _deterministic_record(
        manifest, manifest_path, config, target_path, kinds
    )

    committed_path = _record_path(record_path, expected["variant_id"])
    committed = dict(_load_json(committed_path, "package record"))
    committed.pop("build_observation", None)

    resolved_package_root = (
        package_root
        if package_root is not None
        else _expand_root(PACKAGE_ROOT_TEMPLATE, artifact_root, "package root")
    )
    compiled_root = _expand_root(
        COMPILED_ROOT_TEMPLATE, artifact_root, "compiled output root"
    )
    requests_at = (
        request_directory
        if request_directory is not None
        else _repository_root() / REQUEST_DIRECTORY
    )

    verified: list[dict[str, Any]] = []
    for graph in expected["package"]["graphs"]:
        destination = _package_directory(resolved_package_root, graph)
        for member in graph["files"]:
            _verify_source(
                destination / member["logical_name"], member, graph["graph_kind"]
            )
        preflight = _generate_and_preflight(
            expected,
            graph,
            package_directory=destination,
            compiled_root=compiled_root,
            request_directory=requests_at,
        )
        graph["compile_request"]["request_id"] = preflight["request_id"]
        verified.append(
            {"graph_kind": graph["graph_kind"], "request_id": preflight["request_id"]}
        )

    difference = _first_difference(expected, committed)
    if difference is not None:
        raise QnnPackagingError(
            f"package record no longer matches the candidate manifest at {difference}"
        )
    return {
        "mode": "check",
        "status": "ok",
        "variant_id": expected["variant_id"],
        "record": _repository_label(committed_path),
        "graphs": verified,
        "job_submitted": False,
        "service_contacted": False,
        "package_layout_verified_against_service": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description=(
            "Package a T22 QNN candidate and validate its AI Hub compile "
            "request offline. Submits nothing."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--target",
        type=Path,
        default=_repository_root() / DEFAULT_TARGET_CONFIG,
    )
    parser.add_argument(
        "--graph",
        choices=(*GRAPH_KINDS, "all"),
        default="all",
    )
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--package-root", type=Path, default=None)
    parser.add_argument("--record", type=Path, default=None)
    parser.add_argument("--request-dir", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    kinds = GRAPH_KINDS if args.graph == "all" else (args.graph,)
    try:
        artifact_root = resolve_artifact_root(args.artifact_root)
        runner = check_package if args.check else build_package
        summary = runner(
            args.manifest,
            target_path=args.target,
            artifact_root=artifact_root,
            graph_kinds=kinds,
            record_path=args.record,
            request_directory=args.request_dir,
            package_root=args.package_root,
        )
    except AiHubAdapterError as exc:
        print(f"qnn packaging failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0
