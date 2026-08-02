"""Fail-closed baseline parity preflight run before any weight is quantized.

Parity has two halves and this module never lets a caller confuse them.

Half one is artifact identity: the T20 export attestation, the committed T20
manifests, the frozen T12 tensor contracts, and the bytes on the external
artifact root must still describe one single floating baseline. That half runs
anywhere and needs nothing heavier than :mod:`hashlib`.

Half two is numerical: logit-level agreement between the T11 deterministic
PyTorch reference and the T20 ONNX export. It requires ``torch`` and
``onnxruntime``, which the primary macOS host deliberately does not carry, so
it is recorded as a declared ``not_run`` requirement carrying the command that
completes it and the task that owns it.

Because half two is a declaration rather than a measurement, this module at
this commit can only ever emit ``failed``, ``unavailable``, or ``partial`` as
its overall verdict. There is no input that produces ``verified`` and no code
path that could; closing the numerical half is T21 work that must add a real
measurement here, not flip a flag.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from slm_lab.contracts.static_cache import (
    CONTEXT_VARIANTS,
    MODEL_ID,
    MODEL_REVISION,
    CacheContractError,
    GraphContract,
    TensorSpec,
    build_decode_contract,
    build_prefill_contract,
)


TASK_ID = "T40"
BASELINE_TASK_ID = "T20"
SCHEMA_VERSION = 1
RECORD_TYPE = "baseline_parity_preflight"

DEFAULT_EXPORT_CONFIG = Path("configs/models/qwen3-0.6b-onnx-export.json")
DEFAULT_MODEL_CONTRACT = Path("configs/models/qwen3-0.6b.yaml")
DEFAULT_MANIFEST_DIRECTORY = Path("results/manifests/onnx")
DEFAULT_STORAGE_CONFIG = Path("configs/storage/external-ssd.example.yaml")
DEFAULT_EVIDENCE_DIRECTORY = Path("results/quantization")

GRAPH_KINDS = ("prefill", "decode")
READ_BLOCK_BYTES = 8 * 1024 * 1024
MAX_REPORTED_FAILURES = 24

VERIFY_COMMAND = (
    "SLM_LAB_ARTIFACT_ROOT=<external-root> PYTHONPATH=src "
    "python -m slm_lab.quantization.parity verify"
)
RECORD_COMMAND = (
    "SLM_LAB_ARTIFACT_ROOT=<external-root> PYTHONPATH=src "
    "python -m slm_lab.quantization.parity record"
)

ESTABLISHES = (
    "committed_T20_attestation_and_manifests_still_agree",
    "recorded_model_and_tokenizer_revision_still_matches_T12",
    "recorded_graph_boundaries_still_satisfy_the_frozen_T12_contract",
    "on_disk_graph_and_external_data_bytes_still_hash_to_the_recorded_digests",
)
DOES_NOT_ESTABLISH = (
    "pytorch_versus_onnx_numerical_logit_parity",
    "onnxruntime_execution_or_multi_step_cache_correctness",
    "quantized_quality_delta",
    "compiler_acceptance_or_accelerator_placement",
    "latency_or_memory_performance",
)


class BaselineParityError(ValueError):
    """The floating baseline cannot be established as the recorded artifact."""


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one baseline-parity check.

    ``status`` is one of ``passed``, ``failed``, ``unavailable``, or
    ``skipped``. Only ``passed`` may ever be read as evidence.
    """

    name: str
    status: str
    summary: str
    failures: tuple[str, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a compact, JSON-serializable view of the outcome."""

        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
        }
        if self.failures:
            payload["failures"] = list(self.failures)
        if self.detail:
            payload["detail"] = dict(self.detail)
        return payload


@dataclass(frozen=True)
class FileDigest:
    """One artifact file measured against its committed T20 record."""

    relative_path: str
    recorded_sha256: str
    measured_sha256: str
    recorded_size_bytes: int
    measured_size_bytes: int
    measurement: str = "recomputed_sha256"

    @property
    def matched(self) -> bool:
        """Return whether both the digest and the byte size still agree."""

        return (
            self.recorded_sha256 == self.measured_sha256
            and self.recorded_size_bytes == self.measured_size_bytes
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a compact, JSON-serializable view of the measurement."""

        return {
            "relative_path": self.relative_path,
            "measurement": self.measurement,
            "recorded_sha256": self.recorded_sha256,
            "measured_sha256": self.measured_sha256,
            "recorded_size_bytes": self.recorded_size_bytes,
            "measured_size_bytes": self.measured_size_bytes,
            "matched": self.matched,
        }


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        raise BaselineParityError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BaselineParityError(f"cannot parse {path}: {exc}") from exc


def _sha256(path: Path, *, block_size: int = READ_BLOCK_BYTES) -> str:
    """Return a chunked SHA-256 using the same convention as the T20 exporter."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _cap(failures: Sequence[str]) -> tuple[str, ...]:
    if len(failures) <= MAX_REPORTED_FAILURES:
        return tuple(failures)
    hidden = len(failures) - MAX_REPORTED_FAILURES
    return (
        *failures[:MAX_REPORTED_FAILURES],
        f"... {hidden} further failure(s) omitted",
    )


def load_manifests(repo_root: Path) -> dict[int, Mapping[str, Any]]:
    """Load every committed T20 artifact manifest keyed by context length."""

    directory = repo_root / DEFAULT_MANIFEST_DIRECTORY
    if not directory.is_dir():
        raise BaselineParityError(f"T20 manifest directory is missing: {directory}")
    manifests: dict[int, Mapping[str, Any]] = {}
    for path in sorted(directory.glob("S*.json")):
        try:
            context = int(path.stem[1:])
        except ValueError as exc:
            raise BaselineParityError(
                f"manifest name is not an S<context> variant: {path.name}"
            ) from exc
        document = _load_json(path)
        if not isinstance(document, Mapping):
            raise BaselineParityError(f"manifest is not a JSON object: {path}")
        manifests[context] = document
    if not manifests:
        raise BaselineParityError(f"no T20 manifests found under {directory}")
    return manifests


def artifact_subdirectory(
    export_config: Mapping[str, Any],
    environment_variable: str,
) -> Path:
    """Derive the artifact subdirectory declared by the T20 export config."""

    declared = str(export_config.get("artifact_directory", ""))
    prefix = "${" + environment_variable + "}/"
    if not declared.startswith(prefix):
        raise BaselineParityError(
            f"export config artifact_directory must start with {prefix!r}, "
            f"found {declared!r}"
        )
    relative = PurePosixPath(declared[len(prefix) :])
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise BaselineParityError(
            f"export config artifact_directory is unsafe: {declared!r}"
        )
    return Path(*relative.parts)


def resolve_artifact_root(
    repo_root: Path,
    *,
    override: Path | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    """Resolve the artifact root exactly as the T01 storage preflight does.

    The environment variable name and the primary-machine default both come
    from the committed storage config, so no machine path is hardcoded here.
    Returns ``(None, resolution)`` when the root cannot be resolved; that is a
    refusal, never a pass.
    """

    storage = _load_json(repo_root / DEFAULT_STORAGE_CONFIG)
    environment_variable = str(storage["artifact_root_env"])
    default_root = str(storage["primary_machine_default"])
    if override is not None:
        raw, source = str(override), "explicit_override"
    else:
        from_environment = os.environ.get(environment_variable, "")
        if from_environment.strip():
            raw = from_environment
            source = f"environment:{environment_variable}"
        else:
            raw = default_root
            source = f"storage_config_default:{DEFAULT_STORAGE_CONFIG.as_posix()}"

    requested = Path(raw).expanduser()
    resolution: dict[str, Any] = {
        "environment_variable": environment_variable,
        "source": source,
        "requested": str(requested),
        "command": VERIFY_COMMAND,
    }
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        resolution["available"] = False
        resolution["reason"] = f"artifact root cannot be resolved: {exc}"
        return None, resolution
    if not resolved.is_dir():
        resolution["available"] = False
        resolution["reason"] = f"artifact root is not a directory: {resolved}"
        return None, resolution
    resolution["available"] = True
    resolution["resolved"] = str(resolved)
    return resolved, resolution


def check_attestation_agreement(
    export_config: Mapping[str, Any],
    manifests: Mapping[int, Mapping[str, Any]],
) -> CheckResult:
    """Check that the T20 attestation and the committed manifests still agree."""

    failures: list[str] = []
    attestation = export_config.get("evidence_attestation")
    if not isinstance(attestation, Mapping):
        raise BaselineParityError("export config carries no evidence_attestation")
    graph_hashes = attestation.get("graph_sha256")
    if not isinstance(graph_hashes, Mapping):
        raise BaselineParityError("evidence_attestation carries no graph_sha256")

    declared = tuple(export_config.get("contexts") or ())
    declared_ids = {f"S{context}" for context in declared}
    attested_ids = set(graph_hashes)
    manifest_ids = {f"S{context}" for context in manifests}
    frozen_ids = {f"S{context}" for context in CONTEXT_VARIANTS}

    for label, actual in (
        ("attestation", attested_ids),
        ("manifest directory", manifest_ids),
        ("frozen T12 contract family", frozen_ids),
    ):
        missing = sorted(declared_ids - actual)
        orphan = sorted(actual - declared_ids)
        if missing:
            failures.append(f"{label} is missing context(s): {', '.join(missing)}")
        if orphan:
            failures.append(f"{label} has orphan context(s): {', '.join(orphan)}")

    source_digest = attestation.get("source_artifact_sha256")
    external_digest = attestation.get("external_data_sha256")
    artifact_directory = export_config.get("artifact_directory")

    for context in sorted(set(declared) & set(manifests)):
        variant = f"S{context}"
        manifest = manifests[context]
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Mapping):
            failures.append(f"{variant}: manifest carries no artifacts block")
            continue
        if manifest.get("task_id") != BASELINE_TASK_ID:
            failures.append(
                f"{variant}: manifest task_id is {manifest.get('task_id')!r}, "
                f"expected {BASELINE_TASK_ID!r}"
            )
        if manifest.get("context_length") != context:
            failures.append(
                f"{variant}: manifest context_length is "
                f"{manifest.get('context_length')!r}"
            )
        if manifest.get("variant_id") != variant:
            failures.append(
                f"{variant}: manifest variant_id is {manifest.get('variant_id')!r}"
            )
        if manifest.get("source_artifact_sha256") != source_digest:
            failures.append(
                f"{variant}: manifest source_artifact_sha256 differs from the "
                "export attestation"
            )
        if artifacts.get("root") != artifact_directory:
            failures.append(
                f"{variant}: manifest artifact root is {artifacts.get('root')!r}, "
                f"expected {artifact_directory!r}"
            )
        attested = graph_hashes.get(variant)
        if not isinstance(attested, Mapping):
            failures.append(f"{variant}: attestation records no graph digests")
            attested = {}
        for kind in GRAPH_KINDS:
            record = artifacts.get(kind)
            if not isinstance(record, Mapping):
                failures.append(f"{variant} {kind}: manifest records no artifact")
                continue
            if record.get("sha256") != attested.get(kind):
                failures.append(
                    f"{variant} {kind}: manifest sha256 {record.get('sha256')!r} "
                    f"differs from attested {attested.get(kind)!r}"
                )
            external = record.get("external_data") or ()
            if not external:
                failures.append(
                    f"{variant} {kind}: manifest records no external data file"
                )
            for entry in external:
                if entry.get("sha256") != external_digest:
                    failures.append(
                        f"{variant} {kind} {entry.get('location')!r}: external data "
                        "sha256 differs from the export attestation"
                    )

    return CheckResult(
        name="attestation_manifest_agreement",
        status="failed" if failures else "passed",
        summary=(
            f"{len(manifests)} committed T20 manifest(s) checked against the "
            "export attestation"
        ),
        failures=_cap(failures),
        detail={
            "export_config": DEFAULT_EXPORT_CONFIG.as_posix(),
            "manifest_directory": DEFAULT_MANIFEST_DIRECTORY.as_posix(),
            "run_id": attestation.get("run_id"),
            "exporter_commit": attestation.get("exporter_commit"),
            "contexts": sorted(declared),
        },
    )


def check_revision_agreement(
    export_config: Mapping[str, Any],
    model_contract: Mapping[str, Any],
    manifests: Mapping[int, Mapping[str, Any]],
) -> CheckResult:
    """Check the model and tokenizer revision across contract, config, manifests."""

    failures: list[str] = []
    model = model_contract.get("model") or {}
    tokenizer = model_contract.get("tokenizer") or {}
    chat_template = (tokenizer.get("chat_template") or {}).get("sha256")

    if model.get("id") != MODEL_ID:
        failures.append(
            f"model contract id is {model.get('id')!r}, expected {MODEL_ID!r}"
        )
    for label, value in (
        ("model contract revision", model.get("revision")),
        ("model contract tokenizer revision", tokenizer.get("revision")),
    ):
        if value != MODEL_REVISION:
            failures.append(
                f"{label} is {value!r}, expected the frozen T12 revision "
                f"{MODEL_REVISION!r}"
            )
    declared_contract = export_config.get("model_contract")
    if declared_contract != DEFAULT_MODEL_CONTRACT.as_posix():
        failures.append(
            f"export config model_contract is {declared_contract!r}, expected "
            f"{DEFAULT_MODEL_CONTRACT.as_posix()!r}"
        )

    for context in sorted(manifests):
        variant = f"S{context}"
        manifest = manifests[context]
        if manifest.get("model_id") != MODEL_ID:
            failures.append(
                f"{variant}: manifest model_id is {manifest.get('model_id')!r}"
            )
        for field_name in ("model_revision", "tokenizer_revision"):
            if manifest.get(field_name) != MODEL_REVISION:
                failures.append(
                    f"{variant}: manifest {field_name} is "
                    f"{manifest.get(field_name)!r}, expected {MODEL_REVISION!r}"
                )
        if manifest.get("chat_template_sha256") != chat_template:
            failures.append(
                f"{variant}: manifest chat_template_sha256 differs from the "
                "T00 model contract"
            )
        contract = manifest.get("contract") or {}
        for kind in GRAPH_KINDS:
            nested = contract.get(kind) or {}
            if nested.get("model_id") != MODEL_ID:
                failures.append(
                    f"{variant} {kind}: embedded contract model_id is "
                    f"{nested.get('model_id')!r}"
                )
            if nested.get("model_revision") != MODEL_REVISION:
                failures.append(
                    f"{variant} {kind}: embedded contract model_revision is "
                    f"{nested.get('model_revision')!r}"
                )

    return CheckResult(
        name="model_revision_agreement",
        status="failed" if failures else "passed",
        summary=(
            "model and tokenizer revision agreement across the T00 contract, "
            "the T20 export config, the T20 manifests, and slm_lab.contracts"
        ),
        failures=_cap(failures),
        detail={
            "model_id": MODEL_ID,
            "expected_revision": MODEL_REVISION,
            "revision_source": "slm_lab.contracts.static_cache.MODEL_REVISION",
            "model_contract": DEFAULT_MODEL_CONTRACT.as_posix(),
        },
    )


def _spec_boundary(specs: Sequence[TensorSpec]) -> tuple[tuple[Any, ...], ...]:
    return tuple((spec.name, spec.dtype, tuple(spec.shape)) for spec in specs)


def _manifest_boundary(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            entry.get("name"),
            entry.get("dtype"),
            tuple(entry.get("shape") or ()),
        )
        for entry in entries
    )


def _boundary_failures(
    label: str,
    expected: Sequence[tuple[Any, ...]],
    actual: Sequence[tuple[Any, ...]],
) -> list[str]:
    failures: list[str] = []
    expected_names = [item[0] for item in expected]
    actual_names = [item[0] for item in actual]
    missing = [name for name in expected_names if name not in set(actual_names)]
    extra = [name for name in actual_names if name not in set(expected_names)]
    if missing:
        failures.append(f"{label}: missing tensor(s) {', '.join(map(str, missing))}")
    if extra:
        failures.append(f"{label}: unexpected tensor(s) {', '.join(map(str, extra))}")
    if not missing and not extra and expected_names != actual_names:
        failures.append(f"{label}: tensor order differs from the T12 contract")
    actual_by_name = {item[0]: item for item in actual}
    for name, dtype, shape in expected:
        found = actual_by_name.get(name)
        if found is None:
            continue
        if found[1] != dtype:
            failures.append(
                f"{label} {name}: dtype {found[1]!r}, contract requires {dtype!r}"
            )
        if found[2] != shape:
            failures.append(
                f"{label} {name}: shape {list(found[2])}, contract requires "
                f"{list(shape)}"
            )
    return failures


def check_contract_conformance(
    manifests: Mapping[int, Mapping[str, Any]],
) -> CheckResult:
    """Check recorded graph boundaries against the frozen T12 tensor contracts."""

    failures: list[str] = []
    builders: tuple[tuple[str, Callable[[int], GraphContract]], ...] = (
        ("prefill", build_prefill_contract),
        ("decode", build_decode_contract),
    )
    checked = 0
    for context in sorted(manifests):
        variant = f"S{context}"
        manifest = manifests[context]
        capacity = CONTEXT_VARIANTS.get(context)
        if capacity is None:
            failures.append(
                f"{variant}: context is not part of the frozen T12 contract family"
            )
            continue
        if manifest.get("cache_capacity") != capacity:
            failures.append(
                f"{variant}: manifest cache_capacity is "
                f"{manifest.get('cache_capacity')!r}, contract requires {capacity}"
            )
        artifacts = manifest.get("artifacts") or {}
        for kind, builder in builders:
            try:
                contract = builder(context)
            except CacheContractError as exc:
                failures.append(f"{variant} {kind}: contract cannot be built: {exc}")
                continue
            record = artifacts.get(kind)
            if not isinstance(record, Mapping):
                failures.append(f"{variant} {kind}: manifest records no artifact")
                continue
            if record.get("graph_kind") != kind:
                failures.append(
                    f"{variant} {kind}: manifest graph_kind is "
                    f"{record.get('graph_kind')!r}"
                )
            failures.extend(
                _boundary_failures(
                    f"{variant} {kind} input",
                    _spec_boundary(contract.inputs),
                    _manifest_boundary(record.get("input_tensors") or ()),
                )
            )
            failures.extend(
                _boundary_failures(
                    f"{variant} {kind} output",
                    _spec_boundary(contract.outputs),
                    _manifest_boundary(record.get("output_tensors") or ()),
                )
            )
            checked += 1

    return CheckResult(
        name="t12_contract_conformance",
        status="failed" if failures else "passed",
        summary=(
            f"{checked} recorded graph boundary set(s) checked against the frozen "
            "T12 prefill/decode contracts"
        ),
        failures=_cap(failures),
        detail={
            "contract_source": "slm_lab.contracts.static_cache",
            "contexts": sorted(manifests),
            "compared": ["tensor names", "tensor order", "dtypes", "static shapes"],
        },
    )


def _recorded_size_bytes(record: Mapping[str, Any], label: str) -> int:
    """Return one recorded ``size_bytes``, failing closed on anything else.

    A manifest that carries ``null``, a string, or no size at all is a manifest
    this gate cannot check against, so it raises :class:`BaselineParityError`
    rather than letting ``int()`` escape as an uncaught ``TypeError``.
    """

    value = record.get("size_bytes")
    if isinstance(value, bool) or not isinstance(value, int):
        raise BaselineParityError(
            f"{label}: manifest size_bytes must be an integer, found {value!r}"
        )
    return value


def expected_artifact_files(
    manifests: Mapping[int, Mapping[str, Any]],
) -> list[tuple[str, str, int]]:
    """Return every ``(relative_path, sha256, size_bytes)`` T20 recorded."""

    expected: dict[str, tuple[str, int]] = {}
    for context in sorted(manifests):
        variant = f"S{context}"
        artifacts = manifests[context].get("artifacts") or {}
        for kind in GRAPH_KINDS:
            record = artifacts.get(kind)
            if not isinstance(record, Mapping):
                raise BaselineParityError(
                    f"{variant} {kind}: manifest records no artifact"
                )
            relative = str(record.get("relative_path") or "")
            if not relative:
                raise BaselineParityError(
                    f"{variant} {kind}: manifest records no relative_path"
                )
            entries: list[tuple[str, str, int]] = [
                (
                    relative,
                    str(record.get("sha256") or ""),
                    _recorded_size_bytes(record, f"{variant} {kind} {relative}"),
                )
            ]
            parent = PurePosixPath(relative).parent
            for external in record.get("external_data") or ():
                location = str(external.get("location") or "")
                if not location:
                    raise BaselineParityError(
                        f"{variant} {kind}: external data entry has no location"
                    )
                entries.append(
                    (
                        str(parent / location),
                        str(external.get("sha256") or ""),
                        _recorded_size_bytes(
                            external,
                            f"{variant} {kind} external data {location}",
                        ),
                    )
                )
            for path, digest, size in entries:
                if (
                    PurePosixPath(path).is_absolute()
                    or ".." in PurePosixPath(path).parts
                ):
                    raise BaselineParityError(f"unsafe recorded artifact path: {path}")
                previous = expected.get(path)
                if previous is not None and previous != (digest, size):
                    raise BaselineParityError(
                        f"{path}: committed manifests record conflicting digests"
                    )
                expected[path] = (digest, size)
    return [(path, digest, size) for path, (digest, size) in sorted(expected.items())]


def check_artifact_bytes(
    manifests: Mapping[int, Mapping[str, Any]],
    *,
    artifact_directory: Path | None,
    unavailable_reason: str | None = None,
    enabled: bool = True,
    progress: Callable[[str], None] | None = None,
) -> CheckResult:
    """Re-hash the on-disk T20 graphs and external data against their records.

    A missing artifact root or a missing file yields ``unavailable``, naming
    exactly what could not be checked. It never yields ``passed``.
    """

    expected = expected_artifact_files(manifests)
    expected_bytes = sum(size for _, _, size in expected if size > 0)
    base_detail: dict[str, Any] = {
        "file_count": len(expected),
        "recorded_total_bytes": expected_bytes,
        "command": VERIFY_COMMAND,
    }

    if not enabled:
        return CheckResult(
            name="artifact_byte_identity",
            status="skipped",
            summary=(
                f"byte verification was disabled; {len(expected)} recorded file(s) "
                "were not re-hashed"
            ),
            detail={**base_detail, "measurement": "not_measured"},
        )

    if artifact_directory is None:
        return CheckResult(
            name="artifact_byte_identity",
            status="unavailable",
            summary=(
                "the external artifact root is unavailable; no recorded digest was "
                "re-measured and no parity claim is licensed"
            ),
            failures=_cap([f"not checked: {path}" for path, _, _ in expected]),
            detail={
                **base_detail,
                "measurement": "not_measured",
                "reason": unavailable_reason or "artifact root is unavailable",
            },
        )

    digests: list[FileDigest] = []
    missing: list[str] = []
    failures: list[str] = []
    for relative, recorded_sha256, recorded_size in expected:
        path = artifact_directory.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file():
            missing.append(relative)
            continue
        if progress is not None:
            progress(f"hashing {relative}")
        measured_size = path.stat().st_size
        digest = FileDigest(
            relative_path=relative,
            recorded_sha256=recorded_sha256,
            measured_sha256=_sha256(path),
            recorded_size_bytes=recorded_size,
            measured_size_bytes=measured_size,
        )
        digests.append(digest)
        if not digest.matched:
            failures.append(
                f"{relative}: recorded {recorded_sha256} at {recorded_size} bytes, "
                f"measured {digest.measured_sha256} at {measured_size} bytes"
            )

    if failures:
        status = "failed"
        summary = (
            f"{len(failures)} of {len(digests)} re-hashed file(s) no longer match "
            "the committed T20 record"
        )
    elif missing:
        status = "unavailable"
        summary = (
            f"{len(missing)} recorded file(s) are absent from the artifact root; "
            f"{len(digests)} file(s) matched"
        )
    else:
        status = "passed"
        summary = (
            f"{len(digests)} recorded file(s) re-hashed and matched the committed "
            "T20 digests and sizes"
        )

    return CheckResult(
        name="artifact_byte_identity",
        status=status,
        summary=summary,
        failures=_cap([*failures, *(f"not checked: {path}" for path in missing)]),
        detail={
            **base_detail,
            "measurement": "recomputed_sha256",
            "artifact_directory": str(artifact_directory),
            "measured_file_count": len(digests),
            "measured_total_bytes": sum(
                digest.measured_size_bytes for digest in digests
            ),
            "missing": missing,
            "files": [digest.as_dict() for digest in digests],
        },
    )


def numerical_parity_requirement() -> dict[str, Any]:
    """Return the declared, unexecuted numerical half of baseline parity.

    Nothing in this record is a measurement. It exists so a caller can see that
    the numerical half was never run here, who owns it, and what would run it.
    """

    return {
        "status": "not_run",
        "measurement": "declared_requirement_not_executed",
        "requirement": (
            "Logit-level agreement between the T11 deterministic PyTorch "
            "reference and the T20 float16 ONNX export, for the prefill and "
            "decode graphs at every frozen context."
        ),
        "blocked_by": {
            "reason": (
                "torch and onnxruntime are absent from the primary macOS project "
                "environment and T40 installs no heavy dependency."
            ),
            "missing_dependencies": ["torch", "onnxruntime"],
            "host_class": "primary macOS development host",
        },
        "owner_task": "T21",
        "owner_scope": (
            "T21 owns ONNX Runtime CPU parity under src/slm_lab/backends/ and "
            "tests/onnx/; T40 neither implements nor duplicates it."
        ),
        "consumer_tasks": ["T41", "T42", "T43"],
        "tolerance_policy": {
            "same_model_reference_source": (
                "src/slm_lab/generation/reference.py::DEFAULT_TOLERANCE"
            ),
            "export_parity_policy_status": "not_frozen_at_this_commit",
            "requirement_source": "docs/project/plan.md",
            "note": (
                "DEFAULT_TOLERANCE bounds full-forward versus cached decode drift "
                "for one model, dtype, and device. A distinct export-parity "
                "tolerance must be frozen by T21 before any PyTorch-versus-ONNX "
                "pass or fail claim is made."
            ),
        },
        "commands": [
            {
                "id": "pytorch_reference_logits",
                "status": "implemented_at_this_commit",
                "establishes": "the T11 PyTorch oracle only, not ONNX agreement",
                "command": (
                    "HF_HOME=<local-cache> TRANSFORMERS_OFFLINE=1 PYTHONPATH=src "
                    "<python-with-torch-2.7.1-and-transformers-4.51.3> "
                    "-m slm_lab.generation.reference --fixture raw_ascii "
                    "--device cpu --dtype float32 --max-new-tokens 8"
                ),
            },
            {
                "id": "onnx_runtime_parity",
                "status": "not_implemented_at_this_commit",
                "establishes": (
                    "the missing half: PyTorch-versus-ONNX logit agreement under "
                    "a frozen export-parity tolerance"
                ),
                "command": (
                    "SLM_LAB_ARTIFACT_ROOT=<external-root> HF_HOME=<local-cache> "
                    "TRANSFORMERS_OFFLINE=1 PYTHONPATH=src "
                    "<python-with-torch-and-onnxruntime> -m pytest -q tests/onnx"
                ),
                "delivered_by": "T21",
            },
        ],
    }


def _git_commit(repo_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"commit": None, "tree_clean": None}
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return result
    result["commit"] = commit.stdout.strip() or None
    result["tree_clean"] = not status.stdout.strip()
    return result


def overall_verdict(identity_verdict: str, numerical_status: str) -> str:
    """Compose the two parity halves into the record's overall verdict.

    Split out of :func:`check_baseline_parity` on purpose. The guarantee this
    mapping carries — that the overall verdict can never read ``verified``
    while the numerical half is ``not_run`` — is otherwise only reachable when
    the external nine-gigabyte artifact volume is mounted, which made it
    untestable in the default offline suite. As a pure function it is testable
    over every input in microseconds.

    No branch returns ``verified``. Half two is a declared requirement, never a
    measurement, so at this commit no input produces a fully verified parity
    run — a fully ``verified`` identity half included. An unrecognized
    numerical status fails closed rather than defaulting to ``partial``:
    closing half two is T21 work that must extend this mapping deliberately.
    """

    if identity_verdict == "failed":
        return "failed"
    if identity_verdict == "unavailable":
        return "unavailable"
    if numerical_status == "not_run":
        return "partial"
    raise BaselineParityError(
        f"numerical parity reported an unhandled status {numerical_status!r}; "
        "extend overall_verdict before a record can carry it"
    )


def check_baseline_parity(
    repo_root: Path | str,
    *,
    artifact_root: Path | None = None,
    verify_artifact_bytes: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the pre-quantization baseline parity gate and return its report.

    Half one runs here. Half two is recorded as ``not_run``, so the overall
    verdict is at best ``partial`` until a host with the heavy extras closes it.
    """

    root = Path(repo_root).resolve()
    export_config = _load_json(root / DEFAULT_EXPORT_CONFIG)
    model_contract = _load_json(root / DEFAULT_MODEL_CONTRACT)
    manifests = load_manifests(root)

    artifact_directory: Path | None
    resolved_root, resolution = resolve_artifact_root(root, override=artifact_root)
    if resolved_root is None:
        artifact_directory = None
    else:
        subdirectory = artifact_subdirectory(
            export_config,
            str(resolution["environment_variable"]),
        )
        artifact_directory = resolved_root / subdirectory
        resolution["artifact_directory"] = str(artifact_directory)
        if not artifact_directory.is_dir():
            resolution["available"] = False
            resolution["reason"] = (
                f"T20 artifact directory is missing: {artifact_directory}"
            )
            artifact_directory = None

    checks = [
        check_attestation_agreement(export_config, manifests),
        check_revision_agreement(export_config, model_contract, manifests),
        check_contract_conformance(manifests),
        check_artifact_bytes(
            manifests,
            artifact_directory=artifact_directory,
            unavailable_reason=resolution.get("reason"),
            enabled=verify_artifact_bytes,
            progress=progress,
        ),
    ]

    statuses = {check.status for check in checks}
    if "failed" in statuses:
        identity_verdict = "failed"
    elif "unavailable" in statuses:
        identity_verdict = "unavailable"
    elif "skipped" in statuses:
        identity_verdict = "partial"
    else:
        identity_verdict = "verified"

    numerical = numerical_parity_requirement()
    verdict = overall_verdict(identity_verdict, str(numerical["status"]))

    repository = _git_commit(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "record_type": RECORD_TYPE,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository": {
            # Deliberately no *checkout* path: a record is committed evidence
            # read on other machines, and an absolute checkout path is both
            # meaningless there and a private detail. The commit identifies the
            # tree; ``git_tree_clean`` says whether it was the commit exactly.
            # The record as a whole is not path-free: ``artifact_root`` below
            # carries absolute ``requested``, ``resolved``, and
            # ``artifact_directory`` values on purpose. That root is a
            # published, already-public location (see
            # ``configs/storage/external-ssd.example.yaml``), and *which* root
            # was measured is part of what the record attests.
            "git_commit": repository["commit"],
            "git_tree_clean": repository["tree_clean"],
        },
        "baseline": {
            "task_id": BASELINE_TASK_ID,
            "export_config": DEFAULT_EXPORT_CONFIG.as_posix(),
            "model_contract": DEFAULT_MODEL_CONTRACT.as_posix(),
            "manifest_directory": DEFAULT_MANIFEST_DIRECTORY.as_posix(),
            "contexts": sorted(manifests),
        },
        "artifact_root": resolution,
        "verdict": verdict,
        "verdict_scope": "artifact_identity_only",
        # Scope-explicit on purpose. An unqualified "released_for_calibration:
        # true" beside "verdict: partial" is exactly the partial-result-read-as-
        # complete trap, and this is the field T41 keys on.
        "released_for_calibration_on_artifact_identity": (
            identity_verdict == "verified"
        ),
        "artifact_identity": {
            "verdict": identity_verdict,
            "checks": [check.as_dict() for check in checks],
        },
        "numerical_parity": numerical,
        "claim_boundary": {
            "establishes": list(ESTABLISHES) if identity_verdict == "verified" else [],
            "does_not_establish": list(DOES_NOT_ESTABLISH),
        },
        "commands": {"verify": VERIFY_COMMAND, "record": RECORD_COMMAND},
    }


def default_evidence_path(repo_root: Path, report: Mapping[str, Any]) -> Path:
    """Return the conventional evidence path for one parity run."""

    day = str(report["generated_at"])[:10]
    return repo_root / DEFAULT_EVIDENCE_DIRECTORY / f"t40-baseline-parity-{day}.json"


def write_evidence(path: Path, report: Mapping[str, Any]) -> None:
    """Write one compact, committed parity evidence record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def format_report(report: Mapping[str, Any]) -> str:
    """Render a short, unambiguous human summary of one parity run."""

    identity = report["artifact_identity"]
    lines = [
        f"baseline parity verdict: {report['verdict']} "
        f"(scope: {report['verdict_scope']})",
        f"  artifact identity: {identity['verdict']}",
    ]
    for check in identity["checks"]:
        lines.append(f"    [{check['status']}] {check['name']}: {check['summary']}")
        for failure in check.get("failures", ())[:MAX_REPORTED_FAILURES]:
            lines.append(f"      - {failure}")
    numerical = report["numerical_parity"]
    lines.append(
        f"  numerical parity: {numerical['status']} (owner {numerical['owner_task']})"
    )
    lines.append(
        "  released for calibration (artifact identity only): "
        f"{report['released_for_calibration_on_artifact_identity']}"
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``slm_lab.quantization.parity`` command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("verify", "fail closed unless artifact identity fully verifies"),
        ("record", "verify and write the committed evidence record"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("--artifact-root", type=Path)
        child.add_argument(
            "--no-verify-artifact-bytes",
            action="store_true",
            help="skip re-hashing the external artifacts (never a pass)",
        )
        child.add_argument("--json", action="store_true")
        child.add_argument("--quiet", action="store_true")
    subparsers.choices["record"].add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the parity gate; return zero only when artifact identity verifies."""

    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    progress = None
    if not args.quiet and not args.json:

        def progress(message: str) -> None:
            print(message, file=sys.stderr)

    try:
        report = check_baseline_parity(
            repo_root,
            artifact_root=args.artifact_root,
            verify_artifact_bytes=not args.no_verify_artifact_bytes,
            progress=progress,
        )
        if args.command == "record":
            output = args.output or default_evidence_path(repo_root, report)
            write_evidence(output, report)
    except (BaselineParityError, OSError) as exc:
        print(f"baseline parity failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    elif not args.quiet:
        print(format_report(report))
        if args.command == "record":
            print(f"  evidence written: {output}")
    return 0 if report["artifact_identity"]["verdict"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
