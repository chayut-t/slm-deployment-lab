"""Export the frozen Qwen3-0.6B prefill/decode matrix to static ONNX.

The graph protobufs and external tensor data are intentionally written beneath
``SLM_LAB_ARTIFACT_ROOT``. Only their reproducible manifests belong in Git.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from slm_lab.contracts import (
    CONTEXT_VARIANTS,
    GraphContract,
    TensorSpec,
    build_decode_contract,
    build_prefill_contract,
)
from slm_lab.contracts.static_cache import (
    BATCH_SIZE,
    HEAD_DIM,
    NUM_KEY_VALUE_HEADS,
    VOCAB_SIZE,
)
from slm_lab.evaluation.fixtures import (
    FixtureValidationError,
    canonical_json_sha256,
    validate_documents,
)
from slm_lab.manifests.validation import validate_manifest
from slm_lab.models import load_model_contract, load_reference_model


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/models/qwen3-0.6b-onnx-export.json"
DEFAULT_CONFIG_SPELLING = DEFAULT_CONFIG_PATH.as_posix()
DEFAULT_TOKEN_FIXTURE_PATH = (
    PROJECT_ROOT / "tests/fixtures/t10/token-fixtures-v1.json"
)
DEFAULT_T10_CONTRACT_PATH = (
    PROJECT_ROOT / "configs/workloads/t10-token-fixtures.json"
)
DEFAULT_T10_SOURCE_PATH = (
    PROJECT_ROOT / "tests/fixtures/t10/source-prompts-v1.json"
)
DEFAULT_HOST_MANIFEST_PATH = PROJECT_ROOT / "results/hosts/apple-m4-primary.json"
DEFAULT_MANIFEST_DIRECTORY = PROJECT_ROOT / "results/manifests/onnx"
ARTIFACT_SUBDIRECTORY = Path("onnx/reference/T20")
TASK_ID = "T20"
EXPORTER_SOURCE_PATH = PROJECT_ROOT / "src/slm_lab/export/onnx_matrix.py"
FROZEN_T10_BUNDLE_CANONICAL_SHA256 = (
    "9f9268ae4a366faa4325271492ec52f035bbf3ba0973d2de61f63382e6302745"
)
FROZEN_EXPORT_CONFIG_SHA256 = (
    "be885020992520443d11b883d890d1ceeac424648107007ef8332f37f629d147"
)


class ExportConfigurationError(ValueError):
    """The requested export or an exported artifact violates T20 policy."""


@dataclass(frozen=True)
class ExportAttestation:
    """Independent, committed identity of the completed export run."""

    run_id: str
    exporter_commit: str
    runtime_python_version: str
    source_artifact_sha256: str
    external_data_sha256: str
    graph_hashes: tuple[tuple[int, str, str], ...]

    def graph_sha256(self, prompt_length: int, graph_kind: str) -> str:
        if graph_kind not in {"prefill", "decode"}:
            raise ExportConfigurationError(
                f"attestation has no graph kind {graph_kind!r}"
            )
        match = next(
            (
                (prefill, decode)
                for context, prefill, decode in self.graph_hashes
                if context == prompt_length
            ),
            None,
        )
        if match is None:
            raise ExportConfigurationError(
                f"attestation has no S{prompt_length} graph hashes"
            )
        return match[0] if graph_kind == "prefill" else match[1]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "exporter_commit": self.exporter_commit,
            "runtime_python_version": self.runtime_python_version,
            "source_artifact_sha256": self.source_artifact_sha256,
            "external_data_sha256": self.external_data_sha256,
            "graph_sha256": {
                f"S{context}": {
                    "prefill": prefill,
                    "decode": decode,
                }
                for context, prefill, decode in self.graph_hashes
            },
        }


@dataclass(frozen=True)
class ExportConfig:
    """Exact, declarative settings for the reference ONNX matrix."""

    opset: int
    precision: str
    exporter: str
    torch_version: str
    transformers_version: str
    onnx_version: str
    external_data_threshold_bytes: int
    contexts: tuple[int, ...]
    seed: int
    source_path: Path
    model_contract_path: Path
    token_fixture: TokenFixtureBundle
    evidence_attestation: ExportAttestation
    trusted_config_sha256: str


@dataclass(frozen=True)
class TokenWorkload:
    """One hash-validated frozen T10 context workload."""

    fixture_id: str
    context_length: int
    generated_tokens: int
    prompt_sha256: str
    token_ids_sha256: str
    token_ids: tuple[int, ...]

    def provenance(self) -> dict[str, Any]:
        return {
            "id": self.fixture_id,
            "context_length": self.context_length,
            "generated_tokens": self.generated_tokens,
            "prompt_sha256": self.prompt_sha256,
            "token_ids_sha256": self.token_ids_sha256,
            "token_count": len(self.token_ids),
        }


@dataclass(frozen=True)
class TokenFixtureBundle:
    """Validated content identity and workloads for the configured T10 bundle."""

    source_path: Path
    configured_path: str
    canonical_json_sha256: str
    file_sha256: str
    workloads: tuple[TokenWorkload, ...]

    def workload(self, prompt_length: int) -> TokenWorkload:
        match = next(
            (
                workload
                for workload in self.workloads
                if workload.context_length == prompt_length
            ),
            None,
        )
        if match is None:
            raise ExportConfigurationError(
                f"token fixture has no exact S{prompt_length} workload"
            )
        return match


@dataclass(frozen=True)
class ExternalDataRecord:
    """One external tensor-data file referenced by an ONNX protobuf."""

    location: str
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "location": self.location,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class OnnxArtifactRecord:
    """Validated content identity for one ONNX graph and its data files."""

    graph_kind: str
    relative_path: str
    sha256: str
    size_bytes: int
    external_data: tuple[ExternalDataRecord, ...]
    input_tensors: tuple[dict[str, Any], ...]
    output_tensors: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "graph_kind": self.graph_kind,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "external_data": [record.as_dict() for record in self.external_data],
            "input_tensors": list(self.input_tensors),
            "output_tensors": list(self.output_tensors),
        }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ExportConfigurationError(
            f"exact package {name!r} is required for T20 export"
        ) from exc


def _require_exact_version(actual: str, expected: str, package: str) -> None:
    if actual != expected:
        raise ExportConfigurationError(
            f"{package} version mismatch: expected {expected}, found {actual}"
        )


def _load_json_document(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportConfigurationError(f"invalid {label} {path}: {exc}") from exc


def _resolve_configured_path(value: Any, label: str) -> tuple[Path, str]:
    if not isinstance(value, str) or not value:
        raise ExportConfigurationError(f"{label} must be a non-empty path")
    configured = Path(value)
    path = configured if configured.is_absolute() else PROJECT_ROOT / configured
    path = path.resolve()
    if not path.is_file():
        raise ExportConfigurationError(f"{label} does not exist: {value}")
    return path, value


def _load_token_fixture(path: Path, configured_path: str) -> TokenFixtureBundle:
    bundle = _load_json_document(path, "T10 token fixture")
    try:
        context_records = bundle["context_workloads"]
    except (KeyError, TypeError) as exc:
        raise ExportConfigurationError(
            "configured T10 token fixture lacks context workloads"
        ) from exc
    for record in context_records:
        fixture_id = record.get("id", "<unknown>")
        token_ids = record.get("token_ids")
        if not isinstance(token_ids, list) or not all(
            isinstance(token_id, int) and token_id >= 0
            for token_id in token_ids
        ):
            raise ExportConfigurationError(
                f"{fixture_id}: token IDs must be nonnegative integers"
            )
        if record.get("token_ids_sha256") != canonical_json_sha256(token_ids):
            raise ExportConfigurationError(
                f"{fixture_id}: token ID hash drift"
            )
        prompt = record.get("prompt")
        if not isinstance(prompt, str) or record.get(
            "prompt_sha256"
        ) != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
            raise ExportConfigurationError(f"{fixture_id}: prompt hash drift")

    source = _load_json_document(DEFAULT_T10_SOURCE_PATH, "T10 source fixture")
    t10_contract = _load_json_document(
        DEFAULT_T10_CONTRACT_PATH,
        "T10 workload contract",
    )
    model_contract = _load_json_document(
        DEFAULT_CONFIG_PATH.with_name("qwen3-0.6b.yaml"),
        "Qwen model contract",
    )
    try:
        validate_documents(
            source=source,
            bundle=bundle,
            config=t10_contract,
            model_contract=model_contract,
        )
    except (FixtureValidationError, KeyError, TypeError) as exc:
        raise ExportConfigurationError(
            f"configured T10 token fixture is invalid: {exc}"
        ) from exc

    canonical_digest = canonical_json_sha256(bundle)
    if canonical_digest != FROZEN_T10_BUNDLE_CANONICAL_SHA256:
        raise ExportConfigurationError(
            "configured T10 token fixture differs from the frozen canonical "
            f"digest {FROZEN_T10_BUNDLE_CANONICAL_SHA256}"
        )

    workloads = tuple(
        TokenWorkload(
            fixture_id=record["id"],
            context_length=record["context_length"],
            generated_tokens=record["generated_tokens"],
            prompt_sha256=record["prompt_sha256"],
            token_ids_sha256=record["token_ids_sha256"],
            token_ids=tuple(record["token_ids"]),
        )
        for record in context_records
    )
    if tuple(workload.context_length for workload in workloads) != tuple(
        CONTEXT_VARIANTS
    ):
        raise ExportConfigurationError(
            "configured token fixture does not cover the frozen context matrix"
        )
    return TokenFixtureBundle(
        source_path=path,
        configured_path=configured_path,
        canonical_json_sha256=canonical_digest,
        file_sha256=_sha256(path),
        workloads=workloads,
    )


def _load_export_attestation(value: Any) -> ExportAttestation:
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ExportConfigurationError(
            "evidence_attestation must be a schema-version 1 mapping"
        )
    expected_keys = {
        "schema_version",
        "run_id",
        "exporter_commit",
        "runtime_python_version",
        "source_artifact_sha256",
        "external_data_sha256",
        "graph_sha256",
    }
    if set(value) != expected_keys:
        raise ExportConfigurationError(
            "evidence_attestation fields differ from the frozen contract"
        )
    run_id = value["run_id"]
    exporter_commit = value["exporter_commit"]
    python_version = value["runtime_python_version"]
    if not isinstance(run_id, str) or not re.fullmatch(
        r"T20-[A-Za-z0-9._-]+",
        run_id,
    ):
        raise ExportConfigurationError("invalid T20 evidence run_id")
    if not isinstance(exporter_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}",
        exporter_commit,
    ):
        raise ExportConfigurationError(
            "attested exporter_commit must be a full lowercase Git SHA"
        )
    if not isinstance(python_version, str) or not re.fullmatch(
        r"[0-9]+\.[0-9]+\.[0-9]+",
        python_version,
    ):
        raise ExportConfigurationError(
            "attested runtime_python_version must be exact"
        )
    for field in ("source_artifact_sha256", "external_data_sha256"):
        digest = value[field]
        if not isinstance(digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}",
            digest,
        ):
            raise ExportConfigurationError(
                f"attestation {field} must be a SHA-256 digest"
            )

    graph_document = value["graph_sha256"]
    if not isinstance(graph_document, Mapping) or set(graph_document) != {
        f"S{context}" for context in CONTEXT_VARIANTS
    }:
        raise ExportConfigurationError(
            "attestation graph hashes must cover the exact context matrix"
        )
    graph_hashes: list[tuple[int, str, str]] = []
    for context in CONTEXT_VARIANTS:
        pair = graph_document[f"S{context}"]
        if not isinstance(pair, Mapping) or set(pair) != {"prefill", "decode"}:
            raise ExportConfigurationError(
                f"S{context} attestation must identify prefill and decode"
            )
        for kind in ("prefill", "decode"):
            digest = pair[kind]
            if not isinstance(digest, str) or not re.fullmatch(
                r"[0-9a-f]{64}",
                digest,
            ):
                raise ExportConfigurationError(
                    f"S{context} {kind} attestation is not SHA-256"
                )
        graph_hashes.append((context, pair["prefill"], pair["decode"]))
    return ExportAttestation(
        run_id=run_id,
        exporter_commit=exporter_commit,
        runtime_python_version=python_version,
        source_artifact_sha256=value["source_artifact_sha256"],
        external_data_sha256=value["external_data_sha256"],
        graph_hashes=tuple(graph_hashes),
    )


def _trusted_export_config_bytes(path: str) -> tuple[Path, bytes]:
    """Return the one fixed config only when disk, HEAD, and code agree."""

    if type(path) is not str:
        raise ExportConfigurationError(
            "T20 export config path must be the exact absolute string "
            f"{DEFAULT_CONFIG_SPELLING}"
        )
    if path != DEFAULT_CONFIG_SPELLING:
        raise ExportConfigurationError(
            "T20 accepts only the exact absolute export config spelling "
            f"{DEFAULT_CONFIG_SPELLING}, found {path}"
        )
    source = Path(path)
    expected_source = DEFAULT_CONFIG_PATH
    if source != expected_source:
        raise ExportConfigurationError("exact config spelling resolved unexpectedly")
    if source.is_symlink():
        raise ExportConfigurationError(
            f"T20 export config must not be a symlink: {source}"
        )
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ExportConfigurationError(f"invalid export config {source}: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != FROZEN_EXPORT_CONFIG_SHA256:
        raise ExportConfigurationError(
            "tracked T20 export config differs from its code-pinned SHA-256: "
            f"expected {FROZEN_EXPORT_CONFIG_SHA256}, found {digest}"
        )
    relative = source.relative_to(PROJECT_ROOT).as_posix()
    try:
        committed = subprocess.check_output(
            ["git", "show", f"HEAD:{relative}"],
            cwd=PROJECT_ROOT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExportConfigurationError(
            f"cannot read committed T20 export config HEAD:{relative}: {exc}"
        ) from exc
    if raw != committed:
        raise ExportConfigurationError(
            f"tracked T20 export config must exactly match HEAD:{relative}"
        )
    return source, raw


def load_export_config(path: str = DEFAULT_CONFIG_SPELLING) -> ExportConfig:
    """Load the immutable, committed export configuration."""

    source, raw = _trusted_export_config_bytes(path)
    try:
        payload = json.loads(raw)
        export = payload["export"]
        packages = payload["packages"]
        contexts = tuple(payload["contexts"])
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ExportConfigurationError(f"invalid export config {source}: {exc}") from exc

    if payload.get("schema_version") != 1 or payload.get("task_id") != TASK_ID:
        raise ExportConfigurationError("export config identity must be schema 1 / T20")
    model_contract_path, _ = _resolve_configured_path(
        payload.get("model_contract"),
        "model_contract",
    )
    if model_contract_path != DEFAULT_CONFIG_PATH.with_name(
        "qwen3-0.6b.yaml"
    ).resolve():
        raise ExportConfigurationError(
            "export must use the frozen Qwen3-0.6B model contract"
        )
    token_fixture_path, configured_token_fixture = _resolve_configured_path(
        payload.get("token_fixture"),
        "token_fixture",
    )
    token_fixture = _load_token_fixture(
        token_fixture_path,
        configured_token_fixture,
    )
    evidence_attestation = _load_export_attestation(
        payload.get("evidence_attestation")
    )
    if contexts != tuple(CONTEXT_VARIANTS):
        raise ExportConfigurationError(
            f"contexts must exactly match {tuple(CONTEXT_VARIANTS)}, found {contexts}"
        )
    if export.get("precision") != "float16":
        raise ExportConfigurationError("T12 requires float16 cache export boundaries")
    if export.get("exporter") != "torch.onnx.export":
        raise ExportConfigurationError("unsupported exporter")
    opset = export.get("opset")
    threshold = export.get("external_data_threshold_bytes")
    seed = export.get("seed")
    if not isinstance(opset, int) or opset < 18:
        raise ExportConfigurationError("opset must be an integer >= 18")
    if not isinstance(threshold, int) or threshold < 0:
        raise ExportConfigurationError(
            "external_data_threshold_bytes must be non-negative"
        )
    if not isinstance(seed, int) or seed < 0:
        raise ExportConfigurationError("seed must be a non-negative integer")
    for package in ("torch", "transformers", "onnx"):
        value = packages.get(package)
        if not isinstance(value, str) or not value:
            raise ExportConfigurationError(f"missing exact {package} version")
    return ExportConfig(
        opset=opset,
        precision=export["precision"],
        exporter=export["exporter"],
        torch_version=packages["torch"],
        transformers_version=packages["transformers"],
        onnx_version=packages["onnx"],
        external_data_threshold_bytes=threshold,
        contexts=contexts,
        seed=seed,
        source_path=source.resolve(),
        model_contract_path=model_contract_path,
        token_fixture=token_fixture,
        evidence_attestation=evidence_attestation,
        trusted_config_sha256=FROZEN_EXPORT_CONFIG_SHA256,
    )


def _require_exact_type(value: Any, expected: type, field: str) -> None:
    if type(value) is not expected:
        raise ExportConfigurationError(
            "in-memory T20 export config has non-exact type at "
            f"{field}: expected {expected.__name__}, found {type(value).__name__}"
        )


def _config_primitive(config: ExportConfig) -> dict[str, Any]:
    """Normalize an exactly typed config into JSON-safe primitive values."""

    _require_exact_type(config, ExportConfig, "config")
    for field in (
        "opset",
        "external_data_threshold_bytes",
        "seed",
    ):
        _require_exact_type(getattr(config, field), int, field)
    for field in (
        "precision",
        "exporter",
        "torch_version",
        "transformers_version",
        "onnx_version",
        "trusted_config_sha256",
    ):
        _require_exact_type(getattr(config, field), str, field)
    path_type = type(DEFAULT_CONFIG_PATH)
    _require_exact_type(config.source_path, path_type, "source_path")
    _require_exact_type(config.model_contract_path, path_type, "model_contract_path")
    _require_exact_type(config.contexts, tuple, "contexts")
    for index, context in enumerate(config.contexts):
        _require_exact_type(context, int, f"contexts[{index}]")

    fixture = config.token_fixture
    _require_exact_type(fixture, TokenFixtureBundle, "token_fixture")
    _require_exact_type(fixture.source_path, path_type, "token_fixture.source_path")
    for field in (
        "configured_path",
        "canonical_json_sha256",
        "file_sha256",
    ):
        _require_exact_type(
            getattr(fixture, field),
            str,
            f"token_fixture.{field}",
        )
    _require_exact_type(fixture.workloads, tuple, "token_fixture.workloads")
    workloads: list[dict[str, Any]] = []
    for index, workload in enumerate(fixture.workloads):
        prefix = f"token_fixture.workloads[{index}]"
        _require_exact_type(workload, TokenWorkload, prefix)
        for field in ("context_length", "generated_tokens"):
            _require_exact_type(getattr(workload, field), int, f"{prefix}.{field}")
        for field in (
            "fixture_id",
            "prompt_sha256",
            "token_ids_sha256",
        ):
            _require_exact_type(getattr(workload, field), str, f"{prefix}.{field}")
        _require_exact_type(workload.token_ids, tuple, f"{prefix}.token_ids")
        for token_index, token_id in enumerate(workload.token_ids):
            _require_exact_type(
                token_id,
                int,
                f"{prefix}.token_ids[{token_index}]",
            )
        workloads.append(
            {
                "fixture_id": workload.fixture_id,
                "context_length": workload.context_length,
                "generated_tokens": workload.generated_tokens,
                "prompt_sha256": workload.prompt_sha256,
                "token_ids_sha256": workload.token_ids_sha256,
                "token_ids": list(workload.token_ids),
            }
        )

    attestation = config.evidence_attestation
    _require_exact_type(
        attestation,
        ExportAttestation,
        "evidence_attestation",
    )
    for field in (
        "run_id",
        "exporter_commit",
        "runtime_python_version",
        "source_artifact_sha256",
        "external_data_sha256",
    ):
        _require_exact_type(
            getattr(attestation, field),
            str,
            f"evidence_attestation.{field}",
        )
    _require_exact_type(
        attestation.graph_hashes,
        tuple,
        "evidence_attestation.graph_hashes",
    )
    graph_hashes: list[list[int | str]] = []
    for index, graph_hash in enumerate(attestation.graph_hashes):
        prefix = f"evidence_attestation.graph_hashes[{index}]"
        _require_exact_type(graph_hash, tuple, prefix)
        if len(graph_hash) != 3:
            raise ExportConfigurationError(
                f"in-memory T20 export config has invalid length at {prefix}"
            )
        context, prefill, decode = graph_hash
        _require_exact_type(context, int, f"{prefix}[0]")
        _require_exact_type(prefill, str, f"{prefix}[1]")
        _require_exact_type(decode, str, f"{prefix}[2]")
        graph_hashes.append([context, prefill, decode])

    return {
        "opset": config.opset,
        "precision": config.precision,
        "exporter": config.exporter,
        "torch_version": config.torch_version,
        "transformers_version": config.transformers_version,
        "onnx_version": config.onnx_version,
        "external_data_threshold_bytes": config.external_data_threshold_bytes,
        "contexts": list(config.contexts),
        "seed": config.seed,
        "source_path": config.source_path.as_posix(),
        "model_contract_path": config.model_contract_path.as_posix(),
        "token_fixture": {
            "source_path": fixture.source_path.as_posix(),
            "configured_path": fixture.configured_path,
            "canonical_json_sha256": fixture.canonical_json_sha256,
            "file_sha256": fixture.file_sha256,
            "workloads": workloads,
        },
        "evidence_attestation": {
            "run_id": attestation.run_id,
            "exporter_commit": attestation.exporter_commit,
            "runtime_python_version": attestation.runtime_python_version,
            "source_artifact_sha256": attestation.source_artifact_sha256,
            "external_data_sha256": attestation.external_data_sha256,
            "graph_hashes": graph_hashes,
        },
        "trusted_config_sha256": config.trusted_config_sha256,
    }


def _canonical_config_bytes(config: ExportConfig) -> bytes:
    return _canonical_json_bytes(_config_primitive(config), "export config")


def _exact_json_primitive(value: Any, path: str) -> Any:
    """Copy an exact-builtin JSON tree without invoking caller equality."""

    value_type = type(value)
    if value_type is dict:
        normalized: dict[str, Any] = {}
        for key, nested in value.items():
            if type(key) is not str:
                raise ExportConfigurationError(
                    f"{path} key must be exact str, found {type(key).__name__}"
                )
            normalized[key] = _exact_json_primitive(
                nested,
                f"{path}.{key}",
            )
        return normalized
    if value_type is list:
        return [
            _exact_json_primitive(nested, f"{path}[{index}]")
            for index, nested in enumerate(value)
        ]
    if value_type in {str, int, bool} or value is None:
        return value
    if value_type is float and math.isfinite(value):
        return value
    raise ExportConfigurationError(
        f"{path} must contain exact builtin JSON types, found {value_type.__name__}"
    )


def _canonical_json_bytes(value: Any, path: str) -> bytes:
    primitive = _exact_json_primitive(value, path)
    return json.dumps(
        primitive,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _verify_trusted_config(config: ExportConfig) -> None:
    """Reject parsed or constructed config state outside the trust root."""

    candidate = _canonical_config_bytes(config)
    trusted = load_export_config()
    if candidate != _canonical_config_bytes(trusted):
        raise ExportConfigurationError(
            "in-memory T20 export config differs from the immutable tracked config"
        )


def _legacy_cache(cache: Any) -> tuple[tuple[Any, Any], ...]:
    converter = getattr(cache, "to_legacy_cache", None)
    if callable(converter):
        cache = converter()
    return tuple((layer[0], layer[1]) for layer in cache)


def _torch_module() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ExportConfigurationError(
            "PyTorch is required for T20 wrapper construction and export"
        ) from exc
    return torch


class PrefillWrapper:
    """Factory returning a torch module with the frozen static prefill boundary."""

    def __new__(cls, model: Any, *, prompt_length: int) -> Any:
        torch = _torch_module()
        capacity = CONTEXT_VARIANTS[prompt_length]

        class _Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = model

            def forward(
                self,
                input_ids: Any,
                attention_mask: Any,
                position_ids: Any,
            ) -> tuple[Any, ...]:
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                    return_dict=True,
                )
                result: list[Any] = [
                    output.logits[:, -1, :]
                    .to(torch.float32)
                    .reshape(BATCH_SIZE, VOCAB_SIZE)
                ]
                for key, value in _legacy_cache(output.past_key_values):
                    result.append(
                        torch.nn.functional.pad(
                            key.to(torch.float16),
                            (0, 0, 0, capacity - prompt_length),
                        ).reshape(
                            BATCH_SIZE,
                            NUM_KEY_VALUE_HEADS,
                            capacity,
                            HEAD_DIM,
                        )
                    )
                    result.append(
                        torch.nn.functional.pad(
                            value.to(torch.float16),
                            (0, 0, 0, capacity - prompt_length),
                        ).reshape(
                            BATCH_SIZE,
                            NUM_KEY_VALUE_HEADS,
                            capacity,
                            HEAD_DIM,
                        )
                    )
                result.append(
                    torch.full(
                        (input_ids.shape[0],),
                        prompt_length,
                        dtype=torch.int64,
                        device=input_ids.device,
                    )
                )
                return tuple(result)

        return _Module()


class DecodeWrapper:
    """Factory returning a torch module with explicit fixed-capacity cache I/O."""

    def __new__(cls, model: Any, *, prompt_length: int) -> Any:
        torch = _torch_module()
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError as exc:
            raise ExportConfigurationError(
                "Transformers DynamicCache is required for decode export"
            ) from exc

        class _Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = model

            def forward(
                self,
                input_ids: Any,
                attention_mask: Any,
                position_ids: Any,
                *cache_and_length: Any,
            ) -> tuple[Any, ...]:
                cache_values = cache_and_length[:-1]
                valid_length = cache_and_length[-1]
                cache_pairs = tuple(
                    (
                        torch.narrow(
                            cache_values[2 * layer], 2, 0, valid_length[0]
                        ),
                        torch.narrow(
                            cache_values[2 * layer + 1], 2, 0, valid_length[0]
                        ),
                    )
                    for layer in range(len(cache_values) // 2)
                )
                active_mask = torch.narrow(
                    attention_mask, 1, 0, valid_length[0] + 1
                )
                model_cache = DynamicCache.from_legacy_cache(cache_pairs)
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=active_mask,
                    position_ids=position_ids,
                    past_key_values=model_cache,
                    use_cache=True,
                    return_dict=True,
                )
                result: list[Any] = [
                    output.logits[:, -1, :]
                    .to(torch.float32)
                    .reshape(BATCH_SIZE, VOCAB_SIZE)
                ]
                index = valid_length.reshape(1, 1, 1, 1)
                for layer, (present_key, present_value) in enumerate(
                    _legacy_cache(output.past_key_values)
                ):
                    incoming_key = cache_values[2 * layer]
                    incoming_value = cache_values[2 * layer + 1]
                    expanded_index = index.expand(
                        incoming_key.shape[0],
                        incoming_key.shape[1],
                        1,
                        incoming_key.shape[3],
                    )
                    result.append(
                        incoming_key.scatter(
                            2,
                            expanded_index,
                            present_key[:, :, -1:, :].to(torch.float16),
                        )
                    )
                    result.append(
                        incoming_value.scatter(
                            2,
                            expanded_index,
                            present_value[:, :, -1:, :].to(torch.float16),
                        )
                    )
                result.append(valid_length + 1)
                return tuple(result)

        return _Module()


def _dtype_from_contract(torch: Any, dtype: str) -> Any:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "int64": torch.int64,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ExportConfigurationError(
            f"no example-tensor mapping for contract dtype {dtype}"
        ) from exc


def build_example_inputs(
    contract: GraphContract,
    *,
    token_fixture: TokenFixtureBundle | None = None,
) -> tuple[Any, ...]:
    """Build deterministic concrete tensors for tracing one static graph."""

    torch = _torch_module()
    values: list[Any] = []
    fixture = token_fixture or load_export_config().token_fixture
    workload = fixture.workload(contract.prompt_length)
    prompt_tokens = workload.token_ids
    for spec in contract.inputs:
        dtype = _dtype_from_contract(torch, spec.dtype)
        if spec.name == "input_ids" and contract.graph_kind == "prefill":
            value = torch.tensor([prompt_tokens], dtype=dtype)
        elif spec.name == "input_ids":
            value = torch.tensor([[prompt_tokens[-1]]], dtype=dtype)
        elif spec.name == "attention_mask" and contract.graph_kind == "prefill":
            value = torch.ones(spec.shape, dtype=dtype)
        elif spec.name == "attention_mask":
            value = torch.zeros(spec.shape, dtype=dtype)
            value[:, : contract.prompt_length + 1] = 1
        elif spec.name == "position_ids" and contract.graph_kind == "prefill":
            value = torch.arange(contract.prompt_length, dtype=dtype).reshape(
                spec.shape
            )
        elif spec.name == "position_ids":
            value = torch.full(spec.shape, contract.prompt_length, dtype=dtype)
        elif spec.name == "valid_length":
            value = torch.full(spec.shape, contract.prompt_length, dtype=dtype)
        else:
            value = torch.zeros(spec.shape, dtype=dtype)
        values.append(value)
    return tuple(values)


def _tensor_names(specs: Sequence[TensorSpec]) -> list[str]:
    return [spec.name for spec in specs]


def _sha256(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _onnx_dtype_name(element_type: int) -> str:
    try:
        import onnx
    except ImportError as exc:
        raise ExportConfigurationError("ONNX is required for graph inspection") from exc
    mapping = {
        onnx.TensorProto.FLOAT: "float32",
        onnx.TensorProto.FLOAT16: "float16",
        onnx.TensorProto.INT64: "int64",
    }
    try:
        return mapping[element_type]
    except KeyError as exc:
        raise ExportConfigurationError(
            f"unsupported public ONNX element type {element_type}"
        ) from exc


def _value_info(value: Any) -> dict[str, Any]:
    tensor_type = value.type.tensor_type
    dimensions: list[int] = []
    for dimension in tensor_type.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value < 1:
            raise ExportConfigurationError(
                f"{value.name}: public ONNX dimension is not positive and static"
            )
        dimensions.append(int(dimension.dim_value))
    return {
        "name": value.name,
        "dtype": _onnx_dtype_name(tensor_type.elem_type),
        "shape": dimensions,
    }


def _expected_public_tensor(spec: TensorSpec) -> dict[str, Any]:
    return {"name": spec.name, "dtype": spec.dtype, "shape": list(spec.shape)}


def validate_onnx_contract(model: Any, contract: GraphContract) -> None:
    """Require exact static public ONNX I/O conformance to T12."""

    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs = tuple(
        _value_info(value)
        for value in model.graph.input
        if value.name not in initializer_names
    )
    outputs = tuple(_value_info(value) for value in model.graph.output)
    expected_inputs = tuple(_expected_public_tensor(spec) for spec in contract.inputs)
    expected_outputs = tuple(_expected_public_tensor(spec) for spec in contract.outputs)
    if inputs != expected_inputs:
        raise ExportConfigurationError(
            f"{contract.variant_id} {contract.graph_kind} ONNX inputs differ "
            "from the frozen T12 contract"
        )
    if outputs != expected_outputs:
        raise ExportConfigurationError(
            f"{contract.variant_id} {contract.graph_kind} ONNX outputs differ "
            "from the frozen T12 contract"
        )


def _external_location(initializer: Any) -> str | None:
    fields = {entry.key: entry.value for entry in initializer.external_data}
    return fields.get("location")


def _safe_external_path(onnx_path: Path, location: str) -> Path:
    pure = PurePosixPath(location)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ExportConfigurationError(
            f"unsafe ONNX external-data location {location!r}"
        )
    candidate = onnx_path.parent.joinpath(*pure.parts)
    if not candidate.is_file():
        raise ExportConfigurationError(
            f"missing ONNX external-data file {location!r}"
        )
    return candidate


def inspect_onnx_artifact(
    onnx_path: Path,
    contract: GraphContract,
    *,
    artifact_directory: Path,
    inline_initializer_limit_bytes: int,
) -> OnnxArtifactRecord:
    """Check graph structure/external data and return content-addressed evidence."""

    try:
        import onnx
    except ImportError as exc:
        raise ExportConfigurationError("ONNX is required for graph inspection") from exc
    onnx.checker.check_model(onnx_path)
    model = onnx.load_model(onnx_path, load_external_data=False)
    validate_onnx_contract(model, contract)

    external_locations: set[str] = set()
    largest_inline_initializer = 0
    for initializer in model.graph.initializer:
        location = _external_location(initializer)
        if location is not None:
            external_locations.add(location)
        else:
            largest_inline_initializer = max(
                largest_inline_initializer,
                len(initializer.raw_data),
            )
    if not external_locations:
        raise ExportConfigurationError(
            f"{onnx_path}: model weights are not stored as ONNX external data"
        )
    if largest_inline_initializer > inline_initializer_limit_bytes:
        raise ExportConfigurationError(
            f"{onnx_path}: largest inline initializer is "
            f"{largest_inline_initializer} bytes, exceeding limit "
            f"{inline_initializer_limit_bytes}"
        )

    external_records = tuple(
        ExternalDataRecord(
            location=location,
            sha256=_sha256(_safe_external_path(onnx_path, location)),
            size_bytes=_safe_external_path(onnx_path, location).stat().st_size,
        )
        for location in sorted(external_locations)
    )
    try:
        relative_path = onnx_path.relative_to(artifact_directory).as_posix()
    except ValueError as exc:
        raise ExportConfigurationError(
            f"{onnx_path}: artifact is outside {artifact_directory}"
        ) from exc
    return OnnxArtifactRecord(
        graph_kind=contract.graph_kind,
        relative_path=relative_path,
        sha256=_sha256(onnx_path),
        size_bytes=onnx_path.stat().st_size,
        external_data=external_records,
        input_tensors=tuple(_value_info(value) for value in model.graph.input),
        output_tensors=tuple(_value_info(value) for value in model.graph.output),
    )


def export_onnx_graph(
    wrapper: Any,
    example_inputs: tuple[Any, ...],
    contract: GraphContract,
    destination: Path,
    config: ExportConfig,
) -> None:
    """Trace one graph, force large initializers external, and check it."""

    torch = _torch_module()
    try:
        import onnx
    except ImportError as exc:
        raise ExportConfigurationError("ONNX is required for export") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    inline_path = destination.with_suffix(".inline.onnx")
    if destination.exists() or inline_path.exists():
        raise ExportConfigurationError(
            f"refusing to overwrite an existing export: {destination}"
        )
    torch.onnx.export(
        wrapper,
        example_inputs,
        inline_path,
        export_params=True,
        opset_version=config.opset,
        do_constant_folding=False,
        input_names=_tensor_names(contract.inputs),
        output_names=_tensor_names(contract.outputs),
        dynamic_axes=None,
        dynamo=False,
        external_data=False,
    )
    model = onnx.load_model(inline_path, load_external_data=True)
    validate_onnx_contract(model, contract)
    data_location = f"{destination.name}.data"
    onnx.save_model(
        model,
        destination,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_location,
        size_threshold=config.external_data_threshold_bytes,
        convert_attribute=False,
    )
    inline_path.unlink()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repository_relative(path: Path, label: str) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ExportConfigurationError(
            f"{label} must be a repository file, found {resolved}"
        ) from exc
    tracked = subprocess.run(
        ("git", "ls-files", "--error-unmatch", relative),
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if tracked.returncode != 0:
        raise ExportConfigurationError(
            f"{label} is not tracked by Git: {relative}"
        )
    return relative


def _validate_exporter_commit(commit: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ExportConfigurationError(
            f"exporter commit must be a full lowercase Git SHA, found {commit!r}"
        )
    commit_check = subprocess.run(
        ("git", "cat-file", "-e", f"{commit}^{{commit}}"),
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if commit_check.returncode != 0:
        raise ExportConfigurationError(
            f"exporter commit does not exist locally: {commit}"
        )
    ancestor_check = subprocess.run(
        ("git", "merge-base", "--is-ancestor", commit, "HEAD"),
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if ancestor_check.returncode != 0:
        raise ExportConfigurationError(
            f"exporter commit is not an ancestor of HEAD: {commit}"
        )


def _git_blob(commit: str, relative_path: str) -> bytes:
    try:
        return subprocess.check_output(
            ("git", "show", f"{commit}:{relative_path}"),
            cwd=PROJECT_ROOT,
        )
    except subprocess.CalledProcessError as exc:
        raise ExportConfigurationError(
            f"exporter commit {commit} does not contain {relative_path}"
        ) from exc


def _blob_record(
    commit: str,
    path: Path,
    label: str,
    *,
    require_current_match: bool,
) -> dict[str, str]:
    relative = _repository_relative(path, label)
    commit_digest = hashlib.sha256(_git_blob(commit, relative)).hexdigest()
    if require_current_match:
        current_digest = _sha256(path)
        if commit_digest != current_digest:
            raise ExportConfigurationError(
                f"{label} changed after exporter commit {commit}: "
                f"{commit_digest} != {current_digest}"
            )
    return {"path": relative, "sha256": commit_digest}


def _export_provenance(
    *,
    config: ExportConfig,
    prompt_length: int,
) -> dict[str, Any]:
    git_commit = config.evidence_attestation.exporter_commit
    _validate_exporter_commit(git_commit)
    export_config_relative = _repository_relative(
        config.source_path,
        "export config",
    )
    historical_config = json.loads(
        _git_blob(git_commit, export_config_relative).decode("utf-8")
    )
    current_config = _load_json_document(config.source_path, "export config")
    current_export_settings = dict(current_config)
    current_export_settings.pop("evidence_attestation", None)
    if historical_config != current_export_settings:
        raise ExportConfigurationError(
            "current export settings differ from the attested exporter "
            f"commit {git_commit}"
        )
    workload = config.token_fixture.workload(prompt_length)
    return {
        "commit": git_commit,
        "run_attestation": config.evidence_attestation.as_dict(),
        "attestation_source": {
            "path": export_config_relative,
            "sha256": _sha256(config.source_path),
        },
        "exporter_source": _blob_record(
            git_commit,
            EXPORTER_SOURCE_PATH,
            "exporter source",
            require_current_match=False,
        ),
        "export_config": _blob_record(
            git_commit,
            config.source_path,
            "export config",
            require_current_match=False,
        ),
        "model_contract": _blob_record(
            git_commit,
            config.model_contract_path,
            "model contract",
            require_current_match=True,
        ),
        "t10_workload_contract": _blob_record(
            git_commit,
            DEFAULT_T10_CONTRACT_PATH,
            "T10 workload contract",
            require_current_match=True,
        ),
        "t10_source_fixture": _blob_record(
            git_commit,
            DEFAULT_T10_SOURCE_PATH,
            "T10 source fixture",
            require_current_match=True,
        ),
        "token_fixture_bundle": {
            **_blob_record(
                git_commit,
                config.token_fixture.source_path,
                "configured token fixture",
                require_current_match=True,
            ),
            "canonical_json_sha256": (
                config.token_fixture.canonical_json_sha256
            ),
        },
        "workload": workload.provenance(),
    }


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def _validate_attested_artifacts(
    *,
    prompt_length: int,
    config: ExportConfig,
    prefill: OnnxArtifactRecord,
    decode: OnnxArtifactRecord,
    source_weights_sha256: str,
) -> None:
    attestation = config.evidence_attestation
    if source_weights_sha256 != attestation.source_artifact_sha256:
        raise ExportConfigurationError(
            "source weights differ from the independent export attestation"
        )
    for record in (prefill, decode):
        expected_graph_sha256 = attestation.graph_sha256(
            prompt_length,
            record.graph_kind,
        )
        if record.sha256 != expected_graph_sha256:
            raise ExportConfigurationError(
                f"S{prompt_length} {record.graph_kind} graph differs from "
                "the independent export attestation"
            )
        if not record.external_data or any(
            item.sha256 != attestation.external_data_sha256
            for item in record.external_data
        ):
            raise ExportConfigurationError(
                f"S{prompt_length} {record.graph_kind} external data differs "
                "from the independent export attestation"
            )


def _manifest_payload(
    *,
    prompt_length: int,
    config: ExportConfig,
    prefill: OnnxArtifactRecord,
    decode: OnnxArtifactRecord,
    source_weights_sha256: str,
    host_manifest_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    _validate_attested_artifacts(
        prompt_length=prompt_length,
        config=config,
        prefill=prefill,
        decode=decode,
        source_weights_sha256=source_weights_sha256,
    )
    model_contract = load_model_contract()
    prefill_contract = build_prefill_contract(prompt_length).as_dict()
    decode_contract = build_decode_contract(prompt_length).as_dict()
    command_prefix = (
        "HF_HOME=<local-cache> TRANSFORMERS_OFFLINE=1 "
        "SLM_LAB_ARTIFACT_ROOT=<external-root> PYTHONPATH=src "
        "python -m slm_lab.export.onnx_matrix"
    )
    git_commit = config.evidence_attestation.exporter_commit
    python_version = config.evidence_attestation.runtime_python_version
    return {
        "schema_version": 1,
        "model_id": model_contract.model_id,
        "model_revision": model_contract.revision,
        "tokenizer_revision": model_contract.revision,
        "chat_template_sha256": (
            json.loads(model_contract.source_path.read_text(encoding="utf-8"))[
                "tokenizer"
            ]["chat_template"]["sha256"]
        ),
        "source_artifact_sha256": source_weights_sha256,
        "git_commit": git_commit,
        "task_id": TASK_ID,
        "exporter": config.exporter,
        "exporter_version": config.torch_version,
        "opset": config.opset,
        "input_contract": {
            "prefill": [tensor["name"] for tensor in prefill_contract["inputs"]],
            "decode": [tensor["name"] for tensor in decode_contract["inputs"]],
        },
        "cache_contract": {
            "prefill": prefill_contract["cache_update"],
            "decode": decode_contract["cache_update"],
        },
        "context_length": prompt_length,
        "precision": config.precision,
        "quantization": None,
        "calibration_dataset_revision": None,
        "runtime": None,
        "runtime_version": None,
        "qairt_version": None,
        "target_device": None,
        "device_type": None,
        "compile_options": {},
        "profile_options": {},
        "provider_options": {},
        "host_manifest_sha256": host_manifest_sha256,
        "created_at": created_at,
        "status": "exported_and_shape_validated",
        "export_provenance": _export_provenance(
            config=config,
            prompt_length=prompt_length,
        ),
        "variant_id": f"S{prompt_length}",
        "cache_capacity": CONTEXT_VARIANTS[prompt_length],
        "contract": {
            "prefill": prefill_contract,
            "prefill_sha256": _canonical_sha256(prefill_contract),
            "decode": decode_contract,
            "decode_sha256": _canonical_sha256(decode_contract),
        },
        "toolchain": {
            "python": python_version,
            "torch": config.torch_version,
            "transformers": config.transformers_version,
            "onnx": config.onnx_version,
            "attention_implementation": "eager",
            "device": "cpu",
        },
        "artifacts": {
            "root": "${SLM_LAB_ARTIFACT_ROOT}/onnx/reference/T20",
            "prefill": prefill.as_dict(),
            "decode": decode.as_dict(),
        },
        "commands": {
            "export": f"{command_prefix} export --context {prompt_length}",
            "validate": f"{command_prefix} validate --context {prompt_length}",
        },
        "claim_boundary": {
            "establishes": [
                "pinned_host_export_completed",
                "onnx_checker_accepted_graph",
                "public_io_matches_frozen_T12_contract",
                "external_data_files_exist_and_match_committed_hashes",
            ],
            "does_not_establish": [
                "onnxruntime_numerical_parity",
                "compiler_acceptance",
                "accelerator_placement",
                "latency_or_memory_performance",
            ],
        },
    }


def verify_manifest_evidence(
    manifest: Mapping[str, Any],
    *,
    prompt_length: int,
    config: ExportConfig,
    prefill: OnnxArtifactRecord,
    decode: OnnxArtifactRecord,
    source_weights_sha256: str,
    host_manifest_sha256: str,
) -> None:
    """Reconstruct every deterministic field and reject any claim drift."""

    actual_canonical = _canonical_json_bytes(manifest, "manifest")
    _verify_trusted_config(config)
    validate_manifest("artifact", manifest)
    created_at = manifest.get("created_at")
    if not isinstance(created_at, str):
        raise ExportConfigurationError(
            f"S{prompt_length} manifest lacks creation time"
        )
    expected = _manifest_payload(
        prompt_length=prompt_length,
        config=config,
        prefill=prefill,
        decode=decode,
        source_weights_sha256=source_weights_sha256,
        host_manifest_sha256=host_manifest_sha256,
        created_at=created_at,
    )
    expected_canonical = _canonical_json_bytes(expected, "expected manifest")
    if actual_canonical != expected_canonical:
        raise ExportConfigurationError(
            f"S{prompt_length} manifest deterministic fields differ"
        )


def _artifact_root() -> Path:
    value = os.environ.get("SLM_LAB_ARTIFACT_ROOT")
    if not value:
        raise ExportConfigurationError(
            "SLM_LAB_ARTIFACT_ROOT must identify external artifact storage"
        )
    root = Path(value).expanduser()
    if not root.is_absolute():
        raise ExportConfigurationError("SLM_LAB_ARTIFACT_ROOT must be absolute")
    return root


def _source_weights_path() -> Path:
    cache = os.environ.get("HF_HOME")
    if not cache:
        raise ExportConfigurationError(
            "HF_HOME must identify the pinned local Hugging Face cache"
        )
    return (
        Path(cache)
        / "hub/models--Qwen--Qwen3-0.6B/snapshots"
        / "c1899de289a04d12100db370d81485cdf75e47ca/model.safetensors"
    )


def _selected_contexts(requested: int | None, config: ExportConfig) -> tuple[int, ...]:
    if requested is None:
        return config.contexts
    if requested not in config.contexts:
        raise ExportConfigurationError(
            f"context must be one of {config.contexts}, found {requested}"
        )
    return (requested,)


def _verify_runtime(config: ExportConfig) -> None:
    _verify_trusted_config(config)
    actual_python_version = platform.python_version()
    expected_python_version = config.evidence_attestation.runtime_python_version
    if actual_python_version != expected_python_version:
        raise ExportConfigurationError(
            "Python version mismatch: expected "
            f"{expected_python_version}, found {actual_python_version}"
        )
    _require_exact_version(_package_version("torch"), config.torch_version, "torch")
    _require_exact_version(
        _package_version("transformers"),
        config.transformers_version,
        "transformers",
    )
    _require_exact_version(_package_version("onnx"), config.onnx_version, "onnx")


def run_export(contexts: Iterable[int], config: ExportConfig) -> None:
    """Load the model once and export the selected graph pairs."""

    _verify_runtime(config)
    artifact_directory = _artifact_root() / ARTIFACT_SUBDIRECTORY
    reference = load_reference_model(
        device="cpu",
        dtype=config.precision,
        seed=config.seed,
        local_files_only=True,
        attn_implementation="eager",
    )
    for prompt_length in contexts:
        for contract, wrapper in (
            (
                build_prefill_contract(prompt_length),
                PrefillWrapper(reference.model, prompt_length=prompt_length),
            ),
            (
                build_decode_contract(prompt_length),
                DecodeWrapper(reference.model, prompt_length=prompt_length),
            ),
        ):
            destination = (
                artifact_directory
                / f"S{prompt_length}"
                / f"{contract.graph_kind}.onnx"
            )
            if destination.exists():
                inspect_onnx_artifact(
                    destination,
                    contract,
                    artifact_directory=artifact_directory,
                    inline_initializer_limit_bytes=(
                        config.external_data_threshold_bytes
                    ),
                )
                continue
            export_onnx_graph(
                wrapper,
                build_example_inputs(
                    contract,
                    token_fixture=config.token_fixture,
                ),
                contract,
                destination,
                config,
            )


def run_validate(
    contexts: Iterable[int],
    config: ExportConfig,
    *,
    write_manifests: bool,
) -> None:
    """Validate graph/data hashes and optionally commit-ready manifests."""

    _verify_runtime(config)
    artifact_directory = _artifact_root() / ARTIFACT_SUBDIRECTORY
    source_weights = _source_weights_path()
    if not source_weights.is_file():
        raise ExportConfigurationError(f"missing pinned weights: {source_weights}")
    source_weights_sha256 = _sha256(source_weights)
    host_manifest_sha256 = _sha256(DEFAULT_HOST_MANIFEST_PATH)
    if write_manifests:
        DEFAULT_MANIFEST_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for prompt_length in contexts:
        records: dict[str, OnnxArtifactRecord] = {}
        for contract in (
            build_prefill_contract(prompt_length),
            build_decode_contract(prompt_length),
        ):
            path = (
                artifact_directory
                / f"S{prompt_length}"
                / f"{contract.graph_kind}.onnx"
            )
            records[contract.graph_kind] = inspect_onnx_artifact(
                path,
                contract,
                artifact_directory=artifact_directory,
                inline_initializer_limit_bytes=(
                    config.external_data_threshold_bytes
                ),
            )
        destination = DEFAULT_MANIFEST_DIRECTORY / f"S{prompt_length}.json"
        if write_manifests:
            if destination.exists():
                try:
                    prior = json.loads(destination.read_text(encoding="utf-8"))
                    created_at = prior["created_at"]
                except (
                    OSError,
                    json.JSONDecodeError,
                    KeyError,
                    TypeError,
                ) as exc:
                    raise ExportConfigurationError(
                        f"cannot preserve provenance from {destination}: {exc}"
                    ) from exc
            else:
                created_at = _utc_now()
            manifest = _manifest_payload(
                prompt_length=prompt_length,
                config=config,
                prefill=records["prefill"],
                decode=records["decode"],
                source_weights_sha256=source_weights_sha256,
                host_manifest_sha256=host_manifest_sha256,
                created_at=created_at,
            )
            validate_manifest("artifact", manifest)
            destination.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            try:
                manifest = json.loads(destination.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ExportConfigurationError(
                    f"cannot load committed manifest {destination}: {exc}"
                ) from exc
            verify_manifest_evidence(
                manifest,
                prompt_length=prompt_length,
                config=config,
                prefill=records["prefill"],
                decode=records["decode"],
                source_weights_sha256=source_weights_sha256,
                host_manifest_sha256=host_manifest_sha256,
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_SPELLING)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("export", "validate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--context", type=int)
        if command == "validate":
            subparser.add_argument("--write-manifests", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_export_config(args.config)
    contexts = _selected_contexts(args.context, config)
    if args.command == "export":
        run_export(contexts, config)
    else:
        run_validate(
            contexts,
            config,
            write_manifests=args.write_manifests,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
