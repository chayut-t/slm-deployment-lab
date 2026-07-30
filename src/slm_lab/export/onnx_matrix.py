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
import os
import platform
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
)
from slm_lab.manifests.validation import validate_manifest
from slm_lab.models import load_model_contract, load_reference_model


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/models/qwen3-0.6b-onnx-export.json"
DEFAULT_TOKEN_FIXTURE_PATH = (
    PROJECT_ROOT / "tests/fixtures/t10/token-fixtures-v1.json"
)
DEFAULT_HOST_MANIFEST_PATH = PROJECT_ROOT / "results/hosts/apple-m4-primary.json"
DEFAULT_MANIFEST_DIRECTORY = PROJECT_ROOT / "results/manifests/onnx"
ARTIFACT_SUBDIRECTORY = Path("onnx/reference/T20")
TASK_ID = "T20"


class ExportConfigurationError(ValueError):
    """The requested export or an exported artifact violates T20 policy."""


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


def load_export_config(path: Path | str = DEFAULT_CONFIG_PATH) -> ExportConfig:
    """Load and strictly validate the committed export configuration."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        export = payload["export"]
        packages = payload["packages"]
        contexts = tuple(payload["contexts"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ExportConfigurationError(f"invalid export config {source}: {exc}") from exc

    if payload.get("schema_version") != 1 or payload.get("task_id") != TASK_ID:
        raise ExportConfigurationError("export config identity must be schema 1 / T20")
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
                result: list[Any] = [output.logits[:, -1, :].to(torch.float32)]
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
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=active_mask,
                    position_ids=position_ids,
                    past_key_values=cache_pairs,
                    use_cache=True,
                    return_dict=True,
                )
                result: list[Any] = [output.logits[:, -1, :].to(torch.float32)]
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


def _load_prompt_tokens(prompt_length: int, fixture_path: Path) -> list[int]:
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    match = next(
        (
            workload
            for workload in payload["context_workloads"]
            if workload["context_length"] == prompt_length
        ),
        None,
    )
    if match is None or len(match["token_ids"]) != prompt_length:
        raise ExportConfigurationError(
            f"token fixture has no exact S{prompt_length} workload"
        )
    return list(match["token_ids"])


def build_example_inputs(
    contract: GraphContract,
    *,
    token_fixture_path: Path = DEFAULT_TOKEN_FIXTURE_PATH,
) -> tuple[Any, ...]:
    """Build deterministic concrete tensors for tracing one static graph."""

    torch = _torch_module()
    values: list[Any] = []
    prompt_tokens = _load_prompt_tokens(contract.prompt_length, token_fixture_path)
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
    inline_bytes = 0
    for initializer in model.graph.initializer:
        location = _external_location(initializer)
        if location is not None:
            external_locations.add(location)
        else:
            inline_bytes += len(initializer.raw_data)
    if not external_locations:
        raise ExportConfigurationError(
            f"{onnx_path}: model weights are not stored as ONNX external data"
        )
    if inline_bytes > inline_initializer_limit_bytes:
        raise ExportConfigurationError(
            f"{onnx_path}: {inline_bytes} inline initializer bytes exceed "
            f"limit {inline_initializer_limit_bytes}"
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


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_commit() -> str:
    value = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=PROJECT_ROOT, text=True
    ).strip()
    if len(value) != 40:
        raise ExportConfigurationError("Git did not return a full commit SHA")
    return value


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def _manifest_payload(
    *,
    prompt_length: int,
    config: ExportConfig,
    prefill: OnnxArtifactRecord,
    decode: OnnxArtifactRecord,
    source_weights_sha256: str,
    git_commit: str,
    host_manifest_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    model_contract = load_model_contract()
    prefill_contract = build_prefill_contract(prompt_length).as_dict()
    decode_contract = build_decode_contract(prompt_length).as_dict()
    command_prefix = (
        "HF_HOME=<local-cache> TRANSFORMERS_OFFLINE=1 "
        "SLM_LAB_ARTIFACT_ROOT=<external-root> PYTHONPATH=src "
        "python -m slm_lab.export.onnx_matrix"
    )
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
        "variant_id": f"S{prompt_length}",
        "cache_capacity": CONTEXT_VARIANTS[prompt_length],
        "contract": {
            "prefill": prefill_contract,
            "prefill_sha256": _canonical_sha256(prefill_contract),
            "decode": decode_contract,
            "decode_sha256": _canonical_sha256(decode_contract),
        },
        "toolchain": {
            "python": platform.python_version(),
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
            export_onnx_graph(
                wrapper,
                build_example_inputs(contract),
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
    git_commit = _git_commit()
    host_manifest_sha256 = _sha256(DEFAULT_HOST_MANIFEST_PATH)
    created_at = _utc_now()
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
        manifest = _manifest_payload(
            prompt_length=prompt_length,
            config=config,
            prefill=records["prefill"],
            decode=records["decode"],
            source_weights_sha256=source_weights_sha256,
            git_commit=git_commit,
            host_manifest_sha256=host_manifest_sha256,
            created_at=created_at,
        )
        validate_manifest("artifact", manifest)
        if write_manifests:
            destination = DEFAULT_MANIFEST_DIRECTORY / f"S{prompt_length}.json"
            destination.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
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
