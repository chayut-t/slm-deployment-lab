"""Safe, independently runnable Qualcomm AI Hub Workbench stage adapters.

The external client deliberately remains optional. Tests inject a mock backend,
while command-line runs construct :class:`QaiHubBackend` only after validating
the complete local request. SDK stdout/stderr and exception details are always
discarded because the client may emit private job URLs or service messages.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import re
import shlex
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TypeVar


SCHEMA_VERSION = 1
MANIFEST_TYPE = "slm_lab.qualcomm.ai_hub.stage"
STAGES = {"compile", "inference", "profile"}
SAFE_STATES = {"SUCCESS", "FAILED"}
SAFE_COMPUTE_UNITS = {"CPU", "GPU", "NPU"}
ALLOWED_DTYPES = {
    "float16",
    "float32",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXACT_VERSION_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)+(?:[A-Za-z0-9._-]*)$")
SAFE_TEXT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._()+,:=@-]{0,199}$")
SAFE_LOGICAL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
SAFE_TENSOR_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.:/-]{0,199}$")
PRIVATE_TEXT_PATTERNS = (
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"\b(?:api[_-]?)?token\s*[:=]", re.IGNORECASE),
    re.compile(r"\bauthorization\b", re.IGNORECASE),
    re.compile(r"\bbearer\s+\S+", re.IGNORECASE),
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"/jobs/[A-Za-z0-9_-]+", re.IGNORECASE),
    re.compile(r"\bj[a-z0-9]{7,}\b", re.IGNORECASE),
)
SENSITIVE_OPTION_NAMES = {
    "access-key",
    "account",
    "account-id",
    "api-key",
    "api-token",
    "auth",
    "authorization",
    "bearer",
    "client-id",
    "credential",
    "credentials",
    "email",
    "organization",
    "owner",
    "password",
    "project",
    "project-id",
    "secret",
    "token",
    "user",
    "user-id",
}
QAIRT_OPTION_NAMES = {"qairt-version"}
PATH_OPTION_NAMES = {
    "cache-dir",
    "config",
    "config-file",
    "input-file",
    "model-file",
    "output-dir",
    "path",
}
PATH_OPTION_SUFFIXES = ("-directory", "-file", "-path")

T = TypeVar("T")


class AiHubAdapterError(RuntimeError):
    """A sanitized adapter error safe to print in public task logs."""


class AiHubBackend(Protocol):
    """Minimal external-client surface required by the adapter."""

    @property
    def client_version(self) -> str:
        """Return the exact installed ``qai-hub`` distribution version."""

    def submit_compile(
        self,
        *,
        model: Path,
        device: Mapping[str, Any],
        input_specs: Mapping[str, tuple[tuple[int, ...], str]],
        options: str,
        name: str,
        retry: bool,
    ) -> Any:
        """Submit one compile job."""

    def submit_inference(
        self,
        *,
        model: Path,
        device: Mapping[str, Any],
        inputs: Path,
        options: str,
        name: str,
        retry: bool,
    ) -> Any:
        """Submit one inference job."""

    def submit_profile(
        self,
        *,
        model: Path,
        device: Mapping[str, Any],
        options: str,
        name: str,
        retry: bool,
    ) -> Any:
        """Submit one profile job."""


class QaiHubBackend:
    """Production backend for the optional Qualcomm ``qai-hub`` client."""

    def __init__(self) -> None:
        try:
            with _discard_external_output():
                hub = importlib.import_module("qai_hub")
                hub.set_verbose(False)
                client = hub.Client()
                client.set_verbose(False)
            version = importlib.metadata.version("qai-hub")
        except Exception:
            raise AiHubAdapterError(
                "qai-hub client is unavailable or could not initialize; "
                "private details suppressed"
            ) from None
        self._hub = hub
        self._client = client
        self._client_version = version

    @property
    def client_version(self) -> str:
        return self._client_version

    def _device(self, selector: Mapping[str, Any]) -> Any:
        return self._hub.Device(
            name=selector["name"],
            os=selector["os"],
            attributes=selector["attributes"],
        )

    def submit_compile(
        self,
        *,
        model: Path,
        device: Mapping[str, Any],
        input_specs: Mapping[str, tuple[tuple[int, ...], str]],
        options: str,
        name: str,
        retry: bool,
    ) -> Any:
        return self._client.submit_compile_job(
            model=str(model),
            device=self._device(device),
            input_specs=input_specs,
            options=options,
            name=name,
            single_compile=True,
            retry=retry,
        )

    def submit_inference(
        self,
        *,
        model: Path,
        device: Mapping[str, Any],
        inputs: Path,
        options: str,
        name: str,
        retry: bool,
    ) -> Any:
        return self._client.submit_inference_job(
            model=str(model),
            device=self._device(device),
            inputs=str(inputs),
            options=options,
            name=name,
            retry=retry,
        )

    def submit_profile(
        self,
        *,
        model: Path,
        device: Mapping[str, Any],
        options: str,
        name: str,
        retry: bool,
    ) -> Any:
        return self._client.submit_profile_job(
            model=str(model),
            device=self._device(device),
            options=options,
            name=name,
            retry=retry,
        )


@contextlib.contextmanager
def _discard_external_output():
    """Prevent SDK output from exposing job URLs, IDs, tokens, or accounts."""

    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        yield


def _quiet_call(
    stage: str, operation: Callable[..., T], *args: Any, **kwargs: Any
) -> T:
    try:
        with _discard_external_output():
            return operation(*args, **kwargs)
    except Exception:
        raise AiHubAdapterError(
            f"{stage} stage failed; private service details suppressed"
        ) from None


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        raise AiHubAdapterError("artifact could not be read") from None
    return digest.hexdigest()


def _safe_exact_version(value: Any, field: str) -> str:
    if not isinstance(value, str) or not EXACT_VERSION_PATTERN.fullmatch(value):
        raise AiHubAdapterError(f"{field} must be an exact version")
    return value


def _safe_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SAFE_TEXT_PATTERN.fullmatch(value):
        raise AiHubAdapterError(f"{field} contains unsupported text")
    _assert_safe_string(value, field)
    return value


def _safe_optional_text(value: Any, field: str) -> str:
    if value == "":
        return ""
    return _safe_text(value, field)


def _safe_logical_name(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SAFE_LOGICAL_NAME_PATTERN.fullmatch(value):
        raise AiHubAdapterError(f"{field} must be a path-free logical name")
    _assert_safe_string(value, field)
    return value


def _assert_safe_string(value: str, field: str) -> None:
    if any(pattern.search(value) for pattern in PRIVATE_TEXT_PATTERNS):
        raise AiHubAdapterError(f"{field} contains private or URL-like text")


def _assert_public_safe(value: Any, field: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise AiHubAdapterError(f"{field} contains a non-string key")
            _assert_safe_string(key, field)
            _assert_public_safe(child, f"{field}.{key}")
    elif isinstance(value, list):
        for child in value:
            _assert_public_safe(child, field)
    elif isinstance(value, str):
        _assert_safe_string(value, field)


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AiHubAdapterError(f"{field} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    field: str,
) -> None:
    keys = set(value)
    missing = required - keys
    extra = keys - required - optional
    if missing or extra:
        raise AiHubAdapterError(f"{field} has missing or unsupported fields")


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AiHubAdapterError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AiHubAdapterError(f"{field} must be a nonnegative integer")
    return value


def _load_json(path: Path, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise AiHubAdapterError(f"{field} is not readable valid JSON") from None
    return _require_mapping(value, field)


def load_request(path: Path, expected_stage: str) -> Mapping[str, Any]:
    request = _load_json(path, "request")
    if expected_stage not in STAGES:
        raise AiHubAdapterError("unsupported stage")
    if request.get("schema_version") != SCHEMA_VERSION:
        raise AiHubAdapterError("request schema_version must be 1")
    if request.get("stage") != expected_stage:
        raise AiHubAdapterError("request stage does not match command")
    _assert_public_safe(
        _public_request_projection(request, validate_only=True), "request"
    )
    return request


def _artifact_from_request(
    value: Any,
    *,
    role: str,
) -> tuple[Path, dict[str, Any]]:
    artifact = _require_mapping(value, role)
    _require_exact_keys(
        artifact,
        required={"path", "logical_name", "sha256"},
        field=role,
    )
    path_value = artifact["path"]
    if not isinstance(path_value, str) or not path_value:
        raise AiHubAdapterError(f"{role}.path must be a nonempty path")
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise AiHubAdapterError(f"{role} artifact is missing")
    logical_name = _safe_logical_name(artifact["logical_name"], f"{role}.logical_name")
    expected_sha = artifact["sha256"]
    if not isinstance(expected_sha, str) or not SHA256_PATTERN.fullmatch(expected_sha):
        raise AiHubAdapterError(f"{role}.sha256 must be a lowercase SHA-256")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise AiHubAdapterError(f"{role} artifact digest mismatch")
    try:
        byte_size = path.stat().st_size
    except OSError:
        raise AiHubAdapterError(f"{role} artifact metadata is unavailable") from None
    return path, {
        "role": role,
        "logical_name": logical_name,
        "sha256": actual_sha,
        "byte_size": byte_size,
    }


def _private_output_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AiHubAdapterError(f"{field} must be a nonempty path")
    path = Path(value).expanduser()
    repo_root = Path(__file__).resolve().parents[4]
    try:
        relative = path.resolve(strict=False).relative_to(repo_root)
    except ValueError:
        relative = None
    if relative is not None and (
        not relative.parts or relative.parts[0] not in {".ai-local", "artifacts"}
    ):
        raise AiHubAdapterError(
            f"{field} must be external or under ignored private storage"
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise AiHubAdapterError(f"{field} parent could not be prepared") from None
    return path


def _device(value: Any) -> dict[str, Any]:
    device = _require_mapping(value, "device")
    _require_exact_keys(
        device,
        required={"name"},
        optional={"os", "attributes"},
        field="device",
    )
    normalized = {
        "name": _safe_text(device["name"], "device.name"),
        "os": _safe_optional_text(device.get("os", ""), "device.os"),
        "attributes": [],
    }
    if "attributes" in device:
        attributes = device["attributes"]
        if not isinstance(attributes, list) or any(
            not isinstance(item, str) for item in attributes
        ):
            raise AiHubAdapterError("device.attributes must be a list of strings")
        normalized["attributes"] = [
            _safe_text(item, "device.attributes") for item in attributes
        ]
    return normalized


def _runtime(value: Any) -> dict[str, str]:
    runtime = _require_mapping(value, "runtime")
    _require_exact_keys(runtime, required={"name", "version"}, field="runtime")
    return {
        "name": _safe_text(runtime["name"], "runtime.name"),
        "version": _safe_exact_version(runtime["version"], "runtime.version"),
    }


def _option_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lstrip("-").lower()).strip("-")


def _path_like_option_token(value: str) -> bool:
    candidate = value.partition("=")[2] if "=" in value else value
    return (
        "/" in candidate
        or "\\" in candidate
        or candidate == "~"
        or candidate.startswith(("~/", "~\\", "./", ".\\", "../", "..\\"))
        or bool(re.match(r"^[A-Za-z]:", candidate))
    )


def _validate_options(value: Any, runtime: Mapping[str, str]) -> str:
    if not isinstance(value, str):
        raise AiHubAdapterError("options must be a string")
    _assert_safe_string(value, "options")
    try:
        tokens = shlex.split(value, posix=True)
    except ValueError:
        raise AiHubAdapterError("options could not be parsed safely") from None

    qairt_versions: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        flag, separator, inline_value = token.partition("=")
        name = _option_name(flag) if flag.startswith("-") else ""
        if name in SENSITIVE_OPTION_NAMES:
            raise AiHubAdapterError(
                "options contain credential, account, or identity material"
            )
        if name in PATH_OPTION_NAMES or name.endswith(PATH_OPTION_SUFFIXES):
            raise AiHubAdapterError("options contain path-bearing fields")
        if _path_like_option_token(token):
            raise AiHubAdapterError("options contain path-like material")
        if name in QAIRT_OPTION_NAMES:
            if separator:
                version = inline_value
            else:
                index += 1
                if index >= len(tokens):
                    raise AiHubAdapterError("options omit the QAIRT version value")
                version = tokens[index]
            qairt_versions.append(_safe_exact_version(version, "options QAIRT version"))
        index += 1

    if qairt_versions != [runtime["version"]]:
        raise AiHubAdapterError(
            "options must submit exactly the requested QAIRT version"
        )
    return value


def _input_specs(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    specs = _require_mapping(value, "input_specs")
    if not specs:
        raise AiHubAdapterError("input_specs must not be empty")
    public: dict[str, Any] = {}
    sdk: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, raw_spec in specs.items():
        if not isinstance(name, str) or not SAFE_TENSOR_NAME_PATTERN.fullmatch(name):
            raise AiHubAdapterError("input_specs contains an invalid tensor name")
        spec = _require_mapping(raw_spec, f"input_specs.{name}")
        _require_exact_keys(
            spec,
            required={"shape", "dtype"},
            field=f"input_specs.{name}",
        )
        shape = spec["shape"]
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
                for dimension in shape
            )
        ):
            raise AiHubAdapterError("input_specs shapes must be positive integers")
        dtype = spec["dtype"]
        if dtype not in ALLOWED_DTYPES:
            raise AiHubAdapterError("input_specs contains an unsupported dtype")
        public[name] = {"shape": shape, "dtype": dtype}
        sdk[name] = (tuple(shape), dtype)
    return public, sdk


def _common_request(request: Mapping[str, Any], stage: str) -> dict[str, Any]:
    runtime = _runtime(request.get("runtime"))
    common = {
        "client_version": _safe_exact_version(
            request.get("client_version"), "client_version"
        ),
        "device": _device(request.get("device")),
        "runtime": runtime,
        "options": _validate_options(request.get("options", ""), runtime),
        "job_name": _safe_text(request.get("job_name"), "job_name"),
        "timeout_seconds": _positive_int(
            request.get("timeout_seconds"), "timeout_seconds"
        ),
        "retry": request.get("retry", False),
    }
    if not isinstance(common["retry"], bool):
        raise AiHubAdapterError("retry must be boolean")
    if common["retry"]:
        raise AiHubAdapterError(
            "retry must be false so stage execution has a bounded submission attempt"
        )
    _assert_public_safe(common, f"{stage} request")
    return common


def _public_request_projection(
    request: Mapping[str, Any],
    *,
    validate_only: bool = False,
) -> Mapping[str, Any]:
    """Remove private filesystem paths before request hashing or scanning."""

    projected: dict[str, Any] = {}
    private_path_fields = {
        "path",
        "output_artifact",
        "raw_profile_output",
        "predecessor_manifest",
    }
    for key, value in request.items():
        if key in private_path_fields:
            continue
        if isinstance(value, Mapping):
            projected[key] = {
                child_key: child_value
                for child_key, child_value in value.items()
                if child_key not in private_path_fields
            }
        else:
            projected[key] = value
    if validate_only:
        return projected
    return json.loads(json.dumps(projected))


def _request_id(stage: str, public_request: Mapping[str, Any]) -> str:
    digest = _sha256_bytes(_canonical_bytes(public_request))
    return f"t30-{stage}-{digest[:20]}"


def _observed_at() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _status_success(status: Any) -> bool:
    success = getattr(status, "success", None)
    code = getattr(status, "code", None)
    if isinstance(status, str):
        code = status
        success = status == "SUCCESS"
    if not isinstance(success, bool) or not isinstance(code, str):
        raise AiHubAdapterError("service returned an unsupported job status")
    if code not in SAFE_STATES:
        raise AiHubAdapterError("service returned a non-terminal job status")
    return success and code == "SUCCESS"


def _wait_success(stage: str, job: Any, timeout_seconds: int) -> None:
    status = _quiet_call(stage, job.wait, timeout=timeout_seconds)
    if not _quiet_call(stage, _status_success, status):
        raise AiHubAdapterError(
            f"{stage} stage did not succeed; private service details suppressed"
        )


def _safe_tensor_specs(specs: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(specs, Mapping):
        return normalized
    for graph_name, tensors in specs.items():
        graph = "__default__" if graph_name is None else graph_name
        if not isinstance(graph, str) or not SAFE_TENSOR_NAME_PATTERN.fullmatch(graph):
            raise AiHubAdapterError("target model exposed an unsafe graph name")
        if not isinstance(tensors, Sequence) or isinstance(tensors, (str, bytes)):
            raise AiHubAdapterError("target model exposed malformed tensor specs")
        for tensor in tensors:
            name = getattr(tensor, "name", None)
            shape = getattr(tensor, "shape", None)
            dtype = getattr(tensor, "dtype", None)
            if (
                not isinstance(name, str)
                or not SAFE_TENSOR_NAME_PATTERN.fullmatch(name)
                or not isinstance(shape, Sequence)
                or isinstance(shape, (str, bytes))
            ):
                raise AiHubAdapterError("target model exposed malformed tensor specs")
            dimensions = [
                _positive_int(dimension, "target tensor dimension")
                for dimension in shape
            ]
            dtype_text = str(dtype)
            if dtype_text not in ALLOWED_DTYPES:
                raise AiHubAdapterError("target model exposed an unsupported dtype")
            item: dict[str, Any] = {
                "graph": graph,
                "name": name,
                "shape": dimensions,
                "dtype": dtype_text,
            }
            scale = getattr(tensor, "scale", None)
            zero_point = getattr(tensor, "zero_point", None)
            if scale is not None:
                if not isinstance(scale, (int, float)):
                    raise AiHubAdapterError("target tensor scale is malformed")
                item["scale"] = scale
            if zero_point is not None:
                if not isinstance(zero_point, int):
                    raise AiHubAdapterError("target tensor zero_point is malformed")
                item["zero_point"] = zero_point
            normalized.append(item)
    return normalized


def _target_model(stage: str, job: Any) -> Any:
    target = _quiet_call(stage, job.get_target_model)
    if target is None:
        raise AiHubAdapterError("compile stage returned no target model")
    return target


def _read_target_io(target: Any) -> dict[str, Any]:
    return {
        "inputs": _safe_tensor_specs(getattr(target, "input_spec", None)),
        "outputs": _safe_tensor_specs(getattr(target, "output_spec", None)),
    }


def _target_io(stage: str, target: Any) -> dict[str, Any]:
    return _quiet_call(stage, _read_target_io, target)


def _observed_device(device: Any) -> dict[str, Any]:
    name = _safe_text(getattr(device, "name", None), "observed device.name")
    os_value = getattr(device, "os", "")
    os_text = _safe_optional_text(os_value, "observed device.os")
    raw_attributes = getattr(device, "attributes", [])
    if isinstance(raw_attributes, str):
        raw_attributes = [raw_attributes]
    if not isinstance(raw_attributes, list) or any(
        not isinstance(item, str) for item in raw_attributes
    ):
        raise AiHubAdapterError("service exposed malformed device attributes")
    attributes = [
        _safe_text(item, "observed device.attributes") for item in raw_attributes
    ]
    return {
        "name": name,
        "os": os_text or None,
        "attributes": attributes,
    }


def _metadata_key_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.upper()
    if isinstance(value, str):
        return value.rsplit(".", 1)[-1].upper()
    return ""


def _artifact_runtime(model: Any) -> dict[str, str] | None:
    if model is None:
        return None
    metadata = getattr(model, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    for key, value in metadata.items():
        if _metadata_key_name(key) in {"QAIRT_SDK_VERSION", "QNN_SDK_VERSION"}:
            return {
                "name": "QAIRT",
                "version": _safe_exact_version(value, "target model runtime metadata"),
            }
    return None


def _read_service_evidence(
    job: Any,
    *,
    expected_options: str,
    runtime: Mapping[str, str],
    target_model: Any | None,
) -> dict[str, Any]:
    actual_options = getattr(job, "options", None)
    validated_options = _validate_options(actual_options, runtime)
    if validated_options != expected_options:
        raise AiHubAdapterError("service job options differ from submitted options")
    observed_device = _observed_device(getattr(job, "device", None))
    model = target_model if target_model is not None else getattr(job, "model", None)
    artifact_runtime = _artifact_runtime(model)
    return {
        "device": observed_device,
        "artifact_runtime": artifact_runtime,
    }


def _service_evidence(
    stage: str,
    job: Any,
    *,
    common: Mapping[str, Any],
    target_model: Any | None = None,
) -> dict[str, Any]:
    return _quiet_call(
        stage,
        _read_service_evidence,
        job,
        expected_options=common["options"],
        runtime=common["runtime"],
        target_model=target_model,
    )


def _download_compile_target(stage: str, job: Any, output: Path) -> Path:
    returned = _quiet_call(stage, job.download_target_model, str(output))
    if not isinstance(returned, (str, os.PathLike)):
        raise AiHubAdapterError("compile stage returned no target artifact")
    downloaded = Path(returned)
    if not downloaded.is_file():
        raise AiHubAdapterError("compile target artifact was not downloaded")
    if downloaded.resolve().parent != output.resolve().parent:
        raise AiHubAdapterError(
            "compile target artifact escaped private output storage"
        )
    return downloaded


def _download_inference_output(stage: str, job: Any, output: Path) -> Path:
    returned = _quiet_call(stage, job.download_output_data, str(output))
    if not isinstance(returned, (str, os.PathLike)):
        raise AiHubAdapterError("inference stage returned no output artifact")
    downloaded = Path(returned)
    if not downloaded.is_file():
        raise AiHubAdapterError("inference output artifact was not downloaded")
    if downloaded.resolve().parent != output.resolve().parent:
        raise AiHubAdapterError("inference output escaped private output storage")
    return downloaded


def _artifact_result(path: Path, role: str, logical_name: str) -> dict[str, Any]:
    try:
        byte_size = path.stat().st_size
    except OSError:
        raise AiHubAdapterError(f"{role} artifact metadata is unavailable") from None
    return {
        "role": role,
        "logical_name": _safe_logical_name(logical_name, f"{role}.logical_name"),
        "sha256": sha256_file(path),
        "byte_size": byte_size,
    }


def _privacy_contract() -> dict[str, bool]:
    return {
        "credentials_committed": False,
        "job_ids_committed": False,
        "job_urls_committed": False,
        "account_identifiers_committed": False,
        "raw_service_responses_committed": False,
        "private_paths_committed": False,
    }


def _base_manifest(
    stage: str,
    common: Mapping[str, Any],
    public_request: Mapping[str, Any],
    source_artifacts: list[dict[str, Any]],
    predecessor_sha: str | None,
    service_turnaround_seconds: float,
    service_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_runtime = service_evidence["artifact_runtime"]
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": MANIFEST_TYPE,
        "stage": stage,
        "request_id": _request_id(stage, public_request),
        "observed_at_utc": _observed_at(),
        "status": "success",
        "client": {"name": "qai-hub", "version": common["client_version"]},
        "target": {
            "requested": common["device"],
            "observed": service_evidence["device"],
            "observed_evidence": "successful_service_job_device",
            "exact_request_observation_match_required": False,
        },
        "runtime": {
            "requested": common["runtime"],
            "request_evidence": "successful_service_job_options",
            "artifact": artifact_runtime,
            "artifact_evidence": (
                "target_model_metadata"
                if artifact_runtime is not None
                else "not_exposed_by_target_model_metadata"
            ),
            "observed_execution": None,
            "observed_execution_evidence": "not_exposed_by_stage_adapter",
        },
        "submission": {
            "options": common["options"],
            "retry": common["retry"],
            "timeout_seconds": common["timeout_seconds"],
            "service_turnaround_seconds": round(service_turnaround_seconds, 6),
            "service_turnaround_is_device_latency": False,
            "external_job_reference": "private_not_committed",
        },
        "lineage": {
            "predecessor_manifest_sha256": predecessor_sha,
            "source_artifacts": source_artifacts,
        },
        "privacy": _privacy_contract(),
    }


def _verify_backend_version(common: Mapping[str, Any], backend: AiHubBackend) -> None:
    try:
        actual = _quiet_call(
            "client version check",
            lambda: backend.client_version,
        )
    except AiHubAdapterError:
        raise AiHubAdapterError(
            "qai-hub client version could not be read; private details suppressed"
        ) from None
    if actual != common["client_version"]:
        raise AiHubAdapterError(
            "installed qai-hub client version does not match request"
        )


def run_compile(
    request: Mapping[str, Any],
    *,
    backend: AiHubBackend,
) -> dict[str, Any]:
    _require_exact_keys(
        request,
        required={
            "schema_version",
            "stage",
            "client_version",
            "device",
            "runtime",
            "source_artifact",
            "output_artifact",
            "output_logical_name",
            "input_specs",
            "options",
            "job_name",
            "timeout_seconds",
        },
        optional={"retry"},
        field="compile request",
    )
    if request["schema_version"] != SCHEMA_VERSION or request["stage"] != "compile":
        raise AiHubAdapterError("compile request has wrong schema or stage")
    common = _common_request(request, "compile")
    source_path, source = _artifact_from_request(
        request["source_artifact"], role="source_model"
    )
    public_specs, sdk_specs = _input_specs(request["input_specs"])
    output = _private_output_path(request["output_artifact"], "output_artifact")
    output_logical_name = _safe_logical_name(
        request["output_logical_name"], "output_logical_name"
    )
    public_request = {
        **_public_request_projection(request),
        "source_artifact": source,
        "input_specs": public_specs,
    }
    _assert_public_safe(public_request, "compile request")
    _verify_backend_version(common, backend)

    started = time.monotonic()
    job = _quiet_call(
        "compile",
        backend.submit_compile,
        model=source_path,
        device=common["device"],
        input_specs=sdk_specs,
        options=common["options"],
        name=common["job_name"],
        retry=common["retry"],
    )
    _wait_success("compile", job, common["timeout_seconds"])
    target_model = _target_model("compile", job)
    graph_io = _target_io("compile", target_model)
    service_evidence = _service_evidence(
        "compile", job, common=common, target_model=target_model
    )
    downloaded = _download_compile_target("compile", job, output)
    turnaround = time.monotonic() - started

    manifest = _base_manifest(
        "compile",
        common,
        public_request,
        [source],
        None,
        turnaround,
        service_evidence,
    )
    manifest["graph_contract"] = {"input_specs": public_specs, "target_io": graph_io}
    manifest["result"] = {
        "target_artifact": _artifact_result(
            downloaded, "compiled_model", output_logical_name
        )
    }
    _assert_public_safe(manifest)
    return manifest


def _load_predecessor(
    value: Any,
) -> tuple[Path, Mapping[str, Any], str, dict[str, Any]]:
    if not isinstance(value, str) or not value:
        raise AiHubAdapterError("predecessor_manifest must be a nonempty path")
    path = Path(value).expanduser()
    manifest = _load_json(path, "predecessor manifest")
    _assert_public_safe(manifest, "predecessor manifest")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("manifest_type") != MANIFEST_TYPE
        or manifest.get("stage") != "compile"
        or manifest.get("status") != "success"
    ):
        raise AiHubAdapterError("predecessor manifest is not a successful compile")
    result = _require_mapping(manifest.get("result"), "predecessor result")
    artifact = _require_mapping(
        result.get("target_artifact"), "predecessor target artifact"
    )
    expected_keys = {"role", "logical_name", "sha256", "byte_size"}
    _require_exact_keys(
        artifact,
        required=expected_keys,
        field="predecessor target artifact",
    )
    sha = artifact.get("sha256")
    if not isinstance(sha, str) or not SHA256_PATTERN.fullmatch(sha):
        raise AiHubAdapterError("predecessor target artifact digest is invalid")
    return path, manifest, sha256_file(path), dict(artifact)


def _compiled_artifact(
    value: Any,
    predecessor_artifact: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    artifact = _require_mapping(value, "compiled_artifact")
    _require_exact_keys(
        artifact,
        required={"path", "logical_name", "sha256"},
        field="compiled_artifact",
    )
    if artifact.get("logical_name") != predecessor_artifact.get(
        "logical_name"
    ) or artifact.get("sha256") != predecessor_artifact.get("sha256"):
        raise AiHubAdapterError("compiled artifact does not match predecessor manifest")
    return _artifact_from_request(artifact, role="compiled_model")


def run_inference(
    request: Mapping[str, Any],
    *,
    backend: AiHubBackend,
) -> dict[str, Any]:
    _require_exact_keys(
        request,
        required={
            "schema_version",
            "stage",
            "client_version",
            "device",
            "runtime",
            "predecessor_manifest",
            "compiled_artifact",
            "input_dataset",
            "output_artifact",
            "output_logical_name",
            "options",
            "job_name",
            "timeout_seconds",
        },
        optional={"retry"},
        field="inference request",
    )
    if request["schema_version"] != SCHEMA_VERSION or request["stage"] != "inference":
        raise AiHubAdapterError("inference request has wrong schema or stage")
    common = _common_request(request, "inference")
    _, _, predecessor_sha, predecessor_artifact = _load_predecessor(
        request["predecessor_manifest"]
    )
    model_path, compiled = _compiled_artifact(
        request["compiled_artifact"], predecessor_artifact
    )
    dataset_path, dataset = _artifact_from_request(
        request["input_dataset"], role="input_dataset"
    )
    output = _private_output_path(request["output_artifact"], "output_artifact")
    output_logical_name = _safe_logical_name(
        request["output_logical_name"], "output_logical_name"
    )
    public_request = {
        **_public_request_projection(request),
        "compiled_artifact": compiled,
        "input_dataset": dataset,
        "predecessor_manifest_sha256": predecessor_sha,
    }
    _assert_public_safe(public_request, "inference request")
    _verify_backend_version(common, backend)

    started = time.monotonic()
    job = _quiet_call(
        "inference",
        backend.submit_inference,
        model=model_path,
        device=common["device"],
        inputs=dataset_path,
        options=common["options"],
        name=common["job_name"],
        retry=common["retry"],
    )
    _wait_success("inference", job, common["timeout_seconds"])
    service_evidence = _service_evidence("inference", job, common=common)
    downloaded = _download_inference_output("inference", job, output)
    turnaround = time.monotonic() - started

    manifest = _base_manifest(
        "inference",
        common,
        public_request,
        [compiled, dataset],
        predecessor_sha,
        turnaround,
        service_evidence,
    )
    manifest["result"] = {
        "output_artifact": _artifact_result(
            downloaded, "inference_output", output_logical_name
        ),
        "numerical_validation": "deferred_to_backend_parity_task",
    }
    _assert_public_safe(manifest)
    return manifest


def _find_profile_value(profile: Mapping[str, Any], key: str) -> Any:
    if key in profile:
        return profile[key]
    for container in ("execution_summary", "summary", "profile_summary"):
        child = profile.get(container)
        if isinstance(child, Mapping) and key in child:
            return child[key]
    return None


def _optional_nonnegative(profile: Mapping[str, Any], key: str) -> int | None:
    value = _find_profile_value(profile, key)
    if value is None:
        return None
    return _nonnegative_int(value, f"profile.{key}")


def _all_compute_units(value: Any) -> set[str]:
    units: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"compute_unit", "compute_units", "unit"}:
                candidates = child if isinstance(child, list) else [child]
                for candidate in candidates:
                    if isinstance(candidate, str):
                        upper = candidate.upper()
                        if upper in SAFE_COMPUTE_UNITS:
                            units.add(upper)
            else:
                units.update(_all_compute_units(child))
    elif isinstance(value, list):
        for child in value:
            units.update(_all_compute_units(child))
    return units


def _warning_count(value: Any) -> int:
    count = 0
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key.lower() in {"warning", "warnings"}:
                count += (
                    len(child) if isinstance(child, list) else int(child is not None)
                )
            else:
                count += _warning_count(child)
    elif isinstance(value, list):
        count += sum(_warning_count(child) for child in value)
    return count


def normalize_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    inference_time = _find_profile_value(profile, "estimated_inference_time")
    if inference_time is None:
        inference_time = _find_profile_value(profile, "execution_time")
    memory = _find_profile_value(profile, "estimated_inference_peak_memory")
    estimated_us = _nonnegative_int(inference_time, "profile.estimated_inference_time")
    memory_bytes = _nonnegative_int(memory, "profile.estimated_inference_peak_memory")
    all_times = _find_profile_value(profile, "all_inference_times")
    if all_times is None:
        sample_count = None
        observed_range = None
    else:
        if (
            not isinstance(all_times, list)
            or not all_times
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in all_times
            )
        ):
            raise AiHubAdapterError("profile inference samples are malformed")
        sample_count = len(all_times)
        observed_range = [min(all_times), max(all_times)]
    compute_units = sorted(_all_compute_units(profile))
    return {
        "latency": {
            "estimated_inference_time_us": estimated_us,
            "estimated_inference_time_ms": estimated_us / 1000,
            "inference_sample_count": sample_count,
            "observed_inference_time_range_us": observed_range,
        },
        "memory": {
            "estimated_inference_peak_memory_bytes": memory_bytes,
        },
        "load": {
            "first_load_time_us": _optional_nonnegative(profile, "first_load_time"),
            "first_load_peak_memory_bytes": _optional_nonnegative(
                profile, "first_load_peak_memory"
            ),
            "warm_load_time_us": _optional_nonnegative(profile, "warm_load_time"),
            "warm_load_peak_memory_bytes": _optional_nonnegative(
                profile, "warm_load_peak_memory"
            ),
        },
        "placement": {
            "compute_units": compute_units,
            "evidence": "profile_detail" if compute_units else "not_exposed",
        },
        "warnings": {
            "count": _warning_count(profile),
            "raw_text_committed": False,
        },
        "units": {
            "time": "microseconds",
            "memory": "bytes",
            "source": "qualcomm_ai_hub_profile_contract",
        },
    }


def _write_private_profile(path: Path, profile: Mapping[str, Any]) -> None:
    try:
        path.write_text(
            json.dumps(profile, sort_keys=True, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    except OSError:
        raise AiHubAdapterError("raw profile could not be written") from None


def run_profile(
    request: Mapping[str, Any],
    *,
    backend: AiHubBackend,
) -> dict[str, Any]:
    _require_exact_keys(
        request,
        required={
            "schema_version",
            "stage",
            "client_version",
            "device",
            "runtime",
            "predecessor_manifest",
            "compiled_artifact",
            "raw_profile_output",
            "raw_profile_logical_name",
            "options",
            "job_name",
            "timeout_seconds",
        },
        optional={"retry"},
        field="profile request",
    )
    if request["schema_version"] != SCHEMA_VERSION or request["stage"] != "profile":
        raise AiHubAdapterError("profile request has wrong schema or stage")
    common = _common_request(request, "profile")
    _, _, predecessor_sha, predecessor_artifact = _load_predecessor(
        request["predecessor_manifest"]
    )
    model_path, compiled = _compiled_artifact(
        request["compiled_artifact"], predecessor_artifact
    )
    raw_output = _private_output_path(
        request["raw_profile_output"], "raw_profile_output"
    )
    raw_logical_name = _safe_logical_name(
        request["raw_profile_logical_name"], "raw_profile_logical_name"
    )
    public_request = {
        **_public_request_projection(request),
        "compiled_artifact": compiled,
        "predecessor_manifest_sha256": predecessor_sha,
    }
    _assert_public_safe(public_request, "profile request")
    _verify_backend_version(common, backend)

    started = time.monotonic()
    job = _quiet_call(
        "profile",
        backend.submit_profile,
        model=model_path,
        device=common["device"],
        options=common["options"],
        name=common["job_name"],
        retry=common["retry"],
    )
    _wait_success("profile", job, common["timeout_seconds"])
    service_evidence = _service_evidence("profile", job, common=common)
    profile = _quiet_call("profile", job.download_profile)
    if not isinstance(profile, Mapping):
        raise AiHubAdapterError("profile stage returned a malformed result")
    _write_private_profile(raw_output, profile)
    normalized = normalize_profile(profile)
    turnaround = time.monotonic() - started

    manifest = _base_manifest(
        "profile",
        common,
        public_request,
        [compiled],
        predecessor_sha,
        turnaround,
        service_evidence,
    )
    manifest["result"] = {
        "raw_profile_artifact": _artifact_result(
            raw_output, "raw_profile", raw_logical_name
        ),
        "normalized_profile": normalized,
    }
    _assert_public_safe(manifest)
    return manifest


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> str:
    _assert_public_safe(manifest)
    payload = _canonical_bytes(manifest)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    except OSError:
        raise AiHubAdapterError("sanitized manifest could not be written") from None
    return _sha256_bytes(payload)


def execute_request(
    stage: str,
    request_path: Path,
    manifest_path: Path,
    *,
    backend: AiHubBackend | None = None,
) -> dict[str, Any]:
    request = load_request(request_path, stage)
    selected_backend = backend if backend is not None else QaiHubBackend()
    runners = {
        "compile": run_compile,
        "inference": run_inference,
        "profile": run_profile,
    }
    manifest = runners[stage](request, backend=selected_backend)
    manifest_sha = write_manifest(manifest_path, manifest)
    return {
        "stage": stage,
        "status": "success",
        "request_id": manifest["request_id"],
        "manifest_sha256": manifest_sha,
    }


class _SafeArgumentParser(argparse.ArgumentParser):
    """Keep accidental private CLI values out of argparse diagnostics."""

    def error(self, message: str) -> None:
        del message
        self.print_usage(sys.stderr)
        self.exit(2, "adapter arguments are invalid\n")


def _parser(stage: str) -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description=f"Run one sanitized Qualcomm AI Hub {stage} stage."
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def stage_main(
    stage: str,
    argv: list[str] | None = None,
    *,
    backend: AiHubBackend | None = None,
) -> int:
    args = _parser(stage).parse_args(argv)
    try:
        summary = execute_request(
            stage,
            args.request,
            args.manifest,
            backend=backend,
        )
    except AiHubAdapterError as exc:
        print(f"{stage} adapter failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0
