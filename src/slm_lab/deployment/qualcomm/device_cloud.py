"""Sanitize and validate a Qualcomm Device Cloud generation-loop capture.

The live session input is deliberately private.  It may contain filesystem
paths or raw runtime logs, so the public manifest is constructed from a small,
closed contract and never copies unknown fields.  This module does not launch
or authenticate a Device Cloud session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
MANIFEST_TYPE = "slm_lab.qualcomm.device_cloud.generation"
PRIVATE_REFERENCE = "private_not_committed"
FIXED_PROMPT = (
    "Reply with five consecutive integers beginning at 41, separated by spaces."
)
FIXED_PROMPT_SHA256 = "e36ded0e32a5d70a5b1c3d36d4e625ef98377475295d568b05b69d4719cfa055"
TIMING_COMPONENTS = (
    "artifact_load",
    "model_load",
    "tokenization",
    "prefill",
    "first_decode",
    "decode",
    "generation_total",
    "request_total",
)
TIMING_SOURCES = {
    "geniex_runtime_report",
    "instrumented_host_clock",
    "derived_from_runtime_counters",
}
DEVICE_EVIDENCE_KINDS = {"windows_system_information"}
PLACEMENT_EVIDENCE_KINDS = {"geniex_runtime_log"}
SYNCHRONIZATION_EVIDENCE_KINDS = {
    "instrumented_runtime_trace",
    "timestamped_runtime_log",
}
PRE_TIMER_ACTIONS = {"host_clock_immediately_before_runtime_call"}
POST_TIMER_ACTIONS = {"runtime_completion_before_host_clock_read"}
PRIVATE_TEXT_PATTERNS = (
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b(?:access|api|auth|bearer|refresh|session)[_-]?token\b", re.I),
    re.compile(r"\b(?:password|secret)\s*[:=]", re.IGNORECASE),
    re.compile(r"\b(?:session|job|account)[_-]?id\b", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\"),
    re.compile(r"(?:^|\s)/(?:home|Users|private|tmp|var)/"),
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXACT_VERSION_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)+(?:[A-Za-z0-9._+-]*)$")
SAFE_TEXT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._()+,:=@-]{0,199}$")
RFC3339_UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


class DeviceCloudCaptureError(RuntimeError):
    """A public-safe validation error for a private Device Cloud capture."""


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DeviceCloudCaptureError(f"{field} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    field: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    keys = set(value)
    missing = required - keys
    extra = keys - required - optional
    if missing:
        raise DeviceCloudCaptureError(
            f"{field} is missing required fields: {', '.join(sorted(missing))}"
        )
    if extra:
        raise DeviceCloudCaptureError(
            f"{field} contains unsupported fields: {', '.join(sorted(extra))}"
        )


def _safe_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SAFE_TEXT_PATTERN.fullmatch(value):
        raise DeviceCloudCaptureError(f"{field} contains unsupported text")
    if any(pattern.search(value) for pattern in PRIVATE_TEXT_PATTERNS):
        raise DeviceCloudCaptureError(f"{field} contains private or path-like text")
    return value


def _exact_version(value: Any, field: str) -> str:
    if not isinstance(value, str) or not EXACT_VERSION_PATTERN.fullmatch(value):
        raise DeviceCloudCaptureError(f"{field} must be an exact version")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise DeviceCloudCaptureError(f"{field} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DeviceCloudCaptureError(f"{field} must be an integer >= {minimum}")
    return value


def _nonnegative_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeviceCloudCaptureError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        raise DeviceCloudCaptureError(f"{field} must be a finite number >= 0") from None
    if not math.isfinite(result) or result < 0:
        raise DeviceCloudCaptureError(f"{field} must be a finite number >= 0")
    return result


def _utc_timestamp(value: Any, field: str) -> str:
    if not isinstance(value, str) or not RFC3339_UTC_PATTERN.fullmatch(value):
        raise DeviceCloudCaptureError(f"{field} must be a strict RFC3339 UTC timestamp")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise DeviceCloudCaptureError(
            f"{field} must be a strict RFC3339 UTC timestamp"
        ) from None
    return value


def _private_reference(value: Any, field: str) -> str:
    if value != PRIVATE_REFERENCE:
        raise DeviceCloudCaptureError(
            f"{field} must be the sanitized private evidence reference"
        )
    return PRIVATE_REFERENCE


def _timing(value: Any, field: str) -> dict[str, Any]:
    timing = _require_mapping(value, field)
    _require_exact_keys(
        timing,
        field,
        required={
            "milliseconds",
            "source",
            "evidence_sha256",
            "private_reference",
        },
    )
    source = timing["source"]
    if source not in TIMING_SOURCES:
        raise DeviceCloudCaptureError(
            f"{field}.source must be one of {', '.join(sorted(TIMING_SOURCES))}"
        )
    return {
        "milliseconds": _nonnegative_number(
            timing["milliseconds"], f"{field}.milliseconds"
        ),
        "source": source,
        "evidence_sha256": _sha256(
            timing["evidence_sha256"], f"{field}.evidence_sha256"
        ),
        "private_reference": _private_reference(
            timing["private_reference"], f"{field}.private_reference"
        ),
    }


def _normalize_device(value: Any) -> dict[str, Any]:
    device = _require_mapping(value, "device")
    _require_exact_keys(
        device,
        "device",
        required={
            "product",
            "form_factor",
            "catalog_code",
            "os",
            "chipset",
            "memory_bytes",
            "evidence_kind",
            "evidence_sha256",
            "private_reference",
        },
    )
    if device["product"] != "Snapdragon X Elite":
        raise DeviceCloudCaptureError(
            "device.product must identify Snapdragon X Elite exactly"
        )
    if device["form_factor"] != "Compute Reference Design":
        raise DeviceCloudCaptureError(
            "device.form_factor must be Compute Reference Design"
        )
    if device["evidence_kind"] not in DEVICE_EVIDENCE_KINDS:
        raise DeviceCloudCaptureError("device.evidence_kind is not supported")
    return {
        "product": device["product"],
        "form_factor": device["form_factor"],
        "catalog_code": _safe_text(device["catalog_code"], "device.catalog_code"),
        "os": _safe_text(device["os"], "device.os"),
        "chipset": _safe_text(device["chipset"], "device.chipset"),
        "memory_bytes": _positive_int(device["memory_bytes"], "device.memory_bytes"),
        "evidence_kind": device["evidence_kind"],
        "evidence_sha256": _sha256(device["evidence_sha256"], "device.evidence_sha256"),
        "private_reference": _private_reference(
            device["private_reference"], "device.private_reference"
        ),
    }


def _normalize_runtime(value: Any) -> dict[str, Any]:
    runtime = _require_mapping(value, "runtime")
    _require_exact_keys(
        runtime,
        "runtime",
        required={
            "geniex_version",
            "route",
            "compute_selection",
            "placement",
        },
    )
    if runtime["route"] != "llama_cpp":
        raise DeviceCloudCaptureError(
            "runtime.route must be llama_cpp for the T32 ready-made baseline"
        )
    if runtime["compute_selection"] != "npu":
        raise DeviceCloudCaptureError(
            "runtime.compute_selection must be npu for the pinned T32 baseline"
        )
    placement = _require_mapping(runtime["placement"], "runtime.placement")
    _require_exact_keys(
        placement,
        "runtime.placement",
        required={
            "status",
            "compute_unit",
            "backend",
            "device_id",
            "evidence_kind",
            "evidence_sha256",
            "private_reference",
        },
    )
    if placement["status"] != "observed":
        raise DeviceCloudCaptureError("runtime.placement.status must be observed")
    if (
        placement["compute_unit"] != "NPU"
        or placement["backend"] != "HTP"
        or placement["device_id"] != "HTP0"
    ):
        raise DeviceCloudCaptureError(
            "runtime.placement must affirm observed NPU compute on HTP0"
        )
    if placement["evidence_kind"] not in PLACEMENT_EVIDENCE_KINDS:
        raise DeviceCloudCaptureError(
            "runtime.placement.evidence_kind is not supported"
        )
    return {
        "name": "GenieX",
        "version": _exact_version(runtime["geniex_version"], "runtime.geniex_version"),
        "route": runtime["route"],
        "compute_selection": runtime["compute_selection"],
        "placement": {
            "status": "observed",
            "compute_unit": "NPU",
            "backend": "HTP",
            "device_id": "HTP0",
            "evidence_kind": placement["evidence_kind"],
            "evidence_sha256": _sha256(
                placement["evidence_sha256"],
                "runtime.placement.evidence_sha256",
            ),
            "private_reference": _private_reference(
                placement["private_reference"],
                "runtime.placement.private_reference",
            ),
        },
    }


def _normalize_model(value: Any) -> dict[str, Any]:
    model = _require_mapping(value, "model")
    _require_exact_keys(
        model,
        "model",
        required={
            "logical_name",
            "source",
            "source_version",
            "asset_runtime",
            "precision",
            "artifact_sha256",
            "private_reference",
        },
    )
    expected = {
        "logical_name": "Qwen3-0.6B",
        "source": "Qualcomm AI Hub Models",
        "source_version": "0.58.0",
        "asset_runtime": "geniex_llamacpp",
        "precision": "Q4_0",
    }
    for field, expected_value in expected.items():
        if model[field] != expected_value:
            raise DeviceCloudCaptureError(f"model.{field} must be {expected_value!r}")
    return {
        **expected,
        "artifact_sha256": _sha256(model["artifact_sha256"], "model.artifact_sha256"),
        "private_reference": _private_reference(
            model["private_reference"], "model.private_reference"
        ),
    }


def _normalize_generation(value: Any) -> dict[str, Any]:
    generation = _require_mapping(value, "generation")
    _require_exact_keys(
        generation,
        "generation",
        required={
            "prompt_sha256",
            "prompt_private_reference",
            "prompt_tokens",
            "output_sha256",
            "output_private_reference",
            "output_tokens",
            "finish_reason",
            "valid_multi_token_output_confirmed",
        },
    )
    output_tokens = _positive_int(
        generation["output_tokens"], "generation.output_tokens", minimum=2
    )
    if generation["valid_multi_token_output_confirmed"] is not True:
        raise DeviceCloudCaptureError(
            "generation.valid_multi_token_output_confirmed must be true"
        )
    finish_reason = generation["finish_reason"]
    if finish_reason not in {"eos", "stop", "max_tokens"}:
        raise DeviceCloudCaptureError(
            "generation.finish_reason must be eos, stop, or max_tokens"
        )
    prompt_sha256 = _sha256(generation["prompt_sha256"], "generation.prompt_sha256")
    if prompt_sha256 != FIXED_PROMPT_SHA256:
        raise DeviceCloudCaptureError(
            "generation.prompt_sha256 does not match the pinned normalized prompt"
        )
    return {
        "prompt_contract": "t32_fixed_prompt_utf8_nfc_no_trailing_newline_v1",
        "prompt_sha256": prompt_sha256,
        "prompt_private_reference": _private_reference(
            generation["prompt_private_reference"],
            "generation.prompt_private_reference",
        ),
        "prompt_tokens": _positive_int(
            generation["prompt_tokens"], "generation.prompt_tokens"
        ),
        "output_sha256": _sha256(
            generation["output_sha256"], "generation.output_sha256"
        ),
        "output_private_reference": _private_reference(
            generation["output_private_reference"],
            "generation.output_private_reference",
        ),
        "output_tokens": output_tokens,
        "finish_reason": finish_reason,
        "valid_multi_token_output_confirmed": True,
    }


def _normalize_timings(value: Any) -> dict[str, Any]:
    timings = _require_mapping(value, "timings")
    _require_exact_keys(
        timings,
        "timings",
        required=set(TIMING_COMPONENTS),
    )
    result = {name: _timing(timings[name], f"timings.{name}") for name in timings}
    generation_total = result["generation_total"]["milliseconds"]
    generation_components = sum(
        result[name]["milliseconds"] for name in ("prefill", "first_decode", "decode")
    )
    if generation_total + 0.001 < generation_components:
        raise DeviceCloudCaptureError(
            "timings.generation_total cannot be smaller than "
            "prefill + first_decode + decode"
        )
    request_total = result["request_total"]["milliseconds"]
    request_components = (
        sum(
            result[name]["milliseconds"]
            for name in ("artifact_load", "model_load", "tokenization")
        )
        + generation_total
    )
    if request_total + 0.001 < request_components:
        raise DeviceCloudCaptureError(
            "timings.request_total cannot be smaller than load + tokenization "
            "+ generation_total"
        )
    return result


def normalize_capture(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Return a sanitized public manifest from a complete private capture."""

    capture = _require_mapping(capture, "capture")
    _require_exact_keys(
        capture,
        "capture",
        required={
            "schema_version",
            "observed_at",
            "device",
            "runtime",
            "model",
            "generation",
            "timings",
            "synchronization",
            "cost",
        },
    )
    if (
        isinstance(capture["schema_version"], bool)
        or not isinstance(capture["schema_version"], int)
        or capture["schema_version"] != SCHEMA_VERSION
    ):
        raise DeviceCloudCaptureError(
            f"capture.schema_version must be {SCHEMA_VERSION}"
        )

    synchronization = _require_mapping(capture["synchronization"], "synchronization")
    _require_exact_keys(
        synchronization,
        "synchronization",
        required={
            "backend",
            "method_id",
            "pre_timer_action",
            "post_timer_action",
            "evidence_kind",
            "evidence_sha256",
            "private_reference",
        },
    )
    if synchronization["backend"] != "qualcomm_device_cloud":
        raise DeviceCloudCaptureError(
            "synchronization.backend must be qualcomm_device_cloud"
        )
    if synchronization["method_id"] != "runtime_completion_fence":
        raise DeviceCloudCaptureError(
            "synchronization.method_id must be runtime_completion_fence"
        )
    if synchronization["pre_timer_action"] not in PRE_TIMER_ACTIONS:
        raise DeviceCloudCaptureError(
            "synchronization.pre_timer_action is not supported"
        )
    if synchronization["post_timer_action"] not in POST_TIMER_ACTIONS:
        raise DeviceCloudCaptureError(
            "synchronization.post_timer_action is not supported"
        )
    if synchronization["evidence_kind"] not in SYNCHRONIZATION_EVIDENCE_KINDS:
        raise DeviceCloudCaptureError("synchronization.evidence_kind is not supported")

    cost = _require_mapping(capture["cost"], "cost")
    _require_exact_keys(
        cost,
        "cost",
        required={"paid_resources_used", "cost_usd"},
    )
    cost_usd = _nonnegative_number(cost["cost_usd"], "cost.cost_usd")
    if cost["paid_resources_used"] is not False or cost_usd != 0:
        raise DeviceCloudCaptureError(
            "T32 capture must use no paid resources and report cost_usd 0"
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": MANIFEST_TYPE,
        "status": "completed",
        "observed_at": _utc_timestamp(capture["observed_at"], "observed_at"),
        "measurement_scope": "persistent_device_side_generation_loop",
        "device": _normalize_device(capture["device"]),
        "runtime": _normalize_runtime(capture["runtime"]),
        "model": _normalize_model(capture["model"]),
        "generation": _normalize_generation(capture["generation"]),
        "timings": _normalize_timings(capture["timings"]),
        "synchronization": {
            "backend": synchronization["backend"],
            "method_id": synchronization["method_id"],
            "pre_timer_action": synchronization["pre_timer_action"],
            "post_timer_action": synchronization["post_timer_action"],
            "evidence_kind": synchronization["evidence_kind"],
            "evidence_sha256": _sha256(
                synchronization["evidence_sha256"],
                "synchronization.evidence_sha256",
            ),
            "private_reference": _private_reference(
                synchronization["private_reference"],
                "synchronization.private_reference",
            ),
        },
        "cost": {"paid_resources_used": False, "cost_usd": 0},
        "provenance": {
            "service": "Qualcomm Device Cloud",
            "external_session_reference": PRIVATE_REFERENCE,
            "hosted_graph_latency_included": False,
            "raw_logs_committed": False,
            "private_identifiers_committed": False,
        },
        "claim_boundaries": {
            "proves": (
                "Qwen3-0.6B generated multiple tokens in a persistent GenieX "
                "device-side loop on the observed Snapdragon X Elite"
            ),
            "does_not_prove": (
                "custom QNN static graph compilation or Workbench graph latency"
            ),
        },
    }
    _assert_public_safe(manifest)
    return manifest


def _assert_public_safe(value: Any, field: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise DeviceCloudCaptureError(f"{field} has a non-string key")
            _assert_public_safe(child, f"{field}.{key}")
    elif isinstance(value, list):
        for child in value:
            _assert_public_safe(child, field)
    elif isinstance(value, str):
        if any(pattern.search(value) for pattern in PRIVATE_TEXT_PATTERNS):
            raise DeviceCloudCaptureError(f"{field} contains private or path-like text")


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Return the digest of the canonical public manifest."""

    return hashlib.sha256(_canonical_bytes(manifest)).hexdigest()


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write a normalized manifest without leaking parser errors or input data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        raise DeviceCloudCaptureError(
            "sanitized Device Cloud manifest could not be written"
        ) from None


def _load_capture(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise DeviceCloudCaptureError(
            "private Device Cloud capture could not be read"
        ) from None
    return _require_mapping(value, "capture")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a sanitized manifest from a private T32 capture."
    )
    parser.add_argument(
        "--capture",
        required=True,
        type=Path,
        help="private JSON capture; keep it under ignored .ai-local storage",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="sanitized public JSON manifest",
    )
    args = parser.parse_args(argv)
    try:
        manifest = normalize_capture(_load_capture(args.capture))
        write_manifest(args.manifest, manifest)
    except DeviceCloudCaptureError as error:
        parser.exit(2, f"error: {error}\n")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest_sha256": manifest_sha256(manifest),
                "output_tokens": manifest["generation"]["output_tokens"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
