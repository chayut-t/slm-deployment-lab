"""Reproducible MLX-LM correctness and performance baseline for Apple M4.

The module intentionally imports MLX only when the baseline is executed. That
keeps the repository's normal development environment platform-neutral while
allowing T50 to use an exact, separately pinned macOS environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_CONTRACT = PROJECT_ROOT / "configs/models/qwen3-0.6b.yaml"
DEFAULT_T10_FIXTURES = PROJECT_ROOT / "tests/fixtures/t10/token-fixtures-v1.json"
DEFAULT_T11_FIXTURE = (
    PROJECT_ROOT / "tests/reference/fixtures/qwen3-0.6b-raw-ascii-bf16-cpu-v1.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/raw/apple/baseline"
DEFAULT_REQUIREMENTS = (
    PROJECT_ROOT / "environments/macos-m4/mlx-baseline-requirements.txt"
)
DEFAULT_RESULT_SCHEMA = (
    PROJECT_ROOT / "environments/macos-m4/mlx-baseline-run-v2.schema.json"
)
DEFAULT_BENCHMARK_PROTOCOL = (
    PROJECT_ROOT / "configs/workloads/benchmark-protocol-v1.json"
)

EXPECTED_MODEL_ID = "Qwen/Qwen3-0.6B"
EXPECTED_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
EXPECTED_PROTOCOL_ID = "slm-lab-benchmark-v1"
EXPECTED_PROTOCOL_SHA256 = (
    "2541fa76fb088de3ebb559aeb300aed5cd62e215994b8db0faa2fbc6273f947e"
)
EXPECTED_HOST_MODEL = "Mac16,10"
EXPECTED_HOST_CHIP = "Apple M4"
EXPECTED_MEMORY_BYTES = 16 * 1024**3
EXPECTED_GENERATED_TOKEN_IDS = [576, 8356, 3950]
EXPECTED_WARMUP_REPETITIONS = 2
EXPECTED_MEASURED_REPETITIONS = 10
EXPECTED_LOOKAHEAD_TOKENS = 1
EXPECTED_MODEL_SOURCE_FILES = {
    "config.json": {
        "sha256": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
        "size_bytes": 726,
    },
    "generation_config.json": {
        "sha256": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
        "size_bytes": 239,
    },
    "model.safetensors": {
        "sha256": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
        "size_bytes": 1503300328,
    },
    "tokenizer.json": {
        "sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "size_bytes": 11422654,
    },
    "tokenizer_config.json": {
        "sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        "size_bytes": 9732,
    },
}
REQUIRED_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


class MlxBaselineError(RuntimeError):
    """The MLX baseline cannot produce trustworthy evidence."""


def _canonical_json_sha256(document: Any) -> str:
    canonical = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MlxBaselineError(f"cannot load JSON object {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise MlxBaselineError(f"{path} must contain one JSON object")
    return document


def _write_evidence(path: Path, document: dict[str, Any]) -> Path:
    if "evidence_sha256" in document:
        raise MlxBaselineError("evidence digest is reserved for the writer")
    document["evidence_sha256"] = _canonical_json_sha256(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    anchor_path = path.with_suffix(path.suffix + ".sha256")
    anchor_path.write_text(
        f"{document['evidence_sha256']}  {path.name}\n",
        encoding="utf-8",
    )
    return anchor_path


def validate_evidence(path: Path) -> None:
    """Validate schema, immutable provenance, and cross-field run semantics."""

    document = _load_json(path)
    stored_digest = document.pop("evidence_sha256", None)
    if stored_digest != _canonical_json_sha256(document):
        raise MlxBaselineError(f"{path}: evidence_sha256 does not match content")
    document["evidence_sha256"] = stored_digest
    anchor_path = path.with_suffix(path.suffix + ".sha256")
    try:
        anchored_digest, anchored_name = (
            anchor_path.read_text(encoding="utf-8").strip().split("  ", maxsplit=1)
        )
    except (OSError, ValueError) as exc:
        raise MlxBaselineError(f"{path}: invalid external digest anchor") from exc
    if anchored_digest != stored_digest or anchored_name != path.name:
        raise MlxBaselineError(f"{path}: external digest anchor differs")

    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:
        raise MlxBaselineError(
            "jsonschema is required to validate T50 evidence"
        ) from exc
    schema = _load_json(DEFAULT_RESULT_SCHEMA)
    try:
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(
                schema,
                format_checker=FormatChecker(),
            ).iter_errors(document),
            key=lambda error: list(error.absolute_path),
        )
    except Exception as exc:
        raise MlxBaselineError(f"cannot validate T50 result schema: {exc}") from exc
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "<root>"
        raise MlxBaselineError(f"{path}: schema failure at {location}: {first.message}")

    if document["schema"]["sha256"] != _file_sha256(DEFAULT_RESULT_SCHEMA):
        raise MlxBaselineError(f"{path}: result schema checksum differs")
    if document["runtime"]["requirements_sha256"] != _file_sha256(DEFAULT_REQUIREMENTS):
        raise MlxBaselineError(f"{path}: requirements checksum differs")
    if document["runtime"]["packages"] != _read_pinned_requirements():
        raise MlxBaselineError(f"{path}: runtime package pins differ")
    if document["model"]["source_files"] != EXPECTED_MODEL_SOURCE_FILES:
        raise MlxBaselineError(f"{path}: model source evidence differs")

    protocol = _load_json(DEFAULT_BENCHMARK_PROTOCOL)
    protocol_digest = _protocol_sha256(protocol)
    if (
        protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID
        or protocol.get("contract_sha256") != EXPECTED_PROTOCOL_SHA256
        or protocol_digest != EXPECTED_PROTOCOL_SHA256
        or document["protocol"]["sha256"] != EXPECTED_PROTOCOL_SHA256
    ):
        raise MlxBaselineError(f"{path}: benchmark protocol provenance differs")

    expected_contracts = _fixture_contracts()
    if document["fixture_contracts"] != expected_contracts:
        raise MlxBaselineError(f"{path}: fixture contract provenance differs")
    if document["canary"]["tokenizer_canaries"] != _expected_tokenizer_canaries():
        raise MlxBaselineError(f"{path}: tokenizer canary evidence differs")

    commit = document["source_git_commit"]
    runner = document["runner"]
    if _git_blob_sha256(commit, runner["path"]) != runner["sha256"]:
        raise MlxBaselineError(f"{path}: runner does not match the source commit")
    schema_record = document["schema"]
    if _git_blob_sha256(commit, schema_record["path"]) != schema_record["sha256"]:
        raise MlxBaselineError(f"{path}: schema does not match the source commit")
    requirements_path = document["runtime"]["requirements_path"]
    if (
        _git_blob_sha256(commit, requirements_path)
        != document["runtime"]["requirements_sha256"]
    ):
        raise MlxBaselineError(f"{path}: requirements do not match the source commit")
    for fixture in document["fixture_contracts"].values():
        if _git_blob_sha256(commit, fixture["path"]) != fixture["sha256"]:
            raise MlxBaselineError(f"{path}: fixture does not match the source commit")
    committed_protocol = _git_blob_json(commit, document["protocol"]["path"])
    if _protocol_sha256(committed_protocol) != document["protocol"]["sha256"]:
        raise MlxBaselineError(f"{path}: protocol does not match the source commit")

    compact_time = document["created_at"].replace("-", "").replace(":", "")
    expected_run_id = f"t50-mlx-lm-{compact_time}-{commit[:12]}"
    if document["run_id"] != expected_run_id:
        raise MlxBaselineError(f"{path}: run ID differs from timestamp/source commit")

    validate_repetition_policy(
        document["measurement_policy"]["warmup_repetitions"],
        document["measurement_policy"]["measured_repetitions"],
    )
    samples = document["samples"]
    if [sample["repetition"] for sample in samples] != list(
        range(1, EXPECTED_MEASURED_REPETITIONS + 1)
    ):
        raise MlxBaselineError(f"{path}: repetition indices are not contiguous")
    for sample in samples:
        if sample["generated_token_ids"] != EXPECTED_GENERATED_TOKEN_IDS:
            raise MlxBaselineError(f"{path}: measured generation canary drifted")
        expected_throughput = (
            len(EXPECTED_GENERATED_TOKEN_IDS) / sample["generation_loop_seconds"]
        )
        if not math.isclose(
            sample["returned_output_tokens_per_second_including_prefill_and_lookahead"],
            expected_throughput,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise MlxBaselineError(f"{path}: throughput is inconsistent with latency")

    summary_sources = {
        "ttft_seconds": [sample["ttft_seconds"] for sample in samples],
        "generation_loop_seconds": [
            sample["generation_loop_seconds"] for sample in samples
        ],
        "returned_output_tokens_per_second_including_prefill_and_lookahead": [
            sample["returned_output_tokens_per_second_including_prefill_and_lookahead"]
            for sample in samples
        ],
        "ttft_mlx_peak_memory_bytes": [
            float(sample["ttft_mlx_peak_memory_bytes"]) for sample in samples
        ],
        "generation_loop_mlx_peak_memory_bytes": [
            float(sample["generation_loop_mlx_peak_memory_bytes"]) for sample in samples
        ],
    }
    expected_summary = {
        key: _summarize(values) for key, values in summary_sources.items()
    }
    if document["summary"] != expected_summary:
        raise MlxBaselineError(f"{path}: summary does not match raw samples")


def validate_repetition_policy(
    warmup_repetitions: int,
    measured_repetitions: int,
) -> None:
    """Enforce the frozen T13 generation-loop repetition policy."""

    if (
        warmup_repetitions != EXPECTED_WARMUP_REPETITIONS
        or measured_repetitions != EXPECTED_MEASURED_REPETITIONS
    ):
        raise MlxBaselineError(
            "T50 requires exactly 2 warm-up and 10 measured repetitions"
        )


def _protocol_sha256(protocol: Mapping[str, Any]) -> str:
    hashed = dict(protocol)
    hashed.pop("contract_sha256", None)
    return _canonical_json_sha256(hashed)


def _fixture_contracts() -> dict[str, dict[str, str]]:
    return {
        "model": {
            "path": "configs/models/qwen3-0.6b.yaml",
            "sha256": _file_sha256(DEFAULT_MODEL_CONTRACT),
        },
        "t10": {
            "path": "tests/fixtures/t10/token-fixtures-v1.json",
            "sha256": _file_sha256(DEFAULT_T10_FIXTURES),
        },
        "t11": {
            "path": ("tests/reference/fixtures/qwen3-0.6b-raw-ascii-bf16-cpu-v1.json"),
            "sha256": _file_sha256(DEFAULT_T11_FIXTURE),
        },
    }


def _expected_tokenizer_canaries() -> list[dict[str, Any]]:
    fixtures = _load_json(DEFAULT_T10_FIXTURES)
    records = [*fixtures["raw_canaries"], fixtures["chat_canary"]]
    return [
        {
            "id": record["id"],
            "interface": record["interface"],
            "expected_token_count": len(record["token_ids"]),
            "actual_token_count": len(record["token_ids"]),
            "expected_token_ids_sha256": record["token_ids_sha256"],
            "actual_token_ids_sha256": record["token_ids_sha256"],
            "exact_match": True,
        }
        for record in records
    ]


def _git_blob_sha256(commit: str, path: str) -> str:
    return hashlib.sha256(_git_blob_bytes(commit, path)).hexdigest()


def _git_blob_bytes(commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise MlxBaselineError(
            f"cannot read {path} from source commit {commit}: "
            f"{completed.stderr.decode(errors='replace').strip()}"
        )
    return completed.stdout


def _git_blob_json(commit: str, path: str) -> dict[str, Any]:
    try:
        document = json.loads(_git_blob_bytes(commit, path))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MlxBaselineError(
            f"cannot parse {path} from source commit {commit}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise MlxBaselineError(f"{path} from source commit {commit} is not an object")
    return document


def _clean_source_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise MlxBaselineError(
            "run evidence only from a clean committed worktree; found:\n" + status
        )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise MlxBaselineError(f"required package {name!r} is not installed") from exc


def _read_pinned_requirements() -> dict[str, str]:
    packages = {}
    for line in DEFAULT_REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        if requirement.count("==") != 1:
            raise MlxBaselineError(f"requirement is not exactly pinned: {requirement}")
        name, expected = requirement.split("==")
        packages[name] = expected
    if not packages:
        raise MlxBaselineError("the MLX requirements file contains no packages")
    return packages


def _validate_pinned_environment() -> dict[str, Any]:
    packages = _read_pinned_requirements()
    for name, expected in packages.items():
        actual = _package_version(name)
        if actual != expected:
            raise MlxBaselineError(
                f"installed {name}=={actual}, expected exact version {expected}"
            )
    return {
        "requirements_path": "environments/macos-m4/mlx-baseline-requirements.txt",
        "requirements_sha256": _file_sha256(DEFAULT_REQUIREMENTS),
        "packages": packages,
    }


def _command_result(command: Sequence[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {
            "value": None,
            "check_command": " ".join(command),
            "unavailable_reason": str(exc),
        }
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode:
        return {
            "value": None,
            "check_command": " ".join(command),
            "unavailable_reason": output or f"exit status {completed.returncode}",
        }
    return {
        "value": output,
        "check_command": " ".join(command),
        "unavailable_reason": None,
    }


def _collect_host_runtime(mx: Any) -> dict[str, Any]:
    profiler = _command_result(("system_profiler", "SPHardwareDataType", "-json"))
    if profiler["value"] is None:
        raise MlxBaselineError(
            "system_profiler is required for exact Apple host identification"
        )
    try:
        hardware = json.loads(profiler["value"])["SPHardwareDataType"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise MlxBaselineError("cannot parse system_profiler hardware data") from exc

    model_identifier = hardware.get("machine_model")
    chip = hardware.get("chip_type")
    memory = hardware.get("physical_memory")
    if model_identifier != EXPECTED_HOST_MODEL or chip != EXPECTED_HOST_CHIP:
        raise MlxBaselineError(
            "T50 measurements are restricted to the expected Mac16,10 Apple M4"
        )
    if memory != "16 GB":
        raise MlxBaselineError(f"expected 16 GB physical memory, found {memory!r}")

    os_version = _command_result(("sw_vers", "-productVersion"))
    os_build = _command_result(("sw_vers", "-buildVersion"))
    if os_version["value"] is None or os_build["value"] is None:
        raise MlxBaselineError("sw_vers is required for exact macOS identity")

    device_info = dict(mx.device_info())
    if device_info.get("device_name") != EXPECTED_HOST_CHIP:
        raise MlxBaselineError("MLX device identity differs from host chip")

    command_line_tools = _command_result(
        ("pkgutil", "--pkg-info", "com.apple.pkg.CLTools_Executables")
    )
    clt_version = None
    if command_line_tools["value"]:
        for line in command_line_tools["value"].splitlines():
            if line.startswith("version: "):
                clt_version = line.removeprefix("version: ").strip()
                break

    xcode = _command_result(("xcodebuild", "-version"))
    metal_compiler = _command_result(("xcrun", "metal", "-v"))
    instruments = _command_result(("xcrun", "instruments", "-s", "devices"))
    runtime = {
        "python_version": platform.python_version(),
        **_validate_pinned_environment(),
    }
    return {
        "host": {
            "target_product_name": hardware.get("machine_name"),
            "target_model_identifier": model_identifier,
            "target_model_number": hardware.get("model_number"),
            "target_chip": chip,
            "target_memory_bytes": EXPECTED_MEMORY_BYTES,
            "processor_topology": hardware.get("number_processors"),
            "architecture": platform.machine(),
            "macos_version": os_version["value"],
            "macos_build": os_build["value"],
            "darwin_release": platform.release(),
        },
        "mlx_device": {
            "default_device": str(mx.default_device()),
            "device_name": device_info.get("device_name"),
            "architecture": device_info.get("architecture"),
            "memory_size_bytes": device_info.get("memory_size"),
            "max_recommended_working_set_size_bytes": device_info.get(
                "max_recommended_working_set_size"
            ),
            "max_buffer_length_bytes": device_info.get("max_buffer_length"),
        },
        "runtime": runtime,
        "developer_tools": {
            "command_line_tools_version": clt_version,
            "command_line_tools_check": command_line_tools,
            "xcode_version": xcode,
            "metal_compiler_version": metal_compiler,
            "instruments_version": instruments,
        },
    }


def _validate_model_identity(
    model_path: Path,
    model_contract: Mapping[str, Any],
) -> dict[str, Any]:
    missing = [
        name for name in REQUIRED_MODEL_FILES if not (model_path / name).is_file()
    ]
    if missing:
        raise MlxBaselineError(f"model directory is missing files: {missing}")
    model = model_contract.get("model", {})
    tokenizer = model_contract.get("tokenizer", {})
    if (
        model.get("id") != EXPECTED_MODEL_ID
        or model.get("revision") != EXPECTED_MODEL_REVISION
        or tokenizer.get("revision") != EXPECTED_MODEL_REVISION
    ):
        raise MlxBaselineError(
            "model contract differs from the immutable Qwen revision"
        )

    upstream_config = _load_json(model_path / "config.json")
    if upstream_config.get("model_type") != "qwen3":
        raise MlxBaselineError("model config is not Qwen3")
    expected_config_hash = model_contract["source_metadata"]["config_json_sha256"]
    if _file_sha256(model_path / "config.json") != expected_config_hash:
        raise MlxBaselineError("model config checksum differs from the contract")
    expected_tokenizer_hash = model_contract["source_metadata"][
        "tokenizer_config_json_sha256"
    ]
    if _file_sha256(model_path / "tokenizer_config.json") != expected_tokenizer_hash:
        raise MlxBaselineError("tokenizer config checksum differs from the contract")

    return {
        "id": EXPECTED_MODEL_ID,
        "revision": EXPECTED_MODEL_REVISION,
        "checkpoint_dtype": upstream_config.get("torch_dtype"),
        "weights_format": model.get("weights_format"),
        "trust_remote_code": model.get("trust_remote_code"),
        "source_files": {
            name: {
                "sha256": _file_sha256(model_path / name),
                "size_bytes": (model_path / name).stat().st_size,
            }
            for name in REQUIRED_MODEL_FILES
        },
    }


def _tokenizer_canaries(
    tokenizer: Any,
    fixtures: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    records = [*fixtures["raw_canaries"], fixtures["chat_canary"]]
    evidence = []
    for record in records:
        prompt = record.get("prompt", record.get("rendered_prompt"))
        actual = tokenizer.encode(prompt, add_special_tokens=False)
        actual_ids = [int(token_id) for token_id in actual]
        expected_ids = record["token_ids"]
        passed = (
            actual_ids == expected_ids
            and _canonical_json_sha256(actual_ids) == record["token_ids_sha256"]
        )
        evidence.append(
            {
                "id": record["id"],
                "interface": record["interface"],
                "expected_token_count": len(expected_ids),
                "actual_token_count": len(actual_ids),
                "expected_token_ids_sha256": record["token_ids_sha256"],
                "actual_token_ids_sha256": _canonical_json_sha256(actual_ids),
                "exact_match": passed,
            }
        )
    return evidence, all(record["exact_match"] for record in evidence)


def _summarize(values: Sequence[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise MlxBaselineError("measurement samples must be finite and non-negative")
    ordered = sorted(values)

    def quantile(probability: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        fraction = position - lower
        return ordered[lower] + fraction * (ordered[upper] - ordered[lower])

    median = quantile(0.5)
    deviations = sorted(abs(value - median) for value in ordered)
    if len(deviations) == 1:
        mad = deviations[0]
    else:
        position = (len(deviations) - 1) * 0.5
        lower = math.floor(position)
        upper = math.ceil(position)
        mad = deviations[lower] + (position - lower) * (
            deviations[upper] - deviations[lower]
        )
    return {
        "sample_count_valid": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "sample_standard_deviation": (
            statistics.stdev(values) if len(values) > 1 else 0.0
        ),
        "median": median,
        "p90": quantile(0.9),
        "p95": quantile(0.95),
        "median_absolute_deviation": mad,
        "interquartile_range": quantile(0.75) - quantile(0.25),
    }


def _synchronize_generation_stream(mx: Any, generation_stream: Any) -> None:
    mx.synchronize(generation_stream)


def _measure_ttft(
    *,
    mx: Any,
    generate_step: Any,
    generation_stream: Any,
    model: Any,
    prompt_token_ids: Sequence[int],
) -> tuple[float, int]:
    """Measure first-token materialization without MLX-LM look-ahead."""

    _synchronize_generation_stream(mx, generation_stream)
    mx.reset_peak_memory()
    started = time.perf_counter_ns()
    generator = generate_step(
        mx.array(prompt_token_ids),
        model,
        max_tokens=0,
    )
    try:
        next(generator)
    except StopIteration:
        pass
    else:
        raise MlxBaselineError("max_tokens=0 unexpectedly yielded a token")
    _synchronize_generation_stream(mx, generation_stream)
    finished = time.perf_counter_ns()
    return (finished - started) / 1e9, int(mx.get_peak_memory())


def _measure_generation_loop(
    *,
    mx: Any,
    generate_step: Any,
    generation_stream: Any,
    model: Any,
    prompt_token_ids: Sequence[int],
    max_new_tokens: int,
) -> tuple[list[int], float, int]:
    """Measure MLX-LM's returned tokens plus its one scheduled look-ahead."""

    _synchronize_generation_stream(mx, generation_stream)
    mx.reset_peak_memory()
    started = time.perf_counter_ns()
    generated = [
        int(token)
        for token, _ in generate_step(
            mx.array(prompt_token_ids),
            model,
            max_tokens=max_new_tokens,
        )
    ]
    _synchronize_generation_stream(mx, generation_stream)
    finished = time.perf_counter_ns()
    if len(generated) != max_new_tokens:
        raise MlxBaselineError(
            f"expected {max_new_tokens} generated tokens, found {len(generated)}"
        )
    return generated, (finished - started) / 1e9, int(mx.get_peak_memory())


def run_baseline(
    *,
    model_path: Path,
    output_dir: Path,
    warmup_repetitions: int,
    measured_repetitions: int,
) -> tuple[Path, Path]:
    """Run the pinned correctness canary and warm MLX-LM generation baseline."""

    validate_repetition_policy(warmup_repetitions, measured_repetitions)
    source_commit = _clean_source_commit()

    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import generate_step, generation_stream
    except ImportError as exc:
        raise MlxBaselineError(
            "install the exact environments/macos-m4 MLX baseline requirements"
        ) from exc
    if not mx.metal.is_available():
        raise MlxBaselineError("the MLX Metal backend is unavailable")

    _validate_pinned_environment()
    schema = _load_json(DEFAULT_RESULT_SCHEMA)
    protocol = _load_json(DEFAULT_BENCHMARK_PROTOCOL)
    if (
        protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID
        or protocol.get("contract_sha256") != EXPECTED_PROTOCOL_SHA256
        or _protocol_sha256(protocol) != EXPECTED_PROTOCOL_SHA256
    ):
        raise MlxBaselineError("the frozen benchmark protocol failed validation")
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise MlxBaselineError(f"invalid T50 result schema: {exc}") from exc

    model_contract = _load_json(DEFAULT_MODEL_CONTRACT)
    t10_fixtures = _load_json(DEFAULT_T10_FIXTURES)
    t11_fixture = _load_json(DEFAULT_T11_FIXTURE)
    model_identity = _validate_model_identity(model_path, model_contract)

    mx.reset_peak_memory()
    process_rss_before_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    load_started = time.perf_counter_ns()
    model, tokenizer, loaded_config = load(
        str(model_path),
        lazy=False,
        return_config=True,
    )
    mx.synchronize()
    load_finished = time.perf_counter_ns()
    model_load_seconds = (load_finished - load_started) / 1e9
    process_rss_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    load_peak_memory = int(mx.get_peak_memory())

    if loaded_config.get("model_type") != "qwen3":
        raise MlxBaselineError("MLX-LM loaded a non-Qwen3 model")
    tokenizer_evidence, all_tokenizers_passed = _tokenizer_canaries(
        tokenizer, t10_fixtures
    )

    prompt_ids = [int(token_id) for token_id in t11_fixture["prompt_token_ids"]]
    expected_generated = [
        int(token_id) for token_id in t11_fixture["generated_token_ids"]
    ]
    correctness_tokens, _, _ = _measure_generation_loop(
        mx=mx,
        generate_step=generate_step,
        generation_stream=generation_stream,
        model=model,
        prompt_token_ids=prompt_ids,
        max_new_tokens=len(expected_generated),
    )
    generation_passed = correctness_tokens == expected_generated
    if not all_tokenizers_passed or not generation_passed:
        raise MlxBaselineError("one or more frozen T10/T11 canaries failed")

    for _ in range(warmup_repetitions):
        _measure_ttft(
            mx=mx,
            generate_step=generate_step,
            generation_stream=generation_stream,
            model=model,
            prompt_token_ids=prompt_ids,
        )
        _measure_generation_loop(
            mx=mx,
            generate_step=generate_step,
            generation_stream=generation_stream,
            model=model,
            prompt_token_ids=prompt_ids,
            max_new_tokens=len(expected_generated),
        )

    samples = []
    for repetition in range(measured_repetitions):
        ttft, ttft_peak_memory = _measure_ttft(
            mx=mx,
            generate_step=generate_step,
            generation_stream=generation_stream,
            model=model,
            prompt_token_ids=prompt_ids,
        )
        generated, total, generation_peak_memory = _measure_generation_loop(
            mx=mx,
            generate_step=generate_step,
            generation_stream=generation_stream,
            model=model,
            prompt_token_ids=prompt_ids,
            max_new_tokens=len(expected_generated),
        )
        if generated != expected_generated:
            raise MlxBaselineError(
                f"measured repetition {repetition} drifted from the canary"
            )
        samples.append(
            {
                "repetition": repetition + 1,
                "ttft_seconds": ttft,
                "generation_loop_seconds": total,
                (
                    "returned_output_tokens_per_second_including_prefill_and_lookahead"
                ): len(generated) / total,
                "ttft_mlx_peak_memory_bytes": ttft_peak_memory,
                "generation_loop_mlx_peak_memory_bytes": generation_peak_memory,
                "generated_token_ids": generated,
                "scheduled_lookahead_tokens": EXPECTED_LOOKAHEAD_TOKENS,
            }
        )
    process_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    ttft_values = [sample["ttft_seconds"] for sample in samples]
    total_values = [sample["generation_loop_seconds"] for sample in samples]
    throughput_values = [
        sample["returned_output_tokens_per_second_including_prefill_and_lookahead"]
        for sample in samples
    ]
    ttft_memory_values = [
        float(sample["ttft_mlx_peak_memory_bytes"]) for sample in samples
    ]
    generation_memory_values = [
        float(sample["generation_loop_mlx_peak_memory_bytes"]) for sample in samples
    ]
    created_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    compact_time = created_at.replace("-", "").replace(":", "")
    host_runtime = _collect_host_runtime(mx)
    result_document = {
        "schema_version": 2,
        "schema_id": "slm-lab-t50-mlx-baseline-run-v2",
        "task_id": "T50",
        "run_id": f"t50-mlx-lm-{compact_time}-{source_commit[:12]}",
        "created_at": created_at,
        "source_git_commit": source_commit,
        "protocol": {
            "id": EXPECTED_PROTOCOL_ID,
            "path": "configs/workloads/benchmark-protocol-v1.json",
            "sha256": EXPECTED_PROTOCOL_SHA256,
        },
        "schema": {
            "path": "environments/macos-m4/mlx-baseline-run-v2.schema.json",
            "sha256": _file_sha256(DEFAULT_RESULT_SCHEMA),
        },
        "fixture_contracts": _fixture_contracts(),
        "execution_claim": (
            "MLX-LM generation on the MLX Metal GPU backend; this is not "
            "Apple Neural Engine (ANE) execution evidence."
        ),
        **host_runtime,
        "model": model_identity,
        "runner": {
            "path": "src/slm_lab/backends/mlx_baseline.py",
            "sha256": _file_sha256(Path(__file__)),
        },
        "canary": {
            "tokenizer_canaries": tokenizer_evidence,
            "tokenizer_canaries_passed": all_tokenizers_passed,
            "generation": {
                "fixture_id": t11_fixture["source_fixture"]["id"],
                "prompt_token_ids_sha256": t11_fixture["source_fixture"][
                    "token_ids_sha256"
                ],
                "reference_backend": "PyTorch CPU BF16 eager attention",
                "candidate_backend": "MLX-LM MLX Metal GPU",
                "decoding": "greedy argmax with lowest-token-ID tie break",
                "expected_generated_token_ids": expected_generated,
                "actual_generated_token_ids": correctness_tokens,
                "exact_token_match": generation_passed,
            },
        },
        "workload": {
            "id": "numerical_canary",
            "fixture_id": t11_fixture["source_fixture"]["id"],
            "prompt_tokens": len(prompt_ids),
            "returned_output_tokens": len(expected_generated),
            "scheduled_lookahead_tokens": EXPECTED_LOOKAHEAD_TOKENS,
            "batch_size": 1,
            "model_loading_included": False,
        },
        "measurement_policy": {
            "timing_class": "generation_loop",
            "warmup_repetitions": warmup_repetitions,
            "measured_repetitions": measured_repetitions,
            "clock": "time.perf_counter_ns",
            "generation_stream": "mlx_lm.generate.generation_stream",
            "pre_timer_fence": "mx.synchronize(generation_stream)",
            "post_timer_fence": "mx.synchronize(generation_stream)",
            "ttft_boundary": (
                "Fresh prompt cache through materialization of the first greedy "
                "token using generate_step(max_tokens=0); no later decode is "
                "scheduled or returned."
            ),
            "generation_loop_boundary": (
                "Fresh prompt cache through generate_step(max_tokens=3) exhaustion "
                "and a post-timer fence on MLX-LM's generation stream."
            ),
            "lookahead_accounting": (
                "Pinned mlx-lm 0.31.3 schedules one next-token look-ahead before "
                "each yield, including one unreturned fourth token after the third "
                "returned token. The fenced loop includes that compute and the "
                "throughput name states this explicitly."
            ),
            "model_load_boundary": (
                "Model loading is outside warm steady-state samples and recorded "
                "once; operating-system file cache state was uncontrolled, so the "
                "observation is not labeled cold start."
            ),
            "sample_retention": "all valid samples retained",
            "quantile_method": "Hyndman-Fan type 7 linear",
        },
        "model_load_observation": {
            "seconds": model_load_seconds,
            "mlx_peak_memory_bytes": load_peak_memory,
            "process_peak_rss_before_load_bytes": int(process_rss_before_load),
            "process_peak_rss_after_load_bytes": int(process_rss_after_load),
            "file_cache_state": "uncontrolled",
        },
        "samples": samples,
        "summary": {
            "ttft_seconds": _summarize(ttft_values),
            "generation_loop_seconds": _summarize(total_values),
            (
                "returned_output_tokens_per_second_including_prefill_and_lookahead"
            ): _summarize(throughput_values),
            "ttft_mlx_peak_memory_bytes": _summarize(ttft_memory_values),
            "generation_loop_mlx_peak_memory_bytes": _summarize(
                generation_memory_values
            ),
        },
        "process_peak_rss_bytes": int(process_peak_rss),
        "memory_methods": {
            "mlx_peak_memory": (
                "mx.reset_peak_memory before each run and mx.get_peak_memory "
                "after fencing mlx_lm.generate.generation_stream; includes "
                "resident model allocations"
            ),
            "process_peak_rss": (
                "resource.getrusage(RUSAGE_SELF).ru_maxrss on macOS; process-"
                "lifetime peak including Python and model load"
            ),
        },
        "limitations": [
            "This 18-token prompt and 3-token output is a correctness baseline, "
            "not the T52 four-context performance sweep.",
            "No Instruments, sustained power, thermal, or isolated kernel claim "
            "is made; those belong to T52.",
            "The generation-loop metric includes one MLX-LM look-ahead token "
            "that is computed but not returned.",
        ],
    }

    result_path = output_dir / "mlx-lm-baseline-run-v2.json"
    anchor_path = _write_evidence(result_path, result_document)
    validate_evidence(result_path)
    return result_path, anchor_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Local directory containing the pinned Qwen model and tokenizer files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--warmup-repetitions", type=int, default=2)
    parser.add_argument("--measured-repetitions", type=int, default=10)
    parser.add_argument(
        "--validate",
        type=Path,
        nargs="+",
        help="Validate existing T50 evidence without importing MLX",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.validate:
            for path in args.validate:
                validate_evidence(path.resolve())
            return 0
        if args.model_path is None:
            raise MlxBaselineError("--model-path is required when running a baseline")
        paths = run_baseline(
            model_path=args.model_path.resolve(),
            output_dir=args.output_dir.resolve(),
            warmup_repetitions=args.warmup_repetitions,
            measured_repetitions=args.measured_repetitions,
        )
    except MlxBaselineError as exc:
        print(f"mlx-baseline: {exc}", file=sys.stderr)
        return 2
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
