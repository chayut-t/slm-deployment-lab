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

EXPECTED_MODEL_ID = "Qwen/Qwen3-0.6B"
EXPECTED_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
EXPECTED_HOST_MODEL = "Mac16,10"
EXPECTED_HOST_CHIP = "Apple M4"
EXPECTED_MEMORY_BYTES = 16 * 1024**3
EXPECTED_GENERATED_TOKEN_IDS = [576, 8356, 3950]
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


def _write_evidence(path: Path, document: dict[str, Any]) -> None:
    if "evidence_sha256" in document:
        raise MlxBaselineError("evidence digest is reserved for the writer")
    document["evidence_sha256"] = _canonical_json_sha256(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_evidence(path: Path) -> None:
    """Validate the self-digest and T50 claim boundaries in one result."""

    document = _load_json(path)
    stored_digest = document.pop("evidence_sha256", None)
    if stored_digest != _canonical_json_sha256(document):
        raise MlxBaselineError(f"{path}: evidence_sha256 does not match content")
    if document.get("task_id") != "T50" or document.get("schema_version") != 1:
        raise MlxBaselineError(f"{path}: unexpected task or schema version")
    claim = document.get("execution_claim", "")
    if not (
        "does not establish Apple Neural Engine (ANE) execution" in claim
        or "not Apple Neural Engine (ANE) execution evidence" in claim
    ):
        raise MlxBaselineError(f"{path}: explicit no-ANE boundary is missing")

    report_kind = document.get("report_kind")
    if report_kind is None:
        host = document.get("host", {})
        runtime = document.get("runtime", {})
        if (
            host.get("target_model_identifier") != EXPECTED_HOST_MODEL
            or host.get("target_chip") != EXPECTED_HOST_CHIP
            or host.get("target_memory_bytes") != EXPECTED_MEMORY_BYTES
        ):
            raise MlxBaselineError(f"{path}: exact Apple M4 host identity failed")
        if runtime.get("packages") != _read_pinned_requirements():
            raise MlxBaselineError(f"{path}: runtime package pins differ")
        if runtime.get("requirements_sha256") != _file_sha256(DEFAULT_REQUIREMENTS):
            raise MlxBaselineError(f"{path}: requirements checksum differs")
    elif report_kind == "mlx_lm_parity":
        canaries = document.get("tokenizer_canaries", [])
        generation = document.get("generation_canary", {})
        if (
            document.get("passed") is not True
            or document.get("tokenizer_canaries_passed") is not True
            or len(canaries) != 5
            or not all(canary.get("exact_match") is True for canary in canaries)
            or generation.get("expected_generated_token_ids")
            != EXPECTED_GENERATED_TOKEN_IDS
            or generation.get("actual_generated_token_ids")
            != EXPECTED_GENERATED_TOKEN_IDS
            or generation.get("exact_token_match") is not True
        ):
            raise MlxBaselineError(f"{path}: frozen parity canaries failed")
    elif report_kind == "mlx_lm_baseline_performance":
        policy = document.get("measurement_policy", {})
        samples = document.get("samples", [])
        if (
            policy.get("warmup_repetitions") != 2
            or policy.get("measured_repetitions") != 10
            or len(samples) != 10
        ):
            raise MlxBaselineError(f"{path}: repetition policy differs from T50")
        for sample in samples:
            values = (
                sample.get("time_to_first_token_seconds"),
                sample.get("generation_loop_seconds"),
                sample.get("generation_tokens_per_second_including_prefill"),
                sample.get("mlx_peak_memory_bytes"),
            )
            if sample.get("generated_token_ids") != EXPECTED_GENERATED_TOKEN_IDS or any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in values
            ):
                raise MlxBaselineError(f"{path}: invalid measured sample")
    else:
        raise MlxBaselineError(f"{path}: unknown report_kind {report_kind!r}")


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
        "schema_version": 1,
        "task_id": "T50",
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
        "execution_claim": (
            "MLX default GPU device on Apple M4. MLX/Metal evidence does not "
            "establish Apple Neural Engine (ANE) execution."
        ),
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


def _run_generation(
    *,
    mx: Any,
    generate_step: Any,
    model: Any,
    prompt_token_ids: Sequence[int],
    max_new_tokens: int,
) -> tuple[list[int], float, float, int]:
    mx.synchronize()
    mx.reset_peak_memory()
    started = time.perf_counter_ns()
    generator = generate_step(
        mx.array(prompt_token_ids),
        model,
        max_tokens=max_new_tokens,
    )
    try:
        first_token, _ = next(generator)
    except StopIteration as exc:
        raise MlxBaselineError("MLX-LM returned no generated token") from exc
    first_token_ready = time.perf_counter_ns()
    generated = [int(first_token)]
    generated.extend(int(token) for token, _ in generator)
    mx.synchronize()
    finished = time.perf_counter_ns()
    if len(generated) != max_new_tokens:
        raise MlxBaselineError(
            f"expected {max_new_tokens} generated tokens, found {len(generated)}"
        )
    return (
        generated,
        (first_token_ready - started) / 1e9,
        (finished - started) / 1e9,
        int(mx.get_peak_memory()),
    )


def run_baseline(
    *,
    model_path: Path,
    output_dir: Path,
    warmup_repetitions: int,
    measured_repetitions: int,
) -> tuple[Path, Path, Path]:
    """Run the pinned correctness canary and warm MLX-LM generation baseline."""

    if warmup_repetitions < 1 or measured_repetitions < 1:
        raise MlxBaselineError(
            "at least one warm-up and measured repetition is required"
        )

    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import generate_step
    except ImportError as exc:
        raise MlxBaselineError(
            "install the exact environments/macos-m4 MLX baseline requirements"
        ) from exc
    if not mx.metal.is_available():
        raise MlxBaselineError("the MLX Metal backend is unavailable")

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
    correctness_tokens, _, _, _ = _run_generation(
        mx=mx,
        generate_step=generate_step,
        model=model,
        prompt_token_ids=prompt_ids,
        max_new_tokens=len(expected_generated),
    )
    generation_passed = correctness_tokens == expected_generated
    if not all_tokenizers_passed or not generation_passed:
        raise MlxBaselineError("one or more frozen T10/T11 canaries failed")

    for _ in range(warmup_repetitions):
        _run_generation(
            mx=mx,
            generate_step=generate_step,
            model=model,
            prompt_token_ids=prompt_ids,
            max_new_tokens=len(expected_generated),
        )

    samples = []
    for repetition in range(measured_repetitions):
        generated, ttft, total, peak_memory = _run_generation(
            mx=mx,
            generate_step=generate_step,
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
                "time_to_first_token_seconds": ttft,
                "generation_loop_seconds": total,
                "generation_tokens_per_second_including_prefill": (
                    len(generated) / total
                ),
                "mlx_peak_memory_bytes": peak_memory,
                "generated_token_ids": generated,
            }
        )
    process_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    common = {
        "schema_version": 1,
        "task_id": "T50",
        "model": model_identity,
        "runner": {
            "path": "src/slm_lab/backends/mlx_baseline.py",
            "sha256": _file_sha256(Path(__file__)),
        },
        "execution_claim": (
            "MLX-LM generation on the MLX Metal GPU backend; this is not "
            "Apple Neural Engine (ANE) execution evidence."
        ),
    }
    host_document = _collect_host_runtime(mx)
    parity_document = {
        **common,
        "report_kind": "mlx_lm_parity",
        "tokenizer_canaries": tokenizer_evidence,
        "tokenizer_canaries_passed": all_tokenizers_passed,
        "generation_canary": {
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
        "passed": all_tokenizers_passed and generation_passed,
    }
    ttft_values = [sample["time_to_first_token_seconds"] for sample in samples]
    total_values = [sample["generation_loop_seconds"] for sample in samples]
    throughput_values = [
        sample["generation_tokens_per_second_including_prefill"] for sample in samples
    ]
    memory_values = [float(sample["mlx_peak_memory_bytes"]) for sample in samples]
    performance_document = {
        **common,
        "report_kind": "mlx_lm_baseline_performance",
        "workload": {
            "fixture_id": t11_fixture["source_fixture"]["id"],
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(expected_generated),
            "batch_size": 1,
            "model_loading_included": False,
        },
        "measurement_policy": {
            "timing_class": "generation_loop",
            "warmup_repetitions": warmup_repetitions,
            "measured_repetitions": measured_repetitions,
            "clock": "time.perf_counter_ns",
            "synchronization": (
                "mx.synchronize before each run and after the complete generation "
                "loop; MLX-LM token extraction evaluates the first yielded token"
            ),
            "lazy_execution_boundary": (
                "The time-to-first-token boundary is the first token returned by "
                "mlx_lm.generate_step. MLX-LM may already enqueue the next decode, "
                "so this baseline does not claim isolated prefill latency."
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
            "time_to_first_token_seconds": _summarize(ttft_values),
            "generation_loop_seconds": _summarize(total_values),
            "generation_tokens_per_second_including_prefill": _summarize(
                throughput_values
            ),
            "mlx_peak_memory_bytes": _summarize(memory_values),
        },
        "process_peak_rss_bytes": int(process_peak_rss),
        "memory_methods": {
            "mlx_peak_memory": (
                "mx.reset_peak_memory before each run and mx.get_peak_memory "
                "after synchronization; includes resident model allocations"
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
        ],
    }

    host_path = output_dir / "host-runtime-v1.json"
    parity_path = output_dir / "mlx-lm-parity-v1.json"
    performance_path = output_dir / "mlx-lm-performance-v1.json"
    _write_evidence(host_path, host_document)
    _write_evidence(parity_path, parity_document)
    _write_evidence(performance_path, performance_document)
    for path in (host_path, parity_path, performance_path):
        validate_evidence(path)
    return host_path, parity_path, performance_path


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
