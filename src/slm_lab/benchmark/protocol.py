"""Load and enforce the T13 benchmark and evaluation contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker


PROTOCOL_PATH = Path("configs/workloads/benchmark-protocol-v1.json")
RESULT_SCHEMA_PATH = Path("configs/workloads/benchmark-result-v1.schema.json")
ACADEMIC_PATH = Path("configs/workloads/academic-evaluation-v1.json")
T10_PATH = Path("configs/workloads/t10-token-fixtures.json")

# Intentional changes to the frozen protocol must update both the JSON contract
# and this independently reviewed digest.
EXPECTED_PROTOCOL_SHA256 = (
    "2541fa76fb088de3ebb559aeb300aed5cd62e215994b8db0faa2fbc6273f947e"
)

EXPECTED_TIMING_CLASSES = {
    "single_graph": ("graph", 5, 30),
    "runtime_stage": ("runtime_stage", 5, 30),
    "generation_loop": ("generation_loop", 2, 10),
    "end_to_end_request": ("end_to_end_request", 2, 10),
    "cold_start": ("cold_start", 0, 5),
}

EXPECTED_NON_TIMING_SCOPES = {
    "quality": "evaluation",
    "numerical": "numerical_validation",
    "power_thermal": "resource_observation",
}

EXPECTED_ACADEMIC_TASKS = {
    "wikitext_2_raw": (
        "wikitext",
        "Salesforce/wikitext",
        "b08601e04326c79dfdd32d625aee71d232d685c3",
        "test",
    ),
    "hellaswag_1000": (
        "hellaswag",
        "Rowan/hellaswag",
        "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "validation",
    ),
    "arc_easy_full_validation": (
        "arc_easy",
        "allenai/ai2_arc",
        "210d026faf9955653af8916fad021475a3f00453",
        "validation",
    ),
}

EXPECTED_UNITS = {
    "graph_latency": "seconds",
    "ttft_warm": "seconds",
    "request_ttft": "seconds",
    "cold_ttft": "seconds",
    "prefill_latency": "seconds",
    "prefill_throughput": "tokens_per_second",
    "decode_latency": "seconds",
    "decode_time_per_output_token": "seconds",
    "decode_throughput": "tokens_per_second",
    "generation_throughput_including_prefill": "tokens_per_second",
    "generation_throughput_excluding_prefill": "tokens_per_second",
    "request_total_latency": "seconds",
    "artifact_load_latency": "seconds",
    "model_load_latency": "seconds",
    "peak_memory": "bytes",
    "average_power": "watts",
    "energy_per_output_token": "joules_per_token",
}

SUMMARY_FIELDS = (
    "minimum",
    "maximum",
    "mean",
    "sample_standard_deviation",
    "median",
    "p90",
    "p95",
    "median_absolute_deviation",
    "interquartile_range",
)


class BenchmarkProtocolError(ValueError):
    """A benchmark contract or result violates the frozen T13 protocol."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkProtocolError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkProtocolError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise BenchmarkProtocolError(f"{path} must contain one JSON object")
    return document


def _canonical_json_sha256(document: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def protocol_sha256(protocol: Mapping[str, Any]) -> str:
    """Hash protocol content while excluding its self-describing digest."""

    hashed = dict(protocol)
    hashed.pop("contract_sha256", None)
    return _canonical_json_sha256(hashed)


def load_protocol(root: Path | str) -> dict[str, Any]:
    """Load and validate the frozen benchmark protocol."""

    root_path = Path(root).resolve()
    protocol = _load_json(root_path / PROTOCOL_PATH)
    actual_hash = protocol_sha256(protocol)
    stored_hash = protocol.get("contract_sha256")
    if stored_hash != actual_hash:
        raise BenchmarkProtocolError(
            "benchmark protocol contract_sha256 does not match its content"
        )
    if actual_hash != EXPECTED_PROTOCOL_SHA256:
        raise BenchmarkProtocolError(
            "benchmark protocol differs from the reviewed Python digest"
        )
    _validate_protocol_semantics(protocol)
    return protocol


def _validate_protocol_semantics(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise BenchmarkProtocolError("benchmark protocol schema_version must be 1")
    if protocol.get("protocol_id") != "slm-lab-benchmark-v1":
        raise BenchmarkProtocolError("unexpected benchmark protocol_id")
    timing_classes = protocol.get("timing_classes")
    if not isinstance(timing_classes, dict):
        raise BenchmarkProtocolError("timing_classes must be an object")
    if set(timing_classes) != set(EXPECTED_TIMING_CLASSES):
        raise BenchmarkProtocolError("timing class set differs from frozen policy")
    for class_id, expected in EXPECTED_TIMING_CLASSES.items():
        record = timing_classes[class_id]
        actual = (
            record.get("scope"),
            record.get("warmup_repetitions"),
            record.get("measured_repetitions"),
        )
        if actual != expected:
            raise BenchmarkProtocolError(
                f"{class_id} timing scope/warm-up/repetitions differ from policy"
            )

    non_timing = protocol.get("non_timing_measurements")
    if not isinstance(non_timing, dict):
        raise BenchmarkProtocolError("non_timing_measurements must be an object")
    if set(non_timing) != set(EXPECTED_NON_TIMING_SCOPES):
        raise BenchmarkProtocolError("non-timing measurement set differs from v1")
    for kind, scope in EXPECTED_NON_TIMING_SCOPES.items():
        if non_timing[kind].get("scope") != scope:
            raise BenchmarkProtocolError(
                f"{kind} scope differs from the frozen policy"
            )
        if non_timing[kind].get("warmup_repetitions") != 0:
            raise BenchmarkProtocolError(
                f"{kind} must not use performance warm-up repetitions"
            )

    statistics_policy = protocol.get("statistics", {})
    if statistics_policy.get("quantile_method") != "hf_type_7_linear":
        raise BenchmarkProtocolError("quantile method must remain HF type 7")
    if statistics_policy.get("reported_quantiles") != [0.5, 0.9, 0.95]:
        raise BenchmarkProtocolError("reported quantiles must remain median/p90/p95")
    if statistics_policy.get("outlier_policy", "").startswith("Retain every") is False:
        raise BenchmarkProtocolError("valid-sample retention policy is missing")
    confidence = statistics_policy.get("confidence_interval", {})
    expected_confidence = {
        "required_for_headline_comparisons": True,
        "method": "percentile_bootstrap",
        "resamples": 10000,
        "confidence_level": 0.95,
        "seed": 20260725,
        "statistic": "median",
    }
    if confidence != expected_confidence:
        raise BenchmarkProtocolError("confidence interval policy differs from v1")

    reporting = protocol.get("reporting_policy", {})
    forbidden = reporting.get("forbidden_equivalences", [])
    if not any("graph latency" in item for item in forbidden):
        raise BenchmarkProtocolError("graph/end-to-end claim boundary is missing")
    if not any("runtime software" in item for item in forbidden):
        raise BenchmarkProtocolError("cross-platform system boundary is missing")


def _validate_workloads(
    protocol: Mapping[str, Any],
    t10: Mapping[str, Any],
) -> None:
    expected = [
        {
            "id": item["id"],
            "prompt_tokens": item["context_length"],
            "generated_tokens": item["generated_tokens"],
            "batch_size": item["batch_size"],
            "prompt_fixture_id": item["prompt_fixture_id"],
        }
        for item in t10["context_workloads"]
    ]
    if protocol.get("workloads") != expected:
        raise BenchmarkProtocolError(
            "T13 workloads differ from the frozen T10 context workloads"
        )
    decode_lengths = protocol.get("decode_probes", {}).get("cache_lengths")
    if decode_lengths != [item["prompt_tokens"] for item in expected]:
        raise BenchmarkProtocolError(
            "decode-probe cache lengths differ from the context matrix"
        )


def _validate_academic_contract(academic: Mapping[str, Any]) -> None:
    if academic.get("schema_version") != 1:
        raise BenchmarkProtocolError("academic contract schema_version must be 1")
    if academic.get("suite_id") != "slm-lab-academic-regression-v1":
        raise BenchmarkProtocolError("unexpected academic suite_id")
    if academic.get("prompt_interface") != "raw_completion":
        raise BenchmarkProtocolError("academic suite must use raw completion")
    if academic.get("apply_chat_template") is not False:
        raise BenchmarkProtocolError("academic suite must not apply a chat template")
    if academic.get("fewshot") != 0:
        raise BenchmarkProtocolError("academic suite must remain zero-shot")
    harness = academic.get("harness", {})
    if harness.get("release") != "v0.4.12":
        raise BenchmarkProtocolError("academic harness release differs from v1")
    if (
        harness.get("release_commit")
        != "6d642546f4688648fced259eb3302efd36ece5af"
    ):
        raise BenchmarkProtocolError("academic harness commit differs from v1")

    tasks = academic.get("tasks")
    if not isinstance(tasks, list):
        raise BenchmarkProtocolError("academic tasks must be a list")
    by_id = {task.get("id"): task for task in tasks}
    if set(by_id) != set(EXPECTED_ACADEMIC_TASKS):
        raise BenchmarkProtocolError("academic task set differs from v1")
    for task_id, expected in EXPECTED_ACADEMIC_TASKS.items():
        task = by_id[task_id]
        actual = (
            task.get("harness_task"),
            task.get("dataset_id"),
            task.get("dataset_revision"),
            task.get("split"),
        )
        if actual != expected:
            raise BenchmarkProtocolError(
                f"{task_id} task/dataset/revision/split differs from v1"
            )
        if task.get("data_committed") is not False:
            raise BenchmarkProtocolError(
                f"{task_id} must not commit third-party dataset rows"
            )
    hellaswag = by_id["hellaswag_1000"]["selection"]
    if hellaswag != {
        "policy": "first_n_in_pinned_dataset_order",
        "limit": 1000,
        "shuffle": False,
        "seed": None,
    }:
        raise BenchmarkProtocolError("HellaSwag subset selection differs from v1")


def load_result_schema(root: Path | str) -> dict[str, Any]:
    root_path = Path(root).resolve()
    schema = _load_json(root_path / RESULT_SCHEMA_PATH)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise BenchmarkProtocolError(f"invalid benchmark result schema: {exc}") from exc
    return schema


def validate_repository_contracts(root: Path | str) -> None:
    """Validate protocol, T10 linkage, result schema, and academic subset."""

    root_path = Path(root).resolve()
    protocol = load_protocol(root_path)
    t10 = _load_json(root_path / T10_PATH)
    _validate_workloads(protocol, t10)
    load_result_schema(root_path)
    academic = _load_json(root_path / ACADEMIC_PATH)
    _validate_academic_contract(academic)


def quantile_type_7(values: Sequence[float], probability: float) -> float:
    """Return the Hyndman-Fan type-7 (linear) sample quantile."""

    if not values:
        raise BenchmarkProtocolError("at least one value is required")
    if not 0.0 <= probability <= 1.0:
        raise BenchmarkProtocolError("quantile probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def summarize_samples(values: Iterable[float]) -> dict[str, float | int]:
    """Compute the frozen statistics from unrounded valid base-unit samples."""

    samples = [float(value) for value in values]
    if not samples:
        raise BenchmarkProtocolError("at least one valid sample is required")
    if any(not math.isfinite(value) or value < 0 for value in samples):
        raise BenchmarkProtocolError("samples must be finite and non-negative")
    median = quantile_type_7(samples, 0.5)
    deviations = [abs(value - median) for value in samples]
    return {
        "sample_count_valid": len(samples),
        "minimum": min(samples),
        "maximum": max(samples),
        "mean": statistics.fmean(samples),
        "sample_standard_deviation": (
            statistics.stdev(samples) if len(samples) > 1 else 0.0
        ),
        "median": median,
        "p90": quantile_type_7(samples, 0.9),
        "p95": quantile_type_7(samples, 0.95),
        "median_absolute_deviation": quantile_type_7(deviations, 0.5),
        "interquartile_range": (
            quantile_type_7(samples, 0.75) - quantile_type_7(samples, 0.25)
        ),
    }


def bootstrap_median_confidence_interval(
    values: Sequence[float],
    *,
    resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 20260725,
) -> tuple[float, float]:
    """Compute the protocol's deterministic percentile-bootstrap median CI."""

    samples = [float(value) for value in values]
    if not samples:
        raise BenchmarkProtocolError("at least one valid sample is required")
    if resamples < 1:
        raise BenchmarkProtocolError("bootstrap resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise BenchmarkProtocolError("confidence level must be in (0, 1)")
    if any(not math.isfinite(value) or value < 0 for value in samples):
        raise BenchmarkProtocolError("samples must be finite and non-negative")
    rng = random.Random(seed)
    medians = [
        quantile_type_7(rng.choices(samples, k=len(samples)), 0.5)
        for _ in range(resamples)
    ]
    tail = (1.0 - confidence_level) / 2.0
    return (
        quantile_type_7(medians, tail),
        quantile_type_7(medians, 1.0 - tail),
    )


def _assert_close(field: str, actual: Any, expected: float) -> None:
    if not isinstance(actual, (int, float)) or not math.isclose(
        float(actual),
        expected,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise BenchmarkProtocolError(
            f"summary {field}={actual!r} does not match raw samples ({expected!r})"
        )


def _nonempty_action(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_synchronization(
    result: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    measurement = result["measurement"]
    evidence = measurement["synchronization"]
    backend = evidence["backend"]
    method_policy = protocol["synchronization"]["methods"][backend]
    if evidence["method_id"] != method_policy["method_id"]:
        raise BenchmarkProtocolError(
            f"{backend} must use synchronization method "
            f"{method_policy['method_id']!r}"
        )
    if (
        method_policy["requires_pre_timer_action"]
        and not _nonempty_action(evidence["pre_timer_action"])
    ):
        raise BenchmarkProtocolError(
            f"{backend} requires an explicit pre-timer synchronization action"
        )
    if (
        method_policy["requires_post_timer_action"]
        and not _nonempty_action(evidence["post_timer_action"])
    ):
        raise BenchmarkProtocolError(
            f"{backend} requires an explicit post-timer synchronization action"
        )
    if backend == "qualcomm_workbench" and measurement["scope"] != "graph":
        raise BenchmarkProtocolError(
            "Qualcomm Workbench service-reported timing is graph scope only"
        )
    allowed_by_platform = {
        "cpu_reference": {"pytorch_cpu", "onnxruntime_cpu"},
        "apple": {"mlx", "pytorch_cpu", "onnxruntime_cpu"},
        "nvidia": {
            "onnxruntime_cuda",
            "generic_accelerator",
            "pytorch_cpu",
            "onnxruntime_cpu",
        },
        "qualcomm": {
            "qualcomm_workbench",
            "qualcomm_device_cloud",
            "generic_accelerator",
        },
    }
    platform = result["system"]["platform"]
    allowed = allowed_by_platform.get(platform)
    if allowed is not None and backend not in allowed:
        raise BenchmarkProtocolError(
            f"synchronization backend {backend!r} is incompatible with "
            f"system platform {platform!r}"
        )


def _validate_process_isolation(
    measurement: Mapping[str, Any],
    class_policy: Mapping[str, Any] | None,
) -> None:
    evidence = measurement["process_isolation"]
    expected_fresh = bool(
        class_policy and class_policy["fresh_process_each_repetition"]
    )
    if evidence["fresh_process_each_repetition"] is not expected_fresh:
        raise BenchmarkProtocolError(
            "fresh_process_each_repetition differs from the timing-class policy"
        )
    if expected_fresh:
        if not _nonempty_action(evidence["reset_method"]):
            raise BenchmarkProtocolError(
                "cold-start evidence requires an explicit process reset method"
            )
        if not _nonempty_action(evidence["process_identity_evidence"]):
            raise BenchmarkProtocolError(
                "cold-start evidence requires per-repetition process identity evidence"
            )


def _validate_quality_method(
    result: Mapping[str, Any],
    academic: Mapping[str, Any],
) -> None:
    if result["source"]["workload_id"] != "academic_evaluation":
        raise BenchmarkProtocolError(
            "quality_metric requires source.workload_id='academic_evaluation'"
        )
    method = result["measurement"]["quality_method"]
    tasks = {task["id"]: task for task in academic["tasks"]}
    task = tasks.get(method["task_id"])
    if task is None:
        raise BenchmarkProtocolError(
            f"quality task {method['task_id']!r} is absent from the frozen suite"
        )
    expected = {
        "suite_id": academic["suite_id"],
        "dataset_id": task["dataset_id"],
        "dataset_revision": task["dataset_revision"],
        "dataset_config": task["dataset_config"],
        "harness_release": academic["harness"]["release"],
        "harness_commit": academic["harness"]["release_commit"],
        "split": task["split"],
        "selection": task["selection"],
        "prompt_interface": academic["prompt_interface"],
        "apply_chat_template": academic["apply_chat_template"],
        "fewshot": academic["fewshot"],
    }
    for field, expected_value in expected.items():
        if method[field] != expected_value:
            raise BenchmarkProtocolError(
                f"quality_method.{field} differs from the frozen academic task"
            )
    if method["metric_name"] not in task["metrics"]:
        raise BenchmarkProtocolError(
            f"quality metric {method['metric_name']!r} is not frozen for "
            f"{method['task_id']}"
        )
    unit_by_metric = {
        "acc": "ratio",
        "acc_norm": "ratio",
        "word_perplexity": "perplexity",
        "byte_perplexity": "perplexity",
        "bits_per_byte": "bits_per_byte",
    }
    expected_unit = unit_by_metric[method["metric_name"]]
    if result["measurement"]["unit"] != expected_unit:
        raise BenchmarkProtocolError(
            f"{method['metric_name']} must use unit {expected_unit!r}"
        )


def _validate_result_semantics(
    result: Mapping[str, Any],
    protocol: Mapping[str, Any],
    academic: Mapping[str, Any],
) -> None:
    if result["protocol_sha256"] != protocol["contract_sha256"]:
        raise BenchmarkProtocolError("result references a different protocol digest")

    measurement = result["measurement"]
    kind = measurement["kind"]
    timing_class = measurement["timing_class"]
    if kind in {"performance", "memory"}:
        if timing_class is None:
            raise BenchmarkProtocolError(f"{kind} requires a timing_class")
        class_policy = protocol["timing_classes"][timing_class]
        if measurement["scope"] != class_policy["scope"]:
            raise BenchmarkProtocolError("timing class and scope do not match")
        for field in ("warmup_repetitions", "measured_repetitions"):
            if measurement[field] != class_policy[field]:
                raise BenchmarkProtocolError(
                    f"{field} differs from the {timing_class} policy"
                )
    else:
        if timing_class is not None:
            raise BenchmarkProtocolError(
                f"{kind} is non-timing evidence and timing_class must be null"
            )
        non_timing = protocol["non_timing_measurements"][kind]
        if measurement["scope"] != non_timing["scope"]:
            raise BenchmarkProtocolError(
                f"{kind} scope differs from the non-timing policy"
            )
        if measurement["warmup_repetitions"] != 0:
            raise BenchmarkProtocolError(
                f"{kind} must not use performance warm-up repetitions"
            )
        expected_repetitions = non_timing.get("measured_repetitions")
        if (
            expected_repetitions is not None
            and measurement["measured_repetitions"] != expected_repetitions
        ):
            raise BenchmarkProtocolError(
                f"{kind} measured_repetitions differs from the policy"
            )
        class_policy = None

    metric = measurement["metric"]
    if metric not in protocol["metric_definitions"]:
        raise BenchmarkProtocolError(f"metric {metric!r} is not defined by protocol")
    if kind == "performance" and metric not in class_policy["required_metrics"]:
        raise BenchmarkProtocolError(
            f"{metric} is not allowed for timing class {timing_class}"
        )

    includes = set(measurement["includes"])
    excludes = set(measurement["excludes"])
    if includes & excludes:
        raise BenchmarkProtocolError("measurement includes and excludes overlap")
    expected_inclusion = {
        "compile": False,
        "model_load": bool(class_policy and class_policy["model_load_included"]),
    }
    if timing_class == "cold_start":
        cold_components = {
            "artifact_load_latency": {
                "artifact_load": True,
                "model_load": False,
            },
            "model_load_latency": {
                "artifact_load": False,
                "model_load": True,
            },
            "cold_ttft": {
                "artifact_load": True,
                "model_load": True,
            },
        }
        expected_inclusion.update(cold_components[metric])
    for component, expected_included in expected_inclusion.items():
        collection = includes if expected_included else excludes
        if component not in collection:
            destination = "includes" if expected_included else "excludes"
            raise BenchmarkProtocolError(
                f"{kind} must list {component!r} in {destination}"
            )

    expected_unit = EXPECTED_UNITS.get(metric)
    if expected_unit is not None and measurement["unit"] != expected_unit:
        raise BenchmarkProtocolError(
            f"{metric} must use base unit {expected_unit!r}"
        )
    method_requirements = {
        "peak_memory": "memory_method",
        "average_power": "power_thermal_method",
        "energy_per_output_token": "power_thermal_method",
        "quality_metric": "quality_method",
        "numerical_error": "numerical_method",
    }
    required_method = method_requirements.get(metric)
    if required_method and required_method not in measurement:
        raise BenchmarkProtocolError(
            f"{metric} requires measurement.{required_method}"
        )
    expected_kinds = {
        "peak_memory": "memory",
        "average_power": "power_thermal",
        "energy_per_output_token": "power_thermal",
        "quality_metric": "quality",
        "numerical_error": "numerical",
    }
    expected_kind = expected_kinds.get(metric, "performance")
    if kind != expected_kind:
        raise BenchmarkProtocolError(
            f"{metric} must use measurement kind {expected_kind!r}"
        )
    if kind == "power_thermal":
        power_method = measurement["power_thermal_method"]
        idle_baseline = power_method["idle_baseline_watts"]
        if idle_baseline is not None and (
            isinstance(idle_baseline, bool)
            or not isinstance(idle_baseline, (int, float))
            or not math.isfinite(float(idle_baseline))
            or idle_baseline < 0
        ):
            raise BenchmarkProtocolError(
                "non-null idle_baseline_watts must be finite and non-negative"
            )
        if power_method["baseline_subtracted"] and idle_baseline is None:
            raise BenchmarkProtocolError(
                "baseline_subtracted=true requires non-null idle_baseline_watts"
            )
    _validate_synchronization(result, protocol)
    _validate_process_isolation(measurement, class_policy)

    prompt_denominator_metrics = {"prefill_throughput"}
    generated_denominator_metrics = {
        "decode_time_per_output_token",
        "decode_throughput",
        "generation_throughput_including_prefill",
        "generation_throughput_excluding_prefill",
        "energy_per_output_token",
    }
    if (
        metric in prompt_denominator_metrics
        and measurement.get("actual_prompt_tokens", 0) < 1
    ):
        raise BenchmarkProtocolError(
            f"{metric} requires positive actual_prompt_tokens"
        )
    if (
        metric in generated_denominator_metrics
        and measurement.get("actual_generated_tokens", 0) < 1
    ):
        raise BenchmarkProtocolError(
            f"{metric} requires positive actual_generated_tokens"
        )
    workload = next(
        (
            item
            for item in protocol["workloads"]
            if item["id"] == result["source"]["workload_id"]
        ),
        None,
    )
    if workload is not None:
        actual_prompt = measurement.get("actual_prompt_tokens")
        if (
            actual_prompt is not None
            and actual_prompt != workload["prompt_tokens"]
        ):
            raise BenchmarkProtocolError(
                "actual_prompt_tokens differs from the frozen workload"
            )
        actual_generated = measurement.get("actual_generated_tokens")
        if (
            actual_generated is not None
            and actual_generated > workload["generated_tokens"]
        ):
            raise BenchmarkProtocolError(
                "actual_generated_tokens exceeds the frozen output limit"
            )
    if metric == "quality_metric":
        _validate_quality_method(result, academic)

    samples = result["samples"]
    if len(samples) != measurement["measured_repetitions"]:
        raise BenchmarkProtocolError(
            "raw sample count must equal measured_repetitions"
        )
    indexes = [sample["sample_index"] for sample in samples]
    if indexes != list(range(len(samples))):
        raise BenchmarkProtocolError(
            "sample_index values must be ordered and contiguous from zero"
        )
    for sample in samples:
        if sample["valid"] and sample["invalid_reason"] is not None:
            raise BenchmarkProtocolError("valid sample has an invalid_reason")
        if not sample["valid"] and sample["invalid_reason"] is None:
            raise BenchmarkProtocolError("invalid sample lacks invalid_reason")
    valid_values = [sample["value"] for sample in samples if sample["valid"]]
    expected_summary = summarize_samples(valid_values)
    summary = result["summary"]
    total = len(samples)
    invalid = total - len(valid_values)
    if summary["sample_count_total"] != total:
        raise BenchmarkProtocolError("summary sample_count_total is inconsistent")
    if summary["sample_count_valid"] != len(valid_values):
        raise BenchmarkProtocolError("summary sample_count_valid is inconsistent")
    if summary["sample_count_invalid"] != invalid:
        raise BenchmarkProtocolError("summary sample_count_invalid is inconsistent")
    for field in SUMMARY_FIELDS:
        _assert_close(field, summary[field], float(expected_summary[field]))

    validity = result["validity"]
    if invalid:
        if validity["state"] != "incomplete" or validity["headline_eligible"]:
            raise BenchmarkProtocolError(
                "a series with invalid samples must be incomplete and not headline eligible"
            )
    elif validity["state"] != "valid":
        raise BenchmarkProtocolError(
            "a complete series without invalid samples must be valid"
        )
    if validity["headline_eligible"]:
        interval = summary.get("median_confidence_interval_95")
        if interval is None:
            raise BenchmarkProtocolError(
                "headline-eligible result lacks median 95% confidence interval"
            )
        if not interval[0] <= summary["median"] <= interval[1]:
            raise BenchmarkProtocolError(
                "median confidence interval does not contain the median"
            )
        confidence_policy = protocol["statistics"]["confidence_interval"]
        expected_interval = bootstrap_median_confidence_interval(
            valid_values,
            resamples=confidence_policy["resamples"],
            confidence_level=confidence_policy["confidence_level"],
            seed=confidence_policy["seed"],
        )
        _assert_close(
            "median_confidence_interval_95[0]",
            interval[0],
            expected_interval[0],
        )
        _assert_close(
            "median_confidence_interval_95[1]",
            interval[1],
            expected_interval[1],
        )


def validate_result(
    result: Mapping[str, Any],
    *,
    root: Path | str,
) -> None:
    """Validate one result against JSON Schema and semantic/statistical rules."""

    root_path = Path(root).resolve()
    protocol = load_protocol(root_path)
    academic = _load_json(root_path / ACADEMIC_PATH)
    _validate_academic_contract(academic)
    schema = load_result_schema(root_path)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(result), key=lambda error: list(error.path))
    if errors:
        first = errors[0]
        location = ".".join(str(item) for item in first.absolute_path) or "<root>"
        raise BenchmarkProtocolError(
            f"result schema violation at {location}: {first.message}"
        )
    _validate_result_semantics(result, protocol, academic)


def _self_check_result_validation(root: Path | str) -> None:
    """Exercise the result gate with clearly synthetic, in-memory test data."""

    protocol = load_protocol(root)
    values = [0.001 + index * 0.000001 for index in range(30)]
    computed = summarize_samples(values)
    interval = bootstrap_median_confidence_interval(values)
    summary = {
        "sample_count_total": len(values),
        "sample_count_invalid": 0,
        **computed,
        "median_confidence_interval_95": [
            interval[0],
            interval[1],
        ],
    }
    synthetic = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["contract_sha256"],
        "result_id": "synthetic-validator-self-check-not-a-measurement",
        "task_id": "T13",
        "created_at": "2026-07-25T00:00:00Z",
        "source": {
            "model_id": "synthetic",
            "model_revision": "0000000",
            "tokenizer_revision": "0000000",
            "artifact_id": "synthetic",
            "artifact_sha256": "0" * 64,
            "git_commit": "0" * 40,
            "workload_id": "S128",
            "precision": "synthetic",
            "generation_policy_id": "synthetic",
        },
        "system": {
            "evidence_level": "simulated",
            "platform": "other",
            "device_name": "validator-self-check",
            "device_type": "none",
            "os": "none",
            "runtime": "none",
            "runtime_version": "none",
            "provider_or_compute_unit": "none",
            "placement_evidence": "synthetic schema exercise only",
            "host_manifest_sha256": "0" * 64,
        },
        "measurement": {
            "kind": "performance",
            "timing_class": "single_graph",
            "scope": "graph",
            "metric": "graph_latency",
            "unit": "seconds",
            "timing_boundary": "synthetic schema exercise only",
            "synchronization": {
                "backend": "pytorch_cpu",
                "method_id": "call_return",
                "pre_timer_action": None,
                "post_timer_action": "blocking model call returned",
                "evidence": "synthetic schema exercise only",
            },
            "process_isolation": {
                "fresh_process_each_repetition": False,
                "reset_method": None,
                "process_identity_evidence": None,
            },
            "warmup_repetitions": 5,
            "measured_repetitions": 30,
            "includes": [],
            "excludes": ["model_load", "compile"],
        },
        "samples": [
            {
                "sample_index": index,
                "value": value,
                "valid": True,
                "invalid_reason": None,
            }
            for index, value in enumerate(values)
        ],
        "summary": summary,
        "validity": {
            "state": "valid",
            "reasons": [],
            "headline_eligible": True,
        },
        "comparison": {
            "claim_scope": "system_result",
            "comparable_dimensions": ["synthetic validator shape"],
            "non_comparable_dimensions": ["not a measurement"],
            "system_difference_notes": "In-memory validator exercise only.",
        },
    }
    validate_result(synthetic, root=root)
    synthetic["summary"]["median"] = float(synthetic["summary"]["median"]) + 1
    try:
        validate_result(synthetic, root=root)
    except BenchmarkProtocolError:
        pass
    else:
        raise BenchmarkProtocolError(
            "result validator accepted a summary inconsistent with raw samples"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="validate repository contracts")
    check.add_argument("--root", type=Path, default=Path.cwd())
    check_result = subparsers.add_parser(
        "check-result",
        help="validate one benchmark metric result",
    )
    check_result.add_argument("path", type=Path)
    check_result.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "check":
            validate_repository_contracts(args.root)
            known = summarize_samples([1.0, 2.0, 3.0, 4.0])
            if known["median"] != 2.5 or known["p95"] != 3.8499999999999996:
                raise BenchmarkProtocolError("statistics self-check failed")
            _self_check_result_validation(args.root)
            print("benchmark protocol, schema, and academic subset are valid")
        else:
            result = _load_json(args.path)
            validate_result(result, root=args.root)
            print(f"{args.path}: valid")
    except BenchmarkProtocolError as exc:
        raise SystemExit(f"error: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
