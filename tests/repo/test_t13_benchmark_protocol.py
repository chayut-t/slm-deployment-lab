"""Adversarial regression tests for the frozen T13 benchmark contract."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from typing import Any

from slm_lab.benchmark.protocol import (
    BenchmarkProtocolError,
    load_protocol,
    summarize_samples,
    validate_repository_contracts,
    validate_result,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ACADEMIC_PATH = REPO_ROOT / "configs/workloads/academic-evaluation-v1.json"


class T13BenchmarkProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_protocol(REPO_ROOT)
        cls.academic = json.loads(ACADEMIC_PATH.read_text(encoding="utf-8"))

    def _result(
        self,
        *,
        timing_class: str | None = "single_graph",
        metric: str = "graph_latency",
        kind: str = "performance",
        unit: str = "seconds",
    ) -> dict[str, Any]:
        if timing_class is None:
            non_timing = self.protocol["non_timing_measurements"][kind]
            scope = non_timing["scope"]
            warmup = non_timing["warmup_repetitions"]
            repetitions = non_timing.get("measured_repetitions", 1)
            fresh = False
        else:
            timing = self.protocol["timing_classes"][timing_class]
            scope = timing["scope"]
            warmup = timing["warmup_repetitions"]
            repetitions = timing["measured_repetitions"]
            fresh = timing["fresh_process_each_repetition"]

        values = [0.001 + index * 0.000001 for index in range(repetitions)]
        summary = {
            "sample_count_total": repetitions,
            "sample_count_invalid": 0,
            **summarize_samples(values),
        }
        return {
            "schema_version": 1,
            "protocol_id": self.protocol["protocol_id"],
            "protocol_sha256": self.protocol["contract_sha256"],
            "result_id": "synthetic-adversarial-test-not-a-measurement",
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
                "platform": "cpu_reference",
                "device_name": "validator-test",
                "device_type": "none",
                "os": "none",
                "runtime": "none",
                "runtime_version": "none",
                "provider_or_compute_unit": "cpu",
                "placement_evidence": "synthetic validator test",
                "host_manifest_sha256": "0" * 64,
            },
            "measurement": {
                "kind": kind,
                "timing_class": timing_class,
                "scope": scope,
                "metric": metric,
                "unit": unit,
                "timing_boundary": "synthetic validator test",
                "synchronization": {
                    "backend": "pytorch_cpu",
                    "method_id": "call_return",
                    "pre_timer_action": None,
                    "post_timer_action": "blocking call returned",
                    "evidence": "synthetic validator test",
                },
                "process_isolation": {
                    "fresh_process_each_repetition": fresh,
                    "reset_method": "new process per sample" if fresh else None,
                    "process_identity_evidence": (
                        "distinct synthetic PIDs per sample" if fresh else None
                    ),
                },
                "warmup_repetitions": warmup,
                "measured_repetitions": repetitions,
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
                "headline_eligible": False,
            },
            "comparison": {
                "claim_scope": "system_result",
                "comparable_dimensions": ["synthetic schema shape"],
                "non_comparable_dimensions": ["not a measurement"],
                "system_difference_notes": "Synthetic validator test only.",
            },
        }

    def test_repository_contracts_and_valid_graph_result_pass(self) -> None:
        validate_repository_contracts(REPO_ROOT)
        validate_result(self._result(), root=REPO_ROOT)

    def _energy_result(self) -> dict[str, Any]:
        result = self._result(
            timing_class=None,
            metric="energy_per_output_token",
            kind="power_thermal",
            unit="joules_per_token",
        )
        result["measurement"]["actual_generated_tokens"] = 1
        result["measurement"]["power_thermal_method"] = {
            "instrument": "synthetic",
            "sample_rate_hz": 1,
            "duration_seconds": 600,
            "measurement_domain": "synthetic",
            "idle_baseline_watts": None,
            "baseline_subtracted": False,
            "start_temperature": None,
            "end_temperature": None,
            "thermal_state": "synthetic",
            "ambient_notes": "synthetic",
        }
        return result

    def test_cold_start_load_metrics_are_defined_and_valid(self) -> None:
        definitions = self.protocol["metric_definitions"]
        for metric in ("artifact_load_latency", "model_load_latency"):
            self.assertIn(metric, definitions)

        artifact = self._result(
            timing_class="cold_start",
            metric="artifact_load_latency",
        )
        artifact["measurement"]["includes"] = ["artifact_load"]
        validate_result(artifact, root=REPO_ROOT)

        model = self._result(
            timing_class="cold_start",
            metric="model_load_latency",
        )
        model["measurement"]["includes"] = ["model_load"]
        model["measurement"]["excludes"] = ["artifact_load", "compile"]
        validate_result(model, root=REPO_ROOT)

        cold_ttft = self._result(timing_class="cold_start", metric="cold_ttft")
        cold_ttft["measurement"]["includes"] = ["artifact_load", "model_load"]
        cold_ttft["measurement"]["excludes"] = ["compile"]
        validate_result(cold_ttft, root=REPO_ROOT)

    def test_timing_class_cannot_be_mislabeled_with_another_metric(self) -> None:
        cases = (
            ("single_graph", "request_total_latency"),
            ("end_to_end_request", "graph_latency"),
            ("cold_start", "graph_latency"),
            ("runtime_stage", "request_ttft"),
        )
        for timing_class, metric in cases:
            with self.subTest(timing_class=timing_class, metric=metric):
                result = self._result(timing_class=timing_class, metric=metric)
                with self.assertRaisesRegex(
                    BenchmarkProtocolError,
                    "not allowed for timing class",
                ):
                    validate_result(result, root=REPO_ROOT)

    def test_backend_synchronization_policy_is_enforced(self) -> None:
        wrong_method = self._result()
        wrong_method["measurement"]["synchronization"].update(
            {
                "backend": "onnxruntime_cuda",
                "method_id": "call_return",
            }
        )
        with self.assertRaisesRegex(BenchmarkProtocolError, "must use"):
            validate_result(wrong_method, root=REPO_ROOT)

        missing_fence = self._result()
        missing_fence["measurement"]["synchronization"].update(
            {
                "backend": "onnxruntime_cuda",
                "method_id": "cuda_stream_or_device_fence",
                "pre_timer_action": None,
                "post_timer_action": None,
            }
        )
        with self.assertRaisesRegex(BenchmarkProtocolError, "pre-timer"):
            validate_result(missing_fence, root=REPO_ROOT)

        platform_mismatch = self._result()
        platform_mismatch["measurement"]["synchronization"].update(
            {
                "backend": "onnxruntime_cuda",
                "method_id": "cuda_stream_or_device_fence",
                "pre_timer_action": "cuda synchronize",
                "post_timer_action": "cuda synchronize",
            }
        )
        with self.assertRaisesRegex(BenchmarkProtocolError, "incompatible"):
            validate_result(platform_mismatch, root=REPO_ROOT)

    def test_cold_start_requires_fresh_process_evidence(self) -> None:
        result = self._result(timing_class="cold_start", metric="cold_ttft")
        result["measurement"]["includes"] = ["artifact_load", "model_load"]
        result["measurement"]["excludes"] = ["compile"]
        result["measurement"]["process_isolation"] = {
            "fresh_process_each_repetition": False,
            "reset_method": None,
            "process_identity_evidence": None,
        }
        with self.assertRaisesRegex(
            BenchmarkProtocolError,
            "fresh_process_each_repetition",
        ):
            validate_result(result, root=REPO_ROOT)

    def test_token_denominators_are_required(self) -> None:
        cases = (
            (
                "generation_loop",
                "prefill_throughput",
                "tokens_per_second",
                "actual_prompt_tokens",
            ),
            (
                "generation_loop",
                "decode_throughput",
                "tokens_per_second",
                "actual_generated_tokens",
            ),
            (
                "generation_loop",
                "decode_time_per_output_token",
                "seconds",
                "actual_generated_tokens",
            ),
        )
        for timing_class, metric, unit, field in cases:
            with self.subTest(metric=metric):
                result = self._result(
                    timing_class=timing_class,
                    metric=metric,
                    unit=unit,
                )
                with self.assertRaisesRegex(BenchmarkProtocolError, field):
                    validate_result(result, root=REPO_ROOT)
                result["measurement"][field] = (
                    128 if field == "actual_prompt_tokens" else 1
                )
                validate_result(result, root=REPO_ROOT)

        wrong_prompt_count = self._result(
            timing_class="generation_loop",
            metric="prefill_throughput",
            unit="tokens_per_second",
        )
        wrong_prompt_count["measurement"]["actual_prompt_tokens"] = 127
        with self.assertRaisesRegex(BenchmarkProtocolError, "frozen workload"):
            validate_result(wrong_prompt_count, root=REPO_ROOT)

        energy = self._energy_result()
        energy["measurement"].pop("actual_generated_tokens")
        with self.assertRaisesRegex(
            BenchmarkProtocolError,
            "actual_generated_tokens",
        ):
            validate_result(energy, root=REPO_ROOT)
        energy["measurement"]["actual_generated_tokens"] = 1
        validate_result(energy, root=REPO_ROOT)

    def test_subtracted_power_requires_finite_non_negative_baseline(self) -> None:
        invalid_baselines = (None, -0.1, float("nan"), float("inf"))
        for baseline in invalid_baselines:
            with self.subTest(baseline=baseline):
                result = self._energy_result()
                method = result["measurement"]["power_thermal_method"]
                method["baseline_subtracted"] = True
                method["idle_baseline_watts"] = baseline
                with self.assertRaises(BenchmarkProtocolError):
                    validate_result(result, root=REPO_ROOT)

        for baseline in (0.0, 1.5):
            with self.subTest(valid_baseline=baseline):
                result = self._energy_result()
                method = result["measurement"]["power_thermal_method"]
                method["baseline_subtracted"] = True
                method["idle_baseline_watts"] = baseline
                validate_result(result, root=REPO_ROOT)

    def _quality_result(self) -> dict[str, Any]:
        task = next(
            item
            for item in self.academic["tasks"]
            if item["id"] == "hellaswag_1000"
        )
        result = self._result(
            timing_class=None,
            metric="quality_metric",
            kind="quality",
            unit="ratio",
        )
        result["source"]["workload_id"] = "academic_evaluation"
        result["measurement"]["quality_method"] = {
            "suite_id": self.academic["suite_id"],
            "task_id": task["id"],
            "metric_name": "acc_norm",
            "dataset_id": task["dataset_id"],
            "dataset_revision": task["dataset_revision"],
            "dataset_config": task["dataset_config"],
            "harness_release": self.academic["harness"]["release"],
            "harness_commit": self.academic["harness"]["release_commit"],
            "resolved_task_sha256": "0" * 64,
            "split": task["split"],
            "selection": copy.deepcopy(task["selection"]),
            "prompt_interface": self.academic["prompt_interface"],
            "apply_chat_template": self.academic["apply_chat_template"],
            "fewshot": self.academic["fewshot"],
        }
        return result

    def test_quality_method_matches_frozen_academic_task(self) -> None:
        validate_result(self._quality_result(), root=REPO_ROOT)

        mutations = (
            ("dataset_revision", "deadbeef"),
            ("split", "train"),
            ("dataset_id", "other/dataset"),
            ("harness_commit", "f" * 40),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                result = self._quality_result()
                result["measurement"]["quality_method"][field] = value
                with self.assertRaisesRegex(
                    BenchmarkProtocolError,
                    f"quality_method.{field}",
                ):
                    validate_result(result, root=REPO_ROOT)

        selection = self._quality_result()
        selection["measurement"]["quality_method"]["selection"]["limit"] = 999
        with self.assertRaisesRegex(
            BenchmarkProtocolError,
            "quality_method.selection",
        ):
            validate_result(selection, root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
