"""Regression tests for the sanitized T32 Device Cloud capture."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from slm_lab.deployment.qualcomm.device_cloud import (
    DeviceCloudCaptureError,
    main,
    manifest_sha256,
    normalize_capture,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def timing(milliseconds: float) -> dict[str, object]:
    return {
        "milliseconds": milliseconds,
        "source": "geniex_runtime_report",
    }


def complete_capture() -> dict[str, object]:
    return {
        "schema_version": 1,
        "observed_at": "2026-07-27T18:00:00Z",
        "device": {
            "product": "Snapdragon X Elite",
            "form_factor": "Compute Reference Design",
            "catalog_code": "CRD8380X",
            "os": "Windows 11 build 26100",
            "chipset": "X1E80100",
            "memory_bytes": 34_359_738_368,
        },
        "runtime": {
            "geniex_version": "0.3.1",
            "route": "llama_cpp",
            "compute_selection": "npu",
            "placement_evidence": "GenieX log reported HTP0 NPU device",
        },
        "model": {
            "logical_name": "Qwen3-0.6B",
            "source": "Qualcomm AI Hub Models",
            "asset_runtime": "geniex_llamacpp",
            "precision": "Q4_0",
            "artifact_sha256": SHA_A,
        },
        "generation": {
            "prompt_sha256": SHA_B,
            "prompt_tokens": 14,
            "output_sha256": SHA_C,
            "output_tokens": 9,
            "finish_reason": "eos",
            "valid_multi_token_output_confirmed": True,
        },
        "timings": {
            "artifact_load": timing(10),
            "model_load": timing(20),
            "tokenization": timing(2),
            "prefill": timing(30),
            "first_decode": timing(5),
            "decode": timing(40),
            "generation_total": timing(76),
            "request_total": timing(109),
        },
        "synchronization": {
            "backend": "qualcomm_device_cloud",
            "method_id": "runtime_completion_fence",
            "pre_timer_action": "Host clock recorded immediately before runtime call",
            "post_timer_action": "Final output token materialized before clock read",
            "evidence": "Runtime returned after output stream completion",
        },
        "cost": {"paid_resources_used": False, "cost_usd": 0},
    }


class DeviceCloudCaptureTests(unittest.TestCase):
    def test_complete_capture_normalizes_closed_public_manifest(self) -> None:
        manifest = normalize_capture(complete_capture())

        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(
            manifest["measurement_scope"],
            "persistent_device_side_generation_loop",
        )
        self.assertEqual(manifest["runtime"]["route"], "llama_cpp")
        self.assertEqual(manifest["model"]["asset_runtime"], "geniex_llamacpp")
        self.assertEqual(manifest["model"]["precision"], "Q4_0")
        self.assertEqual(manifest["generation"]["output_tokens"], 9)
        self.assertFalse(manifest["provenance"]["hosted_graph_latency_included"])
        self.assertEqual(len(manifest_sha256(manifest)), 64)

    def test_private_text_is_rejected_without_echoing_value(self) -> None:
        capture = complete_capture()
        capture["runtime"]["placement_evidence"] = "learner@example.invalid"

        with self.assertRaises(DeviceCloudCaptureError) as caught:
            normalize_capture(capture)

        self.assertNotIn("learner@example.invalid", str(caught.exception))

    def test_multi_token_confirmation_is_required(self) -> None:
        for field, value in (
            ("output_tokens", 1),
            ("valid_multi_token_output_confirmed", False),
        ):
            with self.subTest(field=field):
                capture = complete_capture()
                capture["generation"][field] = value
                with self.assertRaises(DeviceCloudCaptureError):
                    normalize_capture(capture)

    def test_ready_made_route_is_not_confused_with_qairt(self) -> None:
        capture = complete_capture()
        capture["runtime"]["route"] = "qairt"

        with self.assertRaisesRegex(DeviceCloudCaptureError, "llama_cpp"):
            normalize_capture(capture)

    def test_all_timing_boundaries_and_consistent_totals_are_required(self) -> None:
        missing = complete_capture()
        del missing["timings"]["prefill"]
        with self.assertRaisesRegex(DeviceCloudCaptureError, "prefill"):
            normalize_capture(missing)

        inconsistent = complete_capture()
        inconsistent["timings"]["generation_total"]["milliseconds"] = 1
        with self.assertRaisesRegex(DeviceCloudCaptureError, "generation_total"):
            normalize_capture(inconsistent)

    def test_paid_resource_record_is_rejected(self) -> None:
        capture = complete_capture()
        capture["cost"] = {"paid_resources_used": True, "cost_usd": 1}

        with self.assertRaisesRegex(DeviceCloudCaptureError, "no paid resources"):
            normalize_capture(capture)

    def test_cli_writes_only_normalized_fields(self) -> None:
        capture = complete_capture()
        capture_with_private_unknown = copy.deepcopy(capture)
        capture_with_private_unknown["raw_session_url"] = (
            "https://private.invalid/session/example"
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture_path = root / "capture.json"
            manifest_path = root / "manifest.json"
            capture_path.write_text(json.dumps(capture), encoding="utf-8")

            self.assertEqual(
                main(
                    [
                        "--capture",
                        str(capture_path),
                        "--manifest",
                        str(manifest_path),
                    ]
                ),
                0,
            )
            public = manifest_path.read_text(encoding="utf-8")
            self.assertNotIn("session", public.lower().replace("session_reference", ""))
            self.assertNotIn(str(root), public)

        with self.assertRaisesRegex(DeviceCloudCaptureError, "unsupported fields"):
            normalize_capture(capture_with_private_unknown)


if __name__ == "__main__":
    unittest.main()
