"""Regression tests for the sanitized T32 Device Cloud capture."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
import unicodedata
from contextlib import redirect_stderr
from pathlib import Path

from slm_lab.deployment.qualcomm.device_cloud import (
    DeviceCloudCaptureError,
    FIXED_PROMPT,
    FIXED_PROMPT_SHA256,
    PRIVATE_REFERENCE,
    main,
    manifest_sha256,
    normalize_capture,
)


SHA_A = "a" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
REPO_ROOT = Path(__file__).resolve().parents[3]


def timing(milliseconds: float) -> dict[str, object]:
    return {
        "milliseconds": milliseconds,
        "source": "geniex_runtime_report",
        "evidence_sha256": SHA_D,
        "private_reference": PRIVATE_REFERENCE,
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
            "evidence_kind": "windows_system_information",
            "evidence_sha256": SHA_A,
            "private_reference": PRIVATE_REFERENCE,
        },
        "runtime": {
            "geniex_version": "0.3.1",
            "route": "llama_cpp",
            "compute_selection": "npu",
            "placement": {
                "status": "observed",
                "compute_unit": "NPU",
                "backend": "HTP",
                "device_id": "HTP0",
                "evidence_kind": "geniex_runtime_log",
                "evidence_sha256": SHA_D,
                "private_reference": PRIVATE_REFERENCE,
            },
        },
        "model": {
            "logical_name": "Qwen3-0.6B",
            "source": "Qualcomm AI Hub Models",
            "source_version": "0.58.0",
            "asset_runtime": "geniex_llamacpp",
            "precision": "Q4_0",
            "artifact_sha256": SHA_A,
            "private_reference": PRIVATE_REFERENCE,
        },
        "generation": {
            "prompt_sha256": FIXED_PROMPT_SHA256,
            "prompt_private_reference": PRIVATE_REFERENCE,
            "prompt_tokens": 14,
            "output_sha256": SHA_C,
            "output_private_reference": PRIVATE_REFERENCE,
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
            "pre_timer_action": "host_clock_immediately_before_runtime_call",
            "post_timer_action": "runtime_completion_before_host_clock_read",
            "evidence_kind": "instrumented_runtime_trace",
            "evidence_sha256": SHA_D,
            "private_reference": PRIVATE_REFERENCE,
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
        self.assertEqual(manifest["model"]["source_version"], "0.58.0")
        self.assertEqual(manifest["model"]["precision"], "Q4_0")
        self.assertEqual(manifest["generation"]["output_tokens"], 9)
        self.assertEqual(
            manifest["generation"]["prompt_contract"],
            "t32_fixed_prompt_utf8_nfc_no_trailing_newline_v1",
        )
        self.assertEqual(
            manifest["runtime"]["placement"],
            {
                "status": "observed",
                "compute_unit": "NPU",
                "backend": "HTP",
                "device_id": "HTP0",
                "evidence_kind": "geniex_runtime_log",
                "evidence_sha256": SHA_D,
                "private_reference": PRIVATE_REFERENCE,
            },
        )
        self.assertFalse(manifest["provenance"]["hosted_graph_latency_included"])
        self.assertEqual(len(manifest_sha256(manifest)), 64)

    def test_private_text_is_rejected_without_echoing_value(self) -> None:
        capture = complete_capture()
        capture["device"]["os"] = "learner@example.invalid"

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

    def test_placement_requires_affirmative_structured_npu_htp_evidence(self) -> None:
        for field, value in (
            ("status", "not_observed"),
            ("compute_unit", "CPU"),
            ("backend", "CPU"),
            ("device_id", "HTP1"),
            ("evidence_kind", "operator_note"),
        ):
            with self.subTest(field=field):
                capture = complete_capture()
                capture["runtime"]["placement"][field] = value
                with self.assertRaises(DeviceCloudCaptureError):
                    normalize_capture(capture)

    def test_all_public_evidence_has_digest_and_private_reference(self) -> None:
        for mutate in (
            lambda capture: capture["device"].update(
                {"private_reference": "raw_log_path"}
            ),
            lambda capture: capture["device"].update(
                {"evidence_kind": "operator_note"}
            ),
            lambda capture: capture["model"].update(
                {"artifact_sha256": "not-a-digest"}
            ),
            lambda capture: capture["runtime"]["placement"].update(
                {"evidence_sha256": "not-a-digest"}
            ),
            lambda capture: capture["timings"]["prefill"].update(
                {"private_reference": "raw_log_path"}
            ),
            lambda capture: capture["synchronization"].update(
                {"evidence_sha256": "not-a-digest"}
            ),
        ):
            capture = complete_capture()
            mutate(capture)
            with self.assertRaises(DeviceCloudCaptureError):
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

    def test_boolean_values_are_rejected_in_numeric_fields(self) -> None:
        for mutate in (
            lambda capture: capture.update({"schema_version": True}),
            lambda capture: capture["cost"].update({"cost_usd": False}),
            lambda capture: capture["timings"]["prefill"].update(
                {"milliseconds": False}
            ),
        ):
            capture = complete_capture()
            mutate(capture)
            with self.assertRaises(DeviceCloudCaptureError):
                normalize_capture(capture)

    def test_huge_integer_timing_is_rejected_through_safe_cli_path(self) -> None:
        capture = complete_capture()
        capture["timings"]["prefill"]["milliseconds"] = 10**400

        with self.assertRaisesRegex(DeviceCloudCaptureError, "finite number"):
            normalize_capture(capture)
        self._assert_cli_rejects(capture)

    def test_huge_integer_cost_is_rejected_through_safe_cli_path(self) -> None:
        capture = complete_capture()
        capture["cost"]["cost_usd"] = 10**400

        with self.assertRaisesRegex(DeviceCloudCaptureError, "finite number"):
            normalize_capture(capture)
        self._assert_cli_rejects(capture)

    def test_observed_at_is_strict_rfc3339_utc(self) -> None:
        for value in (
            "2026-07-27",
            "2026-07-27 18:00:00Z",
            "2026-07-27T18:00:00+00:00",
            "2026-7-27T18:00:00Z",
            "2026-07-27T18:00Z",
            "2026-02-30T18:00:00Z",
        ):
            with self.subTest(value=value):
                capture = complete_capture()
                capture["observed_at"] = value
                with self.assertRaises(DeviceCloudCaptureError):
                    normalize_capture(capture)

    def test_prompt_digest_is_pinned_to_normalized_fixed_prompt(self) -> None:
        normalized = unicodedata.normalize("NFC", FIXED_PROMPT)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        self.assertEqual(digest, FIXED_PROMPT_SHA256)

        capture = complete_capture()
        capture["generation"]["prompt_sha256"] = SHA_C
        with self.assertRaisesRegex(DeviceCloudCaptureError, "pinned"):
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

    def test_powershell_runner_is_no_clobber_and_checks_native_failures(self) -> None:
        script = (
            REPO_ROOT
            / "scripts"
            / "qualcomm"
            / "device_cloud"
            / "run_qwen_baseline.ps1"
        ).read_text(encoding="utf-8")

        self.assertIn("function Invoke-NativeChecked", script)
        self.assertIn("if ($LASTEXITCODE -ne 0)", script)
        self.assertIn("[guid]::NewGuid()", script)
        self.assertIn("Start-Transcript -Path $transcriptPath -NoClobber", script)
        self.assertNotIn("Start-Transcript -Path $transcriptPath -Force", script)
        self.assertEqual(script.count("Invoke-NativeChecked"), 5)
        self.assertIn(FIXED_PROMPT_SHA256, script)

    def test_publication_path_and_lifecycle_match_t32_ownership(self) -> None:
        readme = (
            REPO_ROOT / "scripts" / "qualcomm" / "device_cloud" / "README.md"
        ).read_text(encoding="utf-8")
        result = (
            REPO_ROOT / "docs" / "results" / "qualcomm" / "device-cloud.md"
        ).read_text(encoding="utf-8")
        handoff = (
            REPO_ROOT / "ai" / "handoffs" / "T32-device-cloud-live-boundary.md"
        ).read_text(encoding="utf-8")

        self.assertNotIn("--manifest results/processed/qualcomm", readme)
        self.assertIn(".ai-local/profiles/T32/", readme)
        self.assertIn("Status: in_progress", result)
        self.assertIn("Status: in_progress", handoff)

    def _assert_cli_rejects(self, capture: dict[str, object]) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture_path = root / "capture.json"
            manifest_path = root / "manifest.json"
            capture_path.write_text(json.dumps(capture), encoding="utf-8")

            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                main(
                    [
                        "--capture",
                        str(capture_path),
                        "--manifest",
                        str(manifest_path),
                    ]
                )

            self.assertEqual(caught.exception.code, 2)
            self.assertIn("finite number", stderr.getvalue())
            self.assertNotIn(str(10**400), stderr.getvalue())
            self.assertFalse(manifest_path.exists())


if __name__ == "__main__":
    unittest.main()
