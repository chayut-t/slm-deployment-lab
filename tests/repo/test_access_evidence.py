"""Regression checks for sanitized T02 public-platform evidence."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_EVIDENCE_PATH = (
    REPO_ROOT / "results" / "hosts" / "public-platform-access-2026-07-24.json"
)
EVIDENCE_PATH = (
    REPO_ROOT / "results" / "hosts" / "workbench-toy-lifecycle-2026-07-25.json"
)
HISTORICAL_REPORT_PATH = (
    REPO_ROOT
    / "docs"
    / "results"
    / "access"
    / "2026-07-24-public-platform-access.md"
)
REPORT_PATH = (
    REPO_ROOT
    / "docs"
    / "results"
    / "access"
    / "2026-07-25-workbench-toy-lifecycle.md"
)
FAILURE_PATH = (
    REPO_ROOT
    / "docs"
    / "failures"
    / "access"
    / "2026-07-24-t02-qualcomm-authentication.md"
)
PLAN_PATH = REPO_ROOT / "ai" / "plans" / "completed" / "T02-platform-access.md"


class PublicPlatformAccessEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.historical = json.loads(
            HISTORICAL_EVIDENCE_PATH.read_text(encoding="utf-8")
        )
        cls.evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
        cls.public_paths = (
            HISTORICAL_EVIDENCE_PATH,
            EVIDENCE_PATH,
            HISTORICAL_REPORT_PATH,
            REPORT_PATH,
            FAILURE_PATH,
            PLAN_PATH,
        )
        cls.public_text = "\n".join(
            path.read_text(encoding="utf-8") for path in cls.public_paths
        )

    def test_historical_blocker_is_preserved(self) -> None:
        self.assertEqual(self.historical["status"], "blocked")
        self.assertEqual(
            self.historical["blocker_code"],
            "qualcomm_authentication_required",
        )
        lifecycle = self.historical["workbench"]["toy_lifecycle"]
        self.assertEqual(
            {lifecycle["compile"], lifecycle["inference"], lifecycle["profile"]},
            {"not_run"},
        )

    def test_workbench_lifecycle_is_completed(self) -> None:
        self.assertEqual(self.evidence["task_id"], "T02")
        self.assertEqual(self.evidence["status"], "completed")
        workbench = self.evidence["workbench"]
        self.assertEqual(workbench["account_access"], "authenticated_api")
        lifecycle = workbench["toy_lifecycle"]
        self.assertEqual(
            {
                lifecycle["compile"],
                lifecycle["inference"],
                lifecycle["profile"],
            },
            {"success"},
        )

    def test_exact_client_framework_and_target_versions_are_recorded(self) -> None:
        workbench = self.evidence["workbench"]
        self.assertEqual(workbench["local_client"]["qai_hub_version"], "0.53.0")
        self.assertEqual(workbench["local_client"]["python_version"], "3.11.13")
        self.assertEqual(
            workbench["hosted_frameworks"]["qairt_default"],
            "2.45.0.260326154327",
        )
        self.assertEqual(
            workbench["hosted_frameworks"]["qairt_latest"],
            "2.48.0.260626120635",
        )
        lifecycle = workbench["toy_lifecycle"]
        self.assertEqual(
            lifecycle["compile_evidence"]["target_model_type"],
            "QNN_CONTEXT_BINARY",
        )
        self.assertEqual(lifecycle["compile_evidence"]["backend"], "HTP")
        self.assertEqual(lifecycle["compile_evidence"]["hexagon_version"], "v73")

    def test_inference_is_numerically_validated(self) -> None:
        inference = self.evidence["workbench"]["toy_lifecycle"][
            "inference_evidence"
        ]
        self.assertTrue(inference["allclose"]["result"])
        self.assertLess(inference["max_abs_error"], 1e-6)
        self.assertEqual(inference["allclose"]["rtol"], 1e-5)
        self.assertEqual(inference["allclose"]["atol"], 1e-6)

    def test_profile_has_units_and_npu_placement(self) -> None:
        profile = self.evidence["workbench"]["toy_lifecycle"]["profile_evidence"]
        self.assertEqual(profile["estimated_inference_time_us"], 127)
        self.assertEqual(profile["estimated_inference_time_ms"], 0.127)
        self.assertEqual(profile["inference_sample_count"], 100)
        self.assertGreater(profile["estimated_inference_peak_memory_bytes"], 0)
        self.assertEqual(profile["compute_units"], ["NPU"])
        self.assertEqual(
            profile["units_source"],
            "https://workbench.aihub.qualcomm.com/docs/hub/jobs.html",
        )

    def test_service_turnaround_is_not_reported_as_graph_latency(self) -> None:
        lifecycle = self.evidence["workbench"]["toy_lifecycle"]
        profile = lifecycle["profile_evidence"]
        latency_seconds = profile["estimated_inference_time_us"] / 1_000_000
        for stage in ("compile_evidence", "inference_evidence", "profile_evidence"):
            self.assertGreater(
                lifecycle[stage]["observed_service_turnaround_seconds"],
                latency_seconds,
            )
        self.assertIn("not model latency", REPORT_PATH.read_text(encoding="utf-8"))

    def test_unknown_quota_and_remaining_boundaries_stay_explicit(self) -> None:
        quota = self.evidence["workbench"]["quota"]
        self.assertIsNone(quota["value"])
        self.assertEqual(quota["state"], "not_exposed_by_qai_hub_client")
        device_cloud = self.evidence["device_cloud"]
        self.assertEqual(device_cloud["account_access"], "not_reverified")
        self.assertIsNone(device_cloud["account_minutes"])
        self.assertIsNone(device_cloud["x_elite_live_availability"])

    def test_paid_fallback_was_not_launched(self) -> None:
        fallback = self.evidence["paid_fallback"]
        self.assertTrue(fallback["creation_command_documented"])
        self.assertFalse(fallback["creation_command_executed"])
        self.assertTrue(fallback["approval_required"])
        self.assertEqual(self.evidence["cost"]["workbench_cost_usd"], 0)
        self.assertFalse(self.evidence["cost"]["paid_resources_launched"])

    def test_credential_restart_documents_observed_client_safeguards(self) -> None:
        text = FAILURE_PATH.read_text(encoding="utf-8")
        self.assertIn('importlib.metadata.version("qai-hub")', text)
        self.assertNotIn("qai-hub --version", text)
        self.assertIn("Capture and discard", text)
        self.assertIn("stdout and stderr", text)
        self.assertIn("mode `600`", text)
        self.assertIn("can print the", text)
        self.assertIn("including the credential", text)

    def test_source_urls_are_public_and_have_no_query_or_fragment(self) -> None:
        def visit(value):
            if isinstance(value, dict):
                for child in value.values():
                    yield from visit(child)
            elif isinstance(value, list):
                for child in value:
                    yield from visit(child)
            elif isinstance(value, str) and value.startswith("https://"):
                yield value

        allowed_hosts = {
            "aihub.qualcomm.com",
            "docs.runpod.io",
            "pypi.org",
            "workbench.aihub.qualcomm.com",
        }
        for evidence in (self.historical, self.evidence):
            for url in visit(evidence):
                parsed = urlparse(url)
                self.assertIn(parsed.hostname, allowed_hosts)
                self.assertEqual(parsed.query, "")
                self.assertEqual(parsed.fragment, "")

    def test_public_evidence_excludes_private_markers(self) -> None:
        self.assertNotRegex(
            self.public_text,
            re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
        )
        self.assertNotIn("oauth2/default/v1/authorize", self.public_text)
        self.assertNotRegex(
            self.public_text,
            re.compile(r"\bstate=[A-Za-z0-9_-]+"),
        )
        self.assertNotIn("workbench.aihub.qualcomm.com/jobs/", self.public_text)
        self.assertNotRegex(
            self.public_text,
            re.compile(r"\b(?:api_)?token\s*[:=]\s*\S+", re.I),
        )
        self.assertNotRegex(
            self.public_text,
            re.compile(r"\b(?:job|project|user)_id\s*[:=]\s*\S+", re.I),
        )
        privacy = self.evidence["workbench"]["toy_lifecycle"]["privacy"]
        self.assertEqual(set(privacy.values()), {False})


if __name__ == "__main__":
    unittest.main()
