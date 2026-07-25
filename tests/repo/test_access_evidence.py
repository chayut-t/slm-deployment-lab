"""Regression checks for sanitized T02 public-platform evidence."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = (
    REPO_ROOT / "results" / "hosts" / "public-platform-access-2026-07-24.json"
)
REPORT_PATH = (
    REPO_ROOT
    / "docs"
    / "results"
    / "access"
    / "2026-07-24-public-platform-access.md"
)
FAILURE_PATH = (
    REPO_ROOT
    / "docs"
    / "failures"
    / "access"
    / "2026-07-24-t02-qualcomm-authentication.md"
)
PLAN_PATH = REPO_ROOT / "ai" / "plans" / "active" / "T02-platform-access.md"


class PublicPlatformAccessEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
        cls.public_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (EVIDENCE_PATH, REPORT_PATH, FAILURE_PATH, PLAN_PATH)
        )

    def test_blocked_state_cannot_look_like_completed_lifecycle(self) -> None:
        self.assertEqual(self.evidence["task_id"], "T02")
        self.assertEqual(self.evidence["status"], "blocked")
        self.assertEqual(
            self.evidence["blocker_code"],
            "qualcomm_authentication_required",
        )

        lifecycle = self.evidence["workbench"]["toy_lifecycle"]
        self.assertEqual(
            {lifecycle["compile"], lifecycle["inference"], lifecycle["profile"]},
            {"not_run"},
        )
        self.assertIsNone(lifecycle["job_evidence"])

    def test_access_claims_name_evidence_and_unknowns_remain_null(self) -> None:
        for platform in ("workbench", "device_cloud"):
            self.assertTrue(self.evidence[platform]["evidence"])

        workbench = self.evidence["workbench"]
        self.assertEqual(workbench["account_access"], "authentication_required")
        self.assertIsNone(workbench["quota"]["value"])
        self.assertIsNone(workbench["hosted_versions"]["qairt_default"])

        device_cloud = self.evidence["device_cloud"]
        self.assertIsNone(device_cloud["account_minutes"])
        self.assertIsNone(device_cloud["x_elite_live_availability"])

        colab = self.evidence["free_nvidia"]["colab"]
        self.assertEqual(colab["gpu_allocation"], "not_tested")
        self.assertIsNone(colab["gpu_type"])

    def test_paid_fallback_is_documentation_only(self) -> None:
        fallback = self.evidence["paid_fallback"]
        self.assertTrue(fallback["creation_command_documented"])
        self.assertFalse(fallback["creation_command_executed"])
        self.assertTrue(fallback["approval_required"])

    def test_qualcomm_restart_uses_supported_isolated_client(self) -> None:
        expected_in_order = (
            "uv venv --python 3.11 .ai-local/envs/qai-hub-0.53.0",
            '"qai-hub==0.53.0"',
            ".ai-local/envs/qai-hub-0.53.0/bin/qai-hub --version",
            'read -r -s "T02_QAI_HUB_TOKEN?Qualcomm credential: "',
            ".ai-local/envs/qai-hub-0.53.0/bin/qai-hub configure",
            "unset T02_QAI_HUB_TOKEN",
            ".ai-local/envs/qai-hub-0.53.0/bin/qai-hub list-devices",
            ".ai-local/envs/qai-hub-0.53.0/bin/qai-hub list-frameworks",
        )
        for path in (FAILURE_PATH, PLAN_PATH):
            text = path.read_text(encoding="utf-8")
            positions = [text.index(command) for command in expected_in_order]
            self.assertEqual(positions, sorted(positions), path)
            self.assertIn("Python 3.9.6", text)
            self.assertIn("report `0.53.0`", text)

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
        for url in visit(self.evidence):
            parsed = urlparse(url)
            self.assertIn(parsed.hostname, allowed_hosts)
            self.assertEqual(parsed.query, "")
            self.assertEqual(parsed.fragment, "")

    def test_public_evidence_excludes_common_private_markers(self) -> None:
        self.assertNotRegex(
            self.public_text,
            re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
        )
        self.assertNotIn("oauth2/default/v1/authorize", self.public_text)
        self.assertNotRegex(self.public_text, re.compile(r"\bstate=[A-Za-z0-9_-]+"))
        self.assertNotIn("workbench.aihub.qualcomm.com/jobs/", self.public_text)
        self.assertNotRegex(self.public_text, re.compile(r"\btoken\s*[:=]\s*\S+", re.I))


if __name__ == "__main__":
    unittest.main()
