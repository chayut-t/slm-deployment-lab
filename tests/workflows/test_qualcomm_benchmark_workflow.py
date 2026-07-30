"""Validate the manual Qualcomm workflow without contacting GitHub or AI Hub."""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "qualcomm-benchmark.yml"
GUIDE_PATH = ROOT / "docs" / "learning" / "github_actions_for_ai_hub.md"
ACTION_SHA_PATTERN = re.compile(r"^actions/[a-z-]+@[0-9a-f]{40}$")


def load_workflow() -> tuple[dict[str, Any], str]:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    value = yaml.load(text, Loader=yaml.BaseLoader)
    if not isinstance(value, dict):
        raise AssertionError("workflow must parse as a mapping")
    return value, text


class QualcommBenchmarkWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow, self.text = load_workflow()
        self.dispatch = self.workflow["on"]["workflow_dispatch"]
        self.jobs = self.workflow["jobs"]

    def test_only_manual_dispatch_can_trigger_the_workflow(self) -> None:
        self.assertEqual(set(self.workflow["on"]), {"workflow_dispatch"})
        self.assertNotIn("pull_request_target", self.text)
        self.assertNotIn("schedule:", self.text)

    def test_dispatch_has_required_enumerated_workload_inputs(self) -> None:
        inputs = self.dispatch["inputs"]
        expected = {
            "target": [
                "snapdragon-x-elite",
                "dragonwing-iq-9075",
                "snapdragon-8-elite",
            ],
            "context": ["128", "512", "1024", "4096"],
            "precision": ["fp16", "w8a16", "w8a8", "w4a8"],
            "stage": ["compile", "inference", "profile"],
        }
        for name, options in expected.items():
            with self.subTest(input=name):
                definition = inputs[name]
                self.assertEqual(definition["required"], "true")
                self.assertEqual(definition["type"], "choice")
                self.assertEqual(definition["options"], options)
        run_id = inputs["request_artifact_run_id"]
        self.assertEqual(run_id["required"], "true")
        self.assertEqual(run_id["type"], "string")

    def test_permissions_and_authorization_are_fork_safe(self) -> None:
        self.assertEqual(
            self.workflow["permissions"],
            {"contents": "read", "actions": "read"},
        )
        condition = self.jobs["authorize"]["if"]
        self.assertIn("github.event.repository.fork == false", condition)
        self.assertIn("github.event.repository.default_branch", condition)
        self.assertIn("github.ref ==", condition)
        authorize_text = yaml.safe_dump(self.jobs["authorize"], sort_keys=True)
        self.assertNotIn("QAI_HUB_API_TOKEN", authorize_text)
        self.assertIn("head_repository", authorize_text)
        self.assertIn('run.get("conclusion") == "success"', self.text)

    def test_secret_is_confined_to_protected_submit_job(self) -> None:
        submit = self.jobs["submit"]
        self.assertEqual(submit["environment"], "qualcomm-ai-hub")
        self.assertEqual(self.text.count("secrets.QAI_HUB_API_TOKEN"), 1)
        secret_steps = [
            step
            for step in submit["steps"]
            if "QAI_HUB_API_TOKEN" in yaml.safe_dump(step, sort_keys=True)
        ]
        self.assertEqual(len(secret_steps), 1)
        self.assertEqual(
            secret_steps[0]["name"],
            "Configure the client without printing the secret",
        )
        self.assertNotIn("qai-hub configure", self.text)
        self.assertIn('destination.chmod(0o600)', self.text)
        self.assertIn("if: always()", self.text)
        self.assertIn('unlink "$config_path"', self.text)

    def test_official_actions_and_client_are_immutably_pinned(self) -> None:
        action_uses = [
            step["uses"]
            for step in self.jobs["submit"]["steps"]
            if "uses" in step
        ]
        self.assertGreaterEqual(len(action_uses), 4)
        for action in action_uses:
            with self.subTest(action=action):
                self.assertRegex(action, ACTION_SHA_PATTERN)
        self.assertIn('python-version: "3.11.13"', self.text)
        self.assertIn('"qai-hub==0.53.0"', self.text)
        self.assertIn('version("qai-hub") == "0.53.0"', self.text)

    def test_workflow_delegates_each_stage_to_the_local_t30_script(self) -> None:
        expected_scripts = {
            "scripts/qualcomm/compile.py",
            "scripts/qualcomm/inference.py",
            "scripts/qualcomm/profile.py",
        }
        for script in expected_scripts:
            self.assertEqual(self.text.count(script), 1)
        self.assertIn('PYTHONPATH=src python "$SCRIPT"', self.text)
        self.assertIn('--request "$REQUEST"', self.text)
        self.assertIn('--manifest "$MANIFEST"', self.text)
        self.assertNotIn("submit_compile_job", self.text)
        self.assertNotIn("submit_inference_job", self.text)
        self.assertNotIn("submit_profile_job", self.text)

    def test_embedded_python_is_syntactically_valid(self) -> None:
        scripts = re.findall(
            r"^\s+python(?:3)? - <<'PY'\n(.*?)^\s+PY$",
            self.text,
            flags=re.DOTALL | re.MULTILINE,
        )
        self.assertEqual(len(scripts), 2)
        for index, script in enumerate(scripts):
            lines = script.splitlines()
            indentation = min(
                len(line) - len(line.lstrip()) for line in lines if line.strip()
            )
            source = "\n".join(line[indentation:] for line in lines) + "\n"
            with self.subTest(script=index):
                compile(source, f"<workflow-python-{index}>", "exec")

    def test_private_bundle_is_not_uploaded_and_only_manifest_is_published(
        self,
    ) -> None:
        submit_steps = self.jobs["submit"]["steps"]
        download = next(
            step for step in submit_steps if step["name"].startswith("Download")
        )
        self.assertEqual(download["with"]["name"], "qualcomm-request-bundle")
        self.assertEqual(
            download["with"]["run-id"],
            "${{ needs.authorize.outputs.request-artifact-run-id }}",
        )
        upload = next(
            step for step in submit_steps if step["name"].startswith("Upload")
        )
        self.assertEqual(
            upload["with"]["path"],
            "${{ steps.selection.outputs.manifest }}",
        )
        self.assertEqual(upload["with"]["if-no-files-found"], "error")
        self.assertEqual(upload["with"]["retention-days"], "14")
        self.assertNotIn("artifacts/qualcomm-request-bundle/**", self.text)

    def test_time_and_concurrency_are_bounded(self) -> None:
        self.assertEqual(self.jobs["authorize"]["timeout-minutes"], "5")
        self.assertEqual(self.jobs["submit"]["timeout-minutes"], "120")
        concurrency = self.workflow["concurrency"]
        self.assertEqual(concurrency["cancel-in-progress"], "false")
        for name in ("target", "context", "precision", "stage"):
            self.assertIn(f"inputs.{name}", concurrency["group"])

    def test_guide_teaches_security_and_unverified_external_boundary(self) -> None:
        guide = GUIDE_PATH.read_text(encoding="utf-8")
        for phrase in (
            "learner-owned hands-on",
            "default branch",
            "protected `qualcomm-ai-hub` GitHub environment",
            "request-bundle",
            "service turnaround",
            "does **not** add or inspect a real secret",
            "structurally validated but externally unverified",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, guide)


if __name__ == "__main__":
    unittest.main()
