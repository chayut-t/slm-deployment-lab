"""Regression tests for the T00 model and toolchain-version contract."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "configs" / "models" / "qwen3-0.6b.yaml"
ADR_PATH = REPO_ROOT / "docs" / "decisions" / "0001-model-and-version-pins.md"
PLAN_PATH = REPO_ROOT / "docs" / "project" / "plan.md"
TASK_PATH = REPO_ROOT / "ai" / "tasks" / "definitions" / "T00.yaml"
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        cls.adr = ADR_PATH.read_text(encoding="utf-8")
        cls.plan = PLAN_PATH.read_text(encoding="utf-8")
        cls.task = json.loads(TASK_PATH.read_text(encoding="utf-8"))

    def test_task_outputs_exist(self) -> None:
        for output in self.task["outputs"]:
            self.assertTrue((REPO_ROOT / output).is_file(), output)

    def test_model_and_tokenizer_share_an_immutable_revision(self) -> None:
        model = self.contract["model"]
        tokenizer = self.contract["tokenizer"]
        self.assertEqual(model["id"], "Qwen/Qwen3-0.6B")
        self.assertEqual(tokenizer["id"], model["id"])
        self.assertRegex(model["revision"], FULL_SHA)
        self.assertEqual(tokenizer["revision"], model["revision"])
        self.assertNotIn(model["revision"], {"main", "master", "latest"})
        self.assertFalse(model["trust_remote_code"])

    def test_metadata_digests_are_complete_and_recorded_in_adr(self) -> None:
        metadata = self.contract["source_metadata"]
        digests = (
            metadata["config_json_sha256"],
            metadata["tokenizer_config_json_sha256"],
            self.contract["tokenizer"]["chat_template"]["sha256"],
        )
        for digest in digests:
            self.assertRegex(digest, SHA256)
            self.assertIn(digest, self.adr)
        self.assertIn(self.contract["model"]["revision"], self.adr)

    def test_special_tokens_preserve_model_tokenizer_distinction(self) -> None:
        model_tokens = self.contract["model"][
            "special_token_ids_from_model_config"
        ]
        tokenizer = self.contract["tokenizer"]
        tokenizer_tokens = tokenizer["tokens"]
        self.assertFalse(tokenizer["add_bos_token"])
        self.assertIsNone(tokenizer_tokens["bos"])
        self.assertEqual(model_tokens["bos"], tokenizer_tokens["pad_id"])
        self.assertEqual(model_tokens["eos"], tokenizer_tokens["eos_id"])

    def test_scope_matches_project_plan(self) -> None:
        project = self.contract["project_contract"]
        context_section = self.plan.split("### 6.2 Context matrix", 1)[1].split(
            "### 6.3 Reference levels",
            1,
        )[0]
        plan_contexts = [
            int(value.replace(",", ""))
            for value in re.findall(
                r"^\|\s*([0-9,]+)\s*\|",
                context_section,
                flags=re.MULTILINE,
            )
        ]
        self.assertEqual(project["static_context_lengths"], plan_contexts)
        self.assertEqual(
            project["platform_priority"],
            [
                "qualcomm_public_ai_hub_and_device_cloud",
                "apple_m4_mlx",
                "nvidia_onnx_runtime_cuda",
            ],
        )
        self.assertEqual(
            project["canonical_validation_prompt_interface"],
            "raw_completion",
        )
        self.assertTrue(project["chat_validation_requires_enable_thinking_false"])
        self.assertIn("Primary model: `Qwen/Qwen3-0.6B`", self.plan)
        priority_markers = [
            "**Qualcomm, protected priority:**",
            "**Apple Silicon, second priority:**",
            "**Linux/NVIDIA, third priority:**",
        ]
        positions = [self.plan.index(marker) for marker in priority_markers]
        self.assertEqual(positions, sorted(positions))
        self.assertIn(
            "Qwen3 remains the only primary model",
            self.plan,
        )

    def test_runtime_and_compiler_version_fields_are_mandatory(self) -> None:
        policy = self.contract["toolchain_version_policy"]
        fields = policy["required_fields"]
        self.assertTrue(policy["exact_package_versions_required"])
        self.assertTrue(policy["git_sources_require_full_commit_sha"])
        self.assertIn("compiler_version", fields["common_environment_and_build"])
        self.assertIn("runtime_version", fields["common_environment_and_build"])

        platform_requirements = {
            "qualcomm": {
                "qai_hub_version",
                "aimet_version",
                "qairt_version",
                "qnn_compiler_version",
                "qnn_runtime_version",
            },
            "apple": {
                "mlx_version",
                "macos_version",
                "xcode_version",
                "metal_version",
            },
            "nvidia": {
                "onnxruntime_gpu_version",
                "cuda_version",
                "cudnn_version",
                "nvidia_driver_version",
            },
        }
        for platform, required in platform_requirements.items():
            self.assertLessEqual(required, set(fields[platform]))

    def test_artifact_manifest_fields_match_project_plan(self) -> None:
        manifest_block = self.plan.split("### 17.4 Artifact manifest", 1)[1].split(
            "```yaml",
            1,
        )[1].split("```", 1)[0]
        plan_fields = re.findall(
            r"^([a-z][a-z0-9_]*):\s*$",
            manifest_block,
            flags=re.MULTILINE,
        )
        contract_fields = self.contract["toolchain_version_policy"][
            "required_fields"
        ]["artifact_manifest"]
        self.assertEqual(contract_fields, plan_fields)

    @unittest.skipUnless(
        os.environ.get("SLM_LAB_VERIFY_UPSTREAM") == "1",
        "set SLM_LAB_VERIFY_UPSTREAM=1 to verify pinned public metadata",
    )
    def test_pinned_upstream_metadata(self) -> None:
        model = self.contract["model"]
        revision = model["revision"]
        base_url = f"{model['repository_url']}/raw/{revision}"

        with urllib.request.urlopen(f"{base_url}/config.json", timeout=30) as response:
            config_raw = response.read()
        with urllib.request.urlopen(
            f"{base_url}/tokenizer_config.json",
            timeout=30,
        ) as response:
            tokenizer_raw = response.read()

        metadata = self.contract["source_metadata"]
        self.assertEqual(
            hashlib.sha256(config_raw).hexdigest(),
            metadata["config_json_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(tokenizer_raw).hexdigest(),
            metadata["tokenizer_config_json_sha256"],
        )
        tokenizer = json.loads(tokenizer_raw)
        self.assertEqual(
            hashlib.sha256(tokenizer["chat_template"].encode()).hexdigest(),
            self.contract["tokenizer"]["chat_template"]["sha256"],
        )


if __name__ == "__main__":
    unittest.main()
