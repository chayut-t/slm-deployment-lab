"""Regression tests for T01 manifests and artifact storage."""

from __future__ import annotations

import hashlib
import json
import tempfile
import tomllib
import unittest
from pathlib import Path

from slm_lab.manifests.storage import StoragePreflightError, run_preflight
from slm_lab.manifests.validation import (
    ManifestValidationError,
    load_document,
    load_schema,
    validate_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "manifests"
HOST_MANIFEST = REPO_ROOT / "results" / "hosts" / "apple-m4-primary.json"
MODEL_CONTRACT = REPO_ROOT / "configs" / "models" / "qwen3-0.6b.yaml"
STORAGE_CONFIG = (
    REPO_ROOT / "configs" / "storage" / "external-ssd.example.yaml"
)
TOOLCHAIN_CONFIG = REPO_ROOT / "environments" / "common-toolchain.json"


class ManifestTests(unittest.TestCase):
    def test_representative_artifact_and_host_validate(self) -> None:
        validate_manifest(
            "artifact",
            load_document(FIXTURES / "artifact.valid.json"),
        )
        validate_manifest("host", load_document(HOST_MANIFEST))

    def test_invalid_artifact_reports_provenance_failures(self) -> None:
        with self.assertRaises(ManifestValidationError) as context:
            validate_manifest(
                "artifact",
                load_document(FIXTURES / "artifact.invalid.json"),
            )
        message = str(context.exception)
        self.assertIn("model_revision", message)
        self.assertIn("runtime_version", message)
        self.assertIn("context_length", message)
        self.assertIn("created_at", message)

    def test_artifact_schema_required_fields_match_t00_contract(self) -> None:
        contract = json.loads(MODEL_CONTRACT.read_text(encoding="utf-8"))
        expected = contract["toolchain_version_policy"]["required_fields"][
            "artifact_manifest"
        ]
        self.assertEqual(load_schema("artifact")["required"], expected)

    def test_host_manifest_references_current_lock(self) -> None:
        host = load_document(HOST_MANIFEST)
        lock_digest = hashlib.sha256((REPO_ROOT / "uv.lock").read_bytes()).hexdigest()
        self.assertEqual(
            host["project_environment"]["package_lock_sha256"],
            lock_digest,
        )


class EnvironmentContractTests(unittest.TestCase):
    def test_python_uv_and_direct_dependencies_are_exact(self) -> None:
        toolchain = json.loads(TOOLCHAIN_CONFIG.read_text(encoding="utf-8"))
        project = tomllib.loads(
            (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        python_version = toolchain["python"]["version"]
        self.assertEqual(
            (REPO_ROOT / ".python-version").read_text(encoding="utf-8").strip(),
            python_version,
        )
        self.assertEqual(
            project["project"]["requires-python"],
            f">={python_version},<3.12",
        )
        direct_requirements = (
            project["build-system"]["requires"]
            + project["project"]["dependencies"]
            + project["project"]["optional-dependencies"]["dev"]
        )
        self.assertTrue(direct_requirements)
        for requirement in direct_requirements:
            self.assertRegex(requirement, r"^[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9_.-]*$")

    def test_platform_extensions_have_explicit_task_owners(self) -> None:
        toolchain = json.loads(TOOLCHAIN_CONFIG.read_text(encoding="utf-8"))
        extensions = toolchain["platform_extensions"]
        self.assertEqual(extensions["macos_m4"]["owner_task"], "T50")
        self.assertEqual(extensions["linux_cuda"]["owner_task"], "T60")
        self.assertEqual(extensions["linux_aimet"]["owner_task"], "T40")
        self.assertEqual(extensions["qualcomm_hosted"]["owner_task"], "T30")
        for extension in extensions.values():
            self.assertTrue(extension["status"].startswith("deferred_"))


class StoragePreflightTests(unittest.TestCase):
    def make_config(self, root: Path) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact_root_env": "SLM_LAB_TEST_ARTIFACT_ROOT",
            "primary_machine_default": str(root),
            "required_mount": "/",
            "minimum_free_bytes": 1,
            "write_probe_prefix": ".test-preflight-",
            "expected_directories": ["models", "onnx/reference"],
        }

    def test_preflight_checks_layout_space_and_removes_write_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ("models", "onnx/reference"):
                (root / relative).mkdir(parents=True, exist_ok=True)
            result = run_preflight(self.make_config(root))
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["write_probe"], "passed")
            self.assertEqual(list(root.glob(".test-preflight-*")), [])

    def test_preflight_rejects_incomplete_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "models").mkdir()
            with self.assertRaisesRegex(
                StoragePreflightError,
                "onnx/reference",
            ):
                run_preflight(self.make_config(root), write_probe=False)

    def test_preflight_rejects_layout_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.make_config(root)
            config["expected_directories"] = ["../outside"]
            with self.assertRaisesRegex(
                StoragePreflightError,
                "unsafe paths",
            ):
                run_preflight(config, write_probe=False)

    def test_primary_storage_config_is_internally_consistent(self) -> None:
        config = json.loads(STORAGE_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["schema_version"], 1)
        self.assertGreaterEqual(config["minimum_free_bytes"], 100 * 1024**3)
        self.assertEqual(
            Path(config["primary_machine_default"]).parent,
            Path(config["required_mount"]),
        )


if __name__ == "__main__":
    unittest.main()
