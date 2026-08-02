"""Validate the manual Qualcomm workflow without contacting GitHub or AI Hub."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile
from pathlib import Path
from typing import Any
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "qualcomm-benchmark.yml"
GUIDE_PATH = ROOT / "docs" / "learning" / "github_actions_for_ai_hub.md"
PRODUCER_PATH = ROOT / ".github" / "workflows" / "qualcomm-request-bundle.yml"
ACTION_SHA_PATTERN = re.compile(r"^actions/[a-z-]+@[0-9a-f]{40}$")


def load_workflow() -> tuple[dict[str, Any], str]:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    value = yaml.load(text, Loader=yaml.BaseLoader)
    if not isinstance(value, dict):
        raise AssertionError("workflow must parse as a mapping")
    return value, text


def load_producer() -> tuple[dict[str, Any], str]:
    text = PRODUCER_PATH.read_text(encoding="utf-8")
    value = yaml.load(text, Loader=yaml.BaseLoader)
    if not isinstance(value, dict):
        raise AssertionError("producer workflow must parse as a mapping")
    return value, text


def embedded_python(step: dict[str, Any]) -> str:
    run = step["run"]
    marker = "<<'PY'\n"
    if marker not in run:
        raise AssertionError(f"{step['name']} has no embedded Python")
    source = run.split(marker, 1)[1].rsplit("\nPY", 1)[0]
    return textwrap.dedent(source) + "\n"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prefill_specs(context: int) -> dict[str, dict[str, Any]]:
    return {
        "input_ids": {"shape": [1, context], "dtype": "int64"},
        "attention_mask": {"shape": [1, context], "dtype": "int64"},
        "position_ids": {"shape": [1, context], "dtype": "int64"},
    }


def decode_specs(context: int) -> dict[str, dict[str, Any]]:
    capacity = {128: 160, 512: 576, 1024: 1152, 4096: 4224}[context]
    specs = {
        "input_ids": {"shape": [1, 1], "dtype": "int64"},
        "attention_mask": {"shape": [1, capacity], "dtype": "int64"},
        "position_ids": {"shape": [1, 1], "dtype": "int64"},
    }
    for layer in range(28):
        specs[f"key_cache.{layer}"] = {
            "shape": [1, 8, capacity, 128],
            "dtype": "float16",
        }
        specs[f"value_cache.{layer}"] = {
            "shape": [1, 8, capacity, 128],
            "dtype": "float16",
        }
    specs["valid_length"] = {"shape": [1], "dtype": "int64"}
    return specs


class QualcommBenchmarkWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow, self.text = load_workflow()
        self.dispatch = self.workflow["on"]["workflow_dispatch"]
        self.jobs = self.workflow["jobs"]
        self.authorize_steps = self.jobs["authorize"]["steps"]
        self.submit_steps = self.jobs["submit"]["steps"]
        self.producer, self.producer_text = load_producer()
        self.producer_steps = self.producer["jobs"]["prepare"]["steps"]

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
        for name in (
            "request_artifact_run_id",
            "request_bundle_manifest_sha256",
        ):
            with self.subTest(input=name):
                definition = inputs[name]
                self.assertEqual(definition["required"], "true")
                self.assertEqual(definition["type"], "string")

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
        self.assertIn('run.get("head_sha") ==', self.text)
        self.assertIn('run.get("path") ==', self.text)
        self.assertIn(
            "EXPECTED_PRODUCER_WORKFLOW: .github/workflows/qualcomm-request-bundle.yml",
            self.text,
        )

    def test_successful_default_branch_run_from_wrong_producer_is_rejected(
        self,
    ) -> None:
        provenance_step = next(
            step for step in self.authorize_steps if step.get("id") == "provenance"
        )
        source = embedded_python(provenance_step)
        revision = "a" * 40
        run = {
            "conclusion": "success",
            "head_branch": "main",
            "head_sha": revision,
            "path": ".github/workflows/unreviewed-producer.yml",
            "head_repository": {"full_name": "owner/repository"},
            "event": "workflow_dispatch",
        }
        response = io.BytesIO(json.dumps(run).encode("utf-8"))
        environment = {
            "API_URL": "https://api.github.invalid",
            "DEFAULT_BRANCH": "main",
            "EXPECTED_PRODUCER_REVISION": revision,
            "EXPECTED_PRODUCER_WORKFLOW": (
                ".github/workflows/qualcomm-request-bundle.yml"
            ),
            "GH_TOKEN": "synthetic-github-token",
            "REPOSITORY": "owner/repository",
            "REQUEST_ARTIFACT_RUN_ID": "42",
        }
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch("urllib.request.urlopen", return_value=response),
            self.assertRaisesRegex(SystemExit, "reviewed producer workflow"),
        ):
            exec(compile(source, "<provenance>", "exec"), {})

    def test_producer_is_manual_fork_safe_read_only_and_secret_free(self) -> None:
        self.assertEqual(set(self.producer["on"]), {"workflow_dispatch"})
        self.assertEqual(self.producer["permissions"], {"contents": "read"})
        condition = self.producer["jobs"]["prepare"]["if"]
        self.assertIn("github.event.repository.fork == false", condition)
        self.assertIn("github.event.repository.default_branch", condition)
        self.assertNotIn("QAI_HUB_API_TOKEN", self.producer_text)
        self.assertNotIn("pull_request_target", self.producer_text)
        inputs = self.producer["on"]["workflow_dispatch"]["inputs"]
        for name in (
            "source_release_tag",
            "source_asset_name",
            "source_archive_sha256",
        ):
            self.assertEqual(inputs[name]["required"], "true")
        upload = next(
            step
            for step in self.producer_steps
            if step["name"].startswith("Upload only")
        )
        self.assertEqual(upload["with"]["name"], "qualcomm-request-bundle")
        self.assertEqual(
            upload["with"]["path"],
            "artifacts/qualcomm-request-bundle",
        )

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
        self.assertIn("destination.chmod(0o600)", self.text)
        self.assertIn("if: always()", self.text)
        self.assertIn('unlink "$config_path"', self.text)

    def test_official_actions_and_client_are_immutably_pinned(self) -> None:
        action_uses = [step["uses"] for step in self.submit_steps if "uses" in step]
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
        python_steps = [
            step
            for step in (self.authorize_steps + self.submit_steps + self.producer_steps)
            if "<<'PY'" in step.get("run", "")
        ]
        self.assertEqual(len(python_steps), 5)
        for index, step in enumerate(python_steps):
            with self.subTest(script=index):
                compile(
                    embedded_python(step),
                    f"<workflow-python-{index}>",
                    "exec",
                )

    def test_private_bundle_is_not_uploaded_and_only_manifest_is_published(
        self,
    ) -> None:
        download = next(
            step for step in self.submit_steps if step["name"].startswith("Download")
        )
        self.assertEqual(download["with"]["name"], "qualcomm-request-bundle")
        self.assertEqual(
            download["with"]["run-id"],
            "${{ needs.authorize.outputs.request-artifact-run-id }}",
        )
        upload = next(
            step for step in self.submit_steps if step["name"].startswith("Upload")
        )
        self.assertEqual(
            upload["with"]["path"],
            "${{ steps.selection.outputs.manifest }}",
        )
        self.assertEqual(upload["with"]["if-no-files-found"], "error")
        self.assertEqual(upload["with"]["retention-days"], "14")
        self.assertNotIn("artifacts/qualcomm-request-bundle/**", self.text)

    def make_compile_bundle(
        self,
        request_mutator=None,
    ) -> tuple[Path, dict[str, str]]:
        root = Path(tempfile.mkdtemp(prefix="slm-lab-t72-workflow-test-"))
        self.addCleanup(shutil.rmtree, root, True)
        bundle_root = root / "artifacts" / "qualcomm-request-bundle"
        request_relative = Path("snapdragon-x-elite/128/fp16/compile-request.json")
        request_path = bundle_root / request_relative
        source_path = bundle_root / "inputs" / "model.onnx"
        source_path.parent.mkdir(parents=True)
        request_path.parent.mkdir(parents=True)
        source_path.write_bytes(b"synthetic-onnx")
        outside_path = root / "outside" / "model.onnx"
        outside_path.parent.mkdir()
        outside_path.write_bytes(b"synthetic-outside-onnx")

        request = {
            "schema_version": 2,
            "stage": "compile",
            "client_version": "0.53.0",
            "device": {
                "name": "Snapdragon X Elite CRD",
                "os": "",
                "attributes": [],
            },
            "runtime": {"name": "QAIRT", "version": "2.45.0.260326154327"},
            "source_artifact": {
                "path": "artifacts/qualcomm-request-bundle/inputs/model.onnx",
                "logical_name": "qwen3-0.6b-prefill-s128-fp16.onnx",
                "sha256": sha256(source_path),
            },
            "output_artifact": (
                "artifacts/qualcomm-actions-private/snapdragon-x-elite/"
                "128/fp16/compiled.bin"
            ),
            "output_logical_name": "qwen3-0.6b-prefill-s128-fp16.qnn.bin",
            "input_specs": prefill_specs(128),
            "options": (
                "--target_runtime qnn_context_binary "
                "--qairt_version 2.45.0.260326154327"
            ),
            "job_name": "slm-lab-t72-snapdragon-x-elite-128-fp16-compile",
            "timeout_seconds": 3600,
            "retry": False,
        }
        if request_mutator is not None:
            request_mutator(request)
        request_path.write_text(
            json.dumps(request, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        files = [
            {
                "path": request_relative.as_posix(),
                "sha256": sha256(request_path),
            },
            {"path": "inputs/model.onnx", "sha256": sha256(source_path)},
        ]
        revision = "a" * 40
        bundle_manifest = {
            "schema_version": 1,
            "producer": {
                "workflow_path": (".github/workflows/qualcomm-request-bundle.yml"),
                "revision": revision,
                "run_id": 42,
            },
            "source": {
                "release_tag": "qualcomm-source-v1",
                "asset_name": "qualcomm-source.zip",
                "archive_sha256": "c" * 64,
            },
            "selection": {
                "target": "snapdragon-x-elite",
                "context": 128,
                "precision": "fp16",
                "stage": "compile",
            },
            "request": {
                "path": request_relative.as_posix(),
                "sha256": sha256(request_path),
            },
            "files": files,
        }
        manifest_path = bundle_root / "bundle-manifest.json"
        manifest_path.write_text(
            json.dumps(bundle_manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        environment = {
            **os.environ,
            "BUNDLE_ROOT": "artifacts/qualcomm-request-bundle",
            "CONTEXT": "128",
            "EXPECTED_BUNDLE_MANIFEST_SHA256": sha256(manifest_path),
            "EXPECTED_PRODUCER_REVISION": revision,
            "EXPECTED_PRODUCER_RUN_ID": "42",
            "EXPECTED_PRODUCER_WORKFLOW": (
                ".github/workflows/qualcomm-request-bundle.yml"
            ),
            "MANIFEST": (
                "results/qualcomm-actions/snapdragon-x-elite/128/fp16/"
                "compile-manifest.json"
            ),
            "PRECISION": "fp16",
            "PRIVATE_OUTPUT_ROOT": "artifacts/qualcomm-actions-private",
            "PUBLIC_MANIFEST_ROOT": "results/qualcomm-actions",
            "REQUEST": (
                f"artifacts/qualcomm-request-bundle/{request_relative.as_posix()}"
            ),
            "STAGE": "compile",
            "TARGET": "snapdragon-x-elite",
        }
        return root, environment

    def make_later_stage_bundle(
        self,
        stage: str,
        *,
        predecessor_mutator=None,
        request_mutator=None,
    ) -> tuple[Path, dict[str, str]]:
        if stage not in {"inference", "profile"}:
            raise ValueError(stage)
        root = Path(tempfile.mkdtemp(prefix="slm-lab-t72-later-stage-test-"))
        self.addCleanup(shutil.rmtree, root, True)
        bundle_root = root / "artifacts" / "qualcomm-request-bundle"
        tuple_root = bundle_root / "snapdragon-x-elite" / "128" / "fp16"
        inputs_root = bundle_root / "inputs"
        tuple_root.mkdir(parents=True)
        inputs_root.mkdir(parents=True)

        stem = "qwen3-0.6b-prefill-s128-fp16"
        compiled_path = inputs_root / f"{stem}.qnn.bin"
        compiled_path.write_bytes(b"synthetic-compiled")
        compiled_sha = sha256(compiled_path)
        predecessor = {
            "schema_version": 2,
            "manifest_type": "slm_lab.qualcomm.ai_hub.stage",
            "stage": "compile",
            "status": "success",
            "graph_contract": {"input_specs": prefill_specs(128)},
            "lineage": {
                "source_artifacts": [
                    {
                        "role": "source_model",
                        "logical_name": f"{stem}.onnx",
                        "sha256": "d" * 64,
                        "byte_size": 1,
                    }
                ]
            },
            "result": {
                "target_artifact": {
                    "role": "compiled_model",
                    "logical_name": f"{stem}.qnn.bin",
                    "sha256": compiled_sha,
                    "byte_size": compiled_path.stat().st_size,
                }
            },
        }
        if predecessor_mutator is not None:
            predecessor_mutator(predecessor)
        predecessor_path = inputs_root / f"{stem}.compile-manifest.json"
        predecessor_path.write_text(
            json.dumps(predecessor, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        request = {
            "schema_version": 2,
            "stage": stage,
            "client_version": "0.53.0",
            "device": {
                "name": "Snapdragon X Elite CRD",
                "os": "",
                "attributes": [],
            },
            "runtime": {"name": "QAIRT", "version": "2.45.0.260326154327"},
            "predecessor_manifest": (
                f"artifacts/qualcomm-request-bundle/inputs/{stem}.compile-manifest.json"
            ),
            "compiled_artifact": {
                "path": (f"artifacts/qualcomm-request-bundle/inputs/{stem}.qnn.bin"),
                "logical_name": f"{stem}.qnn.bin",
                "sha256": compiled_sha,
            },
            "options": "--qairt_framework 2.45.0.260326154327",
            "job_name": (f"slm-lab-t72-snapdragon-x-elite-128-fp16-{stage}"),
            "timeout_seconds": 3600,
            "retry": False,
        }
        if stage == "inference":
            dataset_path = inputs_root / f"{stem}.inputs.h5"
            dataset_path.write_bytes(b"synthetic-dataset")
            request.update(
                {
                    "input_dataset": {
                        "path": (
                            f"artifacts/qualcomm-request-bundle/inputs/{stem}.inputs.h5"
                        ),
                        "logical_name": f"{stem}.inputs.h5",
                        "sha256": sha256(dataset_path),
                    },
                    "output_artifact": (
                        "artifacts/qualcomm-actions-private/"
                        "snapdragon-x-elite/128/fp16/outputs.h5"
                    ),
                    "output_logical_name": f"{stem}.outputs.h5",
                }
            )
        else:
            request.update(
                {
                    "raw_profile_output": (
                        "artifacts/qualcomm-actions-private/"
                        "snapdragon-x-elite/128/fp16/profile.json"
                    ),
                    "raw_profile_logical_name": f"{stem}.profile.json",
                }
            )
        if request_mutator is not None:
            request_mutator(request)
        request_relative = Path(f"snapdragon-x-elite/128/fp16/{stage}-request.json")
        request_path = bundle_root / request_relative
        request_path.write_text(
            json.dumps(request, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        files = []
        for path in sorted(bundle_root.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": path.relative_to(bundle_root).as_posix(),
                        "sha256": sha256(path),
                    }
                )
        revision = "a" * 40
        bundle_manifest = {
            "schema_version": 1,
            "producer": {
                "workflow_path": (".github/workflows/qualcomm-request-bundle.yml"),
                "revision": revision,
                "run_id": 42,
            },
            "source": {
                "release_tag": "qualcomm-source-v1",
                "asset_name": "qualcomm-source.zip",
                "archive_sha256": "c" * 64,
            },
            "selection": {
                "target": "snapdragon-x-elite",
                "context": 128,
                "precision": "fp16",
                "stage": stage,
            },
            "request": {
                "path": request_relative.as_posix(),
                "sha256": sha256(request_path),
            },
            "files": files,
        }
        manifest_path = bundle_root / "bundle-manifest.json"
        manifest_path.write_text(
            json.dumps(bundle_manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        environment = {
            **os.environ,
            "BUNDLE_ROOT": "artifacts/qualcomm-request-bundle",
            "CONTEXT": "128",
            "EXPECTED_BUNDLE_MANIFEST_SHA256": sha256(manifest_path),
            "EXPECTED_PRODUCER_REVISION": revision,
            "EXPECTED_PRODUCER_RUN_ID": "42",
            "EXPECTED_PRODUCER_WORKFLOW": (
                ".github/workflows/qualcomm-request-bundle.yml"
            ),
            "MANIFEST": (
                "results/qualcomm-actions/snapdragon-x-elite/128/fp16/"
                f"{stage}-manifest.json"
            ),
            "PRECISION": "fp16",
            "PRIVATE_OUTPUT_ROOT": "artifacts/qualcomm-actions-private",
            "PUBLIC_MANIFEST_ROOT": "results/qualcomm-actions",
            "REQUEST": (
                f"artifacts/qualcomm-request-bundle/{request_relative.as_posix()}"
            ),
            "STAGE": stage,
            "TARGET": "snapdragon-x-elite",
        }
        return root, environment

    def run_bundle_validator(
        self,
        root: Path,
        environment: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        validator = next(
            step
            for step in self.submit_steps
            if step["name"].startswith("Validate bundle manifest")
        )
        return subprocess.run(
            (sys.executable, "-c", embedded_python(validator)),
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def make_source_archive(
        self,
        *,
        request_mutator=None,
        escaped_entry: bool = False,
        unsafe_entry=None,
    ) -> tuple[Path, dict[str, str]]:
        root = Path(tempfile.mkdtemp(prefix="slm-lab-t72-producer-test-"))
        self.addCleanup(shutil.rmtree, root, True)
        archive = root / "artifacts" / "qualcomm-release-source" / "source-bundle.zip"
        archive.parent.mkdir(parents=True)
        request = {
            "schema_version": 2,
            "stage": "compile",
            "device": {
                "name": "Snapdragon X Elite CRD",
                "os": "",
                "attributes": [],
            },
            "job_name": "slm-lab-t72-snapdragon-x-elite-128-fp16-compile",
        }
        if request_mutator is not None:
            request_mutator(request)
        with zipfile.ZipFile(archive, "w") as source:
            source.writestr(
                "snapdragon-x-elite/128/fp16/compile-request.json",
                json.dumps(request, sort_keys=True) + "\n",
            )
            source.writestr("inputs/model.onnx", b"synthetic-onnx")
            if escaped_entry:
                source.writestr("../escaped.txt", b"escape")
            if unsafe_entry is not None:
                source.writestr(unsafe_entry, b"unsafe")
        environment = {
            **os.environ,
            "CONTEXT": "128",
            "PRODUCER_REVISION": "a" * 40,
            "PRODUCER_RUN_ID": "42",
            "PRECISION": "fp16",
            "SOURCE_ARCHIVE_SHA256": sha256(archive),
            "SOURCE_ASSET_NAME": "qualcomm-source.zip",
            "SOURCE_RELEASE_TAG": "qualcomm-source-v1",
            "STAGE": "compile",
            "TARGET": "snapdragon-x-elite",
        }
        return root, environment

    def run_producer_builder(
        self,
        root: Path,
        environment: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        builder = next(
            step
            for step in self.producer_steps
            if step["name"].startswith("Verify source digest")
        )
        return subprocess.run(
            (sys.executable, "-c", embedded_python(builder)),
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def run_producer_input_validator(
        self,
        root: Path,
        environment: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        validator = next(
            step
            for step in self.producer_steps
            if step["name"].startswith("Validate fixed source")
        )
        return subprocess.run(
            ("bash", "-c", validator["run"]),
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_producer_builds_content_addressed_bundle(self) -> None:
        root, environment = self.make_source_archive()
        result = self.run_producer_builder(root, environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(
            result.stdout,
            r"^bundle-manifest-sha256=[0-9a-f]{64}\n$",
        )
        manifest_path = (
            root / "artifacts" / "qualcomm-request-bundle" / "bundle-manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["producer"]["workflow_path"],
            ".github/workflows/qualcomm-request-bundle.yml",
        )
        self.assertEqual(manifest["producer"]["revision"], "a" * 40)
        self.assertEqual(manifest["producer"]["run_id"], 42)
        self.assertEqual(
            manifest["source"],
            {
                "release_tag": "qualcomm-source-v1",
                "asset_name": "qualcomm-source.zip",
                "archive_sha256": environment["SOURCE_ARCHIVE_SHA256"],
            },
        )

    def test_producer_rejects_wrong_reviewed_source_digest(self) -> None:
        root, environment = self.make_source_archive()
        environment["SOURCE_ARCHIVE_SHA256"] = "0" * 64
        result = self.run_producer_builder(root, environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match reviewed SHA-256", result.stderr)

    def test_producer_and_consumer_enforce_source_name_rule_parity(
        self,
    ) -> None:
        fields = (
            ("release_tag", "SOURCE_RELEASE_TAG"),
            ("asset_name", "SOURCE_ASSET_NAME"),
        )
        for manifest_field, environment_field in fields:
            for invalid_value in (".leading-punctuation", "a" * 129):
                with self.subTest(
                    field=manifest_field,
                    value=invalid_value,
                ):
                    producer_root, producer_environment = self.make_source_archive()
                    producer_environment[environment_field] = invalid_value
                    input_result = self.run_producer_input_validator(
                        producer_root,
                        producer_environment,
                    )
                    self.assertNotEqual(input_result.returncode, 0)
                    producer_result = self.run_producer_builder(
                        producer_root,
                        producer_environment,
                    )
                    self.assertNotEqual(producer_result.returncode, 0)
                    self.assertIn(
                        "release tag or asset name is unsafe",
                        producer_result.stderr,
                    )

                    consumer_root, consumer_environment = self.make_compile_bundle()
                    manifest_path = (
                        consumer_root
                        / "artifacts"
                        / "qualcomm-request-bundle"
                        / "bundle-manifest.json"
                    )
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["source"][manifest_field] = invalid_value
                    manifest_path.write_text(
                        json.dumps(manifest, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    consumer_environment["EXPECTED_BUNDLE_MANIFEST_SHA256"] = sha256(
                        manifest_path
                    )
                    consumer_result = self.run_bundle_validator(
                        consumer_root,
                        consumer_environment,
                    )
                    self.assertNotEqual(consumer_result.returncode, 0)
                    self.assertIn(
                        "source provenance value is unsafe",
                        consumer_result.stderr,
                    )

        producer_root, producer_environment = self.make_source_archive()
        producer_environment["SOURCE_RELEASE_TAG"] = "a" * 128
        producer_environment["SOURCE_ASSET_NAME"] = "b" * 128
        input_result = self.run_producer_input_validator(
            producer_root,
            producer_environment,
        )
        self.assertEqual(input_result.returncode, 0, input_result.stderr)
        producer_result = self.run_producer_builder(
            producer_root,
            producer_environment,
        )
        self.assertEqual(producer_result.returncode, 0, producer_result.stderr)

        consumer_root, consumer_environment = self.make_compile_bundle()
        manifest_path = (
            consumer_root
            / "artifacts"
            / "qualcomm-request-bundle"
            / "bundle-manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source"]["release_tag"] = "a" * 128
        manifest["source"]["asset_name"] = "b" * 128
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        consumer_environment["EXPECTED_BUNDLE_MANIFEST_SHA256"] = sha256(manifest_path)
        consumer_result = self.run_bundle_validator(
            consumer_root,
            consumer_environment,
        )
        self.assertEqual(consumer_result.returncode, 0, consumer_result.stderr)

    def test_producer_rejects_source_archive_path_escape(self) -> None:
        root, environment = self.make_source_archive(escaped_entry=True)
        result = self.run_producer_builder(root, environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsafe or duplicate path", result.stderr)
        self.assertFalse((root / "artifacts" / "escaped.txt").exists())

    def test_producer_rejects_noncanonical_source_paths(self) -> None:
        for path in (".", "inputs\\model.bin", "inputs//model.bin"):
            with self.subTest(path=path):
                root, environment = self.make_source_archive(unsafe_entry=path)
                result = self.run_producer_builder(root, environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("unsafe or duplicate path", result.stderr)

    def test_producer_rejects_mislabeled_source_tuple(self) -> None:
        def mislabel(request):
            request["job_name"] = "slm-lab-t72-snapdragon-x-elite-128-w8a8-compile"

        root, environment = self.make_source_archive(request_mutator=mislabel)
        result = self.run_producer_builder(root, environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match workload tuple", result.stderr)

    def test_valid_bundle_is_accepted_before_secret_configuration(self) -> None:
        root, environment = self.make_compile_bundle()
        result = self.run_bundle_validator(root, environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "bundle validation passed\n")
        step_names = [step["name"] for step in self.submit_steps]
        self.assertLess(
            step_names.index(
                "Validate bundle manifest request semantics digests and paths"
            ),
            step_names.index("Configure the client without printing the secret"),
        )

    def test_valid_decode_t12_axes_are_accepted(self) -> None:
        def make_decode(request):
            request["input_specs"] = decode_specs(128)
            request["source_artifact"]["logical_name"] = (
                "qwen3-0.6b-decode-s128-fp16.onnx"
            )
            request["output_logical_name"] = "qwen3-0.6b-decode-s128-fp16.qnn.bin"

        root, environment = self.make_compile_bundle(make_decode)
        result = self.run_bundle_validator(root, environment)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_compile_rejects_prefill_bool_and_float_dimension_aliases(
        self,
    ) -> None:
        substitutions = (
            ("input_ids", 0, True),
            ("attention_mask", 1, 128.0),
        )
        for tensor_name, axis, alias in substitutions:
            with self.subTest(
                tensor=tensor_name,
                axis=axis,
                alias=alias,
            ):

                def substitute(request):
                    request["input_specs"][tensor_name]["shape"][axis] = alias

                root, environment = self.make_compile_bundle(substitute)
                result = self.run_bundle_validator(root, environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "exact builtin JSON tensor types",
                    result.stderr,
                )

    def test_compile_rejects_decode_cache_bool_and_float_axis_aliases(
        self,
    ) -> None:
        substitutions = (
            ("key_cache.0", 0, True),
            ("value_cache.27", 2, 160.0),
        )
        for tensor_name, axis, alias in substitutions:
            with self.subTest(
                tensor=tensor_name,
                axis=axis,
                alias=alias,
            ):

                def substitute(request):
                    request["input_specs"] = decode_specs(128)
                    request["input_specs"][tensor_name]["shape"][axis] = alias
                    request["source_artifact"]["logical_name"] = (
                        "qwen3-0.6b-decode-s128-fp16.onnx"
                    )
                    request["output_logical_name"] = (
                        "qwen3-0.6b-decode-s128-fp16.qnn.bin"
                    )

                root, environment = self.make_compile_bundle(substitute)
                result = self.run_bundle_validator(root, environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "exact builtin JSON tensor types",
                    result.stderr,
                )

    def test_valid_inference_and_profile_predecessors_are_accepted(self) -> None:
        for stage in ("inference", "profile"):
            with self.subTest(stage=stage):
                root, environment = self.make_later_stage_bundle(stage)
                result = self.run_bundle_validator(root, environment)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_later_stages_reject_predecessor_numeric_type_aliases(self) -> None:
        cases = (
            ("inference", "input_ids", 0, True, False),
            ("profile", "key_cache.0", 2, 160.0, True),
        )
        for stage, tensor_name, axis, alias, decode in cases:
            with self.subTest(
                stage=stage,
                tensor=tensor_name,
                axis=axis,
                alias=alias,
            ):

                def substitute(predecessor):
                    if decode:
                        predecessor["graph_contract"]["input_specs"] = decode_specs(128)
                    predecessor["graph_contract"]["input_specs"][tensor_name]["shape"][
                        axis
                    ] = alias

                root, environment = self.make_later_stage_bundle(
                    stage,
                    predecessor_mutator=substitute,
                )
                result = self.run_bundle_validator(root, environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "exact builtin JSON tensor types",
                    result.stderr,
                )

    def test_inference_predecessor_precision_conflict_is_rejected(self) -> None:
        def predecessor_w8(predecessor):
            predecessor["lineage"]["source_artifacts"][0]["logical_name"] = (
                "qwen3-0.6b-prefill-s128-w8a8.onnx"
            )

        root, environment = self.make_later_stage_bundle(
            "inference",
            predecessor_mutator=predecessor_w8,
        )
        result = self.run_bundle_validator(root, environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("consistently bind precision label", result.stderr)

    def test_profile_predecessor_context_conflict_is_rejected(self) -> None:
        def predecessor_s512(predecessor):
            predecessor["graph_contract"]["input_specs"] = prefill_specs(512)

        root, environment = self.make_later_stage_bundle(
            "profile",
            predecessor_mutator=predecessor_s512,
        )
        result = self.run_bundle_validator(root, environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly match a T12 tensor/axis contract", result.stderr)

    def test_bundle_with_mislabeled_request_tuple_is_rejected(self) -> None:
        def mislabel(request):
            request["job_name"] = "slm-lab-t72-snapdragon-x-elite-128-w8a8-compile"

        root, environment = self.make_compile_bundle(mislabel)
        result = self.run_bundle_validator(root, environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("job name does not bind", result.stderr)

    def test_bundle_with_contradictory_full_selector_is_rejected(self) -> None:
        def contradict_selector(request):
            request["device"]["os"] = "Android"

        root, environment = self.make_compile_bundle(contradict_selector)
        result = self.run_bundle_validator(root, environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("full device selector conflicts", result.stderr)

    def test_bundle_with_contradictory_precision_metadata_is_rejected(
        self,
    ) -> None:
        def contradict_precision(request):
            request["source_artifact"]["logical_name"] = (
                "qwen3-0.6b-prefill-s128-w8a8.onnx"
            )

        root, environment = self.make_compile_bundle(contradict_precision)
        result = self.run_bundle_validator(root, environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("consistently bind precision label", result.stderr)

    def test_coincidental_context_dimension_is_not_t12_evidence(self) -> None:
        def contradict_context(request):
            request["input_specs"] = {
                "input_ids": {"shape": [1, 127], "dtype": "int64"},
                "attention_mask": {"shape": [1, 127], "dtype": "int64"},
                "position_ids": {"shape": [1, 127], "dtype": "int64"},
                "decoy": {"shape": [1, 128], "dtype": "int64"},
            }

        root, environment = self.make_compile_bundle(contradict_context)
        result = self.run_bundle_validator(root, environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly match a T12 tensor/axis contract", result.stderr)

    def test_bundle_with_input_path_escape_is_rejected(self) -> None:
        def escape(request):
            request["source_artifact"] = {
                "path": "outside/model.onnx",
                "logical_name": "qwen3-prefill-128-fp16.onnx",
                "sha256": hashlib.sha256(b"synthetic-outside-onnx").hexdigest(),
            }

        root, environment = self.make_compile_bundle(escape)
        result = self.run_bundle_validator(root, environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("escaped its allowed runner-private root", result.stderr)

    def test_normalizing_parent_traversal_is_rejected_before_resolution(
        self,
    ) -> None:
        def traverse(request):
            request["source_artifact"]["path"] = (
                "artifacts/qualcomm-request-bundle/inputs/../inputs/model.onnx"
            )

        root, environment = self.make_compile_bundle(traverse)
        result = self.run_bundle_validator(root, environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("canonical repository-relative syntax", result.stderr)

    def test_bundle_with_wrong_artifact_digest_is_rejected(self) -> None:
        def corrupt_digest(request):
            request["source_artifact"]["sha256"] = "0" * 64

        root, environment = self.make_compile_bundle(corrupt_digest)
        result = self.run_bundle_validator(root, environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("artifact digest does not match", result.stderr)

    def test_bundle_manifest_digest_must_match_reviewed_input(self) -> None:
        root, environment = self.make_compile_bundle()
        environment["EXPECTED_BUNDLE_MANIFEST_SHA256"] = "0" * 64
        result = self.run_bundle_validator(root, environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("reviewed digest", result.stderr)

    def test_bundle_from_wrong_producer_revision_is_rejected(self) -> None:
        root, environment = self.make_compile_bundle()
        manifest_path = (
            root / "artifacts" / "qualcomm-request-bundle" / "bundle-manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["producer"]["revision"] = "b" * 40
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        environment["EXPECTED_BUNDLE_MANIFEST_SHA256"] = sha256(manifest_path)
        result = self.run_bundle_validator(root, environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("identity or revision", result.stderr)

    def test_bundle_with_invalid_source_provenance_is_rejected(self) -> None:
        root, environment = self.make_compile_bundle()
        manifest_path = (
            root / "artifacts" / "qualcomm-request-bundle" / "bundle-manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source"]["archive_sha256"] = "not-a-reviewed-digest"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        environment["EXPECTED_BUNDLE_MANIFEST_SHA256"] = sha256(manifest_path)
        result = self.run_bundle_validator(root, environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source archive digest is invalid", result.stderr)

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
            "fixed producer workflow",
            "path escapes",
            "service turnaround",
            "does **not** add or inspect a real secret",
            "structurally validated but externally",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, guide)


if __name__ == "__main__":
    unittest.main()
