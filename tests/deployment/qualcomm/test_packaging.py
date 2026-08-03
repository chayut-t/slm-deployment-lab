"""Adversarial tests for the T22 QNN candidate package builder.

Every fixture here is synthetic. No test contacts a service, imports
``qai_hub``, reads a real candidate graph, or asserts anything about what
Qualcomm AI Hub would accept.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from slm_lab.deployment.qualcomm import packaging
from slm_lab.deployment.qualcomm.ai_hub import (
    AiHubAdapterError,
    run_compile,
)


REPO_ROOT = Path(packaging.__file__).resolve().parents[4]
TARGET_CONFIG = REPO_ROOT / packaging.DEFAULT_TARGET_CONFIG
QAIRT_VERSION = "2.45.0.260326154327"
CACHE_CAPACITY = 160
LAYER_COUNT = 28


class MockCompileJob:
    """Minimal stand-in so an accepted request can be replayed offline."""

    def __init__(self, options: str) -> None:
        self.options = options
        self.device = _MockDevice()
        self.model = _MockTarget()

    def wait(self, timeout: int | None = None) -> Any:
        return _MockStatus()

    def get_target_model(self) -> Any:
        return self.model

    def download_target_model(self, filename: str) -> str:
        Path(filename).write_bytes(b"synthetic-context-binary")
        return filename


class _MockStatus:
    success = True
    code = "SUCCESS"


class _MockDevice:
    name = "Snapdragon X Elite CRD"
    os = "Windows 11"
    attributes: list[str] = []


class _MockTarget:
    metadata = {"QAIRT_SDK_VERSION": QAIRT_VERSION}
    input_spec: dict[Any, Any] = {}
    output_spec: dict[Any, Any] = {}


class MockBackend:
    def __init__(self, options: str) -> None:
        self.options = options
        self.calls: list[str] = []

    @property
    def client_version(self) -> str:
        return "0.53.0"

    def submit_compile(self, **kwargs: Any) -> Any:
        self.calls.append("compile")
        return MockCompileJob(self.options)

    def submit_inference(self, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("inference must not be reached")

    def submit_profile(self, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("profile must not be reached")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _prefill_inputs(context_length: int) -> list[dict[str, Any]]:
    return [
        {"name": "input_ids", "dtype": "int64", "shape": [1, context_length]},
        {"name": "attention_mask", "dtype": "int64", "shape": [1, context_length]},
        {"name": "position_ids", "dtype": "int64", "shape": [1, context_length]},
    ]


def _decode_inputs(cache_capacity: int) -> list[dict[str, Any]]:
    """Mirror the committed decode boundary: 4 scalars plus 56 cache tensors."""

    tensors: list[dict[str, Any]] = [
        {"name": "input_ids", "dtype": "int64", "shape": [1, 1]},
        {"name": "attention_mask", "dtype": "int64", "shape": [1, cache_capacity]},
        {"name": "position_ids", "dtype": "int64", "shape": [1, 1]},
        {"name": "valid_length", "dtype": "int64", "shape": [1]},
    ]
    for layer in range(LAYER_COUNT):
        for prefix in ("key_cache", "value_cache"):
            tensors.append(
                {
                    "name": f"{prefix}.{layer}",
                    "dtype": "float16",
                    "shape": [1, 8, cache_capacity, 128],
                }
            )
    return tensors


class PackagingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifact_root = self.root / "artifact-root"
        self.candidate_root = self.artifact_root / "onnx" / "qnn-candidate" / "T22"
        self.variant_directory = self.candidate_root / "S128"
        self.variant_directory.mkdir(parents=True)
        self.package_root = self.root / "package-root"
        self.request_directory = self.root / "private-requests"
        self.record_path = self.root / "records" / "S128.json"
        self.payloads: dict[str, bytes] = {}
        for kind in ("prefill", "decode"):
            graph = f"synthetic-{kind}-graph".encode()
            data = f"synthetic-{kind}-external-data".encode() * 8
            (self.variant_directory / f"{kind}.onnx").write_bytes(graph)
            (self.variant_directory / f"{kind}.onnx.data").write_bytes(data)
            self.payloads[f"{kind}.onnx"] = graph
            self.payloads[f"{kind}.onnx.data"] = data
        self.manifest_path = self.root / "S128.json"
        self.write_manifest(self.manifest())

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def manifest(self) -> dict[str, Any]:
        def graph_block(kind: str, inputs: list[dict[str, Any]]) -> dict[str, Any]:
            graph = self.payloads[f"{kind}.onnx"]
            data = self.payloads[f"{kind}.onnx.data"]
            return {
                "graph_kind": kind,
                "relative_path": f"S128/{kind}.onnx",
                "sha256": _sha256(graph),
                "size_bytes": len(graph),
                "external_data": [
                    {
                        "location": f"{kind}.onnx.data",
                        "sha256": _sha256(data),
                        "size_bytes": len(data),
                    }
                ],
                "input_tensors": inputs,
                "output_tensors": [],
            }

        return {
            "schema_version": 1,
            "task_id": "T22",
            "stage": "qnn_candidate",
            "variant_id": "S128",
            "context_length": 128,
            "cache_capacity": CACHE_CAPACITY,
            "opset": 18,
            "precision": "float16",
            "artifacts": {
                "root": "${SLM_LAB_ARTIFACT_ROOT}/onnx/qnn-candidate/T22",
                "prefill": graph_block("prefill", _prefill_inputs(128)),
                "decode": graph_block("decode", _decode_inputs(CACHE_CAPACITY)),
            },
            "transformations": [],
            "structural_delta": {},
            "verification": {"onnx_checker": "not_measured"},
        }

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def build(self, **overrides: Any) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "target_path": TARGET_CONFIG,
            "artifact_root": self.artifact_root,
            "package_root": self.package_root,
            "record_path": self.record_path,
            "request_directory": self.request_directory,
        }
        arguments.update(overrides)
        return packaging.build_package(self.manifest_path, **arguments)

    def check(self, **overrides: Any) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "target_path": TARGET_CONFIG,
            "artifact_root": self.artifact_root,
            "package_root": self.package_root,
            "record_path": self.record_path,
            "request_directory": self.request_directory,
        }
        arguments.update(overrides)
        return packaging.check_package(self.manifest_path, **arguments)

    def record(self) -> dict[str, Any]:
        return json.loads(self.record_path.read_text(encoding="utf-8"))

    def request_path(self, kind: str) -> Path:
        return self.request_directory / "S128" / f"{kind}-compile-request.json"


class PackageAssemblyTests(PackagingTestCase):
    def test_package_contains_graph_sidecar_and_checksums(self) -> None:
        summary = self.build()

        self.assertEqual(summary["status"], "ok")
        self.assertFalse(summary["job_submitted"])
        self.assertFalse(summary["service_contacted"])
        self.assertFalse(summary["package_layout_verified_against_service"])
        for kind in ("prefill", "decode"):
            directory = self.package_root / "S128" / kind
            names = sorted(item.name for item in directory.iterdir())
            self.assertEqual(
                names, sorted([f"{kind}.onnx", f"{kind}.onnx.data", "SHA256SUMS"])
            )
            self.assertEqual(
                (directory / f"{kind}.onnx").read_bytes(),
                self.payloads[f"{kind}.onnx"],
            )
            checksums = (directory / "SHA256SUMS").read_text(encoding="utf-8")
            expected = "".join(
                f"{_sha256(self.payloads[name])}  {name}\n"
                for name in sorted([f"{kind}.onnx", f"{kind}.onnx.data"])
            )
            self.assertEqual(checksums, expected)

    def test_sidecar_stays_beside_the_graph_under_its_recorded_location(self) -> None:
        self.build()
        record = self.record()
        prefill = record["package"]["graphs"][0]
        self.assertEqual(prefill["package_relative_path"], "S128/prefill")
        self.assertEqual(
            [item["logical_name"] for item in prefill["files"]],
            ["prefill.onnx", "prefill.onnx.data", "SHA256SUMS"],
        )
        self.assertEqual(
            prefill["compile_request"]["external_data_members"],
            ["prefill.onnx.data"],
        )

    def test_hardlink_is_preferred_and_shares_an_inode(self) -> None:
        summary = self.build()
        self.assertEqual(summary["graphs"][0]["link_modes"], ["hardlink"])
        source = self.variant_directory / "prefill.onnx"
        placed = self.package_root / "S128" / "prefill" / "prefill.onnx"
        self.assertEqual(source.stat().st_ino, placed.stat().st_ino)
        observation = self.record()["build_observation"]["graphs"][0]["placements"]
        self.assertEqual(
            {item["link_mode"] for item in observation},
            {"hardlink"},
        )
        self.assertEqual(
            {item["digest_evidence"] for item in observation},
            {"same_inode_as_verified_source"},
        )

    def test_copy_fallback_is_used_and_rehashed_when_linking_fails(self) -> None:
        original = os.link

        def refuse(*args: Any, **kwargs: Any) -> None:
            raise OSError("cross-device link not permitted")

        os.link = refuse
        try:
            summary = self.build()
        finally:
            os.link = original

        self.assertEqual(summary["graphs"][0]["link_modes"], ["copy"])
        placed = self.package_root / "S128" / "prefill" / "prefill.onnx"
        source = self.variant_directory / "prefill.onnx"
        self.assertEqual(placed.read_bytes(), source.read_bytes())
        self.assertNotEqual(placed.stat().st_ino, source.stat().st_ino)
        observation = self.record()["build_observation"]["graphs"][0]["placements"]
        self.assertEqual(
            {item["digest_evidence"] for item in observation},
            {"rehashed_after_placement"},
        )

    def test_rebuilding_is_idempotent_over_an_existing_package(self) -> None:
        first = self.build()
        second = self.build()
        self.assertEqual(
            [item["request_id"] for item in first["graphs"]],
            [item["request_id"] for item in second["graphs"]],
        )


class RecordPrivacyTests(PackagingTestCase):
    def test_record_carries_no_filesystem_path(self) -> None:
        self.build()
        text = self.record_path.read_text(encoding="utf-8")
        for fragment in (
            str(self.root),
            str(self.artifact_root),
            str(self.package_root),
            str(self.request_directory),
            str(Path.home()),
        ):
            self.assertNotIn(fragment, text)
        self.assertNotIn("/Users/", text)
        self.assertNotIn("/Volumes/", text)

    def test_record_states_that_no_job_was_submitted(self) -> None:
        self.build()
        record = self.record()
        status = record["submission_status"]
        self.assertFalse(status["job_submitted"])
        self.assertFalse(status["service_contacted"])
        self.assertFalse(status["package_layout_verified_against_service"])
        self.assertEqual(status["first_submission_owner"], "T31")
        self.assertIn("has not been verified", status["caveat"])
        self.assertIn("It does not mean AI Hub accepted it.", status["caveat"])
        self.assertIn(
            "qualcomm_ai_hub_accepted_or_would_accept_this_request",
            record["claim_boundary"]["does_not_establish"],
        )
        self.assertNotIn(
            "compiler_acceptance_of_the_candidate_graph",
            record["claim_boundary"]["establishes"],
        )
        text = json.dumps(record)
        for forbidden in ("latency", "throughput_tokens_per_second"):
            self.assertNotIn(f'"{forbidden}"', text)

    def test_record_pins_only_committed_device_and_runtime_identity(self) -> None:
        self.build()
        record = self.record()
        self.assertEqual(record["device"]["name"], "Snapdragon X Elite CRD")
        self.assertEqual(record["device"]["os"], "Windows 11")
        self.assertEqual(record["device"]["attributes"], [])
        self.assertEqual(record["runtime"], {"name": "QAIRT", "version": QAIRT_VERSION})
        self.assertEqual(
            record["compile"]["options"],
            f"--target_runtime qnn_context_binary --qairt_version {QAIRT_VERSION}",
        )
        self.assertEqual(record["client"], {"name": "qai-hub", "version": "0.53.0"})

    def test_request_is_written_to_private_storage_only(self) -> None:
        self.build()
        for kind in ("prefill", "decode"):
            path = self.request_path(kind)
            self.assertTrue(path.is_file())
            text = path.read_text(encoding="utf-8")
            self.assertIn(str(self.package_root), text)
            self.assertFalse(path.resolve().is_relative_to(REPO_ROOT))

    def test_request_directory_inside_the_public_tree_is_refused(self) -> None:
        with self.assertRaises(AiHubAdapterError) as context:
            self.build(request_directory=REPO_ROOT / "results" / "unsafe")
        self.assertIn("private storage", str(context.exception))
        self.assertFalse((REPO_ROOT / "results" / "unsafe").exists())


class CompileRequestTests(PackagingTestCase):
    def test_decode_specs_cover_every_cache_tensor(self) -> None:
        self.build()
        decode = self.record()["package"]["graphs"][1]
        specs = decode["compile_request"]["input_specs"]
        self.assertEqual(len(specs), 60)
        self.assertEqual(len([name for name in specs if name.endswith("_cache.0")]), 2)
        self.assertEqual(specs["key_cache.0"]["dtype"], "float16")
        self.assertEqual(specs["key_cache.0"]["shape"], [1, 8, CACHE_CAPACITY, 128])
        self.assertEqual(specs["input_ids"]["dtype"], "int64")
        self.assertEqual(specs["valid_length"]["shape"], [1])

    def test_generated_request_is_accepted_by_the_committed_stage_runner(self) -> None:
        self.build()
        record = self.record()
        options = record["compile"]["options"]
        for index, kind in enumerate(("prefill", "decode")):
            request = json.loads(self.request_path(kind).read_text(encoding="utf-8"))
            backend = MockBackend(options)
            manifest = run_compile(request, backend=backend)
            self.assertEqual(backend.calls, ["compile"])
            self.assertEqual(
                manifest["request_id"],
                record["package"]["graphs"][index]["compile_request"]["request_id"],
            )

    def test_request_id_is_independent_of_the_local_artifact_root(self) -> None:
        first = self.build()
        moved_root = self.root / "second-artifact-root"
        (moved_root / "onnx" / "qnn-candidate" / "T22").mkdir(parents=True)
        os.rename(
            self.candidate_root / "S128",
            moved_root / "onnx" / "qnn-candidate" / "T22" / "S128",
        )
        second = packaging.build_package(
            self.manifest_path,
            target_path=TARGET_CONFIG,
            artifact_root=moved_root,
            package_root=self.root / "second-package-root",
            record_path=self.root / "records" / "second.json",
            request_directory=self.root / "second-requests",
        )
        self.assertEqual(
            [item["request_id"] for item in first["graphs"]],
            [item["request_id"] for item in second["graphs"]],
        )

    def test_request_names_only_the_graph_file_and_says_so(self) -> None:
        self.build()
        request = json.loads(self.request_path("prefill").read_text(encoding="utf-8"))
        self.assertTrue(request["source_artifact"]["path"].endswith("prefill.onnx"))
        self.assertEqual(request["retry"], False)
        caveat = self.record()["package"]["graphs"][0]["compile_request"][
            "single_file_source_caveat"
        ]
        self.assertIn("unverified", caveat)


class AdversarialManifestTests(PackagingTestCase):
    def _expect_failure(self, fragment: str, manifest: dict[str, Any]) -> None:
        self.write_manifest(manifest)
        with self.assertRaises(AiHubAdapterError) as context:
            self.build()
        self.assertIn(fragment, str(context.exception))
        self.assertFalse(self.record_path.exists())

    def test_graph_digest_mismatch_fails_before_any_placement(self) -> None:
        manifest = self.manifest()
        manifest["artifacts"]["prefill"]["sha256"] = "0" * 64
        self._expect_failure("digest mismatch", manifest)
        self.assertEqual(list(self.package_root.rglob("*.onnx")), [])

    def test_sidecar_digest_mismatch_fails(self) -> None:
        manifest = self.manifest()
        manifest["artifacts"]["prefill"]["external_data"][0]["sha256"] = "1" * 64
        self._expect_failure("digest mismatch", manifest)

    def test_sidecar_size_mismatch_fails(self) -> None:
        manifest = self.manifest()
        manifest["artifacts"]["decode"]["external_data"][0]["size_bytes"] = 7
        self._expect_failure("wrong size", manifest)

    def test_missing_sidecar_file_fails(self) -> None:
        (self.variant_directory / "prefill.onnx.data").unlink()
        with self.assertRaises(AiHubAdapterError) as context:
            self.build()
        self.assertIn("is missing", str(context.exception))

    def test_nested_external_data_location_is_refused(self) -> None:
        manifest = self.manifest()
        manifest["artifacts"]["prefill"]["external_data"][0]["location"] = (
            "weights/prefill.onnx.data"
        )
        self._expect_failure("plain file name", manifest)

    def test_parent_escaping_relative_path_is_refused(self) -> None:
        manifest = self.manifest()
        manifest["artifacts"]["prefill"]["relative_path"] = "../escape/prefill.onnx"
        self._expect_failure("must stay under", manifest)

    def test_sidecar_colliding_with_the_graph_name_is_refused(self) -> None:
        manifest = self.manifest()
        manifest["artifacts"]["prefill"]["external_data"][0]["location"] = (
            "prefill.onnx"
        )
        self._expect_failure("collides", manifest)

    def test_duplicate_external_data_location_is_refused(self) -> None:
        manifest = self.manifest()
        entry = dict(manifest["artifacts"]["prefill"]["external_data"][0])
        manifest["artifacts"]["prefill"]["external_data"].append(entry)
        self._expect_failure("duplicate external data", manifest)

    def test_unsupported_input_dtype_is_refused(self) -> None:
        manifest = self.manifest()
        manifest["artifacts"]["decode"]["input_tensors"][4]["dtype"] = "bfloat16"
        self._expect_failure("unsupported by the compile adapter", manifest)

    def test_nonpositive_input_dimension_is_refused(self) -> None:
        manifest = self.manifest()
        manifest["artifacts"]["prefill"]["input_tensors"][0]["shape"] = [1, 0]
        self._expect_failure("positive integer", manifest)

    def test_duplicate_input_tensor_name_is_refused(self) -> None:
        manifest = self.manifest()
        tensors = manifest["artifacts"]["prefill"]["input_tensors"]
        tensors.append(dict(tensors[0]))
        self._expect_failure("duplicate input tensor", manifest)

    def test_wrong_stage_or_task_is_refused(self) -> None:
        manifest = self.manifest()
        manifest["stage"] = "reference"
        self._expect_failure("stage must be qnn_candidate", manifest)

        manifest = self.manifest()
        manifest["task_id"] = "T20"
        self._expect_failure("task_id must be T22", manifest)

        manifest = self.manifest()
        manifest["schema_version"] = 2
        self._expect_failure("schema_version must be 1", manifest)

    def test_missing_graph_kind_is_refused(self) -> None:
        manifest = self.manifest()
        del manifest["artifacts"]["decode"]
        self._expect_failure("no decode graph to package", manifest)

    def test_a_single_requested_graph_kind_still_builds(self) -> None:
        manifest = self.manifest()
        del manifest["artifacts"]["decode"]
        self.write_manifest(manifest)
        summary = self.build(graph_kinds=("prefill",))
        self.assertEqual(
            [item["graph_kind"] for item in summary["graphs"]], ["prefill"]
        )


class AdversarialTargetConfigTests(PackagingTestCase):
    def _target(self, **overrides: Any) -> Path:
        config = json.loads(TARGET_CONFIG.read_text(encoding="utf-8"))
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
        path = self.root / "target.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_non_qairt_runtime_is_refused(self) -> None:
        path = self._target(runtime={"name": "QNN", "version": QAIRT_VERSION})
        with self.assertRaisesRegex(AiHubAdapterError, "must be exactly QAIRT"):
            self.build(target_path=path)

    def test_retry_true_is_refused_by_the_config_reader(self) -> None:
        path = self._target(compile={"retry": True})
        with self.assertRaisesRegex(AiHubAdapterError, "retry must be false"):
            self.build(target_path=path)

    def test_option_outside_the_compile_allowlist_is_refused_before_assembly(
        self,
    ) -> None:
        path = self._target(
            compile={"options": f"--qairt_framework {QAIRT_VERSION}"},
        )
        with self.assertRaisesRegex(AiHubAdapterError, "unsupported for the compile"):
            self.build(target_path=path)
        self.assertFalse(self.package_root.exists())

    def test_option_runtime_version_must_match_the_runtime_block(self) -> None:
        path = self._target(
            compile={
                "options": (
                    "--target_runtime qnn_context_binary "
                    "--qairt_version 2.48.0.260626120635"
                )
            },
        )
        with self.assertRaisesRegex(AiHubAdapterError, "exactly the requested QAIRT"):
            self.build(target_path=path)

    def test_credential_like_option_is_refused_without_echoing_it(self) -> None:
        path = self._target(
            compile={
                "options": (
                    "--target_runtime qnn_context_binary "
                    f"--qairt_version {QAIRT_VERSION} "
                    "--access-token synthetic-secret-value"
                )
            },
        )
        with self.assertRaises(AiHubAdapterError) as context:
            self.build(target_path=path)
        self.assertNotIn("synthetic-secret-value", str(context.exception))

    def test_unknown_target_runtime_is_refused(self) -> None:
        path = self._target(
            compile={
                "options": (
                    "--target_runtime synthetic_runtime "
                    f"--qairt_version {QAIRT_VERSION}"
                )
            },
        )
        with self.assertRaisesRegex(AiHubAdapterError, "unsupported target runtime"):
            self.build(target_path=path)


class CheckModeTests(PackagingTestCase):
    def test_check_passes_immediately_after_a_build(self) -> None:
        self.build()
        summary = self.check()
        self.assertEqual(summary["mode"], "check")
        self.assertEqual(summary["status"], "ok")
        self.assertFalse(summary["job_submitted"])

    def test_check_detects_a_corrupted_package_member(self) -> None:
        self.build()
        placed = self.package_root / "S128" / "prefill" / "prefill.onnx.data"
        placed.unlink()
        placed.write_bytes(b"tampered")
        with self.assertRaises(AiHubAdapterError) as context:
            self.check()
        self.assertIn("prefill.onnx.data", str(context.exception))

    def test_check_detects_a_deleted_checksum_file(self) -> None:
        self.build()
        (self.package_root / "S128" / "decode" / "SHA256SUMS").unlink()
        with self.assertRaisesRegex(AiHubAdapterError, "SHA256SUMS is missing"):
            self.check()

    def test_check_detects_a_record_that_no_longer_matches_the_manifest(self) -> None:
        self.build()
        record = self.record()
        record["package"]["graphs"][0]["files"][0]["sha256"] = "2" * 64
        self.record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        with self.assertRaises(AiHubAdapterError) as context:
            self.check()
        message = str(context.exception)
        self.assertIn("no longer matches", message)
        self.assertIn("sha256", message)
        self.assertNotIn("2" * 64, message)

    def test_check_detects_a_tampered_request_id(self) -> None:
        self.build()
        record = self.record()
        request = record["package"]["graphs"][1]["compile_request"]
        request["request_id"] = "t30-compile-000000000000000000000"
        self.record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        with self.assertRaisesRegex(AiHubAdapterError, "request_id"):
            self.check()

    def test_check_does_not_rebuild_a_missing_package(self) -> None:
        self.build()
        (self.package_root / "S128" / "prefill" / "prefill.onnx").unlink()
        with self.assertRaisesRegex(AiHubAdapterError, "prefill.onnx is missing"):
            self.check()
        self.assertFalse(
            (self.package_root / "S128" / "prefill" / "prefill.onnx").exists()
        )

    def test_check_requires_a_committed_record(self) -> None:
        with self.assertRaisesRegex(AiHubAdapterError, "package record"):
            self.check()


class CommandLineTests(PackagingTestCase):
    def _argv(self, *extra: str) -> list[str]:
        return [
            "--manifest",
            str(self.manifest_path),
            "--target",
            str(TARGET_CONFIG),
            "--artifact-root",
            str(self.artifact_root),
            "--package-root",
            str(self.package_root),
            "--record",
            str(self.record_path),
            "--request-dir",
            str(self.request_directory),
            *extra,
        ]

    def test_cli_builds_then_checks_and_prints_a_path_free_summary(self) -> None:
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            self.assertEqual(packaging.main(self._argv()), 0)
            self.assertEqual(packaging.main(self._argv("--check")), 0)
        output = captured.getvalue()
        self.assertNotIn(str(self.root), output)
        summaries = [json.loads(line) for line in output.strip().splitlines()]
        self.assertEqual([item["mode"] for item in summaries], ["build", "check"])
        self.assertTrue(all(item["status"] == "ok" for item in summaries))
        self.assertTrue(all(not item["job_submitted"] for item in summaries))

    def test_cli_failure_is_sanitized_and_returns_one(self) -> None:
        manifest = self.manifest()
        manifest["artifacts"]["prefill"]["sha256"] = "3" * 64
        self.write_manifest(manifest)
        captured_out = io.StringIO()
        captured_err = io.StringIO()
        with (
            contextlib.redirect_stdout(captured_out),
            contextlib.redirect_stderr(captured_err),
        ):
            exit_code = packaging.main(self._argv())
        self.assertEqual(exit_code, 1)
        self.assertEqual(captured_out.getvalue(), "")
        self.assertIn("qnn packaging failed", captured_err.getvalue())
        self.assertNotIn(str(self.root), captured_err.getvalue())

    def test_cli_rejects_unknown_arguments_without_echoing_them(self) -> None:
        captured_err = io.StringIO()
        with contextlib.redirect_stderr(captured_err):
            with self.assertRaises(SystemExit) as context:
                packaging.main(["--api_token=synthetic-secret-value"])
        self.assertEqual(context.exception.code, 2)
        self.assertNotIn("synthetic-secret-value", captured_err.getvalue())


class CommittedTargetConfigTests(unittest.TestCase):
    def test_committed_target_matches_the_recorded_workbench_evidence(self) -> None:
        config = packaging.load_target_config(TARGET_CONFIG)
        self.assertEqual(config["device"]["name"], "Snapdragon X Elite CRD")
        self.assertEqual(config["device"]["os"], "Windows 11")
        self.assertEqual(config["device"]["attributes"], [])
        self.assertEqual(config["runtime"]["version"], QAIRT_VERSION)
        self.assertEqual(config["client"]["version"], "0.53.0")
        self.assertEqual(
            config["compile"]["options"],
            f"--target_runtime qnn_context_binary --qairt_version {QAIRT_VERSION}",
        )
        evidence = REPO_ROOT / config["evidence"]["device_and_runtime"]
        text = evidence.read_text(encoding="utf-8")
        self.assertIn("Snapdragon X Elite CRD, Windows 11", text)
        self.assertIn(QAIRT_VERSION, text)
        self.assertIn("qai-hub==0.53.0", text)


if __name__ == "__main__":
    unittest.main()
