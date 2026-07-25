"""Mocked regression tests for safe Qualcomm AI Hub stage adapters."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_lab.deployment.qualcomm.ai_hub import (
    AiHubAdapterError,
    normalize_profile,
    run_compile,
    run_inference,
    run_profile,
    sha256_file,
    stage_main,
    write_manifest,
)


CLIENT_VERSION = "0.53.0"
QAIRT_VERSION = "2.45.0.260326154327"
SUCCESSOR_QAIRT_VERSION = "2.47.0.260601114230"
COMPILE_OPTIONS = f"--target_runtime qnn_context_binary --qairt_version {QAIRT_VERSION}"
INFERENCE_OPTIONS = f"--qairt_framework {QAIRT_VERSION}"
PROFILE_OPTIONS = f"--qairt_framework {QAIRT_VERSION}"
STAGE_OPTIONS = {
    "compile": COMPILE_OPTIONS,
    "inference": INFERENCE_OPTIONS,
    "profile": PROFILE_OPTIONS,
}
PRIVATE_MARKERS = (
    "https://private.invalid/jobs/jsynthetic000",
    "api_token=synthetic-secret-not-a-credential",
    "learner@example.invalid",
    "jsynthetic000",
)


@dataclass
class MockStatus:
    success: bool = True
    code: str = "SUCCESS"


@dataclass
class MockTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    scale: float | None = None
    zero_point: int | None = None


@dataclass
class MockDevice:
    name: str = "Snapdragon X Elite CRD"
    os: str = "Windows 11"
    attributes: list[str] | None = None

    def __post_init__(self) -> None:
        if self.attributes is None:
            self.attributes = [
                "chipset:qualcomm-snapdragon-x-elite",
                "hexagon:v73",
            ]


class MockTarget:
    metadata = {"QAIRT_SDK_VERSION": QAIRT_VERSION}
    input_spec = {
        None: [
            MockTensor("input_ids", (1, 128), "int32"),
            MockTensor("attention_mask", (1, 128), "int32"),
        ]
    }
    output_spec = {None: [MockTensor("logits", (1, 128, 151936), "float16")]}


class LazyTarget:
    metadata = {"QAIRT_SDK_VERSION": QAIRT_VERSION}

    def __init__(self, failing_property: str) -> None:
        self.failing_property = failing_property

    @property
    def input_spec(self) -> Any:
        print(PRIVATE_MARKERS[0])
        if self.failing_property == "input_spec":
            raise RuntimeError(PRIVATE_MARKERS[1])
        return MockTarget.input_spec

    @property
    def output_spec(self) -> Any:
        print(PRIVATE_MARKERS[2])
        if self.failing_property == "output_spec":
            raise RuntimeError(PRIVATE_MARKERS[3])
        return MockTarget.output_spec


class LazyMetadataTarget:
    input_spec = MockTarget.input_spec
    output_spec = MockTarget.output_spec

    @property
    def metadata(self) -> Any:
        print(PRIVATE_MARKERS[0])
        raise RuntimeError(PRIVATE_MARKERS[1])


class MockCompileJob:
    def __init__(
        self,
        payload: bytes = b"compiled-qnn-context",
        *,
        target: Any | None = None,
        device: MockDevice | None = None,
        options: str = COMPILE_OPTIONS,
    ) -> None:
        self.payload = payload
        self.target = target if target is not None else MockTarget()
        self.device = device if device is not None else MockDevice()
        self.options = options
        self.wait_timeout: int | None = None

    def wait(self, timeout: int | None = None) -> MockStatus:
        self.wait_timeout = timeout
        print(PRIVATE_MARKERS[0])
        return MockStatus()

    def get_target_model(self) -> MockTarget:
        print(PRIVATE_MARKERS[1])
        return self.target

    def download_target_model(self, filename: str) -> str:
        print(PRIVATE_MARKERS[2])
        Path(filename).write_bytes(self.payload)
        return filename


class MockInferenceJob:
    def __init__(
        self,
        payload: bytes = b"mock-hdf5-output",
        *,
        device: MockDevice | None = None,
        model: Any | None = None,
        options: str = INFERENCE_OPTIONS,
    ) -> None:
        self.payload = payload
        self.device = device if device is not None else MockDevice()
        self.model = model if model is not None else MockTarget()
        self.options = options

    def wait(self, timeout: int | None = None) -> MockStatus:
        print(PRIVATE_MARKERS[0])
        return MockStatus()

    def download_output_data(self, filename: str) -> str:
        print(PRIVATE_MARKERS[1])
        Path(filename).write_bytes(self.payload)
        return filename


class MockProfileJob:
    def __init__(
        self,
        profile: dict[str, Any],
        *,
        device: MockDevice | None = None,
        model: Any | None = None,
        options: str = PROFILE_OPTIONS,
    ) -> None:
        self.profile = profile
        self.device = device if device is not None else MockDevice()
        self.model = model if model is not None else MockTarget()
        self.options = options

    def wait(self, timeout: int | None = None) -> MockStatus:
        print(PRIVATE_MARKERS[0])
        return MockStatus()

    def download_profile(self) -> dict[str, Any]:
        print(PRIVATE_MARKERS[1])
        return self.profile


class MockBackend:
    def __init__(
        self,
        *,
        compile_job: Any | None = None,
        inference_job: Any | None = None,
        profile_job: Any | None = None,
        version: str = CLIENT_VERSION,
    ) -> None:
        self._version = version
        self.compile_job = compile_job
        self.inference_job = inference_job
        self.profile_job = profile_job
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def client_version(self) -> str:
        print(PRIVATE_MARKERS[3])
        return self._version

    def submit_compile(self, **kwargs: Any) -> Any:
        print(PRIVATE_MARKERS[0])
        self.calls.append(("compile", kwargs))
        if isinstance(self.compile_job, Exception):
            raise self.compile_job
        return self.compile_job

    def submit_inference(self, **kwargs: Any) -> Any:
        print(PRIVATE_MARKERS[0])
        self.calls.append(("inference", kwargs))
        if isinstance(self.inference_job, Exception):
            raise self.inference_job
        return self.inference_job

    def submit_profile(self, **kwargs: Any) -> Any:
        print(PRIVATE_MARKERS[0])
        self.calls.append(("profile", kwargs))
        if isinstance(self.profile_job, Exception):
            raise self.profile_job
        return self.profile_job


def artifact(path: Path, logical_name: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "logical_name": logical_name,
        "sha256": sha256_file(path),
    }


def common(stage: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "stage": stage,
        "client_version": CLIENT_VERSION,
        "device": {
            "name": "Snapdragon X Elite (Family)",
            "os": "Windows 11",
            "attributes": ["chipset:qualcomm-snapdragon-x-elite", "hexagon:v73"],
        },
        "runtime": {"name": "QAIRT", "version": QAIRT_VERSION},
        "options": STAGE_OPTIONS[stage],
        "job_name": f"slm-lab-t30-{stage}",
        "timeout_seconds": 3600,
        "retry": False,
    }


def profile_payload() -> dict[str, Any]:
    return {
        "execution_summary": {
            "estimated_inference_time": 127,
            "estimated_inference_peak_memory": 14_450_688,
            "first_load_time": 311_671,
            "first_load_peak_memory": 14_786_560,
            "warm_load_time": 198_597,
            "warm_load_peak_memory": 14_573_568,
            "all_inference_times": [127, 129, 1087],
        },
        "execution_details": [{"compute_unit": "NPU", "warning": PRIVATE_MARKERS[1]}],
    }


class AiHubAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "prefill-128.onnx"
        self.source.write_bytes(b"deterministic-source-onnx")
        self.compiled = self.root / "prefill-128.serialized.bin"
        self.compile_manifest_path = self.root / "compile-manifest.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def compile_request(self) -> dict[str, Any]:
        return {
            **common("compile"),
            "source_artifact": artifact(self.source, "prefill-128.onnx"),
            "output_artifact": str(self.compiled),
            "output_logical_name": "prefill-128.serialized.bin",
            "input_specs": {
                "input_ids": {"shape": [1, 128], "dtype": "int32"},
                "attention_mask": {"shape": [1, 128], "dtype": "int32"},
            },
        }

    def run_compile(self) -> tuple[dict[str, Any], MockBackend]:
        backend = MockBackend(compile_job=MockCompileJob())
        manifest = run_compile(self.compile_request(), backend=backend)
        write_manifest(self.compile_manifest_path, manifest)
        return manifest, backend

    def inference_request(self) -> dict[str, Any]:
        dataset = self.root / "inputs.h5"
        dataset.write_bytes(b"mock-hdf5-input")
        return {
            **common("inference"),
            "predecessor_manifest": str(self.compile_manifest_path),
            "compiled_artifact": artifact(self.compiled, "prefill-128.serialized.bin"),
            "input_dataset": artifact(dataset, "prefill-128-inputs.h5"),
            "output_artifact": str(self.root / "outputs.h5"),
            "output_logical_name": "prefill-128-outputs.h5",
        }

    def profile_request(self) -> dict[str, Any]:
        return {
            **common("profile"),
            "predecessor_manifest": str(self.compile_manifest_path),
            "compiled_artifact": artifact(self.compiled, "prefill-128.serialized.bin"),
            "raw_profile_output": str(self.root / "profile-private.json"),
            "raw_profile_logical_name": "prefill-128-profile-private.json",
        }

    def test_compile_manifest_is_sanitized_and_content_addressed(self) -> None:
        captured_out = io.StringIO()
        captured_err = io.StringIO()
        backend = MockBackend(compile_job=MockCompileJob())
        with (
            contextlib.redirect_stdout(captured_out),
            contextlib.redirect_stderr(captured_err),
        ):
            manifest = run_compile(self.compile_request(), backend=backend)

        self.assertEqual(captured_out.getvalue(), "")
        self.assertEqual(captured_err.getvalue(), "")
        self.assertEqual(manifest["stage"], "compile")
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["status"], "success")
        self.assertEqual(manifest["client"]["version"], CLIENT_VERSION)
        self.assertEqual(
            manifest["target"]["requested"]["name"],
            "Snapdragon X Elite (Family)",
        )
        self.assertEqual(
            manifest["target"]["observed"]["name"], "Snapdragon X Elite CRD"
        )
        self.assertFalse(manifest["target"]["exact_request_observation_match_required"])
        self.assertEqual(manifest["runtime"]["requested"]["version"], QAIRT_VERSION)
        self.assertEqual(
            manifest["runtime"]["submitted"],
            {
                "name": "QAIRT",
                "version": QAIRT_VERSION,
                "option": "--qairt_version",
            },
        )
        self.assertEqual(manifest["runtime"]["artifact"]["version"], QAIRT_VERSION)
        self.assertIsNone(manifest["runtime"]["observed_execution"])
        self.assertFalse(manifest["submission"]["service_turnaround_is_device_latency"])
        self.assertEqual(
            manifest["lineage"]["source_artifacts"][0]["sha256"],
            hashlib.sha256(self.source.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            manifest["result"]["target_artifact"]["sha256"],
            hashlib.sha256(b"compiled-qnn-context").hexdigest(),
        )
        self.assertEqual(
            manifest["graph_contract"]["target_io"]["outputs"][0]["name"],
            "logits",
        )
        public_text = json.dumps(manifest, sort_keys=True)
        for marker in PRIVATE_MARKERS:
            self.assertNotIn(marker, public_text)
        self.assertNotIn(str(self.root), public_text)
        self.assertEqual(set(manifest["privacy"].values()), {False})
        self.assertEqual(backend.calls[0][0], "compile")
        self.assertEqual(
            backend.calls[0][1]["input_specs"]["input_ids"],
            ((1, 128), "int32"),
        )
        self.assertEqual(
            backend.calls[0][1]["device"],
            {
                "name": "Snapdragon X Elite (Family)",
                "os": "Windows 11",
                "attributes": [
                    "chipset:qualcomm-snapdragon-x-elite",
                    "hexagon:v73",
                ],
            },
        )

    def test_inference_restarts_from_compile_manifest_and_local_artifacts(self) -> None:
        compile_manifest, _ = self.run_compile()
        compile_manifest_sha = sha256_file(self.compile_manifest_path)
        backend = MockBackend(inference_job=MockInferenceJob())

        manifest = run_inference(self.inference_request(), backend=backend)

        self.assertEqual(manifest["stage"], "inference")
        self.assertEqual(
            manifest["lineage"]["predecessor_manifest_sha256"],
            compile_manifest_sha,
        )
        self.assertEqual(
            manifest["lineage"]["source_artifacts"][0]["sha256"],
            compile_manifest["result"]["target_artifact"]["sha256"],
        )
        self.assertTrue((self.root / "outputs.h5").is_file())
        call = backend.calls[0]
        self.assertEqual(call[0], "inference")
        self.assertEqual(call[1]["model"], self.compiled)
        self.assertEqual(call[1]["inputs"], self.root / "inputs.h5")

    def test_successor_device_reuse_records_current_service_device_independently(
        self,
    ) -> None:
        compile_manifest, _ = self.run_compile()
        request = self.inference_request()
        request["device"] = {
            "name": "Snapdragon X2 Elite (Family)",
            "os": "",
            "attributes": ["chipset:qualcomm-snapdragon-x2-elite"],
        }
        request["runtime"]["version"] = SUCCESSOR_QAIRT_VERSION
        request["options"] = f"--qairt_framework {SUCCESSOR_QAIRT_VERSION}"
        observed = MockDevice(
            name="Snapdragon X2 Elite CRD",
            os="Windows 12",
            attributes=["chipset:qualcomm-snapdragon-x2-elite", "hexagon:v79"],
        )

        manifest = run_inference(
            request,
            backend=MockBackend(
                inference_job=MockInferenceJob(
                    device=observed,
                    options=request["options"],
                ),
            ),
        )

        self.assertEqual(
            compile_manifest["target"]["observed"]["name"],
            "Snapdragon X Elite CRD",
        )
        self.assertEqual(
            manifest["target"]["requested"]["name"],
            "Snapdragon X2 Elite (Family)",
        )
        self.assertEqual(
            manifest["target"]["observed"],
            {
                "name": "Snapdragon X2 Elite CRD",
                "os": "Windows 12",
                "attributes": [
                    "chipset:qualcomm-snapdragon-x2-elite",
                    "hexagon:v79",
                ],
            },
        )
        self.assertFalse(manifest["target"]["exact_request_observation_match_required"])
        self.assertEqual(
            manifest["runtime"]["requested"]["version"],
            SUCCESSOR_QAIRT_VERSION,
        )
        self.assertEqual(
            manifest["runtime"]["submitted"]["option"],
            "--qairt_framework",
        )
        self.assertEqual(
            manifest["runtime"]["artifact"]["version"],
            QAIRT_VERSION,
        )

    def test_profile_restarts_independently_and_normalizes_documented_units(
        self,
    ) -> None:
        self.run_compile()
        backend = MockBackend(profile_job=MockProfileJob(profile_payload()))

        manifest = run_profile(self.profile_request(), backend=backend)

        normalized = manifest["result"]["normalized_profile"]
        self.assertEqual(normalized["latency"]["estimated_inference_time_us"], 127)
        self.assertEqual(normalized["latency"]["estimated_inference_time_ms"], 0.127)
        self.assertEqual(normalized["latency"]["inference_sample_count"], 3)
        self.assertEqual(
            normalized["latency"]["observed_inference_time_range_us"],
            [127, 1087],
        )
        self.assertEqual(
            normalized["memory"]["estimated_inference_peak_memory_bytes"],
            14_450_688,
        )
        self.assertEqual(normalized["placement"]["compute_units"], ["NPU"])
        self.assertEqual(normalized["warnings"]["count"], 1)
        self.assertFalse(normalized["warnings"]["raw_text_committed"])
        raw_path = self.root / "profile-private.json"
        self.assertEqual(
            manifest["result"]["raw_profile_artifact"]["sha256"],
            sha256_file(raw_path),
        )
        self.assertIn(PRIVATE_MARKERS[1], raw_path.read_text(encoding="utf-8"))
        self.assertNotIn(PRIVATE_MARKERS[1], json.dumps(manifest))

    def test_each_stage_uses_only_its_own_backend_method(self) -> None:
        self.run_compile()
        inference_backend = MockBackend(inference_job=MockInferenceJob())
        run_inference(self.inference_request(), backend=inference_backend)
        self.assertEqual([name for name, _ in inference_backend.calls], ["inference"])

        profile_backend = MockBackend(profile_job=MockProfileJob(profile_payload()))
        run_profile(self.profile_request(), backend=profile_backend)
        self.assertEqual([name for name, _ in profile_backend.calls], ["profile"])

    def test_service_failure_suppresses_exception_and_client_output(self) -> None:
        backend = MockBackend(
            compile_job=RuntimeError(
                f"submission rejected {PRIVATE_MARKERS[0]} {PRIVATE_MARKERS[1]}"
            )
        )
        captured_out = io.StringIO()
        captured_err = io.StringIO()
        with (
            contextlib.redirect_stdout(captured_out),
            contextlib.redirect_stderr(captured_err),
        ):
            with self.assertRaises(AiHubAdapterError) as context:
                run_compile(self.compile_request(), backend=backend)
        message = str(context.exception)
        self.assertEqual(captured_out.getvalue(), "")
        self.assertEqual(captured_err.getvalue(), "")
        for marker in PRIVATE_MARKERS:
            self.assertNotIn(marker, message)
        self.assertIn("private service details suppressed", message)

    def test_lazy_target_specs_are_quiet_and_fail_with_sanitized_errors(self) -> None:
        for property_name in ("input_spec", "output_spec"):
            with self.subTest(property_name=property_name):
                captured_out = io.StringIO()
                captured_err = io.StringIO()
                backend = MockBackend(
                    compile_job=MockCompileJob(
                        target=LazyTarget(property_name),
                    )
                )
                with (
                    contextlib.redirect_stdout(captured_out),
                    contextlib.redirect_stderr(captured_err),
                ):
                    with self.assertRaises(AiHubAdapterError) as context:
                        run_compile(self.compile_request(), backend=backend)
                self.assertEqual(captured_out.getvalue(), "")
                self.assertEqual(captured_err.getvalue(), "")
                self.assertIn(
                    "private service details suppressed", str(context.exception)
                )
                for marker in PRIVATE_MARKERS:
                    self.assertNotIn(marker, str(context.exception))

    def test_lazy_runtime_metadata_is_quiet_and_sanitized(self) -> None:
        captured_out = io.StringIO()
        captured_err = io.StringIO()
        backend = MockBackend(compile_job=MockCompileJob(target=LazyMetadataTarget()))
        with (
            contextlib.redirect_stdout(captured_out),
            contextlib.redirect_stderr(captured_err),
        ):
            with self.assertRaises(AiHubAdapterError) as context:
                run_compile(self.compile_request(), backend=backend)
        self.assertEqual(captured_out.getvalue(), "")
        self.assertEqual(captured_err.getvalue(), "")
        self.assertIn("private service details suppressed", str(context.exception))
        for marker in PRIVATE_MARKERS:
            self.assertNotIn(marker, str(context.exception))

    def test_private_request_text_is_rejected_without_echoing_it(self) -> None:
        request = self.compile_request()
        request["options"] += f" --api_token={PRIVATE_MARKERS[3]}"
        with self.assertRaises(AiHubAdapterError) as context:
            run_compile(request, backend=MockBackend(compile_job=MockCompileJob()))
        message = str(context.exception)
        self.assertIn("private or URL-like text", message)
        for marker in PRIVATE_MARKERS:
            self.assertNotIn(marker, message)

    def test_options_reject_whitespace_credentials_accounts_and_paths(self) -> None:
        unsafe_options = (
            f"{COMPILE_OPTIONS} --access-token synthetic-value",
            f"{COMPILE_OPTIONS} --access_token synthetic-value",
            f"{COMPILE_OPTIONS} --billing-account synthetic-account",
            f"{COMPILE_OPTIONS} --model /private/synthetic-model.onnx",
            f"{COMPILE_OPTIONS} --prefixed-api-token synthetic-value",
            f"{COMPILE_OPTIONS} --cache-dir /private/synthetic-cache",
            f"{COMPILE_OPTIONS} --config synthetic-private.ini",
        )
        for options in unsafe_options:
            with self.subTest(kind=options.rsplit(" ", 1)[-1]):
                request = self.compile_request()
                request["options"] = options
                with self.assertRaises(AiHubAdapterError) as context:
                    run_compile(
                        request,
                        backend=MockBackend(compile_job=MockCompileJob()),
                    )
                self.assertNotIn("synthetic", str(context.exception))
                self.assertFalse(self.compiled.exists())

    def test_adversarial_flags_fail_closed_for_every_stage_before_submission(
        self,
    ) -> None:
        self.run_compile()
        requests = {
            "compile": self.compile_request,
            "inference": self.inference_request,
            "profile": self.profile_request,
        }
        backend_jobs = {
            "compile": {"compile_job": MockCompileJob()},
            "inference": {"inference_job": MockInferenceJob()},
            "profile": {"profile_job": MockProfileJob(profile_payload())},
        }
        runners = {
            "compile": run_compile,
            "inference": run_inference,
            "profile": run_profile,
        }
        for stage, request_factory in requests.items():
            for unsafe_flag in (
                "--access-token",
                "--access_token",
                "--billing-account",
                "--model",
            ):
                with self.subTest(stage=stage, flag=unsafe_flag):
                    request = request_factory()
                    request["options"] += f" {unsafe_flag} synthetic-private-value"
                    backend = MockBackend(**backend_jobs[stage])
                    with self.assertRaises(AiHubAdapterError) as context:
                        runners[stage](request, backend=backend)
                    self.assertEqual(backend.calls, [])
                    self.assertNotIn(
                        "synthetic-private-value",
                        str(context.exception),
                    )

    def test_options_require_runtime_identity_in_actual_submission(self) -> None:
        request = self.compile_request()
        request["options"] = "--target_runtime qnn_context_binary"
        with self.assertRaisesRegex(AiHubAdapterError, "qairt_version"):
            run_compile(request, backend=MockBackend(compile_job=MockCompileJob()))

        job = MockCompileJob(
            options=(
                "--target_runtime qnn_context_binary "
                "--qairt_version 2.47.0.260601114230"
            )
        )
        captured_out = io.StringIO()
        with contextlib.redirect_stdout(captured_out):
            with self.assertRaises(AiHubAdapterError) as context:
                run_compile(
                    self.compile_request(),
                    backend=MockBackend(compile_job=job),
                )
        self.assertEqual(captured_out.getvalue(), "")
        self.assertIn("private service details suppressed", str(context.exception))

    def test_safe_nonpath_runtime_options_remain_supported(self) -> None:
        options = f"{COMPILE_OPTIONS} --qnn_options context_enable_graphs=prefill"
        request = self.compile_request()
        request["options"] = options
        manifest = run_compile(
            request,
            backend=MockBackend(compile_job=MockCompileJob(options=options)),
        )
        self.assertEqual(manifest["submission"]["options"], options)

    def test_runtime_option_is_stage_specific_and_name_is_bound(self) -> None:
        self.run_compile()
        profile = self.profile_request()
        profile["options"] = f"--qairt_version {QAIRT_VERSION}"
        with self.assertRaisesRegex(AiHubAdapterError, "unsupported for the profile"):
            run_profile(
                profile,
                backend=MockBackend(profile_job=MockProfileJob(profile_payload())),
            )

        compile_request = self.compile_request()
        compile_request["options"] = f"--qairt_framework {QAIRT_VERSION}"
        with self.assertRaisesRegex(AiHubAdapterError, "unsupported for the compile"):
            run_compile(
                compile_request,
                backend=MockBackend(compile_job=MockCompileJob()),
            )

        compile_request = self.compile_request()
        compile_request["runtime"]["name"] = "QNN"
        with self.assertRaisesRegex(AiHubAdapterError, "exactly QAIRT"):
            run_compile(
                compile_request,
                backend=MockBackend(compile_job=MockCompileJob()),
            )

        compile_request = self.compile_request()
        compile_request["options"] = (
            f"--target_runtime qnn_context_binary --qairt-version {QAIRT_VERSION}"
        )
        with self.assertRaisesRegex(AiHubAdapterError, "pinned SDK spelling"):
            run_compile(
                compile_request,
                backend=MockBackend(compile_job=MockCompileJob()),
            )

        schema_v1_request = self.compile_request()
        schema_v1_request["schema_version"] = 1
        with self.assertRaisesRegex(AiHubAdapterError, "wrong schema"):
            run_compile(
                schema_v1_request,
                backend=MockBackend(compile_job=MockCompileJob()),
            )

    def test_client_version_is_exact_and_must_match_backend(self) -> None:
        with self.assertRaisesRegex(AiHubAdapterError, "does not match"):
            run_compile(
                self.compile_request(),
                backend=MockBackend(
                    compile_job=MockCompileJob(),
                    version="0.54.0",
                ),
            )
        request = self.compile_request()
        request["client_version"] = ">=0.53"
        with self.assertRaisesRegex(AiHubAdapterError, "exact version"):
            run_compile(request, backend=MockBackend(compile_job=MockCompileJob()))

    def test_input_and_predecessor_digests_fail_closed(self) -> None:
        request = self.compile_request()
        request["source_artifact"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(AiHubAdapterError, "digest mismatch"):
            run_compile(request, backend=MockBackend(compile_job=MockCompileJob()))

        self.run_compile()
        inference = self.inference_request()
        inference["compiled_artifact"]["sha256"] = "1" * 64
        with self.assertRaisesRegex(AiHubAdapterError, "predecessor"):
            run_inference(
                inference,
                backend=MockBackend(inference_job=MockInferenceJob()),
            )

        predecessor = json.loads(self.compile_manifest_path.read_text(encoding="utf-8"))
        predecessor["schema_version"] = 1
        self.compile_manifest_path.write_text(
            json.dumps(predecessor),
            encoding="utf-8",
        )
        inference = self.inference_request()
        with self.assertRaisesRegex(AiHubAdapterError, "predecessor"):
            run_inference(
                inference,
                backend=MockBackend(inference_job=MockInferenceJob()),
            )

    def test_raw_and_binary_outputs_cannot_target_public_repo_paths(self) -> None:
        request = self.compile_request()
        request["output_artifact"] = str(
            Path(__file__).resolve().parents[3]
            / "results"
            / "raw"
            / "unsafe-compiled.bin"
        )
        with self.assertRaisesRegex(AiHubAdapterError, "ignored private storage"):
            run_compile(request, backend=MockBackend(compile_job=MockCompileJob()))

    def test_malformed_profile_fails_without_public_manifest(self) -> None:
        self.run_compile()
        malformed = {"estimated_inference_time": 127}
        with self.assertRaisesRegex(AiHubAdapterError, "peak_memory"):
            run_profile(
                self.profile_request(),
                backend=MockBackend(profile_job=MockProfileJob(malformed)),
            )

    def test_profile_normalizer_supports_flat_documented_shape(self) -> None:
        normalized = normalize_profile(
            {
                "estimated_inference_time": 2997,
                "estimated_inference_peak_memory": 69_177_344,
                "all_inference_times": [3000, 2997],
            }
        )
        self.assertEqual(normalized["latency"]["estimated_inference_time_us"], 2997)
        self.assertEqual(
            normalized["memory"]["estimated_inference_peak_memory_bytes"],
            69_177_344,
        )

    def test_cli_prints_only_sanitized_summary_and_manifest(self) -> None:
        request_path = self.root / "compile-request.json"
        request_path.write_text(json.dumps(self.compile_request()), encoding="utf-8")
        manifest_path = self.root / "public-compile.json"
        captured_out = io.StringIO()
        captured_err = io.StringIO()
        with (
            contextlib.redirect_stdout(captured_out),
            contextlib.redirect_stderr(captured_err),
        ):
            exit_code = stage_main(
                "compile",
                [
                    "--request",
                    str(request_path),
                    "--manifest",
                    str(manifest_path),
                ],
                backend=MockBackend(compile_job=MockCompileJob()),
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(captured_err.getvalue(), "")
        summary = json.loads(captured_out.getvalue())
        self.assertEqual(summary["stage"], "compile")
        self.assertEqual(summary["status"], "success")
        self.assertNotIn(str(self.root), captured_out.getvalue())
        public_text = manifest_path.read_text(encoding="utf-8")
        for marker in PRIVATE_MARKERS:
            self.assertNotIn(marker, captured_out.getvalue())
            self.assertNotIn(marker, public_text)

    def test_cli_argument_errors_do_not_echo_unknown_private_values(self) -> None:
        captured_err = io.StringIO()
        with contextlib.redirect_stderr(captured_err):
            with self.assertRaises(SystemExit) as context:
                stage_main("compile", [f"--api_token={PRIVATE_MARKERS[3]}"])
        self.assertEqual(context.exception.code, 2)
        self.assertIn("arguments are invalid", captured_err.getvalue())
        for marker in PRIVATE_MARKERS:
            self.assertNotIn(marker, captured_err.getvalue())


if __name__ == "__main__":
    unittest.main()
