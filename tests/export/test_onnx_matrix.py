from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import slm_lab.export.onnx_matrix as onnx_matrix
from slm_lab.contracts import (
    CONTEXT_VARIANTS,
    build_decode_contract,
    build_prefill_contract,
    validate_tensor_mapping,
)
from slm_lab.contracts.static_cache import NUM_LAYERS
from slm_lab.export.onnx_matrix import (
    DecodeWrapper,
    ExternalDataRecord,
    ExportConfigurationError,
    OnnxArtifactRecord,
    PrefillWrapper,
    build_example_inputs,
    export_onnx_graph,
    inspect_onnx_artifact,
    load_export_config,
    verify_manifest_evidence,
)
from slm_lab.manifests.validation import validate_manifest
from slm_lab.evaluation.fixtures import canonical_json_sha256


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/models/qwen3-0.6b-onnx-export.json"


class _EqualityAdapter:
    """Delegate fixture fields while remaining broadly equality-compatible."""

    def __init__(self, target) -> None:
        self._target = target

    def __getattr__(self, name):
        return getattr(self._target, name)

    def __eq__(self, other) -> bool:
        return True


def test_export_config_freezes_exact_matrix_and_toolchain() -> None:
    config = load_export_config(CONFIG)

    assert config.contexts == (128, 512, 1024, 4096)
    assert config.opset == 18
    assert config.precision == "float16"
    assert config.torch_version == "2.7.1"
    assert config.transformers_version == "4.51.3"
    assert config.onnx_version == "1.18.0"
    assert config.external_data_threshold_bytes == 1024
    assert config.evidence_attestation.exporter_commit == (
        "631fd70bcff9b73b81c08a2a2e0127cad07f09ca"
    )
    assert config.evidence_attestation.runtime_python_version == "3.11.15"
    assert config.token_fixture.source_path == (
        ROOT / "tests/fixtures/t10/token-fixtures-v1.json"
    )
    assert config.token_fixture.canonical_json_sha256 == (
        "9f9268ae4a366faa4325271492ec52f035bbf3ba0973d2de61f63382e6302745"
    )
    assert tuple(
        workload.context_length for workload in config.token_fixture.workloads
    ) == config.contexts


def test_export_config_rejects_caller_supplied_path(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    path = tmp_path / "copied.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExportConfigurationError, match="fixed tracked export config"):
        load_export_config(path)


def test_export_config_rejects_symlink_alias(tmp_path: Path) -> None:
    alias = tmp_path / CONFIG.name
    alias.symlink_to(CONFIG)

    with pytest.raises(ExportConfigurationError, match="fixed tracked export config"):
        load_export_config(alias)


def test_export_config_rejects_symlink_at_configured_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alias = tmp_path / CONFIG.name
    alias.symlink_to(CONFIG)
    monkeypatch.setattr(onnx_matrix, "DEFAULT_CONFIG_PATH", alias)

    with pytest.raises(ExportConfigurationError, match="must not be a symlink"):
        load_export_config(alias)


def test_export_config_rejects_configured_fixture_with_stale_token_hash(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (ROOT / "tests/fixtures/t10/token-fixtures-v1.json").read_text(
            encoding="utf-8"
        )
    )
    fixture["context_workloads"][0]["token_ids"][0] += 1
    fixture_path = tmp_path / "tampered-token-fixtures.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    with pytest.raises(ExportConfigurationError, match="S128.*hash drift"):
        onnx_matrix._load_token_fixture(fixture_path, str(fixture_path))


def test_export_config_rejects_coherent_frozen_token_content_drift(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (ROOT / "tests/fixtures/t10/token-fixtures-v1.json").read_text(
            encoding="utf-8"
        )
    )
    record = fixture["context_workloads"][0]
    record["token_ids"][0] += 1
    record["token_ids_sha256"] = canonical_json_sha256(record["token_ids"])
    fixture_path = tmp_path / "coherently-tampered-token-fixtures.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    with pytest.raises(
        ExportConfigurationError,
        match="configured T10 token fixture is invalid|frozen canonical",
    ):
        onnx_matrix._load_token_fixture(fixture_path, str(fixture_path))


def test_runtime_rejects_actual_python_version_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_export_config(CONFIG)
    monkeypatch.setattr(onnx_matrix.platform, "python_version", lambda: "9.9.9")

    with pytest.raises(ExportConfigurationError, match="Python version mismatch"):
        onnx_matrix._verify_runtime(config)


def test_example_inputs_use_exact_fixture_and_contract() -> None:
    torch = pytest.importorskip("torch")
    fixture = json.loads(
        (ROOT / "tests/fixtures/t10/token-fixtures-v1.json").read_text(
            encoding="utf-8"
        )
    )
    expected = fixture["context_workloads"][0]["token_ids"]

    prefill = build_prefill_contract(128)
    config = load_export_config(CONFIG)
    prefill_inputs = build_example_inputs(
        prefill,
        token_fixture=config.token_fixture,
    )
    validate_tensor_mapping(
        dict(zip((tensor.name for tensor in prefill.inputs), prefill_inputs)),
        prefill.inputs,
    )
    assert prefill_inputs[0].tolist() == [expected]

    decode = build_decode_contract(128)
    decode_inputs = build_example_inputs(
        decode,
        token_fixture=config.token_fixture,
    )
    validate_tensor_mapping(
        dict(zip((tensor.name for tensor in decode.inputs), decode_inputs)),
        decode.inputs,
    )
    assert decode_inputs[1].sum().item() == 129
    assert decode_inputs[2].item() == 128
    assert decode_inputs[-1].item() == 128
    assert decode_inputs[3].dtype == torch.float16


def test_run_export_supplies_the_configured_token_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_export_config(CONFIG)
    observed = []
    monkeypatch.setattr(onnx_matrix, "_verify_runtime", lambda _: None)
    monkeypatch.setattr(onnx_matrix, "_artifact_root", lambda: tmp_path)
    monkeypatch.setattr(
        onnx_matrix,
        "load_reference_model",
        lambda **_: SimpleNamespace(model=object()),
    )
    monkeypatch.setattr(
        onnx_matrix,
        "PrefillWrapper",
        lambda model, prompt_length: object(),
    )
    monkeypatch.setattr(
        onnx_matrix,
        "DecodeWrapper",
        lambda model, prompt_length: object(),
    )

    def capture_inputs(contract, *, token_fixture):
        observed.append((contract.graph_kind, token_fixture))
        return ()

    monkeypatch.setattr(onnx_matrix, "build_example_inputs", capture_inputs)
    monkeypatch.setattr(onnx_matrix, "export_onnx_graph", lambda *args: None)

    onnx_matrix.run_export((128,), config)

    assert observed == [
        ("prefill", config.token_fixture),
        ("decode", config.token_fixture),
    ]


def _tiny_causal_model(torch):
    class TinyCausalModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logit_bias = torch.nn.Parameter(
                torch.linspace(-1.0, 1.0, 151_936, dtype=torch.float16)
            )

        def forward(
            self,
            *,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values=None,
            use_cache,
            return_dict,
        ):
            del use_cache, return_dict
            batch, sequence = input_ids.shape
            token_signal = (
                input_ids
                + attention_mask[:, -sequence:]
                + position_ids
            )
            logits = (
                self.logit_bias.reshape(1, 1, -1).expand(
                    batch, sequence, -1
                )
                + token_signal.to(torch.float16).reshape(batch, sequence, 1)
            )
            value = token_signal.to(torch.float16).reshape(
                batch, 1, sequence, 1
            )
            current = value.expand(batch, 8, sequence, 128)
            if past_key_values is None:
                cache = tuple((current, current + 1) for _ in range(NUM_LAYERS))
            else:
                cache = tuple(
                    (
                        torch.cat((key, current), dim=2),
                        torch.cat((value_cache, current + 1), dim=2),
                    )
                    for key, value_cache in past_key_values
                )
            return SimpleNamespace(logits=logits, past_key_values=cache)

    return TinyCausalModel()


def test_prefill_and_decode_wrappers_implement_t12_cache_transition() -> None:
    torch = pytest.importorskip("torch")
    model = _tiny_causal_model(torch)

    prefill_contract = build_prefill_contract(128)
    prefill_inputs = build_example_inputs(prefill_contract)
    prefill_outputs = PrefillWrapper(model, prompt_length=128)(*prefill_inputs)
    validate_tensor_mapping(
        dict(
            zip(
                (tensor.name for tensor in prefill_contract.outputs),
                prefill_outputs,
            )
        ),
        prefill_contract.outputs,
    )
    assert torch.count_nonzero(prefill_outputs[1][:, :, 128:, :]) == 0
    assert prefill_outputs[-1].item() == 128

    decode_contract = build_decode_contract(128)
    decode_inputs = list(build_example_inputs(decode_contract))
    for layer in range(NUM_LAYERS):
        decode_inputs[3 + 2 * layer][:, :, :128, :] = layer
        decode_inputs[3 + 2 * layer + 1][:, :, :128, :] = layer + 1
    decode_outputs = DecodeWrapper(model, prompt_length=128)(*decode_inputs)
    validate_tensor_mapping(
        dict(
            zip(
                (tensor.name for tensor in decode_contract.outputs),
                decode_outputs,
            )
        ),
        decode_contract.outputs,
    )
    assert torch.equal(decode_outputs[1][:, :, :128, :], decode_inputs[3][:, :, :128, :])
    expected_signal = (
        decode_inputs[0].item()
        + decode_inputs[1][:, -1].item()
        + decode_inputs[2].item()
    )
    assert torch.all(decode_outputs[1][:, :, 128, :] == expected_signal)
    assert torch.count_nonzero(decode_outputs[1][:, :, 129:, :]) == 0
    assert decode_outputs[-1].item() == 129


def test_tiny_export_is_static_checked_and_external(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    config = dataclasses.replace(
        load_export_config(CONFIG),
        external_data_threshold_bytes=0,
    )
    contract = build_prefill_contract(128)
    destination = tmp_path / "artifacts/S128/prefill.onnx"
    export_onnx_graph(
        PrefillWrapper(_tiny_causal_model(torch), prompt_length=128),
        build_example_inputs(contract),
        contract,
        destination,
        config,
    )

    record = inspect_onnx_artifact(
        destination,
        contract,
        artifact_directory=tmp_path / "artifacts",
        inline_initializer_limit_bytes=0,
    )
    assert record.relative_path == "S128/prefill.onnx"
    assert record.size_bytes > 0
    assert record.external_data
    assert all(item.size_bytes > 0 for item in record.external_data)
    assert record.input_tensors[0] == {
        "name": "input_ids",
        "dtype": "int64",
        "shape": [1, 128],
    }


def test_existing_artifact_is_never_overwritten(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    config = load_export_config(CONFIG)
    contract = build_prefill_contract(128)
    destination = tmp_path / "prefill.onnx"
    destination.write_bytes(b"owned")

    with pytest.raises(ExportConfigurationError, match="refusing to overwrite"):
        export_onnx_graph(
            PrefillWrapper(_tiny_causal_model(torch), prompt_length=128),
            build_example_inputs(contract),
            contract,
            destination,
            config,
        )


def test_artifact_subdirectory_covers_every_capacity() -> None:
    assert CONTEXT_VARIANTS == {128: 160, 512: 576, 1024: 1152, 4096: 4224}


def _artifact_record(payload) -> OnnxArtifactRecord:
    return OnnxArtifactRecord(
        graph_kind=payload["graph_kind"],
        relative_path=payload["relative_path"],
        sha256=payload["sha256"],
        size_bytes=payload["size_bytes"],
        external_data=tuple(
            ExternalDataRecord(**item) for item in payload["external_data"]
        ),
        input_tensors=tuple(payload["input_tensors"]),
        output_tensors=tuple(payload["output_tensors"]),
    )


def _verify_s128_manifest(manifest, *, evidence_manifest=None) -> None:
    evidence = evidence_manifest or manifest
    verify_manifest_evidence(
        manifest,
        prompt_length=128,
        config=load_export_config(CONFIG),
        prefill=_artifact_record(evidence["artifacts"]["prefill"]),
        decode=_artifact_record(evidence["artifacts"]["decode"]),
        source_weights_sha256=evidence["source_artifact_sha256"],
        host_manifest_sha256=evidence["host_manifest_sha256"],
    )


def test_committed_manifests_cover_real_matrix_and_exact_public_shapes() -> None:
    manifest_directory = ROOT / "results/manifests/onnx"
    paths = sorted(manifest_directory.glob("S*.json"))
    assert [path.name for path in paths] == [
        "S1024.json",
        "S128.json",
        "S4096.json",
        "S512.json",
    ]

    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        validate_manifest("artifact", manifest)
        prompt_length = manifest["context_length"]
        assert manifest["status"] == "exported_and_shape_validated"
        assert manifest["variant_id"] == f"S{prompt_length}"
        assert manifest["cache_capacity"] == CONTEXT_VARIANTS[prompt_length]
        assert manifest["artifacts"]["root"] == (
            "${SLM_LAB_ARTIFACT_ROOT}/onnx/reference/T20"
        )
        provenance = manifest["export_provenance"]
        assert provenance["commit"] == manifest["git_commit"]
        assert provenance["run_attestation"] == (
            load_export_config(CONFIG).evidence_attestation.as_dict()
        )
        assert provenance["exporter_source"]["path"] == (
            "src/slm_lab/export/onnx_matrix.py"
        )
        assert provenance["export_config"]["path"] == (
            "configs/models/qwen3-0.6b-onnx-export.json"
        )
        assert provenance["token_fixture_bundle"][
            "canonical_json_sha256"
        ] == (
            "9f9268ae4a366faa4325271492ec52f035bbf3ba0973d2de61f63382e6302745"
        )
        assert provenance["workload"]["id"] == f"S{prompt_length}"
        for kind, contract in (
            ("prefill", build_prefill_contract(prompt_length)),
            ("decode", build_decode_contract(prompt_length)),
        ):
            artifact = manifest["artifacts"][kind]
            assert artifact["graph_kind"] == kind
            assert artifact["relative_path"] == (
                f"S{prompt_length}/{kind}.onnx"
            )
            assert artifact["size_bytes"] > 0
            assert len(artifact["sha256"]) == 64
            assert artifact["input_tensors"] == [
                {
                    "name": spec.name,
                    "dtype": spec.dtype,
                    "shape": list(spec.shape),
                }
                for spec in contract.inputs
            ]
            assert artifact["output_tensors"] == [
                {
                    "name": spec.name,
                    "dtype": spec.dtype,
                    "shape": list(spec.shape),
                }
                for spec in contract.outputs
            ]
            assert artifact["external_data"]
            assert all(
                len(record["sha256"]) == 64 and record["size_bytes"] > 0
                for record in artifact["external_data"]
            )


def test_manifest_verification_accepts_untampered_evidence() -> None:
    manifest = json.loads(
        (ROOT / "results/manifests/onnx/S128.json").read_text(encoding="utf-8")
    )
    _verify_s128_manifest(manifest)


@pytest.mark.parametrize(
    "field",
    (
        "run_id",
        "exporter_commit",
        "runtime_python_version",
        "source_artifact_sha256",
        "external_data_sha256",
        "graph_hashes",
    ),
)
def test_manifest_rejects_coherent_in_memory_attestation_substitution(
    field: str,
) -> None:
    manifest = json.loads(
        (ROOT / "results/manifests/onnx/S128.json").read_text(encoding="utf-8")
    )
    tampered = json.loads(json.dumps(manifest))
    config = load_export_config(CONFIG)
    attestation = config.evidence_attestation
    prefill = _artifact_record(manifest["artifacts"]["prefill"])
    decode = _artifact_record(manifest["artifacts"]["decode"])
    source_weights_sha256 = manifest["source_artifact_sha256"]

    if field == "run_id":
        attestation = dataclasses.replace(attestation, run_id="substituted-run")
    elif field == "exporter_commit":
        older_commit = "14518d736110fe12b000e84c0738808002900b8f"
        attestation = dataclasses.replace(
            attestation,
            exporter_commit=older_commit,
        )
        tampered["git_commit"] = older_commit
        tampered["export_provenance"]["commit"] = older_commit
        tampered["export_provenance"]["exporter_source"]["sha256"] = (
            "429bc57e68967f91881e1ed0927320fce76220895dd2625e71ef79b99008c00a"
        )
    elif field == "runtime_python_version":
        attestation = dataclasses.replace(
            attestation,
            runtime_python_version="9.9.9",
        )
        tampered["toolchain"]["python"] = "9.9.9"
    elif field == "source_artifact_sha256":
        source_weights_sha256 = "2" * 64
        attestation = dataclasses.replace(
            attestation,
            source_artifact_sha256=source_weights_sha256,
        )
        tampered["source_artifact_sha256"] = source_weights_sha256
    elif field == "external_data_sha256":
        external_data_sha256 = "3" * 64
        attestation = dataclasses.replace(
            attestation,
            external_data_sha256=external_data_sha256,
        )
        prefill = dataclasses.replace(
            prefill,
            external_data=tuple(
                dataclasses.replace(item, sha256=external_data_sha256)
                for item in prefill.external_data
            ),
        )
        decode = dataclasses.replace(
            decode,
            external_data=tuple(
                dataclasses.replace(item, sha256=external_data_sha256)
                for item in decode.external_data
            ),
        )
        for graph_kind in ("prefill", "decode"):
            for item in tampered["artifacts"][graph_kind]["external_data"]:
                item["sha256"] = external_data_sha256
    else:
        graph_sha256 = "4" * 64
        attestation = dataclasses.replace(
            attestation,
            graph_hashes=tuple(
                (
                    context,
                    graph_sha256 if context == 128 else prefill_sha256,
                    decode_sha256,
                )
                for context, prefill_sha256, decode_sha256 in attestation.graph_hashes
            ),
        )
        prefill = dataclasses.replace(prefill, sha256=graph_sha256)
        tampered["artifacts"]["prefill"]["sha256"] = graph_sha256

    tampered["export_provenance"]["run_attestation"] = attestation.as_dict()
    forged_config = dataclasses.replace(
        config,
        evidence_attestation=attestation,
    )

    with pytest.raises(ExportConfigurationError, match="in-memory"):
        verify_manifest_evidence(
            tampered,
            prompt_length=128,
            config=forged_config,
            prefill=prefill,
            decode=decode,
            source_weights_sha256=source_weights_sha256,
            host_manifest_sha256=manifest["host_manifest_sha256"],
        )


def test_manifest_rejects_modified_in_memory_export_setting() -> None:
    manifest = json.loads(
        (ROOT / "results/manifests/onnx/S128.json").read_text(encoding="utf-8")
    )
    config = dataclasses.replace(
        load_export_config(CONFIG),
        external_data_threshold_bytes=0,
    )

    with pytest.raises(ExportConfigurationError, match="in-memory"):
        verify_manifest_evidence(
            manifest,
            prompt_length=128,
            config=config,
            prefill=_artifact_record(manifest["artifacts"]["prefill"]),
            decode=_artifact_record(manifest["artifacts"]["decode"]),
            source_weights_sha256=manifest["source_artifact_sha256"],
            host_manifest_sha256=manifest["host_manifest_sha256"],
        )


def test_manifest_rejects_nested_equality_adapter() -> None:
    manifest = json.loads(
        (ROOT / "results/manifests/onnx/S128.json").read_text(encoding="utf-8")
    )
    tampered = json.loads(json.dumps(manifest))
    config = load_export_config(CONFIG)
    substituted_attestation = dataclasses.replace(
        config.evidence_attestation,
        run_id="equality-adapter-run",
    )
    tampered["export_provenance"]["run_attestation"] = (
        substituted_attestation.as_dict()
    )
    adapted_config = dataclasses.replace(
        config,
        evidence_attestation=_EqualityAdapter(substituted_attestation),
    )

    with pytest.raises(ExportConfigurationError, match="non-exact type"):
        verify_manifest_evidence(
            tampered,
            prompt_length=128,
            config=adapted_config,
            prefill=_artifact_record(manifest["artifacts"]["prefill"]),
            decode=_artifact_record(manifest["artifacts"]["decode"]),
            source_weights_sha256=manifest["source_artifact_sha256"],
            host_manifest_sha256=manifest["host_manifest_sha256"],
        )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("git_commit",), "0" * 40),
        (("chat_template_sha256",), "0" * 64),
        (("exporter_version",), "0.0.0"),
        (("toolchain", "python"), "0.0.0"),
        (
            (
                "export_provenance",
                "run_attestation",
                "runtime_python_version",
            ),
            "0.0.0",
        ),
        (("toolchain", "torch"), "0.0.0"),
        (("toolchain", "transformers"), "0.0.0"),
        (("toolchain", "onnx"), "0.0.0"),
        (("toolchain", "attention_implementation"), "sdpa"),
        (("toolchain", "device"), "cuda"),
        (("runtime",), "onnxruntime"),
        (("runtime_version",), "0.0.0"),
        (("status",), "compiled"),
        (("commands", "export"), "python unpinned_export.py"),
        (("input_contract",), {}),
        (("cache_contract",), {}),
        (("claim_boundary", "does_not_establish"), []),
        (
            ("export_provenance", "exporter_source", "sha256"),
            "0" * 64,
        ),
        (("artifacts", "decode", "sha256"), "0" * 64),
    ),
)
def test_manifest_verification_rejects_deterministic_claim_drift(
    path: tuple[str, ...],
    value,
) -> None:
    manifest = json.loads(
        (ROOT / "results/manifests/onnx/S128.json").read_text(encoding="utf-8")
    )
    tampered = json.loads(json.dumps(manifest))
    target = tampered
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(
        ExportConfigurationError,
        match="manifest|exporter commit",
    ):
        _verify_s128_manifest(tampered, evidence_manifest=manifest)


def test_manifest_rejects_coherent_switch_to_pre_dynamic_cache_ancestor() -> None:
    manifest = json.loads(
        (ROOT / "results/manifests/onnx/S128.json").read_text(encoding="utf-8")
    )
    tampered = json.loads(json.dumps(manifest))
    older_commit = "14518d736110fe12b000e84c0738808002900b8f"
    tampered["git_commit"] = older_commit
    provenance = tampered["export_provenance"]
    provenance["commit"] = older_commit
    provenance["run_attestation"]["exporter_commit"] = older_commit
    provenance["exporter_source"]["sha256"] = (
        "429bc57e68967f91881e1ed0927320fce76220895dd2625e71ef79b99008c00a"
    )

    with pytest.raises(ExportConfigurationError, match="manifest"):
        _verify_s128_manifest(tampered, evidence_manifest=manifest)


def test_manifest_rejects_paired_runtime_python_tampering() -> None:
    manifest = json.loads(
        (ROOT / "results/manifests/onnx/S128.json").read_text(encoding="utf-8")
    )
    tampered = json.loads(json.dumps(manifest))
    tampered["toolchain"]["python"] = "9.9.9"
    tampered["export_provenance"]["run_attestation"][
        "runtime_python_version"
    ] = "9.9.9"

    with pytest.raises(ExportConfigurationError, match="manifest"):
        _verify_s128_manifest(tampered, evidence_manifest=manifest)
