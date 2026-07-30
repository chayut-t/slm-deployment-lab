from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slm_lab.contracts import (
    CONTEXT_VARIANTS,
    build_decode_contract,
    build_prefill_contract,
    validate_tensor_mapping,
)
from slm_lab.contracts.static_cache import NUM_LAYERS
from slm_lab.export.onnx_matrix import (
    DecodeWrapper,
    ExportConfigurationError,
    PrefillWrapper,
    build_example_inputs,
    export_onnx_graph,
    inspect_onnx_artifact,
    load_export_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/models/qwen3-0.6b-onnx-export.json"


def test_export_config_freezes_exact_matrix_and_toolchain() -> None:
    config = load_export_config(CONFIG)

    assert config.contexts == (128, 512, 1024, 4096)
    assert config.opset == 18
    assert config.precision == "float16"
    assert config.torch_version == "2.7.1"
    assert config.transformers_version == "4.51.3"
    assert config.onnx_version == "1.18.0"
    assert config.external_data_threshold_bytes == 1024


def test_export_config_rejects_context_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["contexts"] = [128]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExportConfigurationError, match="contexts must exactly"):
        load_export_config(path)


def test_example_inputs_use_exact_fixture_and_contract() -> None:
    torch = pytest.importorskip("torch")
    fixture = json.loads(
        (ROOT / "tests/fixtures/t10/token-fixtures-v1.json").read_text(
            encoding="utf-8"
        )
    )
    expected = fixture["context_workloads"][0]["token_ids"]

    prefill = build_prefill_contract(128)
    prefill_inputs = build_example_inputs(prefill)
    validate_tensor_mapping(
        dict(zip((tensor.name for tensor in prefill.inputs), prefill_inputs)),
        prefill.inputs,
    )
    assert prefill_inputs[0].tolist() == [expected]

    decode = build_decode_contract(128)
    decode_inputs = build_example_inputs(decode)
    validate_tensor_mapping(
        dict(zip((tensor.name for tensor in decode.inputs), decode_inputs)),
        decode.inputs,
    )
    assert decode_inputs[1].sum().item() == 129
    assert decode_inputs[2].item() == 128
    assert decode_inputs[-1].item() == 128
    assert decode_inputs[3].dtype == torch.float16


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
