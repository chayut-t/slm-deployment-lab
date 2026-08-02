from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_lab.models.qwen3_reference import (
    EXPECTED_MODEL_ID,
    EXPECTED_MODEL_REVISION,
    ReferenceConfigurationError,
    load_model_contract,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL_CONTRACT = ROOT / "configs/models/qwen3-0.6b.yaml"
TOKEN_FIXTURES = ROOT / "tests/fixtures/t10/token-fixtures-v1.json"
QWEN_REFERENCE_FIXTURE = (
    ROOT / "tests/reference/fixtures/qwen3-0.6b-raw-ascii-bf16-cpu-v1.json"
)


def test_frozen_qwen_contract_loads() -> None:
    contract = load_model_contract()

    assert contract.model_id == EXPECTED_MODEL_ID
    assert contract.revision == EXPECTED_MODEL_REVISION
    assert contract.reference_dtype == "bfloat16"
    assert contract.trust_remote_code is False
    assert contract.eos_token_id == 151645
    assert contract.pad_token_id == 151643
    assert contract.architecture["num_hidden_layers"] == 28
    assert contract.architecture["num_key_value_heads"] == 8


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("model", "id"), "not-qwen"),
        (("model", "revision"), "main"),
        (("model", "trust_remote_code"), True),
        (("model", "reference_dtype"), "auto"),
        (("tokenizer", "tokens", "eos_id"), "151645"),
    ],
)
def test_contract_drift_is_rejected(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    payload = json.loads(MODEL_CONTRACT.read_text(encoding="utf-8"))
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    modified = tmp_path / "model-contract.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReferenceConfigurationError):
        load_model_contract(modified)


def test_qwen_golden_fixture_is_bound_to_t10_and_exact_runtime() -> None:
    token_bundle = json.loads(TOKEN_FIXTURES.read_text(encoding="utf-8"))
    golden = json.loads(QWEN_REFERENCE_FIXTURE.read_text(encoding="utf-8"))
    raw_ascii = next(
        fixture
        for fixture in token_bundle["raw_canaries"]
        if fixture["id"] == "raw_ascii"
    )

    assert golden["model"] == {
        "id": EXPECTED_MODEL_ID,
        "revision": EXPECTED_MODEL_REVISION,
        "weights_format": "safetensors",
        "trust_remote_code": False,
        "attention_implementation": "eager",
    }
    assert golden["source_fixture"]["token_ids_sha256"] == raw_ascii["token_ids_sha256"]
    assert golden["prompt_token_ids"] == raw_ascii["token_ids"]
    assert golden["runtime"] == {
        "python_version": "3.11.13",
        "torch_version": "2.7.1",
        "transformers_version": "4.51.3",
        "safetensors_version": "0.8.0",
        "device": "cpu",
        "dtype": "bfloat16",
        "requested_attention_implementation": "eager",
        "actual_attention_implementation": "eager",
        "deterministic_algorithms": True,
        "seed": 0,
    }
    assert [step["selected_token_id"] for step in golden["steps"]] == golden[
        "generated_token_ids"
    ]
    assert all(step["passed"] for step in golden["steps"])
    assert all(
        step["full_logits_sha256"] == step["cached_logits_sha256"]
        for step in golden["steps"]
    )
