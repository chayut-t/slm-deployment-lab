from __future__ import annotations

# ruff: noqa: E402

import json
import os
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from slm_lab.backends.mlx.config import CacheLayout
from slm_lab.backends.mlx.model import load_custom_qwen3
from slm_lab.generation.mlx import greedy_generate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    PROJECT_ROOT / "tests/reference/fixtures/qwen3-0.6b-raw-ascii-bf16-cpu-v1.json"
)


@pytest.mark.skipif(
    os.environ.get("SLM_LAB_RUN_MLX_QWEN") != "1",
    reason="set SLM_LAB_RUN_MLX_QWEN=1 for the local pinned-weight M4 check",
)
def test_real_qwen_canary_and_layout_parity() -> None:
    model_dir = os.environ.get("SLM_LAB_MLX_MODEL_DIR")
    if not model_dir:
        pytest.fail("SLM_LAB_MLX_MODEL_DIR is required for the real-weight gate")
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    model = load_custom_qwen3(model_dir)

    results = {
        layout: greedy_generate(
            model,
            fixture["prompt_token_ids"],
            max_new_tokens=fixture["max_new_tokens"]
            if "max_new_tokens" in fixture
            else len(fixture["generated_token_ids"]),
            capacity=len(fixture["prompt_token_ids"])
            + len(fixture["generated_token_ids"]),
            eos_token_id=fixture["eos_token_id"] if "eos_token_id" in fixture else None,
            layout=layout,
        )
        for layout in CacheLayout
    }
    mx.synchronize()

    expected = tuple(fixture["generated_token_ids"])
    assert results[CacheLayout.HEAD_MAJOR].generated_token_ids == expected
    assert results[CacheLayout.SEQUENCE_MAJOR].generated_token_ids == expected
    assert results[CacheLayout.HEAD_MAJOR].cache.nbytes == (
        results[CacheLayout.SEQUENCE_MAJOR].cache.nbytes
    )
    assert all(
        layer.num_key_value_heads == model.config.num_key_value_heads == 8
        for result in results.values()
        for layer in result.cache.layers
    )
