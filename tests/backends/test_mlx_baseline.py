from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest
from jsonschema import Draft202012Validator

from slm_lab.backends import mlx_baseline
from slm_lab.backends.mlx_baseline import (
    EXPECTED_GENERATED_TOKEN_IDS,
    MlxBaselineError,
    _measure_generation_loop,
    _measure_ttft,
    validate_repetition_policy,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "environments/macos-m4/mlx-baseline-run-v2.schema.json"


class FakeMx:
    def __init__(self) -> None:
        self.sync_calls: list[Any] = []
        self.reset_count = 0

    def synchronize(self, stream: Any) -> None:
        self.sync_calls.append(stream)

    def reset_peak_memory(self) -> None:
        self.reset_count += 1

    def get_peak_memory(self) -> int:
        return 123

    def array(self, values: Any) -> Any:
        return values


def _fake_generate_step(
    _prompt: Any,
    _model: Any,
    *,
    max_tokens: int,
) -> Iterator[tuple[int, None]]:
    if max_tokens == 0:
        return
    for token in EXPECTED_GENERATED_TOKEN_IDS[:max_tokens]:
        yield token, None


def test_result_schema_is_valid_draft_2020_12() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)


def test_repetition_policy_is_exact() -> None:
    validate_repetition_policy(2, 10)
    for warmups, measurements in ((0, 10), (1, 10), (2, 9), (2, 11)):
        with pytest.raises(MlxBaselineError, match="exactly 2 warm-up and 10"):
            validate_repetition_policy(warmups, measurements)


def test_invalid_repetitions_fail_before_source_or_model_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_accessed = False

    def fail_if_accessed() -> str:
        nonlocal source_accessed
        source_accessed = True
        raise AssertionError("source/model setup must not be reached")

    monkeypatch.setattr(mlx_baseline, "_clean_source_commit", fail_if_accessed)
    with pytest.raises(MlxBaselineError, match="exactly 2 warm-up and 10"):
        mlx_baseline.run_baseline(
            model_path=Path("/not-used"),
            output_dir=Path("/not-used"),
            warmup_repetitions=1,
            measured_repetitions=10,
        )
    assert not source_accessed


def test_ttft_fences_mlx_lm_stream_and_disables_lookahead() -> None:
    mx = FakeMx()
    generation_stream = object()

    seconds, peak_memory = _measure_ttft(
        mx=mx,
        generate_step=_fake_generate_step,
        generation_stream=generation_stream,
        model=object(),
        prompt_token_ids=[1, 2, 3],
    )

    assert seconds > 0
    assert peak_memory == 123
    assert mx.sync_calls == [generation_stream, generation_stream]
    assert mx.reset_count == 1


def test_generation_loop_fences_mlx_lm_stream_and_returns_canary() -> None:
    mx = FakeMx()
    generation_stream = object()

    generated, seconds, peak_memory = _measure_generation_loop(
        mx=mx,
        generate_step=_fake_generate_step,
        generation_stream=generation_stream,
        model=object(),
        prompt_token_ids=[1, 2, 3],
        max_new_tokens=3,
    )

    assert generated == EXPECTED_GENERATED_TOKEN_IDS
    assert seconds > 0
    assert peak_memory == 123
    assert mx.sync_calls == [generation_stream, generation_stream]
    assert mx.reset_count == 1
