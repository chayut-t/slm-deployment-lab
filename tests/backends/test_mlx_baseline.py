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
    _canonical_json_sha256,
    _measure_generation_loop,
    _measure_ttft,
    validate_evidence,
    validate_repetition_policy,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "environments/macos-m4/mlx-baseline-run-v2.schema.json"
RESULT_PATH = ROOT / "results/raw/apple/baseline/mlx-lm-baseline-run-v2.json"


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


def _write_mutated_result(
    tmp_path: Path,
    mutation: Any,
    *,
    update_anchor: bool,
) -> Path:
    document = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    original_digest = document.pop("evidence_sha256")
    mutation(document)
    mutated_digest = _canonical_json_sha256(document)
    document["evidence_sha256"] = mutated_digest

    result_path = tmp_path / RESULT_PATH.name
    result_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    anchored_digest = mutated_digest if update_anchor else original_digest
    result_path.with_suffix(result_path.suffix + ".sha256").write_text(
        f"{anchored_digest}  {result_path.name}\n",
        encoding="utf-8",
    )
    return result_path


def test_committed_result_passes_full_validator() -> None:
    validate_evidence(RESULT_PATH)


def test_recomputed_self_digest_cannot_bypass_external_anchor(
    tmp_path: Path,
) -> None:
    result_path = _write_mutated_result(
        tmp_path,
        lambda document: document["samples"][0].__setitem__("ttft_seconds", 999.0),
        update_anchor=False,
    )

    with pytest.raises(MlxBaselineError, match="external digest anchor differs"):
        validate_evidence(result_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda document: document["host"].__setitem__(
                "target_chip", "Fabricated chip"
            ),
            "schema failure",
        ),
        (
            lambda document: document["runner"].__setitem__("sha256", "0" * 64),
            "runner does not match",
        ),
        (
            lambda document: document["summary"]["ttft_seconds"].__setitem__(
                "median", 999.0
            ),
            "summary does not match",
        ),
        (
            lambda document: document["protocol"].__setitem__("sha256", "0" * 64),
            "benchmark protocol provenance differs",
        ),
        (
            lambda document: document["canary"]["generation"].__setitem__(
                "prompt_token_ids_sha256", "0" * 64
            ),
            "schema failure",
        ),
        (
            lambda document: document["measurement_policy"].__setitem__(
                "ttft_boundary",
                "Includes model loading and every later decode.",
            ),
            "schema failure",
        ),
        (
            lambda document: document["measurement_policy"].__setitem__(
                "generation_loop_boundary",
                "Stops before the generation stream is fenced.",
            ),
            "schema failure",
        ),
        (
            lambda document: document["measurement_policy"].__setitem__(
                "lookahead_accounting",
                "No unreturned look-ahead token is computed.",
            ),
            "schema failure",
        ),
        (
            lambda document: document["measurement_policy"].__setitem__(
                "model_load_boundary",
                "Model loading is included and this is a cold-start result.",
            ),
            "schema failure",
        ),
    ],
)
def test_recomputed_digest_and_anchor_still_fail_semantic_validation(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    result_path = _write_mutated_result(
        tmp_path,
        mutation,
        update_anchor=True,
    )

    with pytest.raises(MlxBaselineError, match=message):
        validate_evidence(result_path)
