from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from slm_lab.generation.reference import (  # noqa: E402
    DEFAULT_TOLERANCE,
    NumericalTolerance,
    ReferenceExecutionError,
    compare_full_and_cached,
    compare_logits,
    generate_cached,
    generate_full_forward,
    load_fixture_token_ids,
)
from slm_lab.models.qwen3_reference import load_reference_model  # noqa: E402
from slm_lab.models.qwen3_reference import (  # noqa: E402
    ReferenceConfigurationError,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests/reference/fixtures/deterministic-causal-reference-v1.json"
QWEN_FIXTURE_PATH = (
    ROOT / "tests/reference/fixtures/qwen3-0.6b-raw-ascii-bf16-cpu-v1.json"
)


class DeterministicCausalModel(torch.nn.Module):
    """Tiny causal model with a transparent, exactly cacheable next-token rule."""

    def __init__(self, vocabulary_size: int = 17) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        past_key_values: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        assert return_dict
        if attention_mask.shape[1] != input_ids.shape[1] + (
            0 if past_key_values is None else past_key_values.shape[1]
        ):
            raise AssertionError("attention mask does not cover the complete prefix")
        full_prefix = (
            input_ids
            if past_key_values is None
            else torch.cat((past_key_values, input_ids), dim=1)
        )
        cumulative = full_prefix.cumsum(dim=1)
        positions = torch.arange(
            1, full_prefix.shape[1] + 1, device=input_ids.device
        ).unsqueeze(0)
        targets = (cumulative + 3 * positions) % self.vocabulary_size
        vocabulary = torch.arange(self.vocabulary_size, device=input_ids.device).view(
            1, 1, -1
        )
        full_logits = -(vocabulary - targets.unsqueeze(-1)).abs().to(torch.float64)
        logits = full_logits[:, -input_ids.shape[1] :, :]
        cache = full_prefix.detach().clone() if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=cache)


class DivergentPrefillModel(DeterministicCausalModel):
    """Select a wrong cached token once and record teacher-forced decode IDs."""

    def __init__(self) -> None:
        super().__init__()
        self.cached_decode_token_ids: list[int] = []

    def forward(self, **kwargs: object) -> SimpleNamespace:
        past_key_values = kwargs.get("past_key_values")
        output = super().forward(**kwargs)
        if kwargs["use_cache"] and past_key_values is None:
            output.logits[:, -1, :] = -1000
            output.logits[:, -1, -1] = 1000
        elif kwargs["use_cache"]:
            token = kwargs["input_ids"]
            assert isinstance(token, torch.Tensor)
            self.cached_decode_token_ids.append(int(token.item()))
        return output


class MissingDecodeCacheModel(DeterministicCausalModel):
    """Return prefill cache correctly, then omit the first decode cache."""

    def forward(self, **kwargs: object) -> SimpleNamespace:
        is_decode = kwargs.get("past_key_values") is not None
        output = super().forward(**kwargs)
        if is_decode:
            output.past_key_values = None
        return output


@pytest.fixture
def frozen_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _inputs(fixture: dict) -> torch.Tensor:
    return torch.tensor([fixture["prompt_token_ids"]], dtype=torch.long)


def test_full_and_cached_generation_match_frozen_tokens(frozen_fixture: dict) -> None:
    model = DeterministicCausalModel(frozen_fixture["model"]["vocabulary_size"])
    input_ids = _inputs(frozen_fixture)
    options = {
        "max_new_tokens": frozen_fixture["max_new_tokens"],
        "eos_token_id": frozen_fixture["eos_token_id"],
    }

    full = generate_full_forward(model, input_ids, **options)
    cached = generate_cached(model, input_ids, **options)

    expected = tuple(frozen_fixture["expected_generated_token_ids"])
    assert full == expected
    assert cached == expected


def test_stepwise_parity_and_evidence_are_deterministic(frozen_fixture: dict) -> None:
    model = DeterministicCausalModel(frozen_fixture["model"]["vocabulary_size"])
    input_ids = _inputs(frozen_fixture)
    options = {
        "max_new_tokens": frozen_fixture["max_new_tokens"],
        "eos_token_id": frozen_fixture["eos_token_id"],
    }

    first = compare_full_and_cached(model, input_ids, **options)
    second = compare_full_and_cached(model, input_ids, **options)

    assert first.passed
    assert len(first.steps) == frozen_fixture["max_new_tokens"]
    assert first.generated_token_ids == tuple(
        frozen_fixture["expected_generated_token_ids"]
    )
    assert first.evidence_sha256 == second.evidence_sha256
    assert all(step.metrics.max_absolute_error == 0 for step in first.steps)
    assert all(step.metrics.top1_agreement for step in first.steps)
    assert all(step.metrics.top5_overlap == 1 for step in first.steps)
    expected_digest = frozen_fixture["expected_evidence_sha256"]
    if expected_digest is not None:
        assert first.evidence_sha256 == expected_digest


def test_eos_is_included_and_stops_both_paths() -> None:
    model = DeterministicCausalModel()
    input_ids = torch.tensor([[2, 5, 1]], dtype=torch.long)

    full = generate_full_forward(model, input_ids, max_new_tokens=6, eos_token_id=3)
    cached = generate_cached(model, input_ids, max_new_tokens=6, eos_token_id=3)
    evidence = compare_full_and_cached(
        model, input_ids, max_new_tokens=6, eos_token_id=3
    )

    assert full == (0, 3)
    assert cached == (0, 3)
    assert evidence.generated_token_ids == (0, 3)
    assert evidence.stopped_on_eos


def test_numerical_tolerance_rejects_logit_and_token_drift() -> None:
    reference = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
    candidate = reference.clone()
    candidate[0, 0] = 8.0

    metrics = compare_logits(
        reference,
        candidate,
        NumericalTolerance(
            atol=0,
            rtol=0,
            protected_relative_max=0,
            cosine_min=1,
            top5_overlap_min=1,
        ),
    )

    assert not metrics.passed
    assert not metrics.allclose
    assert not metrics.top1_agreement


def test_token_mismatch_is_teacher_forced_to_keep_later_prefixes_equal() -> None:
    model = DivergentPrefillModel()
    input_ids = torch.tensor([[2, 5, 1]], dtype=torch.long)

    evidence = compare_full_and_cached(
        model,
        input_ids,
        max_new_tokens=3,
        eos_token_id=None,
    )

    assert evidence.generated_token_ids == (0, 3, 9)
    assert not evidence.passed
    assert not evidence.steps[0].metrics.top1_agreement
    assert evidence.steps[0].metrics.top1_reference == 0
    assert evidence.steps[0].metrics.top1_candidate == 16
    assert evidence.steps[1].metrics.passed
    assert evidence.steps[2].metrics.passed
    assert model.cached_decode_token_ids == [0, 3]


def test_invalid_shapes_missing_cache_and_nonfinite_logits_fail() -> None:
    model = DeterministicCausalModel()
    with pytest.raises(ReferenceExecutionError, match="shape"):
        generate_cached(
            model,
            torch.tensor([1, 2, 3]),
            max_new_tokens=1,
            eos_token_id=None,
        )

    class MissingCacheModel(DeterministicCausalModel):
        def forward(self, **kwargs: object) -> SimpleNamespace:
            output = super().forward(**kwargs)
            output.past_key_values = None
            return output

    with pytest.raises(ReferenceExecutionError, match="past_key_values"):
        generate_cached(
            MissingCacheModel(),
            torch.tensor([[1, 2, 3]]),
            max_new_tokens=1,
            eos_token_id=None,
        )

    for runner in (generate_cached, compare_full_and_cached):
        with pytest.raises(
            ReferenceExecutionError,
            match="cached decode step 1 returned no past_key_values",
        ):
            runner(
                MissingDecodeCacheModel(),
                torch.tensor([[1, 2, 3]]),
                max_new_tokens=2,
                eos_token_id=None,
            )

    finite = torch.zeros((1, 6))
    nonfinite = finite.clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(ReferenceExecutionError, match="NaN"):
        compare_logits(finite, nonfinite, DEFAULT_TOLERANCE)


def test_t10_authored_fixture_ids_are_consumed_without_retokenizing() -> None:
    raw_ascii = load_fixture_token_ids("raw_ascii")

    assert len(raw_ascii) == 18
    assert raw_ascii[:4] == (840, 20772, 3170, 8356)
    with pytest.raises(ReferenceExecutionError, match="unknown"):
        load_fixture_token_ids("not-a-fixture")


def test_loader_records_requested_and_actual_eager_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    class FakeLoadedModel:
        def __init__(self, actual_attention: str) -> None:
            self.config = SimpleNamespace(_attn_implementation=actual_attention)

        def eval(self) -> FakeLoadedModel:
            return self

        def requires_grad_(self, enabled: bool) -> FakeLoadedModel:
            assert enabled is False
            return self

        def to(self, device: str) -> FakeLoadedModel:
            assert device == "cpu"
            return self

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: FakeLoadedModel("eager"),
    )

    reference = load_reference_model(device="cpu", dtype="float32")

    assert reference.runtime.requested_attention_implementation == "eager"
    assert reference.runtime.actual_attention_implementation == "eager"
    assert reference.runtime.as_dict()["actual_attention_implementation"] == "eager"

    with pytest.raises(ReferenceConfigurationError, match="requires.*eager"):
        load_reference_model(attn_implementation="sdpa")

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: FakeLoadedModel("sdpa"),
    )
    with pytest.raises(ReferenceConfigurationError, match="requested.*actual"):
        load_reference_model()


@pytest.mark.skipif(
    os.environ.get("SLM_LAB_RUN_QWEN_REFERENCE") != "1",
    reason="set SLM_LAB_RUN_QWEN_REFERENCE=1 when pinned public weights are local",
)
def test_pinned_qwen_full_cached_parity() -> None:
    golden = json.loads(QWEN_FIXTURE_PATH.read_text(encoding="utf-8"))
    reference = load_reference_model(
        device="cpu",
        dtype="bfloat16",
        local_files_only=True,
    )
    token_ids = load_fixture_token_ids("raw_ascii")
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    evidence = compare_full_and_cached(
        reference.model,
        input_ids,
        max_new_tokens=3,
        eos_token_id=reference.contract.eos_token_id,
    )

    assert evidence.passed
    assert evidence.generated_token_ids == tuple(golden["generated_token_ids"])
    assert evidence.evidence_sha256 == golden["evidence_sha256"]
    assert reference.runtime.as_dict() == golden["runtime"]
