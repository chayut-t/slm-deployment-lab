"""Self-tests for the T21 ONNX Runtime CPU parity runner.

Everything here runs against injected fake sessions on the standard library
alone. No number produced by this file is a parity measurement; the runner
labels every such run ``evidence_tier="fake_session_self_test"`` and one test
proves a fake cannot claim otherwise.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

import slm_lab.backends.onnx_cpu as onnx_cpu
from slm_lab.backends.onnx_cpu import (
    DEFAULT_ORT_CPU_TOLERANCE,
    EvidenceTier,
    FailureKind,
    OnnxCpuError,
    OrtCpuParityRunner,
    ParityInputError,
    ParityTolerance,
    PlainTensor,
    build_parser,
    compare_logits,
    detect_evidence_tier,
    flatten,
    main,
    to_nested_list,
    top_indices,
)
from slm_lab.contracts.static_cache import (
    CONTEXT_VARIANTS,
    TensorSpec,
    build_decode_contract,
    build_prefill_contract,
)


ROOT = Path(__file__).resolve().parents[2]
PROMPT_LENGTH = 128
CAPACITY = CONTEXT_VARIANTS[PROMPT_LENGTH]
CACHE_LAYOUT = ("batch", "kv_head", "cache_position", "head_dim")


# ---------------------------------------------------------------------------
# Reduced-shape contracts and deterministic fake sessions.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """A contract-shaped but cheap graph boundary for fast unit tests."""

    layers: int = 2
    kv_heads: int = 2
    head_dim: int = 4
    vocab: int = 8
    prompt_length: int = PROMPT_LENGTH
    capacity: int = CAPACITY

    @property
    def cache_shape(self) -> tuple[int, ...]:
        return (1, self.kv_heads, self.capacity, self.head_dim)


REAL_GEOMETRY = Geometry(layers=28, kv_heads=8, head_dim=128, vocab=151_936)


def _spec(name: str, dtype: str, shape: tuple[int, ...], layout: tuple[str, ...]):
    return TensorSpec(name, dtype, shape, layout, f"fake {name}")


def _cache_specs(geometry: Geometry, prefixes: tuple[str, str]) -> list[TensorSpec]:
    specs: list[TensorSpec] = []
    for layer in range(geometry.layers):
        for prefix in prefixes:
            specs.append(
                _spec(
                    f"{prefix}.{layer}",
                    "float16",
                    geometry.cache_shape,
                    CACHE_LAYOUT,
                )
            )
    return specs


def reduced_contracts(geometry: Geometry) -> tuple[Any, Any]:
    """Build prefill/decode contracts with the real capacity but tiny tensors."""

    from slm_lab.contracts.static_cache import GraphContract

    sequence = ("batch", "sequence")
    prefill_inputs = (
        _spec("input_ids", "int64", (1, geometry.prompt_length), sequence),
        _spec("attention_mask", "int64", (1, geometry.prompt_length), sequence),
        _spec("position_ids", "int64", (1, geometry.prompt_length), sequence),
    )
    prefill_outputs = (
        _spec("last_logits", "float32", (1, geometry.vocab), ("batch", "vocabulary")),
        *_cache_specs(geometry, ("key_cache", "value_cache")),
        _spec("valid_length", "int64", (1,), ("batch",)),
    )
    decode_inputs = (
        _spec("input_ids", "int64", (1, 1), sequence),
        _spec(
            "attention_mask",
            "int64",
            (1, geometry.capacity),
            ("batch", "cache_position"),
        ),
        _spec("position_ids", "int64", (1, 1), sequence),
        *_cache_specs(geometry, ("key_cache", "value_cache")),
        _spec("valid_length", "int64", (1,), ("batch",)),
    )
    decode_outputs = (
        _spec("next_logits", "float32", (1, geometry.vocab), ("batch", "vocabulary")),
        *_cache_specs(geometry, ("present_key", "present_value")),
        _spec("updated_valid_length", "int64", (1,), ("batch",)),
    )
    prefill = GraphContract(
        graph_kind="prefill",
        prompt_length=geometry.prompt_length,
        cache_capacity=geometry.capacity,
        inputs=prefill_inputs,
        outputs=prefill_outputs,
    )
    decode = GraphContract(
        graph_kind="decode",
        prompt_length=geometry.prompt_length,
        cache_capacity=geometry.capacity,
        inputs=decode_inputs,
        outputs=decode_outputs,
    )
    return prefill, decode


_ROW_CACHE: dict[tuple[int, int], list[list[float]]] = {}


def _rows(head_dim: int, offset_count: int = 61) -> list[list[float]]:
    key = (head_dim, offset_count)
    if key not in _ROW_CACHE:
        _ROW_CACHE[key] = [
            [((offset + dim) % offset_count) / 8.0 + 0.125 for dim in range(head_dim)]
            for offset in range(offset_count)
        ]
    return _ROW_CACHE[key]


def _name_seed(name: str) -> int:
    return sum(ord(character) for character in name)


def prefill_cache_values(spec: TensorSpec, prompt_length: int) -> list[float]:
    """Prompt prefix filled with distinct rows; reserved tail exactly zero."""

    _, kv_heads, capacity, head_dim = spec.shape
    rows = _rows(head_dim)
    seed = _name_seed(spec.name)
    zero_row = [0.0] * head_dim
    values: list[float] = []
    for head in range(kv_heads):
        for position in range(capacity):
            if position < prompt_length:
                values.extend(rows[(seed + head * 7 + position * 13) % len(rows)])
            else:
                values.extend(zero_row)
    return values


def _write_slot(
    flat: list[float],
    spec: TensorSpec,
    index: int,
    marker: int,
    *,
    fill: float | None = None,
) -> None:
    _, kv_heads, capacity, head_dim = spec.shape
    if not 0 <= index < capacity:
        # A faithful graph cannot write outside its own capacity, and a slice
        # assignment past the end would silently grow the tensor into a
        # contract violation instead of the state fault under test.
        return
    rows = _rows(head_dim)
    seed = _name_seed(spec.name)
    for head in range(kv_heads):
        start = head * capacity * head_dim + index * head_dim
        row = rows[(seed + marker * 17 + head * 5) % len(rows)]
        if fill is not None:
            row = [fill] * head_dim
        flat[start : start + head_dim] = row


class FakeReference:
    """Deterministic golden logits and teacher-forced tokens."""

    def __init__(self, *, prompt: Sequence[int], steps: int, vocab: int) -> None:
        self._prompt = tuple(prompt)
        self._logits = [
            [float((index + step) % vocab) for index in range(vocab)]
            for step in range(steps + 1)
        ]

    def prompt_token_ids(self) -> Sequence[int]:
        return self._prompt

    def next_logits(self, step: int) -> Sequence[float]:
        return self._logits[step]

    def expected_token_id(self, step: int) -> int:
        return top_indices(self._logits[step], 1)[0]

    def provenance(self) -> Mapping[str, Any]:
        return {
            "source": "tests.onnx.test_onnx_cpu_parity.FakeReference",
            "tier": "fake_session_self_test",
            "steps": len(self._logits) - 1,
        }


class FakeSession:
    """Common fake ONNX Runtime session surface."""

    def __init__(
        self,
        contract: Any,
        geometry: Geometry,
        reference: FakeReference,
        *,
        declared_names: Sequence[str] | None = None,
        output_overrides: Mapping[str, Any] | None = None,
        drop_last_outputs: int = 0,
    ) -> None:
        self.contract = contract
        self.geometry = geometry
        self.reference = reference
        self.declared_names = list(
            declared_names or [spec.name for spec in contract.outputs]
        )
        self.output_overrides = dict(output_overrides or {})
        self.drop_last_outputs = drop_last_outputs
        self.feeds: list[dict[str, Any]] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=spec.name) for spec in self.contract.inputs]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=name) for name in self.declared_names]

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, Any],
    ) -> list[Any]:
        self.feeds.append(dict(input_feed))
        mapping = self._compute(input_feed)
        mapping.update(self.output_overrides)
        values = [mapping[spec.name] for spec in self.contract.outputs]
        if self.drop_last_outputs:
            values = values[: -self.drop_last_outputs]
        return values

    def _compute(self, feed: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class FakePrefillSession(FakeSession):
    def _compute(self, feed: Mapping[str, Any]) -> dict[str, Any]:
        geometry = self.geometry
        mapping: dict[str, Any] = {
            "last_logits": PlainTensor(
                tuple(self.reference.next_logits(0)),
                (1, geometry.vocab),
                "float32",
            ),
            "valid_length": PlainTensor((geometry.prompt_length,), (1,), "int64"),
        }
        for spec in self.contract.outputs:
            if spec.name.partition(".")[0] in ("key_cache", "value_cache"):
                mapping[spec.name] = PlainTensor(
                    tuple(prefill_cache_values(spec, geometry.prompt_length)),
                    spec.shape,
                    "float16",
                )
        return mapping


class FakeDecodeSession(FakeSession):
    """Faithful by default; every fault mode is an explicit injection."""

    def __init__(
        self,
        *args: Any,
        write_offset: int = 0,
        valid_length_increment: int = 1,
        valid_length_override: int | None = None,
        corrupt_prefix_step: int | None = None,
        corrupt_prefix_position: int = 5,
        corrupt_target: str = "present_key.0",
        tail_write_offset: int | None = None,
        slot_fill: float | None = None,
        logit_bias: float = 0.0,
        logit_bias_steps: Sequence[int] = (),
        non_finite_steps: Sequence[int] = (),
        non_finite_value: float = float("nan"),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.write_offset = write_offset
        self.valid_length_increment = valid_length_increment
        self.valid_length_override = valid_length_override
        self.corrupt_prefix_step = corrupt_prefix_step
        self.corrupt_prefix_position = corrupt_prefix_position
        self.corrupt_target = corrupt_target
        self.tail_write_offset = tail_write_offset
        self.slot_fill = slot_fill
        self.logit_bias = logit_bias
        self.logit_bias_steps = tuple(logit_bias_steps)
        # `output_overrides` replaces an output on *every* step, so it cannot
        # express "NaN at step 1, drifted at step 2". These two are step-indexed
        # for exactly that case.
        self.non_finite_steps = tuple(non_finite_steps)
        self.non_finite_value = non_finite_value
        self.step = 0

    def _compute(self, feed: Mapping[str, Any]) -> dict[str, Any]:
        self.step += 1
        step = self.step
        geometry = self.geometry
        valid_length = feed["valid_length"].values[0]

        logits = list(self.reference.next_logits(step))
        if self.logit_bias and step in self.logit_bias_steps:
            logits = [value + self.logit_bias for value in logits]
        if step in self.non_finite_steps:
            logits = [self.non_finite_value] * len(logits)
        updated = (
            self.valid_length_override
            if self.valid_length_override is not None
            else valid_length + self.valid_length_increment
        )
        mapping: dict[str, Any] = {
            "next_logits": PlainTensor(tuple(logits), (1, geometry.vocab), "float32"),
            "updated_valid_length": PlainTensor((updated,), (1,), "int64"),
        }
        for spec in self.contract.outputs:
            prefix, _, suffix = spec.name.partition(".")
            if prefix not in ("present_key", "present_value"):
                continue
            source = "key_cache" if prefix == "present_key" else "value_cache"
            incoming = feed[f"{source}.{suffix}"]
            flat = list(incoming.values)
            _write_slot(
                flat,
                spec,
                valid_length + self.write_offset,
                step,
                fill=self.slot_fill,
            )
            if self.tail_write_offset is not None:
                _write_slot(
                    flat, spec, valid_length + self.tail_write_offset, step + 100
                )
            if step == self.corrupt_prefix_step and spec.name == self.corrupt_target:
                index = self.corrupt_prefix_position * geometry.head_dim
                flat[index] = flat[index] + 1.0
            mapping[spec.name] = PlainTensor(tuple(flat), spec.shape, "float16")
        return mapping


def build_runner(
    geometry: Geometry = Geometry(),
    *,
    steps: int = 3,
    prefill_kwargs: Mapping[str, Any] | None = None,
    decode_kwargs: Mapping[str, Any] | None = None,
    tolerance: ParityTolerance = DEFAULT_ORT_CPU_TOLERANCE,
) -> tuple[OrtCpuParityRunner, FakePrefillSession, FakeDecodeSession]:
    prefill_contract, decode_contract = reduced_contracts(geometry)
    reference = FakeReference(
        prompt=range(geometry.prompt_length), steps=steps, vocab=geometry.vocab
    )
    prefill_session = FakePrefillSession(
        prefill_contract, geometry, reference, **(prefill_kwargs or {})
    )
    decode_session = FakeDecodeSession(
        decode_contract, geometry, reference, **(decode_kwargs or {})
    )
    runner = OrtCpuParityRunner(
        prefill_session,
        decode_session,
        contract_prefill=prefill_contract,
        contract_decode=decode_contract,
        reference=reference,
        tolerance=tolerance,
    )
    return runner, prefill_session, decode_session


def dirty_prefill_reserve(
    geometry: Geometry,
    name: str = "key_cache.0",
    *,
    value: float = 0.5,
) -> dict[str, Any]:
    """A prefill cache override whose reserved tail is not exactly zero."""

    prefill_contract, _ = reduced_contracts(geometry)
    spec = next(item for item in prefill_contract.outputs if item.name == name)
    flat = prefill_cache_values(spec, geometry.prompt_length)
    # Head 0, the first reserved cache position, first element of the row.
    flat[geometry.prompt_length * geometry.head_dim] = value
    return {name: PlainTensor(tuple(flat), spec.shape, "float16")}


def cache_violations(evidence: Any) -> list[Any]:
    violations = [
        violation
        for step in evidence.cache_report.steps
        for violation in step.violations
    ]
    violations.extend(evidence.cache_report.slot_immutability_violations)
    return violations


# ---------------------------------------------------------------------------
# Pure-python metrics.
# ---------------------------------------------------------------------------


def test_metrics_match_hand_computed_values() -> None:
    reference = [1.0, 2.0, 3.0, 4.0, 10.0]
    candidate = [1.0, 2.5, 3.0, 3.0, 10.0]

    metrics = compare_logits(reference, candidate)

    assert metrics.max_absolute_error == pytest.approx(1.0)
    assert metrics.mean_absolute_error == pytest.approx(1.5 / 5)
    # Denominator is max(|reference|, floor=1.0): 0.5/2 and 1.0/4 both give 0.25.
    assert metrics.max_protected_relative_error == pytest.approx(0.25)
    expected_cosine = 127.0 / ((130.0**0.5) * (125.25**0.5))
    assert metrics.cosine_similarity == pytest.approx(expected_cosine)
    assert metrics.top1_reference == 4
    assert metrics.top1_candidate == 4
    assert metrics.top1_agreement is True
    assert metrics.top5_overlap == pytest.approx(1.0)
    assert metrics.reference_top1_top2_margin == pytest.approx(6.0)
    # Under the derived default, 1.0 <= atol 1.15 + 0.02 * 3.0, so `allclose`
    # holds and the *cosine* is what fails: 0.99529 < cosine_min 0.9993. Both
    # are asserted so this test cannot go green for an unintended reason if a
    # threshold moves again.
    assert metrics.allclose is True
    assert metrics.cosine_similarity < DEFAULT_ORT_CPU_TOLERANCE.cosine_min
    assert metrics.passed is False

    # The `allclose` convention itself -- rtol scales the CANDIDATE, not the
    # reference -- pinned against an explicit tolerance so it does not depend
    # on the default. The worst pair is reference 4.0 against candidate 3.0:
    # 1.0 > 0.5 + 0.1 * 3.0 = 0.8 fails, while 1.0 <= 0.5 + 0.1 * 4.0 = 0.9
    # would have passed had rtol scaled the reference.
    strict = dataclasses.replace(
        DEFAULT_ORT_CPU_TOLERANCE, atol=0.5, rtol=0.1, cosine_min=0.0
    )
    strict_metrics = compare_logits(reference, candidate, strict)
    assert strict_metrics.allclose is False
    assert strict_metrics.passed is False


def test_protected_relative_error_is_floored() -> None:
    metrics = compare_logits([0.1, 0.0], [0.6, 0.0])

    # Unfloored this would be 5.0; the floor of 1.0 keeps near-zero logits
    # from dominating, exactly as in T11.
    assert metrics.max_protected_relative_error == pytest.approx(0.5)


def test_top1_ties_break_to_the_lowest_index() -> None:
    metrics = compare_logits([3.0, 3.0, 1.0], [3.0, 3.0, 1.0])

    assert metrics.top1_reference == 0
    assert metrics.top1_candidate == 0
    assert top_indices([3.0, 3.0, 1.0], 2) == [0, 1]
    assert top_indices([1.0, 5.0, 5.0, 5.0], 3) == [1, 2, 3]


def test_top5_overlap_is_a_fraction_of_the_reference_top_five() -> None:
    reference = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0]
    candidate = [10.0, 9.0, 8.0, 7.0, 1.0, 5.0]

    metrics = compare_logits(reference, candidate)

    assert metrics.top5_overlap == pytest.approx(0.8)


def test_cosine_denominator_is_floored_and_does_not_zero_out() -> None:
    """The denominator is floored at 1e-8; a degenerate pair is not defined away."""

    # ||ref|| * ||cand|| = 1e-10, below the 1e-8 floor, so the denominator
    # becomes 1e-8 and the result is dot/1e-8 = 1e-10/1e-8 = 0.01.
    degenerate = compare_logits([1e-5, 0.0], [1e-5, 0.0])
    assert degenerate.cosine_similarity == pytest.approx(0.01)

    # An all-zero vector still gives zero, because the dot product is zero.
    assert compare_logits([0.0, 0.0], [1.0, 2.0]).cosine_similarity == 0.0

    # Above the floor nothing changes.
    ordinary = compare_logits([3.0, 4.0], [3.0, 4.0])
    assert ordinary.cosine_similarity == pytest.approx(1.0)


def test_cosine_similarity_is_not_clamped_to_one() -> None:
    """`F.cosine_similarity` clamps its denominator, never its output.

    Clamping the result to [-1, 1] here would be a silent divergence from
    `slm_lab.generation.reference.compare_logits`, which calls the real torch
    function, so the tiny float64 overshoot on identical vectors is kept. This
    test fails if the clamp comes back.
    """

    identical = compare_logits([1.0, 5.0], [1.0, 5.0])

    assert identical.cosine_similarity > 1.0
    assert identical.cosine_similarity == pytest.approx(1.0, abs=1e-12)
    # A lower-bound threshold cannot be masked by an above-one value.
    assert identical.passed is True


def test_the_cosine_floor_applies_to_the_norm_product_not_each_norm() -> None:
    """This module's floor is on the norm *product*, pinned on a degenerate input.

    The norms are 1e-12 and 1e+6, whose product 1e-6 is above the 1e-8 floor,
    so nothing is clamped and the vectors are recognized as parallel. Flooring
    each norm separately would divide by 1e-8 * 1e+6 and report 1e-4 instead,
    so this input is the one class that tells the two conventions apart, and
    this test fails if the module ever switches.

    It does *not* pin agreement with or divergence from torch.
    `F.cosine_similarity`'s published formula is the per-norm one, but its
    implementation is reported to clamp the product of the squared norms, in
    which case it agrees with this module here. That is unverified: no host in
    this task has torch. It changes no published number either way, because the
    conventions agree for every norm at or above the floor.
    """

    metrics = compare_logits([1e-12, 0.0], [1e6, 0.0])

    assert metrics.cosine_similarity == pytest.approx(1.0)
    assert metrics.cosine_similarity != pytest.approx(1e-4)


def test_reference_top1_top2_margin() -> None:
    metrics = compare_logits([1.0, 5.0, 3.0], [1.0, 5.0, 3.0])

    assert metrics.reference_top1_top2_margin == pytest.approx(2.0)


def test_allclose_boundary_matches_the_torch_convention() -> None:
    tolerance = ParityTolerance(
        atol=0.25,
        rtol=0.0,
        protected_relative_max=1.0,
        cosine_min=0.0,
        top5_overlap_min=0.0,
        require_top1=False,
    )

    at_threshold = compare_logits([1.25, 0.0], [1.0, 0.0], tolerance)
    beyond = compare_logits([1.2500001, 0.0], [1.0, 0.0], tolerance)

    assert at_threshold.allclose is True
    assert beyond.allclose is False


def test_rtol_scales_the_candidate_operand() -> None:
    tolerance = ParityTolerance(
        atol=0.0,
        rtol=0.5,
        protected_relative_max=10.0,
        cosine_min=0.0,
        top5_overlap_min=0.0,
        require_top1=False,
    )

    # |8 - 4| = 4 > 0.5 * |candidate=4| = 2
    assert compare_logits([8.0, 0.0], [4.0, 0.0], tolerance).allclose is False
    # |4 - 8| = 4 <= 0.5 * |candidate=8| = 4
    assert compare_logits([4.0, 0.0], [8.0, 0.0], tolerance).allclose is True


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_logits_raise(bad: float) -> None:
    with pytest.raises(ParityInputError, match="NaN or infinite"):
        compare_logits([1.0, bad], [1.0, 1.0])
    with pytest.raises(ParityInputError, match="NaN or infinite"):
        compare_logits([1.0, 1.0], [1.0, bad])


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ParityInputError, match="shape mismatch"):
        compare_logits([1.0, 2.0], [1.0])
    with pytest.raises(ParityInputError, match="batch size one"):
        compare_logits([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]])


def test_array_adapter_normalizes_and_rejects_ragged_input() -> None:
    tensor = PlainTensor((1.0, 2.0, 3.0, 4.0), (1, 2, 2), "float32")

    assert to_nested_list(tensor) == [[[1.0, 2.0], [3.0, 4.0]]]
    assert flatten(tensor) == [1.0, 2.0, 3.0, 4.0]
    assert onnx_cpu.shape_of(tensor) == (1, 2, 2)

    with pytest.raises(ParityInputError, match="ragged"):
        to_nested_list([[1.0, 2.0], [3.0]])
    with pytest.raises(ParityInputError, match="ragged"):
        to_nested_list([[1.0, 2.0], 3.0])
    with pytest.raises(ParityInputError, match="real numbers"):
        flatten([["a", "b"]])
    with pytest.raises(ParityInputError, match=r"nested list or expose \.tolist"):
        to_nested_list(object())


# ---------------------------------------------------------------------------
# Clean multi-step run.
# ---------------------------------------------------------------------------


def test_clean_multi_step_run_passes() -> None:
    runner, _, decode_session = build_runner(steps=4)

    evidence = runner.run(4)

    assert evidence.passed is True
    assert evidence.failure_kinds == ()
    assert evidence.evidence_tier == EvidenceTier.FAKE_SESSION_SELF_TEST.value
    assert decode_session.step == 4
    # One prefill report plus one report per decode step.
    assert [report.step for report in evidence.cache_report.steps] == [0, 1, 2, 3, 4]
    assert evidence.cache_report.passed is True
    assert [report.output_valid_length for report in evidence.cache_report.steps] == [
        PROMPT_LENGTH,
        PROMPT_LENGTH + 1,
        PROMPT_LENGTH + 2,
        PROMPT_LENGTH + 3,
        PROMPT_LENGTH + 4,
    ]
    assert [record.step for record in evidence.steps] == [0, 1, 2, 3, 4]
    assert all(record.metrics.passed for record in evidence.steps)
    assert evidence.steps[0].graph_kind == "prefill"
    assert evidence.steps[1].input_token_id is not None


def test_decode_feeds_thread_the_cache_and_build_masks() -> None:
    runner, _, decode_session = build_runner(steps=2)

    runner.run(2)

    first, second = decode_session.feeds
    assert first["valid_length"].values == (PROMPT_LENGTH,)
    assert second["valid_length"].values == (PROMPT_LENGTH + 1,)
    assert first["position_ids"].values == (PROMPT_LENGTH,)
    mask = first["attention_mask"].values
    assert len(mask) == CAPACITY
    assert set(mask[: PROMPT_LENGTH + 1]) == {1}
    assert set(mask[PROMPT_LENGTH + 1 :]) == {0}
    # Step two reads step one's output cache, not the prefill cache.
    assert second["key_cache.0"].values != first["key_cache.0"].values


# ---------------------------------------------------------------------------
# Cache-state faults with perfect logits.
# ---------------------------------------------------------------------------


def test_writing_the_previous_slot_is_a_state_fault_not_a_tolerance_fault() -> None:
    runner, _, _ = build_runner(steps=2, decode_kwargs={"write_offset": -1})

    evidence = runner.run(2)

    assert evidence.failure_kinds == (FailureKind.CACHE_STATE_UPDATE.value,)
    assert all(record.metrics.passed for record in evidence.steps)
    invariants = {violation.invariant for violation in cache_violations(evidence)}
    assert "prefix_preserved" in invariants
    assert "slot_written" in invariants
    prefix = next(
        violation
        for violation in cache_violations(evidence)
        if violation.invariant == "prefix_preserved"
    )
    assert prefix.layer == 0
    assert prefix.position == PROMPT_LENGTH - 1
    assert prefix.tensor.startswith("present_")


def test_prefix_corruption_at_step_three_is_caught_after_two_clean_steps() -> None:
    runner, _, _ = build_runner(
        steps=3,
        decode_kwargs={"corrupt_prefix_step": 3, "corrupt_prefix_position": 5},
    )

    evidence = runner.run(3)

    reports = {report.step: report for report in evidence.cache_report.steps}
    assert reports[1].passed is True
    assert reports[2].passed is True
    assert reports[3].passed is False
    assert evidence.failure_kinds == (FailureKind.CACHE_STATE_UPDATE.value,)
    assert all(record.metrics.passed for record in evidence.steps)
    violation = next(
        item for item in reports[3].violations if item.invariant == "prefix_preserved"
    )
    assert violation.tensor == "present_key.0"
    assert violation.layer == 0
    assert violation.position == 5


def test_missing_valid_length_increment_is_a_state_fault() -> None:
    runner, _, _ = build_runner(steps=2, decode_kwargs={"valid_length_increment": 0})

    evidence = runner.run(2)

    assert evidence.failure_kinds == (FailureKind.CACHE_STATE_UPDATE.value,)
    assert all(record.metrics.passed for record in evidence.steps)
    violation = next(
        item
        for item in cache_violations(evidence)
        if item.invariant == "valid_length_increment"
    )
    assert violation.tensor == "updated_valid_length"
    assert violation.position == PROMPT_LENGTH


def test_writing_past_the_current_slot_is_a_tail_fault() -> None:
    runner, _, _ = build_runner(steps=2, decode_kwargs={"tail_write_offset": 2})

    evidence = runner.run(2)

    assert evidence.failure_kinds == (FailureKind.CACHE_STATE_UPDATE.value,)
    assert all(record.metrics.passed for record in evidence.steps)
    violation = next(
        item
        for item in cache_violations(evidence)
        if item.invariant == "tail_untouched"
    )
    assert violation.layer == 0
    assert violation.position == PROMPT_LENGTH + 2


def test_a_slot_that_changes_later_breaks_written_slot_immutability() -> None:
    """The whole-run invariant: a written slot never changes again.

    ``write_offset=-1`` makes every step write the slot the *previous* step was
    supposed to own, so each step's recorded slot is overwritten by the next
    one. Three steps over four cache tensors leaves the final step's slot
    untouched, so 2 x 4 = 8 recorded slots have moved by the end of the run.
    """

    runner, _, _ = build_runner(steps=3, decode_kwargs={"write_offset": -1})

    evidence = runner.run(3)

    violations = evidence.cache_report.slot_immutability_violations
    assert len(violations) == 8
    assert {violation.invariant for violation in violations} == {
        "written_slot_immutable"
    }
    assert {violation.position for violation in violations} == {
        PROMPT_LENGTH,
        PROMPT_LENGTH + 1,
    }
    assert {violation.tensor for violation in violations} == {
        "present_key.0",
        "present_value.0",
        "present_key.1",
        "present_value.1",
    }
    assert {violation.layer for violation in violations} == {0, 1}
    # It is a whole-run finding, so it has no step and is reported separately
    # from the per-step reports.
    immutability = [
        failure
        for failure in evidence.failures
        if failure.step is None and "written_slot_immutable" in failure.detail
    ]
    assert len(immutability) == 8
    assert evidence.failure_kinds == (FailureKind.CACHE_STATE_UPDATE.value,)


def test_a_clean_run_leaves_every_written_slot_immutable() -> None:
    runner, _, _ = build_runner(steps=4)

    evidence = runner.run(4)

    assert evidence.cache_report.slot_immutability_violations == ()


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_a_non_finite_value_in_the_new_slot_is_a_state_fault(bad: float) -> None:
    runner, _, _ = build_runner(steps=1, decode_kwargs={"slot_fill": bad})

    evidence = runner.run(1)

    assert evidence.failure_kinds == (FailureKind.CACHE_STATE_UPDATE.value,)
    # The logits are untouched, so this is not a tolerance fault.
    assert all(record.metrics.passed for record in evidence.steps)
    violation = next(
        item for item in cache_violations(evidence) if item.invariant == "slot_finite"
    )
    assert violation.position == PROMPT_LENGTH
    assert violation.layer == 0
    assert violation.element == PROMPT_LENGTH * Geometry().head_dim


def test_a_dirty_prefill_reserve_is_a_state_fault() -> None:
    geometry = Geometry()
    runner, _, _ = build_runner(
        geometry,
        steps=1,
        prefill_kwargs={"output_overrides": dirty_prefill_reserve(geometry)},
    )

    evidence = runner.run(1)

    assert evidence.failure_kinds == (FailureKind.CACHE_STATE_UPDATE.value,)
    assert all(record.metrics.passed for record in evidence.steps)
    prefill_report = evidence.cache_report.steps[0]
    assert prefill_report.graph_kind == "prefill"
    violation = next(
        item
        for item in prefill_report.violations
        if item.invariant == "prefill_reserve_zero"
    )
    assert violation.tensor == "key_cache.0"
    assert violation.layer == 0
    assert violation.position == PROMPT_LENGTH
    assert violation.element == PROMPT_LENGTH * geometry.head_dim


def test_a_wrong_prefill_valid_length_is_a_state_fault() -> None:
    runner, _, _ = build_runner(
        steps=1,
        prefill_kwargs={
            "output_overrides": {
                "valid_length": PlainTensor((PROMPT_LENGTH - 1,), (1,), "int64")
            }
        },
    )

    evidence = runner.run(1)

    assert evidence.failure_kinds == (FailureKind.CACHE_STATE_UPDATE.value,)
    violation = next(
        item
        for item in evidence.cache_report.steps[0].violations
        if item.invariant == "prefill_valid_length"
    )
    assert violation.tensor == "valid_length"
    assert f"contract requires {PROMPT_LENGTH}" in violation.detail


def test_a_reported_valid_length_outside_capacity_is_caught_on_the_next_step() -> None:
    """`write_index_within_capacity` is reachable through `run()`.

    The runner threads the graph's own `updated_valid_length` rather than an
    internal counter, so a decode graph that reports a length outside the fixed
    capacity is caught the moment that length would be used as a write index.
    """

    runner, _, _ = build_runner(
        steps=2, decode_kwargs={"valid_length_override": CAPACITY}
    )

    evidence = runner.run(2)

    assert evidence.failure_kinds == (FailureKind.CACHE_STATE_UPDATE.value,)
    assert all(record.metrics.passed for record in evidence.steps)
    reports = {report.step: report for report in evidence.cache_report.steps}
    assert reports[1].passed is False
    violation = next(
        item
        for item in reports[2].violations
        if item.invariant == "write_index_within_capacity"
    )
    assert violation.position == CAPACITY
    assert f"outside capacity {CAPACITY}" in violation.detail
    # Every cache tensor short-circuits on the same fault, and nothing else.
    assert {item.invariant for item in reports[2].violations} == {
        "write_index_within_capacity",
        "valid_length_increment",
    }


# ---------------------------------------------------------------------------
# Classification independence.
# ---------------------------------------------------------------------------


def test_logit_fault_with_a_perfect_cache_is_a_tolerance_fault_only() -> None:
    runner, _, _ = build_runner(
        steps=3, decode_kwargs={"logit_bias": 5.0, "logit_bias_steps": (2,)}
    )

    evidence = runner.run(3)

    assert evidence.failure_kinds == (FailureKind.NUMERICAL_TOLERANCE.value,)
    assert evidence.cache_report.passed is True
    failing = [record.step for record in evidence.steps if not record.metrics.passed]
    assert failing == [2]


def test_simultaneous_faults_report_both_classes() -> None:
    runner, _, _ = build_runner(
        steps=3,
        decode_kwargs={
            "logit_bias": 5.0,
            "logit_bias_steps": (1, 2, 3),
            "write_offset": -1,
        },
    )

    evidence = runner.run(3)

    assert set(evidence.failure_kinds) == {
        FailureKind.NUMERICAL_TOLERANCE.value,
        FailureKind.CACHE_STATE_UPDATE.value,
    }
    # Neither class masked the other: both were computed for every step.
    assert evidence.cache_report.passed is False
    assert not any(record.metrics.passed for record in evidence.steps[1:])
    assert len(evidence.cache_report.steps) == 4
    # Ordering is part of the diagnosis: the state fault must be read before
    # the tolerance fault it also caused, or the operator retolerances a
    # correctness defect. Every cache failure precedes every tolerance one.
    kinds = [failure.kind for failure in evidence.failures]
    assert kinds[0] == FailureKind.CACHE_STATE_UPDATE.value
    last_state = max(
        index
        for index, kind in enumerate(kinds)
        if kind == FailureKind.CACHE_STATE_UPDATE.value
    )
    first_tolerance = min(
        index
        for index, kind in enumerate(kinds)
        if kind == FailureKind.NUMERICAL_TOLERANCE.value
    )
    assert last_state < first_tolerance
    message = _parity_failure_message(evidence)
    assert message.index("cache invariant violations") < message.index("failing steps")


# ---------------------------------------------------------------------------
# Non-finite logits.
# ---------------------------------------------------------------------------


def non_finite_logits(geometry: Geometry, bad: float, *, count: int = 0) -> PlainTensor:
    """Decode logits with `count` non-finite entries (all of them when 0)."""

    values = [float(index) for index in range(geometry.vocab)]
    for index in range(count or geometry.vocab):
        values[index] = bad
    return PlainTensor(tuple(values), (1, geometry.vocab), "float32")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_candidate_logits_are_classified_not_fatal(bad: float) -> None:
    """An FP16 export overflowing to Inf must be evidence, not a lost run.

    This is the logit-side counterpart of `slot_finite`: the run completes, the
    cache invariants are still checked on every step, and the failure is named
    `non_finite_logits` rather than being wrongly folded into
    `numerical_tolerance` or escaping as a configuration error.
    """

    geometry = Geometry()
    runner, _, _ = build_runner(
        geometry,
        steps=3,
        decode_kwargs={
            "output_overrides": {"next_logits": non_finite_logits(geometry, bad)}
        },
    )

    evidence = runner.run(3)

    assert evidence.failure_kinds == (FailureKind.NON_FINITE_LOGITS.value,)
    assert evidence.passed is False
    # The cache side is untouched and still fully checked: prefill plus three
    # decode steps, all clean.
    assert evidence.cache_report.passed is True
    assert len(evidence.cache_report.steps) == 4
    # Every decode step ran; none of them invented a metric.
    assert len(evidence.steps) == 4
    assert evidence.steps[0].metrics is not None
    for record in evidence.steps[1:]:
        assert record.metrics is None
        assert record.non_finite_candidate_logits == geometry.vocab
        assert record.candidate_logits_sha256
    failures = {failure.step: failure for failure in evidence.failures}
    assert set(failures) == {1, 2, 3}
    assert "no metric was computed" in failures[1].detail
    assert evidence.to_json()


def test_a_single_non_finite_candidate_logit_is_counted_exactly() -> None:
    geometry = Geometry()
    runner, _, _ = build_runner(
        geometry,
        steps=1,
        decode_kwargs={
            "output_overrides": {
                "next_logits": non_finite_logits(geometry, float("nan"), count=1)
            }
        },
    )

    evidence = runner.run(1)

    assert evidence.failure_kinds == (FailureKind.NON_FINITE_LOGITS.value,)
    assert evidence.steps[1].non_finite_candidate_logits == 1
    assert "1 NaN or infinite values" in evidence.failures[0].detail


def test_a_state_fault_is_reported_before_a_non_finite_logit_fault() -> None:
    geometry = Geometry()
    runner, _, _ = build_runner(
        geometry,
        steps=2,
        decode_kwargs={
            "write_offset": -1,
            "output_overrides": {
                "next_logits": non_finite_logits(geometry, float("inf"))
            },
        },
    )

    evidence = runner.run(2)

    assert set(evidence.failure_kinds) == {
        FailureKind.CACHE_STATE_UPDATE.value,
        FailureKind.NON_FINITE_LOGITS.value,
    }
    assert evidence.failures[0].kind == FailureKind.CACHE_STATE_UPDATE.value
    assert evidence.failures[-1].kind == FailureKind.NON_FINITE_LOGITS.value


def test_all_three_fault_classes_are_reported_in_the_documented_order() -> None:
    """The full ordering contract, in one run: state, then non-finite, then tolerance.

    `test_a_state_fault_is_reported_before_a_non_finite_logit_fault` and
    `test_simultaneous_faults_report_both_classes` each pin one adjacent pair
    and leave the `non_finite_logits` / `numerical_tolerance` pair unpinned,
    because `output_overrides` replaces an output on every step and so cannot
    make one step non-finite and another merely out of tolerance. The
    step-indexed `non_finite_steps` can, so this run produces all three classes
    and asserts the whole order, not two thirds of it.
    """

    runner, _, _ = build_runner(
        steps=2,
        decode_kwargs={
            "write_offset": -1,
            "non_finite_steps": (1,),
            "logit_bias": 5.0,
            "logit_bias_steps": (2,),
        },
    )

    evidence = runner.run(2)

    assert set(evidence.failure_kinds) == {
        FailureKind.CACHE_STATE_UPDATE.value,
        FailureKind.NON_FINITE_LOGITS.value,
        FailureKind.NUMERICAL_TOLERANCE.value,
    }
    # Step 1 produced no metric at all; step 2 produced one that failed. That is
    # what makes the two logit classes separable in a single run.
    assert evidence.steps[1].metrics is None
    assert evidence.steps[2].metrics is not None
    assert evidence.steps[2].metrics.passed is False

    kinds = [failure.kind for failure in evidence.failures]
    first = {kind: kinds.index(kind) for kind in set(kinds)}
    last = {kind: len(kinds) - 1 - kinds[::-1].index(kind) for kind in set(kinds)}
    assert kinds[0] == FailureKind.CACHE_STATE_UPDATE.value
    assert (
        last[FailureKind.CACHE_STATE_UPDATE.value]
        < first[FailureKind.NON_FINITE_LOGITS.value]
    )
    assert (
        last[FailureKind.NON_FINITE_LOGITS.value]
        < first[FailureKind.NUMERICAL_TOLERANCE.value]
    )
    # The two logit classes, in order, with the step each fired on.
    assert [
        (failure.kind, failure.step)
        for failure in evidence.failures
        if failure.kind != FailureKind.CACHE_STATE_UPDATE.value
    ] == [
        (FailureKind.NON_FINITE_LOGITS.value, 1),
        (FailureKind.NUMERICAL_TOLERANCE.value, 2),
    ]


def test_non_finite_reference_logits_stay_a_configuration_error() -> None:
    """The golden side is an input, not a measurement.

    A NaN in the reference means the fixture or the reference run is broken;
    there is no graph behaviour to classify, so this stays `ParityInputError`
    and the CLI still exits 2.
    """

    geometry = Geometry()
    reference = FakeReference(
        prompt=range(geometry.prompt_length), steps=1, vocab=geometry.vocab
    )
    reference._logits[1][0] = float("nan")  # noqa: SLF001 - fixture surgery
    prefill_contract, decode_contract = reduced_contracts(geometry)
    runner = OrtCpuParityRunner(
        FakePrefillSession(prefill_contract, geometry, reference),
        FakeDecodeSession(decode_contract, geometry, reference),
        contract_prefill=prefill_contract,
        contract_decode=decode_contract,
        reference=reference,
    )

    with pytest.raises(ParityInputError, match="reference logits for step 1"):
        runner.run(1)


# ---------------------------------------------------------------------------
# Contract violations.
# ---------------------------------------------------------------------------


def test_wrong_output_shape_is_a_contract_violation() -> None:
    geometry = Geometry()
    runner, _, _ = build_runner(
        geometry,
        steps=1,
        decode_kwargs={
            "output_overrides": {
                "present_key.0": PlainTensor((0.0,), (1, 1, 1, 1), "float16")
            }
        },
    )

    evidence = runner.run(1)

    assert evidence.failure_kinds == (FailureKind.CONTRACT_VIOLATION.value,)
    assert "expected shape" in evidence.failures[0].detail


def test_wrong_output_dtype_is_a_contract_violation() -> None:
    geometry = Geometry()
    _, decode_contract = reduced_contracts(geometry)
    spec = next(
        item for item in decode_contract.outputs if item.name == "present_value.1"
    )
    runner, _, _ = build_runner(
        geometry,
        steps=1,
        decode_kwargs={
            "output_overrides": {
                spec.name: PlainTensor(
                    (0.0,) * (spec.shape[1] * spec.shape[2] * spec.shape[3]),
                    spec.shape,
                    "float32",
                )
            }
        },
    )

    evidence = runner.run(1)

    assert evidence.failure_kinds == (FailureKind.CONTRACT_VIOLATION.value,)
    assert "expected dtype float16" in evidence.failures[0].detail


def test_missing_output_name_is_a_contract_violation() -> None:
    geometry = Geometry()
    _, decode_contract = reduced_contracts(geometry)
    renamed = [spec.name for spec in decode_contract.outputs]
    renamed[1] = "present_key.renamed"
    runner, _, _ = build_runner(
        geometry, steps=1, decode_kwargs={"declared_names": renamed}
    )

    evidence = runner.run(1)

    assert evidence.failure_kinds == (FailureKind.CONTRACT_VIOLATION.value,)
    assert "declares outputs" in evidence.failures[0].detail


def test_wrong_output_count_is_a_contract_violation() -> None:
    runner, _, _ = build_runner(steps=1, decode_kwargs={"drop_last_outputs": 1})

    evidence = runner.run(1)

    assert evidence.failure_kinds == (FailureKind.CONTRACT_VIOLATION.value,)
    assert "outputs, the T12 contract requires" in evidence.failures[0].detail


def test_session_run_failure_is_a_runtime_error() -> None:
    runner, _, decode_session = build_runner(steps=1)

    def explode(output_names, input_feed):
        raise ValueError("provider unavailable")

    decode_session.run = explode

    evidence = runner.run(1)

    assert evidence.failure_kinds == (FailureKind.RUNTIME_ERROR.value,)
    assert "provider unavailable" in evidence.failures[0].detail


# ---------------------------------------------------------------------------
# Evidence.
# ---------------------------------------------------------------------------


def test_evidence_json_is_deterministic_and_digest_is_sensitive() -> None:
    first = build_runner(steps=2)[0].run(2)
    second = build_runner(steps=2)[0].run(2)

    assert first.to_json() == second.to_json()
    assert first.evidence_sha256 == second.evidence_sha256
    payload = json.loads(first.to_json())
    assert payload["evidence_sha256"] == first.evidence_sha256
    assert payload["tolerance"]["status"].startswith("derived_and_measured")

    perturbed = build_runner(
        steps=2, decode_kwargs={"logit_bias": 0.01, "logit_bias_steps": (1,)}
    )[0].run(2)

    assert perturbed.evidence_sha256 != first.evidence_sha256


def test_runtime_record_carries_the_applied_session_settings() -> None:
    """The evidence records what each session reports, not what was requested."""

    class Options:
        graph_optimization_level = SimpleNamespace(name="ORT_DISABLE_ALL")
        execution_mode = SimpleNamespace(name="ORT_SEQUENTIAL")
        intra_op_num_threads = 1
        inter_op_num_threads = 1

    class ConfiguredSession:
        def __init__(self, level: str) -> None:
            self._options = Options()
            self._options.graph_optimization_level = SimpleNamespace(name=level)

        def get_session_options(self) -> Options:
            return self._options

        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

    record = onnx_cpu.runtime_record(
        {
            "prefill": ConfiguredSession("ORT_DISABLE_ALL"),
            "decode": ConfiguredSession("ORT_DISABLE_ALL"),
        }
    )

    assert record["session_settings"]["prefill"] == {
        "graph_optimization_level": "ORT_DISABLE_ALL",
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "execution_mode": "ORT_SEQUENTIAL",
    }
    assert record["providers"]["decode"] == ["CPUExecutionProvider"]

    # A session that exposes no options records null rather than a guess.
    assert onnx_cpu.applied_session_settings(object()) is None
    assert onnx_cpu.runtime_record({"prefill": object()})["session_settings"] == {
        "prefill": None
    }


def test_the_optimization_level_changes_the_evidence_digest() -> None:
    """Two runs differing only in session settings must be distinguishable."""

    class ConfiguredSession:
        def __init__(self, session: Any, level: str) -> None:
            self._session = session
            self._level = level

        def __getattr__(self, name: str) -> Any:
            return getattr(self._session, name)

        def get_session_options(self) -> SimpleNamespace:
            return SimpleNamespace(
                graph_optimization_level=SimpleNamespace(name=self._level),
                intra_op_num_threads=1,
                inter_op_num_threads=1,
            )

    def evidence_for(level: str) -> Any:
        geometry = Geometry()
        prefill_contract, decode_contract = reduced_contracts(geometry)
        reference = FakeReference(
            prompt=range(geometry.prompt_length), steps=1, vocab=geometry.vocab
        )
        runner = OrtCpuParityRunner(
            ConfiguredSession(
                FakePrefillSession(prefill_contract, geometry, reference), level
            ),
            ConfiguredSession(
                FakeDecodeSession(decode_contract, geometry, reference), level
            ),
            contract_prefill=prefill_contract,
            contract_decode=decode_contract,
            reference=reference,
        )
        return runner.run(1)

    disabled = evidence_for("ORT_DISABLE_ALL")
    enabled = evidence_for("ORT_ENABLE_ALL")

    assert disabled.passed is True
    assert enabled.passed is True
    assert (
        disabled.runtime["session_settings"]["decode"]["graph_optimization_level"]
        == "ORT_DISABLE_ALL"
    )
    assert (
        enabled.runtime["session_settings"]["decode"]["graph_optimization_level"]
        == "ORT_ENABLE_ALL"
    )
    assert disabled.evidence_sha256 != enabled.evidence_sha256


def test_an_unknown_evidence_tier_is_rejected_at_construction() -> None:
    evidence = build_runner(steps=1)[0].run(1)

    with pytest.raises(ParityInputError, match="unknown evidence tier"):
        dataclasses.replace(evidence, evidence_tier="real_onnxruntime_cpu_honest")
    # Every declared member is accepted.
    for tier in EvidenceTier:
        replaced = dataclasses.replace(evidence, evidence_tier=tier.value)
        assert replaced.evidence_tier == tier.value


def test_evidence_tier_cannot_be_forged(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("onnxruntime")

    class InferenceSession:
        pass

    module.InferenceSession = InferenceSession
    module.__version__ = "0.0.0-fake"
    monkeypatch.setitem(sys.modules, "onnxruntime", module)

    class Liar:
        @property
        def __class__(self):  # type: ignore[override]
            return InferenceSession

    liar = Liar()
    # A naive isinstance check is fooled; the tier check is not.
    assert isinstance(liar, InferenceSession)
    assert detect_evidence_tier([liar]) is EvidenceTier.FAKE_SESSION_SELF_TEST
    assert (
        detect_evidence_tier([InferenceSession(), InferenceSession()])
        is EvidenceTier.REAL_ONNXRUNTIME_CPU
    )
    assert (
        detect_evidence_tier([InferenceSession(), liar])
        is EvidenceTier.FAKE_SESSION_SELF_TEST
    )

    runner, _, _ = build_runner(steps=1)
    evidence = runner.run(1)
    assert evidence.evidence_tier == EvidenceTier.FAKE_SESSION_SELF_TEST.value

    # There is no caller-supplied route to a real tier.
    import inspect

    assert (
        "evidence_tier" not in inspect.signature(OrtCpuParityRunner.__init__).parameters
    )
    assert "evidence_tier" not in inspect.signature(OrtCpuParityRunner.run).parameters
    assert "evidence_tier" not in {action.dest for action in build_parser()._actions}


def test_runner_rejects_impossible_configurations() -> None:
    runner, _, _ = build_runner(steps=1)

    with pytest.raises(OnnxCpuError, match="at least one decode step"):
        runner.run(0)
    with pytest.raises(OnnxCpuError, match="overflow capacity"):
        runner.run(CAPACITY)


# ---------------------------------------------------------------------------
# Real T12 contract shapes.
# ---------------------------------------------------------------------------


def test_real_s128_contract_runs_end_to_end_with_fakes() -> None:
    geometry = REAL_GEOMETRY
    prefill_contract = build_prefill_contract(PROMPT_LENGTH)
    decode_contract = build_decode_contract(PROMPT_LENGTH)
    reference = FakeReference(
        prompt=range(PROMPT_LENGTH), steps=1, vocab=geometry.vocab
    )
    prefill_session = FakePrefillSession(prefill_contract, geometry, reference)
    decode_session = FakeDecodeSession(decode_contract, geometry, reference)
    runner = OrtCpuParityRunner(
        prefill_session,
        decode_session,
        contract_prefill=prefill_contract,
        contract_decode=decode_contract,
        reference=reference,
    )

    evidence = runner.run(1)

    assert evidence.passed is True
    assert evidence.variant_id == "S128"
    assert evidence.cache_capacity == CAPACITY
    assert evidence.cache_report.steps[1].tensors_checked == 56
    assert evidence.steps[0].metrics.top1_reference == geometry.vocab - 1


# ---------------------------------------------------------------------------
# Command line.
# ---------------------------------------------------------------------------


def write_manifest(
    tmp_path: Path,
    *,
    prefill_digest: str | None = None,
    decode_digest: str | None = None,
    create_files: bool = True,
) -> tuple[Path, Path]:
    directory = tmp_path / "onnx/reference/T20/S128"
    directory.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for kind in ("prefill", "decode"):
        path = directory / f"{kind}.onnx"
        if create_files:
            path.write_bytes(f"fake-{kind}-graph".encode("utf-8"))
            digests[kind] = onnx_cpu.sha256_file(path)
        else:
            digests[kind] = "0" * 64
    manifest = {
        "context_length": PROMPT_LENGTH,
        "cache_capacity": CAPACITY,
        "variant_id": "S128",
        "artifacts": {
            "prefill": {
                "relative_path": "S128/prefill.onnx",
                "sha256": prefill_digest or digests["prefill"],
            },
            "decode": {
                "relative_path": "S128/decode.onnx",
                "sha256": decode_digest or digests["decode"],
            },
        },
    }
    manifest_path = tmp_path / "S128.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, tmp_path


def test_build_parser_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.steps == 4
    assert args.reference == "torch"
    assert args.graph_optimization_level == "ORT_DISABLE_ALL"
    assert args.manifest.endswith("results/manifests/onnx/S128.json")


def test_cli_digest_mismatch_exits_without_creating_a_session(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path, root = write_manifest(tmp_path, prefill_digest="a" * 64)
    created: list[Path] = []

    def factory(path: Path) -> Any:
        created.append(path)
        raise AssertionError("a session must not be created after a digest mismatch")

    code = main(
        [
            "--manifest",
            str(manifest_path),
            "--artifact-root",
            str(root),
            "--steps",
            "1",
        ],
        session_factory=factory,
    )

    assert code == 2
    assert created == []
    assert "does not match the digest committed" in capsys.readouterr().err


@pytest.mark.parametrize(
    "relative", ["../escape.onnx", "/etc/passwd", "S128/../../escape.onnx", ""]
)
def test_manifest_relative_paths_cannot_escape_the_artifact_root(
    tmp_path: Path, relative: str
) -> None:
    """A manifest is untrusted input, exactly as it is on the inspection side."""

    manifest = {
        "context_length": PROMPT_LENGTH,
        "artifacts": {
            "prefill": {"relative_path": relative, "sha256": "0" * 64},
            "decode": {"relative_path": "S128/decode.onnx", "sha256": "0" * 64},
        },
    }

    with pytest.raises(OnnxCpuError, match="relative_path"):
        onnx_cpu.verified_graph_paths(manifest, tmp_path)

    # The safe form still joins under the given directory.
    assert onnx_cpu.safe_relative(tmp_path, "S128/prefill.onnx") == (
        tmp_path / "S128" / "prefill.onnx"
    )


def test_cli_missing_graph_file_exits_with_a_clear_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path, root = write_manifest(tmp_path, create_files=False)

    code = main(
        ["--manifest", str(manifest_path), "--artifact-root", str(root)],
        session_factory=lambda path: None,
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "missing prefill graph" in captured.err
    assert "Traceback" not in captured.err


def test_cli_without_onnxruntime_names_the_parity_environment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    if "onnxruntime" in sys.modules:  # pragma: no cover - environment dependent
        pytest.skip("onnxruntime is installed in this environment")
    manifest_path, root = write_manifest(tmp_path)

    code = main(["--manifest", str(manifest_path), "--artifact-root", str(root)])

    captured = capsys.readouterr()
    assert code == 2
    assert "onnxruntime is not installed" in captured.err
    assert "environments/onnx-cpu/README.md" in captured.err
    assert "Traceback" not in captured.err


def test_cli_writes_evidence_with_injected_fakes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, root = write_manifest(tmp_path)
    geometry = Geometry()
    prefill_contract, decode_contract = reduced_contracts(geometry)
    reference = FakeReference(
        prompt=range(PROMPT_LENGTH), steps=2, vocab=geometry.vocab
    )
    sessions = {
        "prefill.onnx": FakePrefillSession(prefill_contract, geometry, reference),
        "decode.onnx": FakeDecodeSession(decode_contract, geometry, reference),
    }
    output = tmp_path / "evidence/S128.json"

    monkeypatch.setattr(
        onnx_cpu, "build_prefill_contract", lambda length: prefill_contract
    )
    monkeypatch.setattr(
        onnx_cpu, "build_decode_contract", lambda length: decode_contract
    )
    code = main(
        [
            "--manifest",
            str(manifest_path),
            "--artifact-root",
            str(root),
            "--steps",
            "2",
            "--output",
            str(output),
        ],
        session_factory=lambda path: sessions[path.name],
        reference_factory=lambda document, steps: reference,
        tensor_factory=onnx_cpu.plain_tensor_factory,
    )

    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["evidence_tier"] == EvidenceTier.FAKE_SESSION_SELF_TEST.value
    assert payload["passed"] is True
    # One shape for `graph_digests`, and no host path inside a committed,
    # digest-covered record: the manifest-relative path identifies the graph.
    graph_digests = payload["graph_digests"]
    assert set(graph_digests) == {"prefill", "decode"}
    for kind, record in graph_digests.items():
        assert record == {
            "sha256": onnx_cpu.sha256_file(
                root / "onnx/reference/T20/S128" / f"{kind}.onnx"
            ),
            "relative_path": f"S128/{kind}.onnx",
        }
    assert str(root) not in json.dumps(payload)


def test_cli_exits_one_on_non_finite_logits_not_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 1 means "parity failed, read `failures[]`"; exit 2 means "misconfigured".

    NaN logits out of the graph are a parity failure with a named class, so the
    operator must be pointed at the evidence rather than at the configuration.
    """

    manifest_path, root = write_manifest(tmp_path)
    geometry = Geometry()
    prefill_contract, decode_contract = reduced_contracts(geometry)
    reference = FakeReference(
        prompt=range(PROMPT_LENGTH), steps=2, vocab=geometry.vocab
    )
    sessions = {
        "prefill.onnx": FakePrefillSession(prefill_contract, geometry, reference),
        "decode.onnx": FakeDecodeSession(
            decode_contract,
            geometry,
            reference,
            output_overrides={"next_logits": non_finite_logits(geometry, float("nan"))},
        ),
    }
    output = tmp_path / "evidence/S128-nan.json"

    monkeypatch.setattr(
        onnx_cpu, "build_prefill_contract", lambda length: prefill_contract
    )
    monkeypatch.setattr(
        onnx_cpu, "build_decode_contract", lambda length: decode_contract
    )
    code = main(
        [
            "--manifest",
            str(manifest_path),
            "--artifact-root",
            str(root),
            "--steps",
            "2",
            "--output",
            str(output),
        ],
        session_factory=lambda path: sessions[path.name],
        reference_factory=lambda document, steps: reference,
        tensor_factory=onnx_cpu.plain_tensor_factory,
    )

    assert code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["failure_kinds"] == [FailureKind.NON_FINITE_LOGITS.value]
    assert payload["cache_report"]["passed"] is True
    assert payload["steps"][1]["metrics"] is None
    assert payload["steps"][1]["non_finite_candidate_logits"] == geometry.vocab


# ---------------------------------------------------------------------------
# The tolerance derivation, and the diagnostic that checks it.
# ---------------------------------------------------------------------------


def test_the_tolerance_thresholds_agree_with_one_error_budget() -> None:
    """The four thresholds are one derivation, not four independent knobs.

    The superseded tolerance had `atol=0.25` alongside `cosine_min=0.999`. Those
    imply relative logit errors of 0.25/32 = 0.0078 and sqrt(2 * 1e-3) = 0.0447
    respectively -- a factor of 5.7 apart, so five of the six thresholds could
    never bind and nobody would notice. This pins them to a single budget so
    editing one in isolation fails here rather than silently degrading the
    others. If the derivation changes, change these numbers deliberately; do
    not tune a single threshold to make a measurement agree.
    """

    tolerance = DEFAULT_ORT_CPU_TOLERANCE
    lambda_max = 32.0  # binade ceiling above the measured max |logit| of 30.89

    # The budget every threshold is derived from: G_budget * u_eff.
    rho = tolerance.atol / lambda_max
    assert rho == pytest.approx(0.0359, rel=0.02)

    # 1 - cos ~= rho^2 / 2 for a rounding-noise perturbation.
    assert 1.0 - tolerance.cosine_min == pytest.approx(rho**2 / 2, rel=0.15)

    # max_protected_relative_error ~= 0.93 * max_absolute_error when
    # relative_floor is 1.0 and the logits have RMS ~4.5.
    assert tolerance.protected_relative_max == pytest.approx(
        0.93 * tolerance.atol, rel=0.1
    )

    # rtol covers only the magnitude-proportional term, the final logit
    # rounding, which is a small multiple of bfloat16's unit roundoff 2**-8.
    assert 2.0 * 2**-8 <= tolerance.rtol <= 8.0 * 2**-8

    # And the whole thing still has to be far below a mis-wired graph, whose
    # smallest measured signal is a one-slot cache offset at 13.29 absolute.
    assert tolerance.atol * 10 < 13.29
    assert tolerance.cosine_min > 0.99  # a mis-wired read measured 0.034..0.951

    # Nothing above may relax the state invariants.
    assert tolerance.cache_state == onnx_cpu.EXACT_CACHE_STATE_TOLERANCE
    assert all(tolerance.cache_state.as_dict().values())


def test_committed_diagnostics_show_the_tolerance_is_two_sided() -> None:
    """The committed evidence must accept float32 and reject a cache offset.

    This reads only committed JSON, so it runs everywhere, and it is the check
    that the tolerance change was a repair rather than a widening. The
    superseded `atol=0.25` failed the first half: at S512 step 1 the float32
    reference missed the bfloat16 reference by 0.609, so the old threshold
    rejected the exact answer. A tolerance that rejects the exact answer is
    measuring the reference's dtype, not the graph.

    If a future change makes either half false, the tolerance has stopped
    being a tolerance: too tight and it fails correct implementations, too
    loose and it stops catching a mis-wired cache read.
    """

    directory = ROOT / "results/graph/parity/diagnostics"
    records = sorted(directory.glob("S*-reference-dtype-self-error.json"))
    assert records, f"no committed self-error diagnostics under {directory}"

    for path in records:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["record_kind"] == "diagnostic_reference_dtype_self_error"
        verdict = payload["tolerance_verdict"]
        assert verdict["every_reference_dtype_pair_passes"] is True, path.name
        assert verdict["every_consecutive_step_pair_fails"] is True, path.name

        # And the recorded numbers, not just the rolled-up booleans: float32
        # against bfloat16 is the irreducible floor of this comparison.
        floor = max(
            record["max_absolute_error"]
            for record in payload["pairwise"]["float32_vs_bfloat16"]
        )
        assert floor < DEFAULT_ORT_CPU_TOLERANCE.atol, (
            f"{path.name}: the exact answer misses atol by {floor}"
        )
        weakest_miswire = min(
            record["max_absolute_error"]
            for record in payload["consecutive_step_distance"]
        )
        assert weakest_miswire > 10 * DEFAULT_ORT_CPU_TOLERANCE.atol, path.name


class DtypeFakeReference:
    """A reference whose logits depend on the requested storage dtype.

    Shaped like the real thing in the two ways the diagnostic reads: a coarser
    dtype wobbles the logits more (so it lands further from float32), and
    consecutive steps rotate the whole vector (so the mis-wiring proxy is far
    outside any tolerance). ``wobble`` alternates sign so it perturbs the
    vector's direction rather than merely rescaling it, which a cosine check
    would not see.
    """

    def __init__(self, dtype: str, *, steps: int, vocab: int, wobble: float) -> None:
        self.dtype = dtype
        self._logits = [
            [
                float((index + step) % vocab) + (wobble if index % 2 else -wobble)
                for index in range(vocab)
            ]
            for step in range(steps + 1)
        ]

    def prompt_token_ids(self) -> Sequence[int]:
        return (0, 1, 2)

    def next_logits(self, step: int) -> Sequence[float]:
        return self._logits[step]

    def expected_token_id(self, step: int) -> int:
        return top_indices(self._logits[step], 1)[0]

    def provenance(self) -> Mapping[str, Any]:
        return {"source": "DtypeFakeReference", "runtime": {"dtype": self.dtype}}


def test_reference_self_error_measures_the_reference_against_itself() -> None:
    """The derivation's empirical check must be re-runnable, so it is code."""

    # Roughly the real 8:1 ULP ratio between bfloat16 and float16.
    wobble = {"float32": 0.0, "bfloat16": 0.05, "float16": 0.006}
    report = onnx_cpu.reference_self_error(
        128,
        steps=2,
        reference_factory=lambda name: DtypeFakeReference(
            name, steps=2, vocab=8, wobble=wobble[name]
        ),
    )

    assert report["record_kind"] == "diagnostic_reference_dtype_self_error"
    # It must not be mistakable for a parity measurement.
    assert "evidence_tier" not in report
    assert "passed" not in report
    assert "graph_digests" not in report

    # Both orderings of every pair, once each, keyed left_vs_right.
    assert set(report["pairwise"]) == {
        "float32_vs_bfloat16",
        "float32_vs_float16",
        "bfloat16_vs_float16",
    }
    # The coarser dtype is further from float32, which is the whole point: the
    # reference's own storage dominates this comparison.
    assert (
        report["pairwise"]["float32_vs_bfloat16"][0]["max_absolute_error"]
        > report["pairwise"]["float32_vs_float16"][0]["max_absolute_error"]
    )
    # Lambda, the scale an absolute tolerance binds at.
    assert report["lambda_max_abs_logit"]["float32"][0] == pytest.approx(7.0)
    # The mis-wiring reference scale: consecutive steps, baseline dtype only.
    assert report["consecutive_step_distance_dtype"] == "float32"
    assert len(report["consecutive_step_distance"]) == 2
    assert report["reference_provenance"]["bfloat16"]["runtime"]["dtype"] == "bfloat16"

    # The rolled-up two-sided verdict, which
    # `test_committed_diagnostics_show_the_tolerance_is_two_sided` reads off the
    # committed records. Both halves are exercised here on fakes so the field is
    # covered without a parity host: dtype wobble stays inside the tolerance,
    # a rotated step does not.
    verdict = report["tolerance_verdict"]
    assert verdict["every_reference_dtype_pair_passes"] is True
    assert verdict["every_consecutive_step_pair_fails"] is True


def test_reference_self_error_requires_one_shared_greedy_token_path() -> None:
    """It is a control only while all three dtypes decode the same tokens.

    Each dtype teacher-forces its own greedy token, so if two of them chose
    differently at some step the later rows would compare two *different*
    continuations. The measured "dtype error" would then quietly include a
    whole extra token of context, which is the `consecutive_step_distance`
    scale — three orders of magnitude larger than the effect being measured,
    and in the direction that makes a tolerance look generous. The committed
    records do agree; this is what makes that a checked precondition rather
    than a lucky one.
    """

    class DivergentReference(DtypeFakeReference):
        def expected_token_id(self, step: int) -> int:
            return step + (1 if self.dtype == "float16" else 0)

    wobble = {"float32": 0.0, "bfloat16": 0.05, "float16": 0.006}
    with pytest.raises(OnnxCpuError) as excinfo:
        onnx_cpu.reference_self_error(
            128,
            steps=2,
            reference_factory=lambda name: DivergentReference(
                name, steps=2, vocab=8, wobble=wobble[name]
            ),
        )

    assert "greedy token path" in str(excinfo.value)


def test_cli_self_error_mode_never_builds_a_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diagnostic that touched a graph could be mistaken for a measurement."""

    manifest_path, root = write_manifest(tmp_path)
    output = tmp_path / "diagnostics/self-error.json"

    def explode(path: Path) -> Any:  # pragma: no cover - must never be called
        raise AssertionError(f"self-error mode built a session for {path}")

    monkeypatch.setattr(
        onnx_cpu,
        "reference_self_error",
        lambda context_length, *, steps: {
            "record_kind": "diagnostic_reference_dtype_self_error",
            "context_length": context_length,
            "steps_requested": steps,
        },
    )

    code = main(
        [
            "--manifest",
            str(manifest_path),
            "--artifact-root",
            str(root),
            "--steps",
            "2",
            "--reference-self-error",
            "--output",
            str(output),
        ],
        session_factory=explode,
    )

    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["record_kind"] == "diagnostic_reference_dtype_self_error"
    assert payload["context_length"] == PROMPT_LENGTH
    assert "evidence_tier" not in payload


def test_reference_dtype_defaults_to_the_contract_and_is_constrained() -> None:
    """The committed parity records are a bfloat16-reference comparison."""

    parser = build_parser()

    assert parser.parse_args(["--manifest", "x"]).reference_dtype is None
    assert (
        parser.parse_args(["--manifest", "x", "--reference-dtype", "float16"])
    ).reference_dtype == "float16"
    with pytest.raises(SystemExit):
        parser.parse_args(["--manifest", "x", "--reference-dtype", "int8"])


def test_record_kind_is_derived_from_the_reference_dtype() -> None:
    """A probe must not be structurally identical to a T21 record.

    ``evidence_tier`` cannot carry this: a ``--reference-dtype float16`` run
    creates real ONNX Runtime sessions, so it is honestly
    ``real_onnxruntime_cpu``, and it is honestly ``task_id="T21"`` and honestly
    ``passed`` — against a tolerance derived for a *bfloat16* reference, which
    is roughly 5.7x too loose for that pairing. Before ``record_kind`` the only
    differences were one nested provenance string and the file's name.
    """

    contract_dtype = onnx_cpu.contract_reference_dtype()
    assert contract_dtype == "bfloat16"

    parity = onnx_cpu.classify_record_kind(
        {"runtime": {"dtype": contract_dtype}}
    )
    assert parity is onnx_cpu.RecordKind.T21_ORT_CPU_PARITY

    probe = onnx_cpu.classify_record_kind({"runtime": {"dtype": "float16"}})
    assert probe is onnx_cpu.RecordKind.DIAGNOSTIC_OFF_CONTRACT_REFERENCE_DTYPE

    # A provenance block with no runtime dtype is the in-repository fake, which
    # `evidence_tier` already fences off.
    assert (
        onnx_cpu.classify_record_kind({})
        is onnx_cpu.RecordKind.T21_ORT_CPU_PARITY
    )


def test_record_kind_is_serialized_and_covered_by_the_digest() -> None:
    """A marker outside `evidence_sha256` could be edited off a record."""

    runner, _, _ = build_runner(steps=2)
    evidence = runner.run(2)

    payload = json.loads(evidence.to_json())
    assert payload["record_kind"] == "t21_ort_cpu_parity"

    relabelled = dataclasses.replace(
        evidence,
        record_kind=(
            onnx_cpu.RecordKind.DIAGNOSTIC_OFF_CONTRACT_REFERENCE_DTYPE.value
        ),
    ).with_digest()
    assert relabelled.evidence_sha256 != evidence.evidence_sha256

    with pytest.raises(ParityInputError):
        dataclasses.replace(evidence, record_kind="looks_official")


def test_a_probe_may_not_be_written_to_a_canonical_parity_name(
    tmp_path: Path,
) -> None:
    """Placement is a second claim, so the CLI refuses to let a probe make it.

    ``results/graph/parity/S<N>-ort-cpu.json`` is what the audit tool's glob
    and every citing document read as a T21 measurement. This fires before any
    session is created, so it costs a second rather than a whole run.
    """

    manifest_path, root = write_manifest(tmp_path)
    canonical = tmp_path / "S128-ort-cpu.json"

    def explode(path: Path) -> Any:  # pragma: no cover - must never be called
        raise AssertionError(f"a session was built for {path}")

    code = main(
        [
            "--manifest",
            str(manifest_path),
            "--artifact-root",
            str(root),
            "--steps",
            "1",
            "--reference-dtype",
            "float16",
            "--output",
            str(canonical),
        ],
        session_factory=explode,
    )

    assert code == 2
    assert not canonical.exists()

    # The contract dtype is allowed at that name, and a diagnostic name is
    # allowed at any dtype; neither raises here.
    onnx_cpu._refuse_off_contract_overwrite(
        onnx_cpu.contract_reference_dtype(), str(canonical)
    )
    onnx_cpu._refuse_off_contract_overwrite(
        "float16", str(tmp_path / "S128-ort-cpu-float16-reference-probe.json")
    )


def test_committed_probe_says_what_it_is() -> None:
    """The committed float16-reference probe carries the diagnostic kind."""

    path = (
        ROOT
        / "results/graph/parity/diagnostics"
        / "S128-ort-cpu-float16-reference-probe.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert (
        payload["record_kind"] == "diagnostic_off_contract_reference_dtype"
    ), "the committed probe must not look like a T21 record"
    assert payload["reference_provenance"]["runtime"]["dtype"] == "float16"

    for record in sorted((ROOT / "results/graph/parity").glob("S*-ort-cpu.json")):
        committed = json.loads(record.read_text(encoding="utf-8"))
        assert committed["record_kind"] == "t21_ort_cpu_parity", record.name


# ---------------------------------------------------------------------------
# Guarded real-runtime path.
# ---------------------------------------------------------------------------


def test_the_parity_failure_message_names_both_classes() -> None:
    """The guarded test's diagnostic must work; it only runs on a parity host."""

    runner, _, _ = build_runner(
        steps=2,
        decode_kwargs={
            "logit_bias": 5.0,
            "logit_bias_steps": (1, 2),
            "write_offset": -1,
        },
    )

    message = _parity_failure_message(runner.run(2))

    assert "numerical_tolerance" in message
    assert "cache_state_update" in message
    assert "max_absolute_error=5" in message
    assert "prefix_preserved on present_key.0" in message
    assert "derived_and_measured" in message


def test_real_onnxruntime_cpu_parity_when_available() -> None:
    """Skips cleanly without the runtime; measures parity where it exists.

    History, because this test's green is only meaningful with it. From T21
    until T23 this test could not run at all: `onnxruntime_cpu_session_factory()`
    defaults to ORT_DISABLE_ALL and the reference prefill graphs zero-extended
    the KV cache with a float16 `Pad`, for which the CPU provider registers no
    kernel, so session creation raised. T23 re-exported the prefill graphs with
    `Concat` and sessions now create. That alone did not make this test pass:
    against `DEFAULT_ORT_CPU_TOLERANCE` as T21 proposed it, all 20 measured
    steps failed `numerical_tolerance` -- `protected_relative_max=0.10` was
    exceeded on 20 of 20 by 1.7x-4.9x, and `atol=0.25` on 16 of 20 by up to
    2.3x.

    That failure was NOT the graph. Running the same PyTorch reference at
    float32 and comparing it to the same reference at bfloat16 -- no ONNX
    involved -- reproduces the disagreement almost exactly (S128 step 2: the
    graph misses by 0.4609, float32 misses by 0.4694; S512 step 1: the graph
    misses by 0.5781, float32 misses by 0.6090). The proposed `atol` rejected
    the *exact answer*, so it was measuring bfloat16's own quantization, not
    the graph. It was replaced by a budget derived from dtype ULP and residual
    depth; see the derivation above `DEFAULT_ORT_CPU_TOLERANCE`.

    So the assertions below pin four independent things, and a regression in
    any one of them means something different:

    * sessions create at ORT_DISABLE_ALL -- the float16 `Pad` defect is gone
      and has not come back through a re-export;
    * the evidence tier is real, so no fake session is being asserted about;
    * the cache report passes with zero violations -- the state path, which no
      tolerance may ever absorb;
    * no `numerical_tolerance` failure against the derived budget.

    If the last one ever fires, the correct response is NOT to widen a
    threshold. The derived budget is dominated by the bfloat16 reference (it
    supplies 99% of the error variance), so a numerical failure here means
    either the graph genuinely changed or the derivation is wrong. The
    discriminating measurement already exists and is cheap: rerun with
    `--reference-dtype float16`, which removes the reference's coarseness and
    is 5.7x tighter by the same derivation. The committed probe at
    results/graph/parity/diagnostics/S128-ort-cpu-float16-reference-probe.json
    records the graph clearing that tighter bound with 3x to spare.
    """

    pytest.importorskip("onnxruntime")
    pytest.importorskip("torch")

    manifest_path = ROOT / "results/manifests/onnx/S128.json"
    if not manifest_path.is_file():
        pytest.skip("no committed T20 S128 manifest")
    root = os.environ.get("SLM_LAB_ARTIFACT_ROOT")
    if not root or not Path(root).is_dir():
        pytest.skip("SLM_LAB_ARTIFACT_ROOT is not available")

    manifest = onnx_cpu.load_manifest(manifest_path)
    try:
        graphs = onnx_cpu.verified_graph_paths(manifest, Path(root))
    except OnnxCpuError as exc:
        pytest.skip(f"T20 graphs unavailable: {exc}")

    # No graph_optimization_level argument: the default is ORT_DISABLE_ALL, and
    # creating a session at that level is itself one of the things under test.
    # Do not "fix" a failure here by raising the level; that would hide exactly
    # the class of defect T20/T23 spent two tasks on.
    factory = onnx_cpu.onnxruntime_cpu_session_factory()
    prefill_session = factory(graphs["prefill"][0])
    decode_session = factory(graphs["decode"][0])
    for kind, session in (("prefill", prefill_session), ("decode", decode_session)):
        settings = onnx_cpu.applied_session_settings(session) or {}
        assert settings.get("graph_optimization_level") == "ORT_DISABLE_ALL", (
            f"{kind} session did not apply ORT_DISABLE_ALL: {settings}"
        )

    # dtype is left unset, so it resolves to the contract's reference_dtype,
    # bfloat16. The derived tolerance is derived for that pairing and for no
    # other; passing --reference-dtype here would change what is being measured.
    reference = onnx_cpu.TorchReferenceSource(
        onnx_cpu.load_context_workload_tokens(PROMPT_LENGTH), steps=2
    )
    runner = OrtCpuParityRunner(
        prefill_session,
        decode_session,
        contract_prefill=build_prefill_contract(PROMPT_LENGTH),
        contract_decode=build_decode_contract(PROMPT_LENGTH),
        reference=reference,
        tensor_factory=onnx_cpu.numpy_tensor_factory(),
        # The same payload builder the CLI uses, so this run's
        # `evidence_sha256` is comparable with a CLI run over the same graphs.
        graph_digests=onnx_cpu.graph_digests_payload(graphs),
    )

    evidence = runner.run(2)

    assert evidence.evidence_tier == EvidenceTier.REAL_ONNXRUNTIME_CPU.value
    assert evidence.tolerance is onnx_cpu.DEFAULT_ORT_CPU_TOLERANCE
    assert evidence.tolerance.as_dict()["status"].startswith("derived_and_measured")

    # The state path. Exactness, and no tolerance may ever absorb a violation
    # here; a cache fault also moves the logits, so this is asserted first so a
    # state defect is never read as a tolerance one.
    assert evidence.cache_report.passed, _parity_failure_message(evidence)
    assert not evidence.cache_report.slot_immutability_violations
    assert all(not step.violations for step in evidence.cache_report.steps), (
        _parity_failure_message(evidence)
    )
    assert all(record.non_finite_candidate_logits == 0 for record in evidence.steps), (
        _parity_failure_message(evidence)
    )

    # The acceptance criterion itself. Without this the one path that can
    # produce a real measurement would go green with logits arbitrarily far
    # outside the tolerance. `failure_kinds` is asserted alongside `passed` so
    # the message names the class rather than only the fact.
    assert evidence.failure_kinds == (), _parity_failure_message(evidence)
    assert evidence.passed, _parity_failure_message(evidence)


def _parity_failure_message(evidence: Any) -> str:
    """Say which failure class fired and how far the worst metric missed.

    An operator reading a red run on a parity host has to decide immediately
    whether this is a tolerance problem or a state problem; those have different
    fixes and the second must never be answered by retolerancing.
    """

    tolerance = evidence.tolerance
    non_finite = [
        f"step {record.step} ({record.graph_kind}): "
        f"{record.non_finite_candidate_logits} non-finite candidate logits, "
        "no metric computed"
        for record in evidence.steps
        if record.metrics is None
    ]
    worst = [
        (
            f"step {record.step} ({record.graph_kind}): "
            f"max_absolute_error={record.metrics.max_absolute_error:.6g} "
            f"(atol={tolerance.atol}), "
            f"max_protected_relative_error="
            f"{record.metrics.max_protected_relative_error:.6g} "
            f"(max={tolerance.protected_relative_max}), "
            f"cosine_similarity={record.metrics.cosine_similarity:.9g} "
            f"(min={tolerance.cosine_min}), "
            f"top5_overlap={record.metrics.top5_overlap:.6g} "
            f"(min={tolerance.top5_overlap_min}), "
            f"top1_agreement={record.metrics.top1_agreement}"
        )
        for record in evidence.steps
        if record.metrics is not None and not record.metrics.passed
    ]
    cache = [
        f"{violation.invariant} on {violation.tensor} at position {violation.position}"
        for step in evidence.cache_report.steps
        for violation in step.violations
    ]
    cache.extend(
        f"{violation.invariant} on {violation.tensor} at position {violation.position}"
        for violation in evidence.cache_report.slot_immutability_violations
    )
    return "\n".join(
        [
            f"evidence_sha256={evidence.evidence_sha256}",
            f"failure_kinds={list(evidence.failure_kinds)}",
            f"tolerance.status={tolerance.as_dict()['status']}",
            # Same order as `failures[]`: state first, then non-finite logits,
            # then tolerance. A cache fault moves the logits too, so reading
            # the tolerance line first invites retolerancing a state defect.
            f"cache invariant violations ({len(cache)}):",
            *(f"  {entry}" for entry in cache[:20]),
            "non-finite logit steps:",
            *(non_finite or ["  none"]),
            "failing steps:",
            *(worst or ["  none"]),
            "first failures:",
            *(
                f"  {failure.kind} step={failure.step}: {failure.detail}"
                for failure in evidence.failures[:10]
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Artifact-stage resolution.
#
# The runner reads its graph directory from the manifest's `artifacts.root`
# instead of a hard-coded `onnx/reference/T20`, so one implementation, one
# protocol and one tolerance can measure a later graph stage as well as the T20
# reference. Comparability is the whole point: a second parity implementation
# for a second stage would not produce evidence that could be set beside
# `results/graph/parity/`. That generalization is only safe if it is a no-op for
# everything already committed, which is what the first test checks against the
# committed manifests themselves.
# ---------------------------------------------------------------------------

COMMITTED_MANIFEST_DIRECTORY = ROOT / "results/manifests/onnx"
REFERENCE_ROOT_TEMPLATE = "${SLM_LAB_ARTIFACT_ROOT}/onnx/reference/T20"


def test_committed_manifests_resolve_to_the_hard_coded_reference_directory(
    tmp_path: Path,
) -> None:
    """Every committed manifest resolves exactly where the constant put it.

    `verified_graph_paths` used to join `ARTIFACT_SUBDIRECTORY` to the artifact
    root; it now expands the manifest's own `artifacts.root`. For every manifest
    already committed the two must name the same directory, or re-running the
    protocol would read different files than the records under
    `results/graph/parity/` were measured from -- and nothing in those records
    carries a host path that would reveal the substitution. This proves the
    equality on the committed files instead of asserting it in prose, against
    two unrelated roots so it cannot hold by coincidence of one mount point.
    """

    manifests = sorted(COMMITTED_MANIFEST_DIRECTORY.glob("S*.json"))
    assert [path.name for path in manifests] == [
        "S1024.json",
        "S128.json",
        "S4096.json",
        "S512.json",
    ]
    for manifest_path in manifests:
        artifacts = onnx_cpu.load_manifest(manifest_path)["artifacts"]
        assert artifacts["root"] == REFERENCE_ROOT_TEMPLATE, manifest_path.name
        for root in (tmp_path, Path("/Volumes/T9/slm-deployment-lab")):
            assert onnx_cpu.manifest_graph_directory(artifacts, root) == (
                root / onnx_cpu.ARTIFACT_SUBDIRECTORY
            ), manifest_path.name


def test_a_manifest_without_a_root_still_uses_the_reference_subdirectory(
    tmp_path: Path,
) -> None:
    """An older or hand-written manifest keeps resolving where it always did."""

    assert onnx_cpu.manifest_graph_directory({}, tmp_path) == (
        tmp_path / onnx_cpu.ARTIFACT_SUBDIRECTORY
    )


def test_a_manifest_root_selects_a_different_artifact_stage(tmp_path: Path) -> None:
    """A non-T20 stage is read through the same runner, not a second one."""

    artifacts = {"root": "${SLM_LAB_ARTIFACT_ROOT}/onnx/qnn-candidate/T22"}

    assert onnx_cpu.manifest_graph_directory(artifacts, tmp_path) == (
        tmp_path / "onnx/qnn-candidate/T22"
    )


@pytest.mark.parametrize("template", [None, "", 17, ["a"]])
def test_a_malformed_manifest_root_is_rejected(tmp_path: Path, template: Any) -> None:
    """`root` present but unusable is an error, not a silent T20 fallback.

    A null or empty template is a broken manifest. Falling back to the reference
    stage would answer it by quietly measuring different graphs than the ones
    the manifest's own digests pin.
    """

    with pytest.raises(OnnxCpuError, match="artifacts.root"):
        onnx_cpu.manifest_graph_directory({"root": template}, tmp_path)


def test_a_relative_manifest_root_is_rejected(tmp_path: Path) -> None:
    """The expansion has to land somewhere absolute, not on the working dir."""

    with pytest.raises(OnnxCpuError, match="absolute path"):
        onnx_cpu.manifest_graph_directory({"root": "onnx/reference/T20"}, tmp_path)


@pytest.mark.parametrize(
    "relative", ["../escape.onnx", "/etc/passwd", "S128/../../escape.onnx", ""]
)
def test_the_escape_guard_applies_to_a_manifest_resolved_directory(
    tmp_path: Path, relative: str
) -> None:
    """The `safe_relative` guard covers the expanded directory too.

    Resolving the directory from the manifest does not make the manifest
    trusted: the same file supplies both `root` and `relative_path`.
    """

    manifest = {
        "context_length": PROMPT_LENGTH,
        "artifacts": {
            "root": "${SLM_LAB_ARTIFACT_ROOT}/onnx/qnn-candidate/T22",
            "prefill": {"relative_path": relative, "sha256": "0" * 64},
            "decode": {"relative_path": "S128/decode.onnx", "sha256": "0" * 64},
        },
    }

    with pytest.raises(OnnxCpuError, match="relative_path"):
        onnx_cpu.verified_graph_paths(manifest, tmp_path)


def test_verified_graph_paths_reads_the_stage_the_manifest_names(
    tmp_path: Path,
) -> None:
    """The resolved path follows `artifacts.root`; the evidence path does not.

    `graph_digests_payload` records only the manifest-relative path, so two
    stages are told apart by their digests and by the manifest that pins them,
    never by a host path leaking into a committed record.
    """

    directory = tmp_path / "onnx/qnn-candidate/T22/S128"
    directory.mkdir(parents=True)
    digests: dict[str, str] = {}
    for kind in ("prefill", "decode"):
        path = directory / f"{kind}.onnx"
        path.write_bytes(f"fake-candidate-{kind}".encode("utf-8"))
        digests[kind] = onnx_cpu.sha256_file(path)
    manifest = {
        "artifacts": {
            "root": "${SLM_LAB_ARTIFACT_ROOT}/onnx/qnn-candidate/T22",
            "prefill": {
                "relative_path": "S128/prefill.onnx",
                "sha256": digests["prefill"],
            },
            "decode": {
                "relative_path": "S128/decode.onnx",
                "sha256": digests["decode"],
            },
        }
    }

    graphs = onnx_cpu.verified_graph_paths(manifest, tmp_path)

    assert graphs["prefill"][0] == directory / "prefill.onnx"
    assert graphs["decode"][0] == directory / "decode.onnx"
    payload = onnx_cpu.graph_digests_payload(graphs)
    assert payload == {
        "prefill": {
            "sha256": digests["prefill"],
            "relative_path": "S128/prefill.onnx",
        },
        "decode": {
            "sha256": digests["decode"],
            "relative_path": "S128/decode.onnx",
        },
    }
    assert str(tmp_path) not in json.dumps(payload)


def test_a_missing_graph_in_a_named_stage_still_says_how_to_get_it(
    tmp_path: Path,
) -> None:
    """The message names the missing graph and both ways to fix it."""

    manifest = {
        "artifacts": {
            "root": "${SLM_LAB_ARTIFACT_ROOT}/onnx/qnn-candidate/T22",
            "prefill": {"relative_path": "S128/prefill.onnx", "sha256": "0" * 64},
            "decode": {"relative_path": "S128/decode.onnx", "sha256": "0" * 64},
        }
    }

    with pytest.raises(OnnxCpuError) as caught:
        onnx_cpu.verified_graph_paths(manifest, tmp_path)

    message = str(caught.value)
    assert "missing prefill graph" in message
    assert str(tmp_path / "onnx/qnn-candidate/T22/S128/prefill.onnx") in message
    assert "artifacts.root" in message
    assert "--artifact-root" in message
