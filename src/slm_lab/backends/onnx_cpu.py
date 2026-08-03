"""ONNX Runtime CPU parity and multi-step static-cache validation (T21).

This module drives the eight static graphs exported by T20 through ONNX
Runtime on CPU and compares them against the deterministic T11 PyTorch
reference. It separates two failure modes that look similar in a log and have
completely different fixes:

* ``numerical_tolerance`` -- the graph is wired correctly, but FP16 storage and
  a different backend's accumulation order move the logits outside an explicit,
  written-down tolerance. The fix is precision or retolerancing work.
* ``cache_state_update`` -- the decode graph writes the wrong slot, loses the
  valid prefix, disturbs the reserved tail, or reports the wrong
  ``valid_length``. The fix is a corrected export.

A third numerical class sits beside them because it is a different diagnosis
again, not a wider tolerance:

* ``non_finite_logits`` -- the graph emitted NaN or Inf logits. Nothing is
  "outside tolerance" here; an FP16 export overflowed or divided by zero, and
  no metric computed from those values would mean anything. It is the logit-side
  counterpart of the cache-side ``slot_finite`` invariant.

Neither may mask the other: the logit metrics and the static-cache invariants
are computed on independent code paths, both always run, and the evidence
reports every failure class that fired. ``failures[]`` is ordered state faults
first, then non-finite logits, then tolerance failures, so the most fundamental
diagnosis is read first.

**The real measurement has now been taken.** This module was written on a host
with no ``onnxruntime``, ``torch``, or ``numpy``, so for its first two tasks
every number it could produce came from injected fake sessions labelled
``evidence_tier="fake_session_self_test"``, and the tolerances in
:data:`DEFAULT_ORT_CPU_TOLERANCE` were *proposed and unvalidated*. T23 promoted
the ``Concat`` prefill export, which made the prefill graphs loadable on the
CPU provider at ``ORT_DISABLE_ALL``, and all four contexts have since been
measured there against the PyTorch reference. The proposed thresholds were
replaced by a derived budget; see the derivation above
:data:`DEFAULT_ORT_CPU_TOLERANCE`, and :data:`TOLERANCE_STATUS` for what the
recorded runs did against it.

Everything a fake session could ever produce is still labelled as such. The
tier is derived from the session objects and no flag can supply it.

A machine that has the runtime, the reference model, and the T20 artifact root
produces the real measurement with the separate environment described in
``environments/onnx-cpu/README.md`` (the root ``uv.lock`` deliberately does not
carry ONNX Runtime, exactly as T50 keeps MLX out of it)::

    SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab \\
    HF_HOME=<local-hf-cache> TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \\
    python -m slm_lab.backends.onnx_cpu \\
      --manifest results/manifests/onnx/S128.json \\
      --steps 4 --reference torch \\
      --output results/graph/parity/S128-ort-cpu.json

That run, and only that run, is allowed to be recorded as
``evidence_tier="real_onnxruntime_cpu"``. Inside :meth:`OrtCpuParityRunner.run`
the tier is derived from the session objects themselves and there is no
parameter, flag, or environment variable that can supply it. Code that builds a
:class:`ParityEvidence` directly is not going through that derivation, so the
field is additionally validated against :class:`EvidenceTier` at construction:
an unknown tier is rejected, but a hand-built record carrying a valid tier
string is still not a measurement.

Nothing in this module imports ``onnxruntime``, ``torch``, ``numpy``, or
``onnx`` at import time, and none of them is needed for the unit tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from array import array
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from slm_lab.contracts.static_cache import (
    CONTEXT_VARIANTS,
    CacheContractError,
    GraphContract,
    TensorSpec,
    build_decode_contract,
    build_prefill_contract,
    validate_tensor_mapping,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_SYMLINK = PROJECT_ROOT / "artifacts"
ARTIFACT_SUBDIRECTORY = Path("onnx/reference/T20")
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "results/manifests/onnx/S128.json"
TASK_ID = "T21"
SCHEMA_VERSION = 1
ENVIRONMENT_GUIDE = "environments/onnx-cpu/README.md"
INSTALL_HINT = (
    "ONNX Runtime is not part of the locked root environment; build the "
    f"separate parity environment described in {ENVIRONMENT_GUIDE}"
)

CACHE_POSITION_AXIS = "cache_position"
PREFILL_CACHE_PREFIXES = ("key_cache", "value_cache")
# Decode output cache tensor prefix -> the incoming tensor it must extend.
DECODE_CACHE_PREFIX_SOURCES = {
    "present_key": "key_cache",
    "present_value": "value_cache",
}

REAL_MEASUREMENT_COMMAND = (
    "SLM_LAB_ARTIFACT_ROOT=<external-root> HF_HOME=<local-hf-cache> "
    "TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python -m slm_lab.backends.onnx_cpu "
    "--manifest results/manifests/onnx/S128.json --steps 4 --reference torch "
    "--output results/graph/parity/S128-ort-cpu.json"
)


class OnnxCpuError(RuntimeError):
    """A parity run cannot be configured or executed as specified."""


class OnnxCpuDependencyError(OnnxCpuError):
    """An optional runtime dependency for a real measurement is absent."""


class ParityInputError(OnnxCpuError):
    """A value handed to the pure-Python comparison layer is unusable."""


# ---------------------------------------------------------------------------
# Array adapter: no numpy anywhere in this module.
# ---------------------------------------------------------------------------


def _require_rectangular(node: list[Any], depth: int) -> None:
    """Reject ragged nesting without touching every scalar leaf."""

    if not node or not isinstance(node[0], list):
        # Leaf level. Scalar typing is enforced by ``array("d", ...)`` in
        # ``flatten``, which is a C-level check instead of N isinstance calls.
        return
    width = len(node[0])
    for index, child in enumerate(node):
        if not isinstance(child, list):
            raise ParityInputError(
                f"ragged array-like: depth {depth} entry {index} is a scalar "
                "beside a nested sequence"
            )
        if len(child) != width:
            raise ParityInputError(
                f"ragged array-like: depth {depth} entry {index} has length "
                f"{len(child)}, expected {width}"
            )
        _require_rectangular(child, depth + 1)


def to_nested_list(value: Any) -> list[Any]:
    """Normalize an array-like into rectangular nested Python lists.

    Accepts anything exposing ``.tolist()`` -- which covers numpy arrays as
    returned by ONNX Runtime -- plus plain nested lists and tuples.
    """

    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        nested = value
    else:
        converter = getattr(value, "tolist", None)
        if not callable(converter):
            raise ParityInputError(
                "array-like value must be a nested list or expose .tolist(), "
                f"found {type(value).__name__}"
            )
        nested = converter()
    if not isinstance(nested, list):
        raise ParityInputError(
            "array-like value must have at least one dimension, found a scalar"
        )
    _require_rectangular(nested, 0)
    return nested


def nested_shape(nested: list[Any]) -> tuple[int, ...]:
    """Return the shape of an already-rectangular nested list."""

    shape: list[int] = []
    node: Any = nested
    while isinstance(node, list):
        shape.append(len(node))
        node = node[0] if node else None
    return tuple(shape)


def shape_of(value: Any) -> tuple[int, ...]:
    """Return the shape of any accepted array-like."""

    return nested_shape(to_nested_list(value))


def _flatten_into(node: list[Any], flat: list[float]) -> None:
    if node and isinstance(node[0], list):
        for child in node:
            _flatten_into(child, flat)
        return
    try:
        flat.extend(array("d", node))
    except (TypeError, OverflowError) as exc:
        raise ParityInputError(
            f"array-like leaf values must be real numbers: {exc}"
        ) from exc


def flatten(value: Any) -> list[float]:
    """Return every scalar of an array-like as a flat list of floats."""

    flat: list[float] = []
    _flatten_into(to_nested_list(value), flat)
    return flat


def _scalar_int(value: Any, label: str) -> int:
    """Read a single integer out of a ``[1]``-shaped array-like."""

    nested = to_nested_list(value)
    flat: list[Any] = []
    node: Any = nested
    while isinstance(node, list):
        if len(node) != 1:
            raise ParityInputError(f"{label} must contain exactly one element")
        flat = node
        node = node[0]
    scalar = flat[0]
    if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
        raise ParityInputError(f"{label} must be an integer, found {scalar!r}")
    if isinstance(scalar, float) and not scalar.is_integer():
        raise ParityInputError(f"{label} must be an integer, found {scalar!r}")
    return int(scalar)


def values_sha256(values: Sequence[float]) -> str:
    """Digest a float vector as little-endian float64 for change detection."""

    buffer = array("d", values)
    if sys.byteorder != "little":  # pragma: no cover - little-endian hosts only
        buffer.byteswap()
    return hashlib.sha256(buffer.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Injected surfaces.
# ---------------------------------------------------------------------------


@runtime_checkable
class InferenceSessionLike(Protocol):
    """The minimal ONNX Runtime session surface this module depends on."""

    def get_inputs(self) -> Sequence[Any]: ...

    def get_outputs(self) -> Sequence[Any]: ...

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, Any],
    ) -> Sequence[Any]: ...


SessionFactory = Callable[[Path], InferenceSessionLike]
TensorFactory = Callable[[Sequence[Any], Sequence[int], str], Any]


@dataclass(frozen=True)
class PlainTensor:
    """Dependency-free stand-in for a numpy array at a graph boundary."""

    values: tuple[Any, ...]
    shape: tuple[int, ...]
    dtype: str

    def tolist(self) -> list[Any]:
        nested: list[Any] = list(self.values)
        for size in reversed(self.shape[1:]):
            nested = [
                nested[start : start + size] for start in range(0, len(nested), size)
            ]
        return nested


def plain_tensor_factory(
    values: Sequence[Any],
    shape: Sequence[int],
    dtype: str,
) -> PlainTensor:
    """Build a :class:`PlainTensor`; the default for fake-session self-tests."""

    shape = tuple(int(dimension) for dimension in shape)
    expected = 1
    for dimension in shape:
        expected *= dimension
    flat = tuple(values)
    if len(flat) != expected:
        raise ParityInputError(f"cannot reshape {len(flat)} values into {shape}")
    return PlainTensor(values=flat, shape=shape, dtype=dtype)


def numpy_tensor_factory() -> TensorFactory:
    """Return a factory building real numpy arrays for ONNX Runtime feeds."""

    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - requires the extra
        raise OnnxCpuDependencyError(
            "numpy is required to feed ONNX Runtime. Install the optional "
            f"extra with: {INSTALL_HINT}"
        ) from exc

    def factory(values: Sequence[Any], shape: Sequence[int], dtype: str) -> Any:
        return numpy.asarray(values, dtype=dtype).reshape(tuple(shape))

    return factory


def _require_onnxruntime() -> Any:
    try:
        import onnxruntime
    except ImportError as exc:
        raise OnnxCpuDependencyError(
            "onnxruntime is not installed, so no real ONNX Runtime CPU "
            f"measurement is possible. Install the optional extra with: "
            f"{INSTALL_HINT}"
        ) from exc
    return onnxruntime


#: Session options that change the numbers a run produces, so the evidence has
#: to carry them. Read back off the constructed session, never from the request.
SESSION_SETTING_FIELDS = (
    "graph_optimization_level",
    "intra_op_num_threads",
    "inter_op_num_threads",
    "execution_mode",
)


def _setting_value(value: Any) -> Any:
    """Normalize one session-option value into something JSON can carry."""

    name = getattr(value, "name", None)
    if isinstance(name, str):  # GraphOptimizationLevel / ExecutionMode member
        return name
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return str(value)
    return value


def applied_session_settings(session: Any) -> dict[str, Any] | None:
    """Read back the options a session was actually constructed with.

    This deliberately interrogates the *session*, not the factory's arguments.
    ONNX Runtime is free to normalize or override what it was handed, and a
    run's evidence must describe the configuration that executed. ``None`` means
    the session exposes no options at all, which is the case for every fake.
    """

    getter = getattr(session, "get_session_options", None)
    if not callable(getter):
        return None
    try:
        options = getter()
    except Exception:  # noqa: BLE001 - absence of settings is not a run failure
        return None
    settings: dict[str, Any] = {}
    for name in SESSION_SETTING_FIELDS:
        if not hasattr(options, name):
            continue
        settings[name] = _setting_value(getattr(options, name))
    return settings or None


def onnxruntime_cpu_session_factory(
    *,
    intra_op_num_threads: int = 1,
    inter_op_num_threads: int = 1,
    graph_optimization_level: str = "ORT_DISABLE_ALL",
) -> SessionFactory:
    """Build sessions pinned to a deterministic single-threaded CPU provider.

    Optimizations default to ``ORT_DISABLE_ALL`` so the first parity number
    measures the exported graph rather than ONNX Runtime's fusion choices; a
    second run with ``ORT_ENABLE_ALL`` then isolates the fusion delta.

    The applied configuration is not taken on trust from these arguments:
    :func:`applied_session_settings` reads it back off each constructed session
    and :func:`runtime_record` writes it into the evidence under
    ``runtime.session_settings``, so two runs that differ only in optimization
    level produce different ``evidence_sha256`` values.
    """

    onnxruntime = _require_onnxruntime()
    levels = onnxruntime.GraphOptimizationLevel
    try:
        level = getattr(levels, graph_optimization_level)
    except AttributeError as exc:  # pragma: no cover - requires the extra
        raise OnnxCpuError(
            f"unknown graph optimization level {graph_optimization_level!r}"
        ) from exc

    def factory(path: Path) -> InferenceSessionLike:  # pragma: no cover
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = intra_op_num_threads
        options.inter_op_num_threads = inter_op_num_threads
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = level
        return onnxruntime.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    return factory


@runtime_checkable
class ReferenceSource(Protocol):
    """Golden values the ONNX graphs are measured against."""

    def prompt_token_ids(self) -> Sequence[int]: ...

    def next_logits(self, step: int) -> Sequence[float]: ...

    def expected_token_id(self, step: int) -> int: ...

    def provenance(self) -> Mapping[str, Any]: ...


# ---------------------------------------------------------------------------
# Tolerances.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheStateTolerance:
    """Static-cache criteria. These are exactness rules, not closeness rules.

    A KV cache slot is copied, not recomputed, so any difference in a region
    the contract says is untouched is a wiring defect. Loosening these would
    let a genuine state bug hide inside FP16 noise, which is exactly the
    confusion T21 exists to prevent.
    """

    require_prefix_identical: bool = True
    require_tail_identical: bool = True
    require_slot_written: bool = True
    require_valid_length_increment: bool = True
    require_finite_slot: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "require_prefix_identical": self.require_prefix_identical,
            "require_tail_identical": self.require_tail_identical,
            "require_slot_written": self.require_slot_written,
            "require_valid_length_increment": self.require_valid_length_increment,
            "require_finite_slot": self.require_finite_slot,
        }


EXACT_CACHE_STATE_TOLERANCE = CacheStateTolerance()


@dataclass(frozen=True)
class ParityTolerance:
    """Frozen criteria for BF16-PyTorch versus FP16-ONNX-Runtime parity.

    Field semantics are identical to ``slm_lab.generation.reference``'s
    ``NumericalTolerance`` so a T21 number and a T11 number can be compared
    directly.
    """

    atol: float
    rtol: float
    protected_relative_max: float
    cosine_min: float
    top5_overlap_min: float
    require_top1: bool = True
    relative_floor: float = 1.0
    cache_state: CacheStateTolerance = EXACT_CACHE_STATE_TOLERANCE

    def __post_init__(self) -> None:
        if self.atol < 0 or self.rtol < 0:
            raise ParityInputError(
                "absolute and relative tolerances must be non-negative"
            )
        if self.protected_relative_max < 0 or self.relative_floor <= 0:
            raise ParityInputError("protected-relative parameters must be positive")
        if not 0 <= self.cosine_min <= 1:
            raise ParityInputError("cosine_min must be between zero and one")
        if not 0 <= self.top5_overlap_min <= 1:
            raise ParityInputError("top5_overlap_min must be between zero and one")

    def as_dict(self) -> dict[str, Any]:
        return {
            "atol": self.atol,
            "rtol": self.rtol,
            "protected_relative_max": self.protected_relative_max,
            "cosine_min": self.cosine_min,
            "top5_overlap_min": self.top5_overlap_min,
            "require_top1": self.require_top1,
            "relative_floor": self.relative_floor,
            "cache_state": self.cache_state.as_dict(),
            "status": TOLERANCE_STATUS,
        }


TOLERANCE_STATUS = (
    "derived_and_measured: the proposed thresholds were replaced by a budget "
    "derived from bfloat16-reference and float16-candidate ULP at the measured "
    "logit scale and from 28-layer residual depth, with no observed candidate "
    "error used to set any threshold; measured on onnxruntime 1.28.0 CPU EP at "
    "ORT_DISABLE_ALL over four contexts x five steps, which pass "
    "(results/graph/parity/S*-ort-cpu.json, 2026-08-02)"
)

# DERIVED FROM DTYPE AND DEPTH, THEN MEASURED. Read this before touching a
# number in it; the numbers below are consequences, not settings.
#
# T21 wrote these thresholds without a runtime and marked them
# `proposed_unvalidated`. T23 promoted the `Concat` prefill export, which made
# the prefill graphs loadable at ORT_DISABLE_ALL, and the first real
# measurement -- four contexts, five steps each -- failed on all 20 steps:
# `protected_relative_max=0.10` exceeded on 20 of 20 by 1.7x-4.9x, `atol=0.25`
# on 16 of 20 by up to 2.3x. This block is the re-derivation that measurement
# forced. It is a derivation, not a fit: no observed candidate error appears
# anywhere in it.
#
# ---------------------------------------------------------------------------
# 0. The measurement that makes replacing the threshold legitimate
# ---------------------------------------------------------------------------
#
# Widening a threshold because a measurement missed it is exactly the move this
# task's acceptance criteria forbid, so the justification cannot be "the graph
# is probably fine". It is this, and it is measured:
#
#     RUN THE SAME PYTORCH REFERENCE AT FLOAT32 AND COMPARE IT TO ITSELF AT
#     BFLOAT16. NO ONNX ANYWHERE. IT MISSES atol=0.25 BY THE SAME MARGIN THE
#     GRAPH DOES, AND ON THE WORST STEP IT MISSES BY MORE.
#
#     context/step   graph (fp16) vs bf16 ref   float32 vs bf16 ref
#     S128 step 0                   0.343750               0.344912
#     S128 step 2                   0.460938               0.469389
#     S512 step 0                   0.312500               0.313089
#     S512 step 1                   0.578125               0.608955   <--
#     S512 step 2                   0.189453               0.189295
#
# The two columns agree to about 2%. At S512 step 1 -- the single worst step in
# the whole committed set, the one that missed atol=0.25 by 2.3x -- the EXACT
# ANSWER misses the bfloat16 reference by 0.609, more than the graph's 0.578.
#
# A threshold that rejects float32 is not measuring the graph; it is measuring
# bfloat16's own quantization error and calling it a defect. So atol=0.25 was
# not a tolerance the graph failed. It was a mis-specified instrument that
# every possible implementation fails, including a bit-exact one. Replacing it
# is a repair to the instrument, and the sections below derive what it should
# have been from dtype and depth alone.
#
# Evidence: results/graph/parity/diagnostics/S*-reference-dtype-self-error.json
#
# ---------------------------------------------------------------------------
# 1. What is actually being compared, and which side is coarser
# ---------------------------------------------------------------------------
#
# Candidate: the float16 ONNX graph on the ORT CPU provider.
# Reference: the pinned PyTorch model in *bfloat16* (configs/models/
# qwen3-0.6b.yaml, `reference_dtype: bfloat16`).
#
#     float16   11 significand bits (10 stored + 1 implicit), u16 = 2^-11
#                                                                  = 4.883e-4
#     bfloat16   8 significand bits ( 7 stored + 1 implicit), ubf = 2^-8
#                                                                  = 3.906e-3
#     ubf / u16 = 2^3 = 8
#
# THE REFERENCE IS EIGHT TIMES COARSER THAN THE CANDIDATE. The superseded
# derivation reasoned from "FP16 spacing near 20 is about 0.016" and never
# accounted for that, so it sized the budget from the *finer* side of its own
# comparison. That single omission is worth a factor of 8 before any other
# consideration.
#
# ---------------------------------------------------------------------------
# 2. ULP at the logit scale these models actually produce
# ---------------------------------------------------------------------------
#
# Lambda := max |next-token logit|, measured at float32 over the committed T10
# workloads, five steps each (see "how to reproduce" at the end):
#
#     S128    22.32 .. 25.05        S1024   27.73 .. 30.89
#     S512    19.25 .. 30.26        S4096   27.04 .. 30.06
#
# All 20 lie in the binade [16, 32), where the exponent is 4:
#
#     ULP_fp16(x in [16,32)) = 2^(4-10) = 2^-6 = 0.015625
#     ULP_bf16(x in [16,32)) = 2^(4- 7) = 2^-3 = 0.125
#
# A representation floor follows with no modelling at all. Even if the two
# pipelines computed the *identical real number*, one records it on a 0.125
# grid and the other on a 0.015625 grid:
#
#     floor = 0.125/2 + 0.015625/2 = 0.0703 at Lambda ~ 25
#
# The fp16-only reading of the same bound is 0.0156 -- 4.5x too small.
#
# ---------------------------------------------------------------------------
# 3. Depth: what 28 layers do to one rounding
# ---------------------------------------------------------------------------
#
# The logits are not one rounding. Count the roundings on the signal path that
# reach the output with unit relative gain, for Qwen3-0.6B (28 layers, hidden
# 1024, vocab 151936):
#
#     embedding store                                              1
#     per layer: attention output, residual store,
#                MLP output, residual store          4 x 28  =   112
#     final RMSNorm, lm_head output                                2
#                                                              -----
#     N                                                          115
#
# Round-to-nearest is bounded by u but its RMS is smaller. For x = m * 2^e with
# m log-uniform on [1,2), the absolute error is uniform on +/- ulp/2, so the
# relative error has RMS
#
#     c = sqrt( E[1/m^2] / 3 ),   E[1/m^2] = (1/ln 2) * Int_1^2 m^-3 dm
#                                          = (1/ln 2) * 3/8 = 0.5411
#     c = sqrt(0.5411/3) = 0.4247   (in units of u)
#
# Treating the N roundings as independent and zero-mean, the relative
# perturbation of the final hidden state -- and therefore of the logits, since
# the lm_head is linear -- is
#
#     G = c * sqrt(N) = 0.4247 * sqrt(115) = 0.4247 * 10.724 = 4.55  ULP
#
# Counting only the 2 * 28 = 56 residual stores instead gives
# G = 0.4247 * 7.48 = 3.18, so the counting convention brackets G in
# [3.18, 4.55], a factor of 1.43.
#
# This is checkable without any ONNX graph, and it was checked: running the
# same PyTorch reference at float32, bfloat16 and float16 on the same prompts
# and reading G = rho/u off the cosine gives, over all 20 committed self-error
# steps,
#
#     bfloat16 pipeline   G = 2.08 .. 5.72,  mean 3.57
#     float16  pipeline   G = 1.99 .. 15.04, mean 4.29
#     combined            G = 2.10 .. 5.93,  mean 3.59
#
# The measured mean, 3.59, sits inside the analytic bracket [3.18, 4.55] and
# nearer its conservative end, so the N=115 count is the generous reading and
# N=56 the tight one. That comparison never touches the candidate, so using it
# to check G is not a fit to the quantity under test.
#
# The float16 outlier is real and worth naming: at S512 step 1 the float16
# pipeline excursions to G = 15.0 while bfloat16 stays at 5.7 on the same step.
# It does not propagate, because section 5 shows the *combined* figure is what
# the budget uses and bfloat16 dominates it; the same step's combined G is
# 5.93. But it is why the margin below is sized on the combined spread and not
# on either pipeline alone.
#
# ---------------------------------------------------------------------------
# 4. The geometry that turns a relative hidden-state error into atol
# ---------------------------------------------------------------------------
#
# delta_z = W_lm * delta_h. The rounding-noise vector delta_h is uncorrelated
# in direction with h, and both z_v = <w_v, h> and delta_z_v = <w_v, delta_h>
# are projections of a 1024-dim vector onto the same 151936 rows. They inherit
# the same extreme-value factor over the vocabulary, and it cancels:
#
#     max_v |delta_z_v| ~= (|delta_h| / |h|) * max_v |z_v| = G * u * Lambda
#
# So atol scales with the MAX logit, not the RMS logit, and needs no separate
# tail factor. Two consequences worth stating, because they are structural:
#
# (a) delta_z_v does NOT scale with z_v. The per-element error is roughly the
#     same size everywhere in the vocabulary. Only the final rounding of the
#     logit itself is magnitude-proportional; that is what `rtol` is for, and
#     it is ~1-2 ULP, not a percentage of the logit.
# (b) max_protected_relative_error is therefore NOT independent of atol. With
#     relative_floor = 1.0 and logits of measured RMS 2.19 .. 6.05 (mean 4.74),
#     roughly a fifth of the vocabulary sits below the floor, and the max of
#     |delta_z| over that subset is
#     sqrt(2 ln 0.19V) / sqrt(2 ln V) = 4.55/4.89 = 0.93 of its max over the
#     whole vocabulary. So
#
#         max_protected_relative_error ~= 0.93 * max_absolute_error
#
#     The 20 committed steps show that ratio at 0.67 .. 0.98, mean 0.84.
#     The superseded pair (atol 0.25, protected_relative_max 0.10) asserted a
#     ratio of 0.40, which this geometry says cannot occur: the two thresholds
#     were not two checks, they were one check stated twice, 2.5x apart. The
#     relative one always fired first and never carried information.
#
# ---------------------------------------------------------------------------
# 5. The budget
# ---------------------------------------------------------------------------
#
# The two pipelines round independently, so their errors add in quadrature:
#
#     u_eff = sqrt(ubf^2 + u16^2) = ubf * sqrt(1 + 1/64) = ubf * 1.0078
#           = 3.936e-3
#
# The float16 candidate contributes 0.78% of the amplitude and 1.5% of the
# variance. TO WITHIN A PERCENT, THIS IS A TOLERANCE ON BFLOAT16. That is the
# central fact about this comparison and the reason the superseded number was
# wrong by the factor it was.
#
# Margin. G is an RMS over a distribution, and a threshold that fires on half
# of a healthy model's steps is not a threshold. Two stated uncertainties:
#
#     step-to-step spread of the combined G, 5.93 over a mean 3.59  1.65x
#     counting convention for N, G in [3.18, 4.55]                  1.43x
#     combined, in quadrature                                       2.18x
#
# Rounded DOWN to 2.0, which is the tighter direction; a margin rounded down
# cannot be an accommodation. So G_budget = 2 * 4.55 = 9.11 ULP. Both factors
# are properties of the reference model measured against float32; neither is
# headroom over an observed candidate error.
#
# Lambda for the threshold: the measured maximum is 30.89, and the binade
# ceiling 32 is the largest logit for which the ULP figures in section 2 hold.
# Using 32 states the tolerance's domain of validity rather than pinning it to
# one workload's peak.
#
#     atol = G_budget * u_eff * Lambda = 9.11 * 3.936e-3 * 32 = 1.147
#
# Where that lands, stated plainly rather than buried: 1.89x the largest
# irreducible floor (float32 vs bfloat16, 0.609) and 1.99x the largest measured
# candidate error (0.578). Those two are nearly the same number because they
# are nearly the same quantity -- see section 0. A tolerance sitting at twice
# the error it must not fire on is a tight one, not a generous one, and if the
# derivation had come out below either figure the right answer would have been
# to record the failure. It would have, at Lambda <= 16 (one binade lower), or
# with the N=56 count and no margin (0.399), or against a float16 reference
# (0.201 -- see section 7).
#
# ---------------------------------------------------------------------------
# 6. Does it still fail loudly on a mis-wired graph?
# ---------------------------------------------------------------------------
#
# This is the question a widened tolerance has to answer. A cache read that
# lands one slot off makes the model attend to a shifted context, so the
# distance between consecutive decode steps' logits is a direct proxy for it.
# Measured on the float32 reference over all four contexts, 16 step pairs:
#
#     max_absolute_error   13.29 .. 30.44   (healthy: 0.19 .. 0.58)
#     cosine_similarity     0.034 ..  0.951  (healthy: 0.99976 .. 0.99997)
#     top5_overlap          0.0   ..  0.6    (healthy: 0.8 .. 1.0)
#     top1_agreement        false on all 16
#
# Against atol = 1.15 that is an 11.6x margin at the WEAKEST observed
# mis-wiring signal (13.29), and at that same weakest signal cosine catches the
# fault with (1 - cos) = 0.049, seventy times its 7e-4 threshold. Every one of
# the four logit criteria fires on every one of those comparisons. The
# tolerance's purpose survives.
#
# Note which way round the guards work. atol is the loosest of the three
# because the reference dtype forces it to be; the direction check and the
# argmax check are what actually make a state defect unmissable, and neither
# was loosened here.
#
# ---------------------------------------------------------------------------
# 7. What this does NOT license
# ---------------------------------------------------------------------------
#
# A tolerance of ~1.15 on logits running -22.6 .. +30.9 is loose, and it is
# loose because the *reference* is bfloat16. The measurement that decides
# whether the graph is faithful is float16-vs-float16, where the same
# derivation gives
#
#     atol_fp16ref = 2 * 4.55 * sqrt(2) * u16 * 32 = 0.201
#
# a 5.7x tighter bound. Measured on S128: max absolute error 0.031 .. 0.066,
# against 0.297 .. 0.461 on the same steps with the bfloat16 reference. That is
# 6.9x to 9.8x tighter -- more than the 5.7x the ULP ratio alone predicts,
# because two float16 pipelines make partly correlated rounding errors -- and it
# clears the 0.201 bound with 3x to spare. The graph is faithful; the gap was
# the reference dtype.
#
# That probe is committed as a diagnostic under
# results/graph/parity/diagnostics/ and is NOT a T21 parity record. Tightening
# this comparison is not possible without changing the reference dtype, which
# is a T21 contract decision and out of scope here.
#
# ---------------------------------------------------------------------------
# Per-threshold summary
# ---------------------------------------------------------------------------
#
# atol=1.15                       REPLACED (was 0.25)
#     G_budget * u_eff * Lambda = 9.11 * 3.936e-3 * 32 = 1.147, rounded down.
# rtol=0.02                       CONFIRMED
#     `allclose` uses torch's convention |ref - cand| <= atol + rtol * |cand|,
#     so rtol covers the magnitude-proportional part of the error. Section 4(a)
#     says that part is the final logit rounding alone, ~1-2 ULP; with the same
#     2x margin, 4 * ubf = 0.0156. The existing 0.02 sits just above that and
#     is kept unchanged.
# protected_relative_max=1.05     REPLACED (was 0.10)
#     0.93 * atol = 1.07 by section 4(b), rounded down. It is a restatement of
#     atol at this logit distribution, not an independent check; it is retained
#     only because relative_floor=1.0 keeps T21 numbers comparable with T11's.
# cosine_min=0.9993               REPLACED, TIGHTENED (was 0.999)
#     1 - cos ~= rho^2 / 2 with rho = G_budget * u_eff = 0.0359, giving
#     cos >= 0.99936. Rounded to 0.9993. The superseded 0.999 implied
#     rho <= 0.0447, inconsistent with an atol that implied rho <= 0.0078;
#     the same budget now sets both.
# top5_overlap_min=0.8            CONFIRMED
#     Per-logit noise has std ~ rho * RMS(z) = 0.0359 * 4.74 = 0.17, while the
#     gap between the 5th and 6th logit is ~ RMS(z) / (5 * sqrt(2 ln V)) =
#     0.19. A rank-5/6 swap is therefore expected and a rank-4 loss is a 2-
#     sigma event: "at most one of five may move" is the derived value. One of
#     the 20 measured steps does sit at 0.8, and every other one at 1.0.
# require_top1=True               CONFIRMED, with a caveat now on record
#     Greedy decoding is only reproducible if argmax agrees. The same noise std
#     of 0.17 means a reference top1-top2 margin below ~0.5 makes agreement a
#     coin flip. The measured margins run 0.5 .. 12.9 and top1 held on all 20
#     steps, but a future disagreement at a margin under ~0.5 is a tolerance
#     question, not a wiring one.
# relative_floor=1.0              UNCHANGED
#     Deliberately identical to T11 so a T21 number and a T11 number compare
#     directly. Raising it would make protected_relative_max informative and
#     break that, which is a contract change and not this task's to make.
# cache_state=EXACT_CACHE_STATE_TOLERANCE   UNCHANGED
#     Cache regions the contract calls untouched are compared for exact
#     equality, never closeness. See CacheStateTolerance. Nothing above
#     loosens this, and nothing may.
#
# ---------------------------------------------------------------------------
# How to reproduce the inputs
# ---------------------------------------------------------------------------
#
# Lambda, G, and the mis-wiring proxy come from running the *reference alone*
# at three dtypes -- no ONNX graph involved:
#
#     python -m slm_lab.backends.onnx_cpu --reference-self-error \
#       --manifest results/manifests/onnx/S<N>.json --steps 4 \
#       --output results/graph/parity/diagnostics/S<N>-reference-dtype-self-error.json
#
# The float16-reference parity probe of section 7 is
#
#     python -m slm_lab.backends.onnx_cpu --reference-dtype float16 \
#       --manifest results/manifests/onnx/S128.json --steps 4 --reference torch \
#       --output results/graph/parity/diagnostics/S128-ort-cpu-float16-reference-probe.json
#
# Both write to results/graph/parity/diagnostics/ and are diagnostics, not T21
# parity records. The float16-reference probe carries
# reference_provenance.runtime.dtype = "float16", which is what distinguishes
# it from the committed S<N>-ort-cpu.json measurements.
DEFAULT_ORT_CPU_TOLERANCE = ParityTolerance(
    atol=1.15,
    rtol=0.02,
    protected_relative_max=1.05,
    cosine_min=0.9993,
    top5_overlap_min=0.8,
    require_top1=True,
    relative_floor=1.0,
    cache_state=EXACT_CACHE_STATE_TOLERANCE,
)


# ---------------------------------------------------------------------------
# Pure-python logit metrics.
# ---------------------------------------------------------------------------

#: Lower clamp on the cosine denominator, matching the default ``eps`` of
#: ``torch.nn.functional.cosine_similarity``.
COSINE_DENOMINATOR_FLOOR = 1e-8


@dataclass(frozen=True)
class LogitParityMetrics:
    """Mirrors ``slm_lab.generation.reference.LogitMetrics`` field for field."""

    max_absolute_error: float
    mean_absolute_error: float
    max_protected_relative_error: float
    cosine_similarity: float
    top1_reference: int
    top1_candidate: int
    top1_agreement: bool
    top5_overlap: float
    reference_top1_top2_margin: float
    allclose: bool
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        margin = self.reference_top1_top2_margin
        return {
            "max_absolute_error": self.max_absolute_error,
            "mean_absolute_error": self.mean_absolute_error,
            "max_protected_relative_error": self.max_protected_relative_error,
            "cosine_similarity": self.cosine_similarity,
            "top1_reference": self.top1_reference,
            "top1_candidate": self.top1_candidate,
            "top1_agreement": self.top1_agreement,
            "top5_overlap": self.top5_overlap,
            "reference_top1_top2_margin": (margin if math.isfinite(margin) else None),
            "allclose": self.allclose,
            "passed": self.passed,
        }


def _logit_vector(value: Any, label: str) -> list[float]:
    nested = to_nested_list(value)
    shape = nested_shape(nested)
    if len(shape) == 2:
        if shape[0] != 1:
            raise ParityInputError(
                f"{label} logits must have batch size one, found {shape[0]}"
            )
        nested = nested[0]
        shape = shape[1:]
    if len(shape) != 1 or shape[0] < 1:
        raise ParityInputError(
            f"{label} logits must be a non-empty [vocabulary] or "
            f"[1, vocabulary] array, found shape {shape}"
        )
    return flatten(nested)


def non_finite_count(values: Sequence[float]) -> int:
    """How many of ``values`` are NaN or infinite."""

    return sum(1 for value in values if not math.isfinite(value))


def top_indices(values: Sequence[float], k: int) -> list[int]:
    """Indices of the ``k`` largest values, ties broken to the lowest index.

    This matches ``torch.argmax``'s first-maximum rule and the T10 fixture
    convention that the lowest token ID wins a tie.
    """

    if k <= 0:
        return []
    best: list[tuple[float, int]] = []
    worst = float("-inf")
    for index, value in enumerate(values):
        if len(best) == k and value <= worst:
            continue
        position = 0
        while position < len(best) and best[position][0] >= value:
            position += 1
        best.insert(position, (value, index))
        if len(best) > k:
            best.pop()
        worst = best[-1][0]
    return [index for _, index in best]


def compare_logits(
    reference: Any,
    candidate: Any,
    tolerance: ParityTolerance = DEFAULT_ORT_CPU_TOLERANCE,
) -> LogitParityMetrics:
    """Measure PyTorch-reference versus ONNX-Runtime next-token parity.

    Every definition matches ``slm_lab.generation.reference.compare_logits``
    so the two numbers are directly comparable, but this implementation is
    pure Python and needs neither torch nor numpy. ``allclose`` uses torch's
    convention ``|reference - candidate| <= atol + rtol * |candidate|``: rtol
    scales the *candidate* (the ``other`` operand of ``torch.allclose``).

    Two cosine conventions are stated exactly at the computation below: the
    *product* of the norms is floored at :data:`COSINE_DENOMINATOR_FLOOR`
    rather than each norm separately, and the result is not clamped to
    ``[-1, 1]`` because ``torch.nn.functional.cosine_similarity`` does not
    clamp its output. Whether the first of those differs from torch is
    unverified here; see the comment for what is and is not known.

    Non-finite input is refused rather than measured, on either side, exactly
    as the T11 reference refuses it. A non-finite *candidate* is a real ONNX
    Runtime failure mode rather than a bad call, so
    :class:`OrtCpuParityRunner` screens for it before calling this function and
    records it as :attr:`FailureKind.NON_FINITE_LOGITS`; only a direct caller
    ever sees the exception.
    """

    ref = _logit_vector(reference, "reference")
    cand = _logit_vector(candidate, "candidate")
    if len(ref) != len(cand):
        raise ParityInputError(
            f"logit shape mismatch: reference has {len(ref)} values, "
            f"candidate has {len(cand)}"
        )

    floor = tolerance.relative_floor
    atol = tolerance.atol
    rtol = tolerance.rtol
    max_absolute = 0.0
    total_absolute = 0.0
    max_relative = 0.0
    dot = 0.0
    reference_square = 0.0
    candidate_square = 0.0
    allclose = True
    for value, other in zip(ref, cand, strict=True):
        if not math.isfinite(value) or not math.isfinite(other):
            raise ParityInputError("logits contain NaN or infinite values")
        difference = value - other
        if difference < 0.0:
            difference = -difference
        total_absolute += difference
        if difference > max_absolute:
            max_absolute = difference
        magnitude = value if value >= 0.0 else -value
        denominator = magnitude if magnitude > floor else floor
        relative = difference / denominator
        if relative > max_relative:
            max_relative = relative
        if allclose:
            scale = other if other >= 0.0 else -other
            if difference > atol + rtol * scale:
                allclose = False
        dot += value * other
        reference_square += value * value
        candidate_square += other * other

    # Denominator convention, stated exactly because it is the kind of detail a
    # reproduction gets wrong:
    #
    # * The product of the two norms is clamped from below at
    #   COSINE_DENOMINATOR_FLOOR, so a degenerate pair yields dot/1e-8 (zero
    #   when either vector is all-zero) rather than a defined-away 0.0.
    #   ``test_the_cosine_floor_applies_to_the_norm_product_not_each_norm``
    #   pins that convention on the one input class that can distinguish it:
    #   one norm far below the floor while the other is far above it.
    #   ``torch.nn.functional.cosine_similarity`` also floors its denominator
    #   at its default ``eps``. Its *published formula* writes that floor
    #   per norm -- ``max(||x1||, eps) * max(||x2||, eps)`` -- which would
    #   differ from this module on exactly that degenerate input; its
    #   *implementation* is reported to clamp the product of the squared norms
    #   instead, which would agree with this module everywhere. Which of the
    #   two torch actually does is UNVERIFIED here: no host in this task has
    #   torch installed, so no comparison has been run. It does not affect any
    #   published number, because the two conventions agree for every norm at
    #   or above the floor, which is every non-degenerate logit vector. Treat
    #   the per-norm form as documentation, not as measured torch behaviour,
    #   and settle it on a host with torch before claiming a deviation.
    # * The result is NOT clamped to [-1, 1]. torch does not clamp its output
    #   either: `F.cosine_similarity` can return a value a few ULPs above 1.0
    #   for identical vectors, and clamping here would silently diverge from
    #   `slm_lab.generation.reference.compare_logits`, which calls the real
    #   torch function. `cosine_min` is a lower bound, so an above-one value
    #   cannot mask a failure.
    norm = math.sqrt(reference_square) * math.sqrt(candidate_square)
    cosine = dot / (
        norm if norm > COSINE_DENOMINATOR_FLOOR else COSINE_DENOMINATOR_FLOOR
    )

    width = min(5, len(ref))
    reference_top = top_indices(ref, width)
    candidate_top = top_indices(cand, width)
    top5_overlap = len(set(reference_top) & set(candidate_top)) / len(reference_top)
    top1_reference = reference_top[0]
    top1_candidate = candidate_top[0]
    top1_agreement = top1_reference == top1_candidate
    margin = (
        float("inf")
        if len(ref) < 2
        else ref[reference_top[0]] - ref[top_indices(ref, 2)[1]]
    )

    passed = (
        allclose
        and max_relative <= tolerance.protected_relative_max
        and cosine >= tolerance.cosine_min
        and top5_overlap >= tolerance.top5_overlap_min
        and (top1_agreement or not tolerance.require_top1)
    )
    return LogitParityMetrics(
        max_absolute_error=max_absolute,
        mean_absolute_error=total_absolute / len(ref),
        max_protected_relative_error=max_relative,
        cosine_similarity=cosine,
        top1_reference=top1_reference,
        top1_candidate=top1_candidate,
        top1_agreement=top1_agreement,
        top5_overlap=top5_overlap,
        reference_top1_top2_margin=margin,
        allclose=allclose,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Static-cache state validation.
# ---------------------------------------------------------------------------


class FailureKind(str, Enum):
    """Every failure class this runner distinguishes."""

    NUMERICAL_TOLERANCE = "numerical_tolerance"
    NON_FINITE_LOGITS = "non_finite_logits"
    CACHE_STATE_UPDATE = "cache_state_update"
    CONTRACT_VIOLATION = "contract_violation"
    RUNTIME_ERROR = "runtime_error"


class EvidenceTier(str, Enum):
    """How much a recorded number is worth."""

    REAL_ONNXRUNTIME_CPU = "real_onnxruntime_cpu"
    FAKE_SESSION_SELF_TEST = "fake_session_self_test"


@dataclass(frozen=True)
class CacheInvariantViolation:
    """One specific broken static-cache invariant, located exactly."""

    invariant: str
    tensor: str
    layer: int | None
    position: int | None
    element: int | None
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "invariant": self.invariant,
            "tensor": self.tensor,
            "layer": self.layer,
            "position": self.position,
            "element": self.element,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CacheStepReport:
    """Static-cache verdict for one graph execution."""

    step: int
    graph_kind: str
    input_valid_length: int
    output_valid_length: int
    write_index: int | None
    tensors_checked: int
    violations: tuple[CacheInvariantViolation, ...]

    @property
    def passed(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "graph_kind": self.graph_kind,
            "input_valid_length": self.input_valid_length,
            "output_valid_length": self.output_valid_length,
            "write_index": self.write_index,
            "tensors_checked": self.tensors_checked,
            "passed": self.passed,
            "violations": [item.as_dict() for item in self.violations],
        }


@dataclass(frozen=True)
class CacheStateReport:
    """Per-step reports plus the whole-run slot-immutability verdict."""

    steps: tuple[CacheStepReport, ...]
    slot_immutability_violations: tuple[CacheInvariantViolation, ...]

    @property
    def passed(self) -> bool:
        return not self.slot_immutability_violations and all(
            step.passed for step in self.steps
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "steps": [step.as_dict() for step in self.steps],
            "slot_immutability_violations": [
                item.as_dict() for item in self.slot_immutability_violations
            ],
        }


@dataclass(frozen=True)
class _CacheAxis:
    """Flat-index geometry of a cache tensor's position axis."""

    outer: int
    positions: int
    inner: int


def _cache_axis(spec: TensorSpec) -> _CacheAxis:
    try:
        axis = list(spec.layout).index(CACHE_POSITION_AXIS)
    except ValueError as exc:
        raise OnnxCpuError(
            f"{spec.name}: contract layout has no {CACHE_POSITION_AXIS!r} axis"
        ) from exc
    outer = 1
    for dimension in spec.shape[:axis]:
        outer *= dimension
    inner = 1
    for dimension in spec.shape[axis + 1 :]:
        inner *= dimension
    return _CacheAxis(outer=outer, positions=spec.shape[axis], inner=inner)


def _layer_of(name: str) -> int | None:
    _, _, suffix = name.partition(".")
    return int(suffix) if suffix.isdigit() else None


def _first_difference(
    before: Sequence[float],
    after: Sequence[float],
    start: int,
    stop: int,
) -> int:
    for index in range(start, stop):
        if before[index] != after[index]:
            return index
    return -1


def compare_cache_tensor(
    *,
    name: str,
    spec: TensorSpec,
    before: Sequence[float],
    after: Sequence[float],
    valid_length: int,
    tolerance: CacheStateTolerance,
) -> list[CacheInvariantViolation]:
    """Check one decode cache tensor against the fixed-capacity contract.

    The valid prefix ``[0, valid_length)`` and the reserved tail
    ``(valid_length, capacity)`` must be exactly unchanged; the single slot at
    ``valid_length`` must have been written and must be finite.
    """

    axis = _cache_axis(spec)
    layer = _layer_of(name)
    violations: list[CacheInvariantViolation] = []
    if not 0 <= valid_length < axis.positions:
        return [
            CacheInvariantViolation(
                invariant="write_index_within_capacity",
                tensor=name,
                layer=layer,
                position=valid_length,
                element=None,
                detail=(
                    f"write index {valid_length} is outside capacity {axis.positions}"
                ),
            )
        ]

    row = axis.inner
    for outer in range(axis.outer):
        base = outer * axis.positions * row
        slot_start = base + valid_length * row
        slot_stop = slot_start + row
        block_stop = base + axis.positions * row

        if tolerance.require_prefix_identical and valid_length:
            if before[base:slot_start] != after[base:slot_start]:
                index = _first_difference(before, after, base, slot_start)
                violations.append(
                    CacheInvariantViolation(
                        invariant="prefix_preserved",
                        tensor=name,
                        layer=layer,
                        position=(index - base) // row,
                        element=index - base,
                        detail=(
                            "valid prefix element changed: "
                            f"{before[index]!r} -> {after[index]!r}"
                        ),
                    )
                )

        if tolerance.require_slot_written:
            if before[slot_start:slot_stop] == after[slot_start:slot_stop]:
                violations.append(
                    CacheInvariantViolation(
                        invariant="slot_written",
                        tensor=name,
                        layer=layer,
                        position=valid_length,
                        element=None,
                        detail=(
                            f"slot {valid_length} is unchanged, so the decode "
                            "step wrote nothing there"
                        ),
                    )
                )
        if tolerance.require_finite_slot:
            for index in range(slot_start, slot_stop):
                if not math.isfinite(after[index]):
                    violations.append(
                        CacheInvariantViolation(
                            invariant="slot_finite",
                            tensor=name,
                            layer=layer,
                            position=valid_length,
                            element=index - base,
                            detail=f"written slot value is {after[index]!r}",
                        )
                    )
                    break

        if tolerance.require_tail_identical and slot_stop < block_stop:
            if before[slot_stop:block_stop] != after[slot_stop:block_stop]:
                index = _first_difference(before, after, slot_stop, block_stop)
                violations.append(
                    CacheInvariantViolation(
                        invariant="tail_untouched",
                        tensor=name,
                        layer=layer,
                        position=(index - base) // row,
                        element=index - base,
                        detail=(
                            "reserved tail element changed: "
                            f"{before[index]!r} -> {after[index]!r}"
                        ),
                    )
                )
    return violations


def check_prefill_cache_tensor(
    *,
    name: str,
    spec: TensorSpec,
    values: Sequence[float],
    prompt_length: int,
) -> list[CacheInvariantViolation]:
    """Require the prefill reserve ``[prompt_length, capacity)`` to be zero."""

    axis = _cache_axis(spec)
    layer = _layer_of(name)
    violations: list[CacheInvariantViolation] = []
    row = axis.inner
    for outer in range(axis.outer):
        base = outer * axis.positions * row
        reserve_start = base + prompt_length * row
        reserve_stop = base + axis.positions * row
        if not any(values[reserve_start:reserve_stop]):
            continue
        for index in range(reserve_start, reserve_stop):
            if values[index] != 0.0:
                violations.append(
                    CacheInvariantViolation(
                        invariant="prefill_reserve_zero",
                        tensor=name,
                        layer=layer,
                        position=(index - base) // row,
                        element=index - base,
                        detail=(
                            "reserved position must be zero after prefill, "
                            f"found {values[index]!r}"
                        ),
                    )
                )
                break
    return violations


# ---------------------------------------------------------------------------
# Evidence.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParityFailure:
    """One classified failure, kept alongside every other class that fired."""

    kind: str
    step: int | None
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "step": self.step, "detail": self.detail}


@dataclass(frozen=True)
class ParityStepRecord:
    """Compact per-step evidence; full logits stay outside the repository.

    ``metrics`` is ``None`` exactly when the candidate logits contained
    non-finite values, because no metric can honestly be computed from them:
    every error would be NaN and every comparison against a threshold would be
    silently false. ``non_finite_candidate_logits`` then carries how many
    elements were NaN or infinite, and the run records a
    :attr:`FailureKind.NON_FINITE_LOGITS` failure for the step. The candidate
    digest is still recorded, so the offending output is still identified.
    """

    step: int
    graph_kind: str
    input_token_id: int | None
    input_valid_length: int
    output_valid_length: int
    reference_logits_sha256: str
    candidate_logits_sha256: str
    metrics: LogitParityMetrics | None
    non_finite_candidate_logits: int = 0

    @property
    def passed(self) -> bool:
        return self.metrics is not None and self.metrics.passed

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "graph_kind": self.graph_kind,
            "input_token_id": self.input_token_id,
            "input_valid_length": self.input_valid_length,
            "output_valid_length": self.output_valid_length,
            "reference_logits_sha256": self.reference_logits_sha256,
            "candidate_logits_sha256": self.candidate_logits_sha256,
            "non_finite_candidate_logits": self.non_finite_candidate_logits,
            "metrics": None if self.metrics is None else self.metrics.as_dict(),
        }


@dataclass(frozen=True)
class ParityEvidence:
    """Committable record of one parity run, digest included."""

    evidence_tier: str
    variant_id: str
    prompt_length: int
    cache_capacity: int
    steps_requested: int
    graph_digests: Mapping[str, Any]
    runtime: Mapping[str, Any]
    tolerance: ParityTolerance
    reference_provenance: Mapping[str, Any]
    steps: tuple[ParityStepRecord, ...]
    cache_report: CacheStateReport
    failures: tuple[ParityFailure, ...]
    schema_version: int = SCHEMA_VERSION
    task_id: str = TASK_ID
    evidence_sha256: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        """Reject a tier that is not a member of :class:`EvidenceTier`.

        ``run()`` derives the tier from the session objects, but the dataclass
        is public and ``with_digest()`` reconstructs it, so the field is checked
        here too. This stops a typo or an invented tier from being serialized;
        it cannot, and does not claim to, stop a caller who owns the code from
        writing a *valid* tier onto a record that was never measured.
        """

        try:
            EvidenceTier(self.evidence_tier)
        except ValueError as exc:
            raise ParityInputError(
                f"unknown evidence tier {self.evidence_tier!r}; expected one of "
                f"{[tier.value for tier in EvidenceTier]}"
            ) from exc

    @property
    def failure_kinds(self) -> tuple[str, ...]:
        seen: list[str] = []
        for failure in self.failures:
            if failure.kind not in seen:
                seen.append(failure.kind)
        return tuple(sorted(seen))

    @property
    def passed(self) -> bool:
        return not self.failures

    def digest_payload(self) -> dict[str, Any]:
        """Everything the digest covers -- deterministic, no timestamps."""

        return _json_safe(
            {
                "schema_version": self.schema_version,
                "task_id": self.task_id,
                "evidence_tier": self.evidence_tier,
                "variant_id": self.variant_id,
                "prompt_length": self.prompt_length,
                "cache_capacity": self.cache_capacity,
                "steps_requested": self.steps_requested,
                "graph_digests": dict(self.graph_digests),
                "runtime": dict(self.runtime),
                "tolerance": self.tolerance.as_dict(),
                "reference_provenance": dict(self.reference_provenance),
                "steps": [step.as_dict() for step in self.steps],
                "cache_report": self.cache_report.as_dict(),
                "failures": [failure.as_dict() for failure in self.failures],
                "failure_kinds": list(self.failure_kinds),
                "passed": self.passed,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        payload = self.digest_payload()
        payload["evidence_sha256"] = self.evidence_sha256
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.as_dict(),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )

    def with_digest(self) -> ParityEvidence:
        return ParityEvidence(
            evidence_tier=self.evidence_tier,
            variant_id=self.variant_id,
            prompt_length=self.prompt_length,
            cache_capacity=self.cache_capacity,
            steps_requested=self.steps_requested,
            graph_digests=self.graph_digests,
            runtime=self.runtime,
            tolerance=self.tolerance,
            reference_provenance=self.reference_provenance,
            steps=self.steps,
            cache_report=self.cache_report,
            failures=self.failures,
            schema_version=self.schema_version,
            task_id=self.task_id,
            evidence_sha256=canonical_sha256(self.digest_payload()),
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    # Unreachable by construction, kept as defence in depth: the only
    # non-finite float any payload can carry is
    # `reference_top1_top2_margin`, and `LogitParityMetrics.as_dict` already
    # nulls it, while `compare_logits` refuses non-finite input outright. This
    # branch exists so a future field that forgets to null its own infinity
    # cannot make `json.dumps(..., allow_nan=False)` raise mid-serialization and
    # lose a whole evidence record. No test covers it, and mutating it away
    # leaves the suite green -- that is expected, not a gap.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_sha256(payload: Any) -> str:
    """SHA-256 over canonical JSON, mirroring the T11 evidence digest."""

    encoded = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# Evidence tier and runtime identity.
# ---------------------------------------------------------------------------


def detect_evidence_tier(sessions: Sequence[Any]) -> EvidenceTier:
    """Derive the tier from the session objects, never from a caller claim.

    A run is only ``real_onnxruntime_cpu`` when every session really is an
    instance of ``onnxruntime.InferenceSession``. ``issubclass`` is checked
    against the genuine class object, so a fake cannot answer for itself.
    """

    if not sessions:
        return EvidenceTier.FAKE_SESSION_SELF_TEST
    try:
        import onnxruntime
    except ImportError:
        return EvidenceTier.FAKE_SESSION_SELF_TEST
    real = getattr(onnxruntime, "InferenceSession", None)
    if not isinstance(real, type):
        return EvidenceTier.FAKE_SESSION_SELF_TEST
    for session in sessions:
        if not issubclass(type(session), real):
            return EvidenceTier.FAKE_SESSION_SELF_TEST
    return EvidenceTier.REAL_ONNXRUNTIME_CPU


def runtime_record(sessions: Mapping[str, Any]) -> dict[str, Any]:
    """Record the versions and session options actually loaded, not a pin.

    ``session_settings`` carries the graph optimization level, the two thread
    counts, and the execution mode as each session reports them, so evidence
    from an ``ORT_DISABLE_ALL`` run is distinguishable from an ``ORT_ENABLE_ALL``
    run of the same graphs. A session that exposes no options records ``null``.
    """

    version: str | None = None
    try:
        import onnxruntime
    except ImportError:
        onnxruntime = None  # type: ignore[assignment]
    else:
        version = str(getattr(onnxruntime, "__version__", "unknown"))

    providers: dict[str, Any] = {}
    settings: dict[str, Any] = {}
    for label, session in sessions.items():
        getter = getattr(session, "get_providers", None)
        providers[label] = list(getter()) if callable(getter) else None
        settings[label] = applied_session_settings(session)
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "onnxruntime_version": version,
        "providers": providers,
        "session_settings": settings,
    }


# ---------------------------------------------------------------------------
# The runner.
# ---------------------------------------------------------------------------


class _ClassifiedFault(Exception):
    """Internal signal carrying an already-classified failure."""

    def __init__(self, kind: FailureKind, step: int | None, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind
        self.step = step
        self.detail = detail


class OrtCpuParityRunner:
    """Drive one prefill graph and N decode steps, threading the static cache.

    The decode loop feeds each step's output cache tensors straight back in as
    the next step's input cache tensors. That threading is what makes this a
    multi-step test: a graph that writes the wrong slot still looks correct at
    step one and only diverges once the prefix it corrupted is read back.
    """

    def __init__(
        self,
        prefill_session: InferenceSessionLike,
        decode_session: InferenceSessionLike,
        *,
        contract_prefill: GraphContract,
        contract_decode: GraphContract,
        reference: ReferenceSource,
        tolerance: ParityTolerance = DEFAULT_ORT_CPU_TOLERANCE,
        tensor_factory: TensorFactory = plain_tensor_factory,
        graph_digests: Mapping[str, Any] | None = None,
    ) -> None:
        if contract_prefill.graph_kind != "prefill":
            raise OnnxCpuError("contract_prefill must be a prefill contract")
        if contract_decode.graph_kind != "decode":
            raise OnnxCpuError("contract_decode must be a decode contract")
        if (
            contract_prefill.prompt_length != contract_decode.prompt_length
            or contract_prefill.cache_capacity != contract_decode.cache_capacity
        ):
            raise OnnxCpuError(
                "prefill and decode contracts describe different variants"
            )
        self._prefill_session = prefill_session
        self._decode_session = decode_session
        self._contract_prefill = contract_prefill
        self._contract_decode = contract_decode
        self._reference = reference
        self._tolerance = tolerance
        self._tensor_factory = tensor_factory
        self._graph_digests = dict(graph_digests or {})
        self._cache_pairs = self._build_cache_pairs()
        self._prefill_cache_specs = tuple(
            spec
            for spec in contract_prefill.outputs
            if spec.name.partition(".")[0] in PREFILL_CACHE_PREFIXES
        )

    # -- contract-derived wiring -------------------------------------------

    def _build_cache_pairs(self) -> tuple[tuple[str, str, TensorSpec], ...]:
        inputs = {spec.name: spec for spec in self._contract_decode.inputs}
        pairs: list[tuple[str, str, TensorSpec]] = []
        for spec in self._contract_decode.outputs:
            prefix, _, suffix = spec.name.partition(".")
            source_prefix = DECODE_CACHE_PREFIX_SOURCES.get(prefix)
            if source_prefix is None:
                continue
            source = f"{source_prefix}.{suffix}"
            if source not in inputs:
                raise OnnxCpuError(
                    f"decode contract output {spec.name!r} has no matching "
                    f"input {source!r}"
                )
            if inputs[source].shape != spec.shape:
                raise OnnxCpuError(
                    f"decode contract cache pair {source!r}/{spec.name!r} "
                    "has mismatched shapes"
                )
            pairs.append((source, spec.name, spec))
        if not pairs:
            raise OnnxCpuError("decode contract declares no cache tensor pairs")
        return tuple(pairs)

    # -- execution ---------------------------------------------------------

    def _execute(
        self,
        session: InferenceSessionLike,
        contract: GraphContract,
        feed: Mapping[str, Any],
        step: int,
    ) -> dict[str, Any]:
        expected = [spec.name for spec in contract.outputs]
        declared = self._declared_output_names(session, step)
        if declared is not None and declared != expected:
            raise _ClassifiedFault(
                FailureKind.CONTRACT_VIOLATION,
                step,
                f"{contract.graph_kind} session declares outputs {declared} "
                f"but the T12 contract requires {expected}",
            )
        try:
            values = session.run(expected, dict(feed))
        except Exception as exc:  # noqa: BLE001 - classified, not swallowed
            raise _ClassifiedFault(
                FailureKind.RUNTIME_ERROR,
                step,
                f"{contract.graph_kind} session.run failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc
        values = list(values)
        if len(values) != len(expected):
            raise _ClassifiedFault(
                FailureKind.CONTRACT_VIOLATION,
                step,
                f"{contract.graph_kind} returned {len(values)} outputs, "
                f"the T12 contract requires {len(expected)}",
            )
        mapping = dict(zip(expected, values, strict=True))
        try:
            validate_tensor_mapping(mapping, contract.outputs)
        except CacheContractError as exc:
            raise _ClassifiedFault(
                FailureKind.CONTRACT_VIOLATION,
                step,
                f"{contract.graph_kind} output violates the T12 contract: {exc}",
            ) from exc
        return mapping

    @staticmethod
    def _declared_output_names(
        session: InferenceSessionLike,
        step: int,
    ) -> list[str] | None:
        getter = getattr(session, "get_outputs", None)
        if not callable(getter):
            return None
        try:
            declared = list(getter())
        except Exception as exc:  # noqa: BLE001 - classified, not swallowed
            raise _ClassifiedFault(
                FailureKind.RUNTIME_ERROR,
                step,
                f"session.get_outputs failed: {type(exc).__name__}: {exc}",
            ) from exc
        names: list[str] = []
        for entry in declared:
            name = getattr(entry, "name", None)
            if not isinstance(name, str):
                return None
            names.append(name)
        return names

    def _feed(
        self,
        contract: GraphContract,
        values: Mapping[str, Any],
        passthrough: Mapping[str, Any],
    ) -> dict[str, Any]:
        feed: dict[str, Any] = {}
        for spec in contract.inputs:
            if spec.name in passthrough:
                feed[spec.name] = passthrough[spec.name]
                continue
            if spec.name not in values:
                raise OnnxCpuError(
                    f"no value prepared for {contract.graph_kind} input {spec.name!r}"
                )
            feed[spec.name] = self._tensor_factory(
                values[spec.name], spec.shape, spec.dtype
            )
        return feed

    # -- public entry point ------------------------------------------------

    def run(self, steps: int) -> ParityEvidence:
        """Run prefill plus ``steps`` decode steps and classify every failure."""

        if steps < 1:
            raise OnnxCpuError("steps must be at least one decode step")
        prompt_length = self._contract_prefill.prompt_length
        capacity = self._contract_prefill.cache_capacity
        if prompt_length + steps > capacity:
            raise OnnxCpuError(
                f"{steps} decode steps overflow capacity {capacity} from "
                f"prompt length {prompt_length}"
            )
        prompt = [int(token) for token in self._reference.prompt_token_ids()]
        if len(prompt) != prompt_length:
            raise OnnxCpuError(
                f"reference prompt has {len(prompt)} tokens, the "
                f"{self._contract_prefill.variant_id} contract requires "
                f"{prompt_length}"
            )

        records: list[ParityStepRecord] = []
        cache_steps: list[CacheStepReport] = []
        failures: list[ParityFailure] = []
        written_slots: list[tuple[int, str, int, tuple[float, ...]]] = []
        slot_violations: list[CacheInvariantViolation] = []
        final_cache: dict[str, Any] = {}

        try:
            outputs = self._run_prefill(prompt, records, cache_steps)
            valid_length = _scalar_int(outputs["valid_length"], "valid_length")
            # Prefill emits the decode graph's incoming cache names directly.
            cache = {source: outputs[source] for source, _, _ in self._cache_pairs}
            final_cache = dict(cache)
            for step in range(1, steps + 1):
                token = int(self._reference.expected_token_id(step - 1))
                outputs = self._run_decode(
                    step=step,
                    token=token,
                    valid_length=valid_length,
                    cache=cache,
                    records=records,
                    cache_steps=cache_steps,
                    written_slots=written_slots,
                )
                cache = {
                    source: outputs[target] for source, target, _ in self._cache_pairs
                }
                final_cache = dict(cache)
                # Thread the graph's own reported length, not an internal
                # counter. A validation tool that silently substitutes the
                # value it expected cannot observe the graph disagreeing with
                # it: the disagreement is already recorded as
                # `valid_length_increment`, and carrying the reported value
                # forward is what lets `write_index_within_capacity` fire when
                # a graph reports a length outside the fixed capacity.
                valid_length = _scalar_int(
                    outputs["updated_valid_length"], "updated_valid_length"
                )
        except _ClassifiedFault as fault:
            failures.append(
                ParityFailure(
                    kind=fault.kind.value, step=fault.step, detail=fault.detail
                )
            )

        slot_violations.extend(
            self._check_slot_immutability(written_slots, final_cache)
        )
        cache_report = CacheStateReport(
            steps=tuple(cache_steps),
            slot_immutability_violations=tuple(slot_violations),
        )

        # Ordering is diagnostic, not cosmetic. State faults come first because
        # they are the more fundamental diagnosis: a wrong cache read moves the
        # logits too, so a combined run that listed the tolerance failure first
        # would invite exactly the retolerancing that this task exists to
        # prevent. Non-finite logits precede tolerance failures for the same
        # reason -- "Inf came out of the graph" is a different question from
        # "the answer drifted".
        for report in cache_steps:
            for violation in report.violations:
                failures.append(
                    ParityFailure(
                        kind=FailureKind.CACHE_STATE_UPDATE.value,
                        step=report.step,
                        detail=(
                            f"{violation.invariant} broken on "
                            f"{violation.tensor} at position "
                            f"{violation.position}: {violation.detail}"
                        ),
                    )
                )
        for violation in slot_violations:
            failures.append(
                ParityFailure(
                    kind=FailureKind.CACHE_STATE_UPDATE.value,
                    step=None,
                    detail=(
                        f"{violation.invariant} broken on {violation.tensor} "
                        f"at position {violation.position}: {violation.detail}"
                    ),
                )
            )
        for record in records:
            if record.metrics is None:
                failures.append(
                    ParityFailure(
                        kind=FailureKind.NON_FINITE_LOGITS.value,
                        step=record.step,
                        detail=(
                            f"{record.graph_kind} candidate logits contain "
                            f"{record.non_finite_candidate_logits} NaN or "
                            "infinite values, so no metric was computed"
                        ),
                    )
                )
        for record in records:
            if record.metrics is not None and not record.metrics.passed:
                failures.append(
                    ParityFailure(
                        kind=FailureKind.NUMERICAL_TOLERANCE.value,
                        step=record.step,
                        detail=(
                            f"{record.graph_kind} logits are outside the "
                            "derived ORT CPU tolerance"
                        ),
                    )
                )

        sessions = {
            "prefill": self._prefill_session,
            "decode": self._decode_session,
        }
        evidence = ParityEvidence(
            evidence_tier=detect_evidence_tier(list(sessions.values())).value,
            variant_id=self._contract_prefill.variant_id,
            prompt_length=prompt_length,
            cache_capacity=capacity,
            steps_requested=steps,
            graph_digests=dict(self._graph_digests),
            runtime=runtime_record(sessions),
            tolerance=self._tolerance,
            reference_provenance=dict(self._reference.provenance()),
            steps=tuple(records),
            cache_report=cache_report,
            failures=tuple(failures),
        )
        return evidence.with_digest()

    # -- steps -------------------------------------------------------------

    def _run_prefill(
        self,
        prompt: Sequence[int],
        records: list[ParityStepRecord],
        cache_steps: list[CacheStepReport],
    ) -> dict[str, Any]:
        contract = self._contract_prefill
        prompt_length = contract.prompt_length
        feed = self._feed(
            contract,
            {
                "input_ids": list(prompt),
                "attention_mask": [1] * prompt_length,
                "position_ids": list(range(prompt_length)),
            },
            {},
        )
        outputs = self._execute(self._prefill_session, contract, feed, step=0)

        valid_length = _scalar_int(outputs["valid_length"], "valid_length")
        violations: list[CacheInvariantViolation] = []
        if valid_length != prompt_length:
            violations.append(
                CacheInvariantViolation(
                    invariant="prefill_valid_length",
                    tensor="valid_length",
                    layer=None,
                    position=None,
                    element=None,
                    detail=(
                        f"prefill reported valid_length {valid_length}, the "
                        f"contract requires {prompt_length}"
                    ),
                )
            )
        for spec in self._prefill_cache_specs:
            violations.extend(
                check_prefill_cache_tensor(
                    name=spec.name,
                    spec=spec,
                    values=flatten(outputs[spec.name]),
                    prompt_length=prompt_length,
                )
            )
        cache_steps.append(
            CacheStepReport(
                step=0,
                graph_kind="prefill",
                input_valid_length=0,
                output_valid_length=valid_length,
                write_index=None,
                tensors_checked=len(self._prefill_cache_specs),
                violations=tuple(violations),
            )
        )
        records.append(
            self._logit_record(
                step=0,
                graph_kind="prefill",
                input_token_id=None,
                input_valid_length=0,
                output_valid_length=valid_length,
                candidate=outputs["last_logits"],
            )
        )
        return outputs

    def _run_decode(
        self,
        *,
        step: int,
        token: int,
        valid_length: int,
        cache: Mapping[str, Any],
        records: list[ParityStepRecord],
        cache_steps: list[CacheStepReport],
        written_slots: list[tuple[int, str, int, tuple[float, ...]]],
    ) -> dict[str, Any]:
        contract = self._contract_decode
        capacity = contract.cache_capacity
        mask = [1] * min(valid_length + 1, capacity)
        mask.extend([0] * (capacity - len(mask)))
        feed = self._feed(
            contract,
            {
                "input_ids": [token],
                "attention_mask": mask,
                "position_ids": [valid_length],
                "valid_length": [valid_length],
            },
            cache,
        )
        outputs = self._execute(self._decode_session, contract, feed, step=step)

        updated = _scalar_int(outputs["updated_valid_length"], "updated_valid_length")
        violations: list[CacheInvariantViolation] = []
        tolerance = self._tolerance.cache_state
        if tolerance.require_valid_length_increment and updated != valid_length + 1:
            violations.append(
                CacheInvariantViolation(
                    invariant="valid_length_increment",
                    tensor="updated_valid_length",
                    layer=None,
                    position=valid_length,
                    element=None,
                    detail=(
                        f"updated_valid_length is {updated}, the contract "
                        f"requires {valid_length + 1}"
                    ),
                )
            )
        for source, target, spec in self._cache_pairs:
            before = flatten(cache[source])
            after = flatten(outputs[target])
            violations.extend(
                compare_cache_tensor(
                    name=target,
                    spec=spec,
                    before=before,
                    after=after,
                    valid_length=valid_length,
                    tolerance=tolerance,
                )
            )
            written_slots.append(
                (step, target, valid_length, _slot_values(after, spec, valid_length))
            )
        cache_steps.append(
            CacheStepReport(
                step=step,
                graph_kind="decode",
                input_valid_length=valid_length,
                output_valid_length=updated,
                write_index=valid_length,
                tensors_checked=len(self._cache_pairs),
                violations=tuple(violations),
            )
        )
        records.append(
            self._logit_record(
                step=step,
                graph_kind="decode",
                input_token_id=token,
                input_valid_length=valid_length,
                output_valid_length=updated,
                candidate=outputs["next_logits"],
            )
        )
        return outputs

    def _logit_record(
        self,
        *,
        step: int,
        graph_kind: str,
        input_token_id: int | None,
        input_valid_length: int,
        output_valid_length: int,
        candidate: Any,
    ) -> ParityStepRecord:
        reference_values = _logit_vector(self._reference.next_logits(step), "reference")
        if non_finite_count(reference_values):
            # The golden side is a configuration input, not a measurement: a
            # non-finite reference means the T11 fixture or the reference run
            # is broken, so there is nothing to classify and the run stops with
            # a configuration error (exit 2). The T11 reference refuses the
            # same input for the same reason.
            raise ParityInputError(
                f"reference logits for step {step} contain NaN or infinite "
                "values; the golden reference is unusable"
            )
        candidate_values = _logit_vector(candidate, "candidate")
        # A non-finite *candidate* is the opposite case: an FP16 export that
        # overflows to Inf, or a graph that divides by zero, is one of the most
        # likely real ONNX Runtime failures. It is classified and reported like
        # any other failure instead of aborting the run, so the cache
        # invariants for this and every later step are still checked.
        non_finite = non_finite_count(candidate_values)
        metrics = (
            None
            if non_finite
            else compare_logits(reference_values, candidate_values, self._tolerance)
        )
        return ParityStepRecord(
            step=step,
            graph_kind=graph_kind,
            input_token_id=input_token_id,
            input_valid_length=input_valid_length,
            output_valid_length=output_valid_length,
            reference_logits_sha256=values_sha256(reference_values),
            candidate_logits_sha256=values_sha256(candidate_values),
            metrics=metrics,
            non_finite_candidate_logits=non_finite,
        )

    def _check_slot_immutability(
        self,
        written_slots: Sequence[tuple[int, str, int, tuple[float, ...]]],
        final_cache: Mapping[str, Any],
    ) -> list[CacheInvariantViolation]:
        """Whole-run property: a slot, once written, never changes again."""

        # The `or not final_cache` half is unreachable by construction and kept
        # as defence in depth: `run()` fills `final_cache` from the prefill
        # outputs before the decode loop starts, and `written_slots` is appended
        # to only from inside that loop, so a non-empty `written_slots` always
        # comes with a non-empty `final_cache`. Only the `not written_slots`
        # half ever fires -- on a zero-step run, or when the loop aborted before
        # the first decode. Mutating the `or` away leaves the suite green; that
        # is expected, not a coverage gap.
        if not written_slots or not final_cache:
            return []
        specs = {target: spec for _, target, spec in self._cache_pairs}
        sources = {target: source for source, target, _ in self._cache_pairs}
        cached: dict[str, list[float]] = {}
        violations: list[CacheInvariantViolation] = []
        for step, target, index, expected in written_slots:
            source = sources[target]
            if source not in final_cache:
                continue
            if target not in cached:
                cached[target] = flatten(final_cache[source])
            actual = _slot_values(cached[target], specs[target], index)
            if actual != expected:
                offset = next(
                    (
                        position
                        for position, (left, right) in enumerate(
                            zip(expected, actual, strict=True)
                        )
                        if left != right
                    ),
                    None,
                )
                violations.append(
                    CacheInvariantViolation(
                        invariant="written_slot_immutable",
                        tensor=target,
                        layer=_layer_of(target),
                        position=index,
                        element=offset,
                        detail=(
                            f"slot written at step {step} changed before the "
                            "end of the run"
                        ),
                    )
                )
        return violations


def _slot_values(
    flat: Sequence[float],
    spec: TensorSpec,
    index: int,
) -> tuple[float, ...]:
    axis = _cache_axis(spec)
    if not 0 <= index < axis.positions:
        return ()
    values: list[float] = []
    row = axis.inner
    for outer in range(axis.outer):
        start = outer * axis.positions * row + index * row
        values.extend(flat[start : start + row])
    return tuple(values)


# ---------------------------------------------------------------------------
# PyTorch reference adapter (real runs only).
# ---------------------------------------------------------------------------


class TorchReferenceSource:
    """Golden logits and tokens from the pinned T11 PyTorch reference.

    This adapter exists so the recorded real-measurement command runs; it
    cannot be unit-tested here because torch is absent. It is kept small and
    linear on purpose: prefill once, then teacher-force the reference's own
    greedy token through ``steps`` cached decode steps, recording the
    next-token logits after each. Teacher-forcing mirrors T11 so a single
    disagreeing token diagnoses that step instead of compounding into every
    later one.
    """

    def __init__(
        self,
        prompt_token_ids: Sequence[int],
        *,
        steps: int,
        device: str = "cpu",
        dtype: str | None = None,
        seed: int = 0,
        local_files_only: bool = True,
    ) -> None:
        if steps < 0:
            raise OnnxCpuError("steps must be non-negative")
        self._prompt_token_ids = tuple(int(token) for token in prompt_token_ids)
        self._steps = int(steps)
        self._device = device
        self._dtype = dtype
        self._seed = seed
        self._local_files_only = local_files_only
        self._logits: list[list[float]] = []
        self._tokens: list[int] = []
        self._provenance: dict[str, Any] = {}

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, Any],
        *,
        steps: int,
        fixture_path: Path | None = None,
        **kwargs: Any,
    ) -> TorchReferenceSource:
        """Build the source from a T20 manifest's frozen T10 workload."""

        context_length = manifest.get("context_length")
        if not isinstance(context_length, int):
            raise OnnxCpuError("manifest has no integer context_length")
        return cls(
            load_context_workload_tokens(context_length, path=fixture_path),
            steps=steps,
            **kwargs,
        )

    # -- ReferenceSource ---------------------------------------------------

    def prompt_token_ids(self) -> Sequence[int]:
        return self._prompt_token_ids

    def next_logits(self, step: int) -> Sequence[float]:
        self._materialize()
        if not 0 <= step < len(self._logits):
            raise OnnxCpuError(f"no reference logits for step {step}")
        return self._logits[step]

    def expected_token_id(self, step: int) -> int:
        self._materialize()
        if not 0 <= step < len(self._tokens):
            raise OnnxCpuError(f"no reference token for step {step}")
        return self._tokens[step]

    def provenance(self) -> Mapping[str, Any]:
        self._materialize()
        return dict(self._provenance)

    # -- execution ---------------------------------------------------------

    def _materialize(self) -> None:  # pragma: no cover - requires torch
        if self._logits:
            return
        try:
            import torch
        except ImportError as exc:
            raise OnnxCpuDependencyError(
                "PyTorch is required to produce the T11 reference logits. "
                f"Install the optional extra with: {INSTALL_HINT}"
            ) from exc
        from slm_lab.models.qwen3_reference import load_reference_model

        reference = load_reference_model(
            device=self._device,
            dtype=self._dtype,
            seed=self._seed,
            local_files_only=self._local_files_only,
            attn_implementation="eager",
        )
        model = reference.model
        input_ids = torch.tensor(
            [list(self._prompt_token_ids)], dtype=torch.long, device=self._device
        )
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        logits = output.logits[:, -1, :].to(torch.float32)
        past_key_values = output.past_key_values
        self._logits.append([float(value) for value in logits[0].tolist()])

        for _ in range(self._steps):
            token = int(logits.argmax(dim=-1)[0].item())
            self._tokens.append(token)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones((1, 1), dtype=attention_mask.dtype, device=self._device),
                ),
                dim=1,
            )
            with torch.inference_mode():
                output = model(
                    input_ids=torch.tensor(
                        [[token]], dtype=torch.long, device=self._device
                    ),
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            logits = output.logits[:, -1, :].to(torch.float32)
            past_key_values = output.past_key_values
            self._logits.append([float(value) for value in logits[0].tolist()])

        self._provenance = {
            "source": "slm_lab.models.qwen3_reference.load_reference_model",
            "model_id": reference.contract.model_id,
            "model_revision": reference.contract.revision,
            "runtime": reference.runtime.as_dict(),
            "teacher_forced": True,
            "prompt_token_count": len(self._prompt_token_ids),
            "prompt_token_ids_sha256": canonical_sha256(list(self._prompt_token_ids)),
            "reference_logits_sha256": [
                values_sha256(values) for values in self._logits
            ],
            "expected_token_ids": list(self._tokens),
        }


#: Dtypes the reference self-error diagnostic sweeps, coarsest storage last so
#: the pairwise keys read "more exact vs less exact".
SELF_ERROR_DTYPES = ("float32", "bfloat16", "float16")


def reference_self_error(
    context_length: int,
    *,
    steps: int,
    dtypes: Sequence[str] = SELF_ERROR_DTYPES,
    reference_factory: Callable[[str], ReferenceSource] | None = None,
) -> dict[str, Any]:
    """Measure the *reference model's own* dtype error. No ONNX graph involved.

    This is the empirical check on the depth-and-ULP derivation recorded above
    :data:`DEFAULT_ORT_CPU_TOLERANCE`, and it exists because that derivation is
    otherwise unverifiable from the repository. It runs the same pinned PyTorch
    reference on the same frozen T10 workload at several storage dtypes and
    compares the resulting next-token logits against each other.

    Two quantities come out of it, neither of which involves the candidate:

    * ``lambda_max_abs_logit`` -- the logit scale at which an absolute
      tolerance binds, which the superseded derivation guessed at.
    * the pairwise error, whose ratio to the dtype's unit roundoff is the
      residual stream's error gain ``G``.

    A third, ``consecutive_step_distance``, is the mis-wiring reference scale:
    the distance between one decode step's logits and the next is what a cache
    read landing one slot off would look like, so it bounds from below what a
    state defect must move the logits by.

    This produces a *diagnostic*, never a parity record. There is no session,
    no evidence tier, and no ``passed`` field, precisely so it cannot be
    mistaken for one.
    """

    if steps < 0:
        raise OnnxCpuError("steps must be non-negative")
    names = tuple(dtypes)
    if len(names) < 2:
        raise OnnxCpuError("at least two dtypes are needed to compare")

    build = reference_factory or (
        lambda name: TorchReferenceSource(
            load_context_workload_tokens(context_length), steps=steps, dtype=name
        )
    )

    logits: dict[str, list[list[float]]] = {}
    provenance: dict[str, Any] = {}
    for name in names:
        source = build(name)
        provenance[name] = dict(source.provenance())
        logits[name] = [list(source.next_logits(index)) for index in range(steps + 1)]

    def measured(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
        # relative_floor and the cosine convention are shared with the parity
        # path deliberately, so a number here is directly comparable with a
        # number in an S*-ort-cpu.json record.
        return compare_logits(list(left), list(right)).as_dict()

    pairwise = {
        f"{left}_vs_{right}": [
            measured(logits[left][index], logits[right][index])
            for index in range(steps + 1)
        ]
        for position, left in enumerate(names)
        for right in names[position + 1 :]
    }

    baseline = names[0]
    consecutive = [
        measured(logits[baseline][index], logits[baseline][index + 1])
        for index in range(steps)
    ]
    return {
        "schema_version": 1,
        "task_id": "T23",
        "record_kind": "diagnostic_reference_dtype_self_error",
        "not_a_parity_record": (
            "No ONNX Runtime session was created and no graph was executed. "
            "This compares the PyTorch reference against itself at different "
            "storage dtypes, to check the ORT CPU tolerance derivation. It is "
            "not a T21 parity measurement and carries no evidence tier."
        ),
        # The per-pair `passed` and `allclose` fields nested under `pairwise`
        # and `consecutive_step_distance` are evaluated against
        # DEFAULT_ORT_CPU_TOLERANCE, the same tolerance the parity path uses,
        # so they are directly comparable with a T21 number. Rolled up here
        # because together they are the two-sided property a tolerance has to
        # have, and neither half is worth much without the other.
        "tolerance_verdict": {
            "every_reference_dtype_pair_passes": all(
                record["passed"] for rows in pairwise.values() for record in rows
            ),
            "every_consecutive_step_pair_fails": all(
                not record["passed"] for record in consecutive
            ),
            "means": (
                "The first says the tolerance accepts the exact (float32) "
                "answer, so it is not rejecting correct implementations. The "
                "second says it still rejects a one-slot cache offset, so it "
                "has not been widened into uselessness. A tolerance needs both; "
                "the superseded atol=0.25 had only the second."
            ),
        },
        "context_length": context_length,
        "steps_requested": steps,
        "dtypes": list(names),
        "lambda_max_abs_logit": {
            name: [max(abs(value) for value in row) for row in rows]
            for name, rows in logits.items()
        },
        "max_logit": {
            name: [max(row) for row in rows] for name, rows in logits.items()
        },
        "min_logit": {
            name: [min(row) for row in rows] for name, rows in logits.items()
        },
        "rms_logit": {
            name: [math.sqrt(sum(v * v for v in row) / len(row)) for row in rows]
            for name, rows in logits.items()
        },
        "pairwise": pairwise,
        "consecutive_step_distance": consecutive,
        "consecutive_step_distance_dtype": baseline,
        "reference_provenance": provenance,
        "tolerance": DEFAULT_ORT_CPU_TOLERANCE.as_dict(),
    }


def load_context_workload_tokens(
    context_length: int,
    *,
    path: Path | None = None,
) -> tuple[int, ...]:
    """Load one frozen T10 static-context workload's prompt token IDs."""

    source = path or (PROJECT_ROOT / "tests/fixtures/t10/token-fixtures-v1.json")
    try:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OnnxCpuError(f"cannot load T10 token fixtures: {exc}") from exc
    for record in payload.get("context_workloads", []):
        if record.get("context_length") == context_length:
            tokens = record.get("token_ids")
            if not isinstance(tokens, list) or len(tokens) != context_length:
                raise OnnxCpuError(
                    f"S{context_length} workload token IDs are malformed"
                )
            return tuple(int(token) for token in tokens)
    raise OnnxCpuError(f"T10 fixtures have no S{context_length} workload")


# ---------------------------------------------------------------------------
# Command line.
# ---------------------------------------------------------------------------


def sha256_file(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact_root(explicit: str | None = None) -> Path:
    """``--artifact-root``, then ``SLM_LAB_ARTIFACT_ROOT``, then ``artifacts/``."""

    value = explicit or os.environ.get("SLM_LAB_ARTIFACT_ROOT")
    root = Path(value).expanduser() if value else DEFAULT_ARTIFACT_SYMLINK
    if not root.is_dir():
        raise OnnxCpuError(
            f"artifact root does not exist: {root}. Set SLM_LAB_ARTIFACT_ROOT "
            "or pass --artifact-root."
        )
    return root


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OnnxCpuError(f"cannot load T20 manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OnnxCpuError(f"T20 manifest {path} is not a JSON object")
    return payload


def safe_relative(directory: Path, relative_path: Any) -> Path:
    """Join a manifest-supplied relative path without letting it escape.

    A manifest is an input file like any other, so the same guard
    ``slm_lab.graph.inspection._safe_relative`` applies to the inspection side
    applies here: an absolute path or one containing ``..`` is rejected rather
    than silently resolved outside the artifact root.
    """

    if not isinstance(relative_path, str) or not relative_path:
        raise OnnxCpuError("T20 manifest relative_path must be a non-empty string")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise OnnxCpuError(f"unsafe T20 manifest relative_path {relative_path!r}")
    return directory.joinpath(*pure.parts)


def verified_graph_paths(
    manifest: Mapping[str, Any],
    artifact_root: Path,
) -> dict[str, tuple[Path, str, str]]:
    """Resolve and hash-verify both graphs before any session is created.

    Each entry is ``(resolved_path, sha256, manifest_relative_path)``. The
    resolved path is a host detail used to open the file; the relative path is
    what identifies the graph in committed evidence. See
    :func:`graph_digests_payload`.
    """

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise OnnxCpuError("T20 manifest has no artifacts block")
    directory = artifact_root / ARTIFACT_SUBDIRECTORY
    resolved: dict[str, tuple[Path, str]] = {}
    for kind in ("prefill", "decode"):
        record = artifacts.get(kind)
        if not isinstance(record, Mapping):
            raise OnnxCpuError(f"T20 manifest has no {kind} artifact record")
        relative = record.get("relative_path")
        expected = record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise OnnxCpuError(f"T20 manifest {kind} record lacks relative_path/sha256")
        path = safe_relative(directory, relative)
        if not path.is_file():
            raise OnnxCpuError(
                f"missing {kind} graph {path}. Export it with T20 or point "
                "--artifact-root at the storage that holds it."
            )
        actual = sha256_file(path)
        if actual != expected:
            raise OnnxCpuError(
                f"{kind} graph {path} does not match the digest committed in "
                f"the T20 manifest: expected {expected}, found {actual}"
            )
        resolved[kind] = (path, expected, relative)
    return resolved


def graph_digests_payload(
    graphs: Mapping[str, tuple[Path, str, str]],
) -> dict[str, dict[str, str]]:
    """The single shape of ``graph_digests``, used by the CLI and the tests.

    The record is committed under ``results/graph/parity/`` and covered by
    ``evidence_sha256``, so it must not depend on where the artifact root
    happens to be mounted. Only the manifest-relative path and the verified
    digest go in; the absolute host path stays out of the evidence entirely.
    Two hosts running the same graphs therefore produce comparable digests.
    """

    return {
        kind: {"sha256": sha256, "relative_path": relative}
        for kind, (_, sha256, relative) in graphs.items()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m slm_lab.backends.onnx_cpu",
        description=(
            "Run the T20 ONNX graphs on the ONNX Runtime CPU provider and "
            "classify numerical-tolerance versus static-cache-state failures."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Committed T20 manifest describing one exported variant",
    )
    parser.add_argument(
        "--artifact-root",
        default=None,
        help=(
            "Root holding onnx/reference/T20; defaults to "
            "SLM_LAB_ARTIFACT_ROOT, then the repository artifacts/ symlink"
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of one-token decode steps to thread through the cache",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the evidence JSON; stdout when omitted",
    )
    parser.add_argument(
        "--reference",
        choices=("torch",),
        default="torch",
        help="Source of the golden logits",
    )
    parser.add_argument(
        "--graph-optimization-level",
        default="ORT_DISABLE_ALL",
        help="ONNX Runtime graph optimization level to record and apply",
    )
    parser.add_argument(
        "--reference-dtype",
        default=None,
        choices=("float32", "bfloat16", "float16"),
        help=(
            "Reference model dtype; defaults to the contract's reference_dtype "
            "(bfloat16). DIAGNOSTIC ONLY: a run at any other dtype is not a T21 "
            "parity record, because DEFAULT_ORT_CPU_TOLERANCE is derived for a "
            "bfloat16 reference. The chosen dtype is recorded in "
            "reference_provenance.runtime.dtype, so such a record is "
            "self-identifying; do not write one over results/graph/parity/"
            "S<N>-ort-cpu.json"
        ),
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit downloading the pinned public reference revision",
    )
    parser.add_argument(
        "--reference-self-error",
        action="store_true",
        help=(
            "DIAGNOSTIC: skip ONNX entirely and compare the PyTorch reference "
            "against itself at float32, bfloat16 and float16. This is the "
            "empirical check on the tolerance derivation; it writes a record "
            "with record_kind=diagnostic_reference_dtype_self_error and no "
            "evidence tier, and it is not a parity measurement"
        ),
    )
    return parser


def _run_cli(
    args: argparse.Namespace,
    *,
    session_factory: SessionFactory | None = None,
    reference_factory: Callable[[Mapping[str, Any], int], ReferenceSource]
    | None = None,
    tensor_factory: TensorFactory | None = None,
) -> int:
    manifest = load_manifest(Path(args.manifest))
    context_length = manifest.get("context_length")
    if context_length not in CONTEXT_VARIANTS:
        raise OnnxCpuError(
            f"manifest context_length {context_length!r} is not one of "
            f"{tuple(CONTEXT_VARIANTS)}"
        )

    if getattr(args, "reference_self_error", False):
        # Returns before any session is built, so this branch cannot emit
        # something that looks like a measurement of a graph.
        document = json.dumps(
            _json_safe(reference_self_error(context_length, steps=args.steps)),
            indent=2,
            sort_keys=True,
        )
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(document + "\n", encoding="utf-8")
        else:
            print(document)
        return 0

    artifact_root = resolve_artifact_root(args.artifact_root)
    graphs = verified_graph_paths(manifest, artifact_root)

    factory = session_factory or onnxruntime_cpu_session_factory(
        graph_optimization_level=args.graph_optimization_level
    )
    prefill_session = factory(graphs["prefill"][0])
    decode_session = factory(graphs["decode"][0])

    build_reference = reference_factory
    if build_reference is None:
        if args.reference != "torch":  # pragma: no cover - argparse-guarded
            raise OnnxCpuError(f"unknown reference {args.reference!r}")

        def build_reference(document: Mapping[str, Any], steps: int) -> ReferenceSource:
            return TorchReferenceSource.from_manifest(
                document,
                steps=steps,
                dtype=getattr(args, "reference_dtype", None),
                local_files_only=not args.allow_download,
            )

    reference = build_reference(manifest, args.steps)

    runner = OrtCpuParityRunner(
        prefill_session,
        decode_session,
        contract_prefill=build_prefill_contract(context_length),
        contract_decode=build_decode_contract(context_length),
        reference=reference,
        tolerance=DEFAULT_ORT_CPU_TOLERANCE,
        tensor_factory=tensor_factory or numpy_tensor_factory(),
        graph_digests=graph_digests_payload(graphs),
    )
    evidence = runner.run(args.steps)
    document = evidence.to_json()
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(document + "\n", encoding="utf-8")
    else:
        print(document)
    return 0 if evidence.passed else 1


def main(
    argv: Sequence[str] | None = None,
    *,
    session_factory: SessionFactory | None = None,
    reference_factory: Callable[[Mapping[str, Any], int], ReferenceSource]
    | None = None,
    tensor_factory: TensorFactory | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run_cli(
            args,
            session_factory=session_factory,
            reference_factory=reference_factory,
            tensor_factory=tensor_factory,
        )
    except OnnxCpuError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except CacheContractError as exc:
        print(f"error: contract violation: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
