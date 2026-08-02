"""Rule-driven deployment-risk inspection of the T20 reference ONNX graphs.

The engine reads a :class:`~slm_lab.graph.onnx_reader.GraphSummary`, applies a
declarative catalogue of structural risk rules, and emits a compact ranked
report. It reports what it observed. It never claims that a graph will or will
not compile for a given target; severities are review judgements bound to the
catalogue's stated target context, and only T22 can replace them with measured
compiler evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from slm_lab.graph.onnx_reader import (
    GraphSummary,
    NodeInfo,
    OnnxReadError,
    TensorShape,
    ValueInfo,
    read_onnx_model,
)


SCHEMA_VERSION = 1
MODULE_NAME = "slm_lab.graph.inspection"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RULES_PATH = PROJECT_ROOT / "configs/graph/onnx-risk-rules-v1.json"
DEFAULT_MANIFEST_DIRECTORY = PROJECT_ROOT / "results/manifests/onnx"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "results/graph"
ARTIFACT_ROOT_TOKEN = "${SLM_LAB_ARTIFACT_ROOT}"
GRAPH_KINDS = ("prefill", "decode")
MAX_DYNAMIC_DIMENSION_ENTRIES = 64

# Emitted verbatim into every report so a reader who only ever sees the JSON --
# and therefore only ever sees the word "blocking" next to a rule id -- knows
# what produced it. Mirrors the `claim_boundary` block T20 writes into
# `results/manifests/onnx/S*.json`.
CLAIM_BOUNDARY: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "establishes": (
            "onnx_file_sha256_matches_the_committed_T20_manifest",
            "structural_inventory_of_the_onnx_protobuf",
            "declared_public_boundary_shapes_and_dtypes",
            "risk_rule_matches_against_the_committed_catalogue",
        ),
        "does_not_establish": (
            "compiler_acceptance",
            "operator_support_by_any_vendor_toolchain",
            "accelerator_placement",
            "onnxruntime_numerical_parity",
            "latency_or_memory_performance",
            "severity_derived_from_an_executed_compile_or_conversion_job",
            "onnx_model_validity_beyond_structural_decoding",
        ),
    }
)

SEVERITY_RANKS: Mapping[str, int] = MappingProxyType(
    {
        "blocking": 0,
        "high": 1,
        "medium": 2,
        "low": 3,
        "informational": 4,
    }
)
CATEGORIES = frozenset(
    {
        "dynamic_shape",
        "operator_support",
        "precision",
        "memory_traffic",
        "graph_scale",
        "control_flow",
    }
)
RULE_FIELDS = (
    "id",
    "title",
    "category",
    "severity",
    "detector",
    "params",
    "rationale",
    "mitigation",
    "references",
)

# TensorProto.DataType values. Cast carries its destination as one of these.
ELEMENT_TYPE_NAMES: Mapping[int, str] = MappingProxyType(
    {
        1: "float32",
        2: "uint8",
        3: "int8",
        4: "uint16",
        5: "int16",
        6: "int32",
        7: "int64",
        9: "bool",
        10: "float16",
        11: "float64",
        12: "uint32",
        13: "uint64",
        16: "bfloat16",
    }
)
FLOAT_ELEMENT_TYPES: Mapping[int, str] = MappingProxyType({1: "float32", 10: "float16"})
FLOAT_DTYPE_NAMES = frozenset({"float16", "float32"})

# Operators whose output element type equals the element type of the listed
# input, per the opset 18 type constraints. Used only to resolve the source
# dtype of a Cast when the graph carries no value_info; anything not listed
# here stays unresolved rather than being guessed.
#
# Every entry is read from the opset 18 specification, and only operators that
# exist in the default ONNX domain at opset 18 belong here -- the T20 graphs
# declare opset 18, so a later-opset operator cannot appear in them anyway. An
# earlier revision listed `Gelu`, which entered the default domain at opset 20;
# it is removed rather than annotated so the whole table has one provenance. If
# a later-opset operator ever earns a place, mark its opset inline and soften
# the claim above rather than leaving it silently mixed in.
TYPE_PRESERVING_OPS: Mapping[str, int] = MappingProxyType(
    {
        "Abs": 0,
        "Add": 0,
        "Clip": 0,
        "Concat": 0,
        "Cos": 0,
        "CumSum": 0,
        "Div": 0,
        "Dropout": 0,
        "Einsum": 0,
        "Erf": 0,
        "Exp": 0,
        "Expand": 0,
        "Flatten": 0,
        "Gather": 0,
        "GatherElements": 0,
        "GatherND": 0,
        "Identity": 0,
        "LayerNormalization": 0,
        "Log": 0,
        "MatMul": 0,
        "Max": 0,
        "Min": 0,
        "Mul": 0,
        "Neg": 0,
        "Pad": 0,
        "Pow": 0,
        "Range": 0,
        "Reciprocal": 0,
        "ReduceMax": 0,
        "ReduceMean": 0,
        "ReduceMin": 0,
        "ReduceSum": 0,
        "Relu": 0,
        "Reshape": 0,
        "Resize": 0,
        "ScatterElements": 0,
        "ScatterND": 0,
        "Sigmoid": 0,
        "Sin": 0,
        "Slice": 0,
        "Softmax": 0,
        "Split": 0,
        "Sqrt": 0,
        "Squeeze": 0,
        "Sub": 0,
        "Tanh": 0,
        "Tile": 0,
        "Transpose": 0,
        "Trilu": 0,
        "Unsqueeze": 0,
        "Where": 1,
    }
)

# Operators whose opset 18 output type constraint is a single concrete type,
# independent of the input types. The comparison and logical operators bind
# their output to `T1 : tensor(bool)`; `Shape` binds its output to
# `T1 : tensor(int64)`. Nothing here depends on what flows in, so resolving
# these is a reading of the specification rather than a guess -- the same
# "not guessed" promise that governs TYPE_PRESERVING_OPS.
FIXED_OUTPUT_TYPE_OPS: Mapping[str, str] = MappingProxyType(
    {
        "And": "bool",
        "Equal": "bool",
        "Greater": "bool",
        "GreaterOrEqual": "bool",
        "Less": "bool",
        "LessOrEqual": "bool",
        "Not": "bool",
        "Or": "bool",
        "Shape": "int64",
        "Xor": "bool",
    }
)

# How many *leading* outputs of a TYPE_PRESERVING_OPS node actually carry the
# propagated element type. Everything not listed here propagates to all of its
# outputs, which is correct for the single-output majority and for the ops whose
# every output is constrained to the same type variable (`Split`, for instance).
# The entries below have secondary outputs bound to a different type variable,
# so assigning them the input's dtype would be a guess, not a resolution.
TYPE_PRESERVING_OUTPUT_LIMITS: Mapping[str, int] = MappingProxyType(
    {
        # Dropout: output 0 is T, output 1 `mask` is bool (T2).
        "Dropout": 1,
        # LayerNormalization: output 0 is T, `Mean`/`InvStdDev` are U and are
        # commonly float32 even when Y is float16.
        "LayerNormalization": 1,
    }
)

_DTYPE_BYTES: Mapping[str, int] = MappingProxyType(
    {
        "bool": 1,
        "int8": 1,
        "uint8": 1,
        "int16": 2,
        "uint16": 2,
        "float16": 2,
        "bfloat16": 2,
        "int32": 4,
        "uint32": 4,
        "float32": 4,
        "int64": 8,
        "uint64": 8,
        "float64": 8,
    }
)


class GraphInspectionError(ValueError):
    """A risk catalogue, inspection request, or committed report is invalid."""


@dataclass(frozen=True)
class RiskRule:
    """One declarative deployment-risk rule from the committed catalogue."""

    id: str
    title: str
    category: str
    severity: str
    detector: str
    params: Mapping[str, Any]
    rationale: str
    mitigation: str
    references: tuple[str, ...]

    @property
    def severity_rank(self) -> int:
        return SEVERITY_RANKS.get(self.severity, len(SEVERITY_RANKS))


@dataclass(frozen=True)
class Finding:
    """One observed match of a risk rule against one graph."""

    rule_id: str
    title: str
    category: str
    severity: str
    count: int
    locations: tuple[str, ...]
    detail: str
    rationale: str
    mitigation: str

    @property
    def severity_rank(self) -> int:
        return SEVERITY_RANKS.get(self.severity, len(SEVERITY_RANKS))

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "title": self.title,
            "category": self.category,
            "severity": self.severity,
            "count": self.count,
            "locations": list(self.locations),
            "detail": self.detail,
            "rationale": self.rationale,
            "mitigation": self.mitigation,
        }


@dataclass(frozen=True)
class GraphInspection:
    """Structural summary and ranked findings for one named, hashed graph."""

    schema_version: int
    catalogue_id: str
    variant_id: str
    graph_kind: str
    source_relative_path: str
    source_sha256: str
    producer: str
    ir_version: int
    opset_imports: tuple[tuple[str, int], ...]
    node_count: int
    op_histogram: Mapping[str, int]
    input_count: int
    output_count: int
    initializer_count: int
    external_initializer_count: int
    largest_inline_initializer_bytes: int
    dynamic_dimensions: tuple[str, ...]
    findings: tuple[Finding, ...]

    @property
    def highest_severity(self) -> str:
        if not self.findings:
            return "none"
        return min(self.findings, key=lambda finding: finding.severity_rank).severity

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "catalogue_id": self.catalogue_id,
            "variant_id": self.variant_id,
            "graph_kind": self.graph_kind,
            "source_relative_path": self.source_relative_path,
            "source_sha256": self.source_sha256,
            "producer": self.producer,
            "ir_version": self.ir_version,
            "opset_imports": [
                [domain, version] for domain, version in self.opset_imports
            ],
            "node_count": self.node_count,
            "op_histogram": {
                op_type: self.op_histogram[op_type]
                for op_type in sorted(self.op_histogram)
            },
            "input_count": self.input_count,
            "output_count": self.output_count,
            "initializer_count": self.initializer_count,
            "external_initializer_count": self.external_initializer_count,
            "largest_inline_initializer_bytes": (self.largest_inline_initializer_bytes),
            "dynamic_dimensions": list(self.dynamic_dimensions),
            "highest_severity": self.highest_severity,
            "finding_count": len(self.findings),
            "findings": [finding.as_dict() for finding in self.findings],
        }


def _format_shape(shape: TensorShape | None) -> str:
    if shape is None:
        return "<no shape>"
    parts = ["?" if dimension is None else str(dimension) for dimension in shape.dims]
    return "[" + ", ".join(parts) + "]"


def _is_dynamic(shape: TensorShape | None) -> bool:
    return shape is None or not shape.is_static


def _node_location(node: NodeInfo) -> str:
    scope = f"{node.scope}/" if node.scope else ""
    domain = node.domain or "ai.onnx"
    name = node.name or "<unnamed>"
    return f"{scope}node[{node.index}] {domain}.{node.op_type} {name}"


def _value_location(label: str, value: ValueInfo) -> str:
    return f"{label}: {value.name} {value.dtype} {_format_shape(value.shape)}"


def _int_attribute(node: NodeInfo, name: str) -> int | None:
    for attribute in node.attributes:
        if attribute.name == name and attribute.i is not None:
            return int(attribute.i)
    return None


def _tensor_bytes(value: ValueInfo) -> int | None:
    element_bytes = _DTYPE_BYTES.get(value.dtype)
    if element_bytes is None or value.shape is None or not value.shape.is_static:
        return None
    elements = 1
    for dimension in value.shape.dims:
        elements *= int(dimension)  # type: ignore[arg-type]
    return elements * element_bytes


def _boundary_bytes(values: Sequence[ValueInfo]) -> tuple[int, int]:
    """Return (total sized bytes, number of tensors that could be sized)."""

    total = 0
    known = 0
    for value in values:
        size = _tensor_bytes(value)
        if size is None:
            continue
        total += size
        known += 1
    return total, known


def _make_finding(
    rule: RiskRule,
    *,
    count: int,
    locations: Sequence[str],
    detail: str,
    limit: int,
) -> Finding:
    return Finding(
        rule_id=rule.id,
        title=rule.title,
        category=rule.category,
        severity=rule.severity,
        count=count,
        locations=tuple(locations[:limit]),
        detail=detail,
        rationale=rule.rationale,
        mitigation=rule.mitigation,
    )


def _detect_op_types(
    summary: GraphSummary,
    rule: RiskRule,
    limit: int,
) -> Finding | None:
    watched = {str(op_type) for op_type in rule.params.get("op_types", ())}
    matches = [node for node in summary.nodes if node.op_type in watched]
    if not matches:
        return None
    counts: dict[str, int] = {}
    for node in matches:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    breakdown = ", ".join(f"{op_type}={counts[op_type]}" for op_type in sorted(counts))
    detail = (
        f"{len(matches)} of {len(summary.nodes)} nodes match the watched operator "
        f"set ({breakdown}); watched operators: {', '.join(sorted(watched))}."
    )
    return _make_finding(
        rule,
        count=len(matches),
        locations=[_node_location(node) for node in matches],
        detail=detail,
        limit=limit,
    )


def _detect_node_domain(
    summary: GraphSummary,
    rule: RiskRule,
    limit: int,
) -> Finding | None:
    allowed = {str(domain) for domain in rule.params.get("allowed_domains", ())}
    matches = [node for node in summary.nodes if node.domain not in allowed]
    if not matches:
        return None
    counts: dict[str, int] = {}
    for node in matches:
        counts[node.domain] = counts.get(node.domain, 0) + 1
    breakdown = ", ".join(f"{domain!r}={counts[domain]}" for domain in sorted(counts))
    detail = (
        f"{len(matches)} of {len(summary.nodes)} nodes declare an operator domain "
        f"outside {sorted(allowed)}: {breakdown}."
    )
    return _make_finding(
        rule,
        count=len(matches),
        locations=[_node_location(node) for node in matches],
        detail=detail,
        limit=limit,
    )


def _dynamic_entries(
    pairs: Sequence[tuple[str, Sequence[ValueInfo]]],
) -> tuple[list[str], int]:
    entries: list[str] = []
    total = 0
    for label, values in pairs:
        total += len(values)
        for value in values:
            if _is_dynamic(value.shape):
                entries.append(_value_location(label, value))
    return entries, total


def _detect_dynamic_boundary_dimension(
    summary: GraphSummary,
    rule: RiskRule,
    limit: int,
) -> Finding | None:
    entries, total = _dynamic_entries(
        (("input", summary.inputs), ("output", summary.outputs))
    )
    if not entries:
        return None
    detail = (
        f"{len(entries)} of {total} public boundary tensors "
        f"({len(summary.inputs)} inputs, {len(summary.outputs)} outputs) do not "
        "have a fully static shape."
    )
    return _make_finding(
        rule,
        count=len(entries),
        locations=entries,
        detail=detail,
        limit=limit,
    )


def _detect_dynamic_internal_dimension(
    summary: GraphSummary,
    rule: RiskRule,
    limit: int,
) -> Finding | None:
    entries, total = _dynamic_entries((("value_info", summary.value_info),))
    if not entries:
        return None
    detail = (
        f"{len(entries)} of {total} internal value_info entries do not have a "
        "fully static shape."
    )
    return _make_finding(
        rule,
        count=len(entries),
        locations=entries,
        detail=detail,
        limit=limit,
    )


def _detect_data_dependent_shape(
    summary: GraphSummary,
    rule: RiskRule,
    limit: int,
) -> Finding | None:
    raw_ops = rule.params.get("ops", {})
    watched: dict[str, tuple[int, ...]] = {
        str(op_type): tuple(int(index) for index in indices)
        for op_type, indices in dict(raw_ops).items()
    }
    constant_ops = {
        str(op_type)
        for op_type in rule.params.get("constant_ops", ("Constant", "ConstantOfShape"))
    }
    static_names: set[str] = {initializer.name for initializer in summary.initializers}
    static_names.update(value.name for value in summary.inputs)
    for node in summary.nodes:
        if node.op_type in constant_ops:
            static_names.update(name for name in node.outputs if name)

    entries: list[str] = []
    candidates = 0
    for node in summary.nodes:
        indices = watched.get(node.op_type)
        if not indices:
            continue
        for index in indices:
            if index >= len(node.inputs):
                continue
            name = node.inputs[index]
            if not name:
                continue
            candidates += 1
            if name in static_names:
                continue
            entries.append(f"{_node_location(node)} input[{index}]={name}")
    if not entries:
        return None
    detail = (
        f"{len(entries)} of {candidates} shape-defining operator inputs are "
        "produced by another node rather than by an initializer, a graph input, "
        f"or one of {sorted(constant_ops)}; watched operators: "
        f"{', '.join(f'{op}{list(watched[op])}' for op in sorted(watched))}."
    )
    return _make_finding(
        rule,
        count=len(entries),
        locations=entries,
        detail=detail,
        limit=limit,
    )


def _attribute_tensor(node: NodeInfo) -> Any:
    for attribute in node.attributes:
        if attribute.tensor is not None:
            return attribute.tensor
    return None


def resolve_element_types(summary: GraphSummary) -> dict[str, str]:
    """Resolve tensor element types by declaration, then by a forward pass.

    The T20 graphs carry no ``value_info``, so almost every intermediate tensor
    is undeclared. This pass seeds the declared boundary, initializer, and
    ``Constant`` types and then propagates them through the operators listed in
    ``TYPE_PRESERVING_OPS``, whose opset 18 type constraints tie the output
    element type to a specific input. ``FIXED_OUTPUT_TYPE_OPS`` covers the
    complementary case: operators whose opset 18 output constraint is a single
    concrete type no matter what flows in (``Equal``/``Greater`` and the other
    comparisons produce ``bool``; ``Shape`` produces ``int64``). Operators
    outside both tables are not guessed; their outputs simply stay unresolved.

    Propagation also respects *which* outputs the constraint covers:
    ``TYPE_PRESERVING_OUTPUT_LIMITS`` names the operators whose secondary
    outputs are bound to a different type variable (``Dropout``'s bool ``mask``,
    ``LayerNormalization``'s ``Mean``/``InvStdDev``), and those outputs are left
    unresolved rather than assigned the input's dtype.
    """

    dtypes: dict[str, str] = {}
    for value in (*summary.inputs, *summary.outputs, *summary.value_info):
        dtypes.setdefault(value.name, value.dtype)
    for initializer in summary.initializers:
        dtypes.setdefault(initializer.name, initializer.dtype)

    for node in summary.nodes:
        produced: str | None = None
        resolved_outputs = len(node.outputs)
        if node.op_type == "Cast":
            produced = ELEMENT_TYPE_NAMES.get(_int_attribute(node, "to"))
        elif node.op_type == "Constant":
            tensor = _attribute_tensor(node)
            produced = tensor.dtype if tensor is not None else None
        elif node.op_type == "ConstantOfShape":
            tensor = _attribute_tensor(node)
            # The opset 18 default fill value is a float32 zero.
            produced = tensor.dtype if tensor is not None else "float32"
        elif node.op_type in FIXED_OUTPUT_TYPE_OPS:
            produced = FIXED_OUTPUT_TYPE_OPS[node.op_type]
        else:
            index = TYPE_PRESERVING_OPS.get(node.op_type)
            if index is not None and index < len(node.inputs):
                produced = dtypes.get(node.inputs[index])
                resolved_outputs = TYPE_PRESERVING_OUTPUT_LIMITS.get(
                    node.op_type, resolved_outputs
                )
        if produced is None:
            continue
        for name in node.outputs[:resolved_outputs]:
            if name:
                dtypes.setdefault(name, produced)
    return dtypes


def _detect_precision_cast(
    summary: GraphSummary,
    rule: RiskRule,
    limit: int,
) -> Finding | None:
    dtypes = resolve_element_types(summary)

    entries: list[str] = []
    directions: dict[str, int] = {}
    total_casts = 0
    for node in summary.nodes:
        if node.op_type != "Cast":
            continue
        total_casts += 1
        target = FLOAT_ELEMENT_TYPES.get(_int_attribute(node, "to"))
        if target is None:
            continue
        source = dtypes.get(node.inputs[0]) if node.inputs else None
        if source in FLOAT_DTYPE_NAMES:
            if source == target:
                continue
            direction = f"{source}->{target}"
        elif source is None:
            direction = f"unknown->{target}"
        else:
            continue
        directions[direction] = directions.get(direction, 0) + 1
        entries.append(f"{_node_location(node)} {direction}")
    if not entries:
        return None
    breakdown = ", ".join(
        f"{direction}={directions[direction]}" for direction in sorted(directions)
    )
    unresolved = any(direction.startswith("unknown->") for direction in directions)
    detail = (
        f"{len(entries)} of {total_casts} Cast nodes cross or may cross the "
        f"float16/float32 boundary: {breakdown}."
    )
    if unresolved:
        detail += (
            " An 'unknown->' direction means the source element type could not "
            "be resolved from the graph's declarations, from type-preserving "
            "operators, or from operators whose opset 18 output type is fixed, "
            "so that crossing is unconfirmed rather than observed."
        )
    return _make_finding(
        rule,
        count=len(entries),
        locations=entries,
        detail=detail,
        limit=limit,
    )


def _detect_wide_output_boundary(
    summary: GraphSummary,
    rule: RiskRule,
    limit: int,
) -> Finding | None:
    max_inputs = int(rule.params.get("max_inputs", 0))
    max_outputs = int(rule.params.get("max_outputs", 0))
    exceeded: list[tuple[str, Sequence[ValueInfo]]] = []
    if len(summary.inputs) > max_inputs:
        exceeded.append(("input", summary.inputs))
    if len(summary.outputs) > max_outputs:
        exceeded.append(("output", summary.outputs))
    if not exceeded:
        return None
    locations = [
        _value_location(label, value) for label, values in exceeded for value in values
    ]
    input_bytes, input_known = _boundary_bytes(summary.inputs)
    output_bytes, output_known = _boundary_bytes(summary.outputs)
    detail = (
        f"graph boundary carries {len(summary.inputs)} input tensors "
        f"(limit {max_inputs}) and {len(summary.outputs)} output tensors "
        f"(limit {max_outputs}); {input_known}/{len(summary.inputs)} inputs size "
        f"to {input_bytes} bytes and {output_known}/{len(summary.outputs)} "
        f"outputs size to {output_bytes} bytes, so at least "
        f"{input_bytes + output_bytes} bytes cross the boundary per invocation."
    )
    return _make_finding(
        rule,
        count=len(locations),
        locations=locations,
        detail=detail,
        limit=limit,
    )


def _detect_large_inline_constant(
    summary: GraphSummary,
    rule: RiskRule,
    limit: int,
) -> Finding | None:
    try:
        max_bytes = int(rule.params["max_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GraphInspectionError(
            f"{rule.id}: large_inline_constant requires an integer max_bytes"
        ) from exc
    matches: list[tuple[int, str]] = []
    for initializer in summary.initializers:
        if initializer.external or initializer.inline_bytes <= max_bytes:
            continue
        matches.append(
            (
                initializer.inline_bytes,
                f"initializer: {initializer.name} {initializer.dtype} "
                f"{list(initializer.dims)} {initializer.inline_bytes} bytes",
            )
        )
    for node in summary.nodes:
        for attribute in node.attributes:
            tensor = attribute.tensor
            if tensor is None or tensor.external:
                continue
            if tensor.inline_bytes <= max_bytes:
                continue
            matches.append(
                (
                    tensor.inline_bytes,
                    f"{_node_location(node)} attribute {attribute.name} "
                    f"{tensor.dtype} {list(tensor.dims)} "
                    f"{tensor.inline_bytes} bytes",
                )
            )
    if not matches:
        return None
    matches.sort(key=lambda item: (-item[0], item[1]))
    total = sum(size for size, _ in matches)
    detail = (
        f"{len(matches)} inline constant tensors exceed {max_bytes} bytes; the "
        f"largest is {matches[0][0]} bytes and they total {total} bytes stored "
        "inside the graph protobuf rather than in external data."
    )
    return _make_finding(
        rule,
        count=len(matches),
        locations=[location for _, location in matches],
        detail=detail,
        limit=limit,
    )


def _detect_graph_scale(
    summary: GraphSummary,
    rule: RiskRule,
    limit: int,
) -> Finding | None:
    try:
        max_nodes = int(rule.params["max_nodes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GraphInspectionError(
            f"{rule.id}: graph_scale requires an integer max_nodes"
        ) from exc
    node_count = len(summary.nodes)
    if node_count <= max_nodes:
        return None
    histogram = summary.op_histogram
    ranked = sorted(histogram.items(), key=lambda item: (-item[1], item[0]))[:5]
    busiest = ", ".join(f"{op_type}={count}" for op_type, count in ranked)
    detail = (
        f"graph holds {node_count} nodes across {len(histogram)} distinct "
        f"operator types, above the review threshold of {max_nodes}; busiest "
        f"operators: {busiest}."
    )
    return _make_finding(
        rule,
        count=node_count,
        locations=[f"graph: {summary.graph_name or '<unnamed>'}"],
        detail=detail,
        limit=limit,
    )


def _detect_subgraph_present(
    summary: GraphSummary,
    rule: RiskRule,
    limit: int,
) -> Finding | None:
    matches = [
        node
        for node in summary.nodes
        if any(attribute.has_graph for attribute in node.attributes)
    ]
    if not matches:
        return None
    counts: dict[str, int] = {}
    for node in matches:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    breakdown = ", ".join(f"{op_type}={counts[op_type]}" for op_type in sorted(counts))
    detail = (
        f"{len(matches)} of {len(summary.nodes)} nodes carry at least one "
        f"GRAPH-typed attribute ({breakdown})."
    )
    return _make_finding(
        rule,
        count=len(matches),
        locations=[_node_location(node) for node in matches],
        detail=detail,
        limit=limit,
    )


Detector = Callable[[GraphSummary, RiskRule, int], "Finding | None"]

DETECTORS: Mapping[str, Detector] = MappingProxyType(
    {
        "op_types": _detect_op_types,
        "node_domain": _detect_node_domain,
        "dynamic_boundary_dimension": _detect_dynamic_boundary_dimension,
        "dynamic_internal_dimension": _detect_dynamic_internal_dimension,
        "data_dependent_shape": _detect_data_dependent_shape,
        "precision_cast": _detect_precision_cast,
        "wide_output_boundary": _detect_wide_output_boundary,
        "large_inline_constant": _detect_large_inline_constant,
        "graph_scale": _detect_graph_scale,
        "subgraph_present": _detect_subgraph_present,
    }
)

_REQUIRED_PARAMS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "op_types": ("op_types",),
        "node_domain": ("allowed_domains",),
        "data_dependent_shape": ("ops",),
        "wide_output_boundary": ("max_inputs", "max_outputs"),
        "large_inline_constant": ("max_bytes",),
        "graph_scale": ("max_nodes",),
    }
)


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GraphInspectionError(f"{label} must be a non-empty string")
    return value


def _parse_rule(payload: Any, index: int) -> RiskRule:
    label = f"rule[{index}]"
    if not isinstance(payload, dict):
        raise GraphInspectionError(f"{label} must be a JSON object")
    missing = [field for field in RULE_FIELDS if field not in payload]
    if missing:
        raise GraphInspectionError(f"{label} is missing fields: {sorted(missing)}")
    unexpected = sorted(set(payload) - set(RULE_FIELDS))
    if unexpected:
        raise GraphInspectionError(f"{label} has unexpected fields: {unexpected}")

    rule_id = _require_text(payload["id"], f"{label} id")
    label = f"rule {rule_id}"
    title = _require_text(payload["title"], f"{label} title")
    category = payload["category"]
    if category not in CATEGORIES:
        raise GraphInspectionError(
            f"{label} has unknown category {category!r}; "
            f"known categories: {sorted(CATEGORIES)}"
        )
    severity = payload["severity"]
    if severity not in SEVERITY_RANKS:
        raise GraphInspectionError(
            f"{label} has unknown severity {severity!r}; "
            f"known severities: {sorted(SEVERITY_RANKS)}"
        )
    detector = payload["detector"]
    if detector not in DETECTORS:
        raise GraphInspectionError(
            f"{label} names unknown detector {detector!r}; "
            f"known detectors: {sorted(DETECTORS)}"
        )
    params = payload["params"]
    if not isinstance(params, dict):
        raise GraphInspectionError(f"{label} params must be a JSON object")
    for required in _REQUIRED_PARAMS.get(detector, ()):
        if required not in params:
            raise GraphInspectionError(
                f"{label} detector {detector!r} requires param {required!r}"
            )
    references = payload["references"]
    if not isinstance(references, list) or not references:
        raise GraphInspectionError(f"{label} references must be a non-empty list")
    for position, reference in enumerate(references):
        _require_text(reference, f"{label} references[{position}]")
    return RiskRule(
        id=rule_id,
        title=title,
        category=category,
        severity=severity,
        detector=detector,
        params=MappingProxyType(dict(params)),
        rationale=_require_text(payload["rationale"], f"{label} rationale"),
        mitigation=_require_text(payload["mitigation"], f"{label} mitigation"),
        references=tuple(str(reference) for reference in references),
    )


def load_risk_rules(
    path: Path | str = DEFAULT_RULES_PATH,
) -> tuple[str, tuple[RiskRule, ...]]:
    """Load and validate the declarative risk catalogue at ``path``."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GraphInspectionError(
            f"cannot read risk catalogue {source}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise GraphInspectionError(
            f"invalid JSON in risk catalogue {source}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise GraphInspectionError(f"risk catalogue {source} must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise GraphInspectionError(
            f"risk catalogue {source} must declare schema_version "
            f"{SCHEMA_VERSION}, found {payload.get('schema_version')!r}"
        )
    catalogue_id = _require_text(payload.get("catalogue_id"), "catalogue_id")
    _require_text(payload.get("description"), "catalogue description")
    _require_text(payload.get("target_context"), "catalogue target_context")
    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise GraphInspectionError(
            f"risk catalogue {source} must define a non-empty rules array"
        )
    rules = tuple(_parse_rule(entry, index) for index, entry in enumerate(raw_rules))
    seen: set[str] = set()
    for rule in rules:
        if rule.id in seen:
            raise GraphInspectionError(f"duplicate rule id {rule.id!r} in {source}")
        seen.add(rule.id)
    return catalogue_id, rules


def rank_findings(findings: Iterable[Finding]) -> tuple[Finding, ...]:
    """Rank by deployment impact: severity, then count descending, then id."""

    return tuple(
        sorted(
            findings,
            key=lambda finding: (
                finding.severity_rank,
                -finding.count,
                finding.rule_id,
            ),
        )
    )


def _dynamic_dimension_entries(summary: GraphSummary) -> tuple[str, ...]:
    entries: list[str] = []
    for label, values in (
        ("input", summary.inputs),
        ("output", summary.outputs),
        ("value_info", summary.value_info),
    ):
        for value in values:
            if _is_dynamic(value.shape):
                entries.append(_value_location(label, value))
    if len(entries) > MAX_DYNAMIC_DIMENSION_ENTRIES:
        hidden = len(entries) - MAX_DYNAMIC_DIMENSION_ENTRIES
        entries = entries[:MAX_DYNAMIC_DIMENSION_ENTRIES]
        entries.append(f"... {hidden} further non-static tensors not listed")
    return tuple(entries)


def inspect_graph(
    summary: GraphSummary,
    *,
    variant_id: str,
    graph_kind: str,
    source_relative_path: str,
    source_sha256: str,
    rules: Sequence[RiskRule],
    catalogue_id: str,
    location_sample_limit: int = 8,
) -> GraphInspection:
    """Apply every rule to one hash-identified graph and rank what was found."""

    if graph_kind not in GRAPH_KINDS:
        raise GraphInspectionError(
            f"graph_kind must be one of {list(GRAPH_KINDS)}, found {graph_kind!r}"
        )
    if location_sample_limit < 1:
        raise GraphInspectionError("location_sample_limit must be at least 1")

    findings: list[Finding] = []
    for rule in rules:
        detector = DETECTORS.get(rule.detector)
        if detector is None:
            raise GraphInspectionError(
                f"rule {rule.id} names unknown detector {rule.detector!r}"
            )
        finding = detector(summary, rule, location_sample_limit)
        if finding is not None:
            findings.append(finding)

    inline_sizes = [
        initializer.inline_bytes
        for initializer in summary.initializers
        if not initializer.external
    ]
    producer = " ".join(
        part for part in (summary.producer_name, summary.producer_version) if part
    )
    return GraphInspection(
        schema_version=SCHEMA_VERSION,
        catalogue_id=catalogue_id,
        variant_id=variant_id,
        graph_kind=graph_kind,
        source_relative_path=source_relative_path,
        source_sha256=source_sha256,
        producer=producer or "unknown",
        ir_version=summary.ir_version,
        opset_imports=tuple(summary.opset_imports),
        node_count=len(summary.nodes),
        op_histogram=MappingProxyType(dict(summary.op_histogram)),
        input_count=len(summary.inputs),
        output_count=len(summary.outputs),
        initializer_count=len(summary.initializers),
        external_initializer_count=sum(
            1 for initializer in summary.initializers if initializer.external
        ),
        largest_inline_initializer_bytes=max(inline_sizes, default=0),
        dynamic_dimensions=_dynamic_dimension_entries(summary),
        findings=rank_findings(findings),
    )


def _sha256_file(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_artifact_root(explicit: str | Path | None = None) -> Path:
    """Resolve the external artifact root for the committed T20 graphs."""

    if explicit:
        root = Path(explicit).expanduser()
    else:
        configured = os.environ.get("SLM_LAB_ARTIFACT_ROOT", "").strip()
        root = (
            Path(configured).expanduser() if configured else PROJECT_ROOT / "artifacts"
        )
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    if not root.is_dir():
        raise GraphInspectionError(
            f"artifact root does not exist: {root}; set SLM_LAB_ARTIFACT_ROOT or "
            "pass --artifact-root"
        )
    return root


def _expand_artifact_root(template: Any, artifact_root: Path) -> Path:
    if not isinstance(template, str) or not template:
        raise GraphInspectionError("manifest artifacts.root must be a non-empty string")
    expanded = template.replace(ARTIFACT_ROOT_TOKEN, artifact_root.as_posix())
    directory = Path(expanded)
    if not directory.is_absolute():
        raise GraphInspectionError(
            f"manifest artifacts.root did not resolve to an absolute path: {expanded}"
        )
    return directory


def _safe_relative(directory: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise GraphInspectionError("manifest relative_path must be a non-empty string")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise GraphInspectionError(f"unsafe manifest relative_path {relative_path!r}")
    return directory.joinpath(*pure.parts)


def _manifest_record(manifest: Mapping[str, Any], graph_kind: str) -> Mapping[str, Any]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise GraphInspectionError("manifest has no artifacts object")
    record = artifacts.get(graph_kind)
    if not isinstance(record, dict):
        raise GraphInspectionError(f"manifest has no {graph_kind} artifact record")
    return record


def inspect_manifest(
    manifest_path: Path,
    *,
    rules: Sequence[RiskRule],
    catalogue_id: str,
    rules_path: Path,
    artifact_root: Path,
    graph_kinds: Sequence[str] = GRAPH_KINDS,
    location_sample_limit: int = 8,
) -> dict[str, Any]:
    """Hash-verify and inspect every selected graph named by one T20 manifest."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GraphInspectionError(
            f"cannot read manifest {manifest_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise GraphInspectionError(
            f"invalid JSON in manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise GraphInspectionError(f"manifest {manifest_path} must be a JSON object")

    variant_id = _require_text(manifest.get("variant_id"), "manifest variant_id")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise GraphInspectionError(f"manifest {manifest_path} has no artifacts object")
    directory = _expand_artifact_root(artifacts.get("root"), artifact_root)

    graphs: dict[str, Any] = {}
    for graph_kind in graph_kinds:
        record = _manifest_record(manifest, graph_kind)
        relative_path = record.get("relative_path")
        expected_sha256 = record.get("sha256")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise GraphInspectionError(
                f"{variant_id} {graph_kind}: manifest sha256 is not a digest"
            )
        graph_path = _safe_relative(directory, relative_path)
        if not graph_path.is_file():
            raise GraphInspectionError(
                f"{variant_id} {graph_kind}: graph file is missing: {graph_path}"
            )
        actual_sha256 = _sha256_file(graph_path)
        if actual_sha256 != expected_sha256:
            raise GraphInspectionError(
                f"{variant_id} {graph_kind}: SHA-256 mismatch for {graph_path}; "
                f"manifest records {expected_sha256}, file is {actual_sha256}"
            )
        summary = read_onnx_model(graph_path)
        inspection = inspect_graph(
            summary,
            variant_id=variant_id,
            graph_kind=graph_kind,
            source_relative_path=str(relative_path),
            source_sha256=actual_sha256,
            rules=rules,
            catalogue_id=catalogue_id,
            location_sample_limit=location_sample_limit,
        )
        graphs[graph_kind] = inspection.as_dict()

    return {
        "schema_version": SCHEMA_VERSION,
        "catalogue_id": catalogue_id,
        "task_id": "T21",
        "claim_boundary": {key: list(values) for key, values in CLAIM_BOUNDARY.items()},
        "variant_id": variant_id,
        "context_length": manifest.get("context_length"),
        "cache_capacity": manifest.get("cache_capacity"),
        "opset": manifest.get("opset"),
        "precision": manifest.get("precision"),
        "artifact_root_template": artifacts.get("root"),
        "source_manifest": {
            "path": _repository_relative(manifest_path),
            "sha256": _sha256_file(manifest_path),
        },
        "generated_by": {
            "module": MODULE_NAME,
            "schema_version": SCHEMA_VERSION,
            "rules_path": _repository_relative(rules_path),
            "rules_sha256": _sha256_file(rules_path),
        },
        "graphs": graphs,
    }


def _render(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"python -m {MODULE_NAME}",
        description=(
            "Inspect hash-verified T20 reference ONNX graphs for compiler and "
            "deployment risks and write a ranked, committable report."
        ),
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=None,
        help="committed T20 manifest to inspect; may be repeated",
    )
    parser.add_argument(
        "--all-manifests",
        default=None,
        help=f"directory of S*.json manifests (default {DEFAULT_MANIFEST_DIRECTORY})",
        nargs="?",
        const=str(DEFAULT_MANIFEST_DIRECTORY),
    )
    parser.add_argument(
        "--graph-kind",
        choices=(*GRAPH_KINDS, "both"),
        default="both",
        help="which graph of each variant to inspect (default both)",
    )
    parser.add_argument(
        "--rules",
        default=str(DEFAULT_RULES_PATH),
        help=f"risk catalogue path (default {DEFAULT_RULES_PATH})",
    )
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="external artifact root (default $SLM_LAB_ARTIFACT_ROOT, then ./artifacts)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "output path; only valid for a single manifest "
            f"(default {DEFAULT_OUTPUT_DIRECTORY}/<variant>.json)"
        ),
    )
    parser.add_argument(
        "--output-directory",
        default=str(DEFAULT_OUTPUT_DIRECTORY),
        help=f"directory for per-variant reports (default {DEFAULT_OUTPUT_DIRECTORY})",
    )
    parser.add_argument(
        "--location-sample-limit",
        type=int,
        default=8,
        help="maximum sampled locations recorded per finding (default 8)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-run and fail if the committed report would change",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _selected_manifests(args: argparse.Namespace) -> list[Path]:
    selected: list[Path] = [Path(value) for value in (args.manifest or ())]
    if args.all_manifests:
        directory = Path(args.all_manifests)
        if not directory.is_dir():
            raise GraphInspectionError(f"manifest directory not found: {directory}")
        found = sorted(directory.glob("S*.json"))
        if not found:
            raise GraphInspectionError(f"no S*.json manifests under {directory}")
        selected.extend(found)
    if not selected:
        raise GraphInspectionError(
            "select at least one manifest with --manifest or --all-manifests"
        )
    unique: dict[str, Path] = {}
    for path in selected:
        if not path.is_file():
            raise GraphInspectionError(f"manifest not found: {path}")
        unique.setdefault(str(path.resolve()), path)
    return list(unique.values())


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifests = _selected_manifests(args)
        if args.output and len(manifests) != 1:
            raise GraphInspectionError(
                "--output requires exactly one selected manifest; "
                f"{len(manifests)} were selected"
            )
        rules_path = Path(args.rules)
        catalogue_id, rules = load_risk_rules(rules_path)
        artifact_root = resolve_artifact_root(args.artifact_root)
        graph_kinds = GRAPH_KINDS if args.graph_kind == "both" else (args.graph_kind,)

        changed = False
        for manifest_path in manifests:
            payload = inspect_manifest(
                manifest_path,
                rules=rules,
                catalogue_id=catalogue_id,
                rules_path=rules_path,
                artifact_root=artifact_root,
                graph_kinds=graph_kinds,
                location_sample_limit=args.location_sample_limit,
            )
            text = _render(payload)
            if args.output:
                destination = Path(args.output)
            else:
                destination = (
                    Path(args.output_directory) / f"{payload['variant_id']}.json"
                )
            if args.check:
                if not destination.is_file():
                    print(f"missing report: {destination}", file=sys.stderr)
                    changed = True
                elif destination.read_text(encoding="utf-8") != text:
                    print(f"stale report: {destination}", file=sys.stderr)
                    changed = True
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
            print(f"wrote {destination}")
        if changed:
            return 1
    except (GraphInspectionError, OnnxReadError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
