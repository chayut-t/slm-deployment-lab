"""Tests for the declarative ONNX deployment-risk inspection engine."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import pytest

from slm_lab.graph import inspection
from slm_lab.graph.inspection import (
    CATEGORIES,
    DEFAULT_RULES_PATH,
    DETECTORS,
    SEVERITY_RANKS,
    Finding,
    GraphInspectionError,
    RiskRule,
    inspect_graph,
    load_risk_rules,
    rank_findings,
)
from slm_lab.graph.onnx_reader import (
    AttributeInfo,
    GraphSummary,
    InitializerInfo,
    NodeInfo,
    TensorShape,
    ValueInfo,
)


REPO_ROOT = inspection.PROJECT_ROOT
ELEMENT_TYPES = {
    "float32": 1,
    "uint8": 2,
    "int8": 3,
    "int32": 6,
    "int64": 7,
    "bool": 9,
    "float16": 10,
}


# --------------------------------------------------------------------------
# Fixture builders against the frozen slm_lab.graph.onnx_reader dataclasses.
# --------------------------------------------------------------------------


def make_value(
    name: str,
    *,
    dtype: str = "float32",
    dims: Sequence[int | str | None] | None = (1, 4),
) -> ValueInfo:
    shape = None if dims is None else TensorShape(dims=tuple(dims))
    return ValueInfo(
        name=name,
        elem_type=ELEMENT_TYPES.get(dtype, 0),
        dtype=dtype,
        shape=shape,
    )


def make_initializer(
    name: str,
    *,
    dtype: str = "float16",
    dims: Sequence[int] = (2, 2),
    external: bool = False,
    location: str | None = None,
    inline_bytes: int = 8,
) -> InitializerInfo:
    return InitializerInfo(
        name=name,
        elem_type=ELEMENT_TYPES.get(dtype, 0),
        dtype=dtype,
        dims=tuple(dims),
        external=external,
        external_location=location,
        inline_bytes=0 if external else inline_bytes,
    )


def make_attribute(
    name: str,
    *,
    type_name: str = "INT",
    i: int | None = None,
    tensor: InitializerInfo | None = None,
    has_graph: bool = False,
) -> AttributeInfo:
    return AttributeInfo(
        name=name,
        type=2 if type_name == "INT" else 0,
        type_name=type_name,
        i=i,
        f=None,
        s=None,
        ints=(),
        floats=(),
        tensor=tensor,
        has_graph=has_graph,
    )


def make_node(
    index: int,
    op_type: str,
    *,
    inputs: Sequence[str] = (),
    outputs: Sequence[str] = (),
    name: str = "",
    domain: str = "",
    scope: str = "",
    attributes: Sequence[AttributeInfo] = (),
) -> NodeInfo:
    return NodeInfo(
        index=index,
        scope=scope,
        op_type=op_type,
        name=name or f"{op_type.lower()}_{index}",
        domain=domain,
        inputs=tuple(inputs),
        outputs=tuple(outputs) or (f"{op_type.lower()}_{index}_out",),
        attributes=tuple(attributes),
    )


def make_summary(
    *,
    nodes: Sequence[NodeInfo] = (),
    inputs: Sequence[ValueInfo] = (),
    outputs: Sequence[ValueInfo] = (),
    value_info: Sequence[ValueInfo] = (),
    initializers: Sequence[InitializerInfo] = (),
    graph_name: str = "test_graph",
) -> GraphSummary:
    return GraphSummary(
        ir_version=9,
        producer_name="pytorch",
        producer_version="2.7.0",
        opset_imports=(("", 18),),
        graph_name=graph_name,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        value_info=tuple(value_info),
        initializers=tuple(initializers),
        nodes=tuple(nodes),
    )


def static_summary() -> GraphSummary:
    """A small, fully static, single-domain graph that must stay quiet."""

    return make_summary(
        inputs=(make_value("x", dims=(1, 8)), make_value("w", dims=(8, 8))),
        outputs=(make_value("y", dims=(1, 8)),),
        initializers=(make_initializer("bias", dims=(8,), inline_bytes=16),),
        nodes=(
            make_node(0, "MatMul", inputs=("x", "w"), outputs=("mm",)),
            make_node(1, "Add", inputs=("mm", "bias"), outputs=("y",)),
        ),
    )


def rule_for(detector: str, **params: Any) -> RiskRule:
    return RiskRule(
        id=f"R-{detector.upper().replace('_', '-')}",
        title=f"test rule for {detector}",
        category="dynamic_shape",
        severity="high",
        detector=detector,
        params=dict(params),
        rationale="test rationale",
        mitigation="test mitigation",
        references=("test reference",),
    )


def run_detector(detector: str, summary: GraphSummary, **params: Any) -> Finding | None:
    rule = rule_for(detector, **params)
    return DETECTORS[detector](summary, rule, 8)


def inspect_with(summary: GraphSummary, rules: Sequence[RiskRule]):
    return inspect_graph(
        summary,
        variant_id="S128",
        graph_kind="decode",
        source_relative_path="S128/decode.onnx",
        source_sha256="0" * 64,
        rules=rules,
        catalogue_id="test-catalogue",
    )


# --------------------------------------------------------------------------
# Committed catalogue.
# --------------------------------------------------------------------------


def test_committed_catalogue_loads() -> None:
    catalogue_id, rules = load_risk_rules()
    assert catalogue_id
    assert len(rules) >= 10
    assert DEFAULT_RULES_PATH.is_file()


def test_committed_catalogue_rules_are_well_formed() -> None:
    _, rules = load_risk_rules()
    for rule in rules:
        assert rule.id, "every rule needs an id"
        assert rule.title.strip()
        assert rule.category in CATEGORIES, rule.id
        assert rule.severity in SEVERITY_RANKS, rule.id
        assert rule.detector in DETECTORS, rule.id
        assert rule.rationale.strip(), rule.id
        assert rule.mitigation.strip(), rule.id
        assert rule.references, rule.id
        assert rule.severity_rank == SEVERITY_RANKS[rule.severity]


def test_committed_catalogue_rule_ids_are_unique() -> None:
    _, rules = load_risk_rules()
    ids = [rule.id for rule in rules]
    assert len(ids) == len(set(ids))


def test_committed_catalogue_covers_every_detector() -> None:
    _, rules = load_risk_rules()
    used = {rule.detector for rule in rules}
    assert used == set(DETECTORS), sorted(set(DETECTORS) - used)


def test_committed_catalogue_covers_the_expected_risk_areas() -> None:
    _, rules = load_risk_rules()
    assert {rule.category for rule in rules} == set(CATEGORIES)
    severities = {rule.severity for rule in rules}
    assert "blocking" in severities
    assert "high" in severities


# --------------------------------------------------------------------------
# Malformed catalogues.
# --------------------------------------------------------------------------


def valid_catalogue() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "catalogue_id": "test-catalogue",
        "description": "test catalogue",
        "target_context": "test target context",
        "rules": [
            {
                "id": "R-TEST",
                "title": "test rule",
                "category": "dynamic_shape",
                "severity": "high",
                "detector": "dynamic_boundary_dimension",
                "params": {},
                "rationale": "test rationale",
                "mitigation": "test mitigation",
                "references": ["test reference"],
            }
        ],
    }


def write_catalogue(tmp_path: Path, payload: Any, name: str = "rules.json") -> Path:
    destination = tmp_path / name
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return destination


def test_valid_written_catalogue_loads(tmp_path: Path) -> None:
    catalogue_id, rules = load_risk_rules(write_catalogue(tmp_path, valid_catalogue()))
    assert catalogue_id == "test-catalogue"
    assert len(rules) == 1


def test_unknown_detector_is_rejected(tmp_path: Path) -> None:
    payload = valid_catalogue()
    payload["rules"][0]["detector"] = "no_such_detector"
    with pytest.raises(GraphInspectionError, match="unknown detector"):
        load_risk_rules(write_catalogue(tmp_path, payload))


def test_unknown_severity_is_rejected(tmp_path: Path) -> None:
    payload = valid_catalogue()
    payload["rules"][0]["severity"] = "catastrophic"
    with pytest.raises(GraphInspectionError, match="unknown severity"):
        load_risk_rules(write_catalogue(tmp_path, payload))


def test_unknown_category_is_rejected(tmp_path: Path) -> None:
    payload = valid_catalogue()
    payload["rules"][0]["category"] = "vibes"
    with pytest.raises(GraphInspectionError, match="unknown category"):
        load_risk_rules(write_catalogue(tmp_path, payload))


def test_duplicate_rule_id_is_rejected(tmp_path: Path) -> None:
    payload = valid_catalogue()
    payload["rules"].append(dict(payload["rules"][0]))
    with pytest.raises(GraphInspectionError, match="duplicate rule id"):
        load_risk_rules(write_catalogue(tmp_path, payload))


def test_missing_rule_field_is_rejected(tmp_path: Path) -> None:
    payload = valid_catalogue()
    del payload["rules"][0]["mitigation"]
    with pytest.raises(GraphInspectionError, match="missing fields"):
        load_risk_rules(write_catalogue(tmp_path, payload))


def test_wrong_schema_version_is_rejected(tmp_path: Path) -> None:
    payload = valid_catalogue()
    payload["schema_version"] = 2
    with pytest.raises(GraphInspectionError, match="schema_version"):
        load_risk_rules(write_catalogue(tmp_path, payload))


def test_empty_rationale_is_rejected(tmp_path: Path) -> None:
    payload = valid_catalogue()
    payload["rules"][0]["rationale"] = "   "
    with pytest.raises(GraphInspectionError, match="rationale"):
        load_risk_rules(write_catalogue(tmp_path, payload))


def test_missing_required_detector_param_is_rejected(tmp_path: Path) -> None:
    payload = valid_catalogue()
    payload["rules"][0]["detector"] = "graph_scale"
    payload["rules"][0]["category"] = "graph_scale"
    with pytest.raises(GraphInspectionError, match="requires param"):
        load_risk_rules(write_catalogue(tmp_path, payload))


def test_missing_catalogue_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(GraphInspectionError, match="cannot read risk catalogue"):
        load_risk_rules(tmp_path / "absent.json")


def test_invalid_catalogue_json_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "rules.json"
    destination.write_text("{not json", encoding="utf-8")
    with pytest.raises(GraphInspectionError, match="invalid JSON"):
        load_risk_rules(destination)


# --------------------------------------------------------------------------
# Detectors: positive and negative fixtures.
# --------------------------------------------------------------------------


def test_op_types_detector_fires_and_stays_silent() -> None:
    positive = make_summary(
        nodes=(
            make_node(0, "If", inputs=("cond",)),
            make_node(1, "Add", inputs=("a", "b")),
            make_node(2, "Loop", inputs=("trip",)),
        )
    )
    finding = run_detector("op_types", positive, op_types=["If", "Loop", "Scan"])
    assert finding is not None
    assert finding.count == 2
    assert "If=1" in finding.detail and "Loop=1" in finding.detail
    assert any("If" in location for location in finding.locations)

    assert run_detector("op_types", static_summary(), op_types=["If", "Loop"]) is None


def test_node_domain_detector_fires_and_stays_silent() -> None:
    positive = make_summary(
        nodes=(
            make_node(0, "MatMul", inputs=("x", "w")),
            make_node(1, "FusedAttention", domain="com.vendor.ext", inputs=("x",)),
        )
    )
    finding = run_detector("node_domain", positive, allowed_domains=["", "ai.onnx"])
    assert finding is not None
    assert finding.count == 1
    assert "com.vendor.ext" in finding.detail

    negative = make_summary(
        nodes=(
            make_node(0, "MatMul", inputs=("x", "w")),
            make_node(1, "Add", domain="ai.onnx", inputs=("x", "y")),
        )
    )
    assert (
        run_detector("node_domain", negative, allowed_domains=["", "ai.onnx"]) is None
    )


def test_dynamic_boundary_dimension_detector() -> None:
    positive = make_summary(
        inputs=(
            make_value("input_ids", dtype="int64", dims=(1, "sequence")),
            make_value("mask", dtype="int64", dims=(1, 160)),
        ),
        outputs=(make_value("logits", dims=(1, None)),),
    )
    finding = run_detector("dynamic_boundary_dimension", positive)
    assert finding is not None
    assert finding.count == 2
    assert "2 of 3 public boundary tensors" in finding.detail
    assert any("input_ids" in location for location in finding.locations)

    assert run_detector("dynamic_boundary_dimension", static_summary()) is None


def test_dynamic_boundary_dimension_detector_treats_absent_shape_as_dynamic() -> None:
    summary = make_summary(
        inputs=(make_value("x", dims=None),),
        outputs=(make_value("y", dims=(1, 8)),),
    )
    finding = run_detector("dynamic_boundary_dimension", summary)
    assert finding is not None
    assert finding.count == 1
    assert "<no shape>" in finding.locations[0]


def test_dynamic_internal_dimension_detector() -> None:
    positive = make_summary(
        value_info=(
            make_value("hidden", dims=(1, "sequence", 1024)),
            make_value("proj", dims=(1, 128, 1024)),
        )
    )
    finding = run_detector("dynamic_internal_dimension", positive)
    assert finding is not None
    assert finding.count == 1
    assert "1 of 2 internal value_info entries" in finding.detail

    negative = make_summary(value_info=(make_value("proj", dims=(1, 128, 1024)),))
    assert run_detector("dynamic_internal_dimension", negative) is None


SHAPE_OPS = {"Reshape": [1], "Slice": [1, 2, 3, 4], "Expand": [1], "Tile": [1]}


def test_data_dependent_shape_is_silent_for_initializer_shape() -> None:
    summary = make_summary(
        inputs=(make_value("x", dims=(1, 8)),),
        initializers=(make_initializer("target_shape", dtype="int64", dims=(2,)),),
        nodes=(make_node(0, "Reshape", inputs=("x", "target_shape")),),
    )
    assert run_detector("data_dependent_shape", summary, ops=SHAPE_OPS) is None


def test_data_dependent_shape_is_silent_for_graph_input_shape() -> None:
    summary = make_summary(
        inputs=(
            make_value("x", dims=(1, 8)),
            make_value("target_shape", dtype="int64", dims=(2,)),
        ),
        nodes=(make_node(0, "Reshape", inputs=("x", "target_shape")),),
    )
    assert run_detector("data_dependent_shape", summary, ops=SHAPE_OPS) is None


def test_data_dependent_shape_is_silent_for_constant_shape() -> None:
    summary = make_summary(
        inputs=(make_value("x", dims=(1, 8)),),
        nodes=(
            make_node(0, "Constant", inputs=(), outputs=("target_shape",)),
            make_node(1, "Reshape", inputs=("x", "target_shape")),
            make_node(2, "ConstantOfShape", inputs=("target_shape",), outputs=("z",)),
            make_node(3, "Expand", inputs=("x", "z")),
        ),
    )
    assert run_detector("data_dependent_shape", summary, ops=SHAPE_OPS) is None


def test_data_dependent_shape_fires_for_computed_shape() -> None:
    summary = make_summary(
        inputs=(make_value("x", dims=(1, 8)),),
        nodes=(
            make_node(0, "Shape", inputs=("x",), outputs=("shape",)),
            make_node(1, "Gather", inputs=("shape", "idx"), outputs=("dim",)),
            make_node(2, "Concat", inputs=("dim",), outputs=("computed_shape",)),
            make_node(3, "Reshape", inputs=("x", "computed_shape")),
        ),
    )
    finding = run_detector("data_dependent_shape", summary, ops=SHAPE_OPS)
    assert finding is not None
    assert finding.count == 1
    assert "input[1]=computed_shape" in finding.locations[0]
    assert "1 of 1 shape-defining operator inputs" in finding.detail


def test_data_dependent_shape_ignores_absent_optional_inputs() -> None:
    summary = make_summary(
        inputs=(make_value("x", dims=(1, 8)),),
        initializers=(
            make_initializer("starts", dtype="int64", dims=(1,)),
            make_initializer("ends", dtype="int64", dims=(1,)),
        ),
        nodes=(make_node(0, "Slice", inputs=("x", "starts", "ends")),),
    )
    assert run_detector("data_dependent_shape", summary, ops=SHAPE_OPS) is None


def cast_node(index: int, source: str, to: int) -> NodeInfo:
    return make_node(
        index,
        "Cast",
        inputs=(source,),
        outputs=(f"{source}_cast",),
        attributes=(make_attribute("to", i=to),),
    )


def test_precision_cast_distinguishes_directions() -> None:
    summary = make_summary(
        inputs=(
            make_value("cache", dtype="float16", dims=(1, 8, 160, 128)),
            make_value("hidden", dtype="float32", dims=(1, 1, 1024)),
            make_value("tokens", dtype="int64", dims=(1, 1)),
        ),
        nodes=(
            cast_node(0, "cache", 1),
            cast_node(1, "hidden", 10),
            cast_node(2, "hidden", 10),
            cast_node(3, "tokens", 1),
            make_node(
                4,
                "Cast",
                inputs=("hidden",),
                outputs=("as_int",),
                attributes=(make_attribute("to", i=7),),
            ),
        ),
    )
    finding = run_detector("precision_cast", summary)
    assert finding is not None
    assert finding.count == 3
    assert "float16->float32=1" in finding.detail
    assert "float32->float16=2" in finding.detail
    assert "int64" not in finding.detail
    assert "5 Cast nodes" in finding.detail


def test_precision_cast_is_silent_without_float_crossings() -> None:
    summary = make_summary(
        inputs=(make_value("tokens", dtype="int64", dims=(1, 1)),),
        nodes=(
            cast_node(0, "tokens", 6),
            make_node(1, "Add", inputs=("tokens", "tokens")),
        ),
    )
    assert run_detector("precision_cast", summary) is None


def test_precision_cast_ignores_same_width_float_cast() -> None:
    summary = make_summary(
        inputs=(make_value("hidden", dtype="float32", dims=(1, 8)),),
        nodes=(cast_node(0, "hidden", 1),),
    )
    assert run_detector("precision_cast", summary) is None


def test_precision_cast_flags_unresolved_source_separately() -> None:
    summary = make_summary(nodes=(cast_node(0, "mystery", 10),))
    finding = run_detector("precision_cast", summary)
    assert finding is not None
    assert "unknown->float16=1" in finding.detail
    assert "unconfirmed rather than observed" in finding.detail


def test_precision_cast_resolves_source_through_type_preserving_ops() -> None:
    """The T20 graphs carry no value_info, so propagation is what resolves casts."""

    summary = make_summary(
        inputs=(make_value("cache", dtype="float16", dims=(1, 8, 160, 128)),),
        nodes=(
            make_node(0, "Transpose", inputs=("cache",), outputs=("t",)),
            make_node(1, "Reshape", inputs=("t", "shape"), outputs=("r",)),
            make_node(2, "MatMul", inputs=("r", "w"), outputs=("scores",)),
            cast_node(3, "scores", 1),
        ),
    )
    finding = run_detector("precision_cast", summary)
    assert finding is not None
    assert "float16->float32=1" in finding.detail
    assert "unknown" not in finding.detail.split(". ")[0]


def test_resolve_element_types_seeds_a_cast_chain_from_the_cast_target() -> None:
    """A `Cast` output is typed by its own `to` attribute, not by its source.

    This is the seeding step the rest of the resolver stands on, and the only
    thing that makes the real graphs' rotary-embedding chain resolvable:
    `rotary_emb/Cast_4` reads `Mul_1`, which reads `Cos`/`Sin`, which read two
    `Cast` nodes that declare float32 from an integer position tensor. Nothing
    upstream of those two casts is a float, so without seeding the whole chain
    stays unresolved and the crossing is reported `unknown->float16` instead of
    the observed `float32->float16`.
    """

    summary = make_summary(
        inputs=(make_value("positions", dtype="int64", dims=(1, 4)),),
        nodes=(
            cast_node(0, "positions", ELEMENT_TYPES["float32"]),
            make_node(
                1,
                "Mul",
                inputs=("positions_cast", "positions_cast"),
                outputs=("angles",),
            ),
            cast_node(2, "angles", ELEMENT_TYPES["float16"]),
        ),
    )

    dtypes = inspection.resolve_element_types(summary)
    assert dtypes["positions_cast"] == "float32"
    assert dtypes["angles"] == "float32"

    finding = run_detector("precision_cast", summary)
    assert finding is not None
    # The int64->float32 cast is not a float crossing and is excluded; only the
    # float32->float16 one at the end of the chain is counted.
    assert finding.count == 1
    assert "float32->float16=1" in finding.detail
    assert "unknown" not in finding.detail


def test_resolve_element_types_does_not_guess_unlisted_operators() -> None:
    summary = make_summary(
        inputs=(make_value("cache", dtype="float16", dims=(1, 4)),),
        nodes=(
            # `Loop` output types come from its body subgraph, which this pass
            # deliberately does not model, so they must stay unresolved.
            make_node(0, "Loop", inputs=("cache",), outputs=("looped",)),
            make_node(
                1,
                "Constant",
                outputs=("folded",),
                attributes=(
                    make_attribute(
                        "value",
                        type_name="TENSOR",
                        tensor=make_initializer("t", dtype="float32", dims=(2,)),
                    ),
                ),
            ),
            make_node(2, "ConstantOfShape", inputs=("shape",), outputs=("zeros",)),
        ),
    )
    dtypes = inspection.resolve_element_types(summary)
    assert dtypes["cache"] == "float16"
    assert dtypes["folded"] == "float32"
    assert dtypes["zeros"] == "float32"
    assert "looped" not in dtypes


def test_resolve_element_types_reads_fixed_opset_output_types() -> None:
    """Opset 18 fixes these output types regardless of what flows in.

    `Equal`, `Greater` and the other comparisons bind their output to
    `tensor(bool)` and `Shape` binds its output to `tensor(int64)`, so
    resolving them is reading the specification, not guessing. `Cos` and `Sin`
    are `T -> T` like the rest of `TYPE_PRESERVING_OPS`; they matter because
    the rotary-embedding prologue casts their downstream product.
    """

    summary = make_summary(
        inputs=(
            make_value("angles", dtype="float32", dims=(1, 4)),
            make_value("positions", dtype="int64", dims=(1, 4)),
        ),
        nodes=(
            make_node(0, "Cos", inputs=("angles",), outputs=("cos",)),
            make_node(1, "Sin", inputs=("angles",), outputs=("sin",)),
            make_node(2, "Mul", inputs=("cos", "sin"), outputs=("product",)),
            make_node(3, "Shape", inputs=("angles",), outputs=("shape",)),
            make_node(4, "Range", inputs=("positions",), outputs=("range",)),
            make_node(5, "Greater", inputs=("range", "positions"), outputs=("mask",)),
            make_node(6, "Equal", inputs=("positions", "positions"), outputs=("same",)),
            make_node(7, "Not", inputs=("same",), outputs=("negated",)),
        ),
    )

    dtypes = inspection.resolve_element_types(summary)

    assert dtypes["cos"] == "float32"
    assert dtypes["sin"] == "float32"
    assert dtypes["product"] == "float32"
    assert dtypes["shape"] == "int64"
    assert dtypes["range"] == "int64"
    assert dtypes["mask"] == "bool"
    assert dtypes["same"] == "bool"
    assert dtypes["negated"] == "bool"


def test_precision_cast_excludes_a_cast_from_a_bool_producing_operator() -> None:
    """A `Greater -> Cast(float16)` is a bool widening, not a float crossing.

    This is the decode graph's `Cast_3947`: before `Greater` was resolved it
    was counted as `unknown->float16` and inflated the finding by one.
    """

    summary = make_summary(
        inputs=(
            make_value("positions", dtype="int64", dims=(1, 4)),
            make_value("hidden", dtype="float32", dims=(1, 4)),
        ),
        nodes=(
            make_node(0, "Greater", inputs=("positions", "positions"), outputs=("m",)),
            cast_node(1, "m", 10),
            cast_node(2, "hidden", 10),
        ),
    )

    finding = run_detector("precision_cast", summary)

    assert finding is not None
    assert finding.count == 1
    assert "float32->float16=1" in finding.detail
    assert "unknown" not in finding.detail


def test_resolve_element_types_does_not_type_unconstrained_extra_outputs() -> None:
    """Only the outputs the opset ties to the named input may be resolved.

    `Dropout`'s second output is a bool mask and `LayerNormalization`'s `Mean`
    and `InvStdDev` are type `U` -- commonly float32 while `Y` is float16.
    Assigning them the input's dtype would be a guess, and the documented
    promise is that anything not constrained is left unresolved.
    """

    summary = make_summary(
        inputs=(make_value("x", dtype="float16", dims=(1, 8)),),
        nodes=(
            make_node(0, "Dropout", inputs=("x",), outputs=("y", "mask")),
            make_node(
                1,
                "LayerNormalization",
                inputs=("y", "scale"),
                outputs=("normed", "mean", "inv_std_dev"),
            ),
            # A multi-output operator whose every output really is type T.
            make_node(2, "Split", inputs=("normed",), outputs=("left", "right")),
        ),
    )

    dtypes = inspection.resolve_element_types(summary)

    assert dtypes["y"] == "float16"
    assert "mask" not in dtypes
    assert dtypes["normed"] == "float16"
    assert "mean" not in dtypes
    assert "inv_std_dev" not in dtypes
    assert dtypes["left"] == "float16"
    assert dtypes["right"] == "float16"


def test_unconstrained_extra_outputs_are_reported_as_unresolved_casts() -> None:
    """The consequence for the detector: `unknown->`, never a wrong direction."""

    summary = make_summary(
        inputs=(make_value("x", dtype="float16", dims=(1, 8)),),
        nodes=(
            make_node(0, "Dropout", inputs=("x",), outputs=("y", "mask")),
            cast_node(1, "mask", 1),
            cast_node(2, "y", 1),
        ),
    )

    finding = run_detector("precision_cast", summary)

    assert finding is not None
    assert "float16->float32=1" in finding.detail
    assert "unknown->float32=1" in finding.detail


def test_wide_output_boundary_reports_counts_and_bytes() -> None:
    inputs = [make_value("input_ids", dtype="int64", dims=(1, 1))]
    inputs += [
        make_value(f"key_cache.{layer}", dtype="float16", dims=(1, 8, 160, 128))
        for layer in range(4)
    ]
    outputs = [make_value("next_logits", dtype="float32", dims=(1, 151936))]
    outputs += [
        make_value(f"present_key.{layer}", dtype="float16", dims=(1, 8, 160, 128))
        for layer in range(4)
    ]
    summary = make_summary(inputs=inputs, outputs=outputs)
    finding = run_detector("wide_output_boundary", summary, max_inputs=3, max_outputs=3)
    assert finding is not None
    assert finding.count == len(inputs) + len(outputs)
    expected_cache_bytes = 4 * 1 * 8 * 160 * 128 * 2
    assert f"{expected_cache_bytes + 8} bytes" in finding.detail
    assert "5 input tensors" in finding.detail
    assert "5 output tensors" in finding.detail

    assert (
        run_detector("wide_output_boundary", summary, max_inputs=16, max_outputs=16)
        is None
    )


def test_wide_output_boundary_fires_on_outputs_only() -> None:
    summary = make_summary(
        inputs=(make_value("x", dims=(1, 8)),),
        outputs=tuple(make_value(f"y{i}", dims=(1, 8)) for i in range(5)),
    )
    finding = run_detector("wide_output_boundary", summary, max_inputs=3, max_outputs=3)
    assert finding is not None
    assert finding.count == 5
    assert all(location.startswith("output:") for location in finding.locations)


def test_large_inline_constant_covers_initializers_and_attributes() -> None:
    big_tensor = make_initializer(
        "mask_const",
        dtype="float16",
        dims=(4096, 4096),
        inline_bytes=4096 * 4096 * 2,
    )
    summary = make_summary(
        initializers=(
            make_initializer("small", inline_bytes=64),
            make_initializer(
                "external_weight",
                external=True,
                location="prefill.onnx.data",
                dims=(1024, 1024),
            ),
            make_initializer("medium", dims=(512, 512), inline_bytes=512 * 512 * 2),
        ),
        nodes=(
            make_node(
                0,
                "Constant",
                outputs=("mask",),
                attributes=(
                    make_attribute("value", type_name="TENSOR", tensor=big_tensor),
                ),
            ),
        ),
    )
    finding = run_detector("large_inline_constant", summary, max_bytes=262144)
    assert finding is not None
    assert finding.count == 2
    assert finding.locations[0].startswith("node[0]")
    assert f"largest is {4096 * 4096 * 2} bytes" in finding.detail
    assert "external_weight" not in " ".join(finding.locations)

    quiet = make_summary(initializers=(make_initializer("small", inline_bytes=64),))
    assert run_detector("large_inline_constant", quiet, max_bytes=262144) is None


def test_graph_scale_detector() -> None:
    nodes = tuple(make_node(index, "Add", inputs=("a", "b")) for index in range(12))
    summary = make_summary(nodes=nodes)
    finding = run_detector("graph_scale", summary, max_nodes=10)
    assert finding is not None
    assert finding.count == 12
    assert "12 nodes" in finding.detail
    assert "Add=12" in finding.detail

    assert run_detector("graph_scale", summary, max_nodes=12) is None


def test_graph_scale_requires_integer_param() -> None:
    rule = rule_for("graph_scale")
    with pytest.raises(GraphInspectionError, match="max_nodes"):
        DETECTORS["graph_scale"](make_summary(), rule, 8)


def test_subgraph_present_detector() -> None:
    positive = make_summary(
        nodes=(
            make_node(
                0,
                "If",
                inputs=("cond",),
                attributes=(
                    make_attribute("then_branch", type_name="GRAPH", has_graph=True),
                ),
            ),
            make_node(1, "Add", inputs=("a", "b")),
        )
    )
    finding = run_detector("subgraph_present", positive)
    assert finding is not None
    assert finding.count == 1
    assert "If=1" in finding.detail

    assert run_detector("subgraph_present", static_summary()) is None


def test_location_sample_limit_caps_locations() -> None:
    nodes = tuple(make_node(index, "If", inputs=("c",)) for index in range(20))
    rule = rule_for("op_types", op_types=["If"])
    finding = DETECTORS["op_types"](make_summary(nodes=nodes), rule, 3)
    assert finding is not None
    assert finding.count == 20
    assert len(finding.locations) == 3


# --------------------------------------------------------------------------
# Ranking.
# --------------------------------------------------------------------------


def finding_for(rule_id: str, severity: str, count: int) -> Finding:
    return Finding(
        rule_id=rule_id,
        title=rule_id,
        category="dynamic_shape",
        severity=severity,
        count=count,
        locations=(),
        detail="detail",
        rationale="rationale",
        mitigation="mitigation",
    )


def test_rank_findings_orders_by_severity_then_count_then_id() -> None:
    unordered = [
        finding_for("R-MED-A", "medium", 100),
        finding_for("R-HIGH-B", "high", 2),
        finding_for("R-HIGH-A", "high", 2),
        finding_for("R-HIGH-C", "high", 9),
        finding_for("R-BLOCK", "blocking", 1),
        finding_for("R-LOW", "low", 5000),
        finding_for("R-INFO", "informational", 1),
    ]
    ranked = rank_findings(unordered)
    assert [finding.rule_id for finding in ranked] == [
        "R-BLOCK",
        "R-HIGH-C",
        "R-HIGH-A",
        "R-HIGH-B",
        "R-MED-A",
        "R-LOW",
        "R-INFO",
    ]


# --------------------------------------------------------------------------
# inspect_graph and serialization.
# --------------------------------------------------------------------------


def test_static_graph_produces_no_findings_above_informational() -> None:
    _, rules = load_risk_rules()
    result = inspect_with(static_summary(), rules)
    severe = [
        finding
        for finding in result.findings
        if finding.severity_rank < SEVERITY_RANKS["informational"]
    ]
    assert severe == [], [finding.rule_id for finding in severe]
    assert result.highest_severity == "none"
    assert result.dynamic_dimensions == ()
    assert result.node_count == 2
    assert result.op_histogram == {"MatMul": 1, "Add": 1}


def test_inspect_graph_records_structure_and_provenance() -> None:
    _, rules = load_risk_rules()
    summary = make_summary(
        inputs=(make_value("input_ids", dtype="int64", dims=(1, "sequence")),),
        outputs=(make_value("logits", dims=(1, 151936)),),
        value_info=(make_value("hidden", dims=(1, None, 1024)),),
        initializers=(
            make_initializer("w", external=True, location="decode.onnx.data"),
            make_initializer("b", inline_bytes=4096),
        ),
        nodes=(make_node(0, "MatMul", inputs=("input_ids", "w")),),
    )
    result = inspect_with(summary, rules)
    assert result.variant_id == "S128"
    assert result.graph_kind == "decode"
    assert result.source_relative_path == "S128/decode.onnx"
    assert result.source_sha256 == "0" * 64
    assert result.producer == "pytorch 2.7.0"
    assert result.opset_imports == (("", 18),)
    assert result.initializer_count == 2
    assert result.external_initializer_count == 1
    assert result.largest_inline_initializer_bytes == 4096
    assert len(result.dynamic_dimensions) == 2
    assert result.highest_severity == "blocking"
    assert result.findings[0].rule_id == "R-BOUNDARY-DYNAMIC-SHAPE"


def test_inspect_graph_rejects_unknown_graph_kind() -> None:
    _, rules = load_risk_rules()
    with pytest.raises(GraphInspectionError, match="graph_kind"):
        inspect_graph(
            static_summary(),
            variant_id="S128",
            graph_kind="quantize",
            source_relative_path="S128/decode.onnx",
            source_sha256="0" * 64,
            rules=rules,
            catalogue_id="test",
        )


def test_inspect_graph_rejects_unknown_detector_without_silently_skipping() -> None:
    rogue = RiskRule(
        id="R-ROGUE",
        title="rogue",
        category="dynamic_shape",
        severity="high",
        detector="not_registered",
        params={},
        rationale="r",
        mitigation="m",
        references=("ref",),
    )
    with pytest.raises(GraphInspectionError, match="unknown detector"):
        inspect_with(static_summary(), (rogue,))


def test_inspection_as_dict_is_json_stable() -> None:
    _, rules = load_risk_rules()
    summary = make_summary(
        inputs=(make_value("input_ids", dtype="int64", dims=(1, "sequence")),),
        outputs=tuple(
            make_value(f"present.{layer}", dtype="float16", dims=(1, 8, 160, 128))
            for layer in range(20)
        ),
        nodes=(
            make_node(0, "ScatterND", inputs=("cache", "idx", "upd")),
            make_node(1, "Softmax", inputs=("scores",)),
        ),
    )
    first = json.dumps(inspect_with(summary, rules).as_dict(), sort_keys=True)
    second = json.dumps(inspect_with(summary, rules).as_dict(), sort_keys=True)
    assert first == second
    payload = json.loads(first)
    assert payload["schema_version"] == 1
    assert payload["finding_count"] == len(payload["findings"])
    rule_ids = [finding["rule_id"] for finding in payload["findings"]]
    assert "R-SCATTER-GATHER-INDEXING" in rule_ids
    assert "R-WIDE-IO-BOUNDARY" in rule_ids


# --------------------------------------------------------------------------
# Command line interface.
# --------------------------------------------------------------------------


FAKE_GRAPH_BYTES = b"\x08\x09fake-onnx-protobuf-payload-for-hash-verification"


def build_fake_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create an artifact root, a fake decode graph, and a manifest for it."""

    artifact_root = tmp_path / "artifact-root"
    graph_path = artifact_root / "onnx/reference/T20/S128/decode.onnx"
    graph_path.parent.mkdir(parents=True)
    graph_path.write_bytes(FAKE_GRAPH_BYTES)
    manifest = {
        "variant_id": "S128",
        "context_length": 128,
        "cache_capacity": 160,
        "opset": 18,
        "precision": "float16",
        "artifacts": {
            "root": "${SLM_LAB_ARTIFACT_ROOT}/onnx/reference/T20",
            "decode": {
                "relative_path": "S128/decode.onnx",
                "sha256": hashlib.sha256(FAKE_GRAPH_BYTES).hexdigest(),
            },
        },
    }
    manifest_path = tmp_path / "S128.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return artifact_root, graph_path, manifest_path


@pytest.fixture()
def fake_reader(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Replace the protobuf reader; the SHA-256 check still runs on real bytes."""

    seen: list[Path] = []

    def _read(path: Path, **_: Any) -> GraphSummary:
        seen.append(Path(path))
        return make_summary(
            inputs=(make_value("input_ids", dtype="int64", dims=(1, 1)),),
            outputs=(make_value("next_logits", dims=(1, 151936)),),
            nodes=(make_node(0, "ScatterND", inputs=("cache", "idx", "upd")),),
        )

    monkeypatch.setattr(inspection, "read_onnx_model", _read)
    return seen


def cli_arguments(
    manifest_path: Path,
    artifact_root: Path,
    output: Path,
    *extra: str,
) -> list[str]:
    return [
        "--manifest",
        str(manifest_path),
        "--graph-kind",
        "decode",
        "--artifact-root",
        str(artifact_root),
        "--output",
        str(output),
        *extra,
    ]


def test_cli_writes_expected_report(tmp_path: Path, fake_reader: list[Path]) -> None:
    artifact_root, graph_path, manifest_path = build_fake_artifacts(tmp_path)
    output = tmp_path / "out" / "S128.json"
    assert inspection.main(cli_arguments(manifest_path, artifact_root, output)) == 0
    assert fake_reader == [graph_path]

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["variant_id"] == "S128"
    assert payload["context_length"] == 128
    assert payload["cache_capacity"] == 160
    assert payload["catalogue_id"]
    assert set(payload["graphs"]) == {"decode"}
    decode = payload["graphs"]["decode"]
    assert decode["source_sha256"] == hashlib.sha256(FAKE_GRAPH_BYTES).hexdigest()
    assert decode["source_relative_path"] == "S128/decode.onnx"
    assert payload["generated_by"]["module"] == "slm_lab.graph.inspection"
    assert (
        payload["generated_by"]["rules_sha256"]
        == hashlib.sha256(DEFAULT_RULES_PATH.read_bytes()).hexdigest()
    )
    assert "created_at" not in payload
    assert "timestamp" not in json.dumps(payload)
    assert output.read_text(encoding="utf-8").endswith("}\n")


def test_report_states_its_claim_boundary_in_band(
    tmp_path: Path, fake_reader: list[Path]
) -> None:
    """A report that prints `severity: blocking` must say what produced it.

    The severities are review judgements and no compiler ran; a reader who only
    ever sees this JSON has no other place to learn that. Mirrors the
    `claim_boundary` block T20 writes into its manifests.
    """

    artifact_root, _, manifest_path = build_fake_artifacts(tmp_path)
    output = tmp_path / "out" / "S128.json"
    assert inspection.main(cli_arguments(manifest_path, artifact_root, output)) == 0

    boundary = json.loads(output.read_text(encoding="utf-8"))["claim_boundary"]

    assert set(boundary) == {"establishes", "does_not_establish"}
    assert boundary["establishes"]
    assert boundary["does_not_establish"]
    assert all(
        isinstance(entry, str) and entry
        for entries in boundary.values()
        for entry in entries
    )
    assert "compiler_acceptance" in boundary["does_not_establish"]
    assert (
        "severity_derived_from_an_executed_compile_or_conversion_job"
        in boundary["does_not_establish"]
    )
    # The two lists must not contradict each other.
    assert not set(boundary["establishes"]) & set(boundary["does_not_establish"])


def test_every_committed_report_carries_the_claim_boundary() -> None:
    reports = sorted((REPO_ROOT / "results/graph").glob("S*.json"))
    assert len(reports) == 4
    for report in reports:
        payload = json.loads(report.read_text(encoding="utf-8"))
        boundary = payload.get("claim_boundary")
        assert boundary, f"{report.name} has no claim_boundary"
        assert boundary["establishes"] == list(
            inspection.CLAIM_BOUNDARY["establishes"]
        ), report.name
        assert boundary["does_not_establish"] == list(
            inspection.CLAIM_BOUNDARY["does_not_establish"]
        ), report.name


def test_cli_check_detects_drift(tmp_path: Path, fake_reader: list[Path]) -> None:
    artifact_root, _, manifest_path = build_fake_artifacts(tmp_path)
    output = tmp_path / "out" / "S128.json"
    assert inspection.main(cli_arguments(manifest_path, artifact_root, output)) == 0
    assert (
        inspection.main(cli_arguments(manifest_path, artifact_root, output, "--check"))
        == 0
    )
    output.write_text("{}\n", encoding="utf-8")
    assert (
        inspection.main(cli_arguments(manifest_path, artifact_root, output, "--check"))
        == 1
    )


def test_cli_check_reports_missing_report(
    tmp_path: Path, fake_reader: list[Path]
) -> None:
    artifact_root, _, manifest_path = build_fake_artifacts(tmp_path)
    output = tmp_path / "out" / "S128.json"
    assert (
        inspection.main(cli_arguments(manifest_path, artifact_root, output, "--check"))
        == 1
    )
    assert not output.exists()


def test_cli_rejects_sha256_mismatch(
    tmp_path: Path,
    fake_reader: list[Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact_root, graph_path, manifest_path = build_fake_artifacts(tmp_path)
    graph_path.write_bytes(FAKE_GRAPH_BYTES + b"-tampered")
    output = tmp_path / "out" / "S128.json"
    assert inspection.main(cli_arguments(manifest_path, artifact_root, output)) == 1
    assert not output.exists()
    assert fake_reader == []
    message = capsys.readouterr().err
    assert "SHA-256 mismatch" in message


def test_cli_rejects_missing_graph_file(
    tmp_path: Path,
    fake_reader: list[Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact_root, graph_path, manifest_path = build_fake_artifacts(tmp_path)
    graph_path.unlink()
    output = tmp_path / "out" / "S128.json"
    assert inspection.main(cli_arguments(manifest_path, artifact_root, output)) == 1
    assert not output.exists()
    message = capsys.readouterr().err
    assert "graph file is missing" in message


def test_cli_rejects_missing_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert inspection.main(["--manifest", str(tmp_path / "absent.json")]) == 1
    assert "manifest not found" in capsys.readouterr().err


def test_cli_requires_a_selection(capsys: pytest.CaptureFixture[str]) -> None:
    assert inspection.main([]) == 1
    assert "select at least one manifest" in capsys.readouterr().err


def test_cli_rejects_output_with_multiple_manifests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, _, manifest_path = build_fake_artifacts(tmp_path)
    second = tmp_path / "S512.json"
    second.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    exit_code = inspection.main(
        [
            "--manifest",
            str(manifest_path),
            "--manifest",
            str(second),
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert exit_code == 1
    assert "--output requires exactly one" in capsys.readouterr().err


def test_parser_defaults() -> None:
    args = inspection.parse_args(["--manifest", "results/manifests/onnx/S128.json"])
    assert args.graph_kind == "both"
    assert args.manifest == ["results/manifests/onnx/S128.json"]
    assert args.rules == str(inspection.DEFAULT_RULES_PATH)
    assert args.check is False
    assert args.location_sample_limit == 8


def test_parser_accepts_repeated_manifests_and_all() -> None:
    args = inspection.parse_args(
        ["--manifest", "a.json", "--manifest", "b.json", "--all-manifests"]
    )
    assert args.manifest == ["a.json", "b.json"]
    assert args.all_manifests == str(inspection.DEFAULT_MANIFEST_DIRECTORY)


def test_unsafe_relative_path_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact_root, _, manifest_path = build_fake_artifacts(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["decode"]["relative_path"] = "../../escape.onnx"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "out.json"
    assert inspection.main(cli_arguments(manifest_path, artifact_root, output)) == 1
    assert "unsafe manifest relative_path" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Real T20 graph, guarded.
# --------------------------------------------------------------------------


def real_artifact_root() -> Path:
    configured = os.environ.get("SLM_LAB_ARTIFACT_ROOT", "").strip()
    return Path(configured) if configured else REPO_ROOT / "artifacts"


def test_real_s128_decode_graph_inspects_end_to_end() -> None:
    manifest_path = REPO_ROOT / "results/manifests/onnx/S128.json"
    if not manifest_path.is_file():
        pytest.skip("committed S128 manifest is unavailable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["artifacts"]["decode"]
    graph_path = real_artifact_root() / "onnx/reference/T20" / record["relative_path"]
    if not graph_path.is_file():
        pytest.skip(f"T20 reference graph is unavailable: {graph_path}")

    catalogue_id, rules = load_risk_rules()
    summary = inspection.read_onnx_model(graph_path)
    assert summary.nodes, "the reference decode graph must contain nodes"
    manifest_inputs = {tensor["name"] for tensor in record["input_tensors"]}
    manifest_outputs = {tensor["name"] for tensor in record["output_tensors"]}
    assert manifest_inputs <= {value.name for value in summary.inputs}
    assert manifest_outputs <= {value.name for value in summary.outputs}

    digest = inspection._sha256_file(graph_path)
    assert digest == record["sha256"]

    result = inspect_graph(
        summary,
        variant_id=manifest["variant_id"],
        graph_kind="decode",
        source_relative_path=record["relative_path"],
        source_sha256=digest,
        rules=rules,
        catalogue_id=catalogue_id,
    )
    assert result.node_count == len(summary.nodes)
    assert result.node_count > 0
    assert sum(result.op_histogram.values()) == result.node_count
    ranks = [finding.severity_rank for finding in result.findings]
    assert ranks == sorted(ranks)
    payload = result.as_dict()
    assert json.dumps(payload, sort_keys=True)
    assert payload["source_sha256"] == record["sha256"]
