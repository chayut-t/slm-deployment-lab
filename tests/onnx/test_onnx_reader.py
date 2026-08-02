"""Tests for the dependency-free ONNX protobuf reader.

Fixture models are encoded byte by byte with the tiny protobuf writer helpers
defined below, so these tests need neither ``onnx``/``protobuf`` nor any model
weights. One test additionally reads a real T20 export when it is present on
the machine, and skips with a clear reason otherwise.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import pytest

from slm_lab.graph.onnx_reader import (
    GraphSummary,
    OnnxReadError,
    TensorShape,
    parse_onnx_model,
    read_onnx_model,
)

# ---------------------------------------------------------------------------
# Minimal protobuf writer helpers (test-only)
# ---------------------------------------------------------------------------


def _varint(value: int) -> bytes:
    """Encode an int64 as an unsigned base-128 varint (two's complement)."""
    if value < 0:
        value += 1 << 64
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _tag(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _varint_field(field: int, value: int) -> bytes:
    return _tag(field, 0) + _varint(value)


def _bytes_field(field: int, payload: bytes) -> bytes:
    return _tag(field, 2) + _varint(len(payload)) + payload


def _string_field(field: int, text: str) -> bytes:
    return _bytes_field(field, text.encode("utf-8"))


def _message_field(field: int, payload: bytes) -> bytes:
    return _bytes_field(field, payload)


def _fixed32_float(field: int, value: float) -> bytes:
    return _tag(field, 5) + struct.pack("<f", value)


def _fixed64_field(field: int, value: int) -> bytes:
    return _tag(field, 1) + struct.pack("<Q", value)


def _packed_varints(field: int, values: list[int]) -> bytes:
    return _bytes_field(field, b"".join(_varint(v) for v in values))


def _unpacked_varints(field: int, values: list[int]) -> bytes:
    return b"".join(_varint_field(field, v) for v in values)


def _packed_floats(field: int, values: list[float]) -> bytes:
    return _bytes_field(field, struct.pack(f"<{len(values)}f", *values))


def _unpacked_floats(field: int, values: list[float]) -> bytes:
    return b"".join(_fixed32_float(field, v) for v in values)


# --- ONNX message builders -------------------------------------------------

# AttributeProto.AttributeType values used by the fixtures.
_ATTR_TYPE_FLOAT = 1
_ATTR_TYPE_INT = 2
_ATTR_TYPE_STRING = 3
_ATTR_TYPE_TENSOR = 4
_ATTR_TYPE_GRAPH = 5
_ATTR_TYPE_FLOATS = 8
_ATTR_TYPE_INTS = 9
_ATTR_TYPE_GRAPHS = 12


def _dim_value(value: int) -> bytes:
    return _message_field(1, _varint_field(1, value))


def _dim_param(name: str) -> bytes:
    return _message_field(1, _string_field(2, name))


def _dim_unset() -> bytes:
    return _message_field(1, b"")


def _tensor_type(elem_type: int, dims: bytes | None = None) -> bytes:
    body = _varint_field(1, elem_type)
    if dims is not None:
        body += _message_field(2, dims)
    return _message_field(1, body)


def _value_info(name: str, type_payload: bytes) -> bytes:
    return _string_field(1, name) + _message_field(2, type_payload)


def _string_string_entry(key: str, value: str) -> bytes:
    return _string_field(1, key) + _string_field(2, value)


def _tensor(
    *,
    name: str = "",
    elem_type: int = 1,
    dims: list[int] | None = None,
    packed_dims: bool = True,
    raw_data: bytes | None = None,
    external_location: str | None = None,
    data_location: int | None = None,
) -> bytes:
    body = b""
    if dims:
        body += _packed_varints(1, dims) if packed_dims else _unpacked_varints(1, dims)
    body += _varint_field(2, elem_type)
    if name:
        body += _string_field(8, name)
    if raw_data is not None:
        body += _bytes_field(9, raw_data)
    if external_location is not None:
        body += _message_field(13, _string_string_entry("location", external_location))
    if data_location is not None:
        body += _varint_field(14, data_location)
    return body


def _attribute(
    name: str, attribute_type: int, payload: bytes = b"", *, extra: bytes = b""
) -> bytes:
    return _string_field(1, name) + payload + extra + _varint_field(20, attribute_type)


def _attr_int(name: str, value: int) -> bytes:
    return _attribute(name, _ATTR_TYPE_INT, _varint_field(3, value))


def _attr_float(name: str, value: float) -> bytes:
    return _attribute(name, _ATTR_TYPE_FLOAT, _fixed32_float(2, value))


def _attr_string(name: str, value: bytes) -> bytes:
    return _attribute(name, _ATTR_TYPE_STRING, _bytes_field(4, value))


def _attr_ints(name: str, values: list[int], *, packed: bool = True) -> bytes:
    encoded = _packed_varints(8, values) if packed else _unpacked_varints(8, values)
    return _attribute(name, _ATTR_TYPE_INTS, encoded)


def _attr_floats(name: str, values: list[float], *, packed: bool = True) -> bytes:
    encoded = _packed_floats(7, values) if packed else _unpacked_floats(7, values)
    return _attribute(name, _ATTR_TYPE_FLOATS, encoded)


def _attr_tensor(name: str, tensor_payload: bytes) -> bytes:
    return _attribute(name, _ATTR_TYPE_TENSOR, _message_field(5, tensor_payload))


def _attr_graph(name: str, graph_payload: bytes) -> bytes:
    return _attribute(name, _ATTR_TYPE_GRAPH, _message_field(6, graph_payload))


def _attr_graphs(name: str, graph_payloads: list[bytes]) -> bytes:
    body = b"".join(_message_field(11, payload) for payload in graph_payloads)
    return _attribute(name, _ATTR_TYPE_GRAPHS, body)


def _node(
    op_type: str,
    *,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    name: str = "",
    domain: str = "",
    attributes: list[bytes] | None = None,
    extra: bytes = b"",
) -> bytes:
    body = b"".join(_string_field(1, value) for value in inputs or [])
    body += b"".join(_string_field(2, value) for value in outputs or [])
    if name:
        body += _string_field(3, name)
    body += _string_field(4, op_type)
    body += b"".join(_message_field(5, item) for item in attributes or [])
    if domain:
        body += _string_field(7, domain)
    return body + extra


def _graph(
    name: str,
    *,
    nodes: list[bytes] | None = None,
    initializers: list[bytes] | None = None,
    inputs: list[bytes] | None = None,
    outputs: list[bytes] | None = None,
    value_info: list[bytes] | None = None,
    extra: bytes = b"",
) -> bytes:
    body = b"".join(_message_field(1, item) for item in nodes or [])
    body += _string_field(2, name)
    body += b"".join(_message_field(5, item) for item in initializers or [])
    body += b"".join(_message_field(11, item) for item in inputs or [])
    body += b"".join(_message_field(12, item) for item in outputs or [])
    body += b"".join(_message_field(13, item) for item in value_info or [])
    return body + extra


def _opset(domain: str, version: int) -> bytes:
    return _string_field(1, domain) + _varint_field(2, version)


def _model(
    graph_payload: bytes,
    *,
    ir_version: int = 8,
    producer_name: str = "",
    producer_version: str = "",
    opsets: list[bytes] | None = None,
    extra: bytes = b"",
) -> bytes:
    body = _varint_field(1, ir_version)
    if producer_name:
        body += _string_field(2, producer_name)
    if producer_version:
        body += _string_field(3, producer_version)
    body += _message_field(7, graph_payload)
    body += b"".join(_message_field(8, item) for item in opsets or [])
    return body + extra


def _simple_model(**kwargs: object) -> bytes:
    """A minimal valid model wrapping a one-node graph."""
    graph = _graph("g", nodes=[_node("Add", inputs=["a", "b"], outputs=["c"])])
    return _model(graph, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. Full round trip
# ---------------------------------------------------------------------------


def _full_model_bytes() -> bytes:
    static_input = _value_info(
        "input_ids", _tensor_type(7, _dim_value(1) + _dim_value(128))
    )
    symbolic_input = _value_info(
        "past_key", _tensor_type(10, _dim_value(1) + _dim_param("past_seq"))
    )
    output = _value_info("logits", _tensor_type(1, _dim_value(1) + _dim_value(151936)))
    value_info = _value_info("hidden", _tensor_type(10, _dim_value(1) + _dim_value(8)))

    inline_init = _tensor(
        name="norm.weight",
        elem_type=10,
        dims=[4],
        raw_data=b"\x00\x11\x22\x33\x44\x55\x66\x77",
    )
    external_init = _tensor(
        name="embed.weight",
        elem_type=10,
        dims=[151936, 1024],
        external_location="model.onnx.data",
        data_location=1,
    )

    nodes = [
        _node(
            "MatMul",
            inputs=["input_ids", "embed.weight"],
            outputs=["hidden"],
            name="/model/MatMul",
        ),
        _node(
            "Constant",
            outputs=["const_out"],
            name="/model/Constant",
            attributes=[
                _attr_tensor(
                    "value",
                    _tensor(name="c", elem_type=1, dims=[2], raw_data=b"\x00" * 8),
                )
            ],
        ),
        _node(
            "Cast",
            inputs=["hidden"],
            outputs=["logits"],
            name="/model/Cast",
            domain="com.microsoft",
            attributes=[
                _attr_int("to", 10),
                _attr_float("scale", 0.5),
                _attr_string("label", b"cast-\xff"),
                _attr_ints("axes", [0, 2, -1]),
                _attr_floats("bias", [1.0, -2.5]),
            ],
        ),
    ]
    graph = _graph(
        "main_graph",
        nodes=nodes,
        initializers=[inline_init, external_init],
        inputs=[static_input, symbolic_input],
        outputs=[output],
        value_info=[value_info],
    )
    return _model(
        graph,
        ir_version=8,
        producer_name="pytorch",
        producer_version="2.7.1",
        opsets=[_opset("", 18), _opset("com.microsoft", 1)],
    )


def test_full_model_round_trip() -> None:
    summary = parse_onnx_model(_full_model_bytes())

    assert summary.ir_version == 8
    assert summary.producer_name == "pytorch"
    assert summary.producer_version == "2.7.1"
    assert summary.opset_imports == (("", 18), ("com.microsoft", 1))
    assert summary.graph_name == "main_graph"

    assert [value.name for value in summary.inputs] == ["input_ids", "past_key"]
    static_input, symbolic_input = summary.inputs
    assert static_input.elem_type == 7
    assert static_input.dtype == "int64"
    assert static_input.shape is not None
    assert static_input.shape.as_list() == [1, 128]
    assert static_input.shape.is_static is True
    assert symbolic_input.dtype == "float16"
    assert symbolic_input.shape is not None
    assert symbolic_input.shape.as_list() == [1, "past_seq"]
    assert symbolic_input.shape.is_static is False

    assert len(summary.outputs) == 1
    assert summary.outputs[0].name == "logits"
    assert summary.outputs[0].dtype == "float32"
    assert summary.outputs[0].shape is not None
    assert summary.outputs[0].shape.as_list() == [1, 151936]

    assert len(summary.value_info) == 1
    assert summary.value_info[0].name == "hidden"
    assert summary.value_info[0].dtype == "float16"

    inline_init, external_init = summary.initializers
    assert inline_init.name == "norm.weight"
    assert inline_init.dtype == "float16"
    assert inline_init.dims == (4,)
    assert inline_init.external is False
    assert inline_init.external_location is None
    assert inline_init.inline_bytes == 8
    assert external_init.name == "embed.weight"
    assert external_init.dims == (151936, 1024)
    assert external_init.external is True
    assert external_init.external_location == "model.onnx.data"
    assert external_init.inline_bytes == 0

    assert len(summary.nodes) == 3
    matmul, constant, cast = summary.nodes
    assert (matmul.index, matmul.scope, matmul.op_type) == (0, "", "MatMul")
    assert matmul.name == "/model/MatMul"
    assert matmul.domain == ""
    assert matmul.inputs == ("input_ids", "embed.weight")
    assert matmul.outputs == ("hidden",)
    assert matmul.attributes == ()

    assert constant.op_type == "Constant"
    assert len(constant.attributes) == 1
    value_attr = constant.attributes[0]
    assert value_attr.name == "value"
    assert value_attr.type == 4
    assert value_attr.type_name == "TENSOR"
    assert value_attr.has_graph is False
    assert value_attr.tensor is not None
    assert value_attr.tensor.dtype == "float32"
    assert value_attr.tensor.dims == (2,)
    assert value_attr.tensor.inline_bytes == 8

    assert cast.index == 2
    assert cast.domain == "com.microsoft"
    by_name = {attribute.name: attribute for attribute in cast.attributes}
    assert by_name["to"].type_name == "INT"
    assert by_name["to"].i == 10
    assert by_name["scale"].type_name == "FLOAT"
    assert by_name["scale"].f == pytest.approx(0.5)
    assert by_name["label"].type_name == "STRING"
    assert by_name["label"].s == b"cast-\xff"
    assert by_name["axes"].type_name == "INTS"
    assert by_name["axes"].ints == (0, 2, -1)
    assert by_name["bias"].type_name == "FLOATS"
    assert by_name["bias"].floats == pytest.approx((1.0, -2.5))
    assert all(attribute.has_graph is False for attribute in cast.attributes)


# ---------------------------------------------------------------------------
# 2. op_histogram
# ---------------------------------------------------------------------------


def test_op_histogram_counts_and_is_sorted() -> None:
    graph = _graph(
        "g",
        nodes=[
            _node("Mul", outputs=["m0"]),
            _node("Add", outputs=["a0"]),
            _node("Mul", outputs=["m1"]),
            _node("Cast", outputs=["c0"]),
            _node("Mul", outputs=["m2"]),
        ],
    )
    summary = parse_onnx_model(_model(graph))
    histogram = summary.op_histogram
    assert histogram == {"Add": 1, "Cast": 1, "Mul": 3}
    assert list(histogram) == sorted(histogram)
    assert sum(histogram.values()) == len(summary.nodes)


# ---------------------------------------------------------------------------
# 3. Packed vs unpacked repeated fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("packed", [True, False])
def test_packed_and_unpacked_dims_and_ints_agree(packed: bool) -> None:
    initializer = _tensor(
        name="w", elem_type=1, dims=[2, 3, 5], packed_dims=packed, raw_data=b"\x00" * 4
    )
    node = _node(
        "Reshape",
        outputs=["y"],
        attributes=[_attr_ints("axes", [7, 11, 13], packed=packed)],
    )
    summary = parse_onnx_model(
        _model(_graph("g", nodes=[node], initializers=[initializer]))
    )
    assert summary.initializers[0].dims == (2, 3, 5)
    assert summary.nodes[0].attributes[0].ints == (7, 11, 13)


@pytest.mark.parametrize("packed", [True, False])
def test_packed_and_unpacked_floats_agree(packed: bool) -> None:
    node = _node(
        "Clip", outputs=["y"], attributes=[_attr_floats("v", [0.5, 1.5], packed=packed)]
    )
    summary = parse_onnx_model(_model(_graph("g", nodes=[node])))
    assert summary.nodes[0].attributes[0].floats == pytest.approx((0.5, 1.5))


def test_packed_float_block_with_bad_length_is_rejected() -> None:
    bad_attribute = _attribute("v", _ATTR_TYPE_FLOATS, _bytes_field(7, b"\x00\x00\x00"))
    node = _node("Clip", outputs=["y"], attributes=[bad_attribute])
    with pytest.raises(OnnxReadError, match="not a multiple of 4"):
        parse_onnx_model(_model(_graph("g", nodes=[node])))


# ---------------------------------------------------------------------------
# 4. Negative int64 values
# ---------------------------------------------------------------------------


def test_negative_int64_values_decode_correctly() -> None:
    assert len(_varint(-1)) == 10
    node = _node(
        "Slice",
        outputs=["y"],
        attributes=[
            _attr_int("axis", -1),
            _attr_ints("starts", [-1, 0, -9223372036854775808, 9223372036854775807]),
        ],
    )
    initializer = _tensor(name="w", elem_type=1, dims=[-1, 4])
    summary = parse_onnx_model(
        _model(_graph("g", nodes=[node], initializers=[initializer]))
    )
    attributes = {item.name: item for item in summary.nodes[0].attributes}
    assert attributes["axis"].i == -1
    assert attributes["starts"].ints == (
        -1,
        0,
        -9223372036854775808,
        9223372036854775807,
    )
    assert summary.initializers[0].dims == (-1, 4)


# ---------------------------------------------------------------------------
# 5. Unknown fields are skipped
# ---------------------------------------------------------------------------


def _unknown_field_blob() -> bytes:
    return (
        _varint_field(90, 123456789)
        + _fixed64_field(91, 0xDEADBEEFCAFEBABE)
        + _bytes_field(92, b"opaque payload")
        + _fixed32_float(93, 3.25)
    )


def test_unknown_fields_of_every_wire_type_are_skipped() -> None:
    node = _node(
        "Add",
        inputs=["a", "b"],
        outputs=["c"],
        name="n0",
        extra=_unknown_field_blob(),
    )
    graph = _graph("g", nodes=[node], extra=_unknown_field_blob())
    summary = parse_onnx_model(
        _model(graph, producer_name="p", extra=_unknown_field_blob())
    )
    assert summary.graph_name == "g"
    assert summary.producer_name == "p"
    assert len(summary.nodes) == 1
    assert summary.nodes[0].op_type == "Add"
    assert summary.nodes[0].inputs == ("a", "b")
    assert summary.nodes[0].outputs == ("c",)
    assert summary.nodes[0].name == "n0"


# ---------------------------------------------------------------------------
# 6. Group and invalid wire types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wire_type", [3, 4])
def test_group_wire_types_are_rejected(wire_type: int) -> None:
    graph = _graph("g", nodes=[_node("Add", outputs=["c"], extra=_tag(90, wire_type))])
    with pytest.raises(OnnxReadError, match="group wire type"):
        parse_onnx_model(_model(graph))


@pytest.mark.parametrize("wire_type", [6, 7])
def test_invalid_wire_types_are_rejected(wire_type: int) -> None:
    graph = _graph("g", nodes=[_node("Add", outputs=["c"])], extra=_tag(90, wire_type))
    with pytest.raises(OnnxReadError, match="invalid wire type"):
        parse_onnx_model(_model(graph))


def test_zero_field_number_is_rejected() -> None:
    with pytest.raises(OnnxReadError, match="invalid field number 0"):
        parse_onnx_model(b"\x00")


# ---------------------------------------------------------------------------
# 7 and 8. Truncation and malformed varints
# ---------------------------------------------------------------------------


def test_truncated_varint_is_rejected() -> None:
    with pytest.raises(OnnxReadError, match="truncated varint"):
        parse_onnx_model(_tag(1, 0) + b"\x80\x80")


def test_length_running_past_end_of_buffer_is_rejected() -> None:
    payload = _tag(2, 2) + _varint(64) + b"short"
    with pytest.raises(OnnxReadError, match="truncated field"):
        parse_onnx_model(payload)


def test_truncated_nested_message_is_rejected() -> None:
    truncated_graph_body = _tag(2, 2) + _varint(50) + b"main"
    payload = _varint_field(1, 8) + _message_field(7, truncated_graph_body)
    with pytest.raises(OnnxReadError, match="truncated field"):
        parse_onnx_model(payload)


def test_truncated_node_inside_graph_is_rejected() -> None:
    node_body = _tag(4, 2) + _varint(40) + b"Add"
    graph = _graph("g") + _message_field(1, node_body)
    with pytest.raises(OnnxReadError, match="truncated field"):
        parse_onnx_model(_model(graph))


def test_overlong_varint_is_rejected() -> None:
    payload = _tag(1, 0) + b"\xff" * 11 + b"\x00"
    with pytest.raises(OnnxReadError, match="after 10 bytes"):
        parse_onnx_model(payload)


def test_model_without_graph_is_rejected() -> None:
    with pytest.raises(OnnxReadError, match="contains no graph"):
        parse_onnx_model(_varint_field(1, 8) + _string_field(2, "pytorch"))


@pytest.mark.parametrize("bad", [None, "not bytes", 17, ["bytes"]])
def test_non_bytes_input_raises_the_documented_error(bad: object) -> None:
    """ "Every failure path raises OnnxReadError" has to include this one."""

    with pytest.raises(OnnxReadError, match="serialized bytes of a ModelProto"):
        parse_onnx_model(bad)  # type: ignore[arg-type]


def test_external_data_entries_alone_do_not_make_a_tensor_external() -> None:
    """`data_location` decides; stale `external_data` entries do not.

    Per the ONNX spec the `external_data` key/value entries are only meaningful
    when `data_location` is EXTERNAL. Reporting a tensor that still carries its
    payload inline as external would hide it from the inspection engine's
    large-inline-constant detector, which skips anything marked external.
    """

    initializer = _tensor(
        name="mask",
        elem_type=10,
        dims=[4],
        raw_data=b"\x00" * 8,
        external_location="model.onnx.data",
    )
    model = _message_field(7, _graph(name="g", initializers=[initializer]))

    tensor = parse_onnx_model(model).initializers[0]

    assert tensor.external is False
    assert tensor.inline_bytes == 8
    # The declared location is still recorded for the reader of the report.
    assert tensor.external_location == "model.onnx.data"

    external = _tensor(
        name="weights",
        elem_type=10,
        dims=[4],
        external_location="model.onnx.data",
        data_location=1,
    )
    promoted = parse_onnx_model(
        _message_field(7, _graph(name="g", initializers=[external]))
    ).initializers[0]
    assert promoted.external is True
    assert promoted.inline_bytes == 0


# ---------------------------------------------------------------------------
# 9. Subgraph recursion
# ---------------------------------------------------------------------------


def test_if_subgraphs_are_flattened_with_scope_paths() -> None:
    then_branch = _graph("then", nodes=[_node("Add", outputs=["t"])])
    else_branch = _graph("else", nodes=[_node("Mul", outputs=["e"])])
    if_node = _node(
        "If",
        inputs=["cond"],
        outputs=["out"],
        attributes=[
            _attr_graph("then_branch", then_branch),
            _attr_graph("else_branch", else_branch),
        ],
    )
    graph = _graph("g", nodes=[_node("Relu", outputs=["r"]), if_node])
    summary = parse_onnx_model(_model(graph))

    assert [(node.scope, node.index, node.op_type) for node in summary.nodes] == [
        ("", 0, "Relu"),
        ("", 1, "If"),
        ("node[1]:If.then_branch", 0, "Add"),
        ("node[1]:If.else_branch", 0, "Mul"),
    ]
    assert all(attribute.has_graph is True for attribute in summary.nodes[1].attributes)
    assert summary.nodes[1].attributes[0].type_name == "GRAPH"
    assert summary.op_histogram == {"Add": 1, "If": 1, "Mul": 1, "Relu": 1}
    # Subgraph values are not merged into the top-level summary.
    assert summary.graph_name == "g"
    assert summary.inputs == ()
    assert summary.initializers == ()


def test_nested_subgraphs_produce_nested_scope_paths() -> None:
    loop_body = _graph("body", nodes=[_node("Add", outputs=["a"])])
    loop_node = _node(
        "Loop", outputs=["lo"], attributes=[_attr_graph("body", loop_body)]
    )
    then_branch = _graph("then", nodes=[_node("Cast", outputs=["c"]), loop_node])
    if_node = _node(
        "If", outputs=["out"], attributes=[_attr_graph("then_branch", then_branch)]
    )
    graph = _graph("g", nodes=[_node("Relu", outputs=["r"]), if_node])
    summary = parse_onnx_model(_model(graph))

    assert [(node.scope, node.op_type) for node in summary.nodes] == [
        ("", "Relu"),
        ("", "If"),
        ("node[1]:If.then_branch", "Cast"),
        ("node[1]:If.then_branch", "Loop"),
        ("node[1]:If.then_branch/node[1]:Loop.body", "Add"),
    ]


def test_repeated_graphs_attribute_is_indexed_in_scope() -> None:
    branches = [
        _graph("b0", nodes=[_node("Add", outputs=["a"])]),
        _graph("b1", nodes=[_node("Sub", outputs=["s"])]),
    ]
    scan_node = _node(
        "Scan", outputs=["o"], attributes=[_attr_graphs("bodies", branches)]
    )
    summary = parse_onnx_model(_model(_graph("g", nodes=[scan_node])))
    assert summary.nodes[0].attributes[0].has_graph is True
    assert [(node.scope, node.op_type) for node in summary.nodes[1:]] == [
        ("node[0]:Scan.bodies[0]", "Add"),
        ("node[0]:Scan.bodies[1]", "Sub"),
    ]


def _nested_graph(levels: int) -> bytes:
    payload = _graph("leaf", nodes=[_node("Add", outputs=["a"])])
    for _ in range(levels):
        payload = _graph(
            "wrap",
            nodes=[
                _node(
                    "If",
                    outputs=["o"],
                    attributes=[_attr_graph("then_branch", payload)],
                )
            ],
        )
    return payload


def test_subgraph_depth_limit_boundary() -> None:
    summary = parse_onnx_model(_model(_nested_graph(32)))
    assert summary.nodes[-1].op_type == "Add"
    assert len(summary.nodes) == 33


def test_subgraph_depth_limit_is_enforced() -> None:
    with pytest.raises(OnnxReadError, match="exceeds the maximum depth"):
        parse_onnx_model(_model(_nested_graph(33)))


# ---------------------------------------------------------------------------
# 10 and 11. Shapes and type variants
# ---------------------------------------------------------------------------


def test_non_tensor_type_yields_no_shape() -> None:
    sequence_type = _message_field(4, _message_field(1, _tensor_type(1, _dim_value(3))))
    graph = _graph(
        "g",
        nodes=[_node("Identity", outputs=["y"])],
        inputs=[_value_info("seq_in", sequence_type)],
    )
    summary = parse_onnx_model(_model(graph))
    value = summary.inputs[0]
    assert value.name == "seq_in"
    assert value.shape is None
    assert value.elem_type == 0
    assert value.dtype == "unknown(0)"


def test_tensor_type_without_shape_yields_no_shape() -> None:
    graph = _graph(
        "g",
        nodes=[_node("Identity", outputs=["y"])],
        inputs=[_value_info("x", _tensor_type(1))],
    )
    summary = parse_onnx_model(_model(graph))
    assert summary.inputs[0].dtype == "float32"
    assert summary.inputs[0].shape is None


def test_unknown_elem_type_is_reported_verbatim() -> None:
    graph = _graph(
        "g",
        nodes=[_node("Identity", outputs=["y"])],
        inputs=[_value_info("x", _tensor_type(99, _dim_value(1)))],
    )
    summary = parse_onnx_model(_model(graph))
    assert summary.inputs[0].elem_type == 99
    assert summary.inputs[0].dtype == "unknown(99)"


def test_dimension_without_value_or_param_is_none() -> None:
    dims = _dim_value(1) + _dim_param("seq") + _dim_unset()
    graph = _graph(
        "g",
        nodes=[_node("Identity", outputs=["y"])],
        inputs=[_value_info("x", _tensor_type(1, dims))],
    )
    summary = parse_onnx_model(_model(graph))
    shape = summary.inputs[0].shape
    assert shape is not None
    assert shape.as_list() == [1, "seq", None]
    assert shape.is_static is False


def test_tensor_shape_is_static_helper() -> None:
    assert TensorShape((1, 2, 3)).is_static is True
    assert TensorShape(()).is_static is True
    assert TensorShape((1, "seq")).is_static is False
    assert TensorShape((1, None)).is_static is False
    assert TensorShape((1, "seq")).as_list() == [1, "seq"]


# ---------------------------------------------------------------------------
# 12. read_onnx_model
# ---------------------------------------------------------------------------


def test_read_onnx_model_parses_a_file(tmp_path: Path) -> None:
    path = tmp_path / "tiny.onnx"
    path.write_bytes(_simple_model(producer_name="test"))
    summary = read_onnx_model(path)
    assert summary.producer_name == "test"
    assert summary.nodes[0].op_type == "Add"
    assert read_onnx_model(str(path)).graph_name == "g"


def test_read_onnx_model_rejects_oversized_files(tmp_path: Path) -> None:
    path = tmp_path / "big.onnx"
    path.write_bytes(_simple_model())
    with pytest.raises(OnnxReadError, match="exceeds max_bytes"):
        read_onnx_model(path, max_bytes=4)


def test_read_onnx_model_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OnnxReadError, match="cannot stat"):
        read_onnx_model(tmp_path / "absent.onnx")


def test_read_onnx_model_wraps_parse_errors_with_the_path(tmp_path: Path) -> None:
    path = tmp_path / "broken.onnx"
    path.write_bytes(_tag(1, 0) + b"\x80\x80")
    with pytest.raises(OnnxReadError, match="broken.onnx"):
        read_onnx_model(path)


# ---------------------------------------------------------------------------
# 13. as_dict
# ---------------------------------------------------------------------------


def _assert_no_bytes(value: object) -> None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise AssertionError(f"as_dict() leaked a bytes value: {value!r}")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_no_bytes(key)
            _assert_no_bytes(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_bytes(item)


def test_as_dict_is_json_serializable_and_bytes_free() -> None:
    summary = parse_onnx_model(_full_model_bytes())
    payload = summary.as_dict()
    _assert_no_bytes(payload)
    encoded = json.dumps(payload)
    restored = json.loads(encoded)
    assert restored["graph_name"] == "main_graph"
    assert restored["opset_imports"] == [
        {"domain": "", "version": 18},
        {"domain": "com.microsoft", "version": 1},
    ]
    assert restored["op_histogram"] == {"Cast": 1, "Constant": 1, "MatMul": 1}
    assert restored["inputs"][1]["shape"] == [1, "past_seq"]
    assert restored["initializers"][1]["external"] is True
    cast_attributes = restored["nodes"][2]["attributes"]
    label = next(item for item in cast_attributes if item["name"] == "label")
    assert label["s_hex"] == b"cast-\xff".hex()


# ---------------------------------------------------------------------------
# 14. Real T20 export (guarded)
# ---------------------------------------------------------------------------

_REAL_GRAPH_RELPATH = Path("onnx/reference/T20/S128/decode.onnx")
_REAL_GRAPH_SHA256 = "e200ecd27e1ab83d2bea17de030c0a0c8a0eea08c6f182eed41c04a457c421d2"


def _artifact_root() -> Path:
    env_root = os.environ.get("SLM_LAB_ARTIFACT_ROOT")
    if env_root:
        return Path(env_root)
    return Path(__file__).resolve().parents[2] / "artifacts"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_real_t20_decode_graph_structure() -> None:
    path = _artifact_root() / _REAL_GRAPH_RELPATH
    if not path.is_file():
        pytest.skip(
            f"T20 reference export not present at {path}; set "
            "SLM_LAB_ARTIFACT_ROOT or attach the artifact volume to run this test"
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != _REAL_GRAPH_SHA256:
        pytest.skip(
            f"{path} sha256 {actual_sha256} does not match the digest recorded in "
            f"results/manifests/onnx/S128.json ({_REAL_GRAPH_SHA256})"
        )

    summary = read_onnx_model(path)
    assert isinstance(summary, GraphSummary)
    assert len(summary.nodes) >= 1
    assert sum(summary.op_histogram.values()) == len(summary.nodes)

    default_opsets = dict(summary.opset_imports)
    assert default_opsets[""] == 18

    inputs_by_name = {value.name: value for value in summary.inputs}
    assert "input_ids" in inputs_by_name
    input_ids = inputs_by_name["input_ids"]
    assert input_ids.dtype == "int64"
    assert input_ids.shape is not None
    assert input_ids.shape.as_list() == [1, 1]
    assert input_ids.shape.is_static is True

    assert summary.initializers, "the decode graph must declare initializers"
    for initializer in summary.initializers:
        assert initializer.external or initializer.inline_bytes <= 65536, (
            f"initializer {initializer.name} is inline and unexpectedly large "
            f"({initializer.inline_bytes} bytes)"
        )
        if initializer.external:
            assert initializer.external_location, (
                f"external initializer {initializer.name} has no location"
            )

    # The reader must stay purely structural: no external payload is loaded.
    external = [item for item in summary.initializers if item.external]
    assert external, "the decode graph stores weights as external data"
    assert all(item.inline_bytes == 0 for item in external)

    json.dumps(summary.as_dict())
