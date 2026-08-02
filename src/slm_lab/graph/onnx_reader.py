"""Minimal, read-only ONNX protobuf reader implemented with the standard library.

Why this module exists
----------------------
Task T21 inspects real ONNX graphs exported from Qwen3-0.6B, but this repository
deliberately keeps graph inspection free of heavy runtime dependencies: there is
no ``onnx``, ``protobuf``, ``numpy``, ``torch``, or ``onnxruntime`` requirement
for the inspection path. This module therefore implements just enough of the
protobuf wire format to recover the *structure* of a ``ModelProto`` -- opset
imports, graph inputs/outputs/value_info, initializer metadata, and the node list
with attributes -- from a ``.onnx`` file.

What it does
------------
* Decodes the protobuf wire format directly (varint, 64-bit, length-delimited,
  32-bit), skipping every field number it does not model, by wire type.
* Understands both packed and unpacked encodings of repeated numeric fields.
* Recurses into ``If`` / ``Loop`` / ``Scan`` subgraphs and flattens their nodes
  into a single node list with a readable ``scope`` path.
* Records initializer metadata only: dtype, dims, whether the tensor lives in an
  external data file, and how many bytes of ``raw_data`` were stored inline.

What it deliberately does NOT do
--------------------------------
* It is **not** a validator. It does not check operator schemas, type
  consistency, topological ordering, or opset compatibility.
* It is **not** a writer. There is no serialization path.
* It **never** opens, follows, or reads an external data file. An initializer
  backed by ``model.onnx.data`` is reported as ``external`` together with its
  declared ``location`` string, and nothing more.
* It does not retain tensor payload bytes. Only ``len(raw_data)`` is kept, so
  memory stays bounded even on a 35 MB graph.
* It ignores sequence/map/optional/sparse ``TypeProto`` variants, sparse
  tensors, ``AttributeProto.strings``, and ``AttributeProto.tensors``; those
  fields are skipped correctly rather than guessed at.

Every failure path raises :class:`OnnxReadError` with the offending field path
and absolute byte offset. Truncated input, an over-long varint, a length that
runs past the end of the buffer, deprecated group wire types (3 and 4), and
invalid wire types (6 and 7) are all errors, never silent wrong answers.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

__all__ = [
    "ATTRIBUTE_TYPE_NAMES",
    "TENSOR_DATA_TYPE_NAMES",
    "AttributeInfo",
    "GraphSummary",
    "InitializerInfo",
    "NodeInfo",
    "OnnxReadError",
    "TensorShape",
    "ValueInfo",
    "attribute_type_name",
    "parse_onnx_model",
    "read_onnx_model",
    "tensor_dtype_name",
]


class OnnxReadError(ValueError):
    """The byte stream is not a readable ONNX model."""


# --------------------------------------------------------------------------
# Enum name tables (see onnx.proto3, ONNX 1.18)
# --------------------------------------------------------------------------

TENSOR_DATA_TYPE_NAMES: Mapping[int, str] = MappingProxyType(
    {
        1: "float32",
        2: "uint8",
        3: "int8",
        4: "uint16",
        5: "int16",
        6: "int32",
        7: "int64",
        8: "string",
        9: "bool",
        10: "float16",
        11: "float64",
        12: "uint32",
        13: "uint64",
        14: "complex64",
        15: "complex128",
        16: "bfloat16",
    }
)

ATTRIBUTE_TYPE_NAMES: Mapping[int, str] = MappingProxyType(
    {
        0: "UNDEFINED",
        1: "FLOAT",
        2: "INT",
        3: "STRING",
        4: "TENSOR",
        5: "GRAPH",
        6: "SPARSE_TENSOR",
        7: "TYPE_PROTO",
        8: "FLOATS",
        9: "INTS",
        10: "STRINGS",
        11: "TENSORS",
        12: "GRAPHS",
        13: "SPARSE_TENSORS",
        14: "TYPE_PROTOS",
    }
)

#: Maximum number of nested subgraph levels the reader will follow.
MAX_SUBGRAPH_DEPTH = 32

#: Default ceiling for :func:`read_onnx_model`, chosen so that pointing the
#: reader at a multi-gigabyte ``.onnx.data`` sidecar fails loudly.
DEFAULT_MAX_BYTES = 256 * 1024 * 1024

_UINT64_MASK = (1 << 64) - 1
_INT64_SIGN_BIT = 1 << 63


def tensor_dtype_name(elem_type: int) -> str:
    """Return the readable name of a ``TensorProto.DataType`` enum value."""
    return TENSOR_DATA_TYPE_NAMES.get(elem_type, f"unknown({elem_type})")


def attribute_type_name(attribute_type: int) -> str:
    """Return the readable name of an ``AttributeProto.AttributeType`` value."""
    return ATTRIBUTE_TYPE_NAMES.get(attribute_type, f"UNKNOWN({attribute_type})")


# --------------------------------------------------------------------------
# Public data model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorShape:
    """A tensor rank/shape as declared in a ``TensorShapeProto``.

    ``int`` entries are static ``dim_value`` dimensions, ``str`` entries are
    symbolic ``dim_param`` names, and ``None`` marks a dimension that declared
    neither.
    """

    dims: tuple[int | str | None, ...]

    @property
    def is_static(self) -> bool:
        """True when every dimension is a concrete integer."""
        return all(isinstance(dim, int) for dim in self.dims)

    def as_list(self) -> list[int | str | None]:
        """Return the dimensions as a plain list."""
        return list(self.dims)


@dataclass(frozen=True)
class ValueInfo:
    """A named graph value: input, output, or intermediate ``value_info``."""

    name: str
    elem_type: int
    dtype: str
    shape: TensorShape | None


@dataclass(frozen=True)
class InitializerInfo:
    """Metadata for a ``TensorProto`` -- never its payload."""

    name: str
    elem_type: int
    dtype: str
    dims: tuple[int, ...]
    external: bool
    external_location: str | None
    inline_bytes: int


@dataclass(frozen=True)
class AttributeInfo:
    """A node attribute, decoded for the subset of types this reader models."""

    name: str
    type: int
    type_name: str
    i: int | None
    f: float | None
    s: bytes | None
    ints: tuple[int, ...]
    floats: tuple[float, ...]
    tensor: InitializerInfo | None
    has_graph: bool


@dataclass(frozen=True)
class NodeInfo:
    """A single node, tagged with the graph scope it was found in."""

    index: int
    scope: str
    op_type: str
    name: str
    domain: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attributes: tuple[AttributeInfo, ...]


@dataclass(frozen=True)
class GraphSummary:
    """Structural summary of a ``ModelProto`` and its main ``GraphProto``.

    ``nodes`` is flattened: every node of the main graph first, then the nodes
    contributed by each subgraph in declaration order. Subgraph inputs,
    outputs, ``value_info`` and initializers are intentionally *not* merged.
    """

    ir_version: int
    producer_name: str
    producer_version: str
    opset_imports: tuple[tuple[str, int], ...]
    graph_name: str
    inputs: tuple[ValueInfo, ...]
    outputs: tuple[ValueInfo, ...]
    value_info: tuple[ValueInfo, ...]
    initializers: tuple[InitializerInfo, ...]
    nodes: tuple[NodeInfo, ...]

    @property
    def op_histogram(self) -> dict[str, int]:
        """Map ``op_type`` to occurrence count over ``nodes``, sorted by key."""
        counts: dict[str, int] = {}
        for node in self.nodes:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
        return dict(sorted(counts.items()))

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view. Byte fields are hex-encoded."""
        return {
            "ir_version": self.ir_version,
            "producer_name": self.producer_name,
            "producer_version": self.producer_version,
            "opset_imports": [
                {"domain": domain, "version": version}
                for domain, version in self.opset_imports
            ],
            "graph_name": self.graph_name,
            "inputs": [_value_info_as_dict(value) for value in self.inputs],
            "outputs": [_value_info_as_dict(value) for value in self.outputs],
            "value_info": [_value_info_as_dict(value) for value in self.value_info],
            "initializers": [
                _initializer_as_dict(tensor) for tensor in self.initializers
            ],
            "nodes": [_node_as_dict(node) for node in self.nodes],
            "op_histogram": self.op_histogram,
        }


def _value_info_as_dict(value: ValueInfo) -> dict[str, object]:
    return {
        "name": value.name,
        "elem_type": value.elem_type,
        "dtype": value.dtype,
        "shape": None if value.shape is None else value.shape.as_list(),
        "is_static": None if value.shape is None else value.shape.is_static,
    }


def _initializer_as_dict(tensor: InitializerInfo) -> dict[str, object]:
    return {
        "name": tensor.name,
        "elem_type": tensor.elem_type,
        "dtype": tensor.dtype,
        "dims": list(tensor.dims),
        "external": tensor.external,
        "external_location": tensor.external_location,
        "inline_bytes": tensor.inline_bytes,
    }


def _attribute_as_dict(attribute: AttributeInfo) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": attribute.name,
        "type": attribute.type,
        "type_name": attribute.type_name,
        "i": attribute.i,
        "f": attribute.f,
        "ints": list(attribute.ints),
        "floats": list(attribute.floats),
        "tensor": (
            None if attribute.tensor is None else _initializer_as_dict(attribute.tensor)
        ),
        "has_graph": attribute.has_graph,
    }
    if attribute.s is not None:
        payload["s_hex"] = attribute.s.hex()
    return payload


def _node_as_dict(node: NodeInfo) -> dict[str, object]:
    return {
        "index": node.index,
        "scope": node.scope,
        "op_type": node.op_type,
        "name": node.name,
        "domain": node.domain,
        "inputs": list(node.inputs),
        "outputs": list(node.outputs),
        "attributes": [_attribute_as_dict(item) for item in node.attributes],
    }


# --------------------------------------------------------------------------
# Protobuf wire-format cursor
# --------------------------------------------------------------------------

_WIRE_VARINT = 0
_WIRE_FIXED64 = 1
_WIRE_DELIMITED = 2
_WIRE_START_GROUP = 3
_WIRE_END_GROUP = 4
_WIRE_FIXED32 = 5


class _Cursor:
    """A bounds-checked read cursor over a protobuf message body."""

    __slots__ = ("base", "buf", "path", "pos", "size")

    def __init__(self, buf: memoryview, path: str, base: int = 0) -> None:
        self.buf = buf
        self.size = len(buf)
        self.pos = 0
        self.base = base
        self.path = path

    def fail(self, message: str) -> OnnxReadError:
        return OnnxReadError(
            f"{message} [field path {self.path}, byte offset {self.base + self.pos}]"
        )

    @property
    def exhausted(self) -> bool:
        return self.pos >= self.size

    def read_varint(self) -> int:
        """Read an unsigned varint of at most 10 bytes."""
        result = 0
        shift = 0
        for _ in range(10):
            if self.pos >= self.size:
                raise self.fail("truncated varint: buffer ended mid-value")
            byte = self.buf[self.pos]
            self.pos += 1
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return result
            shift += 7
        raise self.fail("malformed varint: continuation bit still set after 10 bytes")

    def read_tag(self) -> tuple[int, int]:
        """Read a field header and return ``(field_number, wire_type)``."""
        header = self.read_varint()
        field_number = header >> 3
        wire_type = header & 7
        if field_number == 0:
            raise self.fail(f"invalid field number 0 in tag {header}")
        return field_number, wire_type

    def read_raw(self, count: int) -> memoryview:
        end = self.pos + count
        if count < 0 or end > self.size:
            raise self.fail(
                f"truncated field: needed {count} bytes but only "
                f"{self.size - self.pos} remain"
            )
        chunk = self.buf[self.pos : end]
        self.pos = end
        return chunk

    def read_delimited(self) -> memoryview:
        length = self.read_varint()
        return self.read_raw(length)

    def read_float(self) -> float:
        return struct.unpack("<f", self.read_raw(4))[0]

    def submessage(self, field_path: str) -> _Cursor:
        """Read a length-delimited field and return a cursor over its body."""
        start = self.pos
        payload = self.read_delimited()
        header_size = self.pos - start - len(payload)
        return _Cursor(payload, field_path, self.base + start + header_size)

    def skip(self, wire_type: int, field_number: int) -> None:
        """Skip an unmodelled field strictly according to its wire type."""
        if wire_type == _WIRE_VARINT:
            self.read_varint()
        elif wire_type == _WIRE_FIXED64:
            self.read_raw(8)
        elif wire_type == _WIRE_DELIMITED:
            self.read_delimited()
        elif wire_type == _WIRE_FIXED32:
            self.read_raw(4)
        elif wire_type in (_WIRE_START_GROUP, _WIRE_END_GROUP):
            raise self.fail(
                f"field {field_number} uses deprecated group wire type "
                f"{wire_type}, which is not supported"
            )
        else:
            raise self.fail(f"field {field_number} uses invalid wire type {wire_type}")


def _as_int64(value: int) -> int:
    """Reinterpret an unsigned varint as a two's-complement signed int64."""
    value &= _UINT64_MASK
    if value >= _INT64_SIGN_BIT:
        value -= 1 << 64
    return value


def _read_string(cursor: _Cursor, wire_type: int, field_path: str) -> str:
    if wire_type != _WIRE_DELIMITED:
        raise cursor.fail(
            f"{field_path} expected a length-delimited string but saw wire type "
            f"{wire_type}"
        )
    payload = cursor.read_delimited()
    try:
        return bytes(payload).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise cursor.fail(f"{field_path} is not valid UTF-8: {exc}") from exc


def _read_bytes(cursor: _Cursor, wire_type: int, field_path: str) -> bytes:
    if wire_type != _WIRE_DELIMITED:
        raise cursor.fail(
            f"{field_path} expected a length-delimited value but saw wire type "
            f"{wire_type}"
        )
    return bytes(cursor.read_delimited())


def _read_int64s(
    cursor: _Cursor, wire_type: int, out: list[int], field_path: str
) -> None:
    """Append one repeated int64 field occurrence, packed or unpacked."""
    if wire_type == _WIRE_VARINT:
        out.append(_as_int64(cursor.read_varint()))
        return
    if wire_type != _WIRE_DELIMITED:
        raise cursor.fail(
            f"{field_path} expected varint or packed bytes but saw wire type "
            f"{wire_type}"
        )
    packed = cursor.submessage(f"{field_path}[packed]")
    while not packed.exhausted:
        out.append(_as_int64(packed.read_varint()))


def _read_floats(
    cursor: _Cursor, wire_type: int, out: list[float], field_path: str
) -> None:
    """Append one repeated float field occurrence, packed or unpacked."""
    if wire_type == _WIRE_FIXED32:
        out.append(cursor.read_float())
        return
    if wire_type != _WIRE_DELIMITED:
        raise cursor.fail(
            f"{field_path} expected fixed32 or packed bytes but saw wire type "
            f"{wire_type}"
        )
    packed = cursor.read_delimited()
    if len(packed) % 4:
        raise cursor.fail(
            f"{field_path} packed float block has length {len(packed)}, "
            "which is not a multiple of 4"
        )
    out.extend(struct.unpack(f"<{len(packed) // 4}f", packed))


# --------------------------------------------------------------------------
# ONNX message parsers
# --------------------------------------------------------------------------


def _parse_dimension(cursor: _Cursor) -> int | str | None:
    """Parse ``TensorShapeProto.Dimension``."""
    dim: int | str | None = None
    while not cursor.exhausted:
        field_number, wire_type = cursor.read_tag()
        if field_number == 1 and wire_type == _WIRE_VARINT:
            dim = _as_int64(cursor.read_varint())
        elif field_number == 2 and wire_type == _WIRE_DELIMITED:
            dim = _read_string(cursor, wire_type, "Dimension.dim_param")
        else:
            cursor.skip(wire_type, field_number)
    return dim


def _parse_tensor_shape(cursor: _Cursor) -> TensorShape:
    """Parse ``TensorShapeProto``."""
    dims: list[int | str | None] = []
    while not cursor.exhausted:
        field_number, wire_type = cursor.read_tag()
        if field_number == 1 and wire_type == _WIRE_DELIMITED:
            dims.append(
                _parse_dimension(cursor.submessage(f"{cursor.path}.dim[{len(dims)}]"))
            )
        else:
            cursor.skip(wire_type, field_number)
    return TensorShape(tuple(dims))


def _parse_type_proto(cursor: _Cursor) -> tuple[int, TensorShape | None]:
    """Parse ``TypeProto``, modelling only the tensor variant."""
    elem_type = 0
    shape: TensorShape | None = None
    while not cursor.exhausted:
        field_number, wire_type = cursor.read_tag()
        if field_number == 1 and wire_type == _WIRE_DELIMITED:
            tensor = cursor.submessage(f"{cursor.path}.tensor_type")
            while not tensor.exhausted:
                inner_field, inner_wire = tensor.read_tag()
                if inner_field == 1 and inner_wire == _WIRE_VARINT:
                    elem_type = _as_int64(tensor.read_varint())
                elif inner_field == 2 and inner_wire == _WIRE_DELIMITED:
                    shape = _parse_tensor_shape(
                        tensor.submessage(f"{tensor.path}.shape")
                    )
                else:
                    tensor.skip(inner_wire, inner_field)
        else:
            cursor.skip(wire_type, field_number)
    return elem_type, shape


def _parse_value_info(cursor: _Cursor) -> ValueInfo:
    """Parse ``ValueInfoProto``."""
    name = ""
    elem_type = 0
    shape: TensorShape | None = None
    while not cursor.exhausted:
        field_number, wire_type = cursor.read_tag()
        if field_number == 1:
            name = _read_string(cursor, wire_type, f"{cursor.path}.name")
        elif field_number == 2 and wire_type == _WIRE_DELIMITED:
            elem_type, shape = _parse_type_proto(
                cursor.submessage(f"{cursor.path}.type")
            )
        else:
            cursor.skip(wire_type, field_number)
    return ValueInfo(
        name=name,
        elem_type=elem_type,
        dtype=tensor_dtype_name(elem_type),
        shape=shape,
    )


def _parse_string_string_entry(cursor: _Cursor) -> tuple[str, str]:
    """Parse ``StringStringEntryProto``."""
    key = ""
    value = ""
    while not cursor.exhausted:
        field_number, wire_type = cursor.read_tag()
        if field_number == 1:
            key = _read_string(cursor, wire_type, f"{cursor.path}.key")
        elif field_number == 2:
            value = _read_string(cursor, wire_type, f"{cursor.path}.value")
        else:
            cursor.skip(wire_type, field_number)
    return key, value


def _parse_tensor(cursor: _Cursor) -> InitializerInfo:
    """Parse ``TensorProto`` metadata without retaining any payload bytes."""
    dims: list[int] = []
    elem_type = 0
    name = ""
    inline_bytes = 0
    data_location = 0
    external_location: str | None = None
    while not cursor.exhausted:
        field_number, wire_type = cursor.read_tag()
        if field_number == 1:
            _read_int64s(cursor, wire_type, dims, f"{cursor.path}.dims")
        elif field_number == 2 and wire_type == _WIRE_VARINT:
            elem_type = _as_int64(cursor.read_varint())
        elif field_number == 8:
            name = _read_string(cursor, wire_type, f"{cursor.path}.name")
        elif field_number == 9:
            if wire_type != _WIRE_DELIMITED:
                raise cursor.fail(
                    f"{cursor.path}.raw_data expected a length-delimited value "
                    f"but saw wire type {wire_type}"
                )
            inline_bytes = len(cursor.read_delimited())
        elif field_number == 13 and wire_type == _WIRE_DELIMITED:
            key, value = _parse_string_string_entry(
                cursor.submessage(f"{cursor.path}.external_data")
            )
            if key == "location":
                external_location = value
        elif field_number == 14 and wire_type == _WIRE_VARINT:
            data_location = _as_int64(cursor.read_varint())
        else:
            cursor.skip(wire_type, field_number)
    return InitializerInfo(
        name=name,
        elem_type=elem_type,
        dtype=tensor_dtype_name(elem_type),
        dims=tuple(dims),
        # `data_location` alone decides this. Per the ONNX spec the
        # `external_data` entries are only meaningful when `data_location` is
        # EXTERNAL(1); a tensor that carries stale entries while still storing
        # its payload inline is an inline tensor, and treating it as external
        # would hide it from the large-inline-constant detector, which skips
        # anything marked external. The declared location string is still
        # recorded whenever entries are present.
        external=data_location == 1,
        external_location=external_location,
        inline_bytes=inline_bytes,
    )


def _parse_attribute(
    cursor: _Cursor,
) -> tuple[AttributeInfo, list[tuple[str, _Cursor]]]:
    """Parse ``AttributeProto``.

    Returns the attribute plus any subgraph bodies it carries, each paired with
    the label used to build a child scope (``"<attr>"`` for the singular
    ``g`` field, ``"<attr>[k]"`` for entries of the repeated ``graphs`` field).
    """
    name = ""
    attribute_type = 0
    i_value: int | None = None
    f_value: float | None = None
    s_value: bytes | None = None
    ints: list[int] = []
    floats: list[float] = []
    tensor: InitializerInfo | None = None
    subgraphs: list[tuple[str, _Cursor]] = []
    graphs_seen = 0
    has_single_graph = False
    while not cursor.exhausted:
        field_number, wire_type = cursor.read_tag()
        if field_number == 1:
            name = _read_string(cursor, wire_type, f"{cursor.path}.name")
        elif field_number == 20 and wire_type == _WIRE_VARINT:
            attribute_type = _as_int64(cursor.read_varint())
        elif field_number == 2 and wire_type == _WIRE_FIXED32:
            f_value = cursor.read_float()
        elif field_number == 3 and wire_type == _WIRE_VARINT:
            i_value = _as_int64(cursor.read_varint())
        elif field_number == 4:
            s_value = _read_bytes(cursor, wire_type, f"{cursor.path}.s")
        elif field_number == 5 and wire_type == _WIRE_DELIMITED:
            tensor = _parse_tensor(cursor.submessage(f"{cursor.path}.t"))
        elif field_number == 6 and wire_type == _WIRE_DELIMITED:
            has_single_graph = True
            subgraphs.append(("", cursor.submessage(f"{cursor.path}.g")))
        elif field_number == 7:
            _read_floats(cursor, wire_type, floats, f"{cursor.path}.floats")
        elif field_number == 8:
            _read_int64s(cursor, wire_type, ints, f"{cursor.path}.ints")
        elif field_number == 11 and wire_type == _WIRE_DELIMITED:
            subgraphs.append(
                (
                    f"[{graphs_seen}]",
                    cursor.submessage(f"{cursor.path}.graphs[{graphs_seen}]"),
                )
            )
            graphs_seen += 1
        else:
            cursor.skip(wire_type, field_number)
    labelled = [(f"{name}{suffix}", body) for suffix, body in subgraphs]
    attribute = AttributeInfo(
        name=name,
        type=attribute_type,
        type_name=attribute_type_name(attribute_type),
        i=i_value,
        f=f_value,
        s=s_value,
        ints=tuple(ints),
        floats=tuple(floats),
        tensor=tensor,
        has_graph=has_single_graph or graphs_seen > 0,
    )
    return attribute, labelled


def _parse_node(
    cursor: _Cursor, index: int, scope: str
) -> tuple[NodeInfo, list[tuple[str, _Cursor]]]:
    """Parse ``NodeProto`` and collect the subgraph bodies it owns."""
    inputs: list[str] = []
    outputs: list[str] = []
    name = ""
    op_type = ""
    domain = ""
    attributes: list[AttributeInfo] = []
    subgraphs: list[tuple[str, _Cursor]] = []
    while not cursor.exhausted:
        field_number, wire_type = cursor.read_tag()
        if field_number == 1:
            inputs.append(_read_string(cursor, wire_type, f"{cursor.path}.input"))
        elif field_number == 2:
            outputs.append(_read_string(cursor, wire_type, f"{cursor.path}.output"))
        elif field_number == 3:
            name = _read_string(cursor, wire_type, f"{cursor.path}.name")
        elif field_number == 4:
            op_type = _read_string(cursor, wire_type, f"{cursor.path}.op_type")
        elif field_number == 5 and wire_type == _WIRE_DELIMITED:
            attribute, bodies = _parse_attribute(
                cursor.submessage(f"{cursor.path}.attribute[{len(attributes)}]")
            )
            attributes.append(attribute)
            subgraphs.extend(bodies)
        elif field_number == 7:
            domain = _read_string(cursor, wire_type, f"{cursor.path}.domain")
        else:
            cursor.skip(wire_type, field_number)
    node = NodeInfo(
        index=index,
        scope=scope,
        op_type=op_type,
        name=name,
        domain=domain,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        attributes=tuple(attributes),
    )
    label = f"node[{index}]:{op_type or '?'}"
    prefixed = [
        (f"{scope}/{label}.{attr}" if scope else f"{label}.{attr}", body)
        for attr, body in subgraphs
    ]
    return node, prefixed


@dataclass(frozen=True)
class _GraphParts:
    name: str
    inputs: tuple[ValueInfo, ...]
    outputs: tuple[ValueInfo, ...]
    value_info: tuple[ValueInfo, ...]
    initializers: tuple[InitializerInfo, ...]
    nodes: tuple[NodeInfo, ...]


def _parse_graph(cursor: _Cursor, scope: str, depth: int) -> _GraphParts:
    """Parse ``GraphProto``, recursing into subgraphs to flatten node lists."""
    if depth > MAX_SUBGRAPH_DEPTH:
        raise cursor.fail(
            f"subgraph nesting exceeds the maximum depth of {MAX_SUBGRAPH_DEPTH}"
        )
    name = ""
    nodes: list[NodeInfo] = []
    initializers: list[InitializerInfo] = []
    inputs: list[ValueInfo] = []
    outputs: list[ValueInfo] = []
    value_info: list[ValueInfo] = []
    pending: list[tuple[str, _Cursor]] = []
    while not cursor.exhausted:
        field_number, wire_type = cursor.read_tag()
        if field_number == 1 and wire_type == _WIRE_DELIMITED:
            node, bodies = _parse_node(
                cursor.submessage(f"{cursor.path}.node[{len(nodes)}]"),
                len(nodes),
                scope,
            )
            nodes.append(node)
            pending.extend(bodies)
        elif field_number == 2:
            name = _read_string(cursor, wire_type, f"{cursor.path}.name")
        elif field_number == 5 and wire_type == _WIRE_DELIMITED:
            initializers.append(
                _parse_tensor(
                    cursor.submessage(f"{cursor.path}.initializer[{len(initializers)}]")
                )
            )
        elif field_number == 11 and wire_type == _WIRE_DELIMITED:
            inputs.append(
                _parse_value_info(
                    cursor.submessage(f"{cursor.path}.input[{len(inputs)}]")
                )
            )
        elif field_number == 12 and wire_type == _WIRE_DELIMITED:
            outputs.append(
                _parse_value_info(
                    cursor.submessage(f"{cursor.path}.output[{len(outputs)}]")
                )
            )
        elif field_number == 13 and wire_type == _WIRE_DELIMITED:
            value_info.append(
                _parse_value_info(
                    cursor.submessage(f"{cursor.path}.value_info[{len(value_info)}]")
                )
            )
        else:
            cursor.skip(wire_type, field_number)
    for child_scope, body in pending:
        child = _parse_graph(body, child_scope, depth + 1)
        nodes.extend(child.nodes)
    return _GraphParts(
        name=name,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        value_info=tuple(value_info),
        initializers=tuple(initializers),
        nodes=tuple(nodes),
    )


def _parse_operator_set_id(cursor: _Cursor) -> tuple[str, int]:
    """Parse ``OperatorSetIdProto``."""
    domain = ""
    version = 0
    while not cursor.exhausted:
        field_number, wire_type = cursor.read_tag()
        if field_number == 1:
            domain = _read_string(cursor, wire_type, f"{cursor.path}.domain")
        elif field_number == 2 and wire_type == _WIRE_VARINT:
            version = _as_int64(cursor.read_varint())
        else:
            cursor.skip(wire_type, field_number)
    return domain, version


def parse_onnx_model(data: bytes) -> GraphSummary:
    """Parse a serialized ``ModelProto`` into a :class:`GraphSummary`.

    Raises:
        OnnxReadError: if ``data`` is not a bytes-like object, or if the bytes
            are truncated, use an unsupported wire type, or contain no
            ``GraphProto``.
    """
    try:
        buffer = memoryview(data)
    except TypeError as exc:
        raise OnnxReadError(
            "parse_onnx_model expects the serialized bytes of a ModelProto, "
            f"found {type(data).__name__}"
        ) from exc
    cursor = _Cursor(buffer, "ModelProto")
    ir_version = 0
    producer_name = ""
    producer_version = ""
    opset_imports: list[tuple[str, int]] = []
    graph: _GraphParts | None = None
    while not cursor.exhausted:
        field_number, wire_type = cursor.read_tag()
        if field_number == 1 and wire_type == _WIRE_VARINT:
            ir_version = _as_int64(cursor.read_varint())
        elif field_number == 2:
            producer_name = _read_string(cursor, wire_type, "ModelProto.producer_name")
        elif field_number == 3:
            producer_version = _read_string(
                cursor, wire_type, "ModelProto.producer_version"
            )
        elif field_number == 7 and wire_type == _WIRE_DELIMITED:
            graph = _parse_graph(cursor.submessage("GraphProto"), "", 0)
        elif field_number == 8 and wire_type == _WIRE_DELIMITED:
            opset_imports.append(
                _parse_operator_set_id(
                    cursor.submessage(f"ModelProto.opset_import[{len(opset_imports)}]")
                )
            )
        else:
            cursor.skip(wire_type, field_number)
    if graph is None:
        raise OnnxReadError(
            "ModelProto contains no graph (field 7); this is not a readable ONNX model"
        )
    return GraphSummary(
        ir_version=ir_version,
        producer_name=producer_name,
        producer_version=producer_version,
        opset_imports=tuple(opset_imports),
        graph_name=graph.name,
        inputs=graph.inputs,
        outputs=graph.outputs,
        value_info=graph.value_info,
        initializers=graph.initializers,
        nodes=graph.nodes,
    )


def read_onnx_model(
    path: Path | str, *, max_bytes: int = DEFAULT_MAX_BYTES
) -> GraphSummary:
    """Read and parse a single ``.onnx`` file.

    Only the named file is opened. External data sidecars referenced by
    initializers are never opened, followed, or read.

    Raises:
        OnnxReadError: if the path cannot be read, exceeds ``max_bytes``, or
            does not contain a readable ``ModelProto``.
    """
    onnx_path = Path(path)
    try:
        size = onnx_path.stat().st_size
    except OSError as exc:
        raise OnnxReadError(f"cannot stat ONNX file {onnx_path}: {exc}") from exc
    if size > max_bytes:
        raise OnnxReadError(
            f"refusing to read {onnx_path}: {size} bytes exceeds max_bytes "
            f"{max_bytes}; this guard exists so external-data sidecars are "
            "never loaded as graphs"
        )
    try:
        data = onnx_path.read_bytes()
    except OSError as exc:
        raise OnnxReadError(f"cannot read ONNX file {onnx_path}: {exc}") from exc
    try:
        return parse_onnx_model(data)
    except OnnxReadError as exc:
        raise OnnxReadError(f"{onnx_path}: {exc}") from exc
