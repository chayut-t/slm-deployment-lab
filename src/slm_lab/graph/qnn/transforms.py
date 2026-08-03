"""Ordered, declarative graph transformations for the T22 QNN candidate stage.

Each pass is a function over an in-memory ``onnx.ModelProto`` that returns a
structured record of exactly what it changed. The passes, their order, their
parameters, and the T21 rule ids each one addresses are declared in
``configs/graph/qnn-transforms-v1.json``; this module holds only the mechanics
and the measurement.

What these passes establish, and what they do not
-------------------------------------------------
Every effect record here is a count of graph structure before and after a
rewrite. None of it is compiler evidence. A candidate whose
``R-DATA-DEPENDENT-SHAPE-INPUT`` count fell from 804 to a smaller number has
fewer computed shape inputs than the reference; whether any converter accepts
either graph is unmeasured, and ``docs/results/onnx/graph-inspection.md``
section 7 remains the governing statement of that boundary.

Two of the passes deserve a stated limit. ``X-STATIC-SHAPE-FOLD`` evaluates
constant subexpressions with ``onnx.reference.ReferenceEvaluator``, so a folded
value is that implementation's answer, not a target kernel's; the allowlist is
restricted to shape, index, and small elementwise arithmetic so that no
normalization, no ``Softmax``, and no ``MatMul`` is ever evaluated at build
time, and the byte budget bounds both the tensors consumed and the tensors
produced. ``X-INFER-VALUE-INFO`` records how many of how many intermediate
tensors ONNX shape inference actually annotated; partial coverage is reported
as partial.

The frozen T12 static-cache contract is out of scope for every pass. The
decode graph's 56 ``ScatterElements`` writes and the prefill graph's 56
``Concat`` writes are preserved, and :func:`assert_cache_write_preserved`
fails loudly if they are not.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from slm_lab.graph.onnx_reader import GraphSummary


SCHEMA_VERSION = 1
MODULE_NAME = "slm_lab.graph.qnn.transforms"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CATALOGUE_PATH = PROJECT_ROOT / "configs/graph/qnn-transforms-v1.json"

PASS_FIELDS = (
    "id",
    "title",
    "order",
    "applied",
    "addresses",
    "observed_issue",
    "transformation",
    "parameters",
    "rationale",
    "references",
)

#: Every pass id this engine knows how to interpret. A catalogue naming an id
#: outside this set fails to load, so a catalogue can never silently declare a
#: transformation the engine does not implement.
APPLIED_PASS_IDS = (
    "X-CONSTANT-TO-INITIALIZER",
    "X-STATIC-SHAPE-FOLD",
    "X-DEAD-NODE-ELIMINATION",
    "X-EXTERNALIZE-LARGE-TENSORS",
    "X-INFER-VALUE-INFO",
    "X-STAMP-CANDIDATE-PROVENANCE",
)
REJECTED_PASS_IDS = ("X-ORT-CPU-OFFLINE-OPTIMIZATION",)
KNOWN_PASS_IDS = frozenset(APPLIED_PASS_IDS + REJECTED_PASS_IDS)

_PASS_ID_PATTERN = re.compile(r"X-[A-Z0-9]+(?:-[A-Z0-9]+)*")
_RULE_ID_PATTERN = re.compile(r"R-[A-Z0-9]+(?:-[A-Z0-9]+)*")

#: Prefixes of the T12 cache tensors on each graph kind's output boundary.
CACHE_OUTPUT_PREFIXES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "prefill": ("key_cache.", "value_cache."),
        "decode": ("present_key.", "present_value."),
    }
)
#: The operator that writes the cache on each graph kind, and how many hops of
#: producer chain separate it from the graph output. Prefill writes with
#: ``Concat`` into a ``Reshape``; decode's ``ScatterElements`` is the producer
#: of the graph output directly (graph-inspection.md 5.2).
CACHE_WRITE_OPERATORS: Mapping[str, tuple[str, int]] = MappingProxyType(
    {"prefill": ("Concat", 4), "decode": ("ScatterElements", 1)}
)
EXPECTED_BOUNDARY_COUNTS: Mapping[str, tuple[int, int]] = MappingProxyType(
    {"prefill": (3, 58), "decode": (60, 58)}
)
EXPECTED_CACHE_WRITES = 56


class QnnTransformError(ValueError):
    """A transform catalogue, transform input, or post-condition is invalid."""


@dataclass(frozen=True)
class TransformPass:
    """One declared pass from the committed transformation catalogue."""

    id: str
    title: str
    order: int
    applied: bool
    addresses: tuple[str, ...]
    observed_issue: str
    transformation: str
    parameters: Mapping[str, Any]
    rationale: str
    references: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "order": self.order,
            "applied": self.applied,
            "addresses": list(self.addresses),
            "observed_issue": self.observed_issue,
            "transformation": self.transformation,
            "parameters": dict(self.parameters),
            "rationale": self.rationale,
            "references": list(self.references),
        }


# --------------------------------------------------------------------------
# Catalogue loading and validation
# --------------------------------------------------------------------------


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QnnTransformError(f"{label} must be a non-empty string")
    return value


def _parse_pass(payload: Any, index: int) -> TransformPass:
    label = f"pass[{index}]"
    if not isinstance(payload, dict):
        raise QnnTransformError(f"{label} must be a JSON object")
    missing = [field for field in PASS_FIELDS if field not in payload]
    if missing:
        raise QnnTransformError(f"{label} is missing fields: {sorted(missing)}")
    unexpected = sorted(set(payload) - set(PASS_FIELDS))
    if unexpected:
        raise QnnTransformError(f"{label} has unexpected fields: {unexpected}")

    pass_id = _require_text(payload["id"], f"{label} id")
    label = f"pass {pass_id}"
    if not _PASS_ID_PATTERN.fullmatch(pass_id):
        raise QnnTransformError(f"{label} id must match {_PASS_ID_PATTERN.pattern!r}")
    if pass_id not in KNOWN_PASS_IDS:
        raise QnnTransformError(
            f"{label} is not implemented by {MODULE_NAME}; known pass ids: "
            f"{sorted(KNOWN_PASS_IDS)}"
        )
    order = payload["order"]
    if type(order) is not int or order < 1:
        raise QnnTransformError(f"{label} order must be a positive integer")
    applied = payload["applied"]
    if type(applied) is not bool:
        raise QnnTransformError(f"{label} applied must be a boolean")
    if applied and pass_id not in APPLIED_PASS_IDS:
        raise QnnTransformError(
            f"{label} declares applied=true but the engine has no implementation for it"
        )
    if not applied and pass_id in APPLIED_PASS_IDS:
        raise QnnTransformError(
            f"{label} declares applied=false but the engine implements it as an "
            "applied pass; a disabled pass must be a recorded rejection"
        )
    addresses = payload["addresses"]
    if not isinstance(addresses, list):
        raise QnnTransformError(f"{label} addresses must be a list of rule ids")
    for position, rule_id in enumerate(addresses):
        _require_text(rule_id, f"{label} addresses[{position}]")
        if not _RULE_ID_PATTERN.fullmatch(rule_id):
            raise QnnTransformError(
                f"{label} addresses[{position}] is not a risk rule id: {rule_id!r}"
            )
    parameters = payload["parameters"]
    if not isinstance(parameters, dict):
        raise QnnTransformError(f"{label} parameters must be a JSON object")
    references = payload["references"]
    if not isinstance(references, list) or not references:
        raise QnnTransformError(f"{label} references must be a non-empty list")
    for position, reference in enumerate(references):
        _require_text(reference, f"{label} references[{position}]")
    return TransformPass(
        id=pass_id,
        title=_require_text(payload["title"], f"{label} title"),
        order=order,
        applied=applied,
        addresses=tuple(str(rule_id) for rule_id in addresses),
        observed_issue=_require_text(
            payload["observed_issue"], f"{label} observed_issue"
        ),
        transformation=_require_text(
            payload["transformation"], f"{label} transformation"
        ),
        parameters=MappingProxyType(dict(parameters)),
        rationale=_require_text(payload["rationale"], f"{label} rationale"),
        references=tuple(str(reference) for reference in references),
    )


def load_transform_catalogue(
    path: Path | str = DEFAULT_CATALOGUE_PATH,
) -> tuple[str, tuple[TransformPass, ...]]:
    """Load and validate the declarative transformation catalogue at ``path``."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise QnnTransformError(
            f"cannot read transform catalogue {source}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise QnnTransformError(
            f"invalid JSON in transform catalogue {source}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise QnnTransformError(f"transform catalogue {source} must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise QnnTransformError(
            f"transform catalogue {source} must declare schema_version "
            f"{SCHEMA_VERSION}, found {payload.get('schema_version')!r}"
        )
    catalogue_id = _require_text(payload.get("catalogue_id"), "catalogue_id")
    _require_text(payload.get("description"), "catalogue description")
    _require_text(payload.get("target_context"), "catalogue target_context")
    _require_text(payload.get("source_evidence"), "catalogue source_evidence")
    raw_passes = payload.get("passes")
    if not isinstance(raw_passes, list) or not raw_passes:
        raise QnnTransformError(
            f"transform catalogue {source} must define a non-empty passes array"
        )
    passes = tuple(_parse_pass(entry, index) for index, entry in enumerate(raw_passes))
    seen: set[str] = set()
    for entry in passes:
        if entry.id in seen:
            raise QnnTransformError(f"duplicate pass id {entry.id!r} in {source}")
        seen.add(entry.id)
    orders = [entry.order for entry in passes]
    if orders != sorted(orders) or orders != list(range(1, len(orders) + 1)):
        raise QnnTransformError(
            f"transform catalogue {source} must declare orders 1..{len(orders)} "
            f"in ascending declaration order, found {orders}"
        )
    missing = sorted(set(APPLIED_PASS_IDS) - seen)
    if missing:
        raise QnnTransformError(
            f"transform catalogue {source} omits required applied passes: {missing}"
        )
    return catalogue_id, passes


def applied_passes(passes: Sequence[TransformPass]) -> tuple[TransformPass, ...]:
    """Return the applied passes in declared order."""

    return tuple(entry for entry in passes if entry.applied)


def rejected_passes(passes: Sequence[TransformPass]) -> tuple[TransformPass, ...]:
    """Return the recorded-but-not-applied passes in declared order."""

    return tuple(entry for entry in passes if not entry.applied)


# --------------------------------------------------------------------------
# Lazy ONNX access
# --------------------------------------------------------------------------


def _require_onnx() -> Any:
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise QnnTransformError(
            "the onnx package is required to run a transform pass; the locked "
            "root environment deliberately does not have it, so build the "
            "candidates with an environment that does"
        ) from exc
    return onnx


def _require_numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise QnnTransformError(
            "the numpy package is required to evaluate a constant subexpression"
        ) from exc
    return numpy


def _require_reference_evaluator() -> Any:
    try:
        from onnx.reference import ReferenceEvaluator
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise QnnTransformError(
            "onnx.reference.ReferenceEvaluator is required by X-STATIC-SHAPE-FOLD"
        ) from exc
    return ReferenceEvaluator


# --------------------------------------------------------------------------
# Tensor helpers
# --------------------------------------------------------------------------


def _element_size(onnx: Any, elem_type: int) -> int | None:
    try:
        return int(onnx.helper.tensor_dtype_to_np_dtype(elem_type).itemsize)
    except (KeyError, TypeError, ValueError):
        return None


def _shape_bytes(onnx: Any, tensor: Any) -> int:
    element_size = _element_size(onnx, tensor.data_type)
    if element_size is None:
        return 0
    elements = 1
    for dimension in tensor.dims:
        elements *= int(dimension)
    return elements * element_size


def declared_tensor_bytes(tensor: Any) -> int:
    """Return the declared payload size of a ``TensorProto`` without loading it.

    An external tensor is sized from its declared ``length`` entry, so this
    never opens an external-data sidecar.
    """

    onnx = _require_onnx()
    if tensor.data_location == onnx.TensorProto.EXTERNAL:
        for entry in tensor.external_data:
            if entry.key == "length":
                try:
                    return int(entry.value)
                except ValueError:
                    break
        return _shape_bytes(onnx, tensor)
    if tensor.raw_data:
        return len(tensor.raw_data)
    return _shape_bytes(onnx, tensor)


def _is_external(onnx: Any, tensor: Any) -> bool:
    return tensor.data_location == onnx.TensorProto.EXTERNAL


# --------------------------------------------------------------------------
# Pass 1: X-CONSTANT-TO-INITIALIZER
# --------------------------------------------------------------------------


def constant_to_initializer(model: Any) -> dict[str, Any]:
    """Rewrite every ``Constant`` node carrying a tensor as a graph initializer.

    Semantics-preserving: the initializer takes the name the node produced, so
    no consumer edge changes. This is what puts the O(S^2) causal mask and the
    56 float16 zero cache reserves inside ``graph.initializer``, where the
    external-data threshold can see them at all.
    """

    onnx = _require_onnx()
    graph = model.graph
    graph_output_names = {value.name for value in graph.output}
    constants_before = sum(1 for node in graph.node if node.op_type == "Constant")
    nodes_before = len(graph.node)
    initializers_before = len(graph.initializer)

    kept: list[Any] = []
    created: list[Any] = []
    skipped: Counter[str] = Counter()
    moved_bytes = 0
    largest_bytes = 0
    for node in graph.node:
        if node.op_type != "Constant":
            kept.append(node)
            continue
        tensor = next(
            (
                attribute.t
                for attribute in node.attribute
                if attribute.name == "value"
                and attribute.type == onnx.AttributeProto.TENSOR
            ),
            None,
        )
        if tensor is None:
            skipped["unsupported_attribute_form"] += 1
            kept.append(node)
            continue
        outputs = [name for name in node.output if name]
        if len(outputs) != 1:
            skipped["unexpected_output_arity"] += 1
            kept.append(node)
            continue
        if outputs[0] in graph_output_names:
            # An initializer is not a node, so promoting a Constant whose output
            # is also a graph output would leave that output unproduced.
            skipped["output_is_graph_output"] += 1
            kept.append(node)
            continue
        initializer = onnx.TensorProto()
        initializer.CopyFrom(tensor)
        initializer.name = outputs[0]
        size = declared_tensor_bytes(initializer)
        moved_bytes += size
        largest_bytes = max(largest_bytes, size)
        created.append(initializer)

    del graph.node[:]
    graph.node.extend(kept)
    graph.initializer.extend(created)
    return {
        "constant_nodes_before": constants_before,
        "converted_to_initializer": len(created),
        "skipped_by_reason": dict(sorted(skipped.items())),
        "node_count_before": nodes_before,
        "node_count_after": len(graph.node),
        "initializer_count_before": initializers_before,
        "initializer_count_after": len(graph.initializer),
        "bytes_moved_to_initializers": moved_bytes,
        "largest_converted_tensor_bytes": largest_bytes,
    }


# --------------------------------------------------------------------------
# Pass 2: X-STATIC-SHAPE-FOLD
# --------------------------------------------------------------------------


def assert_topological_order(graph: Any) -> None:
    """Fail unless every node input is produced before the node that reads it.

    ``X-STATIC-SHAPE-FOLD`` reaches its fixpoint in one forward sweep only
    because the ONNX specification requires the node list to be topologically
    sorted. That assumption is checked here rather than relied on silently.
    """

    available = {initializer.name for initializer in graph.initializer}
    available.update(value.name for value in graph.input)
    for index, node in enumerate(graph.node):
        for position, name in enumerate(node.input):
            if name and name not in available:
                raise QnnTransformError(
                    f"graph is not topologically ordered: node[{index}] "
                    f"{node.op_type} {node.name or '<unnamed>'} reads "
                    f"input[{position}]={name!r} before it is produced"
                )
        available.update(name for name in node.output if name)


def _fold_model(onnx: Any, model: Any, node: Any, initializers: Sequence[Any]) -> Any:
    outputs = [
        onnx.helper.make_empty_tensor_value_info(name) for name in node.output if name
    ]
    graph = onnx.helper.make_graph(
        [node],
        "static-shape-fold",
        [],
        outputs,
        initializer=list(initializers),
    )
    folded = onnx.helper.make_model(graph, opset_imports=list(model.opset_import))
    folded.ir_version = model.ir_version
    return folded


def static_shape_fold(
    model: Any,
    *,
    allowed_ops: Iterable[str],
    max_input_bytes: int,
    max_output_bytes: int,
) -> dict[str, Any]:
    """Replace nodes whose every input is a known inline constant.

    Addresses ``R-DATA-DEPENDENT-SHAPE-INPUT`` and ``R-SHAPE-COMPUTATION-CHAIN``:
    the shape-defining inputs those rules count are computed by ``Shape`` ->
    ``Gather`` -> arithmetic chains that a static boundary makes constant. Only
    allowlisted operators are evaluated, both the consumed and the produced
    bytes are budgeted, and an initializer that lives in external data is not in
    the constant pool at all, so no float weight can be an input to a fold.
    """

    onnx = _require_onnx()
    numpy = _require_numpy()
    evaluator_type = _require_reference_evaluator()
    graph = model.graph
    assert_topological_order(graph)

    allowlist = frozenset(str(op_type) for op_type in allowed_ops)
    pool: dict[str, Any] = {
        initializer.name: initializer
        for initializer in graph.initializer
        if not _is_external(onnx, initializer)
    }
    external_names = {
        initializer.name
        for initializer in graph.initializer
        if _is_external(onnx, initializer)
    }
    pool_before = len(pool)
    graph_output_names = {value.name for value in graph.output}

    kept: list[Any] = []
    created: list[Any] = []
    skipped: Counter[str] = Counter()
    folded_by_op: Counter[str] = Counter()
    created_bytes = 0
    for node in graph.node:
        outputs = [name for name in node.output if name]
        if node.op_type not in allowlist:
            skipped["operator_not_allowlisted"] += 1
            kept.append(node)
            continue
        if any(
            attribute.type == onnx.AttributeProto.GRAPH for attribute in node.attribute
        ):
            skipped["subgraph_attribute"] += 1
            kept.append(node)
            continue
        if any(name in graph_output_names for name in outputs):
            skipped["output_is_graph_output"] += 1
            kept.append(node)
            continue
        operands = [name for name in node.input if name]
        if any(name not in pool for name in operands):
            reason = (
                "input_in_external_data"
                if any(name in external_names for name in operands)
                else "input_not_constant"
            )
            skipped[reason] += 1
            kept.append(node)
            continue
        input_bytes = sum(declared_tensor_bytes(pool[name]) for name in operands)
        if input_bytes > max_input_bytes:
            skipped["input_bytes_over_budget"] += 1
            kept.append(node)
            continue
        try:
            evaluator = evaluator_type(
                _fold_model(onnx, model, node, [pool[name] for name in operands])
            )
            results = evaluator.run(None, {})
        except Exception:  # noqa: BLE001 - any evaluation failure is a skip
            skipped["evaluation_failed"] += 1
            kept.append(node)
            continue
        if len(results) != len(outputs):
            skipped["evaluation_output_arity_mismatch"] += 1
            kept.append(node)
            continue
        arrays = [numpy.asarray(result) for result in results]
        output_bytes = sum(int(array.nbytes) for array in arrays)
        if output_bytes > max_output_bytes:
            skipped["output_bytes_over_budget"] += 1
            kept.append(node)
            continue
        try:
            tensors = [
                onnx.numpy_helper.from_array(array, name)
                for array, name in zip(arrays, outputs)
            ]
        except Exception:  # noqa: BLE001 - an unrepresentable result is a skip
            skipped["result_not_representable"] += 1
            kept.append(node)
            continue
        for tensor in tensors:
            pool[tensor.name] = tensor
            created.append(tensor)
        created_bytes += output_bytes
        folded_by_op[node.op_type] += 1

    nodes_before = len(graph.node)
    del graph.node[:]
    graph.node.extend(kept)
    graph.initializer.extend(created)
    return {
        "node_count_before": nodes_before,
        "node_count_after": len(graph.node),
        "folded_nodes": sum(folded_by_op.values()),
        "folded_by_operator": dict(sorted(folded_by_op.items())),
        "initializers_created": len(created),
        "bytes_created": created_bytes,
        "skipped_by_reason": dict(sorted(skipped.items())),
        "constant_pool_before": pool_before,
        "constant_pool_after": len(pool),
        "external_initializers_excluded_from_pool": len(external_names),
        "forward_passes": 1,
        "topological_order_verified": True,
        "max_input_bytes": int(max_input_bytes),
        "max_output_bytes": int(max_output_bytes),
    }


# --------------------------------------------------------------------------
# Pass 3: X-DEAD-NODE-ELIMINATION
# --------------------------------------------------------------------------


def dead_node_elimination(model: Any) -> dict[str, Any]:
    """Drop nodes and initializers no graph output transitively depends on.

    Graph inputs are never removed: the public boundary is the frozen T12
    contract, and a decode graph that lost an unused cache input would no
    longer satisfy it.

    The backward liveness sweep visits each node once and so depends on the
    same topological ordering :func:`static_shape_fold` depends on: on an
    unordered node list a live producer that appears after its consumer would
    be seen before the consumer marked its inputs required, and would be
    dropped. That assumption is asserted here rather than relied on silently.
    """

    graph = model.graph
    assert_topological_order(graph)
    nodes_before = len(graph.node)
    initializers_before = len(graph.initializer)
    value_info_before = len(graph.value_info)

    required = {value.name for value in graph.output if value.name}
    kept_reversed: list[Any] = []
    removed_by_op: Counter[str] = Counter()
    for node in reversed(graph.node):
        produced = [name for name in node.output if name]
        if any(name in required for name in produced):
            kept_reversed.append(node)
            required.update(name for name in node.input if name)
        else:
            removed_by_op[node.op_type] += 1
    kept = list(reversed(kept_reversed))

    kept_initializers = [
        initializer for initializer in graph.initializer if initializer.name in required
    ]
    removed_initializer_bytes = sum(
        declared_tensor_bytes(initializer)
        for initializer in graph.initializer
        if initializer.name not in required
    )
    kept_value_info = [value for value in graph.value_info if value.name in required]

    del graph.node[:]
    graph.node.extend(kept)
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)
    del graph.value_info[:]
    graph.value_info.extend(kept_value_info)
    return {
        "node_count_before": nodes_before,
        "node_count_after": len(graph.node),
        "nodes_removed": nodes_before - len(graph.node),
        "nodes_removed_by_operator": dict(sorted(removed_by_op.items())),
        "initializer_count_before": initializers_before,
        "initializer_count_after": len(graph.initializer),
        "initializers_removed": initializers_before - len(graph.initializer),
        "initializer_bytes_removed": removed_initializer_bytes,
        "value_info_removed": value_info_before - len(graph.value_info),
        "graph_inputs_preserved": len(graph.input),
    }


# --------------------------------------------------------------------------
# Pass 4: X-EXTERNALIZE-LARGE-TENSORS
# --------------------------------------------------------------------------


def externalize_large_tensors(
    model: Any,
    *,
    size_threshold_bytes: int,
    location: str,
) -> dict[str, Any]:
    """Decide which initializers the candidate writes into its own sidecar.

    The decision is taken here, over the graph as passes 1 to 3 left it, and is
    realized by :func:`write_candidate` when the candidate is serialized;
    passes 5 and 6 add no initializer, so taking the decision at this point and
    writing it at the end are the same decision. Nothing is mutated here, and
    no sidecar is opened: an external tensor is sized from its declared length.
    """

    onnx = _require_onnx()
    graph = model.graph
    already_external = 0
    newly_external = 0
    kept_inline = 0
    newly_external_bytes = 0
    inline_bytes = 0
    largest_new = 0
    for initializer in graph.initializer:
        size = declared_tensor_bytes(initializer)
        if _is_external(onnx, initializer):
            already_external += 1
            continue
        if size >= size_threshold_bytes:
            newly_external += 1
            newly_external_bytes += size
            largest_new = max(largest_new, size)
        else:
            kept_inline += 1
            inline_bytes += size
    return {
        "size_threshold_bytes": int(size_threshold_bytes),
        "location": location,
        "initializer_count": len(graph.initializer),
        "already_external": already_external,
        "newly_externalized": newly_external,
        "kept_inline": kept_inline,
        "bytes_newly_externalized": newly_external_bytes,
        "bytes_kept_inline": inline_bytes,
        "largest_newly_externalized_bytes": largest_new,
        "realized_at": "serialization",
    }


def write_candidate(
    model: Any,
    destination: Path,
    *,
    source_directory: Path,
    size_threshold_bytes: int,
    location: str,
) -> None:
    """Materialize external data from the source graph and save the candidate.

    The reference sidecar is read, never written. The candidate gets its own
    sidecar at ``location``, so the reference artifacts stay byte-identical and
    independently identifiable.
    """

    onnx = _require_onnx()
    from onnx.external_data_helper import load_external_data_for_model

    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.parent / location
    if sidecar.exists():
        sidecar.unlink()
    load_external_data_for_model(model, str(source_directory))
    onnx.save_model(
        model,
        str(destination),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=location,
        size_threshold=size_threshold_bytes,
        convert_attribute=False,
    )


# --------------------------------------------------------------------------
# Pass 5: X-INFER-VALUE-INFO
# --------------------------------------------------------------------------


def _intermediate_tensor_names(graph: Any) -> set[str]:
    produced: set[str] = set()
    for node in graph.node:
        produced.update(name for name in node.output if name)
    return produced - {value.name for value in graph.output}


def _is_fully_static(value: Any) -> bool:
    tensor_type = value.type.tensor_type
    if not value.type.HasField("tensor_type"):
        return False
    if not tensor_type.HasField("shape"):
        return False
    return all(dimension.HasField("dim_value") for dimension in tensor_type.shape.dim)


def infer_value_info(
    model: Any,
    *,
    check_type: bool = False,
    strict_mode: bool = False,
    data_prop: bool = True,
) -> dict[str, Any]:
    """Run ONNX shape inference and keep the inferred ``value_info``.

    ``docs/results/onnx/graph-inspection.md`` section 7 names this as the
    follow-up that closes the ``R-INTERNAL-DYNAMIC-SHAPE`` evidence boundary:
    the eight reference graphs carry no ``value_info`` at all, so the rule
    inspected 0 of 0 entries. The record below states how many of how many
    intermediate tensors inference actually annotated. Partial coverage is
    reported as partial; no claim is made beyond what was resolved.
    """

    onnx = _require_onnx()
    graph = model.graph
    before = len(graph.value_info)
    intermediate = _intermediate_tensor_names(graph)
    try:
        inferred = onnx.shape_inference.infer_shapes(
            model,
            check_type=check_type,
            strict_mode=strict_mode,
            data_prop=data_prop,
        )
    except Exception as exc:  # noqa: BLE001 - inference failure is recorded
        return {
            "status": "not_measured",
            "reason": f"onnx.shape_inference failed: {exc}",
            "value_info_before": before,
            "value_info_after": before,
            "intermediate_tensors": len(intermediate),
            "intermediate_tensors_annotated": 0,
        }
    del graph.value_info[:]
    graph.value_info.extend(inferred.graph.value_info)

    annotated = [value for value in graph.value_info if value.name in intermediate]
    static = [value for value in annotated if _is_fully_static(value)]
    if not intermediate:
        coverage = "none_required"
    elif len(annotated) == len(intermediate):
        coverage = "complete"
    elif annotated:
        coverage = "partial"
    else:
        coverage = "none"
    return {
        "status": "measured",
        "value_info_before": before,
        "value_info_after": len(graph.value_info),
        "intermediate_tensors": len(intermediate),
        "intermediate_tensors_annotated": len(annotated),
        "intermediate_tensors_unannotated": len(intermediate) - len(annotated),
        "annotated_fully_static": len(static),
        "annotated_not_fully_static": len(annotated) - len(static),
        "coverage": coverage,
        "check_type": check_type,
        "strict_mode": strict_mode,
        "data_prop": data_prop,
    }


# --------------------------------------------------------------------------
# Pass 6: X-STAMP-CANDIDATE-PROVENANCE
# --------------------------------------------------------------------------


def stamp_candidate_provenance(
    model: Any,
    *,
    producer_name: str,
    producer_version: str,
    metadata: Mapping[str, str],
) -> dict[str, Any]:
    """Record the source digest, catalogue identity, and pass list in-band."""

    onnx = _require_onnx()
    before = " ".join(
        part for part in (model.producer_name, model.producer_version) if part
    )
    model.producer_name = producer_name
    model.producer_version = producer_version
    del model.metadata_props[:]
    for key in sorted(metadata):
        entry = onnx.StringStringEntryProto()
        entry.key = key
        entry.value = str(metadata[key])
        model.metadata_props.append(entry)
    return {
        "producer_before": before or "unknown",
        "producer_after": f"{producer_name} {producer_version}",
        "metadata_props": {key: str(metadata[key]) for key in sorted(metadata)},
        "ir_version": int(model.ir_version),
        "opset_imports": [
            [entry.domain, int(entry.version)] for entry in model.opset_import
        ],
    }


# --------------------------------------------------------------------------
# Post-conditions
# --------------------------------------------------------------------------


def _boundary_signature(values: Sequence[Any]) -> tuple[tuple[str, str, tuple], ...]:
    return tuple(
        (
            value.name,
            value.dtype,
            tuple(value.shape.dims) if value.shape is not None else (),
        )
        for value in values
    )


def assert_boundary_preserved(
    reference: GraphSummary,
    candidate: GraphSummary,
    *,
    graph_kind: str,
    label: str,
) -> None:
    """Fail unless the candidate boundary is the reference boundary exactly.

    Names, order, dtypes, and shapes must match on both sides, and the counts
    must be the ones the frozen T12 contract fixes: prefill 3 in / 58 out,
    decode 60 in / 58 out.
    """

    expected = EXPECTED_BOUNDARY_COUNTS.get(graph_kind)
    if expected is None:
        raise QnnTransformError(f"{label}: unknown graph kind {graph_kind!r}")
    for side, reference_values, candidate_values, expected_count in (
        ("input", reference.inputs, candidate.inputs, expected[0]),
        ("output", reference.outputs, candidate.outputs, expected[1]),
    ):
        reference_signature = _boundary_signature(reference_values)
        candidate_signature = _boundary_signature(candidate_values)
        if len(candidate_signature) != expected_count:
            raise QnnTransformError(
                f"{label}: candidate declares {len(candidate_signature)} "
                f"{side} tensors, the T12 {graph_kind} contract fixes "
                f"{expected_count}"
            )
        if reference_signature != candidate_signature:
            missing = [
                item for item in reference_signature if item not in candidate_signature
            ]
            added = [
                item for item in candidate_signature if item not in reference_signature
            ]
            raise QnnTransformError(
                f"{label}: candidate {side} boundary differs from the reference; "
                f"missing={missing[:4]} added={added[:4]}"
            )


def count_cache_write_nodes(
    summary: GraphSummary,
    *,
    graph_kind: str,
) -> int:
    """Count the cache outputs whose producer chain contains the write operator.

    ``graph-inspection.md`` 5.2 records what each graph kind uses: decode's 56
    ``ScatterElements`` produce the ``present_*`` outputs directly, and
    prefill's 56 ``Concat`` writes reach ``key_cache.L`` / ``value_cache.L``
    through a ``Reshape``.
    """

    prefixes = CACHE_OUTPUT_PREFIXES.get(graph_kind)
    write = CACHE_WRITE_OPERATORS.get(graph_kind)
    if prefixes is None or write is None:
        raise QnnTransformError(f"unknown graph kind {graph_kind!r}")
    op_type, max_depth = write
    producers: dict[str, Any] = {}
    for node in summary.nodes:
        for name in node.outputs:
            if name:
                producers.setdefault(name, node)
    found = 0
    for value in summary.outputs:
        if not value.name.startswith(prefixes):
            continue
        frontier = [value.name]
        seen: set[str] = set()
        for _ in range(max_depth):
            next_frontier: list[str] = []
            matched = False
            for name in frontier:
                node = producers.get(name)
                if node is None or name in seen:
                    continue
                seen.add(name)
                if node.op_type == op_type:
                    matched = True
                    break
                next_frontier.extend(item for item in node.inputs if item)
            if matched:
                found += 1
                break
            frontier = next_frontier
            if not frontier:
                break
    return found


def assert_cache_write_preserved(
    summary: GraphSummary,
    *,
    graph_kind: str,
    label: str,
) -> int:
    """Fail unless all 56 T12 cache writes are still present and reachable."""

    found = count_cache_write_nodes(summary, graph_kind=graph_kind)
    if found != EXPECTED_CACHE_WRITES:
        op_type = CACHE_WRITE_OPERATORS[graph_kind][0]
        raise QnnTransformError(
            f"{label}: {found} of {EXPECTED_CACHE_WRITES} T12 cache outputs are "
            f"still written by a {op_type}; the static-cache contract did not "
            "survive the transformation profile"
        )
    return found
