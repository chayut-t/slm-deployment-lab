"""Each transform pass, measured on a tiny graph with the real structural shapes.

These cases need ``onnx`` and are skipped in the locked root environment, which
deliberately has none. The synthetic model reproduces the three structures the
passes exist for: a ``Constant``-fed ``Reshape`` shape chain, a large inline
``Constant``, and a dead branch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_lab.graph.onnx_reader import read_onnx_model
from slm_lab.graph.qnn.transforms import (
    QnnTransformError,
    assert_topological_order,
    constant_to_initializer,
    dead_node_elimination,
    declared_tensor_bytes,
    externalize_large_tensors,
    infer_value_info,
    static_shape_fold,
    stamp_candidate_provenance,
    write_candidate,
)


BIG_SIDE = 512
BIG_BYTES = BIG_SIDE * BIG_SIDE * 4
FOLD_BUDGET = 1024 * 1024
ALLOWED_OPS = (
    "Add",
    "Cast",
    "Concat",
    "ConstantOfShape",
    "Gather",
    "Identity",
    "Mul",
    "Reshape",
    "Shape",
    "Slice",
    "Squeeze",
    "Sub",
    "Unsqueeze",
    "Where",
)


def _constant_node(onnx, numpy, name: str, array, output: str):
    return onnx.helper.make_node(
        "Constant",
        [],
        [output],
        name=name,
        value=onnx.numpy_helper.from_array(numpy.asarray(array), output),
    )


def _synthetic_model(onnx, numpy):
    """Return a model with a shape chain, a large constant, and a dead branch."""

    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 3, 4])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 24])
    live = onnx.helper.make_tensor_value_info(
        "live_root", onnx.TensorProto.FLOAT, [2, 3, 4]
    )
    pinned = onnx.helper.make_tensor_value_info("pinned", onnx.TensorProto.INT64, [1])
    nodes = [
        _constant_node(
            onnx, numpy, "const_a", numpy.array([1], dtype=numpy.int64), "shape_a"
        ),
        _constant_node(
            onnx, numpy, "const_b", numpy.array([24], dtype=numpy.int64), "shape_b"
        ),
        onnx.helper.make_node(
            "Concat", ["shape_a", "shape_b"], ["target_shape"], name="cat", axis=0
        ),
        onnx.helper.make_node("Reshape", ["x", "target_shape"], ["y"], name="rs"),
        _constant_node(
            onnx,
            numpy,
            "const_big",
            numpy.zeros((BIG_SIDE, BIG_SIDE), dtype=numpy.float32),
            "big_const",
        ),
        onnx.helper.make_node(
            "Mul", ["big_const", "big_const"], ["big_squared"], name="mul_big"
        ),
        onnx.helper.make_node("Sqrt", ["x"], ["live_root"], name="sqrt_live"),
        onnx.helper.make_node("Sqrt", ["x"], ["dead_root"], name="sqrt_dead"),
        _constant_node(
            onnx, numpy, "const_pinned", numpy.array([7], dtype=numpy.int64), "pinned"
        ),
    ]
    graph = onnx.helper.make_graph(nodes, "synthetic", [x], [y, live, pinned])
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 18)]
    )
    model.ir_version = 8
    return model


def _op_types(model) -> list[str]:
    return [node.op_type for node in model.graph.node]


def _initializer_names(model) -> set[str]:
    return {initializer.name for initializer in model.graph.initializer}


@pytest.fixture
def onnx_module():
    return pytest.importorskip("onnx")


@pytest.fixture
def numpy_module():
    return pytest.importorskip("numpy")


# --------------------------------------------------------------------------
# X-CONSTANT-TO-INITIALIZER
# --------------------------------------------------------------------------


def test_constant_nodes_become_initializers_under_the_same_names(
    onnx_module, numpy_module
) -> None:
    model = _synthetic_model(onnx_module, numpy_module)

    effect = constant_to_initializer(model)

    assert effect["constant_nodes_before"] == 4
    assert effect["converted_to_initializer"] == 3
    assert effect["node_count_before"] - effect["node_count_after"] == 3
    assert {"shape_a", "shape_b", "big_const"} <= _initializer_names(model)
    assert effect["bytes_moved_to_initializers"] == BIG_BYTES + 16
    assert effect["largest_converted_tensor_bytes"] == BIG_BYTES


def test_a_constant_that_is_a_graph_output_is_left_alone(
    onnx_module, numpy_module
) -> None:
    model = _synthetic_model(onnx_module, numpy_module)

    effect = constant_to_initializer(model)

    assert effect["skipped_by_reason"] == {"output_is_graph_output": 1}
    assert "pinned" not in _initializer_names(model)
    assert "Constant" in _op_types(model)


# --------------------------------------------------------------------------
# X-STATIC-SHAPE-FOLD
# --------------------------------------------------------------------------


def _folded(onnx_module, numpy_module):
    model = _synthetic_model(onnx_module, numpy_module)
    constant_to_initializer(model)
    effect = static_shape_fold(
        model,
        allowed_ops=ALLOWED_OPS,
        max_input_bytes=FOLD_BUDGET,
        max_output_bytes=FOLD_BUDGET,
    )
    return model, effect


def test_a_constant_shape_chain_folds_into_one_initializer(
    onnx_module, numpy_module
) -> None:
    model, effect = _folded(onnx_module, numpy_module)

    assert effect["folded_by_operator"] == {"Concat": 1}
    assert "Concat" not in _op_types(model)
    assert "target_shape" in _initializer_names(model)
    folded = next(
        initializer
        for initializer in model.graph.initializer
        if initializer.name == "target_shape"
    )
    assert list(onnx_module.numpy_helper.to_array(folded)) == [1, 24]


def test_the_reshape_target_shape_stops_being_a_computed_input(
    onnx_module, numpy_module
) -> None:
    model, _ = _folded(onnx_module, numpy_module)

    producers = {name for node in model.graph.node for name in node.output if name}
    reshape = next(node for node in model.graph.node if node.op_type == "Reshape")
    assert reshape.input[1] == "target_shape"
    assert reshape.input[1] not in producers
    assert reshape.input[1] in _initializer_names(model)


def test_a_tensor_over_the_input_budget_is_never_folded(
    onnx_module, numpy_module
) -> None:
    model, effect = _folded(onnx_module, numpy_module)

    assert effect["skipped_by_reason"]["input_bytes_over_budget"] == 1
    assert "Mul" in _op_types(model)
    assert "big_squared" not in _initializer_names(model)


def test_an_evaluation_over_the_output_budget_is_discarded(
    onnx_module, numpy_module
) -> None:
    shape = onnx_module.numpy_helper.from_array(
        numpy_module.array([1024, 512], dtype=numpy_module.int64), "big_shape"
    )
    out = onnx_module.helper.make_tensor_value_info(
        "out", onnx_module.TensorProto.FLOAT, [1024, 512]
    )
    nodes = [
        onnx_module.helper.make_node(
            "ConstantOfShape", ["big_shape"], ["big_zeros"], name="cos"
        ),
        onnx_module.helper.make_node("Sqrt", ["big_zeros"], ["out"], name="keep"),
    ]
    graph = onnx_module.helper.make_graph(
        nodes, "budget", [], [out], initializer=[shape]
    )
    model = onnx_module.helper.make_model(
        graph, opset_imports=[onnx_module.helper.make_opsetid("", 18)]
    )
    model.ir_version = 8

    effect = static_shape_fold(
        model,
        allowed_ops=ALLOWED_OPS,
        max_input_bytes=FOLD_BUDGET,
        max_output_bytes=FOLD_BUDGET,
    )

    assert effect["folded_nodes"] == 0
    assert effect["skipped_by_reason"]["output_bytes_over_budget"] == 1
    assert "ConstantOfShape" in _op_types(model)


def test_an_external_initializer_is_never_in_the_constant_pool(
    onnx_module, numpy_module
) -> None:
    weight = onnx_module.numpy_helper.from_array(
        numpy_module.zeros((4,), dtype=numpy_module.float32), "weight"
    )
    weight.ClearField("raw_data")
    weight.data_location = onnx_module.TensorProto.EXTERNAL
    entry = weight.external_data.add()
    entry.key = "location"
    entry.value = "weights.onnx.data"
    length = weight.external_data.add()
    length.key = "length"
    length.value = "16"
    out = onnx_module.helper.make_tensor_value_info(
        "out", onnx_module.TensorProto.FLOAT, [4]
    )
    nodes = [
        onnx_module.helper.make_node("Identity", ["weight"], ["copy"], name="id"),
        onnx_module.helper.make_node("Sqrt", ["copy"], ["out"], name="keep"),
    ]
    graph = onnx_module.helper.make_graph(
        nodes, "external", [], [out], initializer=[weight]
    )
    model = onnx_module.helper.make_model(
        graph, opset_imports=[onnx_module.helper.make_opsetid("", 18)]
    )
    model.ir_version = 8

    effect = static_shape_fold(
        model,
        allowed_ops=ALLOWED_OPS,
        max_input_bytes=FOLD_BUDGET,
        max_output_bytes=FOLD_BUDGET,
    )

    assert effect["folded_nodes"] == 0
    assert effect["skipped_by_reason"]["input_in_external_data"] == 1
    assert effect["external_initializers_excluded_from_pool"] == 1
    assert declared_tensor_bytes(weight) == 16


def test_a_graph_that_is_not_topologically_ordered_fails_loudly(
    onnx_module, numpy_module
) -> None:
    model = _synthetic_model(onnx_module, numpy_module)
    nodes = list(model.graph.node)
    nodes[2], nodes[3] = nodes[3], nodes[2]
    del model.graph.node[:]
    model.graph.node.extend(nodes)

    with pytest.raises(QnnTransformError, match="not topologically ordered"):
        assert_topological_order(model.graph)


# --------------------------------------------------------------------------
# X-DEAD-NODE-ELIMINATION
# --------------------------------------------------------------------------


def test_dead_node_elimination_removes_only_the_unreachable_branch(
    onnx_module, numpy_module
) -> None:
    model, _ = _folded(onnx_module, numpy_module)

    effect = dead_node_elimination(model)

    names = {node.name for node in model.graph.node}
    assert "sqrt_dead" not in names
    assert "mul_big" not in names
    assert {"rs", "sqrt_live", "const_pinned"} <= names
    assert effect["nodes_removed"] == 2
    assert effect["nodes_removed_by_operator"] == {"Mul": 1, "Sqrt": 1}
    assert "big_const" not in _initializer_names(model)
    # The large constant plus the two 8-byte shape operands the folded Concat
    # consumed, which nothing reads any more.
    assert effect["initializer_bytes_removed"] == BIG_BYTES + 16


def test_dead_node_elimination_never_drops_a_graph_input(
    onnx_module, numpy_module
) -> None:
    model, _ = _folded(onnx_module, numpy_module)

    effect = dead_node_elimination(model)

    assert [value.name for value in model.graph.input] == ["x"]
    assert effect["graph_inputs_preserved"] == 1


def test_dead_node_elimination_refuses_an_unordered_graph(
    onnx_module, numpy_module
) -> None:
    """The backward sweep would drop a live producer that follows its consumer.

    ``cat`` produces ``target_shape``, which ``rs`` reads. Reversing those two
    makes the reverse walk reach ``cat`` before ``rs`` has marked its inputs
    required, so ``cat`` would be removed as dead and ``rs`` would be left
    reading a name nothing produces.
    """

    model = _synthetic_model(onnx_module, numpy_module)
    nodes = list(model.graph.node)
    nodes[2], nodes[3] = nodes[3], nodes[2]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    before = [node.name for node in model.graph.node]

    with pytest.raises(QnnTransformError, match="not topologically ordered"):
        dead_node_elimination(model)

    assert [node.name for node in model.graph.node] == before


# --------------------------------------------------------------------------
# X-EXTERNALIZE-LARGE-TENSORS
# --------------------------------------------------------------------------


def test_externalization_plan_splits_on_the_threshold(
    onnx_module, numpy_module
) -> None:
    model = _synthetic_model(onnx_module, numpy_module)
    constant_to_initializer(model)

    plan = externalize_large_tensors(
        model, size_threshold_bytes=1024, location="prefill.onnx.data"
    )

    assert plan["newly_externalized"] == 1
    assert plan["bytes_newly_externalized"] == BIG_BYTES
    assert plan["largest_newly_externalized_bytes"] == BIG_BYTES
    assert plan["kept_inline"] == 2
    assert plan["bytes_kept_inline"] == 16
    assert plan["realized_at"] == "serialization"
    assert [initializer.data_location for initializer in model.graph.initializer] == [
        0,
        0,
        0,
    ]


# --------------------------------------------------------------------------
# X-INFER-VALUE-INFO
# --------------------------------------------------------------------------


def test_shape_inference_annotates_the_interior_and_says_how_much(
    onnx_module, numpy_module
) -> None:
    model, _ = _folded(onnx_module, numpy_module)
    dead_node_elimination(model)

    effect = infer_value_info(model)

    assert effect["status"] == "measured"
    assert effect["value_info_before"] == 0
    assert effect["intermediate_tensors"] == 0
    assert effect["coverage"] == "none_required"


def test_shape_inference_counts_every_intermediate_tensor(
    onnx_module, numpy_module
) -> None:
    model = _synthetic_model(onnx_module, numpy_module)
    constant_to_initializer(model)

    effect = infer_value_info(model)

    assert effect["status"] == "measured"
    # target_shape, big_squared, and the dead branch's dead_root.
    assert effect["intermediate_tensors"] == 3
    assert effect["intermediate_tensors_annotated"] == 3
    assert effect["annotated_fully_static"] == 3
    assert effect["coverage"] == "complete"


# --------------------------------------------------------------------------
# X-STAMP-CANDIDATE-PROVENANCE
# --------------------------------------------------------------------------


def test_provenance_replaces_the_producer_and_records_the_source(
    onnx_module, numpy_module
) -> None:
    model = _synthetic_model(onnx_module, numpy_module)
    model.producer_name = "pytorch"
    model.producer_version = "2.7.1"

    effect = stamp_candidate_provenance(
        model,
        producer_name="slm_lab.graph.qnn",
        producer_version="qnn-candidate-v1",
        metadata={"slm_lab.source_sha256": "a" * 64, "slm_lab.task_id": "T22"},
    )

    assert effect["producer_before"] == "pytorch 2.7.1"
    assert model.producer_name == "slm_lab.graph.qnn"
    assert model.producer_version == "qnn-candidate-v1"
    assert {entry.key for entry in model.metadata_props} == {
        "slm_lab.source_sha256",
        "slm_lab.task_id",
    }
    assert effect["ir_version"] == 8
    assert effect["opset_imports"] == [["", 18]]


# --------------------------------------------------------------------------
# Serialization round trip
# --------------------------------------------------------------------------


def test_the_written_candidate_reads_back_and_passes_the_onnx_checker(
    onnx_module, numpy_module, tmp_path: Path
) -> None:
    model, _ = _folded(onnx_module, numpy_module)
    dead_node_elimination(model)
    infer_value_info(model)
    stamp_candidate_provenance(
        model,
        producer_name="slm_lab.graph.qnn",
        producer_version="qnn-candidate-v1",
        metadata={"slm_lab.task_id": "T22"},
    )
    destination = tmp_path / "prefill.onnx"

    write_candidate(
        model,
        destination,
        source_directory=tmp_path,
        size_threshold_bytes=1024,
        location="prefill.onnx.data",
    )

    onnx_module.checker.check_model(str(destination))
    summary = read_onnx_model(destination)
    assert summary.producer_name == "slm_lab.graph.qnn"
    assert [value.name for value in summary.inputs] == ["x"]
    assert [value.name for value in summary.outputs] == ["y", "live_root", "pinned"]
