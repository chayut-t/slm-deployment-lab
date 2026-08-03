"""Manifest assembly, contract post-conditions, and ``--check`` staleness.

Everything here runs in the locked root environment. The pieces that need a
real graph are covered in ``test_qnn_transforms.py`` behind an ``onnx`` skip;
the pieces below work on :mod:`slm_lab.graph.onnx_reader` dataclasses, which
are pure standard library.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_lab.graph.inspection import inspect_graph, load_risk_rules
from slm_lab.graph.onnx_reader import (
    GraphSummary,
    InitializerInfo,
    NodeInfo,
    TensorShape,
    ValueInfo,
)
from slm_lab.graph.qnn import build as qnn_build
from slm_lab.graph.qnn.transforms import (
    QnnTransformError,
    assert_boundary_preserved,
    assert_cache_write_preserved,
    count_cache_write_nodes,
)


ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "configs/graph/onnx-risk-rules-v1.json"


def _value(name: str, dtype: str, dims: tuple[int, ...]) -> ValueInfo:
    return ValueInfo(name=name, elem_type=1, dtype=dtype, shape=TensorShape(dims))


def _node(
    index: int,
    op_type: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
) -> NodeInfo:
    return NodeInfo(
        index=index,
        scope="",
        op_type=op_type,
        name=f"/{op_type}_{index}",
        domain="",
        inputs=inputs,
        outputs=outputs,
        attributes=(),
    )


def _initializer(name: str, *, inline_bytes: int, external: bool) -> InitializerInfo:
    return InitializerInfo(
        name=name,
        elem_type=10,
        dtype="float16",
        dims=(inline_bytes // 2,),
        external=external,
        external_location="weights.onnx.data" if external else None,
        inline_bytes=0 if external else inline_bytes,
    )


def _prefill_summary(*, write_op: str = "Concat") -> GraphSummary:
    inputs = tuple(
        _value(name, "int64", (1, 128))
        for name in ("input_ids", "attention_mask", "position_ids")
    )
    outputs = [_value("last_logits", "float32", (1, 151936))]
    nodes: list[NodeInfo] = []
    index = 0
    for layer in range(28):
        for role in ("key_cache", "value_cache"):
            name = f"{role}.{layer}"
            outputs.append(_value(name, "float16", (1, 8, 160, 128)))
            nodes.append(
                _node(index, write_op, ("prefix", "reserve"), (f"{name}/written",))
            )
            index += 1
            nodes.append(_node(index, "Reshape", (f"{name}/written",), (name,)))
            index += 1
    outputs.append(_value("valid_length", "int64", (1,)))
    return GraphSummary(
        ir_version=8,
        producer_name="slm_lab.graph.qnn",
        producer_version="qnn-candidate-v1",
        opset_imports=(("", 18),),
        graph_name="prefill",
        inputs=inputs,
        outputs=tuple(outputs),
        value_info=(),
        initializers=(
            _initializer("weight", inline_bytes=4096, external=True),
            _initializer("small", inline_bytes=64, external=False),
        ),
        nodes=tuple(nodes),
    )


def _decode_summary(*, write_op: str = "ScatterElements") -> GraphSummary:
    inputs = [
        _value("input_ids", "int64", (1, 1)),
        _value("attention_mask", "int64", (1, 160)),
        _value("position_ids", "int64", (1, 1)),
    ]
    outputs = [_value("next_logits", "float32", (1, 151936))]
    nodes: list[NodeInfo] = []
    index = 0
    for layer in range(28):
        for role in ("key", "value"):
            incoming = f"{role}_cache.{layer}"
            present = f"present_{role}.{layer}"
            inputs.append(_value(incoming, "float16", (1, 8, 160, 128)))
            outputs.append(_value(present, "float16", (1, 8, 160, 128)))
            nodes.append(
                _node(index, write_op, (incoming, "index", "update"), (present,))
            )
            index += 1
    inputs.append(_value("valid_length", "int64", (1,)))
    outputs.append(_value("updated_valid_length", "int64", (1,)))
    return GraphSummary(
        ir_version=8,
        producer_name="slm_lab.graph.qnn",
        producer_version="qnn-candidate-v1",
        opset_imports=(("", 18),),
        graph_name="decode",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        value_info=(),
        initializers=(_initializer("weight", inline_bytes=4096, external=True),),
        nodes=tuple(nodes),
    )


# --------------------------------------------------------------------------
# Contract post-conditions
# --------------------------------------------------------------------------


def test_prefill_boundary_counts_match_the_frozen_contract() -> None:
    summary = _prefill_summary()

    assert len(summary.inputs) == 3
    assert len(summary.outputs) == 58
    assert_boundary_preserved(
        summary, summary, graph_kind="prefill", label="S128 prefill"
    )


def test_decode_boundary_counts_match_the_frozen_contract() -> None:
    summary = _decode_summary()

    assert len(summary.inputs) == 60
    assert len(summary.outputs) == 58
    assert_boundary_preserved(
        summary, summary, graph_kind="decode", label="S128 decode"
    )


def test_a_renamed_output_fails_the_boundary_post_condition() -> None:
    reference = _prefill_summary()
    outputs = list(reference.outputs)
    outputs[1] = _value("key_cache.renamed", "float16", (1, 8, 160, 128))
    candidate = GraphSummary(
        ir_version=reference.ir_version,
        producer_name=reference.producer_name,
        producer_version=reference.producer_version,
        opset_imports=reference.opset_imports,
        graph_name=reference.graph_name,
        inputs=reference.inputs,
        outputs=tuple(outputs),
        value_info=reference.value_info,
        initializers=reference.initializers,
        nodes=reference.nodes,
    )

    with pytest.raises(QnnTransformError, match="output boundary differs"):
        assert_boundary_preserved(
            reference, candidate, graph_kind="prefill", label="S128 prefill"
        )


def test_a_dropped_input_fails_the_boundary_post_condition() -> None:
    reference = _decode_summary()
    candidate = GraphSummary(
        ir_version=reference.ir_version,
        producer_name=reference.producer_name,
        producer_version=reference.producer_version,
        opset_imports=reference.opset_imports,
        graph_name=reference.graph_name,
        inputs=reference.inputs[:-1],
        outputs=reference.outputs,
        value_info=reference.value_info,
        initializers=reference.initializers,
        nodes=reference.nodes,
    )

    with pytest.raises(QnnTransformError, match="T12 decode contract fixes 60"):
        assert_boundary_preserved(
            reference, candidate, graph_kind="decode", label="S128 decode"
        )


def test_all_56_prefill_concat_cache_writes_are_found() -> None:
    summary = _prefill_summary()

    assert count_cache_write_nodes(summary, graph_kind="prefill") == 56
    assert (
        assert_cache_write_preserved(
            summary, graph_kind="prefill", label="S128 prefill"
        )
        == 56
    )


def test_all_56_decode_scatter_cache_writes_are_found() -> None:
    summary = _decode_summary()

    assert count_cache_write_nodes(summary, graph_kind="decode") == 56
    assert (
        assert_cache_write_preserved(summary, graph_kind="decode", label="S128 decode")
        == 56
    )


def test_replacing_the_prefill_cache_write_fails_loudly() -> None:
    summary = _prefill_summary(write_op="Pad")

    with pytest.raises(QnnTransformError, match="0 of 56 T12 cache outputs"):
        assert_cache_write_preserved(
            summary, graph_kind="prefill", label="S128 prefill"
        )


def test_replacing_the_decode_cache_write_fails_loudly() -> None:
    summary = _decode_summary(write_op="Where")

    with pytest.raises(QnnTransformError, match="0 of 56 T12 cache outputs"):
        assert_cache_write_preserved(summary, graph_kind="decode", label="S128 decode")


# --------------------------------------------------------------------------
# Manifest assembly
# --------------------------------------------------------------------------


def test_structural_delta_reports_before_and_after_per_rule(tmp_path: Path) -> None:
    catalogue_id, rules = load_risk_rules(RULES_PATH)
    reference = _prefill_summary()
    candidate = _prefill_summary()

    reference_inspection = inspect_graph(
        reference,
        variant_id="S128",
        graph_kind="prefill",
        source_relative_path="S128/prefill.onnx",
        source_sha256="0" * 64,
        rules=rules,
        catalogue_id=catalogue_id,
    )
    candidate_inspection = inspect_graph(
        candidate,
        variant_id="S128",
        graph_kind="prefill",
        source_relative_path="S128/prefill.onnx",
        source_sha256="1" * 64,
        rules=rules,
        catalogue_id=catalogue_id,
    )
    reference_path = tmp_path / "reference.onnx"
    reference_path.write_bytes(b"a" * 32)
    candidate_path = tmp_path / "candidate.onnx"
    candidate_path.write_bytes(b"b" * 16)

    delta = qnn_build.structural_delta(
        reference_inspection=reference_inspection,
        candidate_inspection=candidate_inspection,
        reference_summary=reference,
        candidate_summary=candidate,
        reference_path=reference_path,
        candidate_path=candidate_path,
        reference_external_bytes=1024,
        candidate_external_bytes=2048,
    )

    assert delta["node_count"] == {"before": 112, "after": 112}
    assert delta["input_count"] == {"before": 3, "after": 3}
    assert delta["output_count"] == {"before": 58, "after": 58}
    assert delta["protobuf_bytes"] == {"before": 32, "after": 16}
    assert delta["external_data_bytes"] == {"before": 1024, "after": 2048}
    assert delta["operator_histogram_delta"] == {}
    assert delta["finding_counts"]["R-WIDE-IO-BOUNDARY"] == {
        "before": 58,
        "after": 58,
    }


def test_inline_byte_record_separates_initializers_from_attributes() -> None:
    summary = _prefill_summary()

    record = qnn_build._inline_byte_record(summary)
    assert record == {"initializers": 64, "node_attributes": 0, "total": 64}


def test_histogram_delta_lists_only_changed_operators() -> None:
    delta = qnn_build._histogram_delta(
        {"Constant": 2729, "MatMul": 254}, {"Constant": 0, "MatMul": 254}
    )

    assert delta == {"Constant": {"before": 2729, "after": 0}}


def test_tensor_records_carry_the_T20_shape() -> None:
    summary = _decode_summary()

    records = qnn_build._tensor_records(summary.inputs[:2])
    assert records == [
        {"name": "input_ids", "dtype": "int64", "shape": [1, 1]},
        {"name": "attention_mask", "dtype": "int64", "shape": [1, 160]},
    ]


def test_toolchain_is_read_from_the_running_interpreter() -> None:
    toolchain = qnn_build.toolchain_record()

    assert set(toolchain) == {"python", "onnx", "onnxruntime", "numpy"}
    assert toolchain["python"].count(".") == 2


def test_claim_boundary_names_what_the_build_does_not_establish() -> None:
    boundary = qnn_build.CLAIM_BOUNDARY

    assert "compiler_acceptance" in boundary["does_not_establish"]
    assert (
        "onnxruntime_numerical_parity_of_the_candidate"
        in (boundary["does_not_establish"])
    )
    assert "onnx_checker_accepted_the_candidate_graph" in boundary["establishes"]


# --------------------------------------------------------------------------
# The ORT CPU parity record, read rather than produced
# --------------------------------------------------------------------------


_CANDIDATE_ARTIFACTS = {
    "root": qnn_build.CANDIDATE_ROOT_TEMPLATE,
    "prefill": {
        "graph_kind": "prefill",
        "relative_path": "S128/prefill.onnx",
        "sha256": "a" * 64,
    },
    "decode": {
        "graph_kind": "decode",
        "relative_path": "S128/decode.onnx",
        "sha256": "b" * 64,
    },
}


def _parity_document(**overrides: object) -> dict[str, object]:
    document: dict[str, object] = {
        "schema_version": 1,
        "task_id": "T21",
        "record_kind": "t21_ort_cpu_parity",
        "evidence_sha256": "c" * 64,
        "evidence_tier": "real_onnxruntime_cpu",
        "passed": True,
        "failure_kinds": [],
        "failures": [],
        "steps_requested": 4,
        "cache_report": {"passed": True},
        "tolerance": {"atol": 1.15, "cosine_min": 0.9993},
        "runtime": {"onnxruntime_version": "1.28.0"},
        "reference_provenance": {"model_id": "Qwen/Qwen3-0.6B"},
        "graph_digests": {
            "prefill": {"relative_path": "S128/prefill.onnx", "sha256": "a" * 64},
            "decode": {"relative_path": "S128/decode.onnx", "sha256": "b" * 64},
        },
        "steps": [
            {
                "step": 0,
                "graph_kind": "prefill",
                "metrics": {
                    "passed": True,
                    "top1_agreement": True,
                    "cosine_similarity": 0.99991,
                    "max_absolute_error": 0.34,
                    "max_protected_relative_error": 0.26,
                    "mean_absolute_error": 0.06,
                    "top5_overlap": 1.0,
                },
            },
            {
                "step": 1,
                "graph_kind": "decode",
                "metrics": {
                    "passed": True,
                    "top1_agreement": True,
                    "cosine_similarity": 0.99982,
                    "max_absolute_error": 0.29,
                    "max_protected_relative_error": 0.28,
                    "mean_absolute_error": 0.05,
                    "top5_overlap": 1.0,
                },
            },
            {
                "step": 2,
                "graph_kind": "decode",
                "metrics": {
                    "passed": False,
                    "top1_agreement": False,
                    "cosine_similarity": 0.97,
                    "max_absolute_error": 1.4,
                    "max_protected_relative_error": 1.1,
                    "mean_absolute_error": 0.4,
                    "top5_overlap": 0.6,
                },
            },
        ],
    }
    document.update(overrides)
    return document


def _write_parity(directory: Path, variant_id: str, document: object) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = qnn_build.parity_record_path(directory, variant_id)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def test_a_missing_parity_record_stays_an_explicit_not_measured(tmp_path: Path) -> None:
    record = qnn_build.ort_cpu_parity_record(
        variant_id="S128",
        parity_directory=tmp_path / "parity",
        artifact_records=_CANDIDATE_ARTIFACTS,
    )

    assert record["status"] == "not_measured"
    assert record["expected_record_path"].endswith("S128-ort-cpu.json")
    assert "never" in record["reason"]
    # The reason has to name the command that would produce one, so a reader
    # of the manifest alone can close the gap.
    assert qnn_build.PARITY_RUNNER in record["reason"]
    assert "passed" not in record


def test_a_matching_parity_record_is_carried_into_the_manifest(tmp_path: Path) -> None:
    _write_parity(tmp_path, "S128", _parity_document())

    record = qnn_build.ort_cpu_parity_record(
        variant_id="S128",
        parity_directory=tmp_path,
        artifact_records=_CANDIDATE_ARTIFACTS,
    )

    assert record["status"] == "measured"
    assert record["record_path"].endswith("S128-ort-cpu.json")
    assert len(record["record_sha256"]) == 64
    assert record["record_kind"] == "t21_ort_cpu_parity"
    assert record["record_task_id"] == "T21"
    assert record["evidence_sha256"] == "c" * 64
    assert record["evidence_tier"] == "real_onnxruntime_cpu"
    assert record["passed"] is True
    assert record["steps_requested"] == 4
    assert record["steps_recorded"] == 3
    assert record["graph_digests_match_candidate"] is True


def test_the_parity_summary_reports_the_worst_step_not_the_best(
    tmp_path: Path,
) -> None:
    _write_parity(tmp_path, "S128", _parity_document())

    record = qnn_build.ort_cpu_parity_record(
        variant_id="S128",
        parity_directory=tmp_path,
        artifact_records=_CANDIDATE_ARTIFACTS,
    )

    decode = record["logit_metrics"]["decode"]
    assert decode["steps"] == 2
    assert decode["steps_passed"] == 1
    assert decode["top1_agreements"] == 1
    # The failing step's numbers are the ones that survive the summary.
    assert decode["cosine_similarity_min"] == 0.97
    assert decode["max_absolute_error_max"] == 1.4
    assert decode["max_protected_relative_error_max"] == 1.1
    assert decode["top5_overlap_min"] == 0.6


def test_a_step_with_non_finite_logits_is_counted_not_dropped(tmp_path: Path) -> None:
    document = _parity_document()
    document["steps"] = [  # type: ignore[index]
        {"step": 0, "graph_kind": "prefill", "metrics": None},
        {"step": 1, "graph_kind": "decode", "metrics": None},
    ]
    _write_parity(tmp_path, "S128", document)

    record = qnn_build.ort_cpu_parity_record(
        variant_id="S128",
        parity_directory=tmp_path,
        artifact_records=_CANDIDATE_ARTIFACTS,
    )

    prefill = record["logit_metrics"]["prefill"]
    assert prefill["steps"] == 1
    assert prefill["steps_scored"] == 0
    assert prefill["steps_with_non_finite_candidate_logits"] == 1
    assert prefill["cosine_similarity_min"] is None


def test_a_failing_parity_record_is_carried_in_as_a_failure(tmp_path: Path) -> None:
    _write_parity(
        tmp_path,
        "S128",
        _parity_document(passed=False, failure_kinds=["numerical_tolerance"]),
    )

    record = qnn_build.ort_cpu_parity_record(
        variant_id="S128",
        parity_directory=tmp_path,
        artifact_records=_CANDIDATE_ARTIFACTS,
    )

    assert record["status"] == "measured"
    assert record["passed"] is False
    assert record["failure_kinds"] == ["numerical_tolerance"]


def test_a_record_measuring_other_bytes_is_stale_not_a_verdict(tmp_path: Path) -> None:
    document = _parity_document()
    document["graph_digests"]["prefill"]["sha256"] = "d" * 64  # type: ignore[index]
    _write_parity(tmp_path, "S128", document)

    record = qnn_build.ort_cpu_parity_record(
        variant_id="S128",
        parity_directory=tmp_path,
        artifact_records=_CANDIDATE_ARTIFACTS,
    )

    assert record["status"] == "stale_record"
    assert "passed" not in record
    assert record["measured_graph_digests"]["prefill"]["sha256"] == "d" * 64
    assert record["candidate_graph_digests"]["prefill"]["sha256"] == "a" * 64


def test_a_diagnostic_record_is_not_read_as_a_parity_verdict(tmp_path: Path) -> None:
    _write_parity(
        tmp_path,
        "S128",
        _parity_document(record_kind="diagnostic_off_contract_reference_dtype"),
    )

    record = qnn_build.ort_cpu_parity_record(
        variant_id="S128",
        parity_directory=tmp_path,
        artifact_records=_CANDIDATE_ARTIFACTS,
    )

    assert record["status"] == "not_measured"
    assert "diagnostic_off_contract_reference_dtype" in record["reason"]
    assert "passed" not in record


def test_an_unreadable_parity_record_is_not_read_as_a_verdict(tmp_path: Path) -> None:
    path = qnn_build.parity_record_path(tmp_path, "S128")
    path.write_text("not json at all", encoding="utf-8")

    record = qnn_build.ort_cpu_parity_record(
        variant_id="S128",
        parity_directory=tmp_path,
        artifact_records=_CANDIDATE_ARTIFACTS,
    )

    assert record["status"] == "not_measured"
    assert "not a readable JSON object" in record["reason"]


def test_the_derivation_is_deterministic_for_one_record(tmp_path: Path) -> None:
    _write_parity(tmp_path, "S128", _parity_document())

    first = qnn_build.ort_cpu_parity_record(
        variant_id="S128",
        parity_directory=tmp_path,
        artifact_records=_CANDIDATE_ARTIFACTS,
    )
    second = qnn_build.ort_cpu_parity_record(
        variant_id="S128",
        parity_directory=tmp_path,
        artifact_records=_CANDIDATE_ARTIFACTS,
    )

    assert json.dumps(first, indent=2, sort_keys=True) == json.dumps(
        second, indent=2, sort_keys=True
    )


def test_only_a_passing_measurement_clears_the_parity_claim() -> None:
    unmeasured = qnn_build.claim_boundary_for({"status": "not_measured"})
    passed = qnn_build.claim_boundary_for({"status": "measured", "passed": True})
    failed = qnn_build.claim_boundary_for({"status": "measured", "passed": False})
    stale = qnn_build.claim_boundary_for({"status": "stale_record"})

    for boundary in (unmeasured, stale):
        assert qnn_build.PARITY_NOT_ESTABLISHED in boundary["does_not_establish"]
        assert qnn_build.PARITY_CLAIM_MEASURED not in boundary["establishes"]
    assert qnn_build.PARITY_NOT_ESTABLISHED not in passed["does_not_establish"]
    assert qnn_build.PARITY_CLAIM_PASSED in passed["establishes"]
    assert qnn_build.PARITY_CLAIM_LIMIT in passed["does_not_establish"]
    # A measured failure establishes that it was measured and nothing more.
    assert qnn_build.PARITY_NOT_ESTABLISHED in failed["does_not_establish"]
    assert qnn_build.PARITY_CLAIM_FAILED in failed["establishes"]
    assert qnn_build.PARITY_CLAIM_PASSED not in failed["establishes"]


def test_the_committed_parity_records_are_the_ones_the_manifests_cite() -> None:
    """The four committed manifests must cite live, on-these-bytes records."""

    for variant_id in ("S128", "S512", "S1024", "S4096"):
        manifest = json.loads(
            (ROOT / f"results/manifests/qnn/{variant_id}.json").read_text(
                encoding="utf-8"
            )
        )
        parity = manifest["verification"]["ort_cpu_parity"]
        assert parity["status"] == "measured", variant_id
        record = json.loads((ROOT / parity["record_path"]).read_text(encoding="utf-8"))
        assert record["evidence_sha256"] == parity["evidence_sha256"], variant_id
        assert record["evidence_tier"] == "real_onnxruntime_cpu", variant_id
        assert record["passed"] == parity["passed"], variant_id
        for graph_kind in qnn_build.GRAPH_KINDS:
            assert (
                record["graph_digests"][graph_kind]["sha256"]
                == manifest["artifacts"][graph_kind]["sha256"]
            ), (variant_id, graph_kind)


def test_the_committed_candidate_digests_differ_from_the_reference_digests() -> None:
    """T22's first acceptance criterion, read off the two committed manifests.

    The candidate must cite the exact T20 bytes it read and must not *be* them,
    for all four variants and both graph kinds. Sidecars are deliberately not
    asserted on: pass 4 externalizes nothing new in decode, so the decode
    sidecar comes out byte-identical to the reference one and that is a
    measurement, not a collision.
    """

    for variant_id in ("S128", "S512", "S1024", "S4096"):
        manifest = json.loads(
            (ROOT / f"results/manifests/qnn/{variant_id}.json").read_text(
                encoding="utf-8"
            )
        )
        reference = json.loads(
            (ROOT / manifest["source"]["manifest_path"]).read_text(encoding="utf-8")
        )
        assert manifest["artifacts"]["root"] != reference["artifacts"]["root"]
        for graph_kind in qnn_build.GRAPH_KINDS:
            reference_sha = reference["artifacts"][graph_kind]["sha256"]
            candidate_sha = manifest["artifacts"][graph_kind]["sha256"]
            assert manifest["source"][graph_kind]["sha256"] == reference_sha, (
                variant_id,
                graph_kind,
            )
            assert candidate_sha != reference_sha, (variant_id, graph_kind)


# --------------------------------------------------------------------------
# --check staleness
# --------------------------------------------------------------------------


def test_check_reports_a_missing_committed_report(tmp_path: Path, capsys) -> None:
    destination = tmp_path / "S128.json"

    changed = qnn_build._emit(destination, "{}\n", check=True)

    assert changed is True
    assert "missing report" in capsys.readouterr().err
    assert not destination.exists()


def test_check_reports_a_stale_committed_report(tmp_path: Path, capsys) -> None:
    destination = tmp_path / "S128.json"
    destination.write_text('{"a": 1}\n', encoding="utf-8")

    changed = qnn_build._emit(destination, '{"a": 2}\n', check=True)

    assert changed is True
    assert "stale report" in capsys.readouterr().err
    assert destination.read_text(encoding="utf-8") == '{"a": 1}\n'


def test_check_accepts_an_identical_committed_report(tmp_path: Path) -> None:
    destination = tmp_path / "S128.json"
    destination.write_text('{"a": 1}\n', encoding="utf-8")

    assert qnn_build._emit(destination, '{"a": 1}\n', check=True) is False


def test_writing_creates_the_parent_directory(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "S128.json"

    assert qnn_build._emit(destination, '{"a": 1}\n', check=False) is False
    assert destination.read_text(encoding="utf-8") == '{"a": 1}\n'


# --------------------------------------------------------------------------
# CLI surface
# --------------------------------------------------------------------------


def test_parser_defaults_match_the_committed_locations() -> None:
    args = qnn_build.parse_args([])

    assert args.manifest is None
    assert args.all_manifests is None
    assert Path(args.output_directory) == qnn_build.DEFAULT_OUTPUT_DIRECTORY
    assert Path(args.inspection_directory) == qnn_build.DEFAULT_INSPECTION_DIRECTORY
    assert Path(args.parity_directory) == qnn_build.DEFAULT_PARITY_DIRECTORY
    assert args.check is False


def test_selecting_no_manifest_is_an_error() -> None:
    with pytest.raises(qnn_build.QnnBuildError, match="select at least one manifest"):
        qnn_build._selected_manifests(qnn_build.parse_args([]))


def test_selecting_a_missing_manifest_is_an_error(tmp_path: Path) -> None:
    args = qnn_build.parse_args(["--manifest", str(tmp_path / "absent.json")])

    with pytest.raises(qnn_build.QnnBuildError, match="manifest not found"):
        qnn_build._selected_manifests(args)


def test_all_manifests_globs_the_committed_matrix() -> None:
    args = qnn_build.parse_args(["--all-manifests"])

    selected = qnn_build._selected_manifests(args)
    assert [path.stem for path in selected] == ["S1024", "S128", "S4096", "S512"]


def test_unsafe_relative_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(qnn_build.QnnBuildError, match="unsafe manifest relative_path"):
        qnn_build._safe_relative(tmp_path, "../escape.onnx")


def test_artifact_root_token_expands_to_an_absolute_directory(tmp_path: Path) -> None:
    resolved = qnn_build._expand_artifact_root(
        "${SLM_LAB_ARTIFACT_ROOT}/onnx/reference/T20", tmp_path
    )

    assert resolved == tmp_path / "onnx/reference/T20"
