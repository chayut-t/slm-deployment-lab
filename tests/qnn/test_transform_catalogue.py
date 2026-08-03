"""The committed transformation catalogue and its validation rules.

Everything here runs in the locked root environment: catalogue loading is pure
JSON validation and needs no ``onnx``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_lab.graph.inspection import load_risk_rules
from slm_lab.graph.qnn.transforms import (
    APPLIED_PASS_IDS,
    DEFAULT_CATALOGUE_PATH,
    REJECTED_PASS_IDS,
    QnnTransformError,
    applied_passes,
    load_transform_catalogue,
    rejected_passes,
)


ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "configs/graph/onnx-risk-rules-v1.json"


def _catalogue_payload() -> dict:
    return json.loads(DEFAULT_CATALOGUE_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict) -> Path:
    destination = tmp_path / "catalogue.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return destination


def test_committed_catalogue_loads_with_the_declared_identity() -> None:
    catalogue_id, passes = load_transform_catalogue(DEFAULT_CATALOGUE_PATH)

    assert catalogue_id == "qnn-candidate-v1"
    assert [entry.order for entry in passes] == list(range(1, len(passes) + 1))
    assert tuple(entry.id for entry in applied_passes(passes)) == APPLIED_PASS_IDS
    assert tuple(entry.id for entry in rejected_passes(passes)) == REJECTED_PASS_IDS


def test_every_addressed_rule_id_exists_in_the_committed_risk_catalogue() -> None:
    _, passes = load_transform_catalogue(DEFAULT_CATALOGUE_PATH)
    _, rules = load_risk_rules(RULES_PATH)
    known = {rule.id for rule in rules}

    addressed = {rule_id for entry in passes for rule_id in entry.addresses}
    assert addressed
    assert addressed <= known


def test_the_rejected_pass_is_recorded_as_not_applied() -> None:
    _, passes = load_transform_catalogue(DEFAULT_CATALOGUE_PATH)
    rejected = {entry.id: entry for entry in rejected_passes(passes)}

    assert "X-ORT-CPU-OFFLINE-OPTIMIZATION" in rejected
    entry = rejected["X-ORT-CPU-OFFLINE-OPTIMIZATION"]
    assert entry.applied is False
    assert entry.parameters["graph_optimization_level"] == "ORT_ENABLE_BASIC"
    assert entry.parameters["execution_provider"] == "CPUExecutionProvider"


def test_pass_records_round_trip_to_plain_json(tmp_path: Path) -> None:
    _, passes = load_transform_catalogue(DEFAULT_CATALOGUE_PATH)

    rendered = json.loads(json.dumps([entry.as_dict() for entry in passes]))
    assert [entry["id"] for entry in rendered] == [entry.id for entry in passes]


def test_wrong_schema_version_is_rejected(tmp_path: Path) -> None:
    payload = _catalogue_payload()
    payload["schema_version"] = 2

    with pytest.raises(QnnTransformError, match="schema_version"):
        load_transform_catalogue(_write(tmp_path, payload))


def test_missing_pass_field_is_rejected(tmp_path: Path) -> None:
    payload = _catalogue_payload()
    del payload["passes"][0]["rationale"]

    with pytest.raises(QnnTransformError, match="missing fields"):
        load_transform_catalogue(_write(tmp_path, payload))


def test_unexpected_pass_field_is_rejected(tmp_path: Path) -> None:
    payload = _catalogue_payload()
    payload["passes"][0]["severity"] = "high"

    with pytest.raises(QnnTransformError, match="unexpected fields"):
        load_transform_catalogue(_write(tmp_path, payload))


def test_unknown_pass_id_is_rejected(tmp_path: Path) -> None:
    payload = _catalogue_payload()
    payload["passes"][0]["id"] = "X-INVENTED-PASS"

    with pytest.raises(QnnTransformError, match="not implemented"):
        load_transform_catalogue(_write(tmp_path, payload))


def test_non_contiguous_order_is_rejected(tmp_path: Path) -> None:
    payload = _catalogue_payload()
    payload["passes"][1]["order"] = 9

    with pytest.raises(QnnTransformError, match="must declare orders"):
        load_transform_catalogue(_write(tmp_path, payload))


def test_disabling_an_implemented_pass_is_rejected(tmp_path: Path) -> None:
    payload = _catalogue_payload()
    payload["passes"][0]["applied"] = False

    with pytest.raises(QnnTransformError, match="recorded rejection"):
        load_transform_catalogue(_write(tmp_path, payload))


def test_enabling_the_rejected_pass_is_rejected(tmp_path: Path) -> None:
    payload = _catalogue_payload()
    payload["passes"][-1]["applied"] = True

    with pytest.raises(QnnTransformError, match="no implementation"):
        load_transform_catalogue(_write(tmp_path, payload))


def test_addressed_id_that_is_not_a_rule_id_is_rejected(tmp_path: Path) -> None:
    payload = _catalogue_payload()
    payload["passes"][0]["addresses"] = ["not-a-rule"]

    with pytest.raises(QnnTransformError, match="not a risk rule id"):
        load_transform_catalogue(_write(tmp_path, payload))


def test_dropping_a_required_pass_is_rejected(tmp_path: Path) -> None:
    payload = _catalogue_payload()
    payload["passes"] = [
        entry for entry in payload["passes"] if entry["id"] != "X-STATIC-SHAPE-FOLD"
    ]
    for index, entry in enumerate(payload["passes"], start=1):
        entry["order"] = index

    with pytest.raises(QnnTransformError, match="omits required applied passes"):
        load_transform_catalogue(_write(tmp_path, payload))


def test_duplicate_pass_id_is_rejected(tmp_path: Path) -> None:
    payload = _catalogue_payload()
    payload["passes"].append(dict(payload["passes"][0]))
    for index, entry in enumerate(payload["passes"], start=1):
        entry["order"] = index

    with pytest.raises(QnnTransformError, match="duplicate pass id"):
        load_transform_catalogue(_write(tmp_path, payload))


def test_empty_references_are_rejected(tmp_path: Path) -> None:
    payload = _catalogue_payload()
    payload["passes"][0]["references"] = []

    with pytest.raises(QnnTransformError, match="references must be a non-empty"):
        load_transform_catalogue(_write(tmp_path, payload))


def test_missing_catalogue_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(QnnTransformError, match="cannot read transform catalogue"):
        load_transform_catalogue(tmp_path / "absent.json")


def test_invalid_json_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "catalogue.json"
    destination.write_text("{not json", encoding="utf-8")

    with pytest.raises(QnnTransformError, match="invalid JSON"):
        load_transform_catalogue(destination)
