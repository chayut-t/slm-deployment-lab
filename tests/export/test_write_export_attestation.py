"""The re-attestation tool must derive every digest from bytes it hashed."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "scripts/export/write_export_attestation.py"
CONFIG = ROOT / "configs/models/qwen3-0.6b-onnx-export.json"
EXPORTER = ROOT / "src/slm_lab/export/onnx_matrix.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("write_export_attestation", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _write_graph_pair(root: Path, context: int, payload: bytes, sidecar: bytes) -> None:
    directory = root / f"S{context}"
    directory.mkdir(parents=True, exist_ok=True)
    for graph_kind in ("prefill", "decode"):
        graph = directory / f"{graph_kind}.onnx"
        graph.write_bytes(payload + graph_kind.encode("utf-8"))
        graph.with_name(f"{graph.name}.data").write_bytes(sidecar)


def _stub_matrix(root: Path, *, sidecar: bytes = b"shared-external-data") -> None:
    for context in (128, 512, 1024, 4096):
        _write_graph_pair(root, context, f"graph-{context}".encode("utf-8"), sidecar)


def test_config_bytes_round_trip_the_tracked_spelling() -> None:
    """The tool must reproduce the committed config byte for byte."""

    raw = CONFIG.read_bytes()
    assert tool._config_bytes(json.loads(raw.decode("utf-8"))) == raw


def test_strip_then_restore_is_byte_identical() -> None:
    """Removing and re-inserting the block must not perturb key order."""

    document = json.loads(CONFIG.read_text(encoding="utf-8"))
    attestation = document["evidence_attestation"]

    stripped = tool._place_attestation(document, None)
    assert "evidence_attestation" not in stripped

    restored = tool._place_attestation(stripped, attestation)
    assert tool._config_bytes(restored) == CONFIG.read_bytes()
    assert list(restored) == list(document)


def test_measured_digests_come_from_the_files(tmp_path: Path) -> None:
    artifacts = tmp_path / "T20"
    _stub_matrix(artifacts)

    graph_sha256, external, rows = tool.measure_graphs(
        prefill_root=artifacts,
        decode_root=artifacts,
        contexts=(128, 512, 1024, 4096),
    )

    assert set(graph_sha256) == {"S128", "S512", "S1024", "S4096"}
    assert external == hashlib.sha256(b"shared-external-data").hexdigest()
    assert len(rows) == 8
    for context in (128, 512, 1024, 4096):
        for graph_kind in ("prefill", "decode"):
            path = artifacts / f"S{context}" / f"{graph_kind}.onnx"
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            assert graph_sha256[f"S{context}"][graph_kind] == expected


def test_prefill_and_decode_roots_may_differ(tmp_path: Path) -> None:
    """Seeding an attestation from two trees is what promotion needs."""

    prefill_root = tmp_path / "candidate"
    decode_root = tmp_path / "superseded"
    _stub_matrix(prefill_root)
    _stub_matrix(decode_root)
    for context in (128, 512, 1024, 4096):
        (prefill_root / f"S{context}" / "prefill.onnx").write_bytes(b"fixed-prefill")

    graph_sha256, _, _ = tool.measure_graphs(
        prefill_root=prefill_root,
        decode_root=decode_root,
        contexts=(128, 512, 1024, 4096),
    )

    fixed = hashlib.sha256(b"fixed-prefill").hexdigest()
    decode = hashlib.sha256(b"graph-128decode").hexdigest()
    assert graph_sha256["S128"] == {"prefill": fixed, "decode": decode}


def test_divergent_sidecars_are_refused(tmp_path: Path) -> None:
    """One shared external_data_sha256 is an invariant, not a convention."""

    artifacts = tmp_path / "T20"
    _stub_matrix(artifacts)
    (artifacts / "S4096" / "decode.onnx.data").write_bytes(b"different")

    with pytest.raises(tool.AttestationError, match="distinct sidecar digests"):
        tool.measure_graphs(
            prefill_root=artifacts,
            decode_root=artifacts,
            contexts=(128, 512, 1024, 4096),
        )


def test_missing_sidecar_is_refused(tmp_path: Path) -> None:
    artifacts = tmp_path / "T20"
    _stub_matrix(artifacts)
    (artifacts / "S512" / "prefill.onnx.data").unlink()

    with pytest.raises(tool.AttestationError, match="missing external data sidecar"):
        tool.measure_graphs(
            prefill_root=artifacts,
            decode_root=artifacts,
            contexts=(128, 512, 1024, 4096),
        )


def test_runtime_python_version_is_the_running_interpreter(tmp_path: Path) -> None:
    """The attestation names the interpreter that exports, never an argument."""

    artifacts = tmp_path / "T20"
    _stub_matrix(artifacts)
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")

    attestation, _ = tool.build_attestation(
        run_id="T20-test-run",
        exporter_commit="0" * 40,
        source_weights=weights,
        prefill_root=artifacts,
        decode_root=artifacts,
        contexts=(128, 512, 1024, 4096),
    )

    expected = ".".join(str(part) for part in sys.version_info[:3])
    assert attestation["runtime_python_version"] == expected
    assert attestation["source_artifact_sha256"] == (
        hashlib.sha256(b"weights").hexdigest()
    )
    assert set(attestation) == {
        "schema_version",
        "run_id",
        "exporter_commit",
        "runtime_python_version",
        "source_artifact_sha256",
        "external_data_sha256",
        "graph_sha256",
    }


def test_invalid_run_id_is_refused(tmp_path: Path) -> None:
    artifacts = tmp_path / "T20"
    _stub_matrix(artifacts)
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")

    with pytest.raises(tool.AttestationError, match="run_id must match"):
        tool.build_attestation(
            run_id="T21-wrong-task",
            exporter_commit="0" * 40,
            source_weights=weights,
            prefill_root=artifacts,
            decode_root=artifacts,
            contexts=(128, 512, 1024, 4096),
        )


def test_repin_rewrites_the_single_frozen_literal(tmp_path: Path) -> None:
    exporter = tmp_path / "onnx_matrix.py"
    exporter.write_text(EXPORTER.read_text(encoding="utf-8"), encoding="utf-8")
    replacement = "a" * 64

    assert tool._repin_exporter(exporter, replacement) is True
    text = exporter.read_text(encoding="utf-8")
    assert f'FROZEN_EXPORT_CONFIG_SHA256 = (\n    "{replacement}"\n)' in text
    assert tool._repin_exporter(exporter, replacement) is False


def test_repin_refuses_a_source_without_the_literal(tmp_path: Path) -> None:
    exporter = tmp_path / "onnx_matrix.py"
    exporter.write_text("FROZEN_EXPORT_CONFIG_SHA256 = 'inline'\n", encoding="utf-8")

    with pytest.raises(tool.AttestationError, match="exactly one"):
        tool._repin_exporter(exporter, "b" * 64)


def test_frozen_pin_matches_the_tracked_config() -> None:
    """The committed pin and the committed config must never drift apart."""

    import slm_lab.export.onnx_matrix as onnx_matrix

    assert onnx_matrix.FROZEN_EXPORT_CONFIG_SHA256 == (
        hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    )
