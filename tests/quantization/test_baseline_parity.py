"""Tests for the T40 pre-quantization baseline parity gate.

The suite is offline and deterministic by default. It never mutates a real
repository file: every drift case is injected into a ``tmp_path`` copy of the
committed tree.

Two artifact-touching behaviours are separated on purpose:

* a cheap metadata probe (existence plus ``st_size``) runs whenever the
  external artifact root happens to be mounted;
* the full ~9 GB re-hash runs only when ``SLM_LAB_T40_VERIFY_ARTIFACT_BYTES=1``
  is set, because the offline suite must not depend on a nine-gigabyte read.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from slm_lab.quantization import parity as parity_module
from slm_lab.quantization.parity import (
    BaselineParityError,
    artifact_subdirectory,
    check_baseline_parity,
    expected_artifact_files,
    load_manifests,
    main,
    numerical_parity_requirement,
    overall_verdict,
    resolve_artifact_root,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIRECTORY = Path("results/manifests/onnx")
MODEL_CONTRACT = Path("configs/models/qwen3-0.6b.yaml")
EXPORT_CONFIG = Path("configs/models/qwen3-0.6b-onnx-export.json")
EVIDENCE_RECORD = Path("results/quantization/t40-baseline-parity-2026-08-02.json")
COPIED_PATHS = (
    EXPORT_CONFIG,
    MODEL_CONTRACT,
    Path("configs/storage/external-ssd.example.yaml"),
    MANIFEST_DIRECTORY,
)
BYTE_VERIFICATION_ENV = "SLM_LAB_T40_VERIFY_ARTIFACT_BYTES"


def _clone_repository(tmp_path: Path) -> Path:
    """Copy only the inputs the gate reads into a writable scratch tree."""

    root = tmp_path / "repo"
    for relative in COPIED_PATHS:
        source = REPO_ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    return root


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, document: Any) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _offline_report(root: Path, **kwargs: Any) -> dict[str, Any]:
    """Run the gate against a tree with a deliberately absent artifact root."""

    kwargs.setdefault("artifact_root", Path("/slm-lab-absent-artifact-root"))
    kwargs.setdefault("verify_artifact_bytes", True)
    return check_baseline_parity(root, **kwargs)


def _check(report: dict[str, Any], name: str) -> dict[str, Any]:
    match = next(
        check
        for check in report["artifact_identity"]["checks"]
        if check["name"] == name
    )
    return match


def test_committed_tree_passes_every_offline_identity_check() -> None:
    report = _offline_report(REPO_ROOT)

    for name in (
        "attestation_manifest_agreement",
        "model_revision_agreement",
        "t12_contract_conformance",
    ):
        check = _check(report, name)
        assert check["status"] == "passed", check.get("failures")
    assert report["baseline"]["contexts"] == [128, 512, 1024, 4096]


def test_missing_artifact_root_is_unavailable_and_never_a_pass() -> None:
    report = _offline_report(REPO_ROOT)
    bytes_check = _check(report, "artifact_byte_identity")

    assert bytes_check["status"] == "unavailable"
    assert bytes_check["detail"]["measurement"] == "not_measured"
    assert bytes_check["detail"]["command"]
    assert bytes_check["failures"]
    assert report["artifact_identity"]["verdict"] == "unavailable"
    assert report["verdict"] == "unavailable"
    assert report["released_for_calibration_on_artifact_identity"] is False
    assert report["claim_boundary"]["establishes"] == []
    assert report["artifact_root"]["available"] is False
    assert report["artifact_root"]["reason"]


def test_disabled_byte_verification_is_skipped_not_passed() -> None:
    report = check_baseline_parity(REPO_ROOT, verify_artifact_bytes=False)
    bytes_check = _check(report, "artifact_byte_identity")

    assert bytes_check["status"] == "skipped"
    assert bytes_check["detail"]["measurement"] == "not_measured"
    assert report["artifact_identity"]["verdict"] == "partial"
    assert report["released_for_calibration_on_artifact_identity"] is False


def test_numerical_half_is_declared_not_run_with_a_real_command() -> None:
    report = check_baseline_parity(REPO_ROOT, verify_artifact_bytes=False)
    numerical = report["numerical_parity"]

    assert numerical["status"] == "not_run"
    assert numerical["measurement"] == "declared_requirement_not_executed"
    assert numerical["owner_task"] == "T21"
    assert "T41" in numerical["consumer_tasks"]
    assert numerical["commands"]
    for command in numerical["commands"]:
        assert command["command"].strip()
        assert command["status"] in {
            "implemented_at_this_commit",
            "not_implemented_at_this_commit",
        }
    assert numerical["tolerance_policy"]["same_model_reference_source"]

    # The overall verdict may never read "verified" while half two is not_run.
    assert report["verdict"] != "verified"
    assert report["verdict_scope"] == "artifact_identity_only"
    assert (
        "pytorch_versus_onnx_numerical_logit_parity"
        in report["claim_boundary"]["does_not_establish"]
    )


@pytest.mark.parametrize(
    ("identity_verdict", "expected"),
    (
        ("failed", "failed"),
        ("unavailable", "unavailable"),
        ("skipped", "partial"),
        ("partial", "partial"),
        # The branch that actually matters, and the one an absent artifact root
        # can never reach: a fully verified identity half is still only a
        # partial parity result while half two is a declaration.
        ("verified", "partial"),
    ),
)
def test_overall_verdict_never_reads_verified_while_numerical_is_not_run(
    identity_verdict: str,
    expected: str,
) -> None:
    assert numerical_parity_requirement()["status"] == "not_run"
    assert overall_verdict(identity_verdict, "not_run") == expected


def test_overall_verdict_fails_closed_on_an_unhandled_numerical_status() -> None:
    """T21 must extend the mapping, not inherit ``partial`` by accident."""

    with pytest.raises(BaselineParityError, match="unhandled status"):
        overall_verdict("verified", "verified")


def test_a_verified_identity_half_still_yields_partial_not_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the real gate to a fully ``verified`` identity half, offline.

    ``artifact_byte_identity`` is the only check that needs the external
    volume, so the recorded file list is replaced by three tiny stub files that
    are really written and really re-hashed by production code. Everything else
    — the three offline checks, the identity verdict, the overall verdict, and
    the claim boundary — runs unmodified.
    """

    root = _clone_repository(tmp_path)
    artifact_root = tmp_path / "artifact-root"
    export_config = _read_json(root / EXPORT_CONFIG)
    _, resolution = resolve_artifact_root(root, override=artifact_root)
    subdirectory = artifact_subdirectory(
        export_config,
        str(resolution["environment_variable"]),
    )
    directory = artifact_root / subdirectory
    directory.mkdir(parents=True)

    stubs = {
        "S128/prefill.onnx": b"stub prefill graph",
        "S128/prefill.onnx.data": b"stub prefill external data",
        "S128/decode.onnx": b"stub decode graph",
    }
    declared = []
    for relative, payload in stubs.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        declared.append((relative, hashlib.sha256(payload).hexdigest(), len(payload)))
    monkeypatch.setattr(
        parity_module,
        "expected_artifact_files",
        lambda manifests: declared,
    )

    report = check_baseline_parity(
        root,
        artifact_root=artifact_root,
        verify_artifact_bytes=True,
    )

    bytes_check = _check(report, "artifact_byte_identity")
    assert bytes_check["status"] == "passed", bytes_check.get("failures")
    assert bytes_check["detail"]["measurement"] == "recomputed_sha256"
    assert report["artifact_identity"]["verdict"] == "verified"
    assert report["released_for_calibration_on_artifact_identity"] is True
    assert report["claim_boundary"]["establishes"]

    # The guarantee. Half one verified, half two declared, verdict partial.
    assert report["numerical_parity"]["status"] == "not_run"
    assert report["verdict"] == "partial"
    assert report["verdict_scope"] == "artifact_identity_only"


def test_injected_manifest_digest_drift_fails_the_gate(tmp_path: Path) -> None:
    root = _clone_repository(tmp_path)
    manifest_path = root / MANIFEST_DIRECTORY / "S512.json"
    manifest = _read_json(manifest_path)
    manifest["artifacts"]["prefill"]["sha256"] = "0" * 64
    _write_json(manifest_path, manifest)

    report = _offline_report(root)
    check = _check(report, "attestation_manifest_agreement")

    assert check["status"] == "failed"
    assert any("S512 prefill" in failure for failure in check["failures"])
    assert report["verdict"] == "failed"
    assert report["released_for_calibration_on_artifact_identity"] is False


def test_injected_external_data_digest_drift_fails_the_gate(tmp_path: Path) -> None:
    root = _clone_repository(tmp_path)
    manifest_path = root / MANIFEST_DIRECTORY / "S128.json"
    manifest = _read_json(manifest_path)
    manifest["artifacts"]["decode"]["external_data"][0]["sha256"] = "1" * 64
    _write_json(manifest_path, manifest)

    report = _offline_report(root)
    check = _check(report, "attestation_manifest_agreement")

    assert check["status"] == "failed"
    assert any("external data" in failure for failure in check["failures"])


def test_missing_context_manifest_fails_the_gate(tmp_path: Path) -> None:
    root = _clone_repository(tmp_path)
    (root / MANIFEST_DIRECTORY / "S1024.json").unlink()

    report = _offline_report(root)
    check = _check(report, "attestation_manifest_agreement")

    assert check["status"] == "failed"
    assert any(
        "manifest directory is missing context(s): S1024" in failure
        for failure in check["failures"]
    )
    assert report["verdict"] == "failed"


def test_orphan_context_manifest_fails_the_gate(tmp_path: Path) -> None:
    root = _clone_repository(tmp_path)
    manifest_directory = root / MANIFEST_DIRECTORY
    shutil.copy2(manifest_directory / "S128.json", manifest_directory / "S256.json")

    report = _offline_report(root)
    check = _check(report, "attestation_manifest_agreement")

    assert check["status"] == "failed"
    assert any("orphan context(s): S256" in failure for failure in check["failures"])


def test_wrong_model_revision_fails_the_gate(tmp_path: Path) -> None:
    root = _clone_repository(tmp_path)
    manifest_path = root / MANIFEST_DIRECTORY / "S4096.json"
    manifest = _read_json(manifest_path)
    manifest["model_revision"] = "0123456789abcdef0123456789abcdef01234567"
    _write_json(manifest_path, manifest)

    report = _offline_report(root)
    check = _check(report, "model_revision_agreement")

    assert check["status"] == "failed"
    assert any("model_revision" in failure for failure in check["failures"])
    assert report["verdict"] == "failed"


def test_wrong_contract_revision_fails_the_gate(tmp_path: Path) -> None:
    root = _clone_repository(tmp_path)
    contract_path = root / MODEL_CONTRACT
    contract = _read_json(contract_path)
    contract["model"]["revision"] = "f" * 40
    _write_json(contract_path, contract)

    report = _offline_report(root)
    check = _check(report, "model_revision_agreement")

    assert check["status"] == "failed"
    assert any("model contract revision" in failure for failure in check["failures"])


@pytest.mark.parametrize(
    ("graph_kind", "boundary", "mutation", "expected"),
    (
        ("prefill", "input_tensors", "dtype", "dtype"),
        ("decode", "input_tensors", "shape", "shape"),
        ("prefill", "output_tensors", "name", "missing tensor"),
    ),
)
def test_t12_contract_violations_fail_the_gate(
    tmp_path: Path,
    graph_kind: str,
    boundary: str,
    mutation: str,
    expected: str,
) -> None:
    root = _clone_repository(tmp_path)
    manifest_path = root / MANIFEST_DIRECTORY / "S128.json"
    manifest = _read_json(manifest_path)
    tensor = manifest["artifacts"][graph_kind][boundary][0]
    if mutation == "dtype":
        tensor["dtype"] = "float32"
    elif mutation == "shape":
        tensor["shape"] = [1, 7]
    else:
        tensor["name"] = "renamed_by_test"
    _write_json(manifest_path, manifest)

    report = _offline_report(root)
    check = _check(report, "t12_contract_conformance")

    assert check["status"] == "failed"
    assert any(expected in failure for failure in check["failures"])
    assert report["verdict"] == "failed"


def test_cache_capacity_drift_fails_the_gate(tmp_path: Path) -> None:
    root = _clone_repository(tmp_path)
    manifest_path = root / MANIFEST_DIRECTORY / "S512.json"
    manifest = _read_json(manifest_path)
    manifest["cache_capacity"] = 999
    _write_json(manifest_path, manifest)

    report = _offline_report(root)
    check = _check(report, "t12_contract_conformance")

    assert check["status"] == "failed"
    assert any("cache_capacity" in failure for failure in check["failures"])


def test_missing_manifest_directory_raises(tmp_path: Path) -> None:
    root = _clone_repository(tmp_path)
    shutil.rmtree(root / MANIFEST_DIRECTORY)

    with pytest.raises(BaselineParityError, match="manifest directory is missing"):
        check_baseline_parity(root, verify_artifact_bytes=False)


def test_unparseable_manifest_raises(tmp_path: Path) -> None:
    root = _clone_repository(tmp_path)
    (root / MANIFEST_DIRECTORY / "S128.json").write_text("{", encoding="utf-8")

    with pytest.raises(BaselineParityError, match="cannot parse"):
        check_baseline_parity(root, verify_artifact_bytes=False)


def test_artifact_root_resolution_prefers_the_environment_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _clone_repository(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    monkeypatch.setenv("SLM_LAB_ARTIFACT_ROOT", str(external))

    resolved, resolution = resolve_artifact_root(root)

    assert resolved == external.resolve()
    assert resolution["source"] == "environment:SLM_LAB_ARTIFACT_ROOT"
    assert resolution["available"] is True


def test_expected_artifact_files_cover_every_graph_and_external_data_file() -> None:
    manifests = load_manifests(REPO_ROOT)
    expected = expected_artifact_files(manifests)

    assert len(expected) == 16
    for relative, digest, size in expected:
        assert not Path(relative).is_absolute()
        assert len(digest) == 64
        assert size > 0
    assert {relative for relative, _, _ in expected} >= {
        "S128/prefill.onnx",
        "S128/prefill.onnx.data",
        "S4096/decode.onnx",
        "S4096/decode.onnx.data",
    }


def test_null_size_bytes_fails_closed_instead_of_raising_typeerror(
    tmp_path: Path,
) -> None:
    """A manifest this gate cannot check must refuse, not crash."""

    root = _clone_repository(tmp_path)
    manifest_path = root / MANIFEST_DIRECTORY / "S128.json"
    manifest = _read_json(manifest_path)
    manifest["artifacts"]["prefill"]["size_bytes"] = None
    _write_json(manifest_path, manifest)

    with pytest.raises(BaselineParityError, match="size_bytes must be an integer"):
        expected_artifact_files(load_manifests(root))

    # And the CLI turns it into a clean non-zero exit, not a traceback.
    assert main(["--repo-root", str(root), "verify", "--quiet"]) == 1


def test_committed_evidence_record_has_not_drifted_from_the_manifests() -> None:
    """The committed parity record's 16 digests must still be the recorded ones.

    Without this the record is read by nothing and can rot silently against
    ``results/manifests/onnx/S*.json``.
    """

    record = _read_json(REPO_ROOT / EVIDENCE_RECORD)
    declared = {
        relative: (digest, size)
        for relative, digest, size in expected_artifact_files(load_manifests(REPO_ROOT))
    }
    files = record["artifact_identity"]["checks"]
    measured = next(
        check for check in files if check["name"] == "artifact_byte_identity"
    )["detail"]["files"]

    assert {entry["relative_path"] for entry in measured} == set(declared)
    for entry in measured:
        expected_digest, expected_size = declared[entry["relative_path"]]
        assert entry["recorded_sha256"] == expected_digest, entry["relative_path"]
        assert entry["recorded_size_bytes"] == expected_size, entry["relative_path"]
        assert entry["matched"] is True, entry["relative_path"]


def test_committed_evidence_record_states_its_own_scope_honestly() -> None:
    record = _read_json(REPO_ROOT / EVIDENCE_RECORD)

    assert record["verdict"] == "partial"
    assert record["verdict_scope"] == "artifact_identity_only"
    assert record["released_for_calibration_on_artifact_identity"] is True
    assert record["numerical_parity"]["status"] == "not_run"
    # A throwaway worktree path is not committed evidence.
    assert "root" not in record["repository"]
    assert record["repository"]["git_commit"]
    assert isinstance(record["repository"]["git_tree_clean"], bool)


def test_cli_verify_exits_non_zero_when_bytes_are_not_verified(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "verify",
            "--artifact-root",
            "/slm-lab-absent-artifact-root",
            "--json",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert report["verdict"] == "unavailable"
    assert report["numerical_parity"]["status"] == "not_run"


def test_cli_record_writes_a_compact_evidence_document(tmp_path: Path) -> None:
    output = tmp_path / "evidence.json"
    exit_code = main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "record",
            "--artifact-root",
            "/slm-lab-absent-artifact-root",
            "--output",
            str(output),
            "--quiet",
        ]
    )
    document = _read_json(output)

    assert exit_code == 1
    assert output.stat().st_size < 10 * 1024 * 1024
    assert document["task_id"] == "T40"
    assert document["record_type"] == "baseline_parity_preflight"
    assert document["numerical_parity"]["status"] == "not_run"
    assert document["verdict"] != "verified"


def _mounted_artifact_directory() -> Path | None:
    """Resolve the T20 artifact directory the way the gate itself resolves it.

    The subdirectory comes from the committed export config through
    ``artifact_subdirectory``, never from a literal, so moving
    ``artifact_directory`` cannot leave this probe silently pointing at a path
    that no longer exists.
    """

    resolved, resolution = resolve_artifact_root(REPO_ROOT)
    if resolved is None:
        return None
    export_config = _read_json(REPO_ROOT / EXPORT_CONFIG)
    subdirectory = artifact_subdirectory(
        export_config,
        str(resolution["environment_variable"]),
    )
    directory = resolved / subdirectory
    return directory if directory.is_dir() else None


def test_mounted_artifact_metadata_matches_the_committed_record() -> None:
    """Cheap always-on probe: existence and ``st_size`` only, never a full read."""

    directory = _mounted_artifact_directory()
    if directory is None:
        pytest.skip("external artifact root is not mounted on this host")

    manifests = load_manifests(REPO_ROOT)
    for relative, _, recorded_size in expected_artifact_files(manifests):
        path = directory / relative
        assert path.is_file(), f"recorded artifact is missing: {relative}"
        assert path.stat().st_size == recorded_size, relative


@pytest.mark.skipif(
    os.environ.get(BYTE_VERIFICATION_ENV) != "1",
    reason=f"set {BYTE_VERIFICATION_ENV}=1 to re-hash roughly nine gigabytes",
)
def test_full_byte_verification_reproduces_the_committed_digests() -> None:
    """Opt-in ~9 GB re-hash of every recorded T20 graph and external data file."""

    if _mounted_artifact_directory() is None:
        pytest.skip("external artifact root is not mounted on this host")

    report = check_baseline_parity(REPO_ROOT, verify_artifact_bytes=True)
    check = _check(report, "artifact_byte_identity")

    assert check["status"] == "passed", check.get("failures")
    assert check["detail"]["measurement"] == "recomputed_sha256"
    assert check["detail"]["measured_file_count"] == 16
    for measured in check["detail"]["files"]:
        assert measured["matched"], measured["relative_path"]
    assert report["artifact_identity"]["verdict"] == "verified"
    assert report["released_for_calibration_on_artifact_identity"] is True
    # Half one verified is still not full parity.
    assert report["verdict"] == "partial"
    assert report["numerical_parity"]["status"] == "not_run"
