"""Adversarial tests for the T41 W8 evidence gate.

Three things are pinned here, and each one is a place a partial result could be
read as a complete one:

* :func:`slm_lab.quantization.w8.assess_precision_state` must never return
  ``deployed`` from incomplete, wrong-stage, unchained, or self-asserted
  evidence — and must return it from a complete, correctly chained set, so the
  refusals are refusals rather than a function that cannot say yes;
* :func:`slm_lab.quantization.w8.compare_quality` must refuse to subtract two
  records that are not comparable, and must label its scope from the candidate
  record rather than from its caller;
* the request emitter must stop at the submission boundary, refuse a
  committable output location, and never import ``qai_hub``;
* the read-only capability query must observe the public API surface without
  touching a ``submit_*`` attribute, and must redact anything private out of
  what it writes.

Everything runs offline in seconds against synthetic, clearly labelled stubs.
No test here reaches the network: the capability path is driven entirely by a
fake client and by saved sanitized fixtures.
"""

from __future__ import annotations

import enum
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from slm_lab.benchmark import protocol as benchmark_protocol
from slm_lab.contracts import static_cache
from slm_lab.deployment.qualcomm import ai_hub
from slm_lab.evaluation.fixtures import canonical_json_sha256
from slm_lab.quantization import w8

from tests.quantization.test_w8_candidates import (
    REPO_ROOT,
    clone_repository,
    read_json,
    read_yaml,
    write_json,
)


QUANTIZED_DIGEST = "a" * 64
COMPILED_DIGEST = "b" * 64
DATASET_DIGEST = "c" * 64
OTHER_DIGEST = "d" * 64


def simulation_record(**overrides: Any) -> dict[str, Any]:
    """A synthetic Lane B simulation record. Nothing here was ever run."""

    record = {
        "tool": "aimet-onnx",
        "tool_version": "2.36.0",
        "host": "synthetic-linux-aimet-host-not-a-real-machine",
        "quantized_artifact_sha256": QUANTIZED_DIGEST,
    }
    record.update(overrides)
    return record


def stage_manifest(
    stage: str,
    *,
    status: str = "success",
    source_artifacts: list[dict[str, Any]] | None = None,
    predecessor_sha256: str | None = None,
    target_digest: str | None = None,
) -> dict[str, Any]:
    """A synthetic sanitized AI Hub stage manifest of the shape T30 emits."""

    manifest: dict[str, Any] = {
        "schema_version": ai_hub.SCHEMA_VERSION,
        "manifest_type": ai_hub.MANIFEST_TYPE,
        "stage": stage,
        "request_id": f"t41-{stage}-synthetic",
        "observed_at_utc": "2026-08-03T00:00:00Z",
        "status": status,
        "client": {"name": "qai-hub", "version": w8.QAI_HUB_CLIENT_VERSION},
        "lineage": {
            "predecessor_manifest_sha256": predecessor_sha256,
            "source_artifacts": source_artifacts or [],
        },
    }
    if stage == "compile":
        manifest["result"] = {
            "target_artifact": {
                "role": "compiled_model",
                "logical_name": "w8a16-S128-prefill-qnn-context.bin",
                "sha256": target_digest or COMPILED_DIGEST,
                "byte_size": 1024,
            }
        }
    return manifest


def compile_manifest(**kwargs: Any) -> dict[str, Any]:
    return stage_manifest(
        "compile",
        source_artifacts=[
            {
                "role": "source_model",
                "logical_name": "w8a16-S128-prefill-quantized.onnx",
                "sha256": QUANTIZED_DIGEST,
                "byte_size": 2048,
            }
        ],
        **kwargs,
    )


def downstream_manifest(
    stage: str,
    *,
    predecessor_sha256: str,
    compiled_digest: str = COMPILED_DIGEST,
    **kwargs: Any,
) -> dict[str, Any]:
    sources = [
        {
            "role": "compiled_model",
            "logical_name": "w8a16-S128-prefill-qnn-context.bin",
            "sha256": compiled_digest,
            "byte_size": 1024,
        }
    ]
    if stage == "inference":
        sources.append(
            {
                "role": "input_dataset",
                "logical_name": "w8a16-S128-prefill-inputs.h5",
                "sha256": DATASET_DIGEST,
                "byte_size": 512,
            }
        )
    return stage_manifest(
        stage,
        source_artifacts=sources,
        predecessor_sha256=predecessor_sha256,
        **kwargs,
    )


def complete_evidence() -> dict[str, Any]:
    """The only shape that may ever read ``deployed``."""

    compiled = compile_manifest()
    predecessor = w8.stage_manifest_sha256(compiled)
    return {
        "candidate_id": "w8a16",
        "simulation": simulation_record(),
        "stage_manifests": {
            "compile": compiled,
            "inference": downstream_manifest(
                "inference", predecessor_sha256=predecessor
            ),
            "profile": downstream_manifest("profile", predecessor_sha256=predecessor),
        },
    }


# ---------------------------------------------------------------------------
# Precision state
# ---------------------------------------------------------------------------


def test_no_evidence_is_specified() -> None:
    finding = w8.assess_precision_state(None)

    assert finding.state == "specified"
    assert finding.scope == w8.PRECISION_STATE_SCOPES["specified"]
    assert w8.precision_state(None) == "specified"
    assert w8.precision_state_scope(None) == finding.scope


def test_a_complete_and_chained_evidence_set_reads_deployed() -> None:
    """The refusals below are refusals, not a function that cannot say yes."""

    finding = w8.assess_precision_state(complete_evidence())

    assert finding.state == "deployed"
    assert finding.scope == w8.PRECISION_STATE_SCOPES["deployed"]
    assert finding.unsatisfied == ()


def test_a_valid_simulation_alone_reads_simulated() -> None:
    finding = w8.assess_precision_state({"simulation": simulation_record()})

    assert finding.state == "simulated"
    assert finding.scope == w8.PRECISION_STATE_SCOPES["simulated"]
    assert finding.unsatisfied


@pytest.mark.parametrize(
    "missing_field",
    ["tool", "tool_version", "host", "quantized_artifact_sha256"],
)
def test_an_incomplete_simulation_record_is_only_specified(
    missing_field: str,
) -> None:
    record = simulation_record()
    record.pop(missing_field)

    assert w8.precision_state({"simulation": record}) == "specified"


def test_a_floating_tool_version_is_not_an_exact_version() -> None:
    evidence = complete_evidence()
    evidence["simulation"]["tool_version"] = "latest"

    assert w8.precision_state(evidence) == "specified"


def test_compile_evidence_alone_is_not_deployed() -> None:
    evidence = complete_evidence()
    evidence["stage_manifests"] = {"compile": compile_manifest()}
    finding = w8.assess_precision_state(evidence)

    assert finding.state == "simulated"
    assert "inference_manifest_present" in finding.unsatisfied
    assert "profile_manifest_present" in finding.unsatisfied


def test_compile_and_inference_without_profile_is_not_deployed() -> None:
    evidence = complete_evidence()
    del evidence["stage_manifests"]["profile"]
    finding = w8.assess_precision_state(evidence)

    assert finding.state == "simulated"
    assert "profile_manifest_present" in finding.unsatisfied


def test_a_wrong_stage_manifest_is_not_deployed() -> None:
    """An inference manifest filed under ``profile`` is not profile evidence."""

    evidence = complete_evidence()
    predecessor = w8.stage_manifest_sha256(evidence["stage_manifests"]["compile"])
    evidence["stage_manifests"]["profile"] = downstream_manifest(
        "inference", predecessor_sha256=predecessor
    )
    finding = w8.assess_precision_state(evidence)

    assert finding.state == "simulated"
    assert "profile_manifest_is_the_right_stage" in finding.unsatisfied


def test_a_broken_compiled_artifact_chain_is_not_deployed() -> None:
    evidence = complete_evidence()
    predecessor = w8.stage_manifest_sha256(evidence["stage_manifests"]["compile"])
    evidence["stage_manifests"]["inference"] = downstream_manifest(
        "inference",
        predecessor_sha256=predecessor,
        compiled_digest=OTHER_DIGEST,
    )
    finding = w8.assess_precision_state(evidence)

    assert finding.state == "simulated"
    assert "inference_consumed_the_compiled_artifact" in finding.unsatisfied


def test_a_broken_predecessor_chain_is_not_deployed() -> None:
    evidence = complete_evidence()
    evidence["stage_manifests"]["profile"]["lineage"]["predecessor_manifest_sha256"] = (
        OTHER_DIGEST
    )
    finding = w8.assess_precision_state(evidence)

    assert finding.state == "simulated"
    assert "profile_cites_the_compile_manifest" in finding.unsatisfied


def test_a_compile_source_that_is_not_the_simulated_artifact_is_not_deployed() -> None:
    evidence = complete_evidence()
    evidence["simulation"]["quantized_artifact_sha256"] = OTHER_DIGEST
    finding = w8.assess_precision_state(evidence)

    assert finding.state == "simulated"
    assert "compile_source_is_the_simulated_artifact" in finding.unsatisfied


def test_a_failed_stage_is_not_deployed() -> None:
    evidence = complete_evidence()
    evidence["stage_manifests"]["profile"]["status"] = "failed"
    finding = w8.assess_precision_state(evidence)

    assert finding.state == "simulated"
    assert "profile_manifest_succeeded" in finding.unsatisfied


def test_a_schema_v1_manifest_is_not_deployed() -> None:
    evidence = complete_evidence()
    evidence["stage_manifests"]["compile"]["schema_version"] = 1
    finding = w8.assess_precision_state(evidence)

    assert finding.state == "simulated"
    assert "compile_manifest_schema" in finding.unsatisfied


def test_a_planted_deployed_state_is_never_consulted() -> None:
    """A record may not assert its own conclusion."""

    evidence: dict[str, Any] = {
        "state": "deployed",
        "precision_state": "deployed",
        "verdict": "deployed",
        "deployed": True,
        "simulation": {"state": "deployed"},
        "stage_manifests": {
            stage: {"state": "deployed", "status": "success"}
            for stage in w8.STAGE_ORDER
        },
    }

    assert w8.precision_state(evidence) == "specified"

    planted = complete_evidence()
    planted["stage_manifests"]["profile"] = {"state": "deployed"}
    assert w8.precision_state(planted) != "deployed"


@pytest.mark.parametrize("junk", [None, 42, "deployed", [], {"stage_manifests": 7}])
def test_unparsable_evidence_never_reaches_deployed(junk: Any) -> None:
    assert w8.precision_state(junk) in {"specified", "simulated"}


def test_stage_manifest_digest_matches_the_adapter_that_writes_it(
    tmp_path: Path,
) -> None:
    """The chain check recomputes the digest ``ai_hub`` itself assigns."""

    manifest = compile_manifest()
    path = tmp_path / "compile-manifest.json"
    written = ai_hub.write_manifest(path, manifest)

    assert w8.stage_manifest_sha256(manifest) == written
    assert w8.stage_manifest_sha256(read_json(path)) == written


# ---------------------------------------------------------------------------
# Quality comparison
# ---------------------------------------------------------------------------

UNIT_BY_METRIC = {
    "word_perplexity": "perplexity",
    "acc": "ratio",
    "acc_norm": "ratio",
}


def quality_result(
    *,
    precision: str,
    evidence_level: str,
    value: float,
    result_id: str,
    task_id: str = "wikitext_2_raw",
    metric_name: str = "word_perplexity",
    protocol_sha256: str | None = None,
    artifact_sha256: str = "0" * 64,
) -> dict[str, Any]:
    """Build one synthetic, schema-valid T13 quality result.

    Every value is a labelled test fixture. ``result_id`` says so out loud so
    that a stray copy of one of these can never be mistaken for a measurement.
    """

    protocol = benchmark_protocol.load_protocol(REPO_ROOT)
    academic = read_json(REPO_ROOT / w8.DEFAULT_ACADEMIC_CONTRACT)
    task = next(item for item in academic["tasks"] if item["id"] == task_id)
    summary = benchmark_protocol.summarize_samples([value])
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256 or protocol["contract_sha256"],
        "result_id": result_id,
        "task_id": "T41",
        "created_at": "2026-08-03T00:00:00Z",
        "source": {
            "model_id": static_cache.MODEL_ID,
            "model_revision": static_cache.MODEL_REVISION,
            "tokenizer_revision": static_cache.MODEL_REVISION,
            "artifact_id": "synthetic-fixture-not-a-real-artifact",
            "artifact_sha256": artifact_sha256,
            "git_commit": "0" * 40,
            "workload_id": "academic_evaluation",
            "precision": precision,
            "generation_policy_id": "synthetic",
        },
        "system": {
            "evidence_level": evidence_level,
            "platform": "other",
            "device_name": "synthetic-fixture",
            "device_type": "none",
            "os": "none",
            "runtime": "none",
            "runtime_version": "none",
            "provider_or_compute_unit": "none",
            "placement_evidence": "synthetic test fixture only",
            "host_manifest_sha256": "0" * 64,
        },
        "measurement": {
            "kind": "quality",
            "timing_class": None,
            "scope": "evaluation",
            "metric": "quality_metric",
            "unit": UNIT_BY_METRIC[metric_name],
            "timing_boundary": "synthetic test fixture only",
            "synchronization": {
                "backend": "pytorch_cpu",
                "method_id": "call_return",
                "pre_timer_action": None,
                "post_timer_action": "blocking evaluation call returned",
                "evidence": "synthetic test fixture only",
            },
            "process_isolation": {
                "fresh_process_each_repetition": False,
                "reset_method": None,
                "process_identity_evidence": None,
            },
            "warmup_repetitions": 0,
            "measured_repetitions": 1,
            "includes": [],
            "excludes": ["model_load", "compile"],
            "quality_method": {
                "suite_id": academic["suite_id"],
                "task_id": task["id"],
                "metric_name": metric_name,
                "dataset_id": task["dataset_id"],
                "dataset_revision": task["dataset_revision"],
                "dataset_config": task["dataset_config"],
                "harness_release": academic["harness"]["release"],
                "harness_commit": academic["harness"]["release_commit"],
                "resolved_task_sha256": "1" * 64,
                "split": task["split"],
                "selection": task["selection"],
                "prompt_interface": academic["prompt_interface"],
                "apply_chat_template": academic["apply_chat_template"],
                "fewshot": academic["fewshot"],
            },
        },
        "samples": [
            {
                "sample_index": 0,
                "value": value,
                "valid": True,
                "invalid_reason": None,
            }
        ],
        "summary": {
            "sample_count_total": 1,
            "sample_count_invalid": 0,
            **summary,
        },
        "validity": {
            "state": "valid",
            "reasons": [],
            "headline_eligible": False,
        },
        "comparison": {
            "claim_scope": "system_result",
            "comparable_dimensions": ["synthetic fixture shape"],
            "non_comparable_dimensions": ["not a measurement"],
            "system_difference_notes": "Synthetic test fixture only.",
        },
    }


def baseline_result(**overrides: Any) -> dict[str, Any]:
    defaults = {
        "precision": "float16",
        "evidence_level": "simulated",
        "value": 20.0,
        "result_id": "synthetic-float16-baseline-not-a-measurement",
    }
    defaults.update(overrides)
    return quality_result(**defaults)


def candidate_result(**overrides: Any) -> dict[str, Any]:
    defaults = {
        "precision": w8.precision_label("w8a16", "simulated"),
        "evidence_level": "simulated",
        "value": 21.0,
        "result_id": "synthetic-w8a16-candidate-not-a-measurement",
    }
    defaults.update(overrides)
    return quality_result(**defaults)


def test_the_fixtures_satisfy_the_frozen_evaluation_contract() -> None:
    benchmark_protocol.validate_result(baseline_result(), root=REPO_ROOT)
    benchmark_protocol.validate_result(candidate_result(), root=REPO_ROOT)


def test_compare_quality_labels_the_scope_from_the_record() -> None:
    comparison = w8.compare_quality(
        baseline_result(), candidate_result(), root=REPO_ROOT
    )

    assert comparison["comparison_scope"] == "simulated_vs_float"
    assert comparison["comparison_scope_source"] == "candidate_record.source.precision"
    assert comparison["candidate"]["precision_state"] == "simulated"
    assert comparison["delta"]["absolute_delta"] == pytest.approx(1.0)
    assert comparison["delta"]["relative_delta"] == pytest.approx(0.05)
    assert comparison["delta"]["measurement"] == "delta_of_supplied_records"


def test_compare_quality_has_no_caller_supplied_scope_argument() -> None:
    import inspect

    parameters = inspect.signature(w8.compare_quality).parameters

    assert set(parameters) == {"baseline_record", "candidate_record", "root"}


def test_a_deployed_candidate_record_yields_the_deployed_scope() -> None:
    comparison = w8.compare_quality(
        baseline_result(),
        candidate_result(
            precision=w8.precision_label("w8a8", "deployed"),
            evidence_level="observed_hosted_device",
        ),
        root=REPO_ROOT,
    )

    assert comparison["comparison_scope"] == "deployed_vs_float"
    assert comparison["candidate"]["candidate_id"] == "w8a8"


def test_compare_quality_refuses_a_mismatched_protocol_digest() -> None:
    with pytest.raises(w8.W8EvidenceError, match="not comparable"):
        w8.compare_quality(
            baseline_result(),
            candidate_result(protocol_sha256="9" * 64),
            root=REPO_ROOT,
        )


def test_compare_quality_refuses_a_mismatched_workload() -> None:
    with pytest.raises(w8.W8EvidenceError, match="do not measure the same thing"):
        w8.compare_quality(
            baseline_result(),
            candidate_result(
                task_id="arc_easy_full_validation",
                metric_name="acc",
                value=0.5,
            ),
            root=REPO_ROOT,
        )


def test_compare_quality_refuses_a_candidate_with_no_precision_state() -> None:
    with pytest.raises(w8.W8EvidenceError, match="does not declare a precision state"):
        w8.compare_quality(
            baseline_result(),
            candidate_result(precision="int8"),
            root=REPO_ROOT,
        )


def test_compare_quality_refuses_a_specified_candidate() -> None:
    with pytest.raises(w8.W8EvidenceError, match="precision state"):
        w8.compare_quality(
            baseline_result(),
            candidate_result(precision="w8a16+specified"),
            root=REPO_ROOT,
        )


def test_compare_quality_refuses_a_quantized_baseline() -> None:
    with pytest.raises(w8.W8EvidenceError, match="floating baseline"):
        w8.compare_quality(
            baseline_result(precision=w8.precision_label("w8a16", "simulated")),
            candidate_result(),
            root=REPO_ROOT,
        )


def test_compare_quality_refuses_a_deployed_claim_without_device_evidence() -> None:
    with pytest.raises(w8.W8EvidenceError, match="evidence_level"):
        w8.compare_quality(
            baseline_result(),
            candidate_result(
                precision=w8.precision_label("w8a16", "deployed"),
                evidence_level="simulated",
            ),
            root=REPO_ROOT,
        )


def test_compare_quality_attaches_the_inherited_calibration_bias() -> None:
    comparison = w8.compare_quality(
        baseline_result(), candidate_result(), root=REPO_ROOT
    )
    bias = comparison["inherited_bias"]
    calibration = read_yaml(REPO_ROOT / w8.DEFAULT_CALIBRATION_CONFIG)

    assert bias["inherited_from"] == "T40"
    assert bias["overlaps"] is True
    assert (
        bias["calibration_dataset_revision"]
        == (calibration["calibration_dataset_revision"])
    )
    assert "optimistic" in bias["direction"]
    assert bias["statement"].strip()


def test_compare_quality_refuses_a_record_the_frozen_contract_rejects() -> None:
    broken = candidate_result()
    broken["summary"]["median"] = float(broken["summary"]["median"]) + 1

    with pytest.raises(w8.W8EvidenceError, match="frozen T13 evaluation contract"):
        w8.compare_quality(baseline_result(), broken, root=REPO_ROOT)


# ---------------------------------------------------------------------------
# The AI Hub submission boundary
# ---------------------------------------------------------------------------


def candidate_spec(candidate_id: str = "w8a16") -> dict[str, Any]:
    return read_yaml(REPO_ROOT / w8.candidate_config_path(candidate_id))["candidate"]


def test_a_missing_quantized_artifact_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(w8.W8EvidenceError, match="No W8 artifact exists"):
        w8.build_stage_request(
            candidate_spec(),
            "compile",
            context_length=128,
            quantized_artifact=tmp_path / "absent.onnx",
            output_path=tmp_path / "out.bin",
        )


def test_the_repository_holds_no_quantized_artifact_to_point_at() -> None:
    """The emitter's fail-closed path is the live path, not a hypothetical."""

    assert not list((REPO_ROOT / "configs").rglob("*.onnx"))
    assert not list((REPO_ROOT / "results").rglob("*quantized*.onnx"))


def emitted_compile_request(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    source = tmp_path / "w8a16-S128-prefill-quantized.onnx"
    source.write_bytes(b"synthetic stand-in for a quantized graph")
    arguments: dict[str, Any] = {
        "context_length": 128,
        "quantized_artifact": source,
        "output_path": tmp_path / "compiled.bin",
    }
    arguments.update(overrides)
    return w8.build_stage_request(candidate_spec(), "compile", **arguments)


def test_an_emitted_compile_request_is_accepted_by_the_adapter(
    tmp_path: Path,
) -> None:
    request = emitted_compile_request(tmp_path)
    path = w8.write_stage_request(request, tmp_path / "compile-request.json")
    loaded = ai_hub.load_request(path, "compile")

    assert loaded["schema_version"] == ai_hub.SCHEMA_VERSION
    assert loaded["stage"] == "compile"
    assert loaded["retry"] is False
    assert 0 < loaded["timeout_seconds"] <= 24 * 3600
    assert loaded["client_version"] == "0.53.0"
    assert loaded["runtime"] == {"name": "QAIRT", "version": "2.45.0.260326154327"}
    assert loaded["device"]["name"] == "Snapdragon X Elite CRD"
    # The adapter's own option parser must accept the emitted options.
    ai_hub._common_request(loaded, "compile")


def test_emitted_input_specs_come_from_the_frozen_t12_contract(
    tmp_path: Path,
) -> None:
    for graph_kind, builder in (
        ("prefill", static_cache.build_prefill_contract),
        ("decode", static_cache.build_decode_contract),
    ):
        request = emitted_compile_request(
            tmp_path, context_length=512, graph_kind=graph_kind
        )
        contract = builder(512)
        expected = {
            spec.name: {"shape": list(spec.shape), "dtype": spec.dtype}
            for spec in contract.inputs
        }

        assert request["input_specs"] == expected
        assert request["job_name"].endswith(f"S512-{graph_kind}")


def test_an_unfrozen_context_is_refused(tmp_path: Path) -> None:
    with pytest.raises(w8.W8EvidenceError, match="context length must be one of"):
        emitted_compile_request(tmp_path, context_length=256)


@pytest.mark.parametrize(
    "relative",
    ["results/quantization/leaked-request.json", "configs/quantization/w8/x.json"],
)
def test_a_committable_output_location_is_refused(
    tmp_path: Path,
    relative: str,
) -> None:
    request = emitted_compile_request(tmp_path)

    with pytest.raises(w8.W8EvidenceError, match="private storage"):
        w8.write_stage_request(request, REPO_ROOT / relative)


@pytest.mark.parametrize("private", [".ai-local/profiles/T41", "artifacts/T41"])
def test_ignored_private_storage_is_accepted(private: str) -> None:
    """Mirrors ``ai_hub._private_output_path``: these two roots are allowed."""

    resolved = w8.assert_private_path(
        REPO_ROOT / private / "compile-request.json",
        "request path",
    )

    assert resolved.parts[-2] == "T41"


def test_a_committable_compile_output_artifact_is_refused(tmp_path: Path) -> None:
    with pytest.raises(w8.W8EvidenceError, match="private storage"):
        emitted_compile_request(
            tmp_path,
            output_path=REPO_ROOT / "results/quantization/compiled.bin",
        )


def test_downstream_stages_require_the_compile_predecessor(tmp_path: Path) -> None:
    compiled = tmp_path / "compiled.bin"
    compiled.write_bytes(b"synthetic compiled stand-in")

    with pytest.raises(w8.W8EvidenceError, match="predecessor"):
        w8.build_stage_request(
            candidate_spec(),
            "profile",
            context_length=128,
            compiled_artifact=compiled,
            output_path=tmp_path / "profile.json",
        )


def test_an_emitted_profile_request_is_accepted_by_the_adapter(
    tmp_path: Path,
) -> None:
    compiled = tmp_path / "compiled.bin"
    compiled.write_bytes(b"synthetic compiled stand-in")
    manifest_path = tmp_path / "compile-manifest.json"
    ai_hub.write_manifest(manifest_path, compile_manifest())

    request = w8.build_stage_request(
        candidate_spec(),
        "profile",
        context_length=128,
        compiled_artifact=compiled,
        predecessor_manifest=manifest_path,
        output_path=tmp_path / "profile-private.json",
    )
    path = w8.write_stage_request(request, tmp_path / "profile-request.json")
    loaded = ai_hub.load_request(path, "profile")

    assert loaded["stage"] == "profile"
    assert "--compute_unit npu" in loaded["options"]
    ai_hub._common_request(loaded, "profile")


def test_the_emitter_never_imports_the_external_client(tmp_path: Path) -> None:
    assert "qai_hub" not in sys.modules

    request = emitted_compile_request(tmp_path)
    w8.write_stage_request(request, tmp_path / "compile-request.json")
    w8.assess_precision_state(complete_evidence())

    assert "qai_hub" not in sys.modules


def test_the_cli_request_command_writes_privately_and_prints_the_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "quantized.onnx"
    source.write_bytes(b"synthetic stand-in for a quantized graph")
    request_path = tmp_path / "compile-request.json"

    exit_code = w8.main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "request",
            "--candidate",
            "w8a16",
            "--stage",
            "compile",
            "--context",
            "128",
            "--quantized-artifact",
            str(source),
            "--output-artifact",
            str(tmp_path / "compiled.bin"),
            "--request-out",
            str(request_path),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert request_path.is_file()
    assert "submitted nothing" in output
    assert "scripts/qualcomm/compile.py" in output
    assert "do not commit the request" in output


def test_the_cli_request_command_fails_closed_without_the_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = w8.main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "request",
            "--candidate",
            "w8a8",
            "--stage",
            "compile",
            "--context",
            "128",
            "--quantized-artifact",
            str(tmp_path / "absent.onnx"),
            "--output-artifact",
            str(tmp_path / "compiled.bin"),
            "--request-out",
            str(tmp_path / "compile-request.json"),
        ]
    )

    assert exit_code == 1
    assert "No W8 artifact exists" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The read-only AI Hub capability observation
# ---------------------------------------------------------------------------

PRIVATE_STRINGS = (
    "queued at https://app.example.invalid/jobs/jabcd1234 for review",
    "contact deploy-bot@example.invalid about the run",
    "see /jobs/jzz99887766 for the artifact",
    "api_token = sk-not-a-real-key-0123456789abcdef",
    "Authorization: Bearer not-a-real-credential",
    "job jq7x2m9p1 finished",
)


class SubmitTouched(AssertionError):
    """Raised when anything reaches for a ``submit_*`` attribute."""


class FakeQuantizeDtype(enum.Enum):
    """Stands in for the SDK enum. Values are arbitrary; only names are read."""

    INT4 = 1
    INT8 = 2
    INT16 = 3


class FakeDevice:
    """A stand-in device record with the three fields the query reads."""

    def __init__(self, name: str, os: str, attributes: list[str]) -> None:
        self.name = name
        self.os = os
        self.attributes = attributes


def _trap(name: str) -> Any:
    def submit(*args: Any, **kwargs: Any) -> Any:
        raise SubmitTouched(f"{name} was called")

    return submit


def _fake_submit_quantize_job(
    model: Any = None,
    calibration_data: Any = None,
    weights_dtype: Any = FakeQuantizeDtype.INT8,
    activations_dtype: Any = FakeQuantizeDtype.INT8,
    name: str | None = None,
    options: str = "",
    project: Any = None,
) -> Any:
    raise SubmitTouched("submit_quantize_job was called")


class FakeAiHubClient:
    """A client that raises the moment a ``submit_*`` attribute is *touched*.

    ``__getattribute__`` is overridden rather than ``__getattr__`` so the trap
    fires even for names that are really in the instance dictionary. That is the
    point of the test: the query is allowed to know a submit entry point exists
    and is allowed to read its signature, but the only way to do both without
    tripping this trap is to use ``dir`` and ``inspect.getattr_static``, neither
    of which can run a descriptor, a property, or a call.
    """

    def __init__(self, devices: list[FakeDevice]) -> None:
        self.QuantizeDtype = FakeQuantizeDtype
        self.devices = devices
        self.get_devices_calls = 0
        for name in (
            "submit_compile_and_link_jobs",
            "submit_compile_and_profile_jobs",
            "submit_compile_job",
            "submit_inference_job",
            "submit_link_job",
            "submit_profile_job",
        ):
            setattr(self, name, _trap(name))
        self.submit_quantize_job = _fake_submit_quantize_job

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("submit_"):
            raise SubmitTouched(f"the capability query touched {name}")
        return object.__getattribute__(self, name)

    def get_devices(self) -> list[FakeDevice]:
        self.get_devices_calls += 1
        return list(self.devices)


def fake_devices() -> list[FakeDevice]:
    """The three plan targets plus one device the observation must drop."""

    committed = read_json(REPO_ROOT / w8.DEFAULT_CAPABILITY_RECORD)
    devices = [
        FakeDevice(
            str(entry["name"]),
            str(entry["os"]),
            # Deliberately shuffled: the observation has to sort them.
            list(reversed(list(entry["attributes"]))),
        )
        for entry in committed["observation"]["devices"]
    ]
    devices.append(
        FakeDevice(
            "Synthetic Device Not In Plan 3.2",
            "1",
            ["vendor:synthetic-fixture"],
        )
    )
    return devices


def fake_client() -> FakeAiHubClient:
    return FakeAiHubClient(fake_devices())


def fake_observation(**overrides: Any) -> dict[str, Any]:
    observation = w8.observe_capability(
        fake_client(),
        client_version="0.53.0",
        observation_date="2026-08-03",
    )
    observation.update(overrides)
    return observation


@pytest.mark.parametrize("dirty", PRIVATE_STRINGS)
def test_the_redaction_pass_strips_urls_emails_job_ids_and_tokens(
    dirty: str,
) -> None:
    cleaned = w8.redact_private_text(dirty)

    assert w8.REDACTION_MARKER in cleaned
    for pattern in w8.CAPABILITY_PRIVATE_TEXT_PATTERNS:
        assert pattern.search(cleaned) is None, pattern.pattern
    w8.assert_no_private_text(cleaned)


def test_the_redaction_pass_removes_the_whole_url_not_only_the_scheme() -> None:
    """Detecting a scheme is enough to refuse; removing one is not enough here."""

    cleaned = w8.redact_private_text("job at https://hub.example.invalid/jobs/j1234567")

    assert "hub.example.invalid" not in cleaned
    assert "j1234567" not in cleaned


def test_the_redaction_pass_walks_nested_documents_and_keys() -> None:
    document = {
        "https://leak.example.invalid/x": ["ok", {"deep": PRIVATE_STRINGS[1]}],
        "clean": {"count": 3, "flag": True, "empty": None},
    }
    cleaned = w8.redact_document(document)

    w8.assert_no_private_text(cleaned)
    assert w8.REDACTION_MARKER in cleaned
    assert cleaned[w8.REDACTION_MARKER][0] == "ok"
    assert cleaned["clean"] == {"count": 3, "flag": True, "empty": None}


def test_the_redaction_pass_refuses_a_value_it_does_not_understand() -> None:
    """A live service object is exactly what must never reach the file."""

    with pytest.raises(w8.W8EvidenceError, match="JSON scalars"):
        w8.redact_document({"device": FakeDevice("x", "1", [])})


def test_assert_no_private_text_refuses_a_document_that_kept_a_secret() -> None:
    with pytest.raises(w8.W8EvidenceError, match="survived redaction"):
        w8.assert_no_private_text({"note": PRIVATE_STRINGS[0]})


def test_the_capability_query_never_touches_a_submit_function() -> None:
    """The whole guarantee: names and one signature, never a submission."""

    client = fake_client()
    observation = w8.observe_capability(
        client,
        client_version="0.53.0",
        observation_date="2026-08-03",
    )

    assert observation["submit_entry_points"] == [
        "submit_compile_and_link_jobs",
        "submit_compile_and_profile_jobs",
        "submit_compile_job",
        "submit_inference_job",
        "submit_link_job",
        "submit_profile_job",
        "submit_quantize_job",
    ]
    assert observation["quantize_entry_point"] == "submit_quantize_job"
    assert observation["quantize_parameters"] == [
        "model",
        "calibration_data",
        "weights_dtype",
        "activations_dtype",
        "name",
        "options",
        "project",
    ]
    # Exactly one service call, and it is the device listing.
    assert client.get_devices_calls == 1


def test_the_submit_trap_is_live_rather_than_vacuous() -> None:
    """If the fake could be read normally the test above would prove nothing."""

    client = fake_client()

    with pytest.raises(SubmitTouched):
        getattr(client, "submit_quantize_job")
    with pytest.raises(SubmitTouched):
        getattr(client, "submit_compile_job")
    # The static read the query uses is the one path that is allowed.
    static = inspect.getattr_static(client, "submit_quantize_job")
    assert list(inspect.signature(static).parameters) == [
        "model",
        "calibration_data",
        "weights_dtype",
        "activations_dtype",
        "name",
        "options",
        "project",
    ]


def test_the_capability_query_records_only_the_plan_target_devices() -> None:
    client = fake_client()
    observation = w8.observe_capability(
        client,
        client_version="0.53.0",
        observation_date="2026-08-03",
    )
    names = [device["name"] for device in observation["devices"]]

    assert names == sorted(w8.PLAN_TARGET_DEVICE_NAMES)
    assert "Synthetic Device Not In Plan 3.2" not in names
    # The count is of everything the service listed, not of what was kept.
    assert observation["device_count"] == 4
    for device in observation["devices"]:
        assert device["attributes"] == sorted(device["attributes"])
        assert set(device) == {"name", "os", "attributes"}


def test_a_missing_plan_target_device_is_refused_rather_than_omitted() -> None:
    devices = [
        device for device in fake_devices() if device.name != "Snapdragon X Elite CRD"
    ]

    with pytest.raises(w8.W8EvidenceError, match="Snapdragon X Elite CRD"):
        w8.observe_capability(
            FakeAiHubClient(devices),
            client_version="0.53.0",
            observation_date="2026-08-03",
        )


def test_a_client_without_a_quantize_entry_point_is_refused() -> None:
    class WithoutQuantize(FakeAiHubClient):
        def __init__(self) -> None:
            super().__init__(fake_devices())
            object.__delattr__(self, "submit_quantize_job")

    with pytest.raises(w8.W8EvidenceError, match="exposes no submit_quantize_job"):
        w8.observe_capability(
            WithoutQuantize(),
            client_version="0.53.0",
            observation_date="2026-08-03",
        )


def test_the_capability_record_round_trips_from_a_saved_query(tmp_path: Path) -> None:
    observation = fake_observation()
    record = w8.build_capability_record(observation)
    path = tmp_path / "capability.json"
    w8.write_capability_record(path, record)

    assert w8.load_offline_observation(path) == observation
    assert w8.build_capability_record(w8.load_offline_observation(path)) == record
    # A bare saved query, with no surrounding record, is also accepted.
    bare = tmp_path / "observation.json"
    write_json(bare, observation)
    assert w8.load_offline_observation(bare) == observation


def test_an_offline_input_with_an_observation_must_be_a_capability_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "not-a-capability-record.json"
    write_json(path, {"record_type": "something_else", "observation": {}})

    with pytest.raises(w8.W8EvidenceError, match="not a ai_hub_capability"):
        w8.load_offline_observation(path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"observation_date": "yesterday"}, "ISO calendar date"),
        ({"client_version": "latest"}, "exact client version"),
        ({"device_count": 0}, "positive integer"),
        ({"quantize_entry_point": "submit_compile_job"}, "quantize entry point"),
        ({"quantize_parameters": ["cache_dtype"]}, "does not appear"),
    ],
)
def test_a_malformed_observation_is_refused(
    mutation: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(w8.W8EvidenceError, match=match):
        w8.validate_capability_observation(fake_observation(**mutation))


def test_an_observation_with_an_unsorted_attribute_list_is_refused() -> None:
    observation = fake_observation()
    observation["devices"][0]["attributes"] = list(
        reversed(observation["devices"][0]["attributes"])
    )

    with pytest.raises(w8.W8EvidenceError, match="must be sorted"):
        w8.validate_capability_observation(observation)


def test_a_signature_that_grows_a_cache_argument_stops_the_finding() -> None:
    """The prose says there is no cache knob; it may not outlive the signature."""

    observation = fake_observation()
    observation["quantize_parameters"] = [
        *observation["quantize_parameters"],
        "kv_cache_dtype",
    ]
    observation["quantize_signature"] += ", kv_cache_dtype: 'QuantizeDtype'"

    with pytest.raises(w8.W8EvidenceError, match="may be a cache dtype knob"):
        w8.build_capability_record(observation)


def test_a_candidate_dtype_the_service_no_longer_offers_is_refused() -> None:
    observation = fake_observation(quantize_dtypes=["INT4", "INT8"])

    with pytest.raises(w8.W8EvidenceError, match="no longer expressible"):
        w8.build_candidate_quantize_requests(observation)


def test_the_capabilities_command_rebuilds_the_committed_record_byte_identically(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The committed record is its own saved query, so it is a fixed point."""

    committed = REPO_ROOT / w8.DEFAULT_CAPABILITY_RECORD
    output = tmp_path / "capability.json"
    exit_code = w8.main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "capabilities",
            "--offline-input",
            str(committed),
            "--output",
            str(output),
        ]
    )
    printed = capsys.readouterr().out

    assert exit_code == 0
    assert output.read_bytes() == committed.read_bytes()
    assert "jobs submitted: 0" in printed
    assert "device minutes consumed: 0" in printed
    assert "nothing about compilation, placement, latency, or accuracy" in printed


def test_the_capabilities_command_never_imports_the_external_client(
    tmp_path: Path,
) -> None:
    assert "qai_hub" not in sys.modules

    w8.main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "capabilities",
            "--offline-input",
            str(REPO_ROOT / w8.DEFAULT_CAPABILITY_RECORD),
            "--output",
            str(tmp_path / "capability.json"),
        ]
    )

    assert "qai_hub" not in sys.modules


@pytest.mark.skipif(
    importlib.util.find_spec("qai_hub") is not None,
    reason="a live client is installed; this test must not reach the network",
)
def test_the_live_query_refuses_cleanly_without_a_client() -> None:
    with pytest.raises(w8.W8EvidenceError, match="not importable"):
        w8.query_live_capability(observation_date="2026-08-03")


def test_the_committed_capability_record_states_its_own_scope() -> None:
    record = read_json(REPO_ROOT / w8.DEFAULT_CAPABILITY_RECORD)

    assert record["task_id"] == "T41"
    assert record["record_type"] == w8.CAPABILITY_RECORD_TYPE
    assert record["observation_date"] == w8.CAPABILITY_OBSERVATION_DATE
    assert record["client_version"] == w8.QAI_HUB_CLIENT_VERSION
    assert record["submitted_jobs"] == 0
    assert record["device_minutes_consumed"] == 0
    assert record["cost"] == "none — read-only capability query, no job submitted"
    assert record["reproduction_command"] == w8.CAPABILITIES_COMMAND
    assert record["reproduction"]["offline_command"] == w8.CAPABILITIES_OFFLINE_COMMAND
    assert record["query_scope"]["submit_functions_called"] == []
    assert record["query_scope"]["jobs_created"] == 0
    assert record["query_scope"]["devices_leased"] == 0

    assert record["establishes"] == list(w8.CAPABILITY_ESTABLISHES)
    for entry in (
        "that_either_candidate_compiles",
        "that_any_operator_or_the_whole_graph_is_placed_on_the_NPU",
        "any_latency_throughput_peak_memory_or_energy",
        "any_accuracy_perplexity_or_quality_delta",
        "that_a_W8_artifact_exists_anywhere_to_send_to_that_entry_point",
    ):
        assert entry in record["does_not_establish"]
    assert "T22" in record["boundary"]

    # No wall-clock stamp and no commit: this describes the service on one date,
    # not this tree, and leaving both out is what makes it a fixed point.
    assert "generated_at" not in record
    assert "repository" not in record


def test_the_committed_capability_record_carries_no_redacted_span() -> None:
    """A marker here means the record's own prose tripped a pattern."""

    text = (REPO_ROOT / w8.DEFAULT_CAPABILITY_RECORD).read_text(encoding="utf-8")

    assert w8.REDACTION_MARKER not in text
    w8.assert_no_private_text(read_json(REPO_ROOT / w8.DEFAULT_CAPABILITY_RECORD))


def test_the_capability_record_is_pinned_by_the_configs_that_cite_it() -> None:
    """Without this the record is read by nothing and can rot silently."""

    record = read_json(REPO_ROOT / w8.DEFAULT_CAPABILITY_RECORD)
    digest = canonical_json_sha256(record)

    for candidate_id in w8.CANDIDATE_IDS:
        document = read_yaml(REPO_ROOT / w8.candidate_config_path(candidate_id))
        support = document["deployment_routes"]["lane_a_ai_hub_workbench"][
            "quantization_support"
        ]

        assert support["record"]["path"] == w8.DEFAULT_CAPABILITY_RECORD.as_posix()
        assert support["record"]["canonical_json_sha256"] == digest
        assert support["observation_date"] == record["observation_date"]
        assert support["client_version"] == record["client_version"]
        observed_signature = record["observation"]["quantize_signature"]
        assert support["quantize_signature"] == observed_signature

    readiness = read_json(REPO_ROOT / READINESS_RECORD)
    assert readiness["ai_hub_capability"]["canonical_json_sha256"] == digest


# ---------------------------------------------------------------------------
# The committed readiness record
# ---------------------------------------------------------------------------

READINESS_RECORD = Path("results/quantization/t41-w8-readiness-2026-08-03.json")


def test_committed_readiness_record_is_pinned_to_the_committed_configs() -> None:
    """Without this the record is read by nothing and can rot silently."""

    record = read_json(REPO_ROOT / READINESS_RECORD)
    calibration = read_yaml(REPO_ROOT / w8.DEFAULT_CALIBRATION_CONFIG)
    protocol = read_json(REPO_ROOT / w8.DEFAULT_BENCHMARK_PROTOCOL)

    assert record["task_id"] == "T41"
    assert record["record_type"] == w8.RECORD_TYPE
    assert (
        record["inputs"]["calibration_dataset_revision"]
        == (calibration["calibration_dataset_revision"])
    )
    assert (
        record["inputs"]["benchmark_protocol_sha256"] == (protocol["contract_sha256"])
    )

    by_id = {entry["candidate_id"]: entry for entry in record["candidates"]}
    assert set(by_id) == set(w8.CANDIDATE_IDS)
    for candidate_id, entry in by_id.items():
        config_path = REPO_ROOT / w8.candidate_config_path(candidate_id)
        document = read_yaml(config_path)

        assert entry["config_path"] == w8.candidate_config_path(candidate_id).as_posix()
        assert entry["config_file_sha256"] == w8._sha256_file(config_path)
        assert (
            entry["candidate_canonical_json_sha256"]
            == (document["candidate_canonical_json_sha256"])
        )
        assert (
            entry["weight_storage_projection"]
            == (document["candidate"]["weight_storage_projection"])
        )


def test_committed_readiness_record_states_its_own_scope_honestly() -> None:
    record = read_json(REPO_ROOT / READINESS_RECORD)

    for entry in record["candidates"]:
        assert entry["precision_state"] == "specified"
        assert (
            entry["precision_state_scope"] == (w8.PRECISION_STATE_SCOPES["specified"])
        )
        assert entry["precision_evidence"]["source"] == "absent_at_this_commit"
        assert entry["precision_evidence"]["unsatisfied_checks"]
        assert entry["kv_cache_applied_dtype"] == static_cache.CACHE_DTYPE

    assert "ready" not in record
    assert record["released_for_submission_preparation_only"] is True
    assert record["released_for_submission_preparation_only_meaning"]
    assert record["deployment_routes"]["lane_a_available"] is False
    assert record["deployment_routes"]["lane_b_available"] is False
    assert record["claim_boundary"]["establishes"]
    assert (
        "simulated_or_deployed_precision_for_either_candidate"
        in record["claim_boundary"]["does_not_establish"]
    )
    assert (
        "that_any_job_was_submitted_or_that_this_repository_could_run_a_quantize_job"
        in record["claim_boundary"]["does_not_establish"]
    )
    # A throwaway worktree path is not committed evidence.
    assert "root" not in record["repository"]
    assert record["repository"]["git_commit"]
    assert isinstance(record["repository"]["git_tree_clean"], bool)


def test_the_readiness_ledger_shows_a_capability_gap_not_a_permission_gap() -> None:
    """Permission was granted and no row became satisfied because of it."""

    record = read_json(REPO_ROOT / READINESS_RECORD)
    ledger = record["evidence_requirements"]
    rows = {row["id"]: row for row in ledger["rows"]}

    counts = {entry["status"]: entry["count"] for entry in ledger["summary"]}
    assert counts == {"satisfied": 1, "not_run": 3, "blocked": 6}
    assert [row["id"] for row in ledger["rows"] if row["status"] == "satisfied"] == [
        "calibration_corpus_revision_and_token_budget"
    ]
    for row_id in ("peak_memory", "graph_latency_and_npu_placement"):
        blockers = rows[row_id]["blocked_on"]
        assert "user_authorization:qai_hub_submission" not in blockers
        assert w8.LANE_A_CAPABILITY_BLOCKER in blockers
        assert "hardware:linux_cuda_aimet_host" in blockers
        assert rows[row_id]["status"] == "blocked"

    routes = record["deployment_routes"]
    assert routes["lane_a_available"] is False
    assert routes["lane_a_blocked_on"] == [
        w8.LANE_A_CAPABILITY_BLOCKER,
        "upstream_task:T31",
    ]
    assert routes["lane_a_blocked_on_owners"][w8.LANE_A_CAPABILITY_BLOCKER] == "T22"
    authorization = routes["lane_a_authorization"]
    assert authorization["qai_hub_submission"] == "granted"
    assert authorization["consumed_by_this_repository"]["submitted_jobs"] == 0


def test_the_readiness_record_carries_the_capability_observation_and_its_bounds(
    tmp_path: Path,
) -> None:
    record = read_json(REPO_ROOT / READINESS_RECORD)
    capability = record["ai_hub_capability"]

    assert capability["path"] == w8.DEFAULT_CAPABILITY_RECORD.as_posix()
    assert capability["observation_date"] == w8.CAPABILITY_OBSERVATION_DATE
    assert capability["submitted_jobs"] == 0
    assert capability["device_minutes_consumed"] == 0
    assert capability["candidate_requests"] == {
        "w8a16": {"weights_dtype": "INT8", "activations_dtype": "INT16"},
        "w8a8": {"weights_dtype": "INT8", "activations_dtype": "INT8"},
    }
    assert capability["kv_cache_dtype"]["separate_cache_dtype_argument"] is False
    assert "that_either_candidate_compiles" in capability["does_not_establish"]
    assert (
        record["inputs"]["ai_hub_capability_record"]
        == w8.DEFAULT_CAPABILITY_RECORD.as_posix()
    )

    # And the record regenerates with all of it from the committed tree.
    output = tmp_path / "readiness.json"
    assert (
        w8.main(["--repo-root", str(REPO_ROOT), "record", "--output", str(output)]) == 0
    )
    rebuilt = json.loads(output.read_text(encoding="utf-8"))
    assert rebuilt["ai_hub_capability"] == capability
    assert rebuilt["deployment_routes"] == record["deployment_routes"]


def test_readiness_record_promotes_the_state_when_evidence_appears(
    tmp_path: Path,
) -> None:
    """A future session drops evidence in; nothing else has to change."""

    root = clone_repository(tmp_path)
    evidence_path = w8.precision_evidence_path(root, "w8a16")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(evidence_path, {"simulation": simulation_record()})
    record = w8.build_readiness_record(root)
    by_id = {entry["candidate_id"]: entry for entry in record["candidates"]}

    assert by_id["w8a16"]["precision_state"] == "simulated"
    assert by_id["w8a8"]["precision_state"] == "specified"
    assert by_id["w8a16"]["precision_evidence"]["source"].endswith(".json")


def test_record_command_writes_a_compact_document(tmp_path: Path) -> None:
    output = tmp_path / "readiness.json"
    exit_code = w8.main(
        ["--repo-root", str(REPO_ROOT), "record", "--output", str(output)]
    )
    document = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert output.stat().st_size < 1024 * 1024
    assert document["task_id"] == "T41"
    assert [entry["precision_state"] for entry in document["candidates"]] == [
        "specified",
        "specified",
    ]


def test_status_command_names_the_scope_beside_every_state(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert w8.main(["--repo-root", str(REPO_ROOT), "status"]) == 0
    output = capsys.readouterr().out

    assert "precision_state=specified" in output
    assert w8.PRECISION_STATE_SCOPES["specified"] in output
    assert "not an artifact size" in output
    assert "released for submission preparation only: True" in output
