"""Tests for the frozen T41 W8A16/W8A8 candidate specifications.

Offline and deterministic. No test mutates a committed file: every drift case
is injected into a ``tmp_path`` copy of the inputs the gate actually reads, and
the committed tree is only ever read.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from slm_lab.contracts import static_cache
from slm_lab.deployment.qualcomm import ai_hub
from slm_lab.evaluation.fixtures import canonical_json_sha256
from slm_lab.quantization import w8


REPO_ROOT = Path(__file__).resolve().parents[2]
COPIED_PATHS = (
    Path("configs/models/qwen3-0.6b.yaml"),
    Path("configs/models/qwen3-0.6b-onnx-export.json"),
    Path("configs/quantization/calibration.yaml"),
    Path("configs/quantization/w8"),
    Path("configs/workloads"),
    Path("results/manifests/onnx"),
    Path("results/graph/S128.json"),
    # A committed input since the candidates stopped assuming Lane A's
    # quantization support and started citing an observation of it.
    Path("results/quantization/t41-ai-hub-capability-2026-08-03.json"),
    Path("environments/linux-aimet/aimet-requirements.in"),
)

#: Independently derived from the published Qwen3-0.6B architecture, and
#: reconciled inside the module against the committed float16 export.
EXPECTED_TOTAL_PARAMETERS = 596_049_920


def clone_repository(tmp_path: Path) -> Path:
    """Copy only the inputs the W8 gate reads into a writable scratch tree."""

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


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, document: Any) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def rewrite_candidate(root: Path, candidate_id: str, mutate: Any) -> None:
    """Load, mutate, and rewrite one committed candidate YAML in a clone."""

    path = root / w8.candidate_config_path(candidate_id)
    document = read_yaml(path)
    mutate(document)
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Fixed point and committed-tree validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate_id", w8.CANDIDATE_IDS)
def test_committed_candidate_is_a_byte_identical_fixed_point(
    candidate_id: str,
) -> None:
    inputs = w8.load_inputs(REPO_ROOT)
    expected = w8.render_document(
        w8.build_document(candidate_id, inputs=inputs),
        candidate_id,
    )
    committed = (REPO_ROOT / w8.candidate_config_path(candidate_id)).read_text(
        encoding="utf-8"
    )

    assert committed == expected


def test_generate_leaves_the_committed_tree_unchanged(tmp_path: Path) -> None:
    """Regenerating into a clone must reproduce the committed bytes exactly."""

    root = clone_repository(tmp_path)
    before = {
        candidate_id: (root / w8.candidate_config_path(candidate_id)).read_bytes()
        for candidate_id in w8.CANDIDATE_IDS
    }
    written = w8.generate_repository(root)

    assert len(written) == len(w8.CANDIDATE_IDS)
    for candidate_id, payload in before.items():
        after = (root / w8.candidate_config_path(candidate_id)).read_bytes()
        assert after == payload


def test_check_passes_on_the_committed_tree() -> None:
    assert w8.main(["--repo-root", str(REPO_ROOT), "check"]) == 0


def test_committed_candidates_only_ever_read_specified() -> None:
    for candidate_id in w8.CANDIDATE_IDS:
        document = read_yaml(REPO_ROOT / w8.candidate_config_path(candidate_id))
        candidate = document["candidate"]

        assert candidate["precision_state"] == "specified"
        assert (
            candidate["precision_state_scope"] == w8.PRECISION_STATE_SCOPES["specified"]
        )
        assert candidate["precision_state_note"]


# ---------------------------------------------------------------------------
# Drift detection: every binding must fail closed
# ---------------------------------------------------------------------------


def test_drifted_calibration_revision_fails_check(tmp_path: Path) -> None:
    root = clone_repository(tmp_path)
    path = root / w8.DEFAULT_CALIBRATION_CONFIG
    calibration = read_yaml(path)
    calibration["calibration_dataset_revision"] = "t40-not-the-frozen-corpus-v9"
    path.write_text(yaml.safe_dump(calibration, sort_keys=False), encoding="utf-8")

    with pytest.raises(w8.W8EvidenceError, match="calibration_dataset_revision"):
        w8.validate_repository(root)
    assert w8.main(["--repo-root", str(root), "check"]) == 1


def test_drifted_baseline_manifest_digest_fails_check(tmp_path: Path) -> None:
    root = clone_repository(tmp_path)
    manifest_path = root / w8.DEFAULT_MANIFEST_DIRECTORY / "S512.json"
    manifest = read_json(manifest_path)
    manifest["artifacts"]["prefill"]["sha256"] = "0" * 64
    write_json(manifest_path, manifest)

    with pytest.raises(w8.W8EvidenceError, match="baseline manifest S512"):
        w8.validate_repository(root)


def test_drifted_protocol_digest_fails_check(tmp_path: Path) -> None:
    root = clone_repository(tmp_path)
    protocol_path = root / w8.DEFAULT_BENCHMARK_PROTOCOL
    protocol = read_json(protocol_path)
    protocol["workloads"][0]["generated_tokens"] += 1
    write_json(protocol_path, protocol)

    with pytest.raises(w8.W8EvidenceError, match="T13 benchmark protocol"):
        w8.validate_repository(root)


def test_cache_dtype_claim_inconsistent_with_the_frozen_contract_fails(
    tmp_path: Path,
) -> None:
    root = clone_repository(tmp_path)

    def lower_the_cache(document: Any) -> None:
        document["candidate"]["kv_cache"]["applied_dtype"] = "int8"

    rewrite_candidate(root, "w8a8", lower_the_cache)

    with pytest.raises(w8.W8EvidenceError, match="CACHE_DTYPE"):
        w8.validate_repository(root)


def test_frozen_contract_dtype_claim_is_also_checked(tmp_path: Path) -> None:
    root = clone_repository(tmp_path)

    def lower_the_contract(document: Any) -> None:
        document["candidate"]["kv_cache"]["frozen_contract"]["dtype"] = "int8"

    rewrite_candidate(root, "w8a16", lower_the_contract)

    with pytest.raises(w8.W8EvidenceError, match="frozen_contract.dtype"):
        w8.validate_repository(root)


def test_a_dropped_exclusion_entry_fails_rather_than_flipping_a_class(
    tmp_path: Path,
) -> None:
    """Silently dropping the tied-embedding exclusion must not quantize it."""

    root = clone_repository(tmp_path)

    def drop_the_embedding_exclusion(document: Any) -> None:
        policy = document["candidate"]["excluded_from_quantization"]
        policy["entries"] = [
            entry
            for entry in policy["entries"]
            if entry["id"] != "tied_embedding_table"
        ]

    rewrite_candidate(root, "w8a16", drop_the_embedding_exclusion)

    with pytest.raises(w8.W8EvidenceError, match="tied_embedding_table"):
        w8.validate_repository(root)


def test_edited_projection_fails_the_fixed_point_and_the_recomputation(
    tmp_path: Path,
) -> None:
    root = clone_repository(tmp_path)

    def inflate_the_saving(document: Any) -> None:
        totals = document["candidate"]["weight_storage_projection"]["totals"]
        totals["weight_byte_ratio_float16_over_candidate"] = 2.0

    rewrite_candidate(root, "w8a16", inflate_the_saving)

    with pytest.raises(w8.W8EvidenceError, match="weight_storage_projection drift"):
        w8.validate_repository(root)


def test_prose_edit_fails_the_byte_identical_fixed_point(tmp_path: Path) -> None:
    root = clone_repository(tmp_path)
    path = root / w8.candidate_config_path("w8a8")
    path.write_text(
        path.read_text(encoding="utf-8") + "\nhand_edited: true\n",
        encoding="utf-8",
    )

    with pytest.raises(w8.W8EvidenceError):
        w8.validate_repository(root)


def test_a_planted_quality_record_stops_the_gate(tmp_path: Path) -> None:
    """No W8 quality result exists at this commit, so one appearing is a stop."""

    root = clone_repository(tmp_path)
    evidence = root / w8.DEFAULT_EVIDENCE_DIRECTORY
    evidence.mkdir(parents=True, exist_ok=True)
    write_json(evidence / "t41-w8-quality-w8a16.json", {"perplexity": 12.3})

    with pytest.raises(w8.W8EvidenceError, match="no W8 measurement exists"):
        w8.validate_repository(root)


def test_the_repository_ships_no_w8_quality_record() -> None:
    directory = REPO_ROOT / w8.DEFAULT_EVIDENCE_DIRECTORY
    assert list(directory.glob(w8.QUALITY_RECORD_GLOB)) == []


# ---------------------------------------------------------------------------
# The analytic weight-storage projection
# ---------------------------------------------------------------------------


def test_parameter_total_matches_the_independently_derived_model_size() -> None:
    inputs = w8.load_inputs(REPO_ROOT)
    classes = w8.derive_weight_classes(inputs["model_contract"])

    assert sum(item.parameters for item in classes) == EXPECTED_TOTAL_PARAMETERS


def test_projection_reconciles_with_the_committed_float16_export() -> None:
    """The cross-check is the reason the parameter total can be trusted."""

    inputs = w8.load_inputs(REPO_ROOT)
    document = read_yaml(REPO_ROOT / w8.candidate_config_path("w8a16"))
    cross_check = document["candidate"]["weight_storage_projection"]["cross_check"]

    assert cross_check["derived_parameters"] == EXPECTED_TOTAL_PARAMETERS
    assert cross_check["reconstructed_parameters"] == EXPECTED_TOTAL_PARAMETERS
    assert cross_check["agrees"] is True
    assert cross_check["external_data_bytes"] == inputs["baseline_external_data_bytes"]
    assert (
        cross_check["inline_initializer_count"]
        == 2 * inputs["model_contract"]["model"]["architecture"]["num_hidden_layers"]
    )


def test_projection_cross_check_fails_closed_on_a_changed_export(
    tmp_path: Path,
) -> None:
    root = clone_repository(tmp_path)
    graph_path = root / w8.DEFAULT_GRAPH_INVENTORY
    record = read_json(graph_path)
    record["graphs"]["prefill"]["external_initializer_count"] -= 1
    write_json(graph_path, record)

    with pytest.raises(w8.W8EvidenceError, match="inline initializer"):
        w8.validate_repository(root)


def _mappings(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _mappings(child)


@pytest.mark.parametrize("candidate_id", w8.CANDIDATE_IDS)
def test_every_numeric_projection_mapping_carries_its_label(
    candidate_id: str,
) -> None:
    """No number in the projection may be presented without its label."""

    document = read_yaml(REPO_ROOT / w8.candidate_config_path(candidate_id))
    projection = document["candidate"]["weight_storage_projection"]

    labels = {w8.PROJECTION_MEASUREMENT, "arithmetic_over_committed_inputs"}
    for mapping in _mappings(projection):
        numeric = [
            key
            for key, value in mapping.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if not numeric:
            continue
        assert mapping.get("measurement") in labels, sorted(mapping)

    for entry in (
        "on_disk_quantized_artifact_size",
        "runtime_peak_memory",
        "any_latency_or_throughput_claim",
    ):
        assert entry in projection["does_not_establish"]
    assert any("encoding" in entry for entry in projection["does_not_establish"])


@pytest.mark.parametrize("candidate_id", w8.CANDIDATE_IDS)
def test_excluded_classes_stay_float16_and_dominate_the_exclusion(
    candidate_id: str,
) -> None:
    document = read_yaml(REPO_ROOT / w8.candidate_config_path(candidate_id))
    projection = document["candidate"]["weight_storage_projection"]
    rows = {row["class_id"]: row for row in projection["weight_classes"]}

    assert rows["tied_vocabulary_table"]["quantized"] is False
    assert rows["tied_vocabulary_table"]["kept_precision"] == "float16"
    assert rows["qk_head_norm_scales"]["quantized"] is False
    assert rows["attention_q_proj"]["quantized"] is True
    assert rows["mlp_down_proj"]["quantized"] is True

    totals = projection["totals"]
    assert (
        totals["quantized_parameters"] + totals["excluded_parameters"]
        == (totals["total_parameters"])
    )
    assert totals["float16_weight_bytes"] == totals["total_parameters"] * 2
    # The exclusion is expensive and the file says so rather than hiding it.
    assert (
        rows["tied_vocabulary_table"]["parameters"]
        > 0.25 * (totals["total_parameters"])
    )


def test_scale_storage_is_a_labelled_lower_bound_outside_the_headline_ratio() -> None:
    document = read_yaml(REPO_ROOT / w8.candidate_config_path("w8a16"))
    projection = document["candidate"]["weight_storage_projection"]
    scales = projection["scale_storage_lower_bound"]

    assert scales["measurement"] == w8.PROJECTION_MEASUREMENT
    assert scales["per_output_channel_scale_count"] > 0
    assert (
        scales["weight_byte_ratio_including_scales"]
        < projection["totals"]["weight_byte_ratio_float16_over_candidate"]
    )


def test_kv_cache_projection_uses_the_frozen_capacities() -> None:
    document = read_yaml(REPO_ROOT / w8.candidate_config_path("w8a8"))
    rows = document["candidate"]["weight_storage_projection"][
        "kv_cache_left_at_the_frozen_contract"
    ]["by_context"]

    assert [row["prompt_length"] for row in rows] == sorted(
        static_cache.CONTEXT_VARIANTS
    )
    for row in rows:
        assert row["dtype"] == static_cache.CACHE_DTYPE
        assert row["bytes"] == static_cache.cache_bytes(row["cache_capacity"])


def test_untied_embeddings_are_refused_rather_than_miscounted() -> None:
    contract = read_json(REPO_ROOT / w8.DEFAULT_MODEL_CONTRACT)
    contract["model"]["architecture"]["tie_word_embeddings"] = False

    with pytest.raises(w8.W8EvidenceError, match="tie_word_embeddings"):
        w8.derive_weight_classes(contract)


# ---------------------------------------------------------------------------
# The candidate content that carries the engineering argument
# ---------------------------------------------------------------------------


def test_w8a16_and_w8a8_differ_only_where_the_plan_row_differs() -> None:
    a16 = read_yaml(REPO_ROOT / w8.candidate_config_path("w8a16"))["candidate"]
    a8 = read_yaml(REPO_ROOT / w8.candidate_config_path("w8a8"))["candidate"]

    assert a16["plan_matrix_row"] == "Q1"
    assert a8["plan_matrix_row"] == "Q2"
    assert a16["activations"]["bits"] == 16
    assert a8["activations"]["bits"] == 8
    assert a16["weights"] == a8["weights"]
    assert a16["activations"]["range_estimator"] == "min_max"
    assert a8["activations"]["range_estimator"] == "mse"
    for candidate in (a16, a8):
        assert candidate["activations"]["symmetric"] is True
        assert candidate["activations"]["granularity"] == "per_tensor"
        assert candidate["activations"]["range_estimator_reason"]
        assert candidate["activations"]["rejected_range_estimators"]


def test_kv_cache_block_names_the_owning_tasks_and_the_change_control() -> None:
    a8 = read_yaml(REPO_ROOT / w8.candidate_config_path("w8a8"))["candidate"]
    cache = a8["kv_cache"]

    assert cache["applied_dtype"] == static_cache.CACHE_DTYPE
    assert cache["plan_row_requirement"] == "INT8 or supported type"
    assert cache["satisfied_without_contract_change"] is False
    owners = cache["change_control"]["owner_tasks"]
    assert owners["graph_contract"] == "T12"
    assert owners["onnx_export_boundary"] == "T20"
    assert owners["promoted_prefill_export"] == "T23"
    assert cache["change_control"]["request_status"].startswith("out_of_scope")
    assert cache["frozen_contract"]["decode_graph_cache_inputs"] == (
        2 * static_cache.NUM_LAYERS
    )

    a16 = read_yaml(REPO_ROOT / w8.candidate_config_path("w8a16"))["candidate"]
    assert a16["kv_cache"]["satisfied_without_contract_change"] is True
    assert a16["kv_cache"]["change_control"]["request_status"] == (
        "not_raised_at_this_commit"
    )


def test_exclusion_policy_separates_choice_from_constraint() -> None:
    candidate = read_yaml(REPO_ROOT / w8.candidate_config_path("w8a16"))["candidate"]
    policy = candidate["excluded_from_quantization"]
    by_id = {entry["id"]: entry for entry in policy["entries"]}

    expected = {
        "tied_embedding_table",
        "final_logits_projection",
        "rmsnorm_scales",
        "qwen3_per_head_qk_norm",
        "softmax",
        "rope_sin_cos",
        "residual_adds",
        "kv_cache_read_write",
    }
    assert set(by_id) == expected
    for entry in policy["entries"]:
        assert entry["kind"] in {"policy_choice", "frozen_graph_constraint"}
        assert entry["reason"].strip()
        assert entry["precision_kept"] == "float16"
    assert by_id["tied_embedding_table"]["kind"] == "frozen_graph_constraint"
    assert by_id["kv_cache_read_write"]["kind"] == "frozen_graph_constraint"
    assert by_id["rmsnorm_scales"]["kind"] == "policy_choice"
    assert (
        set(policy["policy_choice_ids"]) | set(policy["frozen_graph_constraint_ids"])
        == expected
    )


def test_graph_inventory_binding_matches_the_committed_t21_record() -> None:
    record = read_json(REPO_ROOT / w8.DEFAULT_GRAPH_INVENTORY)
    prefill = record["graphs"]["prefill"]
    candidate = read_yaml(REPO_ROOT / w8.candidate_config_path("w8a16"))["candidate"]
    inventory = candidate["graph_inventory"]
    counts = {row["op"]: row["count"] for row in inventory["op_counts"]}

    assert inventory["source"]["owner_task"] == record["task_id"]
    assert inventory["node_count"] == prefill["node_count"]
    assert counts["Softmax"] == prefill["op_histogram"]["Softmax"]
    assert counts["ScatterND"] == prefill["op_histogram"]["ScatterND"]
    assert inventory["inline_initializer_count"] == (
        prefill["initializer_count"] - prefill["external_initializer_count"]
    )


def test_calibration_binding_carries_the_t40_bias_and_the_open_question() -> None:
    calibration = read_yaml(REPO_ROOT / w8.DEFAULT_CALIBRATION_CONFIG)
    candidate = read_yaml(REPO_ROOT / w8.candidate_config_path("w8a8"))["candidate"]
    binding = candidate["calibration"]

    assert (
        binding["calibration_dataset_revision"]
        == (calibration["calibration_dataset_revision"])
    )
    assert binding["observe_ranges_on"]["variant_ids"] == [
        f"S{length}" for length in sorted(static_cache.CONTEXT_VARIANTS)
    ]
    bias = binding["inherited_bias"]
    assert bias["total_calibration_tokens"] == 6912
    assert bias["dominant_source_group"] == "context_workloads"
    assert bias["dominant_source_group_token_share"] > 0.8
    assert bias["evaluation_overlap"]["overlaps"] is True
    decode = binding["decode_side_observer_pass"]
    assert decode["open_question_from"] == "T40"
    assert decode["evidence"] == "none_measured_at_this_commit"
    assert decode["blocked_on"] == ["hardware:linux_cuda_aimet_host"]


def test_deployment_routes_are_unavailable_and_say_why() -> None:
    document = read_yaml(REPO_ROOT / w8.candidate_config_path("w8a16"))
    routes = document["deployment_routes"]
    lane_a = routes["lane_a_ai_hub_workbench"]
    lane_b = routes["lane_b_local_aimet"]

    assert lane_a["available"] is False
    # The permission was granted, so it may not still be recorded as a blocker.
    assert "user_authorization:qai_hub_submission" not in lane_a["blocked_on"]
    assert lane_a["blocked_on"] == [
        w8.LANE_A_CAPABILITY_BLOCKER,
        "upstream_task:T31",
    ]
    # The blocker keeps the repository-wide prefix:token shape and the owning
    # task is named beside it rather than inside the string.
    assert lane_a["blocked_on_owners"][w8.LANE_A_CAPABILITY_BLOCKER] == "T22"
    assert set(lane_a["blocked_on_owners"]) == set(lane_a["blocked_on"])
    assert "T22" in " ".join(lane_a["missing"])
    assert any("quantize-stage adapter" in item for item in lane_a["missing"])
    assert lane_a["submission_parameters"]["client"]["version"] == "0.53.0"
    assert lane_a["submission_parameters"]["runtime"] == {
        "name": "QAIRT",
        "version": "2.45.0.260326154327",
    }
    assert lane_a["submission_parameters"]["retry"] is False
    assert lane_a["submission_parameters"]["timeout_seconds"] > 0
    assert lane_a["target_devices"]["primary"]["name"] == "Snapdragon X Elite CRD"
    assert [device["name"] for device in lane_a["target_devices"]["comparison"]] == [
        "Dragonwing IQ-9075 EVK",
        "Snapdragon 8 Elite QRD",
    ]
    for stage in w8.STAGE_ORDER:
        entry = next(
            item
            for item in lane_a["submission_parameters"]["stages"]
            if item["stage"] == stage
        )
        assert entry["script"] == w8.STAGE_SCRIPTS[stage]
        assert entry["command"].strip()

    assert lane_b["available"] is False
    assert lane_b["blocked_on"] == ["hardware:linux_cuda_aimet_host"]
    assert lane_b["pinned_versions"]["aimet-onnx"] == "2.36.0"


def test_lane_a_records_the_permission_without_recording_any_spend() -> None:
    """The grant is recorded as a permission, never as progress."""

    document = read_yaml(REPO_ROOT / w8.candidate_config_path("w8a8"))
    lane_a = document["deployment_routes"]["lane_a_ai_hub_workbench"]
    authorization = lane_a["authorization"]

    assert authorization["qai_hub_submission"] == "granted"
    assert authorization["granted_on"] == w8.CAPABILITY_OBSERVATION_DATE
    assert "120 device minutes" in authorization["scope"]
    assert authorization["consumed_by_this_repository"] == {
        "submitted_jobs": 0,
        "device_minutes": 0,
        "cost": w8.CAPABILITY_COST,
    }
    # The interactive Device Cloud lease is a different lock and stays named.
    assert "not cleared here" in authorization["device_cloud_session"]
    assert lane_a["available"] is False


def test_lane_a_quantization_support_is_an_observation_not_an_assumption() -> None:
    document = read_yaml(REPO_ROOT / w8.candidate_config_path("w8a16"))
    support = document["deployment_routes"]["lane_a_ai_hub_workbench"][
        "quantization_support"
    ]
    capability = read_json(REPO_ROOT / w8.DEFAULT_CAPABILITY_RECORD)

    assert support["measurement"] == "observed_public_api_capability"
    assert support["observation_date"] == w8.CAPABILITY_OBSERVATION_DATE
    assert support["client_version"] == w8.QAI_HUB_CLIENT_VERSION
    assert support["record"]["path"] == w8.DEFAULT_CAPABILITY_RECORD.as_posix()
    assert support["record"]["canonical_json_sha256"] == canonical_json_sha256(
        capability
    )
    assert support["record"]["submitted_jobs"] == 0
    assert support["record"]["device_minutes_consumed"] == 0
    assert support["quantize_entry_point"] == "submit_quantize_job"
    assert "submit_quantize_job" in support["submit_entry_points"]
    assert set(support["quantize_dtypes"]) == {"INT4", "INT8", "INT16"}
    # The boundary travels with the observation wherever it is copied.
    assert "that_either_candidate_compiles" in support["does_not_establish"]
    assert any("expressible" in item for item in support["establishes"])
    assert "assumption" in support["supersedes"]


def test_the_candidate_to_quantize_dtype_mapping_is_recorded_and_derived() -> None:
    document = read_yaml(REPO_ROOT / w8.candidate_config_path("w8a16"))
    requests = document["deployment_routes"]["lane_a_ai_hub_workbench"][
        "quantization_support"
    ]["candidate_requests"]
    by_candidate = requests["by_candidate"]

    assert by_candidate["w8a16"]["weights_dtype"] == "INT8"
    assert by_candidate["w8a16"]["activations_dtype"] == "INT16"
    assert by_candidate["w8a8"]["weights_dtype"] == "INT8"
    assert by_candidate["w8a8"]["activations_dtype"] == "INT8"
    assert requests["entry_point"] == "submit_quantize_job"
    assert requests["dtypes_not_used_by_T41"] == ["INT4"]
    assert "T42" in requests["int4_note"]

    # The mapping is derived from the candidate policies, not transcribed.
    for candidate_id in w8.CANDIDATE_IDS:
        candidate = read_yaml(REPO_ROOT / w8.candidate_config_path(candidate_id))[
            "candidate"
        ]
        entry = by_candidate[candidate_id]
        assert entry["weights_dtype"].lower() == candidate["weights"]["dtype"]
        assert entry["activations_dtype"].lower() == candidate["activations"]["dtype"]
        assert entry["plan_matrix_row"] == candidate["plan_matrix_row"]


def test_the_quantize_api_exposes_no_separate_cache_dtype_knob() -> None:
    """The frozen float16 cache is consistent with this API, not routed around."""

    document = read_yaml(REPO_ROOT / w8.candidate_config_path("w8a8"))
    finding = document["deployment_routes"]["lane_a_ai_hub_workbench"][
        "quantization_support"
    ]["kv_cache_dtype"]

    assert finding["separate_cache_dtype_argument"] is False
    assert not [name for name in finding["quantize_parameters"] if "cache" in name]
    assert "weights_dtype" in finding["quantize_parameters"]
    assert "activations_dtype" in finding["quantize_parameters"]
    assert "T12" in finding["statement"]


def test_target_device_selectors_carry_the_observed_vocabulary() -> None:
    document = read_yaml(REPO_ROOT / w8.candidate_config_path("w8a16"))
    targets = document["deployment_routes"]["lane_a_ai_hub_workbench"]["target_devices"]
    observed = {
        device["name"]: device
        for device in read_json(REPO_ROOT / w8.DEFAULT_CAPABILITY_RECORD)[
            "observation"
        ]["devices"]
    }

    selectors = [targets["primary"], *targets["comparison"]]
    assert {selector["name"] for selector in selectors} == set(
        w8.PLAN_TARGET_DEVICE_NAMES
    )
    for selector in selectors:
        record = observed[selector["name"]]
        assert selector["os"] == record["os"]
        assert selector["attributes"] == record["attributes"]
        assert selector["attributes"], selector["name"]
        assert "framework:qnn" in selector["attributes"]
        assert "htp-supports-fp16:true" in selector["attributes"]
        # A selector is exactly name/os/attributes: the AI Hub adapter rejects
        # any other key, so a citation may not be smuggled into one.
        assert set(selector) == {"name", "os", "attributes"}

    # The SDK's os field is a version string, not a human platform label.
    assert targets["primary"]["os"] == "11"
    assert "os:windows" in targets["primary"]["attributes"]
    assert targets["observed"]["devices_listed_by_the_service"] == 79
    assert targets["observed"]["recorded_here"] == 3
    assert "over-constrains" in targets["attributes_note"]


def test_emitted_selectors_are_accepted_by_the_ai_hub_adapter() -> None:
    """The observed attributes must survive the adapter's own sanitizer."""

    for selector in (w8.PRIMARY_DEVICE, *w8.COMPARISON_DEVICES):
        normalized = ai_hub._device(dict(selector))

        assert normalized["name"] == selector["name"]
        assert normalized["os"] == selector["os"]
        assert normalized["attributes"] == list(selector["attributes"])


def test_a_drifted_observation_breaks_the_citation_the_configs_carry(
    tmp_path: Path,
) -> None:
    """A self-consistent edit still has to fail: both configs cite the digest."""

    root = clone_repository(tmp_path)
    path = root / w8.DEFAULT_CAPABILITY_RECORD
    record = read_json(path)
    record["observation"]["device_count"] = 1000
    # Written the way the module writes it, so the byte-identity and
    # rebuild-from-observation checks both pass and the citation is what fails.
    w8.write_capability_record(path, w8.build_capability_record(record["observation"]))

    with pytest.raises(w8.W8EvidenceError, match="cited AI Hub capability record"):
        w8.validate_repository(root)


def test_an_edited_capability_claim_is_not_the_record_its_observation_derives(
    tmp_path: Path,
) -> None:
    """Widening the claim boundary by hand must not survive a check."""

    root = clone_repository(tmp_path)
    path = root / w8.DEFAULT_CAPABILITY_RECORD
    record = read_json(path)
    record["establishes"].append("that_w8a8_reaches_the_NPU")
    write_json(path, record)

    with pytest.raises(w8.W8EvidenceError, match="not the record its own"):
        w8.validate_repository(root)


def test_a_capability_record_claiming_a_submitted_job_is_refused(
    tmp_path: Path,
) -> None:
    root = clone_repository(tmp_path)
    path = root / w8.DEFAULT_CAPABILITY_RECORD
    record = read_json(path)
    record["submitted_jobs"] = 1
    write_json(path, record)

    with pytest.raises(w8.W8EvidenceError, match="claims a submitted job"):
        w8.validate_repository(root)


def test_a_missing_capability_record_fails_closed(tmp_path: Path) -> None:
    root = clone_repository(tmp_path)
    (root / w8.DEFAULT_CAPABILITY_RECORD).unlink()

    with pytest.raises(w8.W8EvidenceError, match="capability record is missing"):
        w8.validate_repository(root)


def test_a_selector_that_no_longer_matches_the_observation_fails(
    tmp_path: Path,
) -> None:
    """The literal selectors are only trustworthy because this check exists."""

    root = clone_repository(tmp_path)
    capability = read_json(root / w8.DEFAULT_CAPABILITY_RECORD)
    for device in capability["observation"]["devices"]:
        if device["name"] == w8.PRIMARY_DEVICE["name"]:
            device["os"] = "12"
    observation = capability["observation"]

    with pytest.raises(w8.W8EvidenceError, match="selector records os"):
        w8.build_deployment_routes(
            aimet_versions={"aimet-onnx": "2.36.0"},
            capability={**capability, "observation": observation},
        )


def test_evidence_ledger_defaults_to_not_run_and_names_its_blockers() -> None:
    document = read_yaml(REPO_ROOT / w8.candidate_config_path("w8a16"))
    ledger = document["evidence_requirements"]
    rows = {row["id"]: row for row in ledger["rows"]}

    assert ledger["default_status"] == "not_run"
    assert len(rows) == 10
    satisfied = [row for row in ledger["rows"] if row["status"] == "satisfied"]
    assert [row["id"] for row in satisfied] == [
        "calibration_corpus_revision_and_token_budget"
    ]
    # The granted permission cleared no row. Both rows that named it were also
    # missing an artifact, so they now name the two reasons there is none.
    assert not any(
        "user_authorization:qai_hub_submission" in row["blocked_on"]
        for row in ledger["rows"]
    )
    assert rows["peak_memory"]["status"] == "blocked"
    assert rows["peak_memory"]["blocked_on"] == list(w8.NO_W8_ARTIFACT_BLOCKERS)
    assert rows["graph_latency_and_npu_placement"]["status"] == "blocked"
    assert rows["graph_latency_and_npu_placement"]["blocked_on"] == [
        *w8.NO_W8_ARTIFACT_BLOCKERS,
        "upstream_task:T31",
    ]
    for row in ledger["rows"]:
        assert row["status"] in w8.LEDGER_STATUSES
        assert row["plan_reference"] == "docs/project/plan.md section 7.3"
        assert row["command"]["value"].strip()
        assert row["command"]["status"] in w8.COMMAND_STATUSES
        assert row["owner_task"].startswith("T")
        if row["status"] != "satisfied":
            assert row["blocked_on"], row["id"]
        for blocker in row["blocked_on"]:
            assert ":" in blocker
            assert blocker.split(":", 1)[0] in {
                "user_authorization",
                "hardware",
                "upstream_task",
                "dependency",
                "capability",
            }


def test_evaluation_binding_pins_the_frozen_protocol_and_refuses_others() -> None:
    document = read_yaml(REPO_ROOT / w8.candidate_config_path("w8a16"))
    evaluation = document["candidate"]["evaluation"]
    protocol = read_json(REPO_ROOT / w8.DEFAULT_BENCHMARK_PROTOCOL)

    assert (
        evaluation["benchmark_protocol"]["contract_sha256"]
        == (protocol["contract_sha256"])
    )
    assert evaluation["benchmark_protocol"]["owner_task"] == "T13"
    assert evaluation["academic_contract"]["suite_id"] == (
        "slm-lab-academic-regression-v1"
    )
    assert {task["id"] for task in evaluation["academic_contract"]["tasks"]} == {
        "wikitext_2_raw",
        "hellaswag_1000",
        "arc_easy_full_validation",
    }
    assert "not comparable" in evaluation["statement"]
    assert evaluation["comparison"]["function"] == (
        "slm_lab.quantization.w8.compare_quality"
    )
