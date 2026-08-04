"""Build the T31 three-target Qualcomm AI Hub Workbench run plan, offline.

T31 has to compile, run inference on, and profile the committed T22 Qwen
candidate graphs on three public targets. This module produces everything up
to and including the submission boundary and stops there. It never imports
``qai_hub``, never opens a socket, and never submits a job.

What it does:

* reads the three committed target selectors, the four T22 package records,
  the four T22 candidate manifests, the four T22 structural inspections, and
  the four T21/T22 ONNX Runtime CPU parity records;
* derives, for every (target, variant, graph kind), the compile request the
  T30 stage adapter would submit, and validates it through the committed T30
  validators rather than through a private copy of them;
* derives the deterministic ``request_id`` that
  :func:`slm_lab.deployment.qualcomm.ai_hub.preflight_compile_request` would
  record for that request, without touching the artifact bytes;
* specifies the inference and profile stages that follow, and marks them
  ``pending_predecessor`` because each needs a compile manifest naming an
  artifact that does not exist yet;
* orders the whole matrix into one deterministic submission order, with the
  reason for each position derived from committed measurements;
* carries each target's device-evidence strength, read off that target's own
  committed ``claim_boundary`` rather than assigned here;
* carries the known first-failure hypothesis as a field, so a failure at the
  external-data packaging boundary is attributable to packaging rather than
  to the graph.

Two stages are deliberately *not* materialized. An inference or profile
request needs ``predecessor_manifest`` to point at a successful compile
manifest, and ``compiled_artifact`` to carry that manifest's target-artifact
digest (see ``ai_hub._load_predecessor`` and ``ai_hub._compiled_artifact``).
Neither exists before a real compile job runs. Emitting a placeholder so the
request validates would produce a plan that passes here and fails at the
service, which is the failure mode this module exists to avoid.

Errors use the sanitized register of :mod:`ai_hub`: a message names the
logical object that failed, never a filesystem path, a service response, or a
credential.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Deliberate reuse of the committed T30 and T22 validators, not a
# reimplementation: a plan this module accepts must be a plan the stage
# adapters would accept, and a compile request it describes must be one the
# compile preflight would return the same request id for.
from .ai_hub import (
    _COMPILE_REQUEST_FIELDS,
    _COMPILE_REQUEST_OPTIONAL_FIELDS,
    SCHEMA_VERSION as ADAPTER_SCHEMA_VERSION,
    SHA256_PATTERN,
    AiHubAdapterError,
    _assert_public_safe,
    _common_request,
    _input_specs,
    _private_output_path,
    _public_request_projection,
    _request_id,
    _require_exact_keys,
    _safe_logical_name,
    _safe_text,
    _SafeArgumentParser,
    _validate_options,
    preflight_compile_request,
    sha256_file,
)
from .packaging import (
    PACKAGE_ROOT_TEMPLATE,
    QnnPackagingError,
    _expand_root,
    _first_difference,
    _repository_label,
    _repository_root,
    load_target_config,
    normalize_target,
    resolve_artifact_root,
)


SCHEMA_VERSION = 1
RECORD_TYPE = "slm_lab.qualcomm.workbench_run_plan"
TASK_ID = "T31"
SOURCE_TASK_ID = "T22"

#: The two readiness values a stage may hold. ``ready`` means every field the
#: stage adapter requires is fixed by committed inputs and the request is
#: accepted by the committed validators; it is submittable the moment a client
#: and permission exist. ``pending_predecessor`` means the stage is fully
#: specified except for values that only a real predecessor job can produce.
READY = "ready"
PENDING_PREDECESSOR = "pending_predecessor"
READINESS_VALUES = (READY, PENDING_PREDECESSOR)

STAGE_ORDER = ("compile", "inference", "profile")
GRAPH_KINDS = ("prefill", "decode")
VARIANT_IDS = ("S128", "S512", "S1024", "S4096")

TARGET_CONFIG_DIRECTORY = "configs/targets"
TARGET_CONFIG_FILES = (
    "qualcomm-snapdragon-x-elite-crd.json",
    "qualcomm-dragonwing-iq-9075-evk.json",
    "qualcomm-snapdragon-8-elite-qrd.json",
)
PACKAGE_RECORD_DIRECTORY = "results/manifests/qnn/packages"
INSPECTION_DIRECTORY = "results/manifests/qnn/inspection"
PARITY_DIRECTORY = "results/manifests/qnn/parity"

RECORD_DIRECTORY = "results/raw/qualcomm/workbench"
#: Pinned rather than computed from the clock, so ``--check`` keeps resolving
#: the same committed file tomorrow. A regeneration on a later date passes
#: ``--record`` and re-pins this constant in the same change.
RECORD_NAME = "t31-workbench-run-plan-2026-08-04.json"
REQUEST_DIRECTORY = ".ai-local/profiles/T31"
COMPILED_ROOT_TEMPLATE = "${SLM_LAB_ARTIFACT_ROOT}/qualcomm/compiled/T31"

#: The two high-severity structural rules whose residual population on the
#: candidate decides submission order. Both are T21 rule ids and both are read
#: off the committed T22 inspection manifests rather than restated here.
SHAPE_RULE_IDS = ("R-DATA-DEPENDENT-SHAPE-INPUT", "R-INTERNAL-DYNAMIC-SHAPE")

#: Markers a target config publishes about itself. A selector backed by an
#: authenticated device query says so in ``claim_boundary.establishes``; a
#: selector copied from an unauthenticated public catalog listing says the
#: opposite in ``claim_boundary.does_not_establish``. Classifying from the
#: config's own committed boundary means this module cannot promote a target
#: the config did not promote.
AUTHENTICATED_DEVICE_MARKER = "device_and_runtime_identity_match_committed_T02_evidence"
CATALOG_ONLY_DEVICE_MARKER = (
    "an_authenticated_device_query_confirmed_this_selector_resolves"
)
DEVICE_EVIDENCE_AUTHENTICATED = "authenticated_device_query"
DEVICE_EVIDENCE_CATALOG_ONLY = "unauthenticated_public_catalog_listing"
DEVICE_EVIDENCE_RANK = {
    DEVICE_EVIDENCE_AUTHENTICATED: 0,
    DEVICE_EVIDENCE_CATALOG_ONLY: 1,
}

ORDERING_POLICY = {
    "targets": {
        "keys": ["device_evidence_rank", "config_id"],
        "statement": (
            "Authenticated device evidence first, then catalog-only selectors "
            "in config_id order. The first target is the only one whose "
            "device identity an authenticated T02 query has ever returned; "
            "the other two are names read off an unauthenticated public "
            "catalog page, and nothing has confirmed that either resolves for "
            "this account."
        ),
    },
    "graphs": {
        "keys": [
            "residual_high_severity_shape_findings_on_the_candidate",
            "candidate_graph_protobuf_bytes",
            "variant_id",
            "graph_kind",
        ],
        "statement": (
            "Smallest residual shape population first, then smallest "
            "protobuf. A static-shape ahead-of-time compiler is exactly the "
            "consumer that cares about unresolved interior shapes, so the "
            "graph carrying fewest of them is the one whose failure would be "
            "attributable to the pipeline rather than to the graph. The "
            "population is read off the committed T22 inspection manifests, "
            "not asserted here."
        ),
    },
    "matrix": {
        "keys": ["target_order", "graph_order"],
        "statement": (
            "One target is exhausted before the next begins, so a failure "
            "that repeats across every graph on the first target is "
            "distinguishable from one that follows a particular graph across "
            "targets."
        ),
    },
}

NO_PROXY_RULE = (
    "A target that cannot be reached is reported with its exact blocker. A "
    "result measured on one target is never presented as a result for "
    "another, and the T02 toy-model lifecycle on Snapdragon X Elite CRD is "
    "not Qwen evidence for any target including that one."
)

COMPILE_VALIDATION_SCOPE = (
    "the committed T30 compile validation chain except the two filesystem "
    "checks named in deferred_to_submission_time"
)
COMPILE_DEFERRED_CHECKS = (
    "source_artifact_path_exists_and_its_bytes_hash_to_the_recorded_sha256",
    "output_artifact_parent_is_private_storage_and_can_be_prepared",
)
COMPILE_VALIDATED_NOW = (
    "adapter_schema_version_and_stage",
    "exact_compile_request_field_set",
    "client_version_is_an_exact_version",
    "device_selector_through_the_committed_T30_device_validator",
    "runtime_identity_is_QAIRT_at_an_exact_version",
    "option_string_through_the_committed_T30_compile_allowlist",
    "job_name_timeout_and_retry_is_false",
    "input_specs_shapes_are_positive_and_dtypes_are_supported",
    "source_and_output_logical_names_are_path_free",
    "source_digest_is_a_lowercase_sha256_matching_the_committed_package_record",
    "public_request_projection_is_free_of_paths_urls_and_private_text",
)
DOWNSTREAM_VALIDATED_NOW = (
    "device_selector_through_the_committed_T30_device_validator",
    "runtime_identity_is_QAIRT_at_an_exact_version",
    "option_string_through_the_committed_T30_allowlist_for_this_stage",
    "job_name_timeout_and_retry_is_false",
)

#: Why ``--compute_unit`` is absent from the inference and profile option
#: strings. The T30 allowlist accepts it; the plan omits it on purpose.
COMPUTE_UNIT_NOTE = (
    "No --compute_unit flag is set. The T30 allowlist accepts one, but "
    "constraining the compute unit would decide the placement question that "
    "the profile stage exists to observe. The normalized profile reports the "
    "compute units it actually saw, and a plan that pinned them would return "
    "its own input."
)

CLAIM_BOUNDARY = {
    "establishes": [
        "a_deterministic_three_target_run_plan_derived_only_from_committed_inputs",
        "every_compile_request_in_the_plan_satisfies_the_committed_T30_compile_"
        "validation_chain_offline",
        "each_compile_request_id_equals_the_value_the_T30_preflight_records_for_"
        "the_same_request",
        "inference_and_profile_specifications_that_are_explicitly_blocked_on_a_"
        "real_compile_output_rather_than_completed_with_a_placeholder",
        "a_deterministic_submission_order_whose_reason_is_derived_from_the_"
        "committed_T22_inspection_manifests",
        "each_target_device_evidence_strength_read_from_that_target_own_committed_"
        "claim_boundary",
        "a_dated_record_that_no_job_was_submitted_no_service_was_contacted_and_"
        "nothing_was_spent",
    ],
    "does_not_establish": [
        "qualcomm_ai_hub_accepted_or_would_accept_any_request_in_this_plan",
        "qualcomm_ai_hub_accepts_the_external_data_package_layout_this_plan_names",
        "compiler_acceptance_or_operator_support_for_any_qwen3_graph",
        "that_any_of_the_three_devices_resolves_is_schedulable_or_is_reachable_by_"
        "this_account",
        "accelerator_placement_or_fallback_behaviour",
        "latency_throughput_peak_memory_or_energy_on_any_target",
        "device_numerical_parity_or_any_comparison_against_the_T22_reference_logits",
        "that_a_result_on_one_target_transfers_to_another_target",
    ],
}


class WorkbenchPlanError(AiHubAdapterError):
    """A sanitized planning error safe to print in public task logs."""


def _observed_at() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise WorkbenchPlanError(f"{field} is not readable valid JSON") from None
    if not isinstance(value, Mapping):
        raise WorkbenchPlanError(f"{field} must be a JSON object")
    return value


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkbenchPlanError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise WorkbenchPlanError(f"{field} must be a list")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorkbenchPlanError(f"{field} must be a nonempty string")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkbenchPlanError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkbenchPlanError(f"{field} must be a nonnegative integer")
    return value


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise WorkbenchPlanError(f"{field} must be a lowercase SHA-256")
    return value


def _input_reference(path: Path, field: str) -> dict[str, Any]:
    """Bind an input by repository label and digest so drift cannot hide."""

    if not path.is_file():
        raise WorkbenchPlanError(f"{field} is missing")
    return {"path": _repository_label(path), "sha256": sha256_file(path)}


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def device_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    """Classify a target's device evidence from its own committed boundary.

    Returns the strength, the marker that decided it, and the rule that stops
    a catalog-only target from being read as a confirmed one. Raises when a
    config declares neither marker, because guessing the strength of a
    selector is exactly the mistake this field exists to prevent.
    """

    boundary = _mapping(config.get("claim_boundary"), "target config claim_boundary")
    establishes = tuple(
        _text(item, "target config claim_boundary.establishes")
        for item in _sequence(
            boundary.get("establishes"), "target config claim_boundary.establishes"
        )
    )
    does_not = tuple(
        _text(item, "target config claim_boundary.does_not_establish")
        for item in _sequence(
            boundary.get("does_not_establish"),
            "target config claim_boundary.does_not_establish",
        )
    )
    authenticated = AUTHENTICATED_DEVICE_MARKER in establishes
    catalog_only = CATALOG_ONLY_DEVICE_MARKER in does_not
    if authenticated and not catalog_only:
        strength = DEVICE_EVIDENCE_AUTHENTICATED
        marker = AUTHENTICATED_DEVICE_MARKER
        confirmed = True
    elif catalog_only and not authenticated:
        strength = DEVICE_EVIDENCE_CATALOG_ONLY
        marker = CATALOG_ONLY_DEVICE_MARKER
        confirmed = False
    else:
        raise WorkbenchPlanError(
            "target config claim_boundary declares neither an authenticated "
            "device marker nor a catalog-only one, so its device evidence "
            "strength cannot be derived"
        )
    return {
        "strength": strength,
        "rank": DEVICE_EVIDENCE_RANK[strength],
        "decided_by_marker": marker,
        "device_confirmed_by_authenticated_query": confirmed,
        "does_not_establish": list(does_not),
        "no_proxy_rule": NO_PROXY_RULE,
    }


def _stage_job_prefix(compile_prefix: str, stage: str) -> str:
    """Derive a stage job-name prefix from the committed compile prefix.

    The target configs declare only ``compile.job_name_prefix``. Rather than
    invent a second naming scheme, the stage name replaces the first literal
    ``compile`` token in the committed prefix, which keeps every job name for
    a target on one recognisable stem. A prefix without that token is refused
    instead of silently suffixed.
    """

    if stage == "compile":
        return compile_prefix
    if "compile" not in compile_prefix:
        raise WorkbenchPlanError(
            "target config compile.job_name_prefix does not contain the token "
            "compile, so a stage-specific job name cannot be derived from it"
        )
    return compile_prefix.replace("compile", stage, 1)


def _stage_options(stage: str, target: Mapping[str, Any]) -> str:
    """Return the option string for one stage, validated by the T30 allowlist."""

    runtime = target["runtime"]
    if stage == "compile":
        options = target["options"]
    else:
        # Inference and profile bind the runtime with --qairt_framework
        # exactly once. Nothing else is added; see COMPUTE_UNIT_NOTE.
        options = f"--qairt_framework {runtime['version']}"
    return _validate_options(options, runtime, stage)


def load_targets(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load, validate, classify, and order the target selectors."""

    if not paths:
        raise WorkbenchPlanError("no target config was supplied")
    loaded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        config = load_target_config(path)
        normalized = normalize_target(config)
        config_id = _text(config.get("config_id"), "target config config_id")
        if config_id in seen:
            raise WorkbenchPlanError("two target configs declare the same config_id")
        seen.add(config_id)
        compile_block = _mapping(config.get("compile"), "target config compile")
        compile_prefix = _text(
            compile_block.get("job_name_prefix"),
            "target config compile.job_name_prefix",
        )
        evidence = device_evidence(config)
        options = {stage: _stage_options(stage, normalized) for stage in STAGE_ORDER}
        prefixes = {
            stage: _safe_text(
                _stage_job_prefix(compile_prefix, stage), "job_name_prefix"
            )
            for stage in STAGE_ORDER
        }
        loaded.append(
            {
                "config_id": config_id,
                "source": _input_reference(path, "target config"),
                "client": {"name": "qai-hub", "version": normalized["client_version"]},
                "device": normalized["device"],
                "runtime": normalized["runtime"],
                "device_evidence": evidence,
                "stage_options": options,
                "job_name_prefixes": prefixes,
                "job_name_prefix_source": "target_config.compile.job_name_prefix",
                "timeout_seconds": _positive_int(
                    compile_block.get("timeout_seconds"),
                    "target config compile.timeout_seconds",
                ),
                "retry": False,
                "compute_unit_note": COMPUTE_UNIT_NOTE,
            }
        )
    loaded.sort(key=lambda item: (item["device_evidence"]["rank"], item["config_id"]))
    for position, item in enumerate(loaded, start=1):
        item["target_order"] = position
    return loaded


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------


def _shape_residue(
    inspection: Mapping[str, Any], variant_id: str, graph_kind: str
) -> dict[str, Any]:
    graphs = _mapping(inspection.get("graphs"), "inspection graphs")
    pair = _mapping(graphs.get(graph_kind), f"inspection graphs.{graph_kind}")
    residue: dict[str, Any] = {}
    for side in ("reference", "candidate"):
        block = _mapping(pair.get(side), f"inspection graphs.{graph_kind}.{side}")
        counts = {rule: 0 for rule in SHAPE_RULE_IDS}
        for index, raw in enumerate(
            _sequence(block.get("findings"), "inspection findings")
        ):
            finding = _mapping(raw, f"inspection findings[{index}]")
            rule_id = finding.get("rule_id")
            if rule_id in counts:
                counts[rule_id] = _nonnegative_int(
                    finding.get("count"), "inspection finding count"
                )
        residue[side] = {
            **counts,
            "total": sum(counts.values()),
            "node_count": _positive_int(
                block.get("node_count"), "inspection node_count"
            ),
        }
    candidate = _mapping(pair.get("candidate"), "inspection candidate")
    return {
        "rules": list(SHAPE_RULE_IDS),
        "reference": residue["reference"],
        "candidate": residue["candidate"],
        "candidate_source_sha256": _digest(
            candidate.get("source_sha256"), "inspection candidate source_sha256"
        ),
        "note": (
            f"The shape fold takes {variant_id} {graph_kind} from "
            f"{residue['reference']['total']} residual high-severity shape "
            f"findings on the reference to {residue['candidate']['total']} on "
            "the candidate."
            + (
                " On decode the fold converts one rule's population into the "
                "other's rather than removing it, so the candidate scores "
                "worse than the reference by raw count."
                if residue["candidate"]["total"] > residue["reference"]["total"]
                else ""
            )
        ),
    }


def _package_graph(record: Mapping[str, Any], graph_kind: str) -> Mapping[str, Any]:
    graphs = _sequence(
        _mapping(record.get("package"), "package record package").get("graphs"),
        "package record package.graphs",
    )
    for raw in graphs:
        graph = _mapping(raw, "package record graph")
        if graph.get("graph_kind") == graph_kind:
            return graph
    raise WorkbenchPlanError(f"package record has no {graph_kind} graph")


def load_graphs(
    *,
    package_paths: Mapping[str, Path],
    inspection_paths: Mapping[str, Path],
    parity_paths: Mapping[str, Path],
    repository_root: Path,
) -> list[dict[str, Any]]:
    """Load the candidate graph catalogue and derive the submission order."""

    graphs: list[dict[str, Any]] = []
    for variant_id in VARIANT_IDS:
        record = _load_json(package_paths[variant_id], "package record")
        if record.get("schema_version") != 1 or record.get("task_id") != SOURCE_TASK_ID:
            raise WorkbenchPlanError(
                "package record schema_version or task_id is unsupported"
            )
        if record.get("record_type") != "slm_lab.qualcomm.qnn_package":
            raise WorkbenchPlanError("package record record_type is unsupported")
        if record.get("variant_id") != variant_id:
            raise WorkbenchPlanError("package record variant_id does not match")
        inspection = _load_json(inspection_paths[variant_id], "inspection manifest")
        parity = _load_json(parity_paths[variant_id], "parity record")
        if parity.get("record_kind") != "t21_ort_cpu_parity":
            raise WorkbenchPlanError("parity record kind is unsupported")

        source_manifest = _mapping(
            record.get("source_manifest"), "package record source_manifest"
        )
        candidate_manifest_path = repository_root / _text(
            source_manifest.get("path"), "package record source_manifest.path"
        )
        candidate_manifest = _input_reference(
            candidate_manifest_path, "candidate manifest"
        )
        if candidate_manifest["sha256"] != _digest(
            source_manifest.get("sha256"), "package record source_manifest.sha256"
        ):
            raise WorkbenchPlanError(
                "candidate manifest digest no longer matches the package record"
            )

        for graph_kind in GRAPH_KINDS:
            package_graph = _package_graph(record, graph_kind)
            files = _sequence(package_graph.get("files"), "package record files")
            members = [_mapping(item, "package record file") for item in files]
            source = next(
                (item for item in members if item.get("role") == "candidate_graph"),
                None,
            )
            if source is None:
                raise WorkbenchPlanError("package record graph has no candidate graph")
            sidecars = [
                {
                    "logical_name": _safe_logical_name(
                        item.get("logical_name"), "external data logical_name"
                    ),
                    "sha256": _digest(item.get("sha256"), "external data sha256"),
                    "size_bytes": _positive_int(
                        item.get("size_bytes"), "external data size_bytes"
                    ),
                }
                for item in members
                if item.get("role") == "external_data"
            ]
            residue = _shape_residue(inspection, variant_id, graph_kind)
            source_sha = _digest(source.get("sha256"), "candidate graph sha256")
            if residue["candidate_source_sha256"] != source_sha:
                raise WorkbenchPlanError(
                    "inspection manifest and package record disagree about the "
                    "candidate graph digest"
                )
            parity_digests = _mapping(
                parity.get("graph_digests"), "parity record graph_digests"
            )
            parity_graph = _mapping(
                parity_digests.get(graph_kind), "parity record graph digest"
            )
            if (
                _digest(parity_graph.get("sha256"), "parity record graph sha256")
                != source_sha
            ):
                raise WorkbenchPlanError(
                    "parity record and package record disagree about the "
                    "candidate graph digest"
                )
            compile_request = _mapping(
                package_graph.get("compile_request"), "package record compile_request"
            )
            public_specs, _ = _input_specs(compile_request.get("input_specs"))
            graphs.append(
                {
                    "variant_id": variant_id,
                    "graph_kind": graph_kind,
                    "context_length": _positive_int(
                        record.get("context_length"), "package record context_length"
                    ),
                    "cache_capacity": _positive_int(
                        record.get("cache_capacity"), "package record cache_capacity"
                    ),
                    "precision": _text(
                        record.get("precision"), "package record precision"
                    ),
                    "package_relative_path": _text(
                        package_graph.get("package_relative_path"),
                        "package record package_relative_path",
                    ),
                    "source_logical_name": _safe_logical_name(
                        source.get("logical_name"), "candidate graph logical_name"
                    ),
                    "source_sha256": source_sha,
                    "source_byte_size": _positive_int(
                        source.get("size_bytes"), "candidate graph size_bytes"
                    ),
                    "external_data": sidecars,
                    "input_tensor_count": len(public_specs),
                    "input_specs_source": (
                        "results/manifests/qnn/packages/"
                        f"{variant_id}.json package.graphs[].compile_request."
                        "input_specs"
                    ),
                    "shape_residue": residue,
                    "numerical_reference": {
                        "record": _repository_label(parity_paths[variant_id]),
                        "sha256": sha256_file(parity_paths[variant_id]),
                        "evidence_tier": _text(
                            parity.get("evidence_tier"), "parity record evidence_tier"
                        ),
                        "reference_logits_sha256_source": (
                            "reference_provenance.reference_logits_sha256"
                        ),
                        "status": "not_compared",
                        "blocked_by": [
                            "no_inference_stage_has_run_on_any_target",
                            "no_ai_hub_input_dataset_exists_in_this_repository",
                        ],
                        "note": (
                            "The T22 candidates are bit-identical to the "
                            "reference on the ONNX Runtime CPU provider, so a "
                            "divergence observed on a device belongs to the "
                            "compiler, the runtime, or the hardware. Nothing "
                            "was compared here: this field names the reference "
                            "an inference stage would be compared against."
                        ),
                    },
                    "inputs": {
                        "package_record": _input_reference(
                            package_paths[variant_id], "package record"
                        ),
                        "candidate_manifest": candidate_manifest,
                        "inspection_manifest": _input_reference(
                            inspection_paths[variant_id], "inspection manifest"
                        ),
                    },
                }
            )

    graphs.sort(
        key=lambda item: (
            item["shape_residue"]["candidate"]["total"],
            item["source_byte_size"],
            VARIANT_IDS.index(item["variant_id"]),
            GRAPH_KINDS.index(item["graph_kind"]),
        )
    )
    for position, graph in enumerate(graphs, start=1):
        graph["graph_order"] = position
        graph["graph_order_key"] = {
            "residual_high_severity_shape_findings": graph["shape_residue"][
                "candidate"
            ]["total"],
            "candidate_graph_protobuf_bytes": graph["source_byte_size"],
        }
    return graphs


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def compile_request_for(
    target: Mapping[str, Any],
    graph: Mapping[str, Any],
    input_specs: Mapping[str, Any],
    *,
    package_root: str | Path | None = None,
    compiled_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compose the schema-v2 compile request for one (target, graph) pair.

    ``package_root`` and ``compiled_root`` default to the artifact-root token
    forms, which are not real paths. That is deliberate: the two fields they
    fill are dropped by the public projection before hashing, so the request
    id this module derives does not depend on where the bytes happen to live.
    Passing real roots produces a request that ``preflight_compile_request``
    can validate end to end.
    """

    root = PACKAGE_ROOT_TEMPLATE if package_root is None else str(package_root)
    compiled = COMPILED_ROOT_TEMPLATE if compiled_root is None else str(compiled_root)
    output_logical_name = f"{graph['variant_id']}-{graph['graph_kind']}.serialized.bin"
    job_name = (
        f"{target['job_name_prefixes']['compile']}-"
        f"{graph['variant_id']}-{graph['graph_kind']}"
    )
    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "stage": "compile",
        "client_version": target["client"]["version"],
        "device": dict(target["device"]),
        "runtime": dict(target["runtime"]),
        "source_artifact": {
            "path": (
                f"{root}/{graph['package_relative_path']}/"
                f"{graph['source_logical_name']}"
            ),
            "logical_name": graph["source_logical_name"],
            "sha256": graph["source_sha256"],
        },
        "output_artifact": (
            f"{compiled}/{target['config_id']}/{graph['variant_id']}/"
            f"{output_logical_name}"
        ),
        "output_logical_name": output_logical_name,
        "input_specs": json.loads(json.dumps(input_specs)),
        "options": target["stage_options"]["compile"],
        "job_name": job_name,
        "timeout_seconds": target["timeout_seconds"],
        "retry": target["retry"],
    }


def validate_compile_request_offline(
    request: Mapping[str, Any],
    *,
    source_byte_size: int,
) -> dict[str, Any]:
    """Run the committed compile validation chain without touching artifacts.

    This is :func:`ai_hub.preflight_compile_request` minus exactly two steps:
    the source artifact's existence and rehash, and the preparation of the
    output artifact's private parent directory. Both need bytes on a disk that
    a plan does not have, and both are recorded on the plan under
    ``deferred_to_submission_time`` rather than dropped. Every other check
    runs, in the same order, through the same functions.

    ``source_byte_size`` comes from the committed T22 package record, which
    verified it against the file. It enters the public projection exactly
    where ``_artifact_from_request`` would have put ``path.stat().st_size``,
    which is what makes the returned request id equal the one the real
    preflight records.
    """

    _require_exact_keys(
        request,
        required=set(_COMPILE_REQUEST_FIELDS),
        optional=set(_COMPILE_REQUEST_OPTIONAL_FIELDS),
        field="compile request",
    )
    if (
        request["schema_version"] != ADAPTER_SCHEMA_VERSION
        or request["stage"] != "compile"
    ):
        raise WorkbenchPlanError("compile request has wrong schema or stage")
    common = _common_request(request, "compile")
    artifact = _mapping(request["source_artifact"], "source_artifact")
    _require_exact_keys(
        artifact,
        required={"path", "logical_name", "sha256"},
        field="source_artifact",
    )
    source = {
        "role": "source_model",
        "logical_name": _safe_logical_name(
            artifact["logical_name"], "source_artifact.logical_name"
        ),
        "sha256": _digest(artifact["sha256"], "source_artifact.sha256"),
        "byte_size": _positive_int(source_byte_size, "source_artifact byte size"),
    }
    public_specs, _ = _input_specs(request["input_specs"])
    output_logical_name = _safe_logical_name(
        request["output_logical_name"], "output_logical_name"
    )
    public_request = {
        **_public_request_projection(request),
        "source_artifact": source,
        "input_specs": public_specs,
    }
    _assert_public_safe(public_request, "compile request")
    return {
        "request_id": _request_id("compile", public_request),
        "client_version": common["client_version"],
        "output_logical_name": output_logical_name,
        "public_request": public_request,
    }


def _downstream_stage(
    stage: str,
    target: Mapping[str, Any],
    graph: Mapping[str, Any],
    compiled_logical_name: str,
) -> dict[str, Any]:
    """Specify an inference or profile stage that no predecessor exists for."""

    job_name = _safe_text(
        f"{target['job_name_prefixes'][stage]}-"
        f"{graph['variant_id']}-{graph['graph_kind']}",
        "job_name",
    )
    unresolved = ["predecessor_manifest", "compiled_artifact_sha256", "output_path"]
    if stage == "inference":
        unresolved.insert(2, "input_dataset")
    return {
        "readiness": PENDING_PREDECESSOR,
        "depends_on_stage": "compile",
        "job_name": job_name,
        "compiled_artifact_logical_name": compiled_logical_name,
        "compiled_artifact_sha256": None,
        "predecessor_manifest": None,
        "request_id": None,
        "unresolved_input_ids": unresolved,
    }


def _stage_contracts() -> dict[str, Any]:
    """The per-stage rules every plan entry shares, stated once."""

    return {
        "compile": {
            "readiness": READY,
            "readiness_rule": (
                "Every field the T30 compile stage requires is fixed by "
                "committed inputs. The request is submittable the moment a "
                "qai-hub client at the pinned version and explicit user "
                "permission both exist."
            ),
            "validated_now": list(COMPILE_VALIDATED_NOW),
            "validation_scope": COMPILE_VALIDATION_SCOPE,
            "deferred_to_submission_time": list(COMPILE_DEFERRED_CHECKS),
            "validated_by": (
                "slm_lab.deployment.qualcomm.workbench.validate_compile_request_offline"
            ),
            "equivalent_full_check": (
                "slm_lab.deployment.qualcomm.ai_hub.preflight_compile_request"
            ),
        },
        "inference": {
            "readiness": PENDING_PREDECESSOR,
            "readiness_rule": (
                "Specified but not submittable. ai_hub._load_predecessor "
                "requires a successful compile manifest, and "
                "ai_hub._compiled_artifact requires the compiled artifact's "
                "logical name and digest to equal that manifest's. Neither "
                "exists before a real compile job runs."
            ),
            "validated_now": list(DOWNSTREAM_VALIDATED_NOW),
            "unresolved_inputs": [
                {
                    "id": "predecessor_manifest",
                    "produced_by": "a successful compile stage on the same target",
                    "reason": (
                        "The path of a sanitized schema-v2 compile manifest "
                        "with status success. No such manifest exists in this "
                        "repository for any target."
                    ),
                },
                {
                    "id": "compiled_artifact_sha256",
                    "produced_by": "the compile stage's downloaded target model",
                    "reason": (
                        "The digest is read off the bytes the service "
                        "returns, and the adapter refuses a compiled artifact "
                        "whose digest differs from the predecessor manifest's."
                    ),
                },
                {
                    "id": "input_dataset",
                    "produced_by": "T31 at submission time",
                    "reason": (
                        "The inference stage needs an AI Hub-compatible HDF5 "
                        "dataset. This repository contains none at this "
                        "commit; the T22 parity record names the prompt token "
                        "ids by digest but is not that dataset."
                    ),
                },
                {
                    "id": "output_path",
                    "produced_by": "the submitting session",
                    "reason": (
                        "A machine-local private path under ignored storage, "
                        "chosen at submission time and never committed."
                    ),
                },
            ],
            "no_placeholder_rule": (
                "A synthetic predecessor manifest would make the request "
                "validate here and fail at the service. The stage is left "
                "incomplete on purpose and its missing fields are null."
            ),
        },
        "profile": {
            "readiness": PENDING_PREDECESSOR,
            "readiness_rule": (
                "Specified but not submittable, for the same predecessor "
                "reason as inference. Profile needs no dataset."
            ),
            "validated_now": list(DOWNSTREAM_VALIDATED_NOW),
            "unresolved_inputs": [
                {
                    "id": "predecessor_manifest",
                    "produced_by": "a successful compile stage on the same target",
                    "reason": (
                        "The path of a sanitized schema-v2 compile manifest "
                        "with status success."
                    ),
                },
                {
                    "id": "compiled_artifact_sha256",
                    "produced_by": "the compile stage's downloaded target model",
                    "reason": (
                        "Read off the returned bytes and cross-checked "
                        "against the predecessor manifest."
                    ),
                },
                {
                    "id": "output_path",
                    "produced_by": "the submitting session",
                    "reason": (
                        "raw_profile_output is a machine-local private path "
                        "under ignored storage. The raw service profile is "
                        "never committed; only the normalized profile is."
                    ),
                },
            ],
            "no_placeholder_rule": (
                "Same rule as inference: nothing is filled in that only a "
                "real compile job can produce."
            ),
        },
    }


def _first_failure_hypothesis(graphs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "id": "packaging:external_data_sidecar_layout_unverified",
        "affected_stage": "compile",
        "expected_position": 1,
        "statement": (
            "The compile request names only the .onnx file, because the "
            "committed T30 adapter requires source_artifact.path to be one "
            "existing file. Every candidate in the matrix carries an "
            "external-data sidecar of roughly 1.19 GB beside it. Whether the "
            "service reads that sidecar from the same directory, or wants a "
            "directory or an archive instead, has never been tested against "
            "Qualcomm AI Hub."
        ),
        "attribution_rule": (
            "A failure whose diagnostic names the model upload, a missing "
            "external data file, or the source artifact is a PACKAGING "
            "result. It must not be recorded as a graph result, a compiler "
            "result, or an operator-support result, and it says nothing about "
            "the target it happened on. Re-attempt the same graph with a "
            "directory or archive source before concluding anything about the "
            "graph."
        ),
        "distinguishing_observation": (
            "A packaging failure reproduces identically on every target and "
            "for every variant, because the layout is target-independent. A "
            "compiler or operator failure varies with the graph. The plan's "
            "submission order makes that test cheap: the first two positions "
            "are the same graph kind at different residual shape populations."
        ),
        "sources": [
            "ai/handoffs/T22-qnn-candidates.md",
            "ai/handoffs/T31-first-submission.md",
            "results/manifests/qnn/packages/README.md",
        ],
        "external_data_by_graph": [
            {
                "variant_id": graph["variant_id"],
                "graph_kind": graph["graph_kind"],
                "graph_byte_size": graph["source_byte_size"],
                "external_data": [
                    {
                        "logical_name": sidecar["logical_name"],
                        "size_bytes": sidecar["size_bytes"],
                    }
                    for sidecar in graph["external_data"]
                ],
            }
            for graph in sorted(
                graphs, key=lambda item: (item["variant_id"], item["graph_kind"])
            )
        ],
    }


def _submission_boundary() -> dict[str, Any]:
    return {
        "stops_at": "compile_stage_submission",
        "jobs_submitted": 0,
        "service_contacted": False,
        "network_calls_made": 0,
        "required_before_any_submission": [
            {
                "id": "capability:qai_hub_client_absent",
                "kind": "capability",
                "requirement": (
                    "The optional qai-hub client at the exact version every "
                    "target config names. pyproject.toml pins no Qualcomm "
                    "client, and no environment in this repository provides "
                    "one."
                ),
                "owner": "environment",
                "offline_recheck": (
                    'python3 -c "import importlib.util,sys; '
                    "sys.exit(0 if importlib.util.find_spec('qai_hub') else 1)\""
                ),
                "note": (
                    "find_spec does not execute the module, so the recheck "
                    "itself never imports the client."
                ),
            },
            {
                "id": "user_authorization:qai_hub_submission_for_T31",
                "kind": "permission",
                "requirement": (
                    "Explicit user permission, given in the session that "
                    "submits, for hosted AI Hub jobs under T31."
                ),
                "owner": "user",
                "offline_recheck": "not_machine_checkable",
                "note": (
                    "AGENTS.md requires explicit user permission before any "
                    "external job. The budget paragraph in "
                    "ai/handoffs/T31-first-submission.md records an earlier "
                    "grant written into a file by an earlier agent; that is "
                    "context for a future session, not consent held by this "
                    "one."
                ),
            },
        ],
        "deliberately_not_done": [
            "installing_the_qai_hub_client",
            "any_network_call",
            "any_job_submission",
            "any_device_cloud_lease",
        ],
        "resource_lock": "qai_hub_submission",
    }


def _cost_record() -> dict[str, Any]:
    return {
        "jobs_submitted": 0,
        "device_minutes_consumed": 0,
        "currency": "USD",
        "amount_spent": "0.00",
        "free_capacity_only": True,
        "note": (
            "Nothing was submitted, leased, or spent. This row exists so the "
            "external-job ledger AGENTS.md requires has an entry for T31 that "
            "reads zero rather than being absent."
        ),
    }


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def default_paths(repository_root: Path | None = None) -> dict[str, Any]:
    """Resolve every committed input this module reads."""

    root = repository_root if repository_root is not None else _repository_root()
    return {
        "repository_root": root,
        "targets": [
            root / TARGET_CONFIG_DIRECTORY / name for name in TARGET_CONFIG_FILES
        ],
        "packages": {
            variant: root / PACKAGE_RECORD_DIRECTORY / f"{variant}.json"
            for variant in VARIANT_IDS
        },
        "inspections": {
            variant: root / INSPECTION_DIRECTORY / f"{variant}.json"
            for variant in VARIANT_IDS
        },
        "parity": {
            variant: root / PARITY_DIRECTORY / f"{variant}-ort-cpu.json"
            for variant in VARIANT_IDS
        },
        "record": root / RECORD_DIRECTORY / RECORD_NAME,
    }


def build_plan(
    *,
    target_paths: Sequence[Path],
    package_paths: Mapping[str, Path],
    inspection_paths: Mapping[str, Path],
    parity_paths: Mapping[str, Path],
    repository_root: Path,
) -> dict[str, Any]:
    """Build the deterministic plan.

    A pure function of the committed inputs and the policy constants in this
    module. Nothing observed from the local machine enters it, so ``build``
    and ``--check`` can compare their results directly.
    """

    targets = load_targets(target_paths)
    graphs = load_graphs(
        package_paths=package_paths,
        inspection_paths=inspection_paths,
        parity_paths=parity_paths,
        repository_root=repository_root,
    )
    specs_by_graph: dict[tuple[str, str], Mapping[str, Any]] = {}
    for variant_id in VARIANT_IDS:
        record = _load_json(package_paths[variant_id], "package record")
        for graph_kind in GRAPH_KINDS:
            package_graph = _package_graph(record, graph_kind)
            compile_request = _mapping(
                package_graph.get("compile_request"), "package record compile_request"
            )
            specs, _ = _input_specs(compile_request.get("input_specs"))
            specs_by_graph[(variant_id, graph_kind)] = specs

    entries: list[dict[str, Any]] = []
    for target in targets:
        for graph in graphs:
            key = (graph["variant_id"], graph["graph_kind"])
            request = compile_request_for(target, graph, specs_by_graph[key])
            validated = validate_compile_request_offline(
                request, source_byte_size=graph["source_byte_size"]
            )
            compiled_logical_name = validated["output_logical_name"]
            entries.append(
                {
                    "submission_order": 0,
                    "target": target["config_id"],
                    "target_order": target["target_order"],
                    "device_evidence_strength": target["device_evidence"]["strength"],
                    "variant_id": graph["variant_id"],
                    "graph_kind": graph["graph_kind"],
                    "graph_order": graph["graph_order"],
                    "stages": {
                        "compile": {
                            "readiness": READY,
                            "request_id": validated["request_id"],
                            "job_name": request["job_name"],
                            "source_logical_name": graph["source_logical_name"],
                            "output_logical_name": compiled_logical_name,
                        },
                        "inference": _downstream_stage(
                            "inference", target, graph, compiled_logical_name
                        ),
                        "profile": _downstream_stage(
                            "profile", target, graph, compiled_logical_name
                        ),
                    },
                }
            )

    entries.sort(key=lambda item: (item["target_order"], item["graph_order"]))
    for position, entry in enumerate(entries, start=1):
        entry["submission_order"] = position

    first = entries[0]
    plan = {
        "schema_version": SCHEMA_VERSION,
        "record_type": RECORD_TYPE,
        "task_id": TASK_ID,
        "source_task_id": SOURCE_TASK_ID,
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "generated_by": "slm_lab.deployment.qualcomm.workbench",
        "summary": {
            "targets": len(targets),
            "graphs": len(graphs),
            "plan_entries": len(entries),
            "stages_ready": sum(
                1
                for entry in entries
                for stage in STAGE_ORDER
                if entry["stages"][stage]["readiness"] == READY
            ),
            "stages_pending_predecessor": sum(
                1
                for entry in entries
                for stage in STAGE_ORDER
                if entry["stages"][stage]["readiness"] == PENDING_PREDECESSOR
            ),
            "first_submission": {
                "target": first["target"],
                "variant_id": first["variant_id"],
                "graph_kind": first["graph_kind"],
                "stage": "compile",
                "request_id": first["stages"]["compile"]["request_id"],
                "reason": (
                    "The only graph in the matrix whose residual "
                    "high-severity shape population is zero, the smallest "
                    "candidate protobuf in the matrix, and the only target "
                    "whose device identity an authenticated query has "
                    "returned. If it fails, the failure is about the "
                    "pipeline, not about the graph."
                ),
            },
        },
        "ordering_policy": json.loads(json.dumps(ORDERING_POLICY)),
        "job_naming": {
            "rule": (
                "Each job name is the target config's own "
                "compile.job_name_prefix with the stage name substituted for "
                "its first compile token, followed by the variant id and the "
                "graph kind."
            ),
            "note": (
                "The Snapdragon X Elite CRD selector is owned by T22 and its "
                "prefix says t22. That is kept rather than rewritten: the "
                "job name is part of the hashed request, so renaming it would "
                "change the compile request id and break its equality with "
                "the request id the committed T22 package record already "
                "carries."
            ),
        },
        "targets": targets,
        "graphs": graphs,
        "stage_contracts": _stage_contracts(),
        "plan_entry_join": (
            "Every plan entry names a target config_id and a "
            "(variant_id, graph_kind) pair. The option strings, timeouts, "
            "retry policy, and device selector live once under targets[]; the "
            "digests, byte sizes, input tensor counts, and shape residues live "
            "once under graphs[]. A plan entry carries only what is specific "
            "to the pair."
        ),
        "plan": entries,
        "first_failure_hypothesis": _first_failure_hypothesis(graphs),
        "submission_boundary": _submission_boundary(),
        "cost": _cost_record(),
        "no_proxy_rule": NO_PROXY_RULE,
        "claim_boundary": json.loads(json.dumps(CLAIM_BOUNDARY)),
    }
    _assert_public_safe(plan, "workbench run plan")
    return plan


# ---------------------------------------------------------------------------
# Optional full preflight
# ---------------------------------------------------------------------------


def preflight_plan(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
    request_directory: Path,
) -> list[dict[str, Any]]:
    """Run the real T30 compile preflight over every ``ready`` compile stage.

    Optional because it needs the assembled T22 packages on the external
    artifact root. It writes each request into private storage, runs
    :func:`ai_hub.preflight_compile_request`, and refuses any request whose
    real preflight id differs from the id this module derived offline. That
    equality is the whole reason the offline derivation is trustworthy.
    """

    package_root = _expand_root(PACKAGE_ROOT_TEMPLATE, artifact_root, "package root")
    compiled_root = _expand_root(
        COMPILED_ROOT_TEMPLATE, artifact_root, "compiled output root"
    )
    targets = {target["config_id"]: target for target in plan["targets"]}
    graphs = {
        (graph["variant_id"], graph["graph_kind"]): graph for graph in plan["graphs"]
    }
    specs_by_graph: dict[tuple[str, str], Mapping[str, Any]] = {}
    root = _repository_root()
    for variant_id in VARIANT_IDS:
        record = _load_json(
            root / PACKAGE_RECORD_DIRECTORY / f"{variant_id}.json", "package record"
        )
        for graph_kind in GRAPH_KINDS:
            package_graph = _package_graph(record, graph_kind)
            compile_request = _mapping(
                package_graph.get("compile_request"), "package record compile_request"
            )
            specs, _ = _input_specs(compile_request.get("input_specs"))
            specs_by_graph[(variant_id, graph_kind)] = specs

    results: list[dict[str, Any]] = []
    for entry in plan["plan"]:
        target = targets[entry["target"]]
        graph = graphs[(entry["variant_id"], entry["graph_kind"])]
        request = compile_request_for(
            target,
            graph,
            specs_by_graph[(entry["variant_id"], entry["graph_kind"])],
            package_root=package_root,
            compiled_root=compiled_root,
        )
        path = _private_output_path(
            str(
                request_directory
                / entry["target"]
                / entry["variant_id"]
                / f"{entry['graph_kind']}-compile-request.json"
            ),
            "compile request",
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except OSError:
            raise WorkbenchPlanError("compile request could not be written") from None
        preflight = preflight_compile_request(path)
        expected = entry["stages"]["compile"]["request_id"]
        if preflight["request_id"] != expected:
            raise WorkbenchPlanError(
                "the T30 preflight derived a different request id than the "
                "plan did for the same request"
            )
        results.append(
            {
                "target": entry["target"],
                "variant_id": entry["variant_id"],
                "graph_kind": entry["graph_kind"],
                "request_id": preflight["request_id"],
                "service_contacted": preflight["service_contacted"],
                "job_submitted": preflight["job_submitted"],
            }
        )
    return results


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


def _client_probe() -> dict[str, Any]:
    """Observe whether the optional client exists, without importing it."""

    try:
        spec = importlib.util.find_spec("qai_hub")
    except (ImportError, ValueError):
        spec = None
    return {
        "module": "qai_hub",
        "importable": spec is not None,
        "method": "importlib.util.find_spec",
        "module_imported": "qai_hub" in sys.modules,
        "note": (
            "find_spec locates a module without executing it. This planner "
            "never imports the client under any code path."
        ),
    }


def _observed_preflight_flag(preflight: Sequence[Mapping[str, Any]], key: str) -> bool:
    """Fold the per-request preflight observations into one claim.

    The stamped value must come from what the preflight actually observed, so a
    request that reports ``True`` cannot be flattened into a ``False`` literal.
    A missing key is an error rather than a silent ``False``: an unmeasured
    field may not be published as a negative claim.
    """

    observed = False
    for entry in preflight:
        if key not in entry:
            raise WorkbenchPlanError(
                f"a preflight observation does not report {key!r}; the run "
                "observation may not claim what was not measured"
            )
        observed = observed or bool(entry[key])
    return observed


def build_record(
    *,
    record_path: Path,
    paths: Mapping[str, Any],
    preflight: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the plan, attach the dated observation, and write the record."""

    plan = build_plan(
        target_paths=paths["targets"],
        package_paths=paths["packages"],
        inspection_paths=paths["inspections"],
        parity_paths=paths["parity"],
        repository_root=paths["repository_root"],
    )
    record = dict(plan)
    record["run_observation"] = {
        "created_at_utc": _observed_at(),
        "jobs_submitted": 0,
        "service_contacted": False,
        "client_probe": _client_probe(),
        "preflight": (
            {"mode": "not_run", "reason": "not requested on this run"}
            if preflight is None
            else {
                "mode": "ran",
                "requests_validated": len(preflight),
                "all_request_ids_matched_the_plan": True,
                "service_contacted": _observed_preflight_flag(
                    preflight, "service_contacted"
                ),
                "job_submitted": _observed_preflight_flag(preflight, "job_submitted"),
            }
        ),
        "note": (
            "Excluded from the --check comparison on purpose: it is a dated "
            "observation of one machine, not a contract. The plan above is "
            "the part that must re-derive."
        ),
    }
    _assert_public_safe(record, "workbench run plan record")
    payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_bytes(payload)
    except OSError:
        raise WorkbenchPlanError("run plan record could not be written") from None
    return record


def check_record(
    *,
    record_path: Path,
    paths: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-derive the plan and fail on any drift from the committed record."""

    expected = build_plan(
        target_paths=paths["targets"],
        package_paths=paths["packages"],
        inspection_paths=paths["inspections"],
        parity_paths=paths["parity"],
        repository_root=paths["repository_root"],
    )
    committed = dict(_load_json(record_path, "run plan record"))
    observation = committed.pop("run_observation", None)
    difference = _first_difference(expected, committed)
    if difference is not None:
        raise WorkbenchPlanError(
            f"run plan record no longer matches its committed inputs at {difference}"
        )
    if not isinstance(observation, Mapping):
        raise WorkbenchPlanError("run plan record has no run_observation block")
    if (
        observation.get("jobs_submitted") != 0
        or observation.get("service_contacted") is not False
    ):
        raise WorkbenchPlanError(
            "run plan record claims a submitted job or a contacted service; "
            "a real job belongs in a stage manifest, not in a plan"
        )
    return expected


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _summary(mode: str, record_path: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    summary = plan["summary"]
    return {
        "mode": mode,
        "status": "ok",
        "record": _repository_label(record_path),
        "targets": summary["targets"],
        "plan_entries": summary["plan_entries"],
        "stages_ready": summary["stages_ready"],
        "stages_pending_predecessor": summary["stages_pending_predecessor"],
        "first_submission": summary["first_submission"],
        "jobs_submitted": 0,
        "service_contacted": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description=(
            "Build the T31 three-target Qualcomm AI Hub run plan offline. "
            "Submits nothing and imports no Qualcomm client."
        )
    )
    parser.add_argument("--record", type=Path, default=None)
    parser.add_argument("--targets-dir", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--request-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        paths = default_paths()
        if args.targets_dir is not None:
            paths = dict(paths)
            paths["targets"] = [args.targets_dir / name for name in TARGET_CONFIG_FILES]
        record_path = args.record if args.record is not None else paths["record"]
        if args.check:
            if args.preflight:
                raise WorkbenchPlanError(
                    "--preflight re-derives the plan and is not a check mode"
                )
            plan = check_record(record_path=record_path, paths=paths)
            summary = _summary("check", record_path, plan)
        else:
            preflight = None
            if args.preflight:
                plan = build_plan(
                    target_paths=paths["targets"],
                    package_paths=paths["packages"],
                    inspection_paths=paths["inspections"],
                    parity_paths=paths["parity"],
                    repository_root=paths["repository_root"],
                )
                request_directory = (
                    args.request_dir
                    if args.request_dir is not None
                    else paths["repository_root"] / REQUEST_DIRECTORY
                )
                preflight = preflight_plan(
                    plan,
                    artifact_root=resolve_artifact_root(args.artifact_root),
                    request_directory=request_directory,
                )
            record = build_record(
                record_path=record_path, paths=paths, preflight=preflight
            )
            summary = _summary("build", record_path, record)
            if preflight is not None:
                summary["preflighted_requests"] = len(preflight)
    except (AiHubAdapterError, QnnPackagingError) as exc:
        print(f"workbench run plan failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0
