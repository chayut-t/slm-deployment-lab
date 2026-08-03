"""Build the T22 QNN candidate graphs and their committed manifests.

The tool reads a committed T20 manifest, re-hashes the reference graph and its
external-data sidecar against that manifest before touching either, applies the
ordered transformation catalogue at ``configs/graph/qnn-transforms-v1.json``,
writes the candidate beneath ``${SLM_LAB_ARTIFACT_ROOT}/onnx/qnn-candidate/T22``
with its own sidecar, and writes the committed manifest at
``results/manifests/qnn/S<context>.json``.

Structural before and after come from the committed T21 rule engine used as a
library -- :func:`slm_lab.graph.inspection.load_risk_rules` and
:func:`slm_lab.graph.inspection.inspect_graph` -- applied to the reference bytes
and to the candidate bytes. The full findings go to
``results/manifests/qnn/inspection/S<context>.json``; the manifest carries the
compact delta. Nothing is written to ``results/graph/``, which T21 owns.

The rejected pass ``X-ORT-CPU-OFFLINE-OPTIMIZATION`` is measured here rather
than quoted: the tool builds the ONNX Runtime optimized graph into a scratch
directory under the artifact root, reads it back with
:mod:`slm_lab.graph.onnx_reader`, records the operator-histogram delta, the
added opset imports, and the byte sizes, and then deletes the scratch output.

``verification.ort_cpu_parity`` is derived from the committed parity record at
``results/manifests/qnn/parity/S<context>-ort-cpu.json`` when one exists, and is
an explicit ``not_measured`` with a reason when it does not. That record is
produced by the T21 runner, :mod:`slm_lab.backends.onnx_cpu`, never by this
tool: reusing that runner is what makes the candidate's parity directly
comparable to the reference stage's, and reading its output rather than
restating it is what keeps this tool free of a verdict it did not measure. The
record is adopted only if it measured the candidate digests this manifest
records; otherwise it is named as a ``stale_record`` and no verdict is taken
from it.

Nothing in a manifest this tool writes is a compiler result. Every count is a
count of graph structure, every byte size is a file on disk, and anything that
could not be measured is written as an explicit ``not_measured`` with a reason.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import time
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from slm_lab.graph.inspection import (
    DEFAULT_RULES_PATH,
    GraphInspection,
    GraphInspectionError,
    RiskRule,
    inspect_graph,
    load_risk_rules,
    resolve_artifact_root,
)
from slm_lab.graph.onnx_reader import GraphSummary, OnnxReadError, read_onnx_model
from slm_lab.graph.qnn.transforms import (
    DEFAULT_CATALOGUE_PATH,
    QnnTransformError,
    TransformPass,
    applied_passes,
    assert_boundary_preserved,
    assert_cache_write_preserved,
    constant_to_initializer,
    dead_node_elimination,
    externalize_large_tensors,
    infer_value_info,
    load_transform_catalogue,
    rejected_passes,
    static_shape_fold,
    stamp_candidate_provenance,
    write_candidate,
)


SCHEMA_VERSION = 1
TASK_ID = "T22"
STAGE = "qnn_candidate"
MODULE_NAME = "slm_lab.graph.qnn.build"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST_DIRECTORY = PROJECT_ROOT / "results/manifests/onnx"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "results/manifests/qnn"
DEFAULT_INSPECTION_DIRECTORY = DEFAULT_OUTPUT_DIRECTORY / "inspection"
DEFAULT_PARITY_DIRECTORY = DEFAULT_OUTPUT_DIRECTORY / "parity"
ARTIFACT_ROOT_TOKEN = "${SLM_LAB_ARTIFACT_ROOT}"
CANDIDATE_SUBDIRECTORY = "onnx/qnn-candidate/T22"
CANDIDATE_ROOT_TEMPLATE = f"{ARTIFACT_ROOT_TOKEN}/{CANDIDATE_SUBDIRECTORY}"
SCRATCH_DIRECTORY_NAME = ".ort-probe-scratch"
GRAPH_KINDS = ("prefill", "decode")

# The parity record this tool will read, if one exists. The name matches the
# T21 runner's own convention, and the record is produced by that runner --
# this module never executes a graph.
PARITY_RUNNER = "slm_lab.backends.onnx_cpu"
PARITY_RECORD_KIND = "t21_ort_cpu_parity"
PARITY_COMMAND = (
    "SLM_LAB_ARTIFACT_ROOT=<artifact-root> HF_HOME=<local-hf-cache> "
    "TRANSFORMERS_OFFLINE=1 PYTHONPATH=src <parity-env-python> -m "
    f"{PARITY_RUNNER} --manifest results/manifests/qnn/S<context>.json "
    "--steps 4 --reference torch --output "
    "results/manifests/qnn/parity/S<context>-ort-cpu.json"
)

# The structural reader defaults to a 256 MiB ceiling so that a multi-gigabyte
# external-data sidecar can never be parsed as a graph. The ONNX Runtime probe
# has to raise it deliberately, because the graph that probe writes is exactly
# the pathology being measured: the offline optimizer discards the external-data
# layout and inlines every weight back into the protobuf.
ORT_PROBE_MAX_BYTES = 4 * 1024 * 1024 * 1024

CLAIM_BOUNDARY: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "establishes": (
            "reference_graph_and_sidecar_sha256_match_the_committed_T20_manifest",
            "candidate_graph_was_produced_by_the_committed_transform_catalogue",
            "onnx_checker_accepted_the_candidate_graph",
            "candidate_public_boundary_is_identical_to_the_reference_boundary",
            "T12_static_cache_write_survives_in_all_56_cache_outputs",
            "structural_before_and_after_scored_by_the_committed_T21_rule_engine",
            "measured_structural_cost_of_the_rejected_onnxruntime_offline_pass",
        ),
        "does_not_establish": (
            "compiler_acceptance",
            "operator_support_by_any_vendor_toolchain",
            "accelerator_placement",
            "onnxruntime_numerical_parity_of_the_candidate",
            "latency_or_memory_performance",
            "that_a_reduced_finding_count_makes_the_graph_convertible",
        ),
    }
)


#: What a measured, passing parity record adds to the manifest's claim
#: boundary, and what it still does not license. Kept next to
#: :data:`CLAIM_BOUNDARY` so the two are read together.
PARITY_CLAIM_MEASURED = "candidate_was_executed_on_the_onnxruntime_cpu_provider"
PARITY_CLAIM_PASSED = (
    "candidate_logit_and_static_cache_parity_held_on_every_recorded_step_"
    "under_the_T21_protocol_and_tolerance"
)
PARITY_CLAIM_FAILED = (
    "candidate_ort_cpu_parity_was_measured_and_failed_on_at_least_one_step"
)
PARITY_CLAIM_LIMIT = (
    "candidate_parity_beyond_the_recorded_steps_of_one_frozen_workload_on_the_"
    "cpu_execution_provider"
)
PARITY_NOT_ESTABLISHED = "onnxruntime_numerical_parity_of_the_candidate"


class QnnBuildError(ValueError):
    """A build request, source artifact, or committed manifest is invalid."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _sha256_file(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def toolchain_record() -> dict[str, str]:
    """Read the exact runtime versions from the running interpreter."""

    return {
        "python": platform.python_version(),
        "onnx": _package_version("onnx"),
        "onnxruntime": _package_version("onnxruntime"),
        "numpy": _package_version("numpy"),
    }


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QnnBuildError(f"{label} must be a non-empty string")
    return value


def _expand_artifact_root(template: Any, artifact_root: Path) -> Path:
    if not isinstance(template, str) or not template:
        raise QnnBuildError("manifest artifacts.root must be a non-empty string")
    expanded = template.replace(ARTIFACT_ROOT_TOKEN, artifact_root.as_posix())
    directory = Path(expanded)
    if not directory.is_absolute():
        raise QnnBuildError(
            f"manifest artifacts.root did not resolve to an absolute path: {expanded}"
        )
    return directory


def _safe_relative(directory: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise QnnBuildError("manifest relative_path must be a non-empty string")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise QnnBuildError(f"unsafe manifest relative_path {relative_path!r}")
    return directory.joinpath(*pure.parts)


def _tensor_records(values: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": value.name,
            "dtype": value.dtype,
            "shape": list(value.shape.dims) if value.shape is not None else [],
        }
        for value in values
    ]


def _inline_byte_record(summary: GraphSummary) -> dict[str, int]:
    initializers = sum(
        initializer.inline_bytes
        for initializer in summary.initializers
        if not initializer.external
    )
    attributes = sum(
        attribute.tensor.inline_bytes
        for node in summary.nodes
        for attribute in node.attributes
        if attribute.tensor is not None and not attribute.tensor.external
    )
    return {
        "initializers": initializers,
        "node_attributes": attributes,
        "total": initializers + attributes,
    }


def _histogram_delta(
    before: Mapping[str, int], after: Mapping[str, int]
) -> dict[str, dict[str, int]]:
    return {
        op_type: {
            "before": before.get(op_type, 0),
            "after": after.get(op_type, 0),
        }
        for op_type in sorted(set(before) | set(after))
        if before.get(op_type, 0) != after.get(op_type, 0)
    }


# --------------------------------------------------------------------------
# Candidate construction
# --------------------------------------------------------------------------


def _pass_by_id(passes: Sequence[TransformPass], pass_id: str) -> TransformPass:
    for entry in passes:
        if entry.id == pass_id:
            return entry
    raise QnnBuildError(f"transform catalogue has no pass {pass_id!r}")


def _int_parameter(entry: TransformPass, name: str) -> int:
    value = entry.parameters.get(name)
    if type(value) is not int or value < 0:
        raise QnnBuildError(
            f"{entry.id} parameter {name!r} must be a non-negative integer"
        )
    return value


def build_candidate_graph(
    *,
    source_path: Path,
    destination: Path,
    passes: Sequence[TransformPass],
    catalogue_id: str,
    catalogue_sha256: str,
    variant_id: str,
    graph_kind: str,
    source_relative_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Apply the applied passes in order and write the candidate to disk.

    Returns one effect record per applied pass, keyed by pass id.
    """

    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise QnnBuildError(
            "the onnx package is required to build a candidate graph"
        ) from exc

    model = onnx.load_model(str(source_path), load_external_data=False)
    effects: dict[str, Any] = {}

    effects["X-CONSTANT-TO-INITIALIZER"] = constant_to_initializer(model)

    fold = _pass_by_id(passes, "X-STATIC-SHAPE-FOLD")
    allowed_ops = fold.parameters.get("allowed_ops")
    if not isinstance(allowed_ops, list) or not allowed_ops:
        raise QnnBuildError("X-STATIC-SHAPE-FOLD requires a non-empty allowed_ops")
    effects["X-STATIC-SHAPE-FOLD"] = static_shape_fold(
        model,
        allowed_ops=allowed_ops,
        max_input_bytes=_int_parameter(fold, "max_input_bytes"),
        max_output_bytes=_int_parameter(fold, "max_output_bytes"),
    )

    effects["X-DEAD-NODE-ELIMINATION"] = dead_node_elimination(model)

    externalize = _pass_by_id(passes, "X-EXTERNALIZE-LARGE-TENSORS")
    threshold = _int_parameter(externalize, "size_threshold_bytes")
    location = f"{destination.name}.data"
    effects["X-EXTERNALIZE-LARGE-TENSORS"] = externalize_large_tensors(
        model,
        size_threshold_bytes=threshold,
        location=location,
    )

    infer = _pass_by_id(passes, "X-INFER-VALUE-INFO")
    effects["X-INFER-VALUE-INFO"] = infer_value_info(
        model,
        check_type=bool(infer.parameters.get("check_type", False)),
        strict_mode=bool(infer.parameters.get("strict_mode", False)),
        data_prop=bool(infer.parameters.get("data_prop", True)),
    )

    stamp = _pass_by_id(passes, "X-STAMP-CANDIDATE-PROVENANCE")
    prefix = str(stamp.parameters.get("metadata_prefix", "slm_lab."))
    metadata = {
        f"{prefix}task_id": TASK_ID,
        f"{prefix}stage": STAGE,
        f"{prefix}variant_id": variant_id,
        f"{prefix}graph_kind": graph_kind,
        f"{prefix}source_relative_path": source_relative_path,
        f"{prefix}source_sha256": source_sha256,
        f"{prefix}transform_catalogue_id": catalogue_id,
        f"{prefix}transform_catalogue_sha256": catalogue_sha256,
        f"{prefix}applied_passes": ",".join(
            entry.id for entry in applied_passes(passes)
        ),
    }
    effects["X-STAMP-CANDIDATE-PROVENANCE"] = stamp_candidate_provenance(
        model,
        producer_name=str(stamp.parameters.get("producer_name", "slm_lab.graph.qnn")),
        producer_version=catalogue_id,
        metadata=metadata,
    )

    write_candidate(
        model,
        destination,
        source_directory=source_path.parent,
        size_threshold_bytes=threshold,
        location=location,
    )
    del model
    return effects


def check_candidate(destination: Path) -> dict[str, Any]:
    """Run ``onnx.checker.check_model`` on a written candidate."""

    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise QnnBuildError("the onnx package is required by onnx.checker") from exc
    try:
        onnx.checker.check_model(str(destination))
    except Exception as exc:  # noqa: BLE001 - the checker raises several types
        return {
            "status": "failed",
            "checker": "onnx.checker.check_model",
            "full_check": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "status": "passed",
        "checker": "onnx.checker.check_model",
        "full_check": False,
    }


# --------------------------------------------------------------------------
# The rejected pass, measured rather than quoted
# --------------------------------------------------------------------------


def measure_ort_rejection(
    *,
    reference_path: Path,
    reference_summary: GraphSummary,
    reference_external_bytes: int,
    scratch_directory: Path,
    graph_optimization_level: str,
    execution_provider: str,
) -> dict[str, Any]:
    """Build the ONNX Runtime optimized graph and record what it cost.

    The optimized graph is written into ``scratch_directory``, read back with
    the repository's own structural reader, and then deleted. Every number here
    is produced by this function; none of it is copied from a planning probe.
    """

    record: dict[str, Any] = {
        "graph_optimization_level": graph_optimization_level,
        "execution_provider": execution_provider,
        "onnxruntime_version": _package_version("onnxruntime"),
    }
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - environment dependent
        record["status"] = "not_measured"
        record["reason"] = f"onnxruntime is not installed: {exc}"
        return record

    level = getattr(ort.GraphOptimizationLevel, graph_optimization_level, None)
    if level is None:
        record["status"] = "not_measured"
        record["reason"] = (
            f"onnxruntime {ort.__version__} has no graph optimization level "
            f"{graph_optimization_level!r}"
        )
        return record

    shutil.rmtree(scratch_directory, ignore_errors=True)
    scratch_directory.mkdir(parents=True, exist_ok=True)
    optimized_path = scratch_directory / "ort-optimized.onnx"
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.optimized_model_filepath = str(optimized_path)
    started = time.monotonic()
    try:
        ort.InferenceSession(
            str(reference_path), options, providers=[execution_provider]
        )
    except Exception as exc:  # noqa: BLE001 - a session failure is evidence too
        shutil.rmtree(scratch_directory, ignore_errors=True)
        record["status"] = "not_measured"
        record["reason"] = (
            "onnxruntime refused to build a session over the reference graph: "
            f"{type(exc).__name__}: {exc}"
        )
        return record
    elapsed = time.monotonic() - started

    if not optimized_path.is_file():
        shutil.rmtree(scratch_directory, ignore_errors=True)
        record["status"] = "not_measured"
        record["reason"] = (
            "the session was built but onnxruntime wrote no optimized model to "
            "optimized_model_filepath"
        )
        return record

    sidecars = sorted(
        path
        for path in scratch_directory.iterdir()
        if path.is_file() and path != optimized_path
    )
    optimized_bytes = optimized_path.stat().st_size
    sidecar_records = [
        {"name": path.name, "size_bytes": path.stat().st_size} for path in sidecars
    ]
    try:
        optimized = read_onnx_model(optimized_path, max_bytes=ORT_PROBE_MAX_BYTES)
    except OnnxReadError as exc:
        shutil.rmtree(scratch_directory, ignore_errors=True)
        record["status"] = "not_measured"
        record["reason"] = f"the optimized graph could not be read back: {exc}"
        return record

    before_domains = {domain for domain, _ in reference_summary.opset_imports}
    after_imports = list(optimized.opset_imports)
    used_domains = {node.domain or "" for node in optimized.nodes}
    added = [domain for domain, _ in after_imports if domain not in before_domains]
    inline_before = sum(
        initializer.inline_bytes
        for initializer in reference_summary.initializers
        if not initializer.external
    )
    inline_after = sum(
        initializer.inline_bytes
        for initializer in optimized.initializers
        if not initializer.external
    )
    record.update(
        {
            "status": "measured",
            "reference_graph": reference_path.name,
            "node_count": {
                "before": len(reference_summary.nodes),
                "after": len(optimized.nodes),
            },
            "operator_type_count": {
                "before": len(reference_summary.op_histogram),
                "after": len(optimized.op_histogram),
            },
            "operator_histogram_delta": _histogram_delta(
                reference_summary.op_histogram, optimized.op_histogram
            ),
            "opset_imports": {
                "before": [
                    [domain, version]
                    for domain, version in reference_summary.opset_imports
                ],
                "after": [[domain, version] for domain, version in after_imports],
            },
            "added_opset_domains": sorted(added),
            "added_opset_domains_used_by_no_node": sorted(
                domain for domain in added if domain not in used_domains
            ),
            "protobuf_bytes": {
                "before": reference_path.stat().st_size,
                "after": optimized_bytes,
            },
            "external_data_bytes": {
                "before": reference_external_bytes,
                "after": sum(item["size_bytes"] for item in sidecar_records),
            },
            "external_data_files": sidecar_records,
            "initializer_count": {
                "before": len(reference_summary.initializers),
                "after": len(optimized.initializers),
            },
            "external_initializer_count": {
                "before": sum(
                    1
                    for initializer in reference_summary.initializers
                    if initializer.external
                ),
                "after": sum(
                    1 for initializer in optimized.initializers if initializer.external
                ),
            },
            "inline_initializer_bytes": {
                "before": inline_before,
                "after": inline_after,
            },
            "producer": {
                "before": " ".join(
                    part
                    for part in (
                        reference_summary.producer_name,
                        reference_summary.producer_version,
                    )
                    if part
                ),
                "after": " ".join(
                    part
                    for part in (optimized.producer_name, optimized.producer_version)
                    if part
                ),
            },
            "scratch_output_deleted": True,
        }
    )
    shutil.rmtree(scratch_directory, ignore_errors=True)
    record["note"] = (
        "wall-clock duration is deliberately not recorded here so that --check "
        "stays a genuine drift check; the build tool prints the observed "
        "duration on stderr instead. Every other number in this record was "
        "produced by this run, not copied from a planning probe."
    )
    print(
        f"  ort probe {reference_path.parent.name}/{reference_path.name}: "
        f"{elapsed:.1f}s, {len(reference_summary.nodes)} -> "
        f"{len(optimized.nodes)} nodes",
        file=sys.stderr,
    )
    return record


# --------------------------------------------------------------------------
# The ORT CPU parity record, read rather than produced
# --------------------------------------------------------------------------


def parity_record_path(parity_directory: Path, variant_id: str) -> Path:
    """The conventional location of one variant's candidate parity record."""

    return parity_directory / f"{variant_id}-ort-cpu.json"


def _metric_extremum(
    steps: Sequence[Mapping[str, Any]], key: str, *, largest: bool
) -> Any:
    """Return the worst value of one metric, copied verbatim from the record.

    ``min``/``max`` select one of the floats already parsed from the committed
    JSON rather than computing a new one, so re-deriving the manifest from the
    same record reproduces the same literal and ``--check`` stays exact.
    """

    values = [
        step["metrics"][key]
        for step in steps
        if isinstance(step.get("metrics"), Mapping) and key in step["metrics"]
    ]
    if not values:
        return None
    return max(values) if largest else min(values)


def _parity_logit_summary(
    steps: Sequence[Mapping[str, Any]], graph_kind: str
) -> dict[str, Any]:
    """Summarize one graph kind's recorded steps without hiding a failure."""

    selected = [
        step
        for step in steps
        if isinstance(step, Mapping) and step.get("graph_kind") == graph_kind
    ]
    scored = [step for step in selected if isinstance(step.get("metrics"), Mapping)]
    return {
        "steps": len(selected),
        "steps_scored": len(scored),
        "steps_passed": sum(
            1 for step in scored if step["metrics"].get("passed") is True
        ),
        "steps_with_non_finite_candidate_logits": sum(
            1 for step in selected if not isinstance(step.get("metrics"), Mapping)
        ),
        "top1_agreements": sum(
            1 for step in scored if step["metrics"].get("top1_agreement") is True
        ),
        "cosine_similarity_min": _metric_extremum(
            scored, "cosine_similarity", largest=False
        ),
        "max_absolute_error_max": _metric_extremum(
            scored, "max_absolute_error", largest=True
        ),
        "max_protected_relative_error_max": _metric_extremum(
            scored, "max_protected_relative_error", largest=True
        ),
        "mean_absolute_error_max": _metric_extremum(
            scored, "mean_absolute_error", largest=True
        ),
        "top5_overlap_min": _metric_extremum(scored, "top5_overlap", largest=False),
    }


def _record_graph_digests(record: Mapping[str, Any]) -> dict[str, Any]:
    digests = record.get("graph_digests")
    if not isinstance(digests, Mapping):
        return {}
    result: dict[str, Any] = {}
    for graph_kind in GRAPH_KINDS:
        entry = digests.get(graph_kind)
        if isinstance(entry, Mapping):
            result[graph_kind] = {
                "relative_path": entry.get("relative_path"),
                "sha256": entry.get("sha256"),
            }
    return result


def _candidate_graph_digests(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        graph_kind: {
            "relative_path": artifacts[graph_kind]["relative_path"],
            "sha256": artifacts[graph_kind]["sha256"],
        }
        for graph_kind in GRAPH_KINDS
        if isinstance(artifacts.get(graph_kind), Mapping)
    }


def ort_cpu_parity_record(
    *,
    variant_id: str,
    parity_directory: Path,
    artifact_records: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive ``verification.ort_cpu_parity`` from the committed parity record.

    This build tool rewrites and measures graphs; it never executes one. The
    parity record is produced separately by the T21 runner
    (:data:`PARITY_COMMAND`) and is *read* here, so that the manifest points at
    a real measurement of the exact bytes it describes instead of asserting a
    verdict of its own.

    Four outcomes, all deterministic functions of committed files:

    ``measured``
        A record of kind :data:`PARITY_RECORD_KIND` exists at the conventional
        path and its ``graph_digests`` are the digests this manifest records
        for the candidate. Its verdict, tier, tolerance and per-graph-kind
        metric extremes are carried into the manifest, and a failing verdict is
        carried in exactly as faithfully as a passing one.
    ``stale_record``
        A record exists but measured different bytes. It is named, its digests
        and the candidate's are both recorded, and no verdict is adopted --
        a verdict about other bytes is not evidence about these.
    ``not_measured``
        No record exists, or the file at that path is not a readable parity
        record of the expected kind. The reason names the command that would
        produce one.
    """

    path = parity_record_path(parity_directory, variant_id)
    relative = _repository_relative(path)
    if not path.is_file():
        return {
            "status": "not_measured",
            "runner": PARITY_RUNNER,
            "expected_record_path": relative,
            "reason": (
                "this build tool rewrites and measures graphs; it never "
                "executes one. No candidate ONNX Runtime CPU parity record "
                f"exists at {relative}. Produce one with the T21 runner under "
                f"the T23 tolerance: {PARITY_COMMAND}"
            ),
        }

    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        record = None
    if not isinstance(record, Mapping):
        return {
            "status": "not_measured",
            "runner": PARITY_RUNNER,
            "expected_record_path": relative,
            "reason": (
                f"the file at {relative} is not a readable JSON object, so it "
                "is not a parity record and no verdict was read from it"
            ),
        }
    record_kind = record.get("record_kind")
    if record_kind != PARITY_RECORD_KIND:
        return {
            "status": "not_measured",
            "runner": PARITY_RUNNER,
            "expected_record_path": relative,
            "reason": (
                f"the record at {relative} has record_kind {record_kind!r}, "
                f"not {PARITY_RECORD_KIND!r}. A diagnostic run is not a parity "
                "measurement and no verdict was read from it"
            ),
        }

    measured_digests = _record_graph_digests(record)
    candidate_digests = _candidate_graph_digests(artifact_records)
    if measured_digests != candidate_digests:
        return {
            "status": "stale_record",
            "runner": PARITY_RUNNER,
            "record_path": relative,
            "record_sha256": _sha256_file(path),
            "measured_graph_digests": measured_digests,
            "candidate_graph_digests": candidate_digests,
            "reason": (
                f"the record at {relative} measured graph bytes that are not "
                "the candidate bytes this manifest describes, so its verdict "
                "is not evidence about this candidate. Re-run the T21 runner "
                f"against this manifest: {PARITY_COMMAND}"
            ),
        }

    steps = record.get("steps")
    steps = steps if isinstance(steps, list) else []
    cache_report = record.get("cache_report")
    return {
        "status": "measured",
        "runner": PARITY_RUNNER,
        "record_path": relative,
        "record_sha256": _sha256_file(path),
        "record_kind": record_kind,
        "record_task_id": record.get("task_id"),
        "record_schema_version": record.get("schema_version"),
        "evidence_sha256": record.get("evidence_sha256"),
        "evidence_tier": record.get("evidence_tier"),
        "passed": record.get("passed"),
        "failure_kinds": list(record.get("failure_kinds") or ()),
        "steps_requested": record.get("steps_requested"),
        "steps_recorded": len(steps),
        "cache_report_passed": (
            cache_report.get("passed") if isinstance(cache_report, Mapping) else None
        ),
        "graph_digests_match_candidate": True,
        "reference_provenance": record.get("reference_provenance"),
        "runtime": record.get("runtime"),
        "tolerance": record.get("tolerance"),
        "logit_metrics": {
            graph_kind: _parity_logit_summary(steps, graph_kind)
            for graph_kind in GRAPH_KINDS
        },
        "note": (
            "measured by the T21 runner, not by this build tool, and read back "
            f"from {relative}. record_task_id is T21 because the runner stamps "
            "its own task id into a fixed schema field; the record's "
            "graph_digests are what identify it as a measurement of this "
            "candidate. The protocol, reference, step count and tolerance are "
            "the ones the reference-stage records in results/graph/parity/ "
            "use, which is what makes the two directly comparable."
        ),
    }


def claim_boundary_for(parity: Mapping[str, Any]) -> dict[str, list[str]]:
    """Adjust the claim boundary for what the parity record actually shows.

    A manifest that carries a passing, on-these-bytes parity measurement may no
    longer say it does not establish ONNX Runtime numerical parity, and one
    that carries a failing measurement most certainly still must. The
    adjustment is a pure function of the derived block, so ``--check``
    re-derives it byte for byte.
    """

    establishes = list(CLAIM_BOUNDARY["establishes"])
    does_not = list(CLAIM_BOUNDARY["does_not_establish"])
    if parity.get("status") != "measured":
        return {"establishes": establishes, "does_not_establish": does_not}
    establishes.append(PARITY_CLAIM_MEASURED)
    if parity.get("passed") is True:
        establishes.append(PARITY_CLAIM_PASSED)
        does_not = [item for item in does_not if item != PARITY_NOT_ESTABLISHED]
    else:
        establishes.append(PARITY_CLAIM_FAILED)
    does_not.append(PARITY_CLAIM_LIMIT)
    return {"establishes": establishes, "does_not_establish": does_not}


# --------------------------------------------------------------------------
# Structural delta
# --------------------------------------------------------------------------


def structural_delta(
    *,
    reference_inspection: GraphInspection,
    candidate_inspection: GraphInspection,
    reference_summary: GraphSummary,
    candidate_summary: GraphSummary,
    reference_path: Path,
    candidate_path: Path,
    reference_external_bytes: int,
    candidate_external_bytes: int,
) -> dict[str, Any]:
    """Compact before/after produced by the T21 rule engine used as a library."""

    reference_findings = {
        finding.rule_id: finding.count for finding in reference_inspection.findings
    }
    candidate_findings = {
        finding.rule_id: finding.count for finding in candidate_inspection.findings
    }
    rule_ids = sorted(set(reference_findings) | set(candidate_findings))
    return {
        "node_count": {
            "before": reference_inspection.node_count,
            "after": candidate_inspection.node_count,
        },
        "operator_type_count": {
            "before": len(reference_inspection.op_histogram),
            "after": len(candidate_inspection.op_histogram),
        },
        "operator_histogram_delta": _histogram_delta(
            reference_inspection.op_histogram, candidate_inspection.op_histogram
        ),
        "input_count": {
            "before": reference_inspection.input_count,
            "after": candidate_inspection.input_count,
        },
        "output_count": {
            "before": reference_inspection.output_count,
            "after": candidate_inspection.output_count,
        },
        "initializer_count": {
            "before": reference_inspection.initializer_count,
            "after": candidate_inspection.initializer_count,
        },
        "external_initializer_count": {
            "before": reference_inspection.external_initializer_count,
            "after": candidate_inspection.external_initializer_count,
        },
        "largest_inline_initializer_bytes": {
            "before": reference_inspection.largest_inline_initializer_bytes,
            "after": candidate_inspection.largest_inline_initializer_bytes,
        },
        "inline_bytes": {
            "before": _inline_byte_record(reference_summary),
            "after": _inline_byte_record(candidate_summary),
        },
        "value_info_count": {
            "before": len(reference_summary.value_info),
            "after": len(candidate_summary.value_info),
        },
        "protobuf_bytes": {
            "before": reference_path.stat().st_size,
            "after": candidate_path.stat().st_size,
        },
        "external_data_bytes": {
            "before": reference_external_bytes,
            "after": candidate_external_bytes,
        },
        "highest_severity": {
            "before": reference_inspection.highest_severity,
            "after": candidate_inspection.highest_severity,
        },
        "finding_counts": {
            rule_id: {
                "before": reference_findings.get(rule_id, 0),
                "after": candidate_findings.get(rule_id, 0),
            }
            for rule_id in rule_ids
        },
    }


# --------------------------------------------------------------------------
# Per-variant build
# --------------------------------------------------------------------------


def _manifest_graph_record(
    manifest: Mapping[str, Any], graph_kind: str
) -> Mapping[str, Any]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise QnnBuildError("source manifest has no artifacts object")
    record = artifacts.get(graph_kind)
    if not isinstance(record, dict):
        raise QnnBuildError(f"source manifest has no {graph_kind} artifact record")
    return record


def _verify_reference(
    record: Mapping[str, Any],
    directory: Path,
    *,
    label: str,
) -> tuple[Path, dict[str, Any], int]:
    """Re-hash the reference graph and sidecar before anything reads them."""

    expected = record.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise QnnBuildError(f"{label}: source manifest sha256 is not a digest")
    graph_path = _safe_relative(directory, record.get("relative_path"))
    if not graph_path.is_file():
        raise QnnBuildError(f"{label}: reference graph is missing: {graph_path}")
    actual = _sha256_file(graph_path)
    if actual != expected:
        raise QnnBuildError(
            f"{label}: SHA-256 mismatch for {graph_path}; the T20 manifest "
            f"records {expected}, the file is {actual}"
        )
    external_records: list[dict[str, Any]] = []
    external_bytes = 0
    for entry in record.get("external_data", ()):
        if not isinstance(entry, dict):
            raise QnnBuildError(f"{label}: malformed external_data entry")
        sidecar = _safe_relative(graph_path.parent, entry.get("location"))
        if not sidecar.is_file():
            raise QnnBuildError(f"{label}: external data is missing: {sidecar}")
        sidecar_digest = _sha256_file(sidecar)
        if sidecar_digest != entry.get("sha256"):
            raise QnnBuildError(
                f"{label}: SHA-256 mismatch for {sidecar}; the T20 manifest "
                f"records {entry.get('sha256')}, the file is {sidecar_digest}"
            )
        size = sidecar.stat().st_size
        external_bytes += size
        external_records.append(
            {
                "location": str(entry.get("location")),
                "sha256": sidecar_digest,
                "size_bytes": size,
            }
        )
    source_record = {
        "graph_kind": graph_kind_of(record, label),
        "relative_path": str(record.get("relative_path")),
        "sha256": actual,
        "size_bytes": graph_path.stat().st_size,
        "external_data": external_records,
        "sha256_recomputed_from_disk": True,
    }
    return graph_path, source_record, external_bytes


def graph_kind_of(record: Mapping[str, Any], label: str) -> str:
    kind = record.get("graph_kind")
    if kind not in GRAPH_KINDS:
        raise QnnBuildError(f"{label}: unknown graph_kind {kind!r}")
    return str(kind)


def _candidate_external_records(
    destination: Path, summary: GraphSummary
) -> tuple[list[dict[str, Any]], int]:
    locations = sorted(
        {
            initializer.external_location
            for initializer in summary.initializers
            if initializer.external and initializer.external_location
        }
    )
    records: list[dict[str, Any]] = []
    total = 0
    for location in locations:
        sidecar = _safe_relative(destination.parent, location)
        if not sidecar.is_file():
            raise QnnBuildError(f"candidate external data is missing: {sidecar}")
        size = sidecar.stat().st_size
        total += size
        records.append(
            {
                "location": location,
                "sha256": _sha256_file(sidecar),
                "size_bytes": size,
            }
        )
    return records, total


def build_variant(
    manifest_path: Path,
    *,
    passes: Sequence[TransformPass],
    catalogue_id: str,
    catalogue_path: Path,
    rules: Sequence[RiskRule],
    rules_catalogue_id: str,
    rules_path: Path,
    artifact_root: Path,
    location_sample_limit: int,
    inspection_relative_path: str,
    parity_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build both candidate graphs for one variant and return its two payloads."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise QnnBuildError(f"cannot read manifest {manifest_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise QnnBuildError(f"invalid JSON in manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise QnnBuildError(f"manifest {manifest_path} must be a JSON object")

    variant_id = _require_text(manifest.get("variant_id"), "manifest variant_id")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise QnnBuildError(f"manifest {manifest_path} has no artifacts object")
    reference_directory = _expand_artifact_root(artifacts.get("root"), artifact_root)
    candidate_directory = artifact_root / CANDIDATE_SUBDIRECTORY / variant_id
    catalogue_sha256 = _sha256_file(catalogue_path)

    source_records: dict[str, Any] = {
        "manifest_path": _repository_relative(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "root": str(artifacts.get("root")),
    }
    effects: dict[str, dict[str, Any]] = {}
    artifact_records: dict[str, Any] = {"root": CANDIDATE_ROOT_TEMPLATE}
    deltas: dict[str, Any] = {}
    checker: dict[str, Any] = {}
    rejection: dict[str, Any] = {}
    inspections: dict[str, Any] = {}
    contract_checks: dict[str, Any] = {}

    for graph_kind in GRAPH_KINDS:
        label = f"{variant_id} {graph_kind}"
        record = _manifest_graph_record(manifest, graph_kind)
        reference_path, source_record, reference_external_bytes = _verify_reference(
            record, reference_directory, label=label
        )
        source_records[graph_kind] = source_record
        reference_summary = read_onnx_model(reference_path)

        destination = candidate_directory / f"{graph_kind}.onnx"
        print(f"building {label} -> {destination}", file=sys.stderr)
        effects[graph_kind] = build_candidate_graph(
            source_path=reference_path,
            destination=destination,
            passes=passes,
            catalogue_id=catalogue_id,
            catalogue_sha256=catalogue_sha256,
            variant_id=variant_id,
            graph_kind=graph_kind,
            source_relative_path=source_record["relative_path"],
            source_sha256=source_record["sha256"],
        )
        checker[graph_kind] = check_candidate(destination)
        candidate_summary = read_onnx_model(destination)

        assert_boundary_preserved(
            reference_summary,
            candidate_summary,
            graph_kind=graph_kind,
            label=label,
        )
        cache_writes = assert_cache_write_preserved(
            candidate_summary, graph_kind=graph_kind, label=label
        )
        contract_checks[graph_kind] = {
            "boundary_identical_to_reference": True,
            "input_count": len(candidate_summary.inputs),
            "output_count": len(candidate_summary.outputs),
            "cache_writes_preserved": cache_writes,
            "cache_write_operator": (
                "ScatterElements" if graph_kind == "decode" else "Concat"
            ),
        }

        external_records, candidate_external_bytes = _candidate_external_records(
            destination, candidate_summary
        )
        candidate_relative = destination.relative_to(
            artifact_root / CANDIDATE_SUBDIRECTORY
        ).as_posix()
        artifact_records[graph_kind] = {
            "graph_kind": graph_kind,
            "relative_path": candidate_relative,
            "sha256": _sha256_file(destination),
            "size_bytes": destination.stat().st_size,
            "external_data": external_records,
            "input_tensors": _tensor_records(candidate_summary.inputs),
            "output_tensors": _tensor_records(candidate_summary.outputs),
        }

        reference_inspection = inspect_graph(
            reference_summary,
            variant_id=variant_id,
            graph_kind=graph_kind,
            source_relative_path=source_record["relative_path"],
            source_sha256=source_record["sha256"],
            rules=rules,
            catalogue_id=rules_catalogue_id,
            location_sample_limit=location_sample_limit,
        )
        candidate_inspection = inspect_graph(
            candidate_summary,
            variant_id=variant_id,
            graph_kind=graph_kind,
            source_relative_path=candidate_relative,
            source_sha256=artifact_records[graph_kind]["sha256"],
            rules=rules,
            catalogue_id=rules_catalogue_id,
            location_sample_limit=location_sample_limit,
        )
        inspections[graph_kind] = {
            "reference": reference_inspection.as_dict(),
            "candidate": candidate_inspection.as_dict(),
        }
        deltas[graph_kind] = structural_delta(
            reference_inspection=reference_inspection,
            candidate_inspection=candidate_inspection,
            reference_summary=reference_summary,
            candidate_summary=candidate_summary,
            reference_path=reference_path,
            candidate_path=destination,
            reference_external_bytes=reference_external_bytes,
            candidate_external_bytes=candidate_external_bytes,
        )

        rejected = _pass_by_id(passes, "X-ORT-CPU-OFFLINE-OPTIMIZATION")
        rejection[graph_kind] = measure_ort_rejection(
            reference_path=reference_path,
            reference_summary=reference_summary,
            reference_external_bytes=reference_external_bytes,
            scratch_directory=(
                artifact_root
                / CANDIDATE_SUBDIRECTORY
                / SCRATCH_DIRECTORY_NAME
                / variant_id
                / graph_kind
            ),
            graph_optimization_level=str(
                rejected.parameters.get("graph_optimization_level", "ORT_ENABLE_BASIC")
            ),
            execution_provider=str(
                rejected.parameters.get("execution_provider", "CPUExecutionProvider")
            ),
        )

    transformations: list[dict[str, Any]] = []
    for entry in passes:
        payload = {
            "id": entry.id,
            "order": entry.order,
            "applied": entry.applied,
            "title": entry.title,
            "observed_issue": entry.observed_issue,
            "rule_ids": list(entry.addresses),
            "transformation": entry.transformation,
            "parameters": dict(entry.parameters),
        }
        if entry.applied:
            payload["effect"] = {
                graph_kind: effects[graph_kind][entry.id] for graph_kind in GRAPH_KINDS
            }
        else:
            payload["effect"] = {
                graph_kind: {
                    "status": "not_applied",
                    "reason": (
                        "recorded as a rejected transformation; see rejection_evidence"
                    ),
                }
                for graph_kind in GRAPH_KINDS
            }
            payload["rejection_evidence"] = {
                graph_kind: rejection[graph_kind] for graph_kind in GRAPH_KINDS
            }
        transformations.append(payload)

    scratch_root = artifact_root / CANDIDATE_SUBDIRECTORY / SCRATCH_DIRECTORY_NAME
    shutil.rmtree(scratch_root, ignore_errors=True)

    parity = ort_cpu_parity_record(
        variant_id=variant_id,
        parity_directory=parity_directory,
        artifact_records=artifact_records,
    )

    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "stage": STAGE,
        "variant_id": variant_id,
        "context_length": manifest.get("context_length"),
        "cache_capacity": manifest.get("cache_capacity"),
        "opset": manifest.get("opset"),
        "precision": manifest.get("precision"),
        "cache_contract": manifest.get("cache_contract"),
        "source": source_records,
        "transform_catalogue": {
            "path": _repository_relative(catalogue_path),
            "catalogue_id": catalogue_id,
            "sha256": catalogue_sha256,
        },
        "risk_catalogue": {
            "path": _repository_relative(rules_path),
            "catalogue_id": rules_catalogue_id,
            "sha256": _sha256_file(rules_path),
        },
        "toolchain": toolchain_record(),
        "artifacts": artifact_records,
        "transformations": transformations,
        "structural_delta": deltas,
        "contract_preservation": contract_checks,
        "verification": {
            "onnx_checker": checker,
            "graph_inspection": {
                "status": "measured",
                "engine": "slm_lab.graph.inspection",
                "catalogue_id": rules_catalogue_id,
                "report_path": inspection_relative_path,
                "note": (
                    "the reference and the candidate were scored by the same "
                    "committed rule catalogue; the full findings, including "
                    "sampled locations, are in report_path"
                ),
            },
            "ort_cpu_parity": parity,
        },
        "claim_boundary": claim_boundary_for(parity),
        "generated_by": {
            "module": MODULE_NAME,
            "schema_version": SCHEMA_VERSION,
        },
    }

    inspection_payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "stage": STAGE,
        "variant_id": variant_id,
        "catalogue_id": rules_catalogue_id,
        "source_manifest": {
            "path": _repository_relative(manifest_path),
            "sha256": source_records["manifest_sha256"],
        },
        "transform_catalogue": {
            "path": _repository_relative(catalogue_path),
            "catalogue_id": catalogue_id,
            "sha256": catalogue_sha256,
        },
        "generated_by": {
            "module": MODULE_NAME,
            "schema_version": SCHEMA_VERSION,
            "rules_path": _repository_relative(rules_path),
            "rules_sha256": _sha256_file(rules_path),
        },
        "claim_boundary": {key: list(values) for key, values in CLAIM_BOUNDARY.items()},
        "graphs": inspections,
    }
    return manifest_payload, inspection_payload


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _render(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"python -m {MODULE_NAME}",
        description=(
            "Apply the committed QNN transformation catalogue to the hash-"
            "verified T20 reference graphs and write the candidate artifacts "
            "and their committed manifests."
        ),
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=None,
        help="committed T20 manifest to build from; may be repeated",
    )
    parser.add_argument(
        "--all-manifests",
        default=None,
        nargs="?",
        const=str(DEFAULT_MANIFEST_DIRECTORY),
        help=f"directory of S*.json manifests (default {DEFAULT_MANIFEST_DIRECTORY})",
    )
    parser.add_argument(
        "--catalogue",
        default=str(DEFAULT_CATALOGUE_PATH),
        help=f"transformation catalogue path (default {DEFAULT_CATALOGUE_PATH})",
    )
    parser.add_argument(
        "--rules",
        default=str(DEFAULT_RULES_PATH),
        help=f"T21 risk catalogue path (default {DEFAULT_RULES_PATH})",
    )
    parser.add_argument(
        "--artifact-root",
        default=None,
        help=(
            "external artifact root (default $SLM_LAB_ARTIFACT_ROOT, then ./artifacts)"
        ),
    )
    parser.add_argument(
        "--output-directory",
        default=str(DEFAULT_OUTPUT_DIRECTORY),
        help=f"directory for committed manifests (default {DEFAULT_OUTPUT_DIRECTORY})",
    )
    parser.add_argument(
        "--inspection-directory",
        default=str(DEFAULT_INSPECTION_DIRECTORY),
        help=(
            "directory for the full before/after findings "
            f"(default {DEFAULT_INSPECTION_DIRECTORY})"
        ),
    )
    parser.add_argument(
        "--parity-directory",
        default=str(DEFAULT_PARITY_DIRECTORY),
        help=(
            "directory holding the candidate ORT CPU parity records this tool "
            f"reads, never writes (default {DEFAULT_PARITY_DIRECTORY})"
        ),
    )
    parser.add_argument(
        "--location-sample-limit",
        type=int,
        default=8,
        help="maximum sampled locations recorded per finding (default 8)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-derive and fail if a committed manifest or report would change",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _selected_manifests(args: argparse.Namespace) -> list[Path]:
    selected: list[Path] = [Path(value) for value in (args.manifest or ())]
    if args.all_manifests:
        directory = Path(args.all_manifests)
        if not directory.is_dir():
            raise QnnBuildError(f"manifest directory not found: {directory}")
        found = sorted(directory.glob("S*.json"))
        if not found:
            raise QnnBuildError(f"no S*.json manifests under {directory}")
        selected.extend(found)
    if not selected:
        raise QnnBuildError(
            "select at least one manifest with --manifest or --all-manifests"
        )
    unique: dict[str, Path] = {}
    for path in selected:
        if not path.is_file():
            raise QnnBuildError(f"manifest not found: {path}")
        unique.setdefault(str(path.resolve()), path)
    return list(unique.values())


def _emit(destination: Path, text: str, *, check: bool) -> bool:
    """Write ``text`` or, under ``--check``, report whether it would change."""

    if check:
        if not destination.is_file():
            print(f"missing report: {destination}", file=sys.stderr)
            return True
        if destination.read_text(encoding="utf-8") != text:
            print(f"stale report: {destination}", file=sys.stderr)
            return True
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    print(f"wrote {destination}")
    return False


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifests = _selected_manifests(args)
        catalogue_path = Path(args.catalogue)
        catalogue_id, passes = load_transform_catalogue(catalogue_path)
        rules_path = Path(args.rules)
        rules_catalogue_id, rules = load_risk_rules(rules_path)
        known_rule_ids = {rule.id for rule in rules}
        for entry in passes:
            unknown = sorted(set(entry.addresses) - known_rule_ids)
            if unknown:
                raise QnnBuildError(
                    f"{entry.id} addresses rule ids absent from "
                    f"{_repository_relative(rules_path)}: {unknown}"
                )
        if not applied_passes(passes):
            raise QnnBuildError("the transformation catalogue applies no pass")
        if not rejected_passes(passes):
            raise QnnBuildError(
                "the transformation catalogue records no rejected pass; the "
                "measured rejection is part of the deliverable"
            )
        artifact_root = resolve_artifact_root(args.artifact_root)
        output_directory = Path(args.output_directory)
        inspection_directory = Path(args.inspection_directory)
        parity_directory = Path(args.parity_directory)

        changed = False
        for manifest_path in manifests:
            variant_id = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "variant_id"
            )
            inspection_destination = inspection_directory / f"{variant_id}.json"
            manifest_payload, inspection_payload = build_variant(
                manifest_path,
                passes=passes,
                catalogue_id=catalogue_id,
                catalogue_path=catalogue_path,
                rules=rules,
                rules_catalogue_id=rules_catalogue_id,
                rules_path=rules_path,
                artifact_root=artifact_root,
                location_sample_limit=args.location_sample_limit,
                inspection_relative_path=_repository_relative(inspection_destination),
                parity_directory=parity_directory,
            )
            changed |= _emit(
                output_directory / f"{manifest_payload['variant_id']}.json",
                _render(manifest_payload),
                check=args.check,
            )
            changed |= _emit(
                inspection_destination,
                _render(inspection_payload),
                check=args.check,
            )
        if changed:
            return 1
    except (
        QnnBuildError,
        QnnTransformError,
        GraphInspectionError,
        OnnxReadError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
