"""Frozen W8A16/W8A8 candidate specifications and a fail-closed evidence gate.

This module quantizes nothing and its verdict function cannot report a deployed
precision from anything this repository contains at this commit. AIMET is Linux
+ CUDA only and is specified in ``environments/linux-aimet/`` rather than
installed; AI Hub submission is permitted as of 2026-08-03 but there is nothing
to submit, because producing a W8 artifact through Lane A needs a quantize-stage
adapter that ``slm_lab.deployment.qualcomm`` does not have and that T41 does not
own; and the floating Qwen path has not yet traversed the public Workbench
pipeline (T31 and T33 are ``planned``). What T41 freezes here is the
*specification* half of the W8 experiment plus everything up to, and stopping
exactly at, the AI Hub submission boundary.

Three states are kept apart on purpose. ``specified`` is a candidate
specification: prose, policy, and arithmetic. ``simulated`` needs a host
simulation record naming its tool, that tool's exact version, the host, and the
digest of the quantized artifact it produced. ``deployed`` needs all three
sanitized AI Hub schema-v2 stage manifests *and* a verified digest chain
linking the simulated artifact to the compiled artifact to the inference and
profile runs. :func:`assess_precision_state` models the evidence rather than
the wish: it can return ``deployed`` in principle, and there is no input short
of that complete, correctly chained set that makes it do so.

Every number this module emits is either arithmetic over committed repository
inputs, explicitly labelled ``analytic_projection``, a hash or byte size read
off a committed file, or a name, version, or attribute read off the public AI
Hub API by the dated read-only capability query in ``capabilities``, which
submits nothing. There is no measured latency, perplexity, accuracy, artifact
size, or job result here, and none may be added without the artifact that
produced it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from slm_lab.benchmark import protocol as benchmark_protocol
from slm_lab.contracts import static_cache
from slm_lab.deployment.qualcomm import ai_hub
from slm_lab.evaluation.fixtures import canonical_json_sha256

# Deliberate reuse of the T40 parity helper rather than a second, subtly
# different implementation of "which commit was this record written at".
from slm_lab.quantization.parity import _git_commit


SCHEMA_VERSION = 1
TASK_ID = "T41"
RECORD_TYPE = "w8_candidate_readiness"

DEFAULT_MODEL_CONTRACT = Path("configs/models/qwen3-0.6b.yaml")
DEFAULT_CALIBRATION_CONFIG = Path("configs/quantization/calibration.yaml")
DEFAULT_CANDIDATE_DIRECTORY = Path("configs/quantization/w8")
DEFAULT_MANIFEST_DIRECTORY = Path("results/manifests/onnx")
DEFAULT_GRAPH_INVENTORY = Path("results/graph/S128.json")
DEFAULT_ACADEMIC_CONTRACT = Path("configs/workloads/academic-evaluation-v1.json")
DEFAULT_BENCHMARK_PROTOCOL = Path("configs/workloads/benchmark-protocol-v1.json")
DEFAULT_EVIDENCE_DIRECTORY = Path("results/quantization")
DEFAULT_AIMET_REQUIREMENTS = Path("environments/linux-aimet/aimet-requirements.in")

READINESS_RECORD_PREFIX = "t41-w8-readiness-"
PRECISION_EVIDENCE_TEMPLATE = "t41-{candidate_id}-precision-evidence.json"
QUALITY_RECORD_GLOB = "t41-w8-quality-*.json"

CAPABILITY_RECORD_PREFIX = "t41-ai-hub-capability-"
CAPABILITY_RECORD_TYPE = "ai_hub_capability_observation"
#: The single date the committed capability observation was taken on. A new
#: query writes a new dated file and this constant moves with it; it is not a
#: default that quietly re-labels an old observation as a fresh one.
CAPABILITY_OBSERVATION_DATE = "2026-08-03"
DEFAULT_CAPABILITY_RECORD = (
    DEFAULT_EVIDENCE_DIRECTORY
    / f"{CAPABILITY_RECORD_PREFIX}{CAPABILITY_OBSERVATION_DATE}.json"
)

CANDIDATE_IDS = ("w8a16", "w8a8")
PLAN_MATRIX_ROWS = {"w8a16": "Q1", "w8a8": "Q2"}

PRECISION_STATES = ("specified", "simulated", "deployed")
PRECISION_STATE_SCOPES = {
    "specified": "candidate_specification_only_no_weight_was_quantized",
    "simulated": "host_simulation_only_not_compiled_or_executed_on_a_device",
    "deployed": "compiled_inferred_and_profiled_through_the_public_ai_hub_pipeline",
}
MEASURED_PRECISION_STATES = ("simulated", "deployed")
COMPARISON_SCOPES = {
    "simulated": "simulated_vs_float",
    "deployed": "deployed_vs_float",
}
PRECISION_LABEL_SEPARATOR = "+"
FLOAT_BASELINE_PRECISIONS = ("float16", "bfloat16", "float32")
DEPLOYED_EVIDENCE_LEVELS = ("observed_real_device", "observed_hosted_device")

#: Exactly the ``qai-hub`` client and QAIRT build T02 authenticated against and
#: T30 mocked. Both strings are verified in
#: ``ai/handoffs/T30-ai-hub-adapters.md`` and
#: ``results/hosts/workbench-toy-lifecycle-2026-07-25.json``.
QAI_HUB_CLIENT_VERSION = "0.53.0"
QAIRT_VERSION = "2.45.0.260326154327"
RUNTIME_NAME = "QAIRT"
DEFAULT_TIMEOUT_SECONDS = 3600

STAGE_SCRIPTS = {
    "compile": "scripts/qualcomm/compile.py",
    "inference": "scripts/qualcomm/inference.py",
    "profile": "scripts/qualcomm/profile.py",
}
STAGE_OPTIONS = {
    "compile": (f"--target_runtime qnn_context_binary --qairt_version {QAIRT_VERSION}"),
    "inference": f"--qairt_framework {QAIRT_VERSION} --compute_unit npu",
    "profile": f"--qairt_framework {QAIRT_VERSION} --compute_unit npu",
}
STAGE_ORDER = ("compile", "inference", "profile")

#: Device selectors for the plan section 3.2 targets. Every ``name`` below was
#: read from committed T02 access evidence. ``os`` and ``attributes`` are no
#: longer empty and are no longer guesses: they are the values the read-only
#: capability query of ``CAPABILITY_OBSERVATION_DATE`` observed on the live
#: service, recorded in ``DEFAULT_CAPABILITY_RECORD``. The literals are kept
#: here so the default selector stays a pure constant that ``build_stage_request``
#: can use without touching the filesystem, and ``build_deployment_routes``
#: re-checks them against the capability record on every generation run, so a
#: service change fails ``check`` instead of silently re-anchoring a selector.
#:
#: ``os`` is the SDK's own version string. The X Elite CRD reports ``"11"``, not
#: the human label ``"Windows 11"`` this file used to carry; the platform lives
#: in the ``os:windows`` attribute instead.
PRIMARY_DEVICE: dict[str, Any] = {
    "name": "Snapdragon X Elite CRD",
    "os": "11",
    "attributes": [
        "abi:aarch64-windows",
        "chipset:qualcomm-snapdragon-x-elite",
        "chipset:sc8380xp",
        "format:compute",
        "framework:onnx",
        "framework:qnn",
        "hexagon:v73",
        "htp-supports-fp16:true",
        "htp-supports-weight-sharing:true",
        "os:windows",
        "soc-model:60",
        "vendor:qualcomm",
    ],
}
COMPARISON_DEVICES: tuple[dict[str, Any], ...] = (
    {
        "name": "Dragonwing IQ-9075 EVK",
        "os": "1.7",
        "attributes": [
            "abi:aarch64-oe-linux-gcc11",
            "chipset:qcs9075",
            "chipset:qualcomm-dragonwing-iq-9075",
            "chipset:qualcomm-qcs9075",
            "format:iot",
            "framework:onnx",
            "framework:qnn",
            "framework:tflite",
            "hexagon:v73",
            "htp-supports-fp16:true",
            "htp-supports-weight-sharing:true",
            "os:qc_linux",
            "soc-model:77",
            "vendor:qualcomm",
        ],
    },
    {
        "name": "Snapdragon 8 Elite QRD",
        "os": "15",
        "attributes": [
            "abi:aarch64-android",
            "chipset:qualcomm-snapdragon-8-elite",
            "chipset:sm8750",
            "format:phone",
            "framework:onnx",
            "framework:qnn",
            "framework:tflite",
            "hexagon:v79",
            "htp-supports-fp16:true",
            "htp-supports-weight-sharing:true",
            "os:android",
            "soc-model:69",
            "vendor:qualcomm",
        ],
    },
)
#: Plan section 3.2 in selector order: the primary target first, then the two
#: comparison targets. The capability query filters the service's device list
#: down to exactly these names and refuses an observation missing any of them.
PLAN_TARGET_DEVICE_NAMES: tuple[str, ...] = (
    str(PRIMARY_DEVICE["name"]),
    *(str(device["name"]) for device in COMPARISON_DEVICES),
)

GENERATE_COMMAND = "uv run python -m slm_lab.quantization.w8 generate"
CHECK_COMMAND = "uv run python -m slm_lab.quantization.w8 check"
STATUS_COMMAND = "uv run python -m slm_lab.quantization.w8 status"
RECORD_COMMAND = "uv run python -m slm_lab.quantization.w8 record"
COMPARE_COMMAND = (
    "uv run python -m slm_lab.quantization.w8 compare "
    "--baseline <float16-result.json> --candidate <w8-result.json>"
)
REQUEST_COMMAND = (
    "uv run python -m slm_lab.quantization.w8 request --candidate <w8a16|w8a8> "
    "--stage compile --context 128 --graph prefill "
    "--quantized-artifact <private-path>.onnx "
    "--output-artifact <private-path>.serialized.bin "
    "--request-out .ai-local/profiles/T41/compile-request.json"
)
TESTS_COMMAND = "uv run pytest tests/quantization"
CAPABILITIES_COMMAND = "uv run python -m slm_lab.quantization.w8 capabilities"
CAPABILITIES_OFFLINE_COMMAND = (
    "uv run python -m slm_lab.quantization.w8 capabilities "
    f"--offline-input {DEFAULT_CAPABILITY_RECORD.as_posix()}"
)

ESTABLISHES = (
    "two_frozen_regenerable_W8_candidate_specifications_bound_to_committed_inputs",
    "an_explicit_inclusion_exclusion_policy_separating_policy_from_frozen_graph",
    "an_analytic_weight_storage_projection_derived_from_the_model_contract",
    "a_fail_closed_precision_state_function_that_cannot_reach_deployed_here",
    "an_AI_Hub_schema_v2_request_emitter_that_stops_at_the_submission_boundary",
    "a_dated_read_only_observation_that_the_public_quantize_API_can_express_both_candidates",
)
DOES_NOT_ESTABLISH = (
    "any_quantized_weight_activation_or_encoding",
    "simulated_or_deployed_precision_for_either_candidate",
    "quantized_quality_delta_perplexity_or_accuracy",
    "compiler_acceptance_operator_support_or_NPU_placement",
    "on_disk_quantized_artifact_size_or_encoding_overhead",
    "latency_throughput_peak_memory_or_energy",
    "that_the_floating_Qwen_graph_traverses_the_public_pipeline_at_all",
    "that_any_job_was_submitted_or_that_this_repository_could_run_a_quantize_job",
)

PROJECTION_MEASUREMENT = "analytic_projection"
PROJECTION_DOES_NOT_ESTABLISH = (
    "on_disk_quantized_artifact_size",
    "encoding_scale_or_zero_point_storage_overhead_in_the_shipped_format",
    "runtime_peak_memory",
    "any_latency_or_throughput_claim",
    "kv_cache_memory_which_both_candidates_leave_at_the_frozen_float16_contract",
    "quality_impact_of_any_included_or_excluded_weight_class",
)


class W8EvidenceError(ValueError):
    """A W8 candidate specification or its evidence is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise W8EvidenceError(message)


def _load_json(path: Path) -> Any:
    """Read *path* as JSON, turning every read failure into a clean refusal."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        raise W8EvidenceError(f"cannot read {path}: {exc}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise W8EvidenceError(f"cannot parse {path}: {exc}") from exc


def _load_yaml(path: Path) -> Any:
    """Read *path* as YAML with the same fail-closed contract as `_load_json`."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except OSError as exc:
        raise W8EvidenceError(f"cannot read {path}: {exc}") from exc
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise W8EvidenceError(f"cannot parse {path}: {exc}") from exc


def _load_protocol(root: Path) -> Mapping[str, Any]:
    """Load the frozen T13 protocol, failing closed on any drift it reports.

    ``benchmark_protocol.load_protocol`` already refuses a protocol whose
    self-describing digest, reviewed Python digest, or semantics have moved.
    Translating its error keeps every W8 refusal on one exception type.
    """

    try:
        return benchmark_protocol.load_protocol(root)
    except benchmark_protocol.BenchmarkProtocolError as exc:
        raise W8EvidenceError(
            f"the frozen T13 benchmark protocol no longer validates: {exc}"
        ) from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise W8EvidenceError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(ai_hub.SHA256_PATTERN.fullmatch(value))


def _is_exact_version(value: Any) -> bool:
    return isinstance(value, str) and bool(
        ai_hub.EXACT_VERSION_PATTERN.fullmatch(value)
    )


def candidate_config_path(candidate_id: str) -> Path:
    """Return the repository-relative path of one candidate specification."""

    _require(
        candidate_id in CANDIDATE_IDS,
        f"unknown W8 candidate {candidate_id!r}; expected one of {CANDIDATE_IDS}",
    )
    return DEFAULT_CANDIDATE_DIRECTORY / f"{candidate_id}.yaml"


# ---------------------------------------------------------------------------
# Weight classes and the analytic storage projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightClass:
    """One class of stored parameters, derived from the T00 model contract.

    ``shape`` is the *logical* shape a reader reasons about — ``[out, in]`` for
    a projection matrix, ``[n]`` for a scale vector — not necessarily the axis
    order the ONNX initializer carries. Which axis a per-output-channel scale
    indexes in the exported artifact has to be read off the artifact; see
    ``per_channel_axis_note`` on the candidate's ``weights`` block.
    """

    class_id: str
    role: str
    instances: int
    shape: tuple[int, ...]
    exclusion_id: str | None
    note: str

    @property
    def parameters_per_instance(self) -> int:
        return math.prod(self.shape)

    @property
    def parameters(self) -> int:
        return self.instances * self.parameters_per_instance

    @property
    def output_channels(self) -> int | None:
        """Return per-instance output channels, or ``None`` for a vector."""

        return self.shape[0] if len(self.shape) == 2 else None


def derive_weight_classes(model_contract: Mapping[str, Any]) -> tuple[WeightClass, ...]:
    """Derive every stored-parameter class from ``configs/models/qwen3-0.6b.yaml``.

    Nothing here is hardcoded from the published model size: layer count, head
    counts, head dimension, both feature widths, and the vocabulary all come
    from the committed contract, and the tied-embedding flag decides whether
    the vocabulary table is counted once or twice.
    """

    architecture = model_contract["model"]["architecture"]
    layers = int(architecture["num_hidden_layers"])
    hidden = int(architecture["hidden_size"])
    intermediate = int(architecture["intermediate_size"])
    heads = int(architecture["num_attention_heads"])
    kv_heads = int(architecture["num_key_value_heads"])
    head_dim = int(architecture["head_dim"])
    vocab = int(architecture["vocab_size"])
    tied = bool(architecture["tie_word_embeddings"])

    _require(
        tied,
        "this projection counts the vocabulary table once because "
        "tie_word_embeddings is true; an untied contract needs a second "
        "weight class before the projection can be trusted",
    )
    for label, value in (
        ("num_hidden_layers", layers),
        ("hidden_size", hidden),
        ("intermediate_size", intermediate),
        ("num_attention_heads", heads),
        ("num_key_value_heads", kv_heads),
        ("head_dim", head_dim),
        ("vocab_size", vocab),
    ):
        _require(value > 0, f"model contract {label} must be positive, found {value}")
    _require(
        heads % kv_heads == 0,
        f"GQA requires num_attention_heads ({heads}) to be a multiple of "
        f"num_key_value_heads ({kv_heads})",
    )

    query_width = heads * head_dim
    key_value_width = kv_heads * head_dim
    return (
        WeightClass(
            class_id="tied_vocabulary_table",
            role="token embedding lookup and, because the weights are tied, the "
            "final logits projection",
            instances=1,
            shape=(vocab, hidden),
            exclusion_id="tied_embedding_table",
            note=(
                f"One table of {vocab} rows serves both directions. The T21 "
                "inventory records the lookup as a Gather rather than a MatMul, "
                "so the two uses are not even the same operator class."
            ),
        ),
        WeightClass(
            class_id="attention_q_proj",
            role="query projection",
            instances=layers,
            shape=(query_width, hidden),
            exclusion_id=None,
            note=(
                f"{heads} query heads at head_dim {head_dim} give {query_width} "
                "output channels, twice the key/value width under GQA."
            ),
        ),
        WeightClass(
            class_id="attention_k_proj",
            role="key projection",
            instances=layers,
            shape=(key_value_width, hidden),
            exclusion_id=None,
            note=(
                f"{kv_heads} key/value heads give {key_value_width} output "
                "channels; GQA makes this matrix half the size of q_proj."
            ),
        ),
        WeightClass(
            class_id="attention_v_proj",
            role="value projection",
            instances=layers,
            shape=(key_value_width, hidden),
            exclusion_id=None,
            note="Same GQA width as k_proj.",
        ),
        WeightClass(
            class_id="attention_o_proj",
            role="attention output projection",
            instances=layers,
            shape=(hidden, query_width),
            exclusion_id=None,
            note=(
                f"Reduces over {query_width} inputs into {hidden} output "
                "channels and writes straight into the residual stream."
            ),
        ),
        WeightClass(
            class_id="mlp_gate_proj",
            role="SwiGLU gate projection",
            instances=layers,
            shape=(intermediate, hidden),
            exclusion_id=None,
            note=f"{intermediate} output channels feeding the SiLU gate.",
        ),
        WeightClass(
            class_id="mlp_up_proj",
            role="SwiGLU up projection",
            instances=layers,
            shape=(intermediate, hidden),
            exclusion_id=None,
            note=f"{intermediate} output channels multiplied by the gate.",
        ),
        WeightClass(
            class_id="mlp_down_proj",
            role="SwiGLU down projection",
            instances=layers,
            shape=(hidden, intermediate),
            exclusion_id=None,
            note=(
                f"The widest reduction in the block: {intermediate} inputs per "
                f"output channel, so per-output-channel weight error "
                "accumulates over the longest dot product here."
            ),
        ),
        WeightClass(
            class_id="block_rmsnorm_scales",
            role="input_layernorm and post_attention_layernorm scale vectors",
            instances=layers,
            shape=(2 * hidden,),
            exclusion_id="rmsnorm_scales",
            note=(
                "Two vectors of hidden_size per block. They multiply the "
                "normalized residual channel-wise before every matmul."
            ),
        ),
        WeightClass(
            class_id="qk_head_norm_scales",
            role="Qwen3 per-head q_norm and k_norm scale vectors",
            instances=layers,
            shape=(2 * head_dim,),
            exclusion_id="qwen3_per_head_qk_norm",
            note=(
                f"Two vectors of head_dim {head_dim} per block. These are the "
                "56 initializers the T21 inventory leaves inline in the ONNX "
                "protobuf rather than in external data."
            ),
        ),
        WeightClass(
            class_id="final_rmsnorm_scale",
            role="final norm scale vector before the logits projection",
            instances=1,
            shape=(hidden,),
            exclusion_id="rmsnorm_scales",
            note="One vector of hidden_size on the model output path.",
        ),
    )


def _projection_row(
    weight_class: WeightClass,
    *,
    quantized: bool,
    weight_bytes: int,
    float_bytes: int,
) -> dict[str, Any]:
    parameters = weight_class.parameters
    return {
        "class_id": weight_class.class_id,
        "role": weight_class.role,
        "measurement": PROJECTION_MEASUREMENT,
        "instances": weight_class.instances,
        "logical_shape_per_instance": list(weight_class.shape),
        "parameters_per_instance": weight_class.parameters_per_instance,
        "parameters": parameters,
        "quantized": quantized,
        "kept_precision": "int8" if quantized else "float16",
        "exclusion_id": weight_class.exclusion_id,
        "per_output_channel_scales": (
            weight_class.output_channels * weight_class.instances
            if quantized and weight_class.output_channels is not None
            else 0
        ),
        "float16_bytes": parameters * float_bytes,
        "candidate_bytes": parameters * (weight_bytes if quantized else float_bytes),
        "note": weight_class.note,
    }


def _kv_cache_rows() -> list[dict[str, Any]]:
    """Project the float16 KV cache both candidates leave untouched."""

    rows: list[dict[str, Any]] = []
    for prompt_length, capacity in sorted(static_cache.CONTEXT_VARIANTS.items()):
        rows.append(
            {
                "variant_id": f"S{prompt_length}",
                "measurement": PROJECTION_MEASUREMENT,
                "prompt_length": prompt_length,
                "cache_capacity": capacity,
                "dtype": static_cache.CACHE_DTYPE,
                "bytes": static_cache.cache_bytes(capacity),
            }
        )
    return rows


def _excluded_ids(candidate: Mapping[str, Any]) -> set[str]:
    policy = candidate.get("excluded_from_quantization")
    _require(
        isinstance(policy, Mapping),
        "candidate carries no excluded_from_quantization policy",
    )
    assert isinstance(policy, Mapping)
    entries = policy.get("entries")
    _require(
        isinstance(entries, list) and bool(entries),
        "excluded_from_quantization.entries must be a non-empty list",
    )
    assert isinstance(entries, list)
    identifiers: set[str] = set()
    for entry in entries:
        _require(
            isinstance(entry, Mapping) and isinstance(entry.get("id"), str),
            "every exclusion entry needs a string id",
        )
        assert isinstance(entry, Mapping)
        identifiers.add(str(entry["id"]))
    return identifiers


def weight_storage_projection(
    candidate: Mapping[str, Any],
    model_contract: Mapping[str, Any],
    *,
    graph_inventory: Mapping[str, Any] | None = None,
    baseline_external_data_bytes: int | None = None,
) -> dict[str, Any]:
    """Project float16 versus mixed int8/float16 weight storage, analytically.

    Every field is arithmetic over ``configs/models/qwen3-0.6b.yaml`` and the
    candidate's own exclusion policy. Nothing here was weighed, timed, or read
    off a quantized artifact, because no quantized artifact exists; the
    ``does_not_establish`` list on the returned mapping is the authoritative
    statement of that boundary.

    When the committed T20 external-data byte size and the T21 graph inventory
    are supplied, the projection additionally carries an independent
    cross-check: the exported float16 artifact's external data plus its inline
    initializers must account for exactly the derived parameter total.
    """

    classes = derive_weight_classes(model_contract)
    excluded = _excluded_ids(candidate)
    known_exclusions = {
        weight_class.exclusion_id
        for weight_class in classes
        if weight_class.exclusion_id is not None
    }
    missing = sorted(known_exclusions - excluded)
    _require(
        not missing,
        "the candidate's exclusion policy no longer covers weight class(es) "
        f"{', '.join(missing)}; a dropped entry would silently flip them to int8",
    )

    weights = candidate.get("weights")
    _require(isinstance(weights, Mapping), "candidate carries no weights block")
    assert isinstance(weights, Mapping)
    weight_dtype = str(weights.get("dtype"))
    _require(
        weight_dtype == "int8",
        f"W8 candidates quantize weights to int8, found {weight_dtype!r}",
    )
    weight_bytes = 1
    float_bytes = static_cache.DTYPE_BYTES["float16"]
    scale_dtype = str(weights.get("scale_dtype"))
    scale_bytes = static_cache.DTYPE_BYTES.get(scale_dtype)
    _require(
        scale_bytes is not None,
        f"weights.scale_dtype {scale_dtype!r} has no known byte width",
    )
    assert scale_bytes is not None

    rows = [
        _projection_row(
            weight_class,
            quantized=weight_class.exclusion_id not in excluded,
            weight_bytes=weight_bytes,
            float_bytes=float_bytes,
        )
        for weight_class in classes
    ]
    total_parameters = sum(row["parameters"] for row in rows)
    quantized_parameters = sum(row["parameters"] for row in rows if row["quantized"])
    excluded_parameters = total_parameters - quantized_parameters
    float16_bytes = sum(row["float16_bytes"] for row in rows)
    candidate_bytes = sum(row["candidate_bytes"] for row in rows)
    scale_count = sum(row["per_output_channel_scales"] for row in rows)
    scale_storage_bytes = scale_count * scale_bytes

    projection: dict[str, Any] = {
        "measurement": PROJECTION_MEASUREMENT,
        "derived_from": {
            "model_contract": DEFAULT_MODEL_CONTRACT.as_posix(),
            "fields": [
                "num_hidden_layers",
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
                "tie_word_embeddings",
            ],
            "exclusion_policy": "this candidate's excluded_from_quantization block",
        },
        "weight_classes": rows,
        "totals": {
            "measurement": PROJECTION_MEASUREMENT,
            "total_parameters": total_parameters,
            "quantized_parameters": quantized_parameters,
            "excluded_parameters": excluded_parameters,
            "quantized_parameter_fraction": round(
                quantized_parameters / total_parameters, 6
            ),
            "float16_weight_bytes": float16_bytes,
            "candidate_weight_bytes": candidate_bytes,
            "weight_bytes_saved": float16_bytes - candidate_bytes,
            "weight_byte_ratio_float16_over_candidate": round(
                float16_bytes / candidate_bytes, 6
            ),
        },
        "scale_storage_lower_bound": {
            "measurement": PROJECTION_MEASUREMENT,
            "granularity": str(weights.get("granularity")),
            "scale_dtype": scale_dtype,
            "per_output_channel_scale_count": scale_count,
            "scale_bytes": scale_storage_bytes,
            "candidate_weight_bytes_including_scales": candidate_bytes
            + scale_storage_bytes,
            "weight_byte_ratio_including_scales": round(
                float16_bytes / (candidate_bytes + scale_storage_bytes), 6
            ),
            "note": (
                "A lower bound, and the reason the headline ratio above "
                "overstates the saving. It counts one scale per output channel "
                "at the declared scale dtype and nothing else: no zero points "
                "(the policy is symmetric, so there are none), no QDQ node "
                "overhead, no per-tensor activation encodings, and no container "
                "framing. The shipped format is not decided at this commit, so "
                "the real overhead is unknown rather than estimated."
            ),
        },
        "kv_cache_left_at_the_frozen_contract": {
            "measurement": PROJECTION_MEASUREMENT,
            "dtype": static_cache.CACHE_DTYPE,
            "contract_source": "slm_lab.contracts.static_cache.CACHE_DTYPE",
            "by_context": _kv_cache_rows(),
            "note": (
                "Both W8 candidates leave the cache at float16, so none of the "
                "weight saving above applies to it. At the longest frozen "
                "context the cache alone is the same order of magnitude as the "
                "entire float16 weight set, which is why a weight-only W8 "
                "result must never be presented as a memory result for long "
                "context."
            ),
        },
        "does_not_establish": list(PROJECTION_DOES_NOT_ESTABLISH),
    }

    if graph_inventory is not None and baseline_external_data_bytes is not None:
        projection["cross_check"] = _projection_cross_check(
            total_parameters=total_parameters,
            model_contract=model_contract,
            graph_inventory=graph_inventory,
            baseline_external_data_bytes=baseline_external_data_bytes,
        )
    return projection


def _projection_cross_check(
    *,
    total_parameters: int,
    model_contract: Mapping[str, Any],
    graph_inventory: Mapping[str, Any],
    baseline_external_data_bytes: int,
) -> dict[str, Any]:
    """Reconcile the derived parameter total with the committed T20 artifact.

    The exported float16 prefill graph splits its initializers in two: 254 go
    to external data and 56 stay inline in the protobuf. Those 56 are exactly
    ``2 x num_hidden_layers``, and the largest of them is exactly
    ``head_dim x 2`` bytes — the Qwen3 per-head q_norm/k_norm vectors, which sit
    below the exporter's external-data threshold. Adding them back to the
    external-data element count must reproduce the parameter total derived from
    the model contract. Three independent committed sources therefore have to
    agree, and this function fails closed if they stop agreeing.
    """

    architecture = model_contract["model"]["architecture"]
    layers = int(architecture["num_hidden_layers"])
    head_dim = int(architecture["head_dim"])
    float_bytes = static_cache.DTYPE_BYTES["float16"]

    initializers = int(graph_inventory["initializer_count"])
    external = int(graph_inventory["external_initializer_count"])
    largest_inline = int(graph_inventory["largest_inline_initializer_bytes"])
    inline = initializers - external

    _require(
        inline == 2 * layers,
        f"the committed T21 graph inventory records {inline} inline "
        f"initializer(s); the q_norm/k_norm accounting this projection relies "
        f"on requires exactly {2 * layers}. The exported graph changed: "
        f"re-derive the cross-check before trusting the projection.",
    )
    _require(
        largest_inline == head_dim * float_bytes,
        f"the largest inline initializer is {largest_inline} bytes; a per-head "
        f"float16 norm vector is {head_dim * float_bytes} bytes. The inline set "
        "is no longer the q_norm/k_norm vectors.",
    )
    _require(
        baseline_external_data_bytes % float_bytes == 0,
        "the committed external-data size is not a whole number of float16 elements",
    )
    external_elements = baseline_external_data_bytes // float_bytes
    inline_elements = inline * head_dim
    reconstructed = external_elements + inline_elements
    _require(
        reconstructed == total_parameters,
        "the exported float16 artifact accounts for "
        f"{reconstructed} parameters but the model contract derives "
        f"{total_parameters}; one of the two drifted",
    )
    return {
        "measurement": "arithmetic_over_committed_inputs",
        "question": (
            "does the parameter total derived from the model contract equal "
            "the parameter total the committed float16 export actually stores?"
        ),
        "external_data_bytes": baseline_external_data_bytes,
        "external_data_float16_elements": external_elements,
        "external_initializer_count": external,
        "inline_initializer_count": inline,
        "inline_initializer_elements": inline_elements,
        "reconstructed_parameters": reconstructed,
        "derived_parameters": total_parameters,
        "agrees": True,
        "note": (
            "This is a byte-level reconciliation of two independent committed "
            "sources, not a measurement of anything quantized. It is here "
            "because a parameter count that disagrees with the artifact would "
            "silently poison every ratio in this projection."
        ),
    }


# ---------------------------------------------------------------------------
# Precision state: the fail-closed composition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceCheck:
    """One yes/no question about supplied evidence, with its answer."""

    name: str
    satisfied: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "satisfied": self.satisfied,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PrecisionFinding:
    """A precision state that always travels with its scope and its reasons."""

    state: str
    scope: str
    simulation_checks: tuple[EvidenceCheck, ...]
    deployment_checks: tuple[EvidenceCheck, ...]

    @property
    def unsatisfied(self) -> tuple[str, ...]:
        """Return the names of every check the supplied evidence failed."""

        return tuple(
            check.name
            for check in (*self.simulation_checks, *self.deployment_checks)
            if not check.satisfied
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "scope": self.scope,
            "simulation_checks": [check.as_dict() for check in self.simulation_checks],
            "deployment_checks": [check.as_dict() for check in self.deployment_checks],
            "unsatisfied_checks": list(self.unsatisfied),
        }


def stage_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Recompute the digest ``ai_hub.write_manifest`` assigns to a manifest.

    Mirrors ``ai_hub._canonical_bytes``: sorted keys, no whitespace, ASCII
    escaping, one trailing newline. It is recomputed rather than trusted so
    that the downstream stages' ``predecessor_manifest_sha256`` can be checked
    against the compile manifest a caller actually supplied, instead of against
    whatever digest that caller also supplied.
    """

    payload = (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _simulation_checks(evidence: Any) -> tuple[EvidenceCheck, ...]:
    record = evidence.get("simulation") if isinstance(evidence, Mapping) else None
    if not isinstance(record, Mapping):
        return (
            EvidenceCheck(
                "simulation_record_present",
                False,
                "no simulation record was supplied, so the candidate is a "
                "specification and nothing more",
            ),
        )
    checks = [
        EvidenceCheck(
            "simulation_record_present",
            True,
            "a simulation record was supplied",
        )
    ]
    for field, requirement in (
        ("tool", "names the tool that produced the simulated weights"),
        ("host", "names the host the simulation ran on"),
    ):
        value = record.get(field)
        checks.append(
            EvidenceCheck(
                f"simulation_{field}",
                isinstance(value, str) and bool(value.strip()),
                requirement,
            )
        )
    checks.append(
        EvidenceCheck(
            "simulation_tool_version",
            _is_exact_version(record.get("tool_version")),
            "carries an exact tool version; a range or a floating tag is not a "
            "version this repository will record",
        )
    )
    checks.append(
        EvidenceCheck(
            "simulation_quantized_artifact_digest",
            _is_sha256(record.get("quantized_artifact_sha256")),
            "carries the lowercase sha256 of the quantized artifact the "
            "simulation produced",
        )
    )
    return tuple(checks)


def _stage_manifest_checks(
    manifests: Mapping[str, Any],
    stage: str,
) -> tuple[list[EvidenceCheck], Mapping[str, Any] | None]:
    manifest = manifests.get(stage)
    if not isinstance(manifest, Mapping):
        return (
            [
                EvidenceCheck(
                    f"{stage}_manifest_present",
                    False,
                    f"no {stage} stage manifest was supplied",
                )
            ],
            None,
        )
    checks = [
        EvidenceCheck(
            f"{stage}_manifest_present",
            True,
            f"a {stage} stage manifest was supplied",
        ),
        EvidenceCheck(
            f"{stage}_manifest_schema",
            manifest.get("schema_version") == ai_hub.SCHEMA_VERSION
            and manifest.get("manifest_type") == ai_hub.MANIFEST_TYPE,
            f"is a schema-v{ai_hub.SCHEMA_VERSION} {ai_hub.MANIFEST_TYPE} manifest",
        ),
        EvidenceCheck(
            f"{stage}_manifest_is_the_right_stage",
            manifest.get("stage") == stage,
            f"declares stage {stage!r}; a manifest for another stage is not "
            "evidence for this one",
        ),
        EvidenceCheck(
            f"{stage}_manifest_succeeded",
            manifest.get("status") == "success",
            "reports a successful stage",
        ),
    ]
    return checks, manifest


def _source_artifact_digests(manifest: Mapping[str, Any], role: str | None) -> set[str]:
    lineage = manifest.get("lineage")
    if not isinstance(lineage, Mapping):
        return set()
    artifacts = lineage.get("source_artifacts")
    if not isinstance(artifacts, list):
        return set()
    digests: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            continue
        if role is not None and artifact.get("role") != role:
            continue
        digest = artifact.get("sha256")
        if _is_sha256(digest):
            digests.add(str(digest))
    return digests


def _deployment_checks(
    evidence: Any,
    *,
    quantized_digest: str | None,
) -> tuple[EvidenceCheck, ...]:
    manifests = (
        evidence.get("stage_manifests") if isinstance(evidence, Mapping) else None
    )
    if not isinstance(manifests, Mapping):
        manifests = {}

    checks: list[EvidenceCheck] = []
    loaded: dict[str, Mapping[str, Any] | None] = {}
    for stage in STAGE_ORDER:
        stage_checks, manifest = _stage_manifest_checks(manifests, stage)
        checks.extend(stage_checks)
        loaded[stage] = manifest

    compile_manifest = loaded["compile"]
    compiled_digest: str | None = None
    compile_manifest_digest: str | None = None
    if compile_manifest is not None:
        result = compile_manifest.get("result")
        target = result.get("target_artifact") if isinstance(result, Mapping) else None
        candidate_digest = target.get("sha256") if isinstance(target, Mapping) else None
        if _is_sha256(candidate_digest):
            compiled_digest = str(candidate_digest)
        compile_manifest_digest = stage_manifest_sha256(compile_manifest)

    checks.append(
        EvidenceCheck(
            "compile_target_artifact_digest",
            compiled_digest is not None,
            "the compile manifest records the sha256 of the artifact it produced",
        )
    )
    checks.append(
        EvidenceCheck(
            "compile_source_is_the_simulated_artifact",
            bool(
                compile_manifest is not None
                and quantized_digest is not None
                and quantized_digest in _source_artifact_digests(compile_manifest, None)
            ),
            "the compile manifest's source artifact digest equals the "
            "quantized-artifact digest the simulation record declared; without "
            "this link the compiled artifact is not this candidate's",
        )
    )

    for stage in ("inference", "profile"):
        manifest = loaded[stage]
        lineage = manifest.get("lineage") if isinstance(manifest, Mapping) else None
        predecessor = (
            lineage.get("predecessor_manifest_sha256")
            if isinstance(lineage, Mapping)
            else None
        )
        checks.append(
            EvidenceCheck(
                f"{stage}_cites_the_compile_manifest",
                bool(
                    compile_manifest_digest is not None
                    and _is_sha256(predecessor)
                    and predecessor == compile_manifest_digest
                ),
                "the predecessor manifest digest equals the recomputed digest "
                "of the compile manifest supplied here",
            )
        )
        checks.append(
            EvidenceCheck(
                f"{stage}_consumed_the_compiled_artifact",
                bool(
                    manifest is not None
                    and compiled_digest is not None
                    and compiled_digest
                    in _source_artifact_digests(manifest, "compiled_model")
                ),
                "the stage's compiled_model source artifact digest equals the "
                "compile manifest's target artifact digest",
            )
        )
    return tuple(checks)


def assess_precision_state(evidence: Any) -> PrecisionFinding:
    """Compose supplied evidence into one precision state and its scope.

    Pure, and split out for exactly the reason ``parity.overall_verdict`` is:
    the guarantee that matters — that no incomplete or unchained input can
    return ``deployed`` — is only testable if it lives in a function that takes
    data rather than in a code path that needs a hosted device.

    The evidence mapping is read positively and never for a verdict: this
    function looks for a simulation record and three chained stage manifests. A
    ``state``, ``precision_state``, ``verdict``, or ``deployed`` key planted
    anywhere in the input is not consulted, so a record cannot assert its own
    conclusion.
    """

    simulation = _simulation_checks(evidence)
    simulated = all(check.satisfied for check in simulation)
    quantized_digest: str | None = None
    if simulated and isinstance(evidence, Mapping):
        record = evidence.get("simulation")
        if isinstance(record, Mapping):
            quantized_digest = str(record.get("quantized_artifact_sha256"))

    deployment = _deployment_checks(evidence, quantized_digest=quantized_digest)
    deployed = simulated and all(check.satisfied for check in deployment)

    if deployed:
        state = "deployed"
    elif simulated:
        state = "simulated"
    else:
        state = "specified"
    return PrecisionFinding(
        state=state,
        scope=PRECISION_STATE_SCOPES[state],
        simulation_checks=simulation,
        deployment_checks=deployment,
    )


def precision_state(evidence: Any) -> str:
    """Return one of ``specified``, ``simulated``, or ``deployed``."""

    return assess_precision_state(evidence).state


def precision_state_scope(evidence: Any) -> str:
    """Return the scope string that must travel with :func:`precision_state`."""

    return assess_precision_state(evidence).scope


# ---------------------------------------------------------------------------
# Quality comparison under the frozen T13 evaluation
# ---------------------------------------------------------------------------


def precision_label(candidate_id: str, state: str) -> str:
    """Return the ``source.precision`` string a W8 result record must carry.

    The frozen T13 result schema is closed (``additionalProperties: false`` at
    the record root and inside ``source``), so a W8 result has nowhere to add a
    bespoke ``precision_state`` field. The state therefore rides on
    ``source.precision``, which the schema already requires, and is
    cross-checked against ``system.evidence_level``, which the schema already
    constrains. Changing that would mean changing a frozen T13 contract T41
    does not own.
    """

    _require(
        candidate_id in CANDIDATE_IDS,
        f"unknown W8 candidate {candidate_id!r}",
    )
    _require(
        state in MEASURED_PRECISION_STATES,
        f"a measured result may only declare {MEASURED_PRECISION_STATES}, "
        f"found {state!r}; a 'specified' candidate has no measurement",
    )
    return f"{candidate_id}{PRECISION_LABEL_SEPARATOR}{state}"


def parse_precision_label(label: Any) -> tuple[str, str]:
    """Split a candidate ``source.precision`` label into candidate and state."""

    _require(
        isinstance(label, str) and PRECISION_LABEL_SEPARATOR in label,
        "the candidate result does not declare a precision state; "
        f"source.precision must read '<candidate>{PRECISION_LABEL_SEPARATOR}"
        f"<{'|'.join(MEASURED_PRECISION_STATES)}>', found {label!r}",
    )
    assert isinstance(label, str)
    candidate_id, _, state = label.partition(PRECISION_LABEL_SEPARATOR)
    _require(
        candidate_id in CANDIDATE_IDS,
        f"the candidate result declares precision {label!r}, whose candidate "
        f"part is not one of {CANDIDATE_IDS}",
    )
    _require(
        state in MEASURED_PRECISION_STATES,
        f"the candidate result declares precision state {state!r}; only "
        f"{MEASURED_PRECISION_STATES} can carry a measurement",
    )
    return candidate_id, state


def _quality_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    measurement = record["measurement"]
    method = measurement.get("quality_method") or {}
    return {
        "workload_id": record["source"]["workload_id"],
        "metric": measurement["metric"],
        "unit": measurement["unit"],
        "kind": measurement["kind"],
        "task_id": method.get("task_id"),
        "metric_name": method.get("metric_name"),
        "dataset_id": method.get("dataset_id"),
        "dataset_revision": method.get("dataset_revision"),
        "dataset_config": method.get("dataset_config"),
        "split": method.get("split"),
        "selection": method.get("selection"),
        "harness_release": method.get("harness_release"),
        "harness_commit": method.get("harness_commit"),
        "resolved_task_sha256": method.get("resolved_task_sha256"),
        "prompt_interface": method.get("prompt_interface"),
        "apply_chat_template": method.get("apply_chat_template"),
        "fewshot": method.get("fewshot"),
        "actual_prompt_tokens": measurement.get("actual_prompt_tokens"),
        "actual_generated_tokens": measurement.get("actual_generated_tokens"),
    }


def inherited_calibration_bias(root: Path | str) -> dict[str, Any]:
    """Return the T40 calibration/evaluation overlap caveat, read from source.

    Attached to every comparison so a consumer that only reads the numbers
    still carries the reason those numbers are optimistic.
    """

    root_path = Path(root).resolve()
    calibration = _load_yaml(root_path / DEFAULT_CALIBRATION_CONFIG)
    _require(
        isinstance(calibration, Mapping),
        f"calibration contract is not a YAML mapping: {DEFAULT_CALIBRATION_CONFIG}",
    )
    assert isinstance(calibration, Mapping)
    overlap = (calibration.get("licensing") or {}).get("evaluation_overlap")
    _require(
        isinstance(overlap, Mapping),
        "the T40 calibration contract no longer declares its "
        "calibration/evaluation overlap; a W8 quality delta may not be "
        "reported without it",
    )
    assert isinstance(overlap, Mapping)
    corpus = calibration.get("calibration_corpus") or {}
    coverage = corpus.get("coverage") or {}
    budget = corpus.get("token_budget") or {}
    return {
        "inherited_from": "T40",
        "source": DEFAULT_CALIBRATION_CONFIG.as_posix(),
        "calibration_dataset_revision": calibration.get("calibration_dataset_revision"),
        "overlaps": overlap.get("overlaps"),
        "scope": overlap.get("scope"),
        "statement": overlap.get("statement"),
        "total_calibration_tokens": budget.get("total_calibration_tokens"),
        "largest_source_group_token_share": max(
            (coverage.get("token_share_per_source_group") or {}).values(),
            default=None,
        ),
        "distinct_token_ids": coverage.get("distinct_token_ids"),
        "model_vocabulary_size": coverage.get("model_vocabulary_size"),
        "direction": (
            "optimistic: the range observers were fitted on prompts the delta "
            "is measured on, so the measured degradation is a lower bound on "
            "the degradation an unseen prompt would show"
        ),
    }


def compare_quality(
    baseline_record: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    *,
    root: Path | str,
) -> dict[str, Any]:
    """Compare two frozen-protocol quality results and refuse anything else.

    This function computes deltas from records it is *given*. It reads no
    artifact, runs no evaluation, and ships no measurement of its own; at this
    commit the repository contains no such record for either candidate, which
    :func:`validate_repository` enforces.

    ``comparison_scope`` is derived from the candidate record's own declared
    precision state and from nothing else. There is deliberately no caller
    argument that could label a simulated comparison as deployed.
    """

    root_path = Path(root).resolve()
    for label, record in (
        ("baseline", baseline_record),
        ("candidate", candidate_record),
    ):
        _require(
            isinstance(record, Mapping),
            f"the {label} quality record must be a mapping",
        )

    protocol = _load_protocol(root_path)
    repository_digest = protocol["contract_sha256"]
    for label, record in (
        ("baseline", baseline_record),
        ("candidate", candidate_record),
    ):
        declared = record.get("protocol_sha256")
        _require(
            declared == repository_digest,
            f"the {label} record cites protocol digest {declared!r}; this "
            f"repository's frozen protocol is {repository_digest!r}. A quality "
            "delta computed under any other protocol digest is not comparable "
            "and this function will not compute one.",
        )

    for label, record in (
        ("baseline", baseline_record),
        ("candidate", candidate_record),
    ):
        try:
            benchmark_protocol.validate_result(record, root=root_path)
        except benchmark_protocol.BenchmarkProtocolError as exc:
            raise W8EvidenceError(
                f"the {label} record does not satisfy the frozen T13 "
                f"evaluation contract: {exc}"
            ) from exc

    baseline_identity = _quality_identity(baseline_record)
    candidate_identity = _quality_identity(candidate_record)
    differences = sorted(
        key
        for key, value in baseline_identity.items()
        if candidate_identity.get(key) != value
    )
    _require(
        not differences,
        "the two records do not measure the same thing; they differ on "
        f"{', '.join(differences)}. Refusing to subtract them.",
    )

    baseline_precision = baseline_record["source"]["precision"]
    _require(
        baseline_precision in FLOAT_BASELINE_PRECISIONS,
        f"the baseline record declares precision {baseline_precision!r}; a W8 "
        f"delta is anchored on a floating baseline {FLOAT_BASELINE_PRECISIONS}",
    )
    candidate_id, state = parse_precision_label(candidate_record["source"]["precision"])
    evidence_level = candidate_record["system"]["evidence_level"]
    if state == "deployed":
        _require(
            evidence_level in DEPLOYED_EVIDENCE_LEVELS,
            f"the candidate declares precision state 'deployed' but reports "
            f"system.evidence_level {evidence_level!r}; a deployed precision "
            f"requires one of {DEPLOYED_EVIDENCE_LEVELS}",
        )
    else:
        _require(
            evidence_level == "simulated",
            "the candidate declares precision state 'simulated' but reports "
            f"system.evidence_level {evidence_level!r}",
        )

    baseline_median = float(baseline_record["summary"]["median"])
    candidate_median = float(candidate_record["summary"]["median"])
    absolute_delta = candidate_median - baseline_median
    relative_delta = (
        absolute_delta / baseline_median if baseline_median != 0.0 else None
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "record_type": "w8_quality_comparison",
        "comparison_scope": COMPARISON_SCOPES[state],
        "comparison_scope_source": "candidate_record.source.precision",
        "candidate": {
            "candidate_id": candidate_id,
            "precision_state": state,
            "precision_state_scope": PRECISION_STATE_SCOPES[state],
            "precision_label": candidate_record["source"]["precision"],
            "evidence_level": evidence_level,
            "result_id": candidate_record["result_id"],
            "artifact_sha256": candidate_record["source"]["artifact_sha256"],
        },
        "baseline": {
            "precision": baseline_precision,
            "evidence_level": baseline_record["system"]["evidence_level"],
            "result_id": baseline_record["result_id"],
            "artifact_sha256": baseline_record["source"]["artifact_sha256"],
        },
        "protocol": {
            "path": DEFAULT_BENCHMARK_PROTOCOL.as_posix(),
            "protocol_id": protocol["protocol_id"],
            "contract_sha256": repository_digest,
            "statement": (
                "Both records were validated against this exact protocol "
                "digest. A delta computed under any other digest is not "
                "comparable with this one."
            ),
        },
        "measured": dict(candidate_identity),
        "delta": {
            "measurement": "delta_of_supplied_records",
            "statistic": "median",
            "unit": candidate_identity["unit"],
            "baseline_median": baseline_median,
            "candidate_median": candidate_median,
            "absolute_delta": absolute_delta,
            "relative_delta": relative_delta,
            "note": (
                "Sign convention is candidate minus baseline in the metric's "
                "own unit. Whether a positive delta is better or worse depends "
                "on the metric: perplexity down is better, accuracy up is "
                "better. This function does not decide that for the reader."
            ),
        },
        "inherited_bias": inherited_calibration_bias(root_path),
        "does_not_establish": [
            "generalization_beyond_the_frozen_academic_subset",
            "comparability_with_a_delta_measured_under_another_protocol_digest",
            ("deployed_precision_unless_comparison_scope_reads_deployed_vs_float"),
            "latency_memory_or_placement_of_either_side",
        ],
    }


# ---------------------------------------------------------------------------
# The AI Hub capability observation: read the public surface, submit nothing
# ---------------------------------------------------------------------------

#: Duplicated from ``slm_lab.deployment.qualcomm.ai_hub.PRIVATE_TEXT_PATTERNS``
#: rather than imported. That name is private to a module T22 owns, and reaching
#: across a task boundary for a private constant makes this module break the
#: moment T22 reorganizes its internals. The set is seven lines of redaction
#: policy rather than an algorithm, so a local copy is the cheaper coupling and
#: it can be widened here without touching another task's code.
#:
#: Two patterns are deliberately wider than the adapter's. ``ai_hub`` only has to
#: *detect* private text before refusing a manifest, so matching ``https?://`` or
#: ``token:`` is enough for it. This module has to *remove* the text, and
#: removing only the scheme or only the key would leave the host, path, or value
#: behind, so both patterns consume the whole token.
CAPABILITY_PRIVATE_TEXT_PATTERNS = (
    re.compile(r"https?://\S*", re.IGNORECASE),
    re.compile(r"\b(?:api[_-]?)?token\s*[:=]\s*\S*", re.IGNORECASE),
    re.compile(r"\bauthorization\b", re.IGNORECASE),
    re.compile(r"\bbearer\s+\S+", re.IGNORECASE),
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"/jobs/[A-Za-z0-9_-]+", re.IGNORECASE),
    re.compile(r"\bj[a-z0-9]{7,}\b", re.IGNORECASE),
)
#: What a redacted span is replaced with. A committed capability record that
#: contains this marker is a record whose own prose tripped a pattern, which the
#: tests treat as a defect rather than as successful sanitization.
REDACTION_MARKER = "[redacted]"

SUBMIT_ENTRY_POINT_PREFIX = "submit_"
QUANTIZE_ENTRY_POINT = "submit_quantize_job"
CAPABILITY_OBSERVATION_SOURCES = (
    "live_read_only_capability_query",
    "live_read_only_capability_query_transcribed_offline",
)
CAPABILITY_OBSERVATION_FIELDS = (
    "client_version",
    "device_count",
    "devices",
    "observation_date",
    "quantize_dtypes",
    "quantize_entry_point",
    "quantize_parameters",
    "quantize_signature",
    "source",
    "submit_entry_points",
)
CAPABILITY_COST = "none — read-only capability query, no job submitted"
#: The local policy dtype spellings mapped onto the SDK enum member names the
#: query observed. Nothing else in this module knows the SDK spelling.
QUANTIZE_DTYPE_BY_POLICY_DTYPE = {
    "int4": "INT4",
    "int8": "INT8",
    "int16": "INT16",
}

CAPABILITY_ESTABLISHES = (
    "that_the_public_client_at_the_recorded_version_exposes_a_quantize_entry_point",
    "that_its_weights_dtype_and_activations_dtype_arguments_accept_the_recorded_dtypes",
    "that_W8A16_and_W8A8_are_each_expressible_as_one_public_quantize_request",
    "that_the_three_plan_section_3_2_target_devices_were_live_on_the_observation_date",
    "that_those_devices_advertise_framework_qnn_and_htp_supports_fp16",
    "that_the_quantize_entry_point_exposes_no_separate_kv_cache_dtype_argument",
)
CAPABILITY_DOES_NOT_ESTABLISH = (
    "that_either_candidate_compiles",
    "that_the_compiler_accepts_this_graph_at_any_precision",
    "that_any_operator_or_the_whole_graph_is_placed_on_the_NPU",
    "any_latency_throughput_peak_memory_or_energy",
    "any_accuracy_perplexity_or_quality_delta",
    "that_a_W8_artifact_exists_anywhere_to_send_to_that_entry_point",
    "that_this_repository_can_run_a_quantize_job_which_needs_an_adapter_it_lacks",
    "anything_at_all_about_a_device_that_is_not_one_of_the_three_recorded_here",
)


def redact_private_text(value: str) -> str:
    """Return *value* with every private or URL-like span replaced.

    Redaction, not refusal: the capability record carries strings that came off
    an external service, and the useful behaviour for those is to publish the
    sanitized remainder rather than to drop the observation entirely.
    """

    redacted = value
    for pattern in CAPABILITY_PRIVATE_TEXT_PATTERNS:
        redacted = pattern.sub(REDACTION_MARKER, redacted)
    return redacted


def redact_document(value: Any, field: str = "capability record") -> Any:
    """Recursively redact every string in *value*, mapping keys included."""

    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            _require(
                isinstance(key, str),
                f"{field}: a record key must be a string, found {type(key).__name__}",
            )
            redacted[redact_private_text(str(key))] = redact_document(
                child, f"{field}.{key}"
            )
        return redacted
    if isinstance(value, (list, tuple)):
        return [redact_document(item, field) for item in value]
    if isinstance(value, str):
        return redact_private_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise W8EvidenceError(
        f"{field}: a capability record may only carry JSON scalars, lists, and "
        f"objects; found {type(value).__name__}. An unrecognized value is more "
        "likely to be a live service object than a datum, and a live service "
        "object is exactly what must not be written."
    )


def assert_no_private_text(value: Any, field: str = "capability record") -> None:
    """Refuse a document in which any private-looking span survived redaction."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            assert_no_private_text(str(key), f"{field} key")
            assert_no_private_text(child, f"{field}.{key}")
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_no_private_text(item, field)
    elif isinstance(value, str):
        for pattern in CAPABILITY_PRIVATE_TEXT_PATTERNS:
            _require(
                pattern.search(value) is None,
                f"{field}: private or URL-like text survived redaction "
                f"(pattern {pattern.pattern!r}); the record was not written",
            )


def observe_capability(
    client: Any,
    *,
    client_version: str,
    observation_date: str,
    source: str = "live_read_only_capability_query",
) -> dict[str, Any]:
    """Read the public capability surface off *client* and submit nothing.

    This function reads names, one signature, and the public device list. It
    calls exactly one client function, ``get_devices``, and no ``submit_*``
    function ever — not even by attribute access. The entry-point list comes
    from :func:`dir`, which reads names without fetching values, and the quantize
    signature from :func:`inspect.getattr_static`, which reads the object's
    ``__dict__`` without triggering ``__getattr__``, a property, or any other
    descriptor that could have a side effect on the way past.

    It takes the client as an argument rather than importing one, so the
    guarantee above is testable against a fake that raises when a ``submit_*``
    attribute is touched, with no network and no SDK installed.
    """

    _require(
        source in CAPABILITY_OBSERVATION_SOURCES,
        f"unknown capability observation source {source!r}; expected one of "
        f"{CAPABILITY_OBSERVATION_SOURCES}",
    )
    entry_points = sorted(
        name for name in dir(client) if name.startswith(SUBMIT_ENTRY_POINT_PREFIX)
    )
    _require(
        QUANTIZE_ENTRY_POINT in entry_points,
        f"the client exposes no {QUANTIZE_ENTRY_POINT}; the W8 candidates "
        "cannot be expressed as public quantize requests and this observation "
        "would say the opposite of what it found",
    )
    try:
        quantize = inspect.getattr_static(client, QUANTIZE_ENTRY_POINT)
    except AttributeError as exc:
        raise W8EvidenceError(
            f"{QUANTIZE_ENTRY_POINT} is not a plain attribute of the client; "
            "refusing to fetch it through the descriptor protocol, which is the "
            "one path that could have a side effect"
        ) from exc
    try:
        signature = inspect.signature(quantize)
    except (TypeError, ValueError) as exc:
        raise W8EvidenceError(
            f"cannot read the {QUANTIZE_ENTRY_POINT} signature: {exc}"
        ) from exc

    dtype_enum = getattr(client, "QuantizeDtype", None)
    members = getattr(dtype_enum, "__members__", None)
    _require(
        isinstance(members, Mapping) and bool(members),
        "the client exposes no QuantizeDtype members, so the dtypes a quantize "
        "request may name are unknown and must not be assumed",
    )
    assert isinstance(members, Mapping)

    devices = list(client.get_devices())
    observed: list[dict[str, Any]] = []
    for device in devices:
        name = str(getattr(device, "name", ""))
        if name not in PLAN_TARGET_DEVICE_NAMES:
            continue
        attributes = getattr(device, "attributes", ())
        observed.append(
            {
                "name": name,
                "os": str(getattr(device, "os", "")),
                "attributes": sorted(str(item) for item in attributes),
            }
        )
    observed.sort(key=lambda entry: str(entry["name"]))
    missing = sorted(
        set(PLAN_TARGET_DEVICE_NAMES) - {entry["name"] for entry in observed}
    )
    _require(
        not missing,
        f"the service did not list plan section 3.2 target device(s) "
        f"{', '.join(missing)}. A capability record that silently omits a target "
        "reads as evidence the target is fine; re-check the device list and the "
        "plan before recording a partial observation.",
    )

    return {
        "source": source,
        "observation_date": observation_date,
        "client_version": client_version,
        "submit_entry_points": entry_points,
        "quantize_entry_point": QUANTIZE_ENTRY_POINT,
        "quantize_signature": str(signature),
        "quantize_parameters": list(signature.parameters),
        "quantize_dtypes": sorted(str(name) for name in members),
        "device_count": len(devices),
        "devices": observed,
    }


def _installed_client_version() -> str:
    """Return the exact installed ``qai-hub`` distribution version."""

    from importlib import metadata

    try:
        return metadata.version("qai-hub")
    except metadata.PackageNotFoundError as exc:
        raise W8EvidenceError(
            "the qai-hub distribution is not installed, so its exact version "
            "cannot be recorded and this repository will not guess one"
        ) from exc


def query_live_capability(*, observation_date: str) -> dict[str, Any]:
    """Import the client, read its public capability surface, and stop.

    ``qai_hub`` is imported here and nowhere else in this module. Importing it at
    module scope would make ``slm_lab.quantization.w8`` unimportable on every
    host that does not carry the SDK, and every other command in this module —
    ``generate``, ``check``, ``status``, ``record``, ``compare``, ``request`` —
    is offline by construction and must stay that way.
    """

    try:
        import qai_hub  # noqa: PLC0415 - deliberately not a module-scope import
    except ImportError as exc:
        raise W8EvidenceError(
            "the qai-hub client is not importable here. This repository pins no "
            "Qualcomm client in pyproject.toml on purpose, so the live query "
            f"runs from an environment that carries one. Rebuild the record "
            f"offline instead: {CAPABILITIES_OFFLINE_COMMAND}"
        ) from exc
    return observe_capability(
        qai_hub,
        client_version=_installed_client_version(),
        observation_date=observation_date,
    )


def validate_capability_observation(observation: Any) -> dict[str, Any]:
    """Return a shape-checked copy of one capability observation."""

    _require(
        isinstance(observation, Mapping),
        "a capability observation must be a JSON object",
    )
    assert isinstance(observation, Mapping)
    missing = sorted(set(CAPABILITY_OBSERVATION_FIELDS) - set(observation))
    extra = sorted(set(observation) - set(CAPABILITY_OBSERVATION_FIELDS))
    _require(
        not missing and not extra,
        f"a capability observation records exactly "
        f"{list(CAPABILITY_OBSERVATION_FIELDS)}; missing {missing}, unexpected "
        f"{extra}",
    )
    _require(
        observation["source"] in CAPABILITY_OBSERVATION_SOURCES,
        f"capability observation source {observation['source']!r} is not one of "
        f"{CAPABILITY_OBSERVATION_SOURCES}",
    )
    date = observation["observation_date"]
    _require(
        isinstance(date, str) and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)),
        f"observation_date must be an ISO calendar date, found {date!r}",
    )
    _require(
        _is_exact_version(observation["client_version"]),
        "a capability observation must name the exact client version it was "
        f"taken with, found {observation['client_version']!r}",
    )
    for field in ("submit_entry_points", "quantize_dtypes", "quantize_parameters"):
        value = observation[field]
        _require(
            isinstance(value, list)
            and bool(value)
            and all(isinstance(item, str) and item for item in value),
            f"{field} must be a non-empty list of strings",
        )
    _require(
        observation["quantize_entry_point"] == QUANTIZE_ENTRY_POINT,
        f"the quantize entry point must be {QUANTIZE_ENTRY_POINT!r}, found "
        f"{observation['quantize_entry_point']!r}",
    )
    _require(
        QUANTIZE_ENTRY_POINT in observation["submit_entry_points"],
        f"{QUANTIZE_ENTRY_POINT} is not among the observed entry points",
    )
    _require(
        all(
            name.startswith(SUBMIT_ENTRY_POINT_PREFIX)
            for name in observation["submit_entry_points"]
        ),
        f"every observed entry point must start with {SUBMIT_ENTRY_POINT_PREFIX!r}",
    )
    signature = observation["quantize_signature"]
    _require(
        isinstance(signature, str) and bool(signature.strip()),
        "the quantize signature must be the string the client reported",
    )
    for parameter in observation["quantize_parameters"]:
        _require(
            parameter in signature,
            f"the recorded parameter {parameter!r} does not appear in the "
            "recorded signature; the two came from different observations",
        )
    count = observation["device_count"]
    _require(
        isinstance(count, int) and not isinstance(count, bool) and count > 0,
        f"device_count must be a positive integer, found {count!r}",
    )
    devices = observation["devices"]
    _require(
        isinstance(devices, list) and len(devices) == len(PLAN_TARGET_DEVICE_NAMES),
        f"a capability observation records exactly the "
        f"{len(PLAN_TARGET_DEVICE_NAMES)} plan section 3.2 target devices",
    )
    assert isinstance(devices, list)
    for device in devices:
        _require(
            isinstance(device, Mapping) and set(device) == {"name", "os", "attributes"},
            "every observed device records exactly name, os, and attributes",
        )
        assert isinstance(device, Mapping)
        _require(
            device["name"] in PLAN_TARGET_DEVICE_NAMES,
            f"observed device {device['name']!r} is not a plan section 3.2 "
            "target; this record carries the targets and nothing else",
        )
        _require(
            isinstance(device["os"], str) and bool(device["os"]),
            f"{device['name']}: os must be the SDK version string",
        )
        attributes = device["attributes"]
        _require(
            isinstance(attributes, list)
            and bool(attributes)
            and all(isinstance(item, str) and item for item in attributes),
            f"{device['name']}: attributes must be a non-empty list of strings",
        )
        _require(
            list(attributes) == sorted(attributes),
            f"{device['name']}: attributes must be sorted so the record is a "
            "fixed point rather than a service-order snapshot",
        )
    _require(
        {str(device["name"]) for device in devices} == set(PLAN_TARGET_DEVICE_NAMES),
        "the observed devices do not cover the plan section 3.2 targets",
    )
    _require(
        count >= len(devices),
        "device_count is smaller than the number of recorded devices",
    )
    return {field: observation[field] for field in CAPABILITY_OBSERVATION_FIELDS}


def build_candidate_quantize_requests(
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Map each W8 candidate onto the one public quantize request it needs.

    The dtypes are read off the candidate policies rather than transcribed, and
    each one is checked against the dtype list the service actually offered, so
    a service that withdraws INT16 makes this fail rather than leaving a config
    that promises a request the API can no longer express.
    """

    available = list(observation["quantize_dtypes"])
    by_candidate: dict[str, Any] = {}
    used: set[str] = set()
    for candidate_id in CANDIDATE_IDS:
        weights_policy = "int8"
        activations_policy = str(
            CANDIDATE_DEFINITIONS[candidate_id]["activations"]["dtype"]
        )
        pair: dict[str, Any] = {}
        for role, policy_dtype in (
            ("weights_dtype", weights_policy),
            ("activations_dtype", activations_policy),
        ):
            sdk_dtype = QUANTIZE_DTYPE_BY_POLICY_DTYPE.get(policy_dtype)
            _require(
                sdk_dtype is not None,
                f"{candidate_id}: policy dtype {policy_dtype!r} has no known "
                "QuantizeDtype spelling",
            )
            _require(
                sdk_dtype in available,
                f"{candidate_id}: the observed client offers "
                f"{available} and not {sdk_dtype}, so this candidate is no "
                "longer expressible as a public quantize request",
            )
            pair[role] = sdk_dtype
            used.add(str(sdk_dtype))
        by_candidate[candidate_id] = {
            "plan_matrix_row": PLAN_MATRIX_ROWS[candidate_id],
            **pair,
            "request": (
                f"{QUANTIZE_ENTRY_POINT}(weights_dtype=QuantizeDtype."
                f"{pair['weights_dtype']}, activations_dtype=QuantizeDtype."
                f"{pair['activations_dtype']})"
            ),
        }
    return {
        "entry_point": QUANTIZE_ENTRY_POINT,
        "by_candidate": by_candidate,
        "observed_dtypes": available,
        "dtypes_not_used_by_T41": sorted(set(available) - used),
        "int4_note": (
            "INT4 is offered by the same two arguments and is not T41's to use: "
            "the W4A8 row belongs to T42, which is where a four-bit weight "
            "policy, its blocking granularity, and its sensitivity evidence are "
            "argued. Recorded here only so T42 does not have to re-observe it."
        ),
        "note": (
            "Expressible means the public API accepts these two arguments. It is "
            "not compiler acceptance, not operator support, not NPU placement, "
            "and not a claim that an artifact exists to pass as the model."
        ),
    }


def build_kv_cache_dtype_finding(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Record that the public quantize request has no cache dtype argument."""

    parameters = [str(name) for name in observation["quantize_parameters"]]
    cache_parameters = [name for name in parameters if "cache" in name.lower()]
    _require(
        not cache_parameters,
        f"the observed quantize signature now takes {cache_parameters}, which "
        "may be a cache dtype knob. The finding below is written for a signature "
        "that has none; re-argue it against the new signature rather than "
        "letting this record keep the old sentence.",
    )
    return {
        "separate_cache_dtype_argument": False,
        "quantize_parameters": parameters,
        "statement": (
            "The public quantize entry point takes one weights dtype and one "
            "activations dtype, and no argument that names the KV cache. This is "
            "consistent with the frozen float16 CACHE_DTYPE finding already in "
            "both candidates and is not a way around it: the Q2 row's int8 cache "
            "would still have to come from the T12 graph contract and the "
            "T20/T23 export boundary, because there is no quantize option that "
            "could produce one."
        ),
    }


def build_capability_record(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Build the sanitized, committed AI Hub capability record.

    The record carries no wall-clock generation stamp and no repository commit,
    on purpose: it describes the public service on ``observation_date``, not this
    tree, and leaving both out is what makes it regenerate byte-identically from
    its own ``observation`` block on any later commit.
    """

    checked = validate_capability_observation(observation)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "record_type": CAPABILITY_RECORD_TYPE,
        "observation_date": checked["observation_date"],
        "client_version": checked["client_version"],
        "submitted_jobs": 0,
        "device_minutes_consumed": 0,
        "cost": CAPABILITY_COST,
        "query_scope": {
            "operation": "read_only_capability_query",
            "read": [
                "the exact installed client version",
                "the names of every submit_* entry point",
                "the quantize entry point's signature",
                "the QuantizeDtype member names",
                "the number of devices the service lists",
                "name, os, and attributes for the plan section 3.2 targets only",
            ],
            "submit_functions_called": [],
            "jobs_created": 0,
            "models_uploaded": 0,
            "devices_leased": 0,
            "statement": (
                "The query listed names and read one signature. It called one "
                "service function, get_devices, which returns public device "
                "metadata. No submit_* function was called and none was even "
                "fetched through the descriptor protocol: the entry-point names "
                "come from dir() and the signature from inspect.getattr_static, "
                "which reads __dict__ directly. No job was created, no model was "
                "uploaded, and no device was leased."
            ),
        },
        "reproduction_command": CAPABILITIES_COMMAND,
        "reproduction": {
            "live_command": CAPABILITIES_COMMAND,
            "offline_command": CAPABILITIES_OFFLINE_COMMAND,
            "requires": (
                "an authenticated qai-hub client at the recorded version. This "
                "repository pins no Qualcomm client in pyproject.toml, so the "
                "live command does not run here; it ran in an environment that "
                "carried one and the sanitized result was carried in."
            ),
            "offline_note": (
                "The offline command rebuilds this file from the observation "
                "block below, so the record is its own saved query and "
                "regenerates without a network. `check` recomputes it on every "
                "run and fails if the two disagree."
            ),
            "network": "the live command reads the service; the offline command does not",
        },
        "observation": checked,
        "candidate_quantize_requests": build_candidate_quantize_requests(checked),
        "kv_cache_dtype": build_kv_cache_dtype_finding(checked),
        "establishes": list(CAPABILITY_ESTABLISHES),
        "does_not_establish": list(CAPABILITY_DOES_NOT_ESTABLISH),
        "boundary": (
            "This is an observation of a public API surface on one date with one "
            "client, and that is the whole of it. It moves Lane A's quantization "
            "support from an assumption to a dated fact and moves nothing else: "
            "no weight is quantized, no graph is compiled, no operator is placed, "
            "and no number here was measured on a device. What still holds Lane A "
            "is that this repository has no quantize-stage adapter and the module "
            "that would own one belongs to T22."
        ),
    }
    redacted = redact_document(record)
    assert_no_private_text(redacted)
    assert isinstance(redacted, dict)
    return redacted


def load_offline_observation(path: Path | str) -> dict[str, Any]:
    """Load one saved sanitized query, from a bare observation or a record.

    A committed capability record already contains its own ``observation`` block,
    so the record is the saved query and ``--offline-input`` pointed at it
    rebuilds the file it came from.
    """

    document = _load_json(Path(path))
    _require(
        isinstance(document, Mapping),
        f"{path}: an offline capability input must be a JSON object",
    )
    assert isinstance(document, Mapping)
    if "observation" in document:
        _require(
            document.get("record_type") == CAPABILITY_RECORD_TYPE,
            f"{path}: this document carries an observation block but is not a "
            f"{CAPABILITY_RECORD_TYPE}",
        )
        return validate_capability_observation(document["observation"])
    return validate_capability_observation(document)


def default_capability_path(repo_root: Path, record: Mapping[str, Any]) -> Path:
    """Return the conventional evidence path for one capability record."""

    day = str(record["observation_date"])
    return (
        repo_root / DEFAULT_EVIDENCE_DIRECTORY / f"{CAPABILITY_RECORD_PREFIX}{day}.json"
    )


def write_capability_record(path: Path, record: Mapping[str, Any]) -> None:
    """Write one sanitized capability record, refusing to write private text."""

    assert_no_private_text(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_capability_record(repo_root: Path) -> dict[str, Any]:
    """Load and re-derive the committed capability record, failing closed.

    The record is not merely read. It is rebuilt from its own observation block
    and compared, so an edited claim, an edited boundary list, or an edited
    device attribute is a failure rather than a quietly re-anchored citation.
    """

    path = Path(repo_root) / DEFAULT_CAPABILITY_RECORD
    _require(
        path.is_file(),
        f"the AI Hub capability record is missing: {path}. Both candidate "
        f"specifications cite it, so it is a required input; rebuild it with: "
        f"{CAPABILITIES_OFFLINE_COMMAND}",
    )
    document = _load_json(path)
    _require(
        isinstance(document, Mapping),
        f"{path.name}: the capability record is not a JSON object",
    )
    assert isinstance(document, Mapping)
    _require(
        document.get("record_type") == CAPABILITY_RECORD_TYPE,
        f"{path.name}: record_type is not {CAPABILITY_RECORD_TYPE!r}",
    )
    _require(
        document.get("observation_date") == CAPABILITY_OBSERVATION_DATE,
        f"{path.name}: observation_date is "
        f"{document.get('observation_date')!r} but this module is pinned to "
        f"{CAPABILITY_OBSERVATION_DATE!r}. A new observation writes a new dated "
        "file and moves CAPABILITY_OBSERVATION_DATE with it.",
    )
    _require(
        document.get("submitted_jobs") == 0
        and document.get("device_minutes_consumed") == 0,
        f"{path.name}: this record claims a submitted job or consumed device "
        "minutes. A capability observation submits nothing; a record that says "
        "otherwise is a job record and does not belong on this path.",
    )
    rebuilt = build_capability_record(document.get("observation"))
    _require(
        dict(document) == rebuilt,
        f"{path.name}: the committed capability record is not the record its "
        f"own observation derives; rebuild it with: "
        f"{CAPABILITIES_OFFLINE_COMMAND}",
    )
    expected = json.dumps(rebuilt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _require(
        path.read_text(encoding="utf-8") == expected,
        f"{path.name} is not byte-identical to a fresh regeneration; run: "
        f"{CAPABILITIES_OFFLINE_COMMAND}",
    )
    return rebuilt


# ---------------------------------------------------------------------------
# Candidate specifications
# ---------------------------------------------------------------------------

CANDIDATE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "w8a16": {
        "summary": (
            "Conservative post-training quantization (plan section 7.2, Q1): "
            "int8 weights, int16 activations, KV cache left at the frozen "
            "float16 T12 contract."
        ),
        "intent": (
            "Answers one question: is the weight quantizer itself safe on this "
            "model? Activations stay at 16 bits and the cache is untouched, so "
            "any quality movement this candidate shows is attributable to "
            "int8 weight rounding rather than to activation clipping or cache "
            "precision. It is the candidate that isolates the variable, and "
            "the one to reach for if the aggressive candidate degrades and the "
            "cause has to be split."
        ),
        "activations": {
            "dtype": "int16",
            "bits": 16,
            "signed": True,
            "symmetric": True,
            "zero_point": 0,
            "granularity": "per_tensor",
            "range_estimator": "min_max",
            "range_estimator_reason": (
                "At 16 bits a symmetric per-tensor range has 32,768 positive "
                "levels, so covering the full observed range costs very little "
                "resolution. A decoder-only transformer's residual stream "
                "concentrates its largest magnitudes in a small number of "
                "channels that grow with depth; clipping one of those is a "
                "large error on the accumulator every later layer reads, while "
                "the resolution given up to avoid clipping is roughly one part "
                "in ten thousand. At this bit width the trade is one-sided, so "
                "the estimator does not clip at all."
            ),
            "rejected_range_estimators": [
                {
                    "id": "percentile",
                    "reason": (
                        "Buys resolution that int16 does not need and pays for "
                        "it by clipping exactly the outlier channels the "
                        "residual stream depends on."
                    ),
                },
                {
                    "id": "mse",
                    "reason": (
                        "Optimizes a reconstruction objective that is nearly "
                        "flat at 16 bits, so it adds a fitted hyper-parameter "
                        "and a histogram dependency for no expected gain."
                    ),
                },
            ],
            "sensitive_tensors": [
                {
                    "id": "residual_stream",
                    "risk": (
                        "Highest dynamic range in the graph and the tensor "
                        "every block both reads and writes."
                    ),
                },
                {
                    "id": "attention_scores_pre_softmax",
                    "risk": (
                        "Unbounded below and scaled by 1/sqrt(head_dim); its "
                        "range grows with sequence length, so a range fitted "
                        "at S128 does not bound S4096."
                    ),
                },
            ],
            "boundary": (
                "int16 activations are specified here as the Q1 row requires. "
                "Whether the public Workbench compiler accepts a 16-bit "
                "activation datapath for this graph on the named targets is "
                "unverified in this repository: no compile job has been "
                "submitted for any Qwen graph, quantized or floating."
            ),
        },
        "cache_requirement": "FP16/INT16",
        "cache_satisfied_without_contract_change": True,
        "cache_argument": (
            "The Q1 row admits an FP16 cache outright, so this candidate is "
            "fully specifiable against the frozen T12 contract as it stands. "
            "No contract change is requested and none is needed."
        ),
    },
    "w8a8": {
        "summary": (
            "Aggressive post-training quantization (plan section 7.2, Q2): "
            "int8 weights, int8 activations, KV cache left at the frozen "
            "float16 T12 contract because lowering it is not T41's to change."
        ),
        "intent": (
            "Answers the question that decides whether the accelerator's "
            "integer MAC array is usable for this model at all: does the "
            "activation path survive 8 bits? This is where decoder-only "
            "transformers usually break first, because a per-tensor int8 scale "
            "has to cover a residual stream whose largest channels are orders "
            "of magnitude above its typical ones. A quality result from this "
            "candidate is only interpretable next to the conservative "
            "candidate: together they separate weight error from activation "
            "error, and neither does so alone."
        ),
        "activations": {
            "dtype": "int8",
            "bits": 8,
            "signed": True,
            "symmetric": True,
            "zero_point": 0,
            "granularity": "per_tensor",
            "range_estimator": "mse",
            "range_estimator_reason": (
                "At 8 bits a symmetric per-tensor range has 128 positive "
                "levels, so the min-max choice that is free at int16 becomes "
                "the dominant error: one outlier channel sets a scale that "
                "collapses the bulk of the distribution onto a handful of "
                "levels. The estimator therefore has to trade clipping error "
                "against resolution error explicitly. An MSE-minimizing search "
                "over candidate thresholds does that with a stated objective "
                "and no hand-chosen constant, which matters here because no "
                "activation histogram has been measured anywhere in this "
                "repository and a hand-picked percentile would be a guess "
                "wearing a number."
            ),
            "rejected_range_estimators": [
                {
                    "id": "min_max",
                    "reason": (
                        "Guarantees no clipping and guarantees that a single "
                        "outlier channel destroys the resolution of every "
                        "other channel in the tensor. At int8 that is the "
                        "worse failure."
                    ),
                },
                {
                    "id": "percentile",
                    "reason": (
                        "Workable, but requires committing to a percentile "
                        "constant before any histogram exists. Kept as the "
                        "fallback if the MSE search proves unstable on the "
                        "attention tensors."
                    ),
                },
            ],
            "sensitive_tensors": [
                {
                    "id": "residual_stream",
                    "risk": (
                        "The int8 activation path's primary failure mode. If "
                        "this candidate degrades, the residual stream is the "
                        "first place to grant a per-tensor exception."
                    ),
                },
                {
                    "id": "attention_probabilities_post_softmax",
                    "risk": (
                        "Lives in [0, 1] with a long thin tail. A symmetric "
                        "int8 quantizer spends half its range on values that "
                        "cannot occur and rounds the small probabilities that "
                        "carry the long-context contribution to zero."
                    ),
                },
                {
                    "id": "mlp_down_proj_input",
                    "risk": (
                        "The SwiGLU product after the SiLU gate, reduced over "
                        "the widest axis in the block, so activation "
                        "quantization error accumulates over the longest dot "
                        "product here."
                    ),
                },
            ],
            "boundary": (
                "Nothing in this repository has measured any of these ranges. "
                "The sensitivity ordering above is an argument from the "
                "architecture, not a measurement, and the first observer pass "
                "is entitled to contradict it."
            ),
        },
        "cache_requirement": "INT8 or supported type",
        "cache_satisfied_without_contract_change": False,
        "cache_argument": (
            "The Q2 row asks for an int8 cache, and this candidate does not "
            "deliver one. The cache dtype is not a knob T41 owns: "
            "slm_lab.contracts.static_cache freezes CACHE_DTYPE as float16, and "
            "that dtype is not merely internal — it is the declared dtype of "
            "56 key_cache/value_cache graph outputs on every prefill graph and "
            "56 graph inputs plus 56 outputs on every decode graph in the T20 "
            "and T23 exports, and it is asserted by the committed manifests "
            "under results/manifests/onnx/. Lowering it would change a frozen "
            "T12 public contract and re-open the export boundary, which is why "
            "this candidate specifies float16 and files the int8-cache variant "
            "as an out-of-scope change request against its owners rather than "
            "quietly writing int8 here."
        ),
    },
}

EXCLUSION_ENTRIES: tuple[dict[str, Any], ...] = (
    {
        "id": "tied_embedding_table",
        "applies_to": "the single vocabulary x hidden table",
        "kind": "frozen_graph_constraint",
        "precision_kept": "float16",
        "reason": (
            "configs/models/qwen3-0.6b.yaml sets tie_word_embeddings: true, so "
            "the embedding lookup and the final logits projection are the same "
            "stored tensor. They cannot be given two different precisions "
            "without untying them, which changes the model contract and the "
            "exported graph. This exclusion is expensive — the table is over a "
            "quarter of all parameters — and the projection states the cost "
            "rather than hiding it."
        ),
    },
    {
        "id": "final_logits_projection",
        "applies_to": "the output projection that produces last_logits/next_logits",
        "kind": "frozen_graph_constraint",
        "precision_kept": "float16",
        "reason": (
            "At this model revision this is the tied table above, recorded "
            "separately so that an untied future revision does not silently "
            "inherit an exclusion it was never argued for. The T12 contract "
            "also declares the logits tensors float32, so the last matmul "
            "already sits on a wider output path than the rest of the graph."
        ),
    },
    {
        "id": "rmsnorm_scales",
        "applies_to": "input_layernorm, post_attention_layernorm, and the final norm",
        "kind": "policy_choice",
        "precision_kept": "float16",
        "reason": (
            "These vectors multiply the normalized residual channel by "
            "channel, so an int8 rounding error on them is a multiplicative "
            "error applied to the accumulator at every block. They are a "
            "vanishing fraction of the parameters, so the saving is "
            "negligible and the risk is not. The T21 inventory records 113 "
            "ReduceMean nodes in the S128 prefill graph, which is four per "
            "block plus the final norm — this class and the per-head norms "
            "below account for all of them."
        ),
    },
    {
        "id": "qwen3_per_head_qk_norm",
        "applies_to": "Qwen3's per-head q_norm and k_norm scale vectors",
        "kind": "policy_choice",
        "precision_kept": "float16",
        "reason": (
            "Qwen3 normalizes each attention head's query and key before RoPE, "
            "so these vectors act directly on the tensors whose dot product "
            "becomes the attention score. They are also the smallest weights "
            "in the model: the T21 inventory shows they are the 56 "
            "initializers the exporter left inline in the protobuf because "
            "each is only head_dim float16 values. Quantizing them would "
            "perturb attention scores to save a few kilobytes."
        ),
    },
    {
        "id": "softmax",
        "applies_to": "the 28 attention softmax nodes",
        "kind": "policy_choice",
        "precision_kept": "float16",
        "reason": (
            "The input is unbounded below and the output is a probability "
            "distribution whose small entries are what long-context attention "
            "is made of. A symmetric integer quantizer on either side either "
            "wastes half its range or rounds the tail to zero, and the "
            "normalization itself is what makes the operator numerically "
            "well behaved in the first place."
        ),
    },
    {
        "id": "rope_sin_cos",
        "applies_to": "the RoPE sine and cosine tables",
        "kind": "policy_choice",
        "precision_kept": "float16",
        "reason": (
            "They are exact trigonometric constants that encode position. "
            "Quantizing them perturbs the phase applied at every position, "
            "with an error that grows with position index — precisely the "
            "regime the S4096 workload exercises hardest. The T21 inventory "
            "shows one Cos and one Sin node, so the tables are computed once "
            "and cost effectively nothing to keep."
        ),
    },
    {
        "id": "residual_adds",
        "applies_to": "the residual stream additions around attention and the MLP",
        "kind": "policy_choice",
        "precision_kept": "float16",
        "reason": (
            "The residual stream is the model's accumulator. Re-quantizing it "
            "at every block turns a single rounding error into a compounding "
            "one along all 28 layers, and it is the tensor with the widest "
            "dynamic range in the graph. Keeping the adds wide is the standard "
            "way to stop activation quantization error from accumulating with "
            "depth."
        ),
    },
    {
        "id": "kv_cache_read_write",
        "applies_to": "cache reads, the indexed cache write, and the cache boundary",
        "kind": "frozen_graph_constraint",
        "precision_kept": "float16",
        "reason": (
            "slm_lab.contracts.static_cache freezes CACHE_DTYPE as float16 and "
            "the cache crosses the graph boundary as declared inputs and "
            "outputs. The T21 inventory records a single ScatterND in the "
            "prefill graph, which is the cache write; changing its dtype "
            "changes the public contract, not an internal choice. The owning "
            "tasks are named in this candidate's kv_cache block."
        ),
    },
)


def build_weight_policy(model_contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the shared int8 weight policy and what per-channel means here."""

    classes = derive_weight_classes(model_contract)
    return {
        "dtype": "int8",
        "bits": 8,
        "signed": True,
        "symmetric": True,
        "zero_point": 0,
        "granularity": "per_output_channel",
        "scale_dtype": "float32",
        "granularity_reason": (
            "Per-output-channel scales cost one float32 per output feature and "
            "remove the dominant source of int8 weight error: a per-tensor "
            "scale is set by whichever single output channel has the largest "
            "weights, and in this model the channel-wise magnitude spread "
            "inside one matrix is wide enough that a per-tensor scale throws "
            "away most of the range for most channels. Symmetric with a zero "
            "zero-point keeps the integer accumulation free of a cross term."
        ),
        "per_channel_axis_note": (
            "Per output channel means one scale per output feature of the "
            "logical [out, in] weight. Which ONNX axis that is in the exported "
            "artifact is not asserted here: torch.onnx.export emits these "
            "projections as MatMul with a possibly transposed initializer, and "
            "the committed T21 inventory records the node counts but not any "
            "per-initializer axis order. Read the axis off the artifact at "
            "quantization time; do not assume it from this file."
        ),
        "per_class": [
            {
                "class_id": weight_class.class_id,
                "role": weight_class.role,
                "logical_shape_per_instance": list(weight_class.shape),
                "instances": weight_class.instances,
                "output_channels_per_instance": weight_class.output_channels,
                "scales_per_instance": weight_class.output_channels,
                "quantized": weight_class.exclusion_id is None,
                "exclusion_id": weight_class.exclusion_id,
                "note": weight_class.note,
            }
            for weight_class in classes
        ],
    }


def build_kv_cache_policy(candidate_id: str) -> dict[str, Any]:
    """Return the cache block, which is where this candidate meets a contract."""

    definition = CANDIDATE_DEFINITIONS[candidate_id]
    decode_contract = static_cache.build_decode_contract(
        min(static_cache.CONTEXT_VARIANTS)
    )
    cache_inputs = sum(
        1
        for spec in decode_contract.inputs
        if spec.name.startswith(("key_cache.", "value_cache."))
    )
    cache_outputs = sum(
        1
        for spec in decode_contract.outputs
        if spec.name.startswith(("present_key.", "present_value."))
    )
    return {
        "applied_dtype": static_cache.CACHE_DTYPE,
        "plan_row_requirement": definition["cache_requirement"],
        "satisfied_without_contract_change": definition[
            "cache_satisfied_without_contract_change"
        ],
        "argument": definition["cache_argument"],
        "frozen_contract": {
            "dtype": static_cache.CACHE_DTYPE,
            "source": "slm_lab.contracts.static_cache.CACHE_DTYPE",
            "owner_task": "T12",
            "decode_graph_cache_inputs": cache_inputs,
            "decode_graph_cache_outputs": cache_outputs,
            "layout": ["batch", "kv_head", "cache_position", "head_dim"],
            "note": (
                "Counted from the frozen contract rather than quoted: the "
                "cache dtype appears on this many declared graph tensors per "
                "decode variant, which is what makes it a public boundary "
                "rather than an internal representation."
            ),
        },
        "change_control": {
            "lowering_the_cache_dtype_changes": [
                "the frozen T12 static-cache contract (CACHE_DTYPE)",
                "the T20 exported prefill cache outputs and decode cache "
                "inputs/outputs",
                "the T23 promoted prefill export and its committed manifests",
                "results/manifests/onnx/S*.json, which assert float16 on every "
                "cache tensor",
            ],
            "owner_tasks": {
                "graph_contract": "T12",
                "onnx_export_boundary": "T20",
                "promoted_prefill_export": "T23",
            },
            "t41_position": (
                "T41 owns configs/quantization/w8/ and "
                "src/slm_lab/quantization/w8.py. It does not own T12, T20, or "
                "T23, so it cannot unilaterally change the cache dtype, and it "
                "will not specify a cache precision the exported graphs do not "
                "carry. This is a discovered constraint, recorded where it was "
                "discovered."
            ),
            "request_status": (
                "not_raised_at_this_commit"
                if definition["cache_satisfied_without_contract_change"]
                else "out_of_scope_change_request_for_T12_T20_T23"
            ),
        },
        "bytes": {
            "measurement": PROJECTION_MEASUREMENT,
            "by_context": _kv_cache_rows(),
            "note": (
                "Arithmetic from the frozen contract via "
                "slm_lab.contracts.static_cache.cache_bytes. It is here so a "
                "reader can see that W8 weight quantization does not touch the "
                "term that dominates memory at long context."
            ),
        },
    }


def build_calibration_binding(
    calibration: Mapping[str, Any],
    *,
    calibration_path: Path,
) -> dict[str, Any]:
    """Bind this candidate to the frozen T40 corpus and carry its bias."""

    corpus = calibration.get("calibration_corpus") or {}
    coverage = corpus.get("coverage") or {}
    budget = corpus.get("token_budget") or {}
    graph_note = (calibration.get("graph_contract") or {}).get("note")
    revision = calibration.get("calibration_dataset_revision")
    _require(
        isinstance(revision, str) and bool(revision),
        "the T40 calibration contract carries no calibration_dataset_revision",
    )
    shares = coverage.get("token_share_per_source_group") or {}
    dominant_group = max(shares, key=lambda key: shares[key]) if shares else None
    return {
        "calibration_dataset_revision": revision,
        "source": {
            "path": calibration_path.as_posix(),
            "corpus_canonical_json_sha256": calibration.get(
                "calibration_corpus_canonical_json_sha256"
            ),
            "owner_task": "T40",
            "note": (
                "Read at generation time from the committed T40 contract, "
                "never transcribed. A second literal copy of this revision "
                "string is a second place for it to drift."
            ),
        },
        "observe_ranges_on": {
            "graph_kind": "prefill",
            "variant_ids": [
                f"S{prompt_length}"
                for prompt_length in sorted(static_cache.CONTEXT_VARIANTS)
            ],
            "reason": (
                "All four exported prefill contexts, because activation ranges "
                "fitted at 128 tokens do not bound those at 4096 and the "
                "longest context is the one the benchmark exercises hardest."
            ),
        },
        "decode_side_observer_pass": {
            "open_question_from": "T40",
            "question": graph_note,
            "t41_position": "required_pending_measurement",
            "argument": (
                "A decode step reads a materialized cache and produces one "
                "token's activations at cache position valid_length. Neither "
                "the attention denominator over a full cache nor the residual "
                "magnitude at that position is observed by any prefill sample, "
                "so prefill ranges do not bound decode ranges by construction. "
                "For the int8-activation candidate that gap is the difference "
                "between a fitted scale and a guessed one; for the "
                "int16-activation candidate it is a smaller risk at 256 times "
                "the resolution."
            ),
            "evidence": "none_measured_at_this_commit",
            "blocked_on": ["hardware:linux_cuda_aimet_host"],
            "owner_task": "T41",
        },
        "inherited_bias": {
            "from_task": "T40",
            "total_calibration_tokens": budget.get("total_calibration_tokens"),
            "dominant_source_group": dominant_group,
            "dominant_source_group_token_share": (
                shares.get(dominant_group) if dominant_group else None
            ),
            "distinct_token_ids": coverage.get("distinct_token_ids"),
            "model_vocabulary_size": coverage.get("model_vocabulary_size"),
            "vocabulary_fraction": coverage.get("vocabulary_fraction"),
            "evaluation_overlap": (calibration.get("licensing") or {}).get(
                "evaluation_overlap"
            ),
            "statement": (
                "The corpus is small and narrow: most of its token positions "
                "come from one repeated authored seed, it touches a fraction "
                "of a percent of the embedding table, and four of its samples "
                "are also evaluation prompts. Any quality delta measured "
                "against this calibration is optimistic and must be reported "
                "with this block attached."
            ),
        },
    }


def build_baseline_binding(
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind the float16 comparison anchor to the four committed manifests."""

    rows = []
    for variant_id in sorted(manifests, key=lambda name: int(name[1:])):
        manifest = manifests[variant_id]
        rows.append(
            {
                "variant_id": variant_id,
                "path": (DEFAULT_MANIFEST_DIRECTORY / f"{variant_id}.json").as_posix(),
                "canonical_json_sha256": canonical_json_sha256(manifest),
                "precision": manifest.get("precision"),
                "task_id": manifest.get("task_id"),
                "context_length": manifest.get("context_length"),
                "cache_capacity": manifest.get("cache_capacity"),
                "source_artifact_sha256": manifest.get("source_artifact_sha256"),
            }
        )
    return {
        "role": "floating comparison anchor for every W8 delta",
        "precision": "float16",
        "manifests": rows,
        "statement": (
            "A W8 result is only a delta against these exact manifests. The "
            "canonical hashes are recomputed at generation time, so a baseline "
            "manifest that is edited after this file was written makes "
            f"`{CHECK_COMMAND}` fail rather than silently re-anchoring the "
            "comparison."
        ),
    }


def build_evaluation_binding(
    *,
    protocol: Mapping[str, Any],
    academic: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the quality delta to the frozen T13 protocol and academic suite."""

    tasks = [
        {
            "id": task["id"],
            "harness_task": task["harness_task"],
            "dataset_id": task["dataset_id"],
            "dataset_revision": task["dataset_revision"],
            "dataset_config": task.get("dataset_config"),
            "split": task["split"],
            "metrics": list(task["metrics"]),
        }
        for task in academic["tasks"]
    ]
    return {
        "benchmark_protocol": {
            "path": DEFAULT_BENCHMARK_PROTOCOL.as_posix(),
            "protocol_id": protocol["protocol_id"],
            "contract_sha256": protocol["contract_sha256"],
            "digest_source": "slm_lab.benchmark.protocol.protocol_sha256",
            "owner_task": "T13",
        },
        "academic_contract": {
            "path": DEFAULT_ACADEMIC_CONTRACT.as_posix(),
            "canonical_json_sha256": canonical_json_sha256(academic),
            "suite_id": academic["suite_id"],
            "prompt_interface": academic["prompt_interface"],
            "apply_chat_template": academic["apply_chat_template"],
            "fewshot": academic["fewshot"],
            "harness_release": academic["harness"]["release"],
            "harness_release_commit": academic["harness"]["release_commit"],
            "tasks": tasks,
        },
        "quality_delta_metrics": [
            {
                "task_id": task["id"],
                "metrics": task["metrics"],
                "role": (
                    "regression detector for the W8 delta; not a capability "
                    "claim about the model"
                ),
            }
            for task in tasks
        ],
        "comparison": {
            "function": "slm_lab.quantization.w8.compare_quality",
            "command": COMPARE_COMMAND,
            "precision_label_convention": {
                "baseline": list(FLOAT_BASELINE_PRECISIONS),
                "candidate": [
                    f"<candidate_id>{PRECISION_LABEL_SEPARATOR}{state}"
                    for state in MEASURED_PRECISION_STATES
                ],
                "cross_checked_against": "system.evidence_level",
                "reason": (
                    "The frozen T13 result schema sets additionalProperties: "
                    "false at the record root and inside source, so a W8 "
                    "result has nowhere to add a bespoke precision_state "
                    "field. The state rides on source.precision, which the "
                    "schema already requires, and is cross-checked against "
                    "system.evidence_level, which the schema already "
                    "constrains. Widening the schema would mean changing a "
                    "frozen T13 contract T41 does not own."
                ),
            },
        },
        "statement": (
            "A quality delta computed under any other protocol digest is not "
            "comparable with one computed under "
            f"{protocol['contract_sha256']}, and compare_quality refuses to "
            "subtract two records that do not both cite it."
        ),
    }


GRAPH_INVENTORY_OPS = (
    "MatMul",
    "Transpose",
    "Softmax",
    "Sigmoid",
    "ReduceMean",
    "Cos",
    "Sin",
    "Gather",
    "ScatterND",
)


def build_graph_inventory_binding(
    graph_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the exclusion policy to the committed T21 structural inventory."""

    prefill = (graph_record.get("graphs") or {}).get("prefill")
    _require(
        isinstance(prefill, Mapping),
        f"{DEFAULT_GRAPH_INVENTORY} carries no prefill graph inventory",
    )
    assert isinstance(prefill, Mapping)
    histogram = prefill.get("op_histogram") or {}
    return {
        "source": {
            "path": DEFAULT_GRAPH_INVENTORY.as_posix(),
            "canonical_json_sha256": canonical_json_sha256(graph_record),
            "owner_task": graph_record.get("task_id"),
            "variant_id": graph_record.get("variant_id"),
            "graph_kind": "prefill",
            "precision": graph_record.get("precision"),
            "onnx_sha256": prefill.get("source_sha256"),
        },
        "node_count": prefill.get("node_count"),
        "initializer_count": prefill.get("initializer_count"),
        "external_initializer_count": prefill.get("external_initializer_count"),
        "inline_initializer_count": int(prefill["initializer_count"])
        - int(prefill["external_initializer_count"]),
        "largest_inline_initializer_bytes": prefill.get(
            "largest_inline_initializer_bytes"
        ),
        "op_counts": [
            {"op": op, "count": histogram[op]}
            for op in GRAPH_INVENTORY_OPS
            if op in histogram
        ],
        "why_this_is_here": (
            "The exclusion policy below argues about operators and tensors. "
            "This block is the committed structural evidence those arguments "
            "point at, so a reader can check that the operators being excluded "
            "are operators this graph actually contains, and in what number. "
            "It is a structural inventory of one float16 ONNX file; it "
            "establishes nothing about compiler acceptance or placement."
        ),
    }


#: What still holds Lane A now that submission is permitted. The token keeps the
#: repository-wide ``prefix:token`` blocker shape and the owning task is named
#: beside it in ``blocked_on_owners`` rather than inside the string, the way
#: every ledger row already carries an ``owner_task``.
LANE_A_CAPABILITY_BLOCKER = "capability:no_quantize_stage_adapter_in_this_repository"
#: The two independent reasons no W8 artifact exists to compile, profile, or
#: measure. Neither is a permission and neither is resolvable inside T41.
NO_W8_ARTIFACT_BLOCKERS = (
    LANE_A_CAPABILITY_BLOCKER,
    "hardware:linux_cuda_aimet_host",
)
#: Who would have to act to clear each Lane A blocker. ``ai_hub.py`` sits under
#: ``src/slm_lab/deployment/qualcomm/``, an owned path of T22, so T41 may not
#: add a quantize stage to it while that task is open.
LANE_A_BLOCKER_OWNERS = {
    LANE_A_CAPABILITY_BLOCKER: "T22",
    "upstream_task:T31": "T31",
}


def assert_selectors_match_observation(capability: Mapping[str, Any]) -> None:
    """Refuse selectors that no longer match what the service was observed to say.

    ``PRIMARY_DEVICE`` and ``COMPARISON_DEVICES`` carry literal ``os`` and
    ``attribute`` values so the default selector stays a pure constant. This is
    the check that keeps those literals honest: they must equal the capability
    record's observation exactly, or generation fails.
    """

    observed = {
        str(device["name"]): device for device in capability["observation"]["devices"]
    }
    for selector in (PRIMARY_DEVICE, *COMPARISON_DEVICES):
        name = str(selector["name"])
        record = observed.get(name)
        _require(
            record is not None,
            f"the capability record does not observe target device {name!r}",
        )
        assert record is not None
        _require(
            str(selector["os"]) == str(record["os"]),
            f"{name}: the selector records os {selector['os']!r} but the "
            f"capability observation read {record['os']!r}",
        )
        _require(
            list(selector["attributes"]) == list(record["attributes"]),
            f"{name}: the selector's attribute list is not the one the "
            "capability observation read from the service",
        )


def build_quantization_support(capability: Mapping[str, Any]) -> dict[str, Any]:
    """Return Lane A's quantization support as an observation, not an assumption."""

    observation = capability["observation"]
    return {
        "measurement": "observed_public_api_capability",
        "observation_date": observation["observation_date"],
        "client_version": observation["client_version"],
        "record": {
            "path": DEFAULT_CAPABILITY_RECORD.as_posix(),
            "canonical_json_sha256": canonical_json_sha256(capability),
            "record_type": CAPABILITY_RECORD_TYPE,
            "submitted_jobs": capability["submitted_jobs"],
            "device_minutes_consumed": capability["device_minutes_consumed"],
            "cost": capability["cost"],
            "rebuild_command": CAPABILITIES_OFFLINE_COMMAND,
        },
        "submit_entry_points": list(observation["submit_entry_points"]),
        "quantize_entry_point": observation["quantize_entry_point"],
        "quantize_signature": observation["quantize_signature"],
        "quantize_dtypes": list(observation["quantize_dtypes"]),
        "candidate_requests": dict(capability["candidate_quantize_requests"]),
        "kv_cache_dtype": dict(capability["kv_cache_dtype"]),
        "establishes": list(capability["establishes"]),
        "does_not_establish": list(capability["does_not_establish"]),
        "supersedes": (
            "the earlier assumption that Lane A's quantization support was "
            "unverified. It is now a dated read-only observation of the public "
            "API and nothing more: an entry point that can express both "
            "candidates is not a compile, not a placement, and not an artifact."
        ),
    }


def build_deployment_routes(
    *,
    aimet_versions: Mapping[str, str],
    capability: Mapping[str, Any],
) -> dict[str, Any]:
    """Return both lanes with what exists, what is missing, and who is blocked."""

    assert_selectors_match_observation(capability)
    quantization_support = build_quantization_support(capability)
    observation = capability["observation"]
    stage_commands = [
        {
            "stage": stage,
            "order": index,
            "script": STAGE_SCRIPTS[stage],
            "options": STAGE_OPTIONS[stage],
            "command": (
                f"PYTHONPATH=src python3 {STAGE_SCRIPTS[stage]} "
                f"--request <private-request>.json "
                f"--manifest results/processed/qualcomm/"
                f"t41-{stage}-manifest.json"
            ),
        }
        for index, stage in enumerate(STAGE_ORDER)
    ]
    return {
        "lane_a_ai_hub_workbench": {
            "plan_section": "7.1",
            "available": False,
            "blocked_on": [
                LANE_A_CAPABILITY_BLOCKER,
                "upstream_task:T31",
            ],
            "blocked_on_owners": dict(LANE_A_BLOCKER_OWNERS),
            "authorization": {
                "qai_hub_submission": "granted",
                "granted_on": observation["observation_date"],
                "scope": (
                    "free capacity only; no spend; any Device Cloud use capped "
                    "at 120 device minutes"
                ),
                "consumed_by_this_repository": {
                    "submitted_jobs": capability["submitted_jobs"],
                    "device_minutes": capability["device_minutes_consumed"],
                    "cost": capability["cost"],
                },
                "device_cloud_session": (
                    "not cleared here: an interactive Device Cloud lease is a "
                    "separate lock from job submission, and T41 records the "
                    "120-minute ceiling above rather than assuming the lease"
                ),
                "note": (
                    "Recorded because it changed, and recorded as a permission "
                    "rather than as progress. Submission is no longer what holds "
                    "Lane A; nothing has been spent under it, and there is still "
                    "nothing to submit."
                ),
            },
            "quantization_support": quantization_support,
            "ordered_stages": [
                "quantize (Workbench Quantize Job)",
                "compile",
                "inference",
                "profile",
                "download deployable artifact",
            ],
            "provided_by_this_repository": [
                "src/slm_lab/deployment/qualcomm/ai_hub.py: sanitized "
                "schema-v2 compile, inference, and profile stage adapters",
                "scripts/qualcomm/{compile,inference,profile}.py: the three "
                "separate stage processes",
                "src/slm_lab/quantization/w8.py: the request emitter that "
                "composes a schema-v2 request and stops there",
            ],
            "missing": [
                "a quantize-stage adapter: slm_lab.deployment.qualcomm.ai_hub "
                "implements compile, inference, and profile only, so Lane A "
                "cannot by itself produce the quantized artifact at this "
                "commit — the compile stage consumes one, it does not create "
                "one. The public API does expose submit_quantize_job, which is "
                "the point: the gap is this repository's, not the service's, "
                "and src/slm_lab/deployment/qualcomm/ belongs to T22",
                "any W8 artifact to send: an entry point that can express both "
                "candidates is not an artifact, and no weight has been quantized "
                "on either lane",
                "any evidence that a floating Qwen graph compiles at all; T31 "
                "and T33 are planned",
            ],
            "submission_parameters": {
                "client": {"name": "qai-hub", "version": QAI_HUB_CLIENT_VERSION},
                "runtime": {"name": RUNTIME_NAME, "version": QAIRT_VERSION},
                "retry": False,
                "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
                "stages": stage_commands,
                "version_evidence": [
                    "ai/handoffs/T30-ai-hub-adapters.md",
                    "results/hosts/workbench-toy-lifecycle-2026-07-25.json",
                ],
            },
            "target_devices": {
                "policy_section": "3.2",
                "primary": dict(PRIMARY_DEVICE),
                "comparison": [dict(device) for device in COMPARISON_DEVICES],
                "observed": {
                    "path": DEFAULT_CAPABILITY_RECORD.as_posix(),
                    "canonical_json_sha256": canonical_json_sha256(capability),
                    "observation_date": observation["observation_date"],
                    "client_version": observation["client_version"],
                    "devices_listed_by_the_service": observation["device_count"],
                    "recorded_here": len(PLAN_TARGET_DEVICE_NAMES),
                },
                "attributes_note": (
                    "Every name, os, and attribute above was read from the "
                    "service by the read-only capability query recorded at the "
                    "path above, on that date, with that client. It is no "
                    "longer a guess and no longer empty. Two things a reader "
                    "should know. First, os is the SDK's own version string — "
                    "the X Elite CRD reports '11', not the human label 'Windows "
                    "11' this file used to carry, and the platform lives in the "
                    "os:windows attribute. Second, each list is the full "
                    "observed vocabulary, which identifies the device exactly "
                    "and therefore over-constrains the selector: a future "
                    "authorized session that hits a selector mismatch should "
                    "trim to name before concluding the device is gone. The "
                    "selectors are re-checked against the capability record on "
                    "every generation run, so a service change fails check "
                    "rather than silently re-anchoring."
                ),
                "does_not_establish": [
                    "that_any_of_these_devices_will_accept_this_graph",
                    "that_htp_supports_fp16_implies_anything_about_int8_or_int16",
                    "any_queue_time_availability_or_cost_for_these_devices",
                ],
            },
            "quantize_job_note": (
                "Committed T02 access evidence records a Workbench Quantize "
                "Job running AIMET 2.34 per release notes dated 2026-07-06, "
                "while environments/linux-aimet pins aimet-onnx and "
                "aimet-torch at "
                f"{aimet_versions.get('aimet-onnx', 'unknown')}. A Lane A and "
                "a Lane B W8 artifact would therefore not be the same "
                "artifact, and their encodings must not be compared as if a "
                "difference between them were a property of the model."
            ),
        },
        "lane_b_local_aimet": {
            "plan_section": "7.1",
            "available": False,
            "blocked_on": ["hardware:linux_cuda_aimet_host"],
            "ordered_stages": [
                "install the pinned AIMET environment on a Linux + CUDA host",
                "run the range observers over the frozen T40 corpus",
                "simulate W8 and record the quantized artifact digest",
                "export ONNX, external data, and encodings",
                "hand the artifact to Lane A's compile stage",
            ],
            "provided_by_this_repository": [
                "environments/linux-aimet/aimet-requirements.in and "
                ".lock: hash-pinned dependency set",
                "environments/linux-aimet/aimet-cuda-wheels.lock: the CUDA "
                "wheels that cannot be resolved from PyPI",
                "environments/linux-aimet/aimet-host.template.json: the host "
                "manifest a real run must fill in",
                "configs/quantization/calibration.yaml: the frozen inputs",
            ],
            "missing": [
                "the host itself: AIMET publishes Linux x86-64 wheels only and "
                "nothing is installed on the primary macOS development host",
                "any executed observer pass, so no activation histogram exists "
                "anywhere in this repository",
            ],
            "pinned_versions": dict(sorted(aimet_versions.items())),
        },
        "boundary": (
            "Both lanes are unavailable at this commit, and neither is waiting "
            "on permission any more. Lane A is blocked on a capability this "
            "repository does not have — no quantize-stage adapter, in a module "
            "T22 owns — and on T31. Lane B is blocked on hardware. The public "
            "API can express both candidates, which is why the remaining gap is "
            "worth naming precisely: it is ours, not the service's. Neither "
            "blocker is a code defect and neither is resolvable inside this task."
        ),
    }


def read_aimet_versions(root: Path) -> dict[str, str]:
    """Read the pinned Lane B AIMET versions from the committed requirements."""

    path = root / DEFAULT_AIMET_REQUIREMENTS
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise W8EvidenceError(f"cannot read {path}: {exc}") from exc
    versions: dict[str, str] = {}
    for line in text.splitlines():
        entry = line.split("#", 1)[0].strip()
        if "==" not in entry:
            continue
        name, _, version = entry.partition("==")
        if name.strip().startswith("aimet"):
            versions[name.strip()] = version.strip()
    _require(
        bool(versions),
        f"{DEFAULT_AIMET_REQUIREMENTS} pins no aimet distribution",
    )
    return versions


LEDGER_STATUSES = ("satisfied", "not_run", "blocked")
COMMAND_STATUSES = ("implemented_at_this_commit", "not_implemented_at_this_commit")


def _ledger_row(
    *,
    row_id: str,
    requirement: str,
    status: str,
    owner_task: str,
    blocked_on: Sequence[str],
    command: str,
    command_status: str,
    frozen_input_here: str,
) -> dict[str, Any]:
    _require(status in LEDGER_STATUSES, f"{row_id}: unsupported ledger status")
    _require(
        command_status in COMMAND_STATUSES,
        f"{row_id}: unsupported ledger command status",
    )
    _require(bool(command.strip()), f"{row_id}: ledger row needs a command")
    return {
        "id": row_id,
        "plan_reference": "docs/project/plan.md section 7.3",
        "requirement": requirement,
        "status": status,
        "owner_task": owner_task,
        "blocked_on": list(blocked_on),
        "command": {"status": command_status, "value": command},
        "frozen_input_here": frozen_input_here,
    }


def build_evidence_ledger(calibration_binding: Mapping[str, Any]) -> dict[str, Any]:
    """Return plan 7.3's measurement list as a status ledger.

    Exactly one row is ``satisfied``, and only because this module verifies its
    input on every run. Everything else defaults to ``not_run`` or is
    ``blocked`` on a named capability gap, host, authorization, or upstream
    task. A row is never satisfied by a declaration: ``frozen_input_here`` says
    what T41 contributes without pretending that a policy is a report.

    Two rows used to be blocked on ``user_authorization:qai_hub_submission``.
    That permission was granted on ``CAPABILITY_OBSERVATION_DATE`` and the rows
    did not become satisfied, because permission was never the only thing
    missing: both need a W8 artifact, and there is none to profile. They now
    name the two reasons there is none.
    """

    revision = calibration_binding["calibration_dataset_revision"]
    tokens = calibration_binding["inherited_bias"]["total_calibration_tokens"]
    rows = [
        _ledger_row(
            row_id="calibration_corpus_revision_and_token_budget",
            requirement="Calibration corpus revision and token budget.",
            status="satisfied",
            owner_task="T40",
            blocked_on=[],
            command=CHECK_COMMAND,
            command_status="implemented_at_this_commit",
            frozen_input_here=(
                f"Both candidates bind {revision} and carry the corpus token "
                f"budget of {tokens} token positions, re-read from the "
                "committed T40 contract on every validation run."
            ),
        ),
        _ledger_row(
            row_id="quantization_and_calibration_time",
            requirement="Quantization and calibration time.",
            status="blocked",
            owner_task="T41",
            blocked_on=["hardware:linux_cuda_aimet_host"],
            command=(
                "<linux-cuda-aimet-host> python -m <t41-lane-b-runner> "
                "--candidate <w8a16|w8a8> --calibration "
                "configs/quantization/calibration.yaml"
            ),
            command_status="not_implemented_at_this_commit",
            frozen_input_here=(
                "The calibration inputs and the observer policy are frozen; "
                "the runner that would consume them does not exist because "
                "there is no host to run it on."
            ),
        ),
        _ledger_row(
            row_id="layer_inclusion_exclusion_report",
            requirement="Layer inclusion/exclusion report.",
            status="not_run",
            owner_task="T41",
            blocked_on=["hardware:linux_cuda_aimet_host"],
            command=(
                "<linux-cuda-aimet-host> python -m <t41-lane-b-runner> "
                "--candidate <w8a16|w8a8> --emit-inclusion-report"
            ),
            command_status="not_implemented_at_this_commit",
            frozen_input_here=(
                "The intended policy is frozen in "
                "excluded_from_quantization, separated into policy choices and "
                "frozen-graph constraints. That is a declaration of intent, "
                "not a report of what a tool actually excluded, and the two "
                "are not the same evidence."
            ),
        ),
        _ledger_row(
            row_id="encoding_format_and_hashes",
            requirement="Encoding format and hashes.",
            status="not_run",
            owner_task="T41",
            blocked_on=["hardware:linux_cuda_aimet_host"],
            command=(
                "<linux-cuda-aimet-host> sha256sum <quantized>.onnx "
                "<quantized>.onnx.data <quantized>.encodings"
            ),
            command_status="not_implemented_at_this_commit",
            frozen_input_here=(
                "The weight and activation policies that determine what an "
                "encodings file would contain are frozen; no encodings file "
                "exists and the shipped format is not decided."
            ),
        ),
        _ledger_row(
            row_id="perplexity_and_small_benchmark_change",
            requirement="Perplexity and small benchmark change.",
            status="blocked",
            owner_task="T41",
            blocked_on=[
                "hardware:linux_cuda_aimet_host",
                "upstream_task:T31",
            ],
            command=COMPARE_COMMAND,
            command_status="implemented_at_this_commit",
            frozen_input_here=(
                "The evaluation binding, the protocol digest, the metric set, "
                "and the refusal rules are frozen, and compare_quality exists "
                "and runs. It has nothing to compare: this repository contains "
                "no W8 quality result at this commit, which validate_repository "
                "enforces."
            ),
        ),
        _ledger_row(
            row_id="logit_and_cache_error",
            requirement="Logit/cache error.",
            status="blocked",
            owner_task="T41",
            blocked_on=[
                "hardware:linux_cuda_aimet_host",
                "dependency:torch_and_onnxruntime_absent_on_the_primary_host",
            ],
            command=(
                "SLM_LAB_ARTIFACT_ROOT=<external-root> PYTHONPATH=src "
                "<python-with-torch-and-onnxruntime> -m pytest -q tests/onnx"
            ),
            command_status="implemented_at_this_commit",
            frozen_input_here=(
                "The float16 baseline this error would be measured against is "
                "pinned by digest; the quantized side does not exist."
            ),
        ),
        _ledger_row(
            row_id="artifact_size",
            requirement="Artifact size.",
            status="not_run",
            owner_task="T41",
            blocked_on=["hardware:linux_cuda_aimet_host"],
            command="<linux-cuda-aimet-host> wc -c <quantized>.onnx.data",
            command_status="not_implemented_at_this_commit",
            frozen_input_here=(
                "An analytic weight-storage projection labelled "
                "analytic_projection, whose does_not_establish list names "
                "on-disk artifact size first. A projection is not a size."
            ),
        ),
        _ledger_row(
            row_id="peak_memory",
            requirement="Peak memory.",
            status="blocked",
            owner_task="T41",
            blocked_on=list(NO_W8_ARTIFACT_BLOCKERS),
            command=(
                f"PYTHONPATH=src python3 {STAGE_SCRIPTS['profile']} "
                "--request <private-profile-request>.json --manifest "
                "results/processed/qualcomm/t41-profile-manifest.json"
            ),
            command_status="implemented_at_this_commit",
            frozen_input_here=(
                "The projection records the float16 KV cache bytes per context "
                "so a reader knows which term W8 weight quantization does not "
                "reduce. Runtime peak memory is a profile-job measurement. "
                "Submission is permitted as of "
                f"{CAPABILITY_OBSERVATION_DATE}; what is missing is the "
                "quantized artifact to profile, on both lanes."
            ),
        ),
        _ledger_row(
            row_id="graph_latency_and_npu_placement",
            requirement="Graph latency and NPU placement.",
            status="blocked",
            owner_task="T41",
            blocked_on=[*NO_W8_ARTIFACT_BLOCKERS, "upstream_task:T31"],
            command=(
                f"PYTHONPATH=src python3 {STAGE_SCRIPTS['profile']} "
                "--request <private-profile-request>.json --manifest "
                "results/processed/qualcomm/t41-profile-manifest.json"
            ),
            command_status="implemented_at_this_commit",
            frozen_input_here=(
                "The submission parameters, the compute-unit option, and the "
                "target-device selectors are frozen — the selectors now against "
                "the observed device vocabulary rather than a name alone — and "
                "the request emitter will compose the job. Submission is "
                "permitted and no job has been submitted, because there is no "
                "quantized artifact to compile and no compiled artifact to "
                "profile."
            ),
        ),
        _ledger_row(
            row_id="end_to_end_generation_impact",
            requirement=(
                "End-to-end generation impact where a persistent loop exists."
            ),
            status="blocked",
            owner_task="T41",
            blocked_on=[
                "upstream_task:T32",
                "upstream_task:T33",
                "user_authorization:device_cloud_session",
            ],
            command=(
                "<device-cloud-session> PYTHONPATH=src python3 -m "
                "slm_lab.deployment.qualcomm.device_cloud <session-args>"
            ),
            command_status="not_implemented_at_this_commit",
            frozen_input_here=(
                "Nothing. No persistent generation loop exists on any "
                "Qualcomm target in this repository, so this row is recorded "
                "to keep plan 7.3 complete rather than to claim progress."
            ),
        ),
    ]
    counts: dict[str, int] = {status: 0 for status in LEDGER_STATUSES}
    for row in rows:
        counts[str(row["status"])] += 1
    return {
        "plan_section": "7.3",
        "default_status": "not_run",
        "rows": rows,
        "summary": [
            {"status": status, "count": counts[status]} for status in LEDGER_STATUSES
        ],
        "note": (
            "Status is about evidence, not effort. `satisfied` means this "
            "module verified an input on this run; `not_run` means the "
            "measurement was never taken; `blocked` means it cannot be taken "
            "here and names what would unblock it."
        ),
    }


def build_candidate(
    candidate_id: str,
    *,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the candidate core whose canonical JSON is hashed.

    Everything that determines what would actually be quantized lives here —
    the precision policies, the inclusion/exclusion policy, the calibration and
    baseline bindings, the evaluation binding, and the storage projection — and
    nothing that does not, so a prose edit elsewhere in the file never changes
    the candidate revision.
    """

    _require(
        candidate_id in CANDIDATE_IDS,
        f"unknown W8 candidate {candidate_id!r}; expected one of {CANDIDATE_IDS}",
    )
    definition = CANDIDATE_DEFINITIONS[candidate_id]
    model_contract = inputs["model_contract"]
    graph_inventory = build_graph_inventory_binding(inputs["graph_record"])

    candidate: dict[str, Any] = {
        "candidate_id": candidate_id,
        "plan_matrix_row": PLAN_MATRIX_ROWS[candidate_id],
        "summary": definition["summary"],
        "intent": definition["intent"],
        "precision_state": "specified",
        "precision_state_note": (
            "This field is about the candidate's *evidence* state, not about "
            "an achievable precision. `specified` means the policy below is "
            "frozen and nothing has been quantized. `simulated` and `deployed` "
            "are different states reached only by different evidence: a "
            "simulation record naming its tool, that tool's exact version, the "
            "host, and the quantized artifact digest for the first; all three "
            "chained AI Hub stage manifests for the second. "
            "slm_lab.quantization.w8.assess_precision_state is the only thing "
            "that decides which state applies, and it never reads a state "
            "asserted by an input record."
        ),
        "precision_state_scope": PRECISION_STATE_SCOPES["specified"],
        "weights": build_weight_policy(model_contract),
        "activations": dict(definition["activations"]),
        "kv_cache": build_kv_cache_policy(candidate_id),
        "graph_inventory": graph_inventory,
        "excluded_from_quantization": {
            "policy": (
                "Everything not listed here is quantized to the weight or "
                "activation precision declared above. Every entry names why it "
                "is excluded and whether the exclusion is a choice T41 is "
                "making or a constraint the frozen graph imposes."
            ),
            "entries": [dict(entry) for entry in EXCLUSION_ENTRIES],
            "policy_choice_ids": [
                entry["id"]
                for entry in EXCLUSION_ENTRIES
                if entry["kind"] == "policy_choice"
            ],
            "frozen_graph_constraint_ids": [
                entry["id"]
                for entry in EXCLUSION_ENTRIES
                if entry["kind"] == "frozen_graph_constraint"
            ],
            "kind_note": (
                "A policy choice is revisable by T41 on evidence. A "
                "frozen-graph constraint is not: reversing one means changing "
                "the model contract, the T12 graph contract, or the T20/T23 "
                "export boundary, and those belong to other tasks."
            ),
        },
        "calibration": build_calibration_binding(
            inputs["calibration"],
            calibration_path=DEFAULT_CALIBRATION_CONFIG,
        ),
        "baseline": build_baseline_binding(inputs["manifests"]),
        "evaluation": build_evaluation_binding(
            protocol=inputs["protocol"],
            academic=inputs["academic"],
        ),
    }
    candidate["weight_storage_projection"] = weight_storage_projection(
        candidate,
        model_contract,
        graph_inventory=graph_inventory,
        baseline_external_data_bytes=inputs["baseline_external_data_bytes"],
    )
    return candidate


def build_document(
    candidate_id: str,
    *,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the full committed candidate specification document."""

    candidate = build_candidate(candidate_id, inputs=inputs)
    model_contract = inputs["model_contract"]
    model = model_contract["model"]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "candidate_id": candidate_id,
        "plan_matrix_row": PLAN_MATRIX_ROWS[candidate_id],
        "title": f"Qwen3-0.6B {candidate_id.upper()} deployment candidate",
        "summary": candidate["summary"],
        "intent": candidate["intent"],
        "model": {
            "id": model["id"],
            "revision": model["revision"],
            "license": model["license"],
            "reference_dtype": model["reference_dtype"],
            "tie_word_embeddings": model["architecture"]["tie_word_embeddings"],
        },
        "candidate": candidate,
        "candidate_canonical_json_sha256": canonical_json_sha256(candidate),
        "deployment_routes": build_deployment_routes(
            aimet_versions=inputs["aimet_versions"],
            capability=inputs["capability"],
        ),
        "evidence_requirements": build_evidence_ledger(candidate["calibration"]),
        "artifact_manifest_contract": {
            "plan_section": "17.4",
            "required_fields_when_a_W8_artifact_exists": {
                "precision": (
                    f"{candidate_id} plus the precision state, e.g. "
                    f"{precision_label(candidate_id, 'simulated')}"
                ),
                "quantization": (
                    "the weights/activations/kv_cache blocks of this file, "
                    "plus the exclusion policy actually applied"
                ),
                "calibration_dataset_revision": candidate["calibration"][
                    "calibration_dataset_revision"
                ],
                "source_artifact_sha256": (
                    "the float16 graph the quantized artifact was derived from"
                ),
            },
            "instruction": (
                "No artifact manifest may carry this candidate_id without all "
                "four. A manifest whose calibration_dataset_revision differs "
                "from the value above was calibrated on a different corpus and "
                "its quality delta is not comparable."
            ),
        },
        "commands": {
            "generate": GENERATE_COMMAND,
            "check": CHECK_COMMAND,
            "status": STATUS_COMMAND,
            "record": RECORD_COMMAND,
            "capabilities": CAPABILITIES_COMMAND,
            "capabilities_offline": CAPABILITIES_OFFLINE_COMMAND,
            "compare": COMPARE_COMMAND,
            "request": REQUEST_COMMAND,
            "tests": TESTS_COMMAND,
        },
    }


_SHARED_HEADER_POINTS = """\
# 1. This file is a *specification*, not a result. No weight is quantized here
#    and none can be: AIMET publishes Linux + CUDA wheels only and is specified
#    in environments/linux-aimet/ rather than installed, and Lane A cannot make
#    a W8 artifact either because this repository has no quantize-stage adapter.
#    AI Hub submission is permitted as of 2026-08-03 and that changed nothing
#    here, because there is still nothing to submit.
#    `candidate.precision_state` reads `specified` for exactly that reason.
# 2. `specified`, `simulated`, and `deployed` are three different evidence
#    states, and only slm_lab.quantization.w8.assess_precision_state decides
#    which one applies. `simulated` needs a host record naming the tool, its
#    exact version, the host, and the quantized artifact digest. `deployed`
#    additionally needs all three sanitized AI Hub schema-v2 stage manifests
#    with a verified digest chain from the simulated artifact through the
#    compiled artifact into the inference and profile runs. A record that
#    asserts its own state is not consulted.
# 3. Everything numeric here is arithmetic over committed repository inputs or
#    a hash read off a committed file. `weight_storage_projection` is labelled
#    `analytic_projection` on every row and states, in its own
#    `does_not_establish` list, that it is not an artifact size, not an
#    encoding overhead, not a peak memory, and not a latency.
# 4. The KV cache is where this candidate meets a contract it does not own.
#    slm_lab.contracts.static_cache freezes CACHE_DTYPE as float16 and that
#    dtype is declared on the cache tensors at the T20/T23 graph boundary. Read
#    `candidate.kv_cache.change_control` before proposing to lower it.
# 5. The exclusion policy separates what T41 chose from what the frozen graph
#    imposes. Read `candidate.excluded_from_quantization.kind_note`: reversing
#    a `policy_choice` needs evidence, reversing a `frozen_graph_constraint`
#    needs another task's consent.
# 6. Bindings are re-derived and re-hashed on every validation run: the T40
#    calibration revision, the four committed float16 baseline manifests, the
#    frozen T13 protocol digest, the T21 structural graph inventory, and the
#    dated AI Hub capability observation. If any of them moves, `check` fails
#    rather than silently re-anchoring.
# 7. `deployment_routes.lane_a_ai_hub_workbench.quantization_support` is an
#    observation, not an assumption: one read-only query, on one date, with one
#    client, that submitted no job and consumed no device minute. It records
#    that both candidates are *expressible* as public quantize requests. Read
#    its `does_not_establish` list before treating that as compiler acceptance,
#    NPU placement, or an artifact.
"""


def file_header(candidate_id: str) -> str:
    """Return the deterministic comment header for one candidate file."""

    definition = CANDIDATE_DEFINITIONS[candidate_id]
    activations = definition["activations"]
    row = PLAN_MATRIX_ROWS[candidate_id]
    return (
        f"# T41 {candidate_id.upper()} candidate specification for Qwen3-0.6B "
        f"(plan section 7.2, row {row}).\n"
        "#\n"
        "# GENERATED FILE. Do not hand-edit. Regenerate with:\n"
        f"#   {GENERATE_COMMAND}\n"
        "# and verify with:\n"
        f"#   {CHECK_COMMAND}\n"
        "#\n"
        f"# int8 weights, {activations['dtype']} activations, "
        f"{static_cache.CACHE_DTYPE} KV cache.\n"
        "#\n"
        "# What a reader should take away\n"
        "# ------------------------------\n"
        f"{_SHARED_HEADER_POINTS}"
    )


def render_document(document: Mapping[str, Any], candidate_id: str) -> str:
    """Render one candidate specification as deterministic, commented YAML."""

    body = yaml.safe_dump(
        dict(document),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=76,
    )
    return file_header(candidate_id) + "\n" + body


# ---------------------------------------------------------------------------
# Repository generation and validation
# ---------------------------------------------------------------------------


def load_baseline_manifests(repo_root: Path) -> dict[str, Mapping[str, Any]]:
    """Load the four committed float16 T20/T23 manifests keyed by variant."""

    directory = repo_root / DEFAULT_MANIFEST_DIRECTORY
    _require(
        directory.is_dir(),
        f"the float16 baseline manifest directory is missing: {directory}",
    )
    manifests: dict[str, Mapping[str, Any]] = {}
    for path in sorted(directory.glob("S*.json")):
        document = _load_json(path)
        _require(
            isinstance(document, Mapping),
            f"baseline manifest is not a JSON object: {path}",
        )
        assert isinstance(document, Mapping)
        manifests[path.stem] = document
    expected = {f"S{length}" for length in static_cache.CONTEXT_VARIANTS}
    _require(
        set(manifests) == expected,
        f"baseline manifests {sorted(manifests)} do not match the frozen T12 "
        f"context family {sorted(expected)}",
    )
    return manifests


def _baseline_external_data_bytes(manifests: Mapping[str, Mapping[str, Any]]) -> int:
    """Return the single external-data byte size every manifest agrees on."""

    sizes: set[int] = set()
    for variant_id, manifest in manifests.items():
        artifacts = manifest.get("artifacts") or {}
        for kind in ("prefill", "decode"):
            record = artifacts.get(kind) or {}
            for external in record.get("external_data") or ():
                size = external.get("size_bytes")
                _require(
                    isinstance(size, int) and not isinstance(size, bool) and size > 0,
                    f"{variant_id} {kind}: external data size_bytes must be a "
                    f"positive integer, found {size!r}",
                )
                sizes.add(int(size))
    _require(
        len(sizes) == 1,
        "the committed manifests disagree on the float16 weight external-data "
        f"size: {sorted(sizes)}. The projection cross-check needs one weight "
        "set, not several.",
    )
    return sizes.pop()


def load_inputs(repo_root: Path) -> dict[str, Any]:
    """Load every committed document the W8 candidates are derived from."""

    manifests = load_baseline_manifests(repo_root)
    # `configs/models/qwen3-0.6b.yaml` is JSON-formatted despite its extension
    # and every producer in this repository reads it with `json.loads`; follow
    # that convention rather than accepting a YAML superset nothing emits.
    model_contract = _load_json(repo_root / DEFAULT_MODEL_CONTRACT)
    calibration = _load_yaml(repo_root / DEFAULT_CALIBRATION_CONFIG)
    _require(
        isinstance(calibration, Mapping),
        f"calibration contract is not a YAML mapping: {DEFAULT_CALIBRATION_CONFIG}",
    )
    graph_record = _load_json(repo_root / DEFAULT_GRAPH_INVENTORY)
    _require(
        isinstance(graph_record, Mapping),
        f"graph inventory is not a JSON object: {DEFAULT_GRAPH_INVENTORY}",
    )
    academic = _load_json(repo_root / DEFAULT_ACADEMIC_CONTRACT)
    _require(
        isinstance(academic, Mapping),
        f"academic contract is not a JSON object: {DEFAULT_ACADEMIC_CONTRACT}",
    )
    return {
        "model_contract": model_contract,
        "calibration": calibration,
        "manifests": manifests,
        "baseline_external_data_bytes": _baseline_external_data_bytes(manifests),
        "graph_record": graph_record,
        "protocol": _load_protocol(repo_root),
        "academic": academic,
        "aimet_versions": read_aimet_versions(repo_root),
        # Re-derived from its own observation rather than merely read, so a
        # candidate can never cite a capability record that has been edited to
        # claim more than the query saw.
        "capability": load_capability_record(repo_root),
    }


def generate_repository(repo_root: Path) -> list[Path]:
    """Regenerate both committed candidate specifications and return them."""

    inputs = load_inputs(repo_root)
    written: list[Path] = []
    for candidate_id in CANDIDATE_IDS:
        document = build_document(candidate_id, inputs=inputs)
        path = repo_root / candidate_config_path(candidate_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_document(document, candidate_id), encoding="utf-8")
        written.append(path)
    return written


def _validate_candidate_document(
    candidate_id: str,
    config: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> None:
    """Check one committed candidate against every committed input it binds."""

    label = f"{candidate_id}.yaml"
    _require(
        config.get("schema_version") == SCHEMA_VERSION,
        f"{label}: unsupported schema version",
    )
    _require(config.get("task_id") == TASK_ID, f"{label}: unexpected task ID")
    _require(
        config.get("candidate_id") == candidate_id,
        f"{label}: candidate_id drift",
    )
    _require(
        config.get("plan_matrix_row") == PLAN_MATRIX_ROWS[candidate_id],
        f"{label}: plan matrix row drift",
    )

    candidate = config.get("candidate")
    _require(isinstance(candidate, Mapping), f"{label}: no candidate block")
    assert isinstance(candidate, Mapping)

    _require(
        candidate.get("precision_state") in PRECISION_STATES,
        f"{label}: precision_state is not one of {PRECISION_STATES}",
    )
    _require(
        candidate.get("precision_state") == "specified",
        f"{label}: a committed candidate specification may only ever read "
        "precision_state 'specified'; a simulated or deployed state is "
        "evidence, and evidence does not live in a config file",
    )

    # 1. The cache dtype claim must agree with the frozen T12 contract.
    cache = candidate.get("kv_cache") or {}
    frozen = cache.get("frozen_contract") or {}
    for field, value in (
        ("applied_dtype", cache.get("applied_dtype")),
        ("frozen_contract.dtype", frozen.get("dtype")),
    ):
        _require(
            value == static_cache.CACHE_DTYPE,
            f"{label}: kv_cache {field} is {value!r} but "
            f"slm_lab.contracts.static_cache.CACHE_DTYPE is "
            f"{static_cache.CACHE_DTYPE!r}; a candidate may not claim a cache "
            "precision the frozen T12 contract does not carry",
        )

    # 2. The calibration revision must still be the one T40 regenerates.
    expected_revision = inputs["calibration"].get("calibration_dataset_revision")
    recorded_revision = (candidate.get("calibration") or {}).get(
        "calibration_dataset_revision"
    )
    _require(
        recorded_revision == expected_revision,
        f"{label}: calibration_dataset_revision is {recorded_revision!r}; "
        f"configs/quantization/calibration.yaml carries {expected_revision!r}",
    )

    # 3. Every float16 baseline manifest must still hash to its recorded value.
    recorded_baselines = (candidate.get("baseline") or {}).get("manifests")
    _require(
        isinstance(recorded_baselines, list) and bool(recorded_baselines),
        f"{label}: the baseline binding records no manifests",
    )
    assert isinstance(recorded_baselines, list)
    expected_hashes = {
        variant_id: canonical_json_sha256(manifest)
        for variant_id, manifest in inputs["manifests"].items()
    }
    _require(
        {row.get("variant_id") for row in recorded_baselines} == set(expected_hashes),
        f"{label}: the baseline binding does not cover the frozen context family",
    )
    for row in recorded_baselines:
        variant_id = str(row.get("variant_id"))
        _require(
            row.get("canonical_json_sha256") == expected_hashes[variant_id],
            f"{label}: baseline manifest {variant_id} no longer hashes to the "
            "value this candidate was written against",
        )
        _require(
            row.get("precision") == inputs["manifests"][variant_id].get("precision"),
            f"{label}: baseline manifest {variant_id} precision drift",
        )

    # 4. The frozen T13 protocol digest must not have moved.
    recorded_protocol = (candidate.get("evaluation") or {}).get("benchmark_protocol")
    _require(
        isinstance(recorded_protocol, Mapping),
        f"{label}: the evaluation binding records no benchmark protocol",
    )
    assert isinstance(recorded_protocol, Mapping)
    expected_digest = inputs["protocol"]["contract_sha256"]
    _require(
        recorded_protocol.get("contract_sha256") == expected_digest,
        f"{label}: the recorded T13 protocol digest is "
        f"{recorded_protocol.get('contract_sha256')!r}; this repository's "
        f"frozen protocol is {expected_digest!r}, so no quality delta measured "
        "against this candidate would be comparable",
    )

    # 5. The projection must still be the one the inputs derive.
    expected_projection = weight_storage_projection(
        candidate,
        inputs["model_contract"],
        graph_inventory=build_graph_inventory_binding(inputs["graph_record"]),
        baseline_external_data_bytes=inputs["baseline_external_data_bytes"],
    )
    _require(
        candidate.get("weight_storage_projection") == expected_projection,
        f"{label}: weight_storage_projection drift; the recorded projection is "
        "not the one derived from the committed model contract and exclusion "
        "policy",
    )

    # 6. The candidate hash must cover the candidate that is actually there.
    _require(
        config.get("candidate_canonical_json_sha256")
        == canonical_json_sha256(candidate),
        f"{label}: candidate_canonical_json_sha256 drift",
    )

    # 7. The cited AI Hub capability observation must still be the committed one.
    lane_a = (config.get("deployment_routes") or {}).get(
        "lane_a_ai_hub_workbench"
    ) or {}
    support = lane_a.get("quantization_support") or {}
    cited = (support.get("record") or {}).get("canonical_json_sha256")
    expected_capability = canonical_json_sha256(inputs["capability"])
    _require(
        cited == expected_capability,
        f"{label}: the cited AI Hub capability record hashes to {cited!r} but "
        f"{DEFAULT_CAPABILITY_RECORD.as_posix()} hashes to "
        f"{expected_capability!r}. Lane A's quantization support is an "
        "observation of a specific record; a citation that no longer resolves "
        "is an assumption again.",
    )
    _require(
        "user_authorization:qai_hub_submission" not in (lane_a.get("blocked_on") or []),
        f"{label}: Lane A is still recorded as blocked on submission "
        "authorization, which was granted on "
        f"{CAPABILITY_OBSERVATION_DATE}. Regenerate rather than leaving a "
        "blocker that has been cleared.",
    )

    expected_document = build_document(candidate_id, inputs=inputs)
    _require(
        dict(config) == expected_document,
        f"{label}: the committed candidate differs from the candidate derived "
        "from committed inputs",
    )


def _assert_no_measured_quality_record(repo_root: Path) -> None:
    """Refuse to pass while a W8 quality record exists that nothing produced.

    ``compare_quality`` computes deltas from records it is given. At this
    commit no W8 result has been measured anywhere, so a file that looks like
    one appearing under ``results/quantization/`` is either a fabrication or a
    real measurement whose provenance this gate has not been taught to check.
    Both are reasons to stop.
    """

    directory = repo_root / DEFAULT_EVIDENCE_DIRECTORY
    if not directory.is_dir():
        return
    found = sorted(path.name for path in directory.glob(QUALITY_RECORD_GLOB))
    _require(
        not found,
        f"a W8 quality record is present ({', '.join(found)}) but no W8 "
        "measurement exists at this commit. Either it was fabricated, or a "
        "real result arrived and this gate must be extended to validate its "
        "provenance before it is trusted.",
    )


def validate_repository(repo_root: Path) -> None:
    """Validate both committed candidate specifications, failing closed.

    ``load_inputs`` re-derives the committed AI Hub capability record from its
    own observation before either candidate is looked at, so an edited
    observation, an edited claim boundary, or a record that has grown a
    submitted job stops the run here.
    """

    inputs = load_inputs(repo_root)
    for candidate_id in CANDIDATE_IDS:
        path = repo_root / candidate_config_path(candidate_id)
        _require(
            path.is_file(),
            f"W8 candidate specification is missing: {path}",
        )
        config = _load_yaml(path)
        _require(
            isinstance(config, Mapping),
            f"W8 candidate specification is not a YAML mapping: {path}",
        )
        assert isinstance(config, Mapping)
        _validate_candidate_document(candidate_id, config, inputs)

        expected_text = render_document(
            build_document(candidate_id, inputs=inputs),
            candidate_id,
        )
        if path.read_text(encoding="utf-8") != expected_text:
            raise W8EvidenceError(
                f"{path.name} is not byte-identical to a fresh regeneration; "
                f"run: {GENERATE_COMMAND}"
            )
    _assert_no_measured_quality_record(repo_root)


# ---------------------------------------------------------------------------
# The AI Hub submission boundary: compose a request, stop, print the command
# ---------------------------------------------------------------------------

PRIVATE_OUTPUT_ROOTS = frozenset({".ai-local", "artifacts"})


def repository_root() -> Path:
    """Return this checkout's root, the way ``ai_hub`` derives its own."""

    return Path(__file__).resolve().parents[3]


def assert_private_path(
    value: Path | str,
    field: str,
    *,
    repo_root: Path | None = None,
) -> Path:
    """Refuse any output location that is not external or ignored private storage.

    Mirrors ``ai_hub._private_output_path``: a path outside the repository is
    fine, a path under ``.ai-local/`` or ``artifacts/`` is fine, and anything
    else inside the repository is refused. ``scripts/qualcomm/README.md`` is
    explicit that request files carry machine-local paths and must not be
    committed, so the emitter will not write one where git would see it.
    """

    path = Path(value).expanduser()
    root = (repo_root or repository_root()).resolve()
    try:
        relative = path.resolve(strict=False).relative_to(root)
    except ValueError:
        relative = None
    if relative is not None and (
        not relative.parts or relative.parts[0] not in PRIVATE_OUTPUT_ROOTS
    ):
        raise W8EvidenceError(
            f"{field} must be external to this repository or under ignored "
            f"private storage ({', '.join(sorted(PRIVATE_OUTPUT_ROOTS))}/); "
            f"{path} would be committable, and a request file carries "
            "machine-local paths"
        )
    return path


def stage_logical_name(
    candidate_id: str,
    *,
    context_length: int,
    graph_kind: str,
    role: str,
) -> str:
    """Return a deterministic, path-free logical name for one stage artifact."""

    suffixes = {
        "quantized_source": "quantized.onnx",
        "compiled_model": "qnn-context.bin",
        "input_dataset": "inputs.h5",
        "inference_output": "outputs.h5",
        "raw_profile": "profile-private.json",
    }
    _require(role in suffixes, f"unknown stage artifact role {role!r}")
    name = f"{candidate_id}-S{context_length}-{graph_kind}-{suffixes[role]}"
    _require(
        bool(ai_hub.SAFE_LOGICAL_NAME_PATTERN.fullmatch(name)),
        f"derived logical name {name!r} is not path-free",
    )
    return name


def _existing_artifact(
    path: Path | str | None,
    *,
    field: str,
    logical_name: str,
    missing_hint: str,
) -> dict[str, Any]:
    _require(path is not None, f"{field} is required for this stage")
    assert path is not None
    resolved = Path(path).expanduser()
    _require(
        resolved.is_file(),
        f"{field} is missing: {resolved}. {missing_hint}",
    )
    return {
        "path": str(resolved),
        "logical_name": logical_name,
        "sha256": _sha256_file(resolved),
    }


def _input_specs_for(context_length: int, graph_kind: str) -> dict[str, Any]:
    builders = {
        "prefill": static_cache.build_prefill_contract,
        "decode": static_cache.build_decode_contract,
    }
    _require(
        graph_kind in builders,
        f"graph kind must be one of {sorted(builders)}, found {graph_kind!r}",
    )
    _require(
        context_length in static_cache.CONTEXT_VARIANTS,
        f"context length must be one of {sorted(static_cache.CONTEXT_VARIANTS)}, "
        f"found {context_length}",
    )
    try:
        contract = builders[graph_kind](context_length)
    except static_cache.CacheContractError as exc:
        raise W8EvidenceError(f"cannot build the T12 contract: {exc}") from exc
    specs: dict[str, Any] = {}
    for spec in contract.inputs:
        _require(
            spec.dtype in ai_hub.ALLOWED_DTYPES,
            f"{spec.name}: dtype {spec.dtype!r} is not one the AI Hub adapter accepts",
        )
        specs[spec.name] = {"shape": list(spec.shape), "dtype": spec.dtype}
    return specs


MISSING_QUANTIZED_ARTIFACT_HINT = (
    "No W8 artifact exists at this commit: AIMET is Linux + CUDA only and is "
    "specified rather than installed, and this repository's AI Hub adapter "
    "implements compile, inference, and profile but no quantize stage. Produce "
    "the artifact on a Lane B host first, then re-run this emitter."
)


def build_stage_request(
    candidate: Mapping[str, Any],
    stage: str,
    *,
    context_length: int,
    graph_kind: str = "prefill",
    quantized_artifact: Path | str | None = None,
    compiled_artifact: Path | str | None = None,
    predecessor_manifest: Path | str | None = None,
    input_dataset: Path | str | None = None,
    output_path: Path | str,
    device: Mapping[str, Any] | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Compose one schema-v2 AI Hub stage request and submit nothing.

    Everything the request declares that is not a caller-supplied local path
    comes from committed material: the candidate specification, the frozen T12
    tensor contract for the requested variant, the pinned client and QAIRT
    versions, and the plan's target-device policy. ``retry`` is false and the
    timeout is bounded so a future authorized run has one bounded attempt.

    This function never imports ``qai_hub``, never constructs a backend, and
    never calls any submission path. It stops at a mapping.
    """

    _require(stage in STAGE_ORDER, f"unsupported stage {stage!r}")
    candidate_id = str(candidate.get("candidate_id"))
    _require(
        candidate_id in CANDIDATE_IDS,
        f"the supplied specification is not a W8 candidate: {candidate_id!r}",
    )
    _require(
        int(timeout_seconds) > 0,
        "timeout_seconds must be a bounded positive integer",
    )

    selector = dict(device) if device is not None else dict(PRIMARY_DEVICE)
    _require(
        isinstance(selector.get("name"), str) and bool(selector["name"]),
        "the device selector needs an exact device name",
    )
    request: dict[str, Any] = {
        "schema_version": ai_hub.SCHEMA_VERSION,
        "stage": stage,
        "client_version": QAI_HUB_CLIENT_VERSION,
        "device": selector,
        "runtime": {"name": RUNTIME_NAME, "version": QAIRT_VERSION},
        "options": STAGE_OPTIONS[stage],
        "job_name": (
            f"slm-lab-t41-{candidate_id}-{stage}-S{context_length}-{graph_kind}"
        ),
        "timeout_seconds": int(timeout_seconds),
        "retry": False,
    }

    def logical(role: str) -> str:
        return stage_logical_name(
            candidate_id,
            context_length=context_length,
            graph_kind=graph_kind,
            role=role,
        )

    if stage == "compile":
        request["source_artifact"] = _existing_artifact(
            quantized_artifact,
            field="quantized_artifact",
            logical_name=logical("quantized_source"),
            missing_hint=MISSING_QUANTIZED_ARTIFACT_HINT,
        )
        request["input_specs"] = _input_specs_for(context_length, graph_kind)
        request["output_artifact"] = str(
            assert_private_path(output_path, "output_artifact", repo_root=repo_root)
        )
        request["output_logical_name"] = logical("compiled_model")
    else:
        _require(
            predecessor_manifest is not None,
            f"{stage} requires the sanitized compile manifest as its predecessor",
        )
        assert predecessor_manifest is not None
        manifest_path = Path(predecessor_manifest).expanduser()
        _require(
            manifest_path.is_file(),
            f"predecessor_manifest is missing: {manifest_path}. The compile "
            "stage has not been run for this candidate.",
        )
        request["predecessor_manifest"] = str(manifest_path)
        request["compiled_artifact"] = _existing_artifact(
            compiled_artifact,
            field="compiled_artifact",
            logical_name=logical("compiled_model"),
            missing_hint=(
                "The compiled artifact is produced by the compile stage; run "
                "it before requesting this stage."
            ),
        )
        if stage == "inference":
            request["input_dataset"] = _existing_artifact(
                input_dataset,
                field="input_dataset",
                logical_name=logical("input_dataset"),
                missing_hint=(
                    "An AI Hub-compatible HDF5 dataset built from the frozen "
                    "T10 prompt for this variant is required."
                ),
            )
            request["output_artifact"] = str(
                assert_private_path(output_path, "output_artifact", repo_root=repo_root)
            )
            request["output_logical_name"] = logical("inference_output")
        else:
            request["raw_profile_output"] = str(
                assert_private_path(
                    output_path, "raw_profile_output", repo_root=repo_root
                )
            )
            request["raw_profile_logical_name"] = logical("raw_profile")
    return request


def stage_command(stage: str, request_path: Path, manifest_path: str) -> str:
    """Return the exact command a future authorized session would run."""

    return (
        f"PYTHONPATH=src python3 {STAGE_SCRIPTS[stage]} "
        f"--request {request_path} --manifest {manifest_path}"
    )


def write_stage_request(
    request: Mapping[str, Any],
    path: Path | str,
    *,
    repo_root: Path | None = None,
) -> Path:
    """Write one request to a private location and refuse anywhere else."""

    target = assert_private_path(path, "request path", repo_root=repo_root)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise W8EvidenceError(f"cannot write {target}: {exc}") from exc
    return target


# ---------------------------------------------------------------------------
# Readiness record
# ---------------------------------------------------------------------------


def precision_evidence_path(repo_root: Path, candidate_id: str) -> Path:
    """Return where a future session drops one candidate's precision evidence."""

    return (
        repo_root
        / DEFAULT_EVIDENCE_DIRECTORY
        / PRECISION_EVIDENCE_TEMPLATE.format(candidate_id=candidate_id)
    )


def load_precision_evidence(
    repo_root: Path,
    candidate_id: str,
) -> tuple[Mapping[str, Any] | None, str]:
    """Load one candidate's precision evidence if a session has produced any.

    Returns ``(evidence, source)``. An absent file is not an error: it is the
    expected state at this commit and it is what keeps the state ``specified``.
    """

    path = precision_evidence_path(repo_root, candidate_id)
    if not path.is_file():
        return None, "absent_at_this_commit"
    document = _load_json(path)
    _require(
        isinstance(document, Mapping),
        f"{path.name}: precision evidence is not a JSON object",
    )
    assert isinstance(document, Mapping)
    return document, path.relative_to(repo_root).as_posix()


def build_readiness_record(repo_root: Path) -> dict[str, Any]:
    """Build the committed T41 readiness record.

    Runs the full offline validation first, so the record cannot claim a
    frozen, consistent specification that does not validate.
    """

    root = Path(repo_root).resolve()
    validate_repository(root)
    inputs = load_inputs(root)

    candidates: list[dict[str, Any]] = []
    ledger: dict[str, Any] | None = None
    for candidate_id in CANDIDATE_IDS:
        config_path = root / candidate_config_path(candidate_id)
        config = _load_yaml(config_path)
        assert isinstance(config, Mapping)
        candidate = config["candidate"]
        evidence, evidence_source = load_precision_evidence(root, candidate_id)
        finding = assess_precision_state(evidence)
        if ledger is None:
            ledger = dict(config["evidence_requirements"])
        candidates.append(
            {
                "candidate_id": candidate_id,
                "plan_matrix_row": PLAN_MATRIX_ROWS[candidate_id],
                "config_path": candidate_config_path(candidate_id).as_posix(),
                "config_file_sha256": _sha256_file(config_path),
                "candidate_canonical_json_sha256": config[
                    "candidate_canonical_json_sha256"
                ],
                "precision_state": finding.state,
                "precision_state_scope": finding.scope,
                "precision_evidence": {
                    "source": evidence_source,
                    **finding.as_dict(),
                },
                "weight_storage_projection": candidate["weight_storage_projection"],
                "kv_cache_applied_dtype": candidate["kv_cache"]["applied_dtype"],
                "kv_cache_satisfied_without_contract_change": candidate["kv_cache"][
                    "satisfied_without_contract_change"
                ],
            }
        )

    assert ledger is not None
    capability = inputs["capability"]
    routes = build_deployment_routes(
        aimet_versions=inputs["aimet_versions"],
        capability=capability,
    )
    specification_frozen = all(
        entry["precision_state"] in PRECISION_STATES for entry in candidates
    )
    repository = _git_commit(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "record_type": RECORD_TYPE,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository": {
            # No checkout path: a record is committed evidence read on other
            # machines, where an absolute worktree path is both meaningless and
            # a private detail. The commit identifies the tree.
            "git_commit": repository["commit"],
            "git_tree_clean": repository["tree_clean"],
        },
        "inputs": {
            "calibration_dataset_revision": inputs["calibration"][
                "calibration_dataset_revision"
            ],
            "benchmark_protocol_sha256": inputs["protocol"]["contract_sha256"],
            "baseline_manifest_directory": DEFAULT_MANIFEST_DIRECTORY.as_posix(),
            "baseline_precision": "float16",
            "graph_inventory": DEFAULT_GRAPH_INVENTORY.as_posix(),
            "ai_hub_capability_record": DEFAULT_CAPABILITY_RECORD.as_posix(),
        },
        "candidates": candidates,
        "evidence_requirements": ledger,
        "ai_hub_capability": {
            "path": DEFAULT_CAPABILITY_RECORD.as_posix(),
            "canonical_json_sha256": canonical_json_sha256(capability),
            "observation_date": capability["observation_date"],
            "client_version": capability["client_version"],
            "submitted_jobs": capability["submitted_jobs"],
            "device_minutes_consumed": capability["device_minutes_consumed"],
            "cost": capability["cost"],
            "quantize_entry_point": capability["observation"]["quantize_entry_point"],
            "quantize_dtypes": list(capability["observation"]["quantize_dtypes"]),
            "candidate_requests": {
                candidate_id: {
                    "weights_dtype": entry["weights_dtype"],
                    "activations_dtype": entry["activations_dtype"],
                }
                for candidate_id, entry in capability["candidate_quantize_requests"][
                    "by_candidate"
                ].items()
            },
            "kv_cache_dtype": dict(capability["kv_cache_dtype"]),
            "establishes": list(capability["establishes"]),
            "does_not_establish": list(capability["does_not_establish"]),
            "rebuild_command": CAPABILITIES_OFFLINE_COMMAND,
        },
        "deployment_routes": {
            "lane_a_available": routes["lane_a_ai_hub_workbench"]["available"],
            "lane_a_blocked_on": routes["lane_a_ai_hub_workbench"]["blocked_on"],
            "lane_a_blocked_on_owners": routes["lane_a_ai_hub_workbench"][
                "blocked_on_owners"
            ],
            "lane_a_authorization": routes["lane_a_ai_hub_workbench"]["authorization"],
            "lane_b_available": routes["lane_b_local_aimet"]["available"],
            "lane_b_blocked_on": routes["lane_b_local_aimet"]["blocked_on"],
            "boundary": routes["boundary"],
        },
        "claim_boundary": {
            "establishes": list(ESTABLISHES),
            "does_not_establish": list(DOES_NOT_ESTABLISH),
        },
        # Scope-named on purpose. There is no `ready: true` here, because the
        # only thing released is the preparation of a submission that this
        # session was not authorized to make.
        "released_for_submission_preparation_only": specification_frozen,
        "released_for_submission_preparation_only_meaning": (
            "Both candidate specifications regenerate byte-identically from "
            "committed inputs, every binding still hashes to its recorded "
            "value, and the request emitter can compose a schema-v2 stage "
            "request. It does not mean a request was written, a job was "
            "submitted, a weight was quantized, or a precision was achieved. "
            "Submission is permitted as of "
            f"{CAPABILITY_OBSERVATION_DATE} and the scope of this release did "
            "not widen when it was: preparation is still all there is, because "
            "there is still no artifact to submit."
        ),
        "commands": {
            "generate": GENERATE_COMMAND,
            "check": CHECK_COMMAND,
            "status": STATUS_COMMAND,
            "record": RECORD_COMMAND,
            "capabilities": CAPABILITIES_COMMAND,
            "capabilities_offline": CAPABILITIES_OFFLINE_COMMAND,
            "compare": COMPARE_COMMAND,
            "request": REQUEST_COMMAND,
            "tests": TESTS_COMMAND,
        },
    }


def default_readiness_path(repo_root: Path, record: Mapping[str, Any]) -> Path:
    """Return the conventional evidence path for one readiness record."""

    day = str(record["generated_at"])[:10]
    return (
        repo_root / DEFAULT_EVIDENCE_DIRECTORY / f"{READINESS_RECORD_PREFIX}{day}.json"
    )


def write_readiness_record(path: Path, record: Mapping[str, Any]) -> None:
    """Write one compact, committed readiness record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def format_status(record: Mapping[str, Any]) -> str:
    """Render a short, unambiguous human summary of the W8 readiness state."""

    lines = [
        f"T41 W8 candidates at {record['repository']['git_commit']} "
        f"(tree clean: {record['repository']['git_tree_clean']})",
    ]
    for candidate in record["candidates"]:
        totals = candidate["weight_storage_projection"]["totals"]
        lines.append(
            f"  {candidate['candidate_id']} ({candidate['plan_matrix_row']}): "
            f"precision_state={candidate['precision_state']} "
            f"(scope: {candidate['precision_state_scope']})"
        )
        lines.append(
            "    evidence: "
            f"{candidate['precision_evidence']['source']}; unsatisfied "
            f"{len(candidate['precision_evidence']['unsatisfied_checks'])} check(s)"
        )
        lines.append(
            "    analytic weight projection: "
            f"{totals['float16_weight_bytes']} -> "
            f"{totals['candidate_weight_bytes']} bytes "
            f"(x{totals['weight_byte_ratio_float16_over_candidate']}), "
            "not an artifact size"
        )
        lines.append(
            f"    kv cache: {candidate['kv_cache_applied_dtype']} "
            "(frozen T12 contract; "
            f"plan row satisfied without a contract change: "
            f"{candidate['kv_cache_satisfied_without_contract_change']})"
        )
    for entry in record["evidence_requirements"]["summary"]:
        lines.append(f"  ledger {entry['status']}: {entry['count']}")
    lines.append(
        "  released for submission preparation only: "
        f"{record['released_for_submission_preparation_only']}"
    )
    lines.append(
        "  lane A available: "
        f"{record['deployment_routes']['lane_a_available']}; "
        "lane B available: "
        f"{record['deployment_routes']['lane_b_available']}"
    )
    capability = record["ai_hub_capability"]
    lines.append(
        "  lane A blocked on: "
        f"{', '.join(record['deployment_routes']['lane_a_blocked_on'])}"
    )
    lines.append(
        "  AI Hub capability observed "
        f"{capability['observation_date']} with client "
        f"{capability['client_version']}: "
        f"{capability['quantize_entry_point']} accepts "
        f"{', '.join(capability['quantize_dtypes'])}; "
        f"jobs submitted {capability['submitted_jobs']}, device minutes "
        f"{capability['device_minutes_consumed']}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the ``slm_lab.quantization.w8`` command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "generate",
        help="rewrite configs/quantization/w8/*.yaml from committed inputs",
    )
    subparsers.add_parser(
        "check",
        help="full offline validation; the CI and test gate",
    )
    subparsers.add_parser(
        "status",
        help="print the precision state, projection, and ledger summary",
    )
    record = subparsers.add_parser(
        "record",
        help="validate and write the committed readiness record",
    )
    record.add_argument("--output", type=Path)
    record.add_argument("--json", action="store_true")

    capabilities = subparsers.add_parser(
        "capabilities",
        help=(
            "record the read-only public AI Hub capability observation; this "
            "reads names and one signature and never submits a job"
        ),
    )
    capabilities.add_argument(
        "--offline-input",
        type=Path,
        help=(
            "rebuild the record from a saved sanitized query, or from a "
            "committed capability record, without touching the network"
        ),
    )
    capabilities.add_argument("--output", type=Path)
    capabilities.add_argument("--json", action="store_true")

    compare = subparsers.add_parser(
        "compare",
        help="compare one W8 quality result against a floating baseline result",
    )
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)

    request = subparsers.add_parser(
        "request",
        help=(
            "compose one AI Hub schema-v2 stage request and stop; this never "
            "submits anything"
        ),
    )
    request.add_argument("--candidate", choices=list(CANDIDATE_IDS), required=True)
    request.add_argument("--stage", choices=list(STAGE_ORDER), required=True)
    request.add_argument(
        "--context",
        type=int,
        choices=sorted(static_cache.CONTEXT_VARIANTS),
        required=True,
    )
    request.add_argument(
        "--graph",
        choices=["prefill", "decode"],
        default="prefill",
    )
    request.add_argument("--quantized-artifact", type=Path)
    request.add_argument("--compiled-artifact", type=Path)
    request.add_argument("--predecessor-manifest", type=Path)
    request.add_argument("--input-dataset", type=Path)
    request.add_argument("--output-artifact", type=Path, required=True)
    request.add_argument("--request-out", type=Path, required=True)
    request.add_argument(
        "--stage-manifest",
        default="results/processed/qualcomm/t41-{stage}-manifest.json",
        help="public manifest path printed in the follow-up command",
    )
    request.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    return parser


def _run_request(args: argparse.Namespace, repo_root: Path) -> int:
    config_path = repo_root / candidate_config_path(args.candidate)
    config = _load_yaml(config_path)
    _require(
        isinstance(config, Mapping),
        f"W8 candidate specification is not a YAML mapping: {config_path}",
    )
    assert isinstance(config, Mapping)
    request = build_stage_request(
        config["candidate"],
        args.stage,
        context_length=args.context,
        graph_kind=args.graph,
        quantized_artifact=args.quantized_artifact,
        compiled_artifact=args.compiled_artifact,
        predecessor_manifest=args.predecessor_manifest,
        input_dataset=args.input_dataset,
        output_path=args.output_artifact,
        timeout_seconds=args.timeout_seconds,
        repo_root=repo_root,
    )
    written = write_stage_request(request, args.request_out, repo_root=repo_root)
    manifest = str(args.stage_manifest).format(stage=args.stage)
    print(f"wrote {args.stage} request: {written}")
    print(
        "this session submitted nothing. Submission is permitted as of "
        f"{CAPABILITY_OBSERVATION_DATE}; what is missing is a W8 artifact, "
        "which needs a Lane B host or a quantize-stage adapter this repository "
        "does not have."
    )
    print("an authorized session would run, from the repository root:")
    print(f"  {stage_command(args.stage, written, manifest)}")
    print(
        "do not commit the request: it carries machine-local paths "
        "(scripts/qualcomm/README.md)."
    )
    return 0


def _run_capabilities(args: argparse.Namespace, repo_root: Path) -> int:
    """Observe or rebuild the AI Hub capability record; submit nothing."""

    if args.offline_input is not None:
        observation = load_offline_observation(args.offline_input)
        source = f"offline rebuild from {args.offline_input}"
    else:
        observation = query_live_capability(
            observation_date=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
        )
        source = "live read-only capability query"

    record = build_capability_record(observation)
    output = args.output or default_capability_path(repo_root, record)
    write_capability_record(output, record)
    if args.json:
        print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    entry_points = record["observation"]["submit_entry_points"]
    devices = record["observation"]["devices"]
    print(f"AI Hub capability record ({source}): {output}")
    print(
        f"  observed {record['observation_date']} with client "
        f"{record['client_version']}"
    )
    print(
        f"  jobs submitted: {record['submitted_jobs']}; device minutes "
        f"consumed: {record['device_minutes_consumed']}; cost: {record['cost']}"
    )
    print(f"  submit_* entry points: {', '.join(entry_points)}")
    print(
        f"  {record['observation']['quantize_entry_point']} dtypes: "
        f"{', '.join(record['observation']['quantize_dtypes'])}"
    )
    for candidate_id, entry in record["candidate_quantize_requests"][
        "by_candidate"
    ].items():
        print(f"  {candidate_id}: {entry['request']}")
    print(
        f"  plan 3.2 targets live: "
        f"{', '.join(str(device['name']) for device in devices)} "
        f"(service lists {record['observation']['device_count']} devices)"
    )
    print(
        "  this establishes an API surface and nothing about compilation, "
        "placement, latency, or accuracy."
    )
    return 0


def _run_compare(args: argparse.Namespace, repo_root: Path) -> int:
    baseline = _load_json(args.baseline)
    candidate = _load_json(args.candidate)
    _require(
        isinstance(baseline, Mapping) and isinstance(candidate, Mapping),
        "both quality records must be JSON objects",
    )
    assert isinstance(baseline, Mapping) and isinstance(candidate, Mapping)
    comparison = compare_quality(baseline, candidate, root=repo_root)
    print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the W8 candidate CLI; return zero only when the command succeeded."""

    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        if args.command == "generate":
            for path in generate_repository(repo_root):
                print(f"wrote {path}")
        elif args.command == "check":
            validate_repository(repo_root)
            print(f"T41 W8 candidate check passed: {repo_root}")
        elif args.command == "status":
            print(format_status(build_readiness_record(repo_root)))
        elif args.command == "record":
            record = build_readiness_record(repo_root)
            output = args.output or default_readiness_path(repo_root, record)
            write_readiness_record(output, record)
            if args.json:
                print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print(format_status(record))
                print(f"  evidence written: {output}")
        elif args.command == "capabilities":
            return _run_capabilities(args, repo_root)
        elif args.command == "compare":
            return _run_compare(args, repo_root)
        else:
            return _run_request(args, repo_root)
    except (
        W8EvidenceError,
        static_cache.CacheContractError,
        ai_hub.AiHubAdapterError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
