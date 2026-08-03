"""Quantization calibration, parity preflight, and AIMET-facing contracts."""

from __future__ import annotations

import importlib
from typing import Any


#: Public name -> defining submodule. Resolved lazily so that
#: ``python -m slm_lab.quantization.calibration`` does not import the module a
#: second time through the package, which runpy warns about.
_EXPORTS: dict[str, str] = {
    "CalibrationSample": "calibration",
    "CalibrationValidationError": "calibration",
    "build_calibration_samples": "calibration",
    "build_corpus": "calibration",
    "build_coverage_measurements": "calibration",
    "build_document": "calibration",
    "build_prefill_tensors": "calibration",
    "calibration_dataset_revision": "calibration",
    "validate_documents": "calibration",
    "validate_repository": "calibration",
    "BaselineParityError": "parity",
    "check_baseline_parity": "parity",
    "default_evidence_path": "parity",
    "expected_artifact_files": "parity",
    "format_report": "parity",
    "load_manifests": "parity",
    "numerical_parity_requirement": "parity",
    "resolve_artifact_root": "parity",
    "write_evidence": "parity",
    # `w8` also defines build_document, generate_repository, load_inputs, and
    # validate_repository. Those names collide with `calibration`'s and are
    # deliberately not exported: this map is flat, so one name can only ever
    # mean one module. Reach them through `slm_lab.quantization.w8` directly.
    "W8EvidenceError": "w8",
    "assess_precision_state": "w8",
    "build_candidate": "w8",
    "build_readiness_record": "w8",
    "build_stage_request": "w8",
    "compare_quality": "w8",
    "default_readiness_path": "w8",
    "precision_state": "w8",
    "precision_state_scope": "w8",
    "weight_storage_projection": "w8",
    "write_readiness_record": "w8",
    "write_stage_request": "w8",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Import a public quantization symbol from its defining submodule."""

    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the package's lazily exported public names."""

    return sorted({*globals(), *_EXPORTS})
