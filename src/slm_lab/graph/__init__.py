"""Dependency-free ONNX graph reading and deployment-risk inspection.

Attributes resolve lazily so that ``python -m slm_lab.graph.inspection`` does
not import the inspection module a second time through this initialiser, and
so that importing one submodule never forces the other to load.
"""

from typing import Any

_EXPORTS = {
    "AttributeInfo": "slm_lab.graph.onnx_reader",
    "Finding": "slm_lab.graph.inspection",
    "GraphInspection": "slm_lab.graph.inspection",
    "GraphInspectionError": "slm_lab.graph.inspection",
    "GraphSummary": "slm_lab.graph.onnx_reader",
    "InitializerInfo": "slm_lab.graph.onnx_reader",
    "NodeInfo": "slm_lab.graph.onnx_reader",
    "OnnxReadError": "slm_lab.graph.onnx_reader",
    "RiskRule": "slm_lab.graph.inspection",
    "TensorShape": "slm_lab.graph.onnx_reader",
    "ValueInfo": "slm_lab.graph.onnx_reader",
    "inspect_graph": "slm_lab.graph.inspection",
    "load_risk_rules": "slm_lab.graph.inspection",
    "parse_onnx_model": "slm_lab.graph.onnx_reader",
    "rank_findings": "slm_lab.graph.inspection",
    "read_onnx_model": "slm_lab.graph.onnx_reader",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module

    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return list(__all__)
