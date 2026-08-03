"""Target-neutral graph transformations that produce the T22 qnn_candidate stage.

:mod:`slm_lab.graph.qnn.transforms` holds the declarative catalogue loader and
the passes themselves; :mod:`slm_lab.graph.qnn.build` is the CLI that reads a
committed T20 manifest, applies the catalogue, writes the candidate graphs
beneath the external artifact root, and writes the committed T22 manifest.

Nothing in this package modifies a reference artifact, and nothing in it claims
compiler acceptance. It rewrites graphs and measures what the rewrite did.
"""

from __future__ import annotations

from slm_lab.graph.qnn.transforms import (
    QnnTransformError,
    TransformPass,
    load_transform_catalogue,
)

__all__ = [
    "QnnTransformError",
    "TransformPass",
    "load_transform_catalogue",
]
