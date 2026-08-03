#!/usr/bin/env python3
"""Audit committed numeric claims about the T20 reference ONNX graphs.

Two modes:

``citations``
    Strict check. Extracts every claim in the committed hand-written records
    that *binds* a literal to a named reference-graph subject -- a digest, a
    byte size, a node count, or an operator count -- and compares it against
    the measured value. Also cross-checks the generated evidence files against
    each other, and optionally against a re-hash of the artifacts themselves.
    Exits non-zero listing every disagreement.

``claims``
    Exhaustive review queue. Extracts *every* numeric token from the
    hand-written documents and puts each occurrence in exactly one bucket:

    ``MOVES``         the literal matches a pre-promotion measured value and a
                      different post-promotion value exists: edit it.
    ``STATIC``        it matches a measured value that did not move.
    ``AMBIGUOUS``     its readings imply *different actions* -- at least one
                      moved and at least one did not, or they moved to
                      different values. Every reading is printed; none is
                      picked. (Readings that share a fate are not an
                      ambiguity: they are one decision, and they are still
                      listed beside the verdict.)
    ``UNCLASSIFIED``  no measured value explains it. Printed in full.

    The tool asserts that these four account for every token it extracted.

    ``MOVES`` can only be populated when the baseline snapshot differs from
    the current one, which is what ``--baseline-ref`` selects. Before the
    re-export lands the two are the same and the bucket is empty; that is the
    truthful answer, and the report says so rather than implying a clean tree.

Measured truth is read at run time from the generated evidence. Nothing about
the graphs is hardcoded here: change the exports, regenerate the evidence, and
this tool changes its verdict without being edited.

What this tool does NOT prove
-----------------------------

* ``MOVES`` and ``AMBIGUOUS`` are *candidate* lists. Classification is by
  numeric coincidence plus context, not by understanding the sentence. A human
  confirms every one before editing.
* ``UNCLASSIFIED`` is a review queue, not a clean bill of health. A number
  landing there means no measured value explains it -- which may mean it is
  prose arithmetic, a line reference, a date, or a stale claim whose subject
  this tool cannot see.
* ``STATIC`` means "a measured value with this literal exists and did not
  move". It does not mean the sentence around it is true.
* ``citations`` proves agreement only for claims it can *bind* to a subject
  structurally. A claim it cannot bind is not checked here; it is left to
  ``claims``.
* Nothing here establishes that the graphs are correct, loadable, or numerically
  faithful. It compares records to records, and records to file digests.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# Declarative scope. Extend these constants; do not extend the logic.
# --------------------------------------------------------------------------

# Generated evidence. Every measured value the tool knows comes from here.
EXPORT_CONFIG = "configs/models/qwen3-0.6b-onnx-export.json"
MODEL_CONTRACT = "configs/models/qwen3-0.6b.yaml"
MANIFEST_GLOB = "results/manifests/onnx/S*.json"
INSPECTION_GLOB = "results/graph/S*.json"
PARITY_GLOB = "results/graph/parity/S*-ort-cpu.json"
QUANTIZATION_GLOB = "results/quantization/*.json"
RISK_RULES = "configs/graph/onnx-risk-rules-v1.json"

EVIDENCE_GLOBS = (
    MANIFEST_GLOB,
    INSPECTION_GLOB,
    PARITY_GLOB,
    QUANTIZATION_GLOB,
)

# Hand-written documents that describe the reference graphs.
#
# role="reconcile"  the document asserts current fact and must be edited when
#                   a value it cites moves.
# role="historical" the document records a past measurement. Its numbers are
#                   still enumerated -- an unexamined number is the failure
#                   mode this tool exists to prevent -- but editing one would
#                   falsify a record, so they are reported separately and are
#                   not bound by ``citations``.
CLAIM_DOCUMENTS: Tuple[Tuple[str, str], ...] = (
    ("docs/results/onnx/graph-inspection.md", "reconcile"),
    ("docs/results/onnx/ort-cpu-parity.md", "reconcile"),
    ("results/graph/README.md", "reconcile"),
    ("ai/worklogs/2026-07-30-T20-onnx-export-matrix.md", "reconcile"),
    # YAML, not Markdown. Read as raw text on purpose: the numeric claims live
    # inside folded scalars (the LEARN-10 `lede`), so a YAML parse would lose
    # the line numbers a reviewer needs and would gain nothing.
    ("configs/learning/checkpoints.yaml", "reconcile"),
    ("results/quantization/README.md", "reconcile"),
    (
        "docs/failures/runtime/2026-08-02-t20-fp16-prefill-pad-unloadable.md",
        "historical",
    ),
    ("ai/plans/active/T23-prefill-reexport-promotion.md", "historical"),
)

# Markdown table columns that name a measured quantity. Keys are normalized
# header text (lowercased, backticks and punctuation stripped).
COLUMN_MAP: Dict[str, Tuple[str, Optional[str]]] = {
    # header -> (fact family, graph kind implied by the column, if any)
    "nodes": ("nodes", None),
    "node count": ("nodes", None),
    "op types": ("optypes", None),
    "operator types": ("optypes", None),
    "inputs": ("inputs", None),
    "outputs": ("outputs", None),
    "initializers external": ("initializers", None),
    "non static dims": ("dynamic_dims", None),
    "value info": ("value_info", None),
    "onnx bytes": ("onnx_bytes", None),
    "bytes": ("onnx_bytes", None),
    "capacity c": ("cache_capacity", None),
    "graph sha 256": ("digest:graph", None),
    "manifest sha 256": ("digest:manifest", None),
    "prefill count": ("finding", "prefill"),
    "decode count": ("finding", "decode"),
}

GRAPH_KINDS = ("prefill", "decode")

# --------------------------------------------------------------------------
# Readers: worktree, or any git ref.
# --------------------------------------------------------------------------


class Reader:
    """Read repository files from some snapshot."""

    name = "?"

    def read_bytes(self, relpath: str) -> Optional[bytes]:
        raise NotImplementedError

    def paths(self) -> Sequence[str]:
        raise NotImplementedError

    def read_text(self, relpath: str) -> Optional[str]:
        raw = self.read_bytes(relpath)
        if raw is None:
            return None
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def glob(self, pattern: str) -> List[str]:
        return sorted(p for p in self.paths() if fnmatch.fnmatch(p, pattern))


class WorktreeReader(Reader):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.name = "worktree"
        self._paths: Optional[List[str]] = None

    def read_bytes(self, relpath: str) -> Optional[bytes]:
        path = self.root / relpath
        if not path.is_file():
            return None
        return path.read_bytes()

    def paths(self) -> Sequence[str]:
        if self._paths is None:
            out = subprocess.run(
                ["git", "-C", str(self.root), "ls-files", "-z"],
                check=True,
                capture_output=True,
            ).stdout
            tracked = [p for p in out.decode("utf-8").split("\0") if p]
            # Untracked-but-present evidence still counts as the worktree state.
            extra: List[str] = []
            for pattern in EVIDENCE_GLOBS:
                for path in sorted(self.root.glob(pattern)):
                    rel = path.relative_to(self.root).as_posix()
                    if rel not in tracked:
                        extra.append(rel)
            self._paths = sorted(set(tracked) | set(extra))
        return self._paths


class GitReader(Reader):
    def __init__(self, root: Path, ref: str) -> None:
        self.root = root
        self.ref = ref
        self.name = f"git:{ref}"
        self._paths: Optional[List[str]] = None

    def read_bytes(self, relpath: str) -> Optional[bytes]:
        proc = subprocess.run(
            ["git", "-C", str(self.root), "show", f"{self.ref}:{relpath}"],
            capture_output=True,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout

    def paths(self) -> Sequence[str]:
        if self._paths is None:
            out = subprocess.run(
                ["git", "-C", str(self.root), "ls-tree", "-r", "-z",
                 "--name-only", self.ref],
                check=True,
                capture_output=True,
            ).stdout
            self._paths = sorted(p for p in out.decode("utf-8").split("\0") if p)
        return self._paths


# --------------------------------------------------------------------------
# Facts
# --------------------------------------------------------------------------


@dataclass
class Fact:
    identity: str
    kind: str  # digest | count | bytes | percent | ratio | version | date | number
    label: str
    observations: List[Tuple[object, str]] = field(default_factory=list)

    @property
    def value(self) -> object:
        return self.observations[0][0]

    @property
    def values(self) -> List[object]:
        seen: List[object] = []
        for value, _ in self.observations:
            if value not in seen:
                seen.append(value)
        return seen

    @property
    def sources(self) -> List[str]:
        return sorted({source for _, source in self.observations})

    @property
    def conflicted(self) -> bool:
        return len(self.values) > 1


class Snapshot:
    """Every measured value one snapshot of the repository can supply."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.facts: Dict[str, Fact] = {}
        self.notes: List[str] = []
        self.pending_details: List[Tuple[str, str, str, str]] = []
        self.pending_digests: List[Tuple[str, str, str]] = []

    def add(
        self,
        identity: str,
        kind: str,
        label: str,
        value: object,
        source: str,
    ) -> None:
        if value is None:
            return
        fact = self.facts.get(identity)
        if fact is None:
            fact = Fact(identity=identity, kind=kind, label=label)
            self.facts[identity] = fact
        fact.observations.append((value, source))

    # -- indexes -----------------------------------------------------------

    def int_index(self) -> Dict[int, List[str]]:
        index: Dict[int, List[str]] = {}
        for fact in self.facts.values():
            for value in fact.values:
                if isinstance(value, int) and not isinstance(value, bool):
                    index.setdefault(value, []).append(fact.identity)
        return index

    def float_facts(self, kinds: Sequence[str]) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        for fact in self.facts.values():
            if fact.kind not in kinds:
                continue
            for value in fact.values:
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    out.append((fact.identity, float(value)))
        return out

    def digests(self) -> Dict[str, List[str]]:
        index: Dict[str, List[str]] = {}
        for fact in self.facts.values():
            if fact.kind != "digest":
                continue
            for value in fact.values:
                index.setdefault(str(value), []).append(fact.identity)
        return index

    def string_index(self, kinds: Sequence[str]) -> Dict[str, List[str]]:
        index: Dict[str, List[str]] = {}
        for fact in self.facts.values():
            if fact.kind not in kinds:
                continue
            for value in fact.values:
                index.setdefault(str(value), []).append(fact.identity)
        return index

    def conflicts(self) -> List[Fact]:
        return sorted(
            (f for f in self.facts.values() if f.conflicted),
            key=lambda f: f.identity,
        )


# --------------------------------------------------------------------------
# Harvesting
# --------------------------------------------------------------------------

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{40}(?![0-9A-Za-z])")
VARIANT_RE = re.compile(r"\bS(\d+)\b")
DOTTED_RE = re.compile(r"^\d+(?:\.\d+){2,}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _variant_of(path: str) -> Optional[str]:
    stem = Path(path).name.split(".")[0]
    match = re.fullmatch(r"(S\d+)(?:-ort-cpu)?", stem)
    return match.group(1) if match else None


def _json(reader: Reader, relpath: str) -> Optional[object]:
    text = reader.read_text(relpath)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _walk_digests(node: object, prefix: str) -> Iterator[Tuple[str, str]]:
    """Yield (json pointer, sha256) for every 64-hex string in a document."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_digests(value, f"{prefix}/{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk_digests(value, f"{prefix}/{i}")
    elif isinstance(node, str) and SHA256_RE.match(node):
        yield prefix, node


def harvest(reader: Reader, artifact_root: Optional[Path] = None,
            rehash_external: bool = False) -> Snapshot:
    snap = Snapshot(reader.name)

    _harvest_export_config(reader, snap)
    _harvest_model_contract(reader, snap)
    _harvest_manifests(reader, snap)
    _harvest_inspection(reader, snap)
    _harvest_parity(reader, snap)
    _harvest_quantization(reader, snap)
    _harvest_risk_rules(reader, snap)
    _harvest_revisions(reader, snap)
    _harvest_dates(reader, snap)
    _harvest_residual_digests(snap)
    _harvest_finding_details(snap)
    _derive(snap)
    if artifact_root is not None:
        _harvest_artifacts(snap, artifact_root, rehash_external)
    return snap


def _harvest_export_config(reader: Reader, snap: Snapshot) -> None:
    doc = _json(reader, EXPORT_CONFIG)
    if not isinstance(doc, dict):
        snap.notes.append(f"missing or unreadable: {EXPORT_CONFIG}")
        return
    src = EXPORT_CONFIG
    raw = reader.read_bytes(EXPORT_CONFIG)
    if raw is not None:
        snap.add(
            "digest:file:" + EXPORT_CONFIG, "digest",
            "sha256 of the export config file itself",
            hashlib.sha256(raw).hexdigest(), src,
        )
    att = doc.get("evidence_attestation")
    if isinstance(att, dict):
        for variant, kinds in (att.get("graph_sha256") or {}).items():
            if not isinstance(kinds, dict):
                continue
            for kind, digest in kinds.items():
                snap.add(
                    f"digest:graph:{variant}:{kind}", "digest",
                    f"attested sha256 of {variant}/{kind}.onnx", digest, src,
                )
        snap.add("digest:external_data", "digest",
                 "shared external-data sidecar sha256",
                 att.get("external_data_sha256"), src)
        snap.add("digest:source_artifact", "digest",
                 "pinned source model.safetensors sha256",
                 att.get("source_artifact_sha256"), src)
        snap.add("version:runtime_python", "version",
                 "attested exporter interpreter version",
                 att.get("runtime_python_version"), src)
    for name, version in (doc.get("packages") or {}).items():
        snap.add(f"version:package:{name}", "version",
                 f"pinned {name} version", version, src)
    export = doc.get("export") or {}
    snap.add("count:opset", "count", "ONNX opset version", export.get("opset"), src)
    for ctx in doc.get("contexts") or []:
        snap.add(f"count:context_length:S{ctx}", "count",
                 f"S{ctx} context length", ctx, src)


def _harvest_model_contract(reader: Reader, snap: Snapshot) -> None:
    """Read the few architecture numbers the graph documents cite.

    Parsed as text so the tool needs no YAML dependency; the file is JSON-shaped
    anyway. Only scalars under ``architecture`` are taken.
    """
    text = reader.read_text(MODEL_CONTRACT)
    if text is None:
        return
    src = MODEL_CONTRACT
    for match in re.finditer(r'"(\w+)"\s*:\s*(-?\d+)\s*[,}\n]', text):
        key, value = match.group(1), int(match.group(2))
        snap.add(f"count:model:{key}", "count",
                 f"model contract {key}", value, src)


def _harvest_manifests(reader: Reader, snap: Snapshot) -> None:
    paths = reader.glob(MANIFEST_GLOB)
    if not paths:
        snap.notes.append(f"no manifests matched {MANIFEST_GLOB}")
    for relpath in paths:
        variant = _variant_of(relpath)
        doc = _json(reader, relpath)
        raw = reader.read_bytes(relpath)
        if not isinstance(doc, dict) or variant is None or raw is None:
            continue
        snap.add(f"digest:manifest:{variant}", "digest",
                 f"sha256 of {relpath}", hashlib.sha256(raw).hexdigest(), relpath)
        snap.add(f"count:cache_capacity:{variant}", "count",
                 f"{variant} cache capacity", doc.get("cache_capacity"), relpath)
        snap.add(f"count:context_length:{variant}", "count",
                 f"{variant} context length", doc.get("context_length"), relpath)
        for kind, art in (doc.get("artifacts") or {}).items():
            if not isinstance(art, dict):
                continue
            snap.add(f"digest:graph:{variant}:{kind}", "digest",
                     f"manifest sha256 of {variant}/{kind}.onnx",
                     art.get("sha256"), relpath)
            snap.add(f"bytes:onnx:{variant}:{kind}", "bytes",
                     f"{variant}/{kind}.onnx protobuf size",
                     art.get("size_bytes"), relpath)
            for sidecar in art.get("external_data") or []:
                if not isinstance(sidecar, dict):
                    continue
                snap.add("digest:external_data", "digest",
                         "shared external-data sidecar sha256",
                         sidecar.get("sha256"), relpath)
                snap.add("bytes:external_data", "bytes",
                         "external-data sidecar size",
                         sidecar.get("size_bytes"), relpath)
            for field_name in ("input_tensors", "output_tensors"):
                names = art.get(field_name)
                if isinstance(names, list):
                    label = "inputs" if field_name.startswith("input") else "outputs"
                    snap.add(f"count:{label}:{variant}:{kind}", "count",
                             f"{variant} {kind} boundary {label}",
                             len(names), relpath)
        # Every remaining digest in the manifest becomes a known digest so a
        # citation of it resolves rather than reading as unexplained. Deferred
        # so it never becomes a second reading of a digest that already has a
        # specific identity above.
        for pointer, digest in _walk_digests(doc, ""):
            snap.pending_digests.append(
                (f"digest:manifest-field:{variant}{pointer}",
                 digest, relpath))


def _harvest_inspection(reader: Reader, snap: Snapshot) -> None:
    paths = reader.glob(INSPECTION_GLOB)
    if not paths:
        snap.notes.append(f"no inspection reports matched {INSPECTION_GLOB}")
    op_types_all: Set[str] = set()
    for relpath in paths:
        variant = _variant_of(relpath)
        doc = _json(reader, relpath)
        if not isinstance(doc, dict) or variant is None:
            continue
        gen = doc.get("generated_by") or {}
        snap.add("digest:risk_rules", "digest", "risk catalogue sha256",
                 gen.get("rules_sha256"), relpath)
        for kind, graph in (doc.get("graphs") or {}).items():
            if not isinstance(graph, dict):
                continue
            base = f"{variant}:{kind}"
            snap.add(f"count:nodes:{base}", "count",
                     f"{base} node count", graph.get("node_count"), relpath)
            snap.add(f"count:inputs:{base}", "count",
                     f"{base} graph inputs", graph.get("input_count"), relpath)
            snap.add(f"count:outputs:{base}", "count",
                     f"{base} graph outputs", graph.get("output_count"), relpath)
            snap.add(f"count:initializers:{base}", "count",
                     f"{base} initializers", graph.get("initializer_count"), relpath)
            snap.add(f"count:external_initializers:{base}", "count",
                     f"{base} external initializers",
                     graph.get("external_initializer_count"), relpath)
            snap.add(f"bytes:largest_inline_initializer:{base}", "bytes",
                     f"{base} largest inline initializer",
                     graph.get("largest_inline_initializer_bytes"), relpath)
            snap.add(f"count:ir_version:{base}", "count",
                     f"{base} IR version", graph.get("ir_version"), relpath)
            snap.add(f"count:findings:{base}", "count",
                     f"{base} finding count", graph.get("finding_count"), relpath)
            snap.add(f"digest:graph:{base}", "digest",
                     f"inspected sha256 of {variant}/{kind}.onnx",
                     graph.get("source_sha256"), relpath)
            dims = graph.get("dynamic_dimensions")
            if isinstance(dims, list):
                snap.add(f"count:dynamic_dims:{base}", "count",
                         f"{base} non-static dimensions", len(dims), relpath)
            histogram = graph.get("op_histogram") or {}
            snap.add(f"count:optypes:{base}", "count",
                     f"{base} distinct operator types", len(histogram), relpath)
            op_types_all.update(histogram)
            for op, count in histogram.items():
                snap.add(f"count:op:{base}:{op}", "count",
                         f"{base} `{op}` node count", count, relpath)
            for opset in graph.get("opset_imports") or []:
                if isinstance(opset, list) and len(opset) == 2:
                    snap.add("count:opset", "count", "ONNX opset version",
                             opset[1], relpath)
            for finding in graph.get("findings") or []:
                if not isinstance(finding, dict):
                    continue
                rule = finding.get("rule_id")
                if rule:
                    snap.add(f"count:finding:{base}:{rule}", "count",
                             f"{base} {rule} population",
                             finding.get("count"), relpath)
                    snap.pending_details.append(
                        (base, rule, finding.get("detail") or "", relpath))
            # value_info is reported as absent by the inspection schema; the
            # documents cite a literal 0 for it.
            snap.add(f"count:value_info:{base}", "count",
                     f"{base} value_info entries",
                     len(graph.get("value_info") or []), relpath)
        snap.add(f"count:cache_capacity:{variant}", "count",
                 f"{variant} cache capacity", doc.get("cache_capacity"), relpath)
        snap.add(f"count:context_length:{variant}", "count",
                 f"{variant} context length", doc.get("context_length"), relpath)
    if op_types_all:
        snap.add("count:optypes:all", "count",
                 "distinct operator types across all eight graphs",
                 len(op_types_all), INSPECTION_GLOB)


def _harvest_parity(reader: Reader, snap: Snapshot) -> None:
    for relpath in reader.glob(PARITY_GLOB):
        variant = _variant_of(relpath)
        doc = _json(reader, relpath)
        if not isinstance(doc, dict) or variant is None:
            continue
        for kind, entry in (doc.get("graph_digests") or {}).items():
            if isinstance(entry, dict):
                snap.add(f"digest:graph:{variant}:{kind}", "digest",
                         f"parity-verified sha256 of {variant}/{kind}.onnx",
                         entry.get("sha256"), relpath)
        snap.add(f"count:prompt_length:{variant}", "count",
                 f"{variant} prompt length", doc.get("prompt_length"), relpath)
        snap.add(f"count:steps_requested:{variant}", "count",
                 f"{variant} parity steps requested",
                 doc.get("steps_requested"), relpath)
        tol = doc.get("tolerance") or {}
        for key, value in tol.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                snap.add(f"number:tolerance:{key}", "number",
                         f"ORT CPU tolerance {key}", float(value), relpath)
        runtime = doc.get("runtime") or {}
        snap.add("version:onnxruntime", "version", "onnxruntime version",
                 runtime.get("onnxruntime_version"), relpath)
        snap.add("version:parity_python", "version",
                 "parity host interpreter version",
                 runtime.get("python_version"), relpath)
        for i, step in enumerate(doc.get("steps") or []):
            if not isinstance(step, dict):
                continue
            for key, value in (step.get("metrics") or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    snap.add(f"number:parity:{variant}:step{i}:{key}", "number",
                             f"{variant} step {i} {key}", float(value), relpath)
        for pointer, digest in _walk_digests(doc, ""):
            snap.pending_digests.append(
                (f"digest:parity-field:{variant}{pointer}", digest, relpath))


def _harvest_quantization(reader: Reader, snap: Snapshot) -> None:
    for relpath in reader.glob(QUANTIZATION_GLOB):
        doc = _json(reader, relpath)
        if not isinstance(doc, dict):
            continue
        identity = doc.get("artifact_identity") or {}
        for check in identity.get("checks") or []:
            if not isinstance(check, dict):
                continue
            detail = check.get("detail") or {}
            snap.add(f"count:t40:{check.get('name')}:file_count", "count",
                     f"T40 {check.get('name')} file count",
                     detail.get("file_count"), relpath)
            for entry in detail.get("files") or []:
                if not isinstance(entry, dict):
                    continue
                rel = entry.get("relative_path") or "?"
                variant = rel.split("/")[0]
                leaf = rel.split("/")[-1]
                if leaf in ("prefill.onnx", "decode.onnx"):
                    kind = leaf.split(".")[0]
                    snap.add(f"digest:graph:{variant}:{kind}", "digest",
                             f"T40 recorded sha256 of {rel}",
                             entry.get("recorded_sha256"), relpath)
                    snap.add(f"bytes:onnx:{variant}:{kind}", "bytes",
                             f"T40 recorded size of {rel}",
                             entry.get("recorded_size_bytes"), relpath)
                elif leaf.endswith(".onnx.data"):
                    snap.add("digest:external_data", "digest",
                             "shared external-data sidecar sha256",
                             entry.get("recorded_sha256"), relpath)
                    snap.add("bytes:external_data", "bytes",
                             "external-data sidecar size",
                             entry.get("recorded_size_bytes"), relpath)
        for pointer, digest in _walk_digests(doc, ""):
            snap.pending_digests.append(
                (f"digest:quant-field:{Path(relpath).stem}{pointer}",
                 digest, relpath))


def _harvest_risk_rules(reader: Reader, snap: Snapshot) -> None:
    raw = reader.read_bytes(RISK_RULES)
    if raw is None:
        return
    snap.add("digest:risk_rules", "digest", "risk catalogue sha256",
             hashlib.sha256(raw).hexdigest(), RISK_RULES)
    doc = _json(reader, RISK_RULES)
    if isinstance(doc, dict):
        rules = doc.get("rules")
        if isinstance(rules, list):
            snap.add("count:risk_rules", "count",
                     "rules in the risk catalogue", len(rules), RISK_RULES)
        for rule in rules or []:
            if not isinstance(rule, dict):
                continue
            params = rule.get("params") or rule.get("parameters") or {}
            for key, value in params.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    snap.add(f"count:rule:{rule.get('id')}:{key}", "count",
                             f"{rule.get('id')} {key}", value, RISK_RULES)


def _harvest_revisions(reader: Reader, snap: Snapshot) -> None:
    """Register every 40-hex revision the evidence records.

    Model revisions and exporter commits are cited in the documents exactly the
    way graph digests are, so they have to resolve or every one of them reads
    as an unexplained digest.
    """
    sources = [EXPORT_CONFIG, MODEL_CONTRACT]
    for pattern in EVIDENCE_GLOBS:
        sources.extend(reader.glob(pattern))
    for relpath in sources:
        text = reader.read_text(relpath)
        if text is None:
            continue
        for value in set(REVISION_RE.findall(text)):
            snap.add(f"digest:revision:{value}", "digest",
                     f"revision recorded in {relpath}", value, relpath)


def _harvest_dates(reader: Reader, snap: Snapshot) -> None:
    """Register the dates the evidence stamps on itself.

    ``created_at`` and ``generated_at`` are quoted as plain dates in the
    documents, and a re-measurement moves them.
    """
    sources = [EXPORT_CONFIG]
    for pattern in EVIDENCE_GLOBS:
        sources.extend(reader.glob(pattern))
    for relpath in sources:
        doc = _json(reader, relpath)
        if not isinstance(doc, dict):
            continue
        for key in ("created_at", "generated_at", "measured_at"):
            value = doc.get(key)
            if isinstance(value, str) and DATE_RE.match(value):
                snap.add(f"date:{key}:{Path(relpath).stem}", "date",
                         f"{relpath} {key}", value[:10], relpath)


def _harvest_residual_digests(snap: Snapshot) -> None:
    """Register digests that no specific harvester already claimed.

    Every evidence file is walked for stray SHA-256s so that citing one
    resolves. Registering a digest that already has a specific identity would
    give it a second meaning and push every graph-digest citation into
    AMBIGUOUS, so those are dropped here rather than at match time.
    """
    claimed = set(snap.digests())
    for identity, digest, source in snap.pending_digests:
        if digest in claimed:
            continue
        snap.add(identity, "digest", source, digest, source)


_INT_IN_TEXT = re.compile(r"(?<![0-9A-Za-z_.])\d(?:[\d,]*\d)?(?![0-9A-Za-z_.])")


def _harvest_finding_details(snap: Snapshot) -> None:
    """Register populations that only appear inside a finding's prose.

    ``R-DATA-DEPENDENT-SHAPE-INPUT`` reports "804 of 1257 shape-defining
    operator inputs"; the denominator is quoted verbatim in the documents but
    is not a field anywhere. A figure that some other fact already supplies is
    skipped, so this does not manufacture a second reading of a number that
    already has one.
    """
    known = set(snap.int_index())
    for base, rule, detail, relpath in snap.pending_details:
        for i, raw in enumerate(_INT_IN_TEXT.findall(detail)):
            value = int(raw.replace(",", ""))
            if value in known:
                continue
            snap.add(f"count:finding-detail:{base}:{rule}:{i}", "count",
                     f"{base} {rule} population from its detail text",
                     value, relpath)


def _derive(snap: Snapshot) -> None:
    """Derived families.

    Deliberately narrow. A derived family wide enough to explain any number
    would make ``UNCLASSIFIED`` meaningless, so only two are computed, and each
    classification names the derivation it used.
    """
    sizes: Dict[Tuple[str, str], int] = {}
    for identity, fact in list(snap.facts.items()):
        if identity.startswith("bytes:onnx:"):
            _, _, variant, kind = identity.split(":")
            value = fact.value
            if isinstance(value, int):
                sizes[(variant, kind)] = value
    for (v1, k1), s1 in sizes.items():
        for (v2, k2), s2 in sizes.items():
            if k1 != k2 or v1 == v2 or s1 == 0:
                continue
            snap.add(f"ratio:onnx_bytes:{k1}:{v1}->{v2}", "ratio",
                     f"{k1} protobuf size ratio {v1} -> {v2}",
                     s2 / s1, "derived")

    nodes: Dict[Tuple[str, str], int] = {}
    for identity, fact in list(snap.facts.items()):
        if identity.startswith("count:nodes:"):
            _, _, variant, kind = identity.split(":")
            value = fact.value
            if isinstance(value, int):
                nodes[(variant, kind)] = value
    for identity, fact in list(snap.facts.items()):
        if not identity.startswith("count:op:"):
            continue
        _, _, variant, kind, op = identity.split(":", 4)
        total = nodes.get((variant, kind))
        value = fact.value
        if not total or not isinstance(value, int):
            continue
        snap.add(f"percent:op_share:{variant}:{kind}:{op}", "percent",
                 f"`{op}` share of the {variant} {kind} node list",
                 100.0 * value / total, "derived")


def _harvest_artifacts(snap: Snapshot, artifact_root: Path,
                       rehash_external: bool) -> None:
    if not artifact_root.is_dir():
        snap.notes.append(
            f"artifact root not present, skipped re-hash: {artifact_root}")
        return
    hashed = 0
    for variant_dir in sorted(artifact_root.iterdir()):
        if not variant_dir.is_dir() or not re.fullmatch(r"S\d+", variant_dir.name):
            continue
        for kind in GRAPH_KINDS:
            path = variant_dir / f"{kind}.onnx"
            if not path.is_file():
                continue
            snap.add(f"digest:graph:{variant_dir.name}:{kind}", "digest",
                     f"re-hashed {path.name}", _sha256_file(path), "artifact-rehash")
            snap.add(f"bytes:onnx:{variant_dir.name}:{kind}", "bytes",
                     f"measured size of {path.name}", path.stat().st_size,
                     "artifact-rehash")
            hashed += 1
            sidecar = variant_dir / f"{kind}.onnx.data"
            if sidecar.is_file():
                snap.add("bytes:external_data", "bytes",
                         "external-data sidecar size", sidecar.stat().st_size,
                         "artifact-rehash")
                if rehash_external:
                    snap.add("digest:external_data", "digest",
                             "shared external-data sidecar sha256",
                             _sha256_file(sidecar), "artifact-rehash")
    snap.notes.append(
        f"re-hashed {hashed} .onnx protobuf(s) under {artifact_root}"
        + ("" if rehash_external else "; sidecars sized but not hashed"
           " (use --rehash-external-data)")
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Tokenizing
# --------------------------------------------------------------------------


@dataclass
class Token:
    path: str
    role: str
    line: int
    column: int
    raw: str
    form: str  # hex | date | dotted | percent | multiplier | decimal | integer
    numeric: Optional[float]
    integral: Optional[int]
    decimals: int
    line_text: str

    @property
    def where(self) -> str:
        return f"{self.path}:{self.line}"


_HEX_RUN = re.compile(r"(?<![0-9A-Za-z_])[0-9a-f]{7,}(?![0-9A-Za-z_])")
_DATE = re.compile(r"(?<![0-9A-Za-z_])\d{4}-\d{2}-\d{2}(?![0-9])")
_NUMBER = re.compile(
    r"(?<![0-9A-Za-z_.])(\d(?:[\d,]*\d)?(?:\.\d+)*)([%x×])?(?![0-9A-Za-z_])"
)
_SECTION_REF = re.compile(r"(?:section|§|sections)\s*$", re.IGNORECASE)
_LINE_REF = re.compile(r"[A-Za-z0-9_-]+\.(?:md|py|json|yaml|yml|txt):$")


def tokenize(path: str, role: str, text: str,
             known_digests: Iterable[str]) -> List[Token]:
    """Every numeric token in a document, each exactly once.

    One lexical exclusion, applied uniformly: a digit run glued to a preceding
    letter, digit or underscore is part of an identifier (``S128``, ``T20``,
    ``float16``, ``sha256``), not a numeric claim, and is not tokenized. Every
    other digit run in the file is emitted.
    """
    digests = tuple(known_digests)
    tokens: List[Token] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        claimed: List[Tuple[int, int]] = []

        def overlaps(start: int, end: int) -> bool:
            return any(not (end <= s or start >= e) for s, e in claimed)

        for match in _HEX_RUN.finditer(line):
            run = match.group(0)
            has_digit = any(c.isdigit() for c in run)
            has_alpha = any(c.isalpha() for c in run)
            is_prefix = any(d.startswith(run) for d in digests)
            if not ((has_digit and has_alpha) or is_prefix):
                continue
            claimed.append(match.span())
            tokens.append(Token(
                path=path, role=role, line=lineno, column=match.start() + 1,
                raw=run, form="hex", numeric=None, integral=None, decimals=0,
                line_text=line.rstrip(),
            ))
        for match in _DATE.finditer(line):
            if overlaps(*match.span()):
                continue
            claimed.append(match.span())
            tokens.append(Token(
                path=path, role=role, line=lineno, column=match.start() + 1,
                raw=match.group(0), form="date", numeric=None, integral=None,
                decimals=0, line_text=line.rstrip(),
            ))
        for match in _NUMBER.finditer(line):
            if overlaps(*match.span()):
                continue
            claimed.append(match.span())
            raw = match.group(1)
            plain = raw.replace(",", "")
            suffix = match.group(2) or ""
            if DOTTED_RE.match(plain):
                form = "dotted"
                numeric: Optional[float] = None
                integral: Optional[int] = None
                decimals = 0
            else:
                numeric = float(plain)
                integral = int(plain) if "." not in plain else None
                decimals = len(plain.split(".")[1]) if "." in plain else 0
                before = line[max(0, match.start() - 16):match.start()]
                if suffix == "%":
                    form = "percent"
                elif suffix in ("x", "×"):
                    form = "multiplier"
                elif _LINE_REF.search(before):
                    form = "line-ref"
                elif _SECTION_REF.search(before):
                    form = "section-ref"
                elif "." in plain:
                    form = "decimal"
                else:
                    form = "integer"
            tokens.append(Token(
                path=path, role=role, line=lineno, column=match.start() + 1,
                raw=raw + (suffix if suffix in ("%", "x", "×") else ""),
                form=form, numeric=numeric, integral=integral,
                decimals=decimals, line_text=line.rstrip(),
            ))
    tokens.sort(key=lambda t: (t.line, t.column))
    return tokens


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

MATCH_KINDS = {
    "integer": ("count", "bytes", "number"),
    "decimal": ("number", "percent", "ratio"),
    "percent": ("percent",),
    "multiplier": ("ratio",),
}


_VARIANT_SLOT = re.compile(r"(?<=:)S\d+(?=:|$)")
_KIND_SLOT = re.compile(r"(?<=:)(?:prefill|decode)(?=:|$)")


def meaning_key(identity: str) -> str:
    """Collapse the same claim measured at several contexts into one meaning.

    ``count:op:S128:prefill:Cast`` and ``count:op:S512:prefill:Cast`` are one
    claim about prefill, not two, whenever they carry the same value. Without
    this the AMBIGUOUS bucket reports eight readings of every literal and stops
    being a review queue.
    """
    return _KIND_SLOT.sub("*", _VARIANT_SLOT.sub("*", identity))


@dataclass
class Candidate:
    identity: str
    label: str
    role: str  # stale | current | both
    baseline: object
    current: object
    derivation: str

    @property
    def needs_edit(self) -> bool:
        return self.role == "stale"

    @property
    def meaning(self) -> Tuple[str, str, str, str]:
        return (meaning_key(self.identity), repr(self.baseline),
                repr(self.current), self.role)


@dataclass
class Meaning:
    key: str
    role: str
    baseline: object
    current: object
    derivation: str
    identities: List[str]

    @property
    def needs_edit(self) -> bool:
        return self.role == "stale"

    @property
    def sample(self) -> str:
        return self.identities[0]


def to_meanings(candidates: Sequence[Candidate]) -> List[Meaning]:
    grouped: Dict[Tuple[str, str, str, str], Meaning] = {}
    for candidate in candidates:
        key = candidate.meaning
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = Meaning(
                key=key[0], role=candidate.role, baseline=candidate.baseline,
                current=candidate.current, derivation=candidate.derivation,
                identities=[candidate.identity],
            )
        else:
            existing.identities.append(candidate.identity)
    return [grouped[k] for k in sorted(grouped)]


class Classifier:
    def __init__(self, current: Snapshot, baseline: Snapshot,
                 git: Optional["GitObjectResolver"] = None) -> None:
        self.current = current
        self.baseline = baseline
        self.git = git
        self._cur_int = current.int_index()
        self._base_int = baseline.int_index()
        self._cur_digest = current.digests()
        self._base_digest = baseline.digests()
        self._cur_version = current.string_index(("version",))
        self._base_version = baseline.string_index(("version",))
        self._cur_date = current.string_index(("date",))
        self._base_date = baseline.string_index(("date",))
        self._cur_float = {
            kind: current.float_facts((kind,))
            for kind in ("percent", "ratio", "number")
        }
        self._base_float = {
            kind: baseline.float_facts((kind,))
            for kind in ("percent", "ratio", "number")
        }

    def known_digests(self) -> Set[str]:
        return set(self._cur_digest) | set(self._base_digest)

    def _fact_label(self, identity: str) -> str:
        fact = self.current.facts.get(identity) or self.baseline.facts.get(identity)
        return fact.label if fact else identity

    def _candidates(self, cur_hits: Set[str], base_hits: Set[str],
                    derivation: str) -> List[Candidate]:
        out: List[Candidate] = []
        for identity in sorted(cur_hits | base_hits):
            in_cur = identity in cur_hits
            in_base = identity in base_hits
            role = "both" if (in_cur and in_base) else (
                "current" if in_cur else "stale")
            cur_fact = self.current.facts.get(identity)
            base_fact = self.baseline.facts.get(identity)
            out.append(Candidate(
                identity=identity,
                label=self._fact_label(identity),
                role=role,
                baseline=base_fact.value if base_fact else None,
                current=cur_fact.value if cur_fact else None,
                derivation=derivation,
            ))
        return out

    def classify(self, token: Token) -> Tuple[str, List[Meaning]]:
        """Put one occurrence in exactly one bucket.

        AMBIGUOUS is reserved for a literal whose readings imply *different
        actions*. Several fact identities carrying the same number and the same
        fate -- ``count:nodes`` and the node-count rule's own population, say --
        are one decision for a reviewer, not several, and every reading is still
        printed alongside the verdict.
        """
        meanings = to_meanings(self._match(token))
        if not meanings:
            return "UNCLASSIFIED", []
        fates = {(m.needs_edit, repr(m.current)) for m in meanings}
        if len(fates) > 1:
            return "AMBIGUOUS", meanings
        return ("MOVES" if meanings[0].needs_edit else "STATIC"), meanings

    def _match(self, token: Token) -> List[Candidate]:
        if token.form == "hex":
            cur = {
                identity
                for digest, ids in self._cur_digest.items()
                if digest.startswith(token.raw)
                for identity in ids
            }
            base = {
                identity
                for digest, ids in self._base_digest.items()
                if digest.startswith(token.raw)
                for identity in ids
            }
            hits = self._collapse(self._candidates(cur, base, "digest prefix"))
            if not hits and self.git is not None and self.git.resolves(token.raw):
                hits = [Candidate(
                    identity=f"git:object:{token.raw}",
                    label="an object in this repository",
                    role="both", baseline=token.raw, current=token.raw,
                    derivation="git object",
                )]
            return hits
        if token.form == "dotted":
            cur = set(self._cur_version.get(token.raw, []))
            base = set(self._base_version.get(token.raw, []))
            return self._collapse(self._candidates(cur, base, "version string"))
        if token.form == "date":
            cur = set(self._cur_date.get(token.raw, []))
            base = set(self._base_date.get(token.raw, []))
            return self._collapse(self._candidates(cur, base, "recorded date"))
        if token.form in ("line-ref", "section-ref"):
            # Lexically a cross-reference, not a claim. It is still counted and
            # still lands in a bucket; it is just never explained by evidence.
            return []
        out: List[Candidate] = []
        if token.integral is not None:
            cur = set(self._cur_int.get(token.integral, []))
            base = set(self._base_int.get(token.integral, []))
            cur = {i for i in cur if self._kind_of(i) in MATCH_KINDS["integer"]}
            base = {i for i in base if self._kind_of(i) in MATCH_KINDS["integer"]}
            out.extend(self._candidates(cur, base, "exact integer"))
        for kind in MATCH_KINDS.get(token.form, ()):
            if kind not in self._cur_float:
                continue
            cur = self._round_hits(self._cur_float[kind], token)
            base = self._round_hits(self._base_float[kind], token)
            if cur or base:
                out.extend(self._candidates(
                    cur, base, f"{kind} rounded to {token.decimals}dp"))
        return self._collapse(out)

    @staticmethod
    def _round_hits(facts: Sequence[Tuple[str, float]],
                    token: Token) -> Set[str]:
        if token.numeric is None:
            return set()
        hits: Set[str] = set()
        for identity, value in facts:
            if abs(round(value, token.decimals) - token.numeric) < 1e-9:
                hits.add(identity)
        return hits

    def _kind_of(self, identity: str) -> str:
        fact = self.current.facts.get(identity) or self.baseline.facts.get(identity)
        return fact.kind if fact else "?"

    @staticmethod
    def _collapse(candidates: List[Candidate]) -> List[Candidate]:
        seen: Dict[str, Candidate] = {}
        for candidate in candidates:
            existing = seen.get(candidate.identity)
            if existing is None or (candidate.needs_edit and not existing.needs_edit):
                seen[candidate.identity] = candidate
        return [seen[k] for k in sorted(seen)]


# --------------------------------------------------------------------------
# Citations: structural, bound claims
# --------------------------------------------------------------------------


@dataclass
class Finding:
    kind: str
    severity: str  # mismatch | unresolved | stale | conflict
    where: str
    detail: str


def check_evidence_conflicts(snap: Snapshot) -> List[Finding]:
    findings: List[Finding] = []
    for fact in snap.conflicts():
        by_value: Dict[object, List[str]] = {}
        for value, source in fact.observations:
            by_value.setdefault(value, []).append(source)
        parts = "; ".join(
            f"{value!r} from {', '.join(sorted(set(sources)))}"
            for value, sources in by_value.items()
        )
        findings.append(Finding(
            kind="evidence-crosscheck",
            severity="conflict",
            where=fact.identity,
            detail=f"{fact.label}: sources disagree -- {parts}",
        ))
    return findings


def _normalize_header(cell: str) -> str:
    cell = cell.replace("`", " ").replace("*", " ")
    cell = re.sub(r"[^a-z0-9]+", " ", cell.lower()).strip()
    return cell


def _clean_cell(cell: str) -> str:
    return cell.replace("`", "").replace("*", "").strip()


def _iter_tables(lines: Sequence[str]) -> Iterator[Tuple[int, List[str], List[Tuple[int, List[str]]]]]:
    i = 0
    while i < len(lines) - 1:
        if lines[i].lstrip().startswith("|") and re.match(
            r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]
        ):
            header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            rows: List[Tuple[int, List[str]]] = []
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(
                    (j + 1, [c.strip() for c in lines[j].strip().strip("|").split("|")])
                )
                j += 1
            yield i + 1, header, rows
            i = j
        else:
            i += 1


def _row_keys(cells: Sequence[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    variant = kind = rule = None
    for cell in cells:
        text = _clean_cell(cell)
        if re.fullmatch(r"S\d+", text):
            variant = variant or text
        path = re.search(r"\b(S\d+)/(prefill|decode)\.onnx\b", text)
        if path:
            variant = variant or path.group(1)
            kind = kind or path.group(2)
        manifest = re.search(r"results/manifests/onnx/(S\d+)\.json", text)
        if manifest:
            variant = variant or manifest.group(1)
        if text in GRAPH_KINDS:
            kind = kind or text
        if re.fullmatch(r"R-[A-Z0-9-]+", text):
            rule = rule or text
    return variant, kind, rule


def _cell_numbers(cell: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for match in _NUMBER.finditer(cell):
        raw = match.group(1)
        if "." in raw:
            continue
        out.append((raw, int(raw.replace(",", ""))))
    return out


def check_table_citations(path: str, text: str, snap: Snapshot) -> List[Finding]:
    findings: List[Finding] = []
    lines = text.splitlines()
    int_of = {
        identity: fact.value
        for identity, fact in snap.facts.items()
        if isinstance(fact.value, int) and not isinstance(fact.value, bool)
    }
    digests = snap.digests()
    for _, header, rows in _iter_tables(lines):
        columns = [_normalize_header(h) for h in header]
        if not any(c in COLUMN_MAP for c in columns):
            continue
        for lineno, cells in rows:
            variant, kind, rule = _row_keys(cells)
            for index, cell in enumerate(cells):
                if index >= len(columns):
                    break
                mapping = COLUMN_MAP.get(columns[index])
                if mapping is None:
                    continue
                family, column_kind = mapping
                row_kind = column_kind or kind
                findings.extend(_check_cell(
                    path, lineno, cell, family, variant, row_kind, rule,
                    int_of, digests, snap,
                ))
    return findings


def _check_cell(path: str, lineno: int, cell: str, family: str,
                variant: Optional[str], kind: Optional[str],
                rule: Optional[str], int_of: Dict[str, int],
                digests: Dict[str, List[str]], snap: Snapshot) -> List[Finding]:
    where = f"{path}:{lineno}"
    text = _clean_cell(cell)
    if not text:
        return []

    if family.startswith("digest:"):
        token = re.search(r"\b[0-9a-f]{7,64}\b", text)
        if token is None:
            return []
        cited = token.group(0)
        if family == "digest:graph":
            if not variant or not kind:
                return []
            identity = f"digest:graph:{variant}:{kind}"
        else:
            if not variant:
                return []
            identity = f"digest:manifest:{variant}"
        fact = snap.facts.get(identity)
        if fact is None:
            return [Finding("bound-citation", "unresolved", where,
                            f"cites {cited} for {identity}, which no evidence "
                            "file currently records")]
        if not str(fact.value).startswith(cited):
            return [Finding("bound-citation", "mismatch", where,
                            f"{fact.label}: document says {cited}, "
                            f"measured {fact.value}")]
        return []

    numbers = _cell_numbers(text)
    if not numbers:
        return []

    identities: List[str] = []
    if family == "finding":
        if not rule or not kind:
            return []
        identities = [
            i for i in snap.facts
            if i.startswith("count:finding:") and i.endswith(f":{kind}:{rule}")
        ]
    elif family == "cache_capacity":
        if not variant:
            return []
        identities = [f"count:cache_capacity:{variant}"]
    else:
        if not variant or not kind:
            return []
        prefix = "bytes" if family in ("onnx_bytes",) else "count"
        if family == "onnx_bytes":
            identities = [f"bytes:onnx:{variant}:{kind}"]
        else:
            identities = [f"{prefix}:{family}:{variant}:{kind}"]

    expected = [int_of[i] for i in identities if i in int_of]
    if not expected:
        return [Finding("bound-citation", "unresolved", where,
                        f"no measured value for {', '.join(identities)} "
                        f"(cell {text!r})")]

    if family == "initializers":
        external = [
            int_of[f"count:external_initializers:{variant}:{kind}"]
        ] if f"count:external_initializers:{variant}:{kind}" in int_of else []
        expected = expected + external

    if len(numbers) == 1:
        raw, value = numbers[0]
        if value not in expected:
            return [Finding("bound-citation", "mismatch", where,
                            f"{family} for {variant}/{kind or rule}: document "
                            f"says {raw}, measured "
                            f"{', '.join(str(e) for e in sorted(set(expected)))}")]
        return []

    # Annotated cell (a range, or a value plus a qualifier). Require that at
    # least one number in it is a measured value; a wholly stale cell fails.
    if not any(value in expected for _, value in numbers):
        return [Finding("bound-citation", "mismatch", where,
                        f"{family} for {variant}/{kind or rule}: no number in "
                        f"{text!r} matches measured "
                        f"{', '.join(str(e) for e in sorted(set(expected)))}")]
    return []


_OP_EQUALS = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\s*=\s*(\d[\d,]*)\b")
_COUNT_OP = re.compile(r"\b(\d[\d,]*)\s+`([A-Z][A-Za-z0-9_]*)`")
_KIND_WINDOW = 24


def check_operator_citations(path: str, text: str,
                             snap: Snapshot) -> List[Finding]:
    """Check ``Op=N`` and ``N `Op``` spans against the operator histograms."""
    counts: Dict[str, Dict[str, Set[int]]] = {}
    for identity, fact in snap.facts.items():
        if not identity.startswith("count:op:"):
            continue
        _, _, variant, kind, op = identity.split(":", 4)
        if isinstance(fact.value, int):
            counts.setdefault(op, {}).setdefault(kind, set()).add(fact.value)
    if not counts:
        return []

    findings: List[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for regex, op_group, num_group in (
            (_OP_EQUALS, 1, 2), (_COUNT_OP, 2, 1)
        ):
            for match in regex.finditer(line):
                op = match.group(op_group)
                if op not in counts:
                    continue
                value = int(match.group(num_group).replace(",", ""))
                # Bind a graph kind only when the span is immediately
                # qualified. A `prefill` anywhere on the line is not a
                # qualifier: line 403 of graph-inspection.md reads "it carries
                # 459 `Shape` nodes against prefill's 121", where the 459 is
                # decode's.
                prefix = line[max(0, match.start() - _KIND_WINDOW):match.start()]
                near = [k for k in GRAPH_KINDS if k in prefix.lower()]
                kind = near[0] if len(near) == 1 else None
                if kind is not None and kind in counts[op]:
                    allowed = counts[op][kind]
                    scope = f"{kind} `{op}`"
                else:
                    allowed = set().union(*counts[op].values())
                    scope = f"any graph's `{op}`"
                if value not in allowed:
                    findings.append(Finding(
                        "operator-citation", "mismatch", f"{path}:{lineno}",
                        f"{scope}: document says {match.group(num_group)}, "
                        f"measured {', '.join(str(v) for v in sorted(allowed))}"
                    ))
    return findings


_BOUND_DIGEST_CONTEXT = re.compile(
    r"\b(S\d+)/(prefill|decode)\.onnx\b|results/manifests/onnx/(S\d+)\.json"
)


def check_inline_digest_citations(path: str, text: str,
                                  snap: Snapshot) -> List[Finding]:
    """Check any digest quoted on a line that names the graph it belongs to."""
    findings: List[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        contexts = list(_BOUND_DIGEST_CONTEXT.finditer(line))
        if len(contexts) != 1:
            continue
        match = contexts[0]
        if match.group(1):
            identity = f"digest:graph:{match.group(1)}:{match.group(2)}"
        else:
            identity = f"digest:manifest:{match.group(3)}"
        fact = snap.facts.get(identity)
        tokens = [
            m.group(0) for m in re.finditer(r"\b[0-9a-f]{7,64}\b", line)
            if any(c.isdigit() for c in m.group(0))
            and any(c.isalpha() for c in m.group(0))
        ]
        if not tokens or fact is None:
            continue
        if not any(str(fact.value).startswith(t) for t in tokens):
            findings.append(Finding(
                "inline-digest", "mismatch", f"{path}:{lineno}",
                f"{fact.label}: line quotes {', '.join(tokens)}, "
                f"measured {fact.value}",
            ))
    return findings


class GitObjectResolver:
    """Answer 'is this hex string an object in this repository?'.

    A worklog citing a commit is citing a digest, and it has to resolve or it
    reads as an unexplained one. Git is the evidence for that class.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: Dict[str, bool] = {}

    def resolves(self, token: str) -> bool:
        if token in self._cache:
            return self._cache[token]
        proc = subprocess.run(
            ["git", "-C", str(self.root), "rev-parse", "--verify", "--quiet",
             f"{token}^{{object}}"],
            capture_output=True,
        )
        self._cache[token] = proc.returncode == 0
        return self._cache[token]


def check_unresolved_digests(path: str, tokens: Sequence[Token],
                             known: Set[str],
                             git: Optional["GitObjectResolver"] = None
                             ) -> List[Finding]:
    findings: List[Finding] = []
    for token in tokens:
        if token.form != "hex":
            continue
        if any(digest.startswith(token.raw) for digest in known):
            continue
        if git is not None and git.resolves(token.raw):
            continue
        findings.append(Finding(
            "unresolved-digest", "unresolved", token.where,
            f"{token.raw} matches no digest recorded in any evidence file "
            "and is not an object in this repository",
        ))
    return findings


REFERENCE_DIGEST_PREFIXES = ("digest:graph:", "digest:external_data",
                             "digest:manifest:")


def _reference_digests(snap: Snapshot) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for identity, fact in snap.facts.items():
        if identity.startswith(REFERENCE_DIGEST_PREFIXES):
            for value in fact.values:
                out[str(value)] = identity
    return out


def sweep_stale_digests(reader: Reader, current: Snapshot,
                        baseline: Snapshot) -> List[Finding]:
    """Any tracked file still naming a reference digest that no longer exists."""
    now = _reference_digests(current)
    before = _reference_digests(baseline)
    stale = {d: i for d, i in before.items() if d not in now}
    if not stale:
        return []
    findings: List[Finding] = []
    for relpath in reader.paths():
        if relpath.startswith("scripts/audit/") or relpath.startswith("tests/scripts/"):
            continue
        text = reader.read_text(relpath)
        if text is None:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in re.finditer(r"\b[0-9a-f]{7,64}\b", line):
                token = match.group(0)
                if not (any(c.isdigit() for c in token)
                        and any(c.isalpha() for c in token)):
                    continue
                hits = [i for d, i in stale.items() if d.startswith(token)]
                if not hits:
                    continue
                if any(d.startswith(token) for d in now):
                    continue
                findings.append(Finding(
                    "stale-digest", "stale", f"{relpath}:{lineno}",
                    f"{token} is the superseded {hits[0]}",
                ))
    return findings


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _documents(reader: Reader, include_historical: bool,
               extra: Sequence[str]) -> List[Tuple[str, str]]:
    docs = [
        (path, role) for path, role in CLAIM_DOCUMENTS
        if include_historical or role == "reconcile"
    ]
    docs.extend((path, "reconcile") for path in extra)
    present = []
    for path, role in docs:
        if reader.read_text(path) is None:
            print(f"warning: in-scope document missing: {path}", file=sys.stderr)
            continue
        present.append((path, role))
    return present


def _resolve_artifact_root(
    repo_root: Path, override: Optional[str]
) -> Tuple[Optional[Path], Optional[str]]:
    """Decide which artifact tree, if any, to re-hash.

    Returns ``(root, declined_reason)``.

    An explicit ``--artifact-root`` always wins: naming a path is a statement
    that those bytes belong to the records being audited.

    ``SLM_LAB_ARTIFACT_ROOT`` is ambient machine configuration. It names the
    large-artifact store paired with *this* checkout, so it is honoured only
    when the audited root is this checkout. Pointed at any other tree -- a test
    fixture, a second worktree, a bare clone -- the ambient value would hash one
    repository's graphs against a different repository's records and report
    every digest and size as a conflict. That is not a weaker audit, it is a
    wrong one, so the variable is declined and the decision is printed rather
    than taken silently.
    """
    if override:
        return Path(override).expanduser(), None
    template = os.environ.get("SLM_LAB_ARTIFACT_ROOT")
    if not template:
        return None, None
    if repo_root != REPO_ROOT:
        return None, (
            "declined the ambient SLM_LAB_ARTIFACT_ROOT: it describes "
            f"{REPO_ROOT}, not the audited root {repo_root}. Pass "
            "--artifact-root explicitly to re-hash artifacts against this tree."
        )
    return Path(template).expanduser() / "onnx" / "reference" / "T20", None


def run_claims(args: argparse.Namespace) -> int:
    root = Path(args.repo_root).resolve()
    current_reader = WorktreeReader(root)
    baseline_reader = GitReader(root, args.baseline_ref)
    current = harvest(current_reader)
    baseline = harvest(baseline_reader)
    classifier = Classifier(current, baseline, GitObjectResolver(root))
    known = classifier.known_digests()

    docs = _documents(current_reader, args.include_historical, args.document)
    buckets: Dict[str, List[Tuple[Token, List[Meaning]]]] = {
        "MOVES": [], "STATIC": [], "AMBIGUOUS": [], "UNCLASSIFIED": [],
    }
    total = 0
    per_document: Dict[str, Dict[str, int]] = {}
    for path, role in docs:
        text = current_reader.read_text(path) or ""
        tokens = tokenize(path, role, text, known)
        total += len(tokens)
        counts = {k: 0 for k in buckets}
        for token in tokens:
            bucket, candidates = classifier.classify(token)
            buckets[bucket].append((token, candidates))
            counts[bucket] += 1
        per_document[path] = counts

    accounted = sum(len(v) for v in buckets.values())
    if accounted != total:
        raise AssertionError(
            f"bucket invariant violated: {total} tokens, {accounted} classified"
        )

    if args.json:
        print(json.dumps(_claims_json(
            current, baseline, buckets, per_document, total), indent=2))
        return 0

    identical = _snapshots_identical(current, baseline)
    print("=" * 78)
    print("claims -- exhaustive numeric review queue")
    print("=" * 78)
    print(f"repository      {root}")
    print("current state   worktree")
    print(f"baseline state  {baseline_reader.name}")
    if identical:
        print("  NOTE: baseline evidence is identical to the worktree, so MOVES is")
        print("        empty by construction. Pass --baseline-ref <pre-promotion")
        print("        commit> after the re-export lands to populate it.")
    conflicts = current.conflicts()
    if conflicts:
        print(f"  WARNING: {len(conflicts)} measured value(s) disagree between "
              "generated")
        print("        evidence files, so the current snapshot holds both the old")
        print("        and the new reading and MOVES will under-report until the")
        print("        regeneration finishes. Run `citations` for the list.")
    for note in current.notes:
        print(f"  note: {note}")
    print(f"measured facts  {len(current.facts)} current / "
          f"{len(baseline.facts)} baseline")
    print(f"documents       {len(docs)}")
    for path, role in docs:
        counts = per_document[path]
        print(f"  {path} [{role}]  "
              + "  ".join(f"{k}={counts[k]}" for k in
                          ("MOVES", "AMBIGUOUS", "STATIC", "UNCLASSIFIED")))
    print()
    print(f"total numeric tokens {total}")
    for name in ("MOVES", "AMBIGUOUS", "STATIC", "UNCLASSIFIED"):
        print(f"  {name:<14} {len(buckets[name])}")
    print()

    _print_moves(buckets["MOVES"])
    _print_ambiguous(buckets["AMBIGUOUS"], args.limit)
    if args.show_static:
        _print_static(buckets["STATIC"])
    _print_unclassified(buckets["UNCLASSIFIED"], args.limit)

    print()
    print("-" * 78)
    print("What this does not prove")
    print("-" * 78)
    print("  MOVES and AMBIGUOUS are candidate lists produced by numeric")
    print("  coincidence plus context, not by reading the sentence. Confirm each")
    print("  before editing. UNCLASSIFIED is a review queue, not a clean bill of")
    print("  health: it means no measured value explains the literal. STATIC")
    print("  means a measured value with that literal exists and did not move --")
    print("  not that the claim around it is true.")
    return 0


def _snapshots_identical(a: Snapshot, b: Snapshot) -> bool:
    keys = set(a.facts) | set(b.facts)
    for key in keys:
        fa = a.facts.get(key)
        fb = b.facts.get(key)
        if (fa is None) != (fb is None):
            return False
        if fa is not None and fb is not None and fa.values != fb.values:
            return False
    return True


def _print_moves(entries: Sequence[Tuple[Token, List[Meaning]]]) -> None:
    print("-" * 78)
    print(f"MOVES -- literal matches only the pre-promotion value ({len(entries)})")
    print("-" * 78)
    if not entries:
        print("  (none)")
        print()
        return
    for token, meanings in entries:
        meaning = meanings[0]
        tag = "" if token.role == "reconcile" else "  [HISTORICAL: do not edit]"
        print(f"  {token.where:<62} {token.raw}{tag}")
        print(f"      {meaning.key}"
              + ("" if len(meanings) == 1
                 else f"  (+{len(meanings) - 1} reading(s) with the same fate: "
                      + ", ".join(m.key for m in meanings[1:4]) + ")"))
        print(f"      {meaning.baseline!r} -> {meaning.current!r}"
              f"   [{meaning.derivation}]")
        print(f"      | {token.line_text.strip()[:110]}")
    print()


def _print_ambiguous(entries: Sequence[Tuple[Token, List[Meaning]]],
                     limit: int) -> None:
    hot = [e for e in entries if any(m.needs_edit for m in e[1])]
    cool = [e for e in entries if not any(m.needs_edit for m in e[1])]
    print("-" * 78)
    print(f"AMBIGUOUS -- readings imply different actions ({len(entries)}; "
          f"{len(hot)} with at least one reading that moved)")
    print("-" * 78)
    for label, group in (("moves under some reading", hot),
                         ("static under every reading", cool)):
        print(f"  [{label}] {len(group)} occurrence(s)")
        shown = group if limit <= 0 else group[:limit]
        for token, meanings in shown:
            print(f"    {token.where:<60} {token.raw}")
            for meaning in meanings[:6]:
                marker = "*" if meaning.needs_edit else " "
                print(f"      {marker} {meaning.key}  = {meaning.current!r}"
                      f"  ({meaning.derivation})")
            if len(meanings) > 6:
                print(f"        ... {len(meanings) - 6} more reading(s)")
        if limit > 0 and len(group) > limit:
            print(f"    ... {len(group) - limit} more (use --limit 0)")
    print()


def _print_static(entries: Sequence[Tuple[Token, List[Meaning]]]) -> None:
    print("-" * 78)
    print(f"STATIC -- one measured meaning, unchanged ({len(entries)})")
    print("-" * 78)
    for token, meanings in entries:
        print(f"  {token.where:<60} {token.raw:<20} {meanings[0].key}")
    print()


def _print_unclassified(entries: Sequence[Tuple[Token, List[Meaning]]],
                        limit: int) -> None:
    by_value: Dict[Tuple[str, str], List[Token]] = {}
    for token, _ in entries:
        by_value.setdefault((token.form, token.raw), []).append(token)
    forms: Dict[str, int] = {}
    for token, _ in entries:
        forms[token.form] = forms.get(token.form, 0) + 1
    print("-" * 78)
    print(f"UNCLASSIFIED -- no measured value explains it "
          f"({len(entries)} occurrences, {len(by_value)} distinct)")
    print("-" * 78)
    print("  by lexical form: "
          + ", ".join(f"{k}={v}" for k, v in sorted(forms.items())))
    print()
    for (form, raw), tokens in sorted(
        by_value.items(), key=lambda kv: (-len(kv[1]), kv[0])
    ):
        places = ", ".join(t.where for t in tokens[:12])
        more = "" if len(tokens) <= 12 else f", ... (+{len(tokens) - 12})"
        print(f"  {raw:<24} [{form}] x{len(tokens):<4} {places}{more}")
    print()


def _claims_json(current: Snapshot, baseline: Snapshot,
                 buckets: Dict[str, List[Tuple[Token, List[Meaning]]]],
                 per_document: Dict[str, Dict[str, int]],
                 total: int) -> dict:
    def encode(token: Token, meanings: List[Meaning]) -> dict:
        return {
            "path": token.path,
            "line": token.line,
            "column": token.column,
            "literal": token.raw,
            "form": token.form,
            "role": token.role,
            "line_text": token.line_text.strip(),
            "meanings": [
                {
                    "key": m.key,
                    "identities": m.identities,
                    "baseline": m.baseline,
                    "current": m.current,
                    "needs_edit": m.needs_edit,
                    "derivation": m.derivation,
                }
                for m in meanings
            ],
        }

    return {
        "mode": "claims",
        "baseline": baseline.label,
        "current": current.label,
        "fact_count": {"current": len(current.facts),
                       "baseline": len(baseline.facts)},
        "total_tokens": total,
        "per_document": per_document,
        "buckets": {
            name: [encode(t, c) for t, c in entries]
            for name, entries in buckets.items()
        },
    }


def run_citations(args: argparse.Namespace) -> int:
    root = Path(args.repo_root).resolve()
    current_reader = WorktreeReader(root)
    baseline_reader = GitReader(root, args.baseline_ref)

    artifact_root, declined = _resolve_artifact_root(root, args.artifact_root)
    current = harvest(current_reader, artifact_root, args.rehash_external_data)
    if declined:
        current.notes.append(declined)
    baseline = harvest(baseline_reader)
    classifier = Classifier(current, baseline)
    known = classifier.known_digests()

    docs = _documents(current_reader, args.include_historical, args.document)
    git = GitObjectResolver(root)
    findings: List[Finding] = check_evidence_conflicts(current)
    for path, role in docs:
        text = current_reader.read_text(path) or ""
        if role != "reconcile":
            continue
        findings.extend(check_table_citations(path, text, current))
        findings.extend(check_operator_citations(path, text, current))
        findings.extend(check_inline_digest_citations(path, text, current))
        findings.extend(check_unresolved_digests(
            path, tokenize(path, role, text, known), known, git))
    if not args.no_stale_sweep:
        findings.extend(sweep_stale_digests(current_reader, current, baseline))

    if args.json:
        print(json.dumps({
            "mode": "citations",
            "artifact_root": str(artifact_root) if artifact_root else None,
            "notes": current.notes,
            "findings": [f.__dict__ for f in findings],
        }, indent=2))
        return 1 if findings else 0

    print("=" * 78)
    print("citations -- bound numeric claims about the reference graphs")
    print("=" * 78)
    print(f"repository      {root}")
    print(f"baseline state  {baseline_reader.name}")
    print(f"artifact re-hash {artifact_root if artifact_root else 'not available'}")
    for note in current.notes:
        print(f"  note: {note}")
    print(f"measured facts  {len(current.facts)}")
    print(f"documents bound {sum(1 for _, r in docs if r == 'reconcile')}")
    print()

    if not findings:
        print("  no disagreements: every bound citation agrees with the")
        print("  measured evidence, and every generated source agrees with the")
        print("  others.")
    else:
        by_kind: Dict[str, List[Finding]] = {}
        for finding in findings:
            by_kind.setdefault(finding.kind, []).append(finding)
        for kind in sorted(by_kind):
            group = by_kind[kind]
            print("-" * 78)
            print(f"{kind} ({len(group)})")
            print("-" * 78)
            for finding in group:
                print(f"  [{finding.severity}] {finding.where}")
                print(f"      {finding.detail}")
            print()

    print("-" * 78)
    print("What this does not prove")
    print("-" * 78)
    print("  Only claims this tool can bind to a subject structurally are")
    print("  checked: table cells under a recognized header, `Op=N` and")
    print("  ``N `Op``` spans, digests quoted beside the graph they name, and")
    print("  agreement among the generated evidence files. Prose that names a")
    print("  number without naming its subject is not checked here -- run the")
    print("  `claims` mode for that, and expect to read it.")
    print(f"\n{len(findings)} disagreement(s)")
    return 1 if findings else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit_reference_graph_claims.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    for name, handler in (("citations", run_citations), ("claims", run_claims)):
        mode = sub.add_parser(
            name,
            help=("strict check of bound claims; exits non-zero on disagreement"
                  if name == "citations"
                  else "exhaustive numeric review queue over the documents"),
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        mode.set_defaults(handler=handler)
        mode.add_argument("--repo-root", default=str(REPO_ROOT),
                          help="repository root (default: this checkout)")
        mode.add_argument(
            "--baseline-ref", default="HEAD",
            help="git ref supplying the comparison snapshot of the measured "
                 "evidence (default: HEAD). Before the re-export lands this "
                 "equals the worktree and nothing can be reported as moved; "
                 "afterwards, name the pre-promotion commit.")
        mode.add_argument("--include-historical", action="store_true",
                          help="also scan documents recording past "
                               "measurements, which must not be edited")
        mode.add_argument("--document", action="append", default=[],
                          help="additional document to scan (repeatable)")
        mode.add_argument("--json", action="store_true",
                          help="machine-readable output")
        if name == "claims":
            mode.add_argument("--show-static", action="store_true",
                              help="also print the STATIC bucket")
            mode.add_argument("--limit", type=int, default=40,
                              help="cap per-section listing; 0 for no cap")
        else:
            mode.add_argument(
                "--artifact-root",
                help="directory holding S*/{prefill,decode}.onnx to re-hash "
                     "(default: $SLM_LAB_ARTIFACT_ROOT/onnx/reference/T20 when "
                     "present and --repo-root is this checkout; skipped cleanly "
                     "otherwise)")
            mode.add_argument("--rehash-external-data", action="store_true",
                              help="also SHA-256 the ~1.19 GB sidecars")
            mode.add_argument("--no-stale-sweep", action="store_true",
                              help="skip the repository-wide sweep for "
                                   "superseded reference digests")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
