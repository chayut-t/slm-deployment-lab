"""Load and validate versioned artifact and host manifests."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator, FormatChecker


SCHEMAS = {
    "artifact": "artifact-v1.schema.json",
    "host": "host-v1.schema.json",
}


class ManifestValidationError(ValueError):
    """Raised when a manifest does not satisfy its versioned schema."""


def load_document(path: Path) -> Any:
    """Load a JSON or YAML document without accepting an empty document."""

    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            value = json.load(handle)
        else:
            value = yaml.safe_load(handle)
    if value is None:
        raise ManifestValidationError(f"{path}: document is empty")
    return value


def load_schema(kind: str) -> Mapping[str, Any]:
    """Return the packaged schema for *kind*."""

    try:
        schema_name = SCHEMAS[kind]
    except KeyError as exc:
        choices = ", ".join(sorted(SCHEMAS))
        raise ManifestValidationError(
            f"unknown manifest kind {kind!r}; choose one of: {choices}"
        ) from exc
    schema_path = files("slm_lab.manifests").joinpath("schemas", schema_name)
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _display_path(error_path: Any) -> str:
    parts = [str(part) for part in error_path]
    return ".".join(parts) if parts else "<root>"


def validate_manifest(kind: str, document: Any) -> None:
    """Validate *document* and raise one stable, aggregated exception."""

    schema = load_schema(kind)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(document),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        details = "; ".join(
            f"{_display_path(error.absolute_path)}: {error.message}" for error in errors
        )
        raise ManifestValidationError(f"{kind} manifest is invalid: {details}")
