"""Command-line manifest validator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .validation import (
    SCHEMAS,
    ManifestValidationError,
    load_document,
    validate_manifest,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=sorted(SCHEMAS))
    parser.add_argument("path", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        document = load_document(args.path)
        validate_manifest(args.kind, document)
    except (OSError, ValueError, ManifestValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{args.kind} manifest valid: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
