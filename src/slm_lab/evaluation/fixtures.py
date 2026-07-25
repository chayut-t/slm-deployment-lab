"""Build and verify deterministic prompt, token, and evaluation fixtures."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


DEFAULT_MODEL_CONTRACT = Path("configs/models/qwen3-0.6b.yaml")
DEFAULT_SOURCE = Path("tests/fixtures/t10/source-prompts-v1.json")
DEFAULT_BUNDLE = Path("tests/fixtures/t10/token-fixtures-v1.json")
DEFAULT_WORKLOAD_CONFIG = Path("configs/workloads/t10-token-fixtures.json")
TRANSFORMERS_VERSION = "4.51.3"
EXPECTED_CONTEXTS = (128, 512, 1024, 4096)
GENERATED_TOKENS = {128: 32, 512: 64, 1024: 128, 4096: 128}


class FixtureValidationError(ValueError):
    """Raised when a T10 fixture or its provenance is inconsistent."""


class TokenizerLike(Protocol):
    """Narrow tokenizer interface used by fixture generation and verification."""

    chat_template: str | None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Encode text as token IDs."""

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        clean_up_tokenization_spaces: bool,
        skip_special_tokens: bool,
    ) -> str:
        """Decode token IDs to text."""

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        """Render a chat conversation."""


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _canonical_json_bytes(document: Any) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(document: Any) -> str:
    """Return the SHA-256 of a stable canonical JSON representation."""

    return hashlib.sha256(_canonical_json_bytes(document)).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    return canonical_json_sha256(list(token_ids))


def _write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _encode_record(
    tokenizer: TokenizerLike,
    *,
    fixture_id: str,
    purpose: str,
    prompt: str,
) -> dict[str, Any]:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    with_special = tokenizer.encode(prompt, add_special_tokens=True)
    if with_special != token_ids:
        raise FixtureValidationError(
            f"{fixture_id}: raw completion unexpectedly adds special tokens"
        )
    return {
        "id": fixture_id,
        "purpose": purpose,
        "interface": "raw_completion",
        "add_special_tokens": False,
        "prompt": prompt,
        "prompt_sha256": _text_sha256(prompt),
        "token_ids": token_ids,
        "token_ids_sha256": _token_ids_sha256(token_ids),
        "token_count": len(token_ids),
    }


def _exact_context_record(
    tokenizer: TokenizerLike,
    *,
    seed: Mapping[str, Any],
    context_length: int,
) -> dict[str, Any]:
    seed_text = str(seed["text"])
    repeats = 1
    while True:
        candidate = "\n\n".join([seed_text] * repeats)
        candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(candidate_ids) >= context_length:
            break
        repeats *= 2

    token_ids = candidate_ids[:context_length]
    prompt = tokenizer.decode(
        token_ids,
        clean_up_tokenization_spaces=False,
        skip_special_tokens=False,
    )
    round_trip = tokenizer.encode(prompt, add_special_tokens=False)
    if round_trip != token_ids:
        raise FixtureValidationError(
            f"S{context_length}: decoded prompt does not reproduce exact token IDs"
        )

    record = _encode_record(
        tokenizer,
        fixture_id=f"S{context_length}",
        purpose=f"Exact {context_length}-token static workload",
        prompt=prompt,
    )
    record.update(
        {
            "context_length": context_length,
            "generated_tokens": GENERATED_TOKENS[context_length],
            "seed_id": seed["id"],
            "construction": "repeat authored seed, tokenize, truncate IDs, decode",
        }
    )
    return record


def _chat_record(
    tokenizer: TokenizerLike,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if source.get("enable_thinking") is not False:
        raise FixtureValidationError("chat canary must explicitly disable thinking")
    rendered = tokenizer.apply_chat_template(
        source["messages"],
        tokenize=False,
        add_generation_prompt=bool(source["add_generation_prompt"]),
        enable_thinking=False,
    )
    token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    return {
        "id": source["id"],
        "purpose": source["purpose"],
        "interface": "chat_template",
        "messages": source["messages"],
        "enable_thinking": False,
        "add_generation_prompt": bool(source["add_generation_prompt"]),
        "rendered_prompt": rendered,
        "prompt_sha256": _text_sha256(rendered),
        "token_ids": token_ids,
        "token_ids_sha256": _token_ids_sha256(token_ids),
        "token_count": len(token_ids),
    }


def _quality_record(
    tokenizer: TokenizerLike,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    prompt = _encode_record(
        tokenizer,
        fixture_id=str(source["id"]),
        purpose=str(source["category"]),
        prompt=str(source["prompt"]),
    )
    references = []
    for answer in source["reference_answers"]:
        token_ids = tokenizer.encode(str(answer), add_special_tokens=False)
        references.append(
            {
                "text": answer,
                "text_sha256": _text_sha256(str(answer)),
                "token_ids": token_ids,
                "token_ids_sha256": _token_ids_sha256(token_ids),
            }
        )
    return {
        **prompt,
        "category": source["category"],
        "license": source["license"],
        "reference_answers": references,
    }


def build_bundle(
    tokenizer: TokenizerLike,
    source: Mapping[str, Any],
    model_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic token fixture bundle from authored source data."""

    tokenizer_contract = model_contract["tokenizer"]
    chat_template = tokenizer.chat_template
    if not isinstance(chat_template, str):
        raise FixtureValidationError("loaded tokenizer has no chat template")
    chat_template_sha256 = _text_sha256(chat_template)
    expected_template_sha256 = tokenizer_contract["chat_template"]["sha256"]
    if chat_template_sha256 != expected_template_sha256:
        raise FixtureValidationError("loaded chat template does not match T00")

    raw_canaries = [
        _encode_record(
            tokenizer,
            fixture_id=item["id"],
            purpose=item["purpose"],
            prompt=item["prompt"],
        )
        for item in source["raw_canaries"]
    ]
    contexts = [
        _exact_context_record(
            tokenizer,
            seed=source["context_seed"],
            context_length=context_length,
        )
        for context_length in EXPECTED_CONTEXTS
    ]
    quality_subset = [
        _quality_record(tokenizer, item) for item in source["quality_subset"]
    ]

    return {
        "schema_version": 1,
        "task_id": "T10",
        "license": source["license"],
        "source_fixture": {
            "path": DEFAULT_SOURCE.as_posix(),
            "canonical_json_sha256": canonical_json_sha256(source),
        },
        "tokenizer": {
            "id": tokenizer_contract["id"],
            "revision": tokenizer_contract["revision"],
            "class": tokenizer.__class__.__name__,
            "transformers_version": importlib.metadata.version("transformers"),
            "tokenizers_version": importlib.metadata.version("tokenizers"),
            "trust_remote_code": False,
            "add_bos_token": tokenizer_contract["add_bos_token"],
            "pad_id": tokenizer_contract["tokens"]["pad_id"],
            "eos_id": tokenizer_contract["tokens"]["eos_id"],
            "chat_template_sha256": chat_template_sha256,
        },
        "raw_canaries": raw_canaries,
        "chat_canary": _chat_record(tokenizer, source["chat_canary"]),
        "context_workloads": contexts,
        "quality_subset": quality_subset,
        "external_quality_candidates": source["external_quality_candidates"],
    }


def build_workload_config(
    source: Mapping[str, Any],
    bundle: Mapping[str, Any],
    model_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the declarative workload-to-fixture manifest."""

    project_contract = model_contract["project_contract"]
    contexts = []
    for context_length in project_contract["static_context_lengths"]:
        contexts.append(
            {
                "id": f"S{context_length}",
                "context_length": context_length,
                "generated_tokens": GENERATED_TOKENS[context_length],
                "batch_size": 1,
                "prompt_fixture_id": f"S{context_length}",
            }
        )
    return {
        "schema_version": 1,
        "task_id": "T10",
        "model_contract": DEFAULT_MODEL_CONTRACT.as_posix(),
        "prompt_interface": "raw_completion",
        "tokenizer": bundle["tokenizer"],
        "source_fixture": {
            "path": DEFAULT_SOURCE.as_posix(),
            "canonical_json_sha256": canonical_json_sha256(source),
        },
        "token_fixture_bundle": {
            "path": DEFAULT_BUNDLE.as_posix(),
            "canonical_json_sha256": canonical_json_sha256(bundle),
        },
        "context_workloads": contexts,
        "canary_ids": [
            *[item["id"] for item in bundle["raw_canaries"]],
            bundle["chat_canary"]["id"],
        ],
        "quality_subset_ids": [item["id"] for item in bundle["quality_subset"]],
        "generation_command": (
            "uv run --extra tokenizer slm-lab-fixtures generate"
        ),
        "verification_command": (
            "uv run --extra tokenizer slm-lab-fixtures verify"
        ),
    }


def _validate_encoded_record(record: Mapping[str, Any], label: str) -> None:
    prompt = record.get("prompt", record.get("rendered_prompt"))
    if not isinstance(prompt, str):
        raise FixtureValidationError(f"{label}: prompt text is missing")
    token_ids = record.get("token_ids")
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) and token_id >= 0 for token_id in token_ids
    ):
        raise FixtureValidationError(f"{label}: token_ids must be nonnegative ints")
    if record.get("token_count") != len(token_ids):
        raise FixtureValidationError(f"{label}: token count drift")
    if record.get("prompt_sha256") != _text_sha256(prompt):
        raise FixtureValidationError(f"{label}: prompt hash drift")
    if record.get("token_ids_sha256") != _token_ids_sha256(token_ids):
        raise FixtureValidationError(f"{label}: token ID hash drift")


def validate_documents(
    *,
    source: Mapping[str, Any],
    bundle: Mapping[str, Any],
    config: Mapping[str, Any],
    model_contract: Mapping[str, Any],
    tokenizer: TokenizerLike | None = None,
) -> None:
    """Validate fixture structure, hashes, contracts, and optional re-encoding."""

    for label, document in (
        ("source", source),
        ("bundle", bundle),
        ("config", config),
    ):
        if document.get("schema_version") != 1:
            raise FixtureValidationError(f"{label}: unsupported schema version")
        if document.get("task_id") != "T10":
            raise FixtureValidationError(f"{label}: unexpected task ID")
    if source.get("license") != "CC0-1.0":
        raise FixtureValidationError("authored source fixtures must remain CC0-1.0")
    provenance = source.get("provenance", {})
    if provenance.get("contains_private_data") is not False:
        raise FixtureValidationError("source fixture privacy boundary is not explicit")
    if provenance.get("contains_third_party_dataset_rows") is not False:
        raise FixtureValidationError("third-party dataset rows may not be committed")
    for candidate in source["external_quality_candidates"]:
        if candidate.get("data_committed") is not False:
            raise FixtureValidationError(
                f"{candidate['id']}: external evaluation data must remain external"
            )

    source_hash = canonical_json_sha256(source)
    if bundle["source_fixture"]["canonical_json_sha256"] != source_hash:
        raise FixtureValidationError("bundle source hash drift")
    if config["source_fixture"]["canonical_json_sha256"] != source_hash:
        raise FixtureValidationError("config source hash drift")
    if config["token_fixture_bundle"]["canonical_json_sha256"] != (
        canonical_json_sha256(bundle)
    ):
        raise FixtureValidationError("token fixture bundle hash drift")

    expected_tokenizer = model_contract["tokenizer"]
    actual_tokenizer = bundle["tokenizer"]
    for field in ("id", "revision"):
        if actual_tokenizer[field] != expected_tokenizer[field]:
            raise FixtureValidationError(f"tokenizer {field} differs from T00")
    if actual_tokenizer["class"] != expected_tokenizer["class"]:
        raise FixtureValidationError("tokenizer class differs from T00")
    if actual_tokenizer["transformers_version"] != TRANSFORMERS_VERSION:
        raise FixtureValidationError("unexpected Transformers version")
    if actual_tokenizer["chat_template_sha256"] != (
        expected_tokenizer["chat_template"]["sha256"]
    ):
        raise FixtureValidationError("chat template hash differs from T00")

    encoded_records = [
        *bundle["raw_canaries"],
        bundle["chat_canary"],
        *bundle["context_workloads"],
        *bundle["quality_subset"],
    ]
    ids = [record["id"] for record in encoded_records]
    if len(ids) != len(set(ids)):
        raise FixtureValidationError("fixture IDs must be unique")
    for record in encoded_records:
        _validate_encoded_record(record, record["id"])
        if any(
            token_id >= model_contract["model"]["architecture"]["vocab_size"]
            for token_id in record["token_ids"]
        ):
            raise FixtureValidationError(
                f"{record['id']}: token ID exceeds model vocabulary"
            )
    for quality in bundle["quality_subset"]:
        if quality["license"] != "CC0-1.0":
            raise FixtureValidationError(
                f"{quality['id']}: committed quality case is not CC0"
            )
        for reference in quality["reference_answers"]:
            if reference["text_sha256"] != _text_sha256(reference["text"]):
                raise FixtureValidationError(
                    f"{quality['id']}: reference text hash drift"
                )
            if reference["token_ids_sha256"] != _token_ids_sha256(
                reference["token_ids"]
            ):
                raise FixtureValidationError(
                    f"{quality['id']}: reference token hash drift"
                )

    expected_contexts = tuple(
        model_contract["project_contract"]["static_context_lengths"]
    )
    actual_contexts = tuple(
        record["context_length"] for record in bundle["context_workloads"]
    )
    if expected_contexts != EXPECTED_CONTEXTS or actual_contexts != EXPECTED_CONTEXTS:
        raise FixtureValidationError("static context coverage differs from T00")
    for record in bundle["context_workloads"]:
        if record["token_count"] != record["context_length"]:
            raise FixtureValidationError(
                f"{record['id']}: context length is not exact"
            )
        if record["generated_tokens"] != GENERATED_TOKENS[record["context_length"]]:
            raise FixtureValidationError(
                f"{record['id']}: generated-token contract drift"
            )

    expected_config_contexts = [
        {
            "id": record["id"],
            "context_length": record["context_length"],
            "generated_tokens": record["generated_tokens"],
            "batch_size": 1,
            "prompt_fixture_id": record["id"],
        }
        for record in bundle["context_workloads"]
    ]
    if config["context_workloads"] != expected_config_contexts:
        raise FixtureValidationError("workload config differs from token bundle")
    expected_canary_ids = [
        *[record["id"] for record in bundle["raw_canaries"]],
        bundle["chat_canary"]["id"],
    ]
    if config["canary_ids"] != expected_canary_ids:
        raise FixtureValidationError("canary IDs differ from token bundle")
    expected_quality_ids = [record["id"] for record in bundle["quality_subset"]]
    if config["quality_subset_ids"] != expected_quality_ids:
        raise FixtureValidationError("quality IDs differ from token bundle")

    if bundle["chat_canary"].get("enable_thinking") is not False:
        raise FixtureValidationError("chat canary must disable thinking")
    if tokenizer is not None:
        regenerated = build_bundle(tokenizer, source, model_contract)
        if regenerated != bundle:
            raise FixtureValidationError(
                "stored token fixtures differ from pinned tokenizer regeneration"
            )


def validate_repository(
    repo_root: Path,
    *,
    tokenizer: TokenizerLike | None = None,
) -> None:
    """Validate the committed T10 fixtures rooted at *repo_root*."""

    source = _load_json(repo_root / DEFAULT_SOURCE)
    bundle = _load_json(repo_root / DEFAULT_BUNDLE)
    config = _load_json(repo_root / DEFAULT_WORKLOAD_CONFIG)
    model_contract = _load_json(repo_root / DEFAULT_MODEL_CONTRACT)
    validate_documents(
        source=source,
        bundle=bundle,
        config=config,
        model_contract=model_contract,
        tokenizer=tokenizer,
    )


def load_pinned_tokenizer(
    repo_root: Path,
    *,
    local_files_only: bool,
) -> TokenizerLike:
    """Load the immutable T00 tokenizer without importing model code."""

    if importlib.metadata.version("transformers") != TRANSFORMERS_VERSION:
        raise FixtureValidationError(
            f"expected transformers {TRANSFORMERS_VERSION}"
        )
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise FixtureValidationError(
            "install the locked tokenizer extra before upstream verification"
        ) from exc

    contract = _load_json(repo_root / DEFAULT_MODEL_CONTRACT)
    tokenizer_contract = contract["tokenizer"]
    return AutoTokenizer.from_pretrained(
        tokenizer_contract["id"],
        revision=tokenizer_contract["revision"],
        trust_remote_code=False,
        use_fast=False,
        local_files_only=local_files_only,
    )


def generate_repository(
    repo_root: Path,
    *,
    local_files_only: bool,
) -> None:
    """Regenerate both committed T10 manifests from the pinned tokenizer."""

    source = _load_json(repo_root / DEFAULT_SOURCE)
    model_contract = _load_json(repo_root / DEFAULT_MODEL_CONTRACT)
    tokenizer = load_pinned_tokenizer(
        repo_root,
        local_files_only=local_files_only,
    )
    bundle = build_bundle(tokenizer, source, model_contract)
    config = build_workload_config(source, bundle, model_contract)
    _write_json(repo_root / DEFAULT_BUNDLE, bundle)
    _write_json(repo_root / DEFAULT_WORKLOAD_CONFIG, config)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument(
            "--offline",
            action="store_true",
            help="require the pinned tokenizer to exist in the local cache",
        )
    subparsers.add_parser("check")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        if args.command == "generate":
            generate_repository(
                repo_root,
                local_files_only=args.offline,
            )
        elif args.command == "verify":
            tokenizer = load_pinned_tokenizer(
                repo_root,
                local_files_only=args.offline,
            )
            validate_repository(repo_root, tokenizer=tokenizer)
        else:
            validate_repository(repo_root)
    except (FixtureValidationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"T10 fixtures {args.command} passed: {repo_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
