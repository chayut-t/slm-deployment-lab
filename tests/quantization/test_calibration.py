"""Deterministic, offline tests for the frozen T40 calibration corpus."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from slm_lab.contracts import static_cache
from slm_lab.evaluation.fixtures import canonical_json_sha256
from slm_lab.quantization import calibration
from slm_lab.quantization.calibration import (
    CalibrationValidationError,
    build_calibration_samples,
    build_corpus,
    build_document,
    build_prefill_tensors,
    calibration_dataset_revision,
    main,
    render_document,
    validate_repository,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / calibration.DEFAULT_CALIBRATION_CONFIG
COPIED_PATHS = (
    calibration.DEFAULT_CALIBRATION_CONFIG,
    calibration.DEFAULT_TOKEN_BUNDLE,
    calibration.DEFAULT_WORKLOAD_CONFIG,
    calibration.DEFAULT_MODEL_CONTRACT,
    calibration.DEFAULT_EXPORT_CONTRACT,
)

# Independently re-fetched from https://huggingface.co/api/datasets/<id> on
# 2026-08-02. Duplicated here on purpose: if the module constants drift, this
# second copy fails rather than agreeing with the drift.
FETCHED_EXTERNAL_METADATA = {
    "Salesforce/wikitext": (
        "b08601e04326c79dfdd32d625aee71d232d685c3",
        ["cc-by-sa-3.0", "gfdl"],
    ),
    "allenai/c4": (
        "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        ["odc-by"],
    ),
    "wikimedia/wikipedia": (
        "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        ["cc-by-sa-3.0", "gfdl"],
    ),
}


@pytest.fixture(scope="module")
def inputs() -> dict[str, Any]:
    return calibration.load_inputs(REPO_ROOT)


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def samples(inputs: dict[str, Any]) -> list[calibration.CalibrationSample]:
    return build_calibration_samples(inputs["bundle"], inputs["model_contract"])


@pytest.fixture
def repo_copy(tmp_path: Path) -> Path:
    """Copy only the committed files the contract reads into a scratch tree."""

    for relative in COPIED_PATHS:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)
    return tmp_path


def write_config(root: Path, document: Any) -> None:
    """Write a (possibly tampered) contract document into a scratch tree."""

    path = root / calibration.DEFAULT_CALIBRATION_CONFIG
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def load_config(root: Path) -> dict[str, Any]:
    path = root / calibration.DEFAULT_CALIBRATION_CONFIG
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class TestRegeneration:
    """The committed contract must be a fixed point of its own generator."""

    def test_committed_contract_regenerates_byte_identically(
        self,
        inputs: dict[str, Any],
    ) -> None:
        expected = render_document(build_document(**inputs))
        assert CONFIG_PATH.read_text(encoding="utf-8") == expected

    def test_generate_is_idempotent(self, repo_copy: Path) -> None:
        first = (repo_copy / calibration.DEFAULT_CALIBRATION_CONFIG).read_text(
            encoding="utf-8"
        )
        calibration.generate_repository(repo_copy)
        second = (repo_copy / calibration.DEFAULT_CALIBRATION_CONFIG).read_text(
            encoding="utf-8"
        )
        assert first == second

    def test_validate_repository_accepts_the_committed_tree(self) -> None:
        validate_repository(REPO_ROOT)

    def test_check_cli_exits_zero_on_committed_tree(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert main(["--repo-root", str(REPO_ROOT), "check"]) == 0
        assert "check passed" in capsys.readouterr().out

    def test_verify_cli_exits_zero_offline(self) -> None:
        assert main(["--repo-root", str(REPO_ROOT), "verify"]) == 0

    def test_cli_reports_failure_without_raising(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert main(["--repo-root", str(tmp_path), "check"]) == 1
        assert "error:" in capsys.readouterr().err


class TestFrozenRevision:
    """The recorded revision must be the revision that is actually committed."""

    def test_recorded_t10_bundle_hash_is_the_committed_bundle(
        self,
        config: dict[str, Any],
        inputs: dict[str, Any],
    ) -> None:
        actual = canonical_json_sha256(inputs["bundle"])
        assert (
            config["calibration_corpus"]["source_bundle"]["canonical_json_sha256"]
            == actual
        )
        recorded = {
            entry["path"]: entry["canonical_json_sha256"] for entry in config["inputs"]
        }
        assert recorded[calibration.DEFAULT_TOKEN_BUNDLE.as_posix()] == actual
        raw_bundle = json.loads(
            (REPO_ROOT / calibration.DEFAULT_TOKEN_BUNDLE).read_text(encoding="utf-8")
        )
        assert canonical_json_sha256(raw_bundle) == actual

    def test_every_recorded_input_hash_matches_the_file_on_disk(
        self,
        config: dict[str, Any],
        inputs: dict[str, Any],
    ) -> None:
        expected = {
            calibration.DEFAULT_TOKEN_BUNDLE.as_posix(): inputs["bundle"],
            calibration.DEFAULT_WORKLOAD_CONFIG.as_posix(): inputs["workload_config"],
            calibration.DEFAULT_MODEL_CONTRACT.as_posix(): inputs["model_contract"],
            calibration.DEFAULT_EXPORT_CONTRACT.as_posix(): inputs["export_contract"],
        }
        assert len(config["inputs"]) == len(expected)
        for entry in config["inputs"]:
            assert entry["canonical_json_sha256"] == canonical_json_sha256(
                expected[entry["path"]]
            )

    def test_documented_recompute_command_reproduces_the_corpus_hash(
        self,
        config: dict[str, Any],
    ) -> None:
        payload = json.dumps(
            config["calibration_corpus"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        assert config["calibration_corpus_canonical_json_sha256"] == digest

    def test_dataset_revision_is_derived_from_the_corpus_hash(
        self,
        config: dict[str, Any],
    ) -> None:
        digest = config["calibration_corpus_canonical_json_sha256"]
        expected = calibration_dataset_revision(digest)
        assert config["calibration_dataset_revision"] == expected
        assert config["artifact_manifest_contract"]["value"] == expected
        assert digest[:16] in expected

    def test_manifest_field_is_the_one_the_plan_requires(
        self,
        config: dict[str, Any],
    ) -> None:
        contract = config["artifact_manifest_contract"]
        assert contract["field"] == "calibration_dataset_revision"
        model_contract = json.loads(
            (REPO_ROOT / calibration.DEFAULT_MODEL_CONTRACT).read_text(encoding="utf-8")
        )
        required = model_contract["toolchain_version_policy"]["required_fields"][
            "artifact_manifest"
        ]
        assert contract["field"] in required


class TestFreezeIsLoadBearing:
    """Moving a knob in *code* must move the corpus hash and the revision.

    Mutating a copy of the corpus dict and observing that its hash changed would
    prove only that SHA-256 is not a constant function, so no such test lives
    here. That every knob is load-bearing against the *committed* file is proved
    by ``TestDriftGuards``, which routes each mutation through
    ``validate_repository``.
    """

    def test_changing_the_short_fixture_length_in_code_changes_everything(
        self,
        inputs: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        baseline = build_corpus(inputs["bundle"], inputs["model_contract"])
        monkeypatch.setattr(calibration, "SHORT_FIXTURE_CONTEXT_LENGTH", 512)
        mutated = build_corpus(inputs["bundle"], inputs["model_contract"])
        assert canonical_json_sha256(mutated) != canonical_json_sha256(baseline)
        assert (
            mutated["token_budget"]["total_calibration_tokens"]
            != baseline["token_budget"]["total_calibration_tokens"]
        )
        assert calibration_dataset_revision(
            canonical_json_sha256(mutated)
        ) != calibration_dataset_revision(canonical_json_sha256(baseline))


class TestPrefillContractConformance:
    """Emitted tensors must satisfy the frozen T12 prefill contract."""

    def test_every_sample_emits_contract_conformant_tensors(
        self,
        samples: list[calibration.CalibrationSample],
    ) -> None:
        for sample in samples:
            contract = static_cache.build_prefill_contract(sample.context_length)
            tensors = build_prefill_tensors(sample)
            assert set(tensors) == {spec.name for spec in contract.inputs}
            static_cache.validate_tensor_mapping(tensors, contract.inputs)
            for spec in contract.inputs:
                assert tensors[spec.name].dtype == spec.dtype == "int64"
                assert tensors[spec.name].shape == spec.shape
                assert tensors[spec.name].shape[0] == static_cache.BATCH_SIZE

    def test_tensor_payloads_are_plain_python_lists_of_ints(
        self,
        samples: list[calibration.CalibrationSample],
    ) -> None:
        tensors = build_prefill_tensors(samples[0])
        for tensor in tensors.values():
            assert isinstance(tensor.values, list)
            assert all(isinstance(row, list) for row in tensor.values)
            assert all(
                isinstance(value, int) and not isinstance(value, bool)
                for row in tensor.values
                for value in row
            )

    def test_no_sample_is_ever_padded(
        self,
        samples: list[calibration.CalibrationSample],
    ) -> None:
        for sample in samples:
            tensors = build_prefill_tensors(sample)
            mask = tensors["attention_mask"].values[0]
            assert mask == [1] * sample.context_length
            assert len(tensors["input_ids"].values[0]) == sample.context_length
            assert tensors["position_ids"].values[0] == list(
                range(sample.context_length)
            )

    def test_length_mismatch_is_rejected(
        self,
        samples: list[calibration.CalibrationSample],
    ) -> None:
        short = calibration.CalibrationSample(
            **{**samples[0].__dict__, "token_ids": samples[0].token_ids[:-4]}
        )
        with pytest.raises(CalibrationValidationError, match="prefill graph"):
            build_prefill_tensors(short)

    def test_an_unfrozen_prompt_length_is_rejected(self) -> None:
        with pytest.raises(static_cache.CacheContractError):
            static_cache.build_prefill_contract(256)


class TestCoverageAndBudget:
    """The corpus must actually cover the workloads it claims to cover."""

    def test_all_four_deployment_context_lengths_are_present(
        self,
        samples: list[calibration.CalibrationSample],
    ) -> None:
        verbatim = {
            sample.context_length
            for sample in samples
            if sample.source_group == "context_workloads"
        }
        assert verbatim == set(static_cache.CONTEXT_VARIANTS)

    def test_all_token_class_canaries_and_the_chat_path_are_present(
        self,
        samples: list[calibration.CalibrationSample],
    ) -> None:
        fixture_ids = {sample.source_fixture_id for sample in samples}
        assert {
            "raw_ascii",
            "raw_whitespace",
            "raw_unicode",
            "raw_structured",
        } <= fixture_ids
        chat = [sample for sample in samples if sample.source_group == "chat_canary"]
        assert len(chat) == 1
        assert chat[0].interface == "chat_template"
        quality = [
            sample for sample in samples if sample.source_group == "quality_subset"
        ]
        assert len(quality) == 4

    def test_every_sample_states_why_it_is_there(
        self,
        samples: list[calibration.CalibrationSample],
    ) -> None:
        for sample in samples:
            assert len(sample.rationale.split()) >= 15

    def test_token_budget_equals_the_sum_of_sample_token_counts(
        self,
        config: dict[str, Any],
    ) -> None:
        corpus = config["calibration_corpus"]
        rows = corpus["samples"]
        budget = corpus["token_budget"]
        assert budget["total_calibration_tokens"] == sum(
            row["token_count"] for row in rows
        )
        assert budget["sample_count"] == len(rows)
        assert (
            sum(budget["tokens_per_prompt_shape"].values())
            == (budget["total_calibration_tokens"])
        )
        assert (
            sum(budget["tokens_per_source_group"].values())
            == (budget["total_calibration_tokens"])
        )

    def test_recorded_coverage_matches_an_independent_measurement(
        self,
        config: dict[str, Any],
        samples: list[calibration.CalibrationSample],
        inputs: dict[str, Any],
    ) -> None:
        """The coverage block must be a measurement, recomputed here from scratch."""

        coverage = config["calibration_corpus"]["coverage"]
        distinct = {token_id for sample in samples for token_id in sample.token_ids}
        vocab_size = inputs["model_contract"]["model"]["architecture"]["vocab_size"]

        assert coverage["distinct_token_ids"] == len(distinct)
        assert coverage["model_vocabulary_size"] == vocab_size
        assert coverage["vocabulary_fraction"] == round(len(distinct) / vocab_size, 6)

        # Per group as well as overall. These four numbers are quoted verbatim
        # as measurements in docs/learning/calibration_and_aimet.md, so they
        # need an independent union recomputation rather than the overall one.
        per_group: dict[str, set[int]] = {}
        for sample in samples:
            per_group.setdefault(sample.source_group, set()).update(sample.token_ids)
        assert coverage["distinct_token_ids_per_source_group"] == {
            group: len(ids) for group, ids in sorted(per_group.items())
        }

        total = sum(sample.token_count for sample in samples)
        for group, share in coverage["token_share_per_source_group"].items():
            measured = sum(
                sample.token_count for sample in samples if sample.source_group == group
            )
            assert share == round(measured / total, 4)

    def test_the_context_workloads_are_disclosed_as_nested_prefixes(
        self,
        config: dict[str, Any],
        samples: list[calibration.CalibrationSample],
    ) -> None:
        """The study checkpoint fails if the manifest hides this."""

        contexts = sorted(
            (
                sample
                for sample in samples
                if sample.source_group == "context_workloads"
            ),
            key=lambda sample: sample.context_length,
        )
        for shorter, longer in zip(contexts, contexts[1:]):
            assert longer.token_ids[: len(shorter.token_ids)] == shorter.token_ids

        coverage = config["calibration_corpus"]["coverage"]
        assert coverage["context_workloads_are_nested_prefixes"] is True
        assert coverage["context_workload_chain"] == [
            f"S{sample.context_length}" for sample in contexts
        ]

    def test_every_sample_records_its_own_distinct_token_count(
        self,
        config: dict[str, Any],
        samples: list[calibration.CalibrationSample],
    ) -> None:
        rows = {row["id"]: row for row in config["calibration_corpus"]["samples"]}
        assert set(rows) == {sample.sample_id for sample in samples}
        for sample in samples:
            assert rows[sample.sample_id]["distinct_token_ids"] == len(
                set(sample.token_ids)
            )

    def test_every_token_id_is_within_the_model_vocabulary(
        self,
        samples: list[calibration.CalibrationSample],
        inputs: dict[str, Any],
    ) -> None:
        vocab_size = inputs["model_contract"]["model"]["architecture"]["vocab_size"]
        assert vocab_size == static_cache.VOCAB_SIZE
        for sample in samples:
            assert all(0 <= token_id < vocab_size for token_id in sample.token_ids)

    def test_ordering_is_deterministic_and_lexicographic(
        self,
        samples: list[calibration.CalibrationSample],
    ) -> None:
        sample_ids = [sample.sample_id for sample in samples]
        assert sample_ids == sorted(sample_ids)
        assert len(set(sample_ids)) == len(sample_ids)
        groups = [sample.source_group for sample in samples]
        assert groups == [
            *["context_workloads"] * 4,
            *["raw_canaries"] * 4,
            "chat_canary",
            *["quality_subset"] * 4,
        ]

    def test_tiling_repeats_the_source_sequence_exactly(
        self,
        samples: list[calibration.CalibrationSample],
        inputs: dict[str, Any],
    ) -> None:
        by_fixture = {
            record["id"]: record for record in inputs["bundle"]["raw_canaries"]
        }
        tiled = next(
            sample for sample in samples if sample.source_fixture_id == "raw_ascii"
        )
        source = tuple(by_fixture["raw_ascii"]["token_ids"])
        assert tiled.construction == "tiled_then_truncated"
        assert tiled.token_ids[: len(source)] == source
        expected = (source * tiled.tile_repeats)[: tiled.context_length]
        assert tiled.token_ids == expected
        assert set(tiled.token_ids) == set(source)


class TestExternalTier:
    """Tier 2 is declared, revision-pinned, licensed, and never committed."""

    def test_every_candidate_is_declared_only(
        self,
        config: dict[str, Any],
    ) -> None:
        block = config["external_diversity_candidates"]
        assert block["data_committed"] is False
        assert block["candidates"]
        for candidate in block["candidates"]:
            assert candidate["data_committed"] is False
            assert len(candidate["revision"]) == 40
            assert set(candidate["revision"]) <= set("0123456789abcdef")
            assert candidate["license"]
            assert all(name.strip() for name in candidate["license"])
            assert candidate["license_obligations"].strip()

    def test_pinned_revisions_and_licences_match_what_was_fetched(
        self,
        config: dict[str, Any],
    ) -> None:
        recorded = {
            candidate["id"]: (candidate["revision"], candidate["license"])
            for candidate in config["external_diversity_candidates"]["candidates"]
        }
        assert recorded == FETCHED_EXTERNAL_METADATA

    def test_licensing_block_states_the_obligations_and_the_boundary(
        self,
        config: dict[str, Any],
    ) -> None:
        licensing = config["licensing"]
        assert licensing["repository_license"] == "Apache-2.0"
        assert licensing["tier_1_license"] == "CC0-1.0"
        assert licensing["third_party_rows_committed"] is False
        obligations = " ".join(licensing["tier_2_obligations"]).lower()
        assert "share-alike" in obligations
        assert "attribution" in obligations
        assert "odc-by" in obligations
        assert licensing["evaluation_overlap"]["overlaps"] is True

    def test_no_committed_file_contains_a_third_party_dataset_row(
        self,
        config: dict[str, Any],
    ) -> None:
        corpus = config["calibration_corpus"]
        assert corpus["license"] == "CC0-1.0"
        assert corpus["source_bundle"]["license"] == "CC0-1.0"
        assert all(row["tier"] == "t10_derived" for row in corpus["samples"])
        # Token IDs live once, in the CC0 T10 bundle. The contract pins them by
        # hash instead of copying them into a second place that can drift.
        assert all("token_ids" not in row for row in corpus["samples"])


class TestDriftGuards:
    """Every guard in validate_repository must actually fail closed."""

    def test_missing_contract_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(CalibrationValidationError, match="is missing"):
            validate_repository(tmp_path)

    def test_source_bundle_drift_is_rejected(self, repo_copy: Path) -> None:
        bundle_path = repo_copy / calibration.DEFAULT_TOKEN_BUNDLE
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        bundle["raw_canaries"][0]["token_ids"][0] += 1
        bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
        with pytest.raises(CalibrationValidationError, match="input hash drift"):
            validate_repository(repo_copy)

    def test_corpus_hash_drift_is_rejected(self, repo_copy: Path) -> None:
        document = load_config(repo_copy)
        document["calibration_corpus_canonical_json_sha256"] = "0" * 64
        write_config(repo_copy, document)
        with pytest.raises(CalibrationValidationError, match="corpus hash drift"):
            validate_repository(repo_copy)

    def test_token_count_drift_is_rejected(self, repo_copy: Path) -> None:
        document = load_config(repo_copy)
        document["calibration_corpus"]["samples"][0]["token_count"] = 127
        write_config(repo_copy, document)
        with pytest.raises(CalibrationValidationError, match="token count drift"):
            validate_repository(repo_copy)

    def test_context_assignment_drift_is_rejected(self, repo_copy: Path) -> None:
        document = load_config(repo_copy)
        document["calibration_corpus"]["samples"][4]["context_length"] = 512
        write_config(repo_copy, document)
        with pytest.raises(
            CalibrationValidationError,
            match="context assignment drift",
        ):
            validate_repository(repo_copy)

    def test_sample_reordering_is_rejected(self, repo_copy: Path) -> None:
        document = load_config(repo_copy)
        document["calibration_corpus"]["samples"].reverse()
        write_config(repo_copy, document)
        with pytest.raises(CalibrationValidationError, match="ordering drift"):
            validate_repository(repo_copy)

    def test_dropping_a_sample_is_rejected(self, repo_copy: Path) -> None:
        document = load_config(repo_copy)
        del document["calibration_corpus"]["samples"][-1]
        write_config(repo_copy, document)
        with pytest.raises(CalibrationValidationError, match="sample count drift"):
            validate_repository(repo_copy)

    def test_preprocessing_drift_is_rejected(self, repo_copy: Path) -> None:
        document = load_config(repo_copy)
        preprocessing = document["calibration_corpus"]["preprocessing"]
        preprocessing["padding"]["policy"] = "pad_to_context_length"
        write_config(repo_copy, document)
        with pytest.raises(
            CalibrationValidationError,
            match="preprocessing contract drift",
        ):
            validate_repository(repo_copy)

    def test_tiling_target_drift_is_rejected(self, repo_copy: Path) -> None:
        document = load_config(repo_copy)
        assignment = document["calibration_corpus"]["preprocessing"][
            "context_assignment"
        ]
        assignment["short_fixture_target_length"] = 512
        write_config(repo_copy, document)
        with pytest.raises(
            CalibrationValidationError,
            match="preprocessing contract drift",
        ):
            validate_repository(repo_copy)

    def test_coverage_measurement_drift_is_rejected(self, repo_copy: Path) -> None:
        document = load_config(repo_copy)
        coverage = document["calibration_corpus"]["coverage"]
        coverage["distinct_token_ids"] = 151_936
        write_config(repo_copy, document)
        with pytest.raises(
            CalibrationValidationError,
            match="coverage measurement drift",
        ):
            validate_repository(repo_copy)

    def test_token_budget_drift_is_rejected(self, repo_copy: Path) -> None:
        document = load_config(repo_copy)
        budget = document["calibration_corpus"]["token_budget"]
        budget["total_calibration_tokens"] += 1
        write_config(repo_copy, document)
        with pytest.raises(CalibrationValidationError, match="token budget drift"):
            validate_repository(repo_copy)

    def test_dataset_revision_drift_is_rejected(self, repo_copy: Path) -> None:
        document = load_config(repo_copy)
        document["calibration_dataset_revision"] = "t40-handwritten-v9"
        write_config(repo_copy, document)
        with pytest.raises(
            CalibrationValidationError,
            match="calibration_dataset_revision drift",
        ):
            validate_repository(repo_copy)

    def test_committed_tier_two_row_is_rejected(self, repo_copy: Path) -> None:
        document = load_config(repo_copy)
        block = document["external_diversity_candidates"]
        block["candidates"][0]["data_committed"] = True
        write_config(repo_copy, document)
        with pytest.raises(
            CalibrationValidationError,
            match="dataset rows may not be committed",
        ):
            validate_repository(repo_copy)

    def test_tier_two_licence_drift_is_rejected(self, repo_copy: Path) -> None:
        document = load_config(repo_copy)
        block = document["external_diversity_candidates"]
        block["candidates"][0]["license"] = ["mit"]
        write_config(repo_copy, document)
        with pytest.raises(
            CalibrationValidationError,
            match="differs from the pinned, fetched values",
        ):
            validate_repository(repo_copy)

    def test_tier_two_revision_drift_is_rejected(self, repo_copy: Path) -> None:
        document = load_config(repo_copy)
        block = document["external_diversity_candidates"]
        block["candidates"][1]["revision"] = "main"
        write_config(repo_copy, document)
        with pytest.raises(
            CalibrationValidationError,
            match="full 40-hex commit SHA",
        ):
            validate_repository(repo_copy)

    def test_licensing_drift_is_rejected(self, repo_copy: Path) -> None:
        document = load_config(repo_copy)
        document["licensing"]["third_party_rows_committed"] = True
        write_config(repo_copy, document)
        with pytest.raises(
            CalibrationValidationError,
            match="third-party dataset rows may not be committed",
        ):
            validate_repository(repo_copy)

    def test_unrelated_prose_edit_is_rejected(self, repo_copy: Path) -> None:
        document = load_config(repo_copy)
        document["summary"] = "A quantized model was measured here."
        write_config(repo_copy, document)
        with pytest.raises(
            CalibrationValidationError,
            match="differs from the derived contract",
        ):
            validate_repository(repo_copy)

    def test_formatting_drift_is_rejected(self, repo_copy: Path) -> None:
        path = repo_copy / calibration.DEFAULT_CALIBRATION_CONFIG
        path.write_text(
            path.read_text(encoding="utf-8") + "# hand-appended\n",
            encoding="utf-8",
        )
        with pytest.raises(
            CalibrationValidationError,
            match="not byte-identical to a fresh regeneration",
        ):
            validate_repository(repo_copy)

    def test_schema_and_task_identity_are_enforced(self, repo_copy: Path) -> None:
        document = load_config(repo_copy)
        document["task_id"] = "T41"
        write_config(repo_copy, document)
        with pytest.raises(CalibrationValidationError, match="unexpected task ID"):
            validate_repository(repo_copy)


class TestMalformedInputsFailClosed:
    """A corrupt input is a validation failure, never a decoder traceback.

    ``check`` is the CI gate, so an unreadable or unparseable input must exit 1
    with an ``error:`` line the same way a drifted input does. Every case below
    tampers with a `tmp_path` copy; no repository file is ever mutated.
    """

    def test_unparseable_json_input_is_rejected(self, repo_copy: Path) -> None:
        path = repo_copy / calibration.DEFAULT_TOKEN_BUNDLE
        path.write_text("{", encoding="utf-8")
        with pytest.raises(CalibrationValidationError, match="cannot parse"):
            validate_repository(repo_copy)

    def test_non_utf8_json_input_is_rejected(self, repo_copy: Path) -> None:
        path = repo_copy / calibration.DEFAULT_TOKEN_BUNDLE
        path.write_bytes(b"\xff\xfe\x00bad")
        with pytest.raises(CalibrationValidationError, match="cannot parse"):
            validate_repository(repo_copy)

    def test_unparseable_yaml_contract_is_rejected(self, repo_copy: Path) -> None:
        path = repo_copy / calibration.DEFAULT_CALIBRATION_CONFIG
        path.write_text("samples: [1,\nother: 2\n", encoding="utf-8")
        with pytest.raises(CalibrationValidationError, match="cannot parse"):
            validate_repository(repo_copy)

    def test_unreadable_input_is_rejected(self, repo_copy: Path) -> None:
        path = repo_copy / calibration.DEFAULT_TOKEN_BUNDLE
        path.unlink()
        path.mkdir()
        with pytest.raises(CalibrationValidationError, match="cannot read"):
            validate_repository(repo_copy)

    def test_check_cli_reports_a_malformed_input_without_raising(
        self,
        repo_copy: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (repo_copy / calibration.DEFAULT_TOKEN_BUNDLE).write_text("{", encoding="utf-8")
        assert main(["--repo-root", str(repo_copy), "check"]) == 1
        captured = capsys.readouterr()
        assert captured.err.startswith("error: cannot parse")
        assert "Traceback" not in captured.err


class TestBuilderGuards:
    """Guards that protect the builder itself, exercised in memory."""

    def test_out_of_vocabulary_token_is_rejected(
        self,
        inputs: dict[str, Any],
    ) -> None:
        bundle = copy.deepcopy(inputs["bundle"])
        record = bundle["raw_canaries"][0]
        record["token_ids"][0] = static_cache.VOCAB_SIZE
        record["token_ids_sha256"] = canonical_json_sha256(record["token_ids"])
        with pytest.raises(
            CalibrationValidationError,
            match="outside model vocabulary",
        ):
            build_calibration_samples(bundle, inputs["model_contract"])

    def test_t10_token_hash_drift_is_rejected(
        self,
        inputs: dict[str, Any],
    ) -> None:
        bundle = copy.deepcopy(inputs["bundle"])
        bundle["raw_canaries"][0]["token_ids_sha256"] = "0" * 64
        with pytest.raises(CalibrationValidationError, match="token ID hash drift"):
            build_calibration_samples(bundle, inputs["model_contract"])

    def test_a_new_t10_fixture_without_a_rationale_is_rejected(
        self,
        inputs: dict[str, Any],
    ) -> None:
        bundle = copy.deepcopy(inputs["bundle"])
        extra = copy.deepcopy(bundle["raw_canaries"][0])
        extra["id"] = "raw_undocumented"
        bundle["raw_canaries"].append(extra)
        with pytest.raises(
            CalibrationValidationError,
            match="no T40 selection rationale",
        ):
            build_calibration_samples(bundle, inputs["model_contract"])

    def test_empty_token_sequence_is_rejected(
        self,
        inputs: dict[str, Any],
    ) -> None:
        bundle = copy.deepcopy(inputs["bundle"])
        record = bundle["raw_canaries"][0]
        record["token_ids"] = []
        record["token_count"] = 0
        with pytest.raises(CalibrationValidationError, match="has no token IDs"):
            build_calibration_samples(bundle, inputs["model_contract"])

    def test_vocabulary_mismatch_between_contracts_is_rejected(
        self,
        inputs: dict[str, Any],
    ) -> None:
        model_contract = copy.deepcopy(inputs["model_contract"])
        model_contract["model"]["architecture"]["vocab_size"] = 32_000
        with pytest.raises(
            CalibrationValidationError,
            match="differs from the T12 contract",
        ):
            build_calibration_samples(inputs["bundle"], model_contract)

    def test_missing_t10_group_is_rejected(self, inputs: dict[str, Any]) -> None:
        bundle = copy.deepcopy(inputs["bundle"])
        del bundle["chat_canary"]
        with pytest.raises(CalibrationValidationError, match="missing 'chat_canary'"):
            build_calibration_samples(bundle, inputs["model_contract"])

    def test_short_corpus_hash_is_rejected(self) -> None:
        with pytest.raises(CalibrationValidationError, match="64-character SHA-256"):
            calibration_dataset_revision("abc")
