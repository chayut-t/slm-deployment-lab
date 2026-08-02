"""Regression tests for T10 prompt, token, and evaluation fixtures."""

from __future__ import annotations

import copy
import json
import os
import tomllib
import unittest
from pathlib import Path

from slm_lab.evaluation.fixtures import (
    EXPECTED_CONTEXTS,
    FixtureValidationError,
    canonical_json_sha256,
    load_pinned_tokenizer,
    validate_documents,
    validate_repository,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONTRACT_PATH = REPO_ROOT / "configs/models/qwen3-0.6b.yaml"
SOURCE_PATH = REPO_ROOT / "tests/fixtures/t10/source-prompts-v1.json"
BUNDLE_PATH = REPO_ROOT / "tests/fixtures/t10/token-fixtures-v1.json"
CONFIG_PATH = REPO_ROOT / "configs/workloads/t10-token-fixtures.json"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


class T10FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_contract = json.loads(MODEL_CONTRACT_PATH.read_text(encoding="utf-8"))
        cls.source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
        cls.bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_repository_fixtures_validate_without_network(self) -> None:
        validate_repository(REPO_ROOT)

    def test_four_contexts_have_exact_token_counts(self) -> None:
        records = self.bundle["context_workloads"]
        self.assertEqual(
            tuple(record["context_length"] for record in records),
            EXPECTED_CONTEXTS,
        )
        for record in records:
            self.assertEqual(record["token_count"], record["context_length"])
            self.assertEqual(len(record["token_ids"]), record["context_length"])
            self.assertEqual(record["id"], f"S{record['context_length']}")

    def test_tokenizer_identity_and_special_tokens_match_t00(self) -> None:
        expected = self.model_contract["tokenizer"]
        actual = self.bundle["tokenizer"]
        self.assertEqual(actual["id"], expected["id"])
        self.assertEqual(actual["revision"], expected["revision"])
        self.assertEqual(
            actual["chat_template_sha256"],
            expected["chat_template"]["sha256"],
        )
        self.assertFalse(actual["add_bos_token"])
        self.assertEqual(actual["pad_id"], expected["tokens"]["pad_id"])
        self.assertEqual(actual["eos_id"], expected["tokens"]["eos_id"])
        for canary in self.bundle["raw_canaries"]:
            self.assertFalse(canary["add_special_tokens"])

    def test_chat_canary_is_explicitly_non_thinking(self) -> None:
        chat = self.bundle["chat_canary"]
        self.assertEqual(chat["interface"], "chat_template")
        self.assertIs(chat["enable_thinking"], False)
        self.assertTrue(chat["add_generation_prompt"])
        self.assertIn("<think>\n\n</think>", chat["rendered_prompt"])

    def test_committed_quality_cases_are_authored_and_cc0(self) -> None:
        provenance = self.source["provenance"]
        self.assertTrue(provenance["authored_for_repository"])
        self.assertFalse(provenance["contains_private_data"])
        self.assertFalse(provenance["contains_third_party_dataset_rows"])
        for source_case, token_case in zip(
            self.source["quality_subset"],
            self.bundle["quality_subset"],
            strict=True,
        ):
            self.assertEqual(source_case["license"], "CC0-1.0")
            self.assertEqual(token_case["license"], "CC0-1.0")
        for candidate in self.source["external_quality_candidates"]:
            self.assertFalse(candidate["data_committed"])
            self.assertEqual(candidate["selection_owner"], "T13")

    def test_manifest_hash_detects_token_drift(self) -> None:
        tampered = copy.deepcopy(self.bundle)
        tampered["raw_canaries"][0]["token_ids"][0] += 1
        with self.assertRaisesRegex(
            FixtureValidationError,
            "token fixture bundle hash drift",
        ):
            validate_documents(
                source=self.source,
                bundle=tampered,
                config=self.config,
                model_contract=self.model_contract,
            )

    def test_workload_config_must_match_token_bundle(self) -> None:
        tampered = copy.deepcopy(self.config)
        tampered["context_workloads"][0]["generated_tokens"] += 1
        with self.assertRaisesRegex(
            FixtureValidationError,
            "workload config differs from token bundle",
        ):
            validate_documents(
                source=self.source,
                bundle=self.bundle,
                config=tampered,
                model_contract=self.model_contract,
            )

    def test_authoritative_config_rejects_coherent_metadata_tampering(self) -> None:
        mutations = (
            ("prompt_interface", "chat_template"),
            ("model_contract", "configs/models/another-model.json"),
            ("generation_command", "python unpinned_generator.py"),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                tampered = copy.deepcopy(self.config)
                tampered[field] = value
                with self.assertRaisesRegex(
                    FixtureValidationError,
                    "authoritative workload config differs",
                ):
                    validate_documents(
                        source=self.source,
                        bundle=self.bundle,
                        config=tampered,
                        model_contract=self.model_contract,
                    )

    def test_bundle_rejects_coherent_tokenizer_metadata_tampering(self) -> None:
        tampered_bundle = copy.deepcopy(self.bundle)
        tampered_bundle["tokenizer"]["trust_remote_code"] = True
        tampered_config = copy.deepcopy(self.config)
        tampered_config["tokenizer"]["trust_remote_code"] = True
        tampered_config["token_fixture_bundle"]["canonical_json_sha256"] = (
            canonical_json_sha256(tampered_bundle)
        )
        with self.assertRaisesRegex(
            FixtureValidationError,
            "tokenizer metadata differs",
        ):
            validate_documents(
                source=self.source,
                bundle=tampered_bundle,
                config=tampered_config,
                model_contract=self.model_contract,
            )

    def test_generation_policy_is_deterministic_and_explicit(self) -> None:
        policy = self.config["generation_policy"]
        self.assertEqual(policy["decoding"]["strategy"], "greedy")
        self.assertFalse(policy["decoding"]["do_sample"])
        self.assertEqual(policy["decoding"]["argmax_tie_break"], "lowest_token_id")
        self.assertIsNone(policy["seed"]["value"])
        self.assertTrue(policy["stopping"]["stop_on_eos"])
        self.assertEqual(
            policy["stopping"]["eos_token_ids"],
            [self.model_contract["tokenizer"]["tokens"]["eos_id"]],
        )
        self.assertEqual(
            policy["stopping"]["pad_token_id"],
            self.model_contract["tokenizer"]["tokens"]["pad_id"],
        )

    def test_generation_policy_drift_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.config)
        tampered["generation_policy"]["decoding"]["do_sample"] = True
        with self.assertRaisesRegex(
            FixtureValidationError,
            "authoritative workload config differs",
        ):
            validate_documents(
                source=self.source,
                bundle=self.bundle,
                config=tampered,
                model_contract=self.model_contract,
            )

    def test_tokenizer_extra_and_commands_are_exact(self) -> None:
        pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
        tokenizer_extra = pyproject["project"]["optional-dependencies"]["tokenizer"]
        self.assertIn("transformers==4.51.3", tokenizer_extra)
        self.assertIn("jinja2==3.1.6", tokenizer_extra)
        self.assertEqual(
            self.config["generation_command"],
            "uv run --extra tokenizer slm-lab-fixtures generate",
        )
        self.assertEqual(
            self.config["verification_command"],
            "uv run --extra tokenizer slm-lab-fixtures verify",
        )

    @unittest.skipUnless(
        os.environ.get("SLM_LAB_VERIFY_UPSTREAM") == "1",
        "set SLM_LAB_VERIFY_UPSTREAM=1 to re-encode with the pinned tokenizer",
    )
    def test_pinned_tokenizer_exactly_regenerates_bundle(self) -> None:
        tokenizer = load_pinned_tokenizer(
            REPO_ROOT,
            local_files_only=os.environ.get("SLM_LAB_VERIFY_OFFLINE") == "1",
        )
        validate_repository(REPO_ROOT, tokenizer=tokenizer)


if __name__ == "__main__":
    unittest.main()
