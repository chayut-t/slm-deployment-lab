"""Tests for the T31 three-target Workbench run plan builder.

No test here contacts a service, imports ``qai_hub``, reads a candidate
graph, or asserts anything about what Qualcomm AI Hub would accept. The
planner is offline by construction and these tests hold it to that.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from slm_lab.deployment.qualcomm import workbench
from slm_lab.deployment.qualcomm.ai_hub import (
    AiHubAdapterError,
    _validate_options,
)


REPO_ROOT = Path(workbench.__file__).resolve().parents[4]

#: The compile request id the committed T22 package record already carries for
#: Snapdragon X Elite CRD / S128 / prefill. The planner must derive the same
#: value offline, because it is the same request.
COMMITTED_S128_PREFILL_REQUEST_ID = "t30-compile-83b8813c19a37ac036ad"

#: Key names that would only appear if a measurement had been taken. None may
#: appear anywhere in the record.
MEASUREMENT_KEYS = frozenset(
    {
        "estimated_inference_time_us",
        "estimated_inference_time_ms",
        "estimated_inference_peak_memory_bytes",
        "first_load_time_us",
        "warm_load_time_us",
        "compute_units",
        "observed_inference_time_range_us",
        "normalized_profile",
        "service_turnaround_seconds",
        "max_absolute_error",
        "cosine_similarity",
    }
)


def _keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            found.add(key)
            found |= _keys(child)
    elif isinstance(value, list):
        for child in value:
            found |= _keys(child)
    return found


class _PlanFixture:
    """Build the plan once per class. Not a TestCase, so nothing reruns."""

    paths: dict[str, Any]
    plan: dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.paths = workbench.default_paths(REPO_ROOT)
        cls.plan = workbench.build_plan(
            target_paths=cls.paths["targets"],
            package_paths=cls.paths["packages"],
            inspection_paths=cls.paths["inspections"],
            parity_paths=cls.paths["parity"],
            repository_root=cls.paths["repository_root"],
        )


class PlanConstructionTest(_PlanFixture, unittest.TestCase):
    def test_plan_covers_every_target_variant_and_graph_kind(self) -> None:
        self.assertEqual(len(self.plan["targets"]), 3)
        self.assertEqual(len(self.plan["graphs"]), 8)
        self.assertEqual(len(self.plan["plan"]), 24)
        pairs = {
            (entry["target"], entry["variant_id"], entry["graph_kind"])
            for entry in self.plan["plan"]
        }
        self.assertEqual(len(pairs), 24)
        for entry in self.plan["plan"]:
            self.assertEqual(set(entry["stages"]), set(workbench.STAGE_ORDER))

    def test_summary_counts_match_the_stage_readiness(self) -> None:
        summary = self.plan["summary"]
        self.assertEqual(summary["stages_ready"], 24)
        self.assertEqual(summary["stages_pending_predecessor"], 48)

    def test_every_stage_readiness_is_one_of_the_two_declared_values(self) -> None:
        for entry in self.plan["plan"]:
            for stage in workbench.STAGE_ORDER:
                self.assertIn(
                    entry["stages"][stage]["readiness"], workbench.READINESS_VALUES
                )

    def test_compile_is_ready_and_downstream_stages_are_pending(self) -> None:
        for entry in self.plan["plan"]:
            stages = entry["stages"]
            self.assertEqual(stages["compile"]["readiness"], workbench.READY)
            self.assertIsNotNone(stages["compile"]["request_id"])
            for stage in ("inference", "profile"):
                self.assertEqual(
                    stages[stage]["readiness"], workbench.PENDING_PREDECESSOR
                )
                self.assertEqual(stages[stage]["depends_on_stage"], "compile")

    def test_no_downstream_stage_carries_a_fabricated_predecessor(self) -> None:
        for entry in self.plan["plan"]:
            for stage in ("inference", "profile"):
                block = entry["stages"][stage]
                self.assertIsNone(block["predecessor_manifest"])
                self.assertIsNone(block["compiled_artifact_sha256"])
                self.assertIsNone(block["request_id"])
                self.assertIn("predecessor_manifest", block["unresolved_input_ids"])
                self.assertIn("compiled_artifact_sha256", block["unresolved_input_ids"])

    def test_only_the_inference_stage_declares_a_missing_input_dataset(self) -> None:
        for entry in self.plan["plan"]:
            self.assertIn(
                "input_dataset", entry["stages"]["inference"]["unresolved_input_ids"]
            )
            self.assertNotIn(
                "input_dataset", entry["stages"]["profile"]["unresolved_input_ids"]
            )

    def test_compile_request_id_matches_the_committed_package_record(self) -> None:
        first = self.plan["plan"][0]
        self.assertEqual(
            first["stages"]["compile"]["request_id"],
            COMMITTED_S128_PREFILL_REQUEST_ID,
        )
        committed = json.loads(
            (REPO_ROOT / workbench.PACKAGE_RECORD_DIRECTORY / "S128.json").read_text(
                encoding="utf-8"
            )
        )
        prefill = next(
            graph
            for graph in committed["package"]["graphs"]
            if graph["graph_kind"] == "prefill"
        )
        self.assertEqual(
            prefill["compile_request"]["request_id"],
            COMMITTED_S128_PREFILL_REQUEST_ID,
        )

    def test_every_x_elite_request_id_matches_its_committed_package_record(
        self,
    ) -> None:
        """The T22 records were written by the real preflight; these are equal."""

        committed: dict[tuple[str, str], str] = {}
        for variant in workbench.VARIANT_IDS:
            record = json.loads(
                (
                    REPO_ROOT / workbench.PACKAGE_RECORD_DIRECTORY / f"{variant}.json"
                ).read_text(encoding="utf-8")
            )
            for graph in record["package"]["graphs"]:
                committed[(variant, graph["graph_kind"])] = graph["compile_request"][
                    "request_id"
                ]
        checked = 0
        for entry in self.plan["plan"]:
            if entry["target"] != "qualcomm-snapdragon-x-elite-crd":
                continue
            key = (entry["variant_id"], entry["graph_kind"])
            self.assertEqual(entry["stages"]["compile"]["request_id"], committed[key])
            checked += 1
        self.assertEqual(checked, 8)

    def test_the_other_targets_do_not_reuse_an_x_elite_request_id(self) -> None:
        x_elite = {
            entry["stages"]["compile"]["request_id"]
            for entry in self.plan["plan"]
            if entry["target"] == "qualcomm-snapdragon-x-elite-crd"
        }
        others = {
            entry["stages"]["compile"]["request_id"]
            for entry in self.plan["plan"]
            if entry["target"] != "qualcomm-snapdragon-x-elite-crd"
        }
        self.assertEqual(x_elite & others, set())

    def test_every_compile_request_id_is_distinct(self) -> None:
        ids = [entry["stages"]["compile"]["request_id"] for entry in self.plan["plan"]]
        self.assertEqual(len(set(ids)), 24)

    def test_the_planner_never_imports_the_qualcomm_client(self) -> None:
        self.assertNotIn("qai_hub", sys.modules)
        workbench.build_plan(
            target_paths=self.paths["targets"],
            package_paths=self.paths["packages"],
            inspection_paths=self.paths["inspections"],
            parity_paths=self.paths["parity"],
            repository_root=self.paths["repository_root"],
        )
        self.assertNotIn("qai_hub", sys.modules)


class SubmissionOrderTest(_PlanFixture, unittest.TestCase):
    def test_first_submission_is_s128_prefill_on_the_authenticated_target(
        self,
    ) -> None:
        first = self.plan["plan"][0]
        self.assertEqual(first["submission_order"], 1)
        self.assertEqual(first["variant_id"], "S128")
        self.assertEqual(first["graph_kind"], "prefill")
        self.assertEqual(first["target"], "qualcomm-snapdragon-x-elite-crd")
        self.assertEqual(
            first["device_evidence_strength"],
            workbench.DEVICE_EVIDENCE_AUTHENTICATED,
        )
        self.assertEqual(
            self.plan["summary"]["first_submission"]["request_id"],
            first["stages"]["compile"]["request_id"],
        )

    def test_submission_order_is_dense_and_starts_at_one(self) -> None:
        orders = [entry["submission_order"] for entry in self.plan["plan"]]
        self.assertEqual(orders, list(range(1, 25)))

    def test_s128_prefill_is_the_only_graph_with_no_residual_shape_finding(
        self,
    ) -> None:
        zeros = [
            graph
            for graph in self.plan["graphs"]
            if graph["shape_residue"]["candidate"]["total"] == 0
        ]
        self.assertEqual(len(zeros), 1)
        self.assertEqual(zeros[0]["variant_id"], "S128")
        self.assertEqual(zeros[0]["graph_kind"], "prefill")

    def test_graph_order_is_monotone_in_the_declared_keys(self) -> None:
        keys = [
            (
                graph["graph_order_key"]["residual_high_severity_shape_findings"],
                graph["graph_order_key"]["candidate_graph_protobuf_bytes"],
            )
            for graph in self.plan["graphs"]
        ]
        self.assertEqual(keys, sorted(keys))

    def test_no_decode_graph_precedes_a_prefill_graph_on_the_same_target(
        self,
    ) -> None:
        for target in self.plan["targets"]:
            entries = [
                entry
                for entry in self.plan["plan"]
                if entry["target"] == target["config_id"]
            ]
            kinds = [entry["graph_kind"] for entry in entries]
            self.assertEqual(kinds, sorted(kinds, key=lambda k: k != "prefill"))
            self.assertEqual(kinds[:4], ["prefill"] * 4)

    def test_decode_candidates_score_worse_than_their_reference(self) -> None:
        for graph in self.plan["graphs"]:
            if graph["graph_kind"] != "decode":
                continue
            residue = graph["shape_residue"]
            self.assertGreater(
                residue["candidate"]["total"], residue["reference"]["total"]
            )
            self.assertGreater(residue["candidate"]["R-INTERNAL-DYNAMIC-SHAPE"], 0)

    def test_one_target_is_exhausted_before_the_next_begins(self) -> None:
        seen: list[str] = []
        for entry in self.plan["plan"]:
            if not seen or seen[-1] != entry["target"]:
                self.assertNotIn(entry["target"], seen)
                seen.append(entry["target"])
        self.assertEqual(len(seen), 3)


class TargetEvidenceTest(_PlanFixture, unittest.TestCase):
    def test_targets_are_ordered_by_device_evidence_then_config_id(self) -> None:
        ranks = [target["device_evidence"]["rank"] for target in self.plan["targets"]]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual(ranks, [0, 1, 1])
        self.assertEqual(
            [target["config_id"] for target in self.plan["targets"]],
            [
                "qualcomm-snapdragon-x-elite-crd",
                "qualcomm-dragonwing-iq-9075-evk",
                "qualcomm-snapdragon-8-elite-qrd",
            ],
        )

    def test_only_one_target_claims_an_authenticated_device_query(self) -> None:
        confirmed = [
            target
            for target in self.plan["targets"]
            if target["device_evidence"]["device_confirmed_by_authenticated_query"]
        ]
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0]["config_id"], "qualcomm-snapdragon-x-elite-crd")

    def test_catalog_only_targets_carry_their_own_refusals(self) -> None:
        for target in self.plan["targets"][1:]:
            evidence = target["device_evidence"]
            self.assertEqual(
                evidence["strength"], workbench.DEVICE_EVIDENCE_CATALOG_ONLY
            )
            self.assertFalse(evidence["device_confirmed_by_authenticated_query"])
            self.assertIn(
                workbench.CATALOG_ONLY_DEVICE_MARKER, evidence["does_not_establish"]
            )
            self.assertIn(
                "this_account_can_reach_or_schedule_this_device",
                evidence["does_not_establish"],
            )

    def test_every_target_carries_the_no_proxy_rule(self) -> None:
        for target in self.plan["targets"]:
            self.assertEqual(
                target["device_evidence"]["no_proxy_rule"], workbench.NO_PROXY_RULE
            )
        self.assertEqual(self.plan["no_proxy_rule"], workbench.NO_PROXY_RULE)

    def test_device_evidence_refuses_a_config_declaring_neither_marker(self) -> None:
        with self.assertRaises(AiHubAdapterError):
            workbench.device_evidence(
                {"claim_boundary": {"establishes": [], "does_not_establish": []}}
            )

    def test_device_evidence_refuses_a_config_declaring_both_markers(self) -> None:
        with self.assertRaises(AiHubAdapterError):
            workbench.device_evidence(
                {
                    "claim_boundary": {
                        "establishes": [workbench.AUTHENTICATED_DEVICE_MARKER],
                        "does_not_establish": [workbench.CATALOG_ONLY_DEVICE_MARKER],
                    }
                }
            )


class StageOptionTest(_PlanFixture, unittest.TestCase):
    def test_every_stage_option_string_passes_the_t30_allowlist(self) -> None:
        for target in self.plan["targets"]:
            for stage in workbench.STAGE_ORDER:
                options = target["stage_options"][stage]
                self.assertEqual(
                    _validate_options(options, target["runtime"], stage), options
                )

    def test_downstream_stages_bind_the_runtime_with_qairt_framework(self) -> None:
        for target in self.plan["targets"]:
            version = target["runtime"]["version"]
            for stage in ("inference", "profile"):
                self.assertEqual(
                    target["stage_options"][stage], f"--qairt_framework {version}"
                )

    def test_no_stage_pins_a_compute_unit(self) -> None:
        for target in self.plan["targets"]:
            for stage in workbench.STAGE_ORDER:
                self.assertNotIn("compute_unit", target["stage_options"][stage])
            self.assertIn("placement", target["compute_unit_note"])

    def test_job_name_prefixes_carry_the_stage_name(self) -> None:
        for target in self.plan["targets"]:
            for stage in workbench.STAGE_ORDER:
                self.assertIn(stage, target["job_name_prefixes"][stage])

    def test_a_prefix_without_a_compile_token_is_refused(self) -> None:
        with self.assertRaises(AiHubAdapterError):
            workbench._stage_job_prefix("slm-lab-t31", "inference")


class OfflineValidationTest(unittest.TestCase):
    """The offline chain must not be weaker than the committed T30 chain."""

    def setUp(self) -> None:
        paths = workbench.default_paths(REPO_ROOT)
        self.targets = workbench.load_targets(paths["targets"])
        self.graphs = workbench.load_graphs(
            package_paths=paths["packages"],
            inspection_paths=paths["inspections"],
            parity_paths=paths["parity"],
            repository_root=paths["repository_root"],
        )
        record = json.loads(
            (REPO_ROOT / workbench.PACKAGE_RECORD_DIRECTORY / "S128.json").read_text(
                encoding="utf-8"
            )
        )
        self.specs = next(
            graph["compile_request"]["input_specs"]
            for graph in record["package"]["graphs"]
            if graph["graph_kind"] == "prefill"
        )
        self.graph = next(
            graph
            for graph in self.graphs
            if graph["variant_id"] == "S128" and graph["graph_kind"] == "prefill"
        )

    def _request(self) -> dict[str, Any]:
        return workbench.compile_request_for(self.targets[0], self.graph, self.specs)

    def test_the_committed_request_validates_and_reproduces_its_id(self) -> None:
        validated = workbench.validate_compile_request_offline(
            self._request(), source_byte_size=self.graph["source_byte_size"]
        )
        self.assertEqual(validated["request_id"], COMMITTED_S128_PREFILL_REQUEST_ID)
        self.assertNotIn("path", validated["public_request"])
        self.assertNotIn("output_artifact", validated["public_request"])

    def test_a_flag_outside_the_compile_allowlist_is_refused(self) -> None:
        request = self._request()
        request["options"] = f"{request['options']} --compute_unit npu"
        with self.assertRaises(AiHubAdapterError):
            workbench.validate_compile_request_offline(
                request, source_byte_size=self.graph["source_byte_size"]
            )

    def test_a_url_bearing_job_name_is_refused(self) -> None:
        request = self._request()
        request["job_name"] = "https://example.invalid/job"
        with self.assertRaises(AiHubAdapterError):
            workbench.validate_compile_request_offline(
                request, source_byte_size=self.graph["source_byte_size"]
            )

    def test_a_retry_request_is_refused(self) -> None:
        request = self._request()
        request["retry"] = True
        with self.assertRaises(AiHubAdapterError):
            workbench.validate_compile_request_offline(
                request, source_byte_size=self.graph["source_byte_size"]
            )

    def test_an_unknown_field_is_refused(self) -> None:
        request = self._request()
        request["compute_unit"] = "npu"
        with self.assertRaises(AiHubAdapterError):
            workbench.validate_compile_request_offline(
                request, source_byte_size=self.graph["source_byte_size"]
            )

    def test_a_wrong_runtime_version_in_the_options_is_refused(self) -> None:
        request = self._request()
        request["options"] = "--target_runtime qnn_context_binary --qairt_version 1.0.0"
        with self.assertRaises(AiHubAdapterError):
            workbench.validate_compile_request_offline(
                request, source_byte_size=self.graph["source_byte_size"]
            )

    def test_a_nonpositive_source_byte_size_is_refused(self) -> None:
        with self.assertRaises(AiHubAdapterError):
            workbench.validate_compile_request_offline(
                self._request(), source_byte_size=0
            )

    def test_the_two_deferred_checks_are_named_rather_than_dropped(self) -> None:
        contract = workbench._stage_contracts()["compile"]
        self.assertEqual(
            contract["deferred_to_submission_time"],
            list(workbench.COMPILE_DEFERRED_CHECKS),
        )
        self.assertIn(
            "source_artifact_path_exists", contract["deferred_to_submission_time"][0]
        )


class RecordTest(unittest.TestCase):
    """The committed record, and what ``--check`` refuses."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.paths = workbench.default_paths(REPO_ROOT)
        cls.record_path = cls.paths["record"]
        cls.record = json.loads(cls.record_path.read_text(encoding="utf-8"))

    def test_the_committed_record_matches_its_inputs(self) -> None:
        workbench.check_record(record_path=self.record_path, paths=self.paths)

    def test_the_committed_record_records_no_job_and_no_service_call(self) -> None:
        boundary = self.record["submission_boundary"]
        self.assertEqual(boundary["jobs_submitted"], 0)
        self.assertFalse(boundary["service_contacted"])
        self.assertEqual(boundary["network_calls_made"], 0)
        observation = self.record["run_observation"]
        self.assertEqual(observation["jobs_submitted"], 0)
        self.assertFalse(observation["service_contacted"])
        self.assertFalse(observation["client_probe"]["module_imported"])

    def test_the_cost_record_is_zero(self) -> None:
        cost = self.record["cost"]
        self.assertEqual(cost["jobs_submitted"], 0)
        self.assertEqual(cost["device_minutes_consumed"], 0)
        self.assertEqual(cost["amount_spent"], "0.00")
        self.assertEqual(cost["currency"], "USD")

    def test_the_boundary_names_both_missing_preconditions(self) -> None:
        ids = {
            item["id"]
            for item in self.record["submission_boundary"][
                "required_before_any_submission"
            ]
        }
        self.assertEqual(
            ids,
            {
                "capability:qai_hub_client_absent",
                "user_authorization:qai_hub_submission_for_T31",
            },
        )

    def test_the_first_failure_hypothesis_is_a_field_with_an_attribution_rule(
        self,
    ) -> None:
        hypothesis = self.record["first_failure_hypothesis"]
        self.assertEqual(hypothesis["affected_stage"], "compile")
        self.assertIn("PACKAGING", hypothesis["attribution_rule"])
        self.assertEqual(len(hypothesis["external_data_by_graph"]), 8)
        for item in hypothesis["external_data_by_graph"]:
            self.assertTrue(item["external_data"])

    def test_the_claim_boundary_refuses_every_device_claim(self) -> None:
        boundary = self.record["claim_boundary"]
        for refusal in (
            "compiler_acceptance_or_operator_support_for_any_qwen3_graph",
            "latency_throughput_peak_memory_or_energy_on_any_target",
            "that_a_result_on_one_target_transfers_to_another_target",
        ):
            self.assertIn(refusal, boundary["does_not_establish"])

    def test_the_record_carries_no_measurement_key(self) -> None:
        self.assertEqual(_keys(self.record) & MEASUREMENT_KEYS, set())

    def test_the_record_carries_no_absolute_path_or_artifact_token(self) -> None:
        text = self.record_path.read_text(encoding="utf-8")
        self.assertNotIn(str(REPO_ROOT), text)
        self.assertNotIn("/Volumes/", text)
        self.assertNotIn("SLM_LAB_ARTIFACT_ROOT", text)
        self.assertNotIn("http", text)

    def test_every_input_is_bound_by_digest(self) -> None:
        for target in self.record["targets"]:
            self.assertRegex(target["source"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(target["source"]["path"].startswith("configs/targets/"))
        for graph in self.record["graphs"]:
            for key in ("package_record", "candidate_manifest", "inspection_manifest"):
                self.assertRegex(graph["inputs"][key]["sha256"], r"^[0-9a-f]{64}$")

    def test_the_numerical_reference_is_named_and_uncompared(self) -> None:
        for graph in self.record["graphs"]:
            reference = graph["numerical_reference"]
            self.assertEqual(reference["status"], "not_compared")
            self.assertTrue(reference["blocked_by"])
            self.assertTrue(
                reference["record"].startswith("results/manifests/qnn/parity/")
            )


class CheckDriftTest(unittest.TestCase):
    def setUp(self) -> None:
        self.paths = workbench.default_paths(REPO_ROOT)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "record.json"

    def _write(self, record: dict[str, Any]) -> None:
        self.path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _built(self) -> dict[str, Any]:
        return workbench.build_record(record_path=self.path, paths=self.paths)

    def test_a_freshly_built_record_passes_check(self) -> None:
        self._built()
        workbench.check_record(record_path=self.path, paths=self.paths)

    def test_a_changed_request_id_is_caught(self) -> None:
        record = self._built()
        record["plan"][0]["stages"]["compile"]["request_id"] = "t30-compile-0" * 2
        self._write(record)
        with self.assertRaises(AiHubAdapterError) as caught:
            workbench.check_record(record_path=self.path, paths=self.paths)
        self.assertIn("request_id", str(caught.exception))

    def test_a_reordered_plan_is_caught(self) -> None:
        record = self._built()
        record["plan"].reverse()
        self._write(record)
        with self.assertRaises(AiHubAdapterError):
            workbench.check_record(record_path=self.path, paths=self.paths)

    def test_a_promoted_readiness_is_caught(self) -> None:
        record = self._built()
        record["plan"][0]["stages"]["inference"]["readiness"] = workbench.READY
        self._write(record)
        with self.assertRaises(AiHubAdapterError):
            workbench.check_record(record_path=self.path, paths=self.paths)

    def test_a_promoted_device_evidence_strength_is_caught(self) -> None:
        record = self._built()
        record["targets"][1]["device_evidence"]["strength"] = (
            workbench.DEVICE_EVIDENCE_AUTHENTICATED
        )
        self._write(record)
        with self.assertRaises(AiHubAdapterError):
            workbench.check_record(record_path=self.path, paths=self.paths)

    def test_a_record_claiming_a_submitted_job_is_refused(self) -> None:
        record = self._built()
        record["run_observation"]["jobs_submitted"] = 1
        self._write(record)
        with self.assertRaises(AiHubAdapterError) as caught:
            workbench.check_record(record_path=self.path, paths=self.paths)
        self.assertIn("submitted job", str(caught.exception))

    def test_a_record_claiming_a_contacted_service_is_refused(self) -> None:
        record = self._built()
        record["run_observation"]["service_contacted"] = True
        self._write(record)
        with self.assertRaises(AiHubAdapterError):
            workbench.check_record(record_path=self.path, paths=self.paths)

    def test_a_record_without_an_observation_block_is_refused(self) -> None:
        record = self._built()
        record.pop("run_observation")
        self._write(record)
        with self.assertRaises(AiHubAdapterError):
            workbench.check_record(record_path=self.path, paths=self.paths)

    def test_building_never_imports_the_qualcomm_client(self) -> None:
        self._built()
        self.assertNotIn("qai_hub", sys.modules)


class PreflightObservationTest(unittest.TestCase):
    """The stamped preflight claim must come from what preflight observed."""

    def setUp(self) -> None:
        self.paths = workbench.default_paths(REPO_ROOT)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "record.json"

    def _observation(self, preflight: list[dict[str, Any]] | None) -> dict[str, Any]:
        record = workbench.build_record(
            record_path=self.path, paths=self.paths, preflight=preflight
        )
        return record["run_observation"]["preflight"]

    def test_a_clean_preflight_stamps_both_flags_false(self) -> None:
        observation = self._observation(
            [
                {"service_contacted": False, "job_submitted": False},
                {"service_contacted": False, "job_submitted": False},
            ]
        )
        self.assertEqual(observation["requests_validated"], 2)
        self.assertIs(observation["service_contacted"], False)
        self.assertIs(observation["job_submitted"], False)

    def test_a_contacted_request_is_not_flattened_into_a_false_claim(self) -> None:
        observation = self._observation(
            [
                {"service_contacted": False, "job_submitted": False},
                {"service_contacted": True, "job_submitted": False},
            ]
        )
        self.assertIs(observation["service_contacted"], True)
        self.assertIs(observation["job_submitted"], False)

    def test_a_submitted_request_is_not_flattened_into_a_false_claim(self) -> None:
        observation = self._observation(
            [{"service_contacted": True, "job_submitted": True}]
        )
        self.assertIs(observation["job_submitted"], True)

    def test_an_unreported_flag_is_refused_rather_than_read_as_false(self) -> None:
        with self.assertRaises(AiHubAdapterError) as caught:
            self._observation([{"service_contacted": False}])
        self.assertIn("job_submitted", str(caught.exception))

    def test_a_record_built_without_preflight_says_it_did_not_run(self) -> None:
        observation = self._observation(None)
        self.assertEqual(observation["mode"], "not_run")
        self.assertNotIn("service_contacted", observation)


class CommandLineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "record.json"

    def test_build_then_check_round_trips(self) -> None:
        self.assertEqual(
            workbench.main(["--record", str(self.path)]),
            0,
        )
        self.assertEqual(
            workbench.main(["--record", str(self.path), "--check"]),
            0,
        )

    def test_check_of_the_committed_record_succeeds(self) -> None:
        self.assertEqual(workbench.main(["--check"]), 0)

    def test_check_fails_on_a_missing_record(self) -> None:
        self.assertEqual(
            workbench.main(["--record", str(self.path), "--check"]),
            1,
        )

    def test_preflight_is_refused_in_check_mode(self) -> None:
        self.assertEqual(
            workbench.main(["--record", str(self.path), "--check", "--preflight"]),
            1,
        )


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    unittest.main()
