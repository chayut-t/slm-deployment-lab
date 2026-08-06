"""Contract tests for the committed Qualcomm target selectors.

These tests read only committed JSON. Nothing here contacts a service, imports
``qai_hub``, or asserts anything about what Qualcomm AI Hub would accept. The
point is narrower and enforceable offline: every Qualcomm selector in
``configs/targets/`` must be accepted by the committed T30-backed validators,
must be named consistently, and must state an evidence boundary that matches
the evidence this repository actually holds.

Discovery is scoped to ``qualcomm-*.json`` on purpose. ``configs/targets/`` is
documented as also holding Apple host profiles and NVIDIA runtime targets, and
``load_target_config`` requires ``client.name == "qai-hub"``, so a directory-wide
glob would fail the first non-Qualcomm target committed here.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from slm_lab.deployment.qualcomm import packaging


REPO_ROOT = Path(packaging.__file__).resolve().parents[4]
TARGET_DIRECTORY = REPO_ROOT / "configs" / "targets"

# The two selectors whose device identity comes from the unauthenticated
# public model catalog rather than from an authenticated device query.
CATALOG_ONLY_CONFIGS = (
    "qualcomm-dragonwing-iq-9075-evk",
    "qualcomm-snapdragon-8-elite-qrd",
)
# The device names as recorded in
# results/hosts/public-platform-access-2026-07-24.json.
CATALOG_DEVICE_NAMES = {
    "qualcomm-dragonwing-iq-9075-evk": "Dragonwing IQ-9075 EVK",
    "qualcomm-snapdragon-8-elite-qrd": "Snapdragon 8 Elite QRD",
}
PUBLIC_ACCESS_RECORD = (
    REPO_ROOT / "results" / "hosts" / "public-platform-access-2026-07-24.json"
)
# The authenticated record the client and runtime versions are copied from.
TOY_LIFECYCLE_RECORD = (
    REPO_ROOT / "results" / "hosts" / "workbench-toy-lifecycle-2026-07-25.json"
)


#: Only Qualcomm selectors are in scope; see the module docstring.
QUALCOMM_CONFIG_GLOB = "qualcomm-*.json"


def _target_config_paths() -> list[Path]:
    return sorted(TARGET_DIRECTORY.glob(QUALCOMM_CONFIG_GLOB))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class TargetConfigDiscoveryTest(unittest.TestCase):
    """Every committed selector must satisfy the shared contract."""

    def test_directory_holds_the_three_qualcomm_targets(self) -> None:
        stems = {path.stem for path in _target_config_paths()}
        self.assertIn("qualcomm-snapdragon-x-elite-crd", stems)
        for config_id in CATALOG_ONLY_CONFIGS:
            self.assertIn(config_id, stems)

    def test_every_config_is_accepted_and_named_after_its_file(self) -> None:
        paths = _target_config_paths()
        self.assertTrue(paths, "configs/targets/ holds no Qualcomm selectors")
        for path in paths:
            with self.subTest(target=path.name):
                config = packaging.load_target_config(path)
                self.assertEqual(config["config_id"], path.stem)

    def test_every_config_normalizes_through_the_t30_validators(self) -> None:
        for path in _target_config_paths():
            with self.subTest(target=path.name):
                normalized = packaging.normalize_target(_load(path))
                self.assertEqual(normalized["runtime"]["name"], "QAIRT")
                self.assertTrue(normalized["device"]["name"])
                self.assertIn("--qairt_version", normalized["options"])

    def test_every_config_declares_its_evidence_and_boundary(self) -> None:
        for path in _target_config_paths():
            with self.subTest(target=path.name):
                config = _load(path)
                self.assertIsInstance(config.get("evidence"), dict)
                self.assertTrue(config["evidence"])
                notes = config.get("notes")
                self.assertIsInstance(notes, list)
                self.assertTrue(notes)
                boundary = config.get("claim_boundary")
                self.assertIsInstance(boundary, dict)
                self.assertTrue(boundary.get("establishes"))
                self.assertTrue(boundary.get("does_not_establish"))


class CatalogOnlyBoundaryTest(unittest.TestCase):
    """The two T31 selectors must not overstate their weaker evidence."""

    def _config(self, config_id: str) -> dict[str, Any]:
        return _load(TARGET_DIRECTORY / f"{config_id}.json")

    def test_device_names_are_copied_verbatim_from_the_committed_catalog(self) -> None:
        record = _load(PUBLIC_ACCESS_RECORD)
        listed = record["workbench"]["public_qwen_targets"]
        for config_id, device_name in CATALOG_DEVICE_NAMES.items():
            with self.subTest(target=config_id):
                self.assertIn(device_name, listed)
                self.assertEqual(self._config(config_id)["device"]["name"], device_name)

    def test_client_and_runtime_versions_come_from_the_authenticated_record(
        self,
    ) -> None:
        """The one place these selectors may copy authenticated evidence."""

        record = _load(TOY_LIFECYCLE_RECORD)
        client_version = record["workbench"]["local_client"]["qai_hub_version"]
        qairt_version = record["workbench"]["hosted_frameworks"]["qairt_default"]
        for config_id in CATALOG_ONLY_CONFIGS:
            with self.subTest(target=config_id):
                config = self._config(config_id)
                self.assertEqual(config["client"]["version"], client_version)
                self.assertEqual(config["runtime"]["version"], qairt_version)
                self.assertIn(qairt_version, config["compile"]["options"])
                self.assertEqual(
                    config["evidence"]["client_and_runtime_machine_readable"],
                    "results/hosts/workbench-toy-lifecycle-2026-07-25.json",
                )

    def test_evidence_names_the_unauthenticated_catalog_as_the_device_source(
        self,
    ) -> None:
        for config_id in CATALOG_ONLY_CONFIGS:
            with self.subTest(target=config_id):
                evidence = self._config(config_id)["evidence"]
                self.assertEqual(
                    evidence["device_name_source"],
                    "unauthenticated_public_model_catalog",
                )
                self.assertTrue(evidence["device_confirmation"].startswith("none"))
                self.assertEqual(
                    evidence["device_name_machine_readable"],
                    "results/hosts/public-platform-access-2026-07-24.json",
                )

    def test_no_authenticated_device_confirmation_is_claimed(self) -> None:
        for config_id in CATALOG_ONLY_CONFIGS:
            with self.subTest(target=config_id):
                boundary = self._config(config_id)["claim_boundary"]
                self.assertNotIn(
                    "device_and_runtime_identity_match_committed_T02_evidence",
                    boundary["establishes"],
                )
                for claim in boundary["establishes"]:
                    self.assertNotIn("authenticated_device", claim)
                self.assertIn(
                    "an_authenticated_device_query_confirmed_this_selector_resolves",
                    boundary["does_not_establish"],
                )
                self.assertIn(
                    "this_account_can_reach_or_schedule_this_device",
                    boundary["does_not_establish"],
                )

    def test_no_device_attribute_is_invented(self) -> None:
        """No os, chipset, Hexagon, or SoC value exists in committed evidence."""

        for config_id in CATALOG_ONLY_CONFIGS:
            with self.subTest(target=config_id):
                device = self._config(config_id)["device"]
                self.assertNotIn("os", device)
                self.assertEqual(device["attributes"], [])

    def test_x_elite_keeps_its_stronger_authenticated_claim(self) -> None:
        """Guard against normalizing the three boundaries into one wording."""

        config = self._config("qualcomm-snapdragon-x-elite-crd")
        self.assertIn(
            "device_and_runtime_identity_match_committed_T02_evidence",
            config["claim_boundary"]["establishes"],
        )
        self.assertEqual(config["device"]["os"], "11")


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    unittest.main()
