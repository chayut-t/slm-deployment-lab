"""Regression tests for the project dashboard builder."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = REPO_ROOT / "scripts" / "dashboard" / "build_dashboard.py"
DASHBOARD_PATH = REPO_ROOT / "docs" / "dashboard" / "index.html"
GRAPH_PATH = REPO_ROOT / "ai" / "tasks" / "task_graph.yaml"
LANE_PATH = REPO_ROOT / "ai" / "tasks" / "learning_lane.yaml"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_dashboard", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DashboardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = load_builder()
        cls.dashboard_text = DASHBOARD_PATH.read_text(encoding="utf-8")
        cls.graph_text = GRAPH_PATH.read_text(encoding="utf-8")
        cls.lane_text = LANE_PATH.read_text(encoding="utf-8")

    def test_committed_dashboard_is_current(self) -> None:
        rendered, errors = self.builder.build(
            self.dashboard_text, self.graph_text, self.lane_text
        )
        self.assertEqual(errors, [])
        self.assertEqual(
            rendered,
            self.dashboard_text,
            "docs/dashboard/index.html is stale; "
            "run scripts/dashboard/build_dashboard.py",
        )

    def test_build_is_idempotent(self) -> None:
        rendered, _ = self.builder.build(
            self.dashboard_text, self.graph_text, self.lane_text
        )
        rendered_again, errors = self.builder.build(
            rendered, self.graph_text, self.lane_text
        )
        self.assertEqual(errors, [])
        self.assertEqual(rendered_again, rendered)

    def test_tampered_region_is_regenerated(self) -> None:
        tampered = self.dashboard_text.replace(
            "<!-- BEGIN GENERATED: status -->",
            "<!-- BEGIN GENERATED: status -->\n    <p>tampered</p>",
            1,
        )
        rendered, _ = self.builder.build(tampered, self.graph_text, self.lane_text)
        self.assertNotIn("tampered", rendered)
        self.assertEqual(rendered, self.dashboard_text)

    def test_missing_region_is_rejected(self) -> None:
        broken = self.dashboard_text.replace(
            "GENERATED: critical-path", "GENERATED: gone"
        )
        with self.assertRaises(ValueError):
            self.builder.build(broken, self.graph_text, self.lane_text)

    def test_status_drift_is_reported(self) -> None:
        graph = json.loads(self.graph_text)
        done_in_prose = self.builder.CRITICAL_PATH[0][0]
        for task in graph["tasks"]:
            if task["id"] == done_in_prose:
                task["status"] = "planned"
        _, errors = self.builder.build(
            self.dashboard_text, json.dumps(graph), self.lane_text
        )
        self.assertTrue(
            any(done_in_prose in message for message in errors),
            f"expected a drift error naming {done_in_prose}, got: {errors}",
        )

    def test_unlisted_ready_task_is_reported(self) -> None:
        graph = json.loads(self.graph_text)
        completed = next(
            task["id"]
            for task in graph["tasks"]
            if task["status"] == "completed" and not task["depends_on"]
        )
        for task in graph["tasks"]:
            if task["id"] == completed:
                task["status"] = "planned"
                task["worklog"] = None
        _, errors = self.builder.build(
            self.dashboard_text, json.dumps(graph), self.lane_text
        )
        self.assertTrue(
            any("missing from the next section" in message for message in errors),
            f"expected a missing-ready error, got: {errors}",
        )

    def test_uncarded_checkpoint_is_reported(self) -> None:
        lane = json.loads(self.lane_text)
        completed_task = lane["checkpoints"][0]["covers"][0]
        lane["checkpoints"].append(
            {
                "id": "LEARN-98",
                "title": "Synthetic",
                "subject": "Synthetic",
                "covers": [completed_task],
                "sheet": "build/learning/learn-98.html",
                "built": "2026-08-02",
                "sources": {},
            }
        )
        _, errors = self.builder.build(
            self.dashboard_text, self.graph_text, json.dumps(lane)
        )
        self.assertTrue(
            any("LEARN-98" in message for message in errors),
            f"expected an uncarded-checkpoint error, got: {errors}",
        )


if __name__ == "__main__":
    unittest.main()
