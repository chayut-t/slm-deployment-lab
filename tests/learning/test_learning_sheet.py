"""The learning sheets must mirror the repository, not paraphrase it.

These tests hold three properties:

1. every checkpoint cites files that exist and task-graph nodes that are
   completed;
2. every completed task is covered by exactly one checkpoint, so finishing a
   task without adding it to a checkpoint fails here rather than silently;
3. every token of every cited document survives into the rendered page, which
   is what makes "mirrored verbatim" a checked claim instead of a promise.
"""

from __future__ import annotations

import html
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("markdown", reason="the learning sheet needs the dev extra")

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER = REPO_ROOT / "scripts" / "learning" / "build_learning_sheet.py"
CONFIG = REPO_ROOT / "configs" / "learning" / "checkpoints.yaml"
TASK_GRAPH = REPO_ROOT / "ai" / "tasks" / "task_graph.yaml"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_learning_sheet", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so the module's dataclasses can resolve their
    # own module namespace.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = load_builder()
CONFIG_DATA = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
CHECKPOINTS = CONFIG_DATA["checkpoints"]
TASKS = yaml.safe_load(TASK_GRAPH.read_text(encoding="utf-8"))["tasks"]
IDS = [checkpoint["id"] for checkpoint in CHECKPOINTS]


# -- configuration integrity ----------------------------------------------


def test_checkpoint_ids_are_unique_and_ordered():
    assert IDS == sorted(IDS)
    assert len(set(IDS)) == len(IDS)
    for checkpoint_id in IDS:
        assert re.fullmatch(r"LEARN-\d{2}", checkpoint_id)


@pytest.mark.parametrize("checkpoint", CHECKPOINTS, ids=IDS)
def test_checkpoint_shape(checkpoint):
    required_fields = (
        "title",
        "subject",
        "lede",
        "tasks",
        "attention",
        "outcomes",
        "readings",
        "labs",
        "questions",
        "boundaries",
    )
    for field in required_fields:
        assert checkpoint.get(field), f"{checkpoint['id']} is missing {field}"

    keys = [reading["key"] for reading in checkpoint["readings"]]
    assert len(set(keys)) == len(keys), f"{checkpoint['id']} repeats a reading key"
    assert any(reading.get("required") for reading in checkpoint["readings"]), (
        f"{checkpoint['id']} has no required reading"
    )


@pytest.mark.parametrize("checkpoint", CHECKPOINTS, ids=IDS)
def test_cited_sources_exist_and_slices_resolve(checkpoint):
    for spec in checkpoint["readings"]:
        reading = builder.load_reading(spec)
        assert reading.text.strip(), f"{spec['path']} mirrored as empty text"


def test_every_completed_task_belongs_to_exactly_one_checkpoint():
    completed = {task["id"] for task in TASKS if task["status"] == "completed"}
    covered: dict[str, str] = {}
    for checkpoint in CHECKPOINTS:
        for task_id in checkpoint["tasks"]:
            assert task_id not in covered, (
                f"{task_id} is claimed by both {covered[task_id]} and "
                f"{checkpoint['id']}"
            )
            covered[task_id] = checkpoint["id"]

    missing = sorted(completed - set(covered))
    assert not missing, (
        "completed tasks without a learning checkpoint: "
        f"{', '.join(missing)}. Add them to configs/learning/checkpoints.yaml."
    )

    statuses = {task["id"]: task["status"] for task in TASKS}
    for task_id, checkpoint_id in covered.items():
        assert statuses.get(task_id) == "completed", (
            f"{checkpoint_id} covers {task_id}, which is "
            f"{statuses.get(task_id)!r} rather than completed"
        )


# -- the generated learning lane ------------------------------------------


def load_renderer():
    path = REPO_ROOT / "scripts" / "ai" / "render_task_status.py"
    spec = importlib.util.spec_from_file_location("render_task_status_learning", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


renderer = load_renderer()


def test_recorded_lane_matches_the_configuration():
    drift, _ = builder.lane_report(CONFIG_DATA, builder.load_lane())
    assert not drift, (
        "the learning lane no longer matches the config; run "
        "scripts/learning/build_learning_sheet.py --all --record"
    )


def test_lane_drift_reports_a_reading_removed_from_the_config():
    trimmed = {
        **CONFIG_DATA,
        "checkpoints": [
            {**checkpoint, "readings": checkpoint["readings"][:-1]}
            if checkpoint["id"] == "LEARN-00"
            else checkpoint
            for checkpoint in CHECKPOINTS
        ],
    }
    dropped = CHECKPOINTS[0]["readings"][-1]["path"]
    drift, _ = builder.lane_report(trimmed, builder.load_lane())
    assert any(f"-{dropped}" in note for note in drift), drift


def test_record_refuses_a_custom_output_directory(tmp_path, capsys):
    before = builder.LANE_PATH.read_bytes()
    code = builder.main(["--all", "--record", "--output-dir", str(tmp_path / "sheets")])
    assert code == 2
    assert "--record" in capsys.readouterr().err
    assert builder.LANE_PATH.read_bytes() == before


def test_writing_outside_the_repository_reports_an_absolute_path(tmp_path, capsys):
    assert builder.main(["LEARN-00", "--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / "learn-00.html").is_file()
    assert str(tmp_path) in capsys.readouterr().out


def test_renderer_places_every_checkpoint_over_its_tasks():
    lane = renderer.load_learning(renderer.working_text, TASKS)
    assert [entry["id"] for entry in lane] == IDS
    covers = {entry["id"]: entry["covers"] for entry in lane}
    for checkpoint in CHECKPOINTS:
        assert covers[checkpoint["id"]] == checkpoint["tasks"]

    status = renderer.render(
        {"resource_locks": {}},
        TASKS,
        lane,
    )
    assert "## Learning checkpoints" in status
    for checkpoint in CHECKPOINTS:
        node = checkpoint["id"].replace("-", "_")
        for task_id in checkpoint["tasks"]:
            assert f"    {task_id} --> {node}" in status


def test_lane_rejects_a_checkpoint_over_unfinished_work():
    lane = builder.load_lane()
    entry = lane["LEARN-03"]
    downgraded = [
        {**task, "status": "planned"} if task["id"] == entry["covers"][0] else task
        for task in TASKS
    ]

    def read_text(path: str) -> str:
        if path == renderer.LANE_PATH:
            return json.dumps({"checkpoints": [entry]})
        return renderer.working_text(path)

    with pytest.raises(ValueError, match="covers T11"):
        renderer.load_learning(read_text, downgraded)


def test_lane_rejects_an_uncovered_completed_task():
    entry = builder.load_lane()["LEARN-03"]

    def read_text(path: str) -> str:
        if path == renderer.LANE_PATH:
            return json.dumps({"checkpoints": [entry]})
        return renderer.working_text(path)

    with pytest.raises(ValueError, match="without a learning checkpoint"):
        renderer.load_learning(read_text, TASKS)


# -- rendering ------------------------------------------------------------


def rendered(checkpoint) -> str:
    tasks = {task["id"]: task for task in TASKS}
    return builder.build(checkpoint, CONFIG_DATA, tasks)


@pytest.mark.parametrize("checkpoint", CHECKPOINTS, ids=IDS)
def test_page_is_complete(checkpoint):
    page = rendered(checkpoint)
    assert not re.search(r"<!--[A-Z]+-->", page), "an unreplaced placeholder remains"
    assert f"<title>{checkpoint['id']}" in page
    assert page.count("<details") == len(checkpoint["readings"])
    assert page.count("</details>") == len(checkpoint["readings"])
    for task_id in checkpoint["tasks"]:
        assert task_id in page
    # The sheet must keep saying what it is not.
    assert "study surface, not a gate" in page


TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:/-]{2,}")
LINK_TARGET = re.compile(r"\]\([^)]*\)")
FENCE_INFO = re.compile(r"^(\s*```+)[A-Za-z0-9_+-]*\s*$", re.MULTILINE)
TAG = re.compile(r"<[^>]+>")


def visible_text(page: str) -> str:
    body = page.split("</style>", 1)[-1].split("<script>", 1)[0]
    return html.unescape(TAG.sub(" ", body))


@pytest.mark.parametrize("checkpoint", CHECKPOINTS, ids=IDS)
def test_every_source_token_survives(checkpoint):
    page = visible_text(rendered(checkpoint))
    for spec in checkpoint["readings"]:
        reading = builder.load_reading(spec)
        # Two things are markup rather than prose: link targets, which are
        # rendered as inert references, and fence language hints, which become
        # a class attribute on the code block.
        source = FENCE_INFO.sub(r"\1", LINK_TARGET.sub(" ", reading.text))
        missing = sorted(
            {token for token in TOKEN.findall(source) if token not in page}
        )
        assert not missing, (
            f"{checkpoint['id']} / {spec['path']} lost "
            f"{len(missing)} token(s): {missing[:12]}"
        )
