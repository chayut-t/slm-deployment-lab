# Project dashboard

`index.html` is a single-page, self-contained dashboard for the human learner:
current status, completed work, the next ready tasks, a plain-English digest of
every `LEARN-NN` checkpoint, and a glossary. It is aimed at the project's
primary goal — learning SLM deployment with the Qualcomm NPU path first.

## Authored prose, generated numbers

The page is an authored document with two machine-generated regions, delimited
by `<!-- BEGIN GENERATED: <name> -->` / `<!-- END GENERATED: <name> -->`
comments:

| Region | Content | Source |
|---|---|---|
| `status` | Stat tiles and the progress bar | `ai/tasks/task_graph.yaml` |
| `critical-path` | The "road to the NPU" rail with live statuses | `ai/tasks/task_graph.yaml` |

Everything else — task summaries, learning-digest cards, glossary — is authored
prose. The builder never rewrites prose; it cross-checks it instead and fails
with a message telling you what to update by hand:

- task badges in the "What has been done" section must name completed tasks;
- task badges in the "What to do next" section must name exactly the ready
  tasks;
- every checkpoint in `ai/tasks/learning_lane.yaml` must have a digest card and
  a table-of-contents row, and vice versa.

## Commands

```bash
python3 scripts/dashboard/build_dashboard.py          # regenerate the regions
python3 scripts/dashboard/build_dashboard.py --check  # fail if stale or drifted
uv run pytest tests/repo/test_dashboard.py            # regressions
```

Run the builder (and update any prose it complains about) after changing task
statuses or adding a learning checkpoint, in the same change.

## Publishing

The file is an artifact-ready fragment: no `<html>`/`<head>`/`<body>` shell,
inline CSS only, light and dark themes, no external requests. Publish it with
the Claude Code Artifact tool after rebuilding; republish to the existing
artifact URL rather than minting a new one. The published URL is personal
account state and is not committed.
