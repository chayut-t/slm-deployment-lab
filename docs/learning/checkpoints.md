# Learning checkpoints

A learning checkpoint (`LEARN-NN`) is one study unit built from completed
task-graph work. It exists so the subject can be studied directly instead of
being reassembled from several task definitions, worklogs, and result reports.

A checkpoint is **not a gate**. `docs/project/plan.md` section 8.1 rules out
mandatory learner approval, so nothing here blocks implementation. The
per-task attention levels and hands-on requirements remain in
[`docs/project/learning-checkpoints.md`](../project/learning-checkpoints.md);
this page is the study surface those reminders point at.

## The series

The checkpoints below cover every completed task. Every completed task
belongs to exactly one of them, and
`tests/learning/test_learning_sheet.py` fails when a newly completed task has
no checkpoint.

| Checkpoint | Subject | Tasks | Attention |
|---|---|---|---|
| `LEARN-00` | The evidence contract | T00, T01 | Deep study |
| `LEARN-01` | Agentic delivery | T03, T04 | Review |
| `LEARN-02` | Fixtures as a contract | T10 | Deep study |
| `LEARN-03` | Static graphs and the KV-cache contract | T11, T12 | Deep study |
| `LEARN-04` | Benchmarking without false equivalence | T13 | Deep study |
| `LEARN-05` | ONNX export as a compiler contract | T20 | Deep study |
| `LEARN-06` | The Qualcomm public pipeline | T02, T30 | Hands-on and deep study |
| `LEARN-07` | A real generation loop on a real device | T32 | Hands-on and deep study |
| `LEARN-08` | Apple Silicon and MLX | T50, T51 | Deep study |
| `LEARN-09` | CI as a deployment surface | T72 | Hands-on and deep study |
| `LEARN-10` | Reading a graph before a compiler does | T21 | Deep study |

Tasks that are still `planned` have no checkpoint. Add one when the task
completes, in the same change that flips its status.

## What a sheet contains

1. **Covered work** — the completed task nodes and their worklogs.
2. **Outcomes** — what you should be able to explain without the sheet open.
3. **Readings** — the cited documents mirrored *verbatim*, each marked
   required or supplementary, with a note on why it is worth your attention.
4. **Lab** — commands you can run now, plus the Jupyter labs that own the
   numerical experiments for the subject. Notebooks stay the experiment
   surface; the sheet is the reading surface.
5. **Self-check** — questions that are meant to be difficult.
6. **Boundary** — what the evidence behind this checkpoint does not license
   you to claim.
7. **Provenance** — path, line count, and SHA-256 prefix of every mirrored
   source.

The Markdown files are the source of truth. The sheet is a view of them: the
build converts and styles, and never rewrites prose. That property is enforced
by tokenizing every source document and asserting each token survives into the
rendered page.

## In the dependency graph

Checkpoints are **terminal nodes**: they depend on completed tasks, and no task
depends on them. That is what keeps them from becoming gates while still making
the dependency explicit.

`configs/learning/checkpoints.yaml` holds the study content — reading lists,
labs, questions — which does not belong in a dependency graph. The graph-facing
projection of it is `ai/tasks/learning_lane.yaml`: checkpoint ID, subject,
covered tasks, sheet path, build date, and the digest of every mirrored source
at the last build. It is JSON-compatible so `scripts/ai/render_task_status.py`
can validate it with no third-party imports and read identical bytes from a
staged snapshot.

`ai/tasks/status.generated.md` therefore renders the learning lane alongside the
task DAG, and names any sheet whose sources have changed since it was last
built. Rebuilding is a report, never a gate.

## Building a sheet

The generator needs `markdown`, which arrives with the `dev` extra.

```bash
uv sync --extra dev
uv run python scripts/learning/build_learning_sheet.py --all
uv run python scripts/learning/build_learning_sheet.py LEARN-03
uv run python scripts/learning/build_learning_sheet.py --check
```

After republishing a sheet, record it so the lane and task status agree:

```bash
uv run python scripts/learning/build_learning_sheet.py LEARN-03 --record
uv run python scripts/ai/render_task_status.py
```

Output goes to `build/learning/learn-NN.html`, which is generated, ignored by
Git, and self-contained. Each sheet is published as a private Claude Code
artifact; re-run the build and republish the same artifact after editing any
cited document. The published URLs are personal account state, so they are
recorded locally in `.ai-local/learning/published-sheets.md` rather than
committed.

## Adding a checkpoint

Checkpoints are declared in
[`configs/learning/checkpoints.yaml`](../../configs/learning/checkpoints.yaml).
Add an entry with the covered tasks, readings, labs, questions, and
boundaries, then:

```bash
uv run python scripts/learning/build_learning_sheet.py --all --record
uv run python scripts/ai/render_task_status.py
uv run pytest tests/learning/test_learning_sheet.py
```

The renderer refuses a checkpoint over a task that is not `completed`, a task
covered by two checkpoints, and a completed task left uncovered.

Prefer a committed guide, ADR, architecture document, or result report as the
required reading. Use a worklog only where no guide exists yet — `LEARN-05`
and `LEARN-08` do, because T80 has not yet written
`onnx_for_hardware_compilers.md` or `apple_m4_mlx_runtime.md`. When those
guides land, promote them and demote the worklogs to supplementary.
