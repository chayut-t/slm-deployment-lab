# T21: ONNX Runtime CPU parity and graph inspection

Status: completed
Owner: Claude t21-main agent
Updated: 2026-08-02

## Objective

Validate multi-step ONNX Runtime CPU behaviour of the T20 reference graphs
against the T11 deterministic PyTorch reference, and inspect the exported
graphs for compiler and deployment risk before T22 attempts a QNN candidate.

Two failure modes must be told apart, because they have different fixes:

- a **numerical-tolerance failure** — the graph is wired correctly but FP16
  and a different backend move the logits beyond an explicit tolerance;
- a **state-update failure** — the decode graph writes the wrong cache slot,
  loses the prefix, or reports the wrong `valid_length`. Step one can look
  perfect while step four is wrong.

## Scope

### In scope

- A dependency-free ONNX structural reader that parses a real exported graph
  without `onnx`, `protobuf`, or `numpy`.
- A declarative deployment-risk catalogue and an inspection engine that ranks
  findings by deployment impact.
- An ONNX Runtime CPU parity runner with injected session and reference
  factories, explicit tolerances, and multi-step static-cache state checks.
- Committed small evidence under `results/graph/` and reader-facing reports
  under `docs/results/onnx/`.

### Out of scope

- Any compiler, accelerator-placement, quantization, or performance claim.
- Graph rewrites or QNN candidate production (T22).
- Installing heavy dependencies in this environment.

## Dependencies and resources

- Required task dependencies: T20 (completed), which supplies the eight
  reference graphs and four committed manifests. T11/T12 supply the reference
  generation loop and the frozen cache contract.
- Resource locks: none. Graph reads on the external artifact root are
  read-only and small (`.onnx` headers only, never `.onnx.data`).
- External access: none. No paid service, no network.
- Cost boundary: zero.

## Important paths

- Inputs: `results/manifests/onnx/S*.json`,
  `${SLM_LAB_ARTIFACT_ROOT}/onnx/reference/T20/S*/{prefill,decode}.onnx`,
  `src/slm_lab/contracts/static_cache.py`,
  `src/slm_lab/generation/reference.py`.
- Outputs: `src/slm_lab/backends/onnx_cpu.py`, `src/slm_lab/graph/`,
  `tests/onnx/`, `configs/graph/onnx-risk-rules-v1.json`,
  `results/graph/S*.json`, `docs/results/onnx/*.md`.
- Shared contracts: `ai/tasks/task_graph.yaml` (T21 block only),
  `configs/learning/checkpoints.yaml` (`LEARN-10` only). `pyproject.toml` is
  **not** touched: an ONNX Runtime extra was considered and rejected, because
  `uv.lock` records `provides-extras = ["dev", "tokenizer"]` and every
  documented setup uses `uv sync --locked`, so adding an extra offline would
  invalidate the lock for every host including those that will never run a
  parity job. `environments/onnx-cpu/README.md` describes a separate virtual
  environment instead, following the T50/MLX precedent.

## Milestones

- [x] Dependency-free ONNX reader with hand-built protobuf fixtures.
- [x] Declarative risk catalogue plus inspection engine and CLI, provenance
      bound to the committed T20 graph digests.
- [x] ORT CPU parity runner: prefill, N decode steps, tolerance classification
      and static-cache state validation, all exercised with injected fakes.
- [x] Real inspection run over all eight T20 graphs, committed as compact JSON.
- [x] Two reader-facing reports and a `LEARN-10` study checkpoint.

## Verification and acceptance

- Commands:
  - `PYTHONPATH=src python -m pytest tests -q`
  - `PYTHONPATH=src python -m ruff format --check <paths>` and `ruff check`
  - `python scripts/ai/render_task_status.py --check`
  - `python scripts/repo/check_hygiene.py --all`
- Numerical or behavioural criteria:
  - Explicit, written-down tolerances for the ORT-CPU-versus-PyTorch
    comparison, distinct from the T11 same-model tolerance, with the reason
    each threshold was chosen.
  - Multi-step decode preserves the valid prefix, writes exactly one new cache
    slot per step, leaves the tail zero, and increments `valid_length`.
  - A cache-state fault is reported as a state fault even when logits agree,
    and a logit fault is reported as a tolerance fault even when the cache is
    perfect.
- Hardware/profile evidence: none claimed. No ONNX Runtime exists in this
  environment, so parity numbers are **not measured here**; the report records
  the exact command that produces them on a host that has the runtime.

## Artifact and privacy handling

- Committed evidence: risk-rule catalogue, per-variant inspection JSON
  (summaries and counts only, no node dumps), and the two reports.
- External artifacts: the eight `.onnx` graphs and their eight 1.19 GB
  external-data sidecars (one per graph, byte-identical, so one SHA-256 but
  about 8.9 GB of storage) stay under `SLM_LAB_ARTIFACT_ROOT`, referenced by the
  digests already committed in the T20 manifests.
- Private/local material: none.

## Decisions and discoveries

- 2026-08-02: The T20 `.onnx` files are small (1.5–35 MB) because every weight
  is external. A pure-Python protobuf walk over them yields a **real**
  operator inventory, so the graph inspection report is measured evidence
  rather than a prediction, even without `onnx` installed.
- 2026-08-02: Graph inspection verifies the SHA-256 recorded in the committed
  T20 manifest before parsing, so a report is always bound to a named graph.
- 2026-08-02: Parity comparison is implemented in pure Python over the array
  protocol so the runner is testable without `numpy`; real ORT outputs are
  accepted through `.tolist()`.
- 2026-08-02: ORT parity tolerances are proposed and justified but explicitly
  unvalidated until a host with the runtime executes the recorded command.

## Progress and restart instructions

Implementation, review rounds, and evidence generation are complete on
`task/T21-ort-cpu-parity`. The one thing this environment cannot produce is a
real ORT CPU parity measurement; `docs/results/onnx/ort-cpu-parity.md` states
that plainly and records the command. There is no `onnx-cpu` extra to install —
that was rejected for the lock reason above. A machine that has built the
separate parity environment from `environments/onnx-cpu/README.md` and has the
T9 artifact root should run that command and commit the resulting
`results/graph/parity/` record.
