# Graph risk catalogue

`onnx-risk-rules-v1.json` is the declarative catalogue used by
`slm_lab.graph.inspection` to inspect the T20 reference ONNX graphs for
compiler and deployment risk. The engine holds the detection logic; this
directory holds the rule set, the severity assignments, and the review text
that a human reader is expected to be able to explain.

## What a rule is

Each entry in `rules` names:

- `id`, `title`, and `category`
  (`dynamic_shape`, `operator_support`, `precision`, `memory_traffic`,
  `graph_scale`, or `control_flow`);
- `severity` (`blocking`, `high`, `medium`, `low`, `informational`);
- `detector`, one of the detector names registered in
  `slm_lab.graph.inspection.DETECTORS`, plus its `params`;
- `rationale`, explaining why the pattern is a deployment risk in engineering
  terms, and `mitigation`, describing what a fix or workaround looks like;
- `references`, short human-readable pointers into this repository or to a
  named public specification.

A rule that matches nothing produces no finding. Findings are ranked by
severity, then by descending match count, then by rule id.

## What severity means

Severity answers one question: *how much does this pattern threaten the
deployment path described in the catalogue's `target_context`?* That context is
Qualcomm QNN / Hexagon HTP ahead-of-time compilation first, and ONNX Runtime
CPU/CUDA second. A pattern that merely costs a partition boundary on ONNX
Runtime can still be `blocking` for a static-shape accelerator compiler, and
that asymmetry is the point of ranking by deployment impact rather than by
count.

Severities in this file are **review judgements bound to that stated target
context, not measured compiler results.** No severity here was derived from an
executed compile or conversion job. Where a rule encodes a belief about how a
class of compiler behaves rather than an observation, its `rationale` says so
explicitly. T22 is the task that will produce real compiler evidence; when it
does, the affected rationales and severities should be revised to cite it.

The engine never asserts that a graph will fail to compile. It reports the
structure it observed, with counts and byte totals, and leaves the conclusion
to the reader.

## Adding or changing a rule

1. Pick an existing `detector` if one fits. If none does, add the detector
   function to `slm_lab.graph.inspection`, register it in `DETECTORS`, declare
   any required params in `_REQUIRED_PARAMS`, and add positive and negative
   tests in `tests/onnx/test_graph_inspection.py`.
2. Add the rule object with all nine fields. Ids must be unique; an unknown
   detector, category, or severity makes the whole catalogue fail to load.
3. Write the `rationale` so a learner can explain the risk without reading the
   detector source, and state plainly when the rule is a hypothesis rather than
   a measurement. Do not cite a document you have not read, a compiler version
   you have not run, or a support matrix you cannot back up.
4. Keep `schema_version` at `1` while the field set is unchanged. A change to
   the rule field set is a new schema version and a new catalogue file, so that
   previously committed reports stay interpretable.
5. Regenerate the committed reports and confirm they are stable:

   ```bash
   SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab PYTHONPATH=src \
     python -m slm_lab.graph.inspection --all-manifests results/manifests/onnx

   SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab PYTHONPATH=src \
     python -m slm_lab.graph.inspection --all-manifests results/manifests/onnx \
     --check
   ```

   `SLM_LAB_ARTIFACT_ROOT` locates the `.onnx` graphs, which are not committed.
   Without it the tool falls back to `./artifacts`, which only resolves on a
   host where that symlink exists. `--artifact-root` overrides it.

   Reports are written to `results/graph/<variant>.json`. They contain no
   timestamp, so `--check` is a genuine drift check.

# Graph transformation catalogue

`qnn-transforms-v1.json` is the second catalogue in this directory and it is a
different kind of object from the first. `onnx-risk-rules-v1.json` describes
patterns to *look for*; `qnn-transforms-v1.json`, id `qnn-candidate-v1`,
describes rewrites to *apply*, in order, to turn a T20/T23 reference export into
the T22 `qnn_candidate` stage. The engine is `slm_lab.graph.qnn.transforms` and
the build tool is `slm_lab.graph.qnn.build`.

## What a pass is

Each entry in `passes` names:

- `id` (an `X-` prefixed identifier that the engine must implement), `title`,
  and `order` — the orders must be `1..N` in declaration order;
- `applied`, a boolean. A pass the engine implements must declare `applied:
  true`; a pass declaring `applied: false` must be one the engine deliberately
  does not implement. The catalogue therefore cannot silently disable a real
  pass, and cannot silently enable an unimplemented one;
- `addresses`, the list of `onnx-risk-rules-v1.json` rule ids the pass is meant
  to move. The build tool cross-checks every id against the loaded risk
  catalogue and refuses to run if one is unknown. The list may be empty, and is
  empty for `X-STAMP-CANDIDATE-PROVENANCE`, which addresses no risk rule;
- `observed_issue`, citing the numbered finding in
  `docs/results/onnx/graph-inspection.md` that motivated the pass;
- `transformation`, the exact rewrite, and `parameters`, its declarative
  settings — allowlists, byte budgets, thresholds;
- `rationale` and `references`.

## The recorded rejection

`X-ORT-CPU-OFFLINE-OPTIMIZATION` is in the catalogue with `applied: false`. It
is the obvious way to fold these graphs — write an ONNX Runtime session's
`optimized_model_filepath` at `ORT_ENABLE_BASIC` — and it is rejected on
measured grounds. The build tool re-measures it on every graph it builds and
writes the result into each manifest's `rejection_evidence`, so the rejection
is evidence in the repository rather than an assertion in prose. A catalogue
with no rejected pass makes the build tool fail.

## Adding or changing a pass

1. Implement the pass in `slm_lab.graph.qnn.transforms`, add its id to
   `APPLIED_PASS_IDS`, wire it into `slm_lab.graph.qnn.build`, and add positive
   and negative tests in `tests/qnn/`.
2. Add the pass object with all ten fields and renumber `order` so the orders
   stay contiguous.
3. Cite a numbered finding in `docs/results/onnx/graph-inspection.md` for
   `observed_issue`. Do not claim a compiler outcome; no pass here has been
   through a vendor converter, and the catalogue's `target_context` says so.
4. Keep `schema_version` at `1` while the field set is unchanged.
5. Rebuild the candidates and confirm the committed manifests reproduce:

   ```bash
   SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab PYTHONPATH=src \
     python -m slm_lab.graph.qnn.build --all-manifests

   SLM_LAB_ARTIFACT_ROOT=/Volumes/T9/slm-deployment-lab PYTHONPATH=src \
     python -m slm_lab.graph.qnn.build --all-manifests --check
   ```

   Both need an environment with `onnx`, `onnxruntime`, and `numpy`; the locked
   root environment has none of them, which is why the transform tests skip
   there. `results/manifests/qnn/README.md` documents the manifest schema and
   what `--check` does and does not prove.
