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
