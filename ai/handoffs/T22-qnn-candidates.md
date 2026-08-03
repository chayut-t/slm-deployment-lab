# T22 handoff: integrating the QNN candidate stage

Date: 2026-08-03
From: T22
To: whoever merges `task/T22-qnn-candidates`, and then T31

## Why this handoff exists

T22's engineering is complete and its evidence is committed, but the branch is
**not merged**. Merging was not authorized and was not performed, so the task
graph reads `in_progress`, and two repository invariants deliberately block the
rest of the close-out until that changes:

- `scripts/ai/render_task_status.py` raises
  `only completed tasks may set the worklog field`, so `T22.worklog` is `null`
  even though `ai/worklogs/2026-08-03-T22-qnn-candidates-and-packaging.md`
  exists and is committed.
- `configs/learning/checkpoints.yaml` states in its own header that a
  checkpoint must "cover only tasks whose task-graph status is `completed`",
  and `scripts/learning/build_learning_sheet.py` enforces it, failing with
  `LEARN-12 cites T22, whose status is 'in_progress'; checkpoints cover
  completed work only`. That was verified by adding the entry and running the
  build, not inferred from the code.

Neither is a defect. Both are the mechanism that keeps `completed` meaning what
AGENTS.md says it means. This file carries the two things the close-out needs
so they do not have to be re-derived.

## The close-out sequence

Run these in order, on the integration branch, after the merge.

1. In `ai/tasks/task_graph.yaml`, set the `T22` node's `status` to `completed`
   and its `worklog` to
   `ai/worklogs/2026-08-03-T22-qnn-candidates-and-packaging.md`. Both in the
   same edit: the validator refuses a `worklog` on any other status.
2. In `ai/worklogs/2026-08-03-T22-qnn-candidates-and-packaging.md`, change the
   `Status:` line to exactly `Status: completed`. The validator matches
   `^Status:\s*completed\s*$` and will refuse the current descriptive line once
   the task is `completed`. Add a short paragraph to the "Task status" section
   recording the merge commit, as `T23`'s log does.
3. Append the `LEARN-12` block below to `configs/learning/checkpoints.yaml`.
4. Move `ai/plans/active/T22-qnn-candidates.md` to `ai/plans/completed/`.
5. Regenerate and verify:

   ```bash
   python3 scripts/learning/build_learning_sheet.py --all --record
   python3 scripts/ai/render_task_status.py
   python3 scripts/ai/render_task_status.py --check
   python3 scripts/repo/check_hygiene.py --all
   PYTHONPATH=src python -m pytest tests -q
   ```

   `render_task_status.py --check` and the three
   `tests/repo/test_task_automation.py` snapshot cases both fail on a stale
   `ai/tasks/status.generated.md`, so step 5's second command is not optional.

`T31` unblocks at step 1.

## What T31 consumes, and what it must not assume

Stable inputs, all committed:

- `results/manifests/qnn/S<context>.json` — the candidate manifest per variant.
  `artifacts.root` is `${SLM_LAB_ARTIFACT_ROOT}/onnx/qnn-candidate/T22`, and
  `artifacts.<kind>` carries `relative_path`, `sha256`, `size_bytes`,
  `external_data`, `input_tensors` and `output_tensors`. The block is
  deliberately the same shape as the T20 manifest's, so
  `slm_lab.graph.inspection` and `slm_lab.backends.onnx_cpu` both read it.
- `results/manifests/qnn/packages/S<context>.json` — the package record, with
  logical names, digests, byte sizes, input specs, the target selector
  identity, the option string, and a deterministic `request_id` per graph. It
  is path-free by design.
- `configs/targets/qualcomm-snapdragon-x-elite-crd.json` — the compile target
  selector. Every field is copied from committed T02 evidence; the `notes`
  block records what was deliberately *not* copied and why.

Rebuild a package and regenerate its request with
`scripts/qualcomm/package_qnn_candidate.py --manifest <path>`; re-verify one
with `--check`. Generated requests carry machine-local paths and stay under
`.ai-local/profiles/T22/`.

Three things T22 did **not** establish, and that T31 is the first task able to
test:

- **AI Hub acceptance of the package layout.** The layout for an external-data
  ONNX model is unverified against the service. The compile request names only
  the `.onnx` file, because the committed T30 adapter requires
  `source_artifact.path` to be one existing file; whether the service reads the
  `.onnx.data` sidecar from the same directory, or wants a directory or an
  archive, is unknown. Expect this to be the first thing that breaks.
- **Compiler acceptance of any operator.** No converter, no QAIRT tool, no
  device. `qnn_candidate` is a stage name and an intent, not a verdict.
- **Anything about placement, latency, or memory.** No performance number of
  any kind exists for these graphs.

Two candidate-specific facts worth knowing before the first submission:

- The decode candidates carry **1,069 interior tensors whose shape ONNX
  shape inference could not resolve** (9 in prefill at S512 and above, 0 at
  S128 prefill). That is the first real measurement of something the reference
  graphs never declared, and a static-shape ahead-of-time compiler is exactly
  the consumer that will care. Section 6.1 of
  `docs/results/qualcomm/qnn-candidates.md` has the sampled locations.
- The candidates are **bit-identical to the reference** on the ONNX Runtime CPU
  provider, so any numerical divergence T31 observes on a device is
  attributable to the compiler, the runtime, or the hardware — not to the T22
  rewrites. That is the main reason the parity measurement was taken.

## The `LEARN-12` entry, ready to paste

Append verbatim to the end of `configs/learning/checkpoints.yaml`, after
`LEARN-11`. It was validated by `yaml.safe_load` and by
`scripts/learning/build_learning_sheet.py`, which accepted everything except
the `in_progress` status of `T22`.

```yaml
  - id: LEARN-12
    title: Rewriting a graph for a compiler you have not run
    subject: What a transformation catalogue can prove, and where the proof stops
    lede: >-
      T21 ranked what is wrong with the eight reference graphs. T22 rewrites
      them and measures what changed. Six declarative passes turn 7,634 prefill
      nodes into 2,785, collapse a 9.70x protobuf spread to 1.01x, and take the
      rank-1 shape finding from 804 to at most 6 — while the boundary, the 56
      cache writes, the 285 precision crossings and the 367 sensitive operators
      stay exactly where the export put them. Then the candidates are executed,
      and every logit comes back bit-identical to the reference. The two
      instructive parts are the transformation this catalogue *refused*, on
      four measured grounds, and the follow-up that closed a T21 evidence
      boundary and closed it unfavourably.
    tasks: [T22]
    attention: Deep study
    outcomes:
      - Explain why a rewrite catalogue must be declarative, ordered, and
        target-neutral, and what each of those three words rules out.
      - Read a pass ordering as a dependency argument, naming which passes
        cannot work until an earlier one has run, and why.
      - Distinguish a smaller finding count from a more convertible graph, and
        say what evidence would be needed for the second.
      - Argue why bit-identical logits are the expected result of a
        semantics-preserving catalogue, and what a one-ULP difference would
        have meant.
      - Judge an optimizer by the objective it was built for rather than by the
        node count it reports, and name the artifacts a host-specific optimizer
        leaves behind.
      - Say precisely what "ready for submission" is allowed to mean before any
        service has been contacted.
    readings:
      - key: report
        label: QNN candidate report
        title: "The T22 QNN candidate graphs: what six rewrites did"
        path: docs/results/qualcomm/qnn-candidates.md
        required: true
        why: >-
          The measured report: provenance, the six passes with one graph walked
          end to end, the before/after matrix, the rejected pass with its four
          grounds, the parity result, and an explicit list of what none of it
          licenses. Read section 1 and section 10 before the tables.
      - key: catalogue
        label: The transformation catalogue
        title: "qnn-candidate-v1"
        path: configs/graph/qnn-transforms-v1.json
        required: true
        why: >-
          The artifact the build tool actually consumes. Read `target_context`
          first, then each pass's `observed_issue` before its `transformation`
          - the order is deliberate, because a pass with no cited finding is a
          preference rather than a fix. The rejected pass is the last entry.
      - key: manifests
        label: Candidate manifests
        title: "T22 QNN candidate manifests"
        path: results/manifests/qnn/README.md
        required: true
        why: >-
          What each committed manifest holds, what the build tool does in
          order, and what `--check` proves and does not. The section on the
          rejected transformation records a number that disagreed with the
          planning probe, and says the cause is not known rather than guessing
          one.
      - key: packaging
        label: Packaging and preflight
        title: "QNN candidate packaging and the offline compile preflight"
        path: src/slm_lab/deployment/README.md
        required: false
        why: >-
          The package layout, why one directory per graph kind is a
          requirement rather than tidiness, and the three things "ready for
          submission" is allowed to mean.
      - key: worklog
        label: T22 worklog
        title: "T22: QNN candidates and packaging"
        path: ai/worklogs/2026-08-03-T22-qnn-candidates-and-packaging.md
        required: false
        why: >-
          The decisions and their costs, including why the reference-graph
          parity runner was generalized rather than duplicated, and why the
          task was not marked completed.
    labs:
      - what: Run the transform-engine and packaging tests
        run: uv run pytest tests/qnn tests/deployment/qualcomm
        proves: >-
          The catalogue loader's own guards, every pass's positive and negative
          cases, the package builder, and the offline compile preflight. The
          `onnx`-dependent cases skip in the locked root environment, which has
          no `onnx`, `onnxruntime`, or `numpy` - that absence is deliberate.
      - what: Prove the candidate logits are bit-identical to the reference
        run: uv run python -c "import json;[print(v,all(a['candidate_logits_sha256']==b['candidate_logits_sha256'] for a,b in zip(json.load(open(f'results/manifests/qnn/parity/{v}-ort-cpu.json'))['steps'],json.load(open(f'results/graph/parity/{v}-ort-cpu.json'))['steps']))) for v in ('S128','S512','S1024','S4096')]"
        proves: >-
          Every one of the 20 recorded steps produced the same logit digest on
          the candidate as on the reference. Then read `graph_digests` in both
          files and confirm the two runs loaded different bytes.
      - what: Read the rejected pass's evidence off committed JSON
        run: uv run python -c "import json;d=json.load(open('results/manifests/qnn/S4096.json'));r=[t for t in d['transformations'] if not t['applied']][0]['rejection_evidence']['prefill'];print(r['node_count'],r['protobuf_bytes'],r['operator_histogram_delta']['Softmax'],r['operator_histogram_delta']['Cast'],len(r['added_opset_domains_used_by_no_node']))"
        proves: >-
          The four grounds in one line: the node count really does fall, the
          protobuf really is 1,811,439,962 bytes, `Softmax` really goes to
          zero, `Cast` really more than doubles, and all eight added domains
          are used by no node.
      - what: Reproduce the committed manifests from the real graphs
        run: SLM_LAB_ARTIFACT_ROOT=<external-root> <onnx-env-python> -m slm_lab.graph.qnn.build --all-manifests --check
        proves: >-
          Every committed number re-derives from reference bytes whose SHA-256
          matches the T20 manifest, down to the candidate digests. Needs the
          artifact root and an environment with `onnx`, `onnxruntime` and
          `numpy`; exits 0 when current. It re-reads the parity record; it does
          not re-measure it.
    notebooks:
      - name: 03_graph_inspection.ipynb
        status: planned
        owner: T80
        focus: >-
          Graph risks and the rewrites that answer them, with the folding
          experiment run against a graph you can re-inspect.
    questions:
      - Pass 1 changes no value and no edge. Explain why every other pass
        depends on it, and what would have happened to the S4096 causal mask
        without it.
      - The static-shape folder reaches zero residual shape-defining inputs at
        S128 and stops at six above it. Explain that from the byte budget, then
        argue against raising the budget.
      - Dead-node elimination removed zero nodes on all eight graphs. Argue
        that this makes the pass useless, then refute yourself from its own
        recorded effect.
      - R-INTERNAL-DYNAMIC-SHAPE reported nothing on the reference and 1,069 on
        the decode candidate. Say why the first was never a zero, and why the
        second is a statement about shape inference rather than about the
        graph.
      - The rejected pass reduces decode to 4,245 nodes where the committed
        catalogue reaches 5,421. Make the strongest case for adopting it, then
        name the single measured number that defeats that case.
      - Every candidate produced bit-identical logits. Explain why that was
        predicted from the six passes alone, and why the same measurement on
        the rejected pass could not have been read the same way.
      - A colleague writes "the T22 candidates are QNN-ready with verified
        numerical parity". Rewrite it so every clause is supported, and list
        what you removed.
    boundaries:
      - No compiler, no converter, no QAIRT tool, no quantizer and no device
        was run. Nothing here establishes compiler acceptance, operator
        support, accelerator placement, latency or memory.
      - No Qualcomm AI Hub job was submitted and no service call was made. The
        package layout for an external-data ONNX model is unverified against
        the service, and T31 owns the first real submission.
      - The parity result is one frozen workload, four variants, 20 steps, one
        host, one execution provider, one ONNX Runtime build, at
        ORT_DISABLE_ALL. It says nothing about how the candidate behaves under
        a real converter.
      - A reduced finding count is a smaller population of a structural
        pattern, not evidence of convertibility; every manifest's
        claim_boundary says so in-band.
      - The rejected pass's numbers describe what onnxruntime 1.28.0's CPU
        optimizer did on one host. One of them did not reproduce a planning
        probe, and the cause is recorded as unknown rather than explained.
```

Note the one thing that must be re-checked after pasting: `LEARN-10` already
cites `03_graph_inspection.ipynb`, and this entry cites it too. That is
deliberate — the notebook serves both subjects — and no test forbids it, but it
is worth a glance if the notebook list is ever made exclusive.
