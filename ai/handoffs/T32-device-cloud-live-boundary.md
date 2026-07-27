# T32 Device Cloud live boundary handoff

Date: 2026-07-27
Task: `T32`
Status: active; blocked at learner authentication and live-device execution

## Prepared outcome

A reproducible, privacy-safe Qwen3-0.6B GenieX/`llama_cpp` capture path is
ready. No live generation, timing, runtime, or allocated-device result is
claimed, and T32 remains `in_progress`.

The Qualcomm Device Cloud public catalog showed a Snapdragon X Elite Compute
Reference Design (`CRD8380X`, Windows), but the browser session was logged out
and required learner login plus free-minute activation before the complete
catalog or a session could be used. The live Device Cloud tab was left open
for learner handoff.

## Changes ready on the task branch

- A closed-schema capture normalizer refuses to emit a completed manifest
  without exact device/runtime identity, Qwen3-0.6B
  `geniex_llamacpp`/`Q4_0` provenance, observed NPU/HTP evidence, confirmed
  multi-token output, every timing boundary with a source, and zero
  paid-resource use.
- A Windows PowerShell session workflow, private-capture template, and operator
  guide pin the current model-card fetch contract and fixed prompt.
- The public result page separates catalog discovery from allocated-device
  evidence and GenieX/llama.cpp from custom QNN/QAIRT evidence.
- Seven task-scoped regression tests cover the normalizer. T32's original
  owned paths omitted the completed T30 test subtree; the coordinating agent
  explicitly approved the narrow ownership expansion for
  `tests/deployment/qualcomm/test_device_cloud.py`. No existing T30 test was
  modified.

## Evidence and decisions

- T02 proves authenticated Workbench access only; its worklog leaves Device
  Cloud minutes, X Elite availability, and a live session to T32.
- Public catalog discovery on 2026-07-27 exposed the X Elite CRD and an
  `Unlock Free Minutes` label, but not account minutes or allocated hardware.
- Qualcomm AI Hub Models v0.58.0 publishes:

  ```text
  qai-hub-models fetch Qwen3-0.6B --runtime geniex_llamacpp --precision q4_0
  ```

- Current GenieX documentation distinguishes `llama_cpp` GGUF execution from
  `qairt` precompiled-bundle execution. T32's ready-made route uses
  `llama_cpp`; it cannot satisfy the custom-QNN path.
- A completed capture must distinguish artifact load, model load,
  tokenization, prefill, first decode, remaining decode, generation total, and
  request total. Missing boundaries are a blocker, not permission to relabel
  Workbench graph latency or session turnaround.

## Verification at handoff

- `python -m unittest tests.deployment.qualcomm.test_device_cloud -v`:
  7 tests passed.
- `python -m pytest -q`: 125 passed, 3 skipped.
- Repository-wide `unittest discover` was also attempted; it ran 93 tests but
  reported the Torch-absent pytest import-skip as one import error. The same
  suite passes under the repository's configured pytest runner, where that
  dependency is correctly skipped.
- `ruff check` across the T32 Python source, command, and tests: passed.
- `ruff format --check` across the same paths: passed.
- `python scripts/ai/render_task_status.py --check`: passed for 30 tasks.
- `python scripts/repo/check_hygiene.py --all`: passed for 210 tracked and
  untracked public files.
- `git diff --check` and final ignored-status inspection: passed; only
  task-scoped public files are present, with caches/artifacts ignored.
- The PowerShell workflow could not be executed or syntax checked on the macOS
  implementation host.

## Resume procedure

1. Learner signs in at `https://qdc.qualcomm.com/`, confirms free minutes, and
   starts the X Elite CRD session. No paid resource is authorized.
2. Follow `scripts/qualcomm/device_cloud/README.md`; keep raw logs under
   `.ai-local/profiles/T32/`.
3. Record allocated-device identity, exact GenieX version, runtime placement,
   model artifact hash, token counts, output hash, and all timing values.
4. If the installed GenieX build does not expose every timing boundary, add
   trustworthy runtime/API instrumentation or preserve a bounded blocker. Do
   not derive values from unrelated Workbench or session timings.
5. Run the sanitizer and review the private/public split and every timing
   boundary with the learner.
6. Only after every T32 acceptance criterion is satisfied, create the public
   worklog, set the graph worklog/status, render task status, and move the
   active execution plan to completed.
