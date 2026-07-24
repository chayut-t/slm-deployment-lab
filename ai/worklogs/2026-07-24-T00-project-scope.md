# T00: Project scope and platform priorities

Date: 2026-07-24
Task: `T00`
Visibility: `public`
Status: draft

## Outcome

The project scope, platform order, model priority, educational approach,
benchmark contexts, artifact policy, six-week schedule, and minimum success
criteria were consolidated into `docs/project/plan.md`. This is partial T00
evidence; immutable version pins and the formal ADR remain outstanding.

## Changes

- Made the public Qualcomm pipeline the first priority.
- Set Apple M4/MLX as the second platform and NVIDIA/ONNX Runtime CUDA as the
  third comparison platform.
- Selected Qwen3-0.6B as the primary model and defined bounded fallback rules.
- Added hardware-focused readings, notebooks, profiling, quantization, task
  dependencies, agent coordination, and portfolio deliverables.

## Verification

- Confirmed that the plan covers the three platforms, target Qualcomm devices,
  quantization paths, context sweep, profiling, learning artifacts, and
  dependency-aware execution.
- Validated the plan's Markdown fence balance and Mermaid task references.

## Decisions and evidence

- Stable scope lives in `docs/project/plan.md`; dynamic status lives in the
  task graph.
- Learning material assumes transformer fundamentals and concentrates on
  deployment engineering.

## Risks and limitations

- Public Qualcomm capabilities and device availability must be revalidated
  during the access vertical slice.

## Follow-up

- Newly unblocked tasks: none until T00 is integrated.
- Recommended next action: pin immutable model/toolchain revisions, write the
  T00 ADR, integrate the changes, and then mark T00 completed.
