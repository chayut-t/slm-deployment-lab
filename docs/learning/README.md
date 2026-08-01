# Deployment-engineering guides

These guides assume familiarity with transformer architecture. They focus on
engineering concepts needed to deploy and profile SLMs on specific hardware:

- static prefill/decode graphs and explicit KV-cache contracts;
- Qualcomm compilation, quantization, profiling, and runtime integration;
- Apple Silicon GQA/cache layouts, MLX compilation, unified memory, and
  Instruments;
- ONNX Runtime CUDA provider placement and I/O binding;
- numerical validation and cross-platform benchmark interpretation.

Each guide should link to primary references, a runnable notebook or lab, and
the corresponding implementation and result artifacts.

## Learning checkpoints

[`checkpoints.md`](checkpoints.md) groups completed task-graph work into
`LEARN-NN` study units. Each one renders to a self-contained HTML sheet that
mirrors its cited documents verbatim, so a subject can be studied in one place
instead of task by task. The checkpoints are study surfaces, not gates.
