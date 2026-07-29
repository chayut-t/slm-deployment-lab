# SLM Deployment Lab

A deployment-engineering curriculum and reproducible portfolio project for
Qwen3-0.6B across:

1. Qualcomm Snapdragon and Dragonwing targets through public Qualcomm AI Hub
   and Device Cloud workflows.
2. Apple Silicon through a hardware-aware MLX runtime on the project Mac mini.
3. NVIDIA CUDA through ONNX Runtime.

The project emphasizes graph contracts, compilation, quantization, runtime
engineering, profiling, numerical validation, and reproducible agent-assisted
development.

## Start here

- [Development setup](DEVELOPMENT.md)
- [Project plan](docs/project/plan.md)
- [Agent operating rules](AGENTS.md)
- [Execution-plan conventions](PLANS.md)
- [Public AI workspace](ai/README.md)
- [Learning material](docs/learning/README.md)

## Repository state

Implementation is intentionally scaffolded before model and deployment work
begins. The project plan is stable scope guidance; day-to-day progress belongs
in `ai/tasks/task_graph.yaml` and its generated status report.

Large model artifacts, compiled binaries, and traces live outside Git. On the
primary development machine, `artifacts/` points to
`/Volumes/T9/slm-deployment-lab`.
