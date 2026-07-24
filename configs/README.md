# Configuration

Store reviewable, declarative configuration by concern:

- `models/`: model IDs, revisions, tokenizer, and graph-shape contracts;
- `environments/`: runtime and toolchain selections;
- `targets/`: device/compiler/profile targets;
- `workloads/`: prompts, context lengths, and generation settings;
- `quantization/`: calibration and precision policies;
- `storage/`: portable artifact-root and retention examples.

Do not place credentials or machine-local absolute paths in committed configs.
