# Committed results

Commit only compact, reproducible evidence:

- `raw/`: small source metrics or sanitized excerpts;
- `processed/`: normalized comparison tables;
- `plots/`: publication-ready figures;
- `manifests/`: artifact, runtime, compile, and benchmark manifests;
- `hosts/`: sanitized host/device environment manifests;
- `quantization/`: calibration and quantization evidence, including the T40
  pre-quantization baseline parity records.

Weights, compiled binaries, full traces, and large raw profiles belong under
the external artifact root.
