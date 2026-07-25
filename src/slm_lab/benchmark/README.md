# Benchmarking

`protocol.py` loads and enforces the frozen T13 benchmark contracts. It
provides:

- Offline linkage checks against the exact T10 context workloads.
- Type-7 median/p90/p95, sample standard deviation, MAD, and IQR.
- Seeded percentile-bootstrap confidence intervals for headline medians.
- Strict result-schema validation.
- Semantic checks that recompute summaries from retained raw samples.
- Timing-class, synchronization, invalid-series, evidence, and system-claim
  boundaries.

Check the repository contracts:

```bash
uv run python -m slm_lab.benchmark.protocol check --root .
```

Validate one future metric record:

```bash
uv run python -m slm_lab.benchmark.protocol \
  check-result results/raw/example-metric.json --root .
```

The validator does not collect measurements. Platform adapters are responsible
for synchronization and evidence capture, then pass result records through
this shared gate.
