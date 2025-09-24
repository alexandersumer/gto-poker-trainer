# Engineering Notes

## Feature Flags

- `solver.high_precision_cfr` – doubles the local CFR refinement budget for
  postflop normal-form subgames. Enable via
  `GTOTRAINER_FEATURES=solver.high_precision_cfr` or the benchmark `--enable`
  switch when you want tighter frequencies at the cost of extra CPU time.
- `rival.texture_v2` – applies the new texture-aware rival calibration so
  stronger holdings continue more often on dynamic boards. Combine with the
  benchmark harness to compare realism before turning it on by default.

Flags can be combined: `GTOTRAINER_FEATURES=solver.high_precision_cfr,rival.texture_v2`.

## Benchmark Harness

Run the deterministic session benchmark to track exploitability and accuracy:

```bash
scripts/run_benchmark.py --hands 50 --seeds 101,202
```

Typical outputs include `avg_ev_lost` (approximate exploitability in bb per
decision) and `accuracy_pct` (noise-aware hit rate).

To compare feature flags:

```bash
scripts/run_benchmark.py --enable solver.high_precision_cfr --hands 50
scripts/run_benchmark.py --enable rival.texture_v2 --hands 50
```

The benchmark is deterministic for a given seed, making it safe for CI and
regressions.
