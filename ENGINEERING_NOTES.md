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
scripts/run_benchmark.py --hands 40 --seeds 101,202,303
```

Typical outputs include `avg_ev_lost` (approximate exploitability in bb per
decision) and `accuracy_pct` (noise-aware hit rate).

To compare feature flags:

```bash
scripts/run_benchmark.py --enable solver.high_precision_cfr --hands 40
scripts/run_benchmark.py --enable rival.texture_v2 --hands 40
scripts/run_benchmark.py --enable solver.high_precision_cfr,rival.texture_v2 --hands 40
```

The benchmark is deterministic for a given seed, making it safe for CI and
regressions.

### Scenario packs

- `--scenario-pack standard` (default) replays three contrasting spots: BTN vs BB
  single-raised pot, turn probe response, and an aggressive 3-bet defence.
- `--scenario-pack seeded` replays each supplied seed independently—useful when
  you want before/after flag diffs on identical cards.

Example extended run:

```bash
scripts/run_benchmark.py --hands 80 --seeds 101,202,303 --enable solver.high_precision_cfr
```
