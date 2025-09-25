# Engineering Notes

## Trainer Defaults

- Rival personas automatically bias fold/call choices using bet pressure,
  board texture, and tracked hero aggression. No feature toggles required.
- Summary stats now report `accuracy_pct`, `avg_ev_lost`, and `avg_loss_pct`
  to keep scoring, UI, and benchmarks aligned.
- Range sampling and mutable hand-state helpers live in
  `src/gtotrainer/dynamic/range_sampling.py` and
  `src/gtotrainer/dynamic/hand_state.py`. Extend those modules instead of
  inlining new dict math in the policy or generator layers.
- Accuracy uses EV-band weighting (partial credit for yellow feedback) to
  keep noise-level mistakes from tanking headline percentages.

## Benchmark Harness

Run the deterministic session benchmark to track exploitability and accuracy:

```bash
scripts/run_benchmark.py --hands 40 --seeds 101,202,303
```

Typical outputs include `avg_ev_lost` (approximate exploitability in bb per
decision) and `accuracy_pct` (noise-aware hit rate).

To compare solver tweaks, capture a baseline run and then rerun after your
changes:

```bash
scripts/run_benchmark.py --hands 40 --seeds 101,202,303
scripts/run_benchmark.py --hands 40 --seeds 101,202,303 --mc-trials 120
```

The benchmark is deterministic for a given seed, making it safe for CI and
regressions.

### Scenario packs

- `--scenario-pack standard` (default) replays three contrasting spots: BTN vs BB
  single-raised pot, turn probe response, and an aggressive 3-bet defence.
- `--scenario-pack seeded` replays each supplied seed independently—useful when
  you want before/after diffs on identical cards.

Example extended run:

```bash
scripts/run_benchmark.py --hands 80 --seeds 101,202,303 --mc-trials 160
```
