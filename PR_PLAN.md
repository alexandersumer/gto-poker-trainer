# Incremental PR Roadmap

1. **Telemetry & Benchmark Baseline** – tighten summary stats (accuracy, EV per decision) and extend the deterministic benchmark harness so CI can gate on exploitability drift. *Impact:* visible scoring fixes, better regression signals.
2. **Rival Pressure Calibration** – refine size/texture driven fold logic and persona deltas, with regression tests guarding bet-pressure responses. *Impact:* rival play feels closer to GTOWizard, preserves EV sanity across streets.
3. **Action-Space Tuning** – iterate on bet palette pruning and follow-up range carryover, focusing on common SRP/3-bet nodes surfaced by the benchmark. *Impact:* smoother lines, faster convergence; follow with exploitability snapshot to verify gains.
