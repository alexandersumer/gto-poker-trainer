"""Deterministic benchmark harness for trainer policies.

The benchmark runs small self-play sessions using the existing session
manager, then summarises the EV loss and hit rates using the core scoring
helpers.  This keeps the regression cheap while letting us compare different
feature-flag configurations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from ..core import feature_flags
from ..core.scoring import SummaryStats, summarize_records
from ..dynamic.generator import available_rival_styles
from ..core.models import Option
from ..dynamic.policy import reset_bet_sizing_state
from ..features.session.service import SessionConfig, SessionManager, _ensure_active_node, _ensure_options


class _HeroPolicy:
    def __init__(self, mode: str) -> None:
        allowed = {"best", "gto"}
        key = mode.strip().lower()
        if key not in allowed:
            raise ValueError(f"Unknown hero_policy '{mode}'. Options: {', '.join(sorted(allowed))}")
        self.mode = key

    def select(self, options: Sequence[Option], rng: random.Random) -> int:
        if not options:
            raise ValueError("options list is empty")
        if self.mode == "best":
            return max(range(len(options)), key=lambda idx: options[idx].ev)

        weighted: list[tuple[int, float]] = []
        for idx, option in enumerate(options):
            freq = getattr(option, "gto_freq", None)
            if freq is None or freq <= 0:
                continue
            weighted.append((idx, float(freq)))
        if not weighted:
            return max(range(len(options)), key=lambda idx: options[idx].ev)

        total = sum(weight for _, weight in weighted)
        draw = rng.random() * total
        cumulative = 0.0
        for idx, weight in weighted:
            cumulative += weight
            if draw <= cumulative:
                return idx
        return weighted[-1][0]


@dataclass(frozen=True)
class BenchmarkConfig:
    hands: int = 50
    seeds: tuple[int, ...] = (101,)
    mc_trials: int = 96
    rival_style: str = "balanced"
    hero_policy: str = "gto"
    enable_features: tuple[str, ...] = ()
    disable_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.hands <= 0:
            raise ValueError("hands must be positive")
        if self.mc_trials <= 0:
            raise ValueError("mc_trials must be positive")
        allowed_styles = available_rival_styles()
        if self.rival_style.strip().lower() not in allowed_styles:
            raise ValueError(f"Unknown rival_style '{self.rival_style}'. Options: {', '.join(sorted(allowed_styles))}")


@dataclass(frozen=True)
class BenchmarkRun:
    seed: int
    stats: SummaryStats

    @property
    def accuracy_pct(self) -> float:
        return (100.0 * self.stats.hits / self.stats.decisions) if self.stats.decisions else 0.0

    @property
    def exploitability_bb(self) -> float:
        return self.stats.avg_ev_lost


@dataclass(frozen=True)
class BenchmarkResult:
    runs: tuple[BenchmarkRun, ...]
    combined: SummaryStats

    @property
    def accuracy_pct(self) -> float:
        return (100.0 * self.combined.hits / self.combined.decisions) if self.combined.decisions else 0.0

    @property
    def exploitability_bb(self) -> float:
        return self.combined.avg_ev_lost


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    policy = _HeroPolicy(config.hero_policy)
    runs: list[BenchmarkRun] = []
    all_records: list[dict] = []

    with feature_flags.override(enable=config.enable_features, disable=config.disable_features):
        for seed in config.seeds:
            manager = SessionManager()
            reset_bet_sizing_state()
            session_id = manager.create_session(
                SessionConfig(
                    hands=config.hands,
                    mc_trials=config.mc_trials,
                    seed=seed,
                    rival_style=config.rival_style,
                )
            )
            state = manager._sessions[session_id]

            while True:
                node = _ensure_active_node(state)
                if node is None:
                    break
                options = _ensure_options(state, node)
                choice = policy.select(options, state.engine.rng)
                manager.choose(session_id, choice)

            # Copy records before releasing the session.
            run_records = [dict(record) for record in state.records]
            all_records.extend(run_records)
            runs.append(BenchmarkRun(seed=seed, stats=summarize_records(run_records)))
            manager._sessions.pop(session_id, None)

    combined = summarize_records(all_records)
    return BenchmarkResult(runs=tuple(runs), combined=combined)
