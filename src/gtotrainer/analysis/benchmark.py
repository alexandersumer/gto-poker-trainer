"""Deterministic benchmark harness for trainer policies.

The benchmark runs small self-play sessions using the existing session
manager, then summarises the EV loss and hit rates using the core scoring
helpers.  This keeps the regression cheap while letting us compare solver
tweaks without touching production infrastructure.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from ..core.models import Option
from ..core.scoring import SummaryStats, summarize_records
from ..dynamic.episode import Node
from ..dynamic.generator import available_rival_styles
from ..dynamic.policy import reset_bet_sizing_state
from ..features.session.service import SessionConfig, SessionManager


@dataclass(frozen=True)
class BenchmarkScenario:
    """Single deterministic configuration executed inside the benchmark."""

    name: str
    seed: int
    rival_style: str
    hero_policy: str = "gto"
    hands: int | None = None

    def resolve_hands(self, fallback: int) -> int:
        value = self.hands
        return value if value and value > 0 else fallback


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

            def score(idx: int) -> float:
                opt = options[idx]
                meta = getattr(opt, "meta", None) or {}
                baseline = meta.get("baseline_ev")
                value = float(opt.ev)
                if baseline is not None:
                    try:
                        value = max(value, float(baseline))
                    except (TypeError, ValueError):
                        pass
                return value

            return max(range(len(options)), key=score)

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
    scenarios: tuple[BenchmarkScenario, ...] | None = None

    def __post_init__(self) -> None:
        if self.hands <= 0:
            raise ValueError("hands must be positive")
        if self.mc_trials <= 0:
            raise ValueError("mc_trials must be positive")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        allowed_styles = available_rival_styles()
        if self.scenarios:
            for scenario in self.scenarios:
                if scenario.rival_style.strip().lower() not in allowed_styles:
                    raise ValueError(
                        f"Unknown rival_style '{scenario.rival_style}'. Options: {', '.join(sorted(allowed_styles))}"
                    )
        elif self.rival_style.strip().lower() not in allowed_styles:
            raise ValueError(f"Unknown rival_style '{self.rival_style}'. Options: {', '.join(sorted(allowed_styles))}")


@dataclass(frozen=True)
class BenchmarkRun:
    scenario: BenchmarkScenario
    stats: SummaryStats

    @property
    def accuracy_pct(self) -> float:
        return self.stats.accuracy_pct

    @property
    def exploitability_bb(self) -> float:
        return self.stats.avg_ev_lost


@dataclass(frozen=True)
class BenchmarkResult:
    runs: tuple[BenchmarkRun, ...]
    combined: SummaryStats

    @property
    def accuracy_pct(self) -> float:
        return self.combined.accuracy_pct

    @property
    def exploitability_bb(self) -> float:
        return self.combined.avg_ev_lost


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    scenarios = _expand_scenarios(config)
    runs: list[BenchmarkRun] = []
    all_records: list[dict] = []

    for scenario in scenarios:
        policy = _HeroPolicy(scenario.hero_policy)
        manager = SessionManager()
        reset_bet_sizing_state()
        session_id = manager.create_session(
            SessionConfig(
                hands=scenario.resolve_hands(config.hands),
                mc_trials=config.mc_trials,
                seed=scenario.seed,
                rival_style=scenario.rival_style,
            )
        )

        def _chooser(node: Node, options: Sequence[Option], rng: random.Random) -> int:
            return policy.select(options, rng)

        run_records = manager.drive_session(session_id, _chooser, cleanup=True)

        all_records.extend(run_records)
        runs.append(BenchmarkRun(scenario=scenario, stats=summarize_records(run_records)))

    combined = summarize_records(all_records)
    return BenchmarkResult(runs=tuple(runs), combined=combined)


def _expand_scenarios(config: BenchmarkConfig) -> tuple[BenchmarkScenario, ...]:
    if config.scenarios:
        return config.scenarios

    style = config.rival_style.strip().lower()
    policy = config.hero_policy
    scenarios: list[BenchmarkScenario] = []
    for idx, seed in enumerate(config.seeds):
        scenarios.append(
            BenchmarkScenario(
                name=f"seed_{idx}_{seed}",
                seed=seed,
                rival_style=style,
                hero_policy=policy,
            )
        )
    return tuple(scenarios)
