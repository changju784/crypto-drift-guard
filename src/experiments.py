"""
Experiment registry and selection helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from data.data_ingestion import (
    mutate_sentiment_flip,
    mutate_sentiment_fused_eq,
    mutate_sentiment_scale,
    mutate_temporal_lag_by_time,
)


MutationFn = Callable[[pd.DataFrame, str, Optional[int]], pd.DataFrame]

DRIFT_TEMPORAL_COLUMNS = [
    "social_sentiment_score",
    "rsi_technical_indicator",
    "volatility_index",
]

HYBRID_TEMPORAL_COLUMNS = [
    *DRIFT_TEMPORAL_COLUMNS,
    "price_change_24h_percent",
    "fear_greed_index",
]


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    family: str
    description: str
    mutation_fn: MutationFn
    stochastic: bool = False
    requires_seed: bool = False

    def mutate(
        self,
        df: pd.DataFrame,
        *,
        mode: str,
        seed: int | None = None,
    ) -> pd.DataFrame:
        return self.mutation_fn(df, mode, seed)


def _scale(factor: float) -> MutationFn:
    def _inner(df: pd.DataFrame, mode: str, seed: int | None) -> pd.DataFrame:
        del mode, seed
        return mutate_sentiment_scale(df, factor=factor)
    return _inner


def _flip(probability: float) -> MutationFn:
    def _inner(df: pd.DataFrame, mode: str, seed: int | None) -> pd.DataFrame:
        del mode
        return mutate_sentiment_flip(df, p=probability, seed=seed)
    return _inner


def _fused_eq(df: pd.DataFrame, mode: str, seed: int | None) -> pd.DataFrame:
    del mode, seed
    return mutate_sentiment_fused_eq(df)


def _temporal_lag(lag: str) -> MutationFn:
    delta = pd.Timedelta(lag)

    def _inner(df: pd.DataFrame, mode: str, seed: int | None) -> pd.DataFrame:
        del seed
        columns = DRIFT_TEMPORAL_COLUMNS if mode == "drift" else HYBRID_TEMPORAL_COLUMNS
        return mutate_temporal_lag_by_time(df, lag=delta, columns=columns)
    return _inner


def build_registry() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            id="sentiment_scale_k05",
            family="sentiment",
            description="social_sentiment_score x0.5",
            mutation_fn=_scale(0.5),
        ),
        ExperimentSpec(
            id="sentiment_scale_k15",
            family="sentiment",
            description="social_sentiment_score x1.5",
            mutation_fn=_scale(1.5),
        ),
        ExperimentSpec(
            id="sentiment_scale_k20",
            family="sentiment",
            description="social_sentiment_score x2.0",
            mutation_fn=_scale(2.0),
        ),
        ExperimentSpec(
            id="sentiment_flip_p10",
            family="sentiment",
            description="flip social sentiment with p=0.10",
            mutation_fn=_flip(0.10),
            stochastic=True,
            requires_seed=True,
        ),
        ExperimentSpec(
            id="sentiment_flip_p20",
            family="sentiment",
            description="flip social sentiment with p=0.20",
            mutation_fn=_flip(0.20),
            stochastic=True,
            requires_seed=True,
        ),
        ExperimentSpec(
            id="sentiment_flip_p30",
            family="sentiment",
            description="flip social sentiment with p=0.30",
            mutation_fn=_flip(0.30),
            stochastic=True,
            requires_seed=True,
        ),
        ExperimentSpec(
            id="sentiment_fused_eq",
            family="sentiment",
            description="effective sentiment = 0.5*social + 0.5*news",
            mutation_fn=_fused_eq,
        ),
        ExperimentSpec(
            id="temporal_lag_30m",
            family="temporal",
            description="lag consumed signals by 30 minutes within each coin",
            mutation_fn=_temporal_lag("30min"),
        ),
        ExperimentSpec(
            id="temporal_lag_2h",
            family="temporal",
            description="lag consumed signals by 2 hours within each coin",
            mutation_fn=_temporal_lag("2h"),
        ),
        ExperimentSpec(
            id="temporal_lag_6h",
            family="temporal",
            description="lag consumed signals by 6 hours within each coin",
            mutation_fn=_temporal_lag("6h"),
        ),
        ExperimentSpec(
            id="temporal_lag_12h",
            family="temporal",
            description="lag consumed signals by 12 hours within each coin",
            mutation_fn=_temporal_lag("12h"),
        ),
    ]


def parse_csv_arg(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def select_experiments(
    registry: list[ExperimentSpec],
    *,
    families: list[str] | None = None,
    experiment_ids: list[str] | None = None,
) -> list[ExperimentSpec]:
    by_id = {spec.id: spec for spec in registry}
    known_families = {spec.family for spec in registry}

    if experiment_ids:
        unknown = [exp_id for exp_id in experiment_ids if exp_id not in by_id]
        if unknown:
            raise ValueError(f"Unknown experiment id(s): {', '.join(unknown)}")
        return [by_id[exp_id] for exp_id in experiment_ids]

    if families:
        unknown = [family for family in families if family not in known_families]
        if unknown:
            raise ValueError(f"Unknown family name(s): {', '.join(unknown)}")
        return [spec for spec in registry if spec.family in families]

    return list(registry)


def format_catalog(registry: list[ExperimentSpec]) -> str:
    lines = ["=== Experiment Catalog ==="]
    families = sorted({spec.family for spec in registry})
    for family in families:
        lines.append(f"[{family}]")
        for spec in [s for s in registry if s.family == family]:
            suffix = " stochastic" if spec.stochastic else ""
            lines.append(f"  {spec.id:<24} {spec.description}{suffix}")
        lines.append("")
    return "\n".join(lines).rstrip()
