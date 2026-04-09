"""
Experiment registry and selection helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from data.data_ingestion import mutate_intensity, mutate_temporal_jitter


MutationFn = Callable[[pd.DataFrame, str, int | None], pd.DataFrame]


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


def _intensity_k2(df: pd.DataFrame, mode: str, seed: int | None) -> pd.DataFrame:
    del mode, seed
    return mutate_intensity(df, factor=2.0)


def _temporal_jitter_n3(df: pd.DataFrame, mode: str, seed: int | None) -> pd.DataFrame:
    del mode, seed
    return mutate_temporal_jitter(df, shift=3)


def build_registry() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            id="intensity_k2",
            family="sentiment",
            description="sentiment x2.0",
            mutation_fn=_intensity_k2,
        ),
        ExperimentSpec(
            id="temporal_jitter_n3",
            family="temporal",
            description="news_impact shifted 3 steps",
            mutation_fn=_temporal_jitter_n3,
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
