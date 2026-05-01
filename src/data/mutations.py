"""Layer 1 non-adversarial mutation operators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mutate_sentiment_scale(
    df: pd.DataFrame,
    *,
    factor: float,
    score_col: str = "sentiment_score",
) -> pd.DataFrame:
    """Scale a sentiment score column and clip it to [-1, 1]."""
    if score_col not in df.columns:
        raise ValueError(f"missing score column: {score_col}")
    mutant = df.copy()
    mutant[score_col] = (pd.to_numeric(mutant[score_col]) * factor).clip(-1.0, 1.0)
    mutant["mutation_type"] = f"sentiment_scale_k{factor:g}"
    return mutant


def mutate_sentiment_flip(
    df: pd.DataFrame,
    *,
    probability: float,
    seed: int | None = None,
    score_col: str = "sentiment_score",
) -> pd.DataFrame:
    """Flip the sign of a sentiment score with Bernoulli probability."""
    if score_col not in df.columns:
        raise ValueError(f"missing score column: {score_col}")
    mutant = df.copy()
    rng = np.random.default_rng(seed)
    mask = rng.random(len(mutant)) < probability
    mutant.loc[mask, score_col] = -pd.to_numeric(mutant.loc[mask, score_col])
    mutant["mutation_type"] = f"sentiment_flip_p{probability:g}"
    return mutant


def mutate_temporal_lag_by_time(
    df: pd.DataFrame,
    *,
    lag: pd.Timedelta,
    columns: list[str],
    group_col: str = "cryptocurrency",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Replace selected columns with the most recent same-coin values at
    `timestamp - lag`. Rows without valid history keep original values.
    """
    missing = [column for column in [group_col, timestamp_col, *columns] if column not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    mutant = df.copy()
    mutant[timestamp_col] = pd.to_datetime(mutant[timestamp_col], utc=True, errors="coerce")

    for _, idx in mutant.groupby(group_col, sort=False).groups.items():
        group = mutant.loc[idx].sort_values(timestamp_col).copy()
        times = group[timestamp_col].to_numpy()
        lookup = times.searchsorted(times - lag, side="right") - 1
        valid = lookup >= 0
        if not valid.any():
            continue

        for column in columns:
            values = group[column].to_numpy(copy=True)
            source = group[column].to_numpy()
            values[valid] = source[lookup[valid]]
            group[column] = values

        mutant.loc[group.index, columns] = group[columns]

    mutant["mutation_type"] = f"temporal_lag_{lag}"
    return mutant
