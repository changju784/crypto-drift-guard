"""
Layer 1: Data Ingestion & Mutation
Loads the CSV and applies mutation operators to produce baseline and mutant datasets.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
import numpy as np


DATA_PATH = Path(__file__).parent.parent.parent / "data" / "crypto_sentiment_prediction_dataset.csv"


def load_baseline() -> pd.DataFrame:
    """Load and time-sort the raw CSV into a clean baseline DataFrame."""
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def mutate_intensity(df: pd.DataFrame, factor: float = 2.0) -> pd.DataFrame:
    """
    Intensity Mutation: multiply social_sentiment_score by `factor`.
    Scores are clipped to [-1, 1] to stay in a realistic range.
    """
    return mutate_sentiment_scale(df, factor=factor)


def mutate_sentiment_scale(df: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Scale social sentiment by `factor`, clipped to [-1, 1]."""
    mutant = df.copy()
    mutant["social_sentiment_score"] = (
        mutant["social_sentiment_score"] * factor
    ).clip(-1.0, 1.0)
    return mutant


def mutate_sentiment_flip(df: pd.DataFrame, p: float, seed: int | None = None) -> pd.DataFrame:
    """Flip the sign of social sentiment with Bernoulli probability `p`."""
    mutant = df.copy()
    rng = np.random.default_rng(seed)
    mask = rng.random(len(mutant)) < p
    mutant.loc[mask, "social_sentiment_score"] = (
        -mutant.loc[mask, "social_sentiment_score"]
    )
    return mutant


def mutate_sentiment_fused_eq(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the effective social sentiment with equal social/news fusion."""
    mutant = df.copy()
    mutant["social_sentiment_score"] = (
        0.5 * mutant["social_sentiment_score"] +
        0.5 * mutant["news_sentiment_score"]
    ).clip(-1.0, 1.0)
    return mutant


def mutate_temporal_jitter(df: pd.DataFrame, shift: int = 3) -> pd.DataFrame:
    """
    Temporal Jitter: shift news_impact_score by `shift` rows to simulate lag.
    Rows where the shifted value is NaN are filled with the column mean.
    """
    mutant = df.copy()
    mutant["news_impact_score"] = (
        mutant["news_impact_score"]
        .shift(shift)
        .fillna(df["news_impact_score"].mean())
    )
    return mutant
