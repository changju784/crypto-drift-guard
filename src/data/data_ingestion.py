"""
Layer 1: Data Ingestion & Mutation
Loads the CSV and applies mutation operators to produce baseline and mutant datasets.
"""

import pandas as pd
from pathlib import Path


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
    mutant = df.copy()
    mutant["social_sentiment_score"] = (
        mutant["social_sentiment_score"] * factor
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
