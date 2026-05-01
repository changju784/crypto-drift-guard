"""Column contracts for Layer 1 tweet and window datasets."""

from __future__ import annotations

CLEAN_TWEET_COLUMNS = [
    "tweet_id",
    "source_row_id",
    "timestamp",
    "cryptocurrency",
    "tweet_text",
    "hashtags",
    "url",
    "username",
    "is_synthetic",
    "attack_type",
    "target_window_id",
    "source",
]

FINBERT_SCORE_COLUMNS = [
    "finbert_label",
    "p_positive",
    "p_neutral",
    "p_negative",
    "sentiment_score",
    "sentiment_confidence",
]

SCORED_TWEET_COLUMNS = CLEAN_TWEET_COLUMNS + FINBERT_SCORE_COLUMNS

WINDOW_FEATURE_COLUMNS = [
    "window_id",
    "window_start",
    "window_end",
    "cryptocurrency",
    "tweet_count",
    "synthetic_count",
    "positive_count",
    "neutral_count",
    "negative_count",
    "positive_ratio",
    "neutral_ratio",
    "negative_ratio",
    "mean_sentiment_score",
    "sentiment_std",
    "sentiment_intensity",
    "mean_sentiment_confidence",
    "dominant_label",
    "regime",
]
