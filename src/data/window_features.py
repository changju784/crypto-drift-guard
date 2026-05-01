"""Window-level sentiment aggregation and EDA regime labeling."""

from __future__ import annotations

import pandas as pd

from data.layer1_config import DEFAULT_MIN_TWEETS_PER_WINDOW, DEFAULT_WINDOW_FREQ


def make_window_id(cryptocurrency: str, window_start: pd.Timestamp) -> str:
    """Create a stable id for a coin/time window."""
    ts = pd.Timestamp(window_start).strftime("%Y%m%dT%H%M%SZ")
    return f"{cryptocurrency}_{ts}"


def aggregate_window_features(
    scored_df: pd.DataFrame,
    *,
    freq: str = DEFAULT_WINDOW_FREQ,
) -> pd.DataFrame:
    """Aggregate scored tweets into coin-specific time windows."""
    required = {"timestamp", "cryptocurrency", "finbert_label", "sentiment_score"}
    missing = sorted(required - set(scored_df.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    df = scored_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["window_start"] = df["timestamp"].dt.floor(freq)
    if "is_synthetic" not in df.columns:
        df["is_synthetic"] = False
    df["is_synthetic"] = df["is_synthetic"].fillna(False).astype(bool)
    if "sentiment_confidence" not in df.columns:
        df["sentiment_confidence"] = pd.NA

    grouped = df.groupby(["cryptocurrency", "window_start"], sort=True)
    rows: list[dict] = []
    for (coin, window_start), group in grouped:
        label_counts = group["finbert_label"].str.lower().value_counts()
        tweet_count = int(len(group))
        positive_count = int(label_counts.get("positive", 0))
        neutral_count = int(label_counts.get("neutral", 0))
        negative_count = int(label_counts.get("negative", 0))
        dominant_label = max(
            ("positive", "neutral", "negative"),
            key=lambda label: {
                "positive": positive_count,
                "neutral": neutral_count,
                "negative": negative_count,
            }[label],
        )

        window_start = pd.Timestamp(window_start)
        rows.append({
            "window_id": make_window_id(str(coin), window_start),
            "window_start": window_start,
            "window_end": window_start + pd.Timedelta(freq),
            "cryptocurrency": str(coin),
            "tweet_count": tweet_count,
            "synthetic_count": int(group["is_synthetic"].sum()),
            "positive_count": positive_count,
            "neutral_count": neutral_count,
            "negative_count": negative_count,
            "positive_ratio": positive_count / tweet_count if tweet_count else 0.0,
            "neutral_ratio": neutral_count / tweet_count if tweet_count else 0.0,
            "negative_ratio": negative_count / tweet_count if tweet_count else 0.0,
            "mean_sentiment_score": float(group["sentiment_score"].mean()),
            "sentiment_std": float(group["sentiment_score"].std(ddof=0)),
            "sentiment_intensity": float(group["sentiment_score"].abs().mean()),
            "mean_sentiment_confidence": (
                float(pd.to_numeric(group["sentiment_confidence"], errors="coerce").mean())
                if "sentiment_confidence" in group else float("nan")
            ),
            "dominant_label": dominant_label,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        result["regime"] = []
        return result
    return classify_sentiment_regimes(result)


def classify_sentiment_regimes(
    window_df: pd.DataFrame,
    *,
    min_tweet_count: int = DEFAULT_MIN_TWEETS_PER_WINDOW,
    neutral_abs_score: float = 0.15,
    skew_ratio: float = 0.60,
    conflict_ratio: float = 0.35,
) -> pd.DataFrame:
    """Label each window as neutral, skewed, conflict, sparse, or mixed."""
    df = window_df.copy()
    regimes: list[str] = []
    for _, row in df.iterrows():
        if int(row["tweet_count"]) < min_tweet_count:
            regimes.append("sparse")
        elif (
            float(row["positive_ratio"]) >= conflict_ratio
            and float(row["negative_ratio"]) >= conflict_ratio
        ):
            regimes.append("high_conflict")
        elif (
            abs(float(row["mean_sentiment_score"])) <= neutral_abs_score
            and float(row["positive_ratio"]) < 0.45
            and float(row["negative_ratio"]) < 0.45
        ):
            regimes.append("neutral")
        elif float(row["positive_ratio"]) >= skew_ratio:
            regimes.append("positive_skew")
        elif float(row["negative_ratio"]) >= skew_ratio:
            regimes.append("negative_skew")
        else:
            regimes.append("mixed")
    df["regime"] = regimes
    return df


def select_attack_target_windows(
    window_df: pd.DataFrame,
    attack_type: str,
    *,
    max_windows: int | None = None,
) -> pd.DataFrame:
    """Select candidate windows for a specific adversarial injection variant."""
    df = window_df.copy()
    if "regime" not in df.columns:
        df = classify_sentiment_regimes(df)

    if attack_type == "positive_pump":
        selected = df[df["regime"] == "neutral"].copy()
        selected = selected.sort_values(["tweet_count", "sentiment_intensity"], ascending=[False, True])
    elif attack_type == "negative_fud":
        selected = df[df["regime"].isin(["neutral", "positive_skew"])].copy()
        selected = selected.sort_values(["positive_ratio", "tweet_count"], ascending=[False, False])
    elif attack_type == "conflict_balance":
        selected = df[df["regime"].isin(["positive_skew", "negative_skew"])].copy()
        selected = selected.assign(
            skew_strength=(selected["positive_ratio"] - selected["negative_ratio"]).abs()
        ).sort_values(["skew_strength", "tweet_count"], ascending=[False, False])
    else:
        raise ValueError(f"unknown attack_type: {attack_type}")

    if max_windows is not None:
        selected = selected.head(max_windows)
    return selected.reset_index(drop=True)
