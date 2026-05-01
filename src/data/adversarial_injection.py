"""Adversarial tweet injection for Layer 1 mutation datasets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.schemas import CLEAN_TWEET_COLUMNS, FINBERT_SCORE_COLUMNS
from data.window_features import select_attack_target_windows

POSITIVE_TEMPLATES = {
    "BTC": [
        "BTC is breaking out with massive buying pressure building now. #BTC #Bitcoin",
        "Bitcoin momentum looks extremely bullish as traders keep accumulating. #BTC",
        "Huge upside setup forming for BTC. Market sentiment is turning aggressively positive. #Bitcoin",
    ],
    "ETH": [
        "ETH is gaining strong momentum with buyers stepping in fast. #ETH #Ethereum",
        "Ethereum looks extremely bullish as accumulation accelerates. #ETH",
        "ETH breakout signal is getting stronger and sentiment is turning positive. #Ethereum",
    ],
}

NEGATIVE_TEMPLATES = {
    "BTC": [
        "BTC looks weak with heavy sell pressure and panic spreading fast. #BTC #Bitcoin",
        "Bitcoin momentum is breaking down as traders rush to exit positions. #BTC",
        "BTC sentiment is collapsing and another leg down looks likely. #Bitcoin",
    ],
    "ETH": [
        "ETH looks weak with sell pressure building across the market. #ETH #Ethereum",
        "Ethereum sentiment is collapsing as traders dump before another drop. #ETH",
        "ETH downside risk is rising fast and confidence is fading. #Ethereum",
    ],
}


@dataclass(frozen=True)
class InjectionPlanRow:
    window_id: str
    cryptocurrency: str
    attack_type: str
    target_polarity: str
    existing_tweets: int
    current_target_count: int
    target_ratio: float
    synthetic_needed: int


def required_injections(
    total_count: int,
    current_target_count: int,
    target_ratio: float,
) -> int:
    """Return the minimum synthetic rows needed to reach a target ratio."""
    if not 0 < target_ratio < 1:
        raise ValueError("target_ratio must be between 0 and 1")
    needed = (target_ratio * total_count - current_target_count) / (1 - target_ratio)
    return max(0, int(math.ceil(needed)))


def _target_for_window(row: pd.Series, attack_type: str, target_ratio: float) -> InjectionPlanRow:
    if attack_type == "positive_pump":
        target_polarity = "positive"
        current_target_count = int(row["positive_count"])
    elif attack_type == "negative_fud":
        target_polarity = "negative"
        current_target_count = int(row["negative_count"])
    elif attack_type == "conflict_balance":
        if float(row["positive_ratio"]) >= float(row["negative_ratio"]):
            target_polarity = "negative"
            current_target_count = int(row["negative_count"])
        else:
            target_polarity = "positive"
            current_target_count = int(row["positive_count"])
    else:
        raise ValueError(f"unknown attack_type: {attack_type}")

    tweet_count = int(row["tweet_count"])
    synthetic_needed = required_injections(tweet_count, current_target_count, target_ratio)
    return InjectionPlanRow(
        window_id=str(row["window_id"]),
        cryptocurrency=str(row["cryptocurrency"]),
        attack_type=attack_type,
        target_polarity=target_polarity,
        existing_tweets=tweet_count,
        current_target_count=current_target_count,
        target_ratio=float(target_ratio),
        synthetic_needed=synthetic_needed,
    )


def build_injection_plan(
    window_df: pd.DataFrame,
    *,
    attack_type: str,
    target_ratio: float,
    max_windows: int | None = None,
) -> pd.DataFrame:
    """Select attack windows and compute how many synthetic tweets to add."""
    targets = select_attack_target_windows(window_df, attack_type, max_windows=max_windows)
    rows = [_target_for_window(row, attack_type, target_ratio).__dict__ for _, row in targets.iterrows()]
    return pd.DataFrame(rows)


def _template_for(coin: str, polarity: str, rng: np.random.Generator) -> str:
    templates = POSITIVE_TEMPLATES if polarity == "positive" else NEGATIVE_TEMPLATES
    options = templates.get(coin, templates["BTC"])
    return str(rng.choice(options))


def generate_adversarial_tweets(
    window_df: pd.DataFrame,
    *,
    attack_type: str,
    target_ratio: float,
    max_windows: int | None = None,
    seed: int = 527,
) -> pd.DataFrame:
    """
    Generate unscored synthetic tweets for selected windows.

    The returned rows intentionally do not include FinBERT scores. They must be
    passed through the same Layer 1 scoring step as original tweets.
    """
    plan = build_injection_plan(
        window_df,
        attack_type=attack_type,
        target_ratio=target_ratio,
        max_windows=max_windows,
    )
    if plan.empty:
        return pd.DataFrame(columns=CLEAN_TWEET_COLUMNS)

    rng = np.random.default_rng(seed)
    indexed_windows = window_df.set_index("window_id", drop=False)
    records: list[dict] = []
    for _, plan_row in plan.iterrows():
        if int(plan_row["synthetic_needed"]) <= 0:
            continue
        window = indexed_windows.loc[plan_row["window_id"]]
        window_start = pd.Timestamp(window["window_start"])
        window_end = pd.Timestamp(window["window_end"])
        duration_seconds = max(1, int((window_end - window_start).total_seconds()))
        coin = str(plan_row["cryptocurrency"])
        polarity = str(plan_row["target_polarity"])
        hashtags = [coin, "Bitcoin" if coin == "BTC" else "Ethereum", "crypto"]

        for i in range(int(plan_row["synthetic_needed"])):
            offset = int(((i + 1) / (int(plan_row["synthetic_needed"]) + 1)) * duration_seconds)
            timestamp = window_start + pd.Timedelta(seconds=offset)
            tweet_id = f"synthetic:{attack_type}:{plan_row['window_id']}:{i}"
            records.append({
                "tweet_id": tweet_id,
                "source_row_id": tweet_id,
                "timestamp": timestamp,
                "cryptocurrency": coin,
                "tweet_text": _template_for(coin, polarity, rng),
                "hashtags": json.dumps(hashtags),
                "url": "",
                "username": f"synthetic_{attack_type}",
                "is_synthetic": True,
                "attack_type": attack_type,
                "target_window_id": str(plan_row["window_id"]),
                "source": "synthetic_adversarial",
            })

    return pd.DataFrame(records, columns=CLEAN_TWEET_COLUMNS)


def merge_scored_synthetic(
    baseline_scored_df: pd.DataFrame,
    synthetic_scored_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge baseline scored tweets with already-scored synthetic tweets."""
    missing = [col for col in FINBERT_SCORE_COLUMNS if col not in synthetic_scored_df.columns]
    if missing:
        raise ValueError(
            "synthetic tweets must be scored before merge; missing "
            + ", ".join(missing)
        )
    merged = pd.concat([baseline_scored_df, synthetic_scored_df], ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
    return merged.sort_values(["timestamp", "cryptocurrency", "tweet_id"]).reset_index(drop=True)
