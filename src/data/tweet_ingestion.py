"""Crypto10K tweet loading and BTC/ETH cleansing utilities."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from data.layer1_config import resolve_crypto10k_path
from data.schemas import CLEAN_TWEET_COLUMNS

BTC_ALIASES = {"bitcoin", "btc"}
ETH_ALIASES = {"ethereum", "eth"}

BTC_PATTERN = re.compile(r"(?i)(?:\bbitcoin\b|\bbtc\b|#btc\b|\$btc\b)")
ETH_PATTERN = re.compile(r"(?i)(?:\bethereum\b|\beth\b|#eth\b|\$eth\b)")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_tweet_text(text: object) -> str:
    """Collapse whitespace in tweet text while preserving the original words."""
    if pd.isna(text):
        return ""
    return WHITESPACE_PATTERN.sub(" ", str(text)).strip()


def parse_hashtags(value: object) -> list[str]:
    """Parse Kaggle hashtag strings like "['Bitcoin', 'BTC']" into a list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(tag).strip() for tag in value if str(tag).strip()]
    if pd.isna(value):
        return []
    raw = str(value).strip()
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        parsed = [part.strip() for part in raw.split(",")]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, Iterable):
        return []
    return [str(tag).strip() for tag in parsed if str(tag).strip()]


def detect_cryptocurrencies(hashtags: object, text: object) -> list[str]:
    """Return BTC and/or ETH when either hashtags or tweet text mention them."""
    tags = {tag.lower().lstrip("#") for tag in parse_hashtags(hashtags)}
    clean_text = str(text or "")

    has_btc = bool(tags & BTC_ALIASES) or bool(BTC_PATTERN.search(clean_text))
    has_eth = bool(tags & ETH_ALIASES) or bool(ETH_PATTERN.search(clean_text))

    coins: list[str] = []
    if has_btc:
        coins.append("BTC")
    if has_eth:
        coins.append("ETH")
    return coins


def load_crypto10k_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the malformed-tolerant Crypto10K CSV with pandas' Python parser."""
    csv_path = resolve_crypto10k_path(path)
    return pd.read_csv(csv_path, engine="python", on_bad_lines="skip")


def _source_row_id(row: pd.Series, fallback_index: int) -> str:
    for candidate in ("", "Unnamed: 0", "index"):
        if candidate in row and not pd.isna(row[candidate]):
            return str(row[candidate])
    return str(fallback_index)


def clean_crypto10k(
    path: str | Path | None = None,
    *,
    explode_multi_coin: bool = True,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """
    Clean the raw Crypto10K CSV and keep BTC/ETH-relevant tweets.

    Tweets mentioning both BTC and ETH are exploded into two rows by default so
    downstream window features remain coin-specific.
    """
    raw = load_crypto10k_raw(path)
    if max_rows is not None:
        raw = raw.head(max_rows)

    if "Date" not in raw.columns or "Content" not in raw.columns:
        raise ValueError("Crypto10K CSV must contain Date and Content columns")

    raw = raw.copy()
    raw["timestamp"] = pd.to_datetime(raw["Date"], utc=True, errors="coerce")
    raw["tweet_text"] = raw["Content"].map(normalize_tweet_text)
    raw = raw.dropna(subset=["timestamp"])
    raw = raw[raw["tweet_text"].str.len() > 0]

    records: list[dict] = []
    for fallback_index, row in raw.iterrows():
        source_row_id = _source_row_id(row, int(fallback_index))
        hashtags = parse_hashtags(row.get("Hashtags"))
        coins = detect_cryptocurrencies(hashtags, row.get("tweet_text"))
        if not coins:
            continue
        if not explode_multi_coin and len(coins) > 1:
            coins = ["BTC_ETH"]

        for coin in coins:
            records.append({
                "tweet_id": f"{source_row_id}:{coin}",
                "source_row_id": source_row_id,
                "timestamp": row["timestamp"],
                "cryptocurrency": coin,
                "tweet_text": row["tweet_text"],
                "hashtags": json.dumps(hashtags, ensure_ascii=False),
                "url": "" if pd.isna(row.get("URL")) else str(row.get("URL")),
                "username": "" if pd.isna(row.get("Username")) else str(row.get("Username")),
                "is_synthetic": False,
                "attack_type": "baseline",
                "target_window_id": "",
                "source": "crypto_10k",
            })

    cleaned = pd.DataFrame(records, columns=CLEAN_TWEET_COLUMNS)
    if cleaned.empty:
        return cleaned
    cleaned = cleaned.drop_duplicates(subset=["tweet_id", "cryptocurrency"])
    return cleaned.sort_values(["timestamp", "cryptocurrency", "tweet_id"]).reset_index(drop=True)
