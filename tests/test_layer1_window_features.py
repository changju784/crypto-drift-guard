from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.window_features import aggregate_window_features, classify_sentiment_regimes


class WindowFeatureTests(unittest.TestCase):
    def test_aggregate_window_features_counts_labels_and_ratios(self):
        df = pd.DataFrame([
            {
                "timestamp": "2022-11-30T10:00:00Z",
                "cryptocurrency": "BTC",
                "finbert_label": "positive",
                "sentiment_score": 0.8,
                "sentiment_confidence": 0.9,
                "is_synthetic": False,
            },
            {
                "timestamp": "2022-11-30T10:01:00Z",
                "cryptocurrency": "BTC",
                "finbert_label": "negative",
                "sentiment_score": -0.6,
                "sentiment_confidence": 0.8,
                "is_synthetic": True,
            },
            {
                "timestamp": "2022-11-30T10:04:00Z",
                "cryptocurrency": "BTC",
                "finbert_label": "neutral",
                "sentiment_score": 0.0,
                "sentiment_confidence": 0.7,
                "is_synthetic": False,
            },
        ])
        windows = aggregate_window_features(df, freq="5min")
        row = windows.iloc[0]

        self.assertEqual(int(row["tweet_count"]), 3)
        self.assertEqual(int(row["synthetic_count"]), 1)
        self.assertAlmostEqual(float(row["positive_ratio"]), 1 / 3)
        self.assertAlmostEqual(float(row["negative_ratio"]), 1 / 3)

    def test_classify_sentiment_regimes_marks_expected_categories(self):
        df = pd.DataFrame([
            {
                "tweet_count": 20,
                "positive_ratio": 0.30,
                "negative_ratio": 0.30,
                "mean_sentiment_score": 0.02,
            },
            {
                "tweet_count": 20,
                "positive_ratio": 0.70,
                "negative_ratio": 0.10,
                "mean_sentiment_score": 0.50,
            },
            {
                "tweet_count": 20,
                "positive_ratio": 0.40,
                "negative_ratio": 0.40,
                "mean_sentiment_score": 0.00,
            },
        ])
        regimes = classify_sentiment_regimes(df)["regime"].tolist()
        self.assertEqual(regimes, ["neutral", "positive_skew", "high_conflict"])


if __name__ == "__main__":
    unittest.main()
