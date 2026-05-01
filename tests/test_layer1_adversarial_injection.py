from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.adversarial_injection import (
    build_injection_plan,
    generate_adversarial_tweets,
    merge_scored_synthetic,
    required_injections,
)


class AdversarialInjectionTests(unittest.TestCase):
    def test_required_injections_reaches_target_ratio(self):
        self.assertEqual(required_injections(10, 3, 0.5), 4)
        self.assertEqual(required_injections(10, 7, 0.5), 0)

    def test_generate_positive_pump_tweets_for_neutral_window(self):
        windows = pd.DataFrame([{
            "window_id": "BTC_20221130T100000Z",
            "window_start": pd.Timestamp("2022-11-30T10:00:00Z"),
            "window_end": pd.Timestamp("2022-11-30T10:05:00Z"),
            "cryptocurrency": "BTC",
            "tweet_count": 20,
            "positive_count": 4,
            "neutral_count": 12,
            "negative_count": 4,
            "positive_ratio": 0.20,
            "neutral_ratio": 0.60,
            "negative_ratio": 0.20,
            "mean_sentiment_score": 0.0,
            "sentiment_intensity": 0.1,
            "regime": "neutral",
        }])
        plan = build_injection_plan(
            windows,
            attack_type="positive_pump",
            target_ratio=0.7,
        )
        synthetic = generate_adversarial_tweets(
            windows,
            attack_type="positive_pump",
            target_ratio=0.7,
            seed=1,
        )

        self.assertEqual(int(plan.loc[0, "synthetic_needed"]), 34)
        self.assertEqual(len(synthetic), 34)
        self.assertTrue(synthetic["is_synthetic"].all())
        self.assertEqual(set(synthetic["attack_type"]), {"positive_pump"})

    def test_merge_requires_scored_synthetic_rows(self):
        baseline = pd.DataFrame({"tweet_id": ["a"], "timestamp": ["2022-11-30T10:00:00Z"]})
        synthetic = pd.DataFrame({"tweet_id": ["b"], "timestamp": ["2022-11-30T10:01:00Z"]})
        with self.assertRaises(ValueError):
            merge_scored_synthetic(baseline, synthetic)


if __name__ == "__main__":
    unittest.main()
