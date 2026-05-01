from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.tweet_ingestion import clean_crypto10k, detect_cryptocurrencies, normalize_tweet_text


class TweetIngestionTests(unittest.TestCase):
    def test_normalize_tweet_text_collapses_whitespace(self):
        self.assertEqual(
            normalize_tweet_text(" BTC\n\n is\tmoving   fast "),
            "BTC is moving fast",
        )

    def test_detect_cryptocurrencies_from_hashtags_and_text(self):
        self.assertEqual(detect_cryptocurrencies("['BTC']", "hello"), ["BTC"])
        self.assertEqual(detect_cryptocurrencies("[]", "Ethereum looks strong #ETH"), ["ETH"])
        self.assertEqual(
            detect_cryptocurrencies("['Bitcoin', 'ETH']", "mixed tweet"),
            ["BTC", "ETH"],
        )

    def test_clean_crypto10k_filters_and_explodes_btc_eth(self):
        rows = pd.DataFrame([
            {
                "": 0,
                "Date": "2022-11-30 10:00:00+00:00",
                "Username": "alice",
                "Content": "Bitcoin and Ethereum both moving #BTC #ETH",
                "URL": "https://example.com/1",
                "Hashtags": "['BTC', 'ETH']",
            },
            {
                "": 1,
                "Date": "2022-11-30 10:01:00+00:00",
                "Username": "bob",
                "Content": "Random market comment",
                "URL": "https://example.com/2",
                "Hashtags": "['stocks']",
            },
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "tweets.csv"
            rows.to_csv(csv_path, index=False)
            cleaned = clean_crypto10k(csv_path)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned["cryptocurrency"].tolist(), ["BTC", "ETH"])
        self.assertTrue((cleaned["attack_type"] == "baseline").all())
        self.assertFalse(cleaned["is_synthetic"].any())


if __name__ == "__main__":
    unittest.main()
