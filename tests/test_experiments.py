from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import main as app_main
from data.data_ingestion import (
    mutate_sentiment_flip,
    mutate_sentiment_fused_eq,
    mutate_sentiment_scale,
    mutate_temporal_lag_by_time,
)
from experiments import build_registry, select_experiments


def sample_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "timestamp": pd.Timestamp("2025-06-04 10:00"),
            "cryptocurrency": "Bitcoin",
            "current_price_usd": 100.0,
            "price_change_24h_percent": 1.0,
            "trading_volume_24h": 10.0,
            "market_cap_usd": 1000.0,
            "social_sentiment_score": 0.2,
            "news_sentiment_score": 0.6,
            "news_impact_score": 1.0,
            "social_mentions_count": 10,
            "fear_greed_index": 60.0,
            "volatility_index": 20.0,
            "rsi_technical_indicator": 40.0,
            "prediction_confidence": 75.0,
        },
        {
            "timestamp": pd.Timestamp("2025-06-04 10:30"),
            "cryptocurrency": "Ethereum",
            "current_price_usd": 200.0,
            "price_change_24h_percent": -2.0,
            "trading_volume_24h": 20.0,
            "market_cap_usd": 2000.0,
            "social_sentiment_score": -0.3,
            "news_sentiment_score": 0.1,
            "news_impact_score": 2.0,
            "social_mentions_count": 20,
            "fear_greed_index": 35.0,
            "volatility_index": 30.0,
            "rsi_technical_indicator": 55.0,
            "prediction_confidence": 80.0,
        },
        {
            "timestamp": pd.Timestamp("2025-06-04 11:00"),
            "cryptocurrency": "Bitcoin",
            "current_price_usd": 101.0,
            "price_change_24h_percent": 3.0,
            "trading_volume_24h": 11.0,
            "market_cap_usd": 1001.0,
            "social_sentiment_score": 0.8,
            "news_sentiment_score": 0.4,
            "news_impact_score": 3.0,
            "social_mentions_count": 11,
            "fear_greed_index": 65.0,
            "volatility_index": 25.0,
            "rsi_technical_indicator": 45.0,
            "prediction_confidence": 70.0,
        },
        {
            "timestamp": pd.Timestamp("2025-06-04 11:30"),
            "cryptocurrency": "Ethereum",
            "current_price_usd": 201.0,
            "price_change_24h_percent": -4.0,
            "trading_volume_24h": 21.0,
            "market_cap_usd": 2001.0,
            "social_sentiment_score": -0.7,
            "news_sentiment_score": -0.2,
            "news_impact_score": 4.0,
            "social_mentions_count": 21,
            "fear_greed_index": 25.0,
            "volatility_index": 32.0,
            "rsi_technical_indicator": 70.0,
            "prediction_confidence": 79.0,
        },
        {
            "timestamp": pd.Timestamp("2025-06-04 13:30"),
            "cryptocurrency": "Bitcoin",
            "current_price_usd": 103.0,
            "price_change_24h_percent": 5.0,
            "trading_volume_24h": 12.0,
            "market_cap_usd": 1002.0,
            "social_sentiment_score": 0.1,
            "news_sentiment_score": 0.0,
            "news_impact_score": 5.0,
            "social_mentions_count": 12,
            "fear_greed_index": 70.0,
            "volatility_index": 29.0,
            "rsi_technical_indicator": 48.0,
            "prediction_confidence": 81.0,
        },
        {
            "timestamp": pd.Timestamp("2025-06-04 13:45"),
            "cryptocurrency": "Ethereum",
            "current_price_usd": 198.0,
            "price_change_24h_percent": -6.0,
            "trading_volume_24h": 22.0,
            "market_cap_usd": 1998.0,
            "social_sentiment_score": -0.2,
            "news_sentiment_score": 0.3,
            "news_impact_score": 6.0,
            "social_mentions_count": 22,
            "fear_greed_index": 30.0,
            "volatility_index": 35.0,
            "rsi_technical_indicator": 72.0,
            "prediction_confidence": 82.0,
        },
    ])


class ExperimentSelectionTests(unittest.TestCase):
    def test_experiment_ids_override_family_selection(self):
        registry = build_registry()
        chosen = select_experiments(
            registry,
            families=["sentiment"],
            experiment_ids=["temporal_lag_2h"],
        )
        self.assertEqual([spec.id for spec in chosen], ["temporal_lag_2h"])

    def test_family_selection_returns_all_family_experiments(self):
        registry = build_registry()
        chosen = select_experiments(registry, families=["sentiment"])
        self.assertEqual(
            [spec.id for spec in chosen],
            [
                "sentiment_scale_k05",
                "sentiment_scale_k15",
                "sentiment_scale_k20",
                "sentiment_flip_p10",
                "sentiment_flip_p20",
                "sentiment_flip_p30",
                "sentiment_fused_eq",
            ],
        )


class MutationTests(unittest.TestCase):
    def test_sentiment_scale_clips_to_unit_interval(self):
        df = sample_df()
        scaled = mutate_sentiment_scale(df, factor=2.0)
        self.assertEqual(float(scaled.loc[2, "social_sentiment_score"]), 1.0)
        self.assertEqual(float(scaled.loc[3, "social_sentiment_score"]), -1.0)

    def test_sentiment_flip_is_seed_reproducible(self):
        df = sample_df()
        flipped_a = mutate_sentiment_flip(df, p=0.5, seed=527)
        flipped_b = mutate_sentiment_flip(df, p=0.5, seed=527)
        pd.testing.assert_series_equal(
            flipped_a["social_sentiment_score"],
            flipped_b["social_sentiment_score"],
        )

    def test_sentiment_fused_eq_overwrites_effective_sentiment(self):
        df = sample_df()
        fused = mutate_sentiment_fused_eq(df)
        self.assertAlmostEqual(float(fused.loc[0, "social_sentiment_score"]), 0.4)
        self.assertAlmostEqual(float(fused.loc[3, "social_sentiment_score"]), -0.45)

    def test_temporal_lag_by_time_uses_same_coin_history_only(self):
        df = sample_df()
        lagged = mutate_temporal_lag_by_time(
            df,
            lag=pd.Timedelta("2h"),
            columns=["social_sentiment_score", "rsi_technical_indicator"],
        )
        self.assertEqual(float(lagged.loc[4, "social_sentiment_score"]), 0.8)
        self.assertEqual(float(lagged.loc[4, "rsi_technical_indicator"]), 45.0)
        self.assertEqual(float(lagged.loc[5, "social_sentiment_score"]), -0.7)
        self.assertEqual(float(lagged.loc[5, "rsi_technical_indicator"]), 70.0)

    def test_temporal_lag_without_valid_history_keeps_original_value(self):
        df = sample_df()
        lagged = mutate_temporal_lag_by_time(
            df,
            lag=pd.Timedelta("2h"),
            columns=["social_sentiment_score"],
        )
        self.assertEqual(float(lagged.loc[0, "social_sentiment_score"]), 0.2)
        self.assertEqual(float(lagged.loc[1, "social_sentiment_score"]), -0.3)


class OutputLayoutTests(unittest.TestCase):
    def test_drift_and_hybrid_write_family_summaries_and_experiment_dirs(self):
        registry = build_registry()
        experiments = select_experiments(
            registry,
            experiment_ids=["sentiment_scale_k05", "sentiment_fused_eq"],
        )
        df = sample_df()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with patch.object(app_main, "OUTPUTS_ROOT", tmp_path):
                with redirect_stdout(io.StringIO()):
                    app_main.run_drift(df, experiments, seed=527)
                    app_main.run_hybrid(df, experiments, sample=None, backend="heuristic", seed=527)

            drift_summary = tmp_path / "drift" / "sentiment" / "summary.txt"
            hybrid_summary = tmp_path / "hybrid" / "heuristic" / "sentiment" / "summary.txt"
            self.assertTrue(drift_summary.exists())
            self.assertTrue(hybrid_summary.exists())

            drift_text = drift_summary.read_text(encoding="utf-8")
            hybrid_text = hybrid_summary.read_text(encoding="utf-8")
            self.assertIn("sentiment_scale_k05", drift_text)
            self.assertIn("sentiment_fused_eq", drift_text)
            self.assertIn("sentiment_scale_k05", hybrid_text)
            self.assertIn("sentiment_fused_eq", hybrid_text)

            for experiment_id in ("sentiment_scale_k05", "sentiment_fused_eq"):
                drift_dir = tmp_path / "drift" / "sentiment" / experiment_id
                hybrid_dir = tmp_path / "hybrid" / "heuristic" / "sentiment" / experiment_id
                self.assertTrue((drift_dir / "traj_baseline.json").exists())
                self.assertTrue((drift_dir / "traj_mutant.json").exists())
                self.assertTrue((drift_dir / "drift_report.txt").exists())
                self.assertTrue((drift_dir / "pnl_summary.txt").exists())
                self.assertTrue((hybrid_dir / "traj_rule.json").exists())
                self.assertTrue((hybrid_dir / "traj_agent.json").exists())
                self.assertTrue((hybrid_dir / "combined_log.json").exists())
                self.assertTrue((hybrid_dir / "consensus_report.txt").exists())
                self.assertTrue((hybrid_dir / "pnl_summary.txt").exists())
                self.assertTrue((hybrid_dir / "plots" / "plot_action_dist.png").exists())


if __name__ == "__main__":
    unittest.main()
