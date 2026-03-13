"""
Layer 2: Deterministic Trading Agent
Rule-based, non-stochastic engine with a hard Risk Oracle.
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field
from typing import List


ACTION_BUY  = "BUY"
ACTION_SELL = "SELL"
ACTION_HOLD = "HOLD"

# Policy thresholds
SENTIMENT_BUY_THRESHOLD   =  0.5
SENTIMENT_SELL_THRESHOLD  = -0.3
RSI_BUY_MAX               = 70.0
RSI_SELL_MIN              = 80.0

# Risk Oracle threshold
VOLATILITY_ORACLE_LIMIT   = 90.0


@dataclass
class TrajectoryEntry:
    timestamp:        str
    cryptocurrency:   str
    input_sentiment:  float
    rsi:              float
    volatility:       float
    price_usd:        float
    price_change_pct: float  # price_change_24h_percent
    policy_action:    str    # action before oracle
    action_taken:     str    # final action after oracle
    oracle_triggered: bool


@dataclass
class TrajectoryLog:
    entries: List[TrajectoryEntry] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([e.__dict__ for e in self.entries])


class TradingAgent:
    """Stateless, deterministic rule-based agent."""

    def _policy(self, sentiment: float, rsi: float) -> str:
        if sentiment > SENTIMENT_BUY_THRESHOLD and rsi < RSI_BUY_MAX:
            return ACTION_BUY
        if sentiment < SENTIMENT_SELL_THRESHOLD or rsi > RSI_SELL_MIN:
            return ACTION_SELL
        return ACTION_HOLD

    def _risk_oracle(self, volatility: float, policy_action: str) -> tuple[str, bool]:
        """Return (final_action, oracle_triggered)."""
        if volatility > VOLATILITY_ORACLE_LIMIT:
            return ACTION_HOLD, True
        return policy_action, False

    def run(self, df: pd.DataFrame) -> TrajectoryLog:
        log = TrajectoryLog()
        for _, row in df.iterrows():
            sentiment        = float(row["social_sentiment_score"])
            rsi              = float(row["rsi_technical_indicator"])
            volatility       = float(row["volatility_index"])
            price_usd        = float(row["current_price_usd"])
            price_change_pct = float(row["price_change_24h_percent"])
            timestamp        = str(row["timestamp"])
            crypto           = str(row["cryptocurrency"])

            policy_action            = self._policy(sentiment, rsi)
            final_action, oracle_hit = self._risk_oracle(volatility, policy_action)

            log.entries.append(TrajectoryEntry(
                timestamp        = timestamp,
                cryptocurrency   = crypto,
                input_sentiment  = sentiment,
                rsi              = rsi,
                volatility       = volatility,
                price_usd        = price_usd,
                price_change_pct = price_change_pct,
                policy_action    = policy_action,
                action_taken     = final_action,
                oracle_triggered = oracle_hit,
            ))
        return log
