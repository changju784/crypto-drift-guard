"""
P&L Simulator
Takes a TrajectoryLog and computes per-step and cumulative profit/loss.

Model:
  BUY  → long  position: P&L = +POSITION_SIZE * (price_change_pct / 100)
  SELL → short position: P&L = -POSITION_SIZE * (price_change_pct / 100)
  HOLD →                 P&L = 0
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from agents.rule_based_agent import TrajectoryLog

POSITION_SIZE = 1000.0  # USD notional per trade


DIRECTION = {"BUY": 1, "SELL": -1, "HOLD": 0}


@dataclass
class PnLResult:
    total_pnl:      float
    trade_count:    int
    win_count:      int
    loss_count:      int
    win_rate:       float
    detail:         pd.DataFrame   # per-step rows with pnl + cumulative_pnl


def simulate(log: TrajectoryLog) -> PnLResult:
    return simulate_df(log.to_dataframe(), action_col="action_taken")


def simulate_df(
    df: pd.DataFrame,
    action_col: str = "action_taken",
    price_col: str  = "price_change_pct",
) -> PnLResult:
    """Generic P&L simulation — works on any DataFrame with action + price columns."""
    df = df.copy()

    direction     = df[action_col].map(DIRECTION)
    df["pnl"]     = direction * (df[price_col] / 100.0) * POSITION_SIZE
    df["cum_pnl"] = df["pnl"].cumsum()

    trades   = df[df[action_col] != "HOLD"]
    wins     = (trades["pnl"] > 0).sum()
    losses   = (trades["pnl"] < 0).sum()
    n_trades = len(trades)
    win_rate = wins / n_trades if n_trades > 0 else 0.0

    return PnLResult(
        total_pnl   = float(df["pnl"].sum()),
        trade_count = n_trades,
        win_count   = int(wins),
        loss_count  = int(losses),
        win_rate    = float(win_rate),
        detail      = df,
    )
