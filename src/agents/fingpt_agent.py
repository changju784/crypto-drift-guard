"""
Financial LLM Agent
===================
Two backends:

  "heuristic"  (default — no ML, no API key, always works)
      Weighted multi-factor composite score across sentiment, RSI,
      price momentum and fear/greed. Produces human-readable reasoning.

  "finbert"
      Model  : ProsusAI/finbert  (~440 MB, CPU-friendly)
      Output : positive / negative / neutral → BUY / SELL / HOLD

Both backends apply the Risk Oracle post-inference (volatility > 90 → HOLD).
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple

VOLATILITY_ORACLE_LIMIT = 90.0

SENTIMENT_MAP: dict[str, str] = {
    "positive": "BUY",
    "negative": "SELL",
    "neutral":  "HOLD",
}


# ── Prompt builder ────────────────────────────────────────────────────────────

def _finbert_text(row: pd.Series) -> str:
    s    = float(row["social_sentiment_score"])
    rsi  = float(row["rsi_technical_indicator"])
    pct  = float(row["price_change_24h_percent"])
    fg   = float(row["fear_greed_index"])
    sent = "positive" if s > 0.05 else "negative" if s < -0.05 else "neutral"
    rsi_desc = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
    move = "up" if pct > 0 else "down"
    return (
        f"{row['cryptocurrency']} moved {move} {abs(pct):.1f}% with {sent} social sentiment. "
        f"RSI is {rsi_desc} at {rsi:.1f}. "
        f"Fear and greed index is {fg:.0f}."
    )


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FinGPTEntry:
    timestamp:               str
    cryptocurrency:          str
    input_sentiment:         float
    rsi:                     float
    volatility:              float
    price_usd:               float
    price_change_pct:        float
    raw_label:               str
    fingpt_reasoning:        str
    fingpt_policy_action:    str
    fingpt_action_taken:     str
    fingpt_oracle_triggered: bool


@dataclass
class FinGPTLog:
    entries: List[FinGPTEntry] = field(default_factory=list)
    backend: str = "heuristic"

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([e.__dict__ for e in self.entries])


# ── Risk oracle ───────────────────────────────────────────────────────────────

def _oracle(volatility: float, action: str) -> Tuple[str, bool]:
    if volatility > VOLATILITY_ORACLE_LIMIT:
        return "HOLD", True
    return action, False


# ── FinancialLLMAgent ─────────────────────────────────────────────────────────

class FinancialLLMAgent:
    """
    Usage:
        agent = FinancialLLMAgent(backend="heuristic")  # default, no ML
        agent = FinancialLLMAgent(backend="finbert")    # local ~440 MB, CPU ok
        log   = agent.run(df)
    """

    def __init__(self, backend: str = "heuristic"):
        if backend not in ("heuristic", "finbert"):
            raise ValueError(f"backend must be 'heuristic' or 'finbert', got '{backend}'")
        self.backend = backend
        self._pipe   = None
        self._load()

    def _load(self):
        if self.backend == "finbert":
            self._load_finbert()

    def _load_finbert(self):
        from transformers import pipeline as hf_pipeline
        try:
            import torch
            device = 0 if torch.cuda.is_available() else -1
        except ImportError:
            device = -1
        print("  [FinBERT] Loading ProsusAI/finbert ...")
        self._pipe = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            device=device,
            truncation=True,
            max_length=512,
        )
        print("  [FinBERT] Ready")

    # ── Inference ─────────────────────────────────────────────────────────────

    def _infer_finbert(self, row: pd.Series) -> Tuple[str, str]:
        text   = _finbert_text(row)
        result = self._pipe(text)[0]
        label  = result["label"].lower()
        conf   = result["score"]
        return label, f"FinBERT: {label} (conf={conf:.2f}) | {text}"

    def _infer_heuristic(self, row: pd.Series) -> Tuple[str, str]:
        s   = float(row["social_sentiment_score"])
        rsi = float(row["rsi_technical_indicator"])
        pct = float(row["price_change_24h_percent"])
        fg  = float(row["fear_greed_index"])

        rsi_signal = -(rsi - 50) / 50
        mom_signal = max(-1.0, min(1.0, pct / 10.0))
        fg_signal  = (fg - 50) / 50

        composite = 0.50*s + 0.25*rsi_signal + 0.15*mom_signal + 0.10*fg_signal

        if composite > 0.30:
            label = "positive"
        elif composite < -0.20:
            label = "negative"
        else:
            label = "neutral"

        reasoning = (
            f"Composite={composite:+.3f} | "
            f"sentiment={s:+.3f}(x0.50) "
            f"rsi_factor={rsi_signal:+.3f}(x0.25) "
            f"momentum={mom_signal:+.3f}(x0.15) "
            f"fear_greed={fg_signal:+.3f}(x0.10) "
            f"-> {label}"
        )
        return label, reasoning

    def _predict_row(self, row: pd.Series) -> FinGPTEntry:
        vol = float(row["volatility_index"])

        if self.backend == "finbert":
            raw_label, reasoning = self._infer_finbert(row)
        else:
            raw_label, reasoning = self._infer_heuristic(row)

        policy = SENTIMENT_MAP.get(raw_label.lower(), "HOLD")
        action, oracle_hit = _oracle(vol, policy)

        return FinGPTEntry(
            timestamp               = str(row["timestamp"]),
            cryptocurrency          = str(row["cryptocurrency"]),
            input_sentiment         = float(row["social_sentiment_score"]),
            rsi                     = float(row["rsi_technical_indicator"]),
            volatility              = vol,
            price_usd               = float(row["current_price_usd"]),
            price_change_pct        = float(row["price_change_24h_percent"]),
            raw_label               = raw_label,
            fingpt_reasoning        = reasoning,
            fingpt_policy_action    = policy,
            fingpt_action_taken     = action,
            fingpt_oracle_triggered = oracle_hit,
        )

    def run(self, df: pd.DataFrame) -> FinGPTLog:
        log  = FinGPTLog(backend=self.backend)
        rows = list(df.iterrows())
        n    = len(rows)

        print(f"  [{self.backend.upper()}] Running on {n} rows ...")
        for i, (_, row) in enumerate(rows):
            log.entries.append(self._predict_row(row))
            if (i + 1) % 200 == 0 or (i + 1) == n:
                print(f"    {i+1}/{n} rows processed")

        return log
