"""
Reports — unified analysis for both modes.

DriftReport    : mutation analysis (baseline vs mutant Rule Engine)
ConsensusReport: hybrid analysis (Rule Engine vs FinGPT)
format_pnl_table: shared P&L comparison formatter
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

from simulator import PnLResult


# ── Drift Report (Milestone 1) ────────────────────────────────────────────────

@dataclass
class DriftReport:
    total_steps:          int
    mismatched_steps:     int
    adr:                  float   # Action Difference Ratio
    oracle_violations:    int
    oracle_violation_rate:float

    def summary(self) -> str:
        return "\n".join([
            "=== Logic Drift Report ===",
            f"Total steps            : {self.total_steps}",
            f"Mismatched steps       : {self.mismatched_steps}",
            f"ADR                    : {self.adr:.4f}  ({self.adr*100:.2f}%)",
            f"Oracle violations      : {self.oracle_violations}",
            f"Oracle violation rate  : {self.oracle_violation_rate:.4f}  ({self.oracle_violation_rate*100:.2f}%)",
        ])


def compute_drift(baseline_df: pd.DataFrame, mutant_df: pd.DataFrame) -> DriftReport:
    """Compare two Rule Engine trajectory DataFrames."""
    if len(baseline_df) != len(mutant_df):
        raise ValueError(f"Length mismatch: {len(baseline_df)} vs {len(mutant_df)}")
    total      = len(baseline_df)
    mismatches = (baseline_df["action_taken"] != mutant_df["action_taken"]).sum()
    violations = (
        (baseline_df["oracle_triggered"] == True) &
        (mutant_df["action_taken"] != "HOLD")
    ).sum()
    return DriftReport(
        total_steps           = total,
        mismatched_steps      = int(mismatches),
        adr                   = mismatches / total if total else 0.0,
        oracle_violations     = int(violations),
        oracle_violation_rate = violations / total if total else 0.0,
    )


# ── Consensus Report (Milestone 2) ───────────────────────────────────────────

@dataclass
class ConsensusReport:
    total_steps:          int
    logic_gap_steps:      int
    logic_gap_ratio:      float   # post-oracle disagreement %
    policy_gap_steps:     int
    policy_gap_ratio:     float   # pre-oracle disagreement %
    oracle_overrides:     int
    oracle_override_rate: float
    rule_dist:            dict
    agent_dist:           dict
    gap_transitions:      dict

    def summary(self) -> str:
        lines = [
            "=== Consensus & Gap Report ===",
            f"Total steps              : {self.total_steps}",
            "",
            f"Logic Gap (post-oracle)  : {self.logic_gap_steps}  ({self.logic_gap_ratio*100:.2f}%)",
            f"Policy Gap (pre-oracle)  : {self.policy_gap_steps}  ({self.policy_gap_ratio*100:.2f}%)",
            f"Oracle Override Rate     : {self.oracle_overrides}  ({self.oracle_override_rate*100:.2f}%)",
            "",
            f"Rule Engine dist : {self.rule_dist}",
            f"FinGPT dist      : {self.agent_dist}",
            "",
            "Disagreement breakdown (Rule -> FinGPT):",
        ]
        for t, c in sorted(self.gap_transitions.items(), key=lambda x: -x[1]):
            lines.append(f"  {t:<20} : {c}")
        return "\n".join(lines)


def compute_consensus(
    rule_df:  pd.DataFrame,
    agent_df: pd.DataFrame,
    rule_action:  str = "action_taken",
    rule_policy:  str = "policy_action",
    agent_action: str = "fingpt_action_taken",
    agent_policy: str = "fingpt_policy_action",
    agent_oracle: str = "fingpt_oracle_triggered",
) -> ConsensusReport:
    if len(rule_df) != len(agent_df):
        raise ValueError(f"Length mismatch: {len(rule_df)} vs {len(agent_df)}")

    total = len(rule_df)
    ra, aa = rule_df[rule_action].values, agent_df[agent_action].values
    rp, ap = rule_df[rule_policy].values, agent_df[agent_policy].values

    gap_mask    = ra != aa
    policy_mask = rp != ap
    overrides   = int(
        (agent_df[agent_policy].isin(["BUY", "SELL"]) & agent_df[agent_oracle]).sum()
    )
    transitions = pd.Series(
        ra[gap_mask] + " -> " + aa[gap_mask]
    ).value_counts().to_dict()

    return ConsensusReport(
        total_steps          = total,
        logic_gap_steps      = int(gap_mask.sum()),
        logic_gap_ratio      = gap_mask.sum() / total if total else 0.0,
        policy_gap_steps     = int(policy_mask.sum()),
        policy_gap_ratio     = policy_mask.sum() / total if total else 0.0,
        oracle_overrides     = overrides,
        oracle_override_rate = overrides / total if total else 0.0,
        rule_dist            = rule_df[rule_action].value_counts().to_dict(),
        agent_dist           = agent_df[agent_action].value_counts().to_dict(),
        gap_transitions      = transitions,
    )


# ── Shared P&L Table ──────────────────────────────────────────────────────────

def format_pnl_table(
    rows: List[Tuple[str, PnLResult, PnLResult]],
    col_a: str = "Baseline",
    col_b: str = "Mutant",
) -> str:
    header = (
        f"{'Experiment':<28} {col_a:>16} {col_b:>16} {'Delta':>16}\n"
        + "-" * 78
    )
    lines = [f"=== P&L Comparison: {col_a} vs {col_b} ===", header]
    for label, r_a, r_b in rows:
        delta = r_b.total_pnl - r_a.total_pnl
        tag   = "B better" if delta > 0 else ("A better" if delta < 0 else "Tied")
        lines.append(
            f"{label:<28} {r_a.total_pnl:>+16.2f} {r_b.total_pnl:>+16.2f}"
            f" {delta:>+10.2f} ({tag})"
        )
    return "\n".join(lines)
