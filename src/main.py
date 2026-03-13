"""
Crypto-Shield — Unified Entry Point

Two modes, one script:

  drift   Mutation analysis: Rule Engine on baseline vs mutated data.
          Produces trajectory logs and Logic Drift reports.

  hybrid  Parallel agents: Rule Engine vs FinGPT/FinBERT on k=2 data.
          Produces trajectory logs, consensus report, and comparison plots.

  all     Runs drift then hybrid (default).

Usage
  python src/main.py                              # all modes, FinBERT, full dataset
  python src/main.py --mode drift                 # drift only
  python src/main.py --mode hybrid --sample 100   # hybrid, 100 rows
  python src/main.py --mode hybrid --backend fingpt  # full FinGPT (needs GPU)
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from data.data_ingestion import load_baseline, mutate_intensity, mutate_temporal_jitter
from agents.rule_based_agent import TradingAgent
from agents.fingpt_agent import FinancialLLMAgent
from simulator import simulate, simulate_df
from analysis.reports import compute_drift, compute_consensus, format_pnl_table
from analysis.plots import plot_all

OUTPUTS = Path(__file__).parent.parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

SEP = "=" * 65


# ── Shared helpers ────────────────────────────────────────────────────────────

def _save_json(df: pd.DataFrame, name: str) -> Path:
    p = OUTPUTS / f"{name}.json"
    df.to_json(p, orient="records", indent=2, date_format="iso")
    return p


def _save_txt(text: str, name: str) -> Path:
    p = OUTPUTS / f"{name}.txt"
    p.write_text(text, encoding="utf-8")
    return p


def _pnl_line(label: str, r) -> str:
    return (
        f"  {label:<18} | P&L: {r.total_pnl:>+10.2f} USD"
        f" | Trades: {r.trade_count:>4}"
        f" | Win rate: {r.win_rate*100:>5.1f}%"
    )


# ── Mode 1: Drift ─────────────────────────────────────────────────────────────

def run_drift(baseline_df: pd.DataFrame):
    print(f"\n{SEP}")
    print("  MODE: DRIFT  (Rule Engine baseline vs mutations)")
    print(SEP)

    agent      = TradingAgent()
    baseline_log = agent.run(baseline_df)
    baseline_sim = simulate(baseline_log)
    base_df    = baseline_sim.detail

    experiments = [
        ("intensity_k2",      mutate_intensity(baseline_df, factor=2.0), "sentiment x2.0"),
        ("temporal_jitter_n3", mutate_temporal_jitter(baseline_df, shift=3), "news_impact shifted 3 steps"),
    ]

    pnl_rows = []
    for label, mutant_df, desc in experiments:
        print(f"\n[EXPERIMENT: {label}]  {desc}")

        mutant_log = agent.run(mutant_df)
        mutant_sim = simulate(mutant_log)

        drift   = compute_drift(base_df, mutant_sim.detail)
        report  = drift.summary()
        print(report)
        print(_pnl_line("Baseline", baseline_sim))
        print(_pnl_line("Mutant",   mutant_sim))

        _save_json(base_df,           f"traj_baseline_{label}")
        _save_json(mutant_sim.detail, f"traj_mutant_{label}")
        _save_txt(report,            f"drift_report_{label}")

        pnl_rows.append((label, baseline_sim, mutant_sim))

    table = format_pnl_table(pnl_rows, col_a="Baseline", col_b="Mutant")
    _save_txt(table, "drift_pnl_comparison")
    print(f"\n{SEP}\n{table}\n{SEP}")


# ── Mode 2: Hybrid ────────────────────────────────────────────────────────────

def run_hybrid(baseline_df: pd.DataFrame, sample: int | None, backend: str):
    print(f"\n{SEP}")
    print(f"  MODE: HYBRID  (Rule Engine vs {backend.upper()})")
    print(SEP)

    mutant_df = mutate_intensity(baseline_df, factor=2.0)
    if sample:
        mutant_df = mutant_df.head(sample)

    label = f"k2_{'s'+str(sample) if sample else 'full'}_{backend}"
    print(f"\n[INPUT]  rows={len(mutant_df)}  mutation=sentiment x2.0  backend={backend}")

    # Layer A: Rule Engine
    print("\n[LAYER A - Rule Engine]")
    rule_log = TradingAgent().run(mutant_df)
    rule_sim = simulate(rule_log)
    rule_df  = rule_log.to_dataframe()
    print(f"  Actions : {pd.Series([e.action_taken for e in rule_log.entries]).value_counts().to_dict()}")
    print(_pnl_line("Rule Engine", rule_sim))
    _save_json(rule_sim.detail, f"traj_rule_{label}")

    # Layer B: FinGPT / FinBERT
    print(f"\n[LAYER B - {backend.upper()}]")
    fg_log = FinancialLLMAgent(backend=backend).run(mutant_df)
    fg_df  = fg_log.to_dataframe()
    fg_sim = simulate_df(fg_df, action_col="fingpt_action_taken")
    print(f"  Actions : {fg_df['fingpt_action_taken'].value_counts().to_dict()}")
    print(_pnl_line(backend.upper(), fg_sim))

    # Sample reasoning
    print("\n  [Sample reasoning - first 3 non-HOLD]")
    shown = 0
    for e in fg_log.entries:
        if e.fingpt_policy_action != "HOLD":
            print(f"    [{e.fingpt_policy_action}] {e.cryptocurrency}"
                  f" | sentiment={e.input_sentiment:.3f} | label='{e.raw_label}'")
            print(f"    {e.fingpt_reasoning[:110]}")
            shown += 1
            if shown >= 3:
                break
    if shown == 0:
        print("    (no non-HOLD decisions in this sample)")

    _save_json(fg_sim.detail, f"traj_fingpt_{label}")

    # Consensus
    print("\n[CONSENSUS]")
    report = compute_consensus(rule_df, fg_df)
    print(report.summary())
    _save_txt(report.summary(), f"consensus_report_{label}")

    # P&L table
    table = format_pnl_table([(label, rule_sim, fg_sim)],
                              col_a="Rule Engine", col_b=backend.upper())
    _save_txt(table, f"hybrid_pnl_{label}")
    print(f"\n{SEP}\n{table}\n{SEP}")

    # Plots
    print("\n[PLOTS]")
    for p in plot_all(rule_sim.detail, fg_sim.detail, OUTPUTS, label=label):
        print(f"  {p.name}")

    # Combined log
    combined = pd.concat([
        rule_df.add_prefix("rule_").reset_index(drop=True),
        fg_df.reset_index(drop=True),
    ], axis=1)
    combined["pnl_rule"]      = rule_sim.detail["pnl"].values
    combined["pnl_fingpt"]    = fg_sim.detail["pnl"].values
    combined["actions_agree"] = (rule_df["action_taken"].values == fg_df["fingpt_action_taken"].values)
    _save_json(combined, f"combined_log_{label}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Crypto-Shield")
    parser.add_argument("--mode",    default="all",     choices=["drift", "hybrid", "all"])
    parser.add_argument("--sample",  type=int,          default=None,
                        help="Limit hybrid mode to first N rows")
    parser.add_argument("--backend", default="heuristic",
                        choices=["heuristic", "finbert"],
                        help="Agent backend: heuristic (default), finbert (local ~440MB, CPU ok)")
    args = parser.parse_args()

    baseline_df = load_baseline()
    print(f"Loaded {len(baseline_df)} rows | "
          f"{baseline_df['timestamp'].min()} -> {baseline_df['timestamp'].max()}")

    if args.mode in ("drift", "all"):
        run_drift(baseline_df)

    if args.mode in ("hybrid", "all"):
        run_hybrid(baseline_df, sample=args.sample, backend=args.backend)

    print(f"\nAll outputs -> {OUTPUTS}")


if __name__ == "__main__":
    main()
