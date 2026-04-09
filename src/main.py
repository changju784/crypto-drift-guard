"""
Crypto-Shield — Unified Entry Point

Supports batch experiment execution across drift and hybrid modes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from agents.finbert_agent import FinancialLLMAgent
from agents.rule_based_agent import TradingAgent
from analysis.plots import plot_all
from analysis.reports import (
    compute_consensus,
    compute_drift,
    format_drift_family_summary,
    format_hybrid_family_summary,
    format_pnl_table,
)
from data.data_ingestion import load_baseline
from experiments import build_registry, format_catalog, parse_csv_arg, select_experiments
from simulator import simulate, simulate_df

OUTPUTS_ROOT = Path(__file__).parent.parent / "outputs"
SEP = "=" * 65


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_json(df: pd.DataFrame, path: Path) -> Path:
    _ensure_dir(path.parent)
    df.to_json(path, orient="records", indent=2, date_format="iso")
    return path


def _save_txt(text: str, path: Path) -> Path:
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    return path


def _drift_output_dir(family: str, experiment_id: str) -> Path:
    return _ensure_dir(OUTPUTS_ROOT / "drift" / family / experiment_id)


def _hybrid_output_dir(backend: str, family: str, experiment_id: str) -> Path:
    return _ensure_dir(OUTPUTS_ROOT / "hybrid" / backend / family / experiment_id)


def _drift_family_dir(family: str) -> Path:
    return _ensure_dir(OUTPUTS_ROOT / "drift" / family)


def _hybrid_family_dir(backend: str, family: str) -> Path:
    return _ensure_dir(OUTPUTS_ROOT / "hybrid" / backend / family)


def _pnl_line(label: str, r) -> str:
    return (
        f"  {label:<18} | P&L: {r.total_pnl:>+10.2f} USD"
        f" | Trades: {r.trade_count:>4}"
        f" | Win rate: {r.win_rate*100:>5.1f}%"
    )


def _print_selection(experiments) -> None:
    print("Selected experiments:")
    for spec in experiments:
        tag = " stochastic" if spec.stochastic else ""
        print(f"  - [{spec.family}] {spec.id}: {spec.description}{tag}")


def run_drift(baseline_df: pd.DataFrame, experiments, seed: int):
    print(f"\n{SEP}")
    print("  MODE: DRIFT  (Rule Engine baseline vs mutations)")
    print(SEP)

    agent = TradingAgent()
    baseline_log = agent.run(baseline_df)
    baseline_sim = simulate(baseline_log)
    base_df = baseline_sim.detail

    pnl_rows = []
    family_rows: dict[str, list[dict]] = {}
    for spec in experiments:
        print(f"\n[EXPERIMENT: {spec.id}]  family={spec.family}  {spec.description}  seed={seed}")

        mutant_df = spec.mutate(baseline_df, mode="drift", seed=seed)
        mutant_log = agent.run(mutant_df)
        mutant_sim = simulate(mutant_log)

        drift = compute_drift(base_df, mutant_sim.detail)
        report = drift.summary()
        pnl_text = format_pnl_table([(spec.id, baseline_sim, mutant_sim)], col_a="Baseline", col_b="Mutant")

        print(report)
        print(_pnl_line("Baseline", baseline_sim))
        print(_pnl_line("Mutant", mutant_sim))

        out_dir = _drift_output_dir(spec.family, spec.id)
        _save_json(base_df, out_dir / "traj_baseline.json")
        _save_json(mutant_sim.detail, out_dir / "traj_mutant.json")
        _save_txt(report, out_dir / "drift_report.txt")
        _save_txt(pnl_text, out_dir / "pnl_summary.txt")

        pnl_rows.append((spec.id, baseline_sim, mutant_sim))
        family_rows.setdefault(spec.family, []).append({
            "experiment_id": spec.id,
            "description": spec.description,
            "adr_pct": drift.adr * 100.0,
            "baseline_pnl": baseline_sim.total_pnl,
            "mutant_pnl": mutant_sim.total_pnl,
            "delta_pnl": mutant_sim.total_pnl - baseline_sim.total_pnl,
            "baseline_trades": baseline_sim.trade_count,
            "mutant_trades": mutant_sim.trade_count,
            "baseline_win_rate_pct": baseline_sim.win_rate * 100.0,
            "mutant_win_rate_pct": mutant_sim.win_rate * 100.0,
        })

    table = format_pnl_table(pnl_rows, col_a="Baseline", col_b="Mutant")
    print(f"\n{SEP}\n{table}\n{SEP}")

    for family, rows in family_rows.items():
        summary = format_drift_family_summary(family, rows, seed=seed)
        _save_txt(summary, _drift_family_dir(family) / "summary.txt")
        print(f"  [SUMMARY] drift/{family}/summary.txt")


def run_hybrid(
    baseline_df: pd.DataFrame,
    experiments,
    *,
    sample: int | None,
    backend: str,
    seed: int,
):
    print(f"\n{SEP}")
    print(f"  MODE: HYBRID  (Rule Engine vs {backend.upper()})")
    print(SEP)

    pnl_rows = []
    family_rows: dict[str, list[dict]] = {}
    for spec in experiments:
        mutant_df = spec.mutate(baseline_df, mode="hybrid", seed=seed)
        if sample:
            mutant_df = mutant_df.head(sample)

        print(
            f"\n[EXPERIMENT: {spec.id}]  family={spec.family}  "
            f"{spec.description}  rows={len(mutant_df)}  backend={backend}  seed={seed}"
        )

        # Layer A: Rule Engine
        print("\n[LAYER A - Rule Engine]")
        rule_log = TradingAgent().run(mutant_df)
        rule_sim = simulate(rule_log)
        rule_df = rule_log.to_dataframe()
        print(f"  Actions : {pd.Series([e.action_taken for e in rule_log.entries]).value_counts().to_dict()}")
        print(_pnl_line("Rule Engine", rule_sim))

        # Layer B: FinGPT / FinBERT
        print(f"\n[LAYER B - {backend.upper()}]")
        fg_log = FinancialLLMAgent(backend=backend).run(mutant_df)
        fg_df = fg_log.to_dataframe()
        fg_sim = simulate_df(fg_df, action_col="fingpt_action_taken")
        print(f"  Actions : {fg_df['fingpt_action_taken'].value_counts().to_dict()}")
        print(_pnl_line(backend.upper(), fg_sim))

        print("\n  [Sample reasoning - first 3 non-HOLD]")
        shown = 0
        for e in fg_log.entries:
            if e.fingpt_policy_action != "HOLD":
                print(
                    f"    [{e.fingpt_policy_action}] {e.cryptocurrency}"
                    f" | sentiment={e.input_sentiment:.3f} | label='{e.raw_label}'"
                )
                print(f"    {e.fingpt_reasoning[:110]}")
                shown += 1
                if shown >= 3:
                    break
        if shown == 0:
            print("    (no non-HOLD decisions in this sample)")

        print("\n[CONSENSUS]")
        report = compute_consensus(rule_df, fg_df)
        summary = report.summary()
        pnl_text = format_pnl_table([(spec.id, rule_sim, fg_sim)], col_a="Rule Engine", col_b=backend.upper())
        print(summary)
        print(f"\n{SEP}\n{pnl_text}\n{SEP}")

        out_dir = _hybrid_output_dir(backend, spec.family, spec.id)
        plots_dir = _ensure_dir(out_dir / "plots")
        _save_json(rule_sim.detail, out_dir / "traj_rule.json")
        _save_json(fg_sim.detail, out_dir / "traj_agent.json")
        _save_txt(summary, out_dir / "consensus_report.txt")
        _save_txt(pnl_text, out_dir / "pnl_summary.txt")

        print("\n[PLOTS]")
        for p in plot_all(rule_sim.detail, fg_sim.detail, plots_dir):
            print(f"  {p.name}")

        combined = pd.concat([
            rule_df.add_prefix("rule_").reset_index(drop=True),
            fg_df.reset_index(drop=True),
        ], axis=1)
        combined["pnl_rule"] = rule_sim.detail["pnl"].values
        combined["pnl_fingpt"] = fg_sim.detail["pnl"].values
        combined["actions_agree"] = (
            rule_df["action_taken"].values == fg_df["fingpt_action_taken"].values
        )
        _save_json(combined, out_dir / "combined_log.json")

        pnl_rows.append((spec.id, rule_sim, fg_sim))
        family_rows.setdefault(spec.family, []).append({
            "experiment_id": spec.id,
            "description": spec.description,
            "logic_gap_pct": report.logic_gap_ratio * 100.0,
            "policy_gap_pct": report.policy_gap_ratio * 100.0,
            "rule_pnl": rule_sim.total_pnl,
            "agent_pnl": fg_sim.total_pnl,
            "delta_pnl": fg_sim.total_pnl - rule_sim.total_pnl,
            "rule_trades": rule_sim.trade_count,
            "agent_trades": fg_sim.trade_count,
            "rule_win_rate_pct": rule_sim.win_rate * 100.0,
            "agent_win_rate_pct": fg_sim.win_rate * 100.0,
        })

    table = format_pnl_table(pnl_rows, col_a="Rule Engine", col_b=backend.upper())
    print(f"\n{SEP}\n{table}\n{SEP}")

    for family, rows in family_rows.items():
        summary = format_hybrid_family_summary(family, backend, rows, seed=seed)
        _save_txt(summary, _hybrid_family_dir(backend, family) / "summary.txt")
        print(f"  [SUMMARY] hybrid/{backend}/{family}/summary.txt")


def main():
    parser = argparse.ArgumentParser(description="Crypto-Shield")
    parser.add_argument("--mode", default="all", choices=["drift", "hybrid", "all"])
    parser.add_argument("--sample", type=int, default=None, help="Limit hybrid mode to first N rows")
    parser.add_argument(
        "--backend",
        default="heuristic",
        choices=["heuristic", "finbert"],
        help="Agent backend: heuristic (default), finbert (local ~440MB, CPU ok)",
    )
    parser.add_argument("--families", default=None, help="Comma-separated experiment families")
    parser.add_argument("--experiments", default=None, help="Comma-separated experiment ids")
    parser.add_argument("--list-experiments", action="store_true", help="List all available experiments and exit")
    parser.add_argument("--seed", type=int, default=527, help="Random seed for stochastic experiments")
    args = parser.parse_args()

    registry = build_registry()
    if args.list_experiments:
        print(format_catalog(registry))
        return

    family_names = parse_csv_arg(args.families)
    experiment_ids = parse_csv_arg(args.experiments)
    experiments = select_experiments(
        registry,
        families=family_names,
        experiment_ids=experiment_ids,
    )

    baseline_df = load_baseline()
    print(
        f"Loaded {len(baseline_df)} rows | "
        f"{baseline_df['timestamp'].min()} -> {baseline_df['timestamp'].max()}"
    )
    _print_selection(experiments)

    if args.mode in ("drift", "all"):
        run_drift(baseline_df, experiments, seed=args.seed)

    if args.mode in ("hybrid", "all"):
        run_hybrid(
            baseline_df,
            experiments,
            sample=args.sample,
            backend=args.backend,
            seed=args.seed,
        )

    print(f"\nAll outputs -> {OUTPUTS_ROOT}")


if __name__ == "__main__":
    main()
