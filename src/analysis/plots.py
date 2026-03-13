"""
Comparison Plots: Rule Engine vs FinGPT/FinBERT
================================================
Generates 4 PNG charts into the outputs/ directory:

  1. action_dist     — Grouped bar: action counts for each agent
  2. pnl_curves      — Line: cumulative P&L over time
  3. agreement_matrix— Heatmap: Rule action vs FinGPT action confusion matrix
  4. decision_timeline— Scatter/fill: per-step decisions and agreement flag
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on Windows/servers)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


COLORS = {
    "rule":   "#2196F3",   # blue
    "fingpt": "#FF9800",   # orange
    "agree":  "#4CAF50",   # green
    "disagree":"#F44336",  # red
    "BUY":    "#4CAF50",
    "SELL":   "#F44336",
    "HOLD":   "#9E9E9E",
}


def plot_all(
    rule_df:   pd.DataFrame,    # from rule_sim.detail  (has "action_taken", "pnl", "cum_pnl")
    fingpt_df: pd.DataFrame,    # from fingpt_sim.detail (has "fingpt_action_taken", "pnl", "cum_pnl")
    output_dir: Path,
    label: str = "",
) -> list[Path]:
    """
    Run all 4 plots. Returns list of saved file paths.
    rule_df   must contain: action_taken, pnl, cum_pnl
    fingpt_df must contain: fingpt_action_taken, pnl, cum_pnl
    """
    output_dir = Path(output_dir)
    sfx = f"_{label}" if label else ""
    saved = []

    saved.append(_action_distribution(rule_df, fingpt_df, output_dir, sfx))
    saved.append(_pnl_curves(rule_df, fingpt_df, output_dir, sfx))
    saved.append(_agreement_matrix(rule_df, fingpt_df, output_dir, sfx))
    saved.append(_decision_timeline(rule_df, fingpt_df, output_dir, sfx))

    return saved


# ── Plot 1: Action Distribution ───────────────────────────────────────────────

def _action_distribution(rule_df, fingpt_df, out, sfx) -> Path:
    actions = ["BUY", "SELL", "HOLD"]
    rule_c   = [rule_df["action_taken"].value_counts().get(a, 0) for a in actions]
    fingpt_c = [fingpt_df["fingpt_action_taken"].value_counts().get(a, 0) for a in actions]

    x, w = np.arange(len(actions)), 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    b1 = ax.bar(x - w/2, rule_c,   w, label="Rule Engine",   color=COLORS["rule"],   alpha=0.88)
    b2 = ax.bar(x + w/2, fingpt_c, w, label="FinGPT/FinBERT", color=COLORS["fingpt"], alpha=0.88)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 3, str(int(h)),
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(actions, fontsize=11)
    ax.set_ylabel("Decision count")
    ax.set_title(f"Action Distribution: Rule Engine vs FinGPT{sfx}", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = out / f"plot_action_dist{sfx}.png"
    plt.savefig(path, dpi=130)
    plt.close()
    return path


# ── Plot 2: Cumulative P&L Curves ─────────────────────────────────────────────

def _pnl_curves(rule_df, fingpt_df, out, sfx) -> Path:
    rule_cum   = rule_df["pnl"].cumsum()
    fingpt_cum = fingpt_df["pnl"].cumsum()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(rule_cum.values,   color=COLORS["rule"],   lw=1.8, label="Rule Engine")
    ax.plot(fingpt_cum.values, color=COLORS["fingpt"], lw=1.8, label="FinGPT/FinBERT", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)

    ax.fill_between(range(len(rule_cum)),   rule_cum,   0, alpha=0.06, color=COLORS["rule"])
    ax.fill_between(range(len(fingpt_cum)), fingpt_cum, 0, alpha=0.06, color=COLORS["fingpt"])

    final_rule   = rule_cum.iloc[-1]
    final_fingpt = fingpt_cum.iloc[-1]
    ax.annotate(f"${final_rule:+.0f}",
                xy=(len(rule_cum)-1, final_rule),
                xytext=(-40, 10), textcoords="offset points",
                color=COLORS["rule"], fontsize=9, fontweight="bold")
    ax.annotate(f"${final_fingpt:+.0f}",
                xy=(len(fingpt_cum)-1, final_fingpt),
                xytext=(-40, -18), textcoords="offset points",
                color=COLORS["fingpt"], fontsize=9, fontweight="bold")

    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative P&L (USD)")
    ax.set_title(f"Cumulative P&L: Rule Engine vs FinGPT{sfx}", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()

    path = out / f"plot_pnl_curves{sfx}.png"
    plt.savefig(path, dpi=130)
    plt.close()
    return path


# ── Plot 3: Agreement Confusion Matrix ────────────────────────────────────────

def _agreement_matrix(rule_df, fingpt_df, out, sfx) -> Path:
    actions = ["BUY", "SELL", "HOLD"]
    matrix  = pd.crosstab(
        rule_df["action_taken"].rename("Rule Engine"),
        fingpt_df["fingpt_action_taken"].rename("FinGPT"),
    ).reindex(index=actions, columns=actions, fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix.values, cmap="Blues", vmin=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(actions)))
    ax.set_yticks(range(len(actions)))
    ax.set_xticklabels(actions, fontsize=11)
    ax.set_yticklabels(actions, fontsize=11)
    ax.set_xlabel("FinGPT Action", fontsize=11)
    ax.set_ylabel("Rule Engine Action", fontsize=11)
    ax.set_title(f"Decision Agreement Matrix{sfx}", fontsize=12)

    max_val = matrix.values.max()
    for i in range(len(actions)):
        for j in range(len(actions)):
            v = matrix.values[i, j]
            color = "white" if v > max_val * 0.55 else "black"
            ax.text(j, i, str(v), ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = out / f"plot_agreement_matrix{sfx}.png"
    plt.savefig(path, dpi=130)
    plt.close()
    return path


# ── Plot 4: Decision Timeline ─────────────────────────────────────────────────

def _decision_timeline(rule_df, fingpt_df, out, sfx) -> Path:
    action_num = {"BUY": 1, "HOLD": 0, "SELL": -1}
    rule_s   = rule_df["action_taken"].map(action_num).values
    fingpt_s = fingpt_df["fingpt_action_taken"].map(action_num).values
    agree    = (rule_s == fingpt_s).astype(int)
    steps    = np.arange(len(rule_s))

    fig = plt.figure(figsize=(15, 8))
    gs  = gridspec.GridSpec(3, 1, hspace=0.45)

    # Row 0: Rule Engine
    ax0 = fig.add_subplot(gs[0])
    colors_rule = [COLORS.get(a, "#9E9E9E") for a in rule_df["action_taken"]]
    ax0.scatter(steps, rule_s, c=colors_rule, s=12, alpha=0.7, zorder=2)
    ax0.axhline(0, color="black", lw=0.5, ls="--", alpha=0.3)
    ax0.set_yticks([-1, 0, 1])
    ax0.set_yticklabels(["SELL", "HOLD", "BUY"], fontsize=9)
    ax0.set_ylabel("Rule Engine", fontsize=10)
    ax0.grid(alpha=0.2)
    ax0.set_xlim(0, len(steps))

    # Row 1: FinGPT
    ax1 = fig.add_subplot(gs[1])
    colors_fg = [COLORS.get(a, "#9E9E9E") for a in fingpt_df["fingpt_action_taken"]]
    ax1.scatter(steps, fingpt_s, c=colors_fg, s=12, alpha=0.7, zorder=2)
    ax1.axhline(0, color="black", lw=0.5, ls="--", alpha=0.3)
    ax1.set_yticks([-1, 0, 1])
    ax1.set_yticklabels(["SELL", "HOLD", "BUY"], fontsize=9)
    ax1.set_ylabel("FinGPT", fontsize=10)
    ax1.grid(alpha=0.2)
    ax1.set_xlim(0, len(steps))

    # Row 2: Agreement
    ax2 = fig.add_subplot(gs[2])
    ax2.fill_between(steps, agree, alpha=0.55, color=COLORS["agree"],   label="Agree")
    ax2.fill_between(steps, 1 - agree, alpha=0.45, color=COLORS["disagree"],
                     label="Disagree")
    pct_agree = agree.mean() * 100
    ax2.set_ylabel("Agreement", fontsize=10)
    ax2.set_xlabel("Step", fontsize=10)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["No", "Yes"])
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.2)
    ax2.set_xlim(0, len(steps))
    ax2.set_title(f"Agreement: {pct_agree:.1f}% of steps", fontsize=9)

    fig.suptitle(f"Decision Timeline: Rule Engine vs FinGPT{sfx}", fontsize=12, y=1.01)
    plt.tight_layout()

    path = out / f"plot_decision_timeline{sfx}.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    return path
