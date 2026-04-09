# Crypto Drift Guard

A framework for measuring how cryptocurrency trading agents change under controlled input mutations.

The project now supports two execution modes over a shared experiment registry:

- `drift`: compare the same Rule Engine on baseline vs mutated inputs
- `hybrid`: compare the Rule Engine and the financial agent on the same mutated input

## Install

```bash
pip install -r requirements.txt
```

For local verification in `heuristic` mode, only `pandas`, `numpy`, and `matplotlib` are required. `finbert` additionally needs the Hugging Face dependencies in `requirements.txt`.

## Run

List all registered experiments:

```bash
python src/main.py --list-experiments
```

Run every experiment in both families:

```bash
python src/main.py --mode drift
python src/main.py --mode hybrid --backend heuristic
```

Run one full family:

```bash
python src/main.py --mode drift --families sentiment
python src/main.py --mode drift --families temporal
python src/main.py --mode hybrid --families sentiment --backend heuristic
```

Run a single experiment:

```bash
python src/main.py --mode drift --experiments temporal_lag_2h
python src/main.py --mode hybrid --experiments sentiment_fused_eq --backend finbert
```

Useful flags:

- `--families sentiment,temporal`
- `--experiments sentiment_scale_k20,temporal_lag_6h`
- `--seed 527`
- `--sample 100`
- `--backend heuristic|finbert`

Selection precedence is fixed:

1. `--experiments`
2. `--families`
3. otherwise run all registered experiments

## Experiment Families

### Sentiment

- `sentiment_scale_k05`
- `sentiment_scale_k15`
- `sentiment_scale_k20`
- `sentiment_flip_p10`
- `sentiment_flip_p20`
- `sentiment_flip_p30`
- `sentiment_fused_eq`

Definitions:

- `sentiment_scale`: multiply `social_sentiment_score` by `k`, clipped to `[-1, 1]`
- `sentiment_flip`: flip the sign of `social_sentiment_score` with Bernoulli probability `p`
- `sentiment_fused_eq`: replace the effective `social_sentiment_score` with `0.5 * social_sentiment_score + 0.5 * news_sentiment_score`

### Temporal

- `temporal_lag_30m`
- `temporal_lag_2h`
- `temporal_lag_6h`
- `temporal_lag_12h`

Temporal lag is implemented as a time-aware stale-feed mutation:

- sort by `timestamp`
- group by `cryptocurrency`
- for each row, look up the most recent row from the same coin with `timestamp <= current_timestamp - lag`
- replace only the consumed signal columns
- if no valid lagged row exists, keep the original value

Consumed columns:

- `drift` / Rule Engine:
  - `social_sentiment_score`
  - `rsi_technical_indicator`
  - `volatility_index`
- `hybrid`:
  - the same three columns
  - `price_change_24h_percent`
  - `fear_greed_index`

## Results Snapshot

The tables below summarize the current outputs generated from this repo after the mutation framework refactor.

Result scope:

- `drift` tables: full dataset, `2063` rows
- `hybrid` tables: `backend=heuristic`, `sample=100`
- `seed=527`

### Drift: Sentiment Family

Command:

```bash
python src/main.py --mode drift --families sentiment
```

| Experiment | ADR | Baseline P&L | Mutant P&L | Delta | Baseline Trades | Mutant Trades | Baseline Win Rate | Mutant Win Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sentiment_scale_k05` | 11.10% | -369.90 | -147.80 | +222.10 | 296 | 67 | 46.6% | 50.7% |
| `sentiment_scale_k15` | 12.02% | -369.90 | -732.10 | -362.20 | 296 | 544 | 46.6% | 48.2% |
| `sentiment_scale_k20` | 19.29% | -369.90 | -1801.30 | -1431.40 | 296 | 694 | 46.6% | 47.8% |
| `sentiment_flip_p10` | 1.65% | -369.90 | -159.20 | +210.70 | 296 | 303 | 46.6% | 46.9% |
| `sentiment_flip_p20` | 3.64% | -369.90 | -221.50 | +148.40 | 296 | 301 | 46.6% | 46.8% |
| `sentiment_flip_p30` | 5.72% | -369.90 | -740.60 | -370.70 | 296 | 313 | 46.6% | 46.0% |
| `sentiment_fused_eq` | 8.24% | -369.90 | -982.00 | -612.10 | 296 | 264 | 46.6% | 46.6% |

Analysis:

- Scaling sentiment upward makes the Rule Engine increasingly unstable: `k=2.0` produces the largest drift and the worst P&L.
- Mild random sign flips (`p=0.10`, `p=0.20`) change behavior only slightly and do not hurt P&L much, which suggests limited sensitivity to low-rate label noise.
- Fusing `social` and `news` sentiment causes meaningful drift (`ADR 8.24%`) and worsens P&L, showing that the effective sentiment definition matters even without increasing trade count.

### Drift: Temporal Family

Command:

```bash
python src/main.py --mode drift --families temporal
```

| Experiment | ADR | Baseline P&L | Mutant P&L | Delta | Baseline Trades | Mutant Trades | Baseline Win Rate | Mutant Win Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `temporal_lag_30m` | 23.70% | -369.90 | +629.30 | +999.20 | 296 | 285 | 46.6% | 52.3% |
| `temporal_lag_2h` | 24.92% | -369.90 | +1294.10 | +1664.00 | 296 | 295 | 46.6% | 52.2% |
| `temporal_lag_6h` | 25.40% | -369.90 | +1730.40 | +2100.30 | 296 | 301 | 46.6% | 53.2% |
| `temporal_lag_12h` | 25.45% | -369.90 | +1348.70 | +1718.60 | 296 | 323 | 46.6% | 49.8% |

Analysis:

- This family is now materially effective. The old temporal experiment produced `ADR = 0.00%`; the new time-aware stale-feed design produces roughly `24%` to `25%` drift across all lag levels.
- Drift saturates quickly after `2h`, which suggests the Rule Engine is highly sensitive to stale consumed signals even under moderate delay.
- The mutant P&L improving here does not imply delay is beneficial in general; it means this particular 30-day window happened to reward the lagged decisions. The main result is that temporal mutation now meaningfully changes behavior.

### Hybrid: Sentiment Family (`heuristic`, `sample=100`)

Command:

```bash
python src/main.py --mode hybrid --families sentiment --backend heuristic --sample 100
```

| Experiment | Logic Gap | Policy Gap | Rule P&L | Agent P&L | Delta | Rule Trades | Agent Trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sentiment_scale_k05` | 6.00% | 14.00% | -94.50 | +149.30 | +243.80 | 3 | 5 |
| `sentiment_scale_k15` | 14.00% | 26.00% | -314.70 | +232.00 | +546.70 | 31 | 23 |
| `sentiment_scale_k20` | 12.00% | 23.00% | -302.20 | +259.10 | +561.30 | 40 | 30 |
| `sentiment_flip_p10` | 6.00% | 18.00% | -104.20 | +175.10 | +279.30 | 18 | 16 |
| `sentiment_flip_p20` | 7.00% | 18.00% | -169.70 | +160.30 | +330.00 | 17 | 16 |
| `sentiment_flip_p30` | 9.00% | 19.00% | -194.80 | +160.30 | +355.10 | 19 | 16 |
| `sentiment_fused_eq` | 10.00% | 18.00% | -206.30 | +133.10 | +339.40 | 15 | 9 |

Analysis:

- The heuristic agent outperforms the Rule Engine on every sentiment experiment in this validation slice.
- Policy gaps are consistently larger than logic gaps, which means the oracle collapses some pre-oracle disagreement into the same final action.
- Increasing sentiment intensity widens the gap between agents, but not monotonically in logic gap terms; the largest policy divergence in this slice appears at `k=1.5`.

### Hybrid: Temporal Family (`heuristic`, `sample=100`)

Command:

```bash
python src/main.py --mode hybrid --families temporal --backend heuristic --sample 100
```

| Experiment | Logic Gap | Policy Gap | Rule P&L | Agent P&L | Delta | Rule Trades | Agent Trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| `temporal_lag_30m` | 4.00% | 17.00% | -156.80 | +51.00 | +207.80 | 15 | 15 |
| `temporal_lag_2h` | 4.00% | 17.00% | -153.60 | -21.10 | +132.50 | 17 | 17 |
| `temporal_lag_6h` | 2.00% | 20.00% | +42.80 | +110.50 | +67.70 | 16 | 16 |
| `temporal_lag_12h` | 7.00% | 29.00% | -161.10 | +163.00 | +324.10 | 16 | 17 |

Analysis:

- Temporal lag creates only modest final-action disagreement in the `heuristic` hybrid slice, with logic gap between `2%` and `7%`.
- Policy gap is much larger than logic gap, especially at `12h`, again showing that oracle application reduces visible final disagreement.
- The largest hybrid temporal divergence in this sample appears at `12h`, where the heuristic agent remains profitable while the Rule Engine stays negative.

### Overall Takeaways

- The most important improvement from this refactor is experimental validity, not raw strategy optimization.
- `sentiment_scale_k20` reproduces the old `intensity_k2` drift result exactly, which confirms that the new framework preserved the previous benchmark.
- The new temporal family is the clearest improvement: temporal mutation is no longer a dead feature and now produces substantial drift.
- Family-level summaries make the results comparable across multiple experiments in a single run, which was not possible in the previous hardcoded setup.

## Output Layout

Outputs are no longer written to a flat directory.

### Drift

```text
outputs/
  drift/
    <family>/
      summary.txt
      <experiment>/
        traj_baseline.json
        traj_mutant.json
        drift_report.txt
        pnl_summary.txt
```

### Hybrid

```text
outputs/
  hybrid/
    <backend>/
      <family>/
        summary.txt
        <experiment>/
          traj_rule.json
          traj_agent.json
          combined_log.json
          consensus_report.txt
          pnl_summary.txt
          plots/
            plot_action_dist.png
            plot_pnl_curves.png
            plot_agreement_matrix.png
            plot_decision_timeline.png
```

`summary.txt` is produced once per family and aggregates the experiments executed in that run.

## Notes

- Each experiment applies exactly one mutation; mutation families are not stacked.
- `stochastic` experiments are reproducible when run with the same `--seed`.
- `hybrid` uses the same mutated DataFrame for both agents so logic gaps remain comparable.
