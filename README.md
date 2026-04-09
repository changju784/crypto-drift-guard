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
