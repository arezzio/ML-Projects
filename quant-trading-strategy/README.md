# Quant Trading Strategy: BTCUSDT & ETHUSDT

A from-scratch research pipeline that trains a simple autoregressive linear
model to predict future log returns on Binance BTCUSDT and ETHUSDT, then
holds the result to a genuinely out-of-sample standard before calling it a
"strategy."

## What this actually does

1. **Data** — pulls raw tick-level trade data from Binance's public data
   archive and aggregates it into hourly OHLC bars (`binance.py`,
   `research.py`).
2. **Features** — builds the target (next-period log return) and up to 4
   lagged log-return features per asset.
3. **Model search** — benchmarks every combination of up to 3 lag features
   with a linear regression model (PyTorch), ranked by annualized Sharpe on
   a holdout split.
4. **Out-of-sample validation** — because "best of many combinations tried"
   is a data-snooping trap, the notebook re-runs feature selection and
   training on a rolling walk-forward basis, where every test period is
   genuinely unseen during that fold's selection and training.
5. **Statistical significance** — the stitched-together out-of-sample result
   is compared against two baselines:
   - **Buy & hold** (same period, no model)
   - **A random ±1 coin-flip signal**, repeated 1,000 times, to build a null
     distribution — giving a p-value for "could this Sharpe have happened by
     chance?"

## Results (walk-forward, out-of-sample)

| Symbol | OOS Sharpe | Buy & Hold Sharpe | p-value vs. random signal |
|---|---|---|---|
| BTCUSDT | 2.99 | -1.82 | 0.068 |
| ETHUSDT | -0.01 | -2.23 | 0.489 |

**Honest read:** BTCUSDT's out-of-sample result comfortably beats buy-and-hold
and would only be matched by ~7% of random coin-flip signals — suggestive of
a real edge, but not quite past the conventional p<0.05 bar with this much
data. ETHUSDT shows no edge at all (p≈0.49, indistinguishable from chance).
The point of testing both assets wasn't to find two winners — it was to check
whether the BTC result generalizes. It doesn't, which is itself the useful
finding: treat the BTC number as *promising, not proven*, and don't assume it
transfers to other assets.

## Running it

From the repo root (dependencies are shared across projects in this repo):

```bash
uv sync
uv run jupyter lab
```

Open `quant-trading-strategy/quant_trading_strategy.ipynb` and run all cells.
Data is cached to `quant-trading-strategy/cache/` on first run (~168 days of
tick data per symbol, several GB) so re-runs are fast.

> Note: on macOS, PyTorch's and Polars' native thread pools can deadlock
> when Polars does heavy concurrent aggregation right after `torch` is
> imported. The first cell sets `POLARS_MAX_THREADS=1` to avoid this.

## Credits

Built as a follow-along/extension of
[memlabs-research/build-a-quant-trading-strategy](https://github.com/memlabs-research/build-a-quant-trading-strategy),
extended to run the full pipeline on a second asset (ETHUSDT) and to add
walk-forward out-of-sample validation and significance testing, which the
original course material didn't cover.
