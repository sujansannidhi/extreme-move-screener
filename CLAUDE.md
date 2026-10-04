# Extreme Move Screener

Research + daily trading tool for extreme U.S. stock moves. Python, pandas, scikit-learn.
Data: daily OHLCV for ~508 symbols (S&P 500 volatile sectors + ~215 high-beta names + sector ETFs), Jan 2022 →.

## How the user trades (drives every design choice)
- Buys with **market-on-close** orders placed during or before market hours → fills at that day's **closing price**.
- Can **sell at market any time** during the day (or with limit orders).
- Long-only. Wants to be green by the next day's close, and is willing to hold a few days if the system says so.
- Wants the system to be free (no paid data). Daily data refresh uses yfinance (`tools/update_yf.py`).

## Daily routine
```
python3 tools/update_yf.py   # append new bars to data/ohlcv_full.csv.gz
python3 features.py          # rebuild data/features.pkl (~20 s)
python3 daily.py             # buys for the next close + sell prices for positions.csv
```
`positions.csv` (user-maintained, gitignored): `symbol,fill_date,fill_price`.

## The live rule (pre-registered, see nextday_final.py)
- Universe: extreme movers (`backtest.candidate_mask`), price ≥ $3, 20-day avg dollar volume ≥ $15M, leveraged ETFs excluded.
- Pick the 3 most liquid candidates. Buy MOC next session.
- Sell at fill × (1 + 0.25 × ATR%) as soon as it trades (at the open if it gaps above). Else sell at the close of the 5th session. No stop.
- Backtest (`python3 nextday.py && python3 nextday_final.py dev|holdout`):
  - Dev Jul-23→Jun-25: 1,499 trades, 84% sold at a profit, avg +0.59%, worst-5% −10.5%, +122% vs SPY +39%, max DD −49%.
  - Holdout Jul-25→Oct-26 (run once): 944 trades, 83% profitable, avg +0.31%, worst-5% −12.1%, +51% vs SPY +25%, max DD −40%.
  - Green at the day-1 close is only ~49%: the edge is selling into the first small pop, not predicting direction.
  - Daily bars can't order intraday events; fills assume the target trades exactly at the target price.

## Research history (don't repeat these mistakes)
- `backtest.py`: 10-day swing models (gradient boosting, IC-weighted composite, logistic, untuned rules) with monthly
  purged walk-forward. The pre-registered final model **failed the holdout** (−15% vs SPY +25%).
- Shorting extreme movers lost money in every sub-period (−1.3%/trade), even after RSI ≥ 80 or confirmed breakdowns.
- Reversal Score (P of a large 5-day drop) generalized: AUC 0.59 → 0.57. Useful as an avoid/exit signal, not a short trigger.
- Next-day direction models (`nextday_model.py`) had no skill (AUC ≈ 0.51).

## Research protocol (mandatory)
- Features for day t use data through t's close; entry is never earlier than the close of t+1 (lag=1).
- Tune only on development data (≤ 2025-06-30). The holdout (≥ 2025-07-01) has been used; any new idea needs
  fresh out-of-sample data (paper-trade forward from today) before it is trusted.
- Report results against SPY and against the naive baseline (random candidate), with costs.

## Files
| File | Purpose |
|---|---|
| `features.py` | Lagged features, sector/peer-adjusted returns, Extreme Move / rule scores |
| `backtest.py` | Swing-model walk-forward backtest (research) → `out/backtest_results.json` |
| `nextday.py` | Buy-at-close outcome simulation for a grid of exit rules → `data/nextday_lag1.pkl` |
| `nextday_final.py` | The live rule, evaluated on dev / holdout |
| `nextday_model.py` | Walk-forward next-day models (no skill; kept for reference) |
| `daily.py` | Today's plan |
| `screen.py`, `build_report.py` | Swing-model screen and the HTML report (`report.html`) |
| `tools/` | Data converters (Robinhood MCP → CSV) and the yfinance updater |

Large derived files (`data/features.pkl`, `data/nextday_lag1.pkl`, `out/*.pkl`) are gitignored; rebuild with the scripts.
