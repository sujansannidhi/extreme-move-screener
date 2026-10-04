# Extreme Move Screener

Start with `CLAUDE.md` for the full context. Setup: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

Daily screen + regime classifier + walk-forward backtest for extreme U.S. stock moves.

## Files
| File | What it does |
|---|---|
| `build_universe.py` | Universe: volatile-sector S&P 500 names + ~225 curated high-beta names + sector ETFs (`data/universe.csv`) |
| `features.py` | All features (lagged), peer/sector-adjusted returns, Extreme Move / rule scores, evidence counts → `data/features.pkl` |
| `backtest.py` | Candidate events, close-only trade outcomes with stops and costs, monthly purged walk-forward, regimes, portfolio sim, metrics → `out/` |
| `icw_model.py` | IC-weighted composite model (tested, not used in the final regime rules) |
| `screen.py` | Scores the latest close and writes `out/screen_<date>.csv` |
| `build_report.py` | Builds `report.html` |
| `tools/rh_to_csv.py`, `tools/rh_fund_to_csv.py` | Convert Robinhood MCP historical / fundamentals results into the CSV format |

## Daily run
```
python3 tools/update_yf.py   # new bars (free)
python3 features.py
python3 daily.py             # buy-at-close plan + sell prices for positions.csv
```
The swing-model research screen: `python3 backtest.py` (refits models, ~10 min) then `python3 screen.py`.

## Timing / leakage
Features for day t use data through t's close. Entry = close of t+1. Exits are checked on closes only.
Models are refit monthly on events whose outcome window ended ≥16 sessions before the test month.
Development: Jul-2023 → Jun-2025. Holdout (run once): Jul-2025 → Oct-2026.

## Result in one line
The pre-registered final model failed the holdout (−15% vs SPY +25%). Shorting extreme movers lost money in every
sub-period. The Reversal Score's ranking of large 5-day drops held up out of sample (AUC 0.59 → 0.57).
See `report.html` and `out/backtest_results.json`.
