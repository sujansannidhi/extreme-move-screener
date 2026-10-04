"""Daily buy-at-close plan.

Run after the market closes (or before the open next day):
    python3 tools/update_yf.py      # append the newest daily bars (free, via yfinance)
    python3 features.py             # rebuild features (~20 s)
    python3 daily.py                # today's buys + sell prices for open positions

Rule (fixed before the Jul-2025+ holdout, see nextday_final.py):
  - Universe: extreme-move candidates (backtest.candidate_mask), $3+ price, $15M+ avg dollar volume
  - Buy: the 3 most liquid candidates, market-on-close order on the next session
  - Sell: limit sell at fill * (1 + 0.25 * ATR%); if it opens above, sell at the open
  - Time limit: sell at the close of the 5th session after the buy if the target never trades. No stop.

positions.csv (you maintain it): symbol,fill_date,fill_price   one row per open position
"""
import os
import numpy as np
import pandas as pd

from features import ROOT, load_meta
from backtest import candidate_mask

K, TARGET_ATR, MAX_DAYS = 3, 0.25, 5
EXCLUDE = {"SOXL"}


def next_sessions(start, n):
    """Business days after `start` (exchange holidays not modeled — check your broker's calendar)."""
    return pd.bdate_range(start + pd.Timedelta(days=1), periods=n)


def main():
    meta = load_meta()
    f = pd.read_pickle(f"{ROOT}/data/features.pkl")
    asof = f.date.max()
    t = f[(f.date == asof) & f.symbol.isin(meta.query("group != 'etf'").index) & ~f.symbol.isin(EXCLUDE)]
    t = t[(t.price >= 3) & (t.dollar_vol20 >= 15e6)]
    c = t[candidate_mask(t)].sort_values("dollar_vol20", ascending=False)
    buy_day = next_sessions(asof, 1)[0]
    last_day = next_sessions(buy_day, MAX_DAYS)[-1]
    picks = c.head(K).copy()
    picks["target_pct"] = TARGET_ATR * picks.atr_pct
    picks["sell_price_if_fill_at_last_close"] = picks.price * (1 + picks.target_pct)
    picks["buy_on_close_of"] = buy_day.date()
    picks["sell_by_close_of"] = last_day.date()
    cols = ["symbol", "price", "target_pct", "sell_price_if_fill_at_last_close", "buy_on_close_of", "sell_by_close_of",
            "ret_1", "ret_5", "ret_20", "ret_60", "atr_pct", "dollar_vol20"]
    os.makedirs(f"{ROOT}/out", exist_ok=True)
    picks[cols].to_csv(f"{ROOT}/out/plan_{asof.date()}.csv", index=False)

    print(f"Signals from the {asof.date()} close. {len(c)} extreme movers qualified.\n")
    print(f"BUY (market-on-close on {buy_day:%a %b %d}):")
    for r in picks.itertuples():
        print(f"  {r.symbol:6s} last close ${r.price:,.2f} | target +{r.target_pct*100:.2f}% "
              f"-> limit sell ≈ ${r.sell_price_if_fill_at_last_close:,.2f} (recompute from your fill) "
              f"| sell at close {last_day:%a %b %d} if not hit")

    pos_path = f"{ROOT}/positions.csv"
    if os.path.exists(pos_path):
        pos = pd.read_csv(pos_path, parse_dates=["fill_date"])
        if len(pos):
            atr = t.set_index("symbol").atr_pct.reindex(pos.symbol).values
            print("\nOPEN POSITIONS:")
            for (r, a) in zip(pos.itertuples(), atr):
                a = a if np.isfinite(a) else np.nan
                # ATR at the signal date is what the backtest used; current ATR is a close proxy
                tgt = r.fill_price * (1 + TARGET_ATR * a)
                dl = next_sessions(r.fill_date, MAX_DAYS)[-1]
                last = t.set_index("symbol").price.get(r.symbol, np.nan)
                print(f"  {r.symbol:6s} filled ${r.fill_price:,.2f} on {r.fill_date:%b %d} | limit sell ${tgt:,.2f} "
                      f"| last close ${last:,.2f} ({last / r.fill_price - 1:+.1%}) | time exit at close {dl:%a %b %d}"
                      + ("  <-- SELL AT TODAY'S CLOSE" if dl.date() <= next_sessions(asof, 1)[0].date() else ""))


if __name__ == "__main__":
    main()
