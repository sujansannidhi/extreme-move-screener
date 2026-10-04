"""Append the newest daily bars to data/ohlcv_full.csv.gz using yfinance (free, no API key).

    pip install yfinance
    python3 tools/update_yf.py            # fetch from the day after the last stored date
    python3 tools/update_yf.py --days 10  # re-fetch the last 10 calendar days (overwrites those rows)

Bars are split-adjusted (auto_adjust=False, so dividends are NOT adjusted — same as the Robinhood history).
Note: this script was written without network access to Yahoo; run it once and check the printed row counts.
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = f"{ROOT}/data/ohlcv_full.csv.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("pip install yfinance")
    d = pd.read_csv(PATH, parse_dates=["date"])
    syms = sorted(pd.read_csv(f"{ROOT}/data/universe.csv").symbol)
    have = set(d.symbol)
    syms = [s for s in syms if s in have]
    last = d.date.max()
    start = (last - pd.Timedelta(days=a.days)) if a.days else last + pd.Timedelta(days=1)
    if start.date() > pd.Timestamp.today().date():
        print("Already up to date:", last.date())
        return
    raw = yf.download([s.replace(".", "-") for s in syms], start=start.strftime("%Y-%m-%d"),
                      auto_adjust=False, actions=False, group_by="ticker", progress=False, threads=True)
    rows = []
    for s in syms:
        if s not in raw.columns.get_level_values(0):
            continue
        x = raw[s].dropna(subset=["Close"])
        if x.empty:
            continue
        # Yahoo "Close" is split-adjusted but not dividend-adjusted, matching the stored history
        rows.append(pd.DataFrame({"symbol": s, "date": x.index.tz_localize(None).normalize(), "open": x["Open"].round(4),
                                  "high": x["High"].round(4), "low": x["Low"].round(4), "close": x["Close"].round(4),
                                  "volume": x["Volume"].astype("int64")}))
    if not rows:
        print("No new bars returned.")
        return
    new = pd.concat(rows)
    out = pd.concat([d, new]).drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"])
    # sanity: a stock whose history jumps >60% overnight on the seam may need a split re-pull
    seam = out[out.date.isin([last, new.date.min()])].copy()
    seam["r"] = seam.groupby("symbol").close.pct_change()
    bad = seam[seam.r.abs() > 0.6]
    out.to_csv(PATH, index=False)
    print(f"Added {len(new):,} bars for {new.symbol.nunique()} symbols, {new.date.min().date()} → {new.date.max().date()}")
    if len(bad):
        print("Check for an unadjusted split (re-pull full history for these):", bad.symbol.tolist())


if __name__ == "__main__":
    main()
