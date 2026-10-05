"""Trade log: replays every LIVE order (out/plan_*.csv, published before the trade happened) and tags it.

Only real orders are logged: watch-list names and simulated history are never tagged SUCCESS/FAIL.
  tag     SUCCESS   = closed with a net gain
          FAIL      = closed with a net loss (or flat)
          OPEN      = bought, still inside its window (valued at the last close)
          PENDING   = order day has not happened yet
"""
import glob
import os

import numpy as np
import pandas as pd

from strategy import ROOT, exit_cost, borrow_rate

COLS = ["source", "side", "symbol", "signal_date", "order_date", "sell_by", "entry", "target", "exit", "exit_date",
        "days", "exit_reason", "ret", "tag", "atr_pct", "ret_5", "ret_20", "ret_60", "rev_score", "dollar_vol20"]


def load_prices():
    d = pd.read_csv(f"{ROOT}/data/ohlcv_full.csv.gz", parse_dates=["date"])
    return {k: d.pivot(index="date", columns="symbol", values=k).sort_index() for k in ["open", "high", "low", "close"]}


def replay_row(r, px, dates):
    side = "short" if str(r.side).startswith("SHORT") else "long"
    sgn = 1 if side == "long" else -1
    out = dict(entry=np.nan, target=np.nan, exit=np.nan, exit_date=None, days=np.nan, exit_reason="", ret=np.nan)
    sym = r.symbol
    after = dates[dates > pd.Timestamp(r.signal_date)]
    if len(after) == 0 or sym not in px["close"].columns:
        return out, "PENDING"
    eday = after[0]
    entry = px["close"].at[eday, sym]
    if not np.isfinite(entry):
        return out, "PENDING"
    T = entry * (1 + sgn * r.target_atr * r.atr_pct)
    out.update(entry=entry, target=T)
    cost = float(exit_cost([r.dollar_vol20])[0])
    br = float(borrow_rate([1.0], [entry])[0]) if side == "short" else 0.0
    fut = dates[dates > eday][: int(r.max_days)]
    for k, d in enumerate(fut, 1):
        o, h, l, c = (px[f].at[d, sym] for f in ("open", "high", "low", "close"))
        if not np.isfinite(c):
            continue
        hit_gap = (o >= T) if side == "long" else (o <= T)
        hit = (h >= T) if side == "long" else (l <= T)
        exit_px, why = (o, "gapped through target") if hit_gap else ((T, "target hit") if hit else (None, None))
        if exit_px is None and k == int(r.max_days):
            exit_px, why = c, "time limit"
        if exit_px is not None:
            net = sgn * (exit_px / entry - 1) - cost - br * k / 252
            out.update(exit=exit_px, exit_date=d.date(), days=k, exit_reason=f"{why} (day {k})", ret=net)
            return out, ("SUCCESS" if net > 0 else "FAIL")
    last = px["close"][sym].loc[:dates[-1]].dropna()
    out.update(exit=last.iloc[-1], days=len(fut), exit_reason=f"open · day {len(fut)} of {int(r.max_days)}",
               ret=sgn * (last.iloc[-1] / entry - 1))
    return out, "OPEN"


def build_ledger():
    px = load_prices()
    dates = px["close"].index
    frames = []
    live = sorted(glob.glob(f"{ROOT}/out/plan_*.csv"))
    if live:
        frames.append(pd.concat([pd.read_csv(p) for p in live]).assign(source="LIVE"))
    if not frames:
        return pd.DataFrame(columns=COLS)
    plans = pd.concat(frames, ignore_index=True)
    plans = plans[plans.side.isin(["LONG", "SHORT"])]        # real orders only
    if plans.empty:
        return pd.DataFrame(columns=COLS)
    rows = []
    for r in plans.itertuples():
        res, tag = replay_row(r, px, dates)
        rows.append({**{c: getattr(r, c, None) for c in ["source", "side", "symbol", "signal_date", "order_date", "sell_by",
                                                        "atr_pct", "ret_5", "ret_20", "ret_60", "rev_score", "dollar_vol20"]},
                     **res, "tag": tag})
    led = pd.DataFrame(rows)[COLS]
    led.to_csv(f"{ROOT}/out/ledger.csv", index=False)
    return led


if __name__ == "__main__":
    led = build_ledger()
    print(led.groupby(["source", "side", "tag"]).size())
