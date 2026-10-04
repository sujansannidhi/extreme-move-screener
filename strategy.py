"""Single source of truth for the live trading rules (used by daily.py, build_site.py, weekly.py, research).

Execution model (matches the broker):
  - Entry: market-on-close order on the entry day -> fills at that day's CLOSE.
    LONG = buy at the close. SHORT = sell short at the close.
  - Exit: any time from the next session on, at market / limit:
      LONG  target = entry * (1 + m * ATR%)   SHORT target = entry * (1 - m * ATR%)
      if the session OPENS through the target -> exit at the open
      elif the session's range touches the target -> exit at the target
      else after H sessions -> exit at that session's close
  - Costs: 5 bps per side on the market exit (15 bps for names under $50M/day);
    shorts also pay borrow (5% / 25% / 60% a year by volatility tier) for the days held.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = f"{ROOT}/model/params.json"

DEFAULT_PARAMS = {
    "long": {"enabled": True, "k": 3, "target_atr": 0.25, "max_days": 5, "rank_by": "dollar_vol20"},
    "short": {"enabled": False, "k": 3, "target_atr": 0.25, "max_days": 5, "rank_by": "dollar_vol20"},
}

# NYSE full-day closures (update yearly)
NYSE_HOLIDAYS = pd.to_datetime([
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03",
    "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18", "2027-07-05",
    "2027-09-06", "2027-11-25", "2027-12-24",
])


def load_params():
    if os.path.exists(PARAMS_PATH):
        p = json.load(open(PARAMS_PATH))
        for side in DEFAULT_PARAMS:
            p[side] = {**DEFAULT_PARAMS[side], **p.get(side, {})}
        return p
    return json.loads(json.dumps(DEFAULT_PARAMS))


def trading_days_after(day, n):
    """The next n NYSE sessions strictly after `day`."""
    out, d = [], pd.Timestamp(day)
    while len(out) < n:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5 and d not in NYSE_HOLIDAYS:
            out.append(d)
    return out


def exit_cost(dollar_vol):
    return np.where(np.asarray(dollar_vol) < 50e6, 0.0015, 0.0005)


def borrow_rate(rv20, price):
    rv20, price = np.asarray(rv20, float), np.asarray(price, float)
    return np.where(rv20 > 1.5, 0.60, np.where((rv20 > 1.0) | (price < 10), 0.25, 0.05))


def simulate(entry, atr_pct, O, H, L, C, side, m, max_days, cost, borrow=0.0, stop_atr=None):
    """Vectorised exit simulation. O/H/L/C are (n, >=max_days) arrays of the sessions after entry.
    Returns (net_return, days_held, reason, exit_price); NaN where the trade is still open."""
    n = len(entry)
    sgn = 1 if side == "long" else -1
    T = entry * (1 + sgn * m * atr_pct)
    S = entry * (1 - sgn * stop_atr * atr_pct) if stop_atr else None   # protective stop (checked before the target)
    ret = np.full(n, np.nan)
    days = np.full(n, np.nan)
    px = np.full(n, np.nan)
    why = np.full(n, "", dtype=object)
    live = np.isfinite(entry)
    for d in range(max_days):
        o, h, l, c = O[:, d], H[:, d], L[:, d], C[:, d]
        ok = live & np.isfinite(c)
        if side == "long":
            sgap = ok & (o <= S) if S is not None else np.zeros(n, bool)
            stp = ok & ~sgap & (l <= S) if S is not None else np.zeros(n, bool)
            gap = ok & ~sgap & ~stp & (o >= T)
            tgt = ok & ~sgap & ~stp & ~gap & (h >= T)
        else:
            sgap = ok & (o >= S) if S is not None else np.zeros(n, bool)
            stp = ok & ~sgap & (h >= S) if S is not None else np.zeros(n, bool)
            gap = ok & ~sgap & ~stp & (o <= T)
            tgt = ok & ~sgap & ~stp & ~gap & (l <= T)
        tim = ok & ~sgap & ~stp & ~gap & ~tgt & (d == max_days - 1)
        Sx = S if S is not None else c
        for mask, p, code in [(sgap, o, "stop_gap"), (stp, Sx, "stop"), (gap, o, "gap"), (tgt, T, "target"), (tim, c, "time")]:
            px[mask] = p[mask]
            days[mask] = d + 1
            why[mask] = code
            live &= ~mask
    gross = sgn * (px / entry - 1)
    ret = gross - cost - (borrow * days / 252 if side == "short" else 0.0)
    return ret, days, why, px


def select(day_df, side, params, candidate_mask):
    """Pick the side's trades for one signal day."""
    p = params[side]
    x = day_df[(day_df.price >= 3) & (day_df.dollar_vol20 >= 15e6)]
    x = x[candidate_mask(x) & ~x.symbol.isin({"SOXL"})]
    if side == "short" and p.get("min_ret_60") is not None:
        x = x[x.ret_60 >= p["min_ret_60"]]
    return x.sort_values(p["rank_by"], ascending=False).head(p["k"])
