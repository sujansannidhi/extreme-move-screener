"""Daily plan: the next session's orders, with exact dates.

    python3 tools/update_yf.py && python3 features.py && python3 daily.py

Writes out/plan_<signal date>.csv — one row per order (side LONG, or SHORT_WATCH when shorts are off).
These files are the live, forward paper-trade record. Never edit them by hand.
"""
import os
import sys

import numpy as np
import pandas as pd

from features import ROOT, load_meta
from backtest import candidate_mask
from strategy import load_params, select, trading_days_after


def build_plan(day_df, signal_date, params):
    rows = []
    order_day = trading_days_after(signal_date, 1)[0]
    for side in ["long", "short"]:
        p = params[side]
        picks = select(day_df, side, params, candidate_mask)
        if side == "long":
            label = "LONG" if (p["enabled"] and not p.get("paused")) else "LONG_PAUSED"
        else:
            label = "SHORT" if (p["enabled"] and not p.get("paused")) else "SHORT_WATCH"
        sell_by = trading_days_after(order_day, p["max_days"])[-1]
        sgn = 1 if side == "long" else -1
        for r in picks.itertuples():
            rows.append(dict(
                side=label, symbol=r.symbol, signal_date=signal_date.date(), order_date=order_day.date(),
                sell_by=sell_by.date(), last_close=round(r.price, 4), atr_pct=round(r.atr_pct, 5),
                target_atr=p["target_atr"], max_days=p["max_days"],
                target_pct=round(sgn * p["target_atr"] * r.atr_pct, 5),
                target_if_fill_at_last_close=round(r.price * (1 + sgn * p["target_atr"] * r.atr_pct), 4),
                ret_1=round(r.ret_1, 4), ret_5=round(r.ret_5, 4), ret_20=round(r.ret_20, 4), ret_60=round(r.ret_60, 4),
                rev_score=round(r.REV_rule, 1) if np.isfinite(r.REV_rule) else None,
                dollar_vol20=round(r.dollar_vol20), params_version=params.get("version", 1)))
    return pd.DataFrame(rows)


def main(asof=None):
    meta = load_meta()
    f = pd.read_pickle(f"{ROOT}/data/features.pkl")
    f = f[f.symbol.isin(meta.query("group != 'etf'").index)]
    asof = pd.Timestamp(asof) if asof else f.date.max()
    day = f[f.date == asof]
    params = load_params()
    plan = build_plan(day, asof, params)
    os.makedirs(f"{ROOT}/out", exist_ok=True)
    plan.to_csv(f"{ROOT}/out/plan_{asof.date()}.csv", index=False)
    print(f"Signals from the {asof.date()} close (params v{params.get('version', 1)}).")
    for r in plan.itertuples():
        verb = {"LONG": "BUY", "SHORT": "SELL SHORT"}.get(r.side, "WATCH ONLY")
        print(f"  {r.side:12s} {r.symbol:6s} {verb:10s} MOC on {r.order_date} | target {r.target_pct*100:+.2f}% "
              f"(≈ ${r.target_if_fill_at_last_close:,.2f}) | exit by close {r.sell_by}")
    return plan


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
