"""Pre-registered buy-at-close strategy (fixed before looking at the Jul-2025+ holdout).

  universe : extreme-move candidates (backtest.candidate_mask), leveraged ETFs excluded
  picks    : top K=3 per signal day by 20-day average dollar volume
  buy      : market-on-close order on the next session -> fills at that close
  sell     : as soon as price trades at entry * (1 + 0.25 * ATR%)  (or at the open if it gaps above)
             otherwise sell at the close of the 5th session after entry. No stop.
"""
import numpy as np
import pandas as pd
from features import ROOT
from backtest import candidate_mask
from nextday_model import portfolio, tstats

RULE = "T0.25_H5_nostop_hold"
K = 3
EXCLUDE = {"SOXL"}


def picks(period=None):
    nd = pd.read_pickle(f"{ROOT}/data/nextday_lag1.pkl")
    f = pd.read_pickle(f"{ROOT}/data/features.pkl")
    cols = ["date", "symbol", "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60", "EMS", "price", "dollar_vol20"]
    d = nd.merge(f[cols], on=["date", "symbol"])
    d = d[candidate_mask(d) & ~d.symbol.isin(EXCLUDE)]
    if period:
        d = d[(d.date >= period[0]) & (d.date <= period[1])]
    return d.sort_values("dollar_vol20", ascending=False).groupby("date").head(K), d


def evaluate(period):
    p, allc = picks(period)
    net, days = f"net_{RULE}", f"days_{RULE}"
    t = p.dropna(subset=[net])
    stats = tstats(t[net])
    stats["green_at_day1_close"] = float((t.ret_c1 > 0).mean())
    stats["target_hit_day1"] = float(((t[f"why_{RULE}"].isin(["target", "gap_target"])) & (t[days] == 1)).mean())
    stats["exit_with_profit_by_day"] = {int(k): float(v) for k, v in
                                        (t[net] > 0).groupby(t[days]).sum().cumsum().div(len(t)).items()}
    stats["avg_days"] = float(t[days].mean())
    pf, daily = portfolio(t, net, days, K)
    rand = allc.dropna(subset=[net]).groupby("date").sample(n=1, random_state=3, replace=True)
    return dict(trades=stats, portfolio=pf, random_candidate=tstats(allc[net]), worst=t.nsmallest(5, net)[["date", "symbol", net]].to_dict("records")), daily


if __name__ == "__main__":
    import json, sys
    which = sys.argv[1]
    per = {"dev": ("2023-07-01", "2025-06-30"), "holdout": ("2025-07-01", "2026-12-31")}[which]
    r, daily = evaluate(per)
    daily.to_csv(f"{ROOT}/out/nextday_equity_{which}.csv")
    json.dump(r, open(f"{ROOT}/out/nextday_{which}.json", "w"), indent=1, default=str)
    print(json.dumps(r, indent=1, default=str))
