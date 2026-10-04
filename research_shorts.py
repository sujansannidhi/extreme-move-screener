"""Research: does a 'cover into the first small dip' rule work for shorts? (and long variants for comparison)
Pre-registration: choose on development (2023-07..2025-06); report holdout (2025-07..) once."""
import json
import numpy as np
import pandas as pd

from features import ROOT, load_panels, load_meta
from backtest import candidate_mask
from nextday import build_paths
from strategy import simulate, exit_cost, borrow_rate

P = load_panels()
meta = load_meta()
f = pd.read_pickle(f"{ROOT}/data/features.pkl")
f = f[f.symbol.isin(meta.query("group != 'etf'").index) & (f.symbol != "SOXL")]
f = f[(f.price >= 3) & (f.dollar_vol20 >= 15e6) & f.atr_pct.notna()].reset_index(drop=True)
f = f[candidate_mask(f)].reset_index(drop=True)
E, paths, ed = build_paths(f, P, lag=1)
cost = exit_cost(f.dollar_vol20.values)
br = borrow_rate(f.rv20.fillna(1).values, f.price.values)

universes = {
    "liquid": ("dollar_vol20", None),
    "rev_rule": ("REV_rule", None),
    "most_extended": ("ret_60", 0.5),
    "biggest_5d": ("ret_5", None),
    "exhaustion": ("exh_count", None),
}
rows = []
for side in ["short", "long"]:
    for m in [0.25, 0.5, 0.75]:
        for H in [1, 3, 5]:
            r, d, why, _ = simulate(E, f.atr_pct.values, paths["O"], paths["H"], paths["L"], paths["C"], side, m, H, cost, br)
            f["r"], f["d"] = r, d
            for u, (col, minr) in universes.items():
                x = f if minr is None else f[f.ret_60 >= minr]
                pk = x.sort_values([col, "dollar_vol20"], ascending=False).groupby("date").head(3)
                for per, lo, hi in [("dev", "2023-07-01", "2025-06-30"), ("holdout", "2025-07-01", "2026-12-31")]:
                    t = pk[(pk.date >= lo) & (pk.date <= hi)].dropna(subset=["r"])
                    h1 = t[t.date < (pd.Timestamp(lo) + (pd.Timestamp(hi).normalize() - pd.Timestamp(lo)) / 2)].r.mean() if per == "dev" else np.nan
                    rows.append(dict(side=side, universe=u, m=m, H=H, period=per, n=len(t), win=(t.r > 0).mean(),
                                     avg=t.r.mean(), p05=t.r.quantile(0.05), worst=t.r.min(), days=t.d.mean(), dev_first_half=h1))
res = pd.DataFrame(rows)
res.to_csv(f"{ROOT}/out/research_shorts.csv", index=False)
pd.set_option("display.width", 220)
dev = res[res.period == "dev"]
print("DEV — shorts, sorted by avg:")
print(dev[dev.side == "short"].sort_values("avg", ascending=False).head(15).round(4).to_string(index=False))
print("\nDEV — longs (top 8):")
print(dev[dev.side == "long"].sort_values("avg", ascending=False).head(8).round(4).to_string(index=False))
