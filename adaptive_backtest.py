"""Does weekly re-tuning ('learning from recent mistakes') beat the fixed rule?

Each Friday, pick the exit settings (target multiple m, max hold H) that did best on the trailing window of
completed trades, then use them for the next week's signals. Compare the forward results with the fixed rule.
Only trades whose outcome was fully known by the Friday are used for the choice (no look-ahead).
"""
import itertools
import json
import numpy as np
import pandas as pd

from features import ROOT, load_panels, load_meta
from backtest import candidate_mask
from nextday import build_paths
from strategy import simulate, exit_cost

GRID = list(itertools.product([0.25, 0.5, 0.75], [3, 5]))
FIXED = (0.25, 5)


def outcomes():
    P = load_panels()
    meta = load_meta()
    f = pd.read_pickle(f"{ROOT}/data/features.pkl")
    f = f[f.symbol.isin(meta.query("group != 'etf'").index) & (f.symbol != "SOXL")]
    f = f[(f.price >= 3) & (f.dollar_vol20 >= 15e6) & f.atr_pct.notna()].reset_index(drop=True)
    f = f[candidate_mask(f)]
    pk = f.sort_values("dollar_vol20", ascending=False).groupby("date").head(3).reset_index(drop=True)
    E, paths, ed = build_paths(pk, P, 1)
    cost = exit_cost(pk.dollar_vol20.values)
    out = pk[["date", "symbol"]].copy()
    out["entry_date"] = ed
    dates = P["close"].index
    for m, H in GRID:
        r, d, _, _ = simulate(E, pk.atr_pct.values, paths["O"], paths["H"], paths["L"], paths["C"], "long", m, H, cost)
        out[f"r_{m}_{H}"] = r
        # date the outcome became known = entry date + days held (sessions)
        di = np.searchsorted(dates.values, ed)
        known = np.where(np.isfinite(d), di + np.nan_to_num(d).astype(int), len(dates) - 1)
        out[f"k_{m}_{H}"] = dates.values[np.clip(known, 0, len(dates) - 1)]
    return out


def run(out, window_weeks, metric):
    fridays = pd.date_range("2023-06-30", out.date.max(), freq="W-FRI")
    chosen, rows = [], []
    for i, fri in enumerate(fridays[:-1]):
        lo = fri - pd.Timedelta(weeks=window_weeks)
        scores = {}
        for m, H in GRID:
            c = f"r_{m}_{H}"
            hist = out[(out.date >= lo) & (out[f"k_{m}_{H}"] <= fri)][c].dropna()
            if len(hist) < 30:
                continue
            scores[(m, H)] = hist.mean() if metric == "avg" else hist.mean() / (hist.std() + 1e-9)
        pick = max(scores, key=scores.get) if scores else FIXED
        nxt = out[(out.date > fri) & (out.date <= fridays[i + 1])]
        rows.append(nxt.assign(r_adapt=nxt[f"r_{pick[0]}_{pick[1]}"], r_fixed=nxt[f"r_{FIXED[0]}_{FIXED[1]}"],
                               pick=f"{pick[0]}x{pick[1]}"))
    return pd.concat(rows)


def stats(x):
    x = x.dropna()
    return dict(n=len(x), avg=round(float(x.mean()), 5), win=round(float((x > 0).mean()), 3), p05=round(float(x.quantile(.05)), 4))


if __name__ == "__main__":
    out = outcomes()
    res = {}
    for win in [13, 26, 52]:
        for metric in ["avg", "sharpe"]:
            r = run(out, win, metric)
            res[f"{win}w_{metric}"] = dict(adaptive=stats(r.r_adapt), fixed=stats(r.r_fixed),
                                           adaptive_holdout=stats(r[r.date >= "2025-07-01"].r_adapt),
                                           fixed_holdout=stats(r[r.date >= "2025-07-01"].r_fixed),
                                           picks=r.drop_duplicates("date").pick.value_counts().to_dict())
    json.dump(res, open(f"{ROOT}/out/adaptive_backtest.json", "w"), indent=1)
    for k, v in res.items():
        print(f"{k:12s} adaptive avg {v['adaptive']['avg']:+.4f} win {v['adaptive']['win']:.3f} | fixed avg {v['fixed']['avg']:+.4f} win {v['fixed']['win']:.3f}"
              f" || holdout adaptive {v['adaptive_holdout']['avg']:+.4f} fixed {v['fixed_holdout']['avg']:+.4f} | picks {v['picks']}")
