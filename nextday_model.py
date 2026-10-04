"""Walk-forward stock selection for buy-at-close trades.

Models (refit quarterly, rolling 18-month window, purged by 10 sessions):
  pgreen : logistic regression, P(close of day 1 > entry close)          -- the user's literal goal
  er3    : ridge regression on the 3-day forward close return (clipped)   -- expected value
Picks: top K per signal day by score; every exit rule is then evaluated on the same picks.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import ROOT, FEATURE_COLS
from nextday import rule_grid, rule_name

X_COLS = [c for c in FEATURE_COLS if c not in ("exh_count", "cont_count")]
FIRST_TEST, HOLDOUT = "2023-07-01", "2025-07-01"


class Model:
    def __init__(self, kind):
        self.kind = kind

    def fit(self, df, y):
        X = df[X_COLS]
        self.med, self.lo, self.hi = X.median(), X.quantile(0.01), X.quantile(0.99)
        Xc = X.fillna(self.med).clip(self.lo, self.hi, axis=1)
        if self.kind == "pgreen":
            self.m = make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=400)).fit(Xc, y)
        else:
            self.m = make_pipeline(StandardScaler(), Ridge(alpha=500.0)).fit(Xc, y)
        return self

    def predict(self, df):
        Xc = df[X_COLS].fillna(self.med).clip(self.lo, self.hi, axis=1)
        return self.m.predict_proba(Xc)[:, 1] if self.kind == "pgreen" else self.m.predict(Xc)

    def coefs(self):
        return pd.Series(self.m[-1].coef_.ravel(), index=X_COLS)


def walk_forward(df, dates):
    di = {d: i for i, d in enumerate(dates)}
    df = df.assign(di=df.date.map(di))
    preds = []
    last = None
    for q in pd.period_range(FIRST_TEST, df.date.max(), freq="Q"):
        test = df[(df.date >= q.start_time) & (df.date <= q.end_time)]
        if test.empty:
            continue
        cutoff = test.di.min() - 10
        tr = df[(df.di <= cutoff) & (df.date >= q.start_time - pd.DateOffset(months=18))]
        tr = tr.dropna(subset=["ret_c1", "ret_c3"])
        if len(tr) > 120_000:
            tr = tr.sample(120_000, random_state=1)
        mg = Model("pgreen").fit(tr, (tr.ret_c1 > 0).astype(int))
        me = Model("er3").fit(tr, tr.ret_c3.clip(-0.15, 0.15))
        p = test[["date", "symbol"]].copy()
        p["pgreen"], p["er3"] = mg.predict(test), me.predict(test)
        p["base_green"] = (tr.ret_c1 > 0).mean()
        preds.append(p)
        last = (mg, me)
    return pd.concat(preds), last


def pick(df, score, k, universe=None):
    x = df if universe is None else df[universe]
    return x.sort_values(score, ascending=False).groupby("date", sort=True).head(k)


def portfolio(trades, net_col, days_col, k):
    """Realized-P&L equity: each signal day's K picks share 100% of equity / K each, but a position
    only gets capital if it is free (capital locked until exit)."""
    t = trades.dropna(subset=[net_col]).copy()
    t["tret"] = t[net_col].values
    # exit date = entry date + days held (business days; exchange holidays ignored)
    ed = t.entry_date.values.astype("datetime64[D]")
    t["exit_date"] = pd.to_datetime(np.busday_offset(ed, t[days_col].astype(int).values, roll="forward"))
    events = sorted(set(t.entry_date) | set(t.exit_date))
    by_entry = {d: g for d, g in t.groupby("entry_date")}
    eq, cash, open_pos, curve = 1.0, 1.0, [], []
    for d in events:
        still = []
        for p in open_pos:
            if p["exit"] <= d:
                cash += p["alloc"] * (1 + p["ret"])
            else:
                still.append(p)
        open_pos = still
        if d in by_entry:
            g = by_entry[d]
            eq_now = cash + sum(p["alloc"] for p in open_pos)
            per = eq_now / k
            for r in g.itertuples():
                a = min(per, cash)
                if a <= 1e-9:
                    break
                cash -= a
                open_pos.append(dict(exit=r.exit_date, alloc=a, ret=r.tret))
        curve.append((d, cash + sum(p["alloc"] for p in open_pos)))
    e = pd.Series(dict(curve)).sort_index()
    daily = e.resample("B").last().ffill()
    r = daily.pct_change().dropna()
    dd = daily / daily.cummax() - 1
    roll63 = daily.pct_change(63).dropna()
    return dict(total=float(daily.iloc[-1] / daily.iloc[0] - 1), sharpe=float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
                mdd=float(dd.min()), pos3m=float((roll63 > 0).mean()) if len(roll63) else np.nan,
                med3m=float(roll63.median()) if len(roll63) else np.nan), daily


def tstats(x):
    x = pd.Series(x).dropna()
    w, l = x[x > 0].sum(), -x[x < 0].sum()
    return dict(n=len(x), win=float((x > 0).mean()), avg=float(x.mean()), med=float(x.median()),
                p05=float(x.quantile(0.05)), pf=float(w / l) if l > 0 else np.inf)


def main():
    from features import load_panels
    nd = pd.read_pickle(f"{ROOT}/data/nextday_lag1.pkl")
    feats = pd.read_pickle(f"{ROOT}/data/features.pkl")
    df = nd.merge(feats[["date", "symbol"] + [c for c in X_COLS if c not in nd.columns]], on=["date", "symbol"], how="left")
    dates = load_panels()["close"].index
    preds, last = walk_forward(df, dates)
    df = df.merge(preds, on=["date", "symbol"])
    df.to_pickle(f"{ROOT}/out/nextday_oos.pkl")
    return df, last


if __name__ == "__main__":
    import time
    t = time.time()
    df, last = main()
    print("oos rows", len(df), f"{time.time()-t:.0f}s")
