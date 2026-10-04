"""Walk-forward IC-weighted factor composite ("ICW").

For each test month, using ONLY purged training events:
  1. per-month Spearman IC of every cross-sectional feature vs. the after-cost trade return
     (long_net for continuation, short_net for reversal)
  2. keep features whose IC t-stat (mean/std*sqrt(months)) >= T_MIN; weight = mean IC
  3. map each feature to its percentile in the training distribution; composite = weighted mean
  4. calibrate composite -> win probability and expected net return with training deciles
Date-level (market) features are excluded from the composite because their cross-sectional
IC is undefined; they are used only as a separate regime input elsewhere.
"""
import numpy as np
import pandas as pd

from features import FEATURE_COLS

MARKET_COLS = {"mkt_ret20", "mkt_dist50", "mkt_rv20", "smallcap_ret20"}
XS_COLS = [c for c in FEATURE_COLS if c not in MARKET_COLS]
T_MIN = 2.0
MIN_MONTH_N = 30


class ICComposite:
    def __init__(self, target, label):
        self.target, self.label = target, label

    def fit(self, tr):
        tr = tr.dropna(subset=[self.target, self.label])
        ym = tr.date.dt.to_period("M")
        ics = {}
        for m, g in tr.groupby(ym):
            if len(g) < MIN_MONTH_N:
                continue
            ics[m] = g[XS_COLS].corrwith(g[self.target], method="spearman")
        ic = pd.DataFrame(ics).T
        mu, sd, n = ic.mean(), ic.std(), ic.notna().sum()
        t = mu / sd * np.sqrt(n)
        self.ic_mean, self.ic_t = mu, t
        sel = t.abs() >= T_MIN
        self.w = mu[sel]
        self.sorted = {f: np.sort(tr[f].dropna().values) for f in self.w.index}
        s = self.score(tr)
        self.edges = np.unique(np.quantile(s, np.linspace(0, 1, 11)))
        b = np.clip(np.searchsorted(self.edges, s, side="right") - 1, 0, len(self.edges) - 2)
        grp = pd.DataFrame({"b": b, "y": tr[self.label].values, "r": tr[self.target].values}).groupby("b")
        self.p_bin, self.ev_bin = grp.y.mean(), grp.r.mean()
        self.base = tr[self.label].mean()
        return self

    def score(self, df):
        if len(self.w) == 0:
            return np.zeros(len(df))
        tot = np.zeros(len(df))
        for f, w in self.w.items():
            x = df[f].values
            v = self.sorted[f]
            pct = np.searchsorted(v, x, side="right") / max(len(v), 1)
            pct = np.where(np.isfinite(x), pct, 0.5)
            tot += w * (pct - 0.5)
        return tot / self.w.abs().sum()

    def predict(self, df):
        s = self.score(df)
        b = np.clip(np.searchsorted(self.edges, s, side="right") - 1, 0, len(self.edges) - 2)
        p = self.p_bin.reindex(b).values
        ev = self.ev_bin.reindex(b).values
        return s, p, ev
