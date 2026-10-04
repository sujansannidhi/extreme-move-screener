"""Buy-at-close / sell-intraday trade outcomes.

Execution model (matches the user's broker):
  signal day s : features from data through the close of s (computed the evening of s)
  entry day e  : market-on-close buy order placed during day e, filled at CLOSE[e]   (e = s + lag, lag=1 by default)
  exit         : market sell any time from day e+1 on. Simulated from daily OHLC:
                  - opens at/above target  -> sell at the open
                  - opens at/below stop    -> sell at the open
                  - low touches stop       -> sell at stop   (stop checked BEFORE target on the same day: conservative)
                  - high touches target    -> sell at target
                  - green at a close (optional, from day 1) -> sell at that close
                  - last allowed day       -> sell at the close
"""
import itertools
import numpy as np
import pandas as pd

from features import ROOT, load_panels, load_meta

MAXH = 5
# MOC buy fills in the closing auction (no spread); market sell pays about half the spread + impact
SELL_BPS, SELL_ILLIQ_BPS, ILLIQ_DV = 5, 15, 50e6


def rule_grid():
    grid = []
    for m, H, stop, green in itertools.product([0.25, 0.5, 0.75, 1.0], [1, 2, 3, 5], [None, 1.5], [True, False]):
        if H == 1 and not green:          # with H=1 the day-1 close is the exit anyway
            continue
        grid.append(dict(m=m, H=H, stop=stop, green=green))
    return grid


def rule_name(r):
    s = "nostop" if r["stop"] is None else f"stop{r['stop']}"
    return f"T{r['m']}_H{r['H']}_{s}_{'green' if r['green'] else 'hold'}"


def build_paths(feats, P, lag=1):
    """Attach entry price and the next MAXH days of OHLC to each feature row."""
    O, Hh, L, C = (P[k] for k in ["open", "high", "low", "close"])
    dates = C.index
    di = {d: i for i, d in enumerate(dates)}
    sj = {s: j for j, s in enumerate(C.columns)}
    i = feats.date.map(di).values
    j = feats.symbol.map(sj).values
    n = len(dates)
    e = i + lag
    ok = e + 1 < n
    A = {k: v.values for k, v in dict(O=O, H=Hh, L=L, C=C).items()}
    E = np.full(len(feats), np.nan)
    E[ok] = A["C"][e[ok], j[ok]]
    paths = {k: np.full((len(feats), MAXH), np.nan) for k in "OHLC"}
    for d in range(MAXH):
        idx = e + 1 + d
        m = idx < n
        for k in "OHLC":
            paths[k][m, d] = A[k][idx[m], j[m]]
    entry_date = np.full(len(feats), np.datetime64("NaT"), dtype="datetime64[ns]")
    entry_date[ok] = dates.values[e[ok]]
    return E, paths, entry_date


def simulate_rule(E, atr_pct, paths, rule, slip):
    """Vectorised exit simulation. Returns net return, days held, exit reason code."""
    n = len(E)
    T = E * (1 + rule["m"] * atr_pct)
    S = E * (1 - rule["stop"] * atr_pct) if rule["stop"] is not None else np.full(n, -np.inf)
    out = np.full(n, np.nan)
    days = np.full(n, np.nan)
    reason = np.full(n, "", dtype=object)
    open_ = np.isfinite(E)
    last_c = E.copy()
    for d in range(rule["H"]):
        O, Hh, L, C = (paths[k][:, d] for k in "OHLC")
        valid = open_ & np.isfinite(C)
        last_c = np.where(valid, C, last_c)
        # 1) gap through target / stop at the open
        g_up = valid & (O >= T)
        g_dn = valid & ~g_up & (O <= S)
        st = valid & ~g_up & ~g_dn & (L <= S)
        tg = valid & ~g_up & ~g_dn & ~st & (Hh >= T)
        rem = valid & ~(g_up | g_dn | st | tg)
        green = rem & rule["green"] & (C > E)
        final = rem & ~green & (d == rule["H"] - 1)
        for mask, px, code in [(g_up, O, "gap_target"), (g_dn, O, "gap_stop"), (st, S, "stop"), (tg, T, "target"),
                               (green, C, "green_close"), (final, C, "time")]:
            out[mask] = px[mask] / E[mask] - 1
            days[mask] = d + 1
            reason[mask] = code
            open_ &= ~mask
    # positions still open because of missing bars at the end of data: mark as incomplete (NaN)
    net = out - slip
    return net, days, reason


def load_universe():
    meta = load_meta()
    feats = pd.read_pickle(f"{ROOT}/data/features.pkl")
    feats = feats[feats.symbol.isin(meta.query("group != 'etf'").index)]
    liq = (feats.price >= 3) & (feats.dollar_vol20 >= 15e6) & feats.atr_pct.notna()
    return feats[liq].reset_index(drop=True)


def main(lag=1):
    P = load_panels()
    feats = load_universe()
    E, paths, entry_date = build_paths(feats, P, lag)
    slip = np.where(feats.dollar_vol20 < ILLIQ_DV, SELL_BPS + SELL_ILLIQ_BPS, SELL_BPS) / 1e4
    atr = feats.atr_pct.values
    res = {"date": feats.date.values, "symbol": feats.symbol.values, "entry_date": entry_date, "entry": E}
    # reference outcomes
    res["ret_c1"] = paths["C"][:, 0] / E - 1 - slip
    res["ret_c3"] = paths["C"][:, 2] / E - 1 - slip
    res["atr_pct"] = atr            # sell at day-1 close
    res["max_h1"] = paths["H"][:, 0] / E - 1                          # best price available on day 1
    res["min_l1"] = paths["L"][:, 0] / E - 1
    out = pd.DataFrame(res)
    for r in rule_grid():
        net, days, reason = simulate_rule(E, atr, paths, r, slip)
        k = rule_name(r)
        out[f"net_{k}"] = net.astype("float32")
        out[f"days_{k}"] = days.astype("float32")
        out[f"why_{k}"] = pd.Categorical(reason)
    out.to_pickle(f"{ROOT}/data/nextday_lag{lag}.pkl")
    return out


if __name__ == "__main__":
    import sys, time
    lag = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    t = time.time()
    o = main(lag)
    print(o.shape, f"{time.time()-t:.0f}s")
    dev = o[(o.date >= "2023-07-01") & (o.date < "2025-07-01")]
    print("rows dev:", len(dev), "| day-1 close green rate:", round((dev.ret_c1 > 0).mean(), 3), "avg", round(dev.ret_c1.mean(), 4))
    rows = []
    for r in rule_grid():
        k = rule_name(r)
        x = dev[f"net_{k}"].dropna()
        rows.append(dict(rule=k, win=(x > 0).mean(), avg=x.mean(), med=x.median(), p05=x.quantile(0.05),
                         days=dev[f"days_{k}"].mean()))
    print(pd.DataFrame(rows).sort_values("avg", ascending=False).round(4).to_string(index=False))
